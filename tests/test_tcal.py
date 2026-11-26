"""The methods with the temperature selected inside the fit (work-plan card WP-34, ``methods/tcal.py``): the split and
the bags of ``tau_select``, the selection rule, the deployed fits and their bandwidth, the fallback, the factory entries
and the per-metric evaluation of ``main``."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import quapy as qp
from cap.error import vanilla_acc
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection

from uncawe.experiments import main, target_size
from uncawe.experiments import methods as em
from uncawe.experiments import tau_select as ts
from uncawe.experiments.binform import E55_METHODS, E55_TCAL_BASES, _e55_method_order, e55_method
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.per_metric import bag_runs, metric_columns, metric_run
from uncawe.experiments.records import EvaluationContext
from uncawe.methods import tcal
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.plots.common import METHOD_COLORS, METHOD_PLOT_LABELS
from uncawe.tables import tempering as tt
from uncawe.tables.common import METHOD_LATEX_LABELS, STORED_TCAL_BASES, TCAL_METHODS

qp.environ.setdefault("_R_SEED", 0)


def _validation(counts, seed=0, n_features=2):
    """A validation set with the given class counts, features and posteriors of a classifier right ~80% of the time."""
    rng = np.random.default_rng(seed)
    y = rng.permutation(np.repeat(np.arange(len(counts)), counts))
    X = rng.normal(size=(y.size, n_features)) + y[:, None]
    k = len(counts)
    pred = np.where(rng.random(y.size) < 0.8, y, rng.integers(0, k, y.size))
    posteriors = np.full((y.size, k), 0.1 / max(k - 1, 1))
    posteriors[np.arange(y.size), pred] = 0.9
    return LabelledCollection(X, y, classes=np.arange(k)), posteriors


class _Interval:
    def __init__(self, samples):
        self.samples = np.asarray(samples, dtype=float)
        self.point_estimate = float(np.mean(self.samples)) if self.samples.size else np.nan


class _FakeBase:
    """A base at temperature ``tau`` whose interval for each bag is centred on the true value of the bag (read from a
    lookup by the first instance) with half-width ``|tau - best|``-dependent widths, so that the best ``tau`` per metric
    is known; it records every fit and prediction. With ``selects`` it selects a bandwidth that depends on the size of
    the data it is fitted on, as ``bandwidth="logloss"``."""

    log: list = []

    def __init__(self, temperature, bandwidth=None, best=None, selects=False, truth=None):
        self.temperature = float(temperature)
        self.bandwidth = bandwidth
        self.best = best or {}
        self.selects = selects
        self.truth = truth
        self.acc_fn = vanilla_acc

    def fit(self, val, posteriors):
        self.n_fit = len(val.y)
        self.fit_index = np.asarray(val.X[:, 0]).astype(int)
        if self.selects and self.bandwidth is None:
            self.quantifier_ = SimpleNamespace(selects_logloss=True, logloss_bandwidth_=1.0 / self.n_fit)
        _FakeBase.log.append(("fit", self.temperature, self.bandwidth, self.n_fit))
        return self

    def predict_ct_range(self, X, posteriors):
        _FakeBase.log.append(("predict", self.temperature, self.bandwidth, len(X)))
        index = np.asarray(X[:, 0]).astype(int)
        return ("tables", self.temperature, tuple(index))

    def switch(self, acc_fn):
        self.acc_fn = acc_fn
        return self

    def ci_from_cts(self, cts):
        _, tau, index = cts
        metric = next(name for name, fn in _METRICS if fn is self.acc_fn)
        true = self.acc_fn(self.truth.table(index))
        width = 0.01 + abs(np.log(tau) - np.log(self.best.get(metric, 1.0)))
        return _Interval([true - width, true + width])


class _Truth:
    """The realised table of a calibration bag, from the labels and the predictions."""

    def __init__(self, y, pred, n_classes):
        self.y, self.pred, self.n_classes = y, pred, n_classes

    def table(self, index):
        return contingency_table(self.y[list(index)], self.pred[list(index)], self.n_classes)


# the metrics of the fakes, as fixed function objects (gen_acc_measure builds new ones at each call)
_METRICS = list(gen_acc_measure(True))


def _metrics(multiclass):
    return _METRICS


def _indexed(val: LabelledCollection) -> LabelledCollection:
    """``val`` with the position of each instance as its first feature, so that the fakes can read which instances they
    see."""
    X = np.column_stack([np.arange(len(val.y)), val.X])
    return LabelledCollection(X, val.y, classes=val.classes_)


def _method(best=None, selects=False, bandwidth="reuse", n_bags=6, size=40, truth=None, taus=tcal.TAUS):
    def factory(temperature, bandwidth=None):
        return _FakeBase(temperature, bandwidth, best=best, selects=selects, truth=truth)

    return tcal.CalibratedTemperature(
        factory, metrics=_metrics, taus=taus, n_bags=n_bags, cal_bag_size=size, bandwidth=bandwidth
    )


# --------------------------------------------------------------------------------------------------- split and bags


class SplitAndBagsTest(unittest.TestCase):
    def test_tau_select_uses_the_split_and_bags_of_the_method(self):
        self.assertIs(ts.split_index, tcal.split_index)
        self.assertIs(ts.calibration_bags, tcal.calibration_bags)
        self.assertIs(ts.split_seed, tcal.split_seed)
        self.assertIs(ts.calibration_seed, tcal.calibration_seed)
        self.assertEqual(ts.TAUS, tcal.TAUS)
        self.assertEqual(em.TAU_GRID, tcal.TAUS)
        self.assertEqual(ts.N_BAGS, tcal.N_BAGS)
        self.assertEqual(ts.SAMPLE_SIZE, tcal.CAL_BAG_SIZE)

    def test_fits_on_v_fit_and_predicts_the_bags_of_tau_select(self):
        val, post = _validation([30, 50, 20])
        val = _indexed(val)
        y, pred = np.asarray(val.y), post.argmax(axis=1)
        method = _method(truth=_Truth(y, pred, 3), n_bags=4, size=25).set_dataset("toy")
        _FakeBase.log = []
        captured = []
        original = _FakeBase.fit

        def fit(self, data, posteriors):
            captured.append(np.asarray(data.X[:, 0]).astype(int))
            return original(self, data, posteriors)

        with patch.object(_FakeBase, "fit", fit):
            method.fit(val, post)
        fit_index, cal_index = ts.split_index(y, 3, "toy")
        bags = ts.calibration_bags("toy", y, pred, cal_index, 3, n_bags=4, size=25)
        # one fit on V_fit per tau, then the deployed fits on the full V
        for index in captured[: len(tcal.TAUS)]:
            np.testing.assert_array_equal(index, fit_index)
        for index in captured[len(tcal.TAUS) :]:
            np.testing.assert_array_equal(index, np.arange(len(y)))
        predicted = [entry for entry in _FakeBase.log if entry[0] == "predict"]
        self.assertEqual(len(predicted), 4 * len(tcal.TAUS))
        self.assertTrue(all(entry[3] == 25 for entry in predicted))
        self.assertEqual(method.v_fit_size_, fit_index.size)
        self.assertEqual(method.v_cal_size_, cal_index.size)
        self.assertEqual(len(bags), 4)


# --------------------------------------------------------------------------------------------------- selection


class SelectionTest(unittest.TestCase):
    def test_best_tau_is_that_of_the_tables_of_e53(self):
        cases = [
            {0.5: 3.0, 1.0: 2.0, 2.0: 1.0, 4.0: 1.5, 8.0: 4.0},
            {0.5: 1.0, 1.0: 2.0, 2.0: 1.0, 4.0: 1.0, 8.0: 4.0},  # tie: closest to 1 on the log scale, then smaller
            {0.5: 1.0, 1.0: 1.0, 2.0: 1.0, 4.0: 1.0, 8.0: 1.0},
            {0.5: np.nan, 1.0: np.nan, 2.0: 5.0, 4.0: np.nan, 8.0: np.nan},
            {t: np.nan for t in tcal.TAUS},
        ]
        for case in cases:
            a, b = tcal.best_tau(case), tt.best_tau(case)
            self.assertTrue((np.isnan(a) and np.isnan(b)) or a == b, (case, a, b))
        self.assertEqual(tcal.best_tau(cases[1]), 0.5)

    def test_bag_winkler_is_that_of_the_tables(self):
        from uncawe.tables.main import get_acc_winkler

        rng = np.random.default_rng(3)
        for _ in range(5):
            samples, true = rng.random(200), float(rng.random())
            self.assertAlmostEqual(tcal.bag_winkler(samples, true), get_acc_winkler("", true, samples))
        self.assertTrue(np.isnan(tcal.bag_winkler([0.1, np.nan], 0.5)))

    def test_tau_selected_per_metric(self):
        val, post = _validation([40, 60, 30])
        val = _indexed(val)
        best = {"vanilla_accuracy": 2.0, "macro-F1": 0.5, "macro-K": 2.0}
        truth = _Truth(np.asarray(val.y), post.argmax(axis=1), 3)
        method = _method(best=best, truth=truth).set_dataset("toy")
        _FakeBase.log = []
        method.fit(val, post)
        self.assertEqual(method.tau_cal_, best)
        self.assertFalse(any(method.fallback_.values()))
        # two distinct tau: two deployed fits on the full V
        self.assertEqual(set(method.deployed()), {(2.0, False), (0.5, False)})
        full = [e for e in _FakeBase.log if e[0] == "fit" and e[3] == len(val.y)]
        self.assertEqual(sorted(e[1] for e in full), [0.5, 2.0])
        self.assertIs(method.deployed_for("vanilla_accuracy"), method.deployed_for("macro-K"))
        self.assertEqual(method.deployed_for("macro-F1").temperature, 0.5)
        columns = method.metric_columns("macro-F1")
        self.assertEqual(columns["tau_cal"], 0.5)
        self.assertEqual(columns["tau_cal_vanilla_accuracy"], 2.0)
        self.assertEqual(columns["cal_taus"], list(tcal.TAUS))
        self.assertEqual(int(np.argmin(columns["cal_winkler"])), list(tcal.TAUS).index(0.5))
        self.assertFalse(columns["tau_fallback"])
        self.assertEqual(columns["split_seed"], ts.split_seed("toy"))
        self.assertTrue(np.isnan(columns["bandwidth_fit"]) and np.isnan(columns["bandwidth_deployed"]))

    def test_one_deployed_fit_when_the_metrics_agree(self):
        val, post = _validation([40, 60])
        val = _indexed(val)
        truth = _Truth(np.asarray(val.y), post.argmax(axis=1), 2)
        method = _method(best={m: 4.0 for m in get_acc_names()}, truth=truth).set_dataset("toy").fit(val, post)
        self.assertEqual(list(method.deployed()), [(4.0, False)])

    def test_needs_the_dataset(self):
        val, post = _validation([20, 20])
        with self.assertRaisesRegex(RuntimeError, "set_dataset"):
            _method().fit(val, post)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            _method(bandwidth="other")
        with self.assertRaises(ValueError):
            _method(taus=(1.0, 1.0))
        with self.assertRaises(ValueError):
            _method(taus=(0.0, 1.0))


# ------------------------------------------------------------------------------------------- bandwidth and fallback


class BandwidthAndFallbackTest(unittest.TestCase):
    def setUp(self):
        val, post = _validation([40, 60, 30])
        self.val, self.post = _indexed(val), post
        self.truth = _Truth(np.asarray(val.y), post.argmax(axis=1), 3)
        self.best = {m: 2.0 for m in get_acc_names()}

    def test_reuse_the_bandwidth_of_v_fit(self):
        method = _method(best=self.best, selects=True, truth=self.truth).set_dataset("toy").fit(self.val, self.post)
        n_fit = method.v_fit_size_
        self.assertAlmostEqual(method.bandwidth_fit_, 1.0 / n_fit)
        deployed = method.deployed_for("vanilla_accuracy")
        self.assertEqual(deployed.bandwidth, 1.0 / n_fit)
        self.assertEqual(deployed.temperature, 2.0)
        self.assertEqual(method.metric_columns("vanilla_accuracy")["bandwidth_deployed"], 1.0 / n_fit)

    def test_reselect_the_bandwidth_on_v(self):
        method = _method(best=self.best, selects=True, bandwidth="reselect", truth=self.truth)
        method = method.set_dataset("toy").fit(self.val, self.post)
        deployed = method.deployed_for("vanilla_accuracy")
        self.assertIsNone(deployed.bandwidth)
        columns = method.metric_columns("vanilla_accuracy")
        self.assertAlmostEqual(columns["bandwidth_deployed"], 1.0 / len(self.val.y))
        self.assertAlmostEqual(columns["bandwidth_fit"], 1.0 / method.v_fit_size_)

    def test_fallback_when_v_cannot_be_split(self):
        val, post = _validation([40, 1, 30])
        val = _indexed(val)
        _FakeBase.log = []
        method = _method(selects=True, truth=_Truth(np.asarray(val.y), post.argmax(1), 3)).set_dataset("toy")
        method.fit(val, post)
        self.assertEqual(set(method.tau_cal_.values()), {1.0})
        self.assertEqual(set(method.fallback_.values()), {tcal.FALLBACK_SPLIT})
        # no calibration run; the deployed fit is the base itself (tau=1, bandwidth selected on the full V)
        self.assertEqual([e for e in _FakeBase.log if e[0] == "predict"], [])
        self.assertEqual(list(method.deployed()), [(1.0, False)])
        self.assertIsNone(method.deployed_for("macro-K").bandwidth)
        columns = method.metric_columns("macro-K")
        self.assertTrue(columns["tau_fallback"])
        self.assertEqual(columns["tau_fallback_reason"], "split")
        self.assertTrue(np.all(np.isnan(columns["cal_winkler"])))
        self.assertEqual(columns["v_fit_size"], 0)

    def test_fixed_tau_reuses_the_bandwidth_of_v_fit_without_calibration(self):
        _FakeBase.log = []
        method = _method(selects=True, truth=self.truth)
        method = tcal.CalibratedTemperature(
            method.base_factory, metrics=_metrics, fixed_tau=4.0, bandwidth="reuse"
        ).set_dataset("toy")
        method.fit(self.val, self.post)
        self.assertEqual([e for e in _FakeBase.log if e[0] == "predict"], [])
        n_fit = method.v_fit_size_
        self.assertEqual(set(method.tau_cal_.values()), {4.0})
        self.assertEqual(list(method.deployed()), [(4.0, True)])
        deployed = method.deployed_for("macro-F1")
        self.assertEqual((deployed.temperature, deployed.bandwidth), (4.0, 1.0 / n_fit))
        columns = method.metric_columns("macro-F1")
        self.assertEqual((columns["tau_cal"], columns["tau_fallback"]), (4.0, False))
        self.assertEqual((columns["cal_n_bags"], columns["cal_bag_size"]), (0, 0))
        self.assertAlmostEqual(columns["bandwidth_deployed"], 1.0 / n_fit)
        # the fits: one on V_fit at tau (the selection of the bandwidth), one on V at tau with that bandwidth
        fits = [e for e in _FakeBase.log if e[0] == "fit"]
        self.assertEqual(fits, [("fit", 4.0, None, n_fit), ("fit", 4.0, 1.0 / n_fit, len(self.val.y))])

    def test_fixed_tau_fallback_keeps_tau_and_selects_the_bandwidth_on_v(self):
        val, post = _validation([40, 1, 30])
        val = _indexed(val)
        method = _method(selects=True, truth=_Truth(np.asarray(val.y), post.argmax(1), 3))
        method = tcal.CalibratedTemperature(method.base_factory, metrics=_metrics, fixed_tau=2.0).set_dataset("toy")
        method.fit(val, post)
        self.assertEqual(set(method.tau_cal_.values()), {2.0})
        self.assertEqual(set(method.fallback_.values()), {tcal.FALLBACK_SPLIT})
        self.assertEqual(list(method.deployed()), [(2.0, False)])
        self.assertIsNone(method.deployed_for("vanilla_accuracy").bandwidth)

    def test_fixed_tau_needs_reuse(self):
        with self.assertRaises(ValueError):
            tcal.CalibratedTemperature(
                lambda temperature, bandwidth=None: None, _metrics, fixed_tau=2.0, bandwidth="reselect"
            )
        with self.assertRaises(ValueError):
            tcal.CalibratedTemperature(lambda temperature, bandwidth=None: None, _metrics, fixed_tau=0.0)

    def test_fallback_when_no_tau_has_a_score(self):
        class _Failed(_FakeBase):
            def ci_from_cts(self, cts):
                return _Interval([np.nan])

        method = tcal.CalibratedTemperature(
            lambda temperature, bandwidth=None: _Failed(temperature, bandwidth, truth=self.truth),
            metrics=_metrics,
            n_bags=3,
            cal_bag_size=20,
        )
        method.set_dataset("toy").fit(self.val, self.post)
        self.assertEqual(set(method.fallback_.values()), {tcal.FALLBACK_NO_SCORE})
        self.assertEqual(set(method.tau_cal_.values()), {1.0})


# --------------------------------------------------------------------------------------------------- factory


def _attributes(method) -> dict:
    return {k: v for k, v in vars(method).items() if not callable(v) or k == "acc_fn"}


class FactoryTest(unittest.TestCase):
    def setUp(self):
        _, self.acc = next(gen_acc_measure(True))

    def test_names_pool_and_labels(self):
        self.assertEqual(
            em.TCAL_VARIANTS,
            {
                "BayesCAP-tcal": "BayesCAP",
                "BayesCAP-real-tcal": "BayesCAP-real",
                "BayesCAP-Hf-real-tcal": "BayesCAP-Hf-real",
                "BayCAP-KDEy-h-inst-cb-tcal-keep": "BayCAP-KDEy-h-inst-cb",
                "BayCAP-KDEy-h-inst-cb-tcal-resel": "BayCAP-KDEy-h-inst-cb",
            },
        )
        self.assertEqual(set(em.TCAL_VARIANTS.values()), set(STORED_TCAL_BASES))
        self.assertTrue(set(em.TCAL_VARIANTS) <= em.BAYES_POOL_METHODS)
        self.assertTrue(set(em.TCAL_VARIANTS) <= set(em.get_method_names()))
        self.assertFalse(set(em.TCAL_VARIANTS) & set(TCAL_METHODS))
        methods = dict(em.gen_methods())
        for name in em.TCAL_VARIANTS:
            self.assertIsInstance(methods[name], tcal.CalibratedTemperature)
            self.assertEqual(methods[name].taus, tcal.TAUS)
            self.assertEqual(methods[name].cal_bag_size, 1000)
            self.assertIsNone(methods[name].fixed_tau)
            self.assertEqual(methods[name].bandwidth, em.TCAL_BANDWIDTHS[name])
        self.assertEqual(methods["BayCAP-KDEy-h-inst-cb-tcal-keep"].bandwidth, "reuse")
        # the two BayCAP bases with the uniform prior on C (added on 2026-11-25): population and realised pushforward
        self.assertEqual(methods["BayesCAP-tcal"].base_factory(temperature=2.0).estimand, "population")
        self.assertEqual(methods["BayesCAP-real-tcal"].base_factory(temperature=2.0).estimand, "realised")
        self.assertEqual(methods["BayCAP-KDEy-h-inst-cb-tcal-resel"].bandwidth, "reselect")

    def test_labels_and_colours(self):
        for name in (*em.TCAL_VARIANTS, *TCAL_METHODS):
            with self.subTest(method=name):
                self.assertIn(name, METHOD_LATEX_LABELS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_COLORS)
        for base in em.TABLE_TEMPERED_BASES:
            self.assertIn(base, tt.METHOD_LABELS)
        self.assertEqual(METHOD_PLOT_LABELS["BayesCAP-tcal"], "BayCAP-pop-tcal")
        self.assertEqual(METHOD_PLOT_LABELS["BayesCAP-real-tcal"], "BayCAP-sample-tcal")

    def test_default_methods_of_tables_and_plots(self):
        # user's decision of 2026-11-26: each BayCAP family is shown with its calibrated temperature
        from uncawe.plots import amplitude_vs_coverage, shift_profile
        from uncawe.plots.common import DEFAULT_METHODS
        from uncawe.tables import coverage_amplitude, error_split
        from uncawe.tables import main as main_tables

        defaults = ("BayesCAP-tcal", "BayesCAP-real-tcal", "BayesCAP-Hf-real-tcal", "BayCAP-KDEy-h-inst-cb-tcal-keep")
        self.assertEqual(main_tables.MAIN_METHODS[-4:], defaults)
        for untempered in ("BayesCAP", "BayesCAP-real", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst"):
            self.assertNotIn(untempered, main_tables.MAIN_METHODS)
            self.assertNotIn(untempered, DEFAULT_METHODS)
            self.assertNotIn(untempered, shift_profile.DEFAULT_METHODS)
        # every default is a stored method of main (no rerun)
        for name in defaults:
            self.assertIn(name, em.get_method_names())
        # the population method is left out where BayesCAP was
        self.assertEqual(main_tables.MAIN_METHODS_BEFORE_BAYESCAP, main_tables.MAIN_METHODS[:-4] + defaults[1:])
        self.assertEqual(amplitude_vs_coverage.PLOT_METHODS, list(main_tables.MAIN_METHODS_BEFORE_BAYESCAP))
        self.assertNotIn("BayesCAP-tcal", error_split.METHODS)
        self.assertEqual(error_split.METHODS, main_tables.MAIN_METHODS_BEFORE_BAYESCAP[1:])
        for name in error_split.METHODS:
            self.assertIn(name, error_split.SHORT_LABELS)
        self.assertIs(coverage_amplitude.main_tables, main_tables)
        self.assertIn("BayesCAP-tcal", DEFAULT_METHODS)
        self.assertEqual(shift_profile.DEFAULT_METHODS[-2:], defaults[2:])
        for name in shift_profile.DEFAULT_METHODS:
            self.assertIn(name, shift_profile.SHIFT_PROFILE_COLORS)

    def test_keep_grid(self):
        self.assertEqual(
            em.KEEP_VARIANTS,
            {
                "BayCAP-KDEy-h-inst-cb-keep": 1.0,
                "BayCAP-KDEy-h-inst-cb-keep-t0.5": 0.5,
                "BayCAP-KDEy-h-inst-cb-keep-t2": 2.0,
                "BayCAP-KDEy-h-inst-cb-keep-t4": 4.0,
                "BayCAP-KDEy-h-inst-cb-keep-t8": 8.0,
            },
        )
        self.assertTrue(set(em.KEEP_VARIANTS) <= em.BAYES_POOL_METHODS)
        self.assertEqual(em.TABLE_TEMPERED_BASES, (*em.TEMPERED_BASES, em.KEEP_BASE))
        methods = dict(em.gen_methods())
        for name, tau in em.KEEP_VARIANTS.items():
            method = methods[name]
            self.assertEqual((method.fixed_tau, method.taus, method.bandwidth), (tau, (tau,), "reuse"))
            # the base at tau is the tempered method of E5.2 (with its bandwidth fixed in the deployed fit)
            stored = dict(em.gen_methods())[em.KEEP_CALIBRATION_BASE] if tau == 1.0 else None
            if stored is None:
                stored = em.tempered_method(self.acc, f"{em.KEEP_CALIBRATION_BASE}-t{tau:g}")
            self.assertEqual(_attributes(method.base_factory(temperature=tau)), _attributes(stored))
            self.assertEqual(tt.method_name(em.KEEP_BASE, tau), name)

    def test_bases_are_the_tempered_methods_of_e52(self):
        for name, base in em.TCAL_VARIANTS.items():
            factory = em.tcal_method(self.acc, name).base_factory
            for tau in em.TEMPERATURE_LEVELS:
                with self.subTest(name=name, tau=tau):
                    built, stored = factory(temperature=tau), em.tempered_method(self.acc, f"{base}-t{tau:g}")
                    self.assertIs(type(built), type(stored))
                    self.assertEqual(_attributes(built), _attributes(stored))
            # tau=1 is the stored base of main
            reference = dict(em.gen_methods())[base]
            self.assertEqual(_attributes(factory(temperature=1.0)), _attributes(reference))

    def test_kdey_base_with_a_fixed_bandwidth(self):
        factory = em.tcal_method(self.acc, "BayCAP-KDEy-h-inst-cb-tcal-keep").base_factory
        selected, fixed = factory(temperature=2.0), factory(temperature=2.0, bandwidth=0.3)
        self.assertIsInstance(fixed, BayCAPKDEy)
        self.assertEqual(fixed.bandwidth, 0.3)
        a, b = _attributes(selected), _attributes(fixed)
        a.pop("bandwidth"), b.pop("bandwidth")
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            em.tcal_method(self.acc, "BayesCAP-Hf-real-tcal").base_factory(temperature=1.0, bandwidth=0.3)

    def test_prior(self):
        method = em.tcal_method(self.acc, "BayesCAP-Hf-real-tcal", prior=np.array([3.0, 3.0]))
        base = method.base_factory(temperature=2.0)
        self.assertIsInstance(base, BayCAP)
        np.testing.assert_array_equal(base.prior, [3.0, 3.0])

    def test_target_size_and_binform(self):
        # only the methods whose base is in E4.4 and E5.5 (not BayesCAP-tcal and BayesCAP-real-tcal, user's decision of
        # 2026-11-25)
        in_studies = set(em.TCAL_VARIANTS) - {"BayesCAP-tcal", "BayesCAP-real-tcal"}
        self.assertEqual(set(target_size.E44_TCAL_METHODS), in_studies)
        self.assertTrue(in_studies <= set(target_size.E44_METHODS))
        self.assertFalse({"BayesCAP", "BayesCAP-real"} & set(target_size.E44_METHODS))
        for name in in_studies:
            method = target_size.make_method(name, 300)
            self.assertEqual(method.cal_bag_size, 300)
        self.assertEqual(set(E55_TCAL_BASES), in_studies)
        self.assertFalse({"BayesCAP", "BayesCAP-real"} & {base for base, _ in E55_METHODS.values()})
        for base in E55_TCAL_BASES:
            for prior in ("uniform", "centre", "edge"):
                name = f"{base}_{prior}"
                self.assertEqual(E55_METHODS[name], (base, prior))
                method = e55_method(name, 3)
                self.assertIsInstance(method, tcal.CalibratedTemperature)
                self.assertEqual(method.cal_bag_size, 1000)
                self.assertEqual(method.bandwidth, em.TCAL_BANDWIDTHS[base])
        order = [E55_METHODS[n][0] for n in sorted(E55_METHODS, key=_e55_method_order)]
        self.assertLess(order.index(E55_TCAL_BASES[0]), order.index("SQS-H"))


# --------------------------------------------------------------------------------------------------- tables of -keep


class KeepTablesTest(unittest.TestCase):
    """The tables of E5.2 and E5.3 for the -keep variant: its test runs are the grid of -keep in main, its calibration
    runs those of BayCAP-KDEy-h-inst-cb in tau_select."""

    @staticmethod
    def _store(root, base, method_base, experiment, calibration):
        from test_tau_select import _frames

        for tau, df in _frames(base, calibration).items():
            for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                folder = os.path.join(root, experiment, "classic", "vanilla_accuracy", dataset, classifier)
                os.makedirs(folder, exist_ok=True)
                name = tt.method_name(method_base, tau)
                part = part.assign(
                    method=name, true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                )
                part.to_parquet(os.path.join(folder, f"{name}.parquet"))

    def test_keep_reads_the_calibration_of_cb(self):
        from uncawe.tables import tau_select as tables

        cb, keep = em.KEEP_CALIBRATION_BASE, em.KEEP_BASE
        with tempfile.TemporaryDirectory() as root:
            self._store(root, cb, cb, ts.EXPERIMENT, True)
            self._store(root, cb, cb, "main", False)
            self._store(root, cb, keep, "main", False)
            out = tables.compute(["vanilla_accuracy"], root=root)
        pairs = out["pairs"].set_index(["method", "dataset", "classifier"])
        self.assertEqual(list(dict.fromkeys(out["pairs"]["method"])), [cb, keep])
        np.testing.assert_array_equal(pairs.loc[keep, "tau_cal"].to_numpy(), pairs.loc[cb, "tau_cal"].to_numpy())
        self.assertEqual(list(dict.fromkeys(out["calibration"]["method"])), [cb, keep])
        self.assertIn(keep, tt.METHOD_LABELS)


# --------------------------------------------------------------------------------------------------- evaluation


class _Protocol:
    def __init__(self, samples):
        self.samples = samples

    def __call__(self):
        return iter(self.samples)

    def total(self):
        return len(self.samples)


class _Deployed:
    def __init__(self, tau):
        self.tau, self.predict_calls, self.acc_fn = tau, 0, None

    def predict_ct_range(self, X, posteriors):
        self.predict_calls += 1
        return np.array([np.eye(2)])

    def switch(self, acc_fn):
        self.acc_fn = acc_fn

    def ci_from_cts(self, cts):
        return _Interval([self.tau])


class EvaluationTest(unittest.TestCase):
    def _method(self):
        method = tcal.CalibratedTemperature(lambda temperature, bandwidth=None: None, metrics=gen_acc_measure)
        method.dataset = "dataset"
        method.metric_names_ = get_acc_names()
        method.tau_cal_ = {"vanilla_accuracy": 2.0, "macro-F1": 0.5, "macro-K": 2.0}
        method.fallback_ = {name: "" for name in get_acc_names()}
        method.bandwidth_fit_ = np.nan
        method.cal_scores_ = {name: {t: 1.0 for t in tcal.TAUS} for name in get_acc_names()}
        method.v_fit_size_ = method.v_cal_size_ = 10
        method.t_calibration_ = 0.0
        method.deployed_ = {(2.0, False): _Deployed(2.0), (0.5, False): _Deployed(0.5)}
        method.bandwidth_deployed_ = {key: np.nan for key in method.deployed_}
        return method

    def test_runs_per_deployed_fit(self):
        method = self._method()
        runs = bag_runs(method, lambda: iter([(None, None)] * 3), 3, lambda m: None)
        self.assertEqual(set(runs), {(2.0, False), (0.5, False)})
        self.assertEqual([d.predict_calls for d in method.deployed_.values()], [3, 3])
        self.assertIs(metric_run(method, runs, "macro-F1").method, method.deployed_[(0.5, False)])
        columns = metric_columns(method, "macro-K", 3)
        self.assertEqual(columns["tau_cal"], [2.0] * 3)
        self.assertEqual(metric_columns(object(), "macro-K", 3), {})

    @patch("uncawe.experiments.main._result_path", return_value="result.parquet")
    @patch("uncawe.experiments.main.os.path.exists", return_value=False)
    @patch("uncawe.experiments.main.build_result_frame", return_value=Mock())
    def test_main_scores_each_metric_on_its_own_fit(self, build, _exists, _path):
        method = self._method()
        method.fit = Mock(return_value=method)
        pretrain = SimpleNamespace(
            d_info=SimpleNamespace(name="dataset", collection="uci", n_classes=3),
            h_info=SimpleNamespace(name="LR", full_name="LR"),
        )
        samples = [SimpleNamespace(X=np.zeros((1, 1))) for _ in range(2)]
        dataset = SimpleNamespace(test_prot=_Protocol(samples), test_prot_posteriors=[np.ones((1, 2))] * 2)
        context = EvaluationContext(
            pretrain=pretrain, shifts=[0.1, 0.2], true_cts=[np.eye(2)] * 2, train_prev=0.5, val_prev=0.5
        )
        results = main._evaluate_contingency_method(pretrain, "m", method, dataset, Mock(), Mock(), context)
        self.assertEqual(len(results), 3)
        method.fit.assert_called_once()
        self.assertEqual(method.dataset, "dataset")
        self.assertEqual([d.predict_calls for d in method.deployed_.values()], [2, 2])
        by_metric = {call.kwargs["acc_name"]: call.kwargs for call in build.call_args_list}
        self.assertEqual([s[0] for s in by_metric["macro-F1"]["estim_acc_ranges"]], [0.5, 0.5])
        self.assertEqual([s[0] for s in by_metric["vanilla_accuracy"]["estim_acc_ranges"]], [2.0, 2.0])
        self.assertEqual(by_metric["macro-F1"]["extra_columns"]["tau_cal"], [0.5, 0.5])
        self.assertEqual(by_metric["macro-K"]["extra_columns"]["tau_cal_macro-F1"], [0.5, 0.5])


# --------------------------------------------------------------------------------------------------- NUTS


class NutsReproducibilityTest(unittest.TestCase):
    """With a real BayCAP-Hf (short chains): the calibration scores are those of a base fitted on $V_{\\text{fit}}$ and
    run on the bags of ``tau_select``, as ``experiments/tau_select.py`` does, and the deployed fit is a fresh base at
    $\\tau_{\\text{cal}}$ fitted on the full $V$ (the method of E5.2)."""

    def test_calibration_and_deployed_fit(self):
        _, acc = next(gen_acc_measure(False))

        def factory(temperature, bandwidth=None):
            return BayCAP(
                acc, estimand="realised", c_prior="hierarchical-f", num_warmup=30, num_samples=40,
                temperature=temperature,
            )  # fmt: skip

        val, post = _validation([60, 40], seed=4)
        taus = (1.0, 4.0)
        method = tcal.CalibratedTemperature(factory, gen_acc_measure, taus=taus, n_bags=3, cal_bag_size=30)
        method.set_dataset("toy").fit(val, post)
        y, pred = np.asarray(val.y), post.argmax(axis=1)
        fit_index, cal_index = ts.split_index(y, 2, "toy")
        bags = ts.calibration_bags("toy", y, pred, cal_index, 2, n_bags=3, size=30)
        v_fit = LabelledCollection(val.X[fit_index], y[fit_index], classes=val.classes_)
        for tau in taus:
            base = factory(tau).fit(v_fit, post[fit_index])
            cts = [base.predict_ct_range(val.X[b.index], post[b.index]) for b in bags]
            for name, fn in gen_acc_measure(False):
                base.switch(fn)
                values = [tcal.bag_winkler(base.ci_from_cts(ct).samples, fn(b.true_ct)) for ct, b in zip(cts, bags)]
                expected = pd.Series(values).mean()
                self.assertAlmostEqual(method.cal_scores_[name][tau], expected, msg=(name, tau))
        tau = method.tau_cal_["vanilla_accuracy"]
        fresh = factory(tau).fit(val, post)
        deployed = method.deployed_for("vanilla_accuracy")
        np.testing.assert_array_equal(
            deployed.predict_ct_range(val.X[:30], post[:30]), fresh.predict_ct_range(val.X[:30], post[:30])
        )


if __name__ == "__main__":
    unittest.main()
