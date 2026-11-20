"""Tests of AC (methods/ac.py, E8.1 of docs/ext8_average_confidence.md, card WP-29) and of its E8.1 tables."""

import os

# before any import of JAX, as the other test modules: this module sorts first in a discovery, and on a machine with a
# GPU JAX would otherwise start on it, where the NUTS draws of the later test modules are not reproducible
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import itertools
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from cap.calib import LasCal
from cap.error import vanilla_acc
from cap.models.confidence import CBPE, PoissonConfidenceInterval
from quapy.data import LabelledCollection
from scipy.special import softmax

from uncawe.experiments import main
from uncawe.experiments.methods import AC_METHOD, gen_methods, get_method_names
from uncawe.experiments.metrics import get_selection_acc
from uncawe.experiments.records import EvaluationContext
from uncawe.methods import ac as ac_module
from uncawe.methods.ac import (
    AC,
    CentralPoissonInterval,
    central_bounds,
    lascal_calibrate,
    poisson_binomial_pmf,
    recover_temperature,
)
from uncawe.tables import average_confidence as tables
from uncawe.tables import main as main_tables
from uncawe.tables import rank


def _posteriors(rng: np.random.Generator, y: np.ndarray, n_classes: int, signal: float = 2.0) -> np.ndarray:
    """Posteriors of a noisy classifier: a softmax of the one-hot label times ``signal`` plus Gaussian noise."""
    logits = signal * np.eye(n_classes)[y] + rng.normal(scale=1.5, size=(len(y), n_classes))
    return softmax(logits, axis=1)


def _data(n_classes: int, seed: int = 0, n_val: int = 400, n_bag: int = 300):
    """A validation set with uniform prevalence and a bag with the first class at prevalence 0.7 (seeded)."""
    rng = np.random.default_rng(seed)
    y_val = rng.integers(0, n_classes, n_val)
    bag_prev = np.full(n_classes, 0.3 / (n_classes - 1))
    bag_prev[0] = 0.7
    y_bag = rng.choice(n_classes, size=n_bag, p=bag_prev)
    val_post = _posteriors(rng, y_val, n_classes)
    bag_post = _posteriors(rng, y_bag, n_classes)
    val = LabelledCollection(np.zeros((n_val, 1)), y_val, classes=np.arange(n_classes))
    return val, val_post, bag_post, y_bag


def _brute_force_pmf(s: np.ndarray) -> np.ndarray:
    """$P(K=k)=\\sum_{|A|=k}\\prod_{i\\in A}s_i\\prod_{j\\notin A}(1-s_j)$ by enumerating the $2^n$ outcomes."""
    pmf = np.zeros(len(s) + 1)
    for outcome in itertools.product((0, 1), repeat=len(s)):
        outcome = np.asarray(outcome)
        pmf[outcome.sum()] += np.prod(np.where(outcome == 1, s, 1 - s))
    return pmf


class CentralIntervalTest(unittest.TestCase):
    def test_pmf_matches_brute_force(self):
        s = np.random.default_rng(1).uniform(0.05, 0.99, size=8)
        np.testing.assert_allclose(poisson_binomial_pmf(s), _brute_force_pmf(s), atol=1e-12)

    def test_bounds_are_the_quantiles_of_the_brute_force_distribution(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            s = rng.uniform(0.3, 1.0, size=10)
            pmf = _brute_force_pmf(s)
            cdf = np.cumsum(pmf)
            for alpha in (0.05, 0.1, 0.3):
                k_low = min(k for k in range(11) if cdf[k] >= alpha / 2 - 1e-12)
                k_up = min(k for k in range(11) if cdf[k] >= 1 - alpha / 2 - 1e-12)
                interval = CentralPoissonInterval(poisson_binomial_pmf(s), confidence_level=1 - alpha)
                self.assertEqual((interval.low, interval.high), (k_low / 10, k_up / 10))
                # the interval holds at least 1 - alpha of the probability
                self.assertGreaterEqual(pmf[k_low : k_up + 1].sum(), 1 - alpha - 1e-12)

    def test_point_estimate_is_the_mean_confidence(self):
        s = np.random.default_rng(3).uniform(0.5, 1.0, size=500)
        interval = CentralPoissonInterval(poisson_binomial_pmf(s))
        self.assertAlmostEqual(interval.point_estimate, s.mean(), places=10)
        np.testing.assert_allclose(interval.samples, poisson_binomial_pmf(s))

    def test_coverage_on_confidences_calibrated_by_construction(self):
        # labels drawn from Bernoulli(s_i): the realised accuracy follows the Poisson-binomial distribution exactly, so
        # the central interval covers it with probability at least 0.95 (slightly more, since K is discrete)
        rng = np.random.default_rng(4)
        s = rng.beta(8, 2, size=400)
        interval = CentralPoissonInterval(poisson_binomial_pmf(s))
        accs = (rng.uniform(size=(4000, len(s))) < s).mean(axis=1)
        coverage = np.mean([interval.coverage(a) for a in accs])
        self.assertGreater(coverage, 0.94)
        self.assertLess(coverage, 0.97)

    def test_scalar_nan_gives_nan(self):
        for value in (np.nan, float("nan"), np.float64("nan")):
            interval = CentralPoissonInterval(value)
            self.assertTrue(np.isnan(interval.point_estimate))
            self.assertTrue(np.isnan(interval.amplitude()))
            self.assertTrue(np.isnan(interval.coverage(0.5)))

    def test_central_bounds_tolerate_rounding_of_the_fft(self):
        pmf = np.array([-1e-17, 0.5, 0.5 + 1e-15])
        self.assertEqual(central_bounds(pmf, 0.05), (1, 2))


class LasCalTest(unittest.TestCase):
    def test_binary_ac_equals_l_cbpe(self):
        val, val_post, bag_post, _ = _data(2)
        ac = AC(confidence_level=0.95).switch_and_fit(vanilla_acc, val, val_post)
        cbpe = CBPE("vanilla_accuracy", calib_method="lascal").fit(val, val_post)

        confidences = ac.calibrated_confidences(bag_post)
        np.testing.assert_array_equal(confidences, LasCal()(val_post, val.y)(bag_post).max(axis=1))
        pmf, cbpe_pmf = ac.predict_range(None, bag_post), cbpe.predict_range(None, bag_post)
        np.testing.assert_array_equal(pmf, cbpe_pmf)
        self.assertAlmostEqual(ac.predict(None, bag_post), cbpe.predict(None, bag_post), places=12)
        self.assertFalse(ac.last_lascal_columns()["lascal_fallback"])

    def test_temperature_keeps_the_argmax(self):
        val, val_post, bag_post, _ = _data(3, seed=5)
        calibrated, temperature, fallback = lascal_calibrate(val_post, val.y, bag_post)
        self.assertFalse(fallback)
        self.assertTrue(0.1 - 1e-6 <= temperature <= 20.0 + 1e-6)
        np.testing.assert_array_equal(calibrated.argmax(axis=1), bag_post.argmax(axis=1))
        np.testing.assert_allclose(calibrated, softmax(np.log(bag_post) / temperature, axis=1), atol=1e-6)
        # and for any temperature of the grid
        for theta in (0.1, 0.5, 3.0, 20.0):
            scaled = softmax(np.log(bag_post) / theta, axis=1)
            np.testing.assert_array_equal(scaled.argmax(axis=1), bag_post.argmax(axis=1))

    def test_lascal_is_deterministic(self):
        val, val_post, bag_post, _ = _data(3, seed=6)
        first, second = lascal_calibrate(val_post, val.y, bag_post), lascal_calibrate(val_post, val.y, bag_post)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1:], second[1:])

    def test_fallback_on_an_exception(self):
        val, val_post, bag_post, _ = _data(3, seed=7)
        ac = AC().fit(val, val_post)
        with patch.object(ac_module.Calibrator, "calibrate", side_effect=RuntimeError("lascal failed")):
            confidences = ac.calibrated_confidences(bag_post)
        np.testing.assert_array_equal(confidences, bag_post.max(axis=1))
        columns = ac.last_lascal_columns()
        self.assertTrue(columns["lascal_fallback"])
        self.assertTrue(np.isnan(columns["lascal_temperature"]))

    def test_fallback_when_no_temperature_is_selected(self):
        # every loss NaN: lascal keeps its initial temperature -1 and returns the logits divided by -1
        val, val_post, bag_post, _ = _data(3, seed=8)

        def no_temperature(method_name, source_agg, target_agg, train_agg):
            return {"target": {"y_logits": target_agg["y_logits"] / -1, "y_true": None}}

        with patch.object(ac_module.Calibrator, "calibrate", side_effect=no_temperature):
            calibrated, temperature, fallback = lascal_calibrate(val_post, val.y, bag_post)
        self.assertTrue(fallback)
        self.assertTrue(np.isnan(temperature))
        np.testing.assert_array_equal(calibrated, bag_post)

    def test_recover_temperature(self):
        logits = np.log(np.array([[0.2, 0.8], [1.0, 1e-300], [0.5, 0.5]]))
        logits[1, 1] = -np.inf
        self.assertAlmostEqual(recover_temperature(logits, logits / 2.5), 2.5)
        self.assertTrue(np.isnan(recover_temperature(logits, logits * np.array([1.0, 2.0]))))
        self.assertTrue(np.isnan(recover_temperature(np.zeros((2, 2)), np.zeros((2, 2)))))


class ACMetricTest(unittest.TestCase):
    def test_other_metrics_are_nan(self):
        val, val_post, bag_post, _ = _data(3, seed=9)
        ac = AC()
        for name in ("macro-F1", "macro-K"):
            ac.switch_and_fit(get_selection_acc(name, True), val, val_post)
            self.assertIsNone(ac.acc_name)
            self.assertTrue(np.isnan(ac.predict_range(None, bag_post)))
            self.assertIsNone(ac.last_lascal_columns())
            self.assertTrue(np.isnan(ac.ci_from_accs(np.nan).point_estimate))
        ac.switch_and_fit(get_selection_acc("vanilla_accuracy", True), val, val_post)
        self.assertEqual(ac.acc_name, "vanilla_accuracy")


class FactoryTest(unittest.TestCase):
    def test_ac_is_in_main_and_out_of_the_main_tables_and_ranks(self):
        self.assertIn(AC_METHOD, get_method_names())
        methods = dict(gen_methods())
        self.assertIsInstance(methods[AC_METHOD], AC)
        self.assertNotIn(AC_METHOD, main_tables.MAIN_METHODS)
        self.assertIn(AC_METHOD, main_tables.EXCLUDED_METHODS)
        self.assertIn(AC_METHOD, rank.EXCLUDED_METHODS)
        self.assertNotIn(AC_METHOD, rank._methods())


class MainProtocolTest(unittest.TestCase):
    @patch("uncawe.experiments.main._result_path", return_value="result.parquet")
    @patch("uncawe.experiments.main.os.path.exists", return_value=False)
    @patch("uncawe.experiments.main.build_result_frame", return_value=Mock())
    def test_lascal_columns_are_stored_for_vanilla_accuracy_only(self, build, _exists, _path):
        pretrain = SimpleNamespace(
            d_info=SimpleNamespace(name="dataset", collection="uci", n_classes=2),
            h_info=SimpleNamespace(name="LR", full_name="LR"),
        )
        bag_posteriors = [np.array([[0.2, 0.8], [0.6, 0.4]]), np.array([[0.9, 0.1], [0.3, 0.7]])]
        samples = [SimpleNamespace(X=np.zeros((2, 1))) for _ in bag_posteriors]
        test_prot = Mock(side_effect=lambda: iter(samples))
        test_prot.total.return_value = len(samples)
        dataset = SimpleNamespace(test_prot=test_prot, test_prot_posteriors=bag_posteriors)
        context = EvaluationContext(
            pretrain=pretrain, shifts=[0.1, 0.2], true_cts=[np.eye(2), np.eye(2)], train_prev=0.5, val_prev=0.5
        )
        val = LabelledCollection(np.zeros((4, 1)), np.array([0, 1, 0, 1]), classes=np.arange(2))
        temperatures = iter([1.5, 2.5])

        def fake_lascal(val_posteriors, val_y, posteriors):
            return posteriors, next(temperatures), False

        with patch("uncawe.methods.ac.lascal_calibrate", side_effect=fake_lascal):
            main._evaluate_direct_method(pretrain, AC_METHOD, AC(), dataset, val, np.full((4, 2), 0.5), context)

        self.assertEqual(build.call_count, 3)
        by_acc = {call.kwargs["acc_name"]: call.kwargs for call in build.call_args_list}
        self.assertEqual(
            by_acc["vanilla_accuracy"]["extra_columns"],
            {"lascal_temperature": [1.5, 2.5], "lascal_fallback": [False, False]},
        )
        np.testing.assert_allclose(by_acc["vanilla_accuracy"]["estim_accs"], [0.7, 0.8])
        for name in ("macro-F1", "macro-K"):
            self.assertIsNone(by_acc[name]["extra_columns"])
            self.assertTrue(np.isnan(by_acc[name]["estim_accs"]).all())


def _result_rows(method: str, dataset: str, n_classes: int, pmfs: list, true_accs: list, lascal: bool) -> pd.DataFrame:
    """Stored rows of one pair: binary true tables (cap-ml layout) with the given accuracies, 10 instances each."""
    rows = []
    for index, (pmf, acc) in enumerate(zip(pmfs, true_accs)):
        ct = np.zeros((n_classes, n_classes))
        ct[0, 0], ct[0, 1] = acc, 1 - acc
        estimate = np.nan if main_tables._no_estimate(pmf) else CentralPoissonInterval(pmf).point_estimate
        row = {
            "dataset": dataset,
            "classifier": "LR",
            "method": method,
            "n_classes": n_classes,
            "true_cts": ct,
            "estim_acc_ranges": pmf,
            "estim_accs": estimate,
            "acc_err": abs(estimate - acc) if np.isfinite(estimate) else np.nan,
        }
        if lascal:
            row |= {"lascal_temperature": 1.0 + index, "lascal_fallback": index == 0}
        rows.append(row)
    return pd.DataFrame(rows)


class TablesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.pmfs = [poisson_binomial_pmf(rng.uniform(0.6, 1.0, size=10)) for _ in range(3)]
        self.true_accs = [0.8, 0.9, 0.5]
        self.results = {
            AC_METHOD: pd.concat(
                [
                    _result_rows(AC_METHOD, "binary", 2, self.pmfs, self.true_accs, lascal=True),
                    _result_rows(AC_METHOD, "multi", 3, self.pmfs, self.true_accs, lascal=True),
                ],
                ignore_index=True,
            ),
            tables.L_CBPE: pd.concat(
                [
                    _result_rows(tables.L_CBPE, "binary", 2, self.pmfs, self.true_accs, lascal=False),
                    _result_rows(tables.L_CBPE, "multi", 3, [np.nan] * 3, self.true_accs, lascal=False),
                ],
                ignore_index=True,
            ),
        }

    def test_scores_read_the_interval_of_each_method(self):
        pmf, acc = self.pmfs[0], 0.8
        central, hdi = CentralPoissonInterval(pmf), PoissonConfidenceInterval(pmf)
        self.assertEqual(main_tables.get_acc_amplitude(AC_METHOD, pmf), central.amplitude())
        self.assertEqual(main_tables.get_acc_amplitude(tables.L_CBPE, pmf), hdi.amplitude())
        self.assertEqual(main_tables.get_acc_winkler(AC_METHOD, acc, pmf), central.winkler(acc))
        self.assertEqual(main_tables.get_acc_coverage(tables.L_CBPE, acc, pmf), hdi.coverage(acc))
        for method in (AC_METHOD, tables.L_CBPE):
            self.assertTrue(np.isnan(main_tables.get_acc_coverage(method, acc, np.nan)))
            self.assertTrue(np.isnan(main_tables.get_acc_amplitude(method, np.nan)))
            self.assertTrue(np.isnan(main_tables.get_acc_winkler(method, acc, np.nan)))
        # other methods keep the percentiles of their samples
        samples = np.linspace(0.0, 1.0, 1001)
        self.assertAlmostEqual(main_tables.get_acc_amplitude("SQS-H", samples), 0.95)

    def test_pair_tables(self):
        comparison, check = tables.pair_tables(self.results)
        self.assertEqual(comparison["dataset"].tolist(), ["binary", "multi"])
        binary, multi = comparison.iloc[0], comparison.iloc[1]
        expected = np.mean([CentralPoissonInterval(p).winkler(a) for p, a in zip(self.pmfs, self.true_accs)])
        self.assertAlmostEqual(binary[f"{AC_METHOD}_winkler"], expected)
        expected = np.mean([PoissonConfidenceInterval(p).amplitude() for p in self.pmfs])
        self.assertAlmostEqual(binary[f"{tables.L_CBPE}_amplitude"], expected)
        self.assertTrue(np.isnan(multi[f"{tables.L_CBPE}_ae"]))
        self.assertFalse(np.isnan(multi[f"{AC_METHOD}_ae"]))

        self.assertEqual(check.iloc[0]["max_abs_point_diff"], 0.0)
        low_diff = np.mean([CentralPoissonInterval(p).low - PoissonConfidenceInterval(p).low for p in self.pmfs])
        self.assertAlmostEqual(check.iloc[0]["mean_low_diff"], low_diff)
        self.assertTrue(np.isnan(check.iloc[1]["max_abs_point_diff"]))
        self.assertEqual(check.iloc[1]["n_fallback"], 1)
        self.assertEqual((check.iloc[1]["theta_min"], check.iloc[1]["theta_max"]), (2.0, 3.0))

        summary = tables.summary_rows(comparison)
        self.assertEqual(summary["n_pairs"].tolist(), [1, 2])
        self.assertAlmostEqual(summary.iloc[0][f"{tables.L_CBPE}_ae"], binary[f"{tables.L_CBPE}_ae"])
        self.assertTrue(np.isnan(summary.iloc[1][f"{tables.L_CBPE}_ae"]))
        self.assertAlmostEqual(summary.iloc[1][f"{AC_METHOD}_ae"], comparison[f"{AC_METHOD}_ae"].mean())

    def test_unpaired_bags_raise(self):
        self.results[tables.L_CBPE].loc[0, "true_cts"][0, 0] = 0.1
        with self.assertRaises(ValueError):
            tables.pair_tables(self.results)

    def test_write_tables(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tables.write_tables(self.results, out_dir)
            for name in (
                "average_confidence.csv",
                "average_confidence_check.csv",
                "average_confidence_doc.tex",
                "tabular/average_confidence.tex",
                "tabular/average_confidence_check_long.tex",
            ):
                with open(f"{out_dir}/{name}") as f:
                    self.assertTrue(f.read())
            csv = pd.read_csv(f"{out_dir}/average_confidence.csv")
            self.assertEqual(csv["dataset"].tolist()[-2:], [tables.SUMMARY_BOTH, tables.SUMMARY_AC])


if __name__ == "__main__":
    unittest.main()
