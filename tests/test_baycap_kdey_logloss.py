"""Tests of the bandwidth selected by the leave-one-out log-loss of the kernel class probabilities (work-plan card WP-33,
E3.5 of ``docs/ext3_posterior_propagation.md``, §T.16)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from cap.error import vanilla_acc
from scipy.special import logsumexp
from test_baycap_fixed import DATA, SETTINGS

from uncawe.experiments import methods as experiment_methods
from uncawe.methods import baycap_kdey
from uncawe.methods.baycap_kdey import INSTANCE, INSTANCE_SUMMARIES, BayCAPKDEy
from uncawe.methods.baycap_quantifier import FIXED_SUMMARIES
from uncawe.methods.bayes_kdey import (
    LOGLOSS,
    LOGLOSS_SUMMARIES,
    LOO_GRID,
    BayesKDEy,
    class_logloss_curve,
    loo_class_log_probabilities,
    loo_confidence_ece,
    select_from_curve,
)
from uncawe.methods.bayesian_quantifier import confidence_ece, top_label_ece
from uncawe.plots import shift_profile
from uncawe.plots.common import METHOD_COLORS, method_plot_label
from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS

CB = "BayCAP-KDEy-h-inst-cb"
INST = "BayCAP-KDEy-h-inst"


def _simplex_points(rng, n_per_class, concentration=2.0, shift=1.5):
    """Outputs of a noisy classifier on the simplex: the logits of the true class raised by ``shift``."""
    n_classes = len(n_per_class)
    labels = np.repeat(np.arange(n_classes), n_per_class)
    logits = rng.normal(0.0, concentration, size=(len(labels), n_classes))
    logits[np.arange(len(labels)), labels] += shift
    points = np.exp(logits - logits.max(axis=1, keepdims=True))
    return points / points.sum(axis=1, keepdims=True), labels


def _bayes_rule(points, labels, classes, u, b):
    """The class probabilities of instance u from the prevalence and the Gaussian kernel densities of V without u."""
    others = np.arange(len(points)) != u
    rest, rest_labels = points[others], labels[others]
    dim = points.shape[1]
    terms = np.zeros(len(classes))
    for j, class_ in enumerate(classes):
        members = rest[rest_labels == class_]
        if len(members) == 0:
            continue
        prevalence = len(members) / len(rest)
        kernel = np.exp(-np.sum((points[u] - members) ** 2, axis=1) / (2 * b * b)) / (2 * np.pi * b * b) ** (dim / 2)
        terms[j] = prevalence * kernel.mean()
    return terms / terms.sum()


class LeaveOneOutProbabilitiesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.points, self.labels = _simplex_points(rng, (9, 6, 5))
        self.classes = np.arange(3)

    def test_bayes_rule_without_the_instance(self):
        bandwidths = (0.15, 0.4)
        log_w = loo_class_log_probabilities(self.points, self.labels, self.classes, bandwidths)
        self.assertEqual(log_w.shape, (2, len(self.points), 3))
        for g, b in enumerate(bandwidths):
            for u in range(len(self.points)):
                expected = _bayes_rule(self.points, self.labels, self.classes, u, b)
                np.testing.assert_allclose(np.exp(log_w[g, u]), expected, rtol=1e-10, atol=1e-14)

    def test_brute_force_loop_and_blocks(self):
        b = 0.2
        log_w = loo_class_log_probabilities(self.points, self.labels, self.classes, [b])[0]
        # one row per block gives the same values
        blocked = loo_class_log_probabilities(self.points, self.labels, self.classes, [b], block_entries=1)[0]
        np.testing.assert_allclose(blocked, log_w, rtol=1e-12)
        for u in range(len(self.points)):
            kernel = np.exp(-np.sum((self.points[u] - self.points) ** 2, axis=1) / (2 * b * b))
            kernel[u] = 0.0
            expected = np.array([kernel[self.labels == j].sum() for j in self.classes]) / kernel.sum()
            np.testing.assert_allclose(np.exp(log_w[u]), expected, rtol=1e-10, atol=1e-14)

    def test_labels_follow_the_order_of_classes(self):
        labels = np.array(["b", "a", "c"])[self.labels]
        classes = np.array(["b", "a", "c"])
        np.testing.assert_allclose(
            loo_class_log_probabilities(self.points, labels, classes, [0.3]),
            loo_class_log_probabilities(self.points, self.labels, self.classes, [0.3]),
        )
        with self.assertRaises(ValueError):
            loo_class_log_probabilities(self.points, self.labels, np.arange(2), [0.3])

    def test_finite_where_a_direct_computation_underflows(self):
        # two well separated pairs: at b = 0.01 every kernel weight between the pairs is exp(-~2500) = 0
        points = np.array([[0.9, 0.1], [0.88, 0.12], [0.2, 0.8], [0.25, 0.75]])
        labels = np.array([0, 1, 0, 1])
        b = LOO_GRID[0]
        sq = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
        np.fill_diagonal(sq, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            direct = np.log(np.exp(-sq / (2 * b * b))[:, labels == 0].sum(axis=1) / np.exp(-sq / (2 * b * b)).sum(1))
        self.assertFalse(np.all(np.isfinite(direct)))
        log_w = loo_class_log_probabilities(points, labels, np.arange(2), [b])[0]
        self.assertTrue(np.all(np.isfinite(log_w)))
        # the reference in logarithms
        for u in range(4):
            sums = [logsumexp(-sq[u, labels == j] / (2 * b * b)) for j in range(2)]
            np.testing.assert_allclose(log_w[u], np.array(sums) - logsumexp(sums), rtol=1e-12)
        curve, _ = class_logloss_curve(points, labels, np.arange(2), LOO_GRID)
        self.assertTrue(np.all(np.isfinite(curve)))
        # the nearest neighbour of every instance is of the other class: a huge but finite loss at the smallest b
        self.assertGreater(curve[0], 100.0)


class LogLossCurveTest(unittest.TestCase):
    def test_class_balanced_average(self):
        rng = np.random.default_rng(3)
        points, labels = _simplex_points(rng, (30, 4, 8))
        bandwidths = (0.05, 0.2, 0.6)
        curve, used = class_logloss_curve(points, labels, np.arange(3), bandwidths)
        np.testing.assert_array_equal(used, [True, True, True])
        log_w = loo_class_log_probabilities(points, labels, np.arange(3), bandwidths)
        own = log_w[:, np.arange(len(labels)), labels]
        expected = -np.mean([own[:, labels == j].mean(axis=1) for j in range(3)], axis=0)
        np.testing.assert_allclose(curve, expected, rtol=1e-12)
        # not the plain average over V, which weights the classes by their sizes
        self.assertFalse(np.allclose(curve, -own.mean(axis=1)))

    def test_a_class_with_one_instance_stays_in_the_denominators(self):
        rng = np.random.default_rng(4)
        points, labels = _simplex_points(rng, (10, 1, 7))
        b = 0.3
        curve, used = class_logloss_curve(points, labels, np.arange(3), [b])
        np.testing.assert_array_equal(used, [True, False, True])
        own = [
            np.log(_bayes_rule(points, labels, np.arange(3), u, b)[labels[u]])
            for u in range(len(labels))
            if labels[u] != 1
        ]
        kept = labels[labels != 1]
        expected = -np.mean([np.mean(np.asarray(own)[kept == j]) for j in (0, 2)])
        np.testing.assert_allclose(curve[0], expected, rtol=1e-12)
        # the singleton changes the curve through the denominators of the other classes
        without, _ = class_logloss_curve(points[labels != 1], labels[labels != 1], np.arange(3), [b])
        self.assertNotAlmostEqual(curve[0], without[0], places=8)

    def test_no_usable_class(self):
        with self.assertRaises(ValueError):
            class_logloss_curve(np.eye(3), np.arange(3), np.arange(3), [0.1])

    def test_overlapping_classes_minimum_inside_and_better_calibration(self):
        rng = np.random.default_rng(7)
        # two overlapping classes: the outputs (s, 1 - s) with s from two overlapping Beta distributions
        s = np.concatenate([rng.beta(2, 5, size=200), rng.beta(5, 2, size=200)])
        points = np.column_stack([1 - s, s])
        labels = np.repeat([0, 1], 200)
        curve, _ = class_logloss_curve(points, labels, np.arange(2), LOO_GRID)
        best = select_from_curve(LOO_GRID, curve)
        self.assertTrue(0 < best < len(LOO_GRID) - 1)
        ece = loo_confidence_ece(points, labels, np.arange(2), [LOO_GRID[0], LOO_GRID[best]])
        self.assertLess(ece[1], ece[0])

    def test_confidence_ece_of_the_predicted_class(self):
        rng = np.random.default_rng(8)
        points, labels = _simplex_points(rng, (15, 12, 10))
        b = 0.25
        log_w = loo_class_log_probabilities(points, labels, np.arange(3), [b])[0]
        predicted = np.argmax(points, axis=1)
        expected = confidence_ece(np.exp(log_w[np.arange(len(labels)), predicted]), predicted == labels)
        self.assertAlmostEqual(loo_confidence_ece(points, labels, np.arange(3), [b])[0], expected, places=14)

    def test_top_label_ece_unchanged(self):
        rng = np.random.default_rng(9)
        points, labels = _simplex_points(rng, (20, 20, 20))
        correct = np.argmax(points, axis=1) == labels
        self.assertEqual(top_label_ece(points, labels, np.arange(3)), confidence_ece(points.max(axis=1), correct))

    def test_select_from_curve_kept_in_baycap_kdey(self):
        self.assertIs(baycap_kdey.select_from_curve, select_from_curve)


def _method(**kwargs) -> BayCAPKDEy:
    settings = {"input_classifier": "h", "estimand": INSTANCE, "bandwidth": LOGLOSS}
    return BayCAPKDEy(vanilla_acc, **(SETTINGS | settings | kwargs))


def _fitted(**kwargs) -> BayCAPKDEy:
    return _method(**kwargs).fit(DATA.val, DATA.val_posteriors)


class BayCAPKDEyLogLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = _fitted()
        cls.tables = cls.method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        cls.summaries = cls.method.last_posterior_summaries()

    def test_checks(self):
        with self.assertRaises(ValueError):
            _method(input_classifier="mlp")
        with self.assertRaises(ValueError):
            _method(c_prior="hierarchical-f", estimand="realised")
        with self.assertRaises(ValueError):
            _method(kernel="aitchison")
        with self.assertRaises(ValueError):
            _method(kernel="aitchison", shrinkage=0.5)
        with self.assertRaises(ValueError):
            _method(shrinkage=0.5)
        with self.assertRaises(ValueError):
            _method(bandwidth_grid=(0.2, 0.1))
        with self.assertRaises(ValueError):
            BayesKDEy(bandwidth=0.1, bandwidth_grid=(0.1,))

    def test_selection_on_v_alone(self):
        quantifier = self.method.quantifier_
        curve, used = class_logloss_curve(DATA.val_posteriors, DATA.val.y, DATA.val.classes_, LOO_GRID)
        np.testing.assert_allclose(self.summaries["logloss_curve"], curve, rtol=1e-12)
        self.assertEqual(self.summaries["logloss_bandwidth"], LOO_GRID[select_from_curve(LOO_GRID, curve)])
        self.assertEqual(self.summaries["logloss_classes"], int(used.sum()))
        np.testing.assert_allclose(self.method.bandwidths_, self.summaries["logloss_bandwidth"])
        # the bag does not enter the selection: another bag gives the same summaries of the selection
        self.method.predict_ct_range(DATA.X_u[:30], DATA.u_posteriors[:30])
        other = self.method.last_posterior_summaries()
        for name in LOGLOSS_SUMMARIES:
            np.testing.assert_array_equal(other[name], self.summaries[name])
        self.assertEqual(quantifier.logloss_bandwidth_, self.summaries["logloss_bandwidth"])

    def test_summaries(self):
        self.assertEqual(set(self.summaries), set(FIXED_SUMMARIES) | set(INSTANCE_SUMMARIES) | set(LOGLOSS_SUMMARIES))
        grid = np.asarray(LOO_GRID)
        best = int(np.flatnonzero(grid == self.summaries["logloss_bandwidth"])[0])
        default = int(np.flatnonzero(np.isclose(grid, 0.1))[0])
        self.assertEqual(self.summaries["logloss_selected"], self.summaries["logloss_curve"][best])
        self.assertEqual(self.summaries["logloss_default"], self.summaries["logloss_curve"][default])
        self.assertLessEqual(self.summaries["logloss_selected"], self.summaries["logloss_default"])
        self.assertEqual(self.summaries["logloss_edge"], -1 if best == 0 else (1 if best == len(grid) - 1 else 0))
        ece = loo_confidence_ece(
            DATA.val_posteriors, DATA.val.y, DATA.val.classes_, [0.1, self.summaries["logloss_bandwidth"]]
        )
        self.assertAlmostEqual(self.summaries["loo_ece_default"], ece[0], places=14)
        self.assertAlmostEqual(self.summaries["loo_ece_selected"], ece[1], places=14)

    def test_same_draws_as_the_numeric_bandwidth(self):
        # the deployed fit is that of the numeric bandwidth b*: same densities, draws and tables
        numeric = _fitted(bandwidth=self.summaries["logloss_bandwidth"])
        np.testing.assert_array_equal(self.tables, numeric.predict_ct_range(DATA.X_u, DATA.u_posteriors))
        reference = numeric.last_posterior_summaries()
        for name in (*FIXED_SUMMARIES, *INSTANCE_SUMMARIES):
            np.testing.assert_array_equal(self.summaries[name], reference[name], name)

    def test_grid_of_the_reference_bandwidth_gives_bayCAP_KDEy_h_inst(self):
        method = _fitted(bandwidth_grid=(0.1,))
        tables = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        summaries = method.last_posterior_summaries()
        reference = _fitted(bandwidth=0.1)
        np.testing.assert_array_equal(tables, reference.predict_ct_range(DATA.X_u, DATA.u_posteriors))
        expected = reference.last_posterior_summaries()
        self.assertEqual(set(expected), set(FIXED_SUMMARIES) | set(INSTANCE_SUMMARIES))
        for name in expected:
            np.testing.assert_array_equal(summaries[name], expected[name], name)
        self.assertEqual(summaries["logloss_bandwidth"], 0.1)
        self.assertEqual(summaries["logloss_edge"], -1)
        self.assertEqual(summaries["loo_ece_default"], summaries["loo_ece_selected"])

    def test_determinism(self):
        again = _fitted()
        np.testing.assert_array_equal(self.tables, again.predict_ct_range(DATA.X_u, DATA.u_posteriors))
        self.assertEqual(again.last_posterior_summaries()["logloss_bandwidth"], self.summaries["logloss_bandwidth"])

    def test_other_estimands(self):
        # the selection does not depend on the estimand, and the realised method shares the draws of the fit
        realised = _fitted(estimand="realised")
        realised.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        summaries = realised.last_posterior_summaries()
        self.assertEqual(summaries["logloss_bandwidth"], self.summaries["logloss_bandwidth"])
        np.testing.assert_array_equal(summaries["prev_mean"], self.summaries["prev_mean"])


class FactoryTest(unittest.TestCase):
    def test_configuration_of_the_factory_entry(self):
        self.assertEqual(experiment_methods.LOGLOSS_VARIANTS, {CB: INST})
        self.assertIs(experiment_methods.METHOD_CLASS[CB], BayCAPKDEy)
        self.assertIn(CB, experiment_methods.BAYES_POOL_METHODS)
        method = experiment_methods.logloss_method(vanilla_acc, CB)
        reference = experiment_methods.instance_method(vanilla_acc, INST)
        self.assertEqual(method.bandwidth, LOGLOSS)
        self.assertIsNone(method.bandwidth_grid)
        self.assertEqual(reference.bandwidth, 0.1)
        settings = ("input_classifier", "estimand", "c_prior", "kernel", "shrinkage", "num_warmup", "num_samples")
        for name in (*settings, "temperature", "prior", "random_state"):
            self.assertEqual(getattr(method, name), getattr(reference, name), name)
        names = [name for name, _ in experiment_methods.gen_methods()]
        self.assertEqual(names.index(CB), names.index(INST) + 1)
        self.assertEqual(experiment_methods.get_method_names().index(CB), names.index(CB))

    def test_outside_the_main_tables_with_labels(self):
        self.assertNotIn(CB, main_tables.MAIN_METHODS)
        self.assertNotIn(CB, shift_profile.DEFAULT_METHODS)
        self.assertEqual(METHOD_LATEX_LABELS[CB], "\\bcapkdey{}-cb")
        self.assertEqual(method_plot_label(CB), "BayCAP-KDEy-cb")
        self.assertIn(CB, METHOD_COLORS)
        self.assertEqual(list(METHOD_COLORS.values()).count(METHOD_COLORS[CB]), 1)


if __name__ == "__main__":
    unittest.main()
