"""One bandwidth of BayesKDEy selected on calibration bags drawn from half of $V$ (E7.2 of
``docs/ext7_kdey_bandwidth.md``, work-plan card WP-26)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
import quapy as qp
from cap.error import vanilla_acc
from scipy.optimize import minimize
from test_baycap_fixed import DATA, SETTINGS

from uncawe.methods.baycap_kdey import (
    CAL_BAG_STREAM,
    CAL_NUTS_SUMMARIES,
    CAL_SPLIT_STREAM,
    CAL_SUMMARIES,
    BayCAPKDEy,
    accuracy_criterion,
    calibration_bags,
    calibration_seed,
    class_recalls,
    select_from_curve,
    split_halves,
)
from uncawe.methods.bayes_kdey import LOO_GRID, em_prevalences

# the experiment seed of main, the default random_state of the methods
qp.environ.setdefault("_R_SEED", 0)

# a small calibration, so that the tests run fast
CAL = {"bandwidth": "cal", "cal_bags": 8, "cal_bag_size": 200}
GRID = (0.05, 0.1, 0.2)


def _method(**kwargs) -> BayCAPKDEy:
    return BayCAPKDEy(vanilla_acc, estimand="realised", **(SETTINGS | {"input_classifier": "h"} | CAL | kwargs))


def _log_likelihood(prev, log_densities, counts):
    return float(counts @ np.log(np.exp(log_densities).T @ prev))


class SplitTest(unittest.TestCase):
    def test_split_is_stratified_disjoint_covering_and_reproducible(self):
        labels = np.asarray(DATA.val.y)
        classes = DATA.val.classes_
        fit, cal = split_halves(labels, classes, 7)
        self.assertEqual(len(np.intersect1d(fit, cal)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([fit, cal])), np.arange(len(labels)))
        for class_ in classes:
            n = np.sum(labels == class_)
            with self.subTest(class_=class_):
                self.assertEqual(np.sum(labels[cal] == class_), n // 2)
                self.assertEqual(np.sum(labels[fit] == class_), n - n // 2)
        again = split_halves(labels, classes, 7)
        np.testing.assert_array_equal(fit, again[0])
        np.testing.assert_array_equal(cal, again[1])
        self.assertFalse(np.array_equal(cal, split_halves(labels, classes, 8)[1]))

    def test_a_class_with_one_instance_cannot_be_split(self):
        with self.assertRaises(ValueError):
            split_halves(np.array([0, 0, 1]), np.array([0, 1]), 0)

    def test_seeds_depend_on_random_state_and_stream_only(self):
        self.assertEqual(calibration_seed(3, CAL_SPLIT_STREAM), calibration_seed(3, CAL_SPLIT_STREAM))
        self.assertNotEqual(calibration_seed(3, CAL_SPLIT_STREAM), calibration_seed(3, CAL_BAG_STREAM))
        self.assertNotEqual(calibration_seed(3, CAL_SPLIT_STREAM), calibration_seed(4, CAL_SPLIT_STREAM))


class BagsTest(unittest.TestCase):
    def test_bags_are_reproducible_with_their_class_proportions(self):
        labels = np.repeat([0, 1, 2], [5, 9, 7])
        classes = np.array([0, 1, 2])
        index, prevalences = calibration_bags(labels, classes, 6, 50, 11)
        self.assertEqual(index.shape, (6, 50))
        self.assertTrue(np.all((index >= 0) & (index < len(labels))))
        for s in range(6):
            np.testing.assert_allclose(prevalences[s], np.bincount(labels[index[s]], minlength=3) / 50)
        again = calibration_bags(labels, classes, 6, 50, 11)
        np.testing.assert_array_equal(index, again[0])
        np.testing.assert_array_equal(prevalences, again[1])
        self.assertFalse(np.array_equal(index, calibration_bags(labels, classes, 6, 50, 12)[0]))

    def test_bags_leave_the_experiment_seed_and_the_global_state_unchanged(self):
        seed = qp.environ["_R_SEED"]
        np.random.seed(5)
        expected = np.random.random_sample(3)
        np.random.seed(5)
        calibration_bags(np.repeat([0, 1], [4, 6]), np.array([0, 1]), 3, 20, 99)
        self.assertEqual(qp.environ["_R_SEED"], seed)
        np.testing.assert_array_equal(np.random.random_sample(3), expected)


class EMTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        # a toy mixture: three overlapping Gaussian classes on the line, densities known exactly
        centres = np.array([-1.0, 0.0, 1.5])
        z = rng.normal(centres[rng.choice(3, size=60, p=[0.5, 0.3, 0.2])], 1.0)
        self.log_densities = -0.5 * (z[None, :] - centres[:, None]) ** 2 - 0.5 * np.log(2 * np.pi)
        self.counts = rng.integers(0, 4, size=(4, 60)).astype(float)
        self.counts[:, 0] += 1

    def test_em_equals_a_generic_optimiser_of_the_log_likelihood(self):
        estimates, converged = em_prevalences(self.log_densities, self.counts, tol=1e-12, max_iter=200_000)
        self.assertTrue(converged.all())
        for s, counts in enumerate(self.counts):
            result = minimize(
                lambda p, c=counts: -_log_likelihood(p, self.log_densities, c),
                np.full(3, 1 / 3),
                method="SLSQP",
                bounds=[(0, 1)] * 3,
                constraints=[{"type": "eq", "fun": lambda p: p.sum() - 1}],
                options={"ftol": 1e-14, "maxiter": 1000},
            )
            with self.subTest(bag=s):
                np.testing.assert_allclose(estimates[s].sum(), 1.0)
                # the EM estimate is at least as likely as the optimiser's, and close to it
                self.assertGreaterEqual(
                    _log_likelihood(estimates[s], self.log_densities, counts),
                    _log_likelihood(
                        np.clip(result.x, 0, None) / np.clip(result.x, 0, None).sum(), self.log_densities, counts
                    )
                    - 1e-7,
                )
                np.testing.assert_allclose(estimates[s], result.x, atol=1e-3)

    def test_counts_equal_repeated_instances(self):
        counts = self.counts[0].astype(int)
        expanded = np.repeat(np.arange(60), counts)
        by_counts, _ = em_prevalences(self.log_densities, counts[None, :].astype(float))
        by_instances, _ = em_prevalences(self.log_densities[:, expanded], np.ones((1, len(expanded))))
        np.testing.assert_allclose(by_counts, by_instances, atol=1e-8)

    def test_an_instance_with_zero_density_under_every_class_is_left_out(self):
        log_densities = np.column_stack([self.log_densities, np.full(3, -np.inf)])
        counts = np.column_stack([self.counts, np.full(len(self.counts), 5.0)])
        with_it, converged = em_prevalences(log_densities, counts)
        without, _ = em_prevalences(self.log_densities, self.counts)
        self.assertTrue(converged.all())
        np.testing.assert_allclose(with_it, without)

    def test_counts_are_checked(self):
        with self.assertRaises(ValueError):
            em_prevalences(self.log_densities, np.zeros((1, 60)))
        with self.assertRaises(ValueError):
            em_prevalences(self.log_densities, np.ones((1, 59)))


class CriterionTest(unittest.TestCase):
    def test_criterion_is_the_mean_absolute_accuracy_error(self):
        estimates = np.array([[0.5, 0.5], [0.2, 0.8]])
        prevalences = np.array([[0.6, 0.4], [0.2, 0.8]])
        recalls = np.array([0.9, 0.5])
        self.assertAlmostEqual(float(accuracy_criterion(estimates, prevalences, recalls)), (0.04 + 0.0) / 2)

    def test_criterion_is_invariant_to_a_constant_added_to_the_recalls(self):
        rng = np.random.default_rng(2)
        estimates = rng.dirichlet(np.ones(4), size=(5, 7))
        prevalences = rng.dirichlet(np.ones(4), size=7)
        recalls = rng.uniform(size=4)
        np.testing.assert_allclose(
            accuracy_criterion(estimates, prevalences, recalls),
            accuracy_criterion(estimates, prevalences, recalls + 0.37),
        )

    def test_recalls(self):
        labels = np.array([0, 0, 1, 1, 1, 2])
        predicted = np.array([0, 1, 1, 1, 2, 2])
        np.testing.assert_allclose(class_recalls(labels, predicted, np.array([0, 1, 2])), [0.5, 2 / 3, 1.0])

    def test_ties_are_broken_towards_the_reference(self):
        grid = np.array([0.01, 0.05, 0.1, 0.3, 1.0])
        self.assertEqual(select_from_curve(grid, np.array([3.0, 2.0, 2.0, 1.0, 1.0])), 3)
        # at the same distance on the log scale, the smaller bandwidth
        self.assertEqual(select_from_curve(np.array([0.05, 0.1, 0.2]), np.array([1.0, 2.0, 1.0])), 0)
        self.assertEqual(select_from_curve(grid, np.array([1.0, 1.0, 1.0, 1.0, 1.0])), 2)
        self.assertEqual(select_from_curve(grid, np.array([0.5, 1.0, 1.0, 1.0, 1.0])), 0)
        self.assertEqual(select_from_curve(grid, np.array([np.nan, 1.0, 2.0, 0.5, 1.0])), 3)
        with self.assertRaises(ValueError):
            select_from_curve(grid, np.full(5, np.nan))


class MethodTest(unittest.TestCase):
    def test_cal_needs_the_gaussian_kernel_no_shrinkage_a_fixed_c_and_a_valid_grid(self):
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, bandwidth="cal", kernel="aitchison")
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, bandwidth="cal", c_prior="hierarchical-f")
        for grid in ([], [0.1, 0.05], [0.0, 0.1]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                BayCAPKDEy(vanilla_acc, bandwidth="cal", bandwidth_grid=grid)
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, bandwidth="cal", cal_bags=0)
        self.assertEqual(BayCAPKDEy(vanilla_acc, bandwidth="cal")._cal_grid(), LOO_GRID)

    def test_selection_and_summaries(self):
        method = _method(bandwidth_grid=GRID).fit(DATA.val, DATA.val_posteriors)
        self.assertIn(method.cal_bandwidth_, GRID)
        np.testing.assert_array_equal(method.bandwidths_, [method.cal_bandwidth_] * 3)
        self.assertEqual(method.cal_curve_.shape, (len(GRID),))
        self.assertEqual(method.cal_estimates_.shape, (len(GRID), CAL["cal_bags"], 3))
        self.assertAlmostEqual(method.cal_selected_, float(method.cal_curve_.min()))
        self.assertAlmostEqual(method.cal_default_, float(method.cal_curve_[GRID.index(0.1)]))
        np.testing.assert_allclose(
            method.cal_curve_, accuracy_criterion(method.cal_estimates_, method.cal_prevalences_, method.cal_recalls_)
        )
        fit_index, _ = method.cal_split_
        predicted = DATA.val.classes_[np.argmax(DATA.val_posteriors, axis=1)]
        np.testing.assert_allclose(
            method.cal_recalls_, class_recalls(DATA.val.y[fit_index], predicted[fit_index], DATA.val.classes_)
        )
        method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        summaries = method.last_posterior_summaries()
        self.assertTrue(set(CAL_SUMMARIES) <= set(summaries))
        self.assertFalse(set(CAL_NUTS_SUMMARIES) & set(summaries))
        self.assertEqual(summaries["cal_bandwidth"], method.cal_bandwidth_)
        np.testing.assert_array_equal(summaries["cal_curve"], method.cal_curve_)

    def test_reference_outside_the_grid_is_still_reported(self):
        method = _method(bandwidth_grid=(0.05, 0.2)).fit(DATA.val, DATA.val_posteriors)
        reference = _method(bandwidth_grid=(0.1,)).fit(DATA.val, DATA.val_posteriors)
        self.assertEqual(method.cal_curve_.shape, (2,))
        self.assertAlmostEqual(method.cal_default_, reference.cal_default_)

    def test_bags_are_the_same_for_every_bandwidth(self):
        small = _method(bandwidth_grid=(0.05,)).fit(DATA.val, DATA.val_posteriors)
        large = _method(bandwidth_grid=GRID).fit(DATA.val, DATA.val_posteriors)
        np.testing.assert_array_equal(small.cal_bag_index_, large.cal_bag_index_)
        np.testing.assert_array_equal(small.cal_prevalences_, large.cal_prevalences_)
        np.testing.assert_array_equal(small.cal_estimates_[0], large.cal_estimates_[0])

    def test_a_grid_of_0_1_reproduces_the_stored_method(self):
        for source in ("h", "mlp"):
            stored = BayCAPKDEy(vanilla_acc, estimand="realised", **(SETTINGS | {"input_classifier": source}))
            cal = _method(input_classifier=source, bandwidth_grid=(0.1,))
            stored.fit(DATA.val, DATA.val_posteriors)
            cal.fit(DATA.val, DATA.val_posteriors)
            with self.subTest(source=source):
                self.assertEqual(cal.cal_bandwidth_, 0.1)
                np.testing.assert_array_equal(
                    stored.predict_ct_range(DATA.X_u, DATA.u_posteriors),
                    cal.predict_ct_range(DATA.X_u, DATA.u_posteriors),
                )
                np.testing.assert_array_equal(stored._last_nuts[0]["prev"], cal._last_nuts[0]["prev"])
                for acc in ("prev_mean", "acc_sd", "sigma_g", "val_ece"):
                    np.testing.assert_array_equal(
                        stored.last_posterior_summaries()[acc], cal.last_posterior_summaries()[acc]
                    )

    def test_the_deployed_fit_is_that_of_the_selected_bandwidth(self):
        cal = _method(bandwidth_grid=(0.2,)).fit(DATA.val, DATA.val_posteriors)
        numeric = BayCAPKDEy(vanilla_acc, estimand="realised", bandwidth=0.2, **(SETTINGS | {"input_classifier": "h"}))
        numeric.fit(DATA.val, DATA.val_posteriors)
        np.testing.assert_array_equal(
            cal.predict_ct_range(DATA.X_u, DATA.u_posteriors), numeric.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        )

    def test_the_selection_uses_no_target_bag(self):
        method = _method(bandwidth_grid=GRID).fit(DATA.val, DATA.val_posteriors)
        curve, bandwidth = method.cal_curve_.copy(), method.cal_bandwidth_
        for size in (80, 30):
            method.predict_ct_range(DATA.X_u[:size], DATA.u_posteriors[:size])
            summaries = method.last_posterior_summaries()
            np.testing.assert_array_equal(summaries["cal_curve"], curve)
            self.assertEqual(summaries["cal_bandwidth"], bandwidth)
        # refitted, the selection is the same: it depends on V and the seed only
        np.testing.assert_array_equal(_method(bandwidth_grid=GRID).fit(DATA.val, DATA.val_posteriors).cal_curve_, curve)

    def test_mlp_input_and_the_nuts_check(self):
        method = _method(input_classifier="mlp", bandwidth_grid=(0.1, 0.2), cal_bags=3, cal_nuts_check=True)
        method.fit(DATA.val, DATA.val_posteriors)
        plain = _method(input_classifier="mlp", bandwidth_grid=(0.1, 0.2), cal_bags=3).fit(
            DATA.val, DATA.val_posteriors
        )
        # the check does not change the selection
        np.testing.assert_array_equal(method.cal_curve_, plain.cal_curve_)
        self.assertEqual(method.cal_bandwidth_, plain.cal_bandwidth_)
        self.assertEqual(method.cal_nuts_means_.shape, (2, 3, 3))
        np.testing.assert_allclose(method.cal_nuts_means_.sum(axis=-1), 1.0, rtol=1e-5)
        self.assertIn(method.cal_bandwidth_nuts_, (0.1, 0.2))
        # the posterior mean is near the maximum-likelihood estimate (bags of 200 instances)
        self.assertLess(np.abs(method.cal_nuts_means_ - method.cal_estimates_).max(), 0.15)
        method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        summaries = method.last_posterior_summaries()
        self.assertTrue(set(CAL_SUMMARIES) | set(CAL_NUTS_SUMMARIES) <= set(summaries))


if __name__ == "__main__":
    unittest.main()
