"""Per-class leave-one-out bandwidths of BayesKDEy (E7.1 of ``docs/ext7_kdey_bandwidth.md``, work-plan card WP-25)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from cap.error import vanilla_acc
from sklearn.neighbors import KernelDensity
from test_baycap_fixed import DATA, SETTINGS

from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.bayes_kdey import (
    LOO_DEFAULT_BANDWIDTH,
    LOO_GRID,
    LOO_SUMMARIES,
    BayesKDEy,
    _KernelDensities,
    _SelectedKernelDensities,
    loo_log_likelihood,
    simplex_log_density,
)
from uncawe.methods.sqs_b import SQSB


def _simplex_points(rng, n, n_classes, concentration=20.0):
    """Points of the simplex around its centre, away from its edges."""
    return rng.dirichlet(np.full(n_classes, concentration), size=n)


def _brute_force_loo(points, bandwidth):
    """sum_i ln p_{-i}(z_i), each p_{-i} a scikit-learn KernelDensity fitted without z_i, normalised on the simplex."""
    total = 0.0
    for i in range(len(points)):
        kde = KernelDensity(bandwidth=bandwidth).fit(np.delete(points, i, axis=0))
        total += simplex_log_density(kde, points[i : i + 1])[0]
    return total


class LeaveOneOutTest(unittest.TestCase):
    def test_grid_has_21_log_spaced_values_from_0_01_to_1_with_0_1(self):
        self.assertEqual(len(LOO_GRID), 21)
        np.testing.assert_allclose(LOO_GRID, 10 ** np.linspace(-2, 0, 21))
        self.assertTrue(np.isclose(LOO_GRID, LOO_DEFAULT_BANDWIDTH).any())

    def test_closed_form_equals_brute_force(self):
        rng = np.random.default_rng(0)
        for n_classes in (2, 3, 4):
            points = _simplex_points(rng, 40, n_classes)
            bandwidths = [0.01, 0.05, 0.1, 0.5]
            # small blocks, so that several blocks are combined
            got = loo_log_likelihood(points, bandwidths, block_entries=100)
            expected = [_brute_force_loo(points, b) for b in bandwidths]
            with self.subTest(n_classes=n_classes):
                np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_simplex_density_integrates_to_one_on_the_simplex_surface(self):
        rng = np.random.default_rng(1)
        bandwidth = 0.02
        # K=2: the surface is the line z=(t, 1-t); the arc length is sqrt(2) dt
        points = _simplex_points(rng, 15, 2)
        kde = KernelDensity(bandwidth=bandwidth).fit(points)
        t = np.linspace(-0.5, 1.5, 40001)
        density = np.exp(simplex_log_density(kde, np.column_stack([t, 1 - t])))
        self.assertAlmostEqual(np.sum(density) * (t[1] - t[0]) * np.sqrt(2), 1.0, places=6)
        # scikit-learn's normalisation in K dimensions does not integrate to 1 on the surface
        self.assertNotAlmostEqual(np.sum(np.exp(kde.score_samples(np.column_stack([t, 1 - t])))) * (t[1] - t[0]), 1.0)
        # K=3: orthonormal coordinates (u, v) of the plane sum(z)=1 around its centre
        points = _simplex_points(rng, 15, 3)
        kde = KernelDensity(bandwidth=bandwidth).fit(points)
        e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2)
        e2 = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
        grid = np.linspace(-0.4, 0.4, 801)
        u, v = np.meshgrid(grid, grid)
        z = np.full(3, 1 / 3) + u.reshape(-1, 1) * e1 + v.reshape(-1, 1) * e2
        density = np.exp(simplex_log_density(kde, z))
        self.assertAlmostEqual(np.sum(density) * (grid[1] - grid[0]) ** 2, 1.0, places=4)

    def test_selection_is_the_arg_max_of_the_criterion_on_the_grid(self):
        rng = np.random.default_rng(2)
        outputs = np.concatenate([_simplex_points(rng, 60, 3, 50.0), _simplex_points(rng, 30, 3, 5.0)])
        labels = np.repeat([0, 1], [60, 30])
        kde = _SelectedKernelDensities(LOO_GRID).fit(outputs, labels, np.array([0, 1]))
        for j in (0, 1):
            curve = loo_log_likelihood(outputs[labels == j], LOO_GRID)
            with self.subTest(class_=j):
                self.assertEqual(kde.bandwidths[j], LOO_GRID[int(np.argmax(curve))])
                self.assertAlmostEqual(kde.loo_selected[j], curve.max() / np.sum(labels == j))
                self.assertEqual(kde.edges[j], 0)
        # the concentrated class gets the smaller bandwidth
        self.assertLess(kde.bandwidths[0], kde.bandwidths[1])

    def test_duplicated_outputs_select_the_lower_edge(self):
        outputs = np.repeat([[0.98, 0.01, 0.01], [0.01, 0.98, 0.01]], 10, axis=0)
        kde = _SelectedKernelDensities(LOO_GRID).fit(outputs, np.repeat([0, 1], 10), np.array([0, 1]))
        np.testing.assert_array_equal(kde.bandwidths, [LOO_GRID[0], LOO_GRID[0]])
        np.testing.assert_array_equal(kde.edges, [-1, -1])

    def test_a_class_with_one_output_keeps_the_default(self):
        rng = np.random.default_rng(3)
        outputs = np.concatenate([_simplex_points(rng, 20, 3), _simplex_points(rng, 1, 3)])
        kde = _SelectedKernelDensities(LOO_GRID).fit(outputs, np.repeat([0, 1], [20, 1]), np.array([0, 1]))
        self.assertEqual(kde.bandwidths[1], LOO_DEFAULT_BANDWIDTH)
        self.assertEqual((kde.class_sizes[1], kde.edges[1]), (1, 0))
        self.assertTrue(np.isnan(kde.loo_selected[1]) and np.isnan(kde.loo_default[1]))

    def test_densities_with_equal_bandwidths_differ_from_quapy_by_a_common_constant(self):
        rng = np.random.default_rng(4)
        outputs = _simplex_points(rng, 50, 3)
        labels = rng.integers(0, 3, size=50)
        classes = np.arange(3)
        selected = _SelectedKernelDensities([LOO_DEFAULT_BANDWIDTH]).fit(outputs, labels, classes)
        quapy = _KernelDensities("gaussian", LOO_DEFAULT_BANDWIDTH, 0.0).fit(outputs, labels, classes)
        targets = _simplex_points(rng, 7, 3)
        difference = selected.log_densities(targets) - quapy.log_densities(targets)
        np.testing.assert_allclose(difference, 0.5 * np.log(2 * np.pi * LOO_DEFAULT_BANDWIDTH**2))

    def test_grid_is_checked(self):
        for grid in ([], [0.1, 0.05], [0.0, 0.1]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                _SelectedKernelDensities(grid)


class MethodsTest(unittest.TestCase):
    def test_loo_needs_the_gaussian_kernel_no_shrinkage_and_a_fixed_c(self):
        with self.assertRaises(ValueError):
            BayesKDEy(kernel="aitchison", bandwidth="loo")
        with self.assertRaises(ValueError):
            BayesKDEy(kernel="aitchison", bandwidth="loo", shrinkage=0.5)
        with self.assertRaises(ValueError):
            BayesKDEy(bandwidth=0.1, bandwidth_grid=[0.1, 0.2])
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, c_prior="hierarchical-f", bandwidth="loo")

    def test_numeric_bandwidth_reports_it_for_every_class(self):
        quantifier = BayesKDEy(input_classifier="h", **SETTINGS).fit(DATA.val, DATA.val_posteriors)
        np.testing.assert_array_equal(quantifier.bandwidths_, [0.1, 0.1, 0.1])
        with self.assertRaises(RuntimeError):
            quantifier.selection_summaries()

    def test_baycap_kdey_and_sqs_b_with_loo_share_draws_and_report_the_selection(self):
        grid = [0.05, 0.1, 0.2]
        baycap = BayCAPKDEy(vanilla_acc, estimand="realised", bandwidth="loo", bandwidth_grid=grid, **SETTINGS).fit(
            DATA.val, DATA.val_posteriors
        )
        sqsb = SQSB(vanilla_acc, bandwidth="loo", bandwidth_grid=grid, sample_size=100, **SETTINGS).fit(
            DATA.val, DATA.val_posteriors
        )
        baycap.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        sqsb.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        np.testing.assert_array_equal(baycap._last_nuts[0]["prev"], sqsb.quantifier_._last_nuts[0]["prev"])
        np.testing.assert_array_equal(baycap.bandwidths_, sqsb.bandwidths_)
        for method in (baycap, sqsb):
            summaries = method.last_posterior_summaries()
            with self.subTest(method=type(method).__name__):
                self.assertTrue(set(LOO_SUMMARIES) <= set(summaries))
                np.testing.assert_array_equal(summaries["bandwidths"], method.bandwidths_)
                self.assertTrue(set(summaries["bandwidths"]) <= set(grid))
                np.testing.assert_array_equal(summaries["class_sizes"], np.bincount(DATA.val.y))
        # the numeric bandwidth adds no summary
        stored = BayCAPKDEy(vanilla_acc, estimand="realised", **SETTINGS).fit(DATA.val, DATA.val_posteriors)
        stored.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        self.assertFalse(set(LOO_SUMMARIES) & set(stored.last_posterior_summaries()))


if __name__ == "__main__":
    unittest.main()
