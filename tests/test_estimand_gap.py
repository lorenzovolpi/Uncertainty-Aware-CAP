import unittest

import numpy as np
import pandas as pd

from uncawe.tables.estimand_gap import (
    approx_predictive,
    full_prevalence,
    gap_std,
    gate_summary,
    pair_rng,
    precisions,
    pred_counts,
)


class EstimandGapTest(unittest.TestCase):
    def setUp(self):
        # cap-ml layout: true classes on rows, predicted classes on columns
        self.true_ct = np.array([[0.45, 0.05], [0.05, 0.45]])
        self.estim_ct = np.array([[0.4, 0.2], [0.1, 0.3]])

    def test_pred_counts_are_column_sums(self):
        ct = np.array([[0.3, 0.1], [0.2, 0.4]])
        np.testing.assert_array_equal(pred_counts(ct, sample_size=10), [5, 5])

    def test_precisions_normalise_per_predicted_class(self):
        np.testing.assert_allclose(precisions(self.estim_ct), [0.4 / 0.5, 0.3 / 0.5])
        np.testing.assert_allclose(precisions(np.array([[0.5, 0.0], [0.5, 0.0]])), [0.5, 0.0])

    def test_gap_std_hand_computed(self):
        # sqrt(500 * 0.9 * 0.1 + 500 * 0.8 * 0.2) / 1000
        self.assertAlmostEqual(gap_std([500, 500], [0.9, 0.8]), np.sqrt(45 + 80) / 1000)
        self.assertEqual(gap_std([10, 0], [1.0, 0.0]), 0.0)

    def test_approx_predictive_adds_seeded_noise(self):
        samples = np.full(20_000, 0.8)
        noisy = approx_predictive(samples, 0.01, np.random.default_rng(0))
        self.assertAlmostEqual(noisy.mean(), 0.8, places=3)
        self.assertAlmostEqual(noisy.std(), 0.01, places=3)
        np.testing.assert_array_equal(approx_predictive(samples, 0.0, np.random.default_rng(0)), samples)

    def test_pair_rng_depends_only_on_pair(self):
        a = pair_rng("letter", "LR").normal(size=3)
        np.testing.assert_array_equal(a, pair_rng("letter", "LR").normal(size=3))
        self.assertFalse(np.array_equal(a, pair_rng("letter", "kNN").normal(size=3)))

    def test_full_prevalence_restores_first_class(self):
        np.testing.assert_allclose(full_prevalence(0.3, 2), [0.7, 0.3])
        np.testing.assert_allclose(full_prevalence([0.25, 0.65], 3), [0.10, 0.25, 0.65])

    def test_gate_summary_counts(self):
        table = pd.DataFrame(
            {
                "population_coverage": [0.5, 0.95, 0.8],
                "predictive_coverage": [0.94, 0.99, 0.85],
                "population_winkler": [1.0, 0.1, 0.5],
                "predictive_winkler": [0.5, 0.2, 0.4],
                "ratio": [1.2, 0.1, 0.5],
            }
        )
        summary = gate_summary(table)
        self.assertEqual(summary["under_covering"], 2)
        self.assertEqual(summary["fixed_by_predictive"], 1)
        self.assertEqual(summary["ratio_below_small"], 1)
        self.assertEqual(summary["winkler_improved"], 2)


if __name__ == "__main__":
    unittest.main()
