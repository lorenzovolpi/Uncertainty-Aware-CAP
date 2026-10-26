import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.methods.baycap import fit_beta_binomial
from uncawe.plots import eb_prior as eb_prior_plots
from uncawe.tables import eb_prior
from uncawe.tables.displacement import smoothed_recalls
from uncawe.tables.eb_prior import (
    METHODS,
    TABLES,
    _latex_longtable,
    check_table,
    common_pairs,
    diagnostics_table,
    left_out_sd,
    pair_table,
    recall_error,
    recall_estimates,
)

# cap-ml layout: true classes on rows
VAL_COUNTS = {
    ("ternary", "LR"): np.array([[45, 3, 2], [4, 40, 6], [1, 1, 18]]),
    ("binary", "LR"): np.array([[80, 20], [10, 90]]),
}
TRUE_CTS = {
    ("ternary", "LR"): [np.diag([0.3, 0.3, 0.4]), np.array([[0.5, 0.0, 0.0], [0.1, 0.2, 0.0], [0.0, 0.0, 0.2]])],
    ("binary", "LR"): [np.array([[0.45, 0.05], [0.05, 0.45]]), np.array([[0.3, 0.1], [0.1, 0.5]])],
}


def results(estim_shift: float, diagnostics: bool = True) -> pd.DataFrame:
    """Stored results of one method: the mean table moves accuracy by ``estim_shift`` from the realised one."""
    rows = []
    for (dataset, classifier), cts in TRUE_CTS.items():
        n_classes = cts[0].shape[0]
        for bag, true_ct in enumerate(cts):
            estim_ct = true_ct + estim_shift * (np.eye(n_classes) - 1 / n_classes) / (n_classes - 1)
            row = {
                "dataset": dataset,
                "classifier": classifier,
                "n_classes": n_classes,
                "true_cts": true_ct,
                "estim_cts": estim_ct,
                "estim_acc_ranges": np.trace(estim_ct) + np.linspace(-0.02, 0.02, 101),
            }
            if diagnostics:
                row |= {"nuts_divergences": bag, "nuts_max_rhat": 1.0 + 0.01 * bag, "nuts_min_ess": 300.0 - bag}
            rows.append(row)
    return pd.DataFrame(rows)


class CheckTest(unittest.TestCase):
    def test_recall_estimates(self):
        counts = VAL_COUNTS[("ternary", "LR")]
        estimates, fit = recall_estimates(counts)
        f, n = np.array([45, 40, 18]), np.array([50, 50, 20])
        self.assertEqual(fit, fit_beta_binomial(f, n))
        np.testing.assert_allclose(estimates["plain"], f / n)
        np.testing.assert_allclose(estimates["uniform"], (f + 1) / (n + 3))
        np.testing.assert_allclose(estimates["eb"], smoothed_recalls(counts, "empirical-bayes"))

    def test_recall_error_hand_computed_with_absent_class(self):
        true_ct = np.array([[0.4, 0.1, 0.0], [0.1, 0.4, 0.0], [0.0, 0.0, 0.0]])
        # pi = (0.5, 0.5, 0), r = (0.8, 0.8, -): 0.5 (0.9 - 0.8) + 0.5 (0.7 - 0.8); class 2 does not count
        self.assertAlmostEqual(recall_error(np.array([0.9, 0.7, np.nan]), true_ct), 0.0)
        self.assertAlmostEqual(recall_error(np.array([0.9, 0.9, 0.1]), true_ct), 0.1)

    def test_left_out_sd(self):
        fit = fit_beta_binomial(np.array([45, 40, 18]), np.array([50, 50, 20]))
        n = np.array([50, 50, 20])
        expected = np.mean(fit.nu / (n + fit.nu)) * fit.sd_m
        if np.isnan(fit.sd_m):
            self.assertTrue(np.isnan(left_out_sd(n, fit)))
        else:
            self.assertAlmostEqual(left_out_sd(n, fit), expected)
        at_bound = fit_beta_binomial(np.array([5, 5]), np.array([5, 5]))
        self.assertTrue(np.isnan(left_out_sd(np.array([5, 5]), at_bound)))

    def test_check_table(self):
        table = check_table(results(0.0), VAL_COUNTS).set_index(["dataset", "classifier"])
        binary = table.loc[("binary", "LR")]
        # Dir(1) estimates (81/102, 91/102); bag 0: pi = (0.5, 0.5), r = (0.9, 0.9); bag 1: pi = (0.4, 0.6), r = (0.75, 5/6)
        dir1 = np.array([81, 91]) / 102
        errors = [0.5 * (dir1[0] - 0.9) + 0.5 * (dir1[1] - 0.9), 0.4 * dir1[0] + 0.6 * dir1[1] - 0.8]
        self.assertAlmostEqual(binary["check_uniform_mean"], np.mean(errors))
        self.assertAlmostEqual(binary["check_uniform_abs"], np.mean(np.abs(errors)))
        self.assertEqual((binary["n_classes"], binary["min_val_per_class"], binary["n_val"]), (2, 100, 200))
        ternary = table.loc[("ternary", "LR")]
        self.assertEqual(ternary["median_val_per_class"], 50)
        fit = fit_beta_binomial(np.array([45, 40, 18]), np.array([50, 50, 20]))
        self.assertEqual((ternary["eb_m"], ternary["eb_nu"]), (fit.m, fit.nu))


class DiagnosticsTest(unittest.TestCase):
    def test_summary_per_pair(self):
        table = diagnostics_table(results(0.0)).set_index(["dataset", "classifier"])
        np.testing.assert_allclose(table["share_divergent"], 0.5)
        np.testing.assert_allclose(table["max_rhat"], 1.01)
        np.testing.assert_allclose(table["min_ess"], 299.0)

    def test_results_without_diagnostics_give_nan(self):
        table = diagnostics_table(results(0.0, diagnostics=False))
        self.assertTrue(table[["share_divergent", "max_rhat", "min_ess"]].isna().all().all())


class PairTableTest(unittest.TestCase):
    def setUp(self):
        # Dir(1) methods are displaced by -0.1, the empirical-Bayes ones are centred; the Dir(1) ones have no diagnostics
        self.results = {
            "BayesCAP": results(-0.1, diagnostics=False),
            "BayesCAP-real": results(-0.1, diagnostics=False),
            "BayesCAP-EB": results(0.0),
            "BayesCAP-EB-real": results(0.0),
        }
        self.table = pair_table(self.results, VAL_COUNTS)

    def test_rows_columns_and_order(self):
        self.assertEqual(
            list(zip(self.table["dataset"], self.table["classifier"])), [("binary", "LR"), ("ternary", "LR")]
        )
        for method in METHODS:
            for column in ("coverage", "recall_term", "prior_part", "share_divergent"):
                self.assertIn(f"{method}_{column}", self.table)
        for name in ("population", "realised"):
            self.assertIn(f"{name}_winkler_p_holm", self.table)

    def test_scores_tests_and_recall_change(self):
        row = self.table.iloc[0]
        self.assertAlmostEqual(row["BayesCAP_shift"], -0.1)
        self.assertAlmostEqual(row["BayesCAP-EB_shift"], 0.0)
        self.assertAlmostEqual(row["BayesCAP-EB_coverage"], 1.0)
        self.assertAlmostEqual(row["BayesCAP_coverage"], 0.0)
        self.assertLess(row["population_winkler_diff_mean"], 0)
        self.assertLess(row["realised_winkler_ci_high"], 0)
        self.assertAlmostEqual(
            row["observed_recall_change"], row["BayesCAP-EB_recall_term"] - row["BayesCAP_recall_term"]
        )
        self.assertAlmostEqual(row["predicted_recall_change"], row["check_eb_mean"] - row["check_uniform_mean"])
        self.assertTrue(np.isnan(row["BayesCAP_share_divergent"]))
        self.assertAlmostEqual(row["BayesCAP-EB_share_divergent"], 0.5)
        # R_0 of each method uses its own prior
        self.assertAlmostEqual(row["BayesCAP_prior_part"], row["check_uniform_mean"] - row["check_plain_mean"])
        self.assertAlmostEqual(row["BayesCAP-EB_prior_part"], row["check_eb_mean"] - row["check_plain_mean"])

    def test_common_pairs_keep_the_pairs_of_every_method(self):
        partial = dict(self.results)
        partial["BayesCAP-EB"] = self.results["BayesCAP-EB"].query("dataset == 'ternary'")
        self.assertEqual(common_pairs(partial), [("ternary", "LR")])

    def test_compute_pair_table_restricts_to_common_pairs(self):
        partial = dict(self.results)
        partial["BayesCAP-EB-real"] = self.results["BayesCAP-EB-real"].query("dataset == 'binary'")
        with (
            patch.object(eb_prior, "load_results", side_effect=lambda method: partial[method]),
            patch.object(eb_prior, "load_validation_counts", side_effect=lambda pairs: VAL_COUNTS) as counts,
        ):
            table = eb_prior.compute_pair_table()
        self.assertEqual(counts.call_args.args[0], [("binary", "LR")])
        self.assertEqual(list(table["dataset"]), ["binary"])

    def test_missing_method_is_reported(self):
        empty = dict(self.results, **{"BayesCAP-EB": pd.DataFrame()})
        with (
            patch.object(eb_prior, "load_results", side_effect=lambda method: empty[method]),
            self.assertRaisesRegex(RuntimeError, "BayesCAP-EB"),
        ):
            eb_prior.compute_pair_table()

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in TABLES.items():
            with self.subTest(table=name):
                latex = _latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("\\begin{longtable}", latex)
                self.assertIn(label, latex)
        prior = _latex_longtable(TABLES["prior"][0](self.table), "", "x")
        self.assertIn("--", prior)  # diagnostics missing for the Dir(1) methods

    def test_figures(self):
        for name, plotter in eb_prior_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(self.table)
                self.assertEqual(len(fig.axes), 3)  # two panels and the colour bar of K
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
