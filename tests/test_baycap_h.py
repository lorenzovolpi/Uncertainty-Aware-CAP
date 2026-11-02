import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import baycap_h as baycap_h_plots
from uncawe.tables import baycap_h as bh
from uncawe.tables import hierarchical as h

# cap-ml layout: true classes on rows
VAL_COUNTS = {
    ("wdbc", "LR"): np.array([[90, 10], [30, 70]]),
    ("german", "LR"): np.array([[180, 20], [0, 100]]),
    ("letter", "LR"): np.array([[45, 3, 2], [4, 40, 6], [1, 2, 47]]),
}
PAIRS = [("letter", "LR"), ("wdbc", "LR"), ("german", "LR")]
N_BAGS = 4
FAILED = ("wdbc", "LR", 2)  # the bag of BayCAP-H where NUTS produced no draws


def results(method: str, estim_shift: float, seed: int) -> pd.DataFrame:
    """Stored results of one method: the same bags (true tables) for every method, a mean table moved by
    ``estim_shift`` on the diagonal and accuracy draws around its trace; BayCAP-H also has its posterior summaries."""
    bag_rng, rng = np.random.default_rng(0), np.random.default_rng(seed)
    rows = []
    for dataset, classifier in PAIRS:
        k = VAL_COUNTS[(dataset, classifier)].shape[0]
        for bag in range(N_BAGS):
            props = bag_rng.dirichlet(np.ones(k))
            rates = np.full((k, k), 0.1 / (k - 1))
            np.fill_diagonal(rates, 0.9)
            true_ct = props[:, None] * rates
            estim_ct = true_ct.copy()
            estim_ct[np.diag_indices(k)] += estim_shift / k
            estim_ct /= estim_ct.sum()
            draws = np.trace(estim_ct) + rng.normal(0, 0.05, 200)
            row = {
                "dataset": dataset,
                "classifier": classifier,
                "n_classes": k,
                "true_cts": true_ct,
                "estim_cts": estim_ct,
                "estim_acc_ranges": draws,
                "acc_err": abs(np.trace(estim_ct) - np.trace(true_ct)),
                "nuts_divergences": bag % 2 if dataset == "german" else 0,
                "nuts_max_rhat": 1.0,
                "nuts_min_ess": 400.0,
            }
            if method == bh.METHOD:
                row.update(
                    {
                        "nuts_failed": 0,
                        "h_m_mean": 0.8,
                        "h_ln_nu_median": 2.0,
                        "h_ln_nu_q05": 0.5,
                        "h_ln_nu_q95": 4.0,
                        "h_p_nu_above_min_n": 0.9 if dataset == "letter" else 0.1,
                        "h_recall_mean": np.full(k, 0.9),
                        "h_pi_mean": props,
                        "h_pi_cov": (np.eye(k) * 1e-3).ravel(),
                    }
                )
                if (dataset, classifier, bag) == FAILED:
                    row.update(
                        {
                            "estim_acc_ranges": np.full(200, np.nan),
                            "estim_cts": np.full((k, k), np.nan),
                            "acc_err": np.nan,
                            "nuts_failed": 1,
                            "nuts_divergences": np.nan,
                            "h_m_mean": np.nan,
                            "h_p_nu_above_min_n": np.nan,
                            "h_recall_mean": np.full(k, np.nan),
                            "h_pi_cov": np.full(k * k, np.nan),
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def method_results() -> dict[str, pd.DataFrame]:
    shifts = {"BayesCAP-real": -0.05, "BayesCAP-EB-real": 0.0, bh.METHOD: 0.02}
    return {method: results(method, shift, seed=i) for i, (method, shift) in enumerate(shifts.items())}


def synthetic_table() -> pd.DataFrame:
    stored = method_results()
    with (
        patch.object(bh, "load_results", side_effect=lambda method, experiment, domain: stored[method]),
        patch.object(bh, "load_validation_counts", side_effect=lambda pairs, domain: VAL_COUNTS),
    ):
        return bh.compute_table()


class BagFunctionsTest(unittest.TestCase):
    def setUp(self):
        self.stored = method_results()

    def test_failed_bags(self):
        df = self.stored[bh.METHOD]
        wdbc = df.loc[df["dataset"] == "wdbc"].reset_index(drop=True)
        np.testing.assert_array_equal(bh.failed_bags(wdbc), [False, False, True, False])
        self.assertFalse(bh.failed_bags(self.stored["BayesCAP-real"]).any())

    def test_residual_variance_removes_the_part_due_to_the_recalls(self):
        df = self.stored["BayesCAP-real"]
        letter = df.loc[df["dataset"] == "letter"].reset_index(drop=True)
        post = h.uniform_posterior(VAL_COUNTS[("letter", "LR")])
        rho = bh.residual_variances(letter, post)
        for b, row in enumerate(letter.itertuples()):
            props = np.asarray(row.estim_cts).sum(axis=1)
            _, v_c = h.accuracy_moments(post, props / props.sum())
            self.assertAlmostEqual(rho[b], np.var(row.estim_acc_ranges) - v_c[0])

    def test_prevalence_part(self):
        cov = np.array([[0.04, -0.01], [-0.01, 0.09]])
        df = pd.DataFrame({"h_recall_mean": [np.array([0.5, 1.0])], "h_pi_cov": [cov.ravel()]})
        # r' S r = 0.25 * 0.04 + 2 * 0.5 * (-0.01) + 0.09
        np.testing.assert_allclose(bh.prevalence_parts(df), [0.01 - 0.01 + 0.09])

    def test_failure_class(self):
        self.assertEqual(bh.failure_class(0.05, 1, 0.9), "")
        self.assertEqual(bh.failure_class(np.nan, 1, 0.9), "")
        self.assertEqual(bh.failure_class(0.5, 1, 0.9), "boundary")
        self.assertEqual(bh.failure_class(0.5, 0, 0.9), "funnel")
        self.assertEqual(bh.failure_class(0.5, 0, 0.1), "other")


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = synthetic_table()
        cls.stored = method_results()

    def row(self, dataset):
        return self.table.loc[self.table["dataset"].astype(str) == dataset].iloc[0]

    def test_rows_and_columns(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["german", "wdbc", "letter"])
        for method in bh.METHODS:
            for score in bh.SCORES:
                self.assertIn(f"{method}_{score}", self.table)
        for key in bh.REFERENCES:
            for column in bh.TEST_COLUMNS:
                self.assertIn(f"{key}_{column}", self.table)
        for column in ("d4_pred_coverage", "d4_pred_amplitude", "post_alpha_risk", "funnel_risk", *bh.RUN_POSTERIOR):
            self.assertIn(column, self.table)

    def test_failed_bags_are_left_out_of_the_scores_of_baycap_h(self):
        wdbc = self.row("wdbc")
        self.assertAlmostEqual(wdbc["share_failed"], 1 / N_BAGS)
        df = self.stored[bh.METHOD]
        kept = df.loc[df["dataset"] == "wdbc"].reset_index(drop=True).drop(index=FAILED[2])
        self.assertAlmostEqual(wdbc[f"{bh.METHOD}_ae"], kept["acc_err"].mean())
        self.assertAlmostEqual(wdbc[f"{bh.METHOD}_coverage"], h.stored_scores(kept, bh.METHOD)["coverage"])
        self.assertAlmostEqual(wdbc["h_m_mean"], 0.8)
        # the stored methods keep all their bags
        ref = self.stored["BayesCAP-real"]
        ref = ref.loc[ref["dataset"] == "wdbc"].reset_index(drop=True)
        self.assertAlmostEqual(wdbc["BayesCAP-real_coverage"], h.stored_scores(ref, "BayesCAP-real")["coverage"])

    def test_tests_are_holm_adjusted_and_measure_h_minus_reference(self):
        for key, reference in bh.REFERENCES.items():
            p, p_holm = self.table[f"{key}_winkler_p"], self.table[f"{key}_winkler_p_holm"]
            self.assertTrue((p_holm >= p - 1e-12).all())
            letter = self.row("letter")
            diff = letter[f"{bh.METHOD}_winkler"] - letter[f"{reference}_winkler"]
            self.assertAlmostEqual(letter[f"{key}_winkler_diff_mean"], diff)

    def test_sampler_checks_and_classes(self):
        german = self.row("german")
        # half of the bags diverge, and class 2 of german is perfect on V
        self.assertAlmostEqual(german["share_divergent"], 0.5)
        self.assertEqual(german["n_perfect_classes"], 1)
        self.assertEqual(german["failure_class"], "boundary")
        self.assertEqual(self.row("letter")["failure_class"], "")
        self.assertAlmostEqual(self.row("letter")["h_prevalence_part"], 0.9**2 * 3e-3)
        for column in ("post_alpha_risk", "funnel_risk"):
            self.assertTrue(self.table[column].between(0, 1).all())

    def test_d4_matches_the_prediction_of_tables_hierarchical(self):
        letter = self.row("letter")
        counts = VAL_COUNTS[("letter", "LR")]
        f, n = h.diagonal_counts(counts)
        target = h.mixture_posterior(f, n, h.grid_weights(h.class_log_marginals(f, n).sum(axis=0), bh.PRIOR))
        ref = self.stored["BayesCAP-real"]
        ref = ref.loc[ref["dataset"] == "letter"].reset_index(drop=True)
        expected = h.predicted_scores(ref, "BayesCAP-real", h.uniform_posterior(counts), target)
        self.assertAlmostEqual(letter["d4_pred_coverage"], expected["coverage"])
        self.assertAlmostEqual(letter["d4_pred_amplitude"], expected["amplitude"])

    def test_summary(self):
        summary = bh.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "sampler", "checks"})
        overall = summary["sampler"].iloc[0]
        self.assertEqual((overall["K"], overall["pairs"]), ("all", 3))
        self.assertEqual(overall["boundary"], 1)
        self.assertEqual(overall["with_failed_bags"], 1)
        self.assertEqual(overall["above_share"], 1)

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in bh.TABLES.items():
            with self.subTest(table=name):
                latex = bh.eb_prior._latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("longtable", latex)
                self.assertIn(label, latex)

    def test_figures_build_from_the_csv(self):
        table = pd.read_csv(pd.io.common.StringIO(self.table.to_csv(index=False)))
        for name, plotter in baycap_h_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(table)
                self.assertGreaterEqual(len(fig.axes), 2)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
