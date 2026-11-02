import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import baycap_hf as baycap_hf_plots
from uncawe.tables import baycap_hf as hf
from uncawe.tables import eb_prior, pushforward
from uncawe.tables.displacement import error_terms

# cap-ml layout: true classes on rows
VAL_COUNTS = {
    ("wdbc", "LR"): np.array([[90, 10], [30, 70]]),
    ("german", "LR"): np.array([[180, 20], [0, 100]]),
    ("letter", "LR"): np.array([[45, 5, 0], [4, 40, 6], [0, 3, 47]]),
}
PAIRS = [("letter", "LR"), ("wdbc", "LR"), ("german", "LR")]
N_BAGS = 4
FAILED = ("wdbc", "LR", 2)  # the bag of BayCAP-Hf where NUTS produced no draws
SHIFTS = {"BayesCAP-real": -0.05, "BayesCAP-EB-real": 0.0, "BayesCAP-H-real": 0.03, hf.METHOD: 0.01}


def results(method: str, seed: int) -> pd.DataFrame:
    """Stored results of one method: the same bags (true tables) for every method, a mean table moved by the shift of
    the method on the diagonal, accuracy draws around its trace and the summaries of the hierarchical methods."""
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
            estim_ct[np.diag_indices(k)] += SHIFTS[method] / k
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
                "nuts_max_rhat": 1.3 if (dataset == "wdbc" and method == hf.H_METHOD) else 1.0,
                "nuts_min_ess": 400.0,
            }
            if method in hf.HIERARCHICAL:
                row.update(
                    {
                        "nuts_failed": 0,
                        "h_m_mean": 0.8,
                        "h_ln_nu_median": 2.0 if method == hf.METHOD else 4.0,
                        "h_ln_nu_q05": 0.5,
                        "h_ln_nu_q95": 4.0,
                        "h_p_nu_above_min_n": 0.01 if method == hf.METHOD else 0.2,
                    }
                )
            if method == hf.METHOD:
                ln_kappa = np.log(0.3) if k > 2 else np.nan
                row.update(
                    {
                        "h_ln_kappa_median": ln_kappa,
                        "h_ln_kappa_q05": ln_kappa - 1,
                        "h_ln_kappa_q95": ln_kappa + 1,
                        "h_p_kappa_below_1": 0.9 if k > 2 else np.nan,
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
                            "nuts_max_rhat": np.nan,
                            "h_m_mean": np.nan,
                            "h_ln_nu_median": np.nan,
                            "h_p_nu_above_min_n": np.nan,
                        }
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def method_results() -> dict[str, pd.DataFrame]:
    return {method: results(method, seed=i) for i, method in enumerate(hf.METHODS)}


def synthetic_table() -> pd.DataFrame:
    stored = method_results()
    with (
        patch.object(hf, "load_results", side_effect=lambda method, experiment, domain: stored[method]),
        patch.object(hf, "load_validation_counts", side_effect=lambda pairs, domain: VAL_COUNTS),
    ):
        return hf.compute_table()


class FunctionsTest(unittest.TestCase):
    def test_empty_off_diagonal_share(self):
        self.assertAlmostEqual(hf.empty_off_diagonal_share(VAL_COUNTS[("letter", "LR")]), 2 / 6)
        self.assertAlmostEqual(hf.empty_off_diagonal_share(VAL_COUNTS[("german", "LR")]), 1 / 2)
        self.assertAlmostEqual(hf.empty_off_diagonal_share(VAL_COUNTS[("wdbc", "LR")]), 0.0)

    def test_failure_class_checks_the_funnel_first(self):
        # (share divergent, max rhat, P(nu > min n), boundary classes, median ln kappa, empty share)
        self.assertEqual(hf.failure_class(0.05, 1.01, 0.9, 1, -1.0, 0.5), "")
        self.assertEqual(hf.failure_class(np.nan, np.nan, 0.9, 1, -1.0, 0.5), "")
        self.assertEqual(hf.failure_class(0.05, 1.5, 0.2, 1, np.nan, 0.0), "funnel")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.2, 1, np.nan, 0.0), "funnel")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.01, 1, np.nan, 0.0), "boundary")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.01, 0, -0.5, 0.2), "boundary")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.01, 0, -0.5, 0.0), "other")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.01, 0, 0.5, 0.2), "other")
        self.assertEqual(hf.failure_class(0.5, 1.0, 0.01, 0, np.nan, 0.0), "other")


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = synthetic_table()
        cls.stored = method_results()

    def row(self, dataset):
        return self.table.loc[self.table["dataset"].astype(str) == dataset].iloc[0]

    def pair(self, method, dataset):
        df = self.stored[method]
        return df.loc[df["dataset"] == dataset].reset_index(drop=True)

    def test_rows_and_columns(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["german", "wdbc", "letter"])
        for method in hf.METHODS:
            for score in hf.SCORES:
                self.assertIn(f"{method}_{score}", self.table)
        for key in hf.REFERENCES:
            for score in hf.TESTED_SCORES:
                for stat in hf.TEST_STATS:
                    self.assertIn(hf.test_column(key, score, stat), self.table)
        for prefix in hf.SHIFT_METHODS.values():
            for term in hf.SHIFT_TERMS:
                self.assertIn(f"{prefix}_{term}", self.table)
        for column in ("hf_ln_kappa_median", "hf_p_kappa_below_1", "h_ln_nu_median", "hf_max_rhat", "h_max_rhat"):
            self.assertIn(column, self.table)

    def test_failed_bags_are_left_out_of_the_scores_and_tests_of_baycap_hf(self):
        wdbc = self.row("wdbc")
        self.assertAlmostEqual(wdbc["hf_share_failed"], 1 / N_BAGS)
        self.assertAlmostEqual(wdbc["h_share_failed"], 0.0)
        kept = self.pair(hf.METHOD, "wdbc").drop(index=FAILED[2])
        self.assertAlmostEqual(wdbc[f"{hf.METHOD}_ae"], kept["acc_err"].mean())
        # the tests pair the bags where BayCAP-Hf has draws
        ref = self.pair(hf.H_METHOD, "wdbc").drop(index=FAILED[2])
        diff = kept["acc_err"].to_numpy() - ref["acc_err"].to_numpy()
        self.assertAlmostEqual(wdbc[hf.test_column("h", "ae", "diff_mean")], diff.mean())
        # BayCAP-H keeps all its bags in its own scores
        self.assertAlmostEqual(wdbc[f"{hf.H_METHOD}_ae"], self.pair(hf.H_METHOD, "wdbc")["acc_err"].mean())

    def test_tests_are_holm_adjusted_and_measure_hf_minus_reference(self):
        letter = self.row("letter")
        for key, reference in hf.REFERENCES.items():
            for score in hf.TESTED_SCORES:
                p = self.table[hf.test_column(key, score, "p")]
                p_holm = self.table[hf.test_column(key, score, "p_holm")]
                self.assertTrue((p_holm >= p - 1e-12).all())
                np.testing.assert_allclose(p_holm, pushforward.holm(p.to_numpy()))
            diff = letter[f"{hf.METHOD}_winkler"] - letter[f"{reference}_winkler"]
            self.assertAlmostEqual(letter[hf.test_column(key, "winkler", "diff_mean")], diff)
            low, high = letter[hf.test_column(key, "ae", "ci_low")], letter[hf.test_column(key, "ae", "ci_high")]
            self.assertLessEqual(low, letter[hf.test_column(key, "ae", "diff_mean")] + 1e-12)
            self.assertGreaterEqual(high, letter[hf.test_column(key, "ae", "diff_mean")] - 1e-12)

    def test_shift_terms_add_up_to_the_shift(self):
        for prefix in hf.SHIFT_METHODS.values():
            total = self.table[f"{prefix}_recall_term"] + self.table[f"{prefix}_prevalence_term"]
            np.testing.assert_allclose(total, self.table[f"{prefix}_shift"], atol=1e-12)
        df = self.pair(hf.EB_METHOD, "letter")
        terms = [error_terms(e, t)["recall_term"] for e, t in zip(df["estim_cts"], df["true_cts"])]
        self.assertAlmostEqual(self.row("letter")["eb_recall_term"], np.mean(terms))

    def test_sampler_posterior_and_classes(self):
        german, wdbc, letter = self.row("german"), self.row("wdbc"), self.row("letter")
        # half of the bags of german diverge, P(nu > min n) of Hf is low and a class is perfect on V
        self.assertAlmostEqual(german["hf_share_divergent"], 0.5)
        # one divergent transition in every other bag: 0.5 per bag, of NUTS_DRAWS draws
        for prefix in hf.HIERARCHICAL.values():
            self.assertAlmostEqual(german[f"{prefix}_mean_divergences"], 0.5)
            self.assertAlmostEqual(german[f"{prefix}_share_divergent_draws"], 0.5 / hf.NUTS_DRAWS)
        # the failed bag of BayCAP-Hf on wdbc is left out
        self.assertAlmostEqual(wdbc["hf_mean_divergences"], 0.0)
        self.assertEqual(hf.NUTS_DRAWS, 1000)
        self.assertEqual(german["n_perfect_classes"], 1)
        self.assertEqual(german["failure_class"], "boundary")
        # BayCAP-H does not mix on wdbc, BayCAP-Hf does
        self.assertAlmostEqual(wdbc["h_max_rhat"], 1.3)
        self.assertAlmostEqual(wdbc["hf_max_rhat"], 1.0)
        self.assertEqual(wdbc["failure_class"], "")
        self.assertAlmostEqual(letter["hf_ln_kappa_median"], np.log(0.3))
        self.assertTrue(np.isnan(german["hf_ln_kappa_median"]))
        self.assertAlmostEqual(letter["h_ln_nu_median"], 4.0)
        self.assertAlmostEqual(letter["hf_ln_nu_median"], 2.0)
        self.assertAlmostEqual(letter["empty_off_diagonal_share"], 2 / 6)

    def test_diagnostics_match_eb_prior(self):
        df = self.pair(hf.METHOD, "german")
        expected = eb_prior.diagnostics_table(df).iloc[0]
        self.assertAlmostEqual(self.row("german")["hf_min_ess"], expected["min_ess"])

    def test_summary(self):
        summary = hf.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "sampler", "posterior"})
        overall = summary["sampler"].iloc[0]
        self.assertEqual((overall["K"], overall["pairs"]), ("all", 3))
        self.assertEqual(overall["h_not_mixing"], 1)
        self.assertEqual(overall["hf_not_mixing"], 0)
        self.assertEqual(overall["hf_with_failed_bags"], 1)
        self.assertEqual(overall["boundary"], 1)
        posterior = summary["posterior"].iloc[0]
        self.assertEqual(posterior["kappa_below_1"], 1)

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in hf.TABLES.items():
            with self.subTest(table=name):
                latex = eb_prior._latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("longtable", latex)
                self.assertIn(label, latex)

    def test_figures_build_from_the_csv(self):
        table = pd.read_csv(pd.io.common.StringIO(self.table.to_csv(index=False)))
        for name, plotter in baycap_hf_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(table)
                self.assertGreaterEqual(len(fig.axes), 2)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
