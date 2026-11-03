import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import baycap_hfn as baycap_hfn_plots
from uncawe.tables import baycap_hf as hf
from uncawe.tables import baycap_hfn as hfn
from uncawe.tables import eb_prior

# cap-ml layout: true classes on rows; nursery has a class with no observed mistake (row 0)
VAL_COUNTS = {
    ("wdbc", "LR"): np.array([[90, 10], [30, 70]]),
    ("letter", "LR"): np.array([[45, 5, 0], [4, 40, 6], [0, 3, 47]]),
    ("nursery", "LR"): np.array([[50, 0, 0], [5, 40, 5], [0, 10, 40]]),
}
PAIRS = [("letter", "LR"), ("nursery", "LR"), ("wdbc", "LR")]
N_BAGS = 4
FAILED = ("letter", "LR", 1)  # the bag of BayCAP-Hfn where NUTS produced no draws
SHIFTS = {
    "BayesCAP-real": -0.05,
    "BayesCAP-EB-real": 0.0,
    "BayesCAP-H-real": 0.03,
    hf.METHOD: 0.01,
    hfn.METHOD: 0.01,
}
# divergent transitions per bag: BayCAP-Hf diverges on nursery (30% of the draws), BayCAP-Hfn does not
DIVERGENCES = {(hf.METHOD, "nursery"): 300, (hfn.METHOD, "nursery"): 2}
TIMES = {hf.METHOD: 2.0, hfn.METHOD: 3.0}


def results(method: str, seed: int) -> pd.DataFrame:
    """Stored results of one method: the same bags for every method, a mean table moved by the shift of the method on
    the diagonal, accuracy draws around its trace, and the diagnostics and summaries of the hierarchical methods."""
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
            # BayCAP-Hf and BayCAP-Hfn draw the same accuracies on letter, so their scores tie there
            draw_rng = np.random.default_rng(bag) if dataset == "letter" and method in hfn.SAMPLERS else rng
            row = {
                "dataset": dataset,
                "classifier": classifier,
                "n_classes": k,
                "true_cts": true_ct,
                "estim_cts": estim_ct,
                "estim_acc_ranges": np.trace(estim_ct) + draw_rng.normal(0, 0.05, 200),
                "acc_err": abs(np.trace(estim_ct) - np.trace(true_ct)),
                "nuts_divergences": DIVERGENCES.get((method, dataset), 0),
                "nuts_max_rhat": 1.0,
                "nuts_min_ess": 400.0,
                "t_test_ave": TIMES.get(method, 1.0),
            }
            if method in (*hfn.SAMPLERS, hf.H_METHOD):
                row.update(
                    {
                        "nuts_failed": 0,
                        "h_m_mean": 0.8,
                        "h_ln_nu_median": 2.0,
                        "h_ln_nu_q05": 0.5,
                        "h_ln_nu_q95": 4.0,
                        "h_p_nu_above_min_n": 0.01,
                    }
                )
            if method in hfn.SAMPLERS:
                ln_kappa = np.log(0.3 if method == hf.METHOD else 0.25) if k > 2 else np.nan
                row.update(
                    {
                        "h_ln_kappa_median": ln_kappa,
                        "h_ln_kappa_q05": ln_kappa - 1,
                        "h_ln_kappa_q95": ln_kappa + 1,
                        "h_p_kappa_below_1": 0.9 if k > 2 else np.nan,
                    }
                )
            if method == hfn.METHOD and (dataset, classifier, bag) == FAILED:
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
                        "h_ln_kappa_median": np.nan,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_table() -> pd.DataFrame:
    stored = {method: results(method, seed=i) for i, method in enumerate(hfn.METHODS)}
    with (
        patch.object(hfn, "load_results", side_effect=lambda method, experiment, domain: stored[method]),
        patch.object(hfn, "load_validation_counts", side_effect=lambda pairs, domain: VAL_COUNTS),
    ):
        return hfn.compute_table()


class FunctionsTest(unittest.TestCase):
    def test_mistake_indicators(self):
        # letter: 5, 10 and 3 mistakes over K-1 = 2 cells; nursery: a class with none
        self.assertEqual(
            hfn.mistake_indicators(VAL_COUNTS[("letter", "LR")]),
            {"n_no_mistake_columns": 0, "median_mistakes_per_cell": 2.5},
        )
        self.assertEqual(
            hfn.mistake_indicators(VAL_COUNTS[("nursery", "LR")]),
            {"n_no_mistake_columns": 1, "median_mistakes_per_cell": 5.0},
        )

    def test_check_table_and_summary(self):
        hf_table = pd.DataFrame(
            {
                "dataset": ["letter", "nursery", "wdbc"],
                "classifier": ["LR"] * 3,
                "n_classes": [3, 3, 2],
                "min_val_per_class": [50, 50, 100],
                "n_perfect_classes": [0, 1, 0],
                "n_never_right_classes": [0, 0, 0],
                "hf_share_divergent_draws": [0.0, 0.3, 0.0],
                "failure_class": [np.nan, "boundary", np.nan],
            }
        )
        check = hfn.check_table(hf_table, VAL_COUNTS)
        # only the pairs with K >= 3
        self.assertEqual(list(check["dataset"]), ["letter", "nursery"])
        self.assertEqual(list(check["hf_failure_class"]), ["", "boundary"])
        summary = hfn.check_summary(check).set_index("indicator")
        self.assertEqual(set(summary.index), set(hfn.CHECK_INDICATORS))
        self.assertAlmostEqual(summary.loc["n_no_mistake_columns", "spearman_with_divergent_draws"], 1.0)
        self.assertAlmostEqual(summary.loc["n_no_mistake_columns", "mean_boundary"], 1.0)
        self.assertAlmostEqual(summary.loc["n_no_mistake_columns", "mean_other"], 0.0)


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = synthetic_table()

    def row(self, dataset: str) -> pd.Series:
        return self.table.loc[self.table["dataset"] == dataset].iloc[0]

    def test_rows_and_columns(self):
        self.assertEqual(len(self.table), 3)
        # ordered by K
        self.assertEqual(self.table["n_classes"].tolist()[0], 2)
        for method in hfn.METHODS:
            for score in hfn.SCORES:
                self.assertIn(f"{method}_{score}", self.table)
        for score in hfn.TESTED_SCORES:
            for stat in ("diff_mean", "ci_low", "ci_high", "p", "p_holm"):
                self.assertIn(hfn.test_column(score, stat), self.table)

    def test_failed_bags_are_left_out(self):
        letter = self.row("letter")
        self.assertAlmostEqual(letter["hfn_share_failed"], 1 / N_BAGS)
        self.assertAlmostEqual(letter["hf_share_failed"], 0.0)
        # the same accuracy draws on the bags where both have draws: the paired tests find no difference
        self.assertAlmostEqual(letter[hfn.test_column("winkler", "diff_mean")], 0.0)
        self.assertAlmostEqual(letter[hfn.test_column("winkler", "p")], 1.0)

    def test_sampler_classes_and_times(self):
        nursery, letter = self.row("nursery"), self.row("letter")
        self.assertAlmostEqual(nursery["hf_share_divergent_draws"], 300 / hf.NUTS_DRAWS)
        self.assertEqual(nursery["hf_failure_class"], "boundary")
        self.assertEqual(nursery["hfn_failure_class"], "")
        self.assertEqual(letter["hfn_failure_class"], "")
        self.assertAlmostEqual(nursery["hfn_time_per_bag"], 3.0)
        self.assertAlmostEqual(nursery["hf_time_per_bag"], 2.0)
        self.assertEqual(nursery["n_no_mistake_columns"], 1)
        self.assertAlmostEqual(letter["hfn_ln_kappa_median"], np.log(0.25))
        self.assertTrue(np.isnan(self.row("wdbc")["hfn_ln_kappa_median"]))

    def test_diagnostics_match_eb_prior(self):
        df = results(hfn.METHOD, seed=4)
        df = df.loc[df["dataset"] == "nursery"]
        self.assertAlmostEqual(self.row("nursery")["hfn_max_rhat"], eb_prior.diagnostics_table(df).iloc[0]["max_rhat"])

    def test_summary(self):
        summary = hfn.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "sampler", "posterior"})
        sampler = summary["sampler"].set_index("pairs")
        self.assertEqual(sampler.loc["all", "hf_unreliable"], 1)
        self.assertEqual(sampler.loc["all", "hfn_unreliable"], 0)
        self.assertEqual(sampler.loc["all", "became_reliable"], 1)
        self.assertEqual(sampler.loc["all", "became_unreliable"], 0)
        self.assertEqual(sampler.loc["boundary_hf", "n"], 1)
        self.assertEqual(sampler.loc["reliable_hf", "n"], 2)
        scores = summary["scores"].set_index("pairs")
        self.assertEqual(scores.loc["all", "n"], 3)
        posterior = summary["posterior"].iloc[0]
        # letter is the only pair with kappa reliable under both: |ln 0.25 - ln 0.3|
        self.assertAlmostEqual(posterior["ln_kappa_median_max_abs_diff"], abs(np.log(0.25) - np.log(0.3)))
        # the summaries also work on the CSV read back
        read_back = pd.read_csv(pd.io.common.StringIO(self.table.to_csv(index=False)))
        self.assertEqual(hfn.analysis_summary(read_back)["sampler"].iloc[0]["became_reliable"], 1)

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in hfn.TABLES.items():
            with self.subTest(table=name):
                latex = eb_prior._latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("longtable", latex)
                self.assertIn(label, latex)

    def test_figures_build_from_the_csv(self):
        table = pd.read_csv(pd.io.common.StringIO(self.table.to_csv(index=False)))
        check = hfn.check_table(
            pd.DataFrame(
                {
                    "dataset": ["letter", "nursery"],
                    "classifier": ["LR", "LR"],
                    "n_classes": [3, 3],
                    "min_val_per_class": [50, 50],
                    "n_perfect_classes": [0, 1],
                    "n_never_right_classes": [0, 0],
                    "hf_share_divergent_draws": [0.001, 0.3],
                    "failure_class": [np.nan, "boundary"],
                }
            ),
            VAL_COUNTS,
        )
        for name, plotter in baycap_hfn_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(check if name == "check" else table)
                self.assertGreaterEqual(len(fig.axes), 2)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
