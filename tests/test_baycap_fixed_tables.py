import os
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import baycap_fixed as plots
from uncawe.tables import baycap_fixed as bf
from uncawe.tables.displacement import method_c_prior, prior_part, smoothed_recalls

N_BAGS = 12


def _true_ct(rng, n_classes):
    """A realised table (cap-ml layout) with recalls about 0.8."""
    props = rng.dirichlet(np.full(n_classes, 3.0))
    ct = np.outer(props, np.full(n_classes, 0.2 / (n_classes - 1)))
    np.fill_diagonal(ct, props * 0.8)
    return ct


def _frame(method, dataset, classifier, n_classes, true_cts, rng, centre_shift, width):
    rows = []
    for ct in true_cts:
        acc = float(np.trace(ct))
        samples = acc + centre_shift + width * rng.standard_normal(200)
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "n_classes": n_classes,
                "method": method,
                "true_cts": ct,
                "estim_accs": float(samples.mean()),
                "estim_acc_ranges": samples,
                "acc_err": abs(float(samples.mean()) - acc),
                "acc_sd": width,
                "sigma_g": 2 * width,
                "prev_mean": ct.sum(axis=1),
                "val_ece": 0.05 if "-h" in method.replace("-real", "") else 0.08,
                "nuts_divergences": 0,
                "nuts_max_rhat": 1.001,
                "nuts_min_ess": 400.0,
            }
        )
    return pd.DataFrame(rows)


def _results():
    rng = np.random.default_rng(1)
    pairs = [("wine", "LR", 3), ("spam", "kNN", 2)]
    results = {method: [] for method in bf.METHODS}
    val_counts = {}
    for dataset, classifier, n_classes in pairs:
        true_cts = [_true_ct(rng, n_classes) for _ in range(N_BAGS)]
        counts = np.full((n_classes, n_classes), 5) + 40 * np.eye(n_classes, dtype=int)
        val_counts[(dataset, classifier)] = counts
        for method in bf.METHODS:
            realised = method.endswith("-real") or method == "RQBS"
            width = 0.03 if realised else 0.01
            shift = 0.02 if "EMQ" in method else 0.0
            results[method].append(_frame(method, dataset, classifier, n_classes, true_cts, rng, shift, width))
    return {method: pd.concat(frames, ignore_index=True) for method, frames in results.items()}, val_counts


class FixedCTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.val_counts = _results()
        cls.table = bf.pair_table(cls.results, cls.val_counts)

    def test_methods_and_tests(self):
        self.assertEqual(len(bf.FIXED_METHODS), 8)
        self.assertEqual(len(bf.TESTS), 16)
        self.assertIn(bf.HF, bf.METHODS)
        for cell in bf.CELLS:
            self.assertEqual(bf.TESTS[f"{cell}_real_vs_hf"], (bf.method_name(cell, "realised"), bf.HF))
        # the tests against BayCAP-Hf come last, so the generators of the other twelve are unchanged
        self.assertTrue(all(name.endswith("_real_vs_hf") for name in list(bf.TESTS)[12:]))
        for method, reference in bf.TESTS.values():
            self.assertIn(method, bf.METHODS)
            self.assertIn(reference, bf.METHODS)

    def test_one_row_per_pair_with_scores_and_cells(self):
        self.assertEqual(len(self.table), 2)
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "wine"])  # ordered by K
        for method in bf.METHODS:
            for score in bf.SCORES:
                self.assertTrue(self.table[f"{method}_{score}"].notna().all())
        # realised amplitude is three times the population one; sigma_g = 2 sigma_p predicts sqrt(5)
        for cell in bf.CELLS:
            np.testing.assert_allclose(self.table[f"{cell}_widening"], 3.0, rtol=0.25)
            np.testing.assert_allclose(self.table[f"{cell}_widening_pred"], np.sqrt(5))
            np.testing.assert_allclose(self.table[f"{cell}_mismatch"].to_numpy() >= 0, True)

    def test_mismatch_is_zero_when_the_shares_match(self):
        rates = np.array([[0.9, 0.2], [0.1, 0.8]])  # paper layout: columns sum to one
        prev = np.array([0.3, 0.7])
        ct = (rates * prev).T  # cap-ml layout
        self.assertAlmostEqual(bf.mismatch(ct, rates, prev), 0.0)
        self.assertAlmostEqual(bf.mismatch(ct, rates, np.array([0.5, 0.5])), 2 * abs((rates @ (prev - 0.5))[0]))

    def test_predicted_widening(self):
        np.testing.assert_allclose(bf.predicted_widening([0.01, 0.0], [0.02, 0.02]), [np.sqrt(5), np.inf])

    def test_ece_per_input_classifier(self):
        np.testing.assert_allclose(self.table["h_val_ece"], 0.05)
        np.testing.assert_allclose(self.table["mlp_val_ece"], 0.08)

    def test_tests_are_holm_adjusted(self):
        for name in bf.TESTS:
            p, p_holm = self.table[f"{name}_p"].to_numpy(), self.table[f"{name}_p_holm"].to_numpy()
            self.assertTrue(np.all(p_holm >= p - 1e-12))
        # the EMQ methods are shifted by 0.02, so KDEy has the lower Winkler score
        self.assertTrue((self.table["h_kdey_vs_emq_diff_mean"] < 0).all())

    def test_bags_must_be_paired(self):
        results = dict(self.results)
        broken = results["RQBS"].copy()
        broken.at[0, "true_cts"] = np.diag([0.1, 0.0]) + 0.45
        results["RQBS"] = broken
        with self.assertRaises(ValueError):
            bf.pair_table(results, self.val_counts)

    def test_min_prev_bins_cover_every_bag(self):
        table = bf.min_prev_table(self.results)
        self.assertEqual(set(table["method"]), set(bf.FIXED_METHODS))
        for _, part in table.groupby("method"):
            self.assertEqual(part["n_bags"].sum(), 2 * N_BAGS)

    def test_design_rows_rank_the_four_cells(self):
        columns = [f"{m}_{s}" for m in bf.FIXED_METHODS for s in ("coverage", "amplitude", "winkler")]
        records = self.table[["dataset", "classifier"] + columns]
        rows = pd.DataFrame(bf._design_rows(records, "vanilla_accuracy", "all"))
        self.assertEqual(len(rows), 8)
        for _, part in rows.groupby("pushforward"):
            self.assertAlmostEqual(part["winkler_rank"].sum(), 1 + 2 + 3 + 4)
        latex = bf._design_styler(rows).to_latex()
        self.assertIn("rank", latex)
        # the metric name has no underscore, which LaTeX would reject outside math
        self.assertIn("vanilla accuracy", latex)
        self.assertNotIn("_", latex)

    def test_summary_and_stylers(self):
        summary = bf.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "cells", "tests"})
        self.assertEqual(summary["scores"].iloc[0]["K"], "all")
        # the threshold on the largest split-R-hat is that of E2.8
        self.assertEqual(bf.RHAT_HIGH, 1.1)
        table = self.table.copy()
        table["KDEy-h_max_rhat"] = [1.05, 1.2]
        cells = bf.analysis_summary(table)["cells"]
        overall = cells.loc[(cells["K"] == "all") & (cells["cell"] == "KDEy-h")].iloc[0]
        self.assertEqual(overall["pairs_rhat_high"], 1)
        for styler in (bf._realised_styler, bf._population_styler, bf._pushforward_styler, bf._tests_styler):
            latex = styler(self.table).to_latex()
            self.assertIn("wine", latex.lower())

    def test_every_figure_draws_from_the_csv(self):
        table = pd.read_csv(pd.io.common.StringIO(self.table.to_csv(index=False)))
        # draw each figure, so that the mathtext of every label is parsed
        for name, plotter in plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(table)
                fig.canvas.draw()
                plt.close(fig)
        fig = plots.plot_min_prev(bf.min_prev_table(self.results))
        fig.canvas.draw()
        plt.close(fig)

    def test_fixed_methods_have_no_prior_part(self):
        counts = self.val_counts[("wine", "LR")]
        for method in bf.FIXED_METHODS:
            self.assertEqual(method_c_prior(method), "none")
            smoothed = smoothed_recalls(counts, method_c_prior(method))
            self.assertEqual(prior_part(counts, np.full(3, 1 / 3), smoothed), 0.0)
        self.assertEqual(method_c_prior("BayesCAP"), "uniform")


if __name__ == "__main__":
    unittest.main()
