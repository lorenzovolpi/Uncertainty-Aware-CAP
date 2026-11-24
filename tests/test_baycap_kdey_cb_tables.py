"""Tests of the table and plot modules of E3.5 (work-plan card WP-33) on synthetic results."""

import os
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_baycap_fixed_tables import _true_ct
from test_baycap_kdey_hf_tables import _frame

from uncawe.methods.bayes_kdey import LOO_GRID
from uncawe.plots import baycap_kdey_cb as plots
from uncawe.tables import baycap_kdey_cb as cb

N_BAGS = 12
# the selected bandwidth of each synthetic pair: above 0.1 for wine, at 0.1 for spam
BANDWIDTHS = {"wine": LOO_GRID[13], "spam": LOO_GRID[10]}
SIGMA_W = {cb.CB: 0.012, cb.INST: 0.010}
ACC_SD = {cb.CB: 0.006, cb.INST: 0.003}


def _selection(dataset):
    b = BANDWIDTHS[dataset]
    curve = (np.log10(np.asarray(LOO_GRID)) - np.log10(b)) ** 2 + 0.4
    return {
        "logloss_bandwidth": b,
        "logloss_curve": curve,
        "logloss_selected": 0.4,
        "logloss_default": float(curve[10]),
        "logloss_edge": 0,
        "logloss_classes": 3,
        "loo_ece_default": 0.08,
        "loo_ece_selected": 0.03 if b != LOO_GRID[10] else 0.08,
    }


def _results():
    rng = np.random.default_rng(5)
    pairs = [("wine", "LR", 3), ("spam", "kNN", 2)]
    results = {method: [] for method in cb.METHODS}
    val_counts = {}
    for dataset, classifier, n_classes in pairs:
        true_cts = [_true_ct(rng, n_classes) for _ in range(N_BAGS)]
        counts = np.full((n_classes, n_classes), 5) + 40 * np.eye(n_classes, dtype=int)
        val_counts[(dataset, classifier)] = counts
        for method in cb.METHODS:
            # the new method is centred and 1.5 times as wide as BayCAP-KDEy-h-inst; the others shifted by 0.06
            shift = 0.0 if method == cb.CB else 0.06
            width = 0.03 if method == cb.CB else 0.02
            failed = (3,) if method == cb.HF and dataset == "wine" else ()
            frame = _frame(method, dataset, classifier, n_classes, true_cts, rng, shift, width, counts, failed)
            if method in cb.SUMMARY_METHODS:
                frame = frame.assign(
                    acc_sd=ACC_SD[method],
                    sigma_g=0.015,
                    sigma_w=SIGMA_W[method],
                    sigma_g_tilde=0.014,
                    within_class=0.0001,
                    m_w_sd=0.004,
                    m_sd=0.005,
                    centre_shift=0.002,
                )
            if method == cb.CB:
                selection = _selection(dataset)
                frame = frame.assign(**{k: [v] * len(frame) for k, v in selection.items() if k != "logloss_curve"})
                frame["logloss_curve"] = [selection["logloss_curve"]] * len(frame)
            results[method].append(frame)
    return {method: pd.concat(frames, ignore_index=True) for method, frames in results.items()}, val_counts


class BayCAPKDEyCBTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.val_counts = _results()
        cls.table, cls.curves = cb.pair_tables(cls.results, cls.val_counts)

    def test_methods_and_tests(self):
        self.assertEqual(
            cb.METHODS,
            ["BayCAP-KDEy-h-inst-cb", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-real", "BayesCAP-Hf-real"],
        )
        self.assertEqual(
            cb.TESTS,
            {
                "vs_inst": ("BayCAP-KDEy-h-inst-cb", "BayCAP-KDEy-h-inst"),
                "vs_h_real": ("BayCAP-KDEy-h-inst-cb", "BayCAP-KDEy-h-real"),
            },
        )

    def test_one_row_per_pair_with_scores_and_signed_errors(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "wine"])  # ordered by K
        for method in cb.METHODS:
            for score in cb.SCORES:
                self.assertTrue(self.table[f"{method}_{score}"].notna().all())
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        self.assertAlmostEqual(wine[f"{cb.HF}_share_failed"], 1 / N_BAGS)
        # the references are shifted up by 0.06, the new method is centred
        np.testing.assert_allclose(self.table[f"{cb.INST}_bias"], 0.06, atol=0.01)
        np.testing.assert_allclose(self.table[f"{cb.CB}_bias"], 0.0, atol=0.01)
        np.testing.assert_allclose(self.table[f"{cb.CB}_centre_vs_inst"], -0.06, atol=0.01)

    def test_selection_columns(self):
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        spam = self.table.loc[self.table["dataset"] == "spam"].iloc[0]
        self.assertEqual(wine[f"{cb.CB}_bandwidth"], BANDWIDTHS["wine"])
        self.assertAlmostEqual(wine[f"{cb.CB}_log10_ratio"], 0.3)
        self.assertAlmostEqual(spam[f"{cb.CB}_log10_ratio"], 0.0)
        self.assertEqual(wine[f"{cb.CB}_ece_selected"], 0.03)
        self.assertEqual(wine[f"{cb.CB}_n_loss_classes"], 3)

    def test_curves(self):
        self.assertEqual(len(self.curves), 2 * len(LOO_GRID))
        self.assertEqual(list(dict.fromkeys(self.curves["dataset"])), ["spam", "wine"])
        wine = self.curves.loc[self.curves["dataset"] == "wine"]
        np.testing.assert_allclose(wine["bandwidth"], LOO_GRID)
        np.testing.assert_allclose(wine["logloss"], _selection("wine")["logloss_curve"])

    def test_a_selection_that_varies_between_bags_is_an_error(self):
        results = {method: df.copy() for method, df in self.results.items()}
        results[cb.CB].loc[0, "logloss_bandwidth"] = 0.5
        with self.assertRaises(ValueError):
            cb.pair_tables(results, self.val_counts)

    def test_width_columns(self):
        np.testing.assert_allclose(self.table[f"{cb.CB}_ratio_vs_inst"], 1.5, rtol=0.15)
        np.testing.assert_allclose(self.table[f"{cb.CB}_sigma_w"], SIGMA_W[cb.CB])
        np.testing.assert_allclose(self.table[f"{cb.INST}_sigma_w"], SIGMA_W[cb.INST])
        np.testing.assert_allclose(self.table[f"{cb.CB}_acc_sd"], ACC_SD[cb.CB])
        np.testing.assert_allclose(self.table[f"{cb.INST}_acc_sd"], ACC_SD[cb.INST])
        spam = self.table.loc[self.table["dataset"] == "spam"].iloc[0]
        self.assertAlmostEqual(spam[f"{cb.CB}_share_divergent"], 1 / N_BAGS)

    def test_tests_favour_the_centred_method(self):
        for name in cb.TESTS:
            self.assertTrue((self.table[f"{name}_diff_mean"] < 0).all())
            self.assertTrue(np.all(self.table[f"{name}_p_holm"] >= self.table[f"{name}_p"] - 1e-12))

    def test_summary_and_stylers(self):
        summary = cb.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "selection", "widths", "tests"})
        selection = summary["selection"].set_index("K")
        self.assertEqual(selection.loc["all", "pairs_above"], 1)
        self.assertEqual(selection.loc["all", "pairs_at"], 1)
        self.assertEqual(selection.loc["all", "pairs_below"], 0)
        self.assertEqual(selection.loc["all", "pairs_ece_lower"], 1)
        for styler in (
            cb._comparison_styler(self.table),
            cb._selection_styler(self.table),
            cb._tests_styler(self.table),
        ):
            self.assertIn("\\begin{tabular}", styler.to_latex())
        comparison = cb._comparison_styler(self.table).data
        self.assertEqual(comparison.shape[1], 4 * len(cb.METHODS))

    def test_every_figure_draws_from_the_csvs(self):
        with tempfile.TemporaryDirectory() as directory:
            self.table.to_csv(cb.csv_path(directory), index=False)
            self.curves.to_csv(cb.curves_path(directory), index=False)
            table, curves = plots.load_tables(path=cb.csv_path(directory), curves=cb.curves_path(directory))
        self.assertEqual(set(plots.FIGURES), set(plots.PLOTTERS))
        for name, plotter in plots.PLOTTERS.items():
            with self.subTest(figure=name):
                plt.close(plotter(table, curves))


if __name__ == "__main__":
    unittest.main()
