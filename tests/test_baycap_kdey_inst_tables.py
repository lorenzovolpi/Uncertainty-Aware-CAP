"""Tests of the table and plot modules of E3.4 (work-plan card WP-31) on synthetic results."""

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

from uncawe.plots import baycap_kdey_inst as plots
from uncawe.tables import baycap_kdey_inst as bi

N_BAGS = 12
# the label-free summaries of BayCAP-KDEy-h-inst in the synthetic results
SIGMA_W, SIGMA_G, SIGMA_G_TILDE, M_W_SD, M_SD = 0.010, 0.015, 0.014, 0.004, 0.005
WITHIN = SIGMA_G_TILDE**2 - SIGMA_W**2


def _results():
    rng = np.random.default_rng(3)
    pairs = [("wine", "LR", 3), ("spam", "kNN", 2)]
    results = {method: [] for method in bi.METHODS}
    val_counts = {}
    ratio = bi.predicted_ratio(M_W_SD, SIGMA_W, M_SD, SIGMA_G)
    for dataset, classifier, n_classes in pairs:
        true_cts = [_true_ct(rng, n_classes) for _ in range(N_BAGS)]
        counts = np.full((n_classes, n_classes), 5) + 40 * np.eye(n_classes, dtype=int)
        val_counts[(dataset, classifier)] = counts
        for method in bi.METHODS:
            # the per-instance method is centred, its references shifted by 0.06; its width is that of
            # BayCAP-KDEy-h-real times the predicted ratio
            shift = 0.0 if method == bi.INST else 0.06
            width = 0.02 * (ratio if method == bi.INST else 1.0)
            failed = (3,) if method == bi.HF and dataset == "wine" else ()
            frame = _frame(method, dataset, classifier, n_classes, true_cts, rng, shift, width, counts, failed)
            if method == bi.INST:
                frame = frame.assign(
                    acc_sd=0.003,
                    sigma_g=SIGMA_G,
                    sigma_w=SIGMA_W,
                    sigma_g_tilde=SIGMA_G_TILDE,
                    within_class=WITHIN,
                    m_w_sd=M_W_SD,
                    m_sd=M_SD,
                    centre_shift=0.01 if n_classes > 2 else -0.002,
                )
            results[method].append(frame)
    return {method: pd.concat(frames, ignore_index=True) for method, frames in results.items()}, val_counts


class BayCAPKDEyInstTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.val_counts = _results()
        cls.table = bi.pair_table(cls.results, cls.val_counts)

    def test_methods_and_tests(self):
        self.assertEqual(
            bi.METHODS, ["BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-real", "BayCAP-KDEy-mlp-real", "BayesCAP-Hf-real"]
        )
        self.assertEqual(
            bi.TESTS,
            {
                "vs_h_real": ("BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-real"),
                "vs_mlp_real": ("BayCAP-KDEy-h-inst", "BayCAP-KDEy-mlp-real"),
            },
        )

    def test_one_row_per_pair_with_scores(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "wine"])  # ordered by K
        for method in bi.METHODS:
            for score in bi.SCORES:
                self.assertTrue(self.table[f"{method}_{score}"].notna().all())
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        self.assertAlmostEqual(wine[f"{bi.HF}_share_failed"], 1 / N_BAGS)

    def test_predicted_ratio(self):
        np.testing.assert_allclose(bi.predicted_ratio([0.0, 3.0], [0.0, 0.0], [0.0, 0.0], [0.0, 5.0]), [np.nan, 0.6])

    def test_width_columns(self):
        expected = bi.predicted_ratio(M_W_SD, SIGMA_W, M_SD, SIGMA_G)
        np.testing.assert_allclose(self.table[f"{bi.INST}_ratio_predicted"], expected)
        # the synthetic amplitudes follow the predicted ratio up to the noise of 200 samples
        np.testing.assert_allclose(self.table[f"{bi.INST}_ratio_observed"], expected, rtol=0.15)
        np.testing.assert_allclose(self.table[f"{bi.INST}_within_share"], WITHIN / SIGMA_G_TILDE**2)
        np.testing.assert_allclose(self.table[f"{bi.INST}_share_sigma_w_below"], 1.0)
        np.testing.assert_allclose(self.table[f"{bi.INST}_prev_mean_max_diff"], 0.0)
        np.testing.assert_allclose(self.table[f"{bi.INST}_sigma_w"], SIGMA_W)
        spam = self.table.loc[self.table["dataset"] == "spam"].iloc[0]
        self.assertAlmostEqual(spam[f"{bi.INST}_centre_shift"], -0.002)
        self.assertAlmostEqual(spam[f"{bi.INST}_share_divergent"], 1 / N_BAGS)

    def test_tests_favour_the_centred_method(self):
        for name in bi.TESTS:
            self.assertTrue((self.table[f"{name}_diff_mean"] < 0).all())
            self.assertTrue(np.all(self.table[f"{name}_p_holm"] >= self.table[f"{name}_p"] - 1e-12))

    def test_summary_and_stylers(self):
        summary = bi.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "tests", "widths"})
        widths = summary["widths"].set_index("K")
        self.assertEqual(widths.loc["all", "pairs_ratio_below_1"], 2)
        for styler in (bi._comparison_styler(self.table), bi._widths_styler(self.table), bi._tests_styler(self.table)):
            self.assertIn("\\begin{tabular}", styler.to_latex())
        comparison = bi._comparison_styler(self.table).data
        self.assertEqual(len(dict.fromkeys(comparison.columns.get_level_values(0))), 4)
        self.assertEqual(comparison.shape[1], 4 * len(bi.METHODS))

    def test_every_figure_draws_from_the_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "table.csv")
            self.table.to_csv(path, index=False)
            table = plots.load_table(path=path)
        self.assertEqual(set(plots.FIGURES), set(plots.PLOTTERS))
        for name, plotter in plots.PLOTTERS.items():
            with self.subTest(figure=name):
                plt.close(plotter(table))


if __name__ == "__main__":
    unittest.main()
