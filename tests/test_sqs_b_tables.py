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

from uncawe.plots import sqs_b as plots
from uncawe.tables import sqs_b as sb

N_BAGS = 12
# the label-free summaries of SQS-B in the synthetic results: sigma_p, sigma_f, sigma_g
SIGMA_P, SIGMA_F, SIGMA_G = 0.02, 0.012, 0.016


def _results():
    rng = np.random.default_rng(3)
    pairs = [("wine", "LR", 3), ("spam", "kNN", 2)]
    results = {method: [] for method in sb.METHODS}
    val_counts = {}
    for dataset, classifier, n_classes in pairs:
        true_cts = [_true_ct(rng, n_classes) for _ in range(N_BAGS)]
        counts = np.full((n_classes, n_classes), 5) + 40 * np.eye(n_classes, dtype=int)
        val_counts[(dataset, classifier)] = counts
        for method in sb.METHODS:
            # SQS-B is centred, its references shifted by 0.06 (three standard deviations); the width of SQS-B is that
            # of BayCAP-KDEy-mlp-real times the predicted ratio
            shift = 0.0 if method == sb.SQS_B else 0.06
            width = 0.02 * (sb.predicted_ratio(SIGMA_P, SIGMA_F, SIGMA_G) if method == sb.SQS_B else 1.0)
            failed = (3,) if method == sb.HF and dataset == "wine" else ()
            frame = _frame(method, dataset, classifier, n_classes, true_cts, rng, shift, width, counts, failed)
            if method == sb.SQS_B:
                frame = frame.assign(
                    acc_sd=SIGMA_P,
                    sigma_f=SIGMA_F,
                    sigma_g=SIGMA_G,
                    s_rec=0.01,
                    s_prec=0.004 if n_classes > 2 else 0.02,
                )
            results[method].append(frame)
    return {method: pd.concat(frames, ignore_index=True) for method, frames in results.items()}, val_counts


class SQSBTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.val_counts = _results()
        cls.table = sb.pair_table(cls.results, cls.val_counts)

    def test_methods_and_tests(self):
        self.assertEqual(sb.INTERVAL_METHODS, ["SQS-B", "BayCAP-KDEy-mlp-real", "SQS-Hbs", "SQS-H", "BayesCAP-Hf-real"])
        self.assertEqual(sb.METHODS[-1], "BayCAP-KDEy-mlp")
        self.assertEqual(sb.TESTS, {"vs_kdey": ("SQS-B", "BayCAP-KDEy-mlp-real"), "vs_sqs_hbs": ("SQS-B", "SQS-Hbs")})

    def test_one_row_per_pair_with_scores(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "wine"])  # ordered by K
        for method in sb.METHODS:
            for score in sb.SCORES:
                self.assertTrue(self.table[f"{method}_{score}"].notna().all())
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        self.assertAlmostEqual(wine[f"{sb.HF}_share_failed"], 1 / N_BAGS)

    def test_predicted_ratio(self):
        np.testing.assert_allclose(sb.predicted_ratio([0.0, 3.0], [0.0, 4.0], [0.0, 0.0]), [np.nan, 5 / 3])

    def test_width_columns(self):
        expected = np.sqrt((SIGMA_P**2 + SIGMA_F**2) / (SIGMA_P**2 + SIGMA_G**2))
        np.testing.assert_allclose(self.table[f"{sb.SQS_B}_ratio_predicted"], expected)
        # the synthetic amplitudes follow the predicted ratio up to the noise of 200 samples
        np.testing.assert_allclose(self.table[f"{sb.SQS_B}_ratio_observed"], expected, rtol=0.15)
        np.testing.assert_allclose(self.table[f"{sb.SQS_B}_sigma_f"], SIGMA_F)
        np.testing.assert_allclose(self.table[f"{sb.SQS_B}_prev_mean_max_diff"], 0.0)
        spam = self.table.loc[self.table["dataset"] == "spam"].iloc[0]
        self.assertAlmostEqual(spam[f"{sb.SQS_B}_s_diff"], 0.01)
        self.assertAlmostEqual(spam[f"{sb.SQS_B}_share_divergent"], 1 / N_BAGS)

    def test_tests_favour_the_centred_method(self):
        for name in sb.TESTS:
            self.assertTrue((self.table[f"{name}_diff_mean"] < 0).all())
            self.assertTrue(np.all(self.table[f"{name}_p_holm"] >= self.table[f"{name}_p"] - 1e-12))

    def test_summary_and_stylers(self):
        summary = sb.analysis_summary(self.table)
        self.assertEqual(set(summary), {"scores", "tests", "widths"})
        widths = summary["widths"].set_index("K")
        self.assertEqual(widths.loc["all", "pairs_s_prec_larger"], 1)
        for styler in (sb._comparison_styler(self.table), sb._widths_styler(self.table), sb._tests_styler(self.table)):
            self.assertIn("\\begin{tabular}", styler.to_latex())
        groups = list(dict.fromkeys(sb._comparison_styler(self.table).data.columns.get_level_values(0)))
        self.assertEqual(len(groups), 4)
        self.assertEqual(sb._comparison_styler(self.table).data.shape[1], 3 + 3 * len(sb.INTERVAL_METHODS))

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
