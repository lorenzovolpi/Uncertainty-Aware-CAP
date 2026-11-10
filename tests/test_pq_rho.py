"""The table and the plot of E6.2 (card WP-17) on synthetic results of the five ``PQ-rho*`` methods."""

import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import pq_rho as plots
from uncawe.results import RDF
from uncawe.tables import pq_rho as tables

LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]
N_PAIRS = 12
N_BAGS = 4
TRUE_CT = np.array([[30.0, 10.0], [20.0, 40.0]])  # true classes on rows: prevalence (0.4, 0.6), accuracy 0.7


def synthetic_results(missing=(("d11", "kNN", 0.1),)) -> pd.DataFrame:
    """``N_PAIRS`` binary pairs: the absolute error grows with rho (by a factor that differs between pairs), every
    other stored quantity is the same at every level; the (pair, level) in ``missing`` failed."""
    rows = []
    for i in range(N_PAIRS):
        dataset, classifier = f"d{i:02d}", "kNN" if i % 2 else "LR"
        for rho in LEVELS:
            if (dataset, classifier, rho) in missing:
                continue
            for _ in range(N_BAGS):
                rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "method": f"PQ-rho{rho}",
                        "n_classes": 2,
                        "acc_name": "vanilla_accuracy",
                        "true_cts": TRUE_CT.copy(),
                        "estim_cts": TRUE_CT / TRUE_CT.sum(),
                        "estim_acc_ranges": np.linspace(0.6, 0.8, 41),
                        "acc_err": 0.01 * i + rho * (1 + 0.01 * i),
                        "pq_test_prev": [0.5, 0.5],
                        "pq_min_class_support": 0.3 + 0.01 * i,
                    }
                )
    return pd.DataFrame(rows)


class PQRhoTableTest(unittest.TestCase):
    def setUp(self):
        # the pair (d11, kNN) has no result at rho=0.1: a failure
        self.df = synthetic_results(missing=(("d11", "kNN", 0.1),))
        self.pairs, self.failed, self.summary = tables.compute("vanilla_accuracy", self.df)

    def test_quantification_error_from_the_true_table(self):
        self.assertAlmostEqual(tables.quant_error([0.5, 0.5], TRUE_CT), 0.1)
        self.assertTrue(np.allclose(self.pairs["quant_error"], 0.1))

    def test_pair_means_over_the_bags(self):
        self.assertEqual(len(self.pairs), N_PAIRS * len(LEVELS) - 1)
        self.assertTrue((self.pairs["n_bags"] == N_BAGS).all())
        row = self.pairs.loc[(self.pairs["dataset"] == "d03") & (self.pairs["rho"] == 0.7)].iloc[0]
        self.assertAlmostEqual(row["acc_err"], 0.03 + 0.7 * 1.03)
        self.assertAlmostEqual(row["class_support"], 0.33)
        self.assertAlmostEqual(row["coverage"], 1.0)

    def test_failures_are_listed_and_left_out_of_the_summary(self):
        self.assertEqual(self.failed.to_dict("records"), [{"dataset": "d11", "classifier": "kNN", "rho": 0.1}])
        self.assertFalse(self.pairs.loc[self.pairs["dataset"] == "d11", "common"].any())
        self.assertEqual(int(self.pairs.loc[self.pairs["common"], "dataset"].nunique()), N_PAIRS - 1)
        self.assertTrue((self.summary["n_pairs"] == N_PAIRS - 1).all())

    def test_summary_means_over_the_common_pairs(self):
        row = self.summary.set_index("measure").loc["acc_err"]
        common = [i for i in range(N_PAIRS) if i != 11]
        for rho in LEVELS:
            expected = np.mean([0.01 * i + rho * (1 + 0.01 * i) for i in common])
            self.assertAlmostEqual(row[f"mean_{rho}"], expected)

    def test_holm_adjusted_wilcoxon_against_rho_half(self):
        summary = self.summary.set_index("measure")
        for rho in (0.1, 0.3, 0.7, 0.9):
            # the error moves with rho on every pair: significant; the other measures do not move: p=1
            self.assertLess(summary.loc["acc_err", f"p_holm_{rho}"], tables.ALPHA)
            self.assertGreaterEqual(summary.loc["acc_err", f"p_holm_{rho}"], summary.loc["acc_err", f"p_{rho}"])
            for measure in ("quant_error", "coverage", "amplitude", "winkler", "class_support"):
                self.assertEqual(summary.loc[measure, f"p_holm_{rho}"], 1.0)
        self.assertNotIn("p_0.5", self.summary.columns)

    def test_latex_marks_the_significant_cells(self):
        latex = tables.summary_latex({"vanilla_accuracy": self.summary})
        acc_err_line = next(line for line in latex.splitlines() if "AE" in line)
        self.assertEqual(acc_err_line.count(tables.MARK), 4)
        winkler_line = next(line for line in latex.splitlines() if "Winkler" in line)
        self.assertNotIn(tables.MARK, winkler_line)
        self.assertIn("Failed (pair, level): 1", tables.caption({"vanilla_accuracy": self.summary}, 1))

    def test_load_results_reads_the_stored_pq_rho_files(self):
        with tempfile.TemporaryDirectory() as root:
            sub = self.df.loc[self.df["dataset"].isin(["d00", "d01"])]
            for (dataset, classifier, method), frame in sub.groupby(["dataset", "classifier", "method"]):
                folder = os.path.join(root, "main", "classic", "vanilla_accuracy", dataset, classifier)
                os.makedirs(folder, exist_ok=True)
                RDF.from_df(frame.reset_index(drop=True)).save_result(os.path.join(folder, f"{method}.parquet"))
            # a method outside E6.2 is not read
            RDF.from_df(frame.assign(method="PQ").reset_index(drop=True)).save_result(
                os.path.join(folder, "PQ.parquet")
            )
            loaded = tables.load_results("vanilla_accuracy", root=root)
            self.assertEqual(sorted(loaded["method"].unique()), [f"PQ-rho{rho}" for rho in LEVELS])
            self.assertEqual(len(loaded), 2 * len(LEVELS) * N_BAGS)
            np.testing.assert_allclose(np.stack(loaded["true_cts"].to_numpy()), np.stack(sub["true_cts"].to_numpy()))
            self.assertTrue(tables.load_results("macro-F1", root=root).empty)


class PQRhoPlotTest(unittest.TestCase):
    def test_one_panel_per_measure_with_a_line_per_pair_and_the_mean(self):
        pairs, _, _ = tables.compute("vanilla_accuracy", synthetic_results())
        fig = plots.plot_metric(pairs.assign(acc_name="vanilla_accuracy"), "vanilla_accuracy")
        axes = fig.axes
        self.assertEqual(len(axes), len(tables.MEASURES))
        for ax in axes:
            # one line per pair, the bold mean, and the nominal line on the coverage panel
            expected = N_PAIRS + 1 + (ax.get_title() == plots.PANEL_LABELS["coverage"])
            self.assertEqual(len(ax.get_lines()), expected)
        mean_line = axes[list(tables.MEASURES).index("acc_err")].get_lines()[N_PAIRS]
        np.testing.assert_allclose(mean_line.get_xdata(), LEVELS)
        common = [i for i in range(N_PAIRS) if i != 11]
        np.testing.assert_allclose(
            mean_line.get_ydata(), [np.mean([0.01 * i + rho * (1 + 0.01 * i) for i in common]) for rho in LEVELS]
        )
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
