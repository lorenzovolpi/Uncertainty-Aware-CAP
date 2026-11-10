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

from uncawe.plots import baycap_kdey_hf as plots
from uncawe.tables import baycap_kdey_hf as kh
from uncawe.tables.displacement import METHOD_CHOICES, method_c_prior, smoothed_recalls

N_BAGS = 12


def _frame(method, dataset, classifier, n_classes, true_cts, rng, centre_shift, width, val_counts, failed=()):
    rows = []
    for bag, ct in enumerate(true_cts):
        acc = float(np.trace(ct))
        samples = acc + centre_shift + width * rng.standard_normal(200)
        if bag in failed:
            samples = np.full(200, np.nan)
        row = {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": n_classes,
            "method": method,
            "true_cts": ct,
            "estim_cts": ct,
            "estim_accs": float(np.mean(samples)),
            "estim_acc_ranges": samples,
            "acc_err": abs(float(np.mean(samples)) - acc),
            "nuts_divergences": 1 if bag == 0 else 0,
            "nuts_max_rhat": 1.001,
            "nuts_min_ess": 400.0,
            "prev_mean": ct.sum(axis=1),
            "val_ece": 0.05,
        }
        if method in kh.NEW.values():
            # the posterior means reproduce the predicted shares of the bag: no mismatch
            n_s = val_counts.sum(axis=1)
            c_hat = val_counts / n_s[:, None]
            c_mean = np.divide(ct, ct.sum(axis=1, keepdims=True))
            row |= {
                "c_mean": c_mean.ravel(),
                "c_sd": 0.5 * np.sqrt(c_hat * (1 - c_hat) / (n_s[:, None] + 1)).ravel(),
                "h_pi_mean": ct.sum(axis=1),
                "n_empty_cells": 0,
                "n_class_density_cells": 0,
                "h_ln_nu_median": 2.0,
                "h_ln_kappa_median": 0.5 if n_classes > 2 else np.nan,
                "h_p_kappa_below_1": 0.3 if n_classes > 2 else np.nan,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _results():
    rng = np.random.default_rng(2)
    pairs = [("wine", "LR", 3), ("spam", "kNN", 2)]
    results = {method: [] for method in kh.METHODS}
    val_counts = {}
    for dataset, classifier, n_classes in pairs:
        true_cts = [_true_ct(rng, n_classes) for _ in range(N_BAGS)]
        counts = np.full((n_classes, n_classes), 5) + 40 * np.eye(n_classes, dtype=int)
        val_counts[(dataset, classifier)] = counts
        for method in kh.METHODS:
            # the new methods are centred, the references shifted by 0.06 (three standard deviations)
            shift = 0.0 if method in kh.NEW.values() else 0.06
            failed = (3,) if method == kh.NEW["h"] and dataset == "wine" else ()
            frame = _frame(method, dataset, classifier, n_classes, true_cts, rng, shift, 0.02, counts, failed)
            results[method].append(frame)
    return {method: pd.concat(frames, ignore_index=True) for method, frames in results.items()}, val_counts


class KDEyHfTablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results, cls.val_counts = _results()
        cls.table = kh.pair_table(cls.results, cls.val_counts)
        cls.cells = kh.cell_table(cls.results, cls.val_counts)

    def test_methods_and_tests(self):
        self.assertEqual(len(kh.METHODS), 6)
        self.assertEqual(
            kh.TESTS,
            {
                "h_vs_fixed": ("BayCAP-KDEy-Hf-h-real", "BayCAP-KDEy-h-real"),
                "h_vs_hf": ("BayCAP-KDEy-Hf-h-real", "BayesCAP-Hf-real"),
                "mlp_vs_fixed": ("BayCAP-KDEy-Hf-mlp-real", "BayCAP-KDEy-mlp-real"),
                "mlp_vs_hf": ("BayCAP-KDEy-Hf-mlp-real", "BayesCAP-Hf-real"),
            },
        )

    def test_one_row_per_pair_with_scores(self):
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "wine"])  # ordered by K
        for method in kh.METHODS:
            for score in (*kh.SCORES, "std_shift"):
                self.assertTrue(self.table[f"{method}_{score}"].notna().all())
        # the references are shifted by three standard deviations, the new methods are centred
        np.testing.assert_allclose(self.table["SQS-Hbs_std_shift"], 3.0, atol=0.3)
        np.testing.assert_allclose(self.table[f"{kh.NEW['mlp']}_std_shift"], 0.0, atol=0.2)

    def test_failed_bags_are_left_out(self):
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        self.assertAlmostEqual(wine[f"{kh.NEW['h']}_share_failed"], 1 / N_BAGS)
        self.assertTrue(np.isfinite(wine[f"{kh.NEW['h']}_winkler"]))
        self.assertTrue(np.isfinite(wine["h_vs_fixed_p"]))

    def test_sampler_columns(self):
        for source in kh.SOURCES:
            new = kh.NEW[source]
            np.testing.assert_allclose(self.table[f"{new}_mismatch"], 0.0, atol=1e-12)
            np.testing.assert_allclose(self.table[f"{new}_share_divergent"].iloc[0], 1 / N_BAGS)
            self.assertTrue((self.table[f"{kh.FIXED[source]}_mismatch"] >= 0).all())
        spam = self.table.loc[self.table["dataset"] == "spam"].iloc[0]
        self.assertTrue(np.isnan(spam[f"{kh.NEW['h']}_h_ln_kappa_median"]))

    def test_tests_favour_the_centred_methods(self):
        for name in kh.TESTS:
            self.assertTrue((self.table[f"{name}_diff_mean"] < 0).all())
            self.assertTrue(np.all(self.table[f"{name}_p_holm"] >= self.table[f"{name}_p"] - 1e-12))

    def test_mismatch_hf(self):
        ct = np.array([[0.27, 0.03], [0.14, 0.56]])  # cap-ml layout
        pi = ct.sum(axis=1)
        c_mean = ct / pi[:, None]
        self.assertAlmostEqual(kh.mismatch_hf(ct, c_mean.ravel(), pi), 0.0)

    def test_cells_halve_the_sd(self):
        self.assertEqual(len(self.cells), 2 * (9 + 4))
        np.testing.assert_allclose(self.cells["sd_ratio"], 0.5)
        np.testing.assert_allclose(
            self.cells["sd_v"], np.sqrt(self.cells["c_hat"] * (1 - self.cells["c_hat"]) / (self.cells["n_s"] + 1))
        )

    def test_summary_and_stylers(self):
        summary = kh.analysis_summary(self.table, self.cells)
        self.assertEqual(set(summary), {"scores", "tests", "sampler", "cells"})
        self.assertEqual(summary["cells"]["share_below_1"].max(), 1.0)
        for styler in (
            kh._comparison_styler(self.table),
            kh._tests_styler(self.table),
            kh._diagnostics_styler(self.table),
        ):
            self.assertIn("\\begin{tabular}", styler.to_latex())

    def test_every_figure_draws_from_the_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "table.csv")
            self.table.to_csv(path, index=False)
            table = plots.load_table(path=path)
            cells_path = os.path.join(directory, "cells.csv")
            self.cells.to_csv(cells_path, index=False)
            cells = plots.load_table(path=cells_path)
        for name, plotter in plots.PLOTTERS.items():
            with self.subTest(figure=name):
                plt.close(plotter(table))
        plt.close(plots.plot_information(cells))
        self.assertEqual(set(plots.FIGURES), set(plots.PLOTTERS) | {"information"})

    def test_comparison_columns_follow_e31_with_rotated_methods(self):
        styler = kh._comparison_styler(self.table)
        groups = list(dict.fromkeys(styler.data.columns.get_level_values(0)))
        labels = [kh.baycap_fixed.SCORE_LABELS[score] for score in ("ae", "amplitude", "coverage", "winkler")]
        self.assertEqual(groups, [*labels, "$z$"])
        self.assertEqual(styler.data.shape[1], 5 * len(kh.METHODS))
        self.assertTrue(all(m.startswith("\\rotatebox{90}{") for m in styler.data.columns.get_level_values(1)))
        self.assertIn("\\usepackage{graphicx}", kh._latex_document("x.tex"))

    def test_displacement_accepts_the_methods_with_a_nan_prior_part(self):
        for method in kh.NEW.values():
            self.assertIn(method, METHOD_CHOICES)
            self.assertEqual(method_c_prior(method), "hierarchical-f")
        self.assertTrue(np.all(np.isnan(smoothed_recalls(self.val_counts[("wine", "LR")], "hierarchical-f"))))


if __name__ == "__main__":
    unittest.main()
