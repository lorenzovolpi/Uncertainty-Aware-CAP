"""Tables and plots of E7.1 (work-plan card WP-25) on synthetic stored results."""

import os
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from test_baycap_fixed_tables import _true_ct

from uncawe.plots import kdey_loo as plots
from uncawe.tables import kdey_loo as kl

N_BAGS = 12
PAIRS = [("wine", "LR", 3), ("wine", "kNN", 3), ("spam", "LR", 2), ("spam", "kNN", 2)]
# per pair: the selected bandwidths and their edge flags (the first class of wine at the lower edge of the grid)
SELECTION = {
    3: {"bandwidths": [0.01, 0.0631, 0.158], "bandwidth_edge": [-1, 0, 0], "class_sizes": [40, 30, 20]},
    2: {"bandwidths": [0.1, 0.251], "bandwidth_edge": [0, 0], "class_sizes": [50, 25]},
}


def _frames(acc_name: str) -> dict[str, pd.DataFrame]:
    """The four methods on the four pairs; the LOO methods are centred, the stored ones shifted by 0.06."""
    rng = np.random.default_rng(5)
    rows = {method: [] for method in kl.METHODS}
    for dataset, classifier, k in PAIRS:
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        noise = rng.standard_normal((N_BAGS, 200))
        for method in kl.METHODS:
            shift = 0.0 if method in kl.COMPARISONS else 0.06
            for bag, ct in enumerate(true_cts):
                acc = float(np.trace(ct))
                samples = acc + shift + 0.02 * noise[bag]
                row = {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": k,
                    "method": method,
                    "acc_name": acc_name,
                    "true_cts": ct,
                    "estim_cts": ct,
                    "estim_accs": float(np.mean(samples)),
                    "estim_acc_ranges": samples,
                    "acc_err": abs(float(np.mean(samples)) - acc),
                    "nuts_divergences": int(method == "SQS-B-loo" and bag == 0),
                    "nuts_max_rhat": 1.001,
                }
                if method in kl.COMPARISONS:
                    selection = SELECTION[k]
                    row |= {name: np.asarray(value, dtype=float) for name, value in selection.items()}
                    row["loo_selected"] = np.full(k, 1.5)
                    row["loo_default"] = np.full(k, 1.0)
                rows[method].append(row)
    return {method: pd.DataFrame(r) for method, r in rows.items()}


def _load_method(method, acc_name, *args, **kwargs):
    return _frames(acc_name)[method]


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(kl.tempering, "load_method", side_effect=_load_method):
            cls.table = kl.pair_table(["vanilla_accuracy", "macro-F1"])
            cls.classes = kl.class_table()
        cls.summary = kl.summary_table(cls.table)

    def test_methods_and_comparisons(self):
        self.assertEqual(kl.COMPARISONS, {"BayCAP-KDEy-mlp-real-loo": "BayCAP-KDEy-mlp-real", "SQS-B-loo": "SQS-B"})
        self.assertEqual(kl.METHODS, ["BayCAP-KDEy-mlp-real", "BayCAP-KDEy-mlp-real-loo", "SQS-B", "SQS-B-loo"])

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * len(PAIRS))
        vanilla = self.table.loc[self.table["acc_name"] == "vanilla_accuracy"]
        self.assertEqual(list(vanilla["dataset"].astype(str)), ["spam", "spam", "wine", "wine"])  # ordered by K
        for loo in kl.COMPARISONS:
            # the LOO methods are centred: a lower Winkler score on every pair, significant after Holm
            self.assertTrue((vanilla[f"{loo}_diff_mean"] < 0).all())
            self.assertTrue((vanilla[f"{loo}_p_holm"] < kl.TEST_LEVEL).all())
        np.testing.assert_allclose(vanilla["SQS-B-loo_share_divergent"], 1 / N_BAGS)
        wine = vanilla.loc[vanilla["dataset"] == "wine"].iloc[0]
        self.assertEqual((wine["n_val"], wine["n_lower_edge"], wine["n_upper_edge"]), (90, 1, 0))
        self.assertAlmostEqual(wine["b_median"], 0.0631)
        self.assertAlmostEqual(wine["log_b_ratio_median"], np.log10(0.631))
        self.assertAlmostEqual(wine["loo_gain_mean"], 0.5)

    def test_summary(self):
        part = self.summary.loc[(self.summary["acc_name"] == "vanilla_accuracy") & (self.summary["K"] == "all")]
        self.assertEqual(list(part["method"]), kl.METHODS)
        loo = part.set_index("method").loc["BayCAP-KDEy-mlp-real-loo"]
        self.assertEqual((loo["pairs"], loo["pairs_lower"], loo["pairs_higher"]), (4, 4, 0))
        self.assertTrue(np.isnan(part.set_index("method").loc["SQS-B", "pairs_lower"]))

    def test_class_table(self):
        self.assertEqual(len(self.classes), 3 + 3 + 2 + 2)
        wine = self.classes.loc[(self.classes["dataset"] == "wine") & (self.classes["classifier"] == "LR")]
        np.testing.assert_allclose(wine["bandwidth"], SELECTION[3]["bandwidths"])
        np.testing.assert_array_equal(wine["n_j"], SELECTION[3]["class_sizes"])
        np.testing.assert_array_equal(wine["edge"], SELECTION[3]["bandwidth_edge"])

    def test_two_sided_wilcoxon(self):
        self.assertTrue(np.isnan(kl.two_sided_wilcoxon([1.0, 2.0], [1.0, 2.0])))
        self.assertLess(kl.two_sided_wilcoxon(np.arange(10.0), np.arange(10.0) + 1), 0.01)

    def test_latex(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"].reset_index(drop=True)
        latex = kl._summary_styler(part).to_latex()
        self.assertIn("KDEy loo", latex)
        latex = kl._pairs_styler(self.table.loc[self.table["acc_name"] == "vanilla_accuracy"]).to_latex()
        self.assertIn("\\textbf{", latex)
        self.assertIn("\\bfseries", latex)
        for name in kl.TABLES:
            self.assertIn("E7.1", kl.caption("macro-F1", name))

    def test_plots(self):
        for fig in (plots.plot_bandwidths(self.classes), plots.plot_effect(self.table, "vanilla_accuracy")):
            self.assertGreater(len(fig.axes), 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
