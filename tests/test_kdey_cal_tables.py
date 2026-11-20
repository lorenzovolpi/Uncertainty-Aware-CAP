"""Tables and plots of E7.2 (work-plan card WP-26) on synthetic stored results."""

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

from uncawe.plots import kdey_cal as plots
from uncawe.tables import kdey_cal as kc

N_BAGS = 12
PAIRS = [("wine", "LR", 3), ("wine", "kNN", 3), ("spam", "LR", 2), ("spam", "kNN", 2)]
# the stored curve: a minimum at 0.0398 for wine (4 grid steps below 0.1), at 0.1 for spam
CURVES = {3: 0.02 + (np.log10(kc.GRID) - np.log10(0.0398)) ** 2, 2: 0.03 + (np.log10(kc.GRID) + 1) ** 2}
# the pilot check: only the wine pairs store the NUTS curve, which picks one step higher
CHECK = {3: CURVES[3] + 0.001 * (np.log10(kc.GRID) - np.log10(0.0398)) * -20}


def _frames(acc_name: str) -> dict[str, pd.DataFrame]:
    """Both methods on the four pairs; on wine the cal method is centred and the stored one shifted by 0.06, on spam
    they coincide (b_cal = 0.1)."""
    rng = np.random.default_rng(5)
    rows = {method: [] for method in kc.METHODS}
    for dataset, classifier, k in PAIRS:
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        noise = rng.standard_normal((N_BAGS, 200))
        curve = CURVES[k]
        best = int(np.argmin(curve))
        for method in kc.METHODS:
            shift = 0.06 if (method == kc.STORED_METHOD and k == 3) else 0.0
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
                    "nuts_divergences": int(method == kc.CAL_METHOD and bag == 0),
                    "nuts_max_rhat": 1.001,
                }
                if method == kc.CAL_METHOD:
                    row |= {
                        "cal_bandwidth": float(kc.GRID[best]),
                        "cal_curve": curve,
                        "cal_selected": float(curve[best]),
                        "cal_default": float(curve[10]),
                        "cal_em_unconverged": 0,
                        "cal_n_val": 90 if k == 3 else 75,
                        "cal_n_cal": 44 if k == 3 else 37,
                    }
                    if k in CHECK:
                        row["cal_curve_nuts"] = CHECK[k]
                        row["cal_bandwidth_nuts"] = float(kc.GRID[int(np.argmin(CHECK[k]))])
                    else:
                        row["cal_curve_nuts"] = None
                        row["cal_bandwidth_nuts"] = np.nan
                rows[method].append(row)
    return {method: pd.DataFrame(r) for method, r in rows.items()}


def _load_method(method, acc_name, *args, **kwargs):
    return _frames(acc_name)[method]


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(kc.tempering, "load_method", side_effect=_load_method):
            cls.table = kc.pair_table(["vanilla_accuracy", "macro-F1"])
            cls.curves = kc.curve_table()
        cls.summary = kc.summary_table(cls.table)
        cls.check = kc.check_table(cls.table)

    def test_methods(self):
        self.assertEqual(kc.METHODS, ["BayCAP-KDEy-mlp-real", "BayCAP-KDEy-mlp-real-cal"])
        self.assertAlmostEqual(kc.GRID_STEP, 0.1)

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * len(PAIRS))
        vanilla = self.table.loc[self.table["acc_name"] == "vanilla_accuracy"].set_index(["dataset", "classifier"])
        wine, spam = vanilla.loc[("wine", "LR")], vanilla.loc[("spam", "LR")]
        self.assertAlmostEqual(wine["b_cal"], 0.0398, places=4)
        self.assertEqual(wine["b_cal_steps"], -4)
        self.assertEqual(spam["b_cal_steps"], 0)
        self.assertAlmostEqual(wine["cal_gain"], CURVES[3][10] - CURVES[3].min())
        self.assertEqual(spam["cal_gain"], 0.0)
        # on wine the cal method is centred: lower AE and Winkler score, significant after Holm
        for score in kc.TESTED:
            self.assertLess(wine[f"{score}_diff_mean"], 0)
            self.assertLess(wine[f"{score}_p_holm"], kc.TEST_LEVEL)
            self.assertEqual(spam[f"{score}_diff_mean"], 0.0)
        self.assertGreater(wine["test_gain"], 0.0)
        self.assertAlmostEqual(wine["share_divergent"], 1 / N_BAGS)

    def test_check_table(self):
        self.assertEqual(len(self.check), 2)
        self.assertEqual(set(self.check["dataset"]), {"wine"})
        row = self.check.iloc[0]
        self.assertEqual(row["nuts_steps"], kc.steps(row["b_cal_nuts"], row["b_cal"]))
        self.assertAlmostEqual(row["nuts_curve_max_diff"], np.max(np.abs(CHECK[3] - CURVES[3])))

    def test_curve_table(self):
        self.assertEqual(len(self.curves), len(PAIRS) * len(kc.GRID))
        wine = self.curves.loc[(self.curves["dataset"] == "wine") & (self.curves["classifier"] == "LR")]
        np.testing.assert_allclose(wine["criterion"], CURVES[3])
        np.testing.assert_allclose(wine["criterion_nuts"], CHECK[3])
        spam = self.curves.loc[self.curves["dataset"] == "spam"]
        self.assertTrue(spam["criterion_nuts"].isna().all())

    def test_summary(self):
        part = self.summary.loc[(self.summary["acc_name"] == "vanilla_accuracy") & (self.summary["K"] == "all")]
        self.assertEqual(list(part["method"]), kc.METHODS)
        cal = part.set_index("method").loc[kc.CAL_METHOD]
        self.assertEqual((cal["pairs"], cal["ae_pairs_lower"], cal["ae_pairs_higher"]), (4, 2, 0))
        self.assertEqual(cal["pairs_at_reference"], 2)
        self.assertGreater(cal["spearman_gain"], 0.0)
        self.assertTrue(np.isnan(part.set_index("method").loc[kc.STORED_METHOD, "ae_pairs_lower"]))

    def test_latex(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"].reset_index(drop=True)
        self.assertIn("KDEy cal", kc._summary_styler(part).to_latex())
        latex = kc._pairs_styler(self.table.loc[self.table["acc_name"] == "vanilla_accuracy"]).to_latex()
        self.assertIn("\\textbf{", latex)
        self.assertIn("\\bfseries", latex)
        self.assertIn("NUTS", kc._check_styler(self.check).to_latex())
        for name in kc.TABLES:
            self.assertIn("E7.2", kc.caption("macro-F1", name))

    def test_the_cal_method_can_be_read_from_the_archived_pilot(self):
        with patch.object(kc.tempering, "load_method", side_effect=_load_method) as load:
            kc.load_all("vanilla_accuracy", kc.PILOT_EM_EXPERIMENT)
        folders = {call.args[0]: call.kwargs.get("experiment") for call in load.call_args_list}
        self.assertEqual(folders, {kc.CAL_METHOD: kc.PILOT_EM_EXPERIMENT, kc.STORED_METHOD: None})

    def test_plots(self):
        for fig in (plots.plot_bandwidths(self.table), plots.plot_gain(self.table, "vanilla_accuracy")):
            self.assertGreater(len(fig.axes), 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
