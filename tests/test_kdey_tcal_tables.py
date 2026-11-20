"""The ablation table and plot of E7.2 (work-plan card WP-26, E7.2b) on synthetic stored results."""

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

from uncawe.plots import kdey_tcal as plots
from uncawe.tables import kdey_tcal as kt
from uncawe.tables import tempering

N_BAGS = 12
PAIRS = [("wine", "LR", 3), ("wine", "kNN", 3), ("spam", "LR", 2), ("spam", "kNN", 2)]
# tau_cal of E5.3 per pair (spam kNN not stored) and of E7.2b
TAU_E53 = {("wine", "LR"): 8.0, ("wine", "kNN"): 4.0, ("spam", "LR"): 1.0}
TAU_TCAL = {("wine", "LR"): 2.0, ("wine", "kNN"): 2.0, ("spam", "LR"): 1.0, ("spam", "kNN"): 0.5}
# the centre of the interval (shift from the true value) and its spread, per cell: the stored method is off-centre and
# narrow; the tempered ones wider; the cal ones centred
SHIFT = {"stored": 0.06, "tempered": 0.06, "cal": 0.0, "tcal": 0.0}
SPREAD = {"stored": 0.01, "tempered": 0.04, "cal": 0.01, "tcal": 0.02}


def _rows(cell: str, method: str, acc_name: str, pairs) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    rows = []
    for dataset, classifier, k in PAIRS:
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        noise = rng.standard_normal((N_BAGS, 200))
        if (dataset, classifier) not in pairs:
            continue
        for bag, ct in enumerate(true_cts):
            acc = float(np.trace(ct))
            samples = acc + SHIFT[cell] + SPREAD[cell] * noise[bag]
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
                "nuts_divergences": int(cell == "tcal" and bag == 0),
                "nuts_max_rhat": 1.001,
            }
            if cell in ("cal", "tcal"):
                row |= {"cal_bandwidth": 0.05 if k == 3 else 0.1, "cal_n_val": 90 if k == 3 else 75, "cal_n_cal": 44}
            if cell == "tcal":
                tau = TAU_TCAL[(dataset, classifier)]
                winkler = np.array([0.5, 0.4, 0.3, 0.35, 0.45])
                row |= {
                    "cal_temperature": tau,
                    "cal_tau_winkler": winkler,
                    "cal_tau_coverage": np.array([0.2, 0.4, 0.6, 0.8, 0.9]),
                    "cal_tau_divergent": np.array([0, 0, 1, 2, 3]),
                    "cal_tau_max_rhat": np.array([1.0, 1.0, 1.01, 1.02, 1.03]),
                    "cal_tau_min_ess": np.array([400.0, 380, 300, 250, 200]),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def _load_method(method, acc_name, *args, **kwargs):
    everything = {(d, c) for d, c, _ in PAIRS}
    for cell, name in kt.CELL_METHODS.items():
        if method == name and cell != "stored":
            return _rows(cell, method, acc_name, everything)
    if method == kt.STORED_METHOD:
        return _rows("stored", method, acc_name, everything)
    tau = float(method.rsplit("-t", 1)[1])
    pairs = {key for key, value in TAU_E53.items() if value == tau}
    return _rows("tempered", method, acc_name, pairs) if pairs else pd.DataFrame()


def _e53_taus(root=None):
    return pd.DataFrame([{"dataset": d, "classifier": c, "tau_cal_e53": t} for (d, c), t in TAU_E53.items()])


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (
            patch.object(tempering, "load_method", side_effect=_load_method),
            patch.object(kt, "e53_taus", side_effect=_e53_taus),
        ):
            cls.table = kt.pair_table(["vanilla_accuracy", "macro-F1"])
        cls.summary = kt.summary_table(cls.table)
        cls.cross = kt.crosstab(cls.table)
        cls.nuts = kt.nuts_table(cls.table)

    def test_methods(self):
        self.assertEqual(
            kt.CELL_METHODS,
            {
                "stored": "BayCAP-KDEy-mlp-real",
                "cal": "BayCAP-KDEy-mlp-real-cal",
                "tcal": "BayCAP-KDEy-mlp-real-cal-tcal",
            },
        )
        self.assertEqual(kt.TAUS, tempering.TAUS)

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * len(PAIRS))
        vanilla = self.table.loc[self.table["acc_name"] == "vanilla_accuracy"].set_index(["dataset", "classifier"])
        wine = vanilla.loc[("wine", "LR")]
        self.assertEqual((wine["tau_cal"], wine["tau_cal_e53"], wine["b_cal"]), (2.0, 8.0, 0.05))
        self.assertTrue(vanilla["b_cal_same"].all())
        # the E5.3 cell is missing where tau_select is not stored
        self.assertTrue(np.isnan(vanilla.loc[("spam", "kNN"), "e53_winkler"]))
        self.assertTrue(np.isnan(vanilla.loc[("spam", "kNN"), "winkler_vs_e53_p"]))
        # the centred cells have a lower AE than the shifted ones, significant after Holm
        for other in ("stored", "e53"):
            self.assertLess(wine[f"ae_vs_{other}_diff"], 0)
            self.assertLess(wine[f"ae_vs_{other}_p_holm"], kt.TEST_LEVEL)
        self.assertAlmostEqual(wine["share_divergent"], 1 / N_BAGS)
        self.assertEqual(wine["calw_t2"], 0.3)
        self.assertEqual(wine["div_t8"], 3.0)

    def test_the_e53_cell_is_read_at_its_tau(self):
        vanilla = self.table.loc[self.table["acc_name"] == "vanilla_accuracy"].set_index(["dataset", "classifier"])
        self.assertGreater(
            vanilla.loc[("wine", "LR"), "e53_amplitude"], vanilla.loc[("wine", "LR"), "stored_amplitude"]
        )

    def test_summary(self):
        part = self.summary.loc[(self.summary["acc_name"] == "vanilla_accuracy") & (self.summary["K"] == "all")]
        self.assertEqual(list(part["cell"]), list(kt.CELLS))
        by_cell = part.set_index("cell")
        self.assertEqual(by_cell.loc["e53", "pairs"], 3)
        self.assertEqual(by_cell.loc["tcal", "pairs"], 4)
        self.assertEqual(by_cell.loc["tcal", "ae_vs_stored_lower"], 4)
        self.assertEqual(by_cell.loc["tcal", "ae_vs_stored_higher"], 0)

    def test_crosstab_and_nuts(self):
        self.assertEqual(int(self.cross.to_numpy().sum()), len(PAIRS))
        self.assertEqual(self.cross.loc[2.0, 8.0], 1)
        self.assertEqual(self.cross.loc[0.5].iloc[-1], 1)
        self.assertEqual(list(self.nuts["pairs_selected"]), [1, 1, 2, 0, 0])
        self.assertEqual(self.nuts.set_index("tau").loc[8.0, "runs_divergent"], 3 * len(PAIRS))

    def test_latex(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"].reset_index(drop=True)
        self.assertIn("lower", kt._summary_styler(part).to_latex())
        pairs_latex = kt._pairs_styler(self.table.loc[self.table["acc_name"] == "vanilla_accuracy"]).to_latex()
        self.assertIn("tau", pairs_latex)
        self.assertIn("AE", pairs_latex)
        self.assertIn("E5.3", kt._crosstab_styler(self.cross).to_latex())
        self.assertIn("E7.2b", kt.caption("vanilla_accuracy", "summary"))

    def test_plot(self):
        fig = plots.plot_ablation(self.table, "vanilla_accuracy")
        self.assertEqual(len(fig.axes[0].collections), len(kt.CELLS))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
