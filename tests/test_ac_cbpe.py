import unittest

import numpy as np
import pandas as pd

from uncawe.experiments.methods import get_method_names
from uncawe.tables import ac_cbpe
from uncawe.tables import main as main_tables

METRICS = ("acc_err", "winkler")
METHODS = list(ac_cbpe.METHODS)
N = len(METHODS)
CBPE = ("L-CBPE", "L-CBPE-LSB")


def _frame(rng: np.random.Generator) -> pd.DataFrame:
    """Per-bag scores on a binary row (every method) and a multiclass row (no CBPE)."""
    rows = []
    for dataset, methods in (("bin", METHODS), ("multi", [m for m in METHODS if m not in CBPE])):
        for method in methods:
            for value in rng.random(15):
                rows.append(
                    {"classifier": "LR", "dataset": dataset, "method": method, "acc_err": value, "winkler": value}
                )
    return pd.DataFrame(rows)


class ACCBPETableTest(unittest.TestCase):
    def test_methods_are_stored_methods(self):
        self.assertLessEqual(set(METHODS), set(get_method_names()))

    def test_ac_and_cbpe_are_baselines(self):
        self.assertEqual(main_tables._baseline_count(METHODS), 7)
        self.assertEqual(main_tables._baseline_count(list(main_tables.MAIN_METHODS)), 3)

    def test_table(self):
        df = _frame(np.random.default_rng(0))
        latex = main_tables._results_latex_table(df, METRICS, False, ["bin", "multi"], methods=METHODS)
        self.assertEqual(latex.count("\\multicolumn{7}{c|}{Baseline methods}"), 2)
        self.assertIn("\\multicolumn{2}{c|}{Our methods}", latex)
        self.assertIn("\\multicolumn{2}{c}{Our methods}", latex)
        # the columns in the order of METHODS
        header = next(line for line in latex.splitlines() if "\\rotatebox" in line)
        positions = [
            header.index(f"\\rotatebox{{90}}{{{main_tables._latex_display_column_value(m, 'method')}}}")
            for m in METHODS
        ]
        self.assertEqual(positions, sorted(positions))
        # the multiclass row has two empty cells per metric, those of CBPE
        multi = next(line for line in latex.splitlines() if " & multi & " in line)
        self.assertEqual(multi.count(main_tables.EMPTY_CELL), 2 * len(CBPE))
        self.assertIn("Wins vs LEAP", latex)
        self.assertIn("Avg.\\ rank", latex)

    def test_main_table_unchanged_without_methods(self):
        rng = np.random.default_rng(1)
        rows = [
            {"classifier": "LR", "dataset": "d", "method": m, "acc_err": v, "winkler": v}
            for m in [*main_tables.MAIN_METHODS, "AC"]
            for v in rng.random(5)
        ]
        latex = main_tables._results_latex_table(pd.DataFrame(rows), METRICS, False, ["d"])
        self.assertNotIn("{AC}", latex)
        self.assertEqual(latex.count("\\multicolumn{3}{c|}{Baseline methods}"), 2)


if __name__ == "__main__":
    unittest.main()
