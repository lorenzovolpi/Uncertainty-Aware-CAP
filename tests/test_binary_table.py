import unittest
from unittest import mock

import numpy as np
import pandas as pd

from uncawe.experiments.methods import get_method_names
from uncawe.tables import binary
from uncawe.tables import main as main_tables

METRICS = ("acc_err", "winkler")
METHODS = list(binary.METHODS)


class BinaryTableTest(unittest.TestCase):
    def test_methods(self):
        self.assertLessEqual(set(METHODS), set(get_method_names()))
        self.assertEqual(
            METHODS, ["DoC", "QuAcc_NxN", "O-LEAP", "AC-LSB", "L-CBPE-LSB", "BayCAP-KDEy-h-inst-cb-tcal-keep"]
        )
        self.assertEqual(main_tables._baseline_count(METHODS), 5)

    def test_table(self):
        rng = np.random.default_rng(0)
        rows = [
            {"classifier": "LR", "dataset": d, "method": m, "acc_err": v, "winkler": v}
            for d in ("a", "b")
            for m in METHODS
            for v in rng.random(10)
        ]
        latex = main_tables._results_latex_table(pd.DataFrame(rows), METRICS, False, ["a", "b"], methods=METHODS)
        self.assertEqual(latex.count("\\multicolumn{5}{c|}{Baseline methods}"), 2)
        self.assertIn("\\rotatebox{90}{AC}", latex)
        self.assertIn("\\rotatebox{90}{CBPE}", latex)
        self.assertNotIn(main_tables.EMPTY_CELL, latex)
        self.assertIn("Wins vs AC", latex)
        self.assertIn("Avg.\\ rank", latex)

    def test_only_binary_datasets_are_loaded(self):
        rows = [
            {
                "classifier": "LR",
                "dataset": d,
                "collection": c,
                "method": m,
                "acc_err": 0.1,
                "true_cts": np.array([[5, 1], [2, 4]]),
                "estim_acc_ranges": np.array([0.5, 0.9]),
            }
            for d, c in (("haberman", "uci_binary"), ("yeast", "uci_multiclass"))
            for m in ("DoC", "O-LEAP")
        ]
        loaded = mock.Mock(df=pd.DataFrame(rows))
        with (
            mock.patch.object(main_tables, "load_results", return_value=loaded),
            mock.patch.object(main_tables, "get_acc_coverage", return_value=1.0),
            mock.patch.object(main_tables, "get_acc_amplitude", return_value=0.4),
            mock.patch.object(main_tables, "get_acc_winkler", return_value=0.4),
        ):
            df, _ = main_tables.load_table_frame(
                "base", "classic", "vanilla_accuracy", ["LR"], methods=["DoC", "O-LEAP"], collections=binary.COLLECTIONS
            )
        self.assertEqual(df["dataset"].nunique(), 1)
        self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
