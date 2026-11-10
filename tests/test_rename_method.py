"""Tests for scripts/migrate/rename_method.py: renaming a method in stored results keeps every other column."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from uncawe.results import ResultsDataFrame

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "migrate" / "rename_method.py"
spec = importlib.util.spec_from_file_location("rename_method", SCRIPT)
rename_method = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rename_method)


def _result(method: str) -> ResultsDataFrame:
    rng = np.random.default_rng(0)
    return ResultsDataFrame.from_records(
        3,
        true_cts=[rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(3)],
        estim_cts=[rng.dirichlet(np.ones(4)).reshape(2, 2) for _ in range(3)],
        estim_accs=[0.1, 0.2, 0.3],
        estim_acc_ranges=[rng.random(5) for _ in range(3)],
        method=method,
        dataset="d",
    )


class RenameMethodTest(unittest.TestCase):
    def test_rename_keeps_every_other_column(self):
        with tempfile.TemporaryDirectory() as root:
            cell = Path(root) / "classic" / "vanilla_accuracy" / "d" / "LR"
            cell.mkdir(parents=True)
            _result("RQBS").save_result(str(cell / "RQBS.parquet"))
            before = ResultsDataFrame.load_result(str(cell / "RQBS.parquet"))

            outcome = rename_method.rename_file(cell / "RQBS.parquet", "RQBS", "SQS-Hbs", dry_run=False)

            self.assertTrue(outcome.startswith("renamed"))
            self.assertFalse((cell / "RQBS.parquet").exists())
            after = ResultsDataFrame.load_result(str(cell / "SQS-Hbs.parquet"))
            self.assertEqual(list(after.columns), list(before.columns))
            self.assertTrue((after["method"] == "SQS-Hbs").all())
            for column in ("true_cts", "estim_cts", "estim_acc_ranges"):
                for a, b in zip(after[column], before[column]):
                    np.testing.assert_array_equal(a, b)
            pd.testing.assert_frame_equal(
                after.drop(columns=["method", "true_cts", "estim_cts", "estim_acc_ranges"]),
                before.drop(columns=["method", "true_cts", "estim_cts", "estim_acc_ranges"]),
            )

    def test_existing_target_and_other_method_are_left_untouched(self):
        with tempfile.TemporaryDirectory() as root:
            cell = Path(root)
            _result("RQBS").save_result(str(cell / "RQBS.parquet"))
            _result("SQS-Hbs").save_result(str(cell / "SQS-Hbs.parquet"))
            self.assertTrue(
                rename_method.rename_file(cell / "RQBS.parquet", "RQBS", "SQS-Hbs", False).startswith("skip")
            )
            self.assertTrue((cell / "RQBS.parquet").exists())

            other = Path(root) / "other"
            other.mkdir()
            _result("PQ").save_result(str(other / "RQBS.parquet"))
            self.assertTrue(
                rename_method.rename_file(other / "RQBS.parquet", "RQBS", "SQS-Hbs", False).startswith("skip")
            )
            self.assertTrue((other / "RQBS.parquet").exists())

    def test_dry_run_changes_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            cell = Path(root)
            _result("RQBS").save_result(str(cell / "RQBS.parquet"))
            self.assertTrue(
                rename_method.rename_file(cell / "RQBS.parquet", "RQBS", "SQS-Hbs", True).startswith("would")
            )
            self.assertTrue((cell / "RQBS.parquet").exists())
            self.assertFalse((cell / "SQS-Hbs.parquet").exists())


if __name__ == "__main__":
    unittest.main()
