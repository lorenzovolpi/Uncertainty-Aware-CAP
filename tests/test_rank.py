import unittest

import numpy as np
import pandas as pd

from uncawe.experiments.methods import FIXED_C_VARIANTS
from uncawe.tables import rank


def _results(methods: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = [
        {"classifier": "LR", "dataset": dataset, "method": method, "acc_err": rng.random(), "winkler": rng.random()}
        for dataset in ("d1", "d2", "d3")
        for method in methods
        for _ in range(2)
    ]
    return pd.DataFrame(rows)


class TestRankTable(unittest.TestCase):
    def test_included_methods_get_a_column(self):
        # the extended table of E3.1 (--include-e31) ranks the eight methods and must also show them
        main_methods = rank._methods()[:3]
        methods = rank._methods(tuple(FIXED_C_VARIANTS))
        included = [method for method in methods if method in main_methods or method in FIXED_C_VARIANTS]
        latex = rank._rank_latex_table(_results(included), methods=methods)
        value = f"{{w{{c}}{{{rank.VALUE_COLUMN_WIDTH}}}}}"
        group = f"*{{3}}{value}|*{{8}}{value}"
        self.assertIn(f"{{l{group}|{group}}}", latex)
        self.assertIn("\\cmidrule(r){2-12}\\cmidrule(l){13-23}", latex)
        self.assertIn("\\multicolumn{3}{c|}{Baseline methods} & \\multicolumn{8}{c|}{Our methods}", latex)
        self.assertIn("\\multicolumn{1}{c|}{\\rotatebox{90}{LEAP}}", latex)
        self.assertNotIn("\\scriptsize", latex)

    def test_ranked_method_without_a_column_is_an_error(self):
        df = _results(rank._methods()[:2] + [next(iter(FIXED_C_VARIANTS))])
        with self.assertRaises(ValueError):
            rank._rank_latex_table(df)

    def test_one_table_per_metric(self):
        methods = rank._methods()
        df = _results(methods)
        n = len(methods)
        ours = n - 3
        for metric, label in (("acc_err", "$\\mathrm{AE}$"), ("winkler", "$\\mathcal{W}$")):
            with self.subTest(metric=metric):
                latex = rank._rank_latex_table(df, (metric,), methods)
                value = f"{{w{{c}}{{{rank.VALUE_COLUMN_WIDTH}}}}}"
                self.assertIn(f"{{l*{{3}}{value}|*{{{ours}}}{value}}}", latex)
                # the metric heads the columns, with no vertical line after it
                self.assertIn(f" & \\multicolumn{{{n}}}{{c}}{{{label}}} \\\\", latex)
                self.assertIn(
                    f" & \\multicolumn{{3}}{{c|}}{{Baseline methods}} & \\multicolumn{{{ours}}}{{c}}{{Our methods}} \\\\",
                    latex,
                )
                self.assertIn("\\multicolumn{1}{c|}{\\rotatebox{90}{LEAP}}", latex)
        # the ranks of one metric do not depend on the other metric
        both = rank._rank_mean_values(rank._dataset_rank_values(df, rank.METRICS))
        alone = rank._rank_mean_values(rank._dataset_rank_values(df, ("winkler",)))
        pd.testing.assert_frame_equal(
            both.xs("winkler", axis=1, level="metric"), alone.xs("winkler", axis=1, level="metric")
        )

    def test_metric_rules_span_the_columns_of_each_metric(self):
        self.assertEqual(rank._metric_cmidrules(6, 2), "\\cmidrule(r){2-7}\\cmidrule(l){8-13}")


if __name__ == "__main__":
    unittest.main()
