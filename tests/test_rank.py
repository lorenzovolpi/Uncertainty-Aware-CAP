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
        self.assertIn("l*{11}{", latex)
        self.assertIn("\\cmidrule(r){2-12}\\cmidrule(l){13-23}", latex)

    def test_ranked_method_without_a_column_is_an_error(self):
        df = _results(rank._methods()[:2] + [next(iter(FIXED_C_VARIANTS))])
        with self.assertRaises(ValueError):
            rank._rank_latex_table(df)

    def test_metric_rules_span_the_columns_of_each_metric(self):
        self.assertEqual(rank._metric_cmidrules(6, 2), "\\cmidrule(r){2-7}\\cmidrule(l){8-13}")


if __name__ == "__main__":
    unittest.main()
