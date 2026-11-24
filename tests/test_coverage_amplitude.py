import unittest

import numpy as np
import pandas as pd

from uncawe.tables import coverage_amplitude as ca
from uncawe.tables import main as main_tables
from uncawe.tables import rank

METHODS = list(main_tables.MAIN_METHODS)
# the coverage/amplitude table shows the methods of the main table without BayesCAP
N = len(main_tables.MAIN_METHODS_BEFORE_BAYESCAP)


def _frame(coverage: dict[str, float], amplitude: dict[str, float]) -> pd.DataFrame:
    """Two bags per method on one pair, with the given mean coverage and amplitude."""
    rows = []
    for method in METHODS:
        for _ in range(2):
            rows.append(
                {
                    "classifier": "LR",
                    "dataset": "d1",
                    "method": method,
                    "coverage": coverage[method],
                    "amplitude": amplitude[method],
                }
            )
    return pd.DataFrame(rows)


class TestCoverageAmplitude(unittest.TestCase):
    def test_coverage_distance_is_symmetric_around_nominal(self):
        d = ca.coverage_distance(pd.Series([1.0, 0.9, 0.95, 0.0]))
        np.testing.assert_allclose(d, [0.05, 0.05, 0.0, 0.95])

    def test_coverage_color_fades_to_white_at_d_max(self):
        self.assertEqual(ca.coverage_color(0.0), rank._rank_color(1.0))
        self.assertEqual(ca.coverage_color(ca.COVERAGE_D_MAX), "#ffffff")
        self.assertEqual(ca.coverage_color(0.5), "#ffffff")
        self.assertEqual(ca.coverage_color(ca.COVERAGE_D_MAX / 2), rank._rank_color(1.0 + (rank.WHITE_RANK - 1) / 2))
        self.assertEqual(ca.coverage_color(np.nan), "")

    def test_amplitude_ranks_lower_is_better_with_average_ties(self):
        np.testing.assert_allclose(ca.amplitude_ranks(pd.Series([0.3, 0.1, 0.1, 0.2])), [4.0, 1.5, 1.5, 3.0])

    def test_bold_goes_to_closest_coverage_and_narrowest_amplitude(self):
        coverage = dict.fromkeys(METHODS, 0.80)
        coverage["SQS-H"] = 0.99
        coverage["PQ"] = 0.93  # closest to 0.95
        amplitude = {m: 0.1 + 0.01 * i for i, m in enumerate(METHODS)}
        amplitude["SQS-Hbs"] = 0.01  # narrowest
        mean = _frame(coverage, amplitude).groupby(["classifier", "dataset", "method"])[list(ca.METRICS)].mean()
        mean = mean.unstack("method").swaplevel(0, 1, axis=1)
        mean.columns = mean.columns.set_names(["method", "metric"])
        styles = ca.cell_styles(mean)
        row = ("LR", "d1")
        bold = [col for col in styles.columns if "bold" in styles.loc[row, col]]
        self.assertEqual(sorted(bold), [("PQ", "coverage"), ("SQS-Hbs", "amplitude")])
        # coverage 0.80 is 0.15 away: coloured, not white
        self.assertIn("background-color", styles.loc[row, ("DoC", "coverage")])
        self.assertNotIn("#ffffff", styles.loc[row, ("DoC", "coverage")])

    def test_latex_table_has_the_main_table_structure(self):
        coverage = dict.fromkeys(METHODS, 0.95)
        amplitude = dict.fromkeys(METHODS, 0.1)
        latex = ca.latex_table(_frame(coverage, amplitude), ["d1"])
        self.assertIn(f"\\multicolumn{{{N}}}{{c|}}{{$\\mathcal{{C}}$}} & \\multicolumn{{{N}}}{{c}}{{$\\mathcal{{A}}$}}", latex)
        self.assertIn(f"\\multicolumn{{3}}{{c|}}{{Baseline methods}} & \\multicolumn{{{N - 3}}}{{c|}}{{Our methods}}", latex)
        self.assertIn("\\multicolumn{1}{c|}{\\rotatebox{90}{LEAP}}", latex)
        self.assertNotIn("\\pm", latex)
        self.assertNotIn("\\dag", latex)


if __name__ == "__main__":
    unittest.main()
