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

    def test_coverage_color_is_green_within_the_band_and_white_outside(self):
        self.assertEqual(ca.coverage_color(0.0), rank._rank_color(1.0))
        self.assertEqual(ca.coverage_color(ca.COVERAGE_BAND), ca.GREEN)
        self.assertEqual(ca.coverage_color(ca.COVERAGE_BAND + 0.001), "#ffffff")
        self.assertEqual(ca.coverage_color(np.nan), "")

    def test_amplitude_color_is_green_below_the_threshold_and_white_from_it(self):
        self.assertEqual(ca.amplitude_color(0.099), ca.GREEN)
        self.assertEqual(ca.amplitude_color(ca.AMPLITUDE_MAX), "#ffffff")
        self.assertEqual(ca.amplitude_color(0.5), "#ffffff")
        self.assertEqual(ca.amplitude_color(np.nan), "")

    def test_colours_follow_the_printed_values(self):
        # 0.8496 prints as 0.850: in the band; 0.0996 prints as 0.100: not below the threshold
        coverage = dict.fromkeys(METHODS, 0.95)
        coverage["DoC"] = 0.8496
        coverage["PQ"] = 0.8494
        amplitude = dict.fromkeys(METHODS, 0.05)
        amplitude["DoC"] = 0.0996
        mean = _frame(coverage, amplitude).groupby(["classifier", "dataset", "method"])[list(ca.METRICS)].mean()
        mean = mean.unstack("method").swaplevel(0, 1, axis=1)
        mean.columns = mean.columns.set_names(["method", "metric"])
        styles = ca.cell_styles(mean)
        row = ("LR", "d1")
        self.assertIn(ca.GREEN, styles.loc[row, ("DoC", "coverage")])
        self.assertIn("#ffffff", styles.loc[row, ("PQ", "coverage")])
        self.assertIn("#ffffff", styles.loc[row, ("DoC", "amplitude")])
        self.assertIn(ca.GREEN, styles.loc[row, ("PQ", "amplitude")])

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
        # coverage 0.80 is 0.15 away: white; 0.93 is in the band: green
        self.assertIn("#ffffff", styles.loc[row, ("DoC", "coverage")])
        self.assertIn(ca.GREEN, styles.loc[row, ("PQ", "coverage")])

    def test_latex_table_has_the_main_table_structure(self):
        coverage = dict.fromkeys(METHODS, 0.95)
        amplitude = dict.fromkeys(METHODS, 0.1)
        latex = ca.latex_table(_frame(coverage, amplitude), ["d1"])
        self.assertIn(
            f"\\multicolumn{{{N}}}{{c|}}{{$\\mathcal{{C}}$}} & \\multicolumn{{{N}}}{{c}}{{$\\mathcal{{A}}$}}", latex
        )
        self.assertIn(
            f"\\multicolumn{{3}}{{c|}}{{Baseline methods}} & \\multicolumn{{{N - 3}}}{{c|}}{{Our methods}}", latex
        )
        self.assertIn("\\multicolumn{1}{c|}{\\rotatebox{90}{LEAP}}", latex)
        self.assertNotIn("\\pm", latex)
        self.assertNotIn("\\dag", latex)


if __name__ == "__main__":
    unittest.main()
