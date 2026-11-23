import re
import unittest

from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS, METHOD_LATEX_PREAMBLE

METHODS = list(main_tables.MAIN_METHODS)
METRICS = ("acc_err", "winkler")
N = len(METHODS)
OURS = N - 3
NUMERIC = "S[table-format=1.3,table-align-text-post=false]"


class TestMainTableHeader(unittest.TestCase):
    def test_baselines_are_doc_quacc_leap(self):
        self.assertEqual(main_tables._baseline_count(METHODS), 3)
        self.assertEqual(METHODS[:3], ["DoC", "QuAcc_NxN", "O-LEAP"])

    def test_column_format_has_a_line_after_leap_and_between_metrics(self):
        group = NUMERIC * 3 + "|" + NUMERIC * OURS
        self.assertEqual(main_tables._latex_column_format(METHODS, METRICS, 2), "ll" + group + "|" + group)

    def test_family_header_spans_baselines_and_other_methods(self):
        line = main_tables._method_family_header_line(2, METHODS, METRICS)
        self.assertEqual(
            line,
            f" &  & \\multicolumn{{3}}{{c|}}{{Baseline methods}} & \\multicolumn{{{OURS}}}{{c|}}{{Our methods}}"
            f" & \\multicolumn{{3}}{{c|}}{{Baseline methods}} & \\multicolumn{{{OURS}}}{{c}}{{Our methods}} \\\\",
        )

    def test_rules_span_the_methods_of_each_metric(self):
        self.assertEqual(main_tables._metric_cmidrules(2, 10, 2), "\\cmidrule(r){3-12}\\cmidrule(l){13-22}")

    def test_method_header_has_a_line_after_leap(self):
        aligns = [main_tables._method_header_align(i, METHODS, METRICS, 3) for i in range(2 * N)]
        self.assertEqual(aligns[2], "c|")
        self.assertEqual(aligns[N - 1], "c|")
        self.assertEqual(aligns[N + 2], "c|")
        self.assertEqual(aligns[2 * N - 1], "c")
        self.assertEqual(aligns.count("c|"), 3)


class TestMethodLatexMacros(unittest.TestCase):
    def test_every_label_macro_is_defined(self):
        used = {m for label in METHOD_LATEX_LABELS.values() for m in re.findall(r"\\[a-z]+", label)}
        defined = set(re.findall(r"\\newcommand\{(\\[a-z]+)\}", METHOD_LATEX_PREAMBLE))
        self.assertLessEqual(used, defined)

    def test_main_document_defines_the_macros(self):
        self.assertIn(METHOD_LATEX_PREAMBLE, main_tables._results_latex_document([]))


if __name__ == "__main__":
    unittest.main()
