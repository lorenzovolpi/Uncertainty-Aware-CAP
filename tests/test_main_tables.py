import re
import unittest

import numpy as np
import pandas as pd

from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS, METHOD_LATEX_PREAMBLE

METHODS = list(main_tables.MAIN_METHODS)
METRICS = ("acc_err", "winkler")
N = len(METHODS)
# the baselines: DoC, QuAcc, LEAP and, since 2026-11-26, AC-LSB (the default AC)
B = 4
OURS = N - B
NUMERIC = "S[table-format=1.3,table-align-text-post=false]"


class TestMainTableHeader(unittest.TestCase):
    def test_baselines_are_doc_quacc_leap_ac(self):
        self.assertEqual(main_tables._baseline_count(METHODS), B)
        self.assertEqual(METHODS[:B], ["DoC", "QuAcc_NxN", "O-LEAP", "AC-LSB"])

    def test_column_format_has_a_line_after_leap_and_between_metrics(self):
        group = NUMERIC * B + "|" + NUMERIC * OURS
        self.assertEqual(main_tables._latex_column_format(METHODS, METRICS, 2), "ll" + group + "|" + group)

    def test_family_header_spans_baselines_and_other_methods(self):
        line = main_tables._method_family_header_line(2, METHODS, METRICS)
        self.assertEqual(
            line,
            f" &  & \\multicolumn{{{B}}}{{c|}}{{Baseline methods}} & \\multicolumn{{{OURS}}}{{c|}}{{Our methods}}"
            f" & \\multicolumn{{{B}}}{{c|}}{{Baseline methods}} & \\multicolumn{{{OURS}}}{{c}}{{Our methods}} \\\\",
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


def _bags(scores: dict[str, list[float]], dataset: str = "d1", classifier: str = "LR") -> pd.DataFrame:
    """Per-bag rows of one table row: the same scores for AE and the Winkler score."""
    return pd.DataFrame(
        [
            {"classifier": classifier, "dataset": dataset, "method": m, "acc_err": v, "winkler": v}
            for m, values in scores.items()
            for v in values
        ]
    )


def _mean(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.groupby(["classifier", "dataset", "method"], sort=False)[list(METRICS)].mean().unstack("method")
    mean = mean.swaplevel(0, 1, axis=1)
    mean.columns = mean.columns.set_names(["method", "metric"])
    return mean


class TestSummaryRows(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        base = rng.random(30)
        # against the reference: A is clearly better, B clearly worse, C the same up to noise
        self.scores = {
            main_tables.WTL_REFERENCE: list(base + 1.0),
            "A": list(base + 0.2),
            "B": list(base + 3.0),
            "C": list(base + 1.0 + rng.normal(0, 1e-3, 30)),
        }
        self.methods = list(self.scores)

    def test_win_tie_loss(self):
        df = pd.concat([_bags(self.scores, "d1"), _bags(self.scores, "d2")])
        wtl = main_tables.win_tie_loss(df, ["classifier", "dataset"], _mean(df), self.methods, METRICS)
        for metric in METRICS:
            self.assertEqual(wtl[("A", metric)], (2, 0, 0))
            self.assertEqual(wtl[("B", metric)], (0, 0, 2))
            self.assertEqual(wtl[("C", metric)], (0, 2, 0))
            self.assertNotIn((main_tables.WTL_REFERENCE, metric), wtl)

    def test_average_ranks(self):
        df = pd.concat([_bags(self.scores, "d1"), _bags({**self.scores, "A": self.scores["B"]}, "d2")])
        ranks = main_tables.average_ranks(_mean(df), self.methods, METRICS)
        # d1: A 1, C and the reference about 2-3, B 4; d2: A and B tie for last (3.5)
        self.assertAlmostEqual(ranks[("A", "acc_err")], (1 + 3.5) / 2)
        self.assertAlmostEqual(ranks[("B", "winkler")], (4 + 3.5) / 2)

    def test_summary_rows_in_the_table(self):
        methods = [m for m in METHODS]
        rng = np.random.default_rng(1)
        df = pd.concat([_bags({m: list(rng.random(20)) for m in methods}, d) for d in ("d1", "d2")])
        latex = main_tables._results_latex_table(df, METRICS, show_std=False, dataset_order=["d1", "d2"])
        tail = latex[latex.index("\\midrule", latex.index("d2")) :]
        lines = [line for line in tail.splitlines() if line.startswith("\\multicolumn{2}{l}")]
        self.assertEqual(len(lines), 4)
        wins, ties, losses, ranks = lines
        for line, outcome in ((wins, "Wins"), (ties, "Ties"), (losses, "Losses")):
            self.assertIn(f"{{{outcome} vs LEAP}}", line)
        # one cell per method and metric after the label; the reference gets none
        rows = [line.removesuffix(" \\\\").split(" & ")[1:] for line in (wins, ties, losses)]
        reference = methods.index(main_tables.WTL_REFERENCE)
        for cells in rows:
            self.assertEqual(len(cells), 2 * N)
            self.assertEqual(cells[reference], "{--}")
            self.assertEqual(cells[N + reference], "{--}")
        # on each of the two rows of the table a method wins, ties or loses
        for column in range(2 * N):
            if column % N != reference:
                self.assertEqual(sum(int(cells[column].strip("{}")) for cells in rows), 2)
        self.assertIn("Avg.\\ rank", ranks)
        # the best average rank of each metric is in bold (ties all in bold)
        cells = ranks.removesuffix(" \\\\").split(" & ")[1:]
        for metric_cells in (cells[:N], cells[N:]):
            values = [float(c.strip("{}").replace("\\bfseries ", "")) for c in metric_cells]
            bold = [float(c.strip("{}").replace("\\bfseries ", "")) for c in metric_cells if "\\bfseries" in c]
            self.assertTrue(bold)
            self.assertTrue(all(np.isclose(v, min(values)) for v in bold))
        self.assertTrue(latex.rstrip().endswith("\\bottomrule\n\\end{tabular}"))


class TestIncludedMethods(unittest.TestCase):
    def test_included_method_sits_next_to_its_sibling(self):
        # since 2026-11-26 the BayCAP-KDEy of the main tables is BayCAP-KDEy-h-inst-cb-tcal-keep, which follows
        # BayCAP-KDEy-h-inst in the order of the stored methods
        methods = main_tables.table_method_names(("BayCAP-KDEy-h-inst",))
        self.assertEqual(len(methods), N + 1)
        i = methods.index("BayCAP-KDEy-h-inst")
        self.assertEqual(methods[i + 1], "BayCAP-KDEy-h-inst-cb-tcal-keep")
        self.assertEqual(main_tables.table_method_names(), METHODS)

    def test_kdey_labels(self):
        # since WP-31, BayCAP-KDEy is BayCAP-KDEy-h-inst and the cell pushforwards keep their own labels, also together
        both = pd.MultiIndex.from_product(
            [["acc_err"], ["BayCAP-KDEy-h-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-mlp-real"]],
            names=["metric", "method"],
        )
        labels = list(main_tables._latex_display_columns(both).get_level_values("method"))
        self.assertEqual(labels, ["\\bcap{}-KDEy-h-real", "\\bcapkdey", "\\bcap{}-KDEy-mlp-real"])

    def test_table_with_an_included_method(self):
        methods = main_tables.table_method_names(("BayCAP-KDEy-h-real",))
        rng = np.random.default_rng(2)
        df = pd.concat([_bags({m: list(rng.random(10)) for m in methods}, d) for d in ("d1", "d2")])
        latex = main_tables._results_latex_table(df, METRICS, False, ["d1", "d2"], ("BayCAP-KDEy-h-real",))
        self.assertIn(f"\\multicolumn{{{N + 1 - B}}}{{c|}}{{Our methods}}", latex)
        self.assertEqual(latex.count("\\rotatebox{90}{\\bcap{}-KDEy-h-real}"), 2)
        self.assertEqual(latex.count("\\rotatebox{90}{\\bcapkdey{}-cb-tcal-keep}"), 2)
        # without --include the extra method is not shown even if stored
        latex = main_tables._results_latex_table(df, METRICS, False, ["d1", "d2"])
        self.assertNotIn("-h}", latex)
