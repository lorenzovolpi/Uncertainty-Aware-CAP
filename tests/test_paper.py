import os
import re
import tempfile
import unittest

from uncawe import paper
from uncawe.tables.common import METHOD_LATEX_PREAMBLE

MAKEFILE = os.path.join(os.path.dirname(__file__), os.pardir, "Makefile")


def _make_experiments() -> dict[str, str]:
    with open(MAKEFILE) as f:
        return {
            m.group(1): m.group(2).strip() for m in re.finditer(r"^EXP_([\w-]+)\s*:=\s*(.+)$", f.read(), re.MULTILINE)
        }


class TestManifest(unittest.TestCase):
    def test_commands_are_the_make_targets(self):
        experiments = _make_experiments()
        for command in paper.COMMANDS:
            with self.subTest(target=command.target):
                self.assertEqual(experiments[command.target], command.module)
        self.assertEqual(experiments["paper"], "python -m uncawe.paper --generate --collect")

    def test_every_artefact_command_is_run(self):
        self.assertLessEqual({artefact.command for artefact in paper.ARTEFACTS}, set(paper.COMMANDS))
        self.assertEqual(len(set(paper.COMMANDS)), len(paper.COMMANDS))

    def test_destinations_are_unique(self):
        destinations = [d for artefact in paper.ARTEFACTS for _, d in artefact.files()]
        self.assertEqual(len(destinations), len(set(destinations)))

    def test_the_list(self):
        names = [a.paper_name() for a in paper.ARTEFACTS]
        self.assertEqual(sum(a.kind == "tables" for a in paper.ARTEFACTS), 3 + 6 + 3 + 3)
        self.assertIn("amplitude_vs_coverage", names)
        self.assertIn("amplitude_vs_coverage_zoom", names)
        self.assertIn("main_classic_macro-F1_rank_winkler.tex", names)
        self.assertIn("tau_select_compact_macro-K.tex", names)
        figure = next(a for a in paper.ARTEFACTS if a.name == "amplitude_vs_coverage")
        self.assertEqual(
            figure.files(),
            [
                (f"{figure.source}.pdf", "paper/figures/amplitude_vs_coverage.pdf"),
                (f"{figure.source}.png", "paper/figures/amplitude_vs_coverage.png"),
            ],
        )

    def test_argv_runs_the_module_with_its_args(self):
        self.assertEqual(paper.MAIN.argv()[1:], ["-m", "uncawe.tables.main", "--classic", "--tables", "--no-std"])
        self.assertEqual(paper.MAIN.make(), 'make run-tables-main ARGS="--tables --no-std"')
        self.assertEqual(paper.AVC.make(), "make run-plot-avc")


class TestCollect(unittest.TestCase):
    def _write_sources(self, root, artefacts):
        for artefact in artefacts:
            for source, _ in artefact.files():
                os.makedirs(os.path.dirname(os.path.join(root, source)), exist_ok=True)
                with open(os.path.join(root, source), "w") as f:
                    f.write(source)

    def test_collect_copies_and_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_sources(root, paper.ARTEFACTS)
            stale = os.path.join(root, "paper", "tables", "old_table.tex")
            os.makedirs(os.path.dirname(stale))
            open(stale, "w").close()
            copied = paper.collect(root)
            self.assertFalse(os.path.exists(stale))
            self.assertEqual(len(copied), sum(len(a.files()) for a in paper.ARTEFACTS))
            with open(os.path.join(root, "paper", "figures", "amplitude_vs_coverage.png")) as f:
                self.assertEqual(f.read(), "plots/main/amplitude_vs_coverage/classic_vanilla_accuracy.png")
            with open(os.path.join(root, "paper", "macros.tex")) as f:
                self.assertEqual(f.read().strip(), METHOD_LATEX_PREAMBLE)

    def test_missing_source_stops_before_copying(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_sources(root, paper.ARTEFACTS[1:])
            kept = os.path.join(root, "paper", "tables", "kept.tex")
            os.makedirs(os.path.dirname(kept))
            open(kept, "w").close()
            with self.assertRaises(FileNotFoundError):
                paper.collect(root)
            self.assertTrue(os.path.exists(kept))


if __name__ == "__main__":
    unittest.main()
