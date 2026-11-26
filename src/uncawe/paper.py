r"""The tables and figures of the extended paper, regenerated together and collected in one folder (user's decision of
2026-11-24; the list is documented in ``docs/paper.md``).

``ARTEFACTS`` lists every tabular and figure the paper shows, with the command that produces it (``COMMANDS``, each
the ``EXP_<target>`` of the ``Makefile`` with its ``ARGS``) and the file it is copied from, under
``<OUT_DIR>/uncawe``. ``--generate`` runs every command once, in order, and stops at the first failure; ``--collect``
copies the artefacts into ``<OUT_DIR>/uncawe/paper/``: the tabulars in ``tables/``, the figures (pdf and png) in
``figures/``, and the LaTeX macros of the method names (``tables/common.py:METHOD_LATEX_MACROS``), which the tabulars
use, in ``macros.tex``. The two subfolders are emptied first, so they hold exactly the artefacts of the list; a
missing source stops the collection before anything is copied.

Every table and figure scores the realised table of each bag (``true_cts``), as ``main`` does.

On ``bcuda``: ``make run-paper`` (generate and collect), then ``make pull PULL_DIRS=paper``.

One table or figure at a time (user's request of 2026-11-26): ``--only <item>`` (``ITEMS``, one per command, named
``table-<name>`` or ``plot-<name>`` after its ``make`` target) runs that command and copies only its artefacts,
overwriting them in the paper folder and leaving the others in place; ``make run-paper-<item>`` (e.g.
``make run-paper-table-main``), then ``make pull PULL_DIRS=paper``.
"""

import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass

from uncawe import env
from uncawe.config import get_acc_names
from uncawe.tables.common import METHOD_LATEX_PREAMBLE

PAPER_DIR = "paper"
FIGURE_FORMATS = ("pdf", "png")


@dataclass(frozen=True)
class Command:
    """A ``make run-<target>`` with its ``ARGS``: ``module`` is the ``EXP_<target>`` of the ``Makefile``."""

    target: str
    module: str
    args: tuple[str, ...] = ()

    def argv(self) -> list[str]:
        return [sys.executable, "-m", *self.module.split()[2:], *self.args]

    def make(self) -> str:
        return f'make run-{self.target} ARGS="{" ".join(self.args)}"' if self.args else f"make run-{self.target}"


@dataclass(frozen=True)
class Artefact:
    """One tabular (``kind="tables"``, ``source`` with its ``.tex``) or one figure (``kind="figures"``, ``source``
    without extension, copied in every format of ``FIGURE_FORMATS``); ``source`` is relative to ``<OUT_DIR>/uncawe``
    and ``name`` is its name in the paper folder (the source name, unless the paper renames it)."""

    kind: str
    source: str
    command: Command
    name: str | None = None

    def paper_name(self) -> str:
        return self.name or os.path.basename(self.source)

    def files(self) -> list[tuple[str, str]]:
        """Pairs of (source, destination) paths relative to ``<OUT_DIR>/uncawe``."""
        destination = os.path.join(PAPER_DIR, self.kind, self.paper_name())
        if self.kind == "tables":
            return [(self.source, destination)]
        return [(f"{self.source}.{fmt}", f"{destination}.{fmt}") for fmt in FIGURE_FORMATS]


MAIN = Command("tables-main", "python -m uncawe.tables.main --classic", ("--tables", "--no-std"))
RANK = Command("tables-rank", "python -m uncawe.tables.rank --classic", ("--tables",))
TAU_SELECT = Command("tables-tau-select", "python -m uncawe.tables.tau_select", ("--tables",))
COVERAGE_AMPLITUDE = Command(
    "tables-coverage-amplitude", "python -m uncawe.tables.coverage_amplitude --classic", ("--tables",)
)
# the binary datasets only: DoC, QuAcc, LEAP, AC, CBPE and BayCAP-KDEy (user's request of 2026-11-26)
BINARY = Command("tables-binary", "python -m uncawe.tables.binary --classic", ("--tables", "--no-std"))
AVC = Command("plot-avc", "python -m uncawe.plots.amplitude_vs_coverage --domain classic")
# SQS-B left out of the shift profile, the violins of E5.5 and the figures of E4.4 (user's decision of 2026-11-25)
SHIFT_PROFILE = Command(
    "plot-shift-profile",
    "python -m uncawe.plots.shift_profile --domain classic",
    ("--figures", "pooled", "--exclude-methods", "SQS-B"),
)
# the SQS methods of the paper are those with KDEy as quantifier (card WP-36, user's decision of 2026-11-26): the main,
# rank and coverage/amplitude tables and the coverage vs amplitude plot follow tables/main.py:MAIN_METHODS, the shift
# profile its DEFAULT_METHODS; the violins of E5.5 and the figures of E4.4 leave out the PACC versions (passed with
# SQS-B, since --exclude-methods replaces the default), and the figure of E6.2 is that of PQ-KDEy-rho*
PACC_SQS = ("SQS-H", "SQS-Hbs", "PQ")
BINFORM_SIDE_PRIOR = Command(
    "plot-binform-side-prior",
    "python -m uncawe.plots.binform_side_prior",
    (
        "--figures",
        "pair-violins",
        "--groups",
        "all",
        "--metrics",
        "vanilla_accuracy",
        "--exclude-methods",
        "SQS-B",
        *PACC_SQS,
    ),
)
PQ_RHO = Command(
    "plot-pq-rho", "python -m uncawe.plots.pq_rho", ("--metrics", "vanilla_accuracy", "--quantifier", "kdey")
)
TARGET_SIZE_SCORES = Command(
    "plot-target-size-scores", "python -m uncawe.plots.target_size_scores", ("--exclude-methods", "SQS-B", *PACC_SQS)
)
# the tables whose CSV a figure reads, run before it (by --generate, also with --only): the per-pair CSV of E6.2 with
# KDEy and that of the E5.5 violins (which since WP-36 also has the KDEy references)
PQ_RHO_TABLES = Command("tables-pq-rho", "python -m uncawe.tables.pq_rho", ("--tables", "--quantifier", "kdey"))
BINFORM_SIDE_PRIOR_TABLES = Command(
    "tables-binform-side-prior", "python -m uncawe.tables.binform_side_prior", ("--tables",)
)
PREREQUISITES = {PQ_RHO: (PQ_RHO_TABLES,), BINFORM_SIDE_PRIOR: (BINFORM_SIDE_PRIOR_TABLES,)}
COMMANDS = (
    MAIN,
    RANK,
    TAU_SELECT,
    COVERAGE_AMPLITUDE,
    BINARY,
    AVC,
    SHIFT_PROFILE,
    BINFORM_SIDE_PRIOR,
    PQ_RHO,
    TARGET_SIZE_SCORES,
)

ACC_NAMES = tuple(get_acc_names())
SCORES = ("acc_err", "winkler")

ARTEFACTS = (
    # main table: one tabular per accuracy measure
    *(Artefact("tables", f"tables/main/tabular/main_classic_{acc}.tex", MAIN) for acc in ACC_NAMES),
    # rank tables: one tabular per accuracy measure and score
    *(
        Artefact("tables", f"tables/rank/tabular/main_classic_{acc}_rank_{score}.tex", RANK)
        for acc in ACC_NAMES
        for score in SCORES
    ),
    # tau chosen per pair on calibration bags (E5.3), compact table: one tabular per accuracy measure
    *(Artefact("tables", f"tables/tau_select/tabular/tau_select_compact_{acc}.tex", TAU_SELECT) for acc in ACC_NAMES),
    # coverage and amplitude: one tabular per accuracy measure (user's decision of 2026-11-25)
    *(
        Artefact(
            "tables",
            f"tables/coverage_amplitude/tabular/main_classic_{acc}_coverage_amplitude.tex",
            COVERAGE_AMPLITUDE,
        )
        for acc in ACC_NAMES
    ),
    # the binary datasets only, vanilla accuracy only (AC and CBPE estimate no other measure)
    Artefact("tables", "tables/binary/tabular/binary_classic_vanilla_accuracy.tex", BINARY),
    # coverage vs amplitude: linear axes and axes zoomed on the nominal values (user's decision of 2026-11-25)
    Artefact("figures", "plots/main/amplitude_vs_coverage/classic_vanilla_accuracy", AVC, name="amplitude_vs_coverage"),
    Artefact(
        "figures",
        "plots/main/amplitude_vs_coverage/classic_vanilla_accuracy_zoom",
        AVC,
        name="amplitude_vs_coverage_zoom",
    ),
    # the pooled shift profile with bars in place of the one by group of K (user's decision of 2026-11-25)
    Artefact("figures", "plots/main/shift_profile/shift_profile_pooled_classic_vanilla_accuracy", SHIFT_PROFILE),
    *(
        Artefact(
            "figures",
            f"plots/binform_side_prior/binform_side_prior_pair_violins_{position}_all_vanilla_accuracy",
            BINFORM_SIDE_PRIOR,
        )
        for position in ("edge", "centre")
    ),
    # with KDEy since 2026-11-26 (PQ-KDEy-rho*, under kdey/)
    Artefact("figures", "plots/pq_rho/kdey/pq_rho_ae_w_vanilla_accuracy", PQ_RHO),
    Artefact("figures", "plots/target_size/u_scores/target_size_u-scores", TARGET_SIZE_SCORES),
    Artefact("figures", "plots/target_size/v_scores/target_size_v-scores", TARGET_SIZE_SCORES),
)


def item_name(command: Command) -> str:
    """The name of the item of ``command``: its ``make`` target with ``tables-`` shortened to ``table-``."""
    return "table-" + command.target.removeprefix("tables-") if command.target.startswith("tables-") else command.target


# one item per command: name -> (command, its artefacts), in the order of COMMANDS
ITEMS = {item_name(command): (command, tuple(a for a in ARTEFACTS if a.command == command)) for command in COMMANDS}


def with_prerequisites(commands) -> list[Command]:
    """``commands``, each preceded by the commands of ``PREREQUISITES`` it needs (once each)."""
    return list(dict.fromkeys(c for command in commands for c in (*PREREQUISITES.get(command, ()), command)))


def generate(commands=COMMANDS) -> None:
    for command in with_prerequisites(commands):
        print(f"paper: {command.make()}", flush=True)
        subprocess.run(command.argv(), check=True)


def collect(root: str | None = None, artefacts=ARTEFACTS, clean: bool = True) -> list[str]:
    """Copy the artefacts into ``<root>/paper/`` (``root`` defaults to ``<OUT_DIR>/uncawe``); returns the copied
    destination paths. With ``clean`` (the full collection) the two subfolders are emptied first; without it (one item,
    ``--only``) the other files of the paper folder are kept and those of ``artefacts`` overwritten."""
    root = root or env.root_dir
    pairs = [pair for artefact in artefacts for pair in artefact.files()]
    missing = [source for source, _ in pairs if not os.path.exists(os.path.join(root, source))]
    if missing:
        raise FileNotFoundError("Missing paper sources (generate them first):\n" + "\n".join(missing))

    paper_dir = os.path.join(root, PAPER_DIR)
    for kind in ("tables", "figures"):
        if clean:
            shutil.rmtree(os.path.join(paper_dir, kind), ignore_errors=True)
        os.makedirs(os.path.join(paper_dir, kind), exist_ok=True)
    copied = []
    for source, destination in pairs:
        shutil.copy2(os.path.join(root, source), os.path.join(root, destination))
        copied.append(destination)
    with open(os.path.join(paper_dir, "macros.tex"), "w") as f:
        f.write(METHOD_LATEX_PREAMBLE + "\n")
    return copied


def main(argv=None):
    parser = ArgumentParser(description="Regenerate and collect the tables and figures of the extended paper.")
    parser.add_argument("--generate", action="store_true", help="Run every command of the paper artefacts")
    parser.add_argument("--collect", action="store_true", help=f"Copy the artefacts into <OUT_DIR>/uncawe/{PAPER_DIR}/")
    parser.add_argument("--list", action="store_true", help="Print the commands and the artefacts")
    parser.add_argument(
        "--only", choices=list(ITEMS), default=None, help="One table or figure: its command and its artefacts only"
    )
    args = parser.parse_args(argv)
    if not (args.generate or args.collect or args.list):
        parser.error("choose at least one of --generate, --collect, --list")
    commands, artefacts = (COMMANDS, ARTEFACTS) if args.only is None else ((ITEMS[args.only][0],), ITEMS[args.only][1])
    if args.list:
        for command in with_prerequisites(commands):
            print(command.make())
        for artefact in artefacts:
            for source, destination in artefact.files():
                print(f"{source} -> {destination}")
    if args.generate:
        generate(commands)
    if args.collect:
        copied = collect(artefacts=artefacts, clean=args.only is None)
        print(f"paper: {len(copied)} files in {os.path.join(env.root_dir, PAPER_DIR)}")


if __name__ == "__main__":
    main()
