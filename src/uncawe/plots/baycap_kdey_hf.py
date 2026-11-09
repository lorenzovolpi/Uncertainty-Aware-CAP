"""Scatter plots of BayCAP-KDEy with a random $C$ and the Hf prior on the pairs of ``main`` (work-plan card WP-13, E3.2 of
``docs/ext3_posterior_propagation.md``).

The figures read the CSVs of :mod:`uncawe.tables.baycap_kdey_hf` (run ``tables/baycap_kdey_hf.py --tables`` first).
Vanilla accuracy, $|U|=1000$; one panel per input classifier ($h$, mlp). The methods ``BayCAP-KDEy-Hf-<s>-real`` and
their references target the realised table of the target bag and are scored against it.

- ``coverage_fixed``: coverage of each new method against that of the realised BayCAP-KDEy of E3.1 with the same input
  classifier (fixed $\\hat C$), one point per (dataset, classifier) pair, markers by classifier, coloured by $K$, lines
  at $0.95$ and $y=x$;
- ``coverage_hf``: the same against ``BayesCAP-Hf-real``;
- ``information``: label-free, for each cell $(k,j)$ of each pair, the ratio of the posterior standard deviation of
  $C_{kj}$ to its standard deviation from $V$ alone against the count $f_{kj}$ (logarithmic axes), coloured by $K$,
  with the line $y=1$; the cells with $f_{kj}=0$ or $\\hat C_{kj}\\in\\{0,1\\}$ have no ratio and are not shown.

The figures belong to E3.2, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_kdey_hf/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from uncawe import env
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.baycap_kdey_hf import FIXED, HF, NEW, SOURCES, csv_path

PLOT_NAME = "baycap_kdey_hf"
FIGURES = ("coverage_fixed", "coverage_hf", "information")
NOMINAL = 0.95
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(name: str | None = None, path: str | None = None) -> pd.DataFrame:
    path = path or csv_path(name=name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_kdey_hf.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _source_axes(**kwargs):
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(11.0, 5.0), layout="constrained", **kwargs)
    for ax, source in zip(axes, SOURCES):
        ax.set_title(f"$s={source}$", fontsize=font_size(11))
    return fig, axes


def _coverage(table: pd.DataFrame, references: dict[str, str], title: str):
    fig, axes = _source_axes(sharex=True, sharey=True)
    sc = None
    for ax, source in zip(axes, SOURCES):
        reference = references[source]
        sc = scatter_by_classifier(
            ax, table, f"{reference}_coverage", f"{NEW[source]}_coverage", "n_classes", SEQUENTIAL_CMAP, _k_norm(table)
        )
        ax.plot((0, 1), (0, 1), **IDENTITY_LINE)
        ax.axhline(NOMINAL, **REFERENCE_LINE)
        ax.axvline(NOMINAL, **REFERENCE_LINE)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style_scatter_axes(
            ax, f"coverage, {method_plot_label(reference)}", f"coverage, {method_plot_label(NEW[source])}"
        )
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, fontsize=font_size(12))
    return fig


def plot_coverage_fixed(table: pd.DataFrame):
    """Coverage of each new method against the realised BayCAP-KDEy of E3.1 with the same input classifier."""
    return _coverage(table, FIXED, "coverage with a random $C$ against a fixed $\\hat C$")


def plot_coverage_hf(table: pd.DataFrame):
    """Coverage of each new method against ``BayesCAP-Hf-real``."""
    return _coverage(
        table, dict.fromkeys(SOURCES, HF), "coverage with a kernel likelihood against the counts of BayCAP-Hf"
    )


def plot_information(cells: pd.DataFrame):
    """The ratio of the posterior sd of each cell to its sd from $V$ alone, against $f_{kj}$."""
    fig, axes = _source_axes(sharex=True, sharey=True)
    norm = _k_norm(cells)
    sc = None
    hidden = 0
    for ax, source in zip(axes, SOURCES):
        part = cells.loc[cells["input_classifier"] == source]
        shown = part.loc[np.isfinite(part["sd_ratio"]) & (part["sd_ratio"] > 0) & (part["f"] > 0)]
        hidden += len(part) - len(shown)
        sc = ax.scatter(
            shown["f"], shown["sd_ratio"], c=shown["n_classes"], cmap=SEQUENTIAL_CMAP, norm=norm, s=14, alpha=0.6
        )
        ax.axhline(1.0, **IDENTITY_LINE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        style_scatter_axes(ax, "validation count $f_{kj}$", "posterior sd / sd from $V$ alone")
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    suffix = f" ({hidden} cells without a ratio not shown)" if hidden else ""
    fig.suptitle("what the target bag adds on each cell of $C$" + suffix, fontsize=font_size(12))
    return fig


PLOTTERS = {"coverage_fixed": plot_coverage_fixed, "coverage_hf": plot_coverage_hf}


def generate_plots(args):
    table = load_table(path=args.csv)
    out_dir = plot_dir()
    for name in args.figures:
        fig = plot_information(load_table("cells")) if name == "information" else PLOTTERS[name](table)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of BayCAP-KDEy with a random C and the Hf prior (E3.2).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_kdey_hf.py (default: its output)")
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args()
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
