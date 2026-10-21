"""Coverage of the population and realised pushforwards of BayCAP (work-plan card WP-4, E2.3 of ``docs/ext2_baycap.md``).

Both figures read the per-pair table of :mod:`uncawe.tables.pushforward` (one point per (dataset, classifier) pair,
vanilla accuracy, $|U|=1000$; ``BayesCAP`` targets the population table, ``BayesCAP-real`` the realised table, and both
are scored against the realised table), coloured by the ratio $\\hat\\sigma_g/\\hat\\sigma_p$ of E2.1 (log scale):

- ``coverage``: coverage of the population interval against that of the realised predictive, with lines at the nominal
  level on both axes and the line $y=x$;
- ``approximation``: coverage of the realised predictive against the coverage that E2.1 predicted with its
  approximation, with the line $y=x$.

Both figures belong to E2.3, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/main/pushforward/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

from uncawe.config import CONFIDENCE_LEVEL
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    ordered_classifiers,
    plot_dir,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.pushforward import compute_comparison_table

PLOT_NAME = "pushforward"
FIGURES = ("coverage", "approximation")
COVERAGE_LIM = (-0.03, 1.03)


def _coverage_scatter(table: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str, nominal_lines: bool):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    norm = LogNorm(vmin=table["ratio"].min(), vmax=table["ratio"].max())
    sc = scatter_by_classifier(ax, table, x, y, "ratio", SEQUENTIAL_CMAP, norm)
    ax.plot(COVERAGE_LIM, COVERAGE_LIM, color="#595959", linewidth=1.2, linestyle="--", zorder=2)
    if nominal_lines:
        ax.axhline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.axvline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
    ax.set_xlim(COVERAGE_LIM)
    ax.set_ylim(COVERAGE_LIM)
    style_scatter_axes(ax, xlabel, ylabel)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$\\hat\\sigma_g/\\hat\\sigma_p$ (E2.1)", fontsize=font_size(12))
    classifier_legend(fig, table, loc="lower center", ncol=len(ordered_classifiers(table)), bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_coverage(table: pd.DataFrame):
    """Population against realised coverage: points that move up to the nominal line are pairs the predictive repairs."""
    return _coverage_scatter(
        table,
        "population_coverage",
        "realised_coverage",
        "coverage of BayCAP (population)",
        "coverage of BayCAP-real (realised)",
        nominal_lines=True,
    )


def plot_approximation(table: pd.DataFrame):
    """Exact against approximate coverage of the realised predictive: points on $y=x$ are pairs E2.1 got right."""
    return _coverage_scatter(
        table,
        "approx_coverage",
        "realised_coverage",
        "approximate coverage (E2.1)",
        "coverage of BayCAP-real (exact)",
        nominal_lines=False,
    )


PLOTTERS = {"coverage": plot_coverage, "approximation": plot_approximation}


def generate_plots(args):
    table = compute_comparison_table()
    out_dir = plot_dir(PLOT_NAME)
    for name in args.figures:
        fig = PLOTTERS[name](table)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Coverage of the population and realised pushforwards of BayCAP (E2.3).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
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
