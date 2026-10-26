"""Scatter plots of BayCAP with an empirical-Bayes prior on the recalls (work-plan card WP-6, E2.4 of ``docs/ext2_baycap.md``).

Both figures read the per-pair table of :mod:`uncawe.tables.eb_prior` (one point per (dataset, classifier) pair, vanilla
accuracy, $|U|=1000$; ``BayesCAP`` and ``BayesCAP-EB`` target the population table, ``BayesCAP-real`` and
``BayesCAP-EB-real`` the realised table, and all four are scored against the realised table), coloured by $K$:

- ``check``: the change of the recall term predicted by the check (mean recall error of the empirical-Bayes estimates
  minus that of the $\\operatorname{Dir}(\\mathbf 1)$ estimates) against the observed one ($R$ of ``BayesCAP-EB`` minus
  $R$ of ``BayesCAP``), and, in a second panel that validates the check, the $\\operatorname{Dir}(\\mathbf 1)$ check
  against $R$ of ``BayesCAP``; both with the line $y=x$;
- ``coverage``: coverage with the $\\operatorname{Dir}(\\mathbf 1)$ prior against coverage with the empirical-Bayes prior,
  one panel per pushforward, with lines at the nominal level on both axes and the line $y=x$.

Both figures belong to E2.4, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/main/eb_prior/``.
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
    method_plot_label,
    ordered_classifiers,
    plot_dir,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.eb_prior import PUSHFORWARDS, compute_pair_table

PLOT_NAME = "eb_prior"
FIGURES = ("check", "coverage")
COVERAGE_LIM = (-0.03, 1.03)


def _k_colorbar(fig, sc, axes, table: pd.DataFrame):
    ticks = sorted(table["n_classes"].astype(int).unique())
    cbar = fig.colorbar(sc, ax=axes, ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    cbar.set_label("$K$", fontsize=font_size(12))
    cbar.ax.minorticks_off()


def _k_norm(table: pd.DataFrame) -> LogNorm:
    k = table["n_classes"].astype(float)
    return LogNorm(vmin=k.min(), vmax=k.max())


def _diagonal_limits(table: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    lo = float(min(table[x].min(), table[y].min(), 0.0))
    hi = float(max(table[x].max(), table[y].max(), 0.0))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    return lo - pad, hi + pad


def _finish(fig, axes, table: pd.DataFrame, sc):
    _k_colorbar(fig, sc, axes, table)
    classifier_legend(fig, table, loc="lower center", ncol=len(ordered_classifiers(table)), bbox_to_anchor=(0.45, 0))


def plot_check(table: pd.DataFrame):
    """Predicted against observed change of the recall term, and the Dir(1) check against $R$ of BayCAP."""
    panels = (
        (
            "predicted_recall_change",
            "observed_recall_change",
            "predicted change of $R$ (check: EB $-$ $\\operatorname{Dir}(\\mathbf 1)$)",
            "observed change of $R$ (BayCAP-EB $-$ BayCAP)",
            "effect of the prior",
        ),
        (
            "check_uniform_mean",
            "BayesCAP_recall_term",
            "$\\operatorname{Dir}(\\mathbf 1)$ check, mean $\\Delta$",
            "recall term $R$ of BayCAP",
            "validation of the check",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    norm = _k_norm(table)
    sc = None
    for ax, (x, y, xlabel, ylabel, title) in zip(axes, panels):
        sc = scatter_by_classifier(ax, table, x, y, "n_classes", SEQUENTIAL_CMAP, norm)
        lim = _diagonal_limits(table, x, y)
        ax.plot(lim, lim, color="#595959", linewidth=1.2, linestyle="--", zorder=2)
        ax.axhline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.axvline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        style_scatter_axes(ax, xlabel, ylabel)
        ax.set_title(title, fontsize=font_size(11))
    _finish(fig, axes, table, sc)
    return fig


def plot_coverage(table: pd.DataFrame):
    """Coverage with the Dir(1) prior against coverage with the empirical-Bayes prior, one panel per pushforward."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    norm = _k_norm(table)
    sc = None
    for ax, (name, (uniform, eb)) in zip(axes, PUSHFORWARDS.items()):
        sc = scatter_by_classifier(
            ax, table, f"{uniform}_coverage", f"{eb}_coverage", "n_classes", SEQUENTIAL_CMAP, norm
        )
        ax.plot(COVERAGE_LIM, COVERAGE_LIM, color="#595959", linewidth=1.2, linestyle="--", zorder=2)
        ax.axhline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.axvline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.set_xlim(COVERAGE_LIM)
        ax.set_ylim(COVERAGE_LIM)
        style_scatter_axes(ax, f"coverage of {method_plot_label(uniform)}", f"coverage of {method_plot_label(eb)}")
        ax.set_title(f"{name} pushforward", fontsize=font_size(11))
    axes[1].tick_params(labelleft=True)
    _finish(fig, axes, table, sc)
    return fig


PLOTTERS = {"check": plot_check, "coverage": plot_coverage}


def generate_plots(args):
    table = compute_pair_table()
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
    parser = ArgumentParser(description="Scatter plots of BayCAP with an empirical-Bayes prior on the recalls (E2.4).")
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
