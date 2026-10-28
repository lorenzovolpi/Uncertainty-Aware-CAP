"""Scatter plots of the empirical-Bayes prior of BayCAP on the binary datasets of ``quapy`` (work-plan card WP-7, E2.5 of
``docs/ext2_baycap.md``).

The figures read the per-pair table of :mod:`uncawe.tables.eb_binary`: one point per (dataset, classifier) pair of E2.5
and of the binary pairs of ``main`` (E2.4, drawn with a ring), vanilla accuracy, $|U|=1000$. ``BayesCAP`` and
``BayesCAP-EB`` target the population table, ``BayesCAP-real`` and ``BayesCAP-EB-real`` the realised table, and all four
are scored against the realised table. Markers by classifier.

- ``fit``: the gap statistic $|z|$ against $\\hat\\nu$ (log scale), coloured by whether $\\hat\\nu$ is at a bound of the fit,
  with a vertical line at $|z|=1$; pairs with $z$ NaN or infinite cannot be placed and are counted in the title;
- ``collapse``: the left-out standard deviation (log scale) against the change of realised coverage (``BayesCAP-EB-real``
  minus ``BayesCAP-real``), coloured as ``fit``; pairs with a NaN or zero left-out standard deviation are counted in the
  title;
- ``pull``: the predicted change of the recall term (check: EB minus $\\operatorname{Dir}(\\mathbf 1)$) against the
  observed one ($R$ of ``BayesCAP-EB`` minus $R$ of ``BayesCAP``), coloured by $|\\hat r_1-\\hat r_2|$, with the line
  $y=x$.

The three figures belong to E2.5, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/eb_binary/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from uncawe import env
from uncawe.experiments.eb_binary import EXPERIMENT
from uncawe.methods.baycap import EB_NU_BOUNDS
from uncawe.plots.common import (
    CLASSIFIER_MARKERS,
    DEFAULT_MARKER,
    SCATTER_EDGE_COLOR,
    SEQUENTIAL_CMAP,
    classifier_legend_handles,
    font_size,
    marker_size,
    ordered_classifiers,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.eb_binary import compute_table

PLOT_NAME = "eb_binary"
FIGURES = ("fit", "collapse", "pull")
BOUND_COLORS = {False: "#4C78A8", True: "#F58518"}
BOUND_LABELS = {False: "$\\hat\\nu$ inside its bounds", True: "$\\hat\\nu$ at a bound"}
RING_COLOR = "#1f1f1f"


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT)
    os.makedirs(path, exist_ok=True)
    return path


def _rings(ax, table: pd.DataFrame, x: str, y: str):
    """Draw a ring around the pairs of E2.4."""
    sub = table.loc[table["source"] == "E2.4"]
    ax.scatter(sub[x], sub[y], s=marker_size(140), facecolors="none", edgecolors=RING_COLOR, linewidths=1.0, zorder=2)


def _scatter_by_bound(ax, table: pd.DataFrame, x: str, y: str):
    for classifier in ordered_classifiers(table):
        sub = table.loc[table["classifier"].astype(str) == classifier]
        ax.scatter(
            sub[x],
            sub[y],
            c=[BOUND_COLORS[bool(b)] for b in sub["eb_nu_at_bound"]],
            marker=CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER),
            s=marker_size(45),
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=0.6,
            zorder=3,
        )
    _rings(ax, table, x, y)


def _legend(fig, table: pd.DataFrame, colour_by_bound: bool):
    classifiers = ordered_classifiers(table)
    handles, labels = classifier_legend_handles(classifiers), list(classifiers)
    if colour_by_bound:
        for at_bound in (False, True):
            handles.append(Line2D([], [], marker="o", linestyle="", color=BOUND_COLORS[at_bound], markersize=7))
            labels.append(BOUND_LABELS[at_bound])
    handles.append(Line2D([], [], marker="o", linestyle="", markerfacecolor="none", color=RING_COLOR, markersize=10))
    labels.append("binary datasets of main (E2.4)")
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=font_size(10))


def _excluded_title(ax, title: str, excluded: int, reason: str):
    suffix = f" ({excluded} pairs with {reason} not shown)" if excluded else ""
    ax.set_title(title + suffix, fontsize=font_size(11))


def plot_fit(table: pd.DataFrame):
    """$|z|$ against $\\hat\\nu$: the fit should reach the bound mostly when $|z|<1$ (§T.10, prediction 1)."""
    data = table.assign(abs_z=table["gap_z"].abs())
    shown = data.loc[np.isfinite(data["abs_z"])]
    fig, ax = plt.subplots(figsize=(7.5, 6.0), layout="constrained")
    _scatter_by_bound(ax, shown, "abs_z", "eb_nu")
    ax.set_yscale("log")
    ax.axvline(1, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(EB_NU_BOUNDS[1], color="#8c8c8c", linewidth=0.8, zorder=1)
    style_scatter_axes(ax, "gap statistic $|z|$ of the recalls on $V$", "strength $\\hat\\nu$ of the fitted prior")
    _excluded_title(ax, "fit of the empirical-Bayes prior", len(data) - len(shown), "$z$ undefined or infinite")
    _legend(fig, shown, colour_by_bound=True)
    return fig


def plot_collapse(table: pd.DataFrame):
    """Left-out standard deviation against the change of realised coverage (§T.10, prediction 2)."""
    shown = table.loc[table["left_out_sd"] > 0]
    fig, ax = plt.subplots(figsize=(7.5, 6.0), layout="constrained")
    _scatter_by_bound(ax, shown, "left_out_sd", "realised_coverage_change")
    ax.set_xscale("log")
    ax.axhline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
    style_scatter_axes(
        ax,
        "left-out standard deviation $\\sigma_{\\mathrm{lo}}$",
        "change of coverage (BayCAP-EB-real $-$ BayCAP-real)",
    )
    _excluded_title(ax, "collapse of the interval", len(table) - len(shown), "$\\sigma_{\\mathrm{lo}}$ NaN or zero")
    _legend(fig, shown, colour_by_bound=True)
    return fig


def plot_pull(table: pd.DataFrame):
    """Predicted against observed change of the recall term, coloured by the gap of the recalls (prediction 3)."""
    x, y = "predicted_recall_change", "observed_recall_change"
    fig, ax = plt.subplots(figsize=(7.5, 6.0), layout="constrained")
    gap = table["recall_gap"]
    sc = scatter_by_classifier(ax, table, x, y, "recall_gap", SEQUENTIAL_CMAP, Normalize(0, float(gap.max()) or 1.0))
    _rings(ax, table, x, y)
    lo = float(min(table[x].min(), table[y].min(), 0.0))
    hi = float(max(table[x].max(), table[y].max(), 0.0))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    style_scatter_axes(
        ax,
        "predicted change of $R$ (check: EB $-$ $\\operatorname{Dir}(\\mathbf{1})$)",
        "observed change of $R$ (BayCAP-EB $-$ BayCAP)",
    )
    ax.set_title("pull of the empirical-Bayes prior", fontsize=font_size(11))
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("$|\\hat r_1-\\hat r_2|$", fontsize=font_size(12))
    _legend(fig, table, colour_by_bound=False)
    return fig


PLOTTERS = {"fit": plot_fit, "collapse": plot_collapse, "pull": plot_pull}


def generate_plots(args):
    table = compute_table()
    out_dir = plot_dir()
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
    parser = ArgumentParser(description="Scatter plots of the empirical-Bayes prior on binary datasets (E2.5).")
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
