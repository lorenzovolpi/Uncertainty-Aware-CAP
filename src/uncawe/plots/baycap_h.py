"""Scatter plots of BayCAP-H, the hierarchical prior on the recalls of BayCAP, on the pairs of ``main`` (work-plan card
WP-10, E2.7 of ``docs/ext2_baycap.md``).

The figures read the per-pair CSV of :mod:`uncawe.tables.baycap_h` (run ``tables/baycap_h.py --tables`` first): one point
per (dataset, classifier) pair, vanilla accuracy, $|U|=1000$, markers by classifier. ``BayesCAP-H-real``,
``BayesCAP-real`` and ``BayesCAP-EB-real`` target the realised table and are scored against it.

- ``coverage``: coverage of BayCAP-H against that of ``BayesCAP-real`` and of ``BayesCAP-EB-real``, two panels, coloured
  by $K$, with the line $y=x$ and lines at the nominal level;
- ``d4``: observed against predicted (D4) coverage and amplitude, two panels, coloured by $K$, with the line $y=x$;
- ``spread``: the residual variance $\\rho_H$ of BayCAP-H against $\\rho_u$ of ``BayesCAP-real`` and against its
  prevalence part $\\bar r^\\top\\Sigma_\\pi\\bar r$, two panels, log scales, coloured by $K$, with the line $y=x$; pairs
  with a value that is not positive are counted in the title;
- ``d3``: the share of divergent bags of BayCAP-H against $P(\\min_j\\min(a_j,b_j)<1\\mid V)$, $P(\\nu>\\min_jn_j\\mid V)$
  (both from $V$ on the grid of E2.6) and the posterior $P(\\nu>\\min_jn_j)$ of the run, three panels, coloured by
  whether the pair has a class perfect or never right on $V$, with a line at ``DIVERGENT_SHARE``.

The four figures belong to E2.7, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_h/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from uncawe import env
from uncawe.config import CONFIDENCE_LEVEL
from uncawe.plots.common import (
    CLASSIFIER_MARKERS,
    DEFAULT_MARKER,
    SCATTER_EDGE_COLOR,
    SEQUENTIAL_CMAP,
    classifier_legend_handles,
    font_size,
    marker_size,
    method_plot_label,
    ordered_classifiers,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.baycap_h import DIVERGENT_SHARE, METHOD, REFERENCES, csv_path

PLOT_NAME = "baycap_h"
FIGURES = ("coverage", "d4", "spread", "d3")
COVERAGE_LIM = (-0.03, 1.03)
BOUNDARY_COLORS = {False: "#4C78A8", True: "#F58518"}
BOUNDARY_LABELS = {False: "no class perfect or never right on $V$", True: "a class perfect or never right on $V$"}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_h.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _scatter(ax, table: pd.DataFrame, x: str, y: str):
    return scatter_by_classifier(ax, table, x, y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))


def _identity(ax, lim: tuple[float, float]):
    ax.plot(lim, lim, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)


def _limits(table: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    lo = float(np.nanmin([table[x].min(), table[y].min(), 0.0]))
    hi = float(np.nanmax([table[x].max(), table[y].max()]))
    pad = 0.03 * (hi - lo) if hi > lo else 0.01
    return lo - pad, hi + pad


def _legend(fig, table: pd.DataFrame, extra: list[tuple[Line2D, str]] = ()):
    classifiers = ordered_classifiers(table)
    handles, labels = classifier_legend_handles(classifiers), list(classifiers)
    for handle, label in extra:
        handles.append(handle)
        labels.append(label)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=font_size(10))


def _finish(fig, axes, sc, table: pd.DataFrame):
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label("$K$", fontsize=font_size(12))
    _legend(fig, table)


def plot_coverage(table: pd.DataFrame):
    """Coverage of BayCAP-H against that of the two stored methods."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained", sharex=True, sharey=True)
    sc = None
    y = f"{METHOD}_coverage"
    for ax, reference in zip(axes, REFERENCES.values()):
        sc = _scatter(ax, table, f"{reference}_coverage", y)
        _identity(ax, COVERAGE_LIM)
        ax.axhline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
        ax.axvline(CONFIDENCE_LEVEL, color="#8c8c8c", linewidth=0.8, zorder=1)
        style_scatter_axes(
            ax, f"coverage of {method_plot_label(reference)}", f"coverage of {method_plot_label(METHOD)}"
        )
    axes[1].tick_params(labelleft=True)
    fig.suptitle("coverage of the realised table, hierarchical prior against the stored priors", fontsize=font_size(11))
    _finish(fig, axes, sc, table)
    return fig


def plot_d4(table: pd.DataFrame):
    """Observed against predicted (D4) coverage and amplitude of BayCAP-H."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    sc = None
    for ax, score in zip(axes, ("coverage", "amplitude")):
        x, y = f"d4_pred_{score}", f"{METHOD}_{score}"
        sc = _scatter(ax, table, x, y)
        _identity(ax, COVERAGE_LIM if score == "coverage" else _limits(table, x, y))
        style_scatter_axes(ax, f"predicted {score} (D4, from BayCAP-real)", f"observed {score}, BayCAP-H-real")
    fig.suptitle(
        "D4: prediction of the hierarchical prior from $V$ and the draws of BayCAP-real", fontsize=font_size(11)
    )
    _finish(fig, axes, sc, table)
    return fig


def plot_spread(table: pd.DataFrame):
    """Residual variance of BayCAP-H against that of BayCAP-real and against its prevalence part (§T.15)."""
    panels = (
        ("uniform_residual_var", "residual variance $\\rho_u$, BayCAP-real"),
        ("h_prevalence_part", "prevalence part $\\bar r^\\top\\Sigma_\\pi\\bar r$, BayCAP-H-real"),
    )
    y = "h_residual_var"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    sc, excluded = None, []
    for ax, (x, xlabel) in zip(axes, panels):
        shown = table.loc[(table[x] > 0) & (table[y] > 0)]
        excluded.append(len(table) - len(shown))
        sc = _scatter(ax, shown, x, y)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not shown.empty:
            lo = float(min(shown[x].min(), shown[y].min())) / 1.5
            hi = float(max(shown[x].max(), shown[y].max())) * 1.5
            _identity(ax, (lo, hi))
        style_scatter_axes(ax, xlabel, "residual variance $\\rho_H$, BayCAP-H-real")
    suffix = f" ({max(excluded)} pairs with a value that is not positive not shown)" if max(excluded) else ""
    fig.suptitle("split of the spread of the accuracy draws" + suffix, fontsize=font_size(11))
    _finish(fig, axes, sc, table)
    return fig


D3_PANELS = {
    "post_alpha_risk": "$P(\\min_j\\min(a_j,b_j)<1\\mid V)$ (grid)",
    "funnel_risk": "$P(\\nu>\\min_jn_j\\mid V)$ (grid)",
    "h_p_nu_above_min_n": "$P(\\nu>\\min_jn_j)$ (run, mean over the bags)",
}


def plot_d3(table: pd.DataFrame):
    """Share of divergent bags of BayCAP-H against its sampler-risk indicators."""
    y = "share_divergent"
    shown = table.loc[table[y].notna()]
    boundary = (shown["n_perfect_classes"] + shown["n_never_right_classes"]) > 0
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), layout="constrained", sharey=True)
    for ax, (x, xlabel) in zip(axes, D3_PANELS.items()):
        for classifier in ordered_classifiers(shown):
            mask = shown["classifier"].astype(str) == classifier
            ax.scatter(
                shown.loc[mask, x],
                shown.loc[mask, y],
                c=[BOUNDARY_COLORS[bool(b)] for b in boundary[mask]],
                marker=CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER),
                s=marker_size(45),
                edgecolors=SCATTER_EDGE_COLOR,
                linewidths=0.6,
                zorder=3,
            )
        ax.axhline(DIVERGENT_SHARE, color="#8c8c8c", linewidth=0.8, linestyle="--", zorder=1)
        style_scatter_axes(ax, xlabel, "share of divergent bags, BayCAP-H-real")
    excluded = len(table) - len(shown)
    suffix = f" ({excluded} pairs without diagnostics not shown)" if excluded else ""
    fig.suptitle("D3: sampler risk of the hierarchical prior" + suffix, fontsize=font_size(11))
    extra = [
        (Line2D([], [], marker="o", linestyle="", color=BOUNDARY_COLORS[b], markersize=7), BOUNDARY_LABELS[b])
        for b in (False, True)
    ]
    _legend(fig, shown, extra)
    return fig


PLOTTERS = {"coverage": plot_coverage, "d4": plot_d4, "spread": plot_spread, "d3": plot_d3}


def generate_plots(args):
    table = load_table(args.csv)
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
    parser = ArgumentParser(description="Scatter plots of BayCAP-H, the hierarchical prior on the recalls (E2.7).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_h.py (default: its output)")
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
