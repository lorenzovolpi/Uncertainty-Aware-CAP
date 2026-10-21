"""Scatter plots of the causes of BayCAP's displacement (work-plan card WP-3, E2.2 of ``docs/ext2_baycap.md``).

Both figures read the per-pair table of :mod:`uncawe.tables.displacement` (one point per (dataset, classifier) pair,
vanilla accuracy, $|U|=1000$; BayCAP targets the population table and is scored against the realised table):

- ``prior`` (cause 1): mean recall term against its mean prior part, coloured by $K$, with the line $y=x$;
- ``conditioning`` (causes 2 and 3): $1/\\sigma_{\\min}(Q^\\top\\hat CQ)$ (log scale) against the accuracy of $h$ on $V$,
  in two panels coloured by coverage and by shift. Pairs with a singular $\\hat C$ cannot be placed on the log axis and
  are counted in the panel title.

Both figures belong to E2.2, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/main/displacement/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm, Normalize, TwoSlopeNorm

from uncawe.plots.common import (
    CLASSIFIER_MARKERS,
    DEFAULT_MARKER,
    classifier_legend_handles,
    font_size,
    marker_size,
    ordered_classifiers,
    plot_dir,
)
from uncawe.tables.displacement import compute_pair_table

PLOT_NAME = "displacement"
FIGURES = ("prior", "conditioning")
GRID_COLOR = "#d9d9d9"
EDGE_COLOR = "#1f1f1f"
# the lightest quarter of Blues is close to the white background, so it is cut off
SEQUENTIAL_CMAP = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.25, 1.0, 256)), name="Blues_trunc")
DIVERGING_CMAP = "RdBu_r"


def _scatter_by_classifier(ax, table: pd.DataFrame, x: str, y: str, color, cmap, norm):
    """Draw one marker shape per classifier; colour encodes ``color`` through ``cmap`` and ``norm``."""
    for classifier in ordered_classifiers(table):
        sub = table.loc[table["classifier"].astype(str) == classifier]
        sc = ax.scatter(
            sub[x],
            sub[y],
            c=sub[color],
            cmap=cmap,
            norm=norm,
            marker=CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER),
            s=marker_size(45),
            edgecolors=EDGE_COLOR,
            linewidths=0.6,
            zorder=3,
        )
    return sc


def _style_axes(ax, xlabel: str, ylabel: str):
    ax.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_xlabel(xlabel, fontsize=font_size(12))
    ax.set_ylabel(ylabel, fontsize=font_size(12))
    ax.tick_params(axis="both", labelsize=font_size(10))


def _classifier_legend(fig, table: pd.DataFrame, **kwargs):
    classifiers = ordered_classifiers(table)
    fig.legend(classifier_legend_handles(classifiers), classifiers, frameon=True, fontsize=font_size(10), **kwargs)


def plot_prior(table: pd.DataFrame):
    """Cause 1: recall term against its prior part, coloured by $K$; points on $y=x$ have the prior's recall error."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    k = table["n_classes"].astype(float)
    sc = _scatter_by_classifier(
        ax, table, "prior_part", "recall_term", "n_classes", SEQUENTIAL_CMAP, LogNorm(vmin=k.min(), vmax=k.max())
    )
    lo = float(min(table["prior_part"].min(), table["recall_term"].min(), 0.0))
    hi = float(max(table["prior_part"].max(), table["recall_term"].max(), 0.0))
    pad = 0.05 * (hi - lo) if hi > lo else 0.01
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#595959", linewidth=1.2, linestyle="--", zorder=2)
    ax.axhline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
    ax.axvline(0, color="#8c8c8c", linewidth=0.8, zorder=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    _style_axes(ax, "prior part of the recall term $R_0$", "recall term $R$")
    ticks = sorted(table["n_classes"].astype(int).unique())
    cbar = fig.colorbar(sc, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    cbar.set_label("$K$", fontsize=font_size(12))
    cbar.ax.minorticks_off()
    n_classifiers = len(ordered_classifiers(table))
    _classifier_legend(fig, table, loc="lower center", ncol=n_classifiers, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_conditioning(table: pd.DataFrame):
    """Causes 2 and 3: $1/\\sigma_{\\min}$ against the accuracy on $V$, coloured by coverage and by shift."""
    finite = table.loc[np.isfinite(table["inv_sigma_min"])]
    n_singular = len(table) - len(finite)
    shift_lim = float(np.abs(table["shift"]).max()) or 1.0
    panels = (
        ("coverage", "coverage", SEQUENTIAL_CMAP, Normalize(vmin=0.0, vmax=1.0)),
        ("shift", "shift $\\hat A-A$", DIVERGING_CMAP, TwoSlopeNorm(vcenter=0.0, vmin=-shift_lim, vmax=shift_lim)),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, (col, label, cmap, norm) in zip(axes, panels):
        sc = _scatter_by_classifier(ax, finite, "inv_sigma_min", "val_accuracy", col, cmap, norm)
        ax.set_xscale("log")
        _style_axes(ax, "$1/\\sigma_{\\min}(Q^\\top\\hat CQ)$", "accuracy of $h$ on $V$")
        title = f"coloured by {label}"
        if n_singular:
            title += f" ({n_singular} singular {'pair' if n_singular == 1 else 'pairs'} not shown)"
        ax.set_title(title, fontsize=font_size(11))
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label, fontsize=font_size(11))
    axes[1].tick_params(labelleft=True)
    _classifier_legend(fig, table, loc="lower center", ncol=len(ordered_classifiers(table)), bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


PLOTTERS = {"prior": plot_prior, "conditioning": plot_conditioning}


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
    parser = ArgumentParser(description="Scatter plots of the causes of BayCAP's displacement (E2.2).")
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
