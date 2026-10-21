"""Scatter plots of the causes of BayCAP's displacement (work-plan card WP-3, E2.2 of ``docs/ext2_baycap.md``).

Both figures read the per-pair table of :mod:`uncawe.tables.displacement` (one point per (dataset, classifier) pair,
vanilla accuracy, $|U|=1000$; ``BayesCAP`` targets the population table, ``BayesCAP-real`` the realised table, and both
are scored against the realised table). The method is an argument (default ``BayesCAP``):

- ``prior`` (cause 1): mean recall term against its mean prior part, coloured by $K$, with the line $y=x$;
- ``conditioning`` (causes 2 and 3): $1/\\sigma_{\\min}(Q^\\top\\hat CQ)$ (log scale) against the accuracy of $h$ on $V$,
  in two panels coloured by coverage and by shift. Pairs with a singular $\\hat C$ cannot be placed on the log axis and
  are counted in the panel title.

Both figures belong to E2.2, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/main/displacement/`` for ``BayesCAP`` and ``<OUT_DIR>/uncawe/plots/main/displacement/<method>/``
for any other method.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

from uncawe.plots.common import (
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    ordered_classifiers,
    plot_dir,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.displacement import METHOD, compute_pair_table

PLOT_NAME = "displacement"
FIGURES = ("prior", "conditioning")


def method_plot_dir(method: str = METHOD) -> str:
    """Return the output folder of ``method``: the stored E2.2 folder for ``BayesCAP``, a subfolder for the others."""
    return plot_dir(PLOT_NAME if method == METHOD else os.path.join(PLOT_NAME, method))


def plot_prior(table: pd.DataFrame):
    """Cause 1: recall term against its prior part, coloured by $K$; points on $y=x$ have the prior's recall error."""
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    k = table["n_classes"].astype(float)
    sc = scatter_by_classifier(
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
    style_scatter_axes(ax, "prior part of the recall term $R_0$", "recall term $R$")
    ticks = sorted(table["n_classes"].astype(int).unique())
    cbar = fig.colorbar(sc, ax=ax, ticks=ticks)
    cbar.ax.set_yticklabels([str(t) for t in ticks])
    cbar.set_label("$K$", fontsize=font_size(12))
    cbar.ax.minorticks_off()
    n_classifiers = len(ordered_classifiers(table))
    classifier_legend(fig, table, loc="lower center", ncol=n_classifiers, bbox_to_anchor=(0.5, 0))
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
        sc = scatter_by_classifier(ax, finite, "inv_sigma_min", "val_accuracy", col, cmap, norm)
        ax.set_xscale("log")
        style_scatter_axes(ax, "$1/\\sigma_{\\min}(Q^\\top\\hat CQ)$", "accuracy of $h$ on $V$")
        title = f"coloured by {label}"
        if n_singular:
            title += f" ({n_singular} singular {'pair' if n_singular == 1 else 'pairs'} not shown)"
        ax.set_title(title, fontsize=font_size(11))
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(label, fontsize=font_size(11))
    axes[1].tick_params(labelleft=True)
    classifier_legend(fig, table, loc="lower center", ncol=len(ordered_classifiers(table)), bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


PLOTTERS = {"prior": plot_prior, "conditioning": plot_conditioning}


def generate_plots(args):
    table = compute_pair_table(args.method)
    out_dir = method_plot_dir(args.method)
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
    parser.add_argument("--method", default=METHOD, help="Method of main to analyse (default: %(default)s)")
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
