"""Bar plot of the error split of the main-table methods (work-plan card WP-14, E4.1 of
``docs/ext4_uncertainty_budget.md``).

Reads the per-pair table of :mod:`uncawe.tables.error_split` (vanilla accuracy, $|U|=1000$, every method scored against
the realised table of each bag). Each bar stacks the mean square over the bags of $e_1$ (prevalence), $e_2$ (recalls
against the pool) and $e_3$ (luck of the bag, on the bags the method kept), averaged over the classifiers of the
dataset; a black dot marks the mean square of the total error $\\hat A-A$. The stacked height differs from the dot by
twice the mean cross products of the parts, which can be negative. The scale is linear, so that the stacked segments
are comparable.

Figures (``--figures``):

- ``bars``: one panel per dataset, one bar per method, every method of the split except ``QuAcc_NxN``, whose errors are
  one to two orders of magnitude larger and would flatten the other bars (user's decision of 2026-11-10; ``O-LEAP``
  stays as the baseline);
- ``bars-quacc``: ``QuAcc_NxN`` alone, one panel, one bar per dataset.

Output: ``<OUT_DIR>/uncawe/plots/main/error_split/``.
"""

import math
import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from uncawe.plots.common import font_size, method_plot_label, plot_dir
from uncawe.tables import main as main_tables
from uncawe.tables.error_split import METHODS, compute

PLOT_NAME = "error_split"
FIGURES = ("bars", "bars-quacc")
PART_COLORS = {"e1": "#4C78A8", "e2": "#F58518", "e3": "#BAB0AC"}
PART_NAMES = {"e1": "$e_1$ prevalence", "e2": "$e_2$ recalls", "e3": "$e_3$ luck of the bag"}
COL_WRAP = 5
# the method plotted on its own (figure ``bars-quacc``) and left out of the figure ``bars``
SEPARATE = "QuAcc_NxN"
MAIN_METHODS = tuple(m for m in METHODS if m != SEPARATE)


def mean_squares(table: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    """Per dataset and method, the mean square of each part and of the total, averaged over the classifiers.

    The mean square of a pair is the square of its RMS over the bags; $e_3$ is taken on the bags the method kept.
    """
    columns = {"e1": "e1_rms", "e2": "e2_rms", "e3": "e3_rms_kept", "total": "total_rms"}
    rows = []
    for dataset, part in table.groupby("dataset", sort=False, observed=True):
        for m in methods:
            record = {"dataset": str(dataset), "n_classes": int(part["n_classes"].iloc[0]), "method": m}
            for name, column in columns.items():
                record[name] = float((pd.to_numeric(part[f"{m}_{column}"]) ** 2).mean())
            rows.append(record)
    return pd.DataFrame(rows)


def plot_bars(table: pd.DataFrame, methods=MAIN_METHODS):
    data = mean_squares(table, methods)
    datasets = list(dict.fromkeys(data["dataset"]))
    ncols = min(COL_WRAP, len(datasets))
    nrows = math.ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    x = np.arange(len(methods))
    for ax, dataset in zip(axes.flat, datasets):
        part = data.loc[data["dataset"] == dataset].set_index("method").loc[list(methods)]
        bottom = np.zeros(len(methods))
        for name, color in PART_COLORS.items():
            ax.bar(x, part[name].to_numpy(), bottom=bottom, color=color, width=0.75, linewidth=0)
            bottom += part[name].to_numpy()
        ax.scatter(x, part["total"].to_numpy(), color="black", s=14, zorder=3)
        k = int(part["n_classes"].iloc[0])
        ax.set_title(f"{main_tables.dataset_map.get(dataset, dataset)} ($K={k}$)", fontsize=font_size(9))
        ax.set_xticks(x, [method_plot_label(m).replace("-real", "") for m in methods], rotation=60, ha="right")
        ax.tick_params(labelsize=font_size(7))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        ax.yaxis.get_offset_text().set_fontsize(font_size(7))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(datasets) :]:
        ax.set_visible(False)
    _legend(fig)
    fig.supylabel("mean square over the bags", fontsize=font_size(10))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _legend(fig):
    handles = [Patch(color=c, label=PART_NAMES[p]) for p, c in PART_COLORS.items()]
    handles.append(Line2D([], [], marker="o", color="black", linestyle="", label="total $\\hat A-A$"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(9), frameon=False)


def plot_bars_single(table: pd.DataFrame, method=SEPARATE):
    """One panel for ``method``: one stacked bar per dataset, as the bars of :func:`plot_bars`."""
    data = mean_squares(table, (method,))
    datasets = list(dict.fromkeys(data["dataset"]))
    part = data.set_index("dataset").loc[datasets]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(datasets) + 2.0), 4.2))
    x = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    for name, color in PART_COLORS.items():
        ax.bar(x, part[name].to_numpy(), bottom=bottom, color=color, width=0.75, linewidth=0)
        bottom += part[name].to_numpy()
    ax.scatter(x, part["total"].to_numpy(), color="black", s=14, zorder=3)
    labels = [
        f"{main_tables.dataset_map.get(d, d)} ($K={int(k)}$)" for d, k in zip(datasets, part["n_classes"].to_numpy())
    ]
    ax.set_xticks(x, labels, rotation=60, ha="right")
    ax.tick_params(labelsize=font_size(7))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_fontsize(font_size(7))
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(method_plot_label(method).replace("-real", ""), fontsize=font_size(9))
    ax.set_ylabel("mean square over the bags", fontsize=font_size(10))
    _legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


PLOTTERS = {"bars": plot_bars, "bars-quacc": plot_bars_single}


def generate_plots(args):
    _, table = compute()
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
    parser = ArgumentParser(description="Bar plots of the error split of the main-table methods (E4.1).")
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
