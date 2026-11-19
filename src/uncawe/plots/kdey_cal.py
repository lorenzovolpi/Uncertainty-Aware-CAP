"""Plots of the bandwidth of BayesKDEy selected per pair on calibration bags (work-plan card WP-26, E7.2 of
``docs/ext7_kdey_bandwidth.md``).

The figures read the pairs CSV of :mod:`uncawe.tables.kdey_cal` (run ``tables/kdey_cal.py --tables`` first).

- ``bandwidths``: label-free, the selected bandwidth $b_{\\text{cal}}$ of each pair ($\\log_{10}$ scale), one column per
  dataset ordered by $K$, markers by classifier, with $b=0.1$ and the edges of the grid marked;
- ``gain``: per pair, the test gain (mean AE of ``BayCAP-KDEy-mlp-real`` minus that of ``BayCAP-KDEy-mlp-real-cal``,
  scored against the realised table of each bag) against the calibration gain (the criterion at $0.1$ minus at
  $b_{\\text{cal}}$, label-free on the target), coloured by $K$; filled markers where the Holm-adjusted bootstrap test of
  the AE difference gives $p<0.05$. Vanilla accuracy by default.

The figures belong to E7.2, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/kdey_cal/``.
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
from uncawe.experiments.metrics import get_acc_names
from uncawe.methods.baycap_kdey import CAL_REFERENCE
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend,
    classifier_markers,
    font_size,
    ordered_classifiers,
    style_scatter_axes,
)
from uncawe.tables.kdey_cal import GRID, TEST_LEVEL, csv_path

PLOT_NAME = "kdey_cal"
FIGURES = ("bandwidths", "gain")
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
DEFAULT_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}
POINT_COLOR = "#2b6c9e"
# horizontal offset of the markers of the classifiers within the column of a dataset
SPREAD = 0.3


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(name: str = "pairs") -> pd.DataFrame:
    path = csv_path(name=name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/kdey_cal.py --tables first.")
    return pd.read_csv(path)


def plot_bandwidths(table: pd.DataFrame):
    """The selected bandwidth of each pair, one column per dataset ordered by $K$."""
    part = table.loc[table["acc_name"] == table["acc_name"].iloc[0]]
    datasets = list(dict.fromkeys(part.sort_values(["n_classes", "dataset"])["dataset"]))
    classifiers = ordered_classifiers(part)
    markers = classifier_markers(classifiers)
    offsets = dict(zip(classifiers, np.linspace(-SPREAD, SPREAD, len(classifiers)) if len(classifiers) > 1 else [0.0]))
    fig, ax = plt.subplots(figsize=(max(5.0, 0.6 * len(datasets) + 2.0), 4.2), layout="constrained")
    positions = {dataset: i for i, dataset in enumerate(datasets)}
    for classifier, sub in part.groupby("classifier", sort=False):
        x = sub["dataset"].map(positions).to_numpy(dtype=float) + offsets[classifier]
        ax.scatter(x, sub["b_cal"], marker=markers[classifier], s=30, color=POINT_COLOR, alpha=0.8, zorder=3)
    for edge in (GRID[0], GRID[-1]):
        ax.axhline(edge, **REFERENCE_LINE)
    ax.axhline(CAL_REFERENCE, **DEFAULT_LINE)
    ax.set_yscale("log")
    ax.set_ylim(GRID[0] / 1.5, GRID[-1] * 1.5)
    labels = [f"{d} ({int(part.loc[part['dataset'] == d, 'n_classes'].iloc[0])})" for d in datasets]
    ax.set_xticks(range(len(datasets)), labels, rotation=60, ha="right", fontsize=font_size(8))
    ax.set_xlim(-0.6, len(datasets) - 0.4)
    ax.set_ylabel("selected bandwidth $b_{\\mathrm{cal}}$", fontsize=font_size(11))
    ax.set_xlabel("dataset ($K$)", fontsize=font_size(11))
    ax.tick_params(axis="y", labelsize=font_size(8))
    classifier_legend(fig, part, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("bandwidth selected on calibration bags (dashed: $b=0.1$; grey: grid edges)", fontsize=font_size(12))
    return fig


def plot_gain(table: pd.DataFrame, acc_name: str):
    """Per pair, the test gain against the calibration gain, coloured by $K$."""
    part = table.loc[table["acc_name"] == acc_name]
    fig, ax = plt.subplots(figsize=(7.0, 5.0), layout="constrained")
    norm = LogNorm(2, max(float(part["n_classes"].max()), 3.0))
    markers = classifier_markers(ordered_classifiers(part))
    significant = part["ae_p_holm"] < TEST_LEVEL
    for classifier, sub in part.groupby("classifier", sort=False):
        for filled in (True, False):
            rows = sub.loc[significant.loc[sub.index] == filled]
            if rows.empty:
                continue
            colors = SEQUENTIAL_CMAP(norm(rows["n_classes"].to_numpy(dtype=float)))
            ax.scatter(
                rows["cal_gain"],
                rows["test_gain"],
                marker=markers[classifier],
                s=40,
                c=colors if filled else "none",
                edgecolors=colors,
                linewidths=1.0,
                zorder=3,
            )
    ax.axhline(0.0, **REFERENCE_LINE)
    ax.axvline(0.0, **REFERENCE_LINE)
    style_scatter_axes(
        ax,
        "calibration gain: criterion at $0.1$ $-$ at $b_{\\mathrm{cal}}$",
        "test gain: AE stored $-$ AE cal",
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP)
    fig.colorbar(mappable, ax=ax).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, part, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"gain of the calibrated bandwidth ({acc_name}; filled: Holm $p<0.05$ on AE)", fontsize=font_size(12))
    return fig


def generate_plots(args):
    out_dir = plot_dir()
    table = load_table()
    figures = []
    if "bandwidths" in args.figures:
        figures.append(("bandwidths", plot_bandwidths(table)))
    if "gain" in args.figures:
        for acc_name in args.metrics:
            figures.append((f"gain_{acc_name}", plot_gain(table, acc_name)))
    for name, fig in figures:
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of the bandwidth of BayesKDEy selected on calibration bags (E7.2).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=["vanilla_accuracy"])
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
