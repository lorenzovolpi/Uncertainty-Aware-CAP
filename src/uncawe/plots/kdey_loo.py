"""Plots of the per-class leave-one-out bandwidths of BayesKDEy (work-plan card WP-25, E7.1 of
``docs/ext7_kdey_bandwidth.md``).

The figures read the CSVs of :mod:`uncawe.tables.kdey_loo` (run ``tables/kdey_loo.py --tables`` first).

- ``bandwidths``: label-free, one panel per dataset, the selected bandwidth $\\hat b_j$ of each class against its
  number of validation outputs $n_j$ (log scales; one point per class and classifier, markers by classifier), with the
  edges of the grid and $b=0.1$ marked;
- ``effect``: one panel per LOO method, the difference of the mean Winkler score of the pair (LOO minus stored, scored
  against the realised table of each bag) against the median of $\\log_{10}(\\hat b_j/0.1)$ over the classes,
  coloured by $K$; filled markers where the Holm-adjusted bootstrap test gives $p<0.05$. Vanilla accuracy by default.

The figures belong to E7.1, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/kdey_loo/``.
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
from uncawe.methods.bayes_kdey import LOO_DEFAULT_BANDWIDTH, LOO_GRID
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend,
    classifier_markers,
    font_size,
    method_plot_label,
    ordered_classifiers,
    style_scatter_axes,
)
from uncawe.tables.kdey_loo import COMPARISONS, TEST_LEVEL, csv_path

PLOT_NAME = "kdey_loo"
FIGURES = ("bandwidths", "effect")
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
DEFAULT_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}
POINT_COLOR = "#2b6c9e"
COLUMNS = 4


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(name: str) -> pd.DataFrame:
    path = csv_path(name=name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/kdey_loo.py --tables first.")
    return pd.read_csv(path)


def plot_bandwidths(classes: pd.DataFrame):
    """Per dataset, the selected bandwidth of each class against its number of validation outputs."""
    datasets = list(dict.fromkeys(classes.sort_values(["n_classes", "dataset"])["dataset"]))
    rows = int(np.ceil(len(datasets) / COLUMNS))
    fig, axes = plt.subplots(
        rows, COLUMNS, figsize=(3.4 * COLUMNS, 3.0 * rows), layout="constrained", sharey=True, squeeze=False
    )
    markers = classifier_markers(ordered_classifiers(classes))
    for ax, dataset in zip(axes.flat, datasets):
        sub = classes.loc[classes["dataset"] == dataset]
        for classifier, part in sub.groupby("classifier", sort=False):
            ax.scatter(
                part["n_j"], part["bandwidth"], marker=markers[classifier], s=22, color=POINT_COLOR, alpha=0.7, zorder=3
            )
        for edge in (LOO_GRID[0], LOO_GRID[-1]):
            ax.axhline(edge, **REFERENCE_LINE)
        ax.axhline(LOO_DEFAULT_BANDWIDTH, **DEFAULT_LINE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(LOO_GRID[0] / 1.5, LOO_GRID[-1] * 1.5)
        ax.set_title(f"{dataset} ($K={int(sub['n_classes'].iloc[0])}$)", fontsize=font_size(10))
        ax.tick_params(labelsize=font_size(8))
    for ax in axes.flat[len(datasets) :]:
        ax.set_visible(False)
    fig.supxlabel("$n_j$, validation outputs of the class", fontsize=font_size(11))
    fig.supylabel("selected bandwidth $\\hat b_j$", fontsize=font_size(11))
    classifier_legend(fig, classes, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("per-class leave-one-out bandwidths (dashed: $b=0.1$; grey: grid edges)", fontsize=font_size(12))
    return fig


def plot_effect(table: pd.DataFrame, acc_name: str):
    """Per LOO method, the Winkler difference against the median log-ratio of the selected bandwidths to 0.1."""
    part = table.loc[table["acc_name"] == acc_name]
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(11.0, 5.0), layout="constrained", sharey=True)
    norm = LogNorm(2, max(float(part["n_classes"].max()), 3.0))
    markers = classifier_markers(ordered_classifiers(part))
    sc = None
    for ax, (loo, stored) in zip(np.atleast_1d(axes), COMPARISONS.items()):
        significant = part[f"{loo}_p_holm"] < TEST_LEVEL
        for classifier, sub in part.groupby("classifier", sort=False):
            for filled in (True, False):
                rows = sub.loc[significant.loc[sub.index] == filled]
                if rows.empty:
                    continue
                colors = SEQUENTIAL_CMAP(norm(rows["n_classes"].to_numpy(dtype=float)))
                sc = ax.scatter(
                    rows["log_b_ratio_median"],
                    rows[f"{loo}_diff_mean"],
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
            "median $\\log_{10}(\\hat b_j/0.1)$",
            f"$\\mathcal{{W}}$ {method_plot_label(loo)} $-$ {method_plot_label(stored)}",
        )
    if sc is not None:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP)
        fig.colorbar(mappable, ax=axes).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, part, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"effect of the leave-one-out bandwidths ({acc_name}; filled: Holm $p<0.05$)", fontsize=font_size(12))
    return fig


def generate_plots(args):
    out_dir = plot_dir()
    figures = []
    if "bandwidths" in args.figures:
        figures.append(("bandwidths", plot_bandwidths(load_table("classes"))))
    if "effect" in args.figures:
        table = load_table("pairs")
        for acc_name in args.metrics:
            figures.append((f"effect_{acc_name}", plot_effect(table, acc_name)))
    for name, fig in figures:
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of the per-class leave-one-out bandwidths of BayesKDEy (E7.1).")
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
