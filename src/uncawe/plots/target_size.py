r"""Plots of the error split across $|V|$ in the study ``target_size`` (work-plan card WP-21, E4.2b of
``docs/ext4_uncertainty_budget.md``).

Reads the per-cell table of :mod:`uncawe.tables.target_size` (vanilla accuracy, every method scored against the realised
table of each bag; the split is an analysis that uses the true prevalence of the bag and the labels of the pool).

Figures (``--figures``):

- ``v-size``: the RMS of the fixed part of each part ($e_1$, $e_2$, $e_3$, one row each) against $|V|$, log–log, one
  column per method, one colour per dataset, for one classifier (``--classifier``, default LR) and one bag size
  (``--size``, default $1000$). For each $V$ seed a point gives the RMS over the three prevalences of the fixed parts
  of that seed (the full $V$ has one seed), and the line joins, level by level, the means of these points over the
  seeds.

Output: ``<OUT_DIR>/uncawe/plots/target_size/v_size/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from uncawe import env
from uncawe import error_split as es
from uncawe.experiments.target_size import EXPERIMENT, LEVELS, METHODS
from uncawe.plots.common import font_size, method_plot_label
from uncawe.tables import main as main_tables
from uncawe.tables.target_size import PARTS, compute

FIGURES = ("v-size",)
PART_NAMES = {"e1": "$e_1$ prevalence", "e2": "$e_2$ recalls", "e3": "$e_3$ luck of the bag"}
DATASET_CMAP = "tab10"


def seed_points(cells: pd.DataFrame, classifier: str, size: int) -> pd.DataFrame:
    """Per method, dataset, level and $V$ seed: $|V|$ and, per part, the RMS over the prevalences of the fixed parts."""
    cells = cells.loc[(cells["classifier"] == classifier) & (cells["bag_size"] == size)]
    keys = ["method", "dataset", "n_classes", "v_level", "v_seed"]
    rows = []
    for key, group in cells.groupby(keys, sort=False):
        row = dict(zip(keys, key)) | {"v_size": int(group["v_size"].iloc[0])}
        row |= {name: es.rms(group[f"{name}_fixed"]) for name in PARTS}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_v_size(cells: pd.DataFrame, classifier: str = "LR", size: int = 1000):
    points = seed_points(cells, classifier, size)
    if points.empty:
        raise RuntimeError(f"No cell with classifier {classifier} and |U|={size}.")
    datasets = points.drop_duplicates("dataset").sort_values("n_classes", kind="stable")["dataset"].tolist()
    colors = dict(zip(datasets, plt.get_cmap(DATASET_CMAP)(np.arange(len(datasets)) % 10)))
    methods = [m for m in METHODS if m in set(points["method"])]
    fig, axes = plt.subplots(len(PARTS), len(methods), figsize=(4.4 * len(methods), 3.2 * len(PARTS)), squeeze=False)
    for col, method in enumerate(methods):
        for row, name in enumerate(PARTS):
            ax = axes[row, col]
            for dataset in datasets:
                part = points.loc[(points["method"] == method) & (points["dataset"] == dataset)]
                if part.empty:
                    continue
                ax.scatter(part["v_size"], part[name], color=colors[dataset], s=14, alpha=0.6, linewidths=0)
                means = (
                    part.assign(order=part["v_level"].map(list(LEVELS).index))
                    .groupby(["order", "v_size"], as_index=False)[name]
                    .mean()
                    .sort_values("v_size")
                )
                ax.plot(means["v_size"], means[name], color=colors[dataset], linewidth=1.2, marker="o", markersize=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="both", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(7))
            if row == 0:
                ax.set_title(method_plot_label(method).replace("-real", ""), fontsize=font_size(9))
            if col == 0:
                ax.set_ylabel(f"RMS of the fixed part\n{PART_NAMES[name]}", fontsize=font_size(8))
            if row == len(PARTS) - 1:
                ax.set_xlabel("$|V|$", fontsize=font_size(8))
    handles = [
        Line2D([], [], color=colors[d], marker="o", markersize=3, label=main_tables.dataset_map.get(d, d))
        for d in datasets
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(8), frameon=False)
    fig.suptitle(f"{classifier}, $|U|={size}$", fontsize=font_size(9))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


PLOTTERS = {"v-size": plot_v_size}


def out_dir(figure: str) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, figure.replace("-", "_"))
    os.makedirs(path, exist_ok=True)
    return path


def generate_plots(args):
    cells = compute()["cells"]
    for name in args.figures:
        fig = PLOTTERS[name](cells, classifier=args.classifier, size=args.size)
        stem = os.path.join(out_dir(name), f"{EXPERIMENT}_{name}_{args.classifier}_u{args.size}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Plots of the error split across |V| (E4.2b, WP-21).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--classifier", default="LR", help="Classifier of the plotted pairs (default: LR)")
    parser.add_argument("--size", type=int, default=1000, help="Bag size |U| (default: 1000)")
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args(argv)
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
