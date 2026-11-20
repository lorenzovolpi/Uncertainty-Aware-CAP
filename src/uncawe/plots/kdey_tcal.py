"""Plots of the ablation of E7.2 of ``docs/ext7_kdey_bandwidth.md`` (work-plan card WP-26): the bandwidth and the
temperature of BayesKDEy selected per pair on calibration bags, each step alone and both (E7.2b).

The figures read the pairs CSV of :mod:`uncawe.tables.kdey_tcal` (run ``tables/kdey_tcal.py --tables`` first).

- ``ablation``: per pair, the mean Winkler score on the target bags (against the realised table of each bag) of the
  four cells ($b=0.1$ or $b_{\\text{cal}}$, $\\tau=1$ or $\\tau_{\\text{cal}}$), on a $\\log$ scale, the pairs ordered by
  $K$, dataset and classifier, with a separator between datasets. Vanilla accuracy by default.

Output: ``<OUT_DIR>/uncawe/plots/kdey_tcal/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.common import font_size
from uncawe.tables.kdey_tcal import CELLS, csv_path

PLOT_NAME = "kdey_tcal"
FIGURES = ("ablation",)
CELL_STYLES = {
    "stored": {"marker": "o", "color": "#8c8c8c", "label": "$b=0.1$, $\\tau=1$"},
    "e53": {"marker": "v", "color": "#E08214", "label": "$b=0.1$, $\\tau_{\\mathrm{cal}}$ (E5.3)"},
    "cal": {"marker": "s", "color": "#2166AC", "label": "$b_{\\mathrm{cal}}$, $\\tau=1$ (E7.2a)"},
    "tcal": {"marker": "D", "color": "#762A83", "label": "$b_{\\mathrm{cal}}$, $\\tau_{\\mathrm{cal}}$ (E7.2b)"},
}
SEPARATOR = {"color": "#d9d9d9", "linewidth": 0.8, "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table() -> pd.DataFrame:
    path = csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/kdey_tcal.py --tables first.")
    return pd.read_csv(path)


def plot_ablation(table: pd.DataFrame, acc_name: str):
    """Per pair, the mean Winkler score of the four cells ($\\log$ scale)."""
    part = table.loc[table["acc_name"] == acc_name].reset_index(drop=True)
    x = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(max(7.0, 0.16 * len(part) + 2.0), 4.2), layout="constrained")
    for cell in CELLS:
        style = CELL_STYLES[cell]
        ax.scatter(
            x,
            part[f"{cell}_winkler"],
            marker=style["marker"],
            s=18,
            color=style["color"],
            label=style["label"],
            alpha=0.85,
            zorder=3,
        )
    datasets = part["dataset"].to_numpy()
    starts = [0, *[i for i in range(1, len(part)) if datasets[i] != datasets[i - 1]]]
    for start in starts[1:]:
        ax.axvline(start - 0.5, **SEPARATOR)
    ends = [*starts[1:], len(part)]
    centres = [(start + end - 1) / 2 for start, end in zip(starts, ends, strict=True)]
    labels = [f"{datasets[start]} ({int(part['n_classes'].iloc[start])})" for start in starts]
    ax.set_xticks(centres, labels, rotation=60, ha="right", fontsize=font_size(8))
    ax.set_xlim(-0.8, len(part) - 0.2)
    ax.set_yscale("log")
    ax.set_ylabel("mean Winkler score (target bags)", fontsize=font_size(11))
    ax.set_xlabel("dataset ($K$); one point per classifier", fontsize=font_size(11))
    ax.tick_params(axis="y", labelsize=font_size(8))
    fig.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=font_size(9), frameon=False)
    fig.suptitle(f"bandwidth and temperature selected on calibration bags ({acc_name})", fontsize=font_size(12))
    return fig


def generate_plots(args):
    out_dir = plot_dir()
    table = load_table()
    figures = []
    if "ablation" in args.figures:
        for acc_name in args.metrics:
            figures.append((f"ablation_{acc_name}", plot_ablation(table, acc_name)))
    for name, fig in figures:
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of the ablation of the bandwidth and temperature selected (E7.2).")
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
