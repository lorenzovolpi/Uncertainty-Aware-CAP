"""Plot of E8.2 of ``docs/ext8_average_confidence.md`` (card WP-30): AE of vanilla accuracy against the quantification
error, per quantifier.

Reads the AE and split tables of :mod:`uncawe.tables.ctt` (results of ``experiments/ctt.py``; every level scored
against the realised table of each bag). One panel per quantifier (``CTT-KDEy``, ``CTT-EMQ``, ``CTT-ACC``, ``CTT-PACC``); one point
per pair at the mean quantification error $\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$ (x) and the mean AE of vanilla accuracy
of the method (y), with circles for the binary and triangles for the multiclass pairs. On the same pair a hollow grey
marker shows the mean AE of the reference ``CTT-oracle`` (true prevalence of the bag), joined to the first by a
vertical segment, so that the length of the segment shows what quantification adds to the error.

Figures (``--figures``): ``ae_vs_qerr``.

Output: ``<OUT_DIR>/uncawe/plots/ctt/ae_vs_qerr/``.
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
from uncawe.experiments.ctt import ACC_SPLIT, EXPERIMENT, METHODS, ORACLE
from uncawe.plots.common import METHOD_COLORS, font_size, method_plot_label
from uncawe.tables.ctt import SUMMARIES, compute

PLOT_NAME = "ae_vs_qerr"
FIGURES = ("ae_vs_qerr",)
MARKERS = {"binary": "o", "multiclass": "^"}
ORACLE_COLOR = METHOD_COLORS[ORACLE]


def plot_dir(plot_name: str = PLOT_NAME) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, plot_name)
    os.makedirs(path, exist_ok=True)
    return path


def points(ae: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """One row per pair and quantifier: the mean quantification error, the mean AE of vanilla accuracy of the method
    and of the oracle, and whether the pair is binary."""
    ae = ae[~ae["dataset"].isin(SUMMARIES)].set_index(["dataset", "classifier"])
    split = split[~split["dataset"].isin(SUMMARIES)].set_index(["dataset", "classifier"])
    rows = []
    for key in ae.index:
        for level in METHODS:
            rows.append(
                {
                    "dataset": key[0],
                    "classifier": key[1],
                    "level": level,
                    "binary": int(ae.loc[key, "n_classes"]) == 2,
                    "q_l1": float(split.loc[key, f"{level}|q_l1"]) if key in split.index else np.nan,
                    "ae": float(ae.loc[key, f"{ACC_SPLIT}|{level}"]),
                    "ae_oracle": float(ae.loc[key, f"{ACC_SPLIT}|{ORACLE}"]),
                }
            )
    return pd.DataFrame(rows)


def plot_ae_vs_qerr(ae: pd.DataFrame, split: pd.DataFrame):
    data = points(ae, split)
    fig, axes = plt.subplots(1, len(METHODS), figsize=(4.2 * len(METHODS), 4.0), sharey=True, squeeze=False)
    for ax, level in zip(axes.flat, METHODS):
        part = data[data["level"] == level]
        color = METHOD_COLORS.get(level, "#4C78A8")
        ax.vlines(part["q_l1"], part["ae_oracle"], part["ae"], color="#BDBDBD", linewidth=0.8, zorder=1)
        for kind, marker in MARKERS.items():
            sub = part[part["binary"] == (kind == "binary")]
            ax.scatter(sub["q_l1"], sub["ae"], marker=marker, color=color, s=24, zorder=3, linewidths=0)
            ax.scatter(
                sub["q_l1"], sub["ae_oracle"], marker=marker, facecolors="none", edgecolors=ORACLE_COLOR, s=24, zorder=2
            )
        ax.set_title(method_plot_label(level), fontsize=font_size(10))
        ax.set_xlabel("mean $\\|\\hat\\pi^Q-\\bar\\pi\\|_1$", fontsize=font_size(9))
        ax.tick_params(labelsize=font_size(8))
        ax.grid(color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].set_ylabel("mean AE, vanilla accuracy", fontsize=font_size(9))
    handles = [
        Line2D([], [], marker=MARKERS["binary"], color="#595959", linestyle="", label="binary pair (method)"),
        Line2D([], [], marker=MARKERS["multiclass"], color="#595959", linestyle="", label="multiclass pair (method)"),
        Line2D(
            [], [], marker="o", markerfacecolor="none", markeredgecolor=ORACLE_COLOR, linestyle="", label=f"{ORACLE}"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(9), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


PLOTTERS = {"ae_vs_qerr": plot_ae_vs_qerr}


def generate_plots(args):
    ae, split, _ = compute()
    out_dir = plot_dir()
    for name in args.figures:
        fig = PLOTTERS[name](ae, split)
        stem = os.path.join(out_dir, name)
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="E8.2: AE of vanilla accuracy against the quantification error.")
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
