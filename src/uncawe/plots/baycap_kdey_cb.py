"""Plots of BayCAP-KDEy-h-inst with a bandwidth selected for calibrated class probabilities on the pairs of ``main``
(work-plan card WP-33, E3.5 of ``docs/ext3_posterior_propagation.md``).

The figures read the two CSVs of :mod:`uncawe.tables.baycap_kdey_cb` (run ``tables/baycap_kdey_cb.py --tables``
first). Vanilla accuracy, $|U|=1000$; one curve or point per (dataset, classifier) pair, markers by classifier. Both
methods target the realised table of the target bag and are scored against it; the selection uses only $V$.

- ``selection``: the curves $\\mathcal L(b)$ of the pairs over the grid (log scale), one panel per group of $K$ ($K=2$,
  $3\\le K\\le7$, $K\\ge8$), each curve divided by its value at $b=0.1$, with a line at $b=0.1$ and a marker at
  $b^\\star$; and a histogram of $\\log_{10}(b^\\star/0.1)$ over the pairs, in steps of the grid;
- ``ece``: the ECE of the leave-one-out confidences on $V$ at $b=0.1$ (horizontal axis) and at $b^\\star$ (vertical
  axis), coloured by $K$, with $y=x$;
- ``coverage``: coverage of ``BayCAP-KDEy-h-inst`` (horizontal axis) and of ``BayCAP-KDEy-h-inst-cb`` (vertical axis),
  coloured by $\\log_{10}(b^\\star/0.1)$ on a diverging scale centred at $0$, with lines at $0.95$ on both axes and
  $y=x$.

The figures belong to E3.5, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_kdey_cb/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm

from uncawe import env
from uncawe.methods.bayes_kdey import LOO_DEFAULT_BANDWIDTH, LOO_GRID
from uncawe.plots.common import (
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables import hierarchical as h
from uncawe.tables.baycap_kdey_cb import CB, INST, csv_path, curves_path

PLOT_NAME = "baycap_kdey_cb"
FIGURES = ("selection", "ece", "coverage")
NOMINAL = 0.95
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}
CURVE_LINE = {"color": "#2171B5", "linewidth": 0.8, "alpha": 0.6}
SELECTED_MARKER = {"color": "#08306B", "marker": "o", "s": 14, "zorder": 3}
# the step of the grid on the log10 scale (LOO_GRID: 21 values from 10^-2 to 1)
GRID_STEP = float(np.diff(np.log10(LOO_GRID)).mean())


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _read(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_kdey_cb.py --tables first.")
    return pd.read_csv(path)


def load_tables(path: str | None = None, curves: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _read(path or csv_path()), _read(curves or curves_path())


def plot_selection(table: pd.DataFrame, curves: pd.DataFrame):
    """The curves $\\mathcal L(b)/\\mathcal L(0.1)$ per group of $K$, and the histogram of $\\log_{10}(b^\\star/0.1)$."""
    groups = list(h.K_GROUPS)
    fig, axes = plt.subplots(1, len(groups) + 1, figsize=(4.0 * (len(groups) + 1), 3.8), layout="constrained")
    for ax, group in zip(axes, groups):
        part = table.loc[table["n_classes"].map(h._k_group) == group]
        for row in part.to_dict("records"):
            curve = curves.loc[(curves["dataset"] == row["dataset"]) & (curves["classifier"] == row["classifier"])]
            bandwidth, values = curve["bandwidth"].to_numpy(dtype=float), curve["logloss"].to_numpy(dtype=float)
            default = values[np.isclose(bandwidth, LOO_DEFAULT_BANDWIDTH)]
            scale = float(default[0]) if len(default) and default[0] > 0 else 1.0
            ax.plot(bandwidth, values / scale, **CURVE_LINE)
            selected = float(row[f"{CB}_bandwidth"])
            at = np.isclose(bandwidth, selected)
            if at.any():
                ax.scatter(bandwidth[at][:1], values[at][:1] / scale, **SELECTED_MARKER)
        ax.axvline(LOO_DEFAULT_BANDWIDTH, **REFERENCE_LINE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{group} ({len(part)} pairs)", fontsize=font_size(11))
        style_scatter_axes(ax, "bandwidth $b$", "$\\mathcal{L}(b)\\,/\\,\\mathcal{L}(0.1)$")
    ax = axes[-1]
    steps = np.round(np.log10(table[f"{CB}_bandwidth"].to_numpy(dtype=float) / LOO_DEFAULT_BANDWIDTH) / GRID_STEP)
    edges = np.arange(np.nanmin(steps, initial=0) - 0.5, np.nanmax(steps, initial=0) + 1.5) * GRID_STEP
    ax.hist(steps * GRID_STEP, bins=edges, color="#4292C6", edgecolor="white")
    ax.axvline(0.0, **REFERENCE_LINE)
    style_scatter_axes(ax, "$\\log_{10}(b^\\star/0.1)$", "pairs")
    ax.set_title(f"selected bandwidth ({len(table)} pairs)", fontsize=font_size(11))
    fig.suptitle("class-balanced leave-one-out log-loss on $V$", fontsize=font_size(12))
    return fig


def plot_ece(table: pd.DataFrame, curves: pd.DataFrame | None = None):
    """The ECE of the leave-one-out confidences on $V$ at $b^\\star$ against that at $0.1$."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    norm = LogNorm(2, max(float(table["n_classes"].max()), 3.0))
    x, y = f"{CB}_ece_default", f"{CB}_ece_selected"
    sc = scatter_by_classifier(ax, table, x, y, "n_classes", SEQUENTIAL_CMAP, norm)
    high = max(float(np.nanmax(table[[x, y]].to_numpy(dtype=float))), 1e-3) * 1.05
    ax.plot((0, high), (0, high), **IDENTITY_LINE)
    ax.set_xlim(0, high)
    ax.set_ylim(0, high)
    style_scatter_axes(ax, "ECE on $V$ at $b=0.1$", "ECE on $V$ at $b^\\star$")
    fig.colorbar(sc, ax=ax).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("calibration of the leave-one-out confidences", fontsize=font_size(12))
    return fig


def plot_coverage(table: pd.DataFrame, curves: pd.DataFrame | None = None):
    """Coverage of BayCAP-KDEy-h-inst-cb against BayCAP-KDEy-h-inst, coloured by $\\log_{10}(b^\\star/0.1)$."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    ratio = table[f"{CB}_log10_ratio"].to_numpy(dtype=float)
    extent = max(float(np.nanmax(np.abs(ratio))) if np.isfinite(ratio).any() else 0.0, GRID_STEP)
    norm = TwoSlopeNorm(0.0, -extent, extent)
    sc = scatter_by_classifier(
        ax, table, f"{INST}_coverage", f"{CB}_coverage", f"{CB}_log10_ratio", DIVERGING_CMAP, norm
    )
    ax.plot((0, 1), (0, 1), **IDENTITY_LINE)
    ax.axhline(NOMINAL, **REFERENCE_LINE)
    ax.axvline(NOMINAL, **REFERENCE_LINE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_scatter_axes(ax, f"coverage, {method_plot_label(INST)}", f"coverage, {method_plot_label(CB)}")
    fig.colorbar(sc, ax=ax).set_label("$\\log_{10}(b^\\star/0.1)$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("coverage: selected bandwidth against $b=0.1$", fontsize=font_size(12))
    return fig


PLOTTERS = {"selection": plot_selection, "ece": plot_ece, "coverage": plot_coverage}


def generate_plots(args):
    table, curves = load_tables(path=args.csv, curves=args.curves)
    out_dir = plot_dir()
    for name in args.figures:
        fig = PLOTTERS[name](table, curves)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of BayCAP-KDEy-h-inst with the bandwidth of E3.5.")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_kdey_cb.py (default: its output)")
    parser.add_argument("--curves", default=None, help="Curve CSV of tables/baycap_kdey_cb.py (default: its output)")
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
