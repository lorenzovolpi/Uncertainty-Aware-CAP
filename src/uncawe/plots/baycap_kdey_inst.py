"""Scatter plots of BayCAP-KDEy with the per-instance pushforward on the pairs of ``main`` (work-plan card WP-31, E3.4 of
``docs/ext3_posterior_propagation.md``).

The figures read the CSV of :mod:`uncawe.tables.baycap_kdey_inst` (run ``tables/baycap_kdey_inst.py --tables`` first).
Vanilla accuracy, $|U|=1000$; one point per (dataset, classifier) pair, markers by classifier. Both methods target the
realised table of the target bag and are scored against it.

- ``coverage``: coverage of ``BayCAP-KDEy-h-real`` (horizontal axis) and of ``BayCAP-KDEy-h-inst`` (vertical axis),
  coloured by $K$, with lines at $0.95$ on both axes and $y=x$;
- ``width``: label-free, the observed amplitude ratio of ``BayCAP-KDEy-h-inst`` to ``BayCAP-KDEy-h-real`` against the
  predicted one, $\\sqrt{(\\operatorname{sd}(m_w)^2+\\bar\\sigma_w^2)/(\\operatorname{sd}(m)^2+\\bar\\sigma_g^2)}$ (medians
  over the bags), coloured by the share of $\\bar{\\tilde\\sigma}^2_g$ removed by the within-class spread of the
  confidences, with the line $y=x$ and lines at $1$.

The figures belong to E3.4, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_kdey_inst/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, Normalize

from uncawe import env
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.baycap_kdey_inst import H_REAL, INST, csv_path

PLOT_NAME = "baycap_kdey_inst"
FIGURES = ("coverage", "width")
NOMINAL = 0.95
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_kdey_inst.py --tables first.")
    return pd.read_csv(path)


def plot_coverage(table: pd.DataFrame):
    """Coverage of BayCAP-KDEy-h-inst against BayCAP-KDEy-h-real."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    norm = LogNorm(2, max(float(table["n_classes"].max()), 3.0))
    sc = scatter_by_classifier(ax, table, f"{H_REAL}_coverage", f"{INST}_coverage", "n_classes", SEQUENTIAL_CMAP, norm)
    ax.plot((0, 1), (0, 1), **IDENTITY_LINE)
    ax.axhline(NOMINAL, **REFERENCE_LINE)
    ax.axvline(NOMINAL, **REFERENCE_LINE)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    style_scatter_axes(ax, f"coverage, {method_plot_label(H_REAL)}", f"coverage, {method_plot_label(INST)}")
    fig.colorbar(sc, ax=ax).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("coverage: per-instance against cell pushforward", fontsize=font_size(12))
    return fig


def plot_width(table: pd.DataFrame):
    """Observed against predicted amplitude ratio of BayCAP-KDEy-h-inst to BayCAP-KDEy-h-real, coloured by the share of
    the cell term removed by the within-class spread."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    x, y = f"{INST}_ratio_predicted", f"{INST}_ratio_observed"
    share = table[f"{INST}_within_share"].to_numpy(dtype=float)
    norm = Normalize(0.0, max(float(np.nanmax(share)) if np.isfinite(share).any() else 0.0, 1e-12))
    sc = scatter_by_classifier(ax, table, x, y, f"{INST}_within_share", SEQUENTIAL_CMAP, norm)
    values = table[[x, y]].to_numpy(dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    pad = 0.05 * max(high - low, 0.1)
    low, high = min(low, 1.0) - pad, max(high, 1.0) + pad
    ax.plot((low, high), (low, high), **IDENTITY_LINE)
    ax.axhline(1.0, **REFERENCE_LINE)
    ax.axvline(1.0, **REFERENCE_LINE)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    style_scatter_axes(ax, "predicted amplitude ratio", "observed amplitude ratio")
    fig.colorbar(sc, ax=ax).set_label("share of $\\bar{\\tilde\\sigma}^2_g$ within classes", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"amplitude of {method_plot_label(INST)} / {method_plot_label(H_REAL)}", fontsize=font_size(12))
    return fig


PLOTTERS = {"coverage": plot_coverage, "width": plot_width}


def generate_plots(args):
    table = load_table(path=args.csv)
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
    parser = ArgumentParser(description="Plots of BayCAP-KDEy with the per-instance pushforward (E3.4).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_kdey_inst.py (default: its output)")
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
