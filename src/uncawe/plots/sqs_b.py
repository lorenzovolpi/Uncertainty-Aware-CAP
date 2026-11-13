"""Scatter plots of SQS-B on the pairs of ``main`` (work-plan card WP-18, E3.3 of
``docs/ext3_posterior_propagation.md``).

The figures read the CSV of :mod:`uncawe.tables.sqs_b` (run ``tables/sqs_b.py --tables`` first). Vanilla accuracy,
$|U|=1000$; one point per (dataset, classifier) pair, markers by classifier. ``SQS-B`` and its references target the
realised table of the target bag and are scored against it.

- ``coverage``: coverage of ``SQS-B`` against that of ``BayCAP-KDEy-mlp-real`` (left panel; the same prevalence draws)
  and of ``SQS-Hbs`` (right panel), coloured by $K$, with lines at $0.95$ and $y=x$;
- ``width``: label-free, the observed amplitude ratio of ``SQS-B`` to ``BayCAP-KDEy-mlp-real`` against the predicted
  one, $\\sqrt{(\\sigma_p^2+\\bar\\sigma_f^2)/(\\sigma_p^2+\\bar\\sigma_g^2)}$ (medians over the bags), coloured by
  $S_{\\text{prec}}-S_{\\text{rec}}$ (diverging scale centred at $0$), with the line $y=x$ and lines at $1$.

The figures belong to E3.3, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/sqs_b/``.
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
from uncawe.plots.common import (
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.sqs_b import KDEY_REAL, SQS_B, SQS_HBS, csv_path

PLOT_NAME = "sqs_b"
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
        raise FileNotFoundError(f"{path} does not exist. Run tables/sqs_b.py --tables first.")
    return pd.read_csv(path)


def plot_coverage(table: pd.DataFrame):
    """Coverage of SQS-B against BayCAP-KDEy-mlp-real and against SQS-Hbs."""
    references = (KDEY_REAL, SQS_HBS)
    fig, axes = plt.subplots(1, len(references), figsize=(11.0, 5.0), layout="constrained", sharex=True, sharey=True)
    norm = LogNorm(2, max(float(table["n_classes"].max()), 3.0))
    sc = None
    for ax, reference in zip(axes, references):
        sc = scatter_by_classifier(
            ax, table, f"{reference}_coverage", f"{SQS_B}_coverage", "n_classes", SEQUENTIAL_CMAP, norm
        )
        ax.plot((0, 1), (0, 1), **IDENTITY_LINE)
        ax.axhline(NOMINAL, **REFERENCE_LINE)
        ax.axvline(NOMINAL, **REFERENCE_LINE)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style_scatter_axes(ax, f"coverage, {method_plot_label(reference)}", f"coverage, {method_plot_label(SQS_B)}")
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("coverage of SQS-B against its references", fontsize=font_size(12))
    return fig


def plot_width(table: pd.DataFrame):
    """Observed against predicted amplitude ratio of SQS-B to BayCAP-KDEy-mlp-real, coloured by S_prec - S_rec."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    x, y = f"{SQS_B}_ratio_predicted", f"{SQS_B}_ratio_observed"
    diff = table[f"{SQS_B}_s_diff"].to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(diff))), 1e-12)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-bound, vmax=bound)
    sc = scatter_by_classifier(ax, table, x, y, f"{SQS_B}_s_diff", DIVERGING_CMAP, norm)
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
    fig.colorbar(sc, ax=ax).set_label("$S_{\\mathrm{prec}}-S_{\\mathrm{rec}}$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(f"amplitude of {method_plot_label(SQS_B)} / {method_plot_label(KDEY_REAL)}", fontsize=font_size(12))
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
    parser = ArgumentParser(description="Plots of SQS-B (E3.3).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/sqs_b.py (default: its output)")
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
