r"""Scatter plots of the prior on $\pi^Q$ of BayCAP-Hf and BayCAP-KDEy (work-plan card WP-19, E5.1 of
``docs/ext5_baycap_priors_tempering.md``).

The figures read the per-pair CSV of :mod:`uncawe.tables.binform_prior` (run ``tables/binform_prior.py --tables``
first). One figure per method (``BayesCAP-Hf-real``, ``BayCAP-KDEy-mlp-real``) and metric (vanilla accuracy by
default): the mean Winkler score with the uniform prior (horizontal) against that with the well-specified prior (top row)
and with the misspecified prior (bottom row), one point per (dataset, classifier) pair, markers by classifier, coloured
by $K$, one column per bag size, with the diagonal $y=x$ (points below it: the informative prior has a lower Winkler
score). Both methods target the realised table of each bag and are scored against it.

The figures belong to E5.1, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/binform_prior/``.
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
from uncawe.plots.common import (
    ACC_NAME,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.binform_prior import ACC_LABELS, E51_BASE_METHODS, OTHER_PRIORS, PRIOR_LABELS, csv_path

PLOT_NAME = "binform_prior"
FIGURES = ("winkler",)
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/binform_prior.py --tables first.")
    return pd.read_csv(path)


def plot_winkler(
    table: pd.DataFrame,
    method: str,
    acc_name: str = ACC_NAME,
    priors: tuple[str, ...] = OTHER_PRIORS,
    labels: dict[str, str] = PRIOR_LABELS,
    title: str | None = None,
):
    """The Winkler score of the uniform prior against that of each informative prior in ``priors`` (columns
    ``<prior>_winkler`` of ``table``, labelled by ``labels``), one column per bag size. The defaults are those of E5.1;
    E5.4 (``plots/binform_centre_edge.py``) passes its roles and title."""
    rows = table.loc[(table["method"] == method) & (table["acc_name"] == acc_name)]
    if rows.empty:
        return None
    sizes = sorted(rows["size"].unique())
    fig, axes = plt.subplots(
        len(priors),
        len(sizes),
        figsize=(5.0 * len(sizes) + 1.5, 4.6 * len(priors)),
        layout="constrained",
        squeeze=False,
    )
    norm = LogNorm(2, max(float(table["n_classes"].max()), 3.0))
    sc = None
    for col, size in enumerate(sizes):
        part = rows.loc[rows["size"] == size]
        for row, prior in enumerate(priors):
            ax = axes[row, col]
            x, y = "uniform_winkler", f"{prior}_winkler"
            sc = scatter_by_classifier(ax, part, x, y, "n_classes", SEQUENTIAL_CMAP, norm)
            values = part[[x, y]].to_numpy(dtype=float)
            low, high = np.nanmin(values), np.nanmax(values)
            pad = 0.05 * max(high - low, 1e-3)
            low, high = max(low - pad, 0.0), high + pad
            ax.plot((low, high), (low, high), **IDENTITY_LINE)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            style_scatter_axes(ax, "Winkler score, uniform prior", f"Winkler score, {labels[prior]} prior")
            ax.set_title(f"$|U|={size}$, {labels[prior]}", fontsize=font_size(11))
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, rows, loc="center left", bbox_to_anchor=(1.0, 0.5))
    if title is None:
        title = f"E5.1, {method_plot_label(method)}: prior on $\\pi^Q$ ({ACC_LABELS[acc_name]})"
    fig.suptitle(title, fontsize=font_size(12))
    return fig


def generate_plots(args):
    table = load_table(path=args.csv)
    out_dir = plot_dir()
    if "winkler" not in args.figures:
        return
    for acc_name in args.metrics:
        for method in args.methods:
            fig = plot_winkler(table, method, acc_name)
            if fig is None:
                print(f"no rows for {method} ({acc_name}), skipped")
                continue
            stem = os.path.join(out_dir, f"{PLOT_NAME}_{method}_{acc_name}")
            if args.add_stem is not None:
                stem = f"{stem}_{args.add_stem}"
            for fmt in args.formats:
                fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
            plt.close(fig)
            print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of the prior on pi^Q of BayCAP-Hf and BayCAP-KDEy (E5.1).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--methods", nargs="+", choices=E51_BASE_METHODS, default=list(E51_BASE_METHODS))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/binform_prior.py (default: its output)")
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
