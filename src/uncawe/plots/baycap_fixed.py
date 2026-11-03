"""Scatter and line plots of BayCAP with a fixed rate matrix $\\hat C$ on the pairs of ``main`` (work-plan card WP-8, E3.1
of ``docs/ext3_posterior_propagation.md``).

The figures read the CSVs of :mod:`uncawe.tables.baycap_fixed` (run ``tables/baycap_fixed.py --tables`` first). Each
scatter has one panel per cell of the design (rows: quantifier KDEy, EMQ; columns: input classifier $h$, mlp), one point
per (dataset, classifier) pair, vanilla accuracy, $|U|=1000$, markers by classifier. The methods ``BayCAP-<Q>-<s>``
target the population table, ``BayCAP-<Q>-<s>-real`` the realised table; all are scored against the realised table.

- ``coverage``: coverage of each realised method against that of ``BayesCAP-real``, coloured by $K$, lines at $0.95$;
- ``coverage_population``: the same for the population methods against ``BayesCAP``;
- ``width``: observed against predicted widening factor of the realised interval (§T.8), log scales, line $y=x$,
  coloured by the label-free mismatch $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$;
- ``centre``: movement of the centre (realised minus population point estimate) against the mismatch, coloured by $K$;
- ``calibration``: coverage of the realised method against the ECE of its input classifier on $V$, coloured by $K$;
- ``min_prev``: coverage in bins of the smallest true class prevalence of the bag, $\\min_j\\pi^Q_j$, one line per
  method, population and realised in two panels.

The figures belong to E3.1, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_fixed/``.
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
    METHOD_COLORS,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.baycap_fixed import (
    CELLS,
    PUSHFORWARDS,
    QUANTIFIER_LABELS,
    REFERENCES,
    SOURCES,
    csv_path,
    method_name,
)

PLOT_NAME = "baycap_fixed"
FIGURES = ("coverage", "coverage_population", "width", "centre", "calibration", "min_prev")
NOMINAL = 0.95
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(name: str | None = None, path: str | None = None) -> pd.DataFrame:
    path = path or csv_path(name=name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_fixed.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _cell_axes(width: float = 11.0, height: float = 9.0, **kwargs):
    """A 2x2 grid of panels, rows the quantifiers and columns the input classifiers, with the cell of each panel."""
    fig, axes = plt.subplots(
        len(QUANTIFIER_LABELS), len(SOURCES), figsize=(width, height), layout="constrained", **kwargs
    )
    cells = {
        f"{label}-{source}": axes[i, j]
        for i, label in enumerate(QUANTIFIER_LABELS.values())
        for j, source in enumerate(SOURCES)
    }
    for cell, ax in cells.items():
        label, source = cell.split("-")
        ax.set_title(f"{label}, $s={source}$", fontsize=font_size(11))
    return fig, axes, cells


def _finish(fig, axes, sc, table: pd.DataFrame, label: str, title: str):
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes)
        cbar.set_label(label, fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(title, fontsize=font_size(12))


def _coverage(table: pd.DataFrame, pushforward_name: str):
    reference = REFERENCES[pushforward_name]
    fig, axes, cells = _cell_axes(sharex=True, sharey=True)
    sc = None
    x = f"{reference}_coverage"
    for cell, ax in cells.items():
        y = f"{method_name(cell, pushforward_name)}_coverage"
        sc = scatter_by_classifier(ax, table, x, y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))
        ax.plot((0, 1), (0, 1), **IDENTITY_LINE)
        ax.axhline(NOMINAL, **REFERENCE_LINE)
        ax.axvline(NOMINAL, **REFERENCE_LINE)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        style_scatter_axes(ax, f"coverage, {method_plot_label(reference)}", "coverage, fixed $\\hat C$")
    kind = "realised" if pushforward_name == "realised" else "population"
    _finish(fig, axes, sc, table, "$K$", f"coverage of the {kind} methods with a fixed $\\hat C$")
    return fig


def plot_coverage(table: pd.DataFrame):
    """Coverage of the realised methods against ``BayesCAP-real``."""
    return _coverage(table, "realised")


def plot_coverage_population(table: pd.DataFrame):
    """Coverage of the population methods against ``BayesCAP``."""
    return _coverage(table, "population")


def plot_width(table: pd.DataFrame):
    """Observed against predicted widening factor of the realised interval, coloured by the mismatch."""
    fig, axes, cells = _cell_axes()
    columns = [f"{cell}_mismatch" for cell in CELLS]
    norm = Normalize(0.0, max(float(np.nanmax(table[columns].to_numpy())), 1e-3))
    sc = None
    hidden = 0
    for cell, ax in cells.items():
        x, y = f"{cell}_widening_pred", f"{cell}_widening"
        shown = table.loc[np.isfinite(table[x]) & np.isfinite(table[y]) & (table[x] > 0) & (table[y] > 0)]
        hidden += len(table) - len(shown)
        sc = scatter_by_classifier(ax, shown, x, y, f"{cell}_mismatch", SEQUENTIAL_CMAP, norm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not shown.empty:
            values = pd.concat([shown[x], shown[y]])
            lo, hi = float(values.min()) / 1.3, float(values.max()) * 1.3
            ax.plot((lo, hi), (lo, hi), **IDENTITY_LINE)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        style_scatter_axes(ax, "predicted $\\sqrt{1+(\\bar\\sigma_g/\\sigma_p)^2}$", "observed widening")
    suffix = f" ({hidden} infinite or empty points not shown)" if hidden else ""
    _finish(
        fig,
        axes,
        sc,
        table,
        "mismatch $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$",
        "widening of the realised interval" + suffix,
    )
    return fig


def plot_centre(table: pd.DataFrame):
    """Movement of the centre against the mismatch."""
    fig, axes, cells = _cell_axes(sharex=True, sharey=True)
    sc = None
    for cell, ax in cells.items():
        sc = scatter_by_classifier(
            ax, table, f"{cell}_mismatch", f"{cell}_centre_move", "n_classes", SEQUENTIAL_CMAP, _k_norm(table)
        )
        ax.axhline(0.0, **REFERENCE_LINE)
        style_scatter_axes(
            ax, "mismatch $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$", "realised $-$ population estimate"
        )
    _finish(fig, axes, sc, table, "$K$", "movement of the centre by the realised pushforward")
    return fig


def plot_calibration(table: pd.DataFrame):
    """Coverage of the realised method against the ECE of its input classifier on $V$."""
    fig, axes, cells = _cell_axes(sharey=True)
    sc = None
    for cell, ax in cells.items():
        source = cell.split("-")[1]
        y = f"{method_name(cell, 'realised')}_coverage"
        sc = scatter_by_classifier(ax, table, f"{source}_val_ece", y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))
        ax.axhline(NOMINAL, **REFERENCE_LINE)
        ax.set_ylim(-0.02, 1.02)
        style_scatter_axes(ax, f"top-label ECE of ${source}$ on $V$", "coverage, realised")
    _finish(fig, axes, sc, table, "$K$", "coverage of the realised methods and calibration of the input classifier")
    return fig


def plot_min_prev(min_prev: pd.DataFrame):
    """Coverage in bins of $\\min_j\\pi^Q_j$, one line per method, population and realised in two panels."""
    fig, axes = plt.subplots(1, len(PUSHFORWARDS), figsize=(12.0, 5.0), layout="constrained", sharey=True)
    for ax, p in zip(axes, PUSHFORWARDS):
        for cell in CELLS:
            method = method_name(cell, p)
            part = min_prev.loc[min_prev["method"] == method].sort_values("bin_low")
            labels = [f"[{lo:g}, {hi:g})" for lo, hi in zip(part["bin_low"], part["bin_high"])]
            ax.plot(
                range(len(part)),
                part["coverage"],
                marker="o",
                linewidth=2,
                markersize=8,
                color=METHOD_COLORS.get(method),
                label=method_plot_label(method),
            )
            ax.set_xticks(range(len(part)), labels, rotation=30)
        ax.axhline(NOMINAL, **REFERENCE_LINE)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(p, fontsize=font_size(11))
        style_scatter_axes(ax, "smallest true class prevalence of the bag", "coverage")
        ax.legend(fontsize=font_size(9), frameon=True)
    fig.suptitle(
        "coverage of the methods with a fixed $\\hat C$ by the smallest class prevalence", fontsize=font_size(12)
    )
    return fig


PLOTTERS = {
    "coverage": plot_coverage,
    "coverage_population": plot_coverage_population,
    "width": plot_width,
    "centre": plot_centre,
    "calibration": plot_calibration,
}


def generate_plots(args):
    table = load_table(path=args.csv)
    out_dir = plot_dir()
    for name in args.figures:
        fig = plot_min_prev(load_table("min_prev")) if name == "min_prev" else PLOTTERS[name](table)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of BayCAP with a fixed rate matrix (E3.1).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_fixed.py (default: its output)")
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
