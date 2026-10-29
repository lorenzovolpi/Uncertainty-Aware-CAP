"""Scatter plots of the zero-cost screens for a hierarchical prior on the recalls of BayCAP (work-plan card WP-9, E2.6
of ``docs/ext2_baycap.md``).

The figures read the per-pair CSV written by ``tables/hierarchical.py --tables``
(``<OUT_DIR>/uncawe/tables/hierarchical/hierarchical.csv``), whose grid posteriors are too slow to recompute for every
plot: one point per (dataset, classifier) pair of ``main``, vanilla accuracy, $|U|=1000$. Markers by classifier, colour
by $K$. Predicted draws keep the estimand of the method they start from
(``BayesCAP-real``: realised table) and are scored against the realised table; the hierarchical prior shown is the
log-uniform one (``MAIN_PRIOR``).

- ``validation``: predicted against stored realised coverage of the empirical-Bayes prior (D4), with the line $y=x$;
- ``centre``: the diagnosis of D4, realised pushforward, two panels with the line $y=x$: the stored recall part of the
  change of the centre against the shift of D4, and the absolute coverage error of D4 with the stored prevalence part
  added against that of D4;
- ``prediction``: predicted realised coverage with the hierarchical prior against the stored coverage of
``BayesCAP-real``
  and of ``BayesCAP-EB-real``, two panels, with the line $y=x$;
- ``sensitivity``: the range of the predicted realised coverage over the three priors on $\\nu$ against $K$ (log scale)
  (D1);
- ``exchangeability``: the smallest leave-one-class-out p-value (log scale) against the largest vertex shift $|z_j|$
  (D2), with lines at $p=0.01$ and $|z_j|=2$;
- ``sampler``: $P(\\min(\\nu m,\\nu(1-m))<1\\mid V)$ against the stored share of bags with divergent transitions of
  ``BayesCAP-EB`` (D3).

Output: ``<OUT_DIR>/uncawe/plots/hierarchical/``.
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
from uncawe.plots.common import (
    SEQUENTIAL_CMAP,
    classifier_legend_handles,
    font_size,
    ordered_classifiers,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.hierarchical import MAIN_PRIOR, NU_PRIOR_LABELS, P_SMALL, csv_path

PLOT_NAME = "hierarchical"
FIGURES = ("validation", "centre", "prediction", "sensitivity", "exchangeability", "sampler")
P_FLOOR = 1e-12  # p-values below are drawn at the floor


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/hierarchical.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _scatter(ax, table: pd.DataFrame, x: str, y: str):
    return scatter_by_classifier(ax, table, x, y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))


def _identity(ax, table: pd.DataFrame, x: str, y: str, unit: bool = True):
    lo = float(np.nanmin([table[x].min(), table[y].min()] + ([0.0] if unit else [])))
    hi = float(np.nanmax([table[x].max(), table[y].max()] + ([1.0] if unit else [])))
    pad = 0.03 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)


def _finish(fig, axes, sc, table: pd.DataFrame):
    cbar = fig.colorbar(sc, ax=axes)
    cbar.set_label("$K$", fontsize=font_size(12))
    classifiers = ordered_classifiers(table)
    handles, labels = classifier_legend_handles(classifiers), list(classifiers)
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=True, fontsize=font_size(10))


def plot_validation(table: pd.DataFrame):
    """Predicted against stored realised coverage of the empirical-Bayes prior (D4)."""
    x, y = "realised_eb_pred_coverage", "BayesCAP-EB-real_coverage"
    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    sc = _scatter(ax, table, x, y)
    _identity(ax, table, x, y)
    style_scatter_axes(ax, "predicted coverage (EB, from BayCAP-real)", "stored coverage (BayCAP-EB-real)")
    ax.set_title("D4: prediction of the empirical-Bayes prior", fontsize=font_size(11))
    _finish(fig, ax, sc, table)
    return fig


def plot_centre(table: pd.DataFrame):
    """Diagnosis of D4 (realised pushforward): recall part against the D4 shift, and coverage errors with and without
    the stored prevalence part."""
    data = table.assign(
        err_d4=(table["realised_eb_pred_coverage"] - table["BayesCAP-EB-real_coverage"]).abs(),
        err_prev=(table["realised_ebprev_pred_coverage"] - table["BayesCAP-EB-real_coverage"]).abs(),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    panels = (
        ("realised_d4_shift", "realised_recall_part", "shift of D4", "stored recall part $R$", False),
        ("err_d4", "err_prev", "|coverage error| of D4", "|coverage error| of D4 + stored $P$", False),
    )
    sc = None
    for ax, (x, y, xlabel, ylabel, unit) in zip(axes, panels):
        sc = _scatter(ax, data, x, y)
        _identity(ax, data, x, y, unit=unit)
        style_scatter_axes(ax, xlabel, ylabel)
    fig.suptitle("D4: recall part and prevalence part of the empirical-Bayes change", fontsize=font_size(11))
    _finish(fig, axes, sc, data)
    return fig


def plot_prediction(table: pd.DataFrame):
    """Predicted realised coverage with the hierarchical prior against the stored uniform and EB coverage."""
    y = f"realised_{MAIN_PRIOR}_pred_coverage"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained", sharey=True)
    sc = None
    for ax, (x, label) in zip(
        axes, (("BayesCAP-real_coverage", "BayCAP-real"), ("BayesCAP-EB-real_coverage", "BayCAP-EB-real"))
    ):
        sc = _scatter(ax, table, x, y)
        _identity(ax, table, x, y)
        style_scatter_axes(
            ax, f"stored coverage ({label})", f"predicted coverage, hierarchical ({NU_PRIOR_LABELS[MAIN_PRIOR]})"
        )
    fig.suptitle("D4: predicted coverage of the hierarchical prior", fontsize=font_size(11))
    _finish(fig, axes, sc, table)
    return fig


def plot_sensitivity(table: pd.DataFrame):
    """Range of the predicted realised coverage over the three priors on $\\nu$, against $K$ (D1)."""
    x, y = "n_classes", "realised_pred_coverage_range"
    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    sc = _scatter(ax, table, x, y)
    ax.set_xscale("log")
    style_scatter_axes(ax, "number of classes $K$", "range of predicted coverage over the priors on $\\nu$")
    ax.set_title("D1: sensitivity to the prior on $\\nu$", fontsize=font_size(11))
    _finish(fig, ax, sc, table)
    return fig


def plot_exchangeability(table: pd.DataFrame):
    """Smallest leave-one-class-out p-value against the largest vertex shift (D2)."""
    data = table.assign(p_shown=table[f"{MAIN_PRIOR}_min_pvalue"].clip(lower=P_FLOOR))
    x, y = "p_shown", f"{MAIN_PRIOR}_max_abs_z"
    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    sc = _scatter(ax, data, x, y)
    ax.set_xscale("log")
    ax.axvline(P_SMALL, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.axhline(2, color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    style_scatter_axes(ax, f"smallest leave-one-class-out p-value (floor {P_FLOOR:g})", "largest vertex shift $|z_j|$")
    ax.set_title("D2: exchangeability of the classes", fontsize=font_size(11))
    _finish(fig, ax, sc, data)
    return fig


def plot_sampler(table: pd.DataFrame):
    """Posterior probability of a Beta parameter below 1 against the stored divergences of BayCAP-EB (D3)."""
    x, y = f"{MAIN_PRIOR}_alpha_risk", "BayesCAP-EB_share_divergent"
    shown = table.loc[table[y].notna()]
    fig, ax = plt.subplots(figsize=(7.0, 6.0), layout="constrained")
    sc = _scatter(ax, shown, x, y)
    style_scatter_axes(
        ax, "$P(\\min(\\nu m,\\nu(1-m))<1\\mid V)$, hierarchical", "share of divergent bags, BayCAP-EB (stored)"
    )
    excluded = len(table) - len(shown)
    suffix = f" ({excluded} pairs without diagnostics not shown)" if excluded else ""
    ax.set_title("D3: sampler risk" + suffix, fontsize=font_size(11))
    _finish(fig, ax, sc, shown)
    return fig


PLOTTERS = {
    "validation": plot_validation,
    "centre": plot_centre,
    "prediction": plot_prediction,
    "sensitivity": plot_sensitivity,
    "exchangeability": plot_exchangeability,
    "sampler": plot_sampler,
}


def generate_plots(args):
    table = load_table(args.csv)
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
    parser = ArgumentParser(description="Scatter plots of the screens for a hierarchical prior on the recalls (E2.6).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/hierarchical.py (default: its output)")
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
