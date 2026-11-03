"""Scatter plots of BayCAP-Hf, the lighter tail on the strength and learned shares of the mistakes, on the pairs of
``main`` (work-plan card WP-11, E2.8 of ``docs/ext2_baycap.md``).

The figures read the per-pair CSV of :mod:`uncawe.tables.baycap_hf` (run ``tables/baycap_hf.py --tables`` first): one
point per (dataset, classifier) pair, vanilla accuracy, $|U|=1000$, markers by classifier. ``BayesCAP-Hf-real``,
``BayesCAP-H-real`` and ``BayesCAP-EB-real`` target the realised table and are scored against it.

- ``tail``: the largest split-$\\hat R$ of BayCAP-Hf against that of BayCAP-H, log scales, coloured by $K$, with the
  line $y=x$ and lines at ``RHAT_HIGH``;
- ``shares``: the posterior median of $\\kappa$ of BayCAP-Hf (log scale) against the share of empty off-diagonal cells
  of $V$, pairs with $K\\ge3$, coloured by the change of AE from BayCAP-H to BayCAP-Hf (diverging, centred at $0$), with
  a line at $\\kappa=1$;
- ``scores``: the Winkler score of BayCAP-Hf against that of BayCAP-H and that of BayCAP-EB-real, two panels, log
  scales, coloured by $K$, with the line $y=x$.

The three figures belong to E2.8, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_hf/``.
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
from uncawe.tables.baycap_hf import EB_METHOD, H_METHOD, METHOD, RHAT_HIGH, csv_path

PLOT_NAME = "baycap_hf"
FIGURES = ("tail", "shares", "scores")
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_hf.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _identity(ax, lo: float, hi: float):
    ax.plot((lo, hi), (lo, hi), color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _log_limits(table: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    values = pd.concat([table[x], table[y]])
    values = values[values > 0]
    if values.empty:
        return 0.1, 10.0
    return float(values.min()) / 1.3, float(values.max()) * 1.3


def _positive(table: pd.DataFrame, *columns: str) -> pd.DataFrame:
    mask = np.ones(len(table), dtype=bool)
    for column in columns:
        mask &= table[column].to_numpy() > 0
    return table.loc[mask]


def _finish(fig, axes, sc, table: pd.DataFrame, label: str = "$K$"):
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes)
        cbar.set_label(label, fontsize=font_size(12))
    classifier_legend(fig, table, loc="center left", bbox_to_anchor=(1.0, 0.5))


def plot_tail(table: pd.DataFrame):
    """Largest split-$\\hat R$ of BayCAP-Hf against BayCAP-H."""
    x, y = "h_max_rhat", "hf_max_rhat"
    shown = _positive(table, x, y)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), layout="constrained")
    sc = scatter_by_classifier(ax, shown, x, y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))
    ax.set_xscale("log")
    ax.set_yscale("log")
    _identity(ax, *_log_limits(shown, x, y))
    ax.axhline(RHAT_HIGH, **REFERENCE_LINE)
    ax.axvline(RHAT_HIGH, **REFERENCE_LINE)
    style_scatter_axes(
        ax,
        f"largest split-$\\hat R$, {method_plot_label(H_METHOD)}",
        f"largest split-$\\hat R$, {method_plot_label(METHOD)}",
    )
    excluded = len(table) - len(shown)
    suffix = f" ({excluded} pairs without diagnostics not shown)" if excluded else ""
    fig.suptitle("the tail: mixing of the lighter prior on $\\nu$" + suffix, fontsize=font_size(11))
    _finish(fig, ax, sc, shown)
    return fig


def plot_shares(table: pd.DataFrame):
    """Posterior median of $\\kappa$ against the share of empty off-diagonal cells of $V$, $K\\ge3$."""
    shown = table.loc[(table["n_classes"] > 2) & table["hf_ln_kappa_median"].notna()].copy()
    shown["kappa_median"] = np.exp(shown["hf_ln_kappa_median"])
    shown["ae_change"] = shown[f"{METHOD}_ae"] - shown[f"{H_METHOD}_ae"]
    lim = float(np.nanmax(np.abs(shown["ae_change"]))) if not shown.empty else 0.0
    lim = lim if lim > 0 else 1e-3
    fig, ax = plt.subplots(figsize=(7.0, 5.5), layout="constrained")
    sc = scatter_by_classifier(
        ax, shown, "empty_off_diagonal_share", "kappa_median", "ae_change", DIVERGING_CMAP, TwoSlopeNorm(0.0, -lim, lim)
    )
    ax.set_yscale("log")
    ax.axhline(1.0, **REFERENCE_LINE)
    ax.set_xlim(-0.03, 1.03)
    style_scatter_axes(
        ax, "share of empty off-diagonal cells of $V$", f"posterior median of $\\kappa$, {method_plot_label(METHOD)}"
    )
    fig.suptitle("the shares: learned concentration of the mistakes ($K\\geq3$)", fontsize=font_size(11))
    _finish(fig, ax, sc, shown, label=f"AE {method_plot_label(METHOD)} $-$ AE {method_plot_label(H_METHOD)}")
    return fig


def plot_scores(table: pd.DataFrame):
    """Winkler score of BayCAP-Hf against BayCAP-H and against BayCAP-EB-real."""
    y = f"{METHOD}_winkler"
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    sc = None
    for ax, reference in zip(axes, (H_METHOD, EB_METHOD)):
        x = f"{reference}_winkler"
        shown = _positive(table, x, y)
        sc = scatter_by_classifier(ax, shown, x, y, "n_classes", SEQUENTIAL_CMAP, _k_norm(table))
        ax.set_xscale("log")
        ax.set_yscale("log")
        _identity(ax, *_log_limits(shown, x, y))
        style_scatter_axes(
            ax, f"Winkler score, {method_plot_label(reference)}", f"Winkler score, {method_plot_label(METHOD)}"
        )
    fig.suptitle(
        "Winkler score of the realised table, BayCAP-Hf against BayCAP-H and the empirical-Bayes prior",
        fontsize=font_size(11),
    )
    _finish(fig, axes, sc, table)
    return fig


PLOTTERS = {"tail": plot_tail, "shares": plot_shares, "scores": plot_scores}


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
    parser = ArgumentParser(description="Scatter plots of BayCAP-Hf, the lighter tail and learned shares (E2.8).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_hf.py (default: its output)")
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
