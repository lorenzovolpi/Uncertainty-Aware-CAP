"""Scatter plots of BayCAP-Hfn, BayCAP-Hf with non-centred shares of the mistakes, on the pairs of ``main``
(work-plan card WP-12, E2.9 of ``docs/ext2_baycap.md``).

The figures read the CSVs of :mod:`uncawe.tables.baycap_hfn` (run ``tables/baycap_hfn.py --tables``, and ``--check``
for the last figure, first): one point per (dataset, classifier) pair, vanilla accuracy, $|U|=1000$.
``BayesCAP-Hfn-real`` and ``BayesCAP-Hf-real`` sample the same model, target the realised table and are scored against
it.

- ``sampler``: the share of divergent draws of BayCAP-Hfn against that of BayCAP-Hf, log scales (zeros drawn at the
  bottom of the axes), markers by the failure class of BayCAP-Hf (E2.8), coloured by $K$, with the line $y=x$ and
  lines at ``DIVERGENT_DRAWS``;
- ``model``: the Winkler score of BayCAP-Hfn against that of BayCAP-Hf (log scales), and the posterior median of
  $\\ln\\kappa$ of the two ($K\\ge3$), two panels, pairs reliable and unreliable under BayCAP-Hf marked apart, with the
  line $y=x$;
- ``check``: the share of divergent draws of BayCAP-Hf (E2.8) against the median observed mistakes per off-diagonal
  cell of $V$, and against the share of empty off-diagonal cells, two panels, pairs with $K\\ge3$, marked by the
  presence of a column with no observed mistake.

The three figures belong to E2.9, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/baycap_hfn/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from uncawe import env
from uncawe.plots.common import SEQUENTIAL_CMAP, font_size, method_plot_label, style_scatter_axes
from uncawe.tables.baycap_hf import DIVERGENT_DRAWS
from uncawe.tables.baycap_hfn import HF_METHOD, METHOD, check_csv_path, csv_path, is_unreliable

PLOT_NAME = "baycap_hfn"
FIGURES = ("sampler", "model", "check")
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "zorder": 1}
CLASS_MARKERS = {"": "o", "boundary": "s", "funnel": "^", "other": "D"}
CLASS_LABELS = {"": "reliable", "boundary": "boundary", "funnel": "funnel", "other": "other"}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str, script: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_hfn.py {script} first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _identity(ax, lo: float, hi: float):
    ax.plot((lo, hi), (lo, hi), color="#595959", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _scatter_by_group(ax, table, x, y, groups, markers, labels, colour=None, norm=None):
    """One scatter per group of ``groups`` (a Series aligned with ``table``), with its marker; returns the last
    mappable and the legend handles."""
    sc, handles = None, []
    for group, marker in markers.items():
        part = table.loc[groups == group]
        if part.empty:
            continue
        kwargs = {"c": part[colour], "cmap": SEQUENTIAL_CMAP, "norm": norm} if colour else {"color": "#4C78A8"}
        sc = ax.scatter(part[x], part[y], marker=marker, s=36, edgecolors="black", linewidths=0.4, zorder=3, **kwargs)
        handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="",
                markerfacecolor="white",
                markeredgecolor="black",
                label=labels[group],
            )
        )
    return sc, handles


def plot_sampler(table: pd.DataFrame):
    """Share of divergent draws of BayCAP-Hfn against BayCAP-Hf."""
    x, y = "hf_share_divergent_draws", "hfn_share_divergent_draws"
    shown = table.loc[table[x].notna() & table[y].notna()].copy()
    positive = pd.concat([shown[x], shown[y]])
    positive = positive[positive > 0]
    floor = float(positive.min()) / 3 if not positive.empty else 1e-4
    hi = max(float(positive.max()) * 1.5 if not positive.empty else 1.0, DIVERGENT_DRAWS * 3)
    for column in (x, y):
        shown[column] = shown[column].clip(lower=floor)
    fig, ax = plt.subplots(figsize=(7.0, 5.5), layout="constrained")
    groups = shown["hf_failure_class"].fillna("")
    sc, handles = _scatter_by_group(ax, shown, x, y, groups, CLASS_MARKERS, CLASS_LABELS, "n_classes", _k_norm(table))
    ax.set_xscale("log")
    ax.set_yscale("log")
    _identity(ax, floor / 1.3, hi)
    ax.axhline(DIVERGENT_DRAWS, **REFERENCE_LINE)
    ax.axvline(DIVERGENT_DRAWS, **REFERENCE_LINE)
    style_scatter_axes(
        ax,
        f"share of divergent draws, {method_plot_label(HF_METHOD)}",
        f"share of divergent draws, {method_plot_label(METHOD)}",
    )
    fig.suptitle("the sampler: centred and non-centred shares (zeros at the bottom)", fontsize=font_size(11))
    if sc is not None:
        fig.colorbar(sc, ax=ax).set_label("$K$", fontsize=font_size(12))
    fig.legend(handles=handles, title="class under Hf", loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def plot_model(table: pd.DataFrame):
    """Winkler score and posterior median of $\\ln\\kappa$ of BayCAP-Hfn against BayCAP-Hf."""
    groups = np.where(is_unreliable(table, "hf"), "unreliable", "reliable")
    markers = {"reliable": "o", "unreliable": "s"}
    labels = {"reliable": "reliable under Hf", "unreliable": "unreliable under Hf"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    sc, handles = None, []
    panels = (
        (f"{HF_METHOD}_winkler", f"{METHOD}_winkler", "Winkler score", True),
        ("hf_ln_kappa_median", "hfn_ln_kappa_median", "posterior median of $\\ln\\kappa$", False),
    )
    for ax, (x, y, label, log) in zip(axes, panels):
        mask = table[x].notna() & table[y].notna()
        if log:
            mask &= (table[x] > 0) & (table[y] > 0)
        shown = table.loc[mask]
        sc, handles = _scatter_by_group(
            ax, shown, x, y, pd.Series(groups, index=table.index)[mask], markers, labels, "n_classes", _k_norm(table)
        )
        values = pd.concat([shown[x], shown[y]])
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            lo, hi = (float(values.min()) / 1.3, float(values.max()) * 1.3) if not values.empty else (0.1, 10.0)
        else:
            lo, hi = (float(values.min()) - 0.3, float(values.max()) + 0.3) if not values.empty else (-1.0, 1.0)
        _identity(ax, lo, hi)
        style_scatter_axes(ax, f"{label}, {method_plot_label(HF_METHOD)}", f"{label}, {method_plot_label(METHOD)}")
    fig.suptitle("the same model: BayCAP-Hfn against BayCAP-Hf", fontsize=font_size(11))
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


def plot_check(check: pd.DataFrame):
    """Share of divergent draws of BayCAP-Hf against the observed mistakes per cell and the share of empty cells."""
    y = "hf_share_divergent_draws"
    shown = check.loc[check[y].notna()].copy()
    positive = shown.loc[shown[y] > 0, y]
    floor = float(positive.min()) / 3 if not positive.empty else 1e-4
    shown[y] = shown[y].clip(lower=floor)
    groups = np.where(shown["n_no_mistake_columns"] > 0, "none", "some")
    markers = {"some": "o", "none": "s"}
    labels = {"some": "every column has a mistake", "none": "a column with no mistake"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.5), layout="constrained")
    sc, handles = None, []
    panels = (
        ("median_mistakes_per_cell", "median observed mistakes per off-diagonal cell of $V$", True),
        ("empty_off_diagonal_share", "share of empty off-diagonal cells of $V$", False),
    )
    for ax, (x, label, log) in zip(axes, panels):
        part = shown.loc[shown[x] > 0] if log else shown
        sc, handles = _scatter_by_group(
            ax,
            part,
            x,
            y,
            pd.Series(groups, index=shown.index)[part.index],
            markers,
            labels,
            "n_classes",
            _k_norm(check),
        )
        ax.set_yscale("log")
        if log:
            ax.set_xscale("log")
        ax.axhline(DIVERGENT_DRAWS, **REFERENCE_LINE)
        style_scatter_axes(ax, label, f"share of divergent draws, {method_plot_label(HF_METHOD)} (E2.8)")
    excluded = len(shown) - int((shown["median_mistakes_per_cell"] > 0).sum())
    suffix = f" ({excluded} pairs with no median mistake not shown on the left)" if excluded else ""
    fig.suptitle(
        "the check: divergences of the centred shares against the mistakes of $V$" + suffix, fontsize=font_size(11)
    )
    if sc is not None:
        fig.colorbar(sc, ax=axes).set_label("$K$", fontsize=font_size(12))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig


PLOTTERS = {"sampler": plot_sampler, "model": plot_model, "check": plot_check}


def generate_plots(args):
    out_dir = plot_dir()
    for name in args.figures:
        if name == "check":
            table = load_table(args.check_csv or check_csv_path(), "--check")
        else:
            table = load_table(args.csv or csv_path(), "--tables")
        fig = PLOTTERS[name](table)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Scatter plots of BayCAP-Hfn, the non-centred shares of BayCAP-Hf (E2.9).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/baycap_hfn.py (default: its output)")
    parser.add_argument(
        "--check-csv", default=None, help="CSV of the check of tables/baycap_hfn.py (default: its output)"
    )
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
