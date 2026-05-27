import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from copy import copy
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import uncawe.env as env
from uncawe.plots.ae_vs_winkler import (
    DEFAULT_DOMAIN,
    DEFAULT_METHODS,
    EXPERIMENT,
    METHOD_COLORS,
    _ensure_plot_metrics,
    _load_main_results,
    _safe_name,
)
from uncawe.plots.amplitude_vs_coverage import (
    ACC_NAME,
    FONT_SCALE,
    _classifier_legend_handles,
    _classifier_markers,
    _fmt_tick,
    _ordered_classifiers,
)

PLOT_NAME = "mean_tradeoff"
SYMLIM = 0.05


def _plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _plot_stem(domain: str, acc_name: str) -> str:
    filename = f"{_safe_name(domain)}_{_safe_name(acc_name)}"
    return os.path.join(_plot_dir(), filename)


def _load_rank_source(args) -> pd.DataFrame:
    df = _load_main_results(args.domain, args.methods)
    if df.empty:
        raise RuntimeError(f"No results found for domain={args.domain!r} and methods={args.methods}")

    acc_names = getattr(args, "acc_names", None)
    if acc_names:
        df = df.loc[df["acc_name"].astype(str).isin(acc_names), :]
    if args.datasets:
        df = df.loc[df["dataset"].isin(args.datasets), :]
    if args.classifiers:
        df = df.loc[df["classifier"].isin(args.classifiers), :]
    if df.empty:
        raise RuntimeError("No results left after applying filters.")

    df = _ensure_plot_metrics(df)
    if df.empty:
        raise RuntimeError("No finite acc_err/Winkler rows available for plotting.")
    return df


def _rank_test_samples(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    method_group_cols = ["dataset", "classifier", "acc_name", "method"]
    sample_group_cols = ["dataset", "classifier", "acc_name", "sample_id"]

    ranked["sample_id"] = ranked.groupby(method_group_cols, observed=True).cumcount()
    ranked["sample_rank_acc_err"] = ranked.groupby(sample_group_cols, observed=True)["acc_err"].rank(
        method="average", ascending=True
    )
    ranked["sample_rank_winkler"] = ranked.groupby(sample_group_cols, observed=True)["winkler_score"].rank(
        method="average", ascending=True
    )
    return ranked.loc[
        np.isfinite(ranked["sample_rank_acc_err"]) & np.isfinite(ranked["sample_rank_winkler"]),
        :,
    ]


def _mean_sample_ranks(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "acc_name", "method"], observed=True, sort=False)
        .agg(rank_acc_err=("sample_rank_acc_err", "mean"), rank_winkler=("sample_rank_winkler", "mean"))
        .reset_index()
    )


def _aggregate_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "acc_name", "method"], observed=True, sort=False)
        .agg(mean_acc_err=("acc_err", "mean"), mean_winkler=("winkler_score", "mean"))
        .reset_index()
    )


def _axis_limits(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    max_x = df[x_col].max(skipna=True)
    max_y = df[y_col].max(skipna=True)
    xlim = (0.0, float(max_x * 1.08 if max_x > 0 else 1.0))
    ylim = (0.0, float(max_y * 1.08 if max_y > 0 else 1.0))
    return xlim, ylim


def _scale_ticks(limit: float) -> list[float]:
    ticks = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100]
    return [tick for tick in ticks if tick <= limit]


def _apply_axis_scale(ax, axis: str, scale: str, limit: float):
    if scale == "symlog":
        ticks = _scale_ticks(limit)
        if axis == "x":
            ax.set_xscale("symlog", linthresh=SYMLIM, linscale=1.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_fmt_tick(tick) for tick in ticks])
        else:
            ax.set_yscale("symlog", linthresh=SYMLIM, linscale=1.0)
            ax.set_yticks(ticks)
            ax.set_yticklabels([_fmt_tick(tick) for tick in ticks])


def _draw_facet(data: pd.DataFrame, sample_alpha: float, classifiers: list[str], **_):
    if data.empty:
        return

    method = str(data["method"].iloc[0])
    sns.scatterplot(
        data=data,
        x="mean_acc_err",
        y="mean_winkler",
        style="classifier",
        style_order=classifiers,
        markers=_classifier_markers(classifiers),
        color=METHOD_COLORS.get(method, "#4C78A8"),
        alpha=sample_alpha,
        s=34,
        edgecolor=None,
        legend=False,
        ax=plt.gca(),
        zorder=2,
    )
    sns.kdeplot(
        data=data,
        x="mean_acc_err",
        y="mean_winkler",
        color=METHOD_COLORS.get(method, "#4C78A8"),
        legend=False,
        ax=plt.gca(),
        zorder=3,
        alpha=sample_alpha - 0.2,
        fill=True,
    )


def _plot(
    df: pd.DataFrame,
    domain: str,
    acc_name: str,
    methods: list[str],
    sample_alpha: float,
    x_scale: str,
    y_scale: str,
    formats: Iterable[str],
    dpi: int,
    add_stem: str | None,
):
    present_methods = [method for method in methods if method in set(df["method"].astype(str))]
    if not present_methods:
        return

    sns.set_theme(style="whitegrid", context="notebook", font_scale=FONT_SCALE)
    classifiers = _ordered_classifiers(df)
    xlim, ylim = _axis_limits(df, "mean_acc_err", "mean_winkler")

    grid = sns.FacetGrid(
        df,
        col="method",
        col_order=present_methods,
        col_wrap=3,
        sharex=True,
        sharey=True,
        height=4.4,
        aspect=0.74,
        despine=False,
    )
    grid.map_dataframe(_draw_facet, sample_alpha=sample_alpha, classifiers=classifiers)
    grid.set_titles(col_template="{col_name}", size=14)
    grid.set_axis_labels("Absolute error", "Winkler score")

    for ax in grid.axes.flat:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _apply_axis_scale(ax, "x", x_scale, xlim[1])
        _apply_axis_scale(ax, "y", y_scale, ylim[1])
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="both", labelsize=12)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)

    fig = grid.figure
    handles = _classifier_legend_handles(classifiers)
    fig.legend(
        handles,
        classifiers,
        loc="center",
        bbox_to_anchor=(0.82, 0.29),
        ncol=1,
        frameon=True,
        fontsize=12,
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.98, hspace=0.35)

    stem = _plot_stem(domain, acc_name)
    if add_stem is not None:
        stem = f"{stem}_{add_stem}"
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plots(args):
    plot_args = copy(args)
    plot_args.acc_names = [ACC_NAME]
    values = _aggregate_values(_load_rank_source(plot_args))
    if values.empty:
        raise RuntimeError("No aggregated acc_err/Winkler rows available for plotting.")

    acc_df = values.loc[values["acc_name"].astype(str) == ACC_NAME, :]
    if acc_df.empty:
        raise RuntimeError(f"No {ACC_NAME!r} rows available after aggregation.")

    _plot(
        acc_df,
        args.domain,
        ACC_NAME,
        args.methods,
        args.alpha,
        args.x_scale,
        args.y_scale,
        args.formats,
        args.dpi,
        args.add_stem,
    )
    print(f"plot generated: {args.domain}/{ACC_NAME}")


def parse_args():
    parser = ArgumentParser(description="Generate absolute-error vs Winkler trade-off plots.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--x-scale", choices=["symlog", "linear"], default="symlog")
    parser.add_argument("--y-scale", choices=["symlog", "linear"], default="symlog")
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
