import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from copy import copy
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from uncawe.plots.common import (
    ACC_NAME,
    DEFAULT_DOMAIN,
    DEFAULT_METHODS,
    FACET_COL_WRAP,
    FACET_LEGEND_ANCHOR,
    FONT_SCALE,
    METHOD_COLORS,
    axis_limits,
    classifier_legend_handles,
    classifier_markers,
    fmt_tick,
    font_size,
    load_tradeoff_source,
    mean_sample_ranks,
    method_plot_label,
    ordered_classifiers,
    plot_dir,
    rank_test_samples,
    safe_name,
)

_load_rank_source = load_tradeoff_source
_rank_test_samples = rank_test_samples
_mean_sample_ranks = mean_sample_ranks

PLOT_NAME = "ae_vs_winkler"
SYMLIM = 0.05


def _plot_dir() -> str:
    return plot_dir(PLOT_NAME)


def _plot_stem(domain: str, acc_name: str) -> str:
    filename = f"{safe_name(domain)}_{safe_name(acc_name)}"
    return os.path.join(_plot_dir(), filename)


def _aggregate_values(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "acc_name", "method"], observed=True, sort=False)
        .agg(mean_acc_err=("acc_err", "mean"), mean_winkler=("winkler_score", "mean"))
        .reset_index()
    )


def _scale_ticks(limit: float) -> list[float]:
    ticks = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100]
    return [tick for tick in ticks if tick <= limit]


def _apply_axis_scale(ax, axis: str, scale: str, limit: float):
    if scale == "symlog":
        ticks = _scale_ticks(limit)
        if axis == "x":
            ax.set_xscale("symlog", linthresh=SYMLIM, linscale=1.0)
            ax.set_xticks(ticks)
            ax.set_xticklabels([fmt_tick(tick) for tick in ticks])
        else:
            ax.set_yscale("symlog", linthresh=SYMLIM, linscale=1.0)
            ax.set_yticks(ticks)
            ax.set_yticklabels([fmt_tick(tick) for tick in ticks])


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
        markers=classifier_markers(classifiers),
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
    classifiers = ordered_classifiers(df)
    xlim, ylim = axis_limits(df, "mean_acc_err", "mean_winkler")

    grid = sns.FacetGrid(
        df,
        col="method",
        col_order=present_methods,
        col_wrap=FACET_COL_WRAP,
        sharex=True,
        sharey=True,
        height=4.4,
        # aspect=0.74,
        aspect=1,
        despine=False,
    )
    grid.map_dataframe(_draw_facet, sample_alpha=sample_alpha, classifiers=classifiers)
    grid.set_axis_labels("Absolute error", "Winkler score")

    for ax, method in zip(grid.axes.flat, present_methods):
        ax.set_title(method_plot_label(method), fontsize=font_size(14))
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        _apply_axis_scale(ax, "x", x_scale, xlim[1])
        _apply_axis_scale(ax, "y", y_scale, ylim[1])
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="both", labelsize=font_size(12))
        ax.xaxis.label.set_size(font_size(13))
        ax.yaxis.label.set_size(font_size(13))

    fig = grid.figure
    handles = classifier_legend_handles(classifiers)
    fig.legend(
        handles,
        classifiers,
        loc="upper left",
        bbox_to_anchor=FACET_LEGEND_ANCHOR,
        ncol=1,
        frameon=True,
        fontsize=font_size(12),
    )

    fig.tight_layout()
    fig.subplots_adjust(top=0.92, right=0.88, hspace=0.15)

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
