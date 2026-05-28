import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct
from uncawe.plots.common import (
    ACC_NAME,
    DEFAULT_DOMAIN,
    DEFAULT_METHODS,
    FACET_COL_WRAP,
    FACET_LEGEND_ANCHOR,
    FONT_SCALE,
    METHOD_COLORS,
    classifier_legend_handles,
    classifier_markers,
    fmt_tick,
    font_size,
    interval_from_samples,
    load_main_results,
    ordered_classifiers,
    plot_dir,
    safe_name,
)

PLOT_NAME = "amplitude_vs_coverage"


def _plot_dir() -> str:
    return plot_dir(PLOT_NAME)


def _plot_stem(domain: str, acc_name: str, add_stem: str | None) -> str:
    filename = f"{safe_name(domain)}_{safe_name(acc_name)}"
    if add_stem is not None:
        filename += f"_{add_stem}"
    return os.path.join(_plot_dir(), filename)


def _ensure_interval_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    coverages = []
    amplitudes = []

    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(row.acc_name, row.true_cts, type="selection")
        low, high = interval_from_samples(row.estim_acc_ranges)
        if not np.isfinite(true_acc) or not np.isfinite(low) or not np.isfinite(high):
            coverages.append(np.nan)
            amplitudes.append(np.nan)
            continue

        coverages.append(float(low <= true_acc <= high))
        amplitudes.append(float(high - low))

    df["coverage"] = coverages
    df["amplitude"] = amplitudes
    return df.loc[np.isfinite(df["coverage"]) & np.isfinite(df["amplitude"]) & (df["amplitude"] >= 0), :]


def _load_plot_source(args) -> pd.DataFrame:
    df = load_main_results(args.domain, args.methods)
    if df.empty:
        raise RuntimeError(f"No results found for domain={args.domain!r} and methods={args.methods}")

    df = df.loc[df["acc_name"].astype(str) == ACC_NAME, :]
    if args.datasets:
        df = df.loc[df["dataset"].isin(args.datasets), :]
    if args.classifiers:
        df = df.loc[df["classifier"].isin(args.classifiers), :]
    if df.empty:
        raise RuntimeError("No results left after applying filters.")

    df = _ensure_interval_metrics(df)
    if df.empty:
        raise RuntimeError("No finite coverage/amplitude rows available for plotting.")
    return df


def _aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "method"], observed=True, sort=False)
        .agg(mean_coverage=("coverage", "mean"), mean_amplitude=("amplitude", "mean"))
        .reset_index()
    )


def _draw_facet(data: pd.DataFrame, sample_alpha: float, classifiers: list[str], **_):
    if data.empty:
        return

    method = str(data["method"].iloc[0])
    sns.scatterplot(
        data=data,
        x="mean_amplitude",
        y="mean_coverage",
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
        x="mean_amplitude",
        y="mean_coverage",
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
    methods: list[str],
    sample_alpha: float,
    x_scale: str,
    formats: Iterable[str],
    dpi: int,
    add_stem: str,
):
    present_methods = [method for method in methods if method in set(df["method"].astype(str))]
    if not present_methods:
        return

    sns.set_theme(style="whitegrid", context="notebook", font_scale=FONT_SCALE)
    classifiers = ordered_classifiers(df)
    # max_amplitude = df["mean_amplitude"].max(skipna=True)
    # xlim = (0, float(max_amplitude * 1.08 if max_amplitude > 0 else 1.0))
    xlim = (0, 0.55)

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
    grid.set_titles(col_template="{col_name}", size=font_size(14))
    grid.set_axis_labels("Amplitude", "Coverage")

    for ax in grid.axes.flat:
        ax.set_xlim(*xlim)
        if x_scale == "symlog":
            ax.set_xscale("symlog", linthresh=0.05, linscale=1.0)
            x_ticks = [tick for tick in [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5] if tick <= xlim[1]]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([fmt_tick(tick) for tick in x_ticks])
        elif x_scale == "linear":
            x_ticks = [tick for tick in [0, 0.05, 0.2, 0.4, 0.6] if tick <= xlim[1]]
            ax.set_xticks(x_ticks)
            ax.set_xticklabels([fmt_tick(tick) for tick in x_ticks])

        y_ticks = [0, 0.2, 0.4, 0.6, 0.8, 0.95]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([fmt_tick(tick) for tick in y_ticks])

        ax.set_ylim(0, 1.02)
        ax.axvline(1 - CONFIDENCE_LEVEL, color="#333333", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.axhline(CONFIDENCE_LEVEL, color="#333333", linestyle=":", linewidth=1.0, alpha=0.8)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.tick_params(axis="both", labelsize=font_size(12))
        # if x_scale == "symlog":
        #     ax.tick_params(axis="x", labelrotation=35)
        #     for label in ax.get_xticklabels():
        #         label.set_ha("right")
        ax.xaxis.label.set_size(font_size(13))
        ax.yaxis.label.set_size(font_size(13))

    fig = grid.figure
    # fig.suptitle(f"{ACC_NAME} | amplitude vs coverage", fontsize=13)

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
    stem = _plot_stem(domain, ACC_NAME, add_stem)
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plots(args):
    df = _aggregate_metrics(_load_plot_source(args))
    if df.empty:
        raise RuntimeError("No aggregated coverage/amplitude rows available for plotting.")

    _plot(df, args.domain, args.methods, args.alpha, args.x_scale, args.formats, args.dpi, args.add_stem)
    print(f"plot generated: {args.domain}/{ACC_NAME}")


def parse_args():
    parser = ArgumentParser(description="Generate amplitude vs coverage plots from main experiment results.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--x-scale", choices=["symlog", "linear"], default="linear")
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
