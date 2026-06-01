import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.config import get_acc_names
from uncawe.plots.common import (
    DEFAULT_DOMAIN,
    DEFAULT_METHODS,
    CLASSIFIER_MARKERS,
    font_size,
    load_tradeoff_source,
    mean_sample_ranks,
    method_plot_label,
    plot_dir,
    rank_test_samples,
    safe_name,
)

PLOT_NAME = "rank_heatmap"
CLASSIFIER_ORDER = list(CLASSIFIER_MARKERS)
RANK_METRICS = {
    "ae": ("rank_acc_err", "AE mean sample-rank"),
    "winkler": ("rank_winkler", "Winkler mean sample-rank"),
}


def _plot_dir() -> str:
    return plot_dir(PLOT_NAME)


def _plot_stem(domain: str, acc_name: str, metric: str) -> str:
    filename = f"{safe_name(domain)}_{safe_name(acc_name)}_{metric}"
    return os.path.join(_plot_dir(), filename)


def _classifier_sort_key(classifier: str) -> tuple[int, str]:
    if classifier in CLASSIFIER_ORDER:
        return CLASSIFIER_ORDER.index(classifier), classifier
    return len(CLASSIFIER_ORDER), classifier


def _ordered_classifiers(df: pd.DataFrame) -> list[str]:
    classifiers = [str(classifier) for classifier in df["classifier"].dropna().unique()]
    return sorted(classifiers, key=_classifier_sort_key)


def _rank_matrix(df: pd.DataFrame, metric_col: str, methods: list[str]) -> pd.DataFrame:
    matrix = df.pivot_table(
        index="dataset",
        columns=["classifier", "method"],
        values=metric_col,
        aggfunc="mean",
        observed=True,
    )
    dataset_index = sorted(str(dataset) for dataset in df["dataset"].dropna().unique())
    classifiers = _ordered_classifiers(df)
    columns = [
        (classifier, method)
        for classifier in classifiers
        for method in methods
        if (classifier, method) in matrix.columns
    ]
    return matrix.reindex(index=dataset_index, columns=pd.MultiIndex.from_tuples(columns))


def _figure_size(n_rows: int, n_cols: int) -> tuple[float, float]:
    width = max(7.0, 0.30 * n_cols + 2.4)
    height = min(max(4.8, 0.30 * n_rows + 1.9), 14.0)
    return width, height


def _classifier_spans(columns: pd.MultiIndex) -> list[tuple[str, int, int]]:
    spans = []
    if len(columns) == 0:
        return spans

    start = 0
    current = columns[0][0]
    for idx, (classifier, _) in enumerate(columns[1:], start=1):
        if classifier != current:
            spans.append((current, start, idx - 1))
            current = classifier
            start = idx
    spans.append((current, start, len(columns) - 1))
    return spans


def _method_labels(columns: pd.MultiIndex) -> list[str]:
    return [method_plot_label(method) for _, method in columns]


def _plot_heatmap(
    matrix: pd.DataFrame,
    domain: str,
    acc_name: str,
    metric: str,
    title_suffix: str,
    n_methods: int,
    formats: Iterable[str],
    dpi: int,
):
    if matrix.empty or len(matrix.columns) == 0:
        return

    values = matrix.to_numpy(dtype=float)
    masked_values = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#f2f2f2")

    fig, ax = plt.subplots(figsize=_figure_size(len(matrix.index), len(matrix.columns)))
    im = ax.imshow(masked_values, cmap=cmap, vmin=1, vmax=n_methods, aspect="auto")

    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(_method_labels(matrix.columns), rotation=90, ha="center", fontsize=font_size(7))
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=font_size(8))
    ax.set_title(f"{acc_name} | {title_suffix}", fontsize=font_size(12))
    ax.tick_params(axis="both", length=0, grid_alpha=0)
    ax.tick_params(axis="both", which="minor", length=0, grid_alpha=0)

    for classifier, start, end in _classifier_spans(matrix.columns):
        center = (start + end) / 2
        ax.text(
            center,
            1.035,
            classifier,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=font_size(9),
            fontweight="bold",
            clip_on=False,
        )
        if start > 0:
            ax.axvline(start - 0.5, color="#333333", linewidth=1.1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean rank (lower is better)")

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    stem = _plot_stem(domain, acc_name, metric)
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plots(args):
    df = load_tradeoff_source(args)
    ranked = mean_sample_ranks(rank_test_samples(df))
    if ranked.empty:
        raise RuntimeError("No rank rows available for plotting.")

    acc_names = [acc for acc in get_acc_names() if acc in set(ranked["acc_name"].astype(str))]
    if args.acc_names:
        acc_names = [acc for acc in args.acc_names if acc in set(ranked["acc_name"].astype(str))]
    if not acc_names:
        raise RuntimeError("No requested accuracy measure is available after ranking.")

    n_methods = len([method for method in args.methods if method in set(ranked["method"].astype(str))])
    if n_methods == 0:
        raise RuntimeError("No requested method is available after ranking.")

    for acc_name in acc_names:
        acc_df = ranked.loc[ranked["acc_name"].astype(str) == acc_name, :]
        for metric, (metric_col, title_suffix) in RANK_METRICS.items():
            matrix = _rank_matrix(acc_df, metric_col, args.methods)
            _plot_heatmap(matrix, args.domain, acc_name, metric, title_suffix, n_methods, args.formats, args.dpi)
            print(f"plot generated: {args.domain}/{acc_name}/{metric}")


def parse_args():
    parser = ArgumentParser(description="Generate rank heatmaps from main experiment results.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--acc-names", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
