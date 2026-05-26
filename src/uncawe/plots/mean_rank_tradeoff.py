import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

import uncawe.env as env
from uncawe.config import get_acc_names
from uncawe.plots.ae_vs_winkler import (
    DEFAULT_DOMAIN,
    DEFAULT_METHODS,
    EXPERIMENT,
    METHOD_COLORS,
    _ensure_plot_metrics,
    _load_main_results,
    _safe_name,
)

PLOT_NAME = "mean_rank_tradeoff"


def _plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _plot_stem(domain: str, acc_name: str) -> str:
    filename = f"{_safe_name(domain)}_{_safe_name(acc_name)}"
    return os.path.join(_plot_dir(), filename)


def _ellipse_patch(points: np.ndarray, color: str, n_std: float) -> Ellipse | None:
    if points.shape[0] < 3:
        return None

    cov = np.cov(points, rowvar=False)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None

    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    if np.isclose(vals.max(), 0):
        return None

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    center = points.mean(axis=0)
    return Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.4,
        alpha=0.14,
        zorder=2,
    )


def _load_rank_source(args) -> pd.DataFrame:
    df = _load_main_results(args.domain, args.methods)
    if df.empty:
        raise RuntimeError(f"No results found for domain={args.domain!r} and methods={args.methods}")

    if args.acc_names:
        df = df.loc[df["acc_name"].astype(str).isin(args.acc_names), :]
    if args.datasets:
        df = df.loc[df["dataset"].isin(args.datasets), :]
    if args.classifiers:
        df = df.loc[df["classifier"].isin(args.classifiers), :]
    if df.empty:
        raise RuntimeError("No results left after applying filters.")

    df = _ensure_plot_metrics(df)
    if df.empty:
        raise RuntimeError("No finite acc_err/Winkler rows available for ranking.")
    return df


def _mean_method_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "acc_name", "method"], observed=True, sort=False)
        .agg(mean_acc_err=("acc_err", "mean"), mean_winkler=("winkler_score", "mean"))
        .reset_index()
    )


def _rank_methods(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    group_cols = ["dataset", "classifier", "acc_name"]
    ranked["rank_acc_err"] = ranked.groupby(group_cols, observed=True)["mean_acc_err"].rank(
        method="average", ascending=True
    )
    ranked["rank_winkler"] = ranked.groupby(group_cols, observed=True)["mean_winkler"].rank(
        method="average", ascending=True
    )
    return ranked.loc[np.isfinite(ranked["rank_acc_err"]) & np.isfinite(ranked["rank_winkler"]), :]


def _axis_limits(n_methods: int) -> tuple[float, float]:
    pad = 0.35
    return 1 - pad, n_methods + pad


def _legend_handles(methods: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=METHOD_COLORS.get(method, "#4C78A8"),
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.6,
            alpha=1.0,
        )
        for method in methods
    ]


def _draw_method(
    ax,
    df: pd.DataFrame,
    method: str,
    sample_alpha: float,
    show_ellipse: bool,
    ellipse_n_std: float,
):
    color = METHOD_COLORS.get(method, "#4C78A8")
    xs = df["rank_acc_err"].to_numpy(dtype=float)
    ys = df["rank_winkler"].to_numpy(dtype=float)
    points = np.column_stack([xs, ys])

    ax.scatter(xs, ys, s=26, alpha=sample_alpha, color=color, edgecolors="none", zorder=1)

    if show_ellipse:
        ellipse = _ellipse_patch(points, color, ellipse_n_std)
        if ellipse is not None:
            ax.add_patch(ellipse)

    ax.scatter(
        [float(np.mean(xs))],
        [float(np.mean(ys))],
        s=130,
        color=color,
        edgecolors="#1f1f1f",
        linewidths=0.9,
        zorder=4,
    )


def _plot_acc_name(
    df: pd.DataFrame,
    domain: str,
    acc_name: str,
    methods: list[str],
    sample_alpha: float,
    show_ellipse: bool,
    ellipse_n_std: float,
    formats: Iterable[str],
    dpi: int,
):
    present_methods = [method for method in methods if method in set(df["method"].astype(str))]
    if not present_methods:
        return

    n_methods = len(present_methods)
    mid_rank = (n_methods + 1) / 2
    fig, ax = plt.subplots(figsize=(6.4, 5.6))

    for method in present_methods:
        method_df = df.loc[df["method"].astype(str) == method, :]
        if method_df.empty:
            continue
        _draw_method(ax, method_df, method, sample_alpha, show_ellipse, ellipse_n_std)

    low, high = _axis_limits(n_methods)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.axvline(mid_rank, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(mid_rank, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
    ax.set_xlabel("Mean rank on AE")
    ax.set_ylabel("Mean rank on Winkler")
    ax.set_title(f"{acc_name} | lower rank is better", fontsize=12)
    ax.set_aspect("equal", adjustable="box")

    handles = _legend_handles(present_methods)
    ax.legend(
        handles,
        present_methods,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=min(n_methods, 5),
        frameon=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    stem = _plot_stem(domain, acc_name)
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plots(args):
    df = _load_rank_source(args)
    ranked = _rank_methods(_mean_method_metrics(df))
    if ranked.empty:
        raise RuntimeError("No rank rows available for plotting.")

    acc_names = [acc for acc in get_acc_names() if acc in set(ranked["acc_name"].astype(str))]
    if args.acc_names:
        acc_names = [acc for acc in args.acc_names if acc in set(ranked["acc_name"].astype(str))]
    if not acc_names:
        raise RuntimeError("No requested accuracy measure is available after ranking.")

    for acc_name in acc_names:
        acc_df = ranked.loc[ranked["acc_name"].astype(str) == acc_name, :]
        _plot_acc_name(
            acc_df,
            args.domain,
            acc_name,
            args.methods,
            args.alpha,
            args.ellipse,
            args.ellipse_n_std,
            args.formats,
            args.dpi,
        )
        print(f"plot generated: {args.domain}/{acc_name}")


def parse_args():
    parser = ArgumentParser(description="Generate mean-rank AE vs Winkler trade-off plots.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--acc-names", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--alpha", type=float, default=0.22)
    parser.add_argument("--ellipse", action=BooleanOptionalAction, default=True)
    parser.add_argument("--ellipse-n-std", type=float, default=1.0)
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
