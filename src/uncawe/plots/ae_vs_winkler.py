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
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, get_acc_names
from uncawe.results import Results

EXPERIMENT = "main"
PLOT_NAME = "ae_vs_winkler"
DEFAULT_DOMAIN = "classic"
DEFAULT_METHODS = ["O-LEAP", "BayesCAP", "PQ", "RQBS", "DoC"]
METHOD_COLORS = {
    "O-LEAP": "#4C78A8",
    "BayesCAP": "#F58518",
    "PQ": "#54A24B",
    "RQBS": "#B279A2",
    "DoC": "#E45756",
}


def _safe_name(name: str) -> str:
    return str(name).replace(os.sep, "__").replace(" ", "_")


def _plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _plot_stem(domain: str, dataset: str, classifier: str) -> str:
    filename = f"{_safe_name(domain)}_{_safe_name(dataset)}_{_safe_name(classifier)}"
    return os.path.join(_plot_dir(), filename)


def _as_float_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return arr.ravel()


def _interval_from_samples(samples) -> tuple[float, float]:
    arr = _as_float_array(samples)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan, np.nan

    alpha = 1 - CONFIDENCE_LEVEL
    low, high = np.nanpercentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def _winkler_score(true_value: float, low: float, high: float) -> float:
    if not np.isfinite(true_value) or not np.isfinite(low) or not np.isfinite(high):
        return np.nan

    alpha = 1 - CONFIDENCE_LEVEL
    score = high - low
    if true_value < low:
        score += (2 / alpha) * (low - true_value)
    elif true_value > high:
        score += (2 / alpha) * (true_value - high)
    return float(score)


def _ensure_plot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    true_accs = []
    lows = []
    highs = []
    winklers = []
    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(row.acc_name, row.true_cts, type="selection")
        low, high = _interval_from_samples(row.estim_acc_ranges)
        true_accs.append(true_acc)
        lows.append(low)
        highs.append(high)
        winklers.append(_winkler_score(true_acc, low, high))

    df["true_accs"] = true_accs
    df["interval_low"] = lows
    df["interval_high"] = highs
    df["winkler_score"] = winklers
    return df.loc[np.isfinite(df["acc_err"]) & np.isfinite(df["winkler_score"]), :]


def _load_main_results(domain: str, methods: list[str]) -> pd.DataFrame:
    base_dir = os.path.join(env.root_dir, EXPERIMENT)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    frames = []
    for acc_name in get_acc_names():
        try:
            res = Results.load(base_dir=base_dir, acc_name=acc_name, domain=domain, filter_methods=methods)
        except ValueError:
            continue
        if not res.df.empty:
            frames.append(res.df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0, ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=methods, ordered=True)
    df["acc_name"] = pd.Categorical(df["acc_name"], categories=get_acc_names(), ordered=True)
    return df.sort_values(["dataset", "classifier", "acc_name", "method"])


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
        alpha=0.16,
        zorder=2,
    )


def _axis_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    max_x = df["acc_err"].max(skipna=True)
    max_y = df["winkler_score"].max(skipna=True)
    xlim = (0.0, float(max_x * 1.08 if max_x > 0 else 1.0))
    ylim = (0.0, float(max_y * 1.08 if max_y > 0 else 1.0))
    return xlim, ylim


def _draw_group(
    ax,
    group: pd.DataFrame,
    method: str,
    color: str,
    sample_alpha: float,
    ellipse_n_std: float,
):
    xs = group["acc_err"].to_numpy(dtype=float)
    ys = group["winkler_score"].to_numpy(dtype=float)
    ax.scatter(xs, ys, s=18, alpha=sample_alpha, color=color, edgecolors="none", label=method, zorder=1)

    points = np.column_stack([xs, ys])
    ellipse = _ellipse_patch(points, color, ellipse_n_std)
    if ellipse is not None:
        ax.add_patch(ellipse)

    mean_x = float(np.mean(xs))
    mean_y = float(np.mean(ys))
    ax.scatter(
        [mean_x],
        [mean_y],
        s=92,
        color=color,
        edgecolors="#1f1f1f",
        linewidths=0.8,
        zorder=4,
    )


def _plot_classifier(
    df: pd.DataFrame,
    domain: str,
    dataset: str,
    classifier: str,
    methods: list[str],
    sample_alpha: float,
    ellipse_n_std: float,
    formats: Iterable[str],
    dpi: int,
):
    acc_names = [acc for acc in get_acc_names() if acc in set(df["acc_name"].astype(str))]
    if not acc_names:
        return

    fig, axes = plt.subplots(1, len(acc_names), figsize=(5.0 * len(acc_names), 4.6), squeeze=False)
    axes = axes[0]
    xlim, ylim = _axis_limits(df)
    visible_methods = set()

    for ax, acc_name in zip(axes, acc_names):
        acc_df = df.loc[df["acc_name"].astype(str) == acc_name, :]
        for method in methods:
            method_df = acc_df.loc[acc_df["method"].astype(str) == method, :]
            if method_df.empty:
                continue
            color = METHOD_COLORS.get(method, None)
            _draw_group(ax, method_df, method, color, sample_alpha, ellipse_n_std)
            visible_methods.add(method)

        ax.set_title(str(acc_name), fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.7)
        ax.set_xlabel("Absolute error")

    axes[0].set_ylabel("Winkler score")
    fig.suptitle(f"{dataset} | {classifier}", fontsize=13)
    if visible_methods:
        labels = [method for method in methods if method in visible_methods]
        handles = [
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
            for method in labels
        ]
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17, top=0.86)

    stem = _plot_stem(domain, dataset, classifier)
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plots(args):
    df = _load_main_results(args.domain, args.methods)
    if df.empty:
        raise RuntimeError(f"No results found for domain={args.domain!r} and methods={args.methods}")

    if args.datasets:
        df = df.loc[df["dataset"].isin(args.datasets), :]
    if args.classifiers:
        df = df.loc[df["classifier"].isin(args.classifiers), :]
    if df.empty:
        raise RuntimeError("No results left after applying dataset/classifier filters.")

    df = _ensure_plot_metrics(df)
    if df.empty:
        raise RuntimeError("No finite acc_err/Winkler rows available for plotting.")

    for (dataset, classifier), group in df.groupby(["dataset", "classifier"], sort=True, observed=True):
        _plot_classifier(
            group,
            args.domain,
            str(dataset),
            str(classifier),
            args.methods,
            args.alpha,
            args.ellipse_n_std,
            args.formats,
            args.dpi,
        )
        print(f"plot generated: {args.domain}/{dataset}/{classifier}")


def parse_args():
    parser = ArgumentParser(description="Generate acc_err vs Winkler scatter plots from main experiment results.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--classifiers", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--alpha", type=float, default=0.18)
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
