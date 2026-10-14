import os

import numpy as np
import pandas as pd
from cap.models.confidence import ConfidenceInterval
from matplotlib.lines import Line2D

import uncawe.env as env
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, get_acc_names
from uncawe.results import Results

EXPERIMENT = "main"
DEFAULT_DOMAIN = "classic"
# DEFAULT_METHODS = ["DoC", "QuAcc_NxN", "O-LEAP", "B-CBPE", "BE-CBPE", "L-CBPE", "RQBS", "PQ", "BayesCAP"]
DEFAULT_METHODS = ["DoC", "QuAcc_NxN", "O-LEAP", "RQBS", "PQ", "BayesCAP"]
METHOD_COLORS = {
    "O-LEAP": "#4C78A8",
    "BayesCAP": "#F58518",
    "BayesCAP-real": "#9D755D",
    "PQ": "#E45756",
    "RQBS": "#B279A2",
    "DoC": "#54A24B",
    "QuAcc_NxN": "#72B7B2",
    "B-CBPE": "#FF0000",
    "BE-CBPE": "#00FF00",
    "L-CBPE": "#0000FF",
}
METHOD_PLOT_LABELS = {
    "QuAcc_NxN": "QuAcc",
    "PQ": "SQS-S",
    "RQBS": "SQS-H",
    "O-LEAP": "LEAP",
    "BayesCAP": "BayCAP",
    "BayesCAP-real": "BayCAP-real",
}

ACC_NAME = "vanilla_accuracy"
CLASSIFIER_MARKERS = {
    "LR": "o",
    "kNN": "s",
    "SVM": "^",
    "MLP": "D",
    "RF": "P",
}
DEFAULT_MARKER = "X"
FONT_SCALE = 1.25
FACET_COL_WRAP = 3
FACET_LEGEND_ANCHOR = (0.89, 0.65)


def safe_name(name: str) -> str:
    return str(name).replace(os.sep, "__").replace(" ", "_")


def plot_dir(plot_name: str) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, plot_name)
    os.makedirs(path, exist_ok=True)
    return path


def font_size(size: float) -> float:
    return size * FONT_SCALE


def marker_size(size: float) -> float:
    return size * FONT_SCALE


def method_plot_label(method: str) -> str:
    return METHOD_PLOT_LABELS.get(str(method), str(method))


def as_float_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    return arr.ravel()


def confidence_interval_from_samples(samples) -> ConfidenceInterval | None:
    arr = as_float_array(samples)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None

    return ConfidenceInterval(arr, confidence_level=CONFIDENCE_LEVEL)


def interval_from_samples(samples) -> tuple[float, float]:
    ci = confidence_interval_from_samples(samples)
    if ci is None:
        return np.nan, np.nan

    return ci.interval()


def winkler_score(true_value: float, low: float, high: float) -> float:
    if not np.isfinite(true_value) or not np.isfinite(low) or not np.isfinite(high):
        return np.nan

    alpha = 1 - CONFIDENCE_LEVEL
    score = high - low
    if true_value < low:
        score += (2 / alpha) * (low - true_value)
    elif true_value > high:
        score += (2 / alpha) * (true_value - high)
    return float(score)


def interval_metrics(true_value: float, samples) -> tuple[float, float, float, float, float]:
    if not np.isfinite(true_value):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    ci = confidence_interval_from_samples(samples)
    if ci is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    low, high = ci.interval()
    return (
        ci.coverage(true_value),
        ci.amplitude(),
        ci.winkler(true_value),
        low,
        high,
    )


def ensure_tradeoff_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    true_accs = []
    lows = []
    highs = []
    coverages = []
    amplitudes = []
    winklers = []
    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(row.acc_name, row.true_cts, type="selection")
        coverage, amplitude, winkler, low, high = interval_metrics(true_acc, row.estim_acc_ranges)
        true_accs.append(true_acc)
        lows.append(low)
        highs.append(high)
        coverages.append(coverage)
        amplitudes.append(amplitude)
        winklers.append(winkler)

    df["true_accs"] = true_accs
    df["interval_low"] = lows
    df["interval_high"] = highs
    df["coverage"] = coverages
    df["amplitude"] = amplitudes
    df["winkler_score"] = winklers
    return df.loc[np.isfinite(df["acc_err"]) & np.isfinite(df["winkler_score"]), :]


def load_main_results(domain: str, methods: list[str]) -> pd.DataFrame:
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


def load_tradeoff_source(args) -> pd.DataFrame:
    df = load_main_results(args.domain, args.methods)
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

    df = ensure_tradeoff_metrics(df)
    if df.empty:
        raise RuntimeError("No finite acc_err/Winkler rows available for plotting.")
    return df


def rank_test_samples(df: pd.DataFrame) -> pd.DataFrame:
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


def mean_sample_ranks(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "classifier", "acc_name", "method"], observed=True, sort=False)
        .agg(rank_acc_err=("sample_rank_acc_err", "mean"), rank_winkler=("sample_rank_winkler", "mean"))
        .reset_index()
    )


def classifier_legend_handles(classifiers: list[str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER),
            linestyle="",
            markersize=marker_size(6),
            markerfacecolor="#595959",
            markeredgecolor="#1f1f1f",
            markeredgewidth=0.6,
            alpha=1.0,
        )
        for classifier in classifiers
    ]


def ordered_classifiers(df: pd.DataFrame) -> list[str]:
    preferred = list(CLASSIFIER_MARKERS)
    present = [str(classifier) for classifier in df["classifier"].dropna().unique()]
    ordered = [classifier for classifier in preferred if classifier in present]
    ordered.extend(sorted(classifier for classifier in present if classifier not in preferred))
    return ordered


def classifier_markers(classifiers: list[str]) -> dict[str, str]:
    return {classifier: CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER) for classifier in classifiers}


def fmt_tick(x):
    if float(x).is_integer():
        return str(int(x))

    text = f"{x:.2f}"
    # if text.startswith("0."):
    #     return text[1:]
    # if text.startswith("-0."):
    #     return "-" + text[2:]
    return text


def axis_limits(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[tuple[float, float], tuple[float, float]]:
    max_x = df[x_col].max(skipna=True)
    max_y = df[y_col].max(skipna=True)
    xlim = (0.0, float(max_x * 1.08 if max_x > 0 else 1.0))
    ylim = (0.0, float(max_y * 1.08 if max_y > 0 else 1.0))
    return xlim, ylim
