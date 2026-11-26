import os
import re

import numpy as np
import pandas as pd
from cap.models.confidence import ConfidenceInterval
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import uncawe.env as env
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, get_acc_names
from uncawe.results import Results
from uncawe.tables.common import METHOD_LATEX_LABELS, METHOD_LATEX_MACROS

EXPERIMENT = "main"
DEFAULT_DOMAIN = "classic"
# BayCAP-pop with its calibrated temperature (BayesCAP-tcal, WP-34) in place of BayesCAP since 2026-11-26 (user's
# decision), as in tables/main.py:MAIN_METHODS
DEFAULT_METHODS = ["DoC", "QuAcc_NxN", "O-LEAP", "SQS-H", "SQS-Hbs", "PQ", "BayesCAP-tcal"]
METHOD_COLORS = {
    "O-LEAP": "#4C78A8",
    "BayesCAP": "#F58518",
    "BayesCAP-real": "#9D755D",
    "BayesCAP-EB": "#EECA3B",
    "BayesCAP-EB-real": "#FF9DA6",
    "BayesCAP-H-real": "#79706E",
    "BayesCAP-Hf-real": "#BAB0AC",
    "BayesCAP-Hfn-real": "#8C564B",
    # E3.1: one hue per cell, dark for the population and light for the realised pushforward
    "BayCAP-KDEy-h": "#1F77B4",
    "BayCAP-KDEy-h-real": "#AEC7E8",
    # E3.4: the per-instance pushforward with the input classifier h (card WP-31)
    "BayCAP-KDEy-h-inst": "#006D2C",
    # E3.5: the bandwidth selected for calibrated class probabilities (card WP-33)
    "BayCAP-KDEy-h-inst-cb": "#41AB5D",
    "BayCAP-KDEy-mlp": "#2CA02C",
    "BayCAP-KDEy-mlp-real": "#98DF8A",
    "BayCAP-EMQ-h": "#D62728",
    "BayCAP-EMQ-h-real": "#FF9896",
    "BayCAP-EMQ-mlp": "#9467BD",
    "BayCAP-EMQ-mlp-real": "#C5B0D5",
    # E3.2: BayCAP-KDEy with a random C, one hue per input classifier
    "BayCAP-KDEy-Hf-h-real": "#17BECF",
    "BayCAP-KDEy-Hf-mlp-real": "#BCBD22",
    "PQ": "#E45756",
    "SQS-H": "#6B3D9A",
    "SQS-Hbs": "#B279A2",
    "SQS-B": "#FF7F0E",
    # E7.1: the methods with per-class leave-one-out bandwidths
    "BayCAP-KDEy-mlp-real-loo": "#1B7837",
    "SQS-B-loo": "#B35806",
    # E7.2: BayCAP-KDEy with one bandwidth selected on calibration bags (E7.2a), and the temperature after it (E7.2b)
    "BayCAP-KDEy-mlp-real-cal": "#2166AC",
    "BayCAP-KDEy-mlp-real-cal-tcal": "#762A83",
    # the methods of WP-34 (tau selected per metric inside the fit), and the columns of E5.3 assembled from the stored
    # runs (tables/tcal_methods.py, card WP-33; -asm where WP-34 stores the method), in the same colour
    "BayesCAP-tcal": "#B35A00",
    "BayesCAP-real-tcal": "#5C4033",
    "BayesCAP-Hf-real-tcal": "#8C6D31",
    "BayCAP-KDEy-h-inst-cb-tcal-keep": "#A1D99B",
    "BayCAP-KDEy-h-inst-cb-tcal-resel": "#74C476",
    "BayCAP-KDEy-h-inst-cb-keep": "#238B45",
    "BayesCAP-tcal-asm": "#B35A00",
    "BayesCAP-real-tcal-asm": "#5C4033",
    "BayesCAP-Hf-real-tcal-asm": "#8C6D31",
    "BayCAP-KDEy-h-inst-tcal": "#00441B",
    "BayCAP-KDEy-h-inst-cb-tcal-asm": "#74C476",
    "DoC": "#54A24B",
    "QuAcc_NxN": "#72B7B2",
    "B-CBPE": "#FF0000",
    "BE-CBPE": "#00FF00",
    "L-CBPE": "#0000FF",
    # E8.1: Average Confidence with LasCal and the central interval (card WP-29)
    "AC": "#8C564B",
    # E8.3: AC and L-CBPE with label-shift binning (card WP-32)
    "AC-LSB": "#E6550D",
    "L-CBPE-LSB": "#17BECF",
    # E8.2: the rate matrix of V rescaled by a point quantifier, and the oracle reference (card WP-30)
    "CTT-KDEy": "#1F77B4",
    "CTT-EMQ": "#D62728",
    "CTT-ACC": "#E377C2",
    "CTT-PACC": "#6B3D9A",
    "CTT-oracle": "#7F7F7F",
}
METHOD_PLOT_LABELS = {
    "QuAcc_NxN": "QuAcc",
    "PQ": "SQS-S",
    "SQS-H": "SQS-H",
    # mathtext subscripts: the plots do not use LaTeX, so \textsubscript would be printed as is
    "SQS-Hbs": "SQS-H$_{\\mathrm{bs}}$",
    "SQS-B": "SQS-H$_{\\mathrm{bay}}$",
    "SQS-B-loo": "SQS-B-loo",
    "O-LEAP": "LEAP",
    # the default AC and CBPE use label-shift binning (E8.3; user's decision of 2026-11-26): they get the plain names,
    # and the LasCal versions (E8.1) name their calibrator
    "AC-LSB": "AC",
    "L-CBPE-LSB": "CBPE",
    "AC": "AC-LasCal",
    "L-CBPE": "CBPE-LasCal",
    "BayesCAP": "BayCAP-pop",
    "BayesCAP-real": "BayCAP-sample",
    "BayesCAP-EB": "BayCAP-EB",
    "BayesCAP-EB-real": "BayCAP-EB-real",
    "BayesCAP-H-real": "BayCAP-H-real",
    "BayesCAP-Hf-real": "BayCAP-Hf",
    "BayesCAP-Hfn-real": "BayCAP-Hfn-real",
    "BayCAP-KDEy-h": "BayCAP-KDEy-h",
    "BayCAP-KDEy-h-real": "BayCAP-KDEy-h-real",
    # E3.4 (card WP-31): the BayCAP-KDEy of the project since 2026-11-24; before, BayCAP-KDEy was BayCAP-KDEy-mlp-real
    "BayCAP-KDEy-h-inst": "BayCAP-KDEy",
    # E3.5 (card WP-33)
    "BayCAP-KDEy-h-inst-cb": "BayCAP-KDEy-cb",
    "BayCAP-KDEy-mlp": "BayCAP-KDEy-mlp",
    "BayCAP-KDEy-mlp-real": "BayCAP-KDEy-mlp-real",
    "BayCAP-EMQ-h": "BayCAP-EMQ-h",
    "BayCAP-EMQ-h-real": "BayCAP-EMQ-h-real",
    "BayCAP-EMQ-mlp": "BayCAP-EMQ-mlp",
    "BayCAP-EMQ-mlp-real": "BayCAP-EMQ-mlp-real",
    "BayCAP-KDEy-Hf-h-real": "BayCAP-KDEy-Hf-h-real",
    "BayCAP-KDEy-Hf-mlp-real": "BayCAP-KDEy-Hf-mlp-real",
    "BayCAP-KDEy-mlp-real-loo": "BayCAP-KDEy-mlp-real-loo",
    "BayCAP-KDEy-mlp-real-cal": "BayCAP-KDEy-mlp-real-cal",
    "BayCAP-KDEy-mlp-real-cal-tcal": "BayCAP-KDEy-mlp-real-cal-tcal",
    # WP-34, and E5.3 assembled (tables/tcal_methods.py, card WP-33)
    "BayesCAP-tcal": "BayCAP-pop-tcal",
    "BayesCAP-real-tcal": "BayCAP-sample-tcal",
    "BayesCAP-Hf-real-tcal": "BayCAP-Hf-tcal",
    "BayCAP-KDEy-h-inst-cb-tcal-keep": "BayCAP-KDEy-cb-tcal-keep",
    "BayCAP-KDEy-h-inst-cb-tcal-resel": "BayCAP-KDEy-cb-tcal-resel",
    "BayCAP-KDEy-h-inst-cb-keep": "BayCAP-KDEy-cb-keep",
    "BayesCAP-tcal-asm": "BayCAP-pop-tcal-asm",
    "BayesCAP-real-tcal-asm": "BayCAP-sample-tcal-asm",
    "BayesCAP-Hf-real-tcal-asm": "BayCAP-Hf-tcal-asm",
    "BayCAP-KDEy-h-inst-tcal": "BayCAP-KDEy-tcal",
    "BayCAP-KDEy-h-inst-cb-tcal-asm": "BayCAP-KDEy-cb-tcal-asm",
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
# per-pair scatters of the E2 analyses (plots/displacement.py, plots/pushforward.py)
SCATTER_GRID_COLOR = "#d9d9d9"
SCATTER_EDGE_COLOR = "#1f1f1f"
# the lightest quarter of Blues is close to the white background, so it is cut off
SEQUENTIAL_CMAP = ListedColormap(colormaps["Blues"](np.linspace(0.25, 1.0, 256)), name="Blues_trunc")
DIVERGING_CMAP = "RdBu_r"


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


# the bodies of the LaTeX macros of the tables, by macro name (e.g. "sqshbs" -> "SQS-H\\textsubscript{bs}")
_LATEX_MACRO_BODIES = dict(
    re.fullmatch(r"\\newcommand\{\\(\w+)\}\{(.*)\}", macro).groups() for macro in METHOD_LATEX_MACROS
)


def method_table_label(method: str) -> str:
    """The label of ``method`` in the tables (``tables/common.py:METHOD_LATEX_LABELS``) with its LaTeX macros resolved
    for matplotlib: ``\\name`` and ``\\name{}`` become the macro body, ``\\textsubscript{x}`` becomes a mathtext
    subscript (e.g. ``SQS-Hbs`` -> ``SQS-H$_{\\mathrm{bs}}$``, ``BayesCAP-Hf-real`` -> ``BayCAP-Hf``)."""
    label = METHOD_LATEX_LABELS.get(str(method), str(method))
    label = re.sub(r"\\(\w+)(\{\})?", lambda m: _LATEX_MACRO_BODIES.get(m.group(1), m.group(0)), label)
    return re.sub(r"\\textsubscript\{(.*?)\}", r"$_{\\mathrm{\1}}$", label)


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
    """The stored rows of ``methods`` in ``main``, for every metric, plus those that make up the ``-tcal`` methods of
    ``methods`` (``tables/tcal_methods.py``, as in the main tables)."""
    # imported here: tables/tcal_methods.py imports the tables of E5.3, which import this module
    from uncawe.tables.tcal_methods import with_tcal

    base_dir = os.path.join(env.root_dir, EXPERIMENT)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    frames = []
    for acc_name in get_acc_names():
        try:
            res = Results.load(base_dir=base_dir, acc_name=acc_name, domain=domain, filter_methods=methods)
        except ValueError:
            continue
        res = with_tcal(res, base_dir, acc_name, domain, None, methods)
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


def scatter_by_classifier(ax, table: pd.DataFrame, x: str, y: str, color, cmap, norm):
    """Draw one point per row, one marker shape per classifier; colour encodes ``color`` through ``cmap`` and ``norm``."""
    sc = None
    for classifier in ordered_classifiers(table):
        sub = table.loc[table["classifier"].astype(str) == classifier]
        sc = ax.scatter(
            sub[x],
            sub[y],
            c=sub[color],
            cmap=cmap,
            norm=norm,
            marker=CLASSIFIER_MARKERS.get(classifier, DEFAULT_MARKER),
            s=marker_size(45),
            edgecolors=SCATTER_EDGE_COLOR,
            linewidths=0.6,
            zorder=3,
        )
    return sc


def style_scatter_axes(ax, xlabel: str, ylabel: str):
    ax.grid(True, color=SCATTER_GRID_COLOR, linewidth=0.7, alpha=0.7, zorder=0)
    ax.set_xlabel(xlabel, fontsize=font_size(12))
    ax.set_ylabel(ylabel, fontsize=font_size(12))
    ax.tick_params(axis="both", labelsize=font_size(10))


def classifier_legend(fig, table: pd.DataFrame, **kwargs):
    classifiers = ordered_classifiers(table)
    fig.legend(classifier_legend_handles(classifiers), classifiers, frameon=True, fontsize=font_size(10), **kwargs)


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


def dataset_bars(frame: pd.DataFrame, value: str, condition: str, methods: list[str], log: bool = True) -> pd.DataFrame:
    r"""Per method of ``methods`` and value of ``condition``: the mean of ``value`` over the datasets and the ends of
    the bar of $\pm1$ standard error over the datasets after removing each dataset's average level (user's decision of
    2026-11-25, shift profile and E4.4 figures). ``frame`` has one row per method, dataset and condition.

    With ``log`` (for positive scores on a logarithmic axis): $\ell_{d,j}=\log y_{d,j}$ for dataset $d$ and condition
    $j$ (a method and a value of ``condition``), the point $\exp(\bar\ell_j)$ (geometric mean), $w_{d,j}=\ell_{d,j}-
    \bar\ell_{d\cdot}$ with $\bar\ell_{d\cdot}$ the mean of dataset $d$ over all conditions,
    $\mathrm{SE}_j=\operatorname{sd}_d(w_{d,j})/\sqrt n$, and the bar from $\exp(\bar\ell_j-\mathrm{SE}_j)$ to
    $\exp(\bar\ell_j+\mathrm{SE}_j)$. Without ``log`` (e.g. the coverage, which can be $0$): the same on the values,
    with the arithmetic mean. Every dataset must have every condition (and positive values with ``log``); with one
    dataset the bars have zero width. Columns: ``method``, ``condition``, ``value``, ``lower``, ``upper``,
    ``n_datasets``."""
    rows = frame.loc[frame["method"].isin(methods)]
    table = rows.pivot_table(index="dataset", columns=["method", condition], values=value, aggfunc="first")
    if table.isna().any().any():
        raise ValueError(f"Some datasets lack a method or a {condition} of {value}.")
    if log:
        if (table <= 0).any().any():
            raise ValueError(f"Non-positive {value}: the geometric mean is not defined.")
        table = np.log(table)
    n = len(table)
    centred = table.sub(table.mean(axis=1), axis=0)
    mean = table.mean()
    se = centred.std(ddof=1) / np.sqrt(n) if n > 1 else mean * 0.0
    back = np.exp if log else (lambda values: values)
    out = pd.DataFrame({"value": back(mean), "lower": back(mean - se), "upper": back(mean + se)})
    out["n_datasets"] = n
    return out.reset_index()
