import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from cap.models.confidence import ConfidenceInterval
from matplotlib.colors import Normalize, rgb2hex
from scipy.stats import wilcoxon

import uncawe.env as env
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, get_acc_names, get_method_names
from uncawe.main import EXPERIMENT
from uncawe.pretrain.dataset import sort_datasets_by_size
from uncawe.results import Results
from uncawe.tables.common import METHOD_LATEX_LABELS
from uncawe.util import decorate_dataset

# METRICS = ("acc_err", "coverage", "amplitude", "winkler")
METRICS = ("acc_err", "winkler")
LOWER_IS_BETTER = {"acc_err": True, "coverage": False, "amplitude": True, "winkler": True}
VALUE_FORMAT = "{:.3f}"
COLORMAP_SATURATION = 0.65
COLORMAP_LIGHTNESS = 0.35
CLASSIFIER_FILTER = ("LR", "kNN", "SVM", "MLP", "RF")
WORST_RGB = (233, 177, 174)
BEST_RGB = (172, 207, 173)
# WORST_RGB = (251, 160, 159) # old worst
# BEST_RGB = (154, 254, 154) # old best
# BEST_RGB = (138, 154, 91)
# BEST_RGB = (235, 215, 52)
WILCOXON_ALPHA = 0.01
DAGGER_MARK = " $^{\\dag}$"
EMPTY_CELL = "\\multicolumn{1}{c}{-}"
COLOR_MINMAX = "minmax"
COLOR_SIGMA_ZSCORE = "sigma_zscore"
COLOR_BEST_TO_SIGMA = "best_to_sigma"
COLOR_THRESHOLD = "threshold"
GRADIENT_TO_WHITE = "white"
GRADIENT_TO_WORST = "worst"
GRADIENT_END = GRADIENT_TO_WORST
BEST_TO_SIGMA_FACTOR = 1.0
TABLE_COLUMN_SEP = "6pt"

METRIC_COLOR_STRATEGY = {
    # "acc_err": COLOR_MINMAX,
    "acc_err": COLOR_BEST_TO_SIGMA,
    "coverage": COLOR_THRESHOLD,
    "amplitude": COLOR_THRESHOLD,
    # "winkler": COLOR_SIGMA_ZSCORE,
    "winkler": COLOR_BEST_TO_SIGMA,
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
    "stanfordnlp__imdb": "imdb",
    "fancyzhx__yelp_polarity": "yelp-polarity",
    "stanfordnlp__sst2": "sst2",
    "fancyzhx__ag_news": "ag-news",
    "fancyzhx__dbpedia_14": "dbpedia-14",
    "community-datasets__yahoo_answers_topics": "yahoo-answers-topics",
    "yelp_reviews": "yelp-reviews",
    "ag_news-lt": "ag-news-lt",
    "dbpedia_14-lt": "dbpedia-14-lt",
    "ethz__food101": "food101",
}

metrics_map = {
    "acc_err": "$\\mathrm{AE}$",
    "coverage": "$\\mathcal{C}$",
    "amplitude": "$\\mathcal{A}$",
    "winkler": "$\\mathcal{W}$",
}


# methods stored in main but left out of its tables and ranks: the BayCAP variants are used only by the E2 analyses;
# BayesCAP-EB-real is in the tables and ranks (user's request)
EXCLUDED_METHODS = {"BayesCAP-real", "BayesCAP-EB"}


def table_method_names() -> list[str]:
    return [method for method in get_method_names() if method not in EXCLUDED_METHODS]


def get_acc_coverage(method: str, true_acc: float, accs_range: np.ndarray) -> float:
    return _confidence_interval(accs_range).coverage(true_acc)


def get_acc_amplitude(method: str, accs_range: np.ndarray) -> float:
    return _confidence_interval(accs_range).amplitude()


def get_acc_winkler(method: str, true_acc: float, accs_range: np.ndarray) -> float:
    arr = np.asarray(accs_range, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(true_acc):
        return np.nan

    return _confidence_interval(arr).winkler(true_acc)


def _confidence_interval(accs_range: np.ndarray) -> ConfidenceInterval:
    return ConfidenceInterval(accs_range, confidence_level=CONFIDENCE_LEVEL)


def _ordered_columns(methods: list[str], metrics: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([methods, metrics], names=["method", "metric"])


def _latex_column_format(methods: list[str], metrics: tuple[str, ...], index_columns: int = 1) -> str:
    numeric_column = "S[table-format=1.3,table-align-text-post=false]"
    metric_formats = []
    for _ in metrics:
        metric_formats.append(numeric_column * len(methods))
    return "l" * index_columns + "|".join(metric_formats)


def _display_columns(methods: list[str], metrics: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([metrics, methods], names=["metric", "method"])


def _to_display_column_order(df: pd.DataFrame, methods: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    return df.swaplevel("method", "metric", axis=1).reindex(columns=_display_columns(methods, metrics))


def _format_mean_std(mean: pd.DataFrame, std: pd.DataFrame, show_std: bool = True) -> pd.DataFrame:
    formatted = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for col in mean.columns:
        if show_std:
            formatted[col] = [
                EMPTY_CELL
                if pd.isna(m)
                else f"{_format_value(m)}"
                if pd.isna(s)
                else f"{_format_value(m)} \\pm {_format_value(s)}"
                for m, s in zip(mean[col], std[col])
            ]
        else:
            formatted[col] = [EMPTY_CELL if pd.isna(m) else _format_value(m) for m in mean[col]]

    return formatted


def _apply_significance_markers(table: pd.DataFrame, markers: pd.DataFrame) -> pd.DataFrame:
    marked = table.copy()
    for col in marked.columns:
        marked[col] = [
            f"{value}{DAGGER_MARK}" if bool(marker) and value else value
            for value, marker in zip(marked[col], markers[col])
        ]
    return marked


def _format_value(value: float) -> str:
    return VALUE_FORMAT.format(value)


def _gradient_hex(value: float, vmin: float, vmax: float) -> str:
    if pd.isna(value):
        return ""

    if np.isclose(vmin, vmax):
        alpha = 1.0
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
        alpha = float(np.clip(norm(value), 0.0, 1.0))

    best = np.array(BEST_RGB).astype(float) / 256.0
    if GRADIENT_END == GRADIENT_TO_WORST:
        red, green, blue = _best_white_worst_rgb(alpha, best)
    else:
        red, green, blue = np.ones(3) * (1 - alpha) + best * alpha
    return rgb2hex((red, green, blue))


def _best_white_worst_rgb(alpha: float, best: np.ndarray) -> np.ndarray:
    worst = np.array(WORST_RGB).astype(float) / 256.0
    white = np.ones(3)
    if alpha < 0.5:
        return worst * (1 - 2 * alpha) + white * (2 * alpha)
    return white * (2 - 2 * alpha) + best * (2 * alpha - 1)


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return rgb2hex(np.array(rgb).astype(float) / 256.0)


def _text_color(background: str) -> str:
    red = int(background[1:3], 16) / 255
    green = int(background[3:5], 16) / 255
    blue = int(background[5:7], 16) / 255
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#f1f1f1" if luminance < 0.45 else "#000000"


def _sigma_zscore_hex(zscore: float) -> str:
    if pd.isna(zscore):
        return ""
    if zscore <= -1:
        return "#ffffff"
    if zscore >= 1:
        return _rgb_hex(BEST_RGB)
    return _gradient_hex(zscore, -1, 1)


def _metric_scores(values: pd.Series, metric: str) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return -values if LOWER_IS_BETTER[metric] else values


def _minmax_gradient_colors(values: pd.Series, metric: str) -> pd.Series:
    scores = _metric_scores(values, metric)
    vmin = scores.min(skipna=True)
    vmax = scores.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax):
        return pd.Series("", index=values.index)

    return scores.apply(lambda value: _gradient_hex(value, vmin, vmax))


def _sigma_zscore_gradient_colors(zscores: pd.Series) -> pd.Series:
    return zscores.apply(_sigma_zscore_hex)


def _best_to_sigma_gradient_colors(
    values: pd.Series,
    metric: str,
    sigma: float,
    factor: float = BEST_TO_SIGMA_FACTOR,
) -> pd.Series:
    scores = _metric_scores(values, metric)
    best_score = scores.max(skipna=True)
    if pd.isna(best_score):
        return pd.Series("", index=values.index)

    if pd.isna(sigma) or np.isclose(sigma, 0):
        return scores.apply(lambda value: "" if pd.isna(value) else _rgb_hex(BEST_RGB))

    worst_score = best_score - factor * sigma
    return scores.apply(lambda value: _gradient_hex(value, worst_score, best_score))


def _threshold_colors(values: pd.Series, metric: str) -> pd.Series:
    if metric == "coverage":
        return values.apply(lambda value: "" if pd.isna(value) else _rgb_hex(BEST_RGB) if value >= 0.95 else "#ffffff")
    if metric == "amplitude":
        return values.apply(lambda value: "" if pd.isna(value) else _rgb_hex(BEST_RGB) if value <= 0.05 else "#ffffff")

    return pd.Series("", index=values.index)


def _cell_styles(
    mean: pd.DataFrame,
    metrics: tuple[str, ...],
    sigma_zscores: pd.DataFrame | None = None,
    sigmas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    styles = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for metric in metrics:
        metric_values = mean.xs(metric, axis=1, level="metric")

        for dataset, values in metric_values.iterrows():
            colors = _metric_colors(metric, values, dataset, sigma_zscores, sigmas)

            for method, color in colors.items():
                if color:
                    styles.loc[dataset, (method, metric)] = f"background-color: {color}; color: {_text_color(color)};"

        best_values = metric_values.min(axis=1) if LOWER_IS_BETTER[metric] else metric_values.max(axis=1)

        for dataset, best_value in best_values.items():
            if pd.isna(best_value):
                continue
            best_mask = np.isclose(metric_values.loc[dataset], best_value)
            for method in metric_values.columns[best_mask]:
                styles.loc[dataset, (method, metric)] += " font-weight: bold;"

    return styles


def _metric_colors(
    metric: str,
    values: pd.Series,
    row_key,
    sigma_zscores: pd.DataFrame | None,
    sigmas: pd.DataFrame | None,
) -> pd.Series:
    strategy = METRIC_COLOR_STRATEGY.get(metric, COLOR_MINMAX)

    if strategy == COLOR_MINMAX:
        return _minmax_gradient_colors(values, metric)
    if strategy == COLOR_SIGMA_ZSCORE and sigma_zscores is not None:
        zscores = sigma_zscores.xs(metric, axis=1, level="metric").loc[row_key]
        return _sigma_zscore_gradient_colors(zscores)
    if strategy == COLOR_BEST_TO_SIGMA and sigmas is not None:
        sigma = sigmas.loc[row_key, metric] if row_key in sigmas.index and metric in sigmas.columns else np.nan
        return _best_to_sigma_gradient_colors(values, metric, sigma)
    if strategy == COLOR_THRESHOLD:
        return _threshold_colors(values, metric)

    return pd.Series("", index=values.index)


def _row_levels(df: pd.DataFrame) -> list[str]:
    return ["classifier", "dataset"] if "classifier" in df.columns else ["dataset"]


def _ordered_dataset_names(df: pd.DataFrame) -> list[str]:
    ordered = []

    for collection in ("uci_binary", "uci_multiclass"):
        datasets = list(dict.fromkeys(df.loc[df["collection"] == collection, "dataset"]))
        if datasets:
            ordered.extend(sort_datasets_by_size(collection, datasets, descending=True))

    ordered.extend(dataset for dataset in dict.fromkeys(df["dataset"]) if dataset not in ordered)
    return [decorate_dataset(dataset_map.get(dataset, dataset)) for dataset in ordered]


def _order_rows_by_dataset(index: pd.Index, rows: list, dataset_order: list[str] | None) -> list:
    if not dataset_order:
        return rows

    dataset_level = index.names.index("dataset") if isinstance(index, pd.MultiIndex) else None
    ordered_rows = []
    for dataset in dataset_order:
        if dataset_level is None:
            ordered_rows.extend(row for row in rows if row == dataset)
        else:
            ordered_rows.extend(row for row in rows if row[dataset_level] == dataset)
    ordered_rows.extend(row for row in rows if row not in ordered_rows)
    return ordered_rows


def _ordered_row_index(index: pd.Index, dataset_order: list[str] | None = None) -> pd.Index:
    if not isinstance(index, pd.MultiIndex):
        return pd.Index(_order_rows_by_dataset(index, list(index), dataset_order), name=index.name)

    if "classifier" not in index.names:
        return pd.MultiIndex.from_tuples(
            _order_rows_by_dataset(index, list(index), dataset_order),
            names=index.names,
        )

    classifier_level = index.names.index("classifier")
    ordered_rows = []
    for classifier in CLASSIFIER_FILTER:
        rows = [row for row in index if row[classifier_level] == classifier]
        ordered_rows.extend(_order_rows_by_dataset(index, rows, dataset_order))
    ordered_rows.extend(row for row in index if row not in ordered_rows)
    return pd.MultiIndex.from_tuples(ordered_rows, names=index.names)


def _as_row_key(key):
    return key if isinstance(key, tuple) else key


def _metric_sigma_context(
    df: pd.DataFrame,
    row_levels: list[str],
    mean: pd.DataFrame,
    methods: list[str],
    metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zscores = pd.DataFrame(np.nan, index=mean.index, columns=mean.columns)
    sigmas = pd.DataFrame(np.nan, index=mean.index, columns=metrics)

    available_metrics = [metric for metric in metrics if metric in df.columns]

    for key, group in df.groupby(row_levels, sort=False):
        row_key = _as_row_key(key)
        if row_key not in mean.index:
            continue

        for metric in available_metrics:
            if metric not in mean.columns.get_level_values("metric"):
                continue

            pooled_scores = _metric_scores(group[metric], metric)
            pooled_scores = pooled_scores[np.isfinite(pooled_scores)]
            if pooled_scores.empty:
                continue

            mu = pooled_scores.mean()
            sigma = pooled_scores.std(ddof=0)
            sigmas.loc[row_key, metric] = sigma

            for method in methods:
                mean_value = mean.loc[row_key, (method, metric)]
                if pd.isna(mean_value):
                    continue

                mean_score = -mean_value if LOWER_IS_BETTER[metric] else mean_value
                if pd.isna(sigma) or np.isclose(sigma, 0):
                    zscores.loc[row_key, (method, metric)] = 0.0
                else:
                    zscores.loc[row_key, (method, metric)] = (mean_score - mu) / sigma

    return zscores, sigmas


def _significance_markers(
    df: pd.DataFrame,
    row_levels: list[str],
    mean: pd.DataFrame,
    methods: list[str],
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    markers = pd.DataFrame(False, index=mean.index, columns=mean.columns)

    for key, group in df.groupby(row_levels, sort=False):
        row_key = _as_row_key(key)
        if row_key not in mean.index:
            continue

        group = group.copy()
        group["_sample_id"] = group.groupby("method", sort=False).cumcount()

        for metric in metrics:
            values = mean.xs(metric, axis=1, level="metric").loc[row_key]
            best_method = _best_method(values, metric, methods)
            if best_method is None:
                continue

            samples = group.pivot(index="_sample_id", columns="method", values=metric)
            if best_method not in samples:
                continue

            best_samples = pd.to_numeric(samples[best_method], errors="coerce")
            for method in methods:
                if method == best_method or method not in samples:
                    continue

                method_samples = pd.to_numeric(samples[method], errors="coerce")
                paired = pd.concat([best_samples, method_samples], axis=1).dropna()
                if paired.empty:
                    continue

                if _wilcoxon_not_significantly_different(paired.iloc[:, 0], paired.iloc[:, 1]):
                    markers.loc[row_key, (method, metric)] = True

    return markers


def _best_method(values: pd.Series, metric: str, methods: list[str]) -> str | None:
    values = values.dropna()
    if values.empty:
        return None

    best_value = values.min() if LOWER_IS_BETTER[metric] else values.max()
    best_methods = [method for method in methods if method in values.index and np.isclose(values[method], best_value)]
    return best_methods[0] if best_methods else None


def _wilcoxon_not_significantly_different(best_samples: pd.Series, method_samples: pd.Series) -> bool:
    differences = np.asarray(best_samples, dtype=float) - np.asarray(method_samples, dtype=float)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return False
    if np.allclose(differences, 0):
        return True

    try:
        pvalue = wilcoxon(differences, alternative="two-sided").pvalue
    except ValueError:
        return False

    return bool(np.isfinite(pvalue) and pvalue >= WILCOXON_ALPHA)


def _results_latex_table(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = METRICS,
    show_std: bool = True,
    dataset_order: list[str] | None = None,
) -> str:
    row_levels = _row_levels(df)
    stats = df.groupby(row_levels + ["method"], sort=False)[list(metrics)].agg(["mean", "std"])
    mean = stats.xs("mean", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)
    std = stats.xs("std", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)

    methods = [method for method in table_method_names() if method in mean.columns.get_level_values("method")]
    columns = _ordered_columns(methods, metrics)
    rows = _ordered_row_index(mean.index, dataset_order)
    mean = mean.reindex(index=rows)
    std = std.reindex(index=rows)
    mean = mean.reindex(columns=columns)
    std = std.reindex(columns=columns)
    table = _format_mean_std(mean, std, show_std)
    significance_markers = _significance_markers(df, row_levels, mean, methods, metrics)
    table = _apply_significance_markers(table, significance_markers)
    sigma_zscores, sigmas = _metric_sigma_context(df, row_levels, mean, methods, metrics)
    styles = _cell_styles(mean, metrics, sigma_zscores, sigmas)
    table = _to_display_column_order(table, methods, metrics)
    styles = _to_display_column_order(styles, methods, metrics)
    latex_columns = _latex_display_columns(table.columns)
    table.columns = latex_columns
    styles.columns = latex_columns

    latex = table.style.apply(lambda _: styles, axis=None).to_latex(
        column_format=_latex_column_format(methods, metrics, len(row_levels)),
        hrules=True,
        clines="skip-last;data",
        multicol_align="c|",
        convert_css=True,
    )
    return _hide_latex_header_names(latex, methods, metrics, show_metric_names=len(metrics) > 1)


def _hide_latex_header_names(
    latex: str,
    methods: list[str],
    metrics: tuple[str, ...],
    show_metric_names: bool = True,
) -> str:
    lines = latex.splitlines()
    formatted = []
    skip_next_dataset_header = False
    cmidirule = "\\cmidrule(r){3-9}\\cmidrule(l){10-16}"

    for line in lines:
        if line.startswith("metric & "):
            if show_metric_names:
                formatted.append(_format_metric_header_separators(line.replace("metric & ", " & ", 1), len(metrics)))
                # formatted.append("\\midrule")
                formatted.append(cmidirule)
                formatted.append(_method_family_header_line(1, methods, metrics))
                # formatted.append("\\midrule")
                formatted.append(cmidirule)
            skip_next_dataset_header = False
        elif line.startswith(" & metric & "):
            if show_metric_names:
                formatted.append(
                    _format_metric_header_separators(line.replace(" & metric & ", " &  & ", 1), len(metrics))
                )
                # formatted.append("\\midrule")
                formatted.append(cmidirule)
                formatted.append(_method_family_header_line(2, methods, metrics))
                # formatted.append("\\midrule")
                formatted.append(cmidirule)
            skip_next_dataset_header = False
        elif line.startswith("method & "):
            formatted.append(_protect_numeric_header_cells(line.replace("method & ", " & ", 1), 1, methods, metrics))
            skip_next_dataset_header = True
        elif line.startswith(" & method & "):
            formatted.append(
                _protect_numeric_header_cells(line.replace(" & method & ", " &  & ", 1), 2, methods, metrics)
            )
            skip_next_dataset_header = True
        elif skip_next_dataset_header and line.startswith("dataset & "):
            skip_next_dataset_header = False
        elif skip_next_dataset_header and line.startswith("classifier & dataset & "):
            skip_next_dataset_header = False
        else:
            formatted.append(line)
            skip_next_dataset_header = False

    return "\n".join(_drop_cmidrule_before_bottomrule(formatted)) + "\n"


def _drop_cmidrule_before_bottomrule(lines: list[str]) -> list[str]:
    cleaned = []
    for line in lines:
        if line.startswith("\\bottomrule") and cleaned and _is_horizontal_rule(cleaned[-1]):
            cleaned.pop()
        cleaned.append(line)
    return cleaned


def _is_horizontal_rule(line: str) -> bool:
    return line.startswith("\\cmidrule") or line.startswith("\\cline")


def _format_metric_header_separators(line: str, metric_count: int) -> str:
    if metric_count <= 1 or "\\multicolumn" not in line:
        return line

    pieces = line.split("\\multicolumn")
    for idx in range(1, len(pieces)):
        separator = "|" if idx < len(pieces) - 1 else ""
        pieces[idx] = pieces[idx].replace("{c|}", f"{{c{separator}}}", 1)
    return "\\multicolumn".join(pieces)


def _method_family_header_line(index_columns: int, methods: list[str], metrics: tuple[str, ...]) -> str:
    baseline_count = min(4, len(methods))
    ours_count = max(len(methods) - baseline_count, 0)
    cells = [""] * index_columns

    for metric_idx, _ in enumerate(metrics):
        is_last_metric = metric_idx == len(metrics) - 1
        if baseline_count:
            baseline_align = "c|" if ours_count or not is_last_metric else "c"
            cells.append(f"\\multicolumn{{{baseline_count}}}{{{baseline_align}}}{{Baseline methods}}")
        if ours_count:
            ours_align = "c" if is_last_metric else "c|"
            cells.append(f"\\multicolumn{{{ours_count}}}{{{ours_align}}}{{Our methods}}")

    return " & ".join(cells) + " \\\\"


def _protect_numeric_header_cells(
    line: str,
    index_columns: int,
    methods: list[str],
    metrics: tuple[str, ...],
) -> str:
    if " \\\\" not in line:
        return line

    body, suffix = line.rsplit(" \\\\", 1)
    cells = body.split(" & ")
    for idx in range(index_columns, len(cells)):
        cell = cells[idx].strip()
        if cell and not cell.startswith("\\multicolumn"):
            cell = f"\\rotatebox{{90}}{{{cell}}}"
            cells[idx] = (
                f"\\multicolumn{{1}}{{{_method_header_align(idx - index_columns, methods, metrics)}}}{{{cell}}}"
            )
    return " & ".join(cells) + " \\\\" + suffix


def _method_header_align(position: int, methods: list[str], metrics: tuple[str, ...]) -> str:
    method_count = len(methods)
    if method_count == 0:
        return "c"

    metric_index = position // method_count
    method_index = position % method_count
    is_last_method = method_index == method_count - 1
    is_last_metric = metric_index == len(metrics) - 1

    if is_last_method:
        return "c" if is_last_metric else "c|"
    # if method_index == 3:
    #     return "c|"
    return "c"


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


def _latex_display_columns(columns: pd.Index) -> pd.Index:
    if isinstance(columns, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(
            (
                tuple(
                    _latex_display_column_value(value, columns.names[level]) if isinstance(value, str) else value
                    for level, value in enumerate(col)
                )
                for col in columns
            ),
            names=columns.names,
        )

    return _escape_latex_columns(columns)


def _latex_display_column_value(value: str, level_name: str | None) -> str:
    if level_name == "metric":
        return metrics_map.get(value, _escape_latex_text(value))
    if level_name == "method":
        return _escape_latex_text(METHOD_LATEX_LABELS.get(value, value))
    return _escape_latex_text(value)


def _escape_latex_columns(columns: pd.Index) -> pd.Index:
    if isinstance(columns, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(
            (tuple(_escape_latex_text(value) if isinstance(value, str) else value for value in col) for col in columns),
            names=columns.names,
        )

    return pd.Index(
        (_escape_latex_text(value) if isinstance(value, str) else value for value in columns),
        name=columns.name,
    )


def _results_latex_document(tables: list[tuple[str, str, str]]) -> str:
    lines = [
        "\\documentclass{article}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{adjustbox}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\usepackage{graphicx}",
        "\\usepackage{multirow}",
        "\\usepackage{siunitx}",
        "\\geometry{margin=1in}",
        "\\sisetup{detect-all=true, separate-uncertainty=true}",
        f"\\setlength{{\\tabcolsep}}{{{TABLE_COLUMN_SEP}}}",
        "",
        "\\newcommand{\\pq}{SQS-S}",
        "\\newcommand{\\rqbs}{SQS-H}",
        "\\newcommand{\\bcap}{BayesCAP}",
        "",
        "\\begin{document}",
        "",
    ]

    for input_path, caption, label in tables:
        lines.extend(
            [
                "\\begin{table}[htbp]",
                "\\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                "\\begin{adjustbox}{max width=0.8\\textwidth}",
                f"\\input{{{input_path}}}",
                "\\end{adjustbox}",
                "\\end{table}",
                "",
            ]
        )

    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def _ordered_classifiers(base_dir: str, domain: str, acc: str) -> list[str]:
    acc_dir = os.path.join(base_dir, domain, acc)
    classifiers = set()

    if not os.path.isdir(acc_dir):
        return []

    for dataset in os.scandir(acc_dir):
        if not dataset.is_dir():
            continue
        classifiers.update(classifier.name for classifier in os.scandir(dataset.path) if classifier.is_dir())

    return [classifier for classifier in CLASSIFIER_FILTER if classifier in classifiers]


def _selected_metrics(pargs) -> tuple[str, ...]:
    if pargs.metrics is None:
        return METRICS

    unknown_metrics = [metric for metric in pargs.metrics if metric not in METRICS]
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(unknown_metrics)}. Available metrics: {', '.join(METRICS)}.")

    return tuple(metric for metric in METRICS if metric in pargs.metrics)


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", "main")


def gen_tables(pargs):
    experiment = EXPERIMENT
    domain = pargs.domain
    metrics = _selected_metrics(pargs)

    main_base_dir = os.path.join(env.root_dir, "main")
    table_dir = _table_dir(pargs)
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(tabular_dir, exist_ok=True)

    # tbls = []
    document_tables = []
    accs = get_acc_names()
    for acc in accs:
        classifiers = _ordered_classifiers(main_base_dir, domain, acc)
        if not classifiers:
            continue

        name = f"{experiment}_{domain}_{acc}"
        results = Results.load(
            base_dir=main_base_dir,
            acc_name=acc,
            domain=domain,
            filter_classifiers=classifiers,
            filter_methods=table_method_names(),
        )
        dataset_order = _ordered_dataset_names(results.df)
        df = (
            results.map_column_values("dataset", dataset_map)
            .apply_to_column("dataset", decorate_dataset)
            .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            .apply_to_column(["method", "true_accs", "estim_acc_ranges"], get_acc_coverage, new_col="coverage")
            .apply_to_column(["method", "estim_acc_ranges"], get_acc_amplitude, new_col="amplitude")
            .apply_to_column(["method", "true_accs", "estim_acc_ranges"], get_acc_winkler, new_col="winkler")
            .select_columns(
                ["classifier", "dataset", "method", "acc_err", "true_accs", "coverage", "amplitude", "winkler"]
            )
            .df
        )

        tbl = _results_latex_table(df, metrics, pargs.show_std, dataset_order)
        tabular_filename = f"{name}.tex"
        with open(os.path.join(tabular_dir, tabular_filename), "w") as f:
            f.write(tbl)

        # tbls.append(tbl)
        document_tables.append(
            (
                f"tabular/{tabular_filename}",
                f"{_escape_latex_text(acc)} results on {_escape_latex_text(domain)} datasets.",
                f"tab:{experiment}-{domain}-{acc}",
            )
        )
        print(f"table {name} genned")

    if not document_tables:
        raise RuntimeError(f"No tables generated for domain={domain!r}.")

    document_path = os.path.join(table_dir, f"{experiment}_{domain}.tex")
    with open(document_path, "w") as f:
        f.write(_results_latex_document(document_tables))


def gen_pdf(pargs):
    experiment = EXPERIMENT
    domain = pargs.domain
    table_dir = _table_dir(pargs)
    tex_filename = f"{experiment}_{domain}.tex"
    tex_path = os.path.join(table_dir, tex_filename)

    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"{tex_path} does not exist. Generate tables first.")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
        cwd=table_dir,
        check=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=None, help="Metrics to include in tables")
    parser.add_argument("--no-std", action="store_false", dest="show_std", help="Show only means, without std")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
