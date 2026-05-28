import os
import pickle
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from cap.models import ConfidenceInterval
from matplotlib import colormaps
from matplotlib.colors import Normalize, rgb2hex

import uncawe.env as env
from uncawe.config import CONFIDENCE_LEVEL, METHOD_CLASS, acc_from_ct, get_acc_names, get_method_names
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.util import decorate_dataset

METRICS = ("acc_err", "coverage", "amplitude", "winkler")
LOWER_IS_BETTER = {"acc_err": True, "coverage": False, "amplitude": True, "winkler": True}
VALUE_FORMAT = "{:.3f}"
COLORMAP_SATURATION = 0.65
COLORMAP_LIGHTNESS = 0.35
CLASSIFIER_FILTER = ("LR", "kNN", "SVM", "MLP", "RF")
WORST_RGB = (251, 160, 159)
BEST_RGB = (154, 254, 154)

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
    "acc_err": "$\\Delta$",
    "coverage": "$\\mathcal{C}$",
    "amplitude": "$\\mathcal{A}$",
    "winkler": "$\\mathcal{W}$",
}


def get_acc_coverage(method: str, true_acc: float, accs_range: np.ndarray) -> float:
    ci = METHOD_CLASS[method].ci_from_accs(accs_range, confidence_level=CONFIDENCE_LEVEL)
    return ci.coverage(true_acc)


def get_acc_amplitude(method: str, accs_range: np.ndarray) -> float:
    ci = METHOD_CLASS[method].ci_from_accs(accs_range, confidence_level=CONFIDENCE_LEVEL)
    return ci.amplitude()


def get_acc_winkler(method: str, true_acc: float, accs_range: np.ndarray) -> float:
    arr = np.asarray(accs_range, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or not np.isfinite(true_acc):
        return np.nan

    alpha = 1 - CONFIDENCE_LEVEL
    ci = METHOD_CLASS[method].ci_from_accs(accs_range, confidence_level=CONFIDENCE_LEVEL)
    ampl = ci.amplitude()
    low, high = ci.low, ci.high

    if true_acc < low:
        ampl += (2 / alpha) * (low - true_acc)
    elif true_acc > high:
        ampl += (2 / alpha) * (true_acc - high)
    return float(ampl)


def _ordered_columns(methods: list[str], metrics: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([methods, metrics], names=["method", "metric"])


def _latex_column_format(methods: list[str], metrics: tuple[str, ...], index_columns: int = 1) -> str:
    method_formats = ["c" * len(metrics) for _ in methods]
    if len(method_formats) > 3:
        return "l" * index_columns + "|" + "|".join(method_formats[:3]) + "||" + "|".join(method_formats[3:]) + "|"
    return "l" * index_columns + "|" + "|".join(method_formats) + "|"


def _format_mean_std(mean: pd.DataFrame, std: pd.DataFrame, show_std: bool = True) -> pd.DataFrame:
    formatted = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for col in mean.columns:
        if show_std:
            formatted[col] = [
                "" if pd.isna(m) else f"{_format_value(m)} $\\pm$ {('--' if pd.isna(s) else _format_value(s))}"
                for m, s in zip(mean[col], std[col])
            ]
        else:
            formatted[col] = ["" if pd.isna(m) else _format_value(m) for m in mean[col]]

    return formatted


def _format_value(value: float) -> str:
    formatted = VALUE_FORMAT.format(value)
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    if formatted.startswith("0."):
        return formatted[1:]
    return formatted


def _rdylgn_hex(value: float, vmin: float, vmax: float) -> str:
    norm = Normalize(vmin=vmin, vmax=vmax)
    if norm(value) < 0.5:
        red, green, blue = np.array(WORST_RGB).astype(float) / 256.0
        alpha = 1 - (norm(value) / 0.5)
    elif norm(value) > 0.5:
        red, green, blue = np.array(BEST_RGB).astype(float) / 256.0
        alpha = (norm(value) - 0.5) / 0.5
    else:
        red, green, blue = np.array(BEST_RGB).astype(float) / 256.0
        alpha = 0.0

    red, green, blue = np.array([red, green, blue]) * alpha + (1 - alpha)
    return rgb2hex((red, green, blue))


def _rgb_hex(rgb: tuple[int, int, int]) -> str:
    return rgb2hex(np.array(rgb).astype(float) / 256.0)


def _text_color(background: str) -> str:
    red = int(background[1:3], 16) / 255
    green = int(background[3:5], 16) / 255
    blue = int(background[5:7], 16) / 255
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#f1f1f1" if luminance < 0.45 else "#000000"


def _winkler_zscore_hex(zscore: float) -> str:
    if pd.isna(zscore):
        return ""
    if zscore <= -1:
        return _rgb_hex(WORST_RGB)
    if zscore >= 1:
        return _rgb_hex(BEST_RGB)
    return _rdylgn_hex(zscore, -1, 1)


def _cell_styles(
    mean: pd.DataFrame,
    metrics: tuple[str, ...],
    winkler_zscores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    styles = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for metric in metrics:
        metric_values = mean.xs(metric, axis=1, level="metric")

        for dataset, values in metric_values.iterrows():
            if metric == "acc_err":
                scores = -values
                vmin = scores.min(skipna=True)
                vmax = scores.max(skipna=True)

                if pd.isna(vmin) or pd.isna(vmax):
                    continue

                if np.isclose(vmin, vmax):
                    colors = pd.Series("#ffffbf", index=scores.index)
                else:
                    colors = scores.apply(lambda value: "" if pd.isna(value) else _rdylgn_hex(value, vmin, vmax))
            elif metric == "winkler" and winkler_zscores is not None:
                zscores = winkler_zscores.xs("winkler", axis=1, level="metric").loc[dataset]
                colors = zscores.apply(_winkler_zscore_hex)
            elif metric == "winkler":
                scores = -values
                vmin = scores.min(skipna=True)
                vmax = scores.max(skipna=True)

                if pd.isna(vmin) or pd.isna(vmax):
                    continue

                if np.isclose(vmin, vmax):
                    colors = pd.Series("#ffffbf", index=scores.index)
                else:
                    colors = scores.apply(lambda value: "" if pd.isna(value) else _rdylgn_hex(value, vmin, vmax))
            elif metric == "coverage":
                colors = values.apply(
                    lambda value: "" if pd.isna(value) else _rgb_hex(BEST_RGB if value >= 0.95 else WORST_RGB)
                )
            elif metric == "amplitude":
                colors = values.apply(
                    lambda value: "" if pd.isna(value) else _rgb_hex(BEST_RGB if value <= 0.05 else WORST_RGB)
                )
            else:
                continue

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


def _row_levels(df: pd.DataFrame) -> list[str]:
    return ["classifier", "dataset"] if "classifier" in df.columns else ["dataset"]


def _ordered_row_index(index: pd.Index) -> pd.Index:
    if not isinstance(index, pd.MultiIndex) or "classifier" not in index.names:
        return index

    classifier_level = index.names.index("classifier")
    ordered_rows = [row for classifier in CLASSIFIER_FILTER for row in index if row[classifier_level] == classifier]
    ordered_rows.extend(row for row in index if row not in ordered_rows)
    return pd.MultiIndex.from_tuples(ordered_rows, names=index.names)


def _as_row_key(key):
    return key if isinstance(key, tuple) else key


def _winkler_zscores(
    df: pd.DataFrame,
    row_levels: list[str],
    mean: pd.DataFrame,
    methods: list[str],
) -> pd.DataFrame | None:
    if "winkler" not in df.columns or ("winkler" not in mean.columns.get_level_values("metric")):
        return None

    zscores = pd.DataFrame(np.nan, index=mean.index, columns=mean.columns)

    for key, group in df.groupby(row_levels, sort=False):
        row_key = _as_row_key(key)
        if row_key not in mean.index:
            continue

        pooled_scores = -pd.to_numeric(group["winkler"], errors="coerce")
        pooled_scores = pooled_scores[np.isfinite(pooled_scores)]
        if pooled_scores.empty:
            continue

        mu = pooled_scores.mean()
        sigma = pooled_scores.std(ddof=0)

        for method in methods:
            mean_winkler = mean.loc[row_key, (method, "winkler")]
            if pd.isna(mean_winkler):
                continue
            if pd.isna(sigma) or np.isclose(sigma, 0):
                zscores.loc[row_key, (method, "winkler")] = 0.0
            else:
                zscores.loc[row_key, (method, "winkler")] = ((-mean_winkler) - mu) / sigma

    return zscores


def _results_latex_table(df: pd.DataFrame, metrics: tuple[str, ...] = METRICS, show_std: bool = True) -> str:
    row_levels = _row_levels(df)
    stats = df.groupby(row_levels + ["method"], sort=False)[list(metrics)].agg(["mean", "std"])
    mean = stats.xs("mean", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)
    std = stats.xs("std", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)

    methods = [method for method in get_method_names() if method in mean.columns.get_level_values("method")]
    columns = _ordered_columns(methods, metrics)
    rows = _ordered_row_index(mean.index)
    mean = mean.reindex(index=rows)
    std = std.reindex(index=rows)
    mean = mean.reindex(columns=columns)
    std = std.reindex(columns=columns)
    table = _format_mean_std(mean, std, show_std)
    winkler_zscores = _winkler_zscores(df, row_levels, mean, methods) if "winkler" in metrics else None
    styles = _cell_styles(mean, metrics, winkler_zscores)
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
    return _hide_latex_header_names(latex, show_metric_names=len(metrics) > 1)


def _hide_latex_header_names(latex: str, show_metric_names: bool = True) -> str:
    lines = latex.splitlines()
    formatted = []
    skip_next_dataset_header = False

    for line in lines:
        if line.startswith("method & "):
            formatted.append(_add_method_group_separator(line.replace("method & ", " & ", 1)))
            if show_metric_names:
                formatted.append("\\midrule")
            skip_next_dataset_header = False
        elif line.startswith(" & method & "):
            formatted.append(_add_method_group_separator(line.replace(" & method & ", " &  & ", 1)))
            if show_metric_names:
                formatted.append("\\midrule")
            skip_next_dataset_header = False
        elif line.startswith("metric & "):
            if show_metric_names:
                formatted.append(line.replace("metric & ", " & ", 1))
            skip_next_dataset_header = True
        elif line.startswith(" & metric & "):
            if show_metric_names:
                formatted.append(line.replace(" & metric & ", " &  & ", 1))
            skip_next_dataset_header = True
        elif skip_next_dataset_header and line.startswith("dataset & "):
            skip_next_dataset_header = False
        elif skip_next_dataset_header and line.startswith("classifier & dataset & "):
            skip_next_dataset_header = False
        else:
            formatted.append(line)
            skip_next_dataset_header = False

    return "\n".join(formatted) + "\n"


def _add_method_group_separator(line: str) -> str:
    multicol = "\\multicolumn"
    if line.count(multicol) <= 3:
        return line

    pieces = line.split(multicol)
    pieces[3] = pieces[3].replace("{c|}", "{c||}", 1)
    return multicol.join(pieces)


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


def _latex_display_columns(columns: pd.Index) -> pd.Index:
    if isinstance(columns, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(
            (
                tuple(
                    metrics_map.get(value, _escape_latex_text(value)) if isinstance(value, str) else value
                    for value in col
                )
                for col in columns
            ),
            names=columns.names,
        )

    return _escape_latex_columns(columns)


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
        "\\usepackage{multirow}",
        "\\geometry{margin=1in}",
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
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables")


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
        df = (
            Results.load(
                base_dir=main_base_dir,
                acc_name=acc,
                domain=domain,
                filter_classifiers=classifiers,
                filter_methods=get_method_names(),
            )
            .map_column_values("dataset", dataset_map)
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

        tbl = _results_latex_table(df, metrics, pargs.show_std)
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
