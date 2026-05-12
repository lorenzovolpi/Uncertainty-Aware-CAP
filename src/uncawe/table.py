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
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, get_acc_names, get_method_names
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.util import decorate_dataset

METRICS = ("acc_err", "coverage", "amplitude")
LOWER_IS_BETTER = {"acc_err": True, "coverage": False, "amplitude": True}
VALUE_FORMAT = "{:.3f}"
COLORMAP_SATURATION = 0.65
COLORMAP_LIGHTNESS = 0.35

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


def get_acc_coverage(true_acc: float, accs_range: np.ndarray) -> float:
    ci = ConfidenceInterval(accs_range, CONFIDENCE_LEVEL)
    return ci.coverage(true_acc)


def get_acc_amplitude(accs_range: np.ndarray) -> float:
    return float(accs_range.max() - accs_range.min())


def _ordered_columns(methods: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([methods, METRICS], names=["method", "metric"])


def _latex_column_format(methods: list[str]) -> str:
    return "l|" + "|".join("c" * len(METRICS) for _ in methods) + "|"


def _format_mean_std(mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    formatted = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for col in mean.columns:
        formatted[col] = [
            "" if pd.isna(m) else f"{_format_value(m)} $\\pm$ {('--' if pd.isna(s) else _format_value(s))}"
            for m, s in zip(mean[col], std[col])
        ]

    return formatted


def _format_value(value: float) -> str:
    formatted = VALUE_FORMAT.format(value)
    if formatted.startswith("-0."):
        return "-" + formatted[2:]
    if formatted.startswith("0."):
        return formatted[1:]
    return formatted


def _rdylgn_hex(value: float, vmin: float, vmax: float) -> str:
    worst = (251, 160, 159)
    best = (154, 254, 154)

    norm = Normalize(vmin=vmin, vmax=vmax)
    if norm(value) < 0.5:
        red, green, blue = np.array(worst).astype(float) / 256.0
        alpha = 1 - (norm(value) / 0.5)
    elif norm(value) > 0.5:
        red, green, blue = np.array(best).astype(float) / 256.0
        alpha = (norm(value) - 0.5) / 0.5
    else:
        red, green, blue = np.array(best).astype(float) / 256.0
        alpha = 0.0

    red, green, blue = np.array([red, green, blue]) * alpha + (1 - alpha)
    return rgb2hex((red, green, blue))


def _text_color(background: str) -> str:
    red = int(background[1:3], 16) / 255
    green = int(background[3:5], 16) / 255
    blue = int(background[5:7], 16) / 255
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "#f1f1f1" if luminance < 0.45 else "#000000"


def _cell_styles(mean: pd.DataFrame) -> pd.DataFrame:
    styles = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for metric in METRICS:
        metric_values = mean.xs(metric, axis=1, level="metric")

        for dataset, values in metric_values.iterrows():
            scores = -values if LOWER_IS_BETTER[metric] else values
            vmin = scores.min(skipna=True)
            vmax = scores.max(skipna=True)

            if pd.isna(vmin) or pd.isna(vmax):
                continue

            if np.isclose(vmin, vmax):
                colors = pd.Series("#ffffbf", index=scores.index)
            else:
                colors = scores.apply(lambda value: "" if pd.isna(value) else _rdylgn_hex(value, vmin, vmax))

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


def _results_latex_table(df: pd.DataFrame) -> str:
    stats = df.groupby(["dataset", "method"], sort=False)[list(METRICS)].agg(["mean", "std"])
    mean = stats.xs("mean", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)
    std = stats.xs("std", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)

    methods = [method for method in get_method_names() if method in mean.columns.get_level_values("method")]
    columns = _ordered_columns(methods)
    mean = mean.reindex(columns=columns)
    std = std.reindex(columns=columns)
    table = _format_mean_std(mean, std)
    styles = _cell_styles(mean)
    latex_columns = _escape_latex_columns(table.columns)
    table.columns = latex_columns
    styles.columns = latex_columns

    latex = table.style.apply(lambda _: styles, axis=None).to_latex(
        column_format=_latex_column_format(methods),
        hrules=True,
        clines="skip-last;data",
        multicol_align="c|",
        convert_css=True,
    )
    return _hide_latex_header_names(latex)


def _hide_latex_header_names(latex: str) -> str:
    lines = latex.splitlines()
    formatted = []
    skip_next_dataset_header = False

    for line in lines:
        if line.startswith("method & "):
            formatted.append(line.replace("method & ", " & ", 1))
            formatted.append("\\midrule")
            skip_next_dataset_header = False
        elif line.startswith("metric & "):
            formatted.append(line.replace("metric & ", " & ", 1))
            skip_next_dataset_header = True
        elif skip_next_dataset_header and line.startswith("dataset & "):
            skip_next_dataset_header = False
        else:
            formatted.append(line)
            skip_next_dataset_header = False

    return "\n".join(formatted) + "\n"


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


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
                "\\begin{adjustbox}{max width=\\textwidth}",
                f"\\input{{{input_path}}}",
                "\\end{adjustbox}",
                "\\end{table}",
                "",
            ]
        )

    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def gen_tables(pargs):
    experiment = EXPERIMENT
    domain = pargs.domain

    main_base_dir = os.path.join(env.root_dir, "main")
    table_dir = os.path.join(env.root_dir, "tables")
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(tabular_dir, exist_ok=True)

    # tbls = []
    document_tables = []
    accs = get_acc_names()
    for acc in accs:
        name = f"{experiment}_{domain}_{acc}"
        df = (
            Results.load(base_dir=main_base_dir, acc_name=acc, domain=domain)
            .map_column_values("dataset", dataset_map)
            .apply_to_column("dataset", decorate_dataset)
            .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            .apply_to_column(["true_accs", "estim_acc_ranges"], get_acc_coverage, new_col="coverage")
            .apply_to_column("estim_acc_ranges", get_acc_amplitude, new_col="amplitude")
            .select_columns(["dataset", "method", "acc_err", "true_accs", "coverage", "amplitude"])
            .df
        )

        tbl = _results_latex_table(df)
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

    document_path = os.path.join(table_dir, f"{experiment}_{domain}.tex")
    with open(document_path, "w") as f:
        f.write(_results_latex_document(document_tables))


def gen_pdf(pargs):
    experiment = EXPERIMENT
    domain = pargs.domain
    table_dir = os.path.join(env.root_dir, "tables")
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
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
