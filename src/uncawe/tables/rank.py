import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex

import uncawe.env as env
from uncawe.config import acc_from_ct, get_acc_names, get_method_names
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables import main as main_tables

METRICS = ("acc_err", "winkler")
LOWER_IS_BETTER = {"acc_err": True, "winkler": True}
EXCLUDED_METHODS = {"L-CBPE"}
BEST_RGB = (154, 254, 154)
WHITE_RGB = (255, 255, 255)
WHITE_RANK = 6.0
VALUE_FORMAT = "{:.2f}"


def _selected_metrics(pargs) -> tuple[str, ...]:
    if pargs.metrics is None:
        return METRICS

    unknown_metrics = [metric for metric in pargs.metrics if metric not in METRICS]
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(unknown_metrics)}. Available metrics: {', '.join(METRICS)}.")

    return tuple(metric for metric in METRICS if metric in pargs.metrics)


def _methods() -> list[str]:
    return [method for method in get_method_names() if method not in EXCLUDED_METHODS]


def _rgb_array(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.array(rgb).astype(float) / 255.0


def _rank_color(value: float) -> str:
    if pd.isna(value):
        return ""

    alpha = float(np.clip((value - 1.0) / (WHITE_RANK - 1.0), 0.0, 1.0))
    rgb = _rgb_array(BEST_RGB) * (1.0 - alpha) + _rgb_array(WHITE_RGB) * alpha
    return rgb2hex(rgb)


def _format_rank_table(mean_rank: pd.DataFrame) -> pd.DataFrame:
    formatted = pd.DataFrame("", index=mean_rank.index, columns=mean_rank.columns)
    for col in mean_rank.columns:
        formatted[col] = [
            main_tables.EMPTY_CELL if pd.isna(value) else VALUE_FORMAT.format(value) for value in mean_rank[col]
        ]
    return formatted


def _rank_cell_styles(mean_rank: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    styles = pd.DataFrame("", index=mean_rank.index, columns=mean_rank.columns)

    for metric in metrics:
        metric_values = mean_rank.xs(metric, axis=1, level="metric")

        for classifier, values in metric_values.iterrows():
            for method, color in values.apply(_rank_color).items():
                if color:
                    styles.loc[classifier, (method, metric)] = (
                        f"background-color: {color}; color: {main_tables._text_color(color)};"
                    )

            best_value = values.min(skipna=True)
            if pd.isna(best_value):
                continue
            best_mask = np.isclose(values, best_value)
            for method in values.index[best_mask]:
                styles.loc[classifier, (method, metric)] += " font-weight: bold;"

    return styles


def _mean_sample_values(df: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    return (
        df.groupby(["classifier", "dataset", "method"], sort=False, observed=True)[list(metrics)].mean().reset_index()
    )


def _rank_mean_values(df: pd.DataFrame, metrics: tuple[str, ...]) -> pd.DataFrame:
    mean_values = _mean_sample_values(df, metrics)
    long = mean_values.melt(
        id_vars=["classifier", "dataset", "method"],
        value_vars=list(metrics),
        var_name="metric",
        value_name="value",
    )
    long["rank"] = np.nan

    for metric in metrics:
        mask = long["metric"] == metric
        long.loc[mask, "rank"] = (
            long.loc[mask]
            .groupby(["classifier", "dataset", "metric"], sort=False, observed=True)["value"]
            .rank(method="average", ascending=LOWER_IS_BETTER[metric])
        )

    return (
        long.groupby(["classifier", "method", "metric"], sort=False, observed=True)["rank"]
        .mean()
        .unstack(["method", "metric"])
    )


def _rank_latex_table(df: pd.DataFrame, metrics: tuple[str, ...] = METRICS) -> str:
    mean_rank = _rank_mean_values(df, metrics)
    methods = [method for method in _methods() if method in mean_rank.columns.get_level_values("method")]
    columns = main_tables._ordered_columns(methods, metrics)
    rows = pd.Index(
        [classifier for classifier in main_tables.CLASSIFIER_FILTER if classifier in mean_rank.index],
        name="classifier",
    )
    rows = rows.append(pd.Index([classifier for classifier in mean_rank.index if classifier not in rows]))

    mean_rank = mean_rank.reindex(index=rows, columns=columns)
    table = _format_rank_table(mean_rank)
    styles = _rank_cell_styles(mean_rank, metrics)

    table = main_tables._to_display_column_order(table, methods, metrics)
    styles = main_tables._to_display_column_order(styles, methods, metrics)
    latex_columns = main_tables._latex_display_columns(table.columns)
    table.columns = latex_columns
    styles.columns = latex_columns

    latex = table.style.apply(lambda _: styles, axis=None).to_latex(
        column_format=main_tables._latex_column_format(methods, metrics, 1),
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
    skip_classifier_header = False

    for line in lines:
        if line.startswith("metric & "):
            if show_metric_names:
                formatted.append(
                    main_tables._format_metric_header_separators(line.replace("metric & ", " & ", 1), len(metrics))
                )
                # formatted.append("\\midrule")
                formatted.append("\\cmidrule(r){2-7}\\cmidrule(l){8-13}")
            skip_classifier_header = False
        elif line.startswith("method & "):
            formatted.append(_protect_method_header_cells(line.replace("method & ", " & ", 1), 1, methods, metrics))
            skip_classifier_header = True
        elif skip_classifier_header and line.startswith("classifier & "):
            skip_classifier_header = False
        else:
            formatted.append(line)
            skip_classifier_header = False

    return "\n".join(formatted) + "\n"


def _protect_method_header_cells(
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
            cells[idx] = (
                f"\\multicolumn{{1}}{{{main_tables._method_header_align(idx - index_columns, methods, metrics)}}}"
                f"{{{cell}}}"
            )
    return " & ".join(cells) + " \\\\" + suffix


def _prepare_results(acc_name: str, domain: str, methods: list[str]) -> pd.DataFrame:
    main_base_dir = os.path.join(env.root_dir, "main")
    classifiers = main_tables._ordered_classifiers(main_base_dir, domain, acc_name)
    if not classifiers:
        return pd.DataFrame()

    results = Results.load(
        base_dir=main_base_dir,
        acc_name=acc_name,
        domain=domain,
        filter_classifiers=classifiers,
        filter_methods=methods,
    )
    return (
        results.apply_to_column("true_cts", lambda ct: acc_from_ct(acc_name, ct), new_col="true_accs")
        .apply_to_column(["method", "true_accs", "estim_acc_ranges"], main_tables.get_acc_winkler, new_col="winkler")
        .select_columns(["classifier", "dataset", "method", "acc_err", "true_accs", "winkler"])
        .df
    )


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", "rank")


def _rank_latex_document(tables: list[tuple[str, str, str]]) -> str:
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
        f"\\setlength{{\\tabcolsep}}{{{main_tables.TABLE_COLUMN_SEP}}}",
        "",
        "\\newcommand{\\pq}{SQS-S}",
        "\\newcommand{\\bcap}{BayesCAP}",
        "\\newcommand{\\rqbs}{SQS-H}",
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


def gen_tables(pargs):
    domain = pargs.domain
    metrics = _selected_metrics(pargs)
    methods = _methods()

    table_dir = _table_dir(pargs)
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(tabular_dir, exist_ok=True)

    document_tables = []
    for acc_name in get_acc_names():
        try:
            df = _prepare_results(acc_name, domain, methods)
        except ValueError:
            continue
        if df.empty:
            continue

        name = f"{EXPERIMENT}_{domain}_{acc_name}_rank"
        table = _rank_latex_table(df, metrics)
        tabular_filename = f"{name}.tex"
        with open(os.path.join(tabular_dir, tabular_filename), "w") as f:
            f.write(table)

        document_tables.append(
            (
                f"tabular/{tabular_filename}",
                f"Mean rank of {_escape_latex_text(acc_name)} results on {_escape_latex_text(domain)} datasets.",
                f"tab:{EXPERIMENT}-{domain}-{acc_name}-rank",
            )
        )
        print(f"table {name} genned")

    if not document_tables:
        raise RuntimeError(f"No rank tables generated for domain={domain!r}.")

    document_path = os.path.join(table_dir, f"{EXPERIMENT}_{domain}_rank.tex")
    with open(document_path, "w") as f:
        f.write(_rank_latex_document(document_tables))


def _escape_latex_text(text: str) -> str:
    return main_tables._escape_latex_text(text)


def gen_pdf(pargs):
    domain = pargs.domain
    table_dir = _table_dir(pargs)
    tex_filename = f"{EXPERIMENT}_{domain}_rank.tex"
    tex_path = os.path.join(table_dir, tex_filename)

    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"{tex_path} does not exist. Generate rank tables first.")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
        cwd=table_dir,
        check=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate LaTeX mean-rank tables for the main experiment.")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from rank tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=None, help="Metrics to include in tables")
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
