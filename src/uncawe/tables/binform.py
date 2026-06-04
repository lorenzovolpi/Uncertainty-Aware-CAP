import os
import subprocess
from argparse import ArgumentParser

import pandas as pd

import uncawe.env as env
from uncawe.binform import EXPERIMENT, _method_names
from uncawe.config import acc_from_ct, get_acc_names
from uncawe.results import Results
from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS

METRICS = ("acc_err", "winkler")
CLASSIFIER_ORDER = ("LR", "kNN", "SVM", "MLP", "RF")
METHOD_LATEX_OVERRIDES = {
    "BayesCAP_uniform": "\\bcap{}-U",
    "BayesCAP_transformed_val": "\\bcap{}-T",
}
COLOR_BEST_TO_WORST = "best-to-worst"
COLOR_BEST_TO_WHITE = "best-to-white"


def _method_latex_label(method: str) -> str:
    return METHOD_LATEX_OVERRIDES.get(method, METHOD_LATEX_LABELS.get(method, method))


def _escape_latex_text(text: str) -> str:
    return text.replace("_", "\\_")


def _latex_display_columns(columns: pd.Index) -> pd.Index:
    if not isinstance(columns, pd.MultiIndex):
        return pd.Index((_escape_latex_text(value) if isinstance(value, str) else value for value in columns))

    return pd.MultiIndex.from_tuples(
        (
            tuple(
                main_tables.metrics_map.get(value, _escape_latex_text(value))
                if columns.names[level] == "metric"
                else _method_latex_label(value)
                if columns.names[level] == "method"
                else _escape_latex_text(value)
                if isinstance(value, str)
                else value
                for level, value in enumerate(col)
            )
            for col in columns
        ),
        names=columns.names,
    )


def _ordered_columns(methods: list[str], metrics: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([methods, metrics], names=["method", "metric"])


def _display_columns(methods: list[str], metrics: tuple[str, ...]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([metrics, methods], names=["metric", "method"])


def _to_display_column_order(df: pd.DataFrame, methods: list[str], metrics: tuple[str, ...]) -> pd.DataFrame:
    return df.swaplevel("method", "metric", axis=1).reindex(columns=_display_columns(methods, metrics))


def _latex_column_format(methods: list[str], metrics: tuple[str, ...]) -> str:
    numeric_column = "S[table-format=1.3,table-align-text-post=false]"
    return "l" + "|".join(numeric_column * len(methods) for _ in metrics)


def _hide_latex_header_names(latex: str, methods: list[str], metrics: tuple[str, ...]) -> str:
    lines = latex.splitlines()
    formatted = []
    skip_classifier_header = False

    for line in lines:
        if line.startswith("metric & "):
            formatted.append(
                main_tables._format_metric_header_separators(line.replace("metric & ", " & ", 1), len(metrics))
            )
            formatted.append("\\midrule")
            skip_classifier_header = False
        elif line.startswith("method & "):
            formatted.append(
                main_tables._protect_numeric_header_cells(line.replace("method & ", " & ", 1), 1, methods, metrics)
            )
            skip_classifier_header = True
        elif skip_classifier_header and line.startswith("classifier & "):
            skip_classifier_header = False
        else:
            formatted.append(line)
            skip_classifier_header = False

    return "\n".join(formatted) + "\n"


def _ordered_classifiers(df: pd.DataFrame) -> list[str]:
    present = set(df["classifier"])
    ordered = [classifier for classifier in CLASSIFIER_ORDER if classifier in present]
    ordered.extend(classifier for classifier in dict.fromkeys(df["classifier"]) if classifier not in ordered)
    return ordered


def _binform_latex_table(df: pd.DataFrame, methods: list[str], show_std: bool) -> str:
    stats = df.groupby(["classifier", "method"], sort=False)[list(METRICS)].agg(["mean", "std"])
    mean = stats.xs("mean", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)
    std = stats.xs("std", axis=1, level=1).unstack("method").swaplevel(0, 1, axis=1)

    methods = [method for method in methods if method in mean.columns.get_level_values("method")]
    columns = _ordered_columns(methods, METRICS)
    rows = pd.Index(_ordered_classifiers(df), name="classifier")
    mean = mean.reindex(index=rows, columns=columns)
    std = std.reindex(index=rows, columns=columns)

    table = main_tables._format_mean_std(mean, std, show_std)
    sigma_zscores, sigmas = main_tables._metric_sigma_context(df, ["classifier"], mean, methods, METRICS)
    styles = main_tables._cell_styles(mean, METRICS, sigma_zscores, sigmas)

    table = _to_display_column_order(table, methods, METRICS)
    styles = _to_display_column_order(styles, methods, METRICS)
    latex_columns = _latex_display_columns(table.columns)
    table.columns = latex_columns
    styles.columns = latex_columns

    latex = table.style.apply(lambda _: styles, axis=None).to_latex(
        column_format=_latex_column_format(methods, METRICS),
        hrules=True,
        clines="skip-last;data",
        multicol_align="c|",
        convert_css=True,
    )
    return _hide_latex_header_names(latex, methods, METRICS)


def _prepare_results(acc_name: str, methods: list[str]) -> pd.DataFrame:
    base_dir = os.path.join(env.root_dir, EXPERIMENT)
    results = Results.load(
        base_dir=base_dir,
        domain="classic",
        acc_name=acc_name,
        dataset="connect-4",
        filter_classifiers=list(CLASSIFIER_ORDER),
        filter_methods=methods,
    )
    return (
        results.apply_to_column("true_cts", lambda ct: acc_from_ct(acc_name, ct), new_col="true_accs")
        .apply_to_column(["method", "true_accs", "estim_acc_ranges"], main_tables.get_acc_winkler, new_col="winkler")
        .select_columns(["classifier", "method", "acc_err", "true_accs", "winkler"])
        .df
    )


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", "binform")


def _set_coloring(coloring: str):
    if coloring == COLOR_BEST_TO_WORST:
        main_tables.GRADIENT_END = main_tables.GRADIENT_TO_WORST
    elif coloring == COLOR_BEST_TO_WHITE:
        main_tables.GRADIENT_END = main_tables.GRADIENT_TO_WHITE
    else:
        raise ValueError(f"Unknown coloring: {coloring}")


def _latex_document(tables: list[tuple[str, str, str]]) -> str:
    lines = [
        "\\documentclass{article}",
        "\\usepackage[table]{xcolor}",
        "\\usepackage{adjustbox}",
        "\\usepackage{booktabs}",
        "\\usepackage{geometry}",
        "\\usepackage{graphicx}",
        "\\usepackage{siunitx}",
        "\\geometry{margin=1in}",
        "\\sisetup{detect-all=true, separate-uncertainty=true}",
        f"\\setlength{{\\tabcolsep}}{{{main_tables.TABLE_COLUMN_SEP}}}",
        "",
        "\\newcommand{\\bcap}{BayesCAP}",
        "\\newcommand{\\rqbs}{RQBS}",
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
    _set_coloring(pargs.coloring)
    methods = _method_names()
    table_dir = _table_dir(pargs)
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(table_dir, exist_ok=True)
    os.makedirs(tabular_dir, exist_ok=True)

    document_tables = []
    for acc_name in get_acc_names():
        try:
            df = _prepare_results(acc_name, methods)
        except ValueError:
            continue
        if df.empty:
            continue

        name = f"{EXPERIMENT}_classic_connect-4_{acc_name}"
        table = _binform_latex_table(df, methods, pargs.show_std)
        tabular_filename = f"{name}.tex"
        with open(os.path.join(tabular_dir, tabular_filename), "w") as f:
            f.write(table)

        document_tables.append(
            (
                f"tabular/{tabular_filename}",
                f"{_escape_latex_text(acc_name)} results for the binform connect-4 experiment.",
                f"tab:{EXPERIMENT}-connect-4-{acc_name}",
            )
        )
        print(f"table {name} genned")

    if not document_tables:
        raise RuntimeError(f"No {EXPERIMENT} tables generated. Check that experiment results exist.")

    document_path = os.path.join(table_dir, f"{EXPERIMENT}_classic_connect-4.tex")
    with open(document_path, "w") as f:
        f.write(_latex_document(document_tables))


def gen_pdf(pargs):
    table_dir = _table_dir(pargs)
    tex_filename = f"{EXPERIMENT}_classic_connect-4.tex"
    tex_path = os.path.join(table_dir, tex_filename)

    if not os.path.exists(tex_path):
        raise FileNotFoundError(f"{tex_path} does not exist. Generate tables first.")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
        cwd=table_dir,
        check=True,
    )


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate LaTeX tables for the binform experiment.")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--no-std", action="store_false", dest="show_std", help="Show only means, without std")
    parser.add_argument(
        "--coloring",
        choices=(COLOR_BEST_TO_WORST, COLOR_BEST_TO_WHITE),
        default=COLOR_BEST_TO_WORST,
        help="Color gradient style for accuracy error and Winkler score",
    )
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
