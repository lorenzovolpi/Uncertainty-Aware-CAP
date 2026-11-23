"""Coverage and amplitude of the main-table methods, per classifier and dataset (decided on 2026-11-18).

The table has the structure of the main table (``tables/main.py``): one row per classifier and dataset, one column
group per score and, within it, the baselines and the other methods of ``MAIN_METHODS`` except BayesCAP (``MAIN_METHODS_BEFORE_BAYESCAP``). Each cell is the mean over
the bags of the pair of:

- the coverage $\\mathcal C$: the share of bags whose interval contains the accuracy of the realised table of the
  bag (``true_cts``);
- the amplitude $\\mathcal A$: the width of the interval.

Both target the realised table of the target bag, as in ``main``.

Colour and bold, per row and score (the colour scale of the rank table, ``tables/rank.py``: from ``BEST_RGB`` to
white):

- coverage: the distance $d=\\lvert\\bar c-0.95\\rvert$ from the nominal level, the same for over- and
  under-coverage; the colour fades linearly with $d$ on a fixed scale and is white from ``COVERAGE_D_MAX`` on; the
  cell with the smallest $d$ is in bold;
- amplitude: lower is better; the colour follows the rank of the method within the row (ties get the average rank),
  white from rank ``rank.WHITE_RANK`` on; the narrowest cell is in bold.

There is no significance test, no dagger and no standard deviation.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import CONFIDENCE_LEVEL, get_acc_names
from uncawe.main import EXPERIMENT
from uncawe.tables import main as main_tables
from uncawe.tables import rank

TABLE_NAME = "coverage_amplitude"
METRICS = ("coverage", "amplitude")
# the distance from the nominal coverage at which the colour of a coverage cell becomes white
COVERAGE_D_MAX = 0.20


def coverage_distance(values: pd.Series) -> pd.Series:
    """$\\lvert\\bar c-0.95\\rvert$ for each mean coverage of a row."""
    return (pd.to_numeric(values, errors="coerce") - CONFIDENCE_LEVEL).abs()


def coverage_color(distance: float) -> str:
    """From ``rank.BEST_RGB`` at distance 0 to white at ``COVERAGE_D_MAX`` and beyond, linearly."""
    if pd.isna(distance):
        return ""
    # rank._rank_color maps 1 to BEST_RGB and rank.WHITE_RANK to white
    return rank._rank_color(1.0 + min(distance / COVERAGE_D_MAX, 1.0) * (rank.WHITE_RANK - 1.0))


def amplitude_ranks(values: pd.Series) -> pd.Series:
    """Rank of each mean amplitude within a row (1 = narrowest; ties get the average rank)."""
    return pd.to_numeric(values, errors="coerce").rank(method="average", ascending=True)


def _style(color: str) -> str:
    return f"background-color: {color}; color: {main_tables._text_color(color)};" if color else ""


def cell_styles(mean: pd.DataFrame) -> pd.DataFrame:
    """Colour and bold of each cell of ``mean`` (rows: classifier and dataset; columns: (method, metric))."""
    styles = pd.DataFrame("", index=mean.index, columns=mean.columns)

    for metric in METRICS:
        if metric not in mean.columns.get_level_values("metric"):
            continue
        metric_values = mean.xs(metric, axis=1, level="metric")

        for row_key, values in metric_values.iterrows():
            if metric == "coverage":
                keys = coverage_distance(values)
                colors = keys.apply(coverage_color)
            else:
                keys = pd.to_numeric(values, errors="coerce")
                colors = amplitude_ranks(values).apply(rank._rank_color)

            for method, color in colors.items():
                styles.loc[row_key, (method, metric)] = _style(color)

            best = keys.min(skipna=True)
            if pd.isna(best):
                continue
            for method in keys.index[np.isclose(keys, best)]:
                styles.loc[row_key, (method, metric)] += " font-weight: bold;"

    return styles


def latex_table(df: pd.DataFrame, dataset_order: list[str] | None = None) -> str:
    """The LaTeX tabular of one accuracy measure from the per-bag scores of ``main_tables.load_table_frame``."""
    row_levels = main_tables._row_levels(df)
    mean = df.groupby(row_levels + ["method"], sort=False)[list(METRICS)].mean().unstack("method")
    mean = mean.swaplevel(0, 1, axis=1)
    mean.columns = mean.columns.set_names(["method", "metric"])

    # the methods of the main table without BayesCAP, shown there since 2026-11-23 (user's decision of 2026-11-23)
    methods = [
        m
        for m in main_tables.table_method_names()
        if m in main_tables.MAIN_METHODS_BEFORE_BAYESCAP and m in mean.columns.get_level_values("method")
    ]
    mean = mean.reindex(
        index=main_tables._ordered_row_index(mean.index, dataset_order),
        columns=main_tables._ordered_columns(methods, METRICS),
    )
    table = main_tables._format_mean_std(mean, mean, show_std=False)
    styles = cell_styles(mean)
    return main_tables._render_latex_table(table, styles, methods, METRICS, row_levels)


def _caption(acc: str, domain: str) -> str:
    return (
        f"Coverage $\\mathcal{{C}}$ and amplitude $\\mathcal{{A}}$ ({main_tables._escape_latex_text(acc)}, "
        f"{main_tables._escape_latex_text(domain)} datasets), mean over the bags, against the realised table of each "
        f"bag. Coverage: the colour fades with the distance from the nominal level ${CONFIDENCE_LEVEL:g}$, white "
        f"from ${COVERAGE_D_MAX:g}$ on, and the closest is in bold. Amplitude: the colour follows the rank within "
        "the row, and the narrowest is in bold."
    )


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def _document_name(domain: str) -> str:
    return f"{EXPERIMENT}_{domain}_{TABLE_NAME}"


def gen_tables(pargs):
    domain = pargs.domain
    main_base_dir = os.path.join(env.root_dir, "main")
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    document_tables = []
    for acc in get_acc_names():
        classifiers = main_tables._ordered_classifiers(main_base_dir, domain, acc)
        if not classifiers:
            continue

        df, dataset_order = main_tables.load_table_frame(main_base_dir, domain, acc, classifiers)
        name = f"{EXPERIMENT}_{domain}_{acc}_{TABLE_NAME}"
        with open(os.path.join(tabular_dir, f"{name}.tex"), "w") as f:
            f.write(latex_table(df, dataset_order))
        document_tables.append((f"tabular/{name}.tex", _caption(acc, domain), f"tab:{name}"))
        print(f"table {name} genned")

    if not document_tables:
        raise RuntimeError(f"No tables generated for domain={domain!r}.")

    with open(os.path.join(out_dir, f"{_document_name(domain)}.tex"), "w") as f:
        f.write(main_tables._results_latex_document(document_tables))


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    tex_filename = f"{_document_name(pargs.domain)}.tex"
    if not os.path.exists(os.path.join(out_dir, tex_filename)):
        raise FileNotFoundError(f"{os.path.join(out_dir, tex_filename)} does not exist. Generate tables first.")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename], cwd=out_dir, check=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    parser.add_argument("--table-dir", dest="table_dir", default=None, help="Directory of the documents and tabulars")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")
    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
