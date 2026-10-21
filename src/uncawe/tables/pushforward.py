"""Population against realised pushforward of BayCAP on ``main`` (work-plan card WP-4, E2.3 of ``docs/ext2_baycap.md``).

``BayesCAP`` returns the population tables $C^{(r)}\\operatorname{diag}(\\pi^{(r)})$ of its posterior draws and targets
the population table; ``BayesCAP-real`` (``methods/baycap.py:BayCAP`` with ``estimand="realised"``) returns the
realised-table posterior predictive and targets the realised table of the target sample. Both are scored against the
realised table of each target bag (``true_cts``). Vanilla accuracy, $|U|=1000$, the pairs where both methods have
stored results.

Per (dataset, classifier) pair, the table reports coverage, amplitude and Winkler score of both pushforwards (means over
the bags), next to the coverage that E2.1 predicted for the realised pushforward with its approximation and the ratio
$\\hat\\sigma_g/\\hat\\sigma_p$, both read from the per-pair CSV of :mod:`uncawe.tables.estimand_gap`. As a check, it
reports the largest gap between the population coverage computed here and the one stored by E2.1, which is zero when
both read the same results.

Outputs, under ``<OUT_DIR>/uncawe/tables/pushforward/``: a per-pair CSV and one table as a LaTeX tabular fragment, a
longtable and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import acc_from_ct
from uncawe.main import EXPERIMENT
from uncawe.tables import estimand_gap
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import METHOD_LABELS, ordered
from uncawe.tables.estimand_gap import ACC_NAME, booktabs_clines, load_results
from uncawe.util import decorate_dataset

TABLE_NAME = "pushforward"
PUSHFORWARDS = {"population": "BayesCAP", "realised": "BayesCAP-real"}
SCORES = ("coverage", "amplitude", "winkler")
# columns read from the per-pair CSV of E2.1
APPROX_COLUMNS = {
    "n_val": "n_val",
    "ratio": "ratio",
    "predictive_coverage": "approx_coverage",
    "population_coverage": "e21_population_coverage",
}


def interval_scores(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Return, per (dataset, classifier), the mean coverage, amplitude and Winkler score of ``method`` over the bags."""
    rows = []
    for (dataset, classifier), _df in df.groupby(["dataset", "classifier"], sort=False):
        bags = []
        for row in _df.itertuples(index=False):
            true_acc = acc_from_ct(ACC_NAME, np.asarray(row.true_cts, dtype=float))
            samples = np.asarray(row.estim_acc_ranges, dtype=float)
            bags.append(
                {
                    "coverage": main_tables.get_acc_coverage(method, true_acc, samples),
                    "amplitude": main_tables.get_acc_amplitude(method, samples),
                    "winkler": main_tables.get_acc_winkler(method, true_acc, samples),
                }
            )
        bags = pd.DataFrame(bags)
        record = {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": int(_df["n_classes"].iloc[0]),
            "n_bags": len(bags),
        }
        record.update({score: float(bags[score].mean()) for score in SCORES})
        rows.append(record)
    return pd.DataFrame(rows)


def comparison_table(population: pd.DataFrame, realised: pd.DataFrame, approx: pd.DataFrame) -> pd.DataFrame:
    """Join the per-pair scores of the two pushforwards (pairs present in both) with the E2.1 columns.

    ``population`` and ``realised`` come from :func:`interval_scores`; ``approx`` is the per-pair CSV of E2.1.
    """
    keys = ["dataset", "classifier"]
    pop = population.rename(columns={score: f"population_{score}" for score in SCORES})
    real = realised.drop(columns="n_classes").rename(
        columns={score: f"realised_{score}" for score in SCORES} | {"n_bags": "realised_n_bags"}
    )
    table = pop.merge(real, on=keys, how="inner")
    approx = approx[keys + list(APPROX_COLUMNS)].rename(columns=APPROX_COLUMNS)
    table = table.merge(approx, on=keys, how="left")
    missing = table.loc[table["ratio"].isna(), keys]
    if not missing.empty:
        raise ValueError(f"Pairs missing from the E2.1 table: {list(missing.itertuples(index=False, name=None))}")
    table["e21_population_gap"] = (table["population_coverage"] - table["e21_population_coverage"]).abs()
    return ordered(table)


def approx_csv_path() -> str:
    return os.path.join(env.root_dir, "tables", estimand_gap.TABLE_NAME, f"{estimand_gap.TABLE_NAME}.csv")


def compute_comparison_table() -> pd.DataFrame:
    """Load the stored ``main`` results of both pushforwards and the E2.1 CSV, and return the ordered table."""
    scores = {}
    for pushforward, method in PUSHFORWARDS.items():
        df = load_results(method)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT}. Check that experiment results exist.")
        scores[pushforward] = interval_scores(df, method)
    path = approx_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Generate the E2.1 table first (tables.estimand_gap).")
    return comparison_table(scores["population"], scores["realised"], pd.read_csv(path))


CAPTION = (
    "Population (BayCAP) against realised (BayCAP-real) pushforward on \\texttt{main} (vanilla accuracy, "
    "$|U|=1000$). Coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ against the "
    "realised table, means over the 100 bags. E2.1: $\\hat\\sigma_g/\\hat\\sigma_p$, median over bags, and "
    "$\\mathcal{C}_{\\approx}$, coverage of the approximate predictive."
)
LABEL = "tab:pushforward"


def _styled_table(table: pd.DataFrame):
    """Return the Styler of the per-pair table; dataset-level descriptors are index levels, shown once per dataset."""
    cols = {}
    for pushforward, method in PUSHFORWARDS.items():
        for score in SCORES:
            label = (METHOD_LABELS[method], main_tables.metrics_map[score])
            cols[label] = table[f"{pushforward}_{score}"].to_numpy()
    cols[("E2.1", "$\\mathcal{C}_{\\approx}$")] = table["approx_coverage"].to_numpy()
    cols[("E2.1", "$\\hat\\sigma_g/\\hat\\sigma_p$")] = table["ratio"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = pd.MultiIndex.from_arrays(
        [
            table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d))),
            table["n_classes"].astype(int),
            table["classifier"].astype(str),
        ],
        names=["dataset", "$K$", "classifier"],
    )
    return out.style.format(precision=3)


def _latex_table(table: pd.DataFrame) -> str:
    return booktabs_clines(_styled_table(table).to_latex(hrules=True, multicol_align="c", clines="skip-last;data"))


def _latex_longtable(table: pd.DataFrame) -> str:
    latex = _styled_table(table).to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=CAPTION,
        label=LABEL,
    )
    return booktabs_clines(latex)


def _latex_document(longtable_path: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,hmargin=1.5cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{amsmath,booktabs,longtable,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\scriptsize",
            "\\setlength{\\LTcapwidth}{\\textwidth}",
            f"\\input{{{longtable_path}}}",
            "\\end{document}",
            "",
        ]
    )


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", TABLE_NAME)


def gen_tables(pargs):
    table = compute_comparison_table()
    table_dir = _table_dir(pargs)
    tabular_dir = os.path.join(table_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    table.to_csv(os.path.join(table_dir, f"{TABLE_NAME}.csv"), index=False)
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}.tex"), "w") as f:
        f.write(_latex_table(table))
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}_long.tex"), "w") as f:
        f.write(_latex_longtable(table))
    with open(os.path.join(table_dir, f"{TABLE_NAME}_doc.tex"), "w") as f:
        f.write(_latex_document(f"tabular/{TABLE_NAME}_long.tex"))

    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.precision", 3):
        print(table.drop(columns=["e21_population_coverage", "e21_population_gap"]).to_string(index=False))
    under = table["population_coverage"] < estimand_gap.UNDER_COVERAGE
    print(
        f"pairs: {len(table)}; under-covering with BayesCAP: {int(under.sum())}; of these, within "
        f"{list(estimand_gap.NOMINAL_BAND)} with BayesCAP-real: "
        f"{int((under & table['realised_coverage'].between(*estimand_gap.NOMINAL_BAND)).sum())}"
    )
    print(
        f"max |exact - approximate coverage|: {(table['realised_coverage'] - table['approx_coverage']).abs().max():.3f}; "
        f"max |population coverage - E2.1|: {table['e21_population_gap'].max():.2e}"
    )
    print(f"tables written to {table_dir}")


def gen_pdf(pargs):
    table_dir = _table_dir(pargs)
    tex_filename = f"{TABLE_NAME}_doc.tex"
    if not os.path.exists(os.path.join(table_dir, tex_filename)):
        raise FileNotFoundError(f"{tex_filename} does not exist in {table_dir}. Generate tables first.")

    # longtable needs a second pass to settle the column widths
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=table_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Population against realised pushforward of BayCAP on main (E2.3).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSV, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
