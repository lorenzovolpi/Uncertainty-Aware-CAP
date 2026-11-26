r"""Main-style table on the binary datasets only (user's request of 2026-11-26, part of the paper set,
``docs/paper.md``).

Columns, in this order: ``DoC``, ``QuAcc_NxN``, ``O-LEAP``, ``AC-LSB`` (label "AC") and ``L-CBPE-LSB`` (label "CBPE",
which runs only on the binary datasets) under "Baseline methods", then the default BayCAP-KDEy,
``BayCAP-KDEy-h-inst-cb-tcal-keep``, under "Our methods". Rows: the pairs of the ``uci_binary`` datasets of ``main``
(5 datasets, every classifier). The table is built by the code of the main tables (``tables/main.py``): per classifier
and dataset, the mean (and, without ``--no-std``, the standard deviation) over the bags of AE and of the Winkler score
at $\alpha=0.05$, the same colours and daggers, and the rows of wins, ties and losses against LEAP and of the average
rank. Every method is scored against the realised table of the bag (``true_cts``); the intervals of ``AC-LSB`` and
``L-CBPE-LSB`` are read from their stored probability mass function (``tables/main.py:PMF_INTERVALS``). Vanilla
accuracy only, since ``AC`` and ``CBPE`` estimate no other metric.

Output: ``<OUT_DIR>/uncawe/tables/binary/`` (``tabular/binary_classic_vanilla_accuracy.tex`` and the document
``binary_classic.tex``). ``make run-tables-binary ARGS="--tables --no-std"`` (the paper version) or
``make run-paper-table-binary`` (generated and copied into the paper folder).
"""

import os
import subprocess
from argparse import ArgumentParser

from uncawe import env
from uncawe.experiments.methods import AC_LSB_METHOD, L_CBPE_LSB_METHOD
from uncawe.main import EXPERIMENT
from uncawe.tables import main as main_tables

TABLE_NAME = "binary"
ACC_NAME = "vanilla_accuracy"
COLLECTIONS = ("uci_binary",)
METHODS = (
    "DoC",
    "QuAcc_NxN",
    "O-LEAP",
    AC_LSB_METHOD,
    L_CBPE_LSB_METHOD,
    "BayCAP-KDEy-h-inst-cb-tcal-keep",
)


def table_dir(path: str | None = None) -> str:
    return path if path is not None else os.path.join(env.root_dir, "tables", TABLE_NAME)


def tabular_name(domain: str) -> str:
    return f"{TABLE_NAME}_{domain}_{ACC_NAME}"


def table_latex(main_base_dir: str, domain: str, metrics=main_tables.METRICS, show_std: bool = True) -> str:
    classifiers = main_tables._ordered_classifiers(main_base_dir, domain, ACC_NAME)
    if not classifiers:
        raise RuntimeError(f"No results of {ACC_NAME} on {domain} in {main_base_dir}.")
    df, dataset_order = main_tables.load_table_frame(
        main_base_dir, domain, ACC_NAME, classifiers, methods=METHODS, collections=COLLECTIONS
    )
    if df.empty:
        raise RuntimeError(f"No results of the binary datasets on {domain} in {main_base_dir}.")
    return main_tables._results_latex_table(df, metrics, show_std, dataset_order, methods=list(METHODS))


def gen_tables(pargs):
    out = table_dir(pargs.table_dir)
    tabular_dir = os.path.join(out, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    name = tabular_name(pargs.domain)
    tbl = table_latex(
        os.path.join(env.root_dir, EXPERIMENT), pargs.domain, main_tables._selected_metrics(pargs), pargs.show_std
    )
    with open(os.path.join(tabular_dir, f"{name}.tex"), "w") as f:
        f.write(tbl)
    caption = f"{main_tables._escape_latex_text(ACC_NAME)} on the binary {pargs.domain} datasets."
    document = main_tables._results_latex_document(
        [(f"tabular/{name}.tex", caption, f"tab:{TABLE_NAME}-{pargs.domain}-{ACC_NAME}")]
    )
    with open(os.path.join(out, f"{TABLE_NAME}_{pargs.domain}.tex"), "w") as f:
        f.write(document)
    print(f"table {name} genned in {out}")


def gen_pdf(pargs):
    out = table_dir(pargs.table_dir)
    tex_filename = f"{TABLE_NAME}_{pargs.domain}.tex"
    if not os.path.exists(os.path.join(out, tex_filename)):
        raise FileNotFoundError(f"{os.path.join(out, tex_filename)} does not exist. Generate tables first.")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename], cwd=out, check=True)


def parse_args():
    parser = ArgumentParser(description="Main-style table on the binary datasets (DoC, QuAcc, LEAP, AC, CBPE, BayCAP).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    parser.add_argument("--metrics", nargs="+", choices=main_tables.METRICS, default=None)
    parser.add_argument("--no-std", action="store_false", dest="show_std", help="Show only means, without std")
    parser.add_argument("--table-dir", default=None, help="Directory of the table document and tabular fragment")
    pargs = parser.parse_args()
    if pargs.domain is None:
        raise ValueError("Please specify a domain.")
    return pargs


def main():
    pargs = parse_args()
    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)


if __name__ == "__main__":
    main()
