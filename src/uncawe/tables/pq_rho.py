"""The allocation $\\rho$ of SQS-S on the pairs of ``main`` (E6.2 of ``docs/ext6_sqs_s_design.md``, card WP-17).

Inputs: the stored ``main`` results of the five methods ``PQ-rho<ρ>`` (``experiments/methods.py:PQ_RHO_VARIANTS``):
SQS-S as ``PQ`` (PACC(MLP), $n=10{,}000$ candidates, $\\gamma=0.1$, $L_1$, $m=1000$), with the nested split of
``PrediQuant`` at the share $\\rho$ of $V$ that trains the quantifier. Every score is computed against the realised table
of each target bag (``true_cts``); the true prevalence of the bag is read from it, so the quantification error is an
analysis, not part of the method.

Measures, per target bag, level and metric (vanilla accuracy, macro-F1, macro-K):

- ``quant_error``: $\\frac1K\\sum_j\\lvert\\hat\\pi^Q_j-\\pi^Q_j\\rvert$, from the stored ``pq_test_prev`` and the row sums of
  ``true_cts`` (true classes on rows); the same for every metric;
- ``acc_err``: the absolute error of the point estimate (stored);
- ``coverage``, ``amplitude``, ``winkler``: of the $95\\%$ interval, from the stored samples, with the functions of the
  main tables (``tables/main.py``);
- ``class_support``: the minimum over the classes of the class support of the selected bags (``pq_min_class_support``);
  the same for every metric.

Per (pair, level, metric), the means over the bags. A (pair, level) with no stored result (e.g. a failure of the fit at
$\\rho=0.1$, where a class has too few instances in $V_q$ for the $5$-fold cross-validation of PACC, or a class emptied
by the split) is a failure: the failures are listed in a CSV and counted, and the summary uses only the pairs that have
every level (the common pairs), so that every column is a mean over the same pairs. The pairs considered are those with
at least one level stored.

Summary table: for each metric, one row per measure and one column per level, with the mean over the common pairs; a
cell is marked when the per-pair means at that level differ from those at $\\rho=0.5$ under a two-sided paired Wilcoxon
signed-rank test with Holm-adjusted $p<0.05$, the family being the $4\\times6$ tests of the metric. No decision rule
(E6.2 is an analysis only).

Outputs, under ``<OUT_DIR>/uncawe/tables/pq_rho/``: the per-pair CSV (``pq_rho_pairs.csv``, read by
``plots/pq_rho.py``), the failures (``pq_rho_failures.csv``), the summary as CSV (means and Holm-adjusted $p$), a LaTeX
tabular fragment and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncawe import env
from uncawe.experiments.methods import PQ_KDEY_RHO_VARIANTS, PQ_RHO_VARIANTS
from uncawe.experiments.metrics import acc_from_ct, get_acc_names
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables import main as main_tables
from uncawe.tables.estimand_gap import booktabs_clines
from uncawe.tables.pushforward import holm

TABLE_NAME = "pq_rho"
DOMAIN = "classic"
KEYS = ["dataset", "classifier"]
LEVELS = dict(PQ_RHO_VARIANTS)
LEVEL_VALUES = tuple(LEVELS.values())
# the quantifier of SQS-S (card WP-36): key -> (methods name -> rho, label of the captions, subfolder of the outputs);
# PACC(MLP), the methods of E6.2, or KDEyML(MLP), PQ-KDEy at the same levels, under kdey/
QUANTIFIERS = {
    "pacc": (LEVELS, "PACC(MLP)", None),
    "kdey": (dict(PQ_KDEY_RHO_VARIANTS), "KDEyML(MLP)", "kdey"),
}
# KDEy by default since 2026-11-26 (user's decision): the SQS-S of the main tables (PQ-KDEy, WP-36); the tables of E6.2
# with PACC (the stored PQ-rho*) with --quantifier pacc, in the folder of E6.2
DEFAULT_QUANTIFIER = "kdey"
# every stored method of the two sets -> rho
ALL_LEVELS = {**LEVELS, **PQ_KDEY_RHO_VARIANTS}
assert tuple(QUANTIFIERS["kdey"][0].values()) == LEVEL_VALUES
REFERENCE_RHO = 0.5
# measure column -> label
MEASURES = {
    "quant_error": "Quant.\\ error",
    "acc_err": "AE",
    "coverage": "Coverage",
    "amplitude": "Amplitude",
    "winkler": "Winkler",
    "class_support": "Class support",
}
METRIC_LABELS = {"vanilla_accuracy": "vanilla accuracy", "macro-F1": "macro-$F_1$", "macro-K": "macro-$K$"}
ALPHA = 0.05
MARK = "$^{*}$"


# --------------------------------------------------------------------------------------------------- loading


def load_results(
    acc_name: str, domain: str = DOMAIN, root: str | None = None, quantifier: str = DEFAULT_QUANTIFIER
) -> pd.DataFrame:
    """The stored ``main`` rows of the ``PQ-rho*`` methods (``PQ-KDEy-rho*`` with ``quantifier="kdey"``, WP-36) for
    ``acc_name``."""
    root = env.root_dir if root is None else root
    try:
        results = Results.load(
            base_dir=os.path.join(root, EXPERIMENT),
            domain=domain,
            acc_name=acc_name,
            filter_methods=list(QUANTIFIERS[quantifier][0]),
        )
    except ValueError:  # no file to concatenate
        return pd.DataFrame()
    return results.df.reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- computation


def quant_error(test_prev, true_ct) -> float:
    """$\\frac1K\\sum_j\\lvert\\hat\\pi^Q_j-\\pi^Q_j\\rvert$, with $\\pi^Q$ the row sums of ``true_ct`` (true classes on rows),
    normalised."""
    true_ct = np.asarray(true_ct, dtype=float)
    true_prev = true_ct.sum(axis=1) / true_ct.sum()
    return float(np.abs(np.asarray(test_prev, dtype=float) - true_prev).mean())


def bag_scores(df: pd.DataFrame, acc_name: str) -> pd.DataFrame:
    """One row per stored bag of ``df`` (one metric) with the level $\\rho$ and the six measures."""
    rows = []
    for (dataset, classifier, method), _df in df.groupby(KEYS + ["method"], sort=False):
        for bag, row in enumerate(_df.itertuples(index=False)):
            true_ct = np.asarray(row.true_cts, dtype=float)
            true_acc = acc_from_ct(acc_name, true_ct)
            samples = np.asarray(row.estim_acc_ranges, dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": int(row.n_classes),
                    "rho": ALL_LEVELS[method],
                    "bag": bag,
                    "quant_error": quant_error(row.pq_test_prev, true_ct),
                    "acc_err": float(row.acc_err),
                    "coverage": main_tables.get_acc_coverage(method, true_acc, samples),
                    "amplitude": main_tables.get_acc_amplitude(method, samples),
                    "winkler": main_tables.get_acc_winkler(method, true_acc, samples),
                    "class_support": float(row.pq_min_class_support),
                }
            )
    return pd.DataFrame(rows)


def pair_means(bags: pd.DataFrame) -> pd.DataFrame:
    """Per (pair, level): the means of the measures over the bags, with the number of bags."""
    grouped = bags.groupby(KEYS + ["rho"], as_index=False)
    table = grouped.agg(n_classes=("n_classes", "first"), n_bags=("bag", "size"), **{m: (m, "mean") for m in MEASURES})
    return table.sort_values(KEYS + ["rho"]).reset_index(drop=True)


def failures(pairs: pd.DataFrame, levels=LEVEL_VALUES) -> pd.DataFrame:
    """The (pair, level) with no stored result, among the pairs with at least one level."""
    stored = set(map(tuple, pairs[KEYS + ["rho"]].to_numpy()))
    missing = [
        (dataset, classifier, rho)
        for dataset, classifier in pairs[KEYS].drop_duplicates().itertuples(index=False)
        for rho in levels
        if (dataset, classifier, rho) not in stored
    ]
    return pd.DataFrame(missing, columns=KEYS + ["rho"])


def mark_common(pairs: pd.DataFrame, levels=LEVEL_VALUES) -> pd.DataFrame:
    """Add ``common``: the pair has every level."""
    counts = pairs.groupby(KEYS)["rho"].transform("nunique")
    return pairs.assign(common=counts == len(levels))


def summary_table(pairs: pd.DataFrame, levels=LEVEL_VALUES) -> pd.DataFrame:
    """One row per measure: the means over the common pairs at each level, and, for each level other than $0.5$, the
    two-sided paired Wilcoxon $p$ against $0.5$, Holm-adjusted over the measures and levels (``p_holm_<rho>``)."""
    common = pairs.loc[pairs["common"]]
    rows, p_values = [], []
    for measure in MEASURES:
        wide = common.pivot(index=KEYS, columns="rho", values=measure)
        row = {"measure": measure, "n_pairs": len(wide)}
        for rho in levels:
            row[f"mean_{rho}"] = float(wide[rho].mean()) if rho in wide else np.nan
            if rho == REFERENCE_RHO:
                continue
            diff = (wide[rho] - wide[REFERENCE_RHO]).dropna() if rho in wide and REFERENCE_RHO in wide else []
            if len(diff) == 0:
                p = np.nan
            elif (diff != 0).any():
                p = float(wilcoxon(diff).pvalue)
            else:
                p = 1.0
            row[f"p_{rho}"] = p
            p_values.append((len(rows), f"p_holm_{rho}", p))
        rows.append(row)
    table = pd.DataFrame(rows)
    adjusted = holm(np.array([p for _, _, p in p_values]))
    for (i, column, _), value in zip(p_values, adjusted):
        table.loc[i, column] = value
    return table


def compute(acc_name: str, df: pd.DataFrame):
    """The per-pair means (with ``common``), the failures and the summary of one metric."""
    pairs = mark_common(pair_means(bag_scores(df, acc_name)))
    return pairs, failures(pairs), summary_table(pairs)


# --------------------------------------------------------------------------------------------------- LaTeX


def _cell(row: pd.Series, rho: float) -> str:
    text = f"{row[f'mean_{rho}']:.4f}"
    p = row.get(f"p_holm_{rho}", np.nan)
    return text + MARK if pd.notna(p) and p < ALPHA else text


def summary_latex(summaries: dict[str, pd.DataFrame], levels=LEVEL_VALUES) -> str:
    rows = []
    for acc_name, summary in summaries.items():
        for _, row in summary.iterrows():
            cells = {f"$\\rho={rho}$": _cell(row, rho) for rho in levels}
            rows.append({"Metric": METRIC_LABELS.get(acc_name, acc_name), "Measure": MEASURES[row["measure"]]} | cells)
    formatted = pd.DataFrame(rows).set_index(["Metric", "Measure"])
    latex = formatted.style.to_latex(
        hrules=True, clines="skip-last;data", multirow_align="naive", column_format="ll" + "r" * len(levels)
    )
    return booktabs_clines(latex)


def caption(summaries: dict[str, pd.DataFrame], failed: int, quantifier: str = DEFAULT_QUANTIFIER) -> str:
    n_pairs = sorted({int(s["n_pairs"].iloc[0]) for s in summaries.values() if not s.empty})
    pairs = "/".join(str(n) for n in n_pairs)
    return (
        f"SQS-S (PrediQuant with {QUANTIFIERS[quantifier][1]}, $n=10{{,}}000$ candidates, $\\gamma=0.1$, $L_1$, $m=1000$) on the pairs of "
        "\\texttt{main} at five allocations $\\rho=|V_q|/|V|$ with nested splits (E6.2), scored against the realised "
        f"table of each target bag ($|U|=1000$). Means over the {pairs} pairs that have every level of the per-pair "
        "means over the $100$ bags: quantification error $\\frac1K\\sum_j|\\hat\\pi^Q_j-\\pi^Q_j|$, absolute error of "
        "the point estimate (AE), coverage, amplitude and Winkler score of the $95\\%$ interval, and the minimum over "
        "the classes of the class support of the selected bags. "
        f"{MARK}: the per-pair means differ from those at $\\rho=0.5$ (two-sided paired Wilcoxon signed-rank test, "
        f"Holm-adjusted over the $24$ tests of the metric, $p<{ALPHA}$). Failed (pair, level): {failed}."
    )


def _latex_document(tabular_path: str, caption_text: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,hmargin=1.5cm,vmargin=1.5cm]{geometry}",
            "\\usepackage{amsmath,booktabs,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\begin{table}[h]",
            "\\centering\\small",
            f"\\caption{{{caption_text}}}",
            f"\\input{{{tabular_path}}}",
            "\\end{table}",
            "\\end{document}",
            "",
        ]
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and getattr(pargs, "table_dir", None) is not None:
        return pargs.table_dir
    sub = QUANTIFIERS[getattr(pargs, "quantifier", DEFAULT_QUANTIFIER)][2]
    return os.path.join(env.root_dir, "tables", TABLE_NAME, *([sub] if sub else []))


def pairs_csv_path(out_dir: str | None = None) -> str:
    return os.path.join(out_dir or table_dir(), f"{TABLE_NAME}_pairs.csv")


def gen_tables(pargs):
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    all_pairs, all_failures, summaries = [], [], {}
    quantifier = getattr(pargs, "quantifier", DEFAULT_QUANTIFIER)
    levels = QUANTIFIERS[quantifier][0]
    for acc_name in pargs.metrics:
        df = load_results(acc_name, pargs.domain, quantifier=quantifier)
        if df.empty:
            print(f"{acc_name}: no stored results of {sorted(levels)}")
            continue
        pairs, failed, summary = compute(acc_name, df)
        all_pairs.append(pairs.assign(acc_name=acc_name))
        all_failures.append(failed.assign(acc_name=acc_name))
        summaries[acc_name] = summary
    if not summaries:
        raise RuntimeError(f"No {EXPERIMENT} results of {sorted(levels)} found.")

    pairs = pd.concat(all_pairs, ignore_index=True)
    failed = pd.concat(all_failures, ignore_index=True)
    pairs.to_csv(pairs_csv_path(out_dir), index=False)
    failed.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_failures.csv"), index=False)
    pd.concat([s.assign(acc_name=a) for a, s in summaries.items()], ignore_index=True).to_csv(
        os.path.join(out_dir, f"{TABLE_NAME}_summary.csv"), index=False
    )
    n_failed = len(failed.loc[failed["acc_name"] == next(iter(summaries)), KEYS + ["rho"]])
    with open(os.path.join(tabular_dir, f"{TABLE_NAME}_summary.tex"), "w") as f:
        f.write(summary_latex(summaries))
    with open(os.path.join(out_dir, f"{TABLE_NAME}_summary_doc.tex"), "w") as f:
        f.write(_latex_document(f"tabular/{TABLE_NAME}_summary.tex", caption(summaries, n_failed, quantifier)))

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        for acc_name, summary in summaries.items():
            print(f"\n{acc_name}:\n{summary.to_string(index=False)}")
        n_pairs = pairs[KEYS].drop_duplicates().shape[0]
        print(f"\npairs: {n_pairs}; failed (pair, level), {next(iter(summaries))}: {n_failed}")
        if n_failed:
            print(failed.loc[failed["acc_name"] == next(iter(summaries))].to_string(index=False))
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    tex_filename = f"{TABLE_NAME}_summary_doc.tex"
    if not os.path.exists(os.path.join(out_dir, tex_filename)):
        raise FileNotFoundError(f"{tex_filename} does not exist in {out_dir}. Generate tables first.")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
        cwd=out_dir,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def parse_args(argv=None):
    parser = ArgumentParser(description="Tables of the allocation rho of SQS-S on the pairs of main (E6.2).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--domain", default=DOMAIN)
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=get_acc_names())
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSVs, table document and tabular fragment are saved",
    )
    # WP-36: SQS-S with KDEyML(MLP) (PQ-KDEy-rho*, outputs under kdey/) instead of PACC(MLP) (E6.2)
    parser.add_argument("--quantifier", choices=list(QUANTIFIERS), default=DEFAULT_QUANTIFIER)
    return parser.parse_args(argv)


if __name__ == "__main__":
    pargs = parse_args()
    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
