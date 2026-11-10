"""The summary tables of the distance diagnostic ``predist`` (E6.1 of ``docs/ext6_sqs_s_design.md``, card WP-16).

Inputs: the stored ``predist`` results (``experiments/predist.py``) with ``schema_version`` at least
``RESULT_SCHEMA_VERSION`` (PACC(MLP) as the quantifier of PrediQuant), one file per (dataset, classifier) pair, one row
per (distance, test bag). Vanilla accuracy by default. Every score is computed against the realised table of the test
bag (``true_acc``).

Measures, per test bag and distance: the absolute error of the point estimate (``acc_abs_error``), coverage, amplitude
and Winkler score of the $95\\%$ interval, and the mean $L_1$ distance between the true prevalences of the selected
bags and the true prevalence of the test bag (``true_dist_l1_mean``). Two diagnostics of the selection, computed here
from the stored prevalences and not used by the decision rule: the $L_1$ error of the estimated prevalence of the test
bag (``test_prev_l1_error``, the same for every distance, since the fit and the quantifier are shared) and the mean $L_1$
distance between the estimated prevalences of the selected bags and the estimated prevalence of the test bag
(``pred_dist_l1_mean``, the size of the neighbourhood in the space where the selection is made).

Tables (``--tables-set``, default both):

- ``summary``: one row per distance, with the means over all the test bags of the five measures; the number of pairs
  where the distance has the lowest per-pair mean Winkler score (ties count for every tied distance); the number of
  pairs where its per-pair mean Winkler score is lower than that of $L_1$; the $p$-value of the two-sided paired
  Wilcoxon signed-rank test over the per-pair mean Winkler scores against $L_1$;
- ``region``: the same means split by the position of the test bag on the simplex (centre, vertices, random point).

Both tables also show the two diagnostics, not in bold.

The decision rule of E6.1 is evaluated on the summary and printed (``decision``): $L_1$ is kept unless another distance
has a lower mean Winkler score over all the test bags, a lower per-pair mean Winkler score than $L_1$ on a majority of
the pairs, and $p<0.05$; if several distances qualify, the one with the lowest mean Winkler score is chosen.

Outputs, under ``<OUT_DIR>/uncawe/tables/predist/``: a per-pair CSV (``predist_pairs.csv``), a per-pair and per-region
CSV (``predist_pair_region.csv``), the two tables as CSV, and for each table a LaTeX tabular fragment and a standalone
document.
"""

import os
import subprocess
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncawe import env
from uncawe.experiments.predist_config import (
    DEFAULT_ACC_NAME,
    DEFAULT_DOMAIN,
    EXPERIMENT,
    METHOD_NAME,
    RESULT_SCHEMA_VERSION,
)
from uncawe.tables.estimand_gap import booktabs_clines

TABLE_NAME = "predist"
KEYS = ["dataset", "classifier"]
REFERENCE = "l1"
DISTANCE_ORDER = ("l1", "hellinger", "jensen-shannon")
REGION_ORDER = ("centre", "vertex", "random")
# measure column -> (label, lower is better)
MEASURES = {
    "acc_abs_error": ("AE", True),
    "ci_coverage": ("Coverage", False),
    "ci_amplitude": ("Amplitude", True),
    "ci_winkler": ("Winkler", True),
    "true_dist_l1_mean": ("True-space $L_1$", True),
}
# diagnostics of the selection, shown but not bold and not used by the decision rule
DIAGNOSTICS = {
    "test_prev_l1_error": "Test-prev.\\ error",
    "pred_dist_l1_mean": "Estimated-space $L_1$",
}
COLUMNS = list(MEASURES) + list(DIAGNOSTICS)
DISTANCE_LABELS = {"l1": "$L_1$", "hellinger": "Hellinger", "jensen-shannon": "Jensen--Shannon"}
REGION_LABELS = {"centre": "centre", "vertex": "vertices", "random": "random point"}
ALPHA = 0.05
TABLE_SETS = ("summary", "region")


# --------------------------------------------------------------------------------------------------- loading


def load_results(domain: str = DEFAULT_DOMAIN, acc_name: str = DEFAULT_ACC_NAME, root: str | None = None):
    """The stored ``predist`` rows of ``domain`` and ``acc_name`` with ``schema_version >= RESULT_SCHEMA_VERSION``."""
    root = env.root_dir if root is None else root
    paths = sorted(glob(os.path.join(root, EXPERIMENT, domain, acc_name, "*", "*", f"{METHOD_NAME}.parquet")))
    frames = [pd.read_parquet(path) for path in paths]
    frames = [df for df in frames if "schema_version" in df.columns]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df.loc[df["schema_version"] >= RESULT_SCHEMA_VERSION].reset_index(drop=True)


def _ordered(values, order) -> list:
    present = list(dict.fromkeys(values))
    return [v for v in order if v in present] + sorted(v for v in present if v not in order)


# --------------------------------------------------------------------------------------------------- computation


def _l1_rows(prevs, targets) -> np.ndarray:
    """Per row, the mean $L_1$ distance between the prevalences in ``prevs`` (one vector or a list of vectors) and the
    vector in ``targets``."""
    out = []
    for prev, target in zip(prevs, targets):
        prev = np.atleast_2d(np.stack([np.asarray(p, dtype=float) for p in prev]))
        out.append(float(np.abs(prev - np.asarray(target, dtype=float)).sum(axis=1).mean()))
    return np.array(out)


def add_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``test_prev_l1_error`` and ``pred_dist_l1_mean`` from the stored prevalences."""
    df = df.copy()
    df["test_prev_l1_error"] = _l1_rows(df["predicted_test_prev"], df["actual_test_prev"])
    df["pred_dist_l1_mean"] = _l1_rows(df["selected_val_prevs"], df["predicted_test_prev"])
    return df


def complete_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """The rows of the pairs that have every distance on every test bag of the pair, so that the distances are compared
    on the same bags."""
    n_distances = df["distance"].nunique()
    counts = df.groupby(KEYS + ["sample_id"])["distance"].nunique()
    full_bags = counts[counts == n_distances].reset_index()[KEYS + ["sample_id"]]
    per_pair = df.groupby(KEYS)["sample_id"].nunique().rename("n_bags").reset_index()
    full_per_pair = full_bags.groupby(KEYS)["sample_id"].nunique().rename("n_full").reset_index()
    pairs = per_pair.merge(full_per_pair, on=KEYS, how="left")
    pairs = pairs.loc[pairs["n_bags"] == pairs["n_full"], KEYS]
    return df.merge(pairs, on=KEYS).reset_index(drop=True)


def pair_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, classifier, distance): the means of the measures over the test bags of the pair."""
    return df.groupby(KEYS + ["distance"], as_index=False)[COLUMNS].mean()


def pair_region_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, classifier, region, distance): the means over the test bags of that pair and region."""
    return df.groupby(KEYS + ["region", "distance"], as_index=False)[COLUMNS].mean()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distance: means over all test bags, best-pair counts, pairs better than $L_1$, Wilcoxon $p$."""
    pairs = pair_means(df)
    winkler = pairs.pivot(index=KEYS, columns="distance", values="ci_winkler")
    best = winkler.eq(winkler.min(axis=1), axis=0)
    rows = []
    for distance in _ordered(df["distance"], DISTANCE_ORDER):
        row = {"distance": distance, "n_bags": int((df["distance"] == distance).sum())}
        row |= {m: float(df.loc[df["distance"] == distance, m].mean()) for m in COLUMNS}
        row["n_pairs"] = int(winkler[distance].notna().sum())
        row["n_best"] = int(best[distance].sum())
        if distance == REFERENCE or REFERENCE not in winkler:
            row["n_better_than_l1"] = np.nan
            row["p_wilcoxon"] = np.nan
        else:
            diff = (winkler[distance] - winkler[REFERENCE]).dropna()
            row["n_better_than_l1"] = int((diff < 0).sum())
            row["p_wilcoxon"] = float(wilcoxon(diff).pvalue) if (diff != 0).any() else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def region_table(df: pd.DataFrame) -> pd.DataFrame:
    """The means of the measures per (region, distance), over the test bags of that region."""
    table = df.groupby(["region", "distance"], as_index=False).agg(
        n_bags=("ci_winkler", "size"), **{m: (m, "mean") for m in COLUMNS}
    )
    region_rank = {r: i for i, r in enumerate(_ordered(table["region"], REGION_ORDER))}
    distance_rank = {d: i for i, d in enumerate(_ordered(table["distance"], DISTANCE_ORDER))}
    table = table.sort_values(
        ["region", "distance"], key=lambda s: s.map(region_rank if s.name == "region" else distance_rank)
    )
    return table.reset_index(drop=True)


def decision(summary: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """The decision rule of E6.1 on ``summary``: the distances that satisfy its three conditions against $L_1$, and the
    chosen distance ($L_1$ if none does; the one with the lowest mean Winkler score if several do)."""
    reference = summary.loc[summary["distance"] == REFERENCE].iloc[0]
    checks = {}
    for row in summary.loc[summary["distance"] != REFERENCE].itertuples(index=False):
        checks[row.distance] = {
            "lower_mean_winkler": bool(row.ci_winkler < reference.ci_winkler),
            "majority_of_pairs": bool(row.n_better_than_l1 > row.n_pairs / 2),
            "significant": bool(row.p_wilcoxon < alpha),
        }
    qualified = [d for d, c in checks.items() if all(c.values())]
    if qualified:
        winkler = summary.set_index("distance")["ci_winkler"]
        chosen = min(qualified, key=lambda d: winkler[d])
    else:
        chosen = REFERENCE
    return {"checks": checks, "qualified": qualified, "chosen": chosen}


def compute(df: pd.DataFrame):
    df = complete_pairs(df)
    if df.empty:
        raise RuntimeError("No predist pair has every distance on every test bag.")
    df = add_diagnostics(df)
    return pair_means(df), pair_region_means(df), summary_table(df), region_table(df)


# --------------------------------------------------------------------------------------------------- LaTeX


def _bold_best(frame: pd.DataFrame, columns, groups=None) -> pd.DataFrame:
    """Formatted strings, with the best value of each measure in bold (within each group of rows, if given)."""
    out = frame.copy().astype(object)
    groups = [frame.index] if groups is None else groups
    for column in columns:
        lower = MEASURES[column][1]
        for index in groups:
            values = frame.loc[index, column].astype(float)
            target = values.min() if lower else values.max()
            for i in index:
                text = f"{frame.at[i, column]:.4f}"
                out.at[i, column] = f"\\textbf{{{text}}}" if np.isclose(frame.at[i, column], target) else text
    return out


def _format_diagnostics(frame: pd.DataFrame, formatted: pd.DataFrame) -> pd.DataFrame:
    for column in DIAGNOSTICS:
        formatted[column] = frame[column].map(lambda v: f"{v:.4f}")
    return formatted


def summary_latex(summary: pd.DataFrame) -> str:
    formatted = _bold_best(summary, list(MEASURES))
    formatted["distance"] = summary["distance"].map(lambda d: DISTANCE_LABELS.get(d, d))
    formatted["n_better_than_l1"] = summary["n_better_than_l1"].map(lambda v: "--" if pd.isna(v) else f"{int(v)}")
    formatted["p_wilcoxon"] = summary["p_wilcoxon"].map(lambda v: "--" if pd.isna(v) else f"{v:.3g}")
    formatted = _format_diagnostics(summary, formatted)
    columns = {"distance": "Distance"} | {m: label for m, (label, _) in MEASURES.items()} | DIAGNOSTICS
    columns |= {"n_best": "Best pairs", "n_better_than_l1": "Pairs $<L_1$", "p_wilcoxon": "$p$ vs $L_1$"}
    formatted = formatted[list(columns)].rename(columns=columns)
    return booktabs_clines(
        formatted.style.hide(axis="index").to_latex(hrules=True, column_format="l" + "r" * (len(columns) - 1))
    )


def region_latex(table: pd.DataFrame) -> str:
    groups = [table.index[table["region"] == r] for r in table["region"].unique()]
    formatted = _bold_best(table, list(MEASURES), groups)
    formatted["region"] = table["region"].map(lambda r: REGION_LABELS.get(r, r))
    formatted["distance"] = table["distance"].map(lambda d: DISTANCE_LABELS.get(d, d))
    columns = {"region": "Region", "distance": "Distance", "n_bags": "Bags"}
    formatted = _format_diagnostics(table, formatted)
    columns |= {m: label for m, (label, _) in MEASURES.items()} | DIAGNOSTICS
    formatted = formatted[list(columns)].rename(columns=columns).set_index(["Region", "Distance"])
    latex = formatted.style.to_latex(hrules=True, clines="skip-last;data", multirow_align="naive")
    return booktabs_clines(latex)


CAPTIONS = {
    "summary": "SQS-S (PrediQuant with PACC(MLP)) with three distances on the three-class datasets of "
    "\\texttt{predist} (E6.1; vanilla accuracy, $|U|=1000$), scored against the realised table of each test bag. "
    "Means over all the test bags of the absolute error (AE), coverage, amplitude and Winkler score of the $95\\%$ "
    "interval, and of the mean $L_1$ distance between the true prevalences of the selected bags and that of the test "
    "bag; best value in bold. \\textit{Best pairs}: pairs where the distance has the lowest per-pair mean Winkler "
    "score (ties count for every tied distance); \\textit{Pairs $<L_1$}: pairs where its per-pair mean Winkler score is "
    "lower than that of $L_1$; $p$: two-sided paired Wilcoxon signed-rank test over the per-pair mean Winkler scores "
    "against $L_1$. Diagnostics, not in bold: \\textit{Test-prev.\\ error}, the $L_1$ error of the estimated "
    "prevalence of the test bag (the same for every distance); \\textit{Estimated-space $L_1$}, the mean $L_1$ distance "
    "between the estimated prevalences of the selected bags and that of the test bag.",
    "region": "The means of the summary table split by the position of the test bag on the simplex (centre, the three "
    "vertices at $1-0.01$, the random point farthest from them); best value of each region in bold.",
}


def _latex_document(tabular_path: str, caption: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,landscape,hmargin=1cm,vmargin=1cm]{geometry}",
            "\\usepackage{amsmath,booktabs,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\begin{table}[h]",
            "\\centering\\small",
            f"\\caption{{{caption}}}",
            f"\\input{{{tabular_path}}}",
            "\\end{table}",
            "\\end{document}",
            "",
        ]
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def print_decision(result: dict) -> None:
    for distance, checks in result["checks"].items():
        marks = ", ".join(f"{name}={'yes' if ok else 'no'}" for name, ok in checks.items())
        print(f"  {distance}: {marks}")
    print(f"decision: {result['chosen']}" + (" (L1 kept)" if result["chosen"] == REFERENCE else ""))


def gen_tables(pargs):
    df = load_results(pargs.domain, pargs.acc_name)
    if df.empty:
        raise RuntimeError(f"No {EXPERIMENT} results with schema_version >= {RESULT_SCHEMA_VERSION} found.")
    pairs, pair_region, summary, region = compute(df)
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    pairs.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_pairs.csv"), index=False)
    pair_region.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_pair_region.csv"), index=False)
    latex = {"summary": lambda: summary_latex(summary), "region": lambda: region_latex(region)}
    frames = {"summary": summary, "region": region}
    for name in pargs.tables_set:
        frames[name].to_csv(os.path.join(out_dir, f"{_stem(name)}.csv"), index=False)
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(latex[name]())
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{_stem(name)}.tex", CAPTIONS[name]))

    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        print(f"\nsummary:\n{summary.to_string(index=False)}")
        print(f"\nper region:\n{region.to_string(index=False)}")
    print_decision(decision(summary))
    print(f"pairs: {len(pairs[KEYS].drop_duplicates())}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for name in pargs.tables_set:
        tex_filename = f"{_stem(name)}_doc.tex"
        if not os.path.exists(os.path.join(out_dir, tex_filename)):
            raise FileNotFoundError(f"{tex_filename} does not exist in {out_dir}. Generate tables first.")
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Summary tables of the predist distance diagnostic (E6.1).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--acc-name", default=DEFAULT_ACC_NAME)
    parser.add_argument(
        "--tables-set",
        nargs="+",
        choices=TABLE_SETS,
        default=list(TABLE_SETS),
        help="Tables to write: summary (one row per distance), region (by position on the simplex); default: both",
    )
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSVs, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
