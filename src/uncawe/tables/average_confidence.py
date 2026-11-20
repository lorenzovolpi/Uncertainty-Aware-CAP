"""AC against L-CBPE on the pairs of ``main`` (E8.1 of ``docs/ext8_average_confidence.md``, card WP-29).

``AC`` (``methods/ac.py:AC``) is Average Confidence with the posteriors recalibrated per bag by LasCal and the central
interval of the Poisson-binomial distribution of the correct predictions; ``L-CBPE`` (``cap-ml``'s
``CBPE(calib_method="lascal")``, binary datasets only) is the same estimator with the highest-density interval. Both
store the probability mass function of the number of correct predictions in ``estim_acc_ranges``, and their intervals
are read from it by ``tables/main.py:PMF_INTERVALS``. Both target the realised table of each target bag and are scored
against it (``true_cts``). Vanilla accuracy only (both methods give NaN on the other metrics).

Outputs, under ``<OUT_DIR>/uncawe/tables/average_confidence/``:

- **comparison table** (``average_confidence.csv``): one row per pair (dataset × classifier), the means over the bags of
  AE, coverage, amplitude and Winkler score at the level of ``main`` for ``AC`` and ``L-CBPE`` (NaN on the multiclass
  pairs), and two summary rows: the means over the pairs where both methods have results (the binary pairs) and over
  every pair with ``AC`` results (``AC`` only);
- **check table** (``average_confidence_check.csv``): per pair, on the pairs where both methods have results, the
  largest absolute difference of the point estimates, bag by bag (expected $0$ up to rounding, §T.5) and the mean
  differences of the lower and upper bounds (``AC`` minus ``L-CBPE``); for ``AC`` on every pair, the number of bags
  where LasCal fell back to the uncalibrated posteriors and the minimum, median and maximum of the selected
  temperature $\\theta$ over the other bags;
- each table as a LaTeX ``tabular`` fragment and a ``longtable``, and a standalone document with both.

The pairs are those where ``AC`` has stored results, so the module also works on a pilot. The bags of the two methods
are paired by their stored order and checked to have the same true tables.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import acc_from_ct
from uncawe.experiments.methods import AC_METHOD
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables import main as main_tables
from uncawe.tables.estimand_gap import booktabs_clines
from uncawe.util import decorate_dataset

TABLE_NAME = "average_confidence"
DOMAIN = "classic"
ACC_NAME = "vanilla_accuracy"
L_CBPE = "L-CBPE"
METHODS = (AC_METHOD, L_CBPE)
SCORES = ("ae", "coverage", "amplitude", "winkler")
# the column labels of the scores, those of the main tables (where AE is called acc_err)
SCORE_LABELS = {score: main_tables.metrics_map["acc_err" if score == "ae" else score] for score in SCORES}
KEYS = ["dataset", "classifier"]
CLASSIFIER_ORDER = main_tables.CLASSIFIER_FILTER
SUMMARY_BOTH = "mean, pairs with both methods"
SUMMARY_AC = f"mean, every pair ({AC_METHOD} only)"
CHECK_COLUMNS = {
    "max_abs_point_diff": "$\\max\\lvert\\Delta\\hat A\\rvert$",
    "mean_low_diff": "$\\overline{\\Delta l}$",
    "mean_high_diff": "$\\overline{\\Delta u}$",
    "n_fallback": "fallbacks",
    "theta_min": "$\\theta_{\\min}$",
    "theta_median": "$\\theta_{\\mathrm{med}}$",
    "theta_max": "$\\theta_{\\max}$",
}


def load_results(base_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """The stored results of ``main`` (vanilla accuracy) of the two methods, method -> frame (empty if none)."""
    base_dir = base_dir if base_dir is not None else os.path.join(env.root_dir, EXPERIMENT)
    out = {}
    for method in METHODS:
        try:
            out[method] = Results.load(
                base_dir=base_dir, domain=DOMAIN, acc_name=ACC_NAME, filter_methods=[method]
            ).df.reset_index(drop=True)
        except ValueError:
            # Results.load concatenates the files it finds, and pd.concat raises ValueError when there are none
            out[method] = pd.DataFrame()
    return out


def bag_scores(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per bag of one pair and method (stored order): the realised accuracy, the point estimate, AE, the
    bounds of the interval, coverage, amplitude and Winkler score; NaN where the method gave no estimate."""
    rows = []
    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(ACC_NAME, np.asarray(row.true_cts, dtype=float))
        pmf = row.estim_acc_ranges
        if main_tables._no_estimate(pmf) or not np.all(np.isfinite(np.asarray(pmf, dtype=float))):
            rows.append({"true_acc": true_acc, "point": np.nan, "low": np.nan, "high": np.nan} | dict.fromkeys(SCORES))
            continue
        interval = main_tables._confidence_interval(pmf, method)
        rows.append(
            {
                "true_acc": true_acc,
                "point": float(row.estim_accs),
                "low": float(interval.low),
                "high": float(interval.high),
                "ae": float(row.acc_err),
                "coverage": main_tables.get_acc_coverage(method, true_acc, pmf),
                "amplitude": main_tables.get_acc_amplitude(method, pmf),
                "winkler": main_tables.get_acc_winkler(method, true_acc, pmf),
            }
        )
    return pd.DataFrame(rows).astype(float)


def _check_paired(a: pd.DataFrame, b: pd.DataFrame, key: tuple) -> None:
    if len(a) != len(b) or not np.allclose(a["true_acc"], b["true_acc"]):
        raise ValueError(f"{key}: the bags of {AC_METHOD} and {L_CBPE} are not the same")


def _lascal_summary(df: pd.DataFrame) -> dict:
    """The fallbacks and the temperatures of ``AC`` on one pair (NaN if the columns are missing)."""
    if "lascal_fallback" not in df or "lascal_temperature" not in df:
        return {"n_fallback": np.nan, "theta_min": np.nan, "theta_median": np.nan, "theta_max": np.nan}
    fallback = df["lascal_fallback"].astype(bool).to_numpy()
    theta = pd.to_numeric(df["lascal_temperature"], errors="coerce").to_numpy(dtype=float)[~fallback]
    theta = theta[np.isfinite(theta)]
    stats = (np.min(theta), np.median(theta), np.max(theta)) if len(theta) else (np.nan,) * 3
    return {
        "n_fallback": int(fallback.sum()),
        "theta_min": float(stats[0]),
        "theta_median": float(stats[1]),
        "theta_max": float(stats[2]),
    }


def pair_tables(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The comparison table and the check table, one row per pair with ``AC`` results, without the summary rows."""
    groups = {m: dict(list(df.groupby(KEYS, sort=False))) if not df.empty else {} for m, df in results.items()}
    comparison, check = [], []
    for key, ac_df in groups[AC_METHOD].items():
        ac = bag_scores(ac_df, AC_METHOD)
        base = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(ac_df["n_classes"].iloc[0]),
            "n_bags": len(ac),
        }
        row = dict(base)
        check_row = dict(base) | _lascal_summary(ac_df)
        lcbpe = bag_scores(groups[L_CBPE][key], L_CBPE) if key in groups[L_CBPE] else None
        if lcbpe is not None and lcbpe["point"].notna().any():
            _check_paired(ac, lcbpe, key)
            check_row |= {
                "max_abs_point_diff": float(np.nanmax(np.abs(ac["point"] - lcbpe["point"]))),
                "mean_low_diff": float(np.nanmean(ac["low"] - lcbpe["low"])),
                "mean_high_diff": float(np.nanmean(ac["high"] - lcbpe["high"])),
            }
        else:
            lcbpe = None
            check_row |= dict.fromkeys(("max_abs_point_diff", "mean_low_diff", "mean_high_diff"), np.nan)
        for score in SCORES:
            row[f"{AC_METHOD}_{score}"] = float(ac[score].mean())
            row[f"{L_CBPE}_{score}"] = float(lcbpe[score].mean()) if lcbpe is not None else np.nan
        comparison.append(row)
        check.append(check_row)
    return ordered(pd.DataFrame(comparison)), ordered(pd.DataFrame(check))


def summary_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    """The means over the pairs where both methods have results (both columns) and over every pair (``AC`` only)."""
    both = comparison[comparison[f"{L_CBPE}_ae"].notna()]
    score_columns = [f"{m}_{s}" for s in SCORES for m in METHODS]
    rows = []
    row = {"dataset": SUMMARY_BOTH, "classifier": "", "n_pairs": len(both)}
    row |= {c: float(both[c].mean()) if len(both) else np.nan for c in score_columns}
    rows.append(row)
    row = {"dataset": SUMMARY_AC, "classifier": "", "n_pairs": len(comparison)}
    row |= {c: float(comparison[c].mean()) if c.startswith(f"{AC_METHOD}_") else np.nan for c in score_columns}
    rows.append(row)
    return pd.DataFrame(rows)


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Datasets by number of classes, then by name; classifiers in the order of the main tables."""
    if table.empty:
        return table
    table = table.copy()
    datasets = table.drop_duplicates("dataset").sort_values(["n_classes", "dataset"])["dataset"].tolist()
    table["dataset"] = pd.Categorical(table["dataset"], categories=datasets, ordered=True)
    table["classifier"] = pd.Categorical(table["classifier"], categories=list(CLASSIFIER_ORDER), ordered=True)
    table = table.sort_values(["dataset", "classifier"]).reset_index(drop=True)
    table["dataset"] = table["dataset"].astype(str)
    table["classifier"] = table["classifier"].astype(str)
    return table


def _display_dataset(name: str) -> str:
    return decorate_dataset(main_tables.dataset_map.get(name, name))


def _comparison_frame(comparison: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """The comparison table for LaTeX: index (dataset, $K$, classifier), columns (score, method)."""
    table = pd.concat([comparison, summary], ignore_index=True)
    cols = {
        (SCORE_LABELS[score], method): table[f"{method}_{score}"].to_numpy() for score in SCORES for method in METHODS
    }
    out = pd.DataFrame(cols)
    datasets = [d if d in (SUMMARY_BOTH, SUMMARY_AC) else _display_dataset(d) for d in table["dataset"].astype(str)]
    n_classes = ["" if pd.isna(k) else str(int(k)) for k in table.get("n_classes", pd.Series([np.nan] * len(table)))]
    out.index = pd.MultiIndex.from_arrays(
        [datasets, n_classes, table["classifier"].astype(str).tolist()], names=["dataset", "$K$", "classifier"]
    )
    return out


def _check_frame(check: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({label: check[column].to_numpy() for column, label in CHECK_COLUMNS.items()})
    out.index = pd.MultiIndex.from_arrays(
        [
            [_display_dataset(d) for d in check["dataset"]],
            check["n_classes"].astype(int).astype(str).tolist(),
            check["classifier"].tolist(),
        ],
        names=["dataset", "$K$", "classifier"],
    )
    return out


def _styler(frame: pd.DataFrame, formats: dict | None = None):
    return frame.style.format(precision=3, na_rep="-", formatter=formats)


def _latex(frame: pd.DataFrame, formats: dict | None = None, caption: str | None = None, label: str | None = None):
    styler = _styler(frame, formats)
    if caption is None:
        return booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data"))
    latex = styler.to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=caption,
        label=label,
    )
    return booktabs_clines(latex)


CHECK_FORMATS = {
    CHECK_COLUMNS["max_abs_point_diff"]: "{:.1e}",
    CHECK_COLUMNS["mean_low_diff"]: "{:.4f}",
    CHECK_COLUMNS["mean_high_diff"]: "{:.4f}",
    CHECK_COLUMNS["n_fallback"]: "{:.0f}",
    CHECK_COLUMNS["theta_min"]: "{:.2f}",
    CHECK_COLUMNS["theta_median"]: "{:.2f}",
    CHECK_COLUMNS["theta_max"]: "{:.2f}",
}
COMPARISON_CAPTION = (
    f"{AC_METHOD} against {L_CBPE} on \\texttt{{main}} (vanilla accuracy, $|U|=1000$): AE, coverage $\\mathcal{{C}}$, "
    "amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ at $1-\\alpha=0.95$, against the realised table of each "
    f"bag, means over the bags; {L_CBPE} only on the binary datasets. Last rows: means over the pairs."
)
CHECK_CAPTION = (
    f"Check of {AC_METHOD} against {L_CBPE}: largest absolute difference of the point estimates over the bags, mean "
    f"differences of the lower and upper bounds ({AC_METHOD} minus {L_CBPE}, binary datasets); bags where LasCal fell "
    "back to the uncalibrated posteriors, and the selected temperature $\\theta$ over the other bags."
)


def _latex_document(longtables: list[str]) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,landscape,hmargin=1.5cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{booktabs,longtable,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\scriptsize",
            "\\setlength{\\LTcapwidth}{\\textwidth}",
            *[f"\\input{{{path}}}\n\\clearpage" for path in longtables],
            "\\end{document}",
            "",
        ]
    )


def table_dir(pargs=None) -> str:
    if pargs is not None and getattr(pargs, "table_dir", None) is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def write_tables(results: dict[str, pd.DataFrame], out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the CSVs, the LaTeX fragments and the document of the two tables to ``out_dir``."""
    if results[AC_METHOD].empty:
        raise RuntimeError(f"No {AC_METHOD} results found for {EXPERIMENT}. Check that experiment results exist.")
    comparison, check = pair_tables(results)
    summary = summary_rows(comparison)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    pd.concat([comparison, summary], ignore_index=True).to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)
    check.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_check.csv"), index=False)

    frames = {
        TABLE_NAME: (_comparison_frame(comparison, summary), None, COMPARISON_CAPTION),
        f"{TABLE_NAME}_check": (_check_frame(check), CHECK_FORMATS, CHECK_CAPTION),
    }
    longtables = []
    for name, (frame, formats, caption) in frames.items():
        with open(os.path.join(tabular_dir, f"{name}.tex"), "w") as f:
            f.write(_latex(frame, formats))
        with open(os.path.join(tabular_dir, f"{name}_long.tex"), "w") as f:
            f.write(_latex(frame, formats, caption=caption, label=f"tab:{name.replace('_', '-')}"))
        longtables.append(f"tabular/{name}_long.tex")
    with open(os.path.join(out_dir, f"{TABLE_NAME}_doc.tex"), "w") as f:
        f.write(_latex_document(longtables))
    return comparison, check


def gen_tables(pargs):
    out_dir = table_dir(pargs)
    comparison, check = write_tables(load_results(), out_dir)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 4):
        print(pd.concat([comparison, summary_rows(comparison)], ignore_index=True).to_string(index=False))
        print(check.to_string(index=False))
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    tex_filename = f"{TABLE_NAME}_doc.tex"
    if not os.path.exists(os.path.join(out_dir, tex_filename)):
        raise FileNotFoundError(f"{tex_filename} does not exist in {out_dir}. Generate tables first.")
    # longtable needs a second pass to settle the column widths
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="E8.1: AC against L-CBPE on stored main results.")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
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
