r"""AC and L-CBPE with label-shift binning against LasCal, on the pairs of ``main`` (E8.3 of
``docs/ext8_average_confidence.md``, card WP-32).

``AC`` and ``L-CBPE`` (E8.1) recalibrate the posteriors of $h$ on each bag with LasCal; ``AC-LSB`` and ``L-CBPE-LSB``
replace LasCal by ``cap-ml``'s label-shift binning (``cap.calibration.LSBinning``: class-wise uniform-mass bins of $V$,
their class frequencies reweighted by $\hat w=\hat\pi^Q/\pi^V$ with $\hat\pi^Q$ of BBSE-soft, §T.9), and take as the
confidence of an instance the calibrated posterior of the class predicted by $h$. ``AC`` and ``AC-LSB`` read the central
interval of the Poisson-binomial distribution of the correct predictions, ``L-CBPE`` and ``L-CBPE-LSB`` (binary datasets
only) its highest-density interval (``tables/main.py:PMF_INTERVALS``). All four target the realised table of each
target bag and are scored against it (``true_cts``); vanilla accuracy only. The quantification error of BBSE-soft is
scored against the prevalence of the bag, the row sums of the stored ``true_cts`` (true classes on rows in the code,
the column sums of the paper's $T$).

Outputs, under ``<OUT_DIR>/uncawe/tables/ac_lsb/``:

- **comparison table** (``ac_lsb.csv``): one row per pair (dataset × classifier), the means over the bags of AE,
  coverage, amplitude and Winkler score at the level of ``main`` for the four methods (``L-CBPE`` and ``L-CBPE-LSB``
  NaN on the multiclass pairs), and two summary rows: the means over the binary pairs (the pairs where the four methods
  have results) and over every pair (``AC`` and ``AC-LSB`` only);
- **differences table** (``ac_lsb_diff.csv``): per pair, the paired differences of the four scores, ``AC-LSB`` minus
  ``AC`` and ``L-CBPE-LSB`` minus ``L-CBPE`` (means over the bags of the bag-by-bag differences, equal to the
  differences of the means), with the same summary rows;
- **diagnostics table** (``ac_lsb_diagnostics.csv``): per pair, from the per-bag columns of ``AC-LSB``: the selected
  number of bins, the minimum and maximum over the classes of the effective number of bins, the grid fallback, the
  means over the bags of the instances made uniform (``lsb_n_degenerate``) and of the fraction of instances whose
  calibrated argmax differs from the prediction of $h$ (``lsb_argmax_changed``), and the mean quantification error
  $\lVert\hat\pi^Q-\bar\pi\rVert_1$ of BBSE-soft; on the binary pairs, the check of ``AC-LSB`` against ``L-CBPE-LSB``:
  the largest absolute difference of the point estimates over the bags (expected $0$ up to rounding, §T.9);
- each table as a LaTeX ``tabular`` fragment and a ``longtable``, and a standalone document with the three.

The pairs are those where ``AC-LSB`` has stored results, so the module also works on a pilot. The bags of the methods
are paired by their stored order and checked to have the same true tables, as in ``tables/average_confidence.py``.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import AC_LSB_METHOD, AC_METHOD, L_CBPE_LSB_METHOD
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables.average_confidence import (
    ACC_NAME,
    CLASSIFIER_ORDER,
    DOMAIN,
    KEYS,
    L_CBPE,
    SCORE_LABELS,
    SCORES,
    _display_dataset,
    _latex,
    _latex_document,
    bag_scores,
)

TABLE_NAME = "ac_lsb"
METHODS = (AC_METHOD, AC_LSB_METHOD, L_CBPE, L_CBPE_LSB_METHOD)
# the method of each pair of the paired differences: (label-shift binning, LasCal)
PAIRS = ((AC_LSB_METHOD, AC_METHOD), (L_CBPE_LSB_METHOD, L_CBPE))
BINARY_ONLY = (L_CBPE, L_CBPE_LSB_METHOD)
SUMMARY_BINARY = "mean, binary pairs"
SUMMARY_ALL = f"mean, every pair ({AC_METHOD} and {AC_LSB_METHOD} only)"
SUMMARIES = (SUMMARY_BINARY, SUMMARY_ALL)
DIAGNOSTIC_COLUMNS = {
    "nbins": "bins",
    "nbins_eff_min": "$b^{\\mathrm{eff}}_{\\min}$",
    "nbins_eff_max": "$b^{\\mathrm{eff}}_{\\max}$",
    "grid_fallback": "fallback",
    "mean_n_degenerate": "$\\overline{n}_{\\mathrm{deg}}$",
    "mean_argmax_changed": "$\\overline{f}_{\\arg\\max}$",
    "mean_q_l1": "$\\overline{\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1}$",
    "max_abs_point_diff": "$\\max\\lvert\\Delta\\hat A\\rvert$",
}
DIAGNOSTIC_FORMATS = {
    DIAGNOSTIC_COLUMNS["nbins"]: "{:.0f}",
    DIAGNOSTIC_COLUMNS["nbins_eff_min"]: "{:.0f}",
    DIAGNOSTIC_COLUMNS["nbins_eff_max"]: "{:.0f}",
    DIAGNOSTIC_COLUMNS["grid_fallback"]: "{:.0f}",
    DIAGNOSTIC_COLUMNS["mean_n_degenerate"]: "{:.2f}",
    DIAGNOSTIC_COLUMNS["mean_argmax_changed"]: "{:.3f}",
    DIAGNOSTIC_COLUMNS["mean_q_l1"]: "{:.3f}",
    DIAGNOSTIC_COLUMNS["max_abs_point_diff"]: "{:.1e}",
}
LSB_COLUMNS = (
    "lsb_nbins",
    "lsb_nbins_eff",
    "lsb_grid_fallback",
    "lsb_prevalence",
    "lsb_n_degenerate",
    "lsb_argmax_changed",
)


def load_results(base_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """The stored results of ``main`` (vanilla accuracy) of the four methods, method -> frame (empty if none)."""
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


def bag_prevalences(df: pd.DataFrame) -> np.ndarray:
    """$\\bar\\pi$ of each bag, shape (bags, $K$): the row sums of the stored ``true_cts`` (true classes on rows),
    normalised."""
    cts = np.stack([np.asarray(ct, dtype=float) for ct in df["true_cts"]])
    sums = cts.sum(axis=2)
    return sums / sums.sum(axis=1, keepdims=True)


def validation_prevalence(val_prev, n_classes: int) -> np.ndarray:
    """$\\pi^V$ from the stored ``val_prev`` (``util.get_plain_prev``: the last class for $K=2$, the classes after the
    first for $K>2$, rounded to $4$ decimals); the first class takes the rest."""
    rest = np.atleast_1d(np.asarray(val_prev, dtype=float))
    if len(rest) != n_classes - 1:
        raise ValueError(f"val_prev has {len(rest)} values, expected {n_classes - 1}")
    return np.concatenate([[1.0 - rest.sum()], rest])


def bag_shifts(df: pd.DataFrame) -> np.ndarray:
    """The shift $\\lVert\\bar\\pi-\\pi^V\\rVert_1$ of each bag; $\\pi^V$ is rebuilt from the rounded ``val_prev``, so
    the shift is exact to about $K\\cdot5\\cdot10^{-5}$."""
    n_classes = int(df["n_classes"].iloc[0])
    prevalences = bag_prevalences(df)
    val = np.stack([validation_prevalence(v, n_classes) for v in df["val_prev"]])
    return np.abs(prevalences - val).sum(axis=1)


def _check_paired(scores: dict[str, pd.DataFrame], key: tuple) -> None:
    if not scores:
        return
    name, reference = next(iter(scores.items()))
    for method, frame in scores.items():
        if len(frame) != len(reference) or not np.allclose(frame["true_acc"], reference["true_acc"]):
            raise ValueError(f"{key}: the bags of {method} and {name} are not the same")


def _has_estimates(scores: pd.DataFrame | None) -> bool:
    return scores is not None and scores["point"].notna().any()


def _diagnostics(df: pd.DataFrame) -> dict:
    """The diagnostics of label-shift binning on one pair, from the per-bag columns of ``AC-LSB`` (NaN if missing)."""
    out = dict.fromkeys(
        (
            "nbins",
            "nbins_eff_min",
            "nbins_eff_max",
            "grid_fallback",
            "mean_n_degenerate",
            "mean_argmax_changed",
            "mean_q_l1",
        ),
        np.nan,
    )
    if not set(LSB_COLUMNS) <= set(df.columns):
        return out
    estimated = df["lsb_nbins"].notna().to_numpy()
    if not estimated.any():
        return out
    part = df[estimated]
    nbins_eff = np.stack([np.asarray(v, dtype=float) for v in part["lsb_nbins_eff"]])
    estimates = np.stack([np.asarray(v, dtype=float) for v in part["lsb_prevalence"]])
    out |= {
        # the binning is fitted once per pair, so these are the same on every bag
        "nbins": float(part["lsb_nbins"].iloc[0]),
        "nbins_eff_min": float(nbins_eff.min()),
        "nbins_eff_max": float(nbins_eff.max()),
        "grid_fallback": float(part["lsb_grid_fallback"].astype(bool).any()),
        "mean_n_degenerate": float(pd.to_numeric(part["lsb_n_degenerate"]).mean()),
        "mean_argmax_changed": float(pd.to_numeric(part["lsb_argmax_changed"]).mean()),
        "mean_q_l1": float(np.abs(estimates - bag_prevalences(part)).sum(axis=1).mean()),
    }
    return out


def pair_tables(results: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The comparison, differences and diagnostics tables, one row per pair with ``AC-LSB`` results, without the
    summary rows."""
    groups = {m: dict(list(df.groupby(KEYS, sort=False))) if not df.empty else {} for m, df in results.items()}
    comparison, differences, diagnostics = [], [], []
    for key, lsb_df in groups[AC_LSB_METHOD].items():
        scores = {}
        for method in METHODS:
            frame = bag_scores(groups[method][key], method) if key in groups.get(method, {}) else None
            if _has_estimates(frame):
                scores[method] = frame
        _check_paired(scores, key)
        base = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(lsb_df["n_classes"].iloc[0]),
            "n_bags": len(lsb_df),
        }
        row, diff = dict(base), dict(base)
        for score in SCORES:
            for method in METHODS:
                row[f"{method}_{score}"] = float(scores[method][score].mean()) if method in scores else np.nan
            for lsb, lascal in PAIRS:
                both = lsb in scores and lascal in scores
                value = float((scores[lsb][score] - scores[lascal][score]).mean()) if both else np.nan
                diff[f"{lsb}_{score}"] = value
        comparison.append(row)
        differences.append(diff)
        check = np.nan
        if AC_LSB_METHOD in scores and L_CBPE_LSB_METHOD in scores:
            check = float(np.nanmax(np.abs(scores[AC_LSB_METHOD]["point"] - scores[L_CBPE_LSB_METHOD]["point"])))
        diagnostics.append(dict(base) | _diagnostics(lsb_df) | {"max_abs_point_diff": check})
    return ordered(pd.DataFrame(comparison)), ordered(pd.DataFrame(differences)), ordered(pd.DataFrame(diagnostics))


def summary_rows(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """The means over the binary pairs (every column) and over every pair (the columns of ``AC`` and ``AC-LSB``)."""
    binary = table[table["n_classes"] == 2]
    rows = [{"dataset": SUMMARY_BINARY, "classifier": "", "n_pairs": len(binary)}]
    rows[0] |= {c: float(binary[c].mean()) if len(binary) else np.nan for c in columns}
    rows.append({"dataset": SUMMARY_ALL, "classifier": "", "n_pairs": len(table)})
    rows[1] |= {c: np.nan if c.startswith(BINARY_ONLY) else float(table[c].mean()) for c in columns}
    return pd.DataFrame(rows)


def comparison_columns() -> list[str]:
    return [f"{m}_{s}" for s in SCORES for m in METHODS]


def difference_columns() -> list[str]:
    return [f"{lsb}_{s}" for s in SCORES for lsb, _ in PAIRS]


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


def _index(table: pd.DataFrame) -> pd.MultiIndex:
    datasets = [d if d in SUMMARIES else _display_dataset(d) for d in table["dataset"].astype(str)]
    n_classes = ["" if pd.isna(k) else str(int(k)) for k in table.get("n_classes", pd.Series([np.nan] * len(table)))]
    return pd.MultiIndex.from_arrays(
        [datasets, n_classes, table["classifier"].astype(str).tolist()], names=["dataset", "$K$", "classifier"]
    )


def _score_frame(table: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """A table for LaTeX: index (dataset, $K$, classifier), columns (score, method)."""
    cols = {(SCORE_LABELS[s], m): table[f"{m}_{s}"].to_numpy() for s in SCORES for m in methods}
    out = pd.DataFrame(cols)
    out.index = _index(table)
    return out


def _diagnostics_frame(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({label: table[column].to_numpy() for column, label in DIAGNOSTIC_COLUMNS.items()})
    out.index = _index(table)
    return out


COMPARISON_CAPTION = (
    f"{AC_METHOD} and {L_CBPE} with LasCal against {AC_LSB_METHOD} and {L_CBPE_LSB_METHOD} with label-shift binning "
    "on \\texttt{main} (vanilla accuracy, $|U|=1000$): AE, coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$ and "
    "Winkler score $\\mathcal{W}$ at $1-\\alpha=0.95$, against the realised table of each bag, means over the bags; "
    f"{L_CBPE} and {L_CBPE_LSB_METHOD} only on the binary datasets. Last rows: means over the pairs."
)
DIFFERENCES_CAPTION = (
    f"Paired differences of the scores, {AC_LSB_METHOD} minus {AC_METHOD} and {L_CBPE_LSB_METHOD} minus {L_CBPE} "
    "(columns named by the method with label-shift binning), means over the bags; negative AE, amplitude and Winkler "
    "score favour label-shift binning. Last rows: means over the pairs."
)
DIAGNOSTICS_CAPTION = (
    f"Label-shift binning in {AC_LSB_METHOD}: selected number of bins, range of the effective numbers over the "
    "classes, grid fallback, mean number of instances made uniform per bag, mean fraction of instances whose calibrated "
    "argmax differs from the prediction of $h$, mean quantification error of BBSE-soft; last column, on the binary "
    f"datasets, the largest absolute difference of the point estimates of {AC_LSB_METHOD} and {L_CBPE_LSB_METHOD}."
)


def table_dir(pargs=None) -> str:
    if pargs is not None and getattr(pargs, "table_dir", None) is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def write_tables(results: dict[str, pd.DataFrame], out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write the CSVs, the LaTeX fragments and the document of the three tables to ``out_dir``."""
    if results[AC_LSB_METHOD].empty:
        raise RuntimeError(f"No {AC_LSB_METHOD} results found for {EXPERIMENT}. Check that experiment results exist.")
    comparison, differences, diagnostics = pair_tables(results)
    comparison_full = pd.concat([comparison, summary_rows(comparison, comparison_columns())], ignore_index=True)
    differences_full = pd.concat([differences, summary_rows(differences, difference_columns())], ignore_index=True)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    comparison_full.to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)
    differences_full.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_diff.csv"), index=False)
    diagnostics.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_diagnostics.csv"), index=False)

    frames = {
        TABLE_NAME: (_score_frame(comparison_full, list(METHODS)), None, COMPARISON_CAPTION),
        f"{TABLE_NAME}_diff": (
            _score_frame(differences_full, [lsb for lsb, _ in PAIRS]),
            None,
            DIFFERENCES_CAPTION,
        ),
        f"{TABLE_NAME}_diagnostics": (_diagnostics_frame(diagnostics), DIAGNOSTIC_FORMATS, DIAGNOSTICS_CAPTION),
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
    return comparison_full, differences_full, diagnostics


def gen_tables(pargs):
    out_dir = table_dir(pargs)
    tables = write_tables(load_results(), out_dir)
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.precision", 4):
        for table in tables:
            print(table.to_string(index=False))
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
    parser = ArgumentParser(description="E8.3: AC and L-CBPE with label-shift binning against LasCal, on main.")
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
