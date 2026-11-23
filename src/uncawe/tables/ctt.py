"""Tables of E8.2 of ``docs/ext8_average_confidence.md`` (card WP-30): the rate matrix of $V$ rescaled by a point
quantifier, on the protocol of ``main``.

Reads the results of ``experiments/ctt.py`` (``<OUT_DIR>/uncawe/ctt/``): the three methods ``CTT-KDEy``, ``CTT-EMQ``,
``CTT-PACC``, which target the population table $C\\operatorname{diag}(\\pi^Q)$, and the reference ``CTT-oracle`` (true
prevalence of the bag), which targets the expected realised table given the class counts of the bag; all are scored
against the realised table of each bag (``true_cts``).

Outputs, under ``<OUT_DIR>/uncawe/tables/ctt/``:

- **AE tables**, one per metric (``ctt_ae_<metric>.csv`` for vanilla accuracy, macro-F1 and macro-K; ``ctt_ae.csv``
  with the three): one row per pair (dataset × classifier), in a block of binary and one of multiclass pairs, each
  closed by the mean over its pairs, and a final row of the mean over every pair; the mean AE over the bags of the
  three methods, of the point estimates of BayCAP-KDEy (``BayCAP-KDEy-mlp-real``) and BayCAP-Hf (``BayesCAP-Hf-real``)
  stored by ``main`` on the same bags (``MAIN_COMPARED``; their point estimate is the mean of their predictive
  distribution of the realised table, scored against the realised table as here), and of the reference. In LaTeX, in
  each row, the cells of the five methods are coloured from the green of the rank tables (``tables/rank.py:BEST_RGB``,
  the smallest AE) to white (the largest), linearly in the AE, and the smallest AE is in bold; the reference is set
  apart, not coloured and not compared;
- **split table** (``ctt_split.csv``, vanilla accuracy): the same rows; the range $R$ of the recalls of $h$ on $V$;
  per quantifier the mean quantification error $\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$ and the root mean square (RMS)
  over the bags of the quantification part $e_\\pi$; the RMS of the rate part $e_C$, which is the same for every level
  and equals the RMS of the error of ``CTT-oracle`` (§T.8);
- **check table** (``ctt_check.csv``): per pair, the largest $\\lvert e_\\pi+e_C-(\\hat A-A)\\rvert$ over the levels
  and bags (expected $0$ up to rounding), the largest difference between the $e_C$ of a quantifier and that of the
  oracle (expected $0$), the largest $\\lvert e_\\pi\\rvert$ of the oracle (expected $0$), the number of bags where
  $\\lvert e_\\pi\\rvert>\\frac R2\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$ (expected $0$), and the number of bags where
  each quantifier failed (expected $0$);
- each table as a LaTeX ``tabular`` fragment and a ``longtable``, and a standalone document with the five.

The pairs are those with stored results of E8.2, so the module also works on the pilot; a method of ``main`` without
results on a pair has NaN there. The bags of the levels and of the methods of ``main`` are paired by their stored order
and checked to have the same true tables.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.ctt import ACC_SPLIT, DOMAIN, EXPERIMENT, LEVELS, METHODS, ORACLE
from uncawe.experiments.main import EXPERIMENT as MAIN_EXPERIMENT
from uncawe.experiments.metrics import get_acc_names
from uncawe.results import Results
from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS, METHOD_LATEX_PREAMBLE
from uncawe.tables.estimand_gap import booktabs_clines
from uncawe.tables.rank import BEST_RGB, WHITE_RGB
from uncawe.util import decorate_dataset

TABLE_NAME = "ctt"
ACC_NAMES = tuple(get_acc_names())
ACC_LABELS = {"vanilla_accuracy": "vanilla accuracy", "macro-F1": "macro-F1", "macro-K": "macro-K"}
KEYS = ["dataset", "classifier"]
CLASSIFIER_ORDER = main_tables.CLASSIFIER_FILTER
SUMMARY_BINARY = "mean, binary pairs"
SUMMARY_MULTICLASS = "mean, multiclass pairs"
SUMMARY_ALL = "mean, all pairs"
SUMMARIES = (SUMMARY_BINARY, SUMMARY_MULTICLASS, SUMMARY_ALL)
# tolerance of the bound on e_pi in the check table, for rounding
BOUND_TOL = 1e-12
# the methods of main compared in the AE tables (user's decision of 2026-11-23): the point estimates of BayCAP-KDEy and
# BayCAP-Hf, both with the realised-table predictive, as in the main tables
MAIN_COMPARED = ("BayCAP-KDEy-mlp-real", "BayesCAP-Hf-real")
COMPARED = (*METHODS, *MAIN_COMPARED)
AE_COLUMNS = (*COMPARED, ORACLE)


def load_results(base_dir: str | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """The stored results of E8.2, (level, metric) -> frame (empty if none)."""
    base_dir = base_dir if base_dir is not None else os.path.join(env.root_dir, EXPERIMENT)
    out = {}
    for level in LEVELS:
        for acc_name in ACC_NAMES:
            try:
                out[level, acc_name] = Results.load(
                    base_dir=base_dir, domain=DOMAIN, acc_name=acc_name, filter_methods=[level]
                ).df.reset_index(drop=True)
            except ValueError:
                # pd.concat raises ValueError when no file is found
                out[level, acc_name] = pd.DataFrame()
    return out


def load_main_results(base_dir: str | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    """The stored results of ``main`` of ``MAIN_COMPARED``, (method, metric) -> frame (empty if none)."""
    base_dir = base_dir if base_dir is not None else os.path.join(env.root_dir, MAIN_EXPERIMENT)
    out = {}
    for method in MAIN_COMPARED:
        for acc_name in ACC_NAMES:
            try:
                out[method, acc_name] = Results.load(
                    base_dir=base_dir, domain=DOMAIN, acc_name=acc_name, filter_methods=[method]
                ).df.reset_index(drop=True)
            except ValueError:
                out[method, acc_name] = pd.DataFrame()
    return out


def _groups(results: dict[tuple[str, str], pd.DataFrame]) -> dict[tuple[str, str], dict]:
    return {k: dict(list(df.groupby(KEYS, sort=False))) if not df.empty else {} for k, df in results.items()}


def _pairs(groups: dict) -> list[tuple[str, str]]:
    keys = []
    for per_pair in groups.values():
        keys.extend(k for k in per_pair if k not in keys)
    return keys


def _true_accuracy(df: pd.DataFrame) -> np.ndarray:
    """The realised vanilla accuracy of each bag, the trace of its normalised true table."""
    return np.array([np.trace(np.asarray(ct, dtype=float)) for ct in df["true_cts"]])


def _check_paired(frames: dict[str, pd.DataFrame], key: tuple) -> None:
    """The levels of a pair are scored on the same bags, in the same order."""
    reference = None
    for level, df in frames.items():
        cts = np.stack([np.asarray(ct, dtype=float) for ct in df["true_cts"]])
        if reference is None:
            reference = (level, cts)
        elif cts.shape != reference[1].shape or not np.allclose(cts, reference[1]):
            raise ValueError(f"{key}: the bags of {level} and {reference[0]} are not the same")


def _rms(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(values**2))) if len(values) else np.nan


def _nanmean(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def pair_tables(
    results: dict[tuple[str, str], pd.DataFrame], main_results: dict[tuple[str, str], pd.DataFrame] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The AE, split and check tables, one row per pair with stored results of E8.2, without the summary rows; the AE
    table also has the methods of ``main_results`` (``MAIN_COMPARED``) on the same pairs."""
    groups = _groups(results)
    main_groups = _groups(main_results or {})
    ae_rows, split_rows, check_rows = [], [], []
    for key in _pairs(groups):
        frames = {
            (level, acc): groups[level, acc][key] for level in LEVELS for acc in ACC_NAMES if key in groups[level, acc]
        }
        frames |= {(method, acc): per_pair[key] for (method, acc), per_pair in main_groups.items() if key in per_pair}
        _check_paired({f"{level} ({acc})": df for (level, acc), df in frames.items()}, key)
        any_df = next(iter(frames.values()))
        base = {"dataset": key[0], "classifier": key[1], "n_classes": int(any_df["n_classes"].iloc[0])}
        base["n_bags"] = len(any_df)

        ae = dict(base)
        for acc in ACC_NAMES:
            for level in AE_COLUMNS:
                df = frames.get((level, acc))
                ae[f"{acc}|{level}"] = _nanmean(df["acc_err"]) if df is not None else np.nan
        ae_rows.append(ae)

        split, check = dict(base), dict(base)
        split_frames = {level: frames.get((level, ACC_SPLIT)) for level in LEVELS}
        oracle = split_frames[ORACLE]
        any_split = next((df for df in split_frames.values() if df is not None), None)
        split["R"] = float(any_split["recall_range"].iloc[0]) if any_split is not None else np.nan
        residuals, e_c_diffs, bound_violations = [], [], 0
        for level in LEVELS:
            df = split_frames[level]
            if df is None:
                if level != ORACLE:
                    split[f"{level}|q_l1"] = split[f"{level}|e_pi_rms"] = np.nan
                check[f"{level}|n_failed"] = np.nan
                continue
            e_pi = df["e_pi"].to_numpy(dtype=float)
            e_c = df["e_c"].to_numpy(dtype=float)
            q_l1 = df["q_l1"].to_numpy(dtype=float)
            signed = df["estim_accs"].to_numpy(dtype=float) - _true_accuracy(df)
            residuals.append(np.nanmax(np.abs(e_pi + e_c - signed)) if np.isfinite(signed).any() else np.nan)
            if oracle is not None and level != ORACLE:
                e_c_diffs.append(np.nanmax(np.abs(e_c - oracle["e_c"].to_numpy(dtype=float))))
            bound = df["recall_range"].to_numpy(dtype=float) / 2 * q_l1
            bound_violations += int(np.sum(np.abs(e_pi) > bound + BOUND_TOL))
            check[f"{level}|n_failed"] = int(df["q_failed"].astype(bool).sum())
            if level != ORACLE:
                split[f"{level}|q_l1"] = _nanmean(q_l1)
                split[f"{level}|e_pi_rms"] = _rms(e_pi)
        split["e_c_rms"] = _rms(oracle["e_c"]) if oracle is not None else np.nan
        check["max_split_residual"] = float(np.nanmax(residuals)) if residuals else np.nan
        check["max_e_c_diff"] = float(np.nanmax(e_c_diffs)) if e_c_diffs else np.nan
        check["max_oracle_e_pi"] = float(np.nanmax(np.abs(oracle["e_pi"]))) if oracle is not None else np.nan
        check["n_bound_violations"] = bound_violations
        split_rows.append(split)
        check_rows.append(check)
    return ordered(pd.DataFrame(ae_rows)), ordered(pd.DataFrame(split_rows)), ordered(pd.DataFrame(check_rows))


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


def with_summaries(table: pd.DataFrame) -> pd.DataFrame:
    """The binary block and its mean, the multiclass block and its mean, and the mean over every pair; each mean is
    the mean of the per-pair values over the pairs of its block (NaN values skipped)."""
    value_columns = [c for c in table.columns if c not in ("dataset", "classifier", "n_classes", "n_bags")]
    blocks = []
    for label, part in (
        (SUMMARY_BINARY, table[table["n_classes"] == 2]),
        (SUMMARY_MULTICLASS, table[table["n_classes"] > 2]),
    ):
        if part.empty:
            continue
        mean = {"dataset": label, "classifier": "", "n_pairs": len(part)}
        mean |= {c: _nanmean(part[c]) for c in value_columns}
        blocks += [part, pd.DataFrame([mean])]
    overall = {"dataset": SUMMARY_ALL, "classifier": "", "n_pairs": len(table)}
    overall |= {c: _nanmean(table[c]) for c in value_columns}
    blocks.append(pd.DataFrame([overall]))
    return pd.concat(blocks, ignore_index=True)


def _display_dataset(name: str) -> str:
    return name if name in SUMMARIES else decorate_dataset(main_tables.dataset_map.get(name, name))


def _index(table: pd.DataFrame) -> pd.MultiIndex:
    n_classes = ["" if pd.isna(k) else str(int(k)) for k in table["n_classes"]]
    return pd.MultiIndex.from_arrays(
        [[_display_dataset(d) for d in table["dataset"]], n_classes, table["classifier"].astype(str).tolist()],
        names=["dataset", "$K$", "classifier"],
    )


def _level_label(level: str) -> str:
    """The column label of a level or method in the AE tables (the LaTeX label of the methods of ``main``)."""
    return f"{level} (ref.)" if level == ORACLE else METHOD_LATEX_LABELS.get(level, level)


def ae_frame(table: pd.DataFrame, acc: str) -> pd.DataFrame:
    """The AE table of metric ``acc`` for LaTeX: index (dataset, $K$, classifier), one column per level or method."""
    out = pd.DataFrame({_level_label(level): table[f"{acc}|{level}"].to_numpy() for level in AE_COLUMNS})
    out.index = _index(table)
    return out


def _gradient(value: float, best: float, worst: float) -> str:
    """The colour of an AE between the green of the rank tables (``best``) and white (``worst``), linear in the AE."""
    if not np.isfinite(value):
        return ""
    alpha = 1.0 if np.isclose(best, worst) else float(np.clip((worst - value) / (worst - best), 0.0, 1.0))
    rgb = np.array(WHITE_RGB) * (1.0 - alpha) + np.array(BEST_RGB) * alpha
    return "#" + "".join(f"{round(float(c)):02x}" for c in rgb)


def ae_styles(frame: pd.DataFrame) -> pd.DataFrame:
    """The CSS of the cells of an AE table: per row, the five methods coloured from green (smallest AE) to white
    (largest), the smallest in bold; the reference without style."""
    styles = pd.DataFrame("", index=frame.index, columns=frame.columns)
    compared = [_level_label(level) for level in COMPARED]
    for i, (_, row) in enumerate(frame[compared].iterrows()):
        values = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).any():
            continue
        best, worst = np.nanmin(values), np.nanmax(values)
        for column, value in zip(compared, values):
            color = _gradient(value, best, worst)
            if not color:
                continue
            css = f"background-color: {color}; color: {main_tables._text_color(color)};"
            if np.isclose(value, best):
                css += " font-weight: bold;"
            styles.iloc[i, frame.columns.get_loc(column)] = css
    return styles


def split_frame(table: pd.DataFrame) -> pd.DataFrame:
    cols = {("", "$R$"): table["R"].to_numpy()}
    for level in METHODS:
        cols[(level, "$\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$")] = table[f"{level}|q_l1"].to_numpy()
        cols[(level, "RMS $e_\\pi$")] = table[f"{level}|e_pi_rms"].to_numpy()
    cols[("", "RMS $e_C$ (= oracle)")] = table["e_c_rms"].to_numpy()
    out = pd.DataFrame(cols)
    out.index = _index(table)
    return out


CHECK_COLUMNS = {
    "max_split_residual": "$\\max\\lvert e_\\pi+e_C-(\\hat A-A)\\rvert$",
    "max_e_c_diff": "$\\max\\lvert e_C-e_C^{\\text{oracle}}\\rvert$",
    "max_oracle_e_pi": "$\\max\\lvert e_\\pi^{\\text{oracle}}\\rvert$",
    "n_bound_violations": "bound violations",
    **{f"{level}|n_failed": f"failed {level}" for level in METHODS},
}


def check_frame(table: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({label: table[column].to_numpy() for column, label in CHECK_COLUMNS.items()})
    out.index = _index(table)
    return out


def _ae_styler(frame: pd.DataFrame):
    styles = ae_styles(frame)
    return frame.style.format(precision=3, na_rep="-").apply(lambda _: styles, axis=None)


def _latex(styler, caption: str | None = None, label: str | None = None) -> str:
    # convert_css turns the background colours and the bold of the AE tables into \cellcolor and \bfseries
    if caption is None:
        return booktabs_clines(
            styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data", convert_css=True)
        )
    latex = styler.to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=caption,
        label=label,
        convert_css=True,
    )
    return booktabs_clines(latex)


def ae_caption(acc: str) -> str:
    return (
        f"E8.2, {ACC_LABELS[acc]}: mean AE over the bags of the rate matrix of $V$ rescaled by a point quantifier "
        "(CTT-KDEy, CTT-EMQ, CTT-PACC), of the point estimates of \\bcapkdey{} and \\bcaphf{} stored by "
        "\\texttt{main} on the same bags, and of the reference with the true prevalence of the bag (CTT-oracle, not "
        "compared), against the realised table of each bag ($|U|=1000$). In each row, green for the smallest AE of the "
        "five methods to white for the largest; the smallest in bold. Mean rows: means of the per-pair values over the "
        "pairs of the block."
    )


SPLIT_CAPTION = (
    "E8.2, vanilla accuracy: range $R$ of the recalls of $h$ on $V$; per quantifier, mean quantification error and RMS "
    "over the bags of the quantification part $e_\\pi$; RMS of the rate part $e_C$, shared by every level and equal "
    "to the RMS of the error of the oracle."
)
CHECK_CAPTION = (
    "E8.2, checks of the split $\\hat A-A=e_\\pi+e_C$ (all expected $0$): largest residual over levels and bags, "
    "largest difference between the $e_C$ of a quantifier and that of the oracle, largest $\\lvert e_\\pi\\rvert$ of "
    "the oracle, bags beyond the bound $\\frac R2\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$, bags where a quantifier failed."
)
CHECK_FORMATS = {
    CHECK_COLUMNS["max_split_residual"]: "{:.1e}",
    CHECK_COLUMNS["max_e_c_diff"]: "{:.1e}",
    CHECK_COLUMNS["max_oracle_e_pi"]: "{:.1e}",
    **{label: "{:.0f}" for column, label in CHECK_COLUMNS.items() if column.endswith("n_failed")},
    CHECK_COLUMNS["n_bound_violations"]: "{:.0f}",
}


def _latex_document(longtables: list[str]) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,landscape,hmargin=1.5cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{amsmath,amssymb,booktabs,longtable,multirow}",
            "\\usepackage[table]{xcolor}",
            METHOD_LATEX_PREAMBLE,
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


def compute(
    results: dict | None = None, main_results: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The AE, split and check tables with the summary rows (AE and split) from the stored results of E8.2 and of the
    methods of ``main`` in ``MAIN_COMPARED``."""
    results = load_results() if results is None else results
    main_results = load_main_results() if main_results is None else main_results
    if all(df.empty for df in results.values()):
        raise RuntimeError(f"No results found for {EXPERIMENT}. Check that experiment results exist.")
    ae, split, check = pair_tables(results, main_results)
    return with_summaries(ae), with_summaries(split), check


def write_tables(
    results: dict, out_dir: str, main_results: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write the CSVs, the LaTeX fragments and the document of the five tables to ``out_dir``."""
    ae, split, check = compute(results, main_results)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    ae.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_ae.csv"), index=False)
    keys = ["dataset", "classifier", "n_classes", "n_bags", "n_pairs"]
    for acc in ACC_NAMES:
        columns = {f"{acc}|{level}": level for level in AE_COLUMNS}
        ae[keys + list(columns)].rename(columns=columns).to_csv(
            os.path.join(out_dir, f"{TABLE_NAME}_ae_{acc}.csv"), index=False
        )
    split.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_split.csv"), index=False)
    check.to_csv(os.path.join(out_dir, f"{TABLE_NAME}_check.csv"), index=False)

    stylers = {
        **{f"{TABLE_NAME}_ae_{acc}": (_ae_styler(ae_frame(ae, acc)), ae_caption(acc)) for acc in ACC_NAMES},
        f"{TABLE_NAME}_split": (split_frame(split).style.format(precision=3, na_rep="-"), SPLIT_CAPTION),
        f"{TABLE_NAME}_check": (
            check_frame(check).style.format(precision=3, na_rep="-", formatter=CHECK_FORMATS),
            CHECK_CAPTION,
        ),
    }
    longtables = []
    for name, (styler, caption) in stylers.items():
        with open(os.path.join(tabular_dir, f"{name}.tex"), "w") as f:
            f.write(_latex(styler))
        with open(os.path.join(tabular_dir, f"{name}_long.tex"), "w") as f:
            f.write(_latex(styler, caption=caption, label=f"tab:{name.replace('_', '-')}"))
        longtables.append(f"tabular/{name}_long.tex")
    with open(os.path.join(out_dir, f"{TABLE_NAME}_doc.tex"), "w") as f:
        f.write(_latex_document(longtables))
    return ae, split, check


def gen_tables(pargs):
    out_dir = table_dir(pargs)
    ae, split, check = write_tables(load_results(), out_dir, load_main_results())
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.precision", 4):
        for table in (ae, split, check):
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
    parser = ArgumentParser(description="E8.2: tables of the rate matrix of V rescaled by a point quantifier.")
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
