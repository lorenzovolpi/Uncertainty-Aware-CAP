r"""The accuracy and the intervals of six CAP methods across $|U|$ and $|V|$ in the study ``target_size`` (work-plan
card WP-28, E4.4 of ``docs/ext4_uncertainty_budget.md``, §T.7).

Inputs: the stored results of ``experiments/target_size.py`` for the methods ``E44_METHODS``, the classifier LR and
the levels of E4.4 (``E44_LEVELS``), vanilla accuracy, ``schema_version`` at least ``SCHEMA_VERSION``; every dataset
found (the $15$ of ``main`` once both parts have run; ``--datasets`` restricts them). Part a (E4.4a) is the set of
cells with the full $V$, at every $|U|$ found; part b (E4.4b) the set of cells at $|U|=1000$, at every level found.
Every method targets the realised table of each bag and is scored against it (``true_cts``).

Per bag: the realised accuracy $A(h,U)$ (``acc_from_ct`` of the realised table, as in ``tables/main.py``), the absolute
error $\text{AE}=|\hat A-A(h,U)|$ (``acc_err``), and, from the interval of the stored accuracy draws at the level of
``main`` ($1-\alpha=0.95$; ``tables/main.py:get_acc_*``), the coverage $\mathbb 1[l\le A\le u]$, the amplitude $u-l$
and the Winkler score $W=(u-l)+\frac2\alpha\left[(l-A)^++(A-u)^+\right]$; as a diagnostic, the parts $e_1$, $e_2$,
$e_3$ of the error of the mean table (``uncawe.error_split.split``, with the pool recalls of the pair). A bag whose
estimate, interval or mean table is not finite (e.g. NUTS produced no draws) is dropped and counted.

Outputs, under ``<OUT_DIR>/uncawe/tables/target_size/``:

- ``target_size_scores_bags.csv``: one row per kept bag, with its scores and parts;
- ``target_size_scores_seeds.csv``: per (method, dataset, $|U|$, level, $V$ seed), the means over the kept bags (three
  prevalences, $R$ bags each) of AE, $W$, coverage, amplitude and the parts, the bags kept and dropped, $|V|$, the
  smallest class count of the subsample and the flag ``below_threshold`` (a class with fewer than ``MIN_CLASS_COUNT``
  instances, E4.4b);
- ``target_size_scores_pairs.csv``: per (method, dataset, $|U|$, level), the mean and the standard deviation over the
  $V$ seeds of the per-seed means (one seed at the full $V$, standard deviation NaN), ``below_threshold`` if any seed
  is flagged;
- ``target_size_scores_summary.csv``: per part (``a``, ``b``), subset (``all``; ``without poker_hand``, the check of
  E4.4b), method and size (``x``: $|U|$ in part a, the fraction of $V$ in part b), the mean, its standard error
  (standard deviation over the pairs, ``ddof=1``, divided by the square root of the number of pairs; NaN with one
  pair), the median and the 25th and 75th percentiles over the pairs of the per-pair means of AE, $W$, coverage and
  amplitude, the number of pairs and of the flagged ones;
- ``target_size_scores_slopes.csv``: per part, method and dataset with two or more sizes, the least-squares slopes of
  $\log$ mean AE and of $\log$ mean amplitude against $\log|U|$ (part a) or $\log$ of the fraction of $V$ (part b),
  descriptive;
- ``target_size_scores_diagnostics.csv`` and ``target_size_scores_cost.csv``: the NUTS diagnostics and the times per
  method, dataset, size and level (``tables/target_size.py:diagnostics_table`` and ``cost_table``; the methods without
  NUTS count $0$);
- per part, the table of E4.4 (per pair, method and size: mean AE, $W$, coverage, amplitude; in part b also their
  standard deviations over the seeds), as a tabular fragment, a longtable and a standalone document.

The tables are descriptive: no colours, no bold, no test.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe import error_split as es
from uncawe.experiments.metrics import acc_from_ct
from uncawe.experiments.target_size import (
    DOMAIN,
    E44_CLASSIFIERS,
    E44_LEVELS,
    E44_METHODS,
    EXPERIMENT,
    MIN_CLASS_COUNT,
    SAMPLE_SIZE,
    SCHEMA_VERSION,
)
from uncawe.results import ResultsDataFrame
from uncawe.tables import main as main_tables
from uncawe.tables.error_split import load_pool_info
from uncawe.tables.target_size import _write_latex, cost_table, diagnostics_table, result_files
from uncawe.util import decorate_dataset

TABLE_NAME = "target_size_scores"
ACC_NAME = "vanilla_accuracy"
PARTS = ("e1", "e2", "e3")
SCORES = ("ae", "winkler", "coverage", "amplitude")
SEED_KEYS = ["method", "dataset", "classifier", "n_classes", "bag_size", "v_level", "v_fraction", "v_seed"]
PAIR_KEYS = [key for key in SEED_KEYS if key != "v_seed"]
CSV_NAMES = ("bags", "seeds", "pairs", "summary", "slopes", "diagnostics", "cost")
# the parts of E4.4: the column that varies, and the filter of its cells
EXPERIMENT_PARTS = {"a": "bag_size", "b": "v_fraction"}
# the dataset left out in the check of E4.4b (its near-vertex errors could move the multiclass pairs)
CHECK_DATASET = "poker_hand"
# BayCAP-KDEy is BayCAP-KDEy-h-inst since E3.4 (card WP-31); the former one keeps its full name (card WP-33)
SHORT_LABELS = {
    "BayesCAP-Hf-real": "BayCAP-Hf",
    "BayCAP-KDEy-h-inst": "BayCAP-KDEy",
    "BayCAP-KDEy-h-inst-cb": "BayCAP-KDEy-cb",
    "BayCAP-KDEy-mlp-real": "BayCAP-KDEy-mlp-real",
    # WP-34
    "BayesCAP-Hf-real-tcal": "BayCAP-Hf-tcal",
    "BayCAP-KDEy-h-inst-cb-tcal-keep": "BayCAP-KDEy-cb-tcal-keep",
    "BayCAP-KDEy-h-inst-cb-tcal-resel": "BayCAP-KDEy-cb-tcal-resel",
    # card WP-36: the SQS methods with KDEy get the plain labels, the PACC versions name their quantifier
    "SQS-H": "SQS-H-PACC",
    "SQS-Hbs": "SQS-Hbs-PACC",
    "PQ": "SQS-S-PACC",
    "SQS-H-KDEy": "SQS-H",
    "SQS-Hbs-KDEy": "SQS-Hbs",
    "PQ-KDEy": "SQS-S",
}
SCORE_LABELS = {"ae": "AE", "winkler": "$W$", "coverage": "coverage", "amplitude": "amplitude"}
CAPTIONS = {
    "a": (
        "Accuracy and intervals across $|U|$ (E4.4a), full $V$, LR, vanilla accuracy, against the realised accuracy "
        "of each bag. Per pair, method and $|U|$: the mean over the bags (three prevalences, $R$ bags each) of the "
        "absolute error (AE), of the Winkler score $W$ at $1-\\alpha=0.95$, of the coverage and of the amplitude."
    ),
    "b": (
        "Accuracy and intervals across $|V|$ (E4.4b), $|U|=1000$, LR, vanilla accuracy, against the realised accuracy "
        "of each bag. Per pair, method and level of $V$: the mean over the $V$ seeds of the mean over the bags (three "
        "prevalences, $R$ bags each) of AE, $W$ at $1-\\alpha=0.95$, coverage and amplitude, and the standard "
        "deviation over the seeds (sd). $^*$: a class of the subsample keeps fewer than "
        f"{MIN_CLASS_COUNT} instances."
    ),
}


# --------------------------------------------------------------------------------------------------- loading


def load_results(
    root: str | None = None, methods=E44_METHODS, classifiers=E44_CLASSIFIERS, datasets=None
) -> pd.DataFrame:
    """Every stored row of ``methods`` and ``classifiers`` at the levels of E4.4, with ``schema_version >=
    SCHEMA_VERSION``, concatenated; ``datasets`` restricts the datasets (default: every one found)."""
    frames = []
    for path in result_files(root, ACC_NAME):
        df = ResultsDataFrame.load_result(path)
        if "schema_version" not in df.columns or not (df["schema_version"] >= SCHEMA_VERSION).all():
            continue
        keep = df["method"].isin(methods) & df["classifier"].isin(classifiers) & df["v_level"].isin(list(E44_LEVELS))
        if datasets is not None:
            keep &= df["dataset"].isin(datasets)
        frames.append(pd.DataFrame(df.loc[keep]))
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError(f"No {EXPERIMENT} results found for {list(methods)} at the levels of E4.4.")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------------- computation


def _finite(value) -> bool:
    if value is None:
        return False
    array = np.asarray(value, dtype=float)
    return array.size > 0 and bool(np.all(np.isfinite(array)))


def bag_scores(results: pd.DataFrame, pool: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per kept bag, the realised accuracy, AE, $W$, coverage, amplitude and (with ``pool``, which maps (dataset,
    classifier) to its ``d_pool``) the parts $e_1$, $e_2$, $e_3$; and per (method, dataset, size, level, seed) the bags
    stored and dropped."""
    kept_mask = (
        results["estim_accs"].apply(_finite)
        & results["estim_acc_ranges"].apply(_finite)
        & results["estim_cts"].apply(_finite)
    ).to_numpy()
    counts = (
        results.assign(n_dropped=~kept_mask)
        .groupby(SEED_KEYS, sort=False)
        .agg(n_stored=("bag_index", "size"), n_dropped=("n_dropped", "sum"))
        .reset_index()
    )
    kept = results.loc[kept_mask].reset_index(drop=True)
    columns = SEED_KEYS + ["prevalence_kind", "bag_index", "v_size", "v_class_counts", "config_hash"]
    frame = kept[columns].copy()
    frame["true_acc"] = [float(acc_from_ct(ACC_NAME, np.asarray(ct, dtype=float))) for ct in kept["true_cts"]]
    frame["estim_acc"] = kept["estim_accs"].to_numpy(dtype=float)
    frame["ae"] = kept["acc_err"].to_numpy(dtype=float)
    frame["coverage"] = [
        main_tables.get_acc_coverage(m, a, r)
        for m, a, r in zip(kept["method"], frame["true_acc"], kept["estim_acc_ranges"])
    ]
    frame["amplitude"] = [main_tables.get_acc_amplitude(m, r) for m, r in zip(kept["method"], kept["estim_acc_ranges"])]
    frame["winkler"] = [
        main_tables.get_acc_winkler(m, a, r)
        for m, a, r in zip(kept["method"], frame["true_acc"], kept["estim_acc_ranges"])
    ]
    frame["v_min_class_count"] = [int(np.min(c)) for c in kept["v_class_counts"]]
    frame["below_threshold"] = frame["v_min_class_count"] < MIN_CLASS_COUNT
    for name in PARTS:
        frame[name] = np.nan
    if pool is not None:
        for (dataset, classifier), index in frame.groupby(["dataset", "classifier"], sort=False).groups.items():
            out = es.split(
                np.stack(kept.loc[index, "estim_cts"].to_numpy()),
                np.stack(kept.loc[index, "true_cts"].to_numpy()),
                pool[(dataset, classifier)]["d_pool"],
            )
            for name in PARTS:
                frame.loc[index, name] = out[name]
    return frame.drop(columns="v_class_counts"), counts


def seed_table(bags: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, size, level, seed): the means over the kept bags of the scores and parts, $|V|$, the
    smallest class count and the flag, with the bags stored and dropped."""
    means = bags.groupby(SEED_KEYS, sort=False)[list(SCORES + PARTS)].mean().reset_index()
    first = (
        bags.groupby(SEED_KEYS, sort=False)[["v_size", "v_min_class_count", "below_threshold"]].first().reset_index()
    )
    n_bags = bags.groupby(SEED_KEYS, sort=False).size().rename("n_bags").reset_index()
    out = means.merge(first, on=SEED_KEYS).merge(n_bags, on=SEED_KEYS)
    return counts.merge(out, on=SEED_KEYS, how="left")


def pair_table(seeds: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, size, level): the mean and the standard deviation (``ddof=1``; NaN with one seed) over the
    $V$ seeds of the per-seed means, the seeds, the bags, the dropped bags and whether a seed is flagged."""
    rows = []
    for key, group in seeds.groupby(PAIR_KEYS, sort=False):
        row = dict(zip(PAIR_KEYS, key)) | {
            "n_seeds": len(group),
            "v_size_mean": float(group["v_size"].mean()),
            "v_min_class_count": int(group["v_min_class_count"].min()),
            "below_threshold": bool(group["below_threshold"].fillna(False).any()),
            "n_bags": int(group["n_bags"].fillna(0).sum()),
            "n_dropped": int(group["n_dropped"].sum()),
        }
        for name in SCORES + PARTS:
            values = group[name].to_numpy(dtype=float)
            row[name] = float(np.mean(values))
            row[f"{name}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def part_rows(pairs: pd.DataFrame, part: str, size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """The rows of ``pairs`` of a part of E4.4, with the column ``x`` that varies: part ``a``, the full $V$ at every
    $|U|$; part ``b``, $|U|=$ ``size`` at every level (``x`` the fraction of $V$)."""
    if part == "a":
        rows = pairs.loc[pairs["v_level"] == "full"].copy()
    elif part == "b":
        rows = pairs.loc[pairs["bag_size"] == size].copy()
    else:
        raise ValueError(f"Unknown part {part}")
    rows["x"] = rows[EXPERIMENT_PARTS[part]].astype(float)
    return rows


def summary_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per part, subset, method and ``x``: the mean, its standard error (``ddof=1``; NaN with one pair), the median and
    the 25th and 75th percentiles over the pairs of the per-pair means of the scores, the number of pairs and of the
    flagged ones."""
    rows = []
    for part in EXPERIMENT_PARTS:
        frame = part_rows(pairs, part)
        subsets = (("all", frame), (f"without {CHECK_DATASET}", frame.loc[frame["dataset"] != CHECK_DATASET]))
        for subset, sub in subsets:
            for (method, x), group in sub.groupby(["method", "x"], sort=False):
                row = {
                    "part": part,
                    "subset": subset,
                    "method": method,
                    "x": x,
                    "n_pairs": len(group),
                    "n_flagged": int(group["below_threshold"].sum()),
                }
                for name in SCORES:
                    values = group[name].astype(float)
                    row |= {
                        f"{name}_mean": float(values.mean()),
                        f"{name}_se": float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan,
                        f"{name}_median": float(values.median()),
                        f"{name}_q25": float(values.quantile(0.25)),
                        f"{name}_q75": float(values.quantile(0.75)),
                    }
                rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["method"] = pd.Categorical(out["method"], categories=list(E44_METHODS), ordered=True)
    out = out.sort_values(["part", "subset", "method", "x"], kind="stable").reset_index(drop=True)
    out["method"] = out["method"].astype(str)
    return out


def _loglog_slope(x, y) -> float:
    """The least-squares slope of $\\log y$ against $\\log x$ over the points with a finite, positive $y$; NaN with
    fewer than two distinct $x$."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(y) & (y > 0)
    if np.unique(x[ok]).size < 2:
        return np.nan
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def slope_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per part, method and dataset with two or more values of ``x``: the slopes of $\\log$ mean AE and of $\\log$ mean
    amplitude against $\\log x$ (descriptive)."""
    rows = []
    for part in EXPERIMENT_PARTS:
        frame = part_rows(pairs, part)
        for (method, dataset), group in frame.groupby(["method", "dataset"], sort=False):
            if group["x"].nunique() < 2:
                continue
            rows.append(
                {
                    "part": part,
                    "method": method,
                    "dataset": dataset,
                    "n_classes": int(group["n_classes"].iloc[0]),
                    "n_x": int(group["x"].nunique()),
                    "ae_slope": _loglog_slope(group["x"], group["ae"]),
                    "amplitude_slope": _loglog_slope(group["x"], group["amplitude"]),
                }
            )
    columns = ["part", "method", "dataset", "n_classes", "n_x", "ae_slope", "amplitude_slope"]
    return pd.DataFrame(rows, columns=columns)


def compute(
    results: pd.DataFrame | None = None, pool: dict | None = None, split: bool = True
) -> dict[str, pd.DataFrame]:
    """Every table of the module; the pool recalls are loaded only when ``pool`` is not given and ``split`` holds
    (without them the parts are NaN)."""
    results = load_results() if results is None else results
    if pool is None and split:
        pairs = sorted(set(zip(results["dataset"], results["classifier"])))
        pool = load_pool_info(pairs, domain=DOMAIN)
    bags, counts = bag_scores(results, pool)
    seeds = seed_table(bags, counts)
    pairs = _ordered(pair_table(seeds))
    return {
        "bags": bags,
        "seeds": seeds,
        "pairs": pairs,
        "summary": summary_table(pairs),
        "slopes": slope_table(pairs),
        "diagnostics": diagnostics_table(results),
        "cost": cost_table(results),
    }


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order by method (``E44_METHODS``), dataset (increasing $K$, then by name), size and level (full first)."""
    table = table.copy()
    datasets = table.drop_duplicates("dataset").sort_values(["n_classes", "dataset"], kind="stable")["dataset"]
    table["method"] = pd.Categorical(table["method"], categories=list(E44_METHODS), ordered=True)
    table["dataset"] = pd.Categorical(table["dataset"], categories=datasets.tolist(), ordered=True)
    table = table.sort_values(["method", "dataset", "bag_size", "v_fraction"], ascending=[True, True, True, False])
    table = table.reset_index(drop=True)
    for column in ("method", "dataset"):
        table[column] = table[column].astype(str)
    return table


# --------------------------------------------------------------------------------------------------- tables


def _fmt(value, digits: int = 4) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def part_frame(pairs: pd.DataFrame, part: str) -> pd.DataFrame:
    """The displayed table of a part of E4.4 (strings, MultiIndex rows and columns): per pair, method and size, the
    mean AE, $W$, coverage and amplitude; in part b also their standard deviations over the seeds, and a $^*$ on the
    flagged levels."""
    table = part_rows(pairs, part)
    if table.empty or table["x"].nunique() < 2:
        raise RuntimeError(f"No {EXPERIMENT} results of E4.4{part} at two or more sizes.")
    columns = [("", "$|V|$")]
    columns += [("mean", SCORE_LABELS[name]) for name in SCORES]
    if part == "b":
        columns += [("sd over the seeds", SCORE_LABELS[name]) for name in SCORES]
    rows, index = [], []
    for _, r in table.iterrows():
        values = [str(round(r["v_size_mean"]))]
        values += [_fmt(r[name]) for name in SCORES]
        if part == "b":
            values += [_fmt(r[f"{name}_sd"]) for name in SCORES]
        rows.append(values)
        dataset = decorate_dataset(main_tables.dataset_map.get(r["dataset"], r["dataset"]))
        size = str(int(r["bag_size"])) if part == "a" else r["v_level"] + ("$^*$" if r["below_threshold"] else "")
        index.append((SHORT_LABELS.get(r["method"], r["method"]), dataset, size))
    names = ["method", "dataset", "$|U|$" if part == "a" else "level"]
    return pd.DataFrame(
        rows, index=pd.MultiIndex.from_tuples(index, names=names), columns=pd.MultiIndex.from_tuples(columns)
    )


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", EXPERIMENT)


def stem(part: str) -> str:
    return f"{TABLE_NAME}_e44{part}"


def check_summary(out: dict) -> pd.DataFrame:
    """Per method: the bags kept and dropped, the cells, the flagged cells, the values of $R$ and the config hashes."""
    rows = []
    for method, group in out["seeds"].groupby("method", sort=False):
        bags = out["bags"].loc[out["bags"]["method"] == method]
        rows.append(
            {
                "method": method,
                "kept_bags": int(group["n_bags"].fillna(0).sum()),
                "dropped_bags": int(group["n_dropped"].sum()),
                "cells": len(group),
                "flagged_cells": int(group["below_threshold"].fillna(False).sum()),
                "R_values": sorted(set(group["n_stored"].astype(int))),
                "config_hashes": sorted(set(bags["config_hash"])),
            }
        )
    return pd.DataFrame(rows)


def gen_tables(pargs):
    out = compute(load_results(datasets=pargs.datasets))
    out_dir = table_dir(pargs)
    os.makedirs(out_dir, exist_ok=True)
    for name in CSV_NAMES:
        out[name].to_csv(os.path.join(out_dir, f"{TABLE_NAME}_{name}.csv"), index=False)
    for part in EXPERIMENT_PARTS:
        try:
            frame = part_frame(out["pairs"], part)
        except RuntimeError as error:
            print(f"the table of E4.4{part} is not written: {error}")
            continue
        _write_latex(frame, out_dir, stem(part), CAPTIONS[part], f"tab:{TABLE_NAME}-e44{part}")
    checks = check_summary(out)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        print(f"\nchecks:\n{checks.to_string(index=False)}")
    print(f"cells: {len(out['seeds'])}; tables written to {out_dir}")


def gen_pdf(pargs):
    """Compile the documents of the tables of E4.4a and E4.4b, those that exist."""
    out_dir = table_dir(pargs)
    tex_filenames = [f"{stem(part)}_doc.tex" for part in EXPERIMENT_PARTS]
    tex_filenames = [name for name in tex_filenames if os.path.exists(os.path.join(out_dir, name))]
    if not tex_filenames:
        raise FileNotFoundError(f"No table document in {out_dir}. Generate tables first.")
    for tex_filename in tex_filenames:
        # longtable needs a second pass to settle the column widths
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                cwd=out_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Accuracy and intervals across |U| and |V| in the study target_size (E4.4).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict the datasets (default: every one found)")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSVs, table documents and tabular fragments are saved",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    pargs = parse_args()
    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
