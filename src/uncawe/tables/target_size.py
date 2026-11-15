r"""The error split within the cells of the study ``target_size`` (work-plan card WP-21, E4.2b of
``docs/ext4_uncertainty_budget.md``; also read by E4.2a, card WP-22).

Inputs: the stored results of ``experiments/target_size.py`` (vanilla accuracy, every bag size, $|V|$ level and $V$
seed found, ``schema_version`` at least ``SCHEMA_VERSION``) for ``BayesCAP-Hf-real`` and ``BayCAP-KDEy-mlp-real``,
both scored against the realised table of each bag (``true_cts``), and the pool recalls $d^{\text{pool}}$ of each pair
(``tables/error_split.py:load_pool_info``). The split uses the true prevalence of the bag and the labels of the pool,
so it is an analysis, not a method.

Per bag (``uncawe.error_split.split``): the parts $e_1$, $e_2$, $e_3$ of the error of the method's mean table and the
total $\hat A-A$. A cell is (method, dataset, classifier, $|U|$, $|V|$ level, $V$ seed, prevalence), with $R$ bags;
within it each part splits into its fixed part $\bar e$ (mean over the bags) and its bag part $e^{(b)}-\bar e$
(§T.4, ``uncawe.error_split.cell_parts``). Bags with a non-finite mean table (NUTS produced no draws) are dropped and
counted.

Outputs, under ``<OUT_DIR>/uncawe/tables/target_size/``:

- ``target_size_bags.csv``: one row per kept bag, with its parts, fixed parts and bag parts;
- ``target_size_cells.csv``: one row per cell, with $R$, $|V|$, and per part the fixed part, the RMS of the bag parts
  and the standard error of the fixed part ($s/\sqrt R$, the caveat of §T.4);
- ``target_size_seeds.csv``: per (method, dataset, classifier, $|U|$, level, prevalence), the mean and the standard
  deviation over the $V$ seeds of each fixed part, next to its value at the full $V$;
- ``target_size_levels.csv``: per (method, dataset, classifier, $|U|$, level), per part, the RMS over the cells
  (prevalences and seeds) of the fixed parts, the RMS over the bags of the bag parts and the RMS over the cells of the
  standard errors of the fixed parts; and the fixed parts over the seeds, summarised across the three prevalences:
  the mean over the prevalences of the means over the seeds (``<part>_seed_mean``), the square root of the mean over
  the prevalences of the variances over the seeds (``<part>_seed_sd``, a pooled standard deviation) and the mean over
  the prevalences of the fixed part at the full $V$ (``<part>_full``);
- ``target_size_slopes.csv``: per (method, dataset, classifier, $|U|$) and part, the slope of
  $\log(\text{RMS of the fixed parts})$ against $\log|V|$ over the levels (least squares, descriptive; NaN with fewer
  than two levels);
- the table of E4.2b (``levels``): per method, dataset, classifier and level, $|V|$, the RMS of the fixed and of the bag
  parts of $e_1$, $e_2$, $e_3$, and the mean and the pooled standard deviation over the seeds of the fixed part of
  $e_2$ next to its value at the full $V$, for one bag size (``--size``, default $1000$); as a tabular fragment, a
  longtable and a standalone document. Descriptive: no colours, no bold, no test.
"""

import os
import subprocess
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd

from uncawe import env
from uncawe import error_split as es
from uncawe.experiments.target_size import DOMAIN, EXPERIMENT, LEVELS, METHODS, SCHEMA_VERSION
from uncawe.results import ResultsDataFrame
from uncawe.tables import eb_prior
from uncawe.tables import main as main_tables
from uncawe.tables.error_split import load_pool_info
from uncawe.tables.estimand_gap import booktabs_clines
from uncawe.util import decorate_dataset

TABLE_NAME = "target_size"
ACC_NAME = "vanilla_accuracy"
PARTS = ("e1", "e2", "e3")
SPLIT_PARTS = PARTS + ("total",)
LEVEL_KEYS = ["method", "dataset", "classifier", "bag_size", "v_level"]
CELL_KEYS = LEVEL_KEYS + ["v_seed", "prevalence_kind"]
# the columns of the stored rows kept in the per-bag table
BAG_COLUMNS = CELL_KEYS + ["n_classes", "v_size", "bag_index", "bag_seed", "config_hash"]
IDENTITY_TOL = 1e-9
SHORT_LABELS = {"BayesCAP-Hf-real": "BayCAP-Hf", "BayCAP-KDEy-mlp-real": "BayCAP-KDEy"}
PART_LABELS = {"e1": "$e_1$", "e2": "$e_2$", "e3": "$e_3$"}
CAPTION = (
    "Error split across $|V|$ (E4.2b), $|U|={size}$, vanilla accuracy, against the realised table of each bag. "
    "Fixed part: RMS over the cells (three prevalences, and the $V$ seeds) of the mean of each part over the $R$ bags "
    "of a cell; bag part: RMS over the bags of the deviation of each bag from the mean of its cell. Last group: the "
    "fixed part of $e_2$ across the $V$ seeds, mean over the prevalences of its mean over the seeds, pooled standard "
    "deviation over the seeds, and its value at the full $V$ (mean over the prevalences)."
)


# --------------------------------------------------------------------------------------------------- loading


def result_files(root: str | None = None, acc_name: str = ACC_NAME) -> list[str]:
    """The stored files of the study for ``acc_name``, under ``<root>/target_size/u*/v*/s*/<domain>/...``."""
    root = env.root_dir if root is None else root
    pattern = os.path.join(root, EXPERIMENT, "u*", "v*", "s*", DOMAIN, acc_name, "*", "*", "*.parquet")
    return sorted(glob(pattern))


def load_results(root: str | None = None, methods=METHODS) -> pd.DataFrame:
    """Every stored row of ``methods`` with ``schema_version >= SCHEMA_VERSION``, concatenated."""
    frames = []
    for path in result_files(root):
        df = ResultsDataFrame.load_result(path)
        if "schema_version" not in df.columns or not (df["schema_version"] >= SCHEMA_VERSION).all():
            continue
        frames.append(pd.DataFrame(df.loc[df["method"].isin(methods)]))
    if not frames:
        raise RuntimeError(f"No {EXPERIMENT} results found for {list(methods)}.")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------------------- computation


def _finite_table(value) -> bool:
    return value is not None and np.all(np.isfinite(np.asarray(value, dtype=float)))


def bag_table(results: pd.DataFrame, pool: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per kept bag, the parts of the split and their fixed and bag parts within its cell; and per cell the number of
    stored and dropped bags. ``pool`` maps (dataset, classifier) to its ``d_pool``."""
    kept_mask = results["estim_cts"].apply(_finite_table).to_numpy()
    counts = (
        results.assign(n_dropped=~kept_mask)
        .groupby(CELL_KEYS, sort=False)
        .agg(n_stored=("bag_index", "size"), n_dropped=("n_dropped", "sum"))
        .reset_index()
    )
    kept = results.loc[kept_mask].reset_index(drop=True)
    parts = []
    for (dataset, classifier), group in kept.groupby(["dataset", "classifier"], sort=False):
        d_pool = pool[(dataset, classifier)]["d_pool"]
        out = es.split(np.stack(group["estim_cts"].to_numpy()), np.stack(group["true_cts"].to_numpy()), d_pool)
        frame = group[BAG_COLUMNS].copy()
        for name in SPLIT_PARTS + ("identity_gap", "n_empty"):
            frame[name] = out[name]
        frame["estim_gap"] = np.abs(group["estim_accs"].to_numpy(dtype=float) - out["estimate"])
        parts.append(frame)
    bags = pd.concat(parts, ignore_index=True)
    cells = bags[CELL_KEYS].astype(str).agg("|".join, axis=1).to_numpy()
    for name in SPLIT_PARTS:
        bags[f"{name}_fixed"], bags[f"{name}_bag"] = es.cell_parts(bags[name].to_numpy(), cells)
    return bags, counts


def cell_table(bags: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: $R$ (kept bags), $|V|$, and per part the fixed part, the RMS of the bag parts and the standard
    error of the fixed part."""
    rows = []
    for key, group in bags.groupby(CELL_KEYS, sort=False):
        row = dict(zip(CELL_KEYS, key))
        row |= {
            "n_classes": int(group["n_classes"].iloc[0]),
            "v_size": int(group["v_size"].iloc[0]),
            "n_bags": len(group),
        }
        for name in SPLIT_PARTS:
            row[f"{name}_fixed"] = float(group[f"{name}_fixed"].iloc[0])
            row[f"{name}_bag_rms"] = es.rms(group[f"{name}_bag"])
            row[f"{name}_fixed_se"] = es.fixed_se(group[f"{name}_bag"])
        rows.append(row)
    return pd.DataFrame(rows).merge(counts, on=CELL_KEYS, how="left")


def seed_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, classifier, $|U|$, level, prevalence): the mean and the standard deviation over the $V$
    seeds of each fixed part, and the fixed part at the full $V$ of the same method, pair, size and prevalence."""
    keys = LEVEL_KEYS + ["prevalence_kind"]
    full_keys = ["method", "dataset", "classifier", "bag_size", "prevalence_kind"]
    full = cells.loc[cells["v_level"] == "full", full_keys + [f"{p}_fixed" for p in PARTS]]
    full = full.rename(columns={f"{p}_fixed": f"{p}_full" for p in PARTS})
    rows = []
    for key, group in cells.groupby(keys, sort=False):
        row = dict(zip(keys, key)) | {"v_size": int(group["v_size"].iloc[0]), "n_seeds": len(group)}
        for name in PARTS:
            row[f"{name}_seed_mean"], row[f"{name}_seed_sd"] = es.over_seeds(group[f"{name}_fixed"])
        rows.append(row)
    return pd.DataFrame(rows).merge(full, on=full_keys, how="left")


def level_table(bags: pd.DataFrame, cells: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, classifier, $|U|$, level): the RMS of the fixed parts over the cells, of the bag parts over
    the bags and of the standard errors over the cells, and the summaries over the seeds of ``seed_table``."""
    bag_groups = dict(list(bags.groupby(LEVEL_KEYS, sort=False)))
    seed_groups = dict(list(seeds.groupby(LEVEL_KEYS, sort=False)))
    rows = []
    for key, group in cells.groupby(LEVEL_KEYS, sort=False):
        row = dict(zip(LEVEL_KEYS, key))
        bag_rows, seed_rows = bag_groups[key], seed_groups[key]
        row |= {
            "n_classes": int(group["n_classes"].iloc[0]),
            "v_size": int(group["v_size"].iloc[0]),
            "n_seeds": int(group["v_seed"].nunique()),
            "n_cells": len(group),
            "n_bags": int(group["n_bags"].sum()),
            "n_dropped": int(group["n_dropped"].sum()),
        }
        for name in SPLIT_PARTS:
            row[f"{name}_fixed_rms"] = es.rms(group[f"{name}_fixed"])
            row[f"{name}_bag_rms"] = es.rms(bag_rows[f"{name}_bag"])
            row[f"{name}_fixed_se_rms"] = es.rms(group[f"{name}_fixed_se"].dropna())
        for name in PARTS:
            row[f"{name}_seed_mean"] = float(seed_rows[f"{name}_seed_mean"].mean())
            variances = seed_rows[f"{name}_seed_sd"].to_numpy(dtype=float) ** 2
            row[f"{name}_seed_sd"] = float(np.sqrt(np.mean(variances))) if np.isfinite(variances).all() else np.nan
            row[f"{name}_full"] = float(seed_rows[f"{name}_full"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def slope_table(levels: pd.DataFrame) -> pd.DataFrame:
    """Per (method, dataset, classifier, $|U|$) and part, the least-squares slope of $\\log$ RMS of the fixed parts
    against $\\log|V|$ over the levels with a positive RMS (NaN with fewer than two)."""
    keys = ["method", "dataset", "classifier", "bag_size"]
    rows = []
    for key, group in levels.groupby(keys, sort=False):
        row = dict(zip(keys, key)) | {"n_levels": len(group)}
        for name in PARTS:
            values = group[f"{name}_fixed_rms"].to_numpy(dtype=float)
            ok = np.isfinite(values) & (values > 0)
            x = np.log(group["v_size"].to_numpy(dtype=float)[ok])
            row[f"{name}_slope"] = float(np.polyfit(x, np.log(values[ok]), 1)[0]) if np.unique(x).size >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def shared_bag_check(bags: pd.DataFrame, results: pd.DataFrame) -> int:
    """The number of bags (dataset, classifier, $|U|$, prevalence, bag index) whose realised table differs across the
    methods, levels and seeds that stored it: $0$ when every cell of a prevalence and size sees the same bags."""
    keys = ["dataset", "classifier", "bag_size", "prevalence_kind", "bag_index"]
    tables = results[keys].copy()
    tables["table"] = results["true_cts"].apply(lambda t: np.asarray(t, dtype=float).tobytes())
    return int((tables.groupby(keys)["table"].nunique() > 1).sum())


def compute(results: pd.DataFrame | None = None, pool: dict | None = None) -> dict[str, pd.DataFrame]:
    results = load_results() if results is None else results
    pairs = sorted(set(zip(results["dataset"], results["classifier"])))
    pool = load_pool_info(pairs, domain=DOMAIN) if pool is None else pool
    bags, counts = bag_table(results, pool)
    cells = cell_table(bags, counts)
    seeds = seed_table(cells)
    levels = _ordered(level_table(bags, cells, seeds))
    return {
        "bags": bags,
        "cells": cells,
        "seeds": seeds,
        "levels": levels,
        "slopes": slope_table(levels),
        "n_unshared_bags": shared_bag_check(bags, results),
    }


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order by method (``METHODS``), dataset (increasing $K$, then as stored), classifier, size and level (full first)."""
    table = table.copy()
    datasets = table.drop_duplicates("dataset").sort_values("n_classes", kind="stable")["dataset"].tolist()
    table["method"] = pd.Categorical(table["method"], categories=list(METHODS), ordered=True)
    table["dataset"] = pd.Categorical(table["dataset"], categories=datasets, ordered=True)
    table["v_level"] = pd.Categorical(table["v_level"], categories=list(LEVELS), ordered=True)
    table = table.sort_values(["method", "dataset", "classifier", "bag_size", "v_level"]).reset_index(drop=True)
    for column in ("method", "dataset", "v_level"):
        table[column] = table[column].astype(str)
    return table


# --------------------------------------------------------------------------------------------------- tables


def _fmt(value) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{value:.4f}"


def levels_frame(levels: pd.DataFrame, size: int) -> pd.DataFrame:
    """The displayed table of E4.2b for bags of ``size`` instances (strings, MultiIndex rows and columns)."""
    table = levels.loc[levels["bag_size"] == size]
    if table.empty:
        raise RuntimeError(f"No {EXPERIMENT} results with |U|={size}.")
    columns = [("", "$|V|$")]
    columns += [("fixed part, RMS", PART_LABELS[p]) for p in PARTS]
    columns += [("bag part, RMS", PART_LABELS[p]) for p in PARTS]
    columns += [("fixed $e_2$ over the $V$ seeds", label) for label in ("mean", "sd", "full $V$")]
    rows, index = [], []
    for _, r in table.iterrows():
        values = [str(int(r["v_size"]))]
        values += [_fmt(r[f"{p}_fixed_rms"]) for p in PARTS]
        values += [_fmt(r[f"{p}_bag_rms"]) for p in PARTS]
        values += [_fmt(r["e2_seed_mean"]), _fmt(r["e2_seed_sd"]), _fmt(r["e2_full"])]
        rows.append(values)
        dataset = decorate_dataset(main_tables.dataset_map.get(r["dataset"], r["dataset"]))
        index.append((SHORT_LABELS.get(r["method"], r["method"]), dataset, r["classifier"], r["v_level"]))
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["method", "dataset", "classifier", "level"]),
        columns=pd.MultiIndex.from_tuples(columns),
    )


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def _stem(size: int) -> str:
    return f"{TABLE_NAME}_levels_u{size}"


def check_summary(out: dict) -> pd.DataFrame:
    """Per method: the largest identity gap $|e_1+e_2+e_3-(\\hat A-A)|$, the largest gap between the stored estimate
    and the trace of the mean table, the dropped and the empty-column bags, and the sizes $R$ of the cells."""
    rows = []
    for method, group in out["bags"].groupby("method", sort=False):
        cells = out["cells"].loc[out["cells"]["method"] == method]
        rows.append(
            {
                "method": method,
                "max_identity_gap": float(group["identity_gap"].max()),
                "max_estimate_gap": float(group["estim_gap"].max()),
                "dropped_bags": int(cells["n_dropped"].sum()),
                "empty_column_bags": int((group["n_empty"] > 0).sum()),
                "cells": len(cells),
                "R_values": sorted(set(cells["n_stored"].astype(int))),
                "config_hashes": sorted(set(group["config_hash"])),
            }
        )
    return pd.DataFrame(rows)


def gen_tables(pargs):
    out = compute()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    for name in ("bags", "cells", "seeds", "levels", "slopes"):
        out[name].to_csv(os.path.join(out_dir, f"{TABLE_NAME}_{name}.csv"), index=False)

    styler = levels_frame(out["levels"], pargs.size).style
    label = f"tab:{TABLE_NAME}-levels-u{pargs.size}"
    stem = _stem(pargs.size)
    with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
        f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
    with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
        f.write(eb_prior._latex_longtable(styler, CAPTION.format(size=pargs.size), label))
    with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
        f.write(eb_prior._latex_document(f"tabular/{stem}_long.tex"))

    checks = check_summary(out)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.precision", 4):
        print(f"\nchecks:\n{checks.to_string(index=False)}")
    print(f"bags whose realised table differs across the cells that share it: {out['n_unshared_bags']}")
    if (checks["max_identity_gap"] > IDENTITY_TOL).any():
        print(f"WARNING: identity gap above {IDENTITY_TOL:g} (see checks)")
    print(f"cells: {len(out['cells'])}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    tex_filename = f"{_stem(pargs.size)}_doc.tex"
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
    parser = ArgumentParser(description="The error split within the cells of the study target_size (E4.2b, WP-21).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--size", type=int, default=1000, help="Bag size |U| of the LaTeX table (default: 1000)")
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
