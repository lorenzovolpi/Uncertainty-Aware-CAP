r"""The gate of the solver of E7.2a (``docs/ext7_kdey_bandwidth.md``; work-plan card WP-26): EM against the standard
KDEy, QuaPy's ``KDEyML`` (SLSQP), in the selection of the bandwidth of ``BayCAP-KDEy-mlp-real-cal``.

Reads the rows of ``experiments/kdey_cal_solver.py`` (one per pair, bandwidth of the grid and calibration bag) and,
from the stored ``main`` results of the pilot, the test time and the EM criterion curve of the method. Label-free on the
target: only the labels of $V$ enter the selection.

Per pair (``pair_table``):

- $b_{\text{cal}}$ of EM and of SLSQP, the grid steps between them, and the tie gap: the largest, over the two criterion
  curves, of the difference between the curve at the two selected bandwidths ($0$ if they coincide);
- the largest absolute difference of the two criterion curves over the grid;
- the largest $\lVert\hat\pi^{\text{EM}}-\hat\pi^{\text{SLSQP}}\rVert_1$ over the fits where EM converged and over
  those where it did not; the fits where EM did not converge and where SLSQP did not report success;
- the fits where the log-likelihood of the bag at one estimate exceeds that at the other by more than ``LL_TOL``, each
  way, and the largest such excess, each way;
- the times (seconds): of EM and of SLSQP over the grid, of the shared steps (the split, the MLP on $V_{\text{fit}}$,
  the bags, the class densities), of the deployed fit, and of the target bags of the pilot ($100\times$ the stored
  ``t_test_ave``); the run time of the method with EM, their sum; the extra time of SLSQP, its time minus that of EM;
- the largest difference between the EM curve of the gate and the curve stored by the pilot ($0$: the gate reproduces
  the pilot's selection).

The gate (``gate``), with the thresholds decided by the user on 2026-11-20: (a) on every pair the two solvers select
the same bandwidth, or the tie gap is below ``TIE_TOL`` ($10^{-3}$); (b) the extra time of SLSQP, summed over the
pairs, is at most ``TIME_SHARE`` ($10\%$) of the run time of the method with EM, summed over the same pairs.

Outputs, under ``<OUT_DIR>/uncawe/tables/kdey_cal_solver/``: ``kdey_cal_solver_pairs.csv``,
``kdey_cal_solver_gate.csv``, and the per-pair table as a LaTeX tabular fragment, a longtable and a standalone document.
"""

import glob
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.kdey_cal_solver import ACC_FOLDER, DOMAIN, EXPERIMENT, METHOD
from uncawe.methods.baycap_kdey import CAL_REFERENCE
from uncawe.tables import baycap_h, eb_prior, tempering
from uncawe.tables.displacement import ordered
from uncawe.tables.estimand_gap import booktabs_clines
from uncawe.tables.kdey_cal import GRID_STEP

TABLE_NAME = "kdey_cal_solver"
KEYS = ["dataset", "classifier"]
# the thresholds of the gate (user, 2026-11-20)
TIE_TOL = 1e-3
TIME_SHARE = 0.10
# a log-likelihood larger by more than this counts as a better maximum
LL_TOL = 1e-6
# the metric whose stored rows give the test time and the pilot curve (the selection does not depend on the metric)
STORED_ACC = "vanilla_accuracy"


def load_rows(root: str | None = None) -> pd.DataFrame:
    """The stored rows of the gate, empty if none."""
    base = os.path.join(env.root_dir if root is None else root, EXPERIMENT, DOMAIN, ACC_FOLDER)
    paths = sorted(glob.glob(os.path.join(base, "*", "*", f"{METHOD}.parquet")))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def stored_pilot(root: str | None = None) -> pd.DataFrame:
    """Per pair, from the stored ``main`` rows of the method: the time of its target bags ($n$ bags times
    ``t_test_ave``) and its stored criterion curve."""
    df = tempering.load_method(METHOD, STORED_ACC, root)
    if df.empty:
        return pd.DataFrame(columns=[*KEYS, "test_seconds", "pilot_curve"])
    rows = []
    for (dataset, classifier), group in df.groupby(KEYS, sort=False):
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "test_seconds": float(len(group) * group["t_test_ave"].iloc[0]),
                "pilot_curve": np.asarray(group["cal_curve"].iloc[0], dtype=float) if "cal_curve" in group else None,
            }
        )
    return pd.DataFrame(rows)


def _steps(a: float, b: float) -> float:
    return float(np.round(np.log10(a / b) / GRID_STEP, 6))


def _curve(group: pd.DataFrame, solver: str) -> np.ndarray:
    return group.groupby("grid_index")[f"{solver}_criterion"].first().sort_index().to_numpy()


def pair_row(group: pd.DataFrame) -> dict:
    """The measures of one pair (see the module docstring)."""
    first = group.iloc[0]
    grid = group.groupby("grid_index")["bandwidth"].first().sort_index().to_numpy()
    curves = {solver: _curve(group, solver) for solver in ("em", "slsqp")}
    b_em, b_slsqp = float(first["em_bandwidth_cal"]), float(first["slsqp_bandwidth_cal"])
    at = {b: int(np.flatnonzero(np.isclose(grid, b))[0]) for b in (b_em, b_slsqp)}
    tie_gap = max(abs(curve[at[b_slsqp]] - curve[at[b_em]]) for curve in curves.values())
    em = np.stack(group["em_estimate"].to_numpy())
    slsqp = np.stack(group["slsqp_estimate"].to_numpy())
    l1 = np.abs(em - slsqp).sum(axis=1)
    converged = group["em_converged"].to_numpy(dtype=bool)
    ll_gap = (group["slsqp_loglik"] - group["em_loglik"]).to_numpy()
    seconds = group.groupby("grid_index")[["em_seconds", "slsqp_seconds", "densities_seconds"]].first().sum()
    return {
        "dataset": first["dataset"],
        "classifier": first["classifier"],
        "n_classes": int(first["n_classes"]),
        "n_val": int(first["n_val"]),
        "n_cal": int(first["n_cal"]),
        "b_cal_em": b_em,
        "b_cal_slsqp": b_slsqp,
        "steps": _steps(b_slsqp, b_em),
        "tie_gap": float(tie_gap),
        "same_selection": bool(np.isclose(b_em, b_slsqp) or tie_gap < TIE_TOL),
        "curve_max_diff": float(np.max(np.abs(curves["em"] - curves["slsqp"]))),
        "default_diff": float(first["slsqp_default"] - first["em_default"]),
        "fits": len(group),
        "em_unconverged": int(np.sum(~converged)),
        "slsqp_failed": int(np.sum(~group["slsqp_converged"].to_numpy(dtype=bool))),
        "l1_max_converged": float(l1[converged].max()) if converged.any() else np.nan,
        "l1_max_unconverged": float(l1[~converged].max()) if (~converged).any() else np.nan,
        "ll_slsqp_better": int(np.sum(ll_gap > LL_TOL)),
        "ll_em_better": int(np.sum(ll_gap < -LL_TOL)),
        "ll_slsqp_max_gain": float(max(ll_gap.max(), 0.0)),
        "ll_em_max_gain": float(max(-ll_gap.min(), 0.0)),
        "em_seconds": float(seconds["em_seconds"]),
        "slsqp_seconds": float(seconds["slsqp_seconds"]),
        "shared_seconds": float(first["setup_seconds"] + seconds["densities_seconds"]),
        "refit_seconds": float(first["refit_seconds"]),
        "em_curve": curves["em"],
    }


def pair_table(root: str | None = None) -> pd.DataFrame:
    """One row per pair stored by the gate, with the run time and the comparison with the pilot's curve."""
    rows = load_rows(root)
    if rows.empty:
        return pd.DataFrame()
    table = pd.DataFrame([pair_row(group) for _, group in rows.groupby(KEYS, sort=False)])
    table = table.merge(stored_pilot(root), on=KEYS, how="left")
    table["pilot_curve_max_diff"] = [
        float(np.max(np.abs(em - pilot))) if isinstance(pilot, np.ndarray) and len(pilot) == len(em) else np.nan
        for em, pilot in zip(table["em_curve"], table["pilot_curve"], strict=True)
    ]
    table["run_seconds"] = (
        table["shared_seconds"] + table["em_seconds"] + table["refit_seconds"] + table["test_seconds"]
    )
    table["extra_seconds"] = table["slsqp_seconds"] - table["em_seconds"]
    table["extra_share"] = table["extra_seconds"] / table["run_seconds"]
    table = table.drop(columns=["em_curve", "pilot_curve"])
    return ordered(table)


def gate(table: pd.DataFrame) -> dict:
    """The verdict of the gate: (a) the selections, (b) the time, and both."""
    complete = table["run_seconds"].notna()
    extra, run = float(table["extra_seconds"].sum()), float(table.loc[complete, "run_seconds"].sum())
    share = extra / run if complete.all() and run > 0 else np.nan
    same = bool(table["same_selection"].all())
    fast = bool(share <= TIME_SHARE) if np.isfinite(share) else False
    return {
        "pairs": len(table),
        "pairs_same_b": int(np.isclose(table["b_cal_em"], table["b_cal_slsqp"]).sum()),
        "pairs_same_selection": int(table["same_selection"].sum()),
        "max_tie_gap": float(table["tie_gap"].max()),
        "em_seconds": float(table["em_seconds"].sum()),
        "slsqp_seconds": float(table["slsqp_seconds"].sum()),
        "extra_seconds": extra,
        "run_seconds": run,
        "pairs_without_run_time": int((~complete).sum()),
        "extra_share": share,
        "tie_tol": TIE_TOL,
        "time_share": TIME_SHARE,
        "a_selection": same,
        "b_time": fast,
        "passed": same and fast,
    }


# --------------------------------------------------------------------------------------------------- LaTeX


def _fmt(value: float, digits: int = 3) -> str:
    return "--" if pd.isna(value) else f"{value:.{digits}f}"


def _pairs_styler(table: pd.DataFrame):
    out = pd.DataFrame(
        {
            "$b$ EM": table["b_cal_em"].map(_fmt).to_numpy(),
            "$b$ SLSQP": table["b_cal_slsqp"].map(_fmt).to_numpy(),
            "steps": table["steps"].map(lambda v: _fmt(v, 0)).to_numpy(),
            "tie gap": table["tie_gap"].map(lambda v: _fmt(v, 4)).to_numpy(),
            "max $\\lvert\\Delta\\mathrm{crit}\\rvert$": table["curve_max_diff"].map(lambda v: _fmt(v, 4)).to_numpy(),
            "EM unconv.": table["em_unconverged"].astype(int).astype(str).to_numpy(),
            "SLSQP fail": table["slsqp_failed"].astype(int).astype(str).to_numpy(),
            "$\\ell$ SLSQP $>$": table["ll_slsqp_better"].astype(int).astype(str).to_numpy(),
            "$\\ell$ EM $>$": table["ll_em_better"].astype(int).astype(str).to_numpy(),
            "EM (s)": table["em_seconds"].map(lambda v: _fmt(v, 1)).to_numpy(),
            "SLSQP (s)": table["slsqp_seconds"].map(lambda v: _fmt(v, 1)).to_numpy(),
            "run (s)": table["run_seconds"].map(lambda v: _fmt(v, 0)).to_numpy(),
            "extra (\\%)": (100 * table["extra_share"]).map(lambda v: _fmt(v, 1)).to_numpy(),
        }
    )
    out.index = baycap_h._index(table)
    return out.style


def caption(verdict: dict) -> str:
    return (
        "E7.2a, gate of the solver: the bandwidth of \\bcapkdey{} selected on the calibration bags with the "
        "maximum-likelihood estimate of EM and of QuaPy's \\texttt{KDEyML} (SLSQP), on the pilot pairs; label-free on "
        "the target. $b$: the selected bandwidth; steps: grid steps between them; tie gap: the largest, over the two "
        "criterion curves, of the difference at the two selections; max $\\lvert\\Delta\\mathrm{crit}\\rvert$: the "
        "largest difference of the two curves over the grid; EM unconv., SLSQP fail: fits (of $21\\times50$) where EM "
        "reached $10^4$ steps or SLSQP reported no success; $\\ell$ SLSQP $>$, $\\ell$ EM $>$: fits where the "
        f"log-likelihood of the bag is larger by more than ${LL_TOL:g}$ at that estimate; EM, SLSQP: seconds of the "
        "estimates over the grid; run: seconds of the method with EM (selection, deployed fit, 100 target bags of the "
        "pilot); extra: SLSQP minus EM, in \\% of run. Gate: (a) same bandwidth or tie gap below "
        f"${TIE_TOL:g}$ on every pair: {'yes' if verdict['a_selection'] else 'no'} "
        f"({verdict['pairs_same_selection']}/{verdict['pairs']}); (b) total extra time at most "
        f"{100 * TIME_SHARE:.0f}\\% of the total run time: {'yes' if verdict['b_time'] else 'no'} "
        f"({_fmt(100 * verdict['extra_share'], 1)}\\%)."
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str = "pairs") -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_{name}.csv")


STEM = f"{TABLE_NAME}_pairs"


def gen_tables(pargs):
    table = pair_table()
    if table.empty:
        print(f"no result of {EXPERIMENT} under {env.root_dir}")
        return
    verdict = gate(table)
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    pd.DataFrame([verdict]).to_csv(csv_path(out_dir, "gate"), index=False)
    styler = _pairs_styler(table)
    with open(os.path.join(tabular_dir, f"{STEM}.tex"), "w") as f:
        f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
    with open(os.path.join(tabular_dir, f"{STEM}_long.tex"), "w") as f:
        f.write(eb_prior._latex_longtable(styler, caption(verdict), f"tab:{TABLE_NAME}"))
    with open(os.path.join(out_dir, f"{STEM}_doc.tex"), "w") as f:
        f.write(tempering._latex_document(f"tabular/{STEM}_long.tex"))
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 4):
        print(table.to_string(index=False))
        print(pd.Series(verdict).to_string())
    print(f"reference bandwidth {CAL_REFERENCE}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    tex_filename = f"{STEM}_doc.tex"
    if not os.path.exists(os.path.join(out_dir, tex_filename)):
        return
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
            cwd=out_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="The gate of the solver of E7.2a: EM against QuaPy's KDEyML (WP-26).")
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
