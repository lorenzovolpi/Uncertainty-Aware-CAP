"""One bandwidth of BayesKDEy per pair, selected on calibration bags drawn from half of $V$ (work-plan card WP-26, E7.2
of ``docs/ext7_kdey_bandwidth.md``).

``BayCAP-KDEy-mlp-real-cal`` (``experiments/methods.py:CAL_VARIANTS``) is ``BayCAP-KDEy-mlp-real`` with BayesKDEy at
``bandwidth="cal"``: inside the fit, one bandwidth $b_{\\text{cal}}$ shared by the classes is selected over
``methods/bayes_kdey.py:LOO_GRID`` by the mean over $50$ calibration bags drawn from half of $V$ of
$\\lvert\\hat d^{\\text{fit}\\top}(\\hat\\pi_s(b)-\\pi_s)\\rvert$ (§T.7), and the method is then fitted on the full $V$ at
$b_{\\text{cal}}$; everything else, seeds included, is the stored method with $b=0.1$. Both methods target the realised
table of the target bag and are scored against it (``true_cts``); the selection uses only the labels of $V$. All three
metrics of ``main``; the pairs where both methods have stored results (so it also works on the pilot).

Per metric and pair (``pair_table``, one row per metric and pair, every column in the pairs CSV):

- AE, coverage, amplitude and Winkler score of both methods, means over the bags;
- for AE and the Winkler score, the mean per-bag difference (cal minus stored), its paired bootstrap interval and
  $p$-value (``tables/pushforward.py:bootstrap_mean_test``, $10{,}000$ resamples), Holm-adjusted over the pairs, per
  metric and score;
- label-free, from the summaries stored by the cal method: $b_{\\text{cal}}$, its distance from $0.1$ in steps of the
  grid, the criterion at $0.1$ and at $b_{\\text{cal}}$, the calibration gain (their difference), the number of EM
  estimates that did not converge, $\\lvert V\\rvert$ and $\\lvert V_{\\text{cal}}\\rvert$, the share of bags with
  divergent transitions; the test gain, the mean AE of the stored method minus that of the cal method;
- with the pilot check (``main --kdey-cal-check``): the bandwidth selected with the NUTS posterior mean, the steps
  between the two selections and the largest absolute difference of the two criterion curves.

``curve_table``: per pair and bandwidth of the grid, the criterion (and the NUTS criterion where stored).
``summary_table``: per metric and group of $K$, the means over the pairs of the scores of both methods, and for the cal
method the pairs where its AE and its Winkler score are significantly (Holm) lower and higher than the stored ones, the
two-sided Wilcoxon signed-rank $p$-values over the per-pair means, and the Spearman correlation between the calibration
gain and the test gain. ``check_table``: the pilot check, one row per pair that stores it.

Outputs, under ``<OUT_DIR>/uncawe/tables/kdey_cal/``: the CSVs and, per metric, a summary and a per-pair table (and the
pilot check, if any), each as a LaTeX tabular fragment, a longtable and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uncawe import env
from uncawe.experiments.methods import CAL_VARIANTS
from uncawe.experiments.metrics import get_acc_names
from uncawe.methods.baycap_kdey import CAL_REFERENCE
from uncawe.methods.bayes_kdey import LOO_GRID
from uncawe.tables import baycap_h, eb_prior, pushforward, tempering
from uncawe.tables.displacement import ordered
from uncawe.tables.estimand_gap import SEED, booktabs_clines, pair_rng
from uncawe.tables.kdey_loo import two_sided_wilcoxon

TABLE_NAME = "kdey_cal"
KEYS = ["dataset", "classifier"]
SCORES = ("ae", "coverage", "amplitude", "winkler")
# the scores whose per-bag differences are tested
TESTED = ("ae", "winkler")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
CAL_METHOD, STORED_METHOD = next(iter(CAL_VARIANTS.items()))
METHODS = [STORED_METHOD, CAL_METHOD]
# the grid of the selection (the default of the method) and one step of it on the log10 scale
GRID = np.asarray(LOO_GRID)
GRID_STEP = float(np.diff(np.log10(GRID)).mean())
SHORT_LABELS = {STORED_METHOD: "KDEy", CAL_METHOD: "KDEy cal"}


def load_all(acc_name: str) -> dict[str, pd.DataFrame]:
    """The stored ``main`` rows of both methods for ``acc_name``, restricted to the pairs stored for both; empty if
    there is none."""
    frames = {method: tempering.load_method(method, acc_name) for method in METHODS}
    if any(df.empty for df in frames.values()):
        return {}
    pairs = eb_prior.common_pairs(frames)
    if not pairs:
        return {}
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    return {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in frames.items()
    }


def _check_paired(a: pd.DataFrame, b: pd.DataFrame, key) -> None:
    if len(a) != len(b) or not np.allclose(a["true_acc"], b["true_acc"]):
        raise ValueError(f"{CAL_METHOD} and {STORED_METHOD} do not have the same bags on {key}")


def _first(df: pd.DataFrame, name: str):
    """The summary ``name`` stored with the first bag (the selection is fixed by the fit, so the same for every bag);
    NaN if it is not stored."""
    if name not in df:
        return np.nan
    return df[name].iloc[0]


def steps(a: float, b: float) -> float:
    """The distance between two bandwidths in steps of the grid ($\\log_{10}$ scale)."""
    return float(np.round(np.log10(a / b) / GRID_STEP, 6))


def selection_columns(df: pd.DataFrame) -> dict:
    """The label-free summary of the selection of one pair, from the rows of the cal method."""
    bandwidth = float(_first(df, "cal_bandwidth"))
    selected, default = float(_first(df, "cal_selected")), float(_first(df, "cal_default"))
    out = {
        "b_cal": bandwidth,
        "b_cal_steps": steps(bandwidth, CAL_REFERENCE),
        "crit_default": default,
        "crit_selected": selected,
        "cal_gain": default - selected,
        "em_unconverged": float(_first(df, "cal_em_unconverged")),
        "n_val": int(_first(df, "cal_n_val")),
        "n_cal": int(_first(df, "cal_n_cal")),
    }
    nuts = _first(df, "cal_bandwidth_nuts")
    if np.ndim(nuts) == 0 and np.isfinite(float(nuts)):
        curve = np.asarray(_first(df, "cal_curve"), dtype=float)
        curve_nuts = np.asarray(_first(df, "cal_curve_nuts"), dtype=float)
        out |= {
            "b_cal_nuts": float(nuts),
            "nuts_steps": steps(float(nuts), bandwidth),
            "nuts_curve_max_diff": float(np.max(np.abs(curve - curve_nuts))),
        }
    return out


def pair_rows(frames: dict[str, pd.DataFrame], acc_name: str) -> list[dict]:
    rows = []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in frames.items()}
    for key, first in groups[CAL_METHOD].items():
        bags = {method: tempering.bag_frame(groups[method][key], acc_name) for method in METHODS}
        _check_paired(bags[CAL_METHOD], bags[STORED_METHOD], key)
        row = {
            "acc_name": acc_name,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
        for index, score in enumerate(TESTED):
            diff = (bags[CAL_METHOD][score] - bags[STORED_METHOD][score]).to_numpy()
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0:
                low = high = p_value = mean = np.nan
            else:
                low, high, p_value = pushforward.bootstrap_mean_test(diff, pair_rng(*key, seed=SEED + 1000 * index))
                mean = float(diff.mean())
            row |= {f"{score}_diff_mean": mean, f"{score}_ci_low": low, f"{score}_ci_high": high, f"{score}_p": p_value}
        row["share_divergent"] = float(bags[CAL_METHOD]["divergent"].mean())
        row |= selection_columns(first)
        row["test_gain"] = row[f"{STORED_METHOD}_ae"] - row[f"{CAL_METHOD}_ae"]
        rows.append(row)
    return rows


def pair_table(acc_names=None) -> pd.DataFrame:
    """One row per metric and pair with stored results of both methods, ordered as the tables of E3.1."""
    tables = []
    for acc_name in acc_names or get_acc_names():
        frames = load_all(acc_name)
        if not frames:
            continue
        table = pd.DataFrame(pair_rows(frames, acc_name))
        for score in TESTED:
            table[f"{score}_p_holm"] = pushforward.holm(table[f"{score}_p"].to_numpy())
        tables.append(ordered(table))
    if not tables:
        raise RuntimeError(f"No pair of main has stored results for all of {METHODS}.")
    return pd.concat(tables, ignore_index=True)


def curve_table(acc_name: str = "vanilla_accuracy") -> pd.DataFrame:
    """One row per pair and bandwidth of the grid: the criterion, and the NUTS criterion where stored (from the rows of
    the cal method for ``acc_name``; the selection does not depend on the metric)."""
    frames = load_all(acc_name)
    if not frames:
        raise RuntimeError(f"No pair of main has stored results for all of {METHODS}.")
    rows = []
    for key, df in frames[CAL_METHOD].groupby(KEYS, sort=False):
        curve = np.asarray(_first(df, "cal_curve"), dtype=float)
        if len(curve) != len(GRID):
            raise ValueError(f"{key}: the stored criterion has {len(curve)} values, the grid {len(GRID)}")
        nuts = _first(df, "cal_curve_nuts")
        curve_nuts = np.asarray(nuts, dtype=float) if np.ndim(nuts) == 1 else np.full(len(GRID), np.nan)
        for bandwidth, value, value_nuts in zip(GRID, curve, curve_nuts):
            rows.append(
                {
                    "dataset": key[0],
                    "classifier": key[1],
                    "n_classes": int(df["n_classes"].iloc[0]),
                    "bandwidth": float(bandwidth),
                    "criterion": float(value),
                    "criterion_nuts": float(value_nuts),
                }
            )
    return pd.DataFrame(rows)


def check_table(table: pd.DataFrame) -> pd.DataFrame:
    """The pilot check of §T.8: one row per pair that stores the NUTS criterion (vanilla-accuracy rows; the selection
    does not depend on the metric); empty if no pair stores it."""
    columns = ["b_cal", "b_cal_nuts", "nuts_steps", "nuts_curve_max_diff"]
    if "b_cal_nuts" not in table:
        return pd.DataFrame(columns=KEYS + ["n_classes"] + columns)
    part = table.loc[(table["acc_name"] == table["acc_name"].iloc[0]) & table["b_cal_nuts"].notna()]
    return part[KEYS + ["n_classes"] + columns].reset_index(drop=True)


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan, np.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue)


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric and group of $K$, one row per method: the means over the pairs of its scores and, for the cal method,
    the pairs where its AE and its Winkler score are significantly (Holm) lower and higher than the stored ones, the
    Wilcoxon $p$-values over the per-pair means, the pairs with $b_{\\text{cal}}=0.1$ and the Spearman correlation
    between the calibration gain and the test gain."""
    rows = []
    for acc_name, part in table.groupby("acc_name", sort=False):
        for k_group, sub in tempering.k_groups(part):
            for method in METHODS:
                row = {"acc_name": acc_name, "K": k_group, "pairs": len(sub), "method": method}
                row |= {score: float(sub[f"{method}_{score}"].mean()) for score in SCORES}
                if method == CAL_METHOD:
                    for score in TESTED:
                        significant = sub[f"{score}_p_holm"] < TEST_LEVEL
                        lower = sub[f"{score}_diff_mean"] < 0
                        row[f"{score}_pairs_lower"] = int((lower & significant).sum())
                        row[f"{score}_pairs_higher"] = int((~lower & significant).sum())
                        row[f"{score}_wilcoxon_p"] = two_sided_wilcoxon(
                            sub[f"{CAL_METHOD}_{score}"], sub[f"{STORED_METHOD}_{score}"]
                        )
                    row["pairs_at_reference"] = int(np.isclose(sub["b_cal"], CAL_REFERENCE).sum())
                    row["spearman_gain"], row["spearman_p"] = _spearman(sub["cal_gain"], sub["test_gain"])
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

SCORE_LABELS = {
    "ae": "AE",
    "coverage": "$\\mathcal{C}$",
    "amplitude": "$\\mathcal{A}$",
    "winkler": "$\\mathcal{W}$",
}


def _fmt(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _count(value) -> str:
    return "--" if pd.isna(value) else f"{int(value)}"


def _summary_styler(summary: pd.DataFrame):
    """One row per group of $K$ and method: the four scores, and for the cal method the pairs lower/higher and the
    Wilcoxon $p$-value of AE and of the Winkler score, and the Spearman correlation of the gains."""
    columns = {(SCORE_LABELS[score], ""): summary[score].map(_fmt).to_numpy() for score in SCORES}
    for score in TESTED:
        label = f"$\\Delta${SCORE_LABELS[score]}"
        columns[(label, "lower")] = summary[f"{score}_pairs_lower"].map(_count).to_numpy()
        columns[(label, "higher")] = summary[f"{score}_pairs_higher"].map(_count).to_numpy()
        columns[(label, "$p$")] = summary[f"{score}_wilcoxon_p"].map(_fmt).to_numpy()
    columns[("gains", "$\\rho_S$")] = summary["spearman_gain"].map(lambda v: _fmt(v, 2)).to_numpy()
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["K"].map(tempering.K_LABELS).to_numpy(),
            summary["pairs"].astype(int).to_numpy(),
            summary["method"].map(SHORT_LABELS).to_numpy(),
        ],
        names=["$K$", "pairs", "method"],
    )
    return out.style


def _diff_text(diff: float, p_holm: float) -> str:
    if not np.isfinite(diff):
        return "--"
    return f"\\textbf{{{diff:.3f}}}" if p_holm < TEST_LEVEL else f"{diff:.3f}"


def _pairs_styler(table: pd.DataFrame):
    """One row per pair: $b_{\\text{cal}}$, the criterion at $0.1$ and at $b_{\\text{cal}}$, AE, coverage and Winkler
    score of both methods (coloured per row, as in E3.1), and the two differences (bold where Holm $p<0.05$)."""
    columns = {
        ("selection", "$b_{\\mathrm{cal}}$"): table["b_cal"].map(lambda v: _fmt(v, 3)).to_numpy(),
        ("selection", "crit $0.1$"): table["crit_default"].to_numpy(dtype=float),
        ("selection", "crit $b_{\\mathrm{cal}}$"): table["crit_selected"].to_numpy(dtype=float),
    }
    for score in ("ae", "coverage", "winkler"):
        for method in METHODS:
            columns[(SCORE_LABELS[score], SHORT_LABELS[method])] = table[f"{method}_{score}"].to_numpy(dtype=float)
    for score in TESTED:
        columns[("$\\Delta$", SCORE_LABELS[score])] = [
            _diff_text(d, p) for d, p in zip(table[f"{score}_diff_mean"], table[f"{score}_p_holm"])
        ]
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = tempering.score_styles(out, {score: SCORE_LABELS[score] for score in ("ae", "coverage", "winkler")})
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


def _check_styler(check: pd.DataFrame):
    """One row per pilot pair: the bandwidths selected with the EM estimate and with the NUTS posterior mean, the steps
    between them and the largest difference of the two criterion curves."""
    out = pd.DataFrame(
        {
            "$b_{\\mathrm{cal}}$ (EM)": check["b_cal"].map(_fmt).to_numpy(),
            "$b_{\\mathrm{cal}}$ (NUTS)": check["b_cal_nuts"].map(_fmt).to_numpy(),
            "steps": check["nuts_steps"].map(lambda v: _fmt(v, 0)).to_numpy(),
            "max $\\lvert\\Delta\\mathrm{crit}\\rvert$": check["nuts_curve_max_diff"].map(lambda v: _fmt(v, 4)),
        }
    )
    out.index = baycap_h._index(check)
    return out.style


def caption(acc_name: str, name: str) -> str:
    common = (
        f"E7.2, one bandwidth of BayesKDEy per pair selected on calibration bags drawn from half of $V$, on "
        f"\\texttt{{main}} ({tempering.ACC_LABELS[acc_name]}); KDEy: \\bcapkdey{{}} ($b=0.1$); KDEy cal: the same "
        "method with the bandwidth $b_{\\mathrm{cal}}$ that minimises the mean over $50$ calibration bags of the "
        "accuracy error of the quantifier, $\\lvert\\hat d^\\top(\\hat\\pi_s(b)-\\pi_s)\\rvert$. "
    )
    if name == "summary":
        return common + (
            "Scores against the realised table, means over the 100 bags and over the pairs; lower, higher: pairs where "
            "the mean AE or Winkler score of the cal method is significantly lower or higher than that of the stored "
            "method (paired bootstrap, Holm over the pairs, $p<0.05$); $p$: two-sided Wilcoxon signed-rank test over "
            "the per-pair means; $\\rho_S$: Spearman correlation between the calibration gain (criterion at $0.1$ "
            "minus at $b_{\\mathrm{cal}}$) and the test gain (mean AE, stored minus cal)."
        )
    if name == "check":
        return common + (
            "Pilot check of the maximum-likelihood estimate in the criterion: the bandwidth selected with the EM "
            "estimate and with the NUTS posterior mean, the grid steps between them and the largest absolute difference "
            "of the two criterion curves over the grid."
        )
    return common + (
        "Criterion at $0.1$ and at $b_{\\mathrm{cal}}$ (label-free on the target); scores against the realised table, "
        "means over the 100 bags, each coloured per row from the worst (red) to the best (green, in bold; coverage: "
        "closest to $0.95$); $\\Delta$: mean per-bag difference, cal minus stored (bold: Holm $p<0.05$)."
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str = "pairs") -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_{name}.csv")


def _stem(acc_name: str, name: str) -> str:
    return f"{TABLE_NAME}_{name}_{acc_name}"


TABLES = ("summary", "pairs", "check")


def _write(out_dir: str, stem: str, styler, caption_text: str, label: str) -> None:
    tabular_dir = os.path.join(out_dir, "tabular")
    with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
        f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
    with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
        f.write(eb_prior._latex_longtable(styler, caption_text, label))
    with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
        f.write(tempering._latex_document(f"tabular/{stem}_long.tex"))


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    summary = summary_table(table)
    curves = curve_table()
    check = check_table(table)
    out_dir = table_dir(pargs)
    os.makedirs(os.path.join(out_dir, "tabular"), exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    summary.to_csv(csv_path(out_dir, "summary"), index=False)
    curves.to_csv(csv_path(out_dir, "curves"), index=False)
    check.to_csv(csv_path(out_dir, "check"), index=False)

    for acc_name in table["acc_name"].unique():
        stylers = {
            "summary": _summary_styler(summary.loc[summary["acc_name"] == acc_name].reset_index(drop=True)),
            "pairs": _pairs_styler(table.loc[table["acc_name"] == acc_name].reset_index(drop=True)),
        }
        for name, styler in stylers.items():
            _write(
                out_dir, _stem(acc_name, name), styler, caption(acc_name, name), f"tab:{TABLE_NAME}-{name}-{acc_name}"
            )
    if not check.empty:
        _write(
            out_dir,
            f"{TABLE_NAME}_check",
            _check_styler(check),
            caption("vanilla_accuracy", "check"),
            f"tab:{TABLE_NAME}-check",
        )

    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
        first = table.loc[table["acc_name"] == table["acc_name"].iloc[0]]
        print(
            f"pairs: {len(first)}; b_cal = 0.1 on {int(np.isclose(first['b_cal'], CAL_REFERENCE).sum())}; median "
            f"b_cal: {first['b_cal'].median():.3f}; at the grid edges: "
            f"{int(np.isclose(first['b_cal'], GRID[0]).sum() + np.isclose(first['b_cal'], GRID[-1]).sum())}"
        )
        if not check.empty:
            print(check.to_string(index=False))
    print(f"rows: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    stems = [_stem(acc_name, name) for acc_name in pargs.metrics or get_acc_names() for name in ("summary", "pairs")]
    for stem in [*stems, f"{TABLE_NAME}_check"]:
        tex_filename = f"{stem}_doc.tex"
        if not os.path.exists(os.path.join(out_dir, tex_filename)):
            continue
        # longtable needs a second pass to settle the column widths
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                cwd=out_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="One bandwidth of BayesKDEy per pair selected on calibration bags (E7.2).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=None, help="Default: all three")
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
