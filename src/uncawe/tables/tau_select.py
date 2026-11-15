r"""The temperature $\tau$ of BayCAP-Hf and BayCAP-KDEy chosen per pair on calibration bags (work-plan card WP-23, E5.3
of ``docs/ext5_baycap_priors_tempering.md``).

Inputs: the calibration runs of ``experiments/tau_select.py`` (the methods fitted on $V_{\text{fit}}$, $50$ bags per
pair drawn from $V_{\text{cal}}$, at $\tau\in\{0.5,1,2,4,8\}$, under ``<OUT_DIR>/uncawe/tau_select/``) and the stored
test runs of ``main`` and E5.2 (the methods fitted on the full $V$, the $100$ test bags of ``main``, at the same
temperatures, read by ``tables/tempering.py``). Every score is against the realised table of each bag (``true_cts``):
of the calibration bag, from the labels of $V_{\text{cal}}$, and of the test bag. The per-pair means of AE, coverage,
amplitude and Winkler score at $\alpha=0.05$ and the NUTS diagnostics are those of ``tables/tempering.py:pair_rows``.

Selection (``compare``, per metric, method and pair present in both runs):

- $\tau_{\text{cal}}$ (a method: it uses only the labels of $V$): the $\tau$ with the lowest mean Winkler score on the
  calibration bags (``tempering.best_tau``, ties to the $\tau$ closest to $1$ on the log scale, then to the smaller);
  the test scores of the pair are read at $\tau_{\text{cal}}$;
- for comparison, $\tau_{-d}$ of E5.2 (leave one dataset out, computed on all the pairs of the test runs; it uses the
  labels of the other datasets) and, as a reference (it uses the target labels), the oracle-best $\tau$ of each pair.

Decision rule (vanilla accuracy; computed for every metric): per dataset, the mean over its pairs of the per-pair mean
Winkler score on the test bags at $\tau=1$ and at $\tau_{\text{cal}}$; the selection is retained if the second is lower,
by a one-sided paired Wilcoxon signed-rank test over the datasets with $p<0.05$; otherwise $\tau=1$ is kept.

Diagnostics (§T.9): the cross-tabulation of $\tau_{\text{cal}}$ against the oracle-best $\tau$, with the shares of pairs
where $\tau_{\text{cal}}$ is larger, equal and smaller; the expected size of the error from $V$ of the deployed method,
$\hat\sigma_{e_2}=\sqrt{\frac{2}{K(K+1)}\sum_j\hat d_j(1-\hat d_j)/n^s_j}$ (``sigma_e2``, from the recalls $\hat d_j$ of
$h$ on the full $V$ and its class sizes $n^s_j$, stored with the calibration runs), its ratio to the mean half-amplitude
of the test intervals at $\tau=1$, and the Spearman correlations across pairs of $\tau_{\text{cal}}$ with both; the
Spearman correlation between the gain on the calibration bags, $W^{\text{cal}}(1)-W^{\text{cal}}(\tau_{\text{cal}})$,
and the gain on the test bags, $W(1)-W(\tau_{\text{cal}})$; the NUTS diagnostics of the calibration runs per $\tau$.

Outputs, under ``<OUT_DIR>/uncawe/tables/tau_select/``: CSVs of the calibration pairs (every $\tau$), of the pairs
(selections, scores, $\hat\sigma_{e_2}$, gains; read by ``plots/tau_select.py``), of the datasets, of the tests and
correlations, of the summary (per metric, method, group of $K$ and column), of the cross-tabulation, of the
distribution of $\tau_{\text{cal}}$ and of the NUTS diagnostics; per metric, the summary and the cross-tabulation as
LaTeX tabular fragments, longtables and standalone documents.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from uncawe import env
from uncawe.experiments.metrics import get_acc_names
from uncawe.experiments.tau_select import EXPERIMENT
from uncawe.tables import eb_prior
from uncawe.tables import tempering as tt
from uncawe.tables.baycap_hf import DIVERGENT_DRAWS
from uncawe.tables.estimand_gap import booktabs_clines

TABLE_NAME = "tau_select"
KEYS = tt.KEYS
SCORES = tt.SCORES
TAUS = tt.TAUS
REFERENCE_TAU = tt.REFERENCE_TAU
DECISION_METRIC = tt.DECISION_METRIC
DECISION_LEVEL = tt.DECISION_LEVEL
# columns of the summary: tau=1, tau_cal, tau_{-d} of E5.2 and the oracle-best tau (a reference)
COLUMNS = ("t1", "cal", "lodo", "oracle")
REFERENCE_COLUMNS = ("oracle",)


# --------------------------------------------------------------------------------------------------- sigma_e2


def sigma_e2(recalls, counts) -> float:
    """$\\hat\\sigma_{e_2}=\\sqrt{\\frac{2}{K(K+1)}\\sum_j\\hat d_j(1-\\hat d_j)/n^s_j}$ (§T.9): the standard deviation of
    the recall term of the error of a method fitted on $V$, averaged over $\\pi^Q$ uniform on the simplex, with the
    recalls ``recalls`` and the class sizes ``counts`` of $V$; NaN if a class has no instance."""
    recalls, counts = np.asarray(recalls, dtype=float), np.asarray(counts, dtype=float)
    if recalls.shape != counts.shape or recalls.ndim != 1 or np.any(counts <= 0) or not np.all(np.isfinite(recalls)):
        return np.nan
    k = recalls.size
    return float(np.sqrt(2.0 / (k * (k + 1)) * np.sum(recalls * (1 - recalls) / counts)))


# --------------------------------------------------------------------------------------------------- calibration pairs


def _pair_validation(frame: pd.DataFrame) -> pd.DataFrame:
    """Per pair of ``frame`` (the calibration rows of one $\\tau$): $|V|$, $|V_{\\text{fit}}|$, $|V_{\\text{cal}}|$ and
    $\\hat\\sigma_{e_2}$ from the stored recalls and class counts of the full $V$."""
    rows = []
    for (dataset, classifier), part in frame.groupby(KEYS, sort=False):
        first = part.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "v_size": int(first["v_size"]),
                "v_fit_size": int(first["v_fit_size"]),
                "v_cal_size": int(first["v_cal_size"]),
                "sigma_e2": sigma_e2(first["v_recalls"], first["v_class_counts"]),
            }
        )
    return pd.DataFrame(rows)


def calibration_rows(frames: dict[float, pd.DataFrame], acc_name: str, base: str) -> pd.DataFrame:
    """The per-pair rows of ``tempering.pair_rows`` on the calibration bags of one metric and method (the bags are
    checked to be the same at every $\\tau$), with the columns of :func:`_pair_validation`."""
    rows = pd.DataFrame(tt.pair_rows(frames, acc_name, base))
    return rows.merge(_pair_validation(frames[REFERENCE_TAU]), on=KEYS, how="left")


def calibration_table(acc_names: list[str] | None = None, root: str | None = None) -> pd.DataFrame:
    """One row per metric, method and pair of the calibration runs stored at the five temperatures."""
    parts = []
    for acc_name in acc_names or get_acc_names():
        for base, frames in tt.load_all(acc_name, root, EXPERIMENT).items():
            parts.append(calibration_rows(frames, acc_name, base))
    if not parts:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for the five temperatures of a method.")
    return tt.ordered(pd.concat(parts, ignore_index=True))


# --------------------------------------------------------------------------------------------------- selection


def cal_taus(cal: pd.DataFrame) -> np.ndarray:
    """Per row of ``cal``, $\\tau_{\\text{cal}}$: the best $\\tau$ (``tempering.best_tau``) of the mean Winkler score on
    the calibration bags."""
    return tt.oracle_taus(cal)


def _spearman(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return np.nan
    return float(spearmanr(x[keep], y[keep]).statistic)


def compare(cal: pd.DataFrame, test: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The selections, the test of the decision rule and the diagnostics, per metric and method.

    ``cal`` is :func:`calibration_table`, ``test`` is ``tempering.pair_table`` of ``main``. $\\tau_{\\text{cal}}$ is
    computed from ``cal`` only; $\\tau_{-d}$ and the oracle-best $\\tau$ from ``test`` (``tempering.selection``, on all
    its pairs). The outputs cover the pairs present in both.

    - ``pairs``: per pair, the three $\\tau$, the test scores at $\\tau=1$, $\\tau_{\\text{cal}}$, $\\tau_{-d}$ and the
      oracle-best $\\tau$, the calibration Winkler scores at $\\tau=1$ and $\\tau_{\\text{cal}}$, the two gains,
      $\\hat\\sigma_{e_2}$, the half-amplitude at $\\tau=1$ and their ratio;
    - ``datasets``: per dataset, $\\tau_{-d}$ and the means over its pairs of the per-pair mean test Winkler score at
      $\\tau=1$, $\\tau_{\\text{cal}}$ and $\\tau_{-d}$;
    - ``tests``: per metric and method, the number of datasets, of datasets where $\\tau_{\\text{cal}}$ lowers the score,
      the mean difference, the one-sided $p$-value and the decision, the same $p$ for $\\tau_{-d}$ on the same pairs,
      the shares of pairs with $\\tau_{\\text{cal}}$ larger, equal and smaller than the oracle-best, and the Spearman
      correlations of the diagnostics.
    """
    lodo = tt.selection(test)["pairs"].set_index(["acc_name", "method", *KEYS])
    test_index = test.set_index(["acc_name", "method", *KEYS])
    pair_parts, dataset_rows, test_rows = [], [], []
    for (acc_name, base), cal_part in cal.groupby(["acc_name", "method"], sort=False):
        keys = [(acc_name, base, d, c) for d, c in zip(cal_part["dataset"], cal_part["classifier"])]
        present = np.array([key in test_index.index for key in keys])
        if not present.any():
            continue
        cal_part = cal_part.loc[present].reset_index(drop=True)
        keys = [key for key, keep in zip(keys, present) if keep]
        test_part = test_index.loc[keys].reset_index()
        picks = lodo.loc[keys].reset_index()

        tau_cal = cal_taus(cal_part)
        frame = cal_part[["acc_name", "method", "dataset", "classifier", "n_classes"]].copy()
        frame["tau_cal"] = tau_cal
        frame["tau_lodo"] = picks["tau_lodo"].to_numpy(dtype=float)
        frame["tau_oracle"] = picks["tau_oracle"].to_numpy(dtype=float)
        at_cal = tt.scores_at(test_part, tau_cal)
        for score in SCORES:
            frame[f"t1_{score}"] = test_part[f"t1_{score}"].to_numpy(dtype=float)
            frame[f"cal_{score}"] = at_cal[score].to_numpy()
            frame[f"lodo_{score}"] = picks[f"lodo_{score}"].to_numpy(dtype=float)
            frame[f"oracle_{score}"] = picks[f"oracle_{score}"].to_numpy(dtype=float)
        frame["calib_winkler_t1"] = cal_part["t1_winkler"].to_numpy(dtype=float)
        frame["calib_winkler_cal"] = tt.scores_at(cal_part, tau_cal)["winkler"].to_numpy()
        frame["gain_calib"] = frame["calib_winkler_t1"] - frame["calib_winkler_cal"]
        frame["gain_test"] = frame["t1_winkler"] - frame["cal_winkler"]
        frame["sigma_e2"] = cal_part["sigma_e2"].to_numpy(dtype=float)
        frame["half_amplitude_t1"] = frame["t1_amplitude"] / 2
        frame["sigma_ratio"] = tt._ratio(frame["sigma_e2"].to_numpy(), frame["half_amplitude_t1"].to_numpy())
        for column in ("v_size", "v_fit_size", "v_cal_size"):
            frame[column] = cal_part[column].to_numpy()
        pair_parts.append(frame)

        w_1, w_cal, w_lodo = [], [], []
        for dataset, sub in frame.groupby("dataset", sort=False):
            w_1.append(float(np.nanmean(sub["t1_winkler"])))
            w_cal.append(float(np.nanmean(sub["cal_winkler"])))
            w_lodo.append(float(np.nanmean(sub["lodo_winkler"])))
            dataset_rows.append(
                {
                    "acc_name": acc_name,
                    "method": base,
                    "dataset": dataset,
                    "n_classes": int(sub["n_classes"].iloc[0]),
                    "pairs": len(sub),
                    "tau_lodo": float(sub["tau_lodo"].iloc[0]),
                    "taus_cal": " ".join(tt.tau_label(t) for t in sub["tau_cal"]),
                    "winkler_t1": w_1[-1],
                    "winkler_cal": w_cal[-1],
                    "winkler_lodo": w_lodo[-1],
                }
            )
        w_1, w_cal, w_lodo = np.array(w_1), np.array(w_cal), np.array(w_lodo)
        p_value = tt.one_sided_wilcoxon(w_cal, w_1)
        valid = np.isfinite(frame["tau_cal"]) & np.isfinite(frame["tau_oracle"])
        n_valid = max(int(valid.sum()), 1)
        test_rows.append(
            {
                "acc_name": acc_name,
                "method": base,
                "pairs": len(frame),
                "datasets": len(w_1),
                "datasets_lower": int(np.sum(w_cal < w_1)),
                "pairs_tau_cal_not_1": int(np.sum(frame["tau_cal"] != REFERENCE_TAU)),
                "d_winkler": float(np.nanmean(w_cal - w_1)),
                "median_d_winkler": float(np.nanmedian(w_cal - w_1)),
                "p_value": p_value,
                "retained": bool(np.isfinite(p_value) and p_value < DECISION_LEVEL),
                "decision_metric": acc_name == DECISION_METRIC,
                "p_value_lodo": tt.one_sided_wilcoxon(w_lodo, w_1),
                "share_cal_above_oracle": float(np.sum(valid & (frame["tau_cal"] > frame["tau_oracle"])) / n_valid),
                "share_cal_equal_oracle": float(np.sum(valid & (frame["tau_cal"] == frame["tau_oracle"])) / n_valid),
                "share_cal_below_oracle": float(np.sum(valid & (frame["tau_cal"] < frame["tau_oracle"])) / n_valid),
                "spearman_tau_sigma": _spearman(frame["tau_cal"], frame["sigma_e2"]),
                "spearman_tau_ratio": _spearman(frame["tau_cal"], frame["sigma_ratio"]),
                "spearman_gain": _spearman(frame["gain_calib"], frame["gain_test"]),
            }
        )
    if not pair_parts:
        raise RuntimeError("No pair has both calibration runs and stored test runs at the five temperatures.")
    return {
        "pairs": pd.concat(pair_parts, ignore_index=True),
        "datasets": pd.DataFrame(dataset_rows),
        "tests": pd.DataFrame(test_rows),
    }


# --------------------------------------------------------------------------------------------------- summaries


def summary_table(compared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per metric, method, group of $K$ and column of ``COLUMNS`` (``oracle`` is a reference): the number of pairs and
    the means over the pairs of the four test scores."""
    rows = []
    for (acc_name, base), sub in compared["pairs"].groupby(["acc_name", "method"], sort=False):
        for k_group, part in tt.k_groups(sub):
            key = {"acc_name": acc_name, "method": base, "K": k_group, "pairs": len(part)}
            for column in COLUMNS:
                row = key | {"column": column, "reference": column in REFERENCE_COLUMNS}
                row |= {score: float(part[f"{column}_{score}"].mean()) for score in SCORES}
                rows.append(row)
    return pd.DataFrame(rows)


def crosstab(compared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per metric and method, the number of pairs for each ($\\tau_{\\text{cal}}$, oracle-best $\\tau$): one row per
    $\\tau_{\\text{cal}}$, one column per oracle-best $\\tau$ (a reference)."""
    rows = []
    for (acc_name, base), part in compared["pairs"].groupby(["acc_name", "method"], sort=False):
        for tau_cal in TAUS:
            row = {"acc_name": acc_name, "method": base, "tau_cal": tau_cal}
            chosen = part["tau_cal"] == tau_cal
            row |= {tt.tau_label(tau): int(np.sum(chosen & (part["tau_oracle"] == tau))) for tau in TAUS}
            rows.append(row)
    return pd.DataFrame(rows)


def tau_distribution(compared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per metric, method and group of $K$, the number of pairs whose $\\tau_{\\text{cal}}$ is each value."""
    rows = []
    for (acc_name, base), sub in compared["pairs"].groupby(["acc_name", "method"], sort=False):
        for k_group, part in tt.k_groups(sub):
            row = {"acc_name": acc_name, "method": base, "K": k_group, "pairs": len(part)}
            row |= {tt.tau_label(tau): int(np.sum(part["tau_cal"] == tau)) for tau in TAUS}
            rows.append(row)
    return pd.DataFrame(rows)


def nuts_table(cal: pd.DataFrame) -> pd.DataFrame:
    """The NUTS diagnostics of the calibration runs, per metric, method, group of $K$ and $\\tau$: the mean shares of
    bags with divergent transitions and without draws, the mean share of divergent draws and the number of pairs where
    it exceeds ``DIVERGENT_DRAWS`` (the reliability criterion of E2.8), the median over the pairs of the largest
    split-$\\hat R$ and the number of pairs where it exceeds $1.1$."""
    rows = []
    for (acc_name, base), sub in cal.groupby(["acc_name", "method"], sort=False):
        for k_group, part in tt.k_groups(sub):
            for tau in TAUS:
                t = tt.tau_label(tau)
                draws, rhat = part[f"t{t}_share_divergent_draws"], part[f"t{t}_max_rhat"]
                rows.append(
                    {
                        "acc_name": acc_name,
                        "method": base,
                        "K": k_group,
                        "tau": tau,
                        "pairs": len(part),
                        "share_divergent": float(part[f"t{t}_share_divergent"].mean()),
                        "share_failed": float(part[f"t{t}_share_failed"].mean()),
                        "share_divergent_draws": float(draws.mean()),
                        "pairs_divergent_draws": int((draws > DIVERGENT_DRAWS).sum()),
                        "median_max_rhat": float(rhat.median()),
                        "pairs_rhat_above_1.1": int((rhat > 1.1).sum()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

COLUMN_LABELS = {
    "t1": "$\\tau=1$",
    "cal": "$\\tau_{\\mathrm{cal}}$",
    "lodo": "$\\tau_{-d}$",
    "oracle": "oracle",
}


def _summary_styler(summary: pd.DataFrame, tests: pd.DataFrame):
    """One row per method, group of $K$ and score; one column per selection (the oracle in grey), the number of
    datasets where $\\tau_{\\text{cal}}$ lowers the score and the $p$-value of the decision rule (rows of all pairs)."""
    rows = []
    by_method = tests.set_index("method")
    for (base, k_group), part in summary.groupby(["method", "K"], sort=False):
        by_column = part.set_index("column")
        for score in SCORES:
            row = {"method": tt.METHOD_LABELS[base], "$K$": tt.K_LABELS[k_group], "score": tt.SCORE_LABELS[score]}
            row |= {label: by_column.loc[column, score] for column, label in COLUMN_LABELS.items()}
            first = k_group == "all" and score == "winkler" and base in by_method.index
            row["lower"] = (
                f"{by_method.loc[base, 'datasets_lower']}/{by_method.loc[base, 'datasets']}" if first else "--"
            )
            row["$p$"] = by_method.loc[base, "p_value"] if first else np.nan
            rows.append(row)
    out = pd.DataFrame(rows).set_index(["method", "$K$", "score"])
    grey = [COLUMN_LABELS[column] for column in REFERENCE_COLUMNS]
    return (
        out.style.format(precision=3, na_rep="--")
        .map(lambda _: "color:{gray};", subset=grey)
        .map(lambda _: "bfseries: ;", subset=[COLUMN_LABELS["cal"]])
    )


def _crosstab_styler(cross: pd.DataFrame):
    """One row per method and $\\tau_{\\text{cal}}$, one column per oracle-best $\\tau$ (a reference), number of
    pairs."""
    labels = [tt.tau_label(tau) for tau in TAUS]
    out = cross[labels].copy()
    out.columns = pd.MultiIndex.from_tuples([("oracle-best $\\tau$", f"${label}$") for label in labels])
    out.index = pd.MultiIndex.from_arrays(
        [cross["method"].map(tt.METHOD_LABELS), cross["tau_cal"].map(lambda t: f"${tt.tau_label(t)}$")],
        names=["method", "$\\tau_{\\mathrm{cal}}$"],
    )
    return out.style.format("{:d}")


def caption(acc_name: str, name: str) -> str:
    metric = tt.ACC_LABELS[acc_name]
    if name == "summary":
        return (
            f"E5.3, $\\tau$ chosen per pair on calibration bags drawn from half of $V$ ({metric}); means over the "
            "pairs of the per-pair means over the 100 test bags of \\texttt{main} (methods fitted on the full $V$), "
            "scored against the realised table. $\\tau_{\\mathrm{cal}}$: the $\\tau$ with the lowest mean Winkler "
            "score on the 50 calibration bags of the pair; $\\tau_{-d}$: the $\\tau$ of E5.2, chosen on the other "
            "datasets; oracle (grey, a reference: it uses the target labels): the best $\\tau$ of each pair. lower: "
            "datasets where $\\tau_{\\mathrm{cal}}$ lowers the per-dataset Winkler score; $p$: one-sided paired "
            "Wilcoxon signed-rank test over the datasets against $\\tau=1$ (decision rule on vanilla accuracy, "
            "$p<0.05$)."
        )
    return (
        f"E5.3 ({metric}): number of pairs for each $\\tau_{{\\mathrm{{cal}}}}$ (rows) and oracle-best $\\tau$ on the "
        "test bags (columns, a reference chosen with the target labels)."
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


TABLES = ("summary", "crosstab")
CSV_NAMES = ("calibration", "pairs", "datasets", "tests", "summary", "crosstab", "taus", "nuts")


def compute(metrics: list[str] | None = None, root: str | None = None) -> dict[str, pd.DataFrame]:
    """Every output frame (``CSV_NAMES``) from the stored calibration and test runs."""
    cal = calibration_table(metrics, root)
    test = tt.pair_table(metrics, root)
    compared = compare(cal, test)
    return {
        "calibration": cal,
        "pairs": compared["pairs"],
        "datasets": compared["datasets"],
        "tests": compared["tests"],
        "summary": summary_table(compared),
        "crosstab": crosstab(compared),
        "taus": tau_distribution(compared),
        "nuts": nuts_table(cal),
    }


def write_tables(out: dict[str, pd.DataFrame], out_dir: str) -> None:
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    for name in CSV_NAMES:
        out[name].to_csv(csv_path(out_dir, name), index=False)
    for acc_name in out["summary"]["acc_name"].unique():
        stylers = {
            "summary": _summary_styler(
                out["summary"].loc[out["summary"]["acc_name"] == acc_name],
                out["tests"].loc[out["tests"]["acc_name"] == acc_name],
            ),
            "crosstab": _crosstab_styler(out["crosstab"].loc[out["crosstab"]["acc_name"] == acc_name]),
        }
        for name, styler in stylers.items():
            stem = _stem(acc_name, name)
            with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
                f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
            with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
                f.write(
                    eb_prior._latex_longtable(styler, caption(acc_name, name), f"tab:{TABLE_NAME}-{name}-{acc_name}")
                )
            with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
                f.write(tt._latex_document(f"tabular/{stem}_long.tex"))


def gen_tables(pargs):
    out = compute(pargs.metrics)
    out_dir = table_dir(pargs)
    write_tables(out, out_dir)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        summary = out["summary"]
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
        print(out["tests"].to_string(index=False))
        print(out["taus"].loc[out["taus"]["K"] == "all"].to_string(index=False))
    print(f"pairs: {len(out['pairs'])}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for acc_name in pargs.metrics or get_acc_names():
        for name in TABLES:
            tex_filename = f"{_stem(acc_name, name)}_doc.tex"
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
    parser = ArgumentParser(description="Tau chosen per pair on calibration bags for BayCAP-Hf and BayCAP-KDEy (E5.3).")
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
