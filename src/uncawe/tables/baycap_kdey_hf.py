"""BayCAP-KDEy with a random $C$ and the Hf prior on the pairs of ``main`` (work-plan card WP-13, E3.2 of
``docs/ext3_posterior_propagation.md``).

The two methods ``BayCAP-KDEy-Hf-<s>-real`` (``methods/baycap_kdey.py:BayCAPKDEy`` with ``c_prior="hierarchical-f"``,
input classifier $s\\in\\{h$, mlp$\\}$) sample $(\\pi,C)$ jointly, with the class densities of BayesKDEy written as
mixtures of the kernel densities of the cells of $V$ and the Hf prior on $C$. They are compared bag by bag with the
realised BayCAP-KDEy of E3.1 with the same input classifier (``BayCAP-KDEy-<s>-real``, fixed $\\hat C$), with
``BayesCAP-Hf-real`` (E2.8, the same prior on $C$ with the counts of the predictions as likelihood) and with ``SQS-Hbs`` (called ``RQBS`` until 2026-11-10).
All six target the realised table of the target bag and are scored against it (``true_cts``). Vanilla accuracy,
$|U|=1000$, the pairs where the six have stored results (so it also works on the pilot).

Per pair (``compute_table``, one row per pair, every column in the CSV):

- AE, coverage, amplitude and Winkler score of the six methods, means over the bags; the bags where a method with the
  Hf prior produced no draws (``nuts_failed``) are left out of its scores;
- the standardised shift of each method, the mean over the bags of $(\\hat A-A)/\\operatorname{sd}(A^{(r)})$, with
  $\\hat A$ the point estimate, $A$ the realised accuracy and $A^{(r)}$ the accuracy samples (§T.13);
- the split of the mean shift into the recall and the prevalence terms (``tables/displacement.py:error_terms``) of the
  two new methods;
- four paired bootstrap tests of the mean per-bag Winkler difference (``tables/pushforward.py:bootstrap_mean_test``,
  $10{,}000$ resamples), each Holm-adjusted over the pairs on its own: each new method minus the E3.1 method with the
  same input classifier and minus ``BayesCAP-Hf-real``, on the bags where neither method failed;
- for each new method, the NUTS diagnostics (on $\\pi$, $C$ and the Hf hyperparameters: share of bags with divergent
  transitions, mean divergent transitions per bag, largest and median split-$\\hat R$, share of bags with split-$\\hat R$
  above ``RHAT_HIGH``, smallest effective sample size, share of failed bags), the means over the bags of the Hf summaries
  (median of $\\ln\\nu$ and of $\\ln\\kappa$, $P(\\kappa<1)$), the numbers of empty cells of $V$ and of cells with the
  density of their true class (§T.12), and the mean label-free mismatch $\\lVert n^t/N^Q-\\bar C\\bar\\pi\\rVert_1$;
  the mismatch $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$ of the E3.1 methods, for comparison (P14).

Also ``cell_table``, one row per pair, new method and cell $(k,j)$, label-free: the count $f_{kj}$, $n^s_j$,
$\\hat C_{kj}$, the standard deviation that $V$ alone gives the cell, $\\sqrt{\\hat C_{kj}(1-\\hat C_{kj})/(n^s_j+1)}$, the
mean over the bags of the posterior standard deviation of $C_{kj}$, and their ratio (NaN where the standard deviation
from $V$ is $0$, i.e. $\\hat C_{kj}\\in\\{0,1\\}$), §T.13.

The scores, the shifts and the tests use the true tables, so they are analyses; the diagnostics, the Hf summaries, the
mismatch and the cells use no target labels. The E2.2 decomposition table is ``tables/displacement.py --method
<method>`` and the rank table ``tables/rank.py --include-e32``. Outputs, under ``<OUT_DIR>/uncawe/tables/baycap_kdey_hf/``:
the per-pair CSV and the cells CSV, which ``plots/baycap_kdey_hf.py`` reads, and three tables (comparison; tests;
diagnostics), each as a LaTeX tabular fragment, a longtable and a standalone landscape document. Validation counts,
tables and the stored ``c_mean``, ``c_sd`` follow the ``cap-ml`` layout (true classes on rows); $\\hat C$ of
``rate_matrix`` is in the paper's layout.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import acc_from_ct
from uncawe.experiments.methods import KDEY_HF_VARIANTS
from uncawe.main import EXPERIMENT
from uncawe.tables import baycap_fixed, baycap_h, baycap_hf, eb_prior, pushforward
from uncawe.tables import main as main_tables
from uncawe.tables.baycap_hf import RHAT_HIGH
from uncawe.tables.displacement import load_validation_counts, ordered, rate_matrix
from uncawe.tables.estimand_gap import ACC_NAME, DOMAIN, SEED, booktabs_clines, pair_rng

TABLE_NAME = "baycap_kdey_hf"
KEYS = eb_prior.KEYS
SOURCES = ("h", "mlp")
NEW = {source: f"BayCAP-KDEy-Hf-{source}-real" for source in SOURCES}
assert sorted(NEW.values()) == sorted(KDEY_HF_VARIANTS)
FIXED = {source: f"BayCAP-KDEy-{source}-real" for source in SOURCES}
HF = baycap_fixed.HF
SQS_HBS = baycap_fixed.SQS_HBS
METHODS = [*NEW.values(), *FIXED.values(), HF, SQS_HBS]
# the methods whose failed bags (no NUTS draws) are left out of their scores and tests
HIERARCHICAL = (*NEW.values(), HF)
SCORES = ("ae", "coverage", "amplitude", "winkler")
SHIFT_TERMS = baycap_hf.SHIFT_TERMS
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
HF_SUMMARIES = ("h_ln_nu_median", "h_ln_kappa_median", "h_p_kappa_below_1")


def _tests() -> dict[str, tuple[str, str]]:
    """The four tests, name -> (method, reference): the per-bag difference is method minus reference."""
    tests = {}
    for source in SOURCES:
        tests[f"{source}_vs_fixed"] = (NEW[source], FIXED[source])
        tests[f"{source}_vs_hf"] = (NEW[source], HF)
    return tests


TESTS = _tests()


def bag_frame(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per bag of one pair (stored order): the realised accuracy, the point estimate, AE, coverage, amplitude,
    Winkler score, the standardised shift and whether NUTS produced no draws (``failed``)."""
    rows = []
    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(ACC_NAME, np.asarray(row.true_cts, dtype=float))
        samples = np.asarray(row.estim_acc_ranges, dtype=float)
        failed = not np.all(np.isfinite(samples))
        point = float(row.estim_accs)
        sd = float(np.std(samples)) if not failed else np.nan
        rows.append(
            {
                "true_acc": true_acc,
                "failed": failed,
                "ae": np.nan if failed else float(row.acc_err),
                "coverage": np.nan if failed else main_tables.get_acc_coverage(method, true_acc, samples),
                "amplitude": np.nan if failed else main_tables.get_acc_amplitude(method, samples),
                "winkler": np.nan if failed else main_tables.get_acc_winkler(method, true_acc, samples),
                "std_shift": (point - true_acc) / sd if sd > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def standardised_shift(bags: pd.DataFrame) -> float:
    """The mean over the bags (with draws) of $(\\hat A-A)/\\operatorname{sd}(A^{(r)})$ (§T.13)."""
    return float(bags["std_shift"].mean())


def mismatch_hf(true_ct: np.ndarray, c_mean: np.ndarray, pi_mean: np.ndarray) -> float:
    """$\\lVert n^t/N^Q-\\bar C\\bar\\pi\\rVert_1$ from the stored posterior means (``c_mean`` flattened in the ``cap-ml``
    layout, so ``c_mean[j, k]``$=\\bar C_{kj}$); ``true_ct`` gives only its predicted margins $n^t/N^Q$."""
    pi_mean = np.asarray(pi_mean, dtype=float)
    c_mean = np.asarray(c_mean, dtype=float).reshape(len(pi_mean), len(pi_mean))
    shares = np.asarray(true_ct, dtype=float).sum(axis=0)
    return float(np.abs(shares - pi_mean @ c_mean).sum())


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def sampler_columns(df: pd.DataFrame, ok: np.ndarray, prefix: str) -> dict:
    """The NUTS diagnostics, Hf summaries, cells and mismatch of a new method on one pair, over the bags in ``ok``."""
    divergences = _numeric(df, "nuts_divergences")[ok]
    rhat = _numeric(df, "nuts_max_rhat")[ok]
    ess = _numeric(df, "nuts_min_ess")[ok]
    mismatches = [
        mismatch_hf(t, c, p) for t, c, p, keep in zip(df["true_cts"], df["c_mean"], df["h_pi_mean"], ok) if keep
    ]
    out = {
        "share_failed": float(np.mean(~ok)),
        "share_divergent": float(np.mean(divergences > 0)) if ok.any() else np.nan,
        "mean_divergences": float(np.mean(divergences)) if ok.any() else np.nan,
        "max_rhat": float(np.max(rhat)) if ok.any() else np.nan,
        "median_rhat": float(np.median(rhat)) if ok.any() else np.nan,
        "share_rhat_high": float(np.mean(rhat > RHAT_HIGH)) if ok.any() else np.nan,
        "min_ess": float(np.min(ess)) if ok.any() else np.nan,
        "mismatch": float(np.mean(mismatches)) if mismatches else np.nan,
        "n_empty_cells": int(_numeric(df, "n_empty_cells")[0]),
        "n_class_density_cells": int(_numeric(df, "n_class_density_cells")[0]),
        "val_ece": float(_numeric(df, "val_ece")[0]),
    }
    for name in HF_SUMMARIES:
        values = _numeric(df, name)[ok]
        out[name] = float(np.nanmean(values)) if ok.any() and np.isfinite(values).any() else np.nan
    return {f"{prefix}_{name}": value for name, value in out.items()}


def pair_tests(bags: dict[str, pd.DataFrame], key: tuple[str, str]) -> dict:
    """The four tests of one pair, without the Holm adjustment, on the bags where neither method failed; the generator
    of each test depends only on the seed, the pair and the test."""
    row = {}
    for index, (name, (method, reference)) in enumerate(TESTS.items()):
        a, b = bags[method], bags[reference]
        baycap_fixed._check_paired(a, b, key, (method, reference))
        ok = ~(a["failed"].to_numpy() | b["failed"].to_numpy())
        diff = (a["winkler"] - b["winkler"]).to_numpy()[ok]
        if len(diff) == 0:
            low = high = p_value = mean = np.nan
        else:
            low, high, p_value = pushforward.bootstrap_mean_test(diff, pair_rng(*key, seed=SEED + 1000 * index))
            mean = float(diff.mean())
        row |= {f"{name}_diff_mean": mean, f"{name}_ci_low": low, f"{name}_ci_high": high, f"{name}_p": p_value}
    return row


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (method -> stored results, the same pairs and bags for every method)."""
    rows = []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
    for key, first in groups[METHODS[0]].items():
        bags = {method: bag_frame(groups[method][key], method) for method in METHODS}
        for method in METHODS[1:]:
            baycap_fixed._check_paired(bags[METHODS[0]], bags[method], key, (METHODS[0], method))
        counts = val_counts[key]
        rates = rate_matrix(counts)
        row = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
            "n_val": int(counts.sum()),
            "min_val_per_class": int(counts.sum(axis=1).min()),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
            row[f"{method}_std_shift"] = standardised_shift(bags[method])
        for source in SOURCES:
            df = groups[NEW[source]][key]
            ok = ~bags[NEW[source]]["failed"].to_numpy()
            terms = baycap_hf.shift_terms(df.loc[ok]) if ok.any() else dict.fromkeys(SHIFT_TERMS, np.nan)
            row |= {f"{NEW[source]}_{name}": value for name, value in terms.items()}
            row |= sampler_columns(df, ok, NEW[source])
            fixed = groups[FIXED[source]][key]
            row[f"{FIXED[source]}_mismatch"] = float(
                np.mean([baycap_fixed.mismatch(t, rates, p) for t, p in zip(fixed["true_cts"], fixed["prev_mean"])])
            )
        row |= pair_tests(bags, key)
        rows.append(row)
    table = pd.DataFrame(rows)
    for name in TESTS:
        table[f"{name}_p_holm"] = pushforward.holm(table[f"{name}_p"].to_numpy())
    return ordered(table)


def v_alone_sd(c_hat: np.ndarray, n_s: np.ndarray) -> np.ndarray:
    """$\\sqrt{\\hat C_{kj}(1-\\hat C_{kj})/(n^s_j+1)}$, the standard deviation of a cell from $V$ alone (§T.13), for
    ``c_hat`` in the ``cap-ml`` layout (``c_hat[j, k]``$=\\hat C_{kj}$) and the class totals $n^s_j$."""
    c_hat = np.asarray(c_hat, dtype=float)
    return np.sqrt(c_hat * (1 - c_hat) / (np.asarray(n_s, dtype=float)[:, None] + 1))


def cell_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair, new method and cell $(k,j)$: the count, the rates, the standard deviations and their ratio."""
    rows = []
    for source in SOURCES:
        method = NEW[source]
        for key, df in results[method].groupby(KEYS, sort=False):
            counts = np.asarray(val_counts[key], dtype=float)
            n_classes = counts.shape[0]
            n_s = counts.sum(axis=1)
            c_hat = np.divide(counts, n_s[:, None], out=np.zeros_like(counts), where=n_s[:, None] > 0)
            sd_v = v_alone_sd(c_hat, n_s)
            ok = ~baycap_h.failed_bags(df)
            if not ok.any():
                continue
            c_sd = np.mean([np.asarray(s, dtype=float) for s, keep in zip(df["c_sd"], ok) if keep], axis=0)
            c_sd = c_sd.reshape(n_classes, n_classes)
            for j in range(n_classes):
                for k in range(n_classes):
                    rows.append(
                        {
                            "dataset": key[0],
                            "classifier": key[1],
                            "n_classes": n_classes,
                            "method": method,
                            "input_classifier": source,
                            "true_class": j,
                            "pred_class": k,
                            "f": int(counts[j, k]),
                            "n_s": int(n_s[j]),
                            "c_hat": float(c_hat[j, k]),
                            "sd_v": float(sd_v[j, k]),
                            "sd_post": float(c_sd[j, k]),
                            "sd_ratio": float(c_sd[j, k] / sd_v[j, k]) if sd_v[j, k] > 0 else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def load_all(acc_name: str = ACC_NAME) -> tuple[dict[str, pd.DataFrame], list]:
    """The stored results of ``METHODS`` on the pairs stored for all of them."""
    results = {}
    for method in METHODS:
        df = baycap_fixed.load_method(method, acc_name)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT} ({acc_name}).")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {METHODS} ({acc_name}).")
    return baycap_fixed._restrict(results, pairs), pairs


def analysis_summary(table: pd.DataFrame, cells: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the tables to the summaries of the Analysis of E3.2, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage, amplitude, Winkler score and standardised shift of the six methods, and the mean
      over the pairs of the squared standardised shift (near $1$ for draws of a standard normal);
    - ``tests``: per test, the pairs where the new method is significantly (Holm) lower and higher than its reference;
    - ``sampler``: per new method, the pairs with divergent bags, with largest split-$\\hat R$ above ``RHAT_HIGH``, with
      failed bags, and the mean mismatch of the new and of the E3.1 method;
    - ``cells``: per new method, the median ratio of standard deviations and the share of ratios below $1$, for the
      cells with $f_{kj}<10$, $10\\le f_{kj}<100$ and $f_{kj}\\ge100$.
    """
    out = {"scores": [], "tests": [], "sampler": [], "cells": []}
    for k_group, sub in baycap_h._k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        scores = dict(key)
        for method in METHODS:
            for score in (*SCORES, "std_shift"):
                scores[f"{method}_{score}"] = float(sub[f"{method}_{score}"].mean())
            scores[f"{method}_std_shift_sq"] = float((sub[f"{method}_std_shift"] ** 2).mean())
        out["scores"].append(scores)
        tests = dict(key)
        for name in TESTS:
            significant = sub[f"{name}_p_holm"] < TEST_LEVEL
            lower = sub[f"{name}_diff_mean"] < 0
            tests[f"{name}_lower"] = int((lower & significant).sum())
            tests[f"{name}_higher"] = int((~lower & significant).sum())
        out["tests"].append(tests)
        for source in SOURCES:
            new = NEW[source]
            out["sampler"].append(
                key
                | {
                    "method": new,
                    "pairs_divergent": int((sub[f"{new}_share_divergent"] > 0).sum()),
                    "pairs_rhat_high": int((sub[f"{new}_max_rhat"] > RHAT_HIGH).sum()),
                    "pairs_failed": int((sub[f"{new}_share_failed"] > 0).sum()),
                    "mismatch": float(sub[f"{new}_mismatch"].mean()),
                    "mismatch_fixed": float(sub[f"{FIXED[source]}_mismatch"].mean()),
                }
            )
    if not cells.empty:
        bins = pd.cut(cells["f"], [0, 10, 100, np.inf], right=False, labels=["f<10", "10<=f<100", "f>=100"])
        for (method, label), part in cells.groupby(["method", bins], observed=True):
            ratio = part["sd_ratio"].dropna()
            out["cells"].append(
                {
                    "method": method,
                    "f_bin": label,
                    "cells": len(part),
                    "median_ratio": float(ratio.median()) if len(ratio) else np.nan,
                    "share_below_1": float((ratio < 1).mean()) if len(ratio) else np.nan,
                }
            )
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


SHORT_LABELS = {
    NEW["h"]: "Hf-KDEy h",
    NEW["mlp"]: "Hf-KDEy mlp",
    FIXED["h"]: "KDEy h",
    FIXED["mlp"]: "KDEy mlp",
    HF: "Hf",
    SQS_HBS: "SQS-H\\textsubscript{bs}",
}
# the scores of the comparison table, in the order of E3.1
TABLE_SCORES = baycap_fixed.TABLE_SCORES


def _rotated(label: str) -> str:
    """A method label set vertically in the LaTeX header, as in the tables of ``main``."""
    return f"\\rotatebox{{90}}{{{label}}}"


def _comparison_styler(table: pd.DataFrame):
    """Columns grouped by score (AE, amplitude, coverage, Winkler, as in E3.1), then by method, with the standardised
    shift last; each row and score group coloured as in E3.1 (``tables/baycap_fixed.py:_score_cell_styles``); the
    method labels are rotated, as in the tables of ``main``."""
    labels = {score: baycap_fixed.SCORE_LABELS[score] for score in TABLE_SCORES}
    cols = {}
    for score in TABLE_SCORES:
        for method in METHODS:
            cols[(labels[score], _rotated(SHORT_LABELS[method]))] = table[f"{method}_{score}"].to_numpy()
    for method in METHODS:
        cols[("$z$", _rotated(SHORT_LABELS[method]))] = table[f"{method}_std_shift"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = baycap_fixed._score_cell_styles(out, labels)
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


TEST_LABELS = {
    f"{source}_vs_fixed": (f"$s={source}$" if source == "h" else "$s=$ mlp", "$-$ fixed $\\hat C$")
    for source in SOURCES
} | {
    f"{source}_vs_hf": (f"$s={source}$" if source == "h" else "$s=$ mlp", "$-$ \\bcap{}-Hf-real") for source in SOURCES
}


def _tests_styler(table: pd.DataFrame):
    """Per test: the mean Winkler difference (bold where the Holm-adjusted p-value is below 0.05), its $95\\%$ bootstrap
    interval and the Holm-adjusted p-value."""
    cols = {}
    for name in TESTS:
        group = TEST_LABELS[name]
        cols[(*group, "diff.")] = [
            "--" if np.isnan(d) else (f"\\textbf{{{d:.3f}}}" if p < TEST_LEVEL else f"{d:.3f}")
            for d, p in zip(table[f"{name}_diff_mean"], table[f"{name}_p_holm"])
        ]
        cols[(*group, "95\\% CI")] = [
            "--" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"
            for lo, hi in zip(table[f"{name}_ci_low"], table[f"{name}_ci_high"])
        ]
        cols[(*group, "$p_{\\mathrm{Holm}}$")] = ["--" if np.isnan(p) else f"{p:.3f}" for p in table[f"{name}_p_holm"]]
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style


DIAGNOSTIC_COLUMNS = {
    "share_divergent": "div.",
    "max_rhat": "$\\hat R$",
    "min_ess": "ESS",
    "share_failed": "fail",
    "h_ln_nu_median": "$\\ln\\nu$",
    "h_ln_kappa_median": "$\\ln\\kappa$",
    "h_p_kappa_below_1": "$P(\\kappa<1)$",
    "n_empty_cells": "empty",
    "n_class_density_cells": "class",
    "mismatch": "mism.",
}


def _diagnostics_styler(table: pd.DataFrame):
    cols = {}
    for source in SOURCES:
        new = NEW[source]
        for column, header in DIAGNOSTIC_COLUMNS.items():
            cols[(SHORT_LABELS[new], header)] = table[f"{new}_{column}"].to_numpy()
        cols[(SHORT_LABELS[new], "mism.\\ fixed")] = table[f"{FIXED[source]}_mismatch"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    integer = [c for c in out.columns if c[1] in ("empty", "class")]
    return out.style.format(precision=3, na_rep="--").format("{:.0f}", subset=integer, na_rep="--")


COMMON_CAPTION = (
    "BayCAP-KDEy with a random $C$ and the Hf prior on \\texttt{main} (E3.2; vanilla accuracy, $|U|=1000$); means over "
    "the 100 bags, scored against the realised table; Hf-KDEy $s$: \\bcap{}-KDEy-Hf-$s$-real; KDEy $s$: "
    "\\bcap{}-KDEy-$s$-real (fixed $\\hat C$, E3.1); Hf: \\bcap{}-Hf-real. "
)
CAPTIONS = {
    "comparison": COMMON_CAPTION
    + "Absolute error AE, amplitude $\\mathcal{A}$, coverage $\\mathcal{C}$, Winkler score $\\mathcal{W}$ and "
    "standardised shift $z$ "
    "(mean over the bags of $(\\hat A-A)/\\operatorname{sd}(A^{(r)})$). " + baycap_fixed.SCORE_HIGHLIGHT,
    "tests": COMMON_CAPTION
    + "Mean per-bag Winkler difference of each new method minus its reference, $95\\%$ paired bootstrap interval and "
    "Holm-adjusted p-value over the pairs; bold: $p<0.05$.",
    "diagnostics": COMMON_CAPTION
    + "NUTS on $\\pi$, $C$ and the Hf hyperparameters: share of bags with divergent transitions, largest "
    "split-$\\hat R$, smallest effective sample size, share of failed bags; means over the bags of the posterior medians "
    "of $\\ln\\nu$ and $\\ln\\kappa$ and of $P(\\kappa<1)$; empty cells of $V$ and those with the density of their "
    "true class; label-free mismatch $\\lVert n^t/N^Q-\\bar C\\bar\\pi\\rVert_1$, and that of the method with a fixed "
    "$\\hat C$.",
}


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str | None = None) -> str:
    stem = TABLE_NAME if name is None else f"{TABLE_NAME}_{name}"
    return os.path.join(directory or table_dir(), f"{stem}.csv")


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


# the comparison (30 columns) and diagnostics (22 columns) tables are typeset on A3, the tests table on A4
A3_TABLES = ("comparison", "diagnostics")


def _latex_document(longtable_path: str, a3: bool = False) -> str:
    """The document of E3.1, with ``graphicx`` for the rotated method labels."""
    document = baycap_fixed._latex_document(longtable_path, a3=a3)
    return document.replace("\\begin{document}", "\\usepackage{graphicx}\n\\begin{document}", 1)


def gen_tables(pargs):
    results, pairs = load_all()
    val_counts = load_validation_counts(pairs, domain=DOMAIN)
    table = pair_table(results, val_counts)
    cells = cell_table(results, val_counts)
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    cells.to_csv(csv_path(out_dir, "cells"), index=False)

    stylers = {
        "comparison": _comparison_styler(table),
        "tests": _tests_styler(table),
        "diagnostics": _diagnostics_styler(table),
    }
    for name, styler in stylers.items():
        label = f"tab:{TABLE_NAME}-{name}"
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, CAPTIONS[name], label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{_stem(name)}_long.tex", a3=name in A3_TABLES))

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in analysis_summary(table, cells).items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")
    print(f"pairs: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for name in CAPTIONS:
        tex_filename = f"{_stem(name)}_doc.tex"
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
    parser = ArgumentParser(description="BayCAP-KDEy with a random C and the Hf prior (E3.2).")
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
