r"""Tempering of BayCAP-Hf and BayCAP-KDEy on the protocol of ``main`` (work-plan card WP-20, E5.2 of
``docs/ext5_baycap_priors_tempering.md``).

E5.2 runs ``BayesCAP-Hf-real`` (BayCAP-Hf, both likelihood terms divided by $\tau$) and, since WP-33,
``BayCAP-KDEy-h-inst`` and ``BayCAP-KDEy-h-inst-cb`` (BayCAP-KDEy with the input classifier $h$, a fixed $\hat C$ and the
per-instance pushforward, the likelihood of the bag divided by $\tau$, the weights untempered)
(``experiments/methods.py:TEMPERED_BASES``; until WP-33 ``BayCAP-KDEy-mlp-real``, whose results are invalidated; the
predicted widening below is computed for it only, ``WIDENING_METHOD``, since §T.6 derives it for the cell pushforward) at the temperatures
$\tau\in\{0.5,1,2,4,8\}$ on the $75$ pairs of ``main`` (``experiments/methods.py:TEMPERED_VARIANTS``; $\tau=1$ is the
stored untempered method). Both methods target the realised table of each bag and are scored against it
(``true_cts``); the scores use the target labels, so they are analyses. The pairs are those where the five temperatures
of a method are stored, so the table also works on the pilot.

Per pair (``pair_table``, one row per metric, method and pair, every column in the CSV), for each $\tau$: the means over
the bags of AE, coverage, amplitude and Winkler score at $\alpha=0.05$, the shares of bags with divergent transitions
and of bags where NUTS produced no draws (left out of the scores), the share of divergent draws (mean over the bags with
draws of the divergent transitions over the ``NUTS_DRAWS`` draws, as in E2.8), the largest split-$\hat R$, and the median over the
bags of the ratio of the amplitude to that at $\tau=1$ (the observed widening); for BayCAP-KDEy and vanilla accuracy,
the predicted widening, the median over the bags of $\sqrt{(\tau+r^2)/(1+r^2)}$ with $r=\bar\sigma_g/\sigma_p$ stored
per bag at $\tau=1$ (§T.6). The bags of the five temperatures are checked to be the same (equal stored realised tables).

Selection (``selection``, per metric and method):

- leave one dataset out (a method: no label of the held-out dataset is used): for each dataset $d$, $\tau_{-d}$ is the
  $\tau$ with the lowest mean, over the pairs of the other datasets, of the per-pair mean Winkler score (ties go to the
  $\tau$ closest to $1$ on the log scale, then to the smaller one); the pairs of $d$ are scored at $\tau_{-d}$;
- references, chosen with the target labels (never methods): the global best $\tau$, with the lowest mean over all the
  pairs, and the oracle-best $\tau$ of each pair, with its lowest mean Winkler score.

Decision rule (vanilla accuracy; computed for every metric, applied to vanilla accuracy only): per dataset, the mean over
its pairs of the per-pair mean Winkler score at $\tau=1$ and at $\tau_{-d}$; tempering is retained if the second is
lower, by a one-sided paired Wilcoxon signed-rank test over the datasets (alternative: lower at $\tau_{-d}$) with
$p<0.05$; otherwise $\tau=1$ is kept.

Outputs, under ``<OUT_DIR>/uncawe/tables/tempering/``: the per-pair CSV (read by ``plots/tempering.py``), the summary CSV
(per metric, method, group of $K$ and column: the fixed $\tau$, the leave-one-dataset-out selection and the two
references), the per-dataset CSV ($\tau_{-d}$ and the per-dataset Winkler scores), the test CSV and, per metric, the
summary, the per-dataset $\tau_{-d}$ and the per-pair scores (Winkler score and coverage at each $\tau$, $\tau_{-d}$ and
the oracle-best $\tau$) as LaTeX tabular fragments, longtables and standalone documents.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncawe import env
from uncawe.experiments.main import EXPERIMENT
from uncawe.experiments.methods import (
    INVALIDATED_TEMPERED_BASES,
    TEMPERATURE_LEVELS,
    TEMPERED_BASES,
    TEMPERED_VARIANTS,
)
from uncawe.experiments.metrics import acc_from_ct, get_acc_names
from uncawe.results import Results
from uncawe.tables import baycap_fixed, eb_prior, hierarchical
from uncawe.tables import main as main_tables
from uncawe.tables.baycap_hf import DIVERGENT_DRAWS, NUTS_DRAWS
from uncawe.tables.common import METHOD_LATEX_PREAMBLE
from uncawe.tables.estimand_gap import CLASSIFIER_ORDER, booktabs_clines

TABLE_NAME = "tempering"
DOMAIN = "classic"
KEYS = ["dataset", "classifier"]
SCORES = ("ae", "coverage", "amplitude", "winkler")
REFERENCE_TAU = 1.0
TAUS = tuple(sorted((REFERENCE_TAU, *TEMPERATURE_LEVELS)))
DECISION_METRIC = "vanilla_accuracy"
DECISION_LEVEL = 0.05
# the method whose amplitude ratio is compared with the prediction of §T.6, and the metric of its stored sigma_p, sigma_g
WIDENING_METHOD = "BayCAP-KDEy-mlp-real"
WIDENING_METRIC = "vanilla_accuracy"
K_GROUPS = hierarchical.K_GROUPS


def method_name(base: str, tau: float) -> str:
    """The stored method of ``base`` at temperature ``tau``: ``base`` itself at $\\tau=1$."""
    if tau == REFERENCE_TAU:
        return base
    name = f"{base}-t{tau:g}"
    if TEMPERED_VARIANTS.get(name) != (base, tau):
        raise ValueError(f"{name} is not a tempered method of E5.2")
    return name


def tau_label(tau: float) -> str:
    return f"{tau:g}"


# --------------------------------------------------------------------------------------------------- loading


def load_method(method: str, acc_name: str, root: str | None = None, experiment: str = EXPERIMENT) -> pd.DataFrame:
    """The stored rows of ``method`` for ``acc_name`` in ``experiment`` (default ``main``; E5.3 reads its calibration
    runs from ``tau_select``), empty if none."""
    base_dir = os.path.join(env.root_dir if root is None else root, experiment)
    try:
        return Results.load(
            base_dir=base_dir,
            domain=DOMAIN,
            acc_name=acc_name,
            filter_classifiers=list(CLASSIFIER_ORDER),
            filter_methods=[method],
        ).df
    except ValueError:  # no file: pd.concat of an empty list
        return pd.DataFrame()


def load_all(
    acc_name: str, root: str | None = None, experiment: str = EXPERIMENT, bases=TEMPERED_BASES
) -> dict[str, dict[float, pd.DataFrame]]:
    """``base method -> {tau: rows}`` of ``experiment`` for the methods of ``bases``, restricted to the pairs stored for
    the five temperatures; the methods without any such pair are left out."""
    out = {}
    for base in bases:
        frames = {tau: load_method(method_name(base, tau), acc_name, root, experiment) for tau in TAUS}
        if any(df.empty for df in frames.values()):
            continue
        pairs = eb_prior.common_pairs(frames)
        if not pairs:
            continue
        index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
        out[base] = {
            tau: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
            for tau, df in frames.items()
        }
    return out


# --------------------------------------------------------------------------------------------------- scores


def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name not in df:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def bag_frame(df: pd.DataFrame, acc_name: str) -> pd.DataFrame:
    """One row per bag of one pair (stored order): the true value of the metric on the realised table, AE, coverage,
    amplitude and Winkler score, whether NUTS produced no draws (``failed``, scores NaN), whether it had divergent
    transitions, its split-$\\hat R$, and the stored $\\sigma_p$ and $\\bar\\sigma_g$ (NaN where not stored)."""
    rows = []
    for row in df.itertuples(index=False):
        true_acc = acc_from_ct(acc_name, np.asarray(row.true_cts, dtype=float))
        samples = np.asarray(row.estim_acc_ranges, dtype=float)
        failed = not np.all(np.isfinite(samples))
        rows.append(
            {
                "true_acc": true_acc,
                "failed": failed,
                "ae": np.nan if failed else float(row.acc_err),
                "coverage": np.nan if failed else main_tables.get_acc_coverage("", true_acc, samples),
                "amplitude": np.nan if failed else main_tables.get_acc_amplitude("", samples),
                "winkler": np.nan if failed else main_tables.get_acc_winkler("", true_acc, samples),
            }
        )
    out = pd.DataFrame(rows)
    out["divergences"] = _column(df, "nuts_divergences")
    out["divergent"] = np.nan_to_num(out["divergences"]) > 0
    out["rhat"] = _column(df, "nuts_max_rhat")
    out["acc_sd"] = _column(df, "acc_sd")
    out["sigma_g"] = _column(df, "sigma_g")
    return out


def predicted_widening(tau: float, acc_sd: np.ndarray, sigma_g: np.ndarray) -> np.ndarray:
    """$\\sqrt{(\\tau+r^2)/(1+r^2)}$ per bag, with $r=\\bar\\sigma_g/\\sigma_p$ at $\\tau=1$ (§T.6): tempering the bag
    term multiplies the prevalence part $\\sigma_p^2$ of the variance by $\\tau$ and leaves the bag part
    $\\bar\\sigma_g^2$ unchanged; NaN where $\\sigma_p^2+\\bar\\sigma_g^2=0$."""
    acc_sd, sigma_g = np.asarray(acc_sd, dtype=float), np.asarray(sigma_g, dtype=float)
    num, den = tau * acc_sd**2 + sigma_g**2, acc_sd**2 + sigma_g**2
    return np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))


def _check_same_bags(frames: dict[float, pd.DataFrame], key) -> None:
    """The stored realised tables of the bags of one pair are the same at every temperature."""
    reference = np.stack([np.ravel(ct) for ct in frames[REFERENCE_TAU]["true_cts"]])
    for tau, frame in frames.items():
        tables = np.stack([np.ravel(ct) for ct in frame["true_cts"]])
        if tables.shape != reference.shape or not np.allclose(tables, reference):
            raise ValueError(f"{key}: the bags at tau={tau:g} differ from those at tau=1")


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b > 0)


def _finite_max(values) -> float:
    """The largest finite value, NaN if there is none (e.g. no split-$\\hat R$ stored)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else np.nan


def _share_divergent_draws(bags: pd.DataFrame) -> float:
    """The mean over the bags with draws of the share of divergent draws (divergent transitions over the
    ``NUTS_DRAWS`` draws of a bag), as in ``tables/baycap_hf.py``; NaN where no divergences are stored."""
    divergences = bags.loc[~bags["failed"], "divergences"].to_numpy(dtype=float)
    divergences = divergences[np.isfinite(divergences)]
    return float(divergences.mean() / NUTS_DRAWS) if divergences.size else np.nan


def pair_rows(frames: dict[float, pd.DataFrame], acc_name: str, base: str) -> list[dict]:
    """The rows of ``pair_table`` of one metric and base method (``frames``: tau -> stored rows)."""
    groups = {tau: dict(list(df.groupby(KEYS, sort=False))) for tau, df in frames.items()}
    rows = []
    for key, first in groups[REFERENCE_TAU].items():
        _check_same_bags({tau: groups[tau][key] for tau in TAUS}, key)
        bags = {tau: bag_frame(groups[tau][key], acc_name) for tau in TAUS}
        reference = bags[REFERENCE_TAU]
        row = {
            "acc_name": acc_name,
            "method": base,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
        }
        widening = base == WIDENING_METHOD and acc_name == WIDENING_METRIC
        for tau in TAUS:
            t = tau_label(tau)
            for score in SCORES:
                row[f"t{t}_{score}"] = float(bags[tau][score].mean())
            row[f"t{t}_share_failed"] = float(bags[tau]["failed"].mean())
            row[f"t{t}_share_divergent"] = float(bags[tau]["divergent"].mean())
            row[f"t{t}_share_divergent_draws"] = _share_divergent_draws(bags[tau])
            row[f"t{t}_max_rhat"] = _finite_max(bags[tau]["rhat"])
            row[f"t{t}_widening"] = float(np.nanmedian(_ratio(bags[tau]["amplitude"], reference["amplitude"])))
            row[f"t{t}_widening_predicted"] = (
                float(np.nanmedian(predicted_widening(tau, reference["acc_sd"], reference["sigma_g"])))
                if widening
                else np.nan
            )
        rows.append(row)
    return rows


def pair_table(
    acc_names: list[str] | None = None, root: str | None = None, experiment: str = EXPERIMENT, bases=TEMPERED_BASES
) -> pd.DataFrame:
    """One row per metric, base method (of ``bases``) and pair of ``experiment`` (default ``main``), ordered by those
    keys, then by $K$, dataset and classifier (as in ``CLASSIFIER_ORDER``)."""
    rows = []
    for acc_name in acc_names or get_acc_names():
        for base, frames in load_all(acc_name, root, experiment, bases).items():
            rows.extend(pair_rows(frames, acc_name, base))
    if not rows:
        raise RuntimeError(f"No pair of {experiment} has stored results for the five temperatures of a method of E5.2.")
    return ordered(pd.DataFrame(rows))


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["_acc"] = table["acc_name"].map(get_acc_names().index)
    # the bases of the studies first, then the former BayCAP-KDEy (only in tables built from its stored results)
    table["_method"] = table["method"].map((*TEMPERED_BASES, *INVALIDATED_TEMPERED_BASES).index)
    table["_clf"] = table["classifier"].map(lambda c: CLASSIFIER_ORDER.index(c) if c in CLASSIFIER_ORDER else 99)
    table = table.sort_values(["_acc", "_method", "n_classes", "dataset", "_clf"])
    return table.drop(columns=["_acc", "_method", "_clf"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- selection


def best_tau(mean_winkler: dict[float, float]) -> float:
    """The $\\tau$ with the lowest value of ``mean_winkler`` ($\\tau$ -> mean Winkler score); ties go to the $\\tau$
    closest to $1$ on the log scale, then to the smaller one; NaN if every value is NaN."""
    finite = {tau: w for tau, w in mean_winkler.items() if np.isfinite(w)}
    if not finite:
        return np.nan
    lowest = min(finite.values())
    return min((tau for tau, w in finite.items() if w == lowest), key=lambda tau: (abs(np.log(tau)), tau))


def _winkler(part: pd.DataFrame, tau: float) -> np.ndarray:
    return part[f"t{tau_label(tau)}_winkler"].to_numpy(dtype=float)


def lodo_taus(part: pd.DataFrame) -> dict[str, float]:
    """For each dataset $d$ of ``part`` (the pairs of one metric and method), $\\tau_{-d}$: the best $\\tau$
    (:func:`best_tau`) of the means over the pairs of the other datasets of the per-pair mean Winkler score."""
    out = {}
    for dataset in part["dataset"].unique():
        others = part.loc[part["dataset"] != dataset]
        out[dataset] = best_tau({tau: float(np.nanmean(_winkler(others, tau))) for tau in TAUS})
    return out


def oracle_taus(part: pd.DataFrame) -> np.ndarray:
    """Per pair, the oracle-best $\\tau$ (a reference: it uses the target labels): the one with the lowest mean
    Winkler score (:func:`best_tau`)."""
    return np.array([best_tau({tau: row[f"t{tau_label(tau)}_winkler"] for tau in TAUS}) for _, row in part.iterrows()])


def scores_at(part: pd.DataFrame, taus: np.ndarray) -> pd.DataFrame:
    """The scores of each pair of ``part`` at its own $\\tau$ (``taus``, one per pair; NaN gives NaN scores)."""
    out = {}
    for score in SCORES:
        values = np.full(len(part), np.nan)
        for i, tau in enumerate(taus):
            if np.isfinite(tau):
                values[i] = part[f"t{tau_label(tau)}_{score}"].iloc[i]
        out[score] = values
    return pd.DataFrame(out, index=part.index)


def one_sided_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """$p$-value of the one-sided paired Wilcoxon signed-rank test of the alternative "$x$ lower than $y$"; NaN with
    fewer than two finite nonzero differences (e.g. when $\\tau_{-d}=1$ for every dataset)."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    if np.count_nonzero(diff) < 2:
        return np.nan
    return float(wilcoxon(diff, alternative="less").pvalue)


def selection(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The selections and the test of the decision rule, per metric and method.

    - ``pairs``: per pair, $\\tau_{-d}$ of its dataset and the oracle-best $\\tau$, with the scores at each;
    - ``datasets``: per dataset, $\\tau_{-d}$ and the means over its pairs of the per-pair mean Winkler score at
      $\\tau=1$ and at $\\tau_{-d}$;
    - ``tests``: per metric and method, the global best $\\tau$, the number of datasets, the number where $\\tau_{-d}$
      is not $1$ and where it lowers the per-dataset Winkler score, the mean difference, the one-sided $p$-value and
      whether tempering is retained ($p<0.05$; the decision is that of ``DECISION_METRIC``).
    """
    pair_rows_, dataset_rows, test_rows = [], [], []
    for (acc_name, base), part in table.groupby(["acc_name", "method"], sort=False):
        lodo = lodo_taus(part)
        pair_lodo = part["dataset"].map(lodo).to_numpy(dtype=float)
        oracle = oracle_taus(part)
        at_lodo, at_oracle = scores_at(part, pair_lodo), scores_at(part, oracle)
        frame = part[["acc_name", "method", "dataset", "classifier", "n_classes"]].copy()
        frame["tau_lodo"], frame["tau_oracle"] = pair_lodo, oracle
        for score in SCORES:
            frame[f"lodo_{score}"] = at_lodo[score].to_numpy()
            frame[f"oracle_{score}"] = at_oracle[score].to_numpy()
        pair_rows_.append(frame)

        w_1, w_lodo = [], []
        for dataset, tau in lodo.items():
            sub = frame.loc[frame["dataset"] == dataset]
            ref = float(np.nanmean(_winkler(part.loc[part["dataset"] == dataset], REFERENCE_TAU)))
            sel = float(np.nanmean(sub["lodo_winkler"]))
            w_1.append(ref)
            w_lodo.append(sel)
            dataset_rows.append(
                {
                    "acc_name": acc_name,
                    "method": base,
                    "dataset": dataset,
                    "n_classes": int(sub["n_classes"].iloc[0]),
                    "pairs": len(sub),
                    "tau_lodo": tau,
                    "winkler_t1": ref,
                    "winkler_lodo": sel,
                }
            )
        w_1, w_lodo = np.array(w_1), np.array(w_lodo)
        p_value = one_sided_wilcoxon(w_lodo, w_1)
        global_tau = best_tau({tau: float(np.nanmean(_winkler(part, tau))) for tau in TAUS})
        test_rows.append(
            {
                "acc_name": acc_name,
                "method": base,
                "tau_global": global_tau,
                "datasets": len(w_1),
                "datasets_tau_not_1": int(sum(tau != REFERENCE_TAU for tau in lodo.values())),
                "datasets_lower": int(np.sum(w_lodo < w_1)),
                "d_winkler": float(np.nanmean(w_lodo - w_1)),
                "p_value": p_value,
                "retained": bool(np.isfinite(p_value) and p_value < DECISION_LEVEL),
                "decision_metric": acc_name == DECISION_METRIC,
            }
        )
    return {
        "pairs": pd.concat(pair_rows_, ignore_index=True),
        "datasets": pd.DataFrame(dataset_rows),
        "tests": pd.DataFrame(test_rows),
    }


# --------------------------------------------------------------------------------------------------- summary


def k_groups(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("all", table)]
    for name in K_GROUPS:
        part = table.loc[table["n_classes"].map(hierarchical._k_group) == name]
        if not part.empty:
            groups.append((name, part))
    return groups


def summary_table(table: pd.DataFrame, selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per metric, method, group of $K$ and column (each fixed $\\tau$, then ``lodo``, ``global`` and ``oracle``): the
    number of pairs and the means over the pairs of the four scores; for the fixed $\\tau$ also the shares of bags with
    divergent transitions and without draws, the mean share of divergent draws and the number of pairs where it exceeds
    ``DIVERGENT_DRAWS`` (the reliability criterion of E2.8), the median over the pairs of the largest split-$\\hat R$,
    and the medians over the pairs of the observed and predicted widening.
    ``global`` and ``oracle`` are references (they use the target labels)."""
    tests = selected["tests"].set_index(["acc_name", "method"])
    picks = selected["pairs"]
    rows = []
    for (acc_name, base), sub in table.groupby(["acc_name", "method"], sort=False):
        global_tau = tests.loc[(acc_name, base), "tau_global"]
        sub_picks = picks.loc[(picks["acc_name"] == acc_name) & (picks["method"] == base)]
        for k_group, part in k_groups(sub):
            key = {"acc_name": acc_name, "method": base, "K": k_group, "pairs": len(part)}
            for tau in TAUS:
                t = tau_label(tau)
                row = key | {"column": t, "reference": False}
                row |= {score: float(part[f"t{t}_{score}"].mean()) for score in SCORES}
                row["share_divergent"] = float(part[f"t{t}_share_divergent"].mean())
                draws = part[f"t{t}_share_divergent_draws"]
                row["share_divergent_draws"] = float(draws.mean())
                row["pairs_divergent_draws"] = int((draws > DIVERGENT_DRAWS).sum())
                row["median_max_rhat"] = float(part[f"t{t}_max_rhat"].median())
                row["share_failed"] = float(part[f"t{t}_share_failed"].mean())
                row["widening"] = float(part[f"t{t}_widening"].median())
                row["widening_predicted"] = float(part[f"t{t}_widening_predicted"].median())
                rows.append(row)
            chosen = _matching(sub_picks, part)
            rows.append(key | {"column": "lodo", "reference": False} | _means(chosen, "lodo"))
            g = tau_label(global_tau) if np.isfinite(global_tau) else None
            rows.append(
                key
                | {"column": "global", "reference": True}
                | {score: float(part[f"t{g}_{score}"].mean()) if g else np.nan for score in SCORES}
            )
            rows.append(key | {"column": "oracle", "reference": True} | _means(chosen, "oracle"))
    return pd.DataFrame(rows)


def _matching(picks: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``picks`` of the pairs of ``part``."""
    keys = pd.MultiIndex.from_frame(part[KEYS])
    return picks.loc[pd.MultiIndex.from_frame(picks[KEYS]).isin(keys)]


def _means(picks: pd.DataFrame, prefix: str) -> dict:
    return {score: float(picks[f"{prefix}_{score}"].mean()) for score in SCORES}


def oracle_distribution(selected: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per metric and method, the number of pairs whose oracle-best $\\tau$ is each value (a reference)."""
    picks = selected["pairs"]
    rows = []
    for (acc_name, base), part in picks.groupby(["acc_name", "method"], sort=False):
        row = {"acc_name": acc_name, "method": base}
        row |= {tau_label(tau): int(np.sum(part["tau_oracle"] == tau)) for tau in TAUS}
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

# BayCAP-KDEy is BayCAP-KDEy-h-inst since E3.4 (card WP-31); the former one keeps its own label (card WP-33)
METHOD_LABELS = {
    "BayesCAP-Hf-real": "\\bcaphf",
    "BayCAP-KDEy-h-inst": "\\bcapkdey",
    "BayCAP-KDEy-h-inst-cb": "\\bcapkdey{}-cb",
    "BayCAP-KDEy-mlp-real": "\\bcap{}-KDEy-mlp-real",
}
K_LABELS = {"all": "all", "K=2": "$K=2$", "3<=K<=7": "$3\\le K\\le7$", "K>=8": "$K\\ge8$"}
SCORE_LABELS = {
    "ae": "AE",
    "coverage": main_tables.metrics_map["coverage"],
    "amplitude": main_tables.metrics_map["amplitude"],
    "winkler": main_tables.metrics_map["winkler"],
}
COLUMN_LABELS = {
    **{tau_label(tau): f"$\\tau={tau_label(tau)}$" for tau in TAUS},
    "lodo": "$\\tau_{-d}$",
    "global": "global",
    "oracle": "oracle",
}
REFERENCE_COLUMNS = ("global", "oracle")
ACC_LABELS = {"vanilla_accuracy": "vanilla accuracy", "macro-F1": "macro-$F_1$", "macro-K": "macro-$K$"}


def _summary_styler(summary: pd.DataFrame, tests: pd.DataFrame):
    """One row per method, group of $K$ and score, one column per fixed $\\tau$, then $\\tau_{-d}$ and the two
    references (grey), and the $p$-value of the test of the decision rule (on the rows of all pairs)."""
    rows = []
    p_values = tests.set_index("method")["p_value"]
    for (base, k_group), part in summary.groupby(["method", "K"], sort=False):
        by_column = part.set_index("column")
        for score in SCORES:
            row = {"method": METHOD_LABELS[base], "$K$": K_LABELS[k_group], "score": SCORE_LABELS[score]}
            row |= {label: by_column.loc[column, score] for column, label in COLUMN_LABELS.items()}
            row["$p$"] = p_values.get(base, np.nan) if (k_group == "all" and score == "winkler") else np.nan
            rows.append(row)
    out = pd.DataFrame(rows).set_index(["method", "$K$", "score"])
    grey = [COLUMN_LABELS[column] for column in REFERENCE_COLUMNS]
    return (
        out.style.format(precision=3, na_rep="--")
        .map(lambda _: "color:{gray};", subset=grey)
        .map(lambda _: "bfseries: ;", subset=[COLUMN_LABELS["lodo"]])
    )


def _datasets_styler(datasets: pd.DataFrame):
    """One row per method and dataset: $\\tau_{-d}$ and the per-dataset Winkler scores at $\\tau=1$ and $\\tau_{-d}$."""
    out = pd.DataFrame(
        {
            "$K$": datasets["n_classes"].to_numpy(),
            "$\\tau_{-d}$": datasets["tau_lodo"].map(lambda t: tau_label(t) if np.isfinite(t) else "--").to_numpy(),
            "$\\mathcal{W}_{\\tau=1}$": datasets["winkler_t1"].to_numpy(),
            "$\\mathcal{W}_{\\tau_{-d}}$": datasets["winkler_lodo"].to_numpy(),
        }
    )
    out.index = pd.MultiIndex.from_arrays(
        [
            datasets["method"].map(METHOD_LABELS),
            datasets["dataset"].map(lambda d: f"\\texttt{{{main_tables._escape_latex_text(d)}}}"),
        ],
        names=["method", "dataset"],
    )
    return out.style.format(precision=3, na_rep="--").format("{:.0f}", subset=["$K$"])


def score_styles(out: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """Per row and score group (``labels``: score -> first level of the column labels of the group), the colour scale of
    the tables of E3.1 (``tables/baycap_fixed.py:_score_cell_styles``, the min-max scale of ``tables/main.py``): from
    the worst value of the row in the group (red) to the best (green, in bold); lower is better for the Winkler score,
    closer to the nominal level for the coverage."""
    return baycap_fixed._score_cell_styles(out, labels)


def _tau_text(tau: float) -> str:
    return tau_label(tau) if np.isfinite(tau) else "--"


def _pair_index(pairs: pd.DataFrame) -> pd.MultiIndex:
    """Method, dataset and classifier of each row, as LaTeX."""
    return pd.MultiIndex.from_arrays(
        [
            pairs["method"].map(METHOD_LABELS),
            pairs["dataset"].map(lambda d: f"\\texttt{{{main_tables._escape_latex_text(d)}}}"),
            pairs["classifier"].map(main_tables._escape_latex_text),
        ],
        names=["method", "dataset", "classifier"],
    )


def _pairs_styler(table: pd.DataFrame, picks: pd.DataFrame):
    """One row per method and pair (``table``: the pair table of one metric; ``picks``: the ``pairs`` frame of
    :func:`selection`): $K$, the mean Winkler score and the coverage at each fixed $\\tau$ (each coloured per row from
    the worst to the best value, the best in bold: for the Winkler score, the oracle-best $\\tau$), $\\tau_{-d}$ and the oracle-best $\\tau$ (grey, a reference)."""
    merged = table.merge(picks[["acc_name", "method", *KEYS, "tau_lodo", "tau_oracle"]], how="left")
    columns = {("", "$K$"): merged["n_classes"].to_numpy()}
    labels = [COLUMN_LABELS[tau_label(tau)] for tau in TAUS]
    for score in ("winkler", "coverage"):
        for tau, label in zip(TAUS, labels):
            columns[(SCORE_LABELS[score], label)] = merged[f"t{tau_label(tau)}_{score}"].to_numpy(dtype=float)
    columns[("selected $\\tau$", COLUMN_LABELS["lodo"])] = merged["tau_lodo"].map(_tau_text).to_numpy()
    columns[("selected $\\tau$", COLUMN_LABELS["oracle"])] = merged["tau_oracle"].map(_tau_text).to_numpy()
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _pair_index(merged)
    styles = score_styles(out, {score: SCORE_LABELS[score] for score in ("winkler", "coverage")})
    return (
        out.style.format(precision=3, na_rep="--")
        .format("{:.0f}", subset=[("", "$K$")])
        .apply(lambda _: styles, axis=None)
        .map(lambda _: "color:{gray};", subset=[("selected $\\tau$", COLUMN_LABELS["oracle"])])
    )


def caption(acc_name: str, name: str) -> str:
    metric = ACC_LABELS[acc_name]
    common = (
        f"E5.2, tempering on \\texttt{{main}} ({metric}); means over the pairs of the per-pair means over the 100 bags, "
        "scored against the realised table. "
    )
    if name == "summary":
        return common + (
            "$\\tau_{-d}$: each dataset scored at the $\\tau$ chosen on the other datasets (leave one dataset out); "
            "references, chosen with the target labels (grey): global, the $\\tau$ with the lowest mean Winkler score "
            "over all pairs; oracle, the best $\\tau$ of each pair. $p$: one-sided paired Wilcoxon signed-rank test over "
            "the datasets of the per-dataset Winkler score at $\\tau_{-d}$ against $\\tau=1$ (decision rule on vanilla "
            "accuracy, $p<0.05$)."
        )
    if name == "pairs":
        return (
            f"E5.2, tempering on \\texttt{{main}} ({metric}); one row per pair: means over the 100 bags, scored "
            "against the realised table, at each $\\tau$, each score coloured per row from the worst (red) to the best (green, in bold; "
            "coverage: closest to $0.95$). "
            "$\\tau_{-d}$: the $\\tau$ chosen on the other datasets (leave one dataset out); oracle (grey, a reference "
            "chosen with the target labels): the best $\\tau$ of the pair."
        )
    return common + (
        "Per held-out dataset: the $\\tau_{-d}$ chosen on the other datasets and the mean over its pairs of the "
        "Winkler score at $\\tau=1$ and at $\\tau_{-d}$."
    )


def _latex_document(longtable_path: str) -> str:
    return eb_prior._latex_document(longtable_path).replace(
        "\\begin{document}", "\\usepackage[table]{xcolor}\n" + METHOD_LATEX_PREAMBLE + "\n\\begin{document}", 1
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


TABLES = ("summary", "datasets", "pairs")


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    selected = selection(table)
    summary = summary_table(table, selected)
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    summary.to_csv(csv_path(out_dir, "summary"), index=False)
    selected["pairs"].to_csv(csv_path(out_dir, "selection"), index=False)
    selected["datasets"].to_csv(csv_path(out_dir, "datasets"), index=False)
    selected["tests"].to_csv(csv_path(out_dir, "tests"), index=False)
    oracle = oracle_distribution(selected)
    oracle.to_csv(csv_path(out_dir, "oracle"), index=False)

    for acc_name in summary["acc_name"].unique():
        stylers = {
            "summary": _summary_styler(
                summary.loc[summary["acc_name"] == acc_name],
                selected["tests"].loc[selected["tests"]["acc_name"] == acc_name],
            ),
            "datasets": _datasets_styler(selected["datasets"].loc[selected["datasets"]["acc_name"] == acc_name]),
            "pairs": _pairs_styler(table.loc[table["acc_name"] == acc_name], selected["pairs"]),
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
                f.write(_latex_document(f"tabular/{stem}_long.tex"))

    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
        print(selected["tests"].to_string(index=False))
        print(oracle.to_string(index=False))
    print(f"rows: {len(table)}; tables written to {out_dir}")


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
    parser = ArgumentParser(description="Tempering of BayCAP-Hf and BayCAP-KDEy on main (E5.2).")
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
