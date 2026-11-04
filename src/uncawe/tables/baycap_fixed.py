"""BayCAP with a fixed rate matrix $\\hat C$, quantifier × input classifier × pushforward, on the pairs of ``main``
(work-plan card WP-8, E3.1 of ``docs/ext3_posterior_propagation.md``).

The eight methods ``BayCAP-<Q>-<s>`` and ``BayCAP-<Q>-<s>-real`` (``methods/baycap_fixed.py:BayCAPFixed``, quantifier
$Q\\in\\{$KDEy, EMQ$\\}$, input classifier $s\\in\\{h$, mlp$\\}$) push the posterior draws $\\pi^{(r)}$ of a Bayesian
quantifier through the rate matrix $\\hat C$ of $h$ on $V$: the four without ``-real`` target the population table
$\\hat C\\operatorname{diag}(\\pi^Q)$, the four with ``-real`` the realised table of the target bag. All are scored against
the realised table of each bag (``true_cts``), next to the stored ``BayesCAP``, ``BayesCAP-real``, ``BayesCAP-Hf-real``
(E2.8) and ``RQBS``. Vanilla accuracy, $|U|=1000$, the pairs where the eight methods and the four references have
stored results (so it also works on the pilot).

Per pair (``compute_table``, one row per pair, every column in the CSV):

- AE, coverage, amplitude and Winkler score of the twelve methods, means over the bags;
- for each cell (a quantifier and an input classifier): the observed widening factor, the median over the bags of the
  amplitude of the realised method divided by that of the population method, and the predicted one, the median over the
  bags of $\\sqrt{1+(\\bar\\sigma_g/\\sigma_p)^2}$ from the stored ``sigma_g`` and ``acc_sd`` (§T.8); the movement of the
  centre, the mean over the bags of the realised minus the population point estimate; the label-free mismatch, the mean
  over the bags of $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$, with $n^t/N^Q$ the predicted-class shares of the bag (the
  predicted margins of ``true_cts``, which do not use the labels) and $\\bar\\pi$ the stored ``prev_mean``; the NUTS
  diagnostics of $\\pi$ (share of bags with divergent transitions, mean divergent transitions per bag, largest
  split-$\\hat R$, smallest effective sample size; the two methods of a cell share their draws, so they are read from
  the realised one);
- the top-label ECE of the two input classifiers on the validation outputs the quantifier is fitted on (``val_ece``);
  the spread of the recalls of $h$ on $V$, $\\operatorname{sd}_j(\\hat C_{jj})$;
- sixteen paired bootstrap tests of the mean per-bag Winkler difference (``tables/pushforward.py:bootstrap_mean_test``,
  $10{,}000$ resamples), each Holm-adjusted over the pairs on its own: in each cell, the realised method minus the
  population one, minus ``BayesCAP-real`` and minus ``BayesCAP-Hf-real``; among the realised methods, ``h`` minus
  ``mlp`` for each quantifier and ``KDEy`` minus ``EMQ`` for each input classifier. All pairs are reported, also those
  where NUTS is unreliable for BayCAP-Hf in E2.8, with no flag.

Also:

- ``min_prev_table``: the coverage of the eight methods in bins of the smallest true class prevalence of the bag,
  $\\min_j\\pi^Q_j$ (``MIN_PREV_BINS``), over all the bags of the pairs;
- ``design_summary``: for each metric of ``main`` (vanilla accuracy, macro-F1, macro-K) and each pushforward, the $2\\times2$
  design (quantifier × input classifier) of the mean rank of the Winkler score among the four cells of the pushforward
  (rank $1$ the lowest mean Winkler score of the pair), the mean coverage and the mean amplitude, over the pairs and per
  classifier $h$.

The scores, the bins of $\\min_j\\pi^Q_j$ and the tests use the true tables, so they are analyses; $n^t$, $\\bar\\pi$ and
the ECE use no target labels. The split of the shift of E2.2 is ``tables/displacement.py --method <method>``, with
prior part $R_0=0$ for these methods. Outputs, under ``<OUT_DIR>/uncawe/tables/baycap_fixed/``: the per-pair CSV, which
``plots/baycap_fixed.py`` reads, the CSVs of the two other summaries, and five tables (realised; population;
pushforward; tests; design), each as a LaTeX tabular fragment, a longtable and a standalone landscape document.
Validation counts and tables follow the ``cap-ml`` layout (true classes on rows); $\\hat C$ of ``rate_matrix`` is in the
paper's layout.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import acc_from_ct, get_acc_names
from uncawe.experiments.methods import FIXED_C_VARIANTS
from uncawe.main import EXPERIMENT
from uncawe.results import Results
from uncawe.tables import baycap_h, eb_prior, estimand_gap, pushforward
from uncawe.tables import main as main_tables
from uncawe.tables.baycap_hf import RHAT_HIGH
from uncawe.tables.common import METHOD_LATEX_LABELS
from uncawe.tables.displacement import load_validation_counts, ordered, rate_matrix
from uncawe.tables.estimand_gap import ACC_NAME, CLASSIFIER_ORDER, SEED, booktabs_clines, pair_rng

TABLE_NAME = "baycap_fixed"
KEYS = eb_prior.KEYS
DOMAIN = estimand_gap.DOMAIN
QUANTIFIER_LABELS = {"kdey": "KDEy", "emq": "EMQ"}
SOURCES = ("h", "mlp")
CELLS = tuple(f"{label}-{source}" for label in QUANTIFIER_LABELS.values() for source in SOURCES)
PUSHFORWARDS = ("population", "realised")
REFERENCES = {"population": "BayesCAP", "realised": "BayesCAP-real"}
# the best BayCAP prior of E2.8, realised only
HF = "BayesCAP-Hf-real"
RQBS = "RQBS"
SCORES = ("ae", "coverage", "amplitude", "winkler")
SCORE_LABELS = {"ae": "AE"} | {score: main_tables.metrics_map[score] for score in SCORES[1:]}
# bins of the smallest true class prevalence of a bag, left-closed (the smallest prevalence is at most 1/2)
MIN_PREV_BINS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0)
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL


def method_name(cell: str, pushforward_name: str) -> str:
    """The name in ``main`` of the method of a cell (e.g. ``"KDEy-h"``) and a pushforward."""
    return f"BayCAP-{cell}" + ("-real" if pushforward_name == "realised" else "")


FIXED_METHODS = [method_name(cell, p) for cell in CELLS for p in PUSHFORWARDS]
assert sorted(FIXED_METHODS) == sorted(FIXED_C_VARIANTS)
METHODS = FIXED_METHODS + list(REFERENCES.values()) + [HF, RQBS]


def _tests() -> dict[str, tuple[str, str]]:
    """The sixteen tests, name -> (method, reference): the per-bag difference is method minus reference. The tests
    against BayCAP-Hf come last, so the generators of the other twelve do not depend on them."""
    tests = {}
    for cell in CELLS:
        real = method_name(cell, "realised")
        tests[f"{cell}_real_vs_pop"] = (real, method_name(cell, "population"))
        tests[f"{cell}_real_vs_bcap"] = (real, REFERENCES["realised"])
    for label in QUANTIFIER_LABELS.values():
        tests[f"{label}_h_vs_mlp"] = (method_name(f"{label}-h", "realised"), method_name(f"{label}-mlp", "realised"))
    for source in SOURCES:
        tests[f"{source}_kdey_vs_emq"] = (
            method_name(f"KDEy-{source}", "realised"),
            method_name(f"EMQ-{source}", "realised"),
        )
    for cell in CELLS:
        tests[f"{cell}_real_vs_hf"] = (method_name(cell, "realised"), HF)
    return tests


TESTS = _tests()


def load_method(method: str, acc_name: str = ACC_NAME) -> pd.DataFrame:
    """The stored ``main`` results of ``method`` for ``acc_name`` (``classic`` domain)."""
    results = Results.load(
        base_dir=os.path.join(env.root_dir, EXPERIMENT),
        domain=DOMAIN,
        acc_name=acc_name,
        filter_classifiers=list(CLASSIFIER_ORDER),
        filter_methods=[method],
    )
    return results.df


def bag_frame(df: pd.DataFrame, method: str, acc_name: str = ACC_NAME) -> pd.DataFrame:
    """One row per bag of one pair (stored order): the true value of the metric, the point estimate, AE, coverage,
    amplitude and Winkler score, and the smallest true class prevalence."""
    rows = []
    for row in df.itertuples(index=False):
        true_ct = np.asarray(row.true_cts, dtype=float)
        true_acc = acc_from_ct(acc_name, true_ct)
        samples = np.asarray(row.estim_acc_ranges, dtype=float)
        rows.append(
            {
                "true_acc": true_acc,
                "point": float(row.estim_accs),
                "ae": float(row.acc_err),
                "coverage": main_tables.get_acc_coverage(method, true_acc, samples),
                "amplitude": main_tables.get_acc_amplitude(method, samples),
                "winkler": main_tables.get_acc_winkler(method, true_acc, samples),
                "min_prev": float(true_ct.sum(axis=1).min()),
            }
        )
    return pd.DataFrame(rows)


def mismatch(true_ct: np.ndarray, rates: np.ndarray, prev_mean: np.ndarray) -> float:
    """$\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$: ``true_ct`` gives only its predicted margins $n^t/N^Q$ (column sums in
    the ``cap-ml`` layout), ``rates`` is $\\hat C$ in the paper's layout."""
    shares = np.asarray(true_ct, dtype=float).sum(axis=0)
    return float(np.abs(shares - np.asarray(rates, dtype=float) @ np.asarray(prev_mean, dtype=float)).sum())


def predicted_widening(acc_sd: np.ndarray, sigma_g: np.ndarray) -> np.ndarray:
    """$\\sqrt{1+(\\bar\\sigma_g/\\sigma_p)^2}$ per bag (§T.8); infinite where $\\sigma_p=0$."""
    acc_sd, sigma_g = np.asarray(acc_sd, dtype=float), np.asarray(sigma_g, dtype=float)
    ratio = np.divide(sigma_g, acc_sd, out=np.full_like(sigma_g, np.inf), where=acc_sd > 0)
    return np.sqrt(1 + ratio**2)


def _median_ratio(num: np.ndarray, den: np.ndarray) -> float:
    ratio = np.divide(num, den, out=np.full_like(num, np.inf, dtype=float), where=den > 0)
    return float(np.median(ratio))


def _check_paired(a: pd.DataFrame, b: pd.DataFrame, key, names) -> None:
    if len(a) != len(b) or not np.allclose(a["true_acc"].to_numpy(), b["true_acc"].to_numpy()):
        raise ValueError(f"The bags of {key} differ between {names[0]} and {names[1]}")


def cell_columns(real_df: pd.DataFrame, bags: dict[str, pd.DataFrame], cell: str, rates: np.ndarray) -> dict:
    """The per-pair columns of one cell: widening factors, movement of the centre, mismatch, NUTS diagnostics, ECE."""
    pop, real = bags[method_name(cell, "population")], bags[method_name(cell, "realised")]
    acc_sd = pd.to_numeric(real_df["acc_sd"]).to_numpy()
    sigma_g = pd.to_numeric(real_df["sigma_g"]).to_numpy()
    divergences = pd.to_numeric(real_df["nuts_divergences"]).to_numpy()
    mismatches = [mismatch(t, rates, p) for t, p in zip(real_df["true_cts"], real_df["prev_mean"])]
    return {
        f"{cell}_widening": _median_ratio(real["amplitude"].to_numpy(), pop["amplitude"].to_numpy()),
        f"{cell}_widening_pred": float(np.median(predicted_widening(acc_sd, sigma_g))),
        f"{cell}_centre_move": float((real["point"] - pop["point"]).mean()),
        f"{cell}_mismatch": float(np.mean(mismatches)),
        f"{cell}_sigma_p": float(np.mean(acc_sd)),
        f"{cell}_sigma_g": float(np.mean(sigma_g)),
        f"{cell}_share_divergent": float(np.mean(divergences > 0)),
        f"{cell}_mean_divergences": float(np.mean(divergences)),
        f"{cell}_max_rhat": float(pd.to_numeric(real_df["nuts_max_rhat"]).max()),
        f"{cell}_min_ess": float(pd.to_numeric(real_df["nuts_min_ess"]).min()),
        f"{cell}_val_ece": float(pd.to_numeric(real_df["val_ece"]).iloc[0]),
    }


def pair_tests(bags: dict[str, pd.DataFrame], key: tuple[str, str]) -> dict:
    """The sixteen tests of one pair, without the Holm adjustment; the generator of each test depends only on the seed,
    the pair and the test."""
    row = {}
    for index, (name, (method, reference)) in enumerate(TESTS.items()):
        _check_paired(bags[method], bags[reference], key, (method, reference))
        diff = (bags[method]["winkler"] - bags[reference]["winkler"]).to_numpy()
        low, high, p_value = pushforward.bootstrap_mean_test(diff, pair_rng(*key, seed=SEED + 1000 * index))
        row.update(
            {
                f"{name}_diff_mean": float(diff.mean()),
                f"{name}_ci_low": low,
                f"{name}_ci_high": high,
                f"{name}_p": p_value,
            }
        )
    return row


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (method -> stored results, the same pairs and bags for every method)."""
    rows = []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
    for key, first in groups[METHODS[0]].items():
        bags = {method: bag_frame(groups[method][key], method) for method in METHODS}
        for method in METHODS[1:]:
            _check_paired(bags[METHODS[0]], bags[method], key, (METHODS[0], method))
        counts = val_counts[key]
        rates = rate_matrix(counts)
        row = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
            "n_val": int(counts.sum()),
            "sd_recalls": float(np.std(np.diag(rates))),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
        for cell in CELLS:
            row |= cell_columns(groups[method_name(cell, "realised")][key], bags, cell, rates)
        for source in SOURCES:
            eces = {row[f"{label}-{source}_val_ece"] for label in QUANTIFIER_LABELS.values()}
            if not np.allclose(list(eces), min(eces)):
                raise ValueError(f"The ECE of {source} differs between the quantifiers on {key}")
            row[f"{source}_val_ece"] = min(eces)
        row |= pair_tests(bags, key)
        rows.append(row)
    table = pd.DataFrame(rows)
    for name in TESTS:
        table[f"{name}_p_holm"] = pushforward.holm(table[f"{name}_p"].to_numpy())
    return ordered(table)


def _restrict(results: dict[str, pd.DataFrame], pairs) -> dict[str, pd.DataFrame]:
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    return {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in results.items()
    }


def load_all(acc_name: str = ACC_NAME, methods=None) -> tuple[dict[str, pd.DataFrame], list]:
    """The stored results of ``methods`` (default ``METHODS``) on the pairs stored for all of them."""
    methods = METHODS if methods is None else methods
    results = {}
    for method in methods:
        df = load_method(method, acc_name)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT} ({acc_name}).")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {methods} ({acc_name}).")
    return _restrict(results, pairs), pairs


def compute_table() -> pd.DataFrame:
    """Load the stored results of the twelve methods and the validation counts, and return the ordered table."""
    results, pairs = load_all()
    return pair_table(results, load_validation_counts(pairs, domain=DOMAIN))


def min_prev_table(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Coverage of the eight methods in the bins ``MIN_PREV_BINS`` of $\\min_j\\pi^Q_j$, over all the bags."""
    rows = []
    for method in FIXED_METHODS:
        bags = pd.concat(
            [bag_frame(df, method) for _, df in results[method].groupby(KEYS, sort=False)], ignore_index=True
        )
        bins = pd.cut(bags["min_prev"], MIN_PREV_BINS, right=False)
        for interval, part in bags.groupby(bins, observed=True):
            rows.append(
                {
                    "method": method,
                    "bin_low": float(interval.left),
                    "bin_high": float(interval.right),
                    "n_bags": len(part),
                    "coverage": float(part["coverage"].mean()),
                    "amplitude": float(part["amplitude"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _design_rows(scores: pd.DataFrame, acc_name: str, group: str) -> list[dict]:
    rows = []
    for p in PUSHFORWARDS:
        methods = [method_name(cell, p) for cell in CELLS]
        ranks = scores[[f"{m}_winkler" for m in methods]].rank(axis=1, method="average")
        for cell, method in zip(CELLS, methods):
            label, source = cell.split("-")
            rows.append(
                {
                    "acc_name": acc_name,
                    "classifier": group,
                    "pushforward": p,
                    "quantifier": label,
                    "input_classifier": source,
                    "pairs": len(scores),
                    "winkler_rank": float(ranks[f"{method}_winkler"].mean()),
                    "coverage": float(scores[f"{method}_coverage"].mean()),
                    "amplitude": float(scores[f"{method}_amplitude"].mean()),
                    "winkler": float(scores[f"{method}_winkler"].mean()),
                }
            )
    return rows


def design_summary(acc_names=None) -> pd.DataFrame:
    """The $2\\times2$ design of each pushforward and metric: mean Winkler rank among the four cells, mean coverage and
    amplitude, over the pairs stored for the eight methods (``"all"``) and per classifier $h$."""
    rows = []
    for acc_name in acc_names or get_acc_names():
        try:
            results, _ = load_all(acc_name, FIXED_METHODS)
        except RuntimeError:
            continue
        records = []
        groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
        for key in groups[FIXED_METHODS[0]]:
            record = dict(zip(KEYS, key))
            for method in FIXED_METHODS:
                bags = bag_frame(groups[method][key], method, acc_name)
                for score in ("coverage", "amplitude", "winkler"):
                    record[f"{method}_{score}"] = float(bags[score].mean())
            records.append(record)
        scores = pd.DataFrame(records)
        rows += _design_rows(scores, acc_name, "all")
        for classifier in CLASSIFIER_ORDER:
            part = scores.loc[scores["classifier"] == classifier]
            if not part.empty:
                rows += _design_rows(part, acc_name, classifier)
    return pd.DataFrame(rows)


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E3.1, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage, amplitude and Winkler score of the twelve methods;
    - ``cells``: per cell, the mean and median observed and predicted widening factor, the mean movement of the centre,
      the mean mismatch, the pairs with divergent bags and with largest split-$\\hat R$ above ``RHAT_HIGH`` ($1.1$, as E2.8);
    - ``tests``: per test, the pairs where the method is significantly (Holm) lower and higher than its reference.
    """
    out = {"scores": [], "cells": [], "tests": []}
    for k_group, sub in baycap_h._k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        scores = dict(key)
        for method in METHODS:
            for score in SCORES:
                scores[f"{method}_{score}"] = float(sub[f"{method}_{score}"].mean())
        out["scores"].append(scores)
        for cell in CELLS:
            out["cells"].append(
                key
                | {
                    "cell": cell,
                    "widening_mean": float(sub[f"{cell}_widening"].replace(np.inf, np.nan).mean()),
                    "widening_median": float(sub[f"{cell}_widening"].median()),
                    "widening_pred_median": float(sub[f"{cell}_widening_pred"].median()),
                    "centre_move": float(sub[f"{cell}_centre_move"].mean()),
                    "mismatch": float(sub[f"{cell}_mismatch"].mean()),
                    "pairs_divergent": int((sub[f"{cell}_share_divergent"] > 0).sum()),
                    "pairs_rhat_high": int((sub[f"{cell}_max_rhat"] > RHAT_HIGH).sum()),
                }
            )
        tests = dict(key)
        for name in TESTS:
            significant = sub[f"{name}_p_holm"] < TEST_LEVEL
            lower = sub[f"{name}_diff_mean"] < 0
            tests[f"{name}_lower"] = int((lower & significant).sum())
            tests[f"{name}_higher"] = int((~lower & significant).sum())
        out["tests"].append(tests)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


CELL_LABELS = {cell: cell.replace("-", " ") for cell in CELLS}


def _scores_styler(table: pd.DataFrame, methods: list[str], extra: dict[tuple[str, str], str]):
    cols = {}
    for method in methods:
        for score in ("coverage", "amplitude", "winkler"):
            cols[(METHOD_LATEX_LABELS.get(method, method), SCORE_LABELS[score])] = table[f"{method}_{score}"].to_numpy()
    for label, column in extra.items():
        cols[label] = table[column].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style.format(precision=3, na_rep="--")


def _realised_styler(table: pd.DataFrame):
    methods = [method_name(cell, "realised") for cell in CELLS] + [REFERENCES["realised"], HF, RQBS]
    extra = {("ECE", "$h$"): "h_val_ece", ("ECE", "mlp"): "mlp_val_ece"}
    return _scores_styler(table, methods, extra)


def _population_styler(table: pd.DataFrame):
    methods = [method_name(cell, "population") for cell in CELLS] + [REFERENCES["population"], RQBS]
    return _scores_styler(table, methods, {("$V$", "$\\operatorname{sd}_j\\hat C_{jj}$"): "sd_recalls"})


def _pushforward_styler(table: pd.DataFrame):
    cols = {}
    for cell in CELLS:
        label = CELL_LABELS[cell]
        cols[(label, "wid.")] = table[f"{cell}_widening"].to_numpy()
        cols[(label, "pred.")] = table[f"{cell}_widening_pred"].to_numpy()
        cols[(label, "move")] = table[f"{cell}_centre_move"].to_numpy()
        cols[(label, "mism.")] = table[f"{cell}_mismatch"].to_numpy()
        cols[(label, "div.")] = table[f"{cell}_share_divergent"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style.format(precision=3, na_rep="--").format(
        lambda v: "$\\infty$" if np.isinf(v) else f"{v:.2f}",
        subset=[c for c in out.columns if c[1] in ("wid.", "pred.")],
    )


TEST_LABELS = (
    {f"{cell}_real_vs_pop": (CELL_LABELS[cell], "real$-$pop") for cell in CELLS}
    | {f"{cell}_real_vs_bcap": (CELL_LABELS[cell], "real$-$\\bcap{}-real") for cell in CELLS}
    | {f"{cell}_real_vs_hf": (CELL_LABELS[cell], "real$-$\\bcap{}-Hf-real") for cell in CELLS}
    | {f"{label}_h_vs_mlp": (label, "$h-$mlp") for label in QUANTIFIER_LABELS.values()}
    | {f"{source}_kdey_vs_emq": (source, "KDEy$-$EMQ") for source in SOURCES}
)


def _tests_styler(table: pd.DataFrame):
    """The mean Winkler differences of the sixteen tests, in bold where the Holm-adjusted p-value is below 0.05."""
    cols = {}
    for name in TESTS:
        cols[TEST_LABELS[name]] = [
            "--" if np.isnan(d) else (f"\\textbf{{{d:.3f}}}" if p < TEST_LEVEL else f"{d:.3f}")
            for d, p in zip(table[f"{name}_diff_mean"], table[f"{name}_p_holm"])
        ]
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style


def _design_styler(summary: pd.DataFrame):
    part = summary.loc[summary["classifier"] == "all"]
    cols = {}
    index = []
    for (acc_name, p, label), rows in part.groupby(["acc_name", "pushforward", "quantifier"], sort=False):
        index.append((acc_name, p, label))
        for _, row in rows.iterrows():
            for column, header in (
                ("winkler_rank", "rank"),
                ("coverage", "$\\mathcal{C}$"),
                ("amplitude", "$\\mathcal{A}$"),
            ):
                cols.setdefault((f"$s={row['input_classifier']}$", header), []).append(row[column])
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = pd.MultiIndex.from_tuples(index, names=["metric", "pushforward", "quantifier"])
    return out.style.format(precision=3, na_rep="--")


COMMON_CAPTION = (
    "BayCAP with a fixed $\\hat C$ on \\texttt{main} (E3.1; vanilla accuracy, $|U|=1000$); means over the 100 bags, "
    "scored against the realised table. "
)
CAPTIONS = {
    "realised": COMMON_CAPTION
    + "Realised methods, \\bcap{}-real, \\bcap{}-Hf-real and SQS-H: coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$, Winkler score "
    "$\\mathcal{W}$; ECE: top-label calibration error (15 bins) of the input classifier on its validation outputs.",
    "population": COMMON_CAPTION
    + "Population methods, \\bcap{} and SQS-H; $\\operatorname{sd}_j\\hat C_{jj}$: spread of the recalls of $h$ on $V$.",
    "pushforward": COMMON_CAPTION
    + "Per cell: observed widening factor (median over bags of the amplitude ratio realised/population), predicted "
    "factor $\\sqrt{1+(\\bar\\sigma_g/\\sigma_p)^2}$ (median over bags), movement of the centre (realised minus "
    "population point estimate), label-free mismatch $\\lVert n^t/N^Q-\\hat C\\bar\\pi\\rVert_1$, share of bags with "
    "divergent transitions.",
    "tests": COMMON_CAPTION
    + "Mean per-bag Winkler difference of each test; bold: paired bootstrap, Holm-adjusted over the pairs, $p<0.05$.",
    "design": "BayCAP with a fixed $\\hat C$ (E3.1): mean rank of the Winkler score among the four cells of each "
    "pushforward, mean coverage and amplitude over the pairs, for each metric of \\texttt{main}.",
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


def _latex_document(longtable_path: str) -> str:
    return eb_prior._latex_document(longtable_path).replace(
        "\\begin{document}", "\\newcommand{\\bcap}{BayCAP}\n\\newcommand{\\rqbs}{SQS-H}\n\\begin{document}"
    )


def gen_tables(pargs):
    table = compute_table()
    results, _ = load_all()
    min_prev = min_prev_table(results)
    design = design_summary()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    min_prev.to_csv(csv_path(out_dir, "min_prev"), index=False)
    design.to_csv(csv_path(out_dir, "design"), index=False)

    stylers = {
        "realised": _realised_styler(table),
        "population": _population_styler(table),
        "pushforward": _pushforward_styler(table),
        "tests": _tests_styler(table),
        "design": _design_styler(design),
    }
    for name, styler in stylers.items():
        label = f"tab:{TABLE_NAME}-{name}"
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, CAPTIONS[name], label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{_stem(name)}_long.tex"))

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in analysis_summary(table).items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")
        print(f"\nmin_prev:\n{min_prev.to_string(index=False)}")
        print(f"\ndesign (all classifiers):\n{design.loc[design['classifier'] == 'all'].to_string(index=False)}")
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
    parser = ArgumentParser(
        description="BayCAP with a fixed rate matrix, quantifier x input classifier x pushforward (E3.1)."
    )
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
