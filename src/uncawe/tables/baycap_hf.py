"""BayCAP-Hf, a lighter tail on the strength and learned shares of the mistakes, on the pairs of ``main`` (work-plan card
WP-11, E2.8 of ``docs/ext2_baycap.md``).

``BayesCAP-Hf-real`` is ``methods/baycap.py:BayCAP`` with ``c_prior="hierarchical-f"`` (§T.16): $m\\sim U(0,1)$,
$\\rho=1/(1+\\nu)\\sim U(0,1)$, $C_{jj}\\mid m,\\nu\\sim\\operatorname{Beta}(\\nu m,\\nu(1-m))$, the mistakes of each true
class shared by $\\operatorname{Dir}(\\kappa\\mathbf 1_{K-1})$ with $1/(1+\\kappa)\\sim U(0,1)$ (no $\\kappa$ with $K=2$),
sampled by NUTS with $C$ and $\\pi^Q$. It is compared bag by bag with the stored ``BayesCAP-H-real`` (E2.7),
``BayesCAP-EB-real`` (empirical-Bayes prior) and ``BayesCAP-real`` ($\\operatorname{Dir}(\\mathbf 1)$ prior). All four
target the realised table of the target sample and are scored against it (``true_cts``). Vanilla accuracy, $|U|=1000$,
the pairs where the four have stored results (so it also works on the pilot).

Per pair:

- the comparison: AE, coverage, amplitude and Winkler score of the four methods (means over the bags), and the paired
  bootstrap tests of the mean per-bag difference of AE and of the Winkler score, BayCAP-Hf minus BayCAP-H and minus
  the empirical-Bayes method (``tables/pushforward.py:bootstrap_mean_test``), each Holm-adjusted over the pairs; the
  bags where a hierarchical method produced no draws (``nuts_failed``) are left out of its scores and of its tests;
- the split of the mean shift $\\overline{\\hat A-A}$ into the recall and the prevalence terms
  (``tables/displacement.py:error_terms``) for BayCAP-Hf, BayCAP-H and the empirical-Bayes method;
- the sampler of BayCAP-Hf and BayCAP-H: the share of bags with divergent transitions, the largest split-$\\hat R$ and the
  smallest effective sample size (``tables/eb_prior.py:diagnostics_table``), the mean number of divergent transitions
  per bag and the share of divergent draws (that mean over the ``NUTS_DRAWS`` draws of a bag), both over the bags with
  draws, and the share of failed bags;
- the posterior of the runs: means over the bags of the posterior mean of $m$, of the median and the $5\\%$ and $95\\%$
  quantiles of $\\ln\\nu$ and of $P(\\nu>\\min_jn_j)$, for both; of the median and the $5\\%$ and $95\\%$ quantiles of
  $\\ln\\kappa$ and of $P(\\kappa<1)$, for BayCAP-Hf ($K\\ge3$);
- from $V$: the share of off-diagonal cells with $f_{ij}=0$, the classes perfect and never right;
- a failure class of BayCAP-Hf for each pair that does not mix (largest split-$\\hat R$ above ``RHAT_HIGH``) or whose
  share of divergent draws exceeds ``DIVERGENT_DRAWS``: ``funnel`` if the mean posterior $P(\\nu>\\min_jn_j)$ exceeds
  ``FUNNEL_HIGH``; else ``boundary`` if a class is perfect or never right on $V$, or if the posterior median of $\\kappa$
  (the mean over the bags of the median of $\\ln\\kappa$) is below $1$ and some off-diagonal cell is empty on $V$; else
  ``other``.

The scores and the split of the shift use the true tables, so they are analyses. Outputs, under
``<OUT_DIR>/uncawe/tables/baycap_hf/``: a per-pair CSV with every column, which ``plots/baycap_hf.py`` reads, and two
tables (comparison; sampler and posterior), each as a LaTeX tabular fragment, a longtable and a standalone landscape
document. Validation counts and tables follow the ``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import BOOTSTRAP_ITERS
from uncawe.main import EXPERIMENT
from uncawe.tables import baycap_h, eb_prior, estimand_gap, pushforward
from uncawe.tables import hierarchical as h
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import METHOD_LABELS, error_terms, load_validation_counts
from uncawe.tables.estimand_gap import SEED, booktabs_clines, load_results, pair_rng

TABLE_NAME = "baycap_hf"
KEYS = eb_prior.KEYS
DOMAIN = estimand_gap.DOMAIN
METHOD = "BayesCAP-Hf-real"
H_METHOD = baycap_h.METHOD
EB_METHOD = baycap_h.REFERENCES["eb"]
UNIFORM_METHOD = baycap_h.REFERENCES["uniform"]
# the stored methods BayCAP-Hf is tested against, by key of the test columns
REFERENCES = {"h": H_METHOD, "eb": EB_METHOD}
METHODS = [UNIFORM_METHOD, EB_METHOD, H_METHOD, METHOD]
# the methods with a hierarchical prior, with per-bag posterior summaries and possibly failed bags; column prefixes
HIERARCHICAL = {METHOD: "hf", H_METHOD: "h"}
# the methods whose mean shift is split into the recall and the prevalence terms
SHIFT_METHODS = {METHOD: "hf", H_METHOD: "h", EB_METHOD: "eb"}
SHIFT_TERMS = ("shift", "recall_term", "prevalence_term")
SCORES = ("ae", *pushforward.SCORES)
TESTED_SCORES = ("ae", "winkler")
TEST_STATS = ("diff_mean", "ci_low", "ci_high", "p", "p_holm")
NU_POSTERIOR = ("h_m_mean", "h_ln_nu_median", "h_ln_nu_q05", "h_ln_nu_q95", "h_p_nu_above_min_n")
KAPPA_POSTERIOR = ("h_ln_kappa_median", "h_ln_kappa_q05", "h_ln_kappa_q95", "h_p_kappa_below_1")
# a pair fails if more than this share of its draws is divergent (E2.8, decided after the pilot: the share of bags
# with divergent transitions counted a single divergent transition as a failure)
DIVERGENT_DRAWS = 0.01
# the NUTS draws retained after warm-up in each bag of main (``experiments/methods.py``)
NUTS_DRAWS = BOOTSTRAP_ITERS
RHAT_HIGH = 1.1
# E2.7: the mean posterior P(nu > min_j n_j) was at most 0.008 on spambase (divergent, but mixing) and 0.12-0.88 on
# the funnel pairs of mammographic and hand_digits
FUNNEL_HIGH = 0.05
FAILURE_CLASSES = ("funnel", "boundary", "other")


def test_column(key: str, score: str, stat: str) -> str:
    """Name of a test column: ``<reference key>_<score>_<stat>``, e.g. ``h_winkler_p_holm``."""
    return f"{key}_{score}_{stat}"


def empty_off_diagonal_share(val_counts: np.ndarray) -> float:
    """Share of the off-diagonal cells of a validation table with no instance ($f_{ij}=0$, $i\\ne j$)."""
    val_counts = np.asarray(val_counts)
    off = ~np.eye(val_counts.shape[0], dtype=bool)
    return float((val_counts[off] == 0).mean())


def failure_class(
    share_divergent_draws: float,
    max_rhat: float,
    p_funnel: float,
    n_boundary: int,
    ln_kappa_median: float,
    empty_share: float,
) -> str:
    """Class of a pair with failures (E2.8): ``funnel``, ``boundary`` or ``other``; empty for a pair that mixes and
    has at most ``DIVERGENT_DRAWS`` of its draws divergent. The funnel is checked first."""
    if not (share_divergent_draws > DIVERGENT_DRAWS or max_rhat > RHAT_HIGH):
        return ""
    if p_funnel > FUNNEL_HIGH:
        return "funnel"
    if n_boundary > 0 or (ln_kappa_median < 0 and empty_share > 0):
        return "boundary"
    return "other"


def divergence_load(df: pd.DataFrame, ok: np.ndarray) -> dict:
    """Mean number of divergent transitions per bag and share of divergent draws, over the bags in ``ok`` (the bags with
    draws); NaN where the results were stored without the NUTS diagnostics."""
    if "nuts_divergences" not in df:
        return {"mean_divergences": np.nan, "share_divergent_draws": np.nan}
    mean = _mean(df, "nuts_divergences", ok)
    return {"mean_divergences": mean, "share_divergent_draws": mean / NUTS_DRAWS}


def bag_frame(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """One row per bag of one pair (stored order): AE, coverage, amplitude, Winkler score and the true accuracy."""
    bags = pushforward.bag_scores(df, method)
    bags["ae"] = pd.to_numeric(df["acc_err"], errors="coerce").to_numpy()
    return bags


def shift_terms(df: pd.DataFrame) -> dict:
    """Means over the bags of the shift and of its recall and prevalence terms (``displacement.error_terms``)."""
    terms = [
        error_terms(np.asarray(e, dtype=float), np.asarray(t, dtype=float))
        for e, t in zip(df["estim_cts"], df["true_cts"])
    ]
    return {name: float(np.mean([t[name] for t in terms])) if terms else np.nan for name in SHIFT_TERMS}


def _mean(df: pd.DataFrame, column: str, ok: np.ndarray) -> float:
    if column not in df or not ok.any():
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce")[ok].mean())


def pair_tests(bags: dict[str, pd.DataFrame], ok: np.ndarray, key: tuple[str, str]) -> dict:
    """The paired bootstrap tests of one pair, BayCAP-Hf minus each reference, on the bags in ``ok`` (the bags where
    no hierarchical method failed); without the Holm adjustment."""
    hf = bags[METHOD]
    row = {}
    for name, reference in REFERENCES.items():
        ref = bags[reference]
        if len(ref) != len(hf) or not np.allclose(ref["true_acc"].to_numpy(), hf["true_acc"].to_numpy()):
            raise ValueError(f"The bags of {key} differ between {METHOD} and {reference}")
        for score in TESTED_SCORES:
            diff = (hf[score] - ref[score]).to_numpy()[ok]
            if len(diff) == 0:
                low = high = p_value = mean = np.nan
            else:
                # the generator depends only on the seed, the pair, the reference and the score
                rng = pair_rng(*key, seed=SEED + 1000 * list(REFERENCES).index(name) + TESTED_SCORES.index(score))
                low, high, p_value = pushforward.bootstrap_mean_test(diff, rng)
                mean = float(diff.mean())
            row.update(
                {
                    test_column(name, score, "diff_mean"): mean,
                    test_column(name, score, "ci_low"): low,
                    test_column(name, score, "ci_high"): high,
                    test_column(name, score, "p"): p_value,
                }
            )
    return row


def pair_row(results: dict[str, pd.DataFrame], val_counts: np.ndarray, key: tuple[str, str]) -> dict:
    """The per-pair columns: scores, tests (without Holm), shift terms, sampler, posterior of the runs, $V$ and class.

    ``results`` maps each method of ``METHODS`` to its stored results on the pair, in the stored order of the bags.
    """
    f, n = h.diagonal_counts(val_counts)
    failed = {method: baycap_h.failed_bags(results[method]) for method in HIERARCHICAL}
    row = {
        "n_classes": int(val_counts.shape[0]),
        "min_val_per_class": int(n.min()),
        "n_bags": len(results[METHOD]),
        "empty_off_diagonal_share": empty_off_diagonal_share(val_counts),
        "n_perfect_classes": int((f == n).sum()),
        "n_never_right_classes": int((f == 0).sum()),
    }
    bags = {}
    for method in METHODS:
        df = results[method]
        ok = ~failed[method] if method in failed else np.ones(len(df), dtype=bool)
        bags[method] = bag_frame(df, method)
        kept = bags[method].loc[ok]
        for score in SCORES:
            row[f"{method}_{score}"] = float(kept[score].mean()) if len(kept) else np.nan
        if method in SHIFT_METHODS:
            terms = shift_terms(df.loc[ok])
            row.update({f"{SHIFT_METHODS[method]}_{term}": value for term, value in terms.items()})

    for method, prefix in HIERARCHICAL.items():
        df, ok = results[method], ~failed[method]
        row[f"{prefix}_share_failed"] = float(failed[method].mean())
        diagnostics = eb_prior.diagnostics_table(df).iloc[0]
        row.update({f"{prefix}_{name}": float(diagnostics[name]) for name in eb_prior.DIAGNOSTICS})
        row.update({f"{prefix}_{name}": value for name, value in divergence_load(df, ok).items()})
        names = NU_POSTERIOR + (KAPPA_POSTERIOR if method == METHOD else ())
        row.update({f"{prefix}_{name.removeprefix('h_')}": _mean(df, name, ok) for name in names})

    both_ok = ~failed[METHOD] & ~failed[H_METHOD]
    row.update(pair_tests(bags, both_ok, key))
    row["failure_class"] = failure_class(
        row["hf_share_divergent_draws"],
        row["hf_max_rhat"],
        row["hf_p_nu_above_min_n"],
        row["n_perfect_classes"] + row["n_never_right_classes"],
        row["hf_ln_kappa_median"],
        row["empty_off_diagonal_share"],
    )
    return row


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (each method restricted to the same pairs), ordered by $K$ as E2.6; the
    p-values of each test are Holm-adjusted over the pairs."""
    rows = []
    for (dataset, classifier), _ in results[METHOD].groupby(KEYS, sort=False):
        pair_results = {
            method: df.loc[(df["dataset"] == dataset) & (df["classifier"] == classifier)].reset_index(drop=True)
            for method, df in results.items()
        }
        key = (dataset, classifier)
        rows.append({"dataset": dataset, "classifier": classifier, **pair_row(pair_results, val_counts[key], key)})
    table = pd.DataFrame(rows)
    for name in REFERENCES:
        for score in TESTED_SCORES:
            p = table[test_column(name, score, "p")].to_numpy()
            table[test_column(name, score, "p_holm")] = pushforward.holm(p)
    return h.ordered(table)


def compute_table() -> pd.DataFrame:
    """Load the stored results of the four methods and the validation counts, and return the ordered table."""
    results = {}
    for method in METHODS:
        df = load_results(method, experiment=EXPERIMENT, domain=DOMAIN)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT}.")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {METHODS}.")
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    results = {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in results.items()
    }
    return pair_table(results, load_validation_counts(pairs, domain=DOMAIN))


def is_unreliable(table: pd.DataFrame) -> pd.Series:
    """The pairs where BayCAP-Hf has a failure class (empty in the table, NaN once read back from the CSV)."""
    return table["failure_class"].fillna("").astype(str) != ""


def _scores_summary(sub: pd.DataFrame, k_group: str) -> dict:
    scores = {"K": k_group, "pairs": len(sub)}
    for method in METHODS:
        for score, label in (("ae", "AE"), ("coverage", "cov"), ("winkler", "W")):
            scores[f"{method}_{label}"] = float(sub[f"{method}_{score}"].mean())
    for name in REFERENCES:
        for score in TESTED_SCORES:
            significant = sub[test_column(name, score, "p_holm")] < pushforward.WINKLER_TEST_LEVEL
            lower = sub[test_column(name, score, "diff_mean")] < 0
            scores[f"{score}_lower_than_{name}"] = int((lower & significant).sum())
            scores[f"{score}_higher_than_{name}"] = int((~lower & significant).sum())
    return scores


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E2.8, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage and Winkler score of the four methods, and, for each test, the pairs where
      BayCAP-Hf is significantly (Holm) lower and higher;
    - ``scores_reliable``: the same, on the pairs where BayCAP-Hf has no failure class (the pairs on which E2.8
      decides; the p-values keep their Holm adjustment over all the pairs);
    - ``sampler``: for BayCAP-Hf and BayCAP-H, the pairs that do not mix (largest $\\hat R$ above ``RHAT_HIGH``), with
      more than ``DIVERGENT_DRAWS`` of their draws divergent, with failed bags; the pairs of BayCAP-Hf in each failure
      class;
    - ``posterior``: the median over the pairs of the median of $\\ln\\nu$ under both methods and of $\\ln\\kappa$, the
      pairs with a posterior median of $\\kappa$ below $1$, the Spearman correlation across pairs of the median of
      $\\ln\\kappa$ with the share of empty off-diagonal cells, and the mean recall and prevalence terms of the three
      methods.
    """
    out = {"scores": [], "scores_reliable": [], "sampler": [], "posterior": []}
    reliable = table.loc[~is_unreliable(table)]
    out["scores_reliable"] = [_scores_summary(sub, k_group) for k_group, sub in baycap_h._k_groups(reliable)]
    for k_group, sub in baycap_h._k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        out["scores"].append(_scores_summary(sub, k_group))

        sampler = dict(key)
        for prefix in HIERARCHICAL.values():
            sampler[f"{prefix}_not_mixing"] = int((sub[f"{prefix}_max_rhat"] > RHAT_HIGH).sum())
            sampler[f"{prefix}_above_draws"] = int((sub[f"{prefix}_share_divergent_draws"] > DIVERGENT_DRAWS).sum())
            sampler[f"{prefix}_with_failed_bags"] = int((sub[f"{prefix}_share_failed"] > 0).sum())
        sampler.update({cls: int((sub["failure_class"] == cls).sum()) for cls in FAILURE_CLASSES})
        out["sampler"].append(sampler)

        posterior = dict(key)
        for prefix in HIERARCHICAL.values():
            posterior[f"{prefix}_median_ln_nu"] = float(sub[f"{prefix}_ln_nu_median"].median())
        kappa = sub["hf_ln_kappa_median"]
        posterior["hf_median_ln_kappa"] = float(kappa.median())
        posterior["kappa_below_1"] = int((kappa < 0).sum())
        many = kappa.notna().sum() > 2
        posterior["spearman_ln_kappa_empty"] = (
            float(kappa.corr(sub["empty_off_diagonal_share"], method="spearman")) if many else np.nan
        )
        for prefix in SHIFT_METHODS.values():
            for term in ("recall_term", "prevalence_term"):
                posterior[f"{prefix}_{term}"] = float(sub[f"{prefix}_{term}"].mean())
        out["posterior"].append(posterior)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


COMPARISON_CAPTION = (
    "BayCAP-Hf-real (hierarchical prior with $1/(1+\\nu)$ uniform and learned shares of the mistakes) against the "
    "stored BayCAP-real ($\\operatorname{Dir}(\\mathbf 1)$ prior), BayCAP-EB-real (empirical-Bayes prior) and BayCAP-H-real "
    "(hierarchical prior), realised pushforward, on \\texttt{main} (E2.8, vanilla accuracy, $|U|=1000$). AE, coverage "
    "$\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ against the realised table, means over the "
    "bags (for the hierarchical methods, over the bags where NUTS produced draws). Tests: mean per-bag difference of AE "
    "and of $\\mathcal{W}$, BayCAP-Hf minus BayCAP-H and minus BayCAP-EB-real, and its two-sided paired-bootstrap "
    "p-value, Holm-adjusted over the pairs."
)
SAMPLER_CAPTION = (
    "Sampler and posterior of BayCAP-Hf-real and BayCAP-H-real on \\texttt{main} (E2.8, vanilla accuracy, $|U|=1000$). "
    "NUTS: share of bags with divergent transitions, mean number of divergent transitions per bag and share of "
    f"divergent draws (of the {NUTS_DRAWS} draws of a bag), over the bags with draws, largest split-$\\hat R$, share of "
    "bags without draws (Hf only). Run: "
    "means over the bags of the median of $\\ln\\nu$ and of $P(\\nu>\\min_jn_j)$, and, for BayCAP-Hf, of the median and "
    "the 90\\% interval of $\\ln\\kappa$ and of $P(\\kappa<1)$. $V$: share of empty off-diagonal cells, classes perfect "
    "or never right. Mean recall and prevalence terms of the shift of BayCAP-Hf, BayCAP-H and BayCAP-EB-real "
    "($\\times10^2$). Class of BayCAP-Hf for the pairs that do not mix or have more than 1\\% of the draws divergent."
)
COMPARISON_LABEL = "tab:baycap-hf-comparison"
SAMPLER_LABEL = "tab:baycap-hf-sampler"
TEST_LABELS = {"h": "Hf $-$ H", "eb": "Hf $-$ EB-real"}
SCORE_LABELS = {"ae": "AE", **{score: main_tables.metrics_map[score] for score in pushforward.SCORES}}
SHIFT_LABELS = {"hf": "Hf", "h": "H", "eb": "EB"}
TERM_SCALE = 1e2


def _comparison_styler(table: pd.DataFrame):
    cols = {}
    for method in METHODS:
        for score in SCORES:
            cols[(METHOD_LABELS[method], SCORE_LABELS[score])] = table[f"{method}_{score}"].to_numpy()
    for name, label in TEST_LABELS.items():
        for score in TESTED_SCORES:
            cols[(label, f"$\\Delta${SCORE_LABELS[score]}")] = table[test_column(name, score, "diff_mean")].to_numpy()
            cols[(label, f"$p_{{{SCORE_LABELS[score].strip('$')}}}$")] = table[
                test_column(name, score, "p_holm")
            ].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style.format(precision=3, na_rep="--")


def _sampler_styler(table: pd.DataFrame):
    cols = {}
    for prefix, label in (("hf", "Hf"), ("h", "H")):
        cols[(f"NUTS {label}", "div.")] = table[f"{prefix}_share_divergent"].to_numpy()
        cols[(f"NUTS {label}", "\\#div.")] = table[f"{prefix}_mean_divergences"].to_numpy()
        cols[(f"NUTS {label}", "draws")] = table[f"{prefix}_share_divergent_draws"].to_numpy()
        cols[(f"NUTS {label}", "$\\hat R$")] = table[f"{prefix}_max_rhat"].to_numpy()
    cols[("NUTS Hf", "fail.")] = table["hf_share_failed"].to_numpy()
    for prefix, label in (("hf", "Hf"), ("h", "H")):
        cols[(f"$\\nu$ {label}", "$\\ln\\nu$")] = table[f"{prefix}_ln_nu_median"].to_numpy()
        cols[(f"$\\nu$ {label}", "$P(\\nu>n)$")] = table[f"{prefix}_p_nu_above_min_n"].to_numpy()
    cols[("$\\kappa$", "$\\ln\\kappa$")] = table["hf_ln_kappa_median"].to_numpy()
    cols[("$\\kappa$", "90\\%")] = [
        baycap_h._fmt_interval(lo, hi) for lo, hi in zip(table["hf_ln_kappa_q05"], table["hf_ln_kappa_q95"])
    ]
    cols[("$\\kappa$", "$P(\\kappa<1)$")] = table["hf_p_kappa_below_1"].to_numpy()
    cols[("$V$", "empty")] = table["empty_off_diagonal_share"].to_numpy()
    cols[("$V$", "bound.")] = (table["n_perfect_classes"] + table["n_never_right_classes"]).to_numpy()
    for prefix, label in SHIFT_LABELS.items():
        cols[(f"terms {label}", "rec.")] = table[f"{prefix}_recall_term"].to_numpy() * TERM_SCALE
        cols[(f"terms {label}", "prev.")] = table[f"{prefix}_prevalence_term"].to_numpy() * TERM_SCALE
    cols[("", "class")] = table["failure_class"].fillna("").to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styler = out.style.format(precision=2, na_rep="--")
    formats = {("NUTS Hf", "$\\hat R$"): "{:.3f}", ("NUTS H", "$\\hat R$"): "{:.3f}"}
    for label in ("Hf", "H"):
        formats[(f"NUTS {label}", "\\#div.")] = "{:.1f}"
        formats[(f"NUTS {label}", "draws")] = "{:.3f}"
    for key, fmt in formats.items():
        styler = styler.format(fmt, subset=[key], na_rep="--")
    return styler


TABLES = {
    "comparison": (_comparison_styler, COMPARISON_CAPTION, COMPARISON_LABEL),
    "sampler": (_sampler_styler, SAMPLER_CAPTION, SAMPLER_LABEL),
}


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}.csv")


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def gen_tables(pargs):
    table = compute_table()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    for name, (styler_fn, caption, label) in TABLES.items():
        styler = styler_fn(table)
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, caption, label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(eb_prior._latex_document(f"tabular/{_stem(name)}_long.tex"))
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in analysis_summary(table).items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for name in TABLES:
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
    parser = ArgumentParser(description="BayCAP-Hf, the lighter tail and learned shares of BayCAP-H, on main (E2.8).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSV, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
