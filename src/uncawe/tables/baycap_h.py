"""BayCAP-H, the hierarchical prior on the recalls of BayCAP, on the pairs of ``main`` (work-plan card WP-10, E2.7 of
``docs/ext2_baycap.md``).

``BayesCAP-H-real`` is ``methods/baycap.py:BayCAP`` with ``c_prior="hierarchical"`` (§T.14, §T.15): $m\\sim U(0,1)$,
$u=1/\\sqrt{1+\\nu}\\sim U(0,1)$, $C_{jj}\\mid m,\\nu\\sim\\operatorname{Beta}(\\nu m,\\nu(1-m))$, the mistakes of each true
class shared by $\\operatorname{Dir}(\\mathbf 1_{K-1})$, sampled by NUTS with $C$ and $\\pi^Q$. It is compared bag by bag
with the stored ``BayesCAP-real`` ($\\operatorname{Dir}(\\mathbf 1)$ prior) and ``BayesCAP-EB-real`` (empirical-Bayes
prior). All three target the realised table of the target sample and are scored against it (``true_cts``). Vanilla
accuracy, $|U|=1000$, the pairs where the three have stored results (so it also works on the pilot).

Per pair:

- the comparison: AE, coverage, amplitude and Winkler score of the three methods (means over the bags), and the paired
  bootstrap test of the mean per-bag Winkler difference, BayCAP-H minus each of the two (``tables/pushforward.py``),
  Holm-adjusted over the pairs; the scores of BayCAP-H and the tests leave out the bags where NUTS produced no draws
  (``nuts_failed``), the scores of the stored methods use all bags;
- the sampler: the share of bags with divergent transitions, the largest split-$\\hat R$ and the smallest effective
  sample size (``tables/eb_prior.py:diagnostics_table``), the share of failed bags;
- the posterior of the run: means over the bags of the posterior mean of $m$, of the median and the $5\\%$ and $95\\%$
  quantiles of $\\ln\\nu$, and of $P(\\nu>\\min_jn_j)$;
- D3 (E2.6, ``tables/hierarchical.py``) for the prior of BayCAP-H on the grid (``PRIOR``): $P(\\min_j\\min(a_j,b_j)<1\\mid
  V)$, the classes perfect and never right on $V$, $P(\\nu>\\min_jn_j\\mid V)$; pairs whose share of divergent bags
  exceeds ``DIVERGENT_SHARE`` are classed as boundary (a class perfect or never right on $V$), funnel (mean posterior
  $P(\\nu>\\min_jn_j)$ of the run above ``FUNNEL_HIGH``) or other;
- D4: the coverage, amplitude and Winkler score predicted from the stored draws of ``BayesCAP-real`` for the prior of
  BayCAP-H on the grid (``tables/hierarchical.py:predicted_scores``), next to the observed ones;
- the split of the spread (§T.15): per bag, the residual variance $\\rho=s^2-V_C(\\bar\\pi)$ of the accuracy draws of
  ``BayesCAP-real`` (with $V_C$ of $\\operatorname{Dir}(\\mathbf 1)$) and of BayCAP-H (with $V_C$ of its prior on the
  grid), $\\bar\\pi$ the prevalences of the mean table of the bag (``estim_cts``, as in D4), and the prevalence part
  $\\bar r^\\top\\Sigma_\\pi\\bar r$ of BayCAP-H from its stored posterior means of the recalls and of $\\pi^Q$ and its
  covariance of $\\pi^Q$; means over the bags.

D3 and D4 use only $V$ and the stored draws; the scores use the true tables, so they are analyses. Outputs, under
``<OUT_DIR>/uncawe/tables/baycap_h/``: a per-pair CSV with every column, which ``plots/baycap_h.py`` reads, and two
tables (comparison; sampler and checks), each as a LaTeX tabular fragment, a longtable and a standalone landscape
document. Validation counts and tables follow the ``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.main import EXPERIMENT
from uncawe.tables import eb_prior, estimand_gap, pushforward
from uncawe.tables import hierarchical as h
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import METHOD_LABELS, load_validation_counts
from uncawe.tables.estimand_gap import booktabs_clines, load_results
from uncawe.util import decorate_dataset

TABLE_NAME = "baycap_h"
KEYS = eb_prior.KEYS
DOMAIN = estimand_gap.DOMAIN
METHOD = "BayesCAP-H-real"
# the stored methods BayCAP-H is compared with: Dir(1) and empirical-Bayes prior, realised pushforward
REFERENCES = {"uniform": "BayesCAP-real", "eb": "BayesCAP-EB-real"}
METHODS = [*REFERENCES.values(), METHOD]
SCORES = ("ae", *pushforward.SCORES)
PRIOR = "unifu"  # the prior of BayesCAP-H-real on the grid of E2.6 (tables/hierarchical.py:NU_PRIORS)
RUN_POSTERIOR = ("h_m_mean", "h_ln_nu_median", "h_ln_nu_q05", "h_ln_nu_q95", "h_p_nu_above_min_n")
DIVERGENT_SHARE = 0.10
FUNNEL_HIGH = 0.5
FAILURE_CLASSES = ("boundary", "funnel", "other")
TEST_COLUMNS = (*eb_prior.TEST_COLUMNS, "winkler_p_holm")


def failed_bags(df: pd.DataFrame) -> np.ndarray:
    """Flags of the bags of one pair (stored order) whose accuracy draws are not all finite: NUTS produced none."""
    return np.array([not np.all(np.isfinite(np.asarray(r, dtype=float))) for r in df["estim_acc_ranges"]])


def residual_variances(df: pd.DataFrame, post: h.RecallPosterior) -> np.ndarray:
    """Per bag, $\\rho=s^2-V_C(\\bar\\pi)$: the variance of the accuracy draws minus the part due to the recalls under
    ``post``, at the prevalences $\\bar\\pi$ of the mean table of the bag (§T.15)."""
    props = np.stack([h._bag_props(ct) for ct in df["estim_cts"]])
    _, v_c = h.accuracy_moments(post, props)
    s2 = np.array([np.var(np.asarray(r, dtype=float)) for r in df["estim_acc_ranges"]])
    return s2 - v_c


def prevalence_parts(df: pd.DataFrame) -> np.ndarray:
    """Per bag, $\\bar r^\\top\\Sigma_\\pi\\bar r$ from the stored posterior summaries of BayCAP-H (§T.15)."""
    parts = []
    for recalls, cov in zip(df["h_recall_mean"], df["h_pi_cov"]):
        r = np.asarray(recalls, dtype=float)
        parts.append(float(r @ np.asarray(cov, dtype=float).reshape(len(r), len(r)) @ r))
    return np.array(parts)


def failure_class(share_divergent: float, n_boundary: int, p_funnel: float) -> str:
    """Class of a pair with failures (E2.7): ``boundary``, ``funnel`` or ``other``; empty below ``DIVERGENT_SHARE``."""
    if not share_divergent > DIVERGENT_SHARE:
        return ""
    if n_boundary > 0:
        return "boundary"
    return "funnel" if p_funnel > FUNNEL_HIGH else "other"


def pair_row(results: dict[str, pd.DataFrame], val_counts: np.ndarray) -> dict:
    """The per-pair columns that do not need the tests: scores, sampler, run posterior, D3, D4 and the spread.

    ``results`` maps each method of ``METHODS`` to its stored results on the pair, in the stored order of the bags.
    """
    f, n = h.diagonal_counts(val_counts)
    hdf = results[METHOD]
    failed = failed_bags(hdf)
    ok = ~failed
    row = {
        "n_classes": int(val_counts.shape[0]),
        "min_val_per_class": int(n.min()),
        "median_val_per_class": float(np.median(n)),
        "n_bags": len(hdf),
        "share_failed": float(failed.mean()),
    }
    for method, df in results.items():
        df = df.loc[ok].reset_index(drop=True) if method == METHOD else df
        row[f"{method}_ae"] = float(df["acc_err"].mean()) if len(df) else np.nan
        scores = h.stored_scores(df, method) if len(df) else dict.fromkeys(pushforward.SCORES, np.nan)
        row.update({f"{method}_{score}": value for score, value in scores.items()})

    diagnostics = eb_prior.diagnostics_table(hdf).iloc[0]
    row.update({name: float(diagnostics[name]) for name in eb_prior.DIAGNOSTICS})
    for name in RUN_POSTERIOR:
        row[name] = float(pd.to_numeric(hdf[name], errors="coerce")[ok].mean()) if name in hdf and ok.any() else np.nan

    weights = h.grid_weights(h.class_log_marginals(f, n).sum(axis=0), PRIOR)
    target = h.mixture_posterior(f, n, weights)
    uniform = h.uniform_posterior(val_counts)
    row.update(
        {
            "post_alpha_risk": h.posterior_alpha_risk(f, n, weights),
            "funnel_risk": h.funnel_risk(n, weights),
            "n_perfect_classes": int((f == n).sum()),
            "n_never_right_classes": int((f == 0).sum()),
        }
    )
    reference = REFERENCES["uniform"]
    predicted = h.predicted_scores(results[reference], reference, uniform, target)
    row.update({f"d4_pred_{score}": value for score, value in predicted.items()})

    row["uniform_residual_var"] = float(residual_variances(results[reference], uniform).mean())
    if ok.any():
        hok = hdf.loc[ok].reset_index(drop=True)
        row["h_residual_var"] = float(residual_variances(hok, target).mean())
        row["h_prevalence_part"] = float(prevalence_parts(hok).mean()) if "h_pi_cov" in hok else np.nan
    else:
        row["h_residual_var"] = row["h_prevalence_part"] = np.nan
    row["failure_class"] = failure_class(
        row["share_divergent"], row["n_perfect_classes"] + row["n_never_right_classes"], row["h_p_nu_above_min_n"]
    )
    return row


def winkler_tests(results: dict[str, pd.DataFrame], failed: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """The paired bootstrap tests of BayCAP-H minus each reference, on the bags where BayCAP-H did not fail, Holm over
    the pairs; columns ``<reference key>_<test column>``."""
    scores = {}
    for method, df in results.items():
        bags = pushforward.bag_scores(df, method)
        keep = [not failed[(d, c)][b] for d, c, b in zip(bags["dataset"], bags["classifier"], bags["bag"])]
        scores[method] = bags.loc[keep].reset_index(drop=True)
    table = None
    for key, reference in REFERENCES.items():
        # winkler_tests subtracts its first argument from its second
        tests = pushforward.winkler_tests(scores[reference], scores[METHOD])
        tests = tests.rename(columns={col: f"{key}_{col}" for col in TEST_COLUMNS})
        table = tests if table is None else table.merge(tests, on=KEYS, how="outer")
    return table


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (each method restricted to the same pairs), ordered by $K$ as E2.6."""
    rows, failed = [], {}
    for (dataset, classifier), _ in results[METHOD].groupby(KEYS, sort=False):
        pair_results = {
            method: df.loc[(df["dataset"] == dataset) & (df["classifier"] == classifier)].reset_index(drop=True)
            for method, df in results.items()
        }
        failed[(dataset, classifier)] = failed_bags(pair_results[METHOD])
        rows.append(
            {"dataset": dataset, "classifier": classifier, **pair_row(pair_results, val_counts[(dataset, classifier)])}
        )
    table = pd.DataFrame(rows).merge(winkler_tests(results, failed), on=KEYS, how="left")
    return h.ordered(table)


def compute_table() -> pd.DataFrame:
    """Load the stored results of the three methods and the validation counts, and return the ordered table."""
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


def _k_groups(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("all", table)]
    for name in h.K_GROUPS:
        part = table.loc[table["n_classes"].map(h._k_group) == name]
        if not part.empty:
            groups.append((name, part))
    return groups


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E2.7, over all pairs and per group of $K$.

    - ``scores``: mean coverage and Winkler score of the three methods, and, for each reference, the pairs where
      BayCAP-H has a significantly (Holm) lower and higher Winkler score;
    - ``sampler``: pairs with divergences, with a share of divergent bags above ``DIVERGENT_SHARE`` and in each failure
      class, pairs with failed bags, the largest $\\hat R$ and the smallest ESS; the Spearman correlation of the share of
      divergent bags with each indicator of D3 and with the posterior $P(\\nu>\\min_jn_j)$ of the run;
    - ``checks``: for D4, the mean absolute difference of the predicted and observed coverage and amplitude, the share of
      pairs within $0.05$ in coverage and the correlations; for the spread, the median residual variances and prevalence
      part, and the correlation across pairs of the change of the residual variance with the miss of D4 in amplitude.
    """
    out = {"scores": [], "sampler": [], "checks": []}
    for k_group, sub in _k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        many = len(sub) > 2
        scores = dict(key)
        for method in METHODS:
            scores[f"{method}_cov"] = float(sub[f"{method}_coverage"].mean())
            scores[f"{method}_W"] = float(sub[f"{method}_winkler"].mean())
        for name, reference in REFERENCES.items():
            significant = sub[f"{name}_winkler_p_holm"] < pushforward.WINKLER_TEST_LEVEL
            lower = sub[f"{METHOD}_winkler"] < sub[f"{reference}_winkler"]
            scores[f"lower_than_{name}"] = int((lower & significant).sum())
            scores[f"higher_than_{name}"] = int((~lower & significant).sum())
        out["scores"].append(scores)

        divergent = sub["share_divergent"]
        sampler = {
            **key,
            "with_divergences": int((divergent > 0).sum()),
            "above_share": int((divergent > DIVERGENT_SHARE).sum()),
            **{cls: int((sub["failure_class"] == cls).sum()) for cls in FAILURE_CLASSES},
            "with_failed_bags": int((sub["share_failed"] > 0).sum()),
            "max_rhat": float(sub["max_rhat"].max()),
            "min_ess": float(sub["min_ess"].min()),
        }
        for column in ("post_alpha_risk", "funnel_risk", "h_p_nu_above_min_n"):
            sampler[f"spearman_{column}"] = float(sub[column].corr(divergent, method="spearman")) if many else np.nan
        out["sampler"].append(sampler)

        checks = dict(key)
        for score in ("coverage", "amplitude"):
            pred, obs = sub[f"d4_pred_{score}"], sub[f"{METHOD}_{score}"]
            checks[f"mean_abs_{score}_diff"] = float((pred - obs).abs().mean())
            checks[f"corr_{score}"] = float(pred.corr(obs)) if many else np.nan
        checks["coverage_within_0.05"] = float(
            ((sub["d4_pred_coverage"] - sub[f"{METHOD}_coverage"]).abs() <= 0.05).mean()
        )
        for column in ("uniform_residual_var", "h_residual_var", "h_prevalence_part"):
            checks[f"median_{column}"] = float(sub[column].median())
        change = sub["h_residual_var"] - sub["uniform_residual_var"]
        miss = sub[f"{METHOD}_amplitude"] - sub["d4_pred_amplitude"]
        checks["corr_residual_change_amplitude_miss"] = float(change.corr(miss)) if many else np.nan
        out["checks"].append(checks)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


COMPARISON_CAPTION = (
    "BayCAP with the hierarchical prior on the recalls (BayCAP-H-real), against the stored BayCAP-real "
    "($\\operatorname{Dir}(\\mathbf 1)$ prior) and BayCAP-EB-real (empirical-Bayes prior), realised pushforward, on "
    "\\texttt{main} (E2.7, vanilla accuracy, $|U|=1000$). AE, coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler "
    "score $\\mathcal{W}$ against the realised table, means over the bags (for BayCAP-H, over the bags where NUTS "
    "produced draws). Tests: 95\\% paired-bootstrap interval of the mean per-bag Winkler difference, BayCAP-H minus "
    "each stored method, and its two-sided p-value, Holm-adjusted over the pairs."
)
SAMPLER_CAPTION = (
    "Sampler and checks of BayCAP-H-real on \\texttt{main} (E2.7, vanilla accuracy, $|U|=1000$). NUTS: share of bags with "
    "divergent transitions, largest split-$\\hat R$, smallest effective sample size, share of bags without draws. Run: "
    "means over the bags of the posterior mean of $m$, of the median and the 90\\% interval of $\\ln\\nu$, and of "
    "$P(\\nu>\\min_jn_j)$. D3, from $V$ on the grid of E2.6 with the prior of BayCAP-H: $P(\\min_j\\min(a_j,b_j)<1\\mid V)$, "
    "classes perfect and never right on $V$, $P(\\nu>\\min_jn_j\\mid V)$ (funnel). D4: coverage and amplitude predicted "
    "from the draws of BayCAP-real and observed. Spread: mean residual variances $\\rho=s^2-V_C(\\bar\\pi)$ of BayCAP-real "
    "and BayCAP-H and the prevalence part $\\bar r^\\top\\Sigma_\\pi\\bar r$ of BayCAP-H ($\\times10^4$). Class of the "
    "pairs with more than 10\\% divergent bags."
)
COMPARISON_LABEL = "tab:baycap-h-comparison"
SAMPLER_LABEL = "tab:baycap-h-sampler"
TEST_LABELS = {"uniform": "H $-$ real", "eb": "H $-$ EB-real"}
SCORE_LABELS = {"ae": "AE", **{score: main_tables.metrics_map[score] for score in pushforward.SCORES}}
SPREAD_SCALE = 1e4


def _index(table: pd.DataFrame) -> pd.MultiIndex:
    names = table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d)))
    return pd.MultiIndex.from_arrays(
        [names, table["n_classes"].astype(int), table["classifier"].astype(str)], names=["dataset", "$K$", "classifier"]
    )


def _comparison_styler(table: pd.DataFrame):
    cols = {}
    for method in METHODS:
        for score in SCORES:
            cols[(METHOD_LABELS[method], SCORE_LABELS[score])] = table[f"{method}_{score}"].to_numpy()
    for name, label in TEST_LABELS.items():
        cols[(label, "$\\Delta\\mathcal{W}$ 95\\%")] = [
            eb_prior._fmt_ci(low, high)
            for low, high in zip(table[f"{name}_winkler_ci_low"], table[f"{name}_winkler_ci_high"])
        ]
        cols[(label, "$p_{\\mathcal{W}}$")] = table[f"{name}_winkler_p_holm"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table)
    return out.style.format(precision=3, na_rep="--")


def _fmt_interval(low: float, high: float) -> str:
    return "--" if np.isnan(low) else f"[{low:.1f}, {high:.1f}]"


def _sampler_styler(table: pd.DataFrame):
    cols = {
        ("NUTS", "div."): table["share_divergent"].to_numpy(),
        ("NUTS", "$\\hat R$"): table["max_rhat"].to_numpy(),
        ("NUTS", "ESS"): table["min_ess"].to_numpy(),
        ("NUTS", "fail."): table["share_failed"].to_numpy(),
        ("run", "$\\bar m$"): table["h_m_mean"].to_numpy(),
        ("run", "$\\ln\\nu$"): table["h_ln_nu_median"].to_numpy(),
        ("run", "90\\%"): [_fmt_interval(lo, hi) for lo, hi in zip(table["h_ln_nu_q05"], table["h_ln_nu_q95"])],
        ("run", "$P(\\nu>n)$"): table["h_p_nu_above_min_n"].to_numpy(),
        ("D3", "$P(a,b<1)$"): table["post_alpha_risk"].to_numpy(),
        ("D3", "perf."): table["n_perfect_classes"].to_numpy(),
        ("D3", "never"): table["n_never_right_classes"].to_numpy(),
        ("D3", "funnel"): table["funnel_risk"].to_numpy(),
    }
    for score, label in (("coverage", "$\\mathcal{C}$"), ("amplitude", "$\\mathcal{A}$")):
        cols[("D4", f"{label} pred.")] = table[f"d4_pred_{score}"].to_numpy()
        cols[("D4", f"{label} obs.")] = table[f"{METHOD}_{score}"].to_numpy()
    cols[("spread", "$\\rho_u$")] = table["uniform_residual_var"].to_numpy() * SPREAD_SCALE
    cols[("spread", "$\\rho_H$")] = table["h_residual_var"].to_numpy() * SPREAD_SCALE
    cols[("spread", "prev.")] = table["h_prevalence_part"].to_numpy() * SPREAD_SCALE
    cols[("", "class")] = table["failure_class"].fillna("").to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table)
    styler = out.style.format(precision=3, na_rep="--")
    formats = {("NUTS", "div."): "{:.2f}", ("NUTS", "ESS"): "{:.0f}", ("NUTS", "fail."): "{:.2f}"}
    for spread in ("$\\rho_u$", "$\\rho_H$", "prev."):
        formats[("spread", spread)] = "{:.2f}"
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
    parser = ArgumentParser(description="BayCAP-H, the hierarchical prior on the recalls of BayCAP, on main (E2.7).")
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
