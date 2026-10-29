"""Zero-cost screens for a hierarchical prior on the recalls of BayCAP (work-plan card WP-9, E2.6 of
``docs/ext2_baycap.md``).

The hierarchical prior gives each recall $r_j=C_{jj}$ of a classifier the prior $\\operatorname{Beta}(\\nu m,\\nu(1-m))$
and puts a prior on $(m,\\nu)$ instead of plugging in the fit $(\\hat m,\\hat\\nu)$ of the empirical-Bayes prior of
``BayesCAP-EB``. Its posterior is computed from the validation counts alone (§T.14), on the grid of
``scripts/toys/ext2_hierarchical.py``: $m$ uniform, and one of three priors on $\\nu\\in[0.1,10^4]$ (``NU_PRIORS``,
written as log-densities in $\\ln\\nu$). No NUTS is run. Four screens, per (dataset, classifier) pair of ``main``,
vanilla accuracy, $|U|=1000$:

- D4, prediction from $V$ alone: the stored accuracy draws of ``BayesCAP`` (population table) and ``BayesCAP-real``
  (realised table) are moved by the change of the posterior-mean recalls weighted by the posterior-mean prevalence of
  the bag (from ``estim_cts``, no labels) and rescaled so that their variance swaps the part due to $C$ under
  $\\operatorname{Dir}(\\mathbf 1)$ for that under another prior (``predict_draws``). The prediction for the
  empirical-Bayes prior is validated against the stored ``BayesCAP-EB`` and ``BayesCAP-EB-real``; the same prediction
  is then made for each hierarchical prior. Its misses are diagnosed by splitting, per bag, the change of the centre
  between the stored uniform and empirical-Bayes tables into a recall part and a prevalence part (``centre_split``),
  and by repeating the prediction with the stored prevalence part added (``ebprev``) or with the stored centre
  (``ebcentre``). Predicted draws keep the estimand of the method they start from, and every score is computed against
  the realised table of each bag (``true_cts``), so D4 is an analysis, not a method;
- D1, prior sensitivity: the ratio of the 90% widths of the posterior and of the prior of $\\ln\\nu$, the posterior mean
  of $m$, the posterior mass with $\\nu\\ge10^3$, the change of the posterior-mean accuracy from
  $\\operatorname{Dir}(\\mathbf 1)$ and the standard deviation of accuracy due to $C$, at the uniform prevalence and
  at the vertices; the range of the D4 predictions over the three priors;
- D2, exchangeability: per class, the leave-one-class-out predictive p-value of $f_{jj}$ and the vertex shift
  $z_j=(\\mathbb E[r_j\\mid V]-f_{jj}/n_j)/\\operatorname{sd}(r_j\\mid V)$; per pair the smallest p-value, the number of
  classes below ``P_SMALL`` and the largest $|z_j|$;
- D3, sampler risk: $P(\\min(\\nu m,\\nu(1-m))<1\\mid V)$ and the 90% width of the posterior of $\\ln\\nu$, next to the
  stored share of bags with divergent transitions of ``BayesCAP-EB``.

The pairs are those of ``main`` (all $75$) with all four BayCAP methods stored. Outputs, under
``<OUT_DIR>/uncawe/tables/hierarchical/``: a per-pair CSV, which ``plots/hierarchical.py`` reads, and two tables
(validation; screens), each as a LaTeX tabular fragment, a longtable and a standalone landscape document. Validation
counts and tables follow the ``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln

from uncawe import env
from uncawe.config import acc_from_ct
from uncawe.main import EXPERIMENT
from uncawe.methods.baycap import fit_beta_binomial
from uncawe.tables import eb_prior, estimand_gap
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import class_props_and_recalls, load_validation_counts
from uncawe.tables.estimand_gap import ACC_NAME, CLASSIFIER_ORDER, booktabs_clines
from uncawe.util import decorate_dataset

TABLE_NAME = "hierarchical"
KEYS = eb_prior.KEYS
DOMAIN = estimand_gap.DOMAIN
PUSHFORWARDS = eb_prior.PUSHFORWARDS  # name -> (uniform method, empirical-Bayes method)
SCORES = ("coverage", "amplitude", "winkler")

GRID_M = np.linspace(0.0025, 0.9975, 200)
GRID_LN_NU = np.linspace(np.log(0.1), np.log(1e4), 200)
# prior on nu as a log-density in ln(nu): p(nu) d(nu) = p(nu) nu d(ln nu)
NU_PRIORS = {"nu32": -0.5, "loguniform": 0.0, "flat": 1.0}
NU_PRIOR_LABELS = {"nu32": "$\\nu^{-3/2}$", "loguniform": "log-unif.", "flat": "flat"}
MAIN_PRIOR = "loguniform"
TAIL = 0.05  # 90% intervals of ln(nu)
TOP_NU = 1e3
P_SMALL = 0.01
MASS_KEPT = 1 - 1e-6  # grid points kept, by decreasing weight
PMF_CHUNK = 128
K_GROUPS = {"K=2": (2, 2), "3<=K<=7": (3, 7), "K>=8": (8, np.inf)}


@dataclass
class RecallPosterior:
    """The posterior of the recalls as a mixture over points $(m_p,\\nu_p)$ with weights $w_p$ (§T.14).

    ``mean[j, p]`` and ``var[j, p]`` are the mean and variance of $r_j$ given $V$ and the point $p$; a prior with a
    fixed
    concentration (uniform, empirical Bayes) has a single point of weight $1$.
    """

    mean: np.ndarray
    var: np.ndarray
    weights: np.ndarray


def _beta_moments(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = a / (a + b)
    return mean, mean * (1 - mean) / (a + b + 1)


def _log_binom(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def diagonal_counts(val_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return $f_{jj}$ and $n_j$ of a ``cap-ml`` validation table (true classes on rows)."""
    val_counts = np.asarray(val_counts, dtype=float)
    return np.diag(val_counts), val_counts.sum(axis=1)


def uniform_posterior(val_counts: np.ndarray) -> RecallPosterior:
    """$\\operatorname{Dir}(\\mathbf 1)$ prior: by aggregation $r_j\\mid
    V\\sim\\operatorname{Beta}(f_{jj}+1,n_j-f_{jj}+K-1)$."""
    f, n = diagonal_counts(val_counts)
    mean, var = _beta_moments(f + 1, n - f + len(f) - 1)
    return RecallPosterior(mean[:, None], var[:, None], np.ones(1))


def eb_posterior(val_counts: np.ndarray) -> RecallPosterior:
    """Empirical-Bayes prior of ``BayesCAP-EB``: $r_j\\mid V\\sim\\operatorname{Beta}(f_{jj}+\\hat\\nu\\hat
    m,n_j-f_{jj}+
    \\hat\\nu(1-\\hat m))$, with the fit of ``BayCAP.fit``."""
    f, n = diagonal_counts(val_counts)
    fit = fit_beta_binomial(f, n)
    mean, var = _beta_moments(f + fit.nu * fit.m, n - f + fit.nu * (1 - fit.m))
    return RecallPosterior(mean[:, None], var[:, None], np.ones(1))


def class_log_marginals(f: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Return, for each class, $\\ln\\operatorname{BetaBin}(f_{jj}\\mid n_j,\\nu m,\\nu(1-m))$ on the grid, shape (K,
    M, N)."""
    m = GRID_M[None, :, None]
    nu = np.exp(GRID_LN_NU)[None, None, :]
    a, b = nu * m, nu * (1 - m)
    f, n = np.asarray(f, dtype=float)[:, None, None], np.asarray(n, dtype=float)[:, None, None]
    return _log_binom(n, f) + betaln(a + f, b + n - f) - betaln(a, b)


def grid_weights(log_lik: np.ndarray, prior: str) -> np.ndarray:
    """Normalised posterior weights on the grid, from the log-likelihood summed over classes, shape (M, N)."""
    log_w = log_lik + NU_PRIORS[prior] * GRID_LN_NU[None, :]
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


def prior_weights(prior: str) -> np.ndarray:
    """The prior on the grid (uniform in $m$), shape (M, N)."""
    return grid_weights(np.zeros((GRID_M.size, GRID_LN_NU.size)), prior)


def _kept_points(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return $m$, $\\nu$ and the renormalised weights of the grid points that carry ``MASS_KEPT`` of the mass."""
    flat = weights.ravel()
    order = np.argsort(flat)[::-1]
    keep = order[: int(np.searchsorted(np.cumsum(flat[order]), MASS_KEPT)) + 1]
    i, j = np.unravel_index(keep, weights.shape)
    w = flat[keep]
    return GRID_M[i], np.exp(GRID_LN_NU[j]), w / w.sum()


def mixture_posterior(f: np.ndarray, n: np.ndarray, weights: np.ndarray) -> RecallPosterior:
    """The recalls given $V$ under the grid posterior ``weights``: at each point the conjugate
    $\\operatorname{Beta}(\\nu m+f_{jj},\\nu(1-m)+n_j-f_{jj})$."""
    m, nu, w = _kept_points(weights)
    f, n = np.asarray(f, dtype=float)[:, None], np.asarray(n, dtype=float)[:, None]
    mean, var = _beta_moments(f + nu * m, n - f + nu * (1 - m))
    return RecallPosterior(mean, var, w)


def recall_moments(post: RecallPosterior) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of each recall, by the laws of total expectation and variance."""
    mean = post.mean @ post.weights
    var = post.var @ post.weights + ((post.mean - mean[:, None]) ** 2) @ post.weights
    return mean, var


def accuracy_moments(post: RecallPosterior, props: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance $V_C(\\pi)$ of $A=\\sum_j\\pi_jr_j$ for each row of ``props`` (shape (B, K)), by the law of
    total
    variance over the mixture points (the recalls are independent given a point)."""
    props = np.atleast_2d(np.asarray(props, dtype=float))
    cond_mean = props @ post.mean
    cond_var = (props**2) @ post.var
    mean = cond_mean @ post.weights
    var = cond_var @ post.weights + ((cond_mean - mean[:, None]) ** 2) @ post.weights
    return mean, var


def predict_draws(samples: np.ndarray, shift: float, var_from: float, var_to: float) -> np.ndarray | None:
    """Move the draws by ``shift`` and rescale them so that their variance loses ``var_from`` and gains ``var_to``;
    None if
    the variance is not positive or the predicted variance would not be (§T.14, D4)."""
    samples = np.asarray(samples, dtype=float)
    centre, s2 = float(np.mean(samples)), float(np.var(samples))
    new_var = s2 - var_from + var_to
    if not (s2 > 0 and new_var > 0):
        return None
    return centre + shift + (samples - centre) * np.sqrt(new_var / s2)


def ln_nu_quantiles(weights: np.ndarray, probs=(TAIL, 1 - TAIL)) -> np.ndarray:
    """Quantiles of the marginal of $\\ln\\nu$ on the grid, by linear interpolation of its cumulative distribution."""
    marginal = weights.sum(axis=0)
    cdf = np.cumsum(marginal) - marginal / 2
    return np.interp(probs, cdf, GRID_LN_NU)


def ln_nu_width(weights: np.ndarray) -> float:
    lo, hi = ln_nu_quantiles(weights)
    return float(hi - lo)


def alpha_risk(weights: np.ndarray) -> float:
    """$P(\\min(\\nu m,\\nu(1-m))<1\\mid V)$ on the grid."""
    m = GRID_M[:, None]
    nu = np.exp(GRID_LN_NU)[None, :]
    return float(weights[np.minimum(nu * m, nu * (1 - m)) < 1].sum())


def loo_pvalue(f_j: float, n_j: float, weights: np.ndarray) -> float:
    """Two-sided predictive p-value of the count $f_{jj}$ under the beta-binomial averaged over ``weights``, the grid
    posterior fitted without class $j$."""
    m, nu, w = _kept_points(weights)
    a, b = nu * m, nu * (1 - m)
    n_j, f_j = round(n_j), round(f_j)
    below = above = 0.0
    for start in range(0, n_j + 1, PMF_CHUNK):
        k = np.arange(start, min(start + PMF_CHUNK, n_j + 1), dtype=float)[:, None]
        log_pmf = _log_binom(n_j, k) + betaln(a + k, b + n_j - k) - betaln(a, b)
        pmf = np.exp(log_pmf) @ w
        below += pmf[k[:, 0] <= f_j].sum()
        above += pmf[k[:, 0] >= f_j].sum()
    return float(min(1.0, 2 * min(below, above)))


def hierarchical_screens(val_counts: np.ndarray) -> tuple[dict, dict[str, RecallPosterior]]:
    """Return the per-pair screens D1–D3 and the mixture posterior of the recalls for each prior on $\\nu$."""
    f, n = diagonal_counts(val_counts)
    k = len(f)
    class_ll = class_log_marginals(f, n)
    total_ll = class_ll.sum(axis=0)
    uniform = uniform_posterior(val_counts)
    unif_mean, _ = recall_moments(uniform)
    vertices = np.eye(k)
    centre = np.full((1, k), 1 / k)
    nu_grid = np.exp(GRID_LN_NU)
    out, posteriors = {}, {}
    for prior in NU_PRIORS:
        weights = grid_weights(total_ll, prior)
        post = mixture_posterior(f, n, weights)
        posteriors[prior] = post
        mean, var = recall_moments(post)
        acc_mean_c, acc_var_c = accuracy_moments(post, centre)
        unif_acc_c, _ = accuracy_moments(uniform, centre)
        _, acc_var_v = accuracy_moments(post, vertices)
        z = (mean - f / n) / np.sqrt(var)
        pvalues = [loo_pvalue(f[j], n[j], grid_weights(total_ll - class_ll[j], prior)) for j in range(k)]
        out.update(
            {
                f"{prior}_width_ratio": ln_nu_width(weights) / ln_nu_width(prior_weights(prior)),
                f"{prior}_ln_nu_width": ln_nu_width(weights),
                f"{prior}_m_mean": float((weights.sum(axis=1) * GRID_M).sum()),
                f"{prior}_top_nu_mass": float(weights[:, nu_grid >= TOP_NU].sum()),
                f"{prior}_acc_change_centre": float(acc_mean_c[0] - unif_acc_c[0]),
                f"{prior}_acc_change_vertex": float(np.max(np.abs(mean - unif_mean))),
                f"{prior}_acc_sd_centre": float(np.sqrt(acc_var_c[0])),
                f"{prior}_acc_sd_vertex": float(np.sqrt(acc_var_v.max())),
                f"{prior}_alpha_risk": alpha_risk(weights),
                f"{prior}_min_pvalue": float(min(pvalues)),
                f"{prior}_n_small_pvalue": int(sum(p < P_SMALL for p in pvalues)),
                f"{prior}_max_abs_z": float(np.max(np.abs(z))),
            }
        )
    return out, posteriors


def _bag_props(estim_ct) -> np.ndarray:
    props, _ = class_props_and_recalls(np.asarray(estim_ct, dtype=float))
    return props / props.sum()


def stored_scores(df: pd.DataFrame, method: str) -> dict:
    """Mean coverage, amplitude and Winkler score over the bags of one pair, from the stored draws."""
    rows = [_scores(method, row.true_cts, row.estim_acc_ranges) for row in df.itertuples()]
    return {score: float(np.mean([r[score] for r in rows])) for score in SCORES}


def _scores(method: str, true_ct, samples) -> dict:
    true_acc = acc_from_ct(ACC_NAME, np.asarray(true_ct, dtype=float))
    samples = np.asarray(samples, dtype=float)
    return {
        "coverage": main_tables.get_acc_coverage(method, true_acc, samples),
        "amplitude": main_tables.get_acc_amplitude(method, samples),
        "winkler": main_tables.get_acc_winkler(method, true_acc, samples),
    }


def predicted_scores(
    df: pd.DataFrame,
    method: str,
    uniform: RecallPosterior,
    target: RecallPosterior,
    extra_shift: np.ndarray | None = None,
) -> dict:
    """Mean scores over the bags of one pair of the draws of ``method`` (uniform prior) predicted for the prior
    ``target`` (D4), and the share of bags where the prediction is undefined.

    ``extra_shift`` (one value per bag) is added to the shift of D4; the diagnosis of D4 uses it to add a part of the
    stored change of the centre (``centre_split``).
    """
    props = np.stack([_bag_props(ct) for ct in df["estim_cts"]])
    mean_u, var_u = accuracy_moments(uniform, props)
    mean_t, var_t = accuracy_moments(target, props)
    shifts = mean_t - mean_u if extra_shift is None else mean_t - mean_u + np.asarray(extra_shift, dtype=float)
    rows, undefined = [], 0
    for b, row in enumerate(df.itertuples()):
        draws = predict_draws(row.estim_acc_ranges, shifts[b], var_u[b], var_t[b])
        if draws is None:
            undefined += 1
            continue
        rows.append(_scores(method, row.true_cts, draws))
    out = {score: float(np.mean([r[score] for r in rows])) if rows else np.nan for score in SCORES}
    out["undefined"] = undefined / len(df)
    return out


def _props_and_recalls(ct) -> tuple[np.ndarray, np.ndarray]:
    ct = np.asarray(ct, dtype=float)
    return class_props_and_recalls(ct / ct.sum())


def centre_split(
    uniform_df: pd.DataFrame, eb_df: pd.DataFrame, uniform: RecallPosterior, eb: RecallPosterior
) -> dict[str, np.ndarray]:
    """Split, per bag, the change of the centre of accuracy from the stored uniform to the stored empirical-Bayes
    method into a recall part and a prevalence part (§T.14, diagnosis of D4).

    With $\\hat\\pi$, $\\hat r$ the prevalences and recalls of the posterior-mean table (``estim_cts``) of each method,
    $\\sum_j\\hat\\pi^{EB}_j\\hat r^{EB}_j-\\sum_j\\hat\\pi^{u}_j\\hat r^{u}_j=R+P$ exactly, with
    $R=\\sum_j\\hat\\pi^{u}_j(\\hat r^{EB}_j-\\hat r^{u}_j)$ and $P=\\sum_j(\\hat\\pi^{EB}_j-\\hat\\pi^{u}_j)\\hat r^{EB}_j$. Also returned:
    the shift of D4 for the same bag, and $\\lVert\\hat\\pi^{EB}-\\hat\\pi^{u}\\rVert_1$. The bags of the two methods are
    matched by their stored order, and must have the same realised table.
    """
    if len(uniform_df) != len(eb_df):
        raise ValueError(f"different numbers of bags: {len(uniform_df)} and {len(eb_df)}")
    parts = []
    for u, e in zip(uniform_df.itertuples(), eb_df.itertuples()):
        if not np.allclose(np.asarray(u.true_cts, dtype=float), np.asarray(e.true_cts, dtype=float)):
            raise ValueError("the bags of the two methods do not match")
        props_u, recalls_u = _props_and_recalls(u.estim_cts)
        props_e, recalls_e = _props_and_recalls(e.estim_cts)
        parts.append(
            (
                props_u,
                float(props_u @ (recalls_e - recalls_u)),
                float((props_e - props_u) @ recalls_e),
                float(np.abs(props_e - props_u).sum()),
            )
        )
    props = np.stack([p[0] for p in parts])
    mean_u, _ = accuracy_moments(uniform, props)
    mean_e, _ = accuracy_moments(eb, props)
    return {
        "d4_shift": mean_e - mean_u,
        "recall_part": np.array([p[1] for p in parts]),
        "prevalence_part": np.array([p[2] for p in parts]),
        "props_l1": np.array([p[3] for p in parts]),
    }


def pair_row(results: dict[str, pd.DataFrame], val_counts: np.ndarray) -> dict:
    """All the columns of one pair: stored scores of the four methods, the D4 predictions, the diagnosis of D4 and the
    screens."""
    row = {
        "n_classes": int(val_counts.shape[0]),
        "min_val_per_class": int(val_counts.sum(axis=1).min()),
        "median_val_per_class": float(np.median(val_counts.sum(axis=1))),
    }
    screens, posteriors = hierarchical_screens(val_counts)
    row.update(screens)
    uniform, eb = uniform_posterior(val_counts), eb_posterior(val_counts)
    targets = {"eb": eb, **posteriors}
    for name, (uniform_method, eb_method) in PUSHFORWARDS.items():
        for method in (uniform_method, eb_method):
            for score, value in stored_scores(results[method], method).items():
                row[f"{method}_{score}"] = value
        for target, post in targets.items():
            for score, value in predicted_scores(results[uniform_method], uniform_method, uniform, post).items():
                row[f"{name}_{target}_pred_{score}"] = value
        split = centre_split(results[uniform_method], results[eb_method], uniform, eb)
        for part, values in split.items():
            row[f"{name}_{part}"] = float(values.mean())
        row[f"{name}_abs_prevalence_part"] = float(np.abs(split["prevalence_part"]).mean())
        # the stored centre: D4 shift + (R - D4 shift) + P
        diagnoses = {
            "ebprev": split["prevalence_part"],
            "ebcentre": split["recall_part"] + split["prevalence_part"] - split["d4_shift"],
        }
        for target, extra in diagnoses.items():
            scores = predicted_scores(results[uniform_method], uniform_method, uniform, eb, extra_shift=extra)
            for score, value in scores.items():
                row[f"{name}_{target}_pred_{score}"] = value
    for score in ("coverage", "winkler"):
        values = [row[f"realised_{prior}_pred_{score}"] for prior in NU_PRIORS]
        row[f"realised_pred_{score}_range"] = float(np.nanmax(values) - np.nanmin(values))
    return row


def compute_table() -> pd.DataFrame:
    """The ordered per-pair table: the pairs of ``main`` with all four BayCAP methods stored."""
    results = {}
    for method in eb_prior.METHODS:
        df = estimand_gap.load_results(method, experiment=EXPERIMENT, domain=DOMAIN)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT}.")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {eb_prior.METHODS}.")
    val_counts = load_validation_counts(pairs, domain=DOMAIN)
    divergences = eb_prior.diagnostics_table(results["BayesCAP-EB"]).set_index(KEYS)["share_divergent"]
    rows = []
    for dataset, classifier in pairs:
        pair_results = {
            method: df.loc[(df["dataset"] == dataset) & (df["classifier"] == classifier)].reset_index(drop=True)
            for method, df in results.items()
        }
        row = {"dataset": dataset, "classifier": classifier}
        row.update(pair_row(pair_results, val_counts[(dataset, classifier)]))
        row["BayesCAP-EB_share_divergent"] = float(divergences.get((dataset, classifier), np.nan))
        rows.append(row)
    return ordered(pd.DataFrame(rows))


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order the pairs by $K$, dataset and classifier as in ``main``."""
    table = table.copy()
    order = table.drop_duplicates("dataset").sort_values(["n_classes", "dataset"])["dataset"].tolist()
    table["dataset"] = pd.Categorical(table["dataset"], categories=order, ordered=True)
    table["classifier"] = pd.Categorical(table["classifier"], categories=CLASSIFIER_ORDER, ordered=True)
    return table.sort_values(["dataset", "classifier"]).reset_index(drop=True)


def _k_group(n_classes: int) -> str:
    return next(name for name, (lo, hi) in K_GROUPS.items() if lo <= n_classes <= hi)


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E2.6, over all pairs and per group of $K$.

    - ``validation``: for each pushforward, the mean absolute difference of the predicted and stored coverage of the
      empirical-Bayes method, the share of pairs within $0.05$, the correlation of predicted and stored coverage, and
      the share of pairs where the predicted and the stored change of the Winkler score from the uniform prior have the
      same sign; for the diagnosis, the mean absolute coverage difference and the share within $0.05$ with the stored
      prevalence part added (``prev``) and with the stored centre (``centre``), the correlation across pairs of the
      shift of D4 and the stored recall part, and the mean absolute prevalence part;
    - ``prediction``: mean realised coverage and Winkler score, stored (uniform, empirical Bayes) and predicted (each
      prior on $\\nu$);
    - ``screens``: median width ratio of $\\ln\\nu$ and median range of the predicted realised coverage over the priors;
      pairs with a p-value below ``P_SMALL`` and with $|z_j|>2$ (``MAIN_PRIOR``); median sampler risk, and its Spearman
      correlation with the stored share of divergent bags of ``BayesCAP-EB``.
    """
    groups = [("all", table)]
    for name in K_GROUPS:
        part = table.loc[table["n_classes"].map(_k_group) == name]
        if not part.empty:
            groups.append((name, part))
    out = {"validation": [], "prediction": [], "screens": []}
    for k_group, sub in groups:
        key = {"K": k_group, "pairs": len(sub)}
        for name, (uniform_method, eb_method) in PUSHFORWARDS.items():
            stored = sub[f"{eb_method}_coverage"]
            pred = sub[f"{name}_eb_pred_coverage"]
            pred_dw = sub[f"{name}_eb_pred_winkler"] - sub[f"{uniform_method}_winkler"]
            stored_dw = sub[f"{eb_method}_winkler"] - sub[f"{uniform_method}_winkler"]
            row = {
                **key,
                "pushforward": name,
                "mean_abs_cov_diff": float((pred - stored).abs().mean()),
                "within_0.05": float(((pred - stored).abs() <= 0.05).mean()),
                "corr_cov": float(pred.corr(stored)) if len(sub) > 2 else np.nan,
                "same_sign_dW": float((np.sign(pred_dw) == np.sign(stored_dw)).mean()),
                "undefined": float(sub[f"{name}_eb_pred_undefined"].mean()),
            }
            for target, label in (("ebprev", "prev"), ("ebcentre", "centre")):
                diff = (sub[f"{name}_{target}_pred_coverage"] - stored).abs()
                row[f"mean_abs_cov_diff_{label}"] = float(diff.mean())
                row[f"within_0.05_{label}"] = float((diff <= 0.05).mean())
            shift, recall = sub[f"{name}_d4_shift"], sub[f"{name}_recall_part"]
            row["corr_shift_recall"] = float(shift.corr(recall)) if len(sub) > 2 else np.nan
            row["mean_abs_prevalence_part"] = float(sub[f"{name}_abs_prevalence_part"].mean())
            out["validation"].append(row)
        prediction = {
            **key,
            "uniform_cov": float(sub["BayesCAP-real_coverage"].mean()),
            "eb_cov": float(sub["BayesCAP-EB-real_coverage"].mean()),
            "uniform_W": float(sub["BayesCAP-real_winkler"].mean()),
            "eb_W": float(sub["BayesCAP-EB-real_winkler"].mean()),
        }
        for prior in NU_PRIORS:
            prediction[f"{prior}_cov"] = float(sub[f"realised_{prior}_pred_coverage"].mean())
            prediction[f"{prior}_W"] = float(sub[f"realised_{prior}_pred_winkler"].mean())
        out["prediction"].append(prediction)
        risk, divergent = sub[f"{MAIN_PRIOR}_alpha_risk"], sub["BayesCAP-EB_share_divergent"]
        out["screens"].append(
            {
                **key,
                **{f"{prior}_width_ratio": float(sub[f"{prior}_width_ratio"].median()) for prior in NU_PRIORS},
                "cov_range": float(sub["realised_pred_coverage_range"].median()),
                "small_pvalue": int((sub[f"{MAIN_PRIOR}_min_pvalue"] < P_SMALL).sum()),
                "large_z": int((sub[f"{MAIN_PRIOR}_max_abs_z"] > 2).sum()),
                "alpha_risk": float(risk.median()),
                "spearman_risk_div": float(risk.corr(divergent, method="spearman")) if len(sub) > 2 else np.nan,
            }
        )
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


VALIDATION_CAPTION = (
    "Prediction of the empirical-Bayes (EB) prior of BayCAP from the stored draws of BayCAP with the "
    "$\\operatorname{Dir}(\\mathbf 1)$ prior and the validation counts alone (E2.6, D4), vanilla accuracy, $|U|=1000$. Coverage $\\mathcal{C}$, "
    "amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ against the realised table, means over the bags: stored "
    "(BayCAP-EB, BayCAP-EB-real) and predicted, for the population and the realised pushforward. Diagnosis of the "
    "prediction: coverage with the stored change of the estimated prevalences added (+$P$) and with the stored centre "
    "of the EB method (centre)."
)
SCREENS_CAPTION = (
    "Screens of a hierarchical prior on the recalls of BayCAP (E2.6), vanilla accuracy, $|U|=1000$. For each prior on $\\nu$: the "
    "ratio of the 90\\% widths of the posterior and of the prior of $\\ln\\nu$ (D1), and the predicted coverage and "
    "Winkler score of the realised pushforward (D4). Stored realised coverage with the "
    "$\\operatorname{Dir}(\\mathbf 1)$ and EB priors. Log-uniform prior: smallest leave-one-class-out p-value, "
    "largest vertex shift $|z_j|$ (D2), and "
    "$P(\\min(\\nu m,\\nu(1-m))<1\\mid V)$ (D3), next to the stored share of bags with divergent transitions of "
    "BayCAP-EB."
)
VALIDATION_LABEL = "tab:hierarchical-validation"
SCREENS_LABEL = "tab:hierarchical-screens"


def _index(table: pd.DataFrame) -> pd.MultiIndex:
    names = table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d)))
    return pd.MultiIndex.from_arrays(
        [names, table["n_classes"].astype(int), table["classifier"].astype(str)],
        names=["dataset", "$K$", "classifier"],
    )


SCORE_LABELS = {"coverage": "$\\mathcal{C}$", "amplitude": "$\\mathcal{A}$", "winkler": "$\\mathcal{W}$"}


def _validation_styler(table: pd.DataFrame):
    cols = {}
    for name, (_, eb_method) in PUSHFORWARDS.items():
        for score in SCORES:
            cols[(name, SCORE_LABELS[score], "stored")] = table[f"{eb_method}_{score}"].to_numpy()
            cols[(name, SCORE_LABELS[score], "pred.")] = table[f"{name}_eb_pred_{score}"].to_numpy()
            if score == "coverage":
                cols[(name, SCORE_LABELS[score], "+$P$")] = table[f"{name}_ebprev_pred_{score}"].to_numpy()
                cols[(name, SCORE_LABELS[score], "centre")] = table[f"{name}_ebcentre_pred_{score}"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table)
    return out.style.format(precision=3, na_rep="--")


def _fmt_p(value: float) -> str:
    return "--" if np.isnan(value) else (f"{value:.0e}" if value < 0.001 else f"{value:.3f}")


def _screens_styler(table: pd.DataFrame):
    cols = {}
    for prior in NU_PRIORS:
        label = NU_PRIOR_LABELS[prior]
        cols[(label, "width")] = table[f"{prior}_width_ratio"].to_numpy()
        cols[(label, "$\\mathcal{C}$")] = table[f"realised_{prior}_pred_coverage"].to_numpy()
        cols[(label, "$\\mathcal{W}$")] = table[f"realised_{prior}_pred_winkler"].to_numpy()
    cols[("stored $\\mathcal{C}$", "unif.")] = table["BayesCAP-real_coverage"].to_numpy()
    cols[("stored $\\mathcal{C}$", "EB")] = table["BayesCAP-EB-real_coverage"].to_numpy()
    cols[("D2", "min $p$")] = [_fmt_p(v) for v in table[f"{MAIN_PRIOR}_min_pvalue"]]
    cols[("D2", "max $|z_j|$")] = table[f"{MAIN_PRIOR}_max_abs_z"].to_numpy()
    cols[("D3", "$P(\\alpha<1)$")] = table[f"{MAIN_PRIOR}_alpha_risk"].to_numpy()
    cols[("D3", "EB div.")] = table["BayesCAP-EB_share_divergent"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table)
    return out.style.format(precision=3, na_rep="--")


TABLES = {
    "validation": (_validation_styler, VALIDATION_CAPTION, VALIDATION_LABEL),
    "screens": (_screens_styler, SCREENS_CAPTION, SCREENS_LABEL),
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
    parser = ArgumentParser(description="Zero-cost screens for a hierarchical prior on the recalls of BayCAP (E2.6).")
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
