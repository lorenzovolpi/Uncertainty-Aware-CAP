"""BayCAP with a population or a realised-table estimand.

This module implements the method of work-plan card WP-1, motivated by the
realised-table posterior predictive of ``docs/ext2_baycap.md``. The
posterior model, the priors and the NUTS sampler are copied from ``cap-ml``
``cap.models.confidence.ACC_BayesCAP`` (commit ``bb20aa5`` of branch
``devel``); only the pushforward of the posterior draws to contingency tables
is extended. Work-plan card WP-5 adds the empirical-Bayes prior on the
columns of $C$ (``c_prior``, §T.9 of ``docs/ext2_baycap.md``), work-plan
card WP-6 the NUTS diagnostics of each ``predict_ct_range`` call
(``last_nuts_diagnostics``), and work-plan card WP-10 the hierarchical prior
on the recalls (``c_prior="hierarchical"``, §T.14 and §T.15) with its
per-bag posterior summaries (``last_posterior_summaries``). Work-plan card
WP-11 adds its variant ``c_prior="hierarchical-f"`` (§T.16): a lighter tail
on the strength $\\nu$ and a learned concentration $\\kappa$ of the shares of
the mistakes. Work-plan card WP-12 adds a non-centred form of those shares
(``shares="non-centred"``, §T.17): the same model, in other coordinates.
"""

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Number
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
from cap.models.confidence import P_C_COND_Y, P_TEST_C, P_TEST_Y, BayesCAP, CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from quapy.data import LabelledCollection
from scipy.optimize import minimize
from scipy.special import betaln, digamma, expit, logit
from sklearn.metrics import confusion_matrix

ESTIMANDS = ("population", "realised")
C_PRIORS = ("uniform", "empirical-bayes", "hierarchical", "hierarchical-f")
# the priors with a hierarchical prior on the recalls, sampled by NUTS with C and pi (§T.15, §T.16)
HIERARCHICAL_PRIORS = ("hierarchical", "hierarchical-f")
# coordinates of the shares of the mistakes of "hierarchical-f" (§T.17): the Dirichlet itself, or normalised Gammas
# built from variables whose law does not depend on kappa
SHARES_FORMS = ("centred", "non-centred")

# per-bag NUTS diagnostics stored by the experiments (``nuts_diagnostics``)
NUTS_DIAGNOSTICS = ("nuts_divergences", "nuts_max_rhat", "nuts_min_ess")
# per-bag posterior summaries of the hierarchical prior stored by the experiments (``last_posterior_summaries``);
# the vectors are stored flat, the covariance of pi row by row
HIERARCHICAL_SUMMARIES = (
    "nuts_failed",
    "h_m_mean",
    "h_ln_nu_median",
    "h_ln_nu_q05",
    "h_ln_nu_q95",
    "h_p_nu_above_min_n",
    "h_recall_mean",
    "h_pi_mean",
    "h_pi_cov",
)
# the posterior summaries of the concentration kappa of the shares of the mistakes, added by "hierarchical-f" (§T.16);
# NaN with K=2, where there are no shares
KAPPA_SUMMARIES = ("h_ln_kappa_median", "h_ln_kappa_q05", "h_ln_kappa_q95", "h_p_kappa_below_1")

# sites of the hierarchical prior (§T.15): the centre m, u = 1/sqrt(1+nu), the strength nu, the recalls and the shares
# of the mistakes of each true class
H_M, H_U, H_NU, H_RECALLS, H_SHARES = "h_m", "h_u", "h_nu", "h_recalls", "h_shares"
# sites added by "hierarchical-f" (§T.16): rho = 1/(1+nu), which replaces u, and rho_kappa = 1/(1+kappa), with the
# concentration kappa of the shares of the mistakes (only for K >= 3)
H_RHO, H_RHO_KAPPA, H_KAPPA = "h_rho", "h_rho_kappa", "h_kappa"
# sites of the non-centred shares (§T.17): G' ~ Gamma(kappa + 1, 1) and e ~ Exp(1) in every off-diagonal cell
H_SHARES_GAMMA, H_SHARES_EXP = "h_shares_gamma", "h_shares_exp"

# bounds of the empirical-Bayes fit (§T.9); the lower bound on nu is only a numerical safeguard
EB_M_BOUNDS = (1e-3, 1 - 1e-3)
EB_NU_BOUNDS = (1e-3, 1e4)


def realised_tables(cts: np.ndarray, pred_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Sample one realised contingency table of the target sample per table draw.

    For each draw $r$ and each predicted class $\omega_i$, the true classes of
    the $n^t_i$ target instances predicted as $\omega_i$ are drawn as
    $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, with
    $\rho^{(r)}_{ij}=C^{(r)}_{ij}\pi^{(r)}_j/\sum_l C^{(r)}_{il}\pi^{(r)}_l$,
    i.e. the draw's table restricted to predicted class $\omega_i$ and
    normalised to one. The sampled counts are divided by
    $N^Q=\sum_i n^t_i$.

    Tables follow the ``cap-ml`` layout (true classes on rows, predicted
    classes on columns), so each column of a draw is resampled.

    :param cts: table draws of shape ``(n_draws, n_classes, n_classes)``,
        non-negative, in the ``cap-ml`` layout.
    :param pred_counts: number of target instances predicted as each class,
        of shape ``(n_classes,)``.
    :param rng: NumPy generator used for the multinomial draws.
    :return: normalised realised tables with the shape of ``cts``; the column
        sums of every table equal ``pred_counts / pred_counts.sum()``.
    """
    cts = np.asarray(cts, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=np.int64)
    if cts.ndim != 3 or cts.shape[1] != cts.shape[2]:
        raise ValueError(f"cts must have shape (n_draws, n_classes, n_classes); found {cts.shape}")
    if pred_counts.shape != (cts.shape[2],):
        raise ValueError(f"pred_counts must have shape ({cts.shape[2]},); found {pred_counts.shape}")
    if np.any(pred_counts < 0) or pred_counts.sum() == 0:
        raise ValueError("pred_counts must be non-negative and not all zero")
    if np.any(cts < 0) or not np.all(np.isfinite(cts)):
        raise ValueError("cts must contain finite non-negative values")

    # rho[r, c, y] = P(Y=y | Yhat=c) under draw r; empty columns fall back to uniform
    rho = np.swapaxes(cts, 1, 2)
    col_mass = rho.sum(axis=2, keepdims=True)
    n_classes = cts.shape[1]
    rho = np.divide(rho, col_mass, out=np.full_like(rho, 1.0 / n_classes), where=col_mass > 0)

    n = np.broadcast_to(pred_counts, rho.shape[:2])
    counts = rng.multinomial(n, rho)
    return np.swapaxes(counts, 1, 2) / pred_counts.sum()


def nuts_diagnostics(draws: dict[str, np.ndarray], diverging: np.ndarray) -> dict[str, float]:
    r"""Summarise the convergence of one NUTS run.

    :param draws: posterior draws of each parameter, grouped by chain, of shape
        ``(n_chains, n_draws, ...)``.
    :param diverging: flags of the divergent transitions after warm-up.
    :return: ``nuts_divergences``, the number of divergent transitions;
        ``nuts_max_rhat``, the largest split-$\hat R$ over all the entries of
        all the parameters (each chain is split in two halves, so a single
        chain gives two); ``nuts_min_ess``, the smallest effective sample size
        over the same entries.
    """
    rhat = [np.asarray(split_gelman_rubin(np.asarray(x))).ravel() for x in draws.values()]
    ess = [np.asarray(effective_sample_size(np.asarray(x))).ravel() for x in draws.values()]
    return {
        "nuts_divergences": int(np.sum(diverging)),
        "nuts_max_rhat": float(np.max(np.concatenate(rhat))),
        "nuts_min_ess": float(np.min(np.concatenate(ess))),
    }


def beta_binomial_loglik(m: float, nu: float, f: np.ndarray, n: np.ndarray) -> float:
    r"""Beta-binomial log-likelihood of $(m,\nu)$ for the diagonal validation counts.

    $\ell(m,\nu)=\sum_j[\ln B(f_j+\nu m,\,n_j-f_j+\nu(1-m))-\ln B(\nu m,\,\nu(1-m))]$,
    i.e. the log-probability of the counts $f_j$ out of $n_j$ when each recall
    is drawn from $\operatorname{Beta}(\nu m,\nu(1-m))$, without the binomial
    coefficients (which do not depend on $(m,\nu)$).

    :param m: centre of the Beta prior, in $(0,1)$.
    :param nu: strength of the Beta prior, positive.
    :param f: correctly classified validation instances of each class.
    :param n: validation instances of each class.
    """
    f, n = np.asarray(f, dtype=float), np.asarray(n, dtype=float)
    a, b = nu * m, nu * (1 - m)
    return float(np.sum(betaln(f + a, n - f + b) - betaln(a, b)))


def _loglik_grad(theta: np.ndarray, f: np.ndarray, n: np.ndarray) -> tuple[float, np.ndarray]:
    """Log-likelihood and its gradient in the coordinates (logit m, ln nu)."""
    m, nu = expit(theta[0]), np.exp(theta[1])
    a, b = nu * m, nu * (1 - m)
    common = digamma(nu) - digamma(n + nu)
    d_a = np.sum(digamma(f + a) - digamma(a) + common)
    d_b = np.sum(digamma(n - f + b) - digamma(b) + common)
    grad = np.array([nu * m * (1 - m) * (d_a - d_b), a * d_a + b * d_b])
    return beta_binomial_loglik(m, nu, f, n), grad


@dataclass(frozen=True)
class BetaBinomialFit:
    r"""Maximum-likelihood estimate of the Beta prior on the recalls (§T.9).

    :param m: the centre $\hat m$ (the typical recall).
    :param nu: the strength $\hat\nu$.
    :param m_at_bound: whether $\hat m$ is on a bound of ``EB_M_BOUNDS``.
    :param nu_at_bound: whether $\hat\nu$ is on a bound of ``EB_NU_BOUNDS``.
    :param sd_m: $\operatorname{sd}(\hat m)$ from the curvature of the
        log-likelihood at its maximum; NaN when $\hat m$ is on a bound or the
        curvature is not negative.
    :param loglik: the log-likelihood at the maximum.
    """

    m: float
    nu: float
    m_at_bound: bool
    nu_at_bound: bool
    sd_m: float
    loglik: float

    @property
    def at_bound(self) -> bool:
        return self.m_at_bound or self.nu_at_bound


def fit_beta_binomial(
    f: np.ndarray, n: np.ndarray, starts_ln_nu: tuple[float, ...] = (0.0, 3.0, 6.0)
) -> BetaBinomialFit:
    r"""Fit $(\hat m,\hat\nu)$ by maximising the beta-binomial log-likelihood.

    The maximum is searched over $(\operatorname{logit}m,\ln\nu)$ with L-BFGS-B
    and the analytic gradient, within ``EB_M_BOUNDS`` and ``EB_NU_BOUNDS``,
    from the pooled recall $\sum_jf_j/\sum_jn_j$ and each value of
    ``starts_ln_nu``; the best maximum is kept.

    $\operatorname{sd}(\hat m)=\hat m(1-\hat m)\operatorname{sd}(\operatorname{logit}\hat m)$,
    with the variance of $\operatorname{logit}\hat m$ from the inverse of the
    observed information (minus the Hessian of the log-likelihood, by central
    differences of the gradient). When $\hat\nu$ is on a bound, $\nu$ is
    treated as fixed and only the curvature in $\operatorname{logit}m$ is used.

    :param f: correctly classified validation instances of each class
        (the diagonal of the validation table).
    :param n: validation instances of each class (its class totals).
    """
    f, n = np.asarray(f, dtype=float), np.asarray(n, dtype=float)
    if f.shape != n.shape or f.ndim != 1:
        raise ValueError(f"f and n must be vectors of the same shape; found {f.shape} and {n.shape}")
    if np.any(f < 0) or np.any(f > n) or n.sum() == 0:
        raise ValueError("counts must satisfy 0 <= f <= n, with at least one validation instance")

    bounds = [tuple(logit(EB_M_BOUNDS)), tuple(np.log(EB_NU_BOUNDS))]
    m0 = np.clip(f.sum() / n.sum(), *EB_M_BOUNDS)

    def neg(theta):
        ll, grad = _loglik_grad(theta, f, n)
        return -ll, -grad

    best = None
    for ln_nu in starts_ln_nu:
        res = minimize(neg, np.array([logit(m0), ln_nu]), jac=True, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    theta = best.x

    at_bound = [np.isclose(theta[i], bounds[i], rtol=0, atol=1e-6).any() for i in range(2)]
    free = [i for i in range(2) if not at_bound[i]]
    sd_m = np.nan
    if 0 in free:
        eps = 1e-5
        hess = np.empty((2, 2))
        for i in range(2):
            step = np.zeros(2)
            step[i] = eps
            hess[:, i] = (_loglik_grad(theta + step, f, n)[1] - _loglik_grad(theta - step, f, n)[1]) / (2 * eps)
        info = -(hess + hess.T)[np.ix_(free, free)] / 2
        if np.all(np.linalg.eigvalsh(info) > 0):
            var_logit_m = np.linalg.inv(info)[0, 0]
            m = expit(theta[0])
            sd_m = float(m * (1 - m) * np.sqrt(var_logit_m))

    return BetaBinomialFit(
        m=float(expit(theta[0])),
        nu=float(np.exp(theta[1])),
        m_at_bound=bool(at_bound[0]),
        nu_at_bound=bool(at_bound[1]),
        sd_m=sd_m,
        loglik=-float(best.fun),
    )


def eb_concentration(n_classes: int, m: float, nu: float) -> np.ndarray:
    r"""Dirichlet concentrations of the empirical-Bayes prior on the columns of $C$.

    Column $j$ of $C$ gets $\alpha_{jj}=\nu m$ and $\alpha_{ij}=\nu(1-m)/(K-1)$
    for $i\ne j$, so that the recall $C_{jj}$ is $\operatorname{Beta}(\nu m,\nu(1-m))$.
    The matrix is returned in the ``cap-ml`` layout of ``p_c_cond_y`` (row $j$
    is the column $C_{:j}$, i.e. the true class on rows), so each row sums to
    $\nu$; with a constant diagonal and constant off-diagonal entries it is
    symmetric, so the two layouts coincide.
    """
    if n_classes < 2:
        raise ValueError(f"n_classes must be at least 2; found {n_classes}")
    off = nu * (1 - m) / (n_classes - 1)
    alpha = np.full((n_classes, n_classes), off)
    np.fill_diagonal(alpha, nu * m)
    return alpha


def nu_from_u(u):
    r"""Return the strength $\nu=1/u^2-1$ of the hierarchical prior from $u=1/\sqrt{1+\nu}$ (§T.15).

    With $u\sim U(0,1)$, $\nu$ has the density $\frac12(1+\nu)^{-3/2}$ on $(0,\infty)$, with median $3$.
    Works on NumPy and JAX arrays.
    """
    return 1.0 / u**2 - 1.0


def strength_from_rho(rho):
    r"""Return $1/\rho-1$, the strength $\nu$ (or the concentration $\kappa$) of "hierarchical-f" from
    $\rho=1/(1+\nu)$ (§T.16).

    With $\rho\sim U(0,1)$, $\nu$ has the density $(1+\nu)^{-2}$ on $(0,\infty)$, with $P(\nu>x)=1/(1+x)$ and median
    $1$. Works on NumPy and JAX arrays.
    """
    return 1.0 / rho - 1.0


def noncentred_shares(gamma, exponential, kappa):
    r"""The shares of the mistakes from their non-centred coordinates (§T.17).

    With $G'_{ij}\sim\operatorname{Gamma}(\kappa+1,1)$ and $e_{ij}\sim\operatorname{Exp}(1)$ independent,
    $g_{ij}=G'_{ij}e^{-e_{ij}/\kappa}=G'_{ij}U_{ij}^{1/\kappa}$ (with $U_{ij}=e^{-e_{ij}}$ uniform) is
    $\operatorname{Gamma}(\kappa,1)$, so the normalised $s_{ij}=g_{ij}/\sum_kg_{kj}$ of each true class are
    $\operatorname{Dir}(\kappa\mathbf 1_{K-1})$ exactly. The normalisation is a softmax of $\ln G'_{ij}-e_{ij}/\kappa$
    over the last axis, computed in logs since $e_{ij}/\kappa$ can be large. Works on NumPy and JAX arrays.

    :param gamma: $G'$, shape ``(..., K, K-1)``, positive.
    :param exponential: $e$, the same shape, non-negative.
    :param kappa: the concentration $\kappa>0$, broadcast against them.
    :return: the shares, the same shape, summing to $1$ over the last axis.
    """
    return jax.nn.softmax(jnp.log(gamma) - exponential / kappa, axis=-1)


def off_diagonal_index(n_classes: int) -> np.ndarray:
    """For each class $j$, the other classes in increasing order, shape ``(n_classes, n_classes - 1)``."""
    return np.array([[i for i in range(n_classes) if i != j] for j in range(n_classes)], dtype=int)


def hierarchical_rates(recalls, shares):
    r"""Build the rate matrix from the recalls and the shares of the mistakes, in the ``cap-ml`` layout.

    Row $j$ is the column $C_{:j}$ of the paper: $C_{jj}=r_j$ and, for the other classes $i$ in increasing order,
    $C_{ij}=(1-r_j)s_{j,i}$, so each row sums to $1$ when the shares of each class sum to $1$.

    :param recalls: the recalls $r_j$, shape ``(K,)``.
    :param shares: the shares of the mistakes of each true class, shape ``(K, K-1)``.
    """
    n_classes = recalls.shape[0]
    rows = np.repeat(np.arange(n_classes), n_classes - 1)
    cols = off_diagonal_index(n_classes).ravel()
    off = ((1 - recalls)[:, None] * shares).ravel()
    diag = np.arange(n_classes)
    return jnp.zeros((n_classes, n_classes)).at[rows, cols].set(off).at[diag, diag].set(recalls)


def hierarchical_rate_sites(n_classes: int, light: bool, shares: str = "centred"):
    r"""Sample the hierarchical prior on $C$ inside a NumPyro model and return ``p_c_cond_y`` in the ``cap-ml`` layout.

    With ``light=False``, the prior of ``c_prior="hierarchical"`` (§T.15 of ``docs/ext2_baycap.md``): $m\sim U(0,1)$,
    $u=1/\sqrt{1+\nu}\sim U(0,1)$, centred recalls $C_{jj}\mid m,\nu\sim\operatorname{Beta}(\nu m,\nu(1-m))$ and
    shares of the mistakes $\operatorname{Dir}(\mathbf 1_{K-1})$. With ``light=True``, that of ``"hierarchical-f"``
    (§T.16): $\rho=1/(1+\nu)\sim U(0,1)$ instead of $u$, and shares $\operatorname{Dir}(\kappa\mathbf 1_{K-1})$ with
    $1/(1+\kappa)\sim U(0,1)$, in the coordinates of ``shares`` (§T.17). With $K=2$ there are no shares and no
    $\kappa$. Used by :class:`BayCAP` and by ``methods/baycap_kdey.py:BayCAPKDEy`` (E3.2 of
    ``docs/ext3_posterior_propagation.md``), so that the two models have the same sites, in the same order.
    """
    m = numpyro.sample(H_M, dist.Uniform(0.0, 1.0))
    if light:
        rho = numpyro.sample(H_RHO, dist.Uniform(0.0, 1.0))
        nu = numpyro.deterministic(H_NU, strength_from_rho(rho))
    else:
        u = numpyro.sample(H_U, dist.Uniform(0.0, 1.0))
        nu = numpyro.deterministic(H_NU, nu_from_u(u))
    with numpyro.plate("classes", n_classes):
        recalls = numpyro.sample(H_RECALLS, dist.Beta(nu * m, nu * (1 - m)))
    if n_classes > 2:
        if light:
            rho_kappa = numpyro.sample(H_RHO_KAPPA, dist.Uniform(0.0, 1.0))
            kappa = numpyro.deterministic(H_KAPPA, strength_from_rho(rho_kappa))
        if light and shares == "non-centred":
            cells = (n_classes, n_classes - 1)
            gamma = numpyro.sample(H_SHARES_GAMMA, dist.Gamma(kappa + 1.0, 1.0).expand(cells).to_event(2))
            exponential = numpyro.sample(H_SHARES_EXP, dist.Exponential(1.0).expand(cells).to_event(2))
            share_draws = numpyro.deterministic(H_SHARES, noncentred_shares(gamma, exponential, kappa))
        else:
            concentration = (kappa if light else 1.0) * jnp.ones((n_classes, n_classes - 1))
            share_draws = numpyro.sample(H_SHARES, dist.Dirichlet(concentration))
    else:
        share_draws = jnp.ones((n_classes, 1))
    return numpyro.deterministic(P_C_COND_Y, hierarchical_rates(recalls, share_draws))


def hierarchical_summaries(draws: dict[str, np.ndarray], min_val_count: float, with_kappa: bool = False) -> dict:
    r"""Summarise the posterior of the hierarchical prior from the draws of one NUTS run (E2.7, E2.8).

    :param draws: the draws of ``H_M``, ``H_NU``, ``H_RECALLS`` and ``P_TEST_Y``, and of ``H_KAPPA`` for
        "hierarchical-f" with $K\ge3$, with the draws on the first axis.
    :param min_val_count: $\min_jn_j$, the validation instances of the smallest class.
    :param with_kappa: add the summaries of $\kappa$ (``KAPPA_SUMMARIES``), NaN when ``draws`` has no ``H_KAPPA``.
    :return: ``nuts_failed`` ($0$); the posterior mean of $m$; the median and the $5\%$ and $95\%$ quantiles of
        $\ln\nu$; $P(\nu>\min_jn_j)$; the posterior mean of the recalls; the posterior mean of $\pi^Q$ and its
        covariance, flattened row by row; with ``with_kappa``, the median and the $5\%$ and $95\%$ quantiles of
        $\ln\kappa$ and $P(\kappa<1)$.
    """
    ln_nu = np.log(np.asarray(draws[H_NU], dtype=float))
    pi = np.asarray(draws[P_TEST_Y], dtype=float)
    q05, median, q95 = np.quantile(ln_nu, [0.05, 0.5, 0.95])
    return {
        "nuts_failed": 0,
        "h_m_mean": float(np.mean(draws[H_M])),
        "h_ln_nu_median": float(median),
        "h_ln_nu_q05": float(q05),
        "h_ln_nu_q95": float(q95),
        "h_p_nu_above_min_n": float(np.mean(np.asarray(draws[H_NU]) > min_val_count)),
        "h_recall_mean": np.asarray(draws[H_RECALLS], dtype=float).mean(axis=0),
        "h_pi_mean": pi.mean(axis=0),
        "h_pi_cov": np.cov(pi, rowvar=False).ravel(),
    } | (kappa_summaries(draws.get(H_KAPPA)) if with_kappa else {})


def kappa_summaries(kappa: np.ndarray | None) -> dict:
    r"""The median and the $5\%$ and $95\%$ quantiles of $\ln\kappa$ and $P(\kappa<1)$ from the draws of $\kappa$; NaN
    if there are none ($K=2$)."""
    if kappa is None:
        return dict.fromkeys(KAPPA_SUMMARIES, np.nan)
    kappa = np.asarray(kappa, dtype=float)
    q05, median, q95 = np.quantile(np.log(kappa), [0.05, 0.5, 0.95])
    return dict(zip(KAPPA_SUMMARIES, (float(median), float(q05), float(q95), float(np.mean(kappa < 1)))))


def failed_summaries(n_classes: int, with_kappa: bool = False) -> dict:
    """The posterior summaries of a bag where NUTS produced no draws: ``nuts_failed`` is $1$, the rest NaN."""
    return {
        **{name: np.nan for name in HIERARCHICAL_SUMMARIES},
        "nuts_failed": 1,
        "h_recall_mean": np.full(n_classes, np.nan),
        "h_pi_mean": np.full(n_classes, np.nan),
        "h_pi_cov": np.full(n_classes * n_classes, np.nan),
    } | (dict.fromkeys(KAPPA_SUMMARIES, np.nan) if with_kappa else {})


class BayCAP(CAPContingencyTable, CTCAPWithConfidence, BayesCAP):
    r"""BayCAP with a selectable estimand.

    The posterior over the target prevalence $\pi^Q$ and the rate matrix $C$
    is the BayesianCC model of ``ACC_BayesCAP``: a Dirichlet prior on
    $\pi^Q$ (uniform by default), a Dirichlet prior on each column of $C$, a
    multinomial likelihood for the validation contingency table and a
    multinomial likelihood for the predicted counts $n^t$ of the target
    sample. It is sampled with NumPyro NUTS from ``PRNGKey(random_state)``.

    The prior on the columns of $C$ is selected by ``c_prior``:

    - ``"uniform"`` is $\operatorname{Dir}(\mathbf 1_K)$, as in ``ACC_BayesCAP``.
    - ``"empirical-bayes"`` gives column $j$ the prior
      $\operatorname{Dir}(\alpha_{\cdot j})$ with $\alpha_{jj}=\hat\nu\hat m$
      and $\alpha_{ij}=\hat\nu(1-\hat m)/(K-1)$, so that the recall $C_{jj}$
      is $\operatorname{Beta}(\hat\nu\hat m,\hat\nu(1-\hat m))$; ``fit``
      estimates $(\hat m,\hat\nu)$ from the diagonal and the class totals of
      the validation table (:func:`fit_beta_binomial`) and stores the fit in
      ``eb_fit``.
    - ``"hierarchical"`` (§T.14, §T.15) gives the recalls a common prior with
      its own parameters, sampled by NUTS with $C$ and $\pi^Q$: $m\sim U(0,1)$,
      $u=1/\sqrt{1+\nu}\sim U(0,1)$ (so $p(\nu)=\frac12(1+\nu)^{-3/2}$),
      $C_{jj}\mid m,\nu\sim\operatorname{Beta}(\nu m,\nu(1-m))$
      independently, and the mistakes of each true class shared among the
      other classes by $\operatorname{Dir}(\mathbf 1_{K-1})$
      (:func:`hierarchical_rates`); with $K=2$ the shares are $1$. The
      parametrisation is centred: each recall is a parameter of its own. After
      each ``predict_ct_range`` call, :meth:`last_posterior_summaries` returns
      the posterior summaries of $m$, $\nu$, the recalls and $\pi^Q$
      (:func:`hierarchical_summaries`). If NUTS cannot start (no valid initial
      parameters), the bag gets tables of NaN, NaN diagnostics and
      ``nuts_failed`` equal to $1$, so that the other bags of the pair are kept.
    - ``"hierarchical-f"`` (§T.16) is ``"hierarchical"`` with two changes:
      $\rho=1/(1+\nu)\sim U(0,1)$ instead of $u$ (so $p(\nu)=(1+\nu)^{-2}$,
      a lighter tail), and the shares of the mistakes of each true class
      $\operatorname{Dir}(\kappa\mathbf 1_{K-1})$ with a common concentration
      $\kappa$, $1/(1+\kappa)\sim U(0,1)$ (so $p(\kappa)=(1+\kappa)^{-2}$),
      sampled by NUTS too; with $K=2$ there are no shares and no $\kappa$. Its
      posterior summaries add those of $\kappa$ (``KAPPA_SUMMARIES``, NaN with
      $K=2$), and a failed run is handled as for ``"hierarchical"``.
      ``shares`` selects the coordinates in which NUTS samples its shares:
      ``"centred"`` samples the Dirichlet itself; ``"non-centred"`` (§T.17)
      samples, in every off-diagonal cell, $G'_{ij}\sim\operatorname{Gamma}(\kappa+1,1)$
      and $e_{ij}\sim\operatorname{Exp}(1)$, and computes the shares from
      them (:func:`noncentred_shares`), with the same law given $\kappa$: the
      law of $e_{ij}$ does not depend on $\kappa$, and that of
      $\ln G'_{ij}$ keeps a left tail of rate at least $1$, so their
      coordinates do not widen as $\kappa$ falls. The model and the posterior
      are unchanged, the draws are not.

    After each ``predict_ct_range`` call, :meth:`last_nuts_diagnostics` returns
    the number of divergent transitions, the largest split-$\hat R$ and the
    smallest effective sample size of that NUTS run, over the entries of
    $\pi^Q$ and $C$, and also of $m$ and $u$ with the hierarchical prior, of
    $m$, $\rho$ and $1/(1+\kappa)$ with ``"hierarchical-f"``
    (:func:`nuts_diagnostics`). Recording them does not change the draws.

    The estimand decides how each posterior draw becomes a contingency table:

    - ``"population"`` returns the population table
      $C^{(r)}\operatorname{diag}(\pi^{(r)})$, exactly as ``ACC_BayesCAP``.
      It represents the uncertainty on $\pi^Q$ and on $C$, not the
      finite-sample fluctuation of the target sample's table.
    - ``"realised"`` returns the realised-table posterior predictive (see
      :func:`realised_tables`): for each draw, the true classes behind the
      observed predictions are sampled. It represents the uncertainty on
      $\pi^Q$ and on $C$ and the finite-sample fluctuation of the realised
      table of the target sample, which is what the experiments score
      (``true_cts``).

    Both estimands share the NUTS draws for the same ``random_state``; the
    multinomial draws of ``"realised"`` use a NumPy generator seeded with
    ``random_state``. As in ``cap-ml``, tables have true classes on rows, so
    the paper's $C\operatorname{diag}(\pi)$ is returned transposed, as
    $\operatorname{diag}(\pi)C^\top$.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param estimand: ``"population"`` or ``"realised"``.
    :param c_prior: prior on the columns of $C$, ``"uniform"``,
        ``"empirical-bayes"``, ``"hierarchical"`` or ``"hierarchical-f"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param shares: coordinates of the shares of ``"hierarchical-f"``,
        ``"centred"`` (default) or ``"non-centred"``; only ``"centred"`` is
        allowed with the other priors, which have no learned concentration.
    :param prior: Dirichlet parameter of the prior on $\pi^Q$: ``"uniform"``,
        a positive scalar repeated for every class, or a vector with one
        value per class.
    :param random_state: seed of the NUTS sampler and of the multinomial
        draws; ``qp.environ["_R_SEED"]`` if ``None``, as in ``cap-ml``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        estimand: Literal["population", "realised"] = "population",
        c_prior: Literal["uniform", "empirical-bayes", "hierarchical", "hierarchical-f"] = "uniform",
        num_warmup: int = 500,
        num_samples: int = 1000,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
        shares: Literal["centred", "non-centred"] = "centred",
    ):
        if estimand not in ESTIMANDS:
            raise ValueError(f"estimand must be one of {ESTIMANDS}; found {estimand!r}")
        if c_prior not in C_PRIORS:
            raise ValueError(f"c_prior must be one of {C_PRIORS}; found {c_prior!r}")
        if shares not in SHARES_FORMS:
            raise ValueError(f"shares must be one of {SHARES_FORMS}; found {shares!r}")
        if shares != "centred" and c_prior != "hierarchical-f":
            raise ValueError(f"shares={shares!r} needs c_prior='hierarchical-f'; found {c_prior!r}")
        self.shares = shares
        CAPContingencyTable.__init__(self, acc_fn)
        self.estimand = estimand
        self.c_prior = c_prior
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state
        self.prior = prior

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        val_yhat = np.argmax(posteriors, axis=-1)
        self.classes = val.classes
        self.val_ct = confusion_matrix(val.y, val_yhat, labels=val.classes)
        n_classes = len(self.classes)
        if self.c_prior == "empirical-bayes":
            # cap-ml layout: true classes on rows, so the diagonal holds f_j and the row sums n_j
            self.eb_fit = fit_beta_binomial(np.diag(self.val_ct), self.val_ct.sum(axis=1))
            self.c_alpha = eb_concentration(n_classes, self.eb_fit.m, self.eb_fit.nu)
        elif self.c_prior in HIERARCHICAL_PRIORS:
            self.eb_fit = None
            self.c_alpha = None
        else:
            self.eb_fit = None
            self.c_alpha = np.ones((n_classes, n_classes))
        return self

    def model(
        self,
        pred_posterior_count: np.ndarray,
        train_class_cond_count: np.ndarray,
        alpha: np.ndarray,
        c_alpha: np.ndarray,
    ):
        train_class_count = train_class_cond_count.sum(axis=1)

        L = len(train_class_count)

        pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.asarray(alpha, dtype=jnp.float32)))
        if c_alpha is None:
            p_c_cond_y = self._hierarchical_model(L)
        else:
            # row y of p_c_cond_y is the column C_{:y} of the paper, with concentrations c_alpha[y]
            p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.asarray(c_alpha, dtype=jnp.float32)))

        with numpyro.plate("plate", L):
            numpyro.sample("F_yc", dist.Multinomial(train_class_count, p_c_cond_y), obs=train_class_cond_count)

        p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
        numpyro.sample("N_c", dist.Multinomial(jnp.sum(pred_posterior_count), p_c), obs=pred_posterior_count)

    def _hierarchical_model(self, n_classes: int):
        """The hierarchical prior on $C$ of ``c_prior`` (:func:`hierarchical_rate_sites`); returns ``p_c_cond_y`` in the
        ``cap-ml`` layout."""
        return hierarchical_rate_sites(n_classes, light=self.c_prior == "hierarchical-f", shares=self.shares)

    def sample_posterior(self, posterior_count: np.ndarray, alpha: np.ndarray) -> dict:
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self.model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(self.random_state)
        mcmc.run(
            rng_key,
            pred_posterior_count=posterior_count,
            train_class_cond_count=self.val_ct,
            alpha=alpha,
            c_alpha=self.c_alpha,
            extra_fields=("diverging",),
        )
        # kept for last_nuts_diagnostics, which is computed on request so that it does not count in the test time
        grouped = mcmc.get_samples(group_by_chain=True)
        names = (P_TEST_Y, P_C_COND_Y) + self._hyper_sites()
        self._last_nuts = (
            {name: np.asarray(jax.device_get(grouped[name])) for name in names},
            np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])),
        )
        samples = mcmc.get_samples()
        if self.c_prior in HIERARCHICAL_PRIORS:
            # kept for last_posterior_summaries, computed on request as the diagnostics
            summary_sites = (H_M, H_NU, H_RECALLS, P_TEST_Y) + ((H_KAPPA,) if H_KAPPA in samples else ())
            self._last_summaries = (
                {name: np.asarray(jax.device_get(samples[name])) for name in summary_sites},
                float(self.val_ct.sum(axis=1).min()),
            )
        return samples

    def _hyper_sites(self) -> tuple[str, ...]:
        r"""The sampled sites of the hierarchical priors whose convergence is checked with that of $\pi^Q$ and $C$."""
        if self.c_prior == "hierarchical":
            return (H_M, H_U)
        if self.c_prior == "hierarchical-f":
            return (H_M, H_RHO) + ((H_RHO_KAPPA,) if len(self.classes) > 2 else ())
        return ()

    def last_nuts_diagnostics(self) -> dict[str, float]:
        """Return the NUTS diagnostics (:func:`nuts_diagnostics`) of the last ``sample_posterior`` call; NaN if NUTS
        could not start (hierarchical priors only)."""
        if getattr(self, "_last_nuts", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        if self._last_nuts == "failed":
            return {name: np.nan for name in NUTS_DIAGNOSTICS}
        return nuts_diagnostics(*self._last_nuts)

    def last_posterior_summaries(self) -> dict:
        r"""Return the posterior summaries of the hierarchical priors (:func:`hierarchical_summaries`, with those of
        $\kappa$ for ``"hierarchical-f"``) of the last ``sample_posterior`` call, or of a failed run
        (:func:`failed_summaries`); empty for the other priors."""
        if self.c_prior not in HIERARCHICAL_PRIORS:
            return {}
        if getattr(self, "_last_summaries", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        with_kappa = self.c_prior == "hierarchical-f"
        if self._last_summaries == "failed":
            return failed_summaries(len(self.classes), with_kappa)
        return hierarchical_summaries(*self._last_summaries, with_kappa=with_kappa)

    def _alpha(self) -> np.ndarray:
        n_classes = len(self.classes)
        if isinstance(self.prior, str) and self.prior == "uniform":
            return np.ones(n_classes, dtype=float)
        if isinstance(self.prior, Number):
            return np.full(n_classes, float(self.prior), dtype=float)
        alpha = np.asarray(self.prior, dtype=float)
        if alpha.ndim != 1 or len(alpha) != n_classes:
            raise ValueError(f"wrong shape for prior; expected {n_classes} values, found shape {alpha.shape}")
        return alpha

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return one normalised contingency table per posterior draw (true classes on rows)."""
        yhat = np.argmax(posteriors, axis=-1)
        posterior_count = qp.functional.counts_from_labels(yhat, self.classes)

        if self.c_prior in HIERARCHICAL_PRIORS:
            try:
                samples = self.sample_posterior(posterior_count, alpha=self._alpha())
            except RuntimeError as error:
                # numpyro raises a RuntimeError when it finds no valid initial parameters
                if "initial parameters" not in str(error):
                    raise
                self._last_nuts = self._last_summaries = "failed"
                n_classes = len(self.classes)
                return np.full((self.num_samples, n_classes, n_classes), np.nan)
        else:
            samples = self.sample_posterior(posterior_count, alpha=self._alpha())

        # compute P(Y,C) for all samples from P(Y) and P(C|Y)
        p_y_and_c_test = jnp.einsum("sy,syc->syc", samples[P_TEST_Y], samples[P_C_COND_Y])
        cts = np.array(jax.device_get(p_y_and_c_test))
        if self.estimand == "realised":
            cts = realised_tables(cts, posterior_count, np.random.default_rng(self.random_state))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)
