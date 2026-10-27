"""BayCAP with a population or a realised-table estimand.

This module implements the method of work-plan card WP-1, motivated by the
realised-table posterior predictive of ``docs/ext2_baycap.md``. The
posterior model, the priors and the NUTS sampler are copied from ``cap-ml``
``cap.models.confidence.ACC_BayesCAP`` (commit ``bb20aa5`` of branch
``devel``); only the pushforward of the posterior draws to contingency tables
is extended. Work-plan card WP-5 adds the empirical-Bayes prior on the
columns of $C$ (``c_prior``, §T.9 of ``docs/ext2_baycap.md``), and work-plan
card WP-6 the NUTS diagnostics of each ``predict_ct_range`` call
(``last_nuts_diagnostics``).
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
C_PRIORS = ("uniform", "empirical-bayes")

# per-bag NUTS diagnostics stored by the experiments (``nuts_diagnostics``)
NUTS_DIAGNOSTICS = ("nuts_divergences", "nuts_max_rhat", "nuts_min_ess")

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

    After each ``predict_ct_range`` call, :meth:`last_nuts_diagnostics` returns
    the number of divergent transitions, the largest split-$\hat R$ and the
    smallest effective sample size of that NUTS run, over the entries of
    $\pi^Q$ and $C$ (:func:`nuts_diagnostics`). Recording them does not
    change the draws.

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
    :param c_prior: prior on the columns of $C$, ``"uniform"`` or
        ``"empirical-bayes"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
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
        c_prior: Literal["uniform", "empirical-bayes"] = "uniform",
        num_warmup: int = 500,
        num_samples: int = 1000,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ):
        if estimand not in ESTIMANDS:
            raise ValueError(f"estimand must be one of {ESTIMANDS}; found {estimand!r}")
        if c_prior not in C_PRIORS:
            raise ValueError(f"c_prior must be one of {C_PRIORS}; found {c_prior!r}")
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
        # row y of p_c_cond_y is the column C_{:y} of the paper, with concentrations c_alpha[y]
        p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.asarray(c_alpha, dtype=jnp.float32)))

        with numpyro.plate("plate", L):
            numpyro.sample("F_yc", dist.Multinomial(train_class_count, p_c_cond_y), obs=train_class_cond_count)

        p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
        numpyro.sample("N_c", dist.Multinomial(jnp.sum(pred_posterior_count), p_c), obs=pred_posterior_count)

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
        self._last_nuts = (
            {name: np.asarray(jax.device_get(grouped[name])) for name in (P_TEST_Y, P_C_COND_Y)},
            np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])),
        )
        return mcmc.get_samples()

    def last_nuts_diagnostics(self) -> dict[str, float]:
        """Return the NUTS diagnostics (:func:`nuts_diagnostics`) of the last ``sample_posterior`` call."""
        if getattr(self, "_last_nuts", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        return nuts_diagnostics(*self._last_nuts)

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

        samples = self.sample_posterior(posterior_count, alpha=self._alpha())

        # compute P(Y,C) for all samples from P(Y) and P(C|Y)
        p_y_and_c_test = jnp.einsum("sy,syc->syc", samples[P_TEST_Y], samples[P_C_COND_Y])
        cts = np.array(jax.device_get(p_y_and_c_test))
        if self.estimand == "realised":
            cts = realised_tables(cts, posterior_count, np.random.default_rng(self.random_state))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)
