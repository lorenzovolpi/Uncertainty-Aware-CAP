r"""BayCAP-KDEy: BayCAP with the posterior of BayesKDEy, with a fixed rate matrix $\hat C$ (E3.1) or a random $C$ (E3.2).

``c_prior="fixed"`` gives the methods ``BayCAP-KDEy-<s>`` and ``BayCAP-KDEy-<s>-real`` of E3.1 of
``docs/ext3_posterior_propagation.md`` (work-plan card WP-8): the posterior draws of $\pi$ of BayesKDEy are pushed
through the rate matrix $\hat C$ of $h$ on $V$. Work-plan card WP-13 moved them from
``methods/baycap_fixed.py:BayCAPFixed`` (``quantifier="kdey"``) to this class, with the same draws
(``tests/test_baycap_split.py``). The NUTS model of BayesKDEy is copied from QuaPy's ``quapy.method._bayesian``
(``BayesianKDEy``, commit ``31f36f6`` of branch ``devel``), since QuaPy builds the class only when ``pystan`` is
installed (decision of the user, 2026-11-03); the copy gives the draws of QuaPy's ``sample_from_posterior``
(``tests/test_baycap_fixed.py``). The kernel density estimates are QuaPy's (``quapy.method._kdey.KDEBase``).

``c_prior="hierarchical-f"`` gives the methods ``BayCAP-KDEy-Hf-<s>-real`` of E3.2 (work-plan card WP-13, §T.10–§T.13):
each class of $V$ is split into cells by the prediction of $h$, the class density of BayesKDEy becomes the mixture
$f_j=\sum_kC_{kj}\hat\phi_{kj}$ of the kernel densities of its cells, and $(\pi,C)$ are sampled jointly by NUTS, with the
Hf prior on $C$ of ``methods/baycap.py`` (centred shares) and the validation counts observed as multinomials, as in
``BayCAP``. This model is the project's own, not a copy of QuaPy's.
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from cap.models.confidence import P_C_COND_Y, P_TEST_Y
from jax.scipy.special import logsumexp
from quapy.method._bayesian import _resolve_dirichlet_prior
from quapy.method._kdey import KDEBase

from uncawe.methods.baycap import (
    H_KAPPA,
    H_M,
    H_NU,
    H_RECALLS,
    H_RHO,
    H_RHO_KAPPA,
    HIERARCHICAL_SUMMARIES,
    KAPPA_SUMMARIES,
    failed_summaries,
    hierarchical_rate_sites,
    hierarchical_summaries,
    realised_tables,
)
from uncawe.methods.baycap_quantifier import PREV, QuantifierBayCAP, _validate_prevalence_samples

# the prior on the columns of C: "fixed" holds C at the rates of h on V (E3.1), "hierarchical-f" samples C with the Hf
# prior of BayCAP (E3.2)
KDEY_C_PRIORS = ("fixed", "hierarchical-f")
# source of the density of a cell (§T.12): its own validation outputs; those of all the validation instances that h
# predicts as the class of the cell (the cell is empty); those of the true class of the cell (h never predicts the
# class of the cell on V)
CELL_OWN, CELL_PREDICTED, CELL_CLASS = 0, 1, 2
# per-bag summaries of the cells added by "hierarchical-f" (E3.2): the posterior mean and standard deviation of C, in
# the cap-ml layout (true classes on rows), flattened row by row; the label-free mismatch ||n^t/N^Q - mean(C) mean(pi)||_1
# (§T.8); the numbers of empty cells of V and, among them, of cells that take the density of their true class; the
# top-label ECE of the input classifier on its validation outputs
CELL_SUMMARIES = ("c_mean", "c_sd", "mismatch", "n_empty_cells", "n_class_density_cells", "val_ece")
KDEY_HF_SUMMARIES = HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES + CELL_SUMMARIES


def kdey_model(test_log_densities, alpha, temperature: float = 1.0) -> None:
    r"""The NUTS model of BayesKDEy (QuaPy's ``BayesianKDEy._numpyro_model``).

    $\pi\sim\operatorname{Dir}(\alpha)$, and the log-likelihood $\sum_n\ln\sum_j\pi_jp_j(x_n)$ of the outputs $x_n$ of
    the bag, with $p_j$ the kernel density of class $\omega_j$ on $V$ (``test_log_densities[j, n]``$=\ln p_j(x_n)$),
    divided by the temperature $\tau$.
    """
    prev = numpyro.sample(PREV, dist.Dirichlet(jnp.asarray(alpha)))
    log_likelihood = jnp.sum(logsumexp(jnp.log(prev)[:, None] + test_log_densities, axis=0))
    numpyro.factor("loglik", (1.0 / temperature) * log_likelihood)


def kdey_hf_loglik(prev, rates, log_cells, pred=None):
    r"""The log-likelihood of the target bag of E3.2 (§T.10), for one $(\pi,C)$.

    With ``pred=None`` (input classifier ``h``), the likelihood of the outputs alone,
    $\sum_u\ln\sum_j\pi_j\sum_kC_{kj}\hat\phi_{kj}(z_u)$, with ``log_cells[u, j, k]``$=\ln\hat\phi_{kj}(z_u)$. With
    ``pred`` the predictions $\hat y_u$ of $h$ on the bag (input classifier ``mlp``), the joint likelihood of the outputs
    and the predictions, $\sum_u\ln\sum_j\pi_jC_{\hat y_uj}\hat\phi_{\hat y_uj}(z_u)$, with
    ``log_cells[u, j]``$=\ln\hat\phi_{\hat y_uj}(z_u)$. Both are computed by log-sum-exp from the logarithms.

    :param prev: $\pi$, shape ``(K,)``.
    :param rates: $C$ in the ``cap-ml`` layout, ``rates[j, k]``$=C_{kj}$.
    :param log_cells: the log-densities of the cells at the target outputs, shape ``(N, K, K)`` or ``(N, K)``.
    :param pred: the indices of the predictions of $h$ on the bag, shape ``(N,)``, or ``None``.
    """
    # log_w[j, k] = ln(pi_j C_kj)
    log_w = jnp.log(prev)[:, None] + jnp.log(rates)
    if pred is None:
        return jnp.sum(logsumexp(log_w[None, :, :] + log_cells, axis=(1, 2)))
    return jnp.sum(logsumexp(log_w[:, pred].T + log_cells, axis=1))


def kdey_hf_model(log_cells, val_counts, alpha, pred=None, temperature: float = 1.0) -> None:
    r"""The NUTS model of E3.2 (§T.10, §T.11): BayesKDEy with a random $C$ and the Hf prior.

    $\pi\sim\operatorname{Dir}(\alpha)$; $C$ from the Hf prior with centred shares
    (:func:`uncawe.methods.baycap.hierarchical_rate_sites`, the sites of ``BayCAP`` with ``c_prior="hierarchical-f"``);
    the validation counts of each true class $\operatorname{Mult}(n^s_j,C_{\cdot j})$, observed, as in ``BayCAP.model``;
    the log-likelihood of the bag (:func:`kdey_hf_loglik`) divided by the temperature $\tau$.

    :param log_cells: see :func:`kdey_hf_loglik`.
    :param val_counts: the validation counts in the ``cap-ml`` layout, ``val_counts[j, k]``$=f_{kj}$.
    :param alpha: the Dirichlet parameter of the prior on $\pi$.
    :param pred: see :func:`kdey_hf_loglik`.
    :param temperature: $\tau$.
    """
    n_classes = val_counts.shape[0]
    prev = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.asarray(alpha)))
    p_c_cond_y = hierarchical_rate_sites(n_classes, light=True, shares="centred")
    with numpyro.plate("plate", n_classes):
        numpyro.sample("F_yc", dist.Multinomial(val_counts.sum(axis=1), p_c_cond_y), obs=val_counts)
    numpyro.factor("loglik", (1.0 / temperature) * kdey_hf_loglik(prev, p_c_cond_y, log_cells, pred))


class _KernelDensities(KDEBase):
    """QuaPy's kernel density estimates of the outputs within each class, as fitted by ``BayesianKDEy``."""

    def __init__(self, kernel: str, bandwidth, shrinkage: float):
        self.kernel = KDEBase._check_kernel(kernel)
        self.bandwidth = KDEBase._check_bandwidth(bandwidth, self.kernel)
        if not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be in [0,1)")
        if self.kernel == "gaussian" and shrinkage != 0:
            raise ValueError("shrinkage is only supported for Aitchison/ILR kernels")
        self.shrinkage = float(shrinkage)

    def fit(self, outputs: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> Self:
        self.mix_densities = self.get_mixture_components(outputs, labels, classes, self.bandwidth, self.kernel)
        return self

    def log_densities(self, outputs: np.ndarray) -> np.ndarray:
        """``[j, n]``: the log-density of class $j$ at the outputs of instance $n$."""
        return np.asarray([self.pdf(kde, outputs, self.kernel, log_densities=True) for kde in self.mix_densities])

    def density(self, outputs: np.ndarray):
        """A kernel density with the kernel and bandwidth of the class densities, fitted on ``outputs``."""
        return self.get_kde_function(outputs, self.bandwidth, self.kernel)

    def log_density(self, kde, outputs: np.ndarray) -> np.ndarray:
        """The log-density of ``kde`` at ``outputs``."""
        return self.pdf(kde, outputs, self.kernel, log_densities=True)


class BayCAPKDEy(QuantifierBayCAP):
    r"""BayCAP with the posterior of BayesKDEy, with a fixed $\hat C$ (E3.1) or a random $C$ with the Hf prior (E3.2).

    BayesKDEy estimates the densities of the outputs of the input classifier within each class from $V$ by kernel
    density estimation (``kernel``, ``bandwidth``, ``shrinkage``) and puts a Dirichlet prior on $\pi$. The input
    classifier, the estimands, the seeds and the NUTS diagnostics are those of
    :class:`uncawe.methods.baycap_quantifier.QuantifierBayCAP`.

    - ``c_prior="fixed"`` (E3.1): NUTS samples $\pi$ alone (:func:`kdey_model`), and each draw is pushed through the
      rate matrix $\hat C$ of $h$ on $V$. After each ``predict_ct_range`` call, :meth:`last_posterior_summaries` returns
      ``FIXED_SUMMARIES``.
    - ``c_prior="hierarchical-f"`` (E3.2, §T.10–§T.13): ``fit`` splits each class $\omega_j$ of $V$ into the cells
      $V_{kj}$ of the instances that $h$ predicts as $\omega_k$ and fits one kernel density $\hat\phi_{kj}$ per cell on
      the validation outputs of the input classifier (cross-fitted for ``mlp``). An empty cell takes the density of all
      the validation instances that $h$ predicts as $\omega_k$, or, if $h$ never predicts $\omega_k$ on $V$, the class
      density $\hat f_j$ (§T.12). NUTS samples $(\pi,C)$ jointly (:func:`kdey_hf_model`): $C$ has the Hf prior of
      ``BayCAP`` with centred shares and the validation counts as data, and the bag enters through
      :func:`kdey_hf_loglik`, with the outputs alone for ``h`` and the outputs and the predictions of $h$ for ``mlp``.
      Each draw becomes the table $C^{(r)}\operatorname{diag}(\pi^{(r)})$, then the estimand applies. The NUTS
      diagnostics cover $\pi$, $C$, $m$, $\rho$ and $1/(1+\kappa)$; :meth:`last_posterior_summaries` returns
      ``KDEY_HF_SUMMARIES``. If NUTS cannot start, the bag gets tables of NaN, NaN diagnostics and ``nuts_failed``
      equal to $1$, as in ``BayesCAP-Hf-real``.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param estimand: ``"population"`` or ``"realised"``.
    :param c_prior: ``"fixed"`` or ``"hierarchical-f"``.
    :param kernel: kernel of BayesKDEy (``"gaussian"``, ``"aitchison"`` or ``"ilr"``).
    :param bandwidth: bandwidth of BayesKDEy, a positive number, ``"scott"`` or ``"silverman"``.
    :param shrinkage: shrinkage of BayesKDEy, in $[0,1)$, $0$ with the Gaussian kernel.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the bag.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS, of the MLP and of the multinomial draws; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        input_classifier: Literal["h", "mlp"] = "mlp",
        estimand: Literal["population", "realised"] = "population",
        c_prior: Literal["fixed", "hierarchical-f"] = "fixed",
        kernel: Literal["gaussian", "aitchison", "ilr"] = "gaussian",
        bandwidth: float | Literal["scott", "silverman"] = 0.1,
        shrinkage: float = 0.0,
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        if c_prior not in KDEY_C_PRIORS:
            raise ValueError(f"c_prior must be one of {KDEY_C_PRIORS}; found {c_prior!r}")
        super().__init__(
            acc_fn,
            input_classifier=input_classifier,
            estimand=estimand,
            num_warmup=num_warmup,
            num_samples=num_samples,
            temperature=temperature,
            prior=prior,
            random_state=random_state,
        )
        self.c_prior = c_prior
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.shrinkage = shrinkage

    def _fit_quantifier(self, val_outputs: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> None:
        self.kde_ = _KernelDensities(self.kernel, self.bandwidth, self.shrinkage)
        self.kde_.fit(val_outputs, labels, self.quantifier_classes_)
        if self.c_prior == "hierarchical-f":
            self._fit_cells(val_outputs, labels, predicted)

    def _fit_cells(self, val_outputs: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> None:
        r"""Fit the densities of the cells (§T.10) and choose those of the empty cells (§T.12).

        ``cell_source_[j, k]`` is ``CELL_OWN``, ``CELL_PREDICTED`` or ``CELL_CLASS``; ``_cell_key[j][k]`` names the
        density of cell $(k,j)$ in ``_densities``, where each distinct density is fitted once.
        """
        classes = self.classes_
        n_classes = len(classes)
        self.cell_source_ = np.full((n_classes, n_classes), CELL_OWN, dtype=int)
        self._densities, self._cell_key = {}, [[None] * n_classes for _ in range(n_classes)]
        for k in range(n_classes):
            predicted_k = predicted == classes[k]
            for j in range(n_classes):
                in_cell = predicted_k & (labels == classes[j])
                if in_cell.any():
                    key = ("cell", j, k)
                    if key not in self._densities:
                        self._densities[key] = self.kde_.density(val_outputs[in_cell])
                elif predicted_k.any():
                    key = ("predicted", k)
                    self.cell_source_[j, k] = CELL_PREDICTED
                    if key not in self._densities:
                        self._densities[key] = self.kde_.density(val_outputs[predicted_k])
                else:
                    key = ("class", j)
                    self.cell_source_[j, k] = CELL_CLASS
                    if key not in self._densities:
                        self._densities[key] = self.kde_.mix_densities[self._quantifier_to_rate_order[j]]
                self._cell_key[j][k] = key

    def cell_log_densities(self, outputs: np.ndarray, pred: np.ndarray | None = None) -> np.ndarray:
        r"""The log-densities of the cells at the outputs of a bag (§T.10), computed before sampling.

        :param outputs: the outputs of the input classifier on the bag, shape ``(N, K)``.
        :param pred: ``None`` for all cells, ``[u, j, k]``$=\ln\hat\phi_{kj}(z_u)$, shape ``(N, K, K)``; or the indices
            of the predictions of $h$ on the bag, for ``[u, j]``$=\ln\hat\phi_{\hat y_uj}(z_u)$, shape ``(N, K)``.
        """
        n_classes = len(self.classes_)
        if pred is None:
            cache = {}
            out = np.empty((len(outputs), n_classes, n_classes))
            for j in range(n_classes):
                for k in range(n_classes):
                    key = self._cell_key[j][k]
                    if key not in cache:
                        cache[key] = self.kde_.log_density(self._densities[key], outputs)
                    out[:, j, k] = cache[key]
            return out
        out = np.empty((len(outputs), n_classes))
        for k in np.unique(pred):
            rows = pred == k
            cache = {}
            for j in range(n_classes):
                key = self._cell_key[j][k]
                if key not in cache:
                    cache[key] = self.kde_.log_density(self._densities[key], outputs[rows])
                out[rows, j] = cache[key]
        return out

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model of BayesKDEy, its arguments and the initial parameters, as QuaPy's ``sample_from_posterior``."""
        alpha = _resolve_dirichlet_prior(self.prior, outputs.shape[1])
        args = {"test_log_densities": self.kde_.log_densities(outputs), "alpha": alpha}
        return kdey_model, args | {"temperature": self.temperature}, None

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        return np.asarray(jax.device_get(grouped[PREV]))

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalised contingency table per posterior draw (true classes on rows).

        ``posteriors`` are the outputs of $h$ on the bag: they give $n^t$, the predictions $\hat y_u$ and, with
        ``input_classifier="h"``, the outputs the quantifier reads; with ``"mlp"`` it reads the outputs of its MLP on
        ``X``.
        """
        if self.c_prior == "fixed":
            return super().predict_ct_range(X, posteriors)
        self._check_is_fitted()
        posteriors = np.asarray(posteriors, dtype=float)
        n_classes = len(self.classes_)
        pred = np.argmax(posteriors, axis=1)
        pred_counts = np.bincount(pred, minlength=n_classes)
        cell_pred = None if self.input_classifier == "h" else pred
        model_args = {
            "log_cells": self.cell_log_densities(self.outputs(X, posteriors), cell_pred),
            "val_counts": self.val_counts_,
            "alpha": _resolve_dirichlet_prior(self.prior, n_classes),
            "pred": cell_pred,
            "temperature": self.temperature,
        }
        try:
            mcmc = self._run_nuts(kdey_hf_model, model_args, None)
        except RuntimeError as error:
            # numpyro raises a RuntimeError when it finds no valid initial parameters
            if "initial parameters" not in str(error):
                raise
            self._last_nuts = "failed"
            self._last_summaries = failed_summaries(n_classes, with_kappa=True) | self._cell_summaries(None, None, None)
            return np.full((self.num_samples, n_classes, n_classes), np.nan)

        grouped = mcmc.get_samples(group_by_chain=True)
        sites = (P_TEST_Y, P_C_COND_Y, H_M, H_RHO) + ((H_RHO_KAPPA,) if n_classes > 2 else ())
        self._last_nuts = (
            {name: np.asarray(jax.device_get(grouped[name])) for name in sites},
            np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])),
        )
        samples = {name: np.asarray(jax.device_get(value)) for name, value in mcmc.get_samples().items()}
        prevalences, rates = samples[P_TEST_Y].astype(float), samples[P_C_COND_Y].astype(float)
        _validate_prevalence_samples(prevalences, n_classes)
        summary_sites = (H_M, H_NU, H_RECALLS, P_TEST_Y) + ((H_KAPPA,) if H_KAPPA in samples else ())
        self._last_summaries = hierarchical_summaries(
            {name: samples[name] for name in summary_sites},
            float(self.val_counts_.sum(axis=1).min()),
            with_kappa=True,
        ) | self._cell_summaries(prevalences, rates, pred_counts)

        cts = prevalences[:, :, None] * rates
        if self.estimand == "realised":
            cts = realised_tables(cts, pred_counts, np.random.default_rng(self.random_state))
        return cts

    def _cell_summaries(self, prevalences, rates, pred_counts) -> dict:
        """The summaries ``CELL_SUMMARIES`` of a bag; NaN for the posterior ones when NUTS failed (``rates=None``)."""
        n_classes = len(self.classes_)
        counts = {
            "n_empty_cells": int(np.sum(self.val_counts_ == 0)),
            "n_class_density_cells": int(np.sum(self.cell_source_ == CELL_CLASS)),
            "val_ece": self.val_ece_,
        }
        if rates is None:
            nan = np.full(n_classes * n_classes, np.nan)
            return {"c_mean": nan, "c_sd": nan.copy(), "mismatch": np.nan} | counts
        c_mean = rates.mean(axis=0)
        shares = np.asarray(pred_counts, dtype=float) / np.sum(pred_counts)
        return {
            "c_mean": c_mean.ravel(),
            "c_sd": rates.std(axis=0).ravel(),
            # the predicted share of class k under the posterior means: sum_j mean(pi)_j mean(C)_kj
            "mismatch": float(np.abs(shares - prevalences.mean(axis=0) @ c_mean).sum()),
        } | counts


__all__ = [
    "CELL_CLASS",
    "CELL_OWN",
    "CELL_PREDICTED",
    "CELL_SUMMARIES",
    "KDEY_C_PRIORS",
    "KDEY_HF_SUMMARIES",
    "BayCAPKDEy",
    "kdey_hf_loglik",
    "kdey_hf_model",
    "kdey_model",
]
