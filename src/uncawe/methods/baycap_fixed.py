r"""BayCAP with a fixed rate matrix $\hat C$ and the prevalence posterior of a Bayesian quantifier.

This module implements the methods of E3.1 of ``docs/ext3_posterior_propagation.md`` (work-plan card WP-8). It
generalises the former ``methods/baycap_kdey.py:BayCAPKDEy`` (BayesKDEy on an MLP, population pushforward) along three
factors: the quantifier (BayesKDEy or BayesianMAPLS), the input classifier whose outputs the quantifier reads (the
classifier $h$ itself or a separate MLP trained on $V$), and the pushforward (population or realised table).

The NUTS models of the two quantifiers and their preparation are copied from QuaPy's ``quapy.method._bayesian``
(``BayesianKDEy`` and ``BayesianMAPLS``, commit ``31f36f6`` of branch ``devel``): QuaPy builds neither class unless
``pystan`` is installed, and ``pystan`` is not in the environment of the project (decision of the user, 2026-11-03).
The copies keep QuaPy's sites, priors, likelihoods, initialisation and seeds, so where ``pystan`` is installed they give
the draws of QuaPy's ``sample_from_posterior`` (``tests/test_baycap_fixed.py``). The kernel density estimates are
QuaPy's (``quapy.method._kdey.KDEBase``) and the EM start of BayesianMAPLS is QuaPy's ``mapls``; neither needs
``pystan``.
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
import quapy.functional as F
from cap.models.confidence import CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from jax.scipy.special import logsumexp
from numpyro.infer import MCMC, NUTS
from quapy.data import LabelledCollection
from quapy.method._bayesian import _resolve_dirichlet_prior, mapls
from quapy.method._kdey import KDEBase
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier

from uncawe.methods.baycap import ESTIMANDS, NUTS_DIAGNOSTICS, nuts_diagnostics, realised_tables

QUANTIFIERS = ("kdey", "emq")
INPUT_CLASSIFIERS = ("h", "mlp")
# number of folds of the cross-fitted validation outputs of the MLP (QuaPy's default ``val_split``); the folds run in
# the worker's process, so that a BayCAP worker keeps one core busy (docs/work_plan.md, Part V.1)
MLP_FOLDS = 5
# equal-width bins of the top-label expected calibration error
ECE_BINS = 15
# per-bag summaries stored by the experiments (``last_posterior_summaries``), vanilla accuracy: the posterior mean of
# pi (a vector of length K), the standard deviation sigma_p of the population accuracies of the draws, the bag term
# sigma_g of the realised pushforward (§T.8), the top-label ECE of the input classifier on its validation outputs
FIXED_SUMMARIES = ("prev_mean", "acc_sd", "sigma_g", "val_ece")
# sampled sites of the two models (QuaPy's names): pi for BayesKDEy, its ILR coordinates for BayesianMAPLS
PREV, ILR_Z = "prev", "z"


def top_label_ece(outputs: np.ndarray, labels: np.ndarray, classes: np.ndarray, n_bins: int = ECE_BINS) -> float:
    r"""Expected calibration error of the top-label confidence, with ``n_bins`` equal-width bins.

    Each instance has confidence $c_n=\max_k p_{nk}$ and is correct when the class of largest output is its label. Bin
    $b$ holds the instances with $c_n\in((b-1)/B,b/B]$; the ECE is $\sum_b\frac{|B_b|}{N}\lvert\mathrm{acc}_b-
    \bar c_b\rvert$, with $\mathrm{acc}_b$ the share of correct instances and $\bar c_b$ the mean confidence in bin $b$.

    :param outputs: the outputs, of shape ``(N, K)``, with columns in the order of ``classes``.
    :param labels: the true labels, of shape ``(N,)``.
    :param classes: the class labels of the columns of ``outputs``.
    """
    outputs = np.asarray(outputs, dtype=float)
    confidence = outputs.max(axis=1)
    correct = np.asarray(classes)[outputs.argmax(axis=1)] == np.asarray(labels)
    bins = np.clip(np.ceil(confidence * n_bins).astype(int) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def accuracy_moments(prevalences: np.ndarray, rates: np.ndarray, pred_counts: np.ndarray) -> tuple[float, float]:
    r"""Return $(\sigma_p,\bar\sigma_g)$ of the vanilla accuracy for the draws of $\pi$ and a fixed $\hat C$ (§T.5, §T.8).

    $\sigma_p$ is the standard deviation of the population accuracies $a^{(r)}=\sum_j\hat C_{jj}\pi^{(r)}_j$ over the
    draws. $\bar\sigma_g=\sqrt{\frac1R\sum_r\sigma_g^2(\pi^{(r)})}$, with
    $\sigma_g^2(\pi)=\frac{1}{(N^Q)^2}\sum_in^t_i\rho_{ii}(\pi)(1-\rho_{ii}(\pi))$ the variance of the realised accuracy
    given the draw, and $\rho_{ii}(\pi)=\hat C_{ii}\pi_i/\sum_l\hat C_{il}\pi_l$ the precision of class $\omega_i$. As
    in :func:`uncawe.methods.baycap.realised_tables`, a predicted class that a draw makes impossible gets $\rho_{ii}=1/K$.

    :param prevalences: draws of $\pi$, of shape ``(R, K)``.
    :param rates: the rate matrix in the ``cap-ml`` layout, row $j$ the column $\hat C_{:j}$ of the paper.
    :param pred_counts: $n^t$, the number of target instances predicted as each class.
    """
    prevalences = np.asarray(prevalences, dtype=float)
    rates = np.asarray(rates, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=float)
    n_classes = rates.shape[0]
    accuracies = prevalences @ np.diag(rates)
    # predicted[r, i] = sum_j pi_j C_ij, the share of the population predicted as class i under draw r
    predicted = prevalences @ rates
    correct = prevalences * np.diag(rates)
    precision = np.divide(correct, predicted, out=np.full_like(correct, 1.0 / n_classes), where=predicted > 0)
    sigma_g2 = (precision * (1 - precision)) @ pred_counts / pred_counts.sum() ** 2
    return float(np.std(accuracies)), float(np.sqrt(np.mean(sigma_g2)))


def ilr_basis(n_classes: int) -> np.ndarray:
    """The orthonormal Helmert basis of QuaPy's ILR transformation (``quapy.functional.ILRtransformation.get_V``)."""
    return F.ILRtransformation().get_V(n_classes)


def ilr(prevalence) -> jnp.ndarray:
    """ILR coordinates of a prevalence vector, as QuaPy's ``_JaxILRTransformation.__call__`` (smoothed by $10^{-12}$)."""
    smoothed = jnp.asarray(qp.error.smooth(np.asarray(prevalence), F.CompositionalTransformation.EPSILON))
    return jnp.log(smoothed) @ jnp.asarray(ilr_basis(smoothed.shape[-1])).T


def ilr_inverse(z) -> jnp.ndarray:
    """Prevalence vectors of ILR coordinates, as QuaPy's ``_JaxILRTransformation.inverse``."""
    z = jnp.asarray(z)
    p = jnp.exp(z @ jnp.asarray(ilr_basis(z.shape[-1] + 1)))
    return p / jnp.sum(p, axis=-1, keepdims=True)


def kdey_model(test_log_densities, alpha, temperature: float = 1.0) -> None:
    r"""The NUTS model of BayesKDEy (QuaPy's ``BayesianKDEy._numpyro_model``).

    $\pi\sim\operatorname{Dir}(\alpha)$, and the log-likelihood $\sum_n\ln\sum_j\pi_jp_j(x_n)$ of the outputs $x_n$ of
    the bag, with $p_j$ the kernel density of class $\omega_j$ on $V$ (``test_log_densities[j, n]``$=\ln p_j(x_n)$),
    divided by the temperature $\tau$.
    """
    prev = numpyro.sample(PREV, dist.Dirichlet(jnp.asarray(alpha)))
    log_likelihood = jnp.sum(logsumexp(jnp.log(prev)[:, None] + test_log_densities, axis=0))
    numpyro.factor("loglik", (1.0 / temperature) * log_likelihood)


def emq_model(test_posteriors, alpha, train_prev, temperature: float = 1.0) -> None:
    r"""The NUTS model of BayesianMAPLS (QuaPy's ``BayesianMAPLS._numpyro_model``).

    The ILR coordinates $z$ of $\pi$ are $\mathcal N(\mathbf 0,I)$, the density $\operatorname{Dir}(\alpha)$ of $\pi$ is
    added as a factor (with no Jacobian, so the effective prior on $\pi$ is not the Dirichlet, §T.4), and the
    log-likelihood $\sum_n\ln\sum_jp_{nj}\pi_j/\pi^P_j$ of the outputs $p_{nj}$ of the bag, with $\pi^P$ the source
    prevalence, is divided by the temperature $\tau$.
    """
    test_posteriors = jnp.asarray(test_posteriors)
    n_classes = test_posteriors.shape[1]
    z = numpyro.sample(ILR_Z, dist.Normal(jnp.zeros(n_classes - 1), 1.0))
    prev = ilr_inverse(z)
    numpyro.factor("dirichlet_prior", dist.Dirichlet(jnp.asarray(alpha)).log_prob(prev))
    log_w = jnp.log(prev) - jnp.log(jnp.asarray(train_prev))
    log_likelihood = jnp.sum(logsumexp(jnp.log(test_posteriors) + log_w, axis=-1))
    numpyro.factor("likelihood", (1.0 / temperature) * log_likelihood)


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


class BayCAPFixed(CAPContingencyTable, CTCAPWithConfidence):
    r"""Push the joint posterior draws of a Bayesian quantifier through the fixed rate matrix of $h$ on $V$.

    ``fit`` measures $\hat C_{ij}=f_{ij}/n^s_j$ from the predictions of $h$ on $V$ (no prior, not sampled) and fits the
    quantifier. ``predict_ct_range`` samples the posterior of $\pi^Q$ with NUTS and turns each draw into a table.

    - ``quantifier="kdey"``: BayesKDEy, which estimates the densities of the outputs within each class from $V$ by
      kernel density estimation (``kernel``, ``bandwidth``, ``shrinkage``) and puts a Dirichlet prior on $\pi$
      (:func:`kdey_model`).
    - ``quantifier="emq"``: BayesianMAPLS, which reads the outputs as calibrated probabilities (:func:`emq_model`). Its
      source prevalence $\pi^P$ is that of the labels of $V$ (QuaPy's ``exact_train_prev=True``), its chain starts at
      the EM estimate of QuaPy's ``mapls`` (``mapls_chain_init=True``), and its effective prior on $\pi$ is
      logistic-normal even with ``prior="uniform"`` (§T.4).
    - ``input_classifier="h"``: the quantifier reads the outputs of $h$, the ``posteriors`` arguments of ``fit`` (on
      $V$, out of sample since $h$ was trained on $L$) and of ``predict_ct_range`` (on the target bag); no classifier is
      trained, and the columns of the outputs follow ``val.classes_``.
    - ``input_classifier="mlp"``: the quantifier reads the outputs of a separate ``MLPClassifier()`` (default settings,
      ``random_state``) trained on $V$; its validation outputs are cross-fitted (``MLP_FOLDS`` folds, as QuaPy's
      ``fit``), its target outputs come from the MLP trained on all of $V$.

    The estimand decides how each draw $\pi^{(r)}$ becomes a contingency table:

    - ``"population"``: the population table $\hat C\operatorname{diag}(\pi^{(r)})$. It represents the uncertainty of
      $\pi^Q$ given the target bag; not the error of the quantifier fitted on a finite $V$, nor that of $\hat C$, nor
      the finite-sample fluctuation of the bag's table.
    - ``"realised"``: :func:`uncawe.methods.baycap.realised_tables` applied to the population tables, with $n^t$ the
      arg-max predictions of $h$ on the bag. It adds the finite-sample fluctuation of the realised table of the target
      sample, which is what the experiments score (``true_cts``).

    The two estimands share the prevalence draws for the same ``random_state``, which seeds NUTS, the MLP and the
    multinomial draws of ``"realised"``. Tables follow the ``cap-ml`` layout (true classes on rows), so the paper's
    $\hat C\operatorname{diag}(\pi)$ is returned transposed, as $\operatorname{diag}(\pi)\hat C^\top$.

    After each ``predict_ct_range`` call, :meth:`last_nuts_diagnostics` returns the divergent transitions, the largest
    split-$\hat R$ and the smallest effective sample size of $\pi$ (:func:`uncawe.methods.baycap.nuts_diagnostics`),
    and :meth:`last_posterior_summaries` the summaries ``FIXED_SUMMARIES`` of vanilla accuracy.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param quantifier: ``"kdey"`` or ``"emq"``.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param estimand: ``"population"`` or ``"realised"``.
    :param kernel: kernel of BayesKDEy (``"gaussian"``, ``"aitchison"`` or ``"ilr"``).
    :param bandwidth: bandwidth of BayesKDEy, a positive number, ``"scott"`` or ``"silverman"``.
    :param shrinkage: shrinkage of BayesKDEy, in $[0,1)$, $0$ with the Gaussian kernel.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the quantifier.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS, of the MLP and of the multinomial draws; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        quantifier: Literal["kdey", "emq"] = "kdey",
        input_classifier: Literal["h", "mlp"] = "mlp",
        estimand: Literal["population", "realised"] = "population",
        kernel: Literal["gaussian", "aitchison", "ilr"] = "gaussian",
        bandwidth: float | Literal["scott", "silverman"] = 0.1,
        shrinkage: float = 0.0,
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        if quantifier not in QUANTIFIERS:
            raise ValueError(f"quantifier must be one of {QUANTIFIERS}; found {quantifier!r}")
        if input_classifier not in INPUT_CLASSIFIERS:
            raise ValueError(f"input_classifier must be one of {INPUT_CLASSIFIERS}; found {input_classifier!r}")
        if estimand not in ESTIMANDS:
            raise ValueError(f"estimand must be one of {ESTIMANDS}; found {estimand!r}")
        if isinstance(prior, str) and prior != "uniform":
            raise ValueError(f"prior must be 'uniform', a positive scalar or a vector; found {prior!r}")
        if not temperature > 0:
            raise ValueError(f"temperature must be positive; found {temperature!r}")
        CAPContingencyTable.__init__(self, acc_fn)
        self.quantifier = quantifier
        self.input_classifier = input_classifier
        self.estimand = estimand
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.shrinkage = shrinkage
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.prior = prior
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        """Measure the fixed rates of $h$ on $V$ and fit the quantifier on the outputs of the input classifier."""
        posteriors = np.asarray(posteriors, dtype=float)
        classes = np.asarray(val.classes_)
        _validate_outputs(posteriors, len(val), len(classes))
        self.classes_ = classes
        predicted = classes[np.argmax(posteriors, axis=1)]
        # cap-ml layout: row j is the column C_{:j} of the paper, the rates of the instances of class j
        self.rate_matrix_ = confusion_matrix(val.y, predicted, labels=classes, normalize="true")

        labels = np.asarray(val.y)
        if self.input_classifier == "h":
            self.mlp_ = None
            self.quantifier_classes_ = classes
            val_outputs = posteriors
        else:
            # QuaPy's AggregativeQuantifier.classifier_fit_predict with val_split=5: cross-fitted outputs, then the
            # MLP trained on all of V
            self.mlp_ = MLPClassifier(random_state=self.random_state)
            val_outputs = cross_val_predict(self.mlp_, val.X, labels, cv=MLP_FOLDS, n_jobs=1, method="predict_proba")
            self.mlp_.fit(val.X, labels)
            self.quantifier_classes_ = np.asarray(self.mlp_.classes_)
        self._quantifier_to_rate_order = _class_order(self.quantifier_classes_, classes)
        self.val_outputs_ = val_outputs
        self.val_ece_ = top_label_ece(val_outputs, labels, self.quantifier_classes_)

        if self.quantifier == "kdey":
            self.kde_ = _KernelDensities(self.kernel, self.bandwidth, self.shrinkage)
            self.kde_.fit(val_outputs, labels, self.quantifier_classes_)
        else:
            # BayesianMAPLS.aggregation_fit with exact_train_prev=True
            self.train_prevalence_ = F.prevalence_from_labels(labels, classes=self.quantifier_classes_)
        return self

    def outputs(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """The outputs of the input classifier on a bag: ``posteriors`` for $h$, those of the MLP on ``X`` otherwise."""
        if self.input_classifier == "h":
            return np.asarray(posteriors, dtype=float)
        return self.mlp_.predict_proba(X)

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model, its arguments and the initial parameters, as the ``sample_from_posterior`` of the quantifier
        in QuaPy."""
        n_classes = outputs.shape[1]
        alpha = _resolve_dirichlet_prior(self.prior, n_classes)
        if self.quantifier == "kdey":
            args = {"test_log_densities": self.kde_.log_densities(outputs), "alpha": alpha}
            return kdey_model, args | {"temperature": self.temperature}, None
        map_estimate = mapls(self.val_outputs_, test_probs=outputs, pz=self.train_prevalence_, return_lambda=True)[0]
        args = {"test_posteriors": outputs, "alpha": alpha, "train_prev": self.train_prevalence_}
        return emq_model, args | {"temperature": self.temperature}, {ILR_Z: ilr(map_estimate)}

    def sample_prevalences(self, outputs: np.ndarray) -> np.ndarray:
        r"""Sample the posterior of $\pi$ given the outputs of the input classifier on the bag, recording the divergent
        transitions.

        :return: the draws of $\pi$, of shape ``(num_samples, K)``, columns in the order of ``quantifier_classes_``.
        """
        model, model_args, init_params = self._model(outputs)
        mcmc = MCMC(
            NUTS(model), num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=1, progress_bar=False
        )
        mcmc.run(
            jax.random.PRNGKey(self.random_state), init_params=init_params, extra_fields=("diverging",), **model_args
        )
        grouped = mcmc.get_samples(group_by_chain=True)
        if self.quantifier == "kdey":
            prev = np.asarray(jax.device_get(grouped[PREV]))
        else:
            prev = np.asarray(ilr_inverse(grouped[ILR_Z]))
        self._last_nuts = ({PREV: prev}, np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])))
        return prev[0]

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalised contingency table per posterior draw (true classes on rows).

        ``posteriors`` are the outputs of $h$ on the bag: they give $n^t$ and, with ``input_classifier="h"``, the
        outputs the quantifier reads; with ``"mlp"`` the quantifier reads the outputs of its MLP on ``X``.
        """
        self._check_is_fitted()
        posteriors = np.asarray(posteriors, dtype=float)
        prevalences = self.sample_prevalences(self.outputs(X, posteriors))[:, self._quantifier_to_rate_order]
        _validate_prevalence_samples(prevalences, len(self.classes_))

        pred_counts = np.bincount(np.argmax(posteriors, axis=1), minlength=len(self.classes_))
        acc_sd, sigma_g = accuracy_moments(prevalences, self.rate_matrix_, pred_counts)
        self._last_summaries = {
            "prev_mean": prevalences.mean(axis=0),
            "acc_sd": acc_sd,
            "sigma_g": sigma_g,
            "val_ece": self.val_ece_,
        }

        cts = prevalences[:, :, None] * self.rate_matrix_[None, :, :]
        if self.estimand == "realised":
            cts = realised_tables(cts, pred_counts, np.random.default_rng(self.random_state))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return the mean of the tables of :meth:`predict_ct_range`."""
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    def last_nuts_diagnostics(self) -> dict[str, float]:
        r"""Return the NUTS diagnostics of $\pi$ of the last ``predict_ct_range`` call (``NUTS_DIAGNOSTICS``)."""
        if getattr(self, "_last_nuts", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        return nuts_diagnostics(*self._last_nuts)

    def last_posterior_summaries(self) -> dict:
        """Return the summaries ``FIXED_SUMMARIES`` of the last ``predict_ct_range`` call."""
        if getattr(self, "_last_summaries", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        return dict(self._last_summaries)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "rate_matrix_"):
            raise RuntimeError("BayCAPFixed must be fitted before prediction")


def _validate_outputs(outputs: np.ndarray, n_instances: int, n_classes: int) -> None:
    expected = (n_instances, n_classes)
    if outputs.ndim != 2 or outputs.shape != expected:
        raise ValueError(f"posteriors must have shape {expected}; found {outputs.shape}")
    if not np.all(np.isfinite(outputs)):
        raise ValueError("posteriors must contain only finite values")


def _class_order(quantifier_classes: np.ndarray, rate_classes: np.ndarray) -> np.ndarray:
    if len(quantifier_classes) != len(rate_classes):
        raise ValueError("quantifier and validation data have different numbers of classes")
    order = []
    for class_ in rate_classes:
        matches = np.flatnonzero(quantifier_classes == class_)
        if len(matches) != 1:
            raise ValueError("quantifier and validation data must contain the same classes")
        order.append(int(matches[0]))
    return np.asarray(order, dtype=int)


def _validate_prevalence_samples(samples: np.ndarray, n_classes: int) -> None:
    if samples.ndim != 2 or samples.shape[1] != n_classes or samples.shape[0] == 0:
        raise ValueError(f"prevalence draws must have shape (n_samples, {n_classes}); found {samples.shape}")
    if not np.all(np.isfinite(samples)):
        raise ValueError("prevalence draws must contain only finite values")
    if np.any(samples < 0) or not np.allclose(samples.sum(axis=1), 1.0):
        raise ValueError("prevalence draws must lie on the probability simplex")


__all__ = [
    "FIXED_SUMMARIES",
    "INPUT_CLASSIFIERS",
    "NUTS_DIAGNOSTICS",
    "QUANTIFIERS",
    "BayCAPFixed",
    "accuracy_moments",
    "emq_model",
    "ilr",
    "ilr_inverse",
    "kdey_model",
    "top_label_ece",
]
