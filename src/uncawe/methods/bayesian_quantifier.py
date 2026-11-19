r"""Bayesian quantifiers: the posterior of the target prevalence $\pi^Q$ given the outputs of an input classifier.

Work-plan card WP-25 (Step 1) separated the quantifiers from the CAP methods that use them. Before, BayesKDEy and
BayesianMAPLS lived inside the BayCAP methods (``methods/baycap_quantifier.py:QuantifierBayCAP``), and SQS-B borrowed a
whole BayCAP method for its posterior draws. This module holds what the two quantifiers share: the input classifier
whose outputs the quantifier reads ($h$ itself, or a separate MLP trained on $V$ with cross-fitted validation outputs),
the NUTS run with its divergences and diagnostics, and the top-label ECE of the input classifier. The quantifiers are
``methods/bayes_kdey.py:BayesKDEy`` and ``methods/bayes_mapls.py:BayesMAPLS``; the code is moved unchanged, so the
draws are those of the classes before WP-25 (``tests/test_wp25_restructure.py``).
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import numpy as np
import quapy as qp
from numpyro.infer import MCMC, NUTS
from quapy.data import LabelledCollection
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier

from uncawe.methods.baycap import NUTS_DIAGNOSTICS, nuts_diagnostics

INPUT_CLASSIFIERS = ("h", "mlp")
# number of folds of the cross-fitted validation outputs of the MLP (QuaPy's default ``val_split``); the folds run in
# the worker's process, so that a BayCAP worker keeps one core busy (docs/work_plan.md, Part V.1)
MLP_FOLDS = 5
# equal-width bins of the top-label expected calibration error
ECE_BINS = 15
# the site of the prevalences in QuaPy's model of BayesKDEy, also the key of the draws of pi in the NUTS record
PREV = "prev"


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


class BayesianQuantifier:
    r"""Base class of the Bayesian quantifiers: the posterior of $\pi^Q$ given the outputs of an input classifier.

    - ``input_classifier="h"``: the quantifier reads the outputs of $h$, the ``posteriors`` arguments of ``fit`` (on
      $V$, out of sample since $h$ was trained on $L$) and of ``outputs`` (on the target bag); no classifier is trained,
      and the columns of the outputs follow ``val.classes_``.
    - ``input_classifier="mlp"``: the quantifier reads the outputs of a separate ``MLPClassifier()`` (default settings,
      ``random_state``) trained on $V$; its validation outputs are cross-fitted (``MLP_FOLDS`` folds, as QuaPy's
      ``fit``), its target outputs come from the MLP trained on all of $V$.

    A subclass fits its model on the validation outputs (:meth:`_fit`), and gives the NUTS model of $\pi$ for the
    outputs of a bag (:meth:`_model`) and the draws of $\pi$ from the NUTS samples (:meth:`_prevalence_draws`).
    :meth:`sample_prevalences` returns the draws with columns in the order of ``val.classes_``.

    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the bag.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS and of the MLP; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        input_classifier: Literal["h", "mlp"] = "mlp",
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        if input_classifier not in INPUT_CLASSIFIERS:
            raise ValueError(f"input_classifier must be one of {INPUT_CLASSIFIERS}; found {input_classifier!r}")
        if isinstance(prior, str) and prior != "uniform":
            raise ValueError(f"prior must be 'uniform', a positive scalar or a vector; found {prior!r}")
        if not temperature > 0:
            raise ValueError(f"temperature must be positive; found {temperature!r}")
        self.input_classifier = input_classifier
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.prior = prior
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        """Obtain the validation outputs of the input classifier and fit the quantifier on them and their labels.

        :param val: the validation set $V$.
        :param posteriors: the outputs of $h$ on $V$, columns in the order of ``val.classes_``.
        """
        posteriors = np.asarray(posteriors, dtype=float)
        classes = np.asarray(val.classes_)
        _validate_outputs(posteriors, len(val), len(classes))
        self.classes_ = classes
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
        # class_order_[j]: the column of the outputs (and of the draws of the model) of class val.classes_[j]
        self.class_order_ = _class_order(self.quantifier_classes_, classes)
        self.val_outputs_ = val_outputs
        self.val_labels_ = labels
        self.val_ece_ = top_label_ece(val_outputs, labels, self.quantifier_classes_)
        self._fit(val_outputs, labels)
        return self

    def _fit(self, val_outputs: np.ndarray, labels: np.ndarray) -> None:
        """Fit the quantifier on the validation outputs of the input classifier and their true labels."""
        raise NotImplementedError

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model of $\\pi$ for the outputs of a bag, its arguments and the initial parameters."""
        raise NotImplementedError

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        """The draws of $\\pi$, grouped by chain, from the grouped NUTS samples of :meth:`_model`."""
        raise NotImplementedError

    def outputs(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """The outputs of the input classifier on a bag: ``posteriors`` for $h$, those of the MLP on ``X`` otherwise."""
        self._check_is_fitted()
        if self.input_classifier == "h":
            return np.asarray(posteriors, dtype=float)
        return self.mlp_.predict_proba(X)

    def run_nuts(self, model: Callable, model_args: dict, init_params: dict | None) -> MCMC:
        """Run one chain of NUTS from ``PRNGKey(random_state)``, recording the divergent transitions."""
        mcmc = MCMC(
            NUTS(model), num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=1, progress_bar=False
        )
        mcmc.run(
            jax.random.PRNGKey(self.random_state), init_params=init_params, extra_fields=("diverging",), **model_args
        )
        return mcmc

    def sample_prevalences(self, outputs: np.ndarray) -> np.ndarray:
        r"""Sample the posterior of $\pi$ given the outputs of the input classifier on the bag, recording the divergent
        transitions.

        :return: the draws of $\pi$, of shape ``(num_samples, K)``, columns in the order of ``val.classes_``.
        """
        self._check_is_fitted()
        mcmc = self.run_nuts(*self._model(outputs))
        prev = self._prevalence_draws(mcmc.get_samples(group_by_chain=True))
        self._last_nuts = ({PREV: prev}, np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])))
        return prev[0][:, self.class_order_]

    def last_nuts_diagnostics(self) -> dict[str, float]:
        """Return the NUTS diagnostics (``NUTS_DIAGNOSTICS``) of the last :meth:`sample_prevalences` call."""
        if getattr(self, "_last_nuts", None) is None:
            raise RuntimeError("No NUTS run yet: call sample_prevalences first")
        return nuts_diagnostics(*self._last_nuts)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "val_outputs_"):
            raise RuntimeError(f"{type(self).__name__} must be fitted before prediction")


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


__all__ = [
    "ECE_BINS",
    "INPUT_CLASSIFIERS",
    "MLP_FOLDS",
    "NUTS_DIAGNOSTICS",
    "PREV",
    "BayesianQuantifier",
    "top_label_ece",
]
