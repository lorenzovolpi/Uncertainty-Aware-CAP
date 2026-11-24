r"""Code shared by the BayCAP methods built on a Bayesian quantifier (E3.1 and E3.2 of
``docs/ext3_posterior_propagation.md``).

Work-plan card WP-13 split the former ``methods/baycap_fixed.py:BayCAPFixed`` (card WP-8) into
``methods/baycap_kdey.py:BayCAPKDEy`` (BayesKDEy) and ``methods/baycap_mapls.py:BayCAPMAPLS`` (BayesianMAPLS), with the
draws of ``BayCAPFixed`` (``tests/test_baycap_split.py``). Work-plan card WP-25 moved the quantifiers into classes of
their own (``methods/bayesian_quantifier.py:BayesianQuantifier``, ``methods/bayes_kdey.py:BayesKDEy``,
``methods/bayes_mapls.py:BayesMAPLS``), with the same draws (``tests/test_wp25_restructure.py``). This module holds what
the two BayCAP classes share: the fixed rate matrix $\hat C$ of $h$ on $V$, the population and realised pushforwards of
a fixed $\hat C$, and the moments of accuracy (§T.8); the quantifier is built by each subclass from its own
hyperparameters.
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import numpy as np
import quapy as qp
from cap.models.confidence import CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from quapy.data import LabelledCollection
from sklearn.metrics import confusion_matrix

from uncawe.methods.baycap import ESTIMANDS, NUTS_DIAGNOSTICS, nuts_diagnostics, realised_tables
from uncawe.methods.bayesian_quantifier import (
    ECE_BINS,
    INPUT_CLASSIFIERS,
    MLP_FOLDS,
    PREV,
    BayesianQuantifier,
    _validate_outputs,
    top_label_ece,
)

# per-bag summaries stored by the experiments (``last_posterior_summaries``) for a fixed rate matrix, vanilla accuracy:
# the posterior mean of pi (a vector of length K), the standard deviation sigma_p of the population accuracies of the
# draws, the bag term sigma_g of the realised pushforward (§T.8), the top-label ECE of the input classifier on its
# validation outputs
FIXED_SUMMARIES = ("prev_mean", "acc_sd", "sigma_g", "val_ece")


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
    accuracies = prevalences @ np.diag(rates)
    precision = class_precisions(prevalences, rates)
    sigma_g2 = (precision * (1 - precision)) @ pred_counts / pred_counts.sum() ** 2
    return float(np.std(accuracies)), float(np.sqrt(np.mean(sigma_g2)))


def class_precisions(prevalences: np.ndarray, rates: np.ndarray) -> np.ndarray:
    r"""The precisions $\rho_{ii}(\pi^{(r)})=\hat C_{ii}\pi^{(r)}_i/\sum_l\hat C_{il}\pi^{(r)}_l$ of each draw (§T.7),
    $1/K$ for a predicted class that the draw makes impossible.

    :param prevalences: draws of $\pi$, of shape ``(R, K)``.
    :param rates: the rate matrix in the ``cap-ml`` layout, row $j$ the column $\hat C_{:j}$ of the paper.
    :return: ``[r, i]``$=\rho_{ii}(\pi^{(r)})$, shape ``(R, K)``.
    """
    prevalences = np.asarray(prevalences, dtype=float)
    rates = np.asarray(rates, dtype=float)
    n_classes = rates.shape[0]
    # predicted[r, i] = sum_j pi_j C_ij, the share of the population predicted as class i under draw r
    predicted = prevalences @ rates
    correct = prevalences * np.diag(rates)
    return np.divide(correct, predicted, out=np.full_like(correct, 1.0 / n_classes), where=predicted > 0)


class QuantifierBayCAP(CAPContingencyTable, CTCAPWithConfidence):
    r"""Base class of the BayCAP methods built on a Bayesian quantifier.

    ``fit`` measures $\hat C_{ij}=f_{ij}/n^s_j$ from the predictions of $h$ on $V$ and the validation counts $f_{ij}$,
    and fits the quantifier (``quantifier_``, built by :meth:`_make_quantifier` from the hyperparameters of the method),
    which reads the outputs of the input classifier (:class:`uncawe.methods.bayesian_quantifier.BayesianQuantifier`:
    $h$ itself, or a separate MLP trained on $V$ with cross-fitted validation outputs). With a fixed rate matrix,
    ``predict_ct_range`` samples the posterior of $\pi^Q$ with NUTS and turns each draw into a table through $\hat C$;
    a subclass with a random $C$ overrides it.

    The estimand decides how each draw becomes a contingency table:

    - ``"population"``: the population table $C\operatorname{diag}(\pi^{(r)})$;
    - ``"realised"``: :func:`uncawe.methods.baycap.realised_tables` applied to the population tables, with $n^t$ the
      arg-max predictions of $h$ on the bag, the realised table of the target sample, which is what the experiments
      score (``true_cts``).

    Tables follow the ``cap-ml`` layout (true classes on rows), so the paper's $C\operatorname{diag}(\pi)$ is returned
    transposed, as $\operatorname{diag}(\pi)C^\top$. ``random_state`` seeds NUTS, the MLP and the multinomial draws of
    ``"realised"``, so the two estimands share their posterior draws.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param estimand: ``"population"`` or ``"realised"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the quantifier.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS, of the MLP and of the multinomial draws; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    # the estimands that the class accepts; a subclass may add its own (BayCAPKDEy: "realised-instance", E3.4)
    estimands: tuple[str, ...] = ESTIMANDS

    def __init__(
        self,
        acc_fn: Callable,
        input_classifier: Literal["h", "mlp"] = "mlp",
        estimand: Literal["population", "realised"] = "population",
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        if input_classifier not in INPUT_CLASSIFIERS:
            raise ValueError(f"input_classifier must be one of {INPUT_CLASSIFIERS}; found {input_classifier!r}")
        if estimand not in self.estimands:
            raise ValueError(f"estimand must be one of {self.estimands}; found {estimand!r}")
        if isinstance(prior, str) and prior != "uniform":
            raise ValueError(f"prior must be 'uniform', a positive scalar or a vector; found {prior!r}")
        if not temperature > 0:
            raise ValueError(f"temperature must be positive; found {temperature!r}")
        CAPContingencyTable.__init__(self, acc_fn)
        self.input_classifier = input_classifier
        self.estimand = estimand
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.temperature = float(temperature)
        self.prior = prior
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def _quantifier_settings(self) -> dict:
        """The hyperparameters that every quantifier takes, as set on the method."""
        return {
            "input_classifier": self.input_classifier,
            "num_warmup": self.num_warmup,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "prior": self.prior,
            "random_state": self.random_state,
        }

    def _make_quantifier(self) -> BayesianQuantifier:
        """The unfitted quantifier, built from the hyperparameters of the method."""
        raise NotImplementedError

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        """Measure the rates of $h$ on $V$ and fit the quantifier on the outputs of the input classifier."""
        posteriors = np.asarray(posteriors, dtype=float)
        classes = np.asarray(val.classes_)
        _validate_outputs(posteriors, len(val), len(classes))
        self.classes_ = classes
        predicted = classes[np.argmax(posteriors, axis=1)]
        # cap-ml layout: row j is the column C_{:j} of the paper, the counts and rates of the instances of class j
        self.val_counts_ = confusion_matrix(val.y, predicted, labels=classes)
        self.rate_matrix_ = confusion_matrix(val.y, predicted, labels=classes, normalize="true")
        self.quantifier_ = self._make_quantifier().fit(val, posteriors)
        self._after_fit(np.asarray(val.y), predicted)
        return self

    def _after_fit(self, labels: np.ndarray, predicted: np.ndarray) -> None:
        """Fit what the method adds to the quantifier, from the true labels and the predictions of $h$ on $V$."""

    @property
    def mlp_(self):
        """The MLP of the input classifier ``"mlp"``, ``None`` with ``"h"``."""
        return self.quantifier_.mlp_

    @property
    def val_outputs_(self) -> np.ndarray:
        """The validation outputs of the input classifier (cross-fitted for the MLP)."""
        return self.quantifier_.val_outputs_

    @property
    def val_ece_(self) -> float:
        """The top-label ECE of the input classifier on its validation outputs."""
        return self.quantifier_.val_ece_

    @property
    def quantifier_classes_(self) -> np.ndarray:
        """The classes of the columns of the outputs of the input classifier."""
        return self.quantifier_.quantifier_classes_

    @property
    def _quantifier_to_rate_order(self) -> np.ndarray:
        """The column of the outputs of each class of ``classes_``."""
        return self.quantifier_.class_order_

    def outputs(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """The outputs of the input classifier on a bag: ``posteriors`` for $h$, those of the MLP on ``X`` otherwise."""
        return self.quantifier_.outputs(X, posteriors)

    def sample_prevalences(self, outputs: np.ndarray) -> np.ndarray:
        r"""Sample the posterior of $\pi$ given the outputs of the input classifier on the bag, recording the divergent
        transitions.

        :return: the draws of $\pi$, of shape ``(num_samples, K)``, columns in the order of ``classes_``.
        """
        prevalences = self.quantifier_.sample_prevalences(outputs)
        self._last_nuts = self.quantifier_._last_nuts
        return prevalences

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalised contingency table per posterior draw (true classes on rows), with the fixed $\hat C$.

        ``posteriors`` are the outputs of $h$ on the bag: they give $n^t$ and, with ``input_classifier="h"``, the
        outputs the quantifier reads; with ``"mlp"`` the quantifier reads the outputs of its MLP on ``X``.
        """
        self._check_is_fitted()
        posteriors = np.asarray(posteriors, dtype=float)
        prevalences = self.sample_prevalences(self.outputs(X, posteriors))
        _validate_prevalence_samples(prevalences, len(self.classes_))

        pred_counts = np.bincount(np.argmax(posteriors, axis=1), minlength=len(self.classes_))
        acc_sd, sigma_g = accuracy_moments(prevalences, self.rate_matrix_, pred_counts)
        self._last_summaries = {
            "prev_mean": prevalences.mean(axis=0),
            "acc_sd": acc_sd,
            "sigma_g": sigma_g,
            "val_ece": self.val_ece_,
        }

        return self._pushforward(prevalences, posteriors, pred_counts)

    def _pushforward(self, prevalences: np.ndarray, posteriors: np.ndarray, pred_counts: np.ndarray) -> np.ndarray:
        r"""Turn the draws of $\pi$ into tables through $\hat C$, with the estimand of the method (``"population"`` or
        ``"realised"``); a subclass with an estimand of its own overrides it."""
        cts = prevalences[:, :, None] * self.rate_matrix_[None, :, :]
        if self.estimand == "realised":
            cts = realised_tables(cts, pred_counts, np.random.default_rng(self.random_state))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return the mean of the tables of :meth:`predict_ct_range`."""
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    def last_nuts_diagnostics(self) -> dict[str, float]:
        """Return the NUTS diagnostics (``NUTS_DIAGNOSTICS``) of the last ``predict_ct_range`` call."""
        if getattr(self, "_last_nuts", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        if self._last_nuts == "failed":
            return {name: np.nan for name in NUTS_DIAGNOSTICS}
        return nuts_diagnostics(*self._last_nuts)

    def last_posterior_summaries(self) -> dict:
        """Return the per-bag summaries of the last ``predict_ct_range`` call."""
        if getattr(self, "_last_summaries", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        return dict(self._last_summaries)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "rate_matrix_"):
            raise RuntimeError(f"{type(self).__name__} must be fitted before prediction")


def _validate_prevalence_samples(samples: np.ndarray, n_classes: int) -> None:
    if samples.ndim != 2 or samples.shape[1] != n_classes or samples.shape[0] == 0:
        raise ValueError(f"prevalence draws must have shape (n_samples, {n_classes}); found {samples.shape}")
    if not np.all(np.isfinite(samples)):
        raise ValueError("prevalence draws must contain only finite values")
    if np.any(samples < 0) or not np.allclose(samples.sum(axis=1), 1.0):
        raise ValueError("prevalence draws must lie on the probability simplex")


__all__ = [
    "ECE_BINS",
    "FIXED_SUMMARIES",
    "INPUT_CLASSIFIERS",
    "MLP_FOLDS",
    "NUTS_DIAGNOSTICS",
    "PREV",
    "QuantifierBayCAP",
    "accuracy_moments",
    "class_precisions",
    "top_label_ece",
]
