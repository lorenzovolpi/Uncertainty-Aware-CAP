r"""Code shared by the BayCAP methods built on a Bayesian quantifier (E3.1 and E3.2 of
``docs/ext3_posterior_propagation.md``).

Work-plan card WP-13 split the former ``methods/baycap_fixed.py:BayCAPFixed`` (card WP-8) into
``methods/baycap_kdey.py:BayCAPKDEy`` (BayesKDEy) and ``methods/baycap_mapls.py:BayCAPMAPLS`` (BayesianMAPLS). This
module holds what the two classes share: the input classifier whose outputs the quantifier reads (the classifier $h$
itself or a separate MLP trained on $V$, with cross-fitted validation outputs), the fixed rate matrix $\hat C$ of $h$ on
$V$, the NUTS run with its divergences, the population and realised pushforwards of a fixed $\hat C$, the moments of
accuracy (§T.8) and the top-label ECE. The code is moved unchanged, so the two classes give the draws of
``BayCAPFixed`` for the same configuration and seeds (``tests/test_baycap_split.py``).
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import numpy as np
import quapy as qp
from cap.models.confidence import CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from numpyro.infer import MCMC, NUTS
from quapy.data import LabelledCollection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.neural_network import MLPClassifier

from uncawe.methods.baycap import ESTIMANDS, NUTS_DIAGNOSTICS, nuts_diagnostics, realised_tables

INPUT_CLASSIFIERS = ("h", "mlp")
# number of folds of the cross-fitted validation outputs of the MLP (QuaPy's default ``val_split``); the folds run in
# the worker's process, so that a BayCAP worker keeps one core busy (docs/work_plan.md, Part V.1)
MLP_FOLDS = 5
# equal-width bins of the top-label expected calibration error
ECE_BINS = 15
# per-bag summaries stored by the experiments (``last_posterior_summaries``) for a fixed rate matrix, vanilla accuracy:
# the posterior mean of pi (a vector of length K), the standard deviation sigma_p of the population accuracies of the
# draws, the bag term sigma_g of the realised pushforward (§T.8), the top-label ECE of the input classifier on its
# validation outputs
FIXED_SUMMARIES = ("prev_mean", "acc_sd", "sigma_g", "val_ece")
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


class QuantifierBayCAP(CAPContingencyTable, CTCAPWithConfidence):
    r"""Base class of the BayCAP methods built on a Bayesian quantifier.

    ``fit`` measures $\hat C_{ij}=f_{ij}/n^s_j$ from the predictions of $h$ on $V$ and the validation counts $f_{ij}$,
    obtains the validation outputs of the input classifier and fits the quantifier on them (:meth:`_fit_quantifier`).
    With a fixed rate matrix, ``predict_ct_range`` samples the posterior of $\pi^Q$ with NUTS and turns each draw into a
    table through $\hat C$; a subclass with a random $C$ overrides it.

    - ``input_classifier="h"``: the quantifier reads the outputs of $h$, the ``posteriors`` arguments of ``fit`` (on
      $V$, out of sample since $h$ was trained on $L$) and of ``predict_ct_range`` (on the target bag); no classifier is
      trained, and the columns of the outputs follow ``val.classes_``.
    - ``input_classifier="mlp"``: the quantifier reads the outputs of a separate ``MLPClassifier()`` (default settings,
      ``random_state``) trained on $V$; its validation outputs are cross-fitted (``MLP_FOLDS`` folds, as QuaPy's
      ``fit``), its target outputs come from the MLP trained on all of $V$.

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
        if estimand not in ESTIMANDS:
            raise ValueError(f"estimand must be one of {ESTIMANDS}; found {estimand!r}")
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
        self._fit_quantifier(val_outputs, labels, predicted)
        return self

    def _fit_quantifier(self, val_outputs: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> None:
        """Fit the quantifier on the validation outputs of the input classifier, their true labels and the predictions
        of $h$ on $V$."""
        raise NotImplementedError

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model of $\\pi$ for the outputs of a bag, its arguments and the initial parameters."""
        raise NotImplementedError

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        """The draws of $\\pi$, grouped by chain, from the grouped NUTS samples of :meth:`_model`."""
        raise NotImplementedError

    def outputs(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """The outputs of the input classifier on a bag: ``posteriors`` for $h$, those of the MLP on ``X`` otherwise."""
        if self.input_classifier == "h":
            return np.asarray(posteriors, dtype=float)
        return self.mlp_.predict_proba(X)

    def _run_nuts(self, model: Callable, model_args: dict, init_params: dict | None) -> MCMC:
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

        :return: the draws of $\pi$, of shape ``(num_samples, K)``, columns in the order of ``quantifier_classes_``.
        """
        mcmc = self._run_nuts(*self._model(outputs))
        prev = self._prevalence_draws(mcmc.get_samples(group_by_chain=True))
        self._last_nuts = ({PREV: prev}, np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])))
        return prev[0]

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalised contingency table per posterior draw (true classes on rows), with the fixed $\hat C$.

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
    "ECE_BINS",
    "FIXED_SUMMARIES",
    "INPUT_CLASSIFIERS",
    "MLP_FOLDS",
    "NUTS_DIAGNOSTICS",
    "QuantifierBayCAP",
    "accuracy_moments",
    "top_label_ece",
]
