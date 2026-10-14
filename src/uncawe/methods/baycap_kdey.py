"""BayCAP-KDEy with a fixed classifier rate matrix.

This module implements the proposal documented in ``docs/baycap_kdey.md``.
The method follows the ``cap-ml`` contingency-table API while delegating the
Bayesian prevalence posterior to an internally configured QuaPy
``BayesianKDEy`` quantifier.
"""

from collections.abc import Callable
from typing import Literal, Self

import numpy as np
from cap.models.confidence import CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from quapy.data import LabelledCollection
from quapy.method._bayesian import BayesianKDEy
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


class BayCAPKDEy(CAPContingencyTable, CTCAPWithConfidence):
    r"""Push a BayesKDEy prevalence posterior through a fixed rate matrix.

    The method internally instantiates ``BayesianKDEy`` with an
    ``MLPClassifier`` surrogate. This surrogate and the target classifier whose
    performance is estimated are deliberately distinct. ``posteriors`` passed
    to :meth:`fit` are the target classifier's out-of-sample predictions on the
    labelled validation set; they determine the fixed rate matrix
    $P(\hat Y\mid Y)$. BayesKDEy and its surrogate are fitted independently on
    the validation instances and labels.

    At prediction time the target classifier's ``posteriors`` argument is
    accepted for compatibility with ``cap-ml`` but is not used: target
    prevalence is inferred from ``X`` by the internal surrogate. Joint
    posterior draws $\pi^{(r)}$ are then pushed through the fixed rate matrix;
    uncertainty in that matrix is not modelled.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param kernel: KDE geometry used by BayesKDEy. One of ``"gaussian"``,
        ``"aitchison"``, or ``"ilr"``.
    :param bandwidth: KDE bandwidth, either a positive numeric value or a
        QuaPy-supported bandwidth rule such as ``"scott"`` or ``"silverman"``.
    :param shrinkage: shrinkage towards the uniform composition before an
        Aitchison or ILR transformation. It must be in $[0,1)$ and is not
        supported with the Gaussian kernel.
    :param num_warmup: number of NumPyro NUTS warm-up iterations.
    :param num_samples: number of joint posterior prevalence draws retained
        after warm-up.
    :param mcmc_seed: random seed used by the NumPyro MCMC sampler. It does not
        set the random state of the internal ``MLPClassifier``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        kernel: Literal["gaussian", "aitchison", "ilr"] = "gaussian",
        bandwidth: float | Literal["scott", "silverman"] = 0.1,
        shrinkage: float = 0.0,
        num_warmup: int = 500,
        num_samples: int = 1_000,
        mcmc_seed: int = 0,
    ) -> None:
        CAPContingencyTable.__init__(self, acc_fn)
        self._q_params = {
            "kernel": kernel,
            "bandwidth": bandwidth,
            "shrinkage": shrinkage,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "mcmc_seed": mcmc_seed,
        }
        self.q = BayesianKDEy(MLPClassifier(), **self._q_params)

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        """Fit BayesKDEy and measure the target classifier's fixed rates."""
        posteriors = np.asarray(posteriors)
        classes = np.asarray(val.classes_)
        self._validate_validation_posteriors(posteriors, len(val), len(classes))

        self.q.fit(*val.Xy)
        self.classes_ = classes
        self._quantifier_to_rate_order = self._class_order(np.asarray(self.q.classes_), classes)

        predicted_labels = classes[np.argmax(posteriors, axis=1)]
        self.rate_matrix_ = confusion_matrix(
            val.y,
            predicted_labels,
            labels=classes,
            normalize="true",
        )
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalized contingency table per posterior draw.

        ``cap-ml`` represents contingency tables with true classes on rows and
        predicted classes on columns. Thus the proposal's
        $C^h\operatorname{diag}(\pi)$ is returned transposed, equivalently as
        $\operatorname{diag}(\pi)(C^h)^\top$.
        """
        self._check_is_fitted()
        quantifier_posteriors = self.q.classify(X)
        prevalence_samples = np.asarray(self.q.sample_from_posterior(quantifier_posteriors), dtype=float)
        prevalence_samples = prevalence_samples[
            :, self._quantifier_to_rate_order
        ]  # all prevalence samples drawn during the Bayesian walk
        self._validate_prevalence_samples(prevalence_samples, len(self.classes_))

        return prevalence_samples[:, :, None] * self.rate_matrix_[None, :, :]

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return the posterior-mean normalized contingency table."""
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    @staticmethod
    def _validate_validation_posteriors(posteriors: np.ndarray, n_instances: int, n_classes: int) -> None:
        expected = (n_instances, n_classes)
        if posteriors.ndim != 2 or posteriors.shape != expected:
            raise ValueError(f"posteriors must have shape {expected}; found {posteriors.shape}")
        if not np.all(np.isfinite(posteriors)):
            raise ValueError("posteriors must contain only finite values")

    @staticmethod
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

    @staticmethod
    def _validate_prevalence_samples(samples: np.ndarray, n_classes: int) -> None:
        if samples.ndim != 2 or samples.shape[1] != n_classes or samples.shape[0] == 0:
            raise ValueError(
                f"BayesianKDEy posterior samples must have shape (n_samples, {n_classes}); found {samples.shape}"
            )
        if not np.all(np.isfinite(samples)):
            raise ValueError("BayesianKDEy posterior samples must contain only finite values")
        if np.any(samples < 0) or not np.allclose(samples.sum(axis=1), 1.0):
            raise ValueError("BayesianKDEy posterior samples must lie on the probability simplex")

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "rate_matrix_"):
            raise RuntimeError("BayCAPKDEy must be fitted before prediction")
