r"""SQS-B: SQS on the posterior of a Bayesian quantifier (E3.3 of ``docs/ext3_posterior_propagation.md``, work-plan card
WP-18).

The SQS methods of ``main`` estimate the table of $h$ on the target bag by drawing bags from $V$ at an estimated target
prevalence: ``SQS-H`` draws all of them at one estimate of PACC, ``SQS-Hbs`` one bag at the estimate of PACC on each
bootstrap resample of the target (``cap-ml``'s ``RQBS``). :class:`SQSB` draws one bag at each draw $\pi^{(r)}$ of the
posterior of a Bayesian quantifier, so that the uncertainty on $\pi^Q$ comes from the posterior instead of the
bootstrap. Its structure is that of ``RQBS`` (user's decision of 2026-11-10); the quantifier is a
:class:`uncawe.methods.baycap_quantifier.QuantifierBayCAP` with a fixed rate matrix, reused through its ``fit``,
``outputs`` and ``sample_prevalences``, so that, for the same seeds, SQS-B and the BayCAP method of the same quantifier
share their prevalence draws (``SQS-B`` and ``BayCAP-KDEy-mlp-real`` in ``main``).
"""

from collections.abc import Callable
from typing import Self

import numpy as np
import quapy as qp
from cap.models.confidence import CTCAPWithConfidence, _temp_np_seed
from cap.models.cont_table import CAPContingencyTable
from cap.models.utils import smooth
from quapy.data import LabelledCollection

from uncawe.methods.baycap_quantifier import (
    FIXED_SUMMARIES,
    QuantifierBayCAP,
    _validate_prevalence_samples,
    accuracy_moments,
)

# the smoothing of the prevalence draws before the bags are drawn, as in RQBS (cap.models.utils.smooth)
SMOOTHING = 1e-5
# per-bag summaries of SQS-B, vanilla accuracy: those of the BayCAP methods with a fixed rate matrix (E3.1), the bag
# term sigma_f of the forward bag, and the spreads S_rec of the recalls and S_prec of the precisions at the posterior
# mean of pi (§T.14)
SQSB_SUMMARIES = (*FIXED_SUMMARIES, "sigma_f", "s_rec", "s_prec")


def forward_bag_variances(prevalences: np.ndarray, rates: np.ndarray, sample_size: int) -> np.ndarray:
    r"""$\sigma_f^2(\pi^{(r)})=\frac{1}{N^Q}\sum_j\pi^{(r)}_j\hat d_j(1-\hat d_j)$ for each draw (§T.14): the variance of
    the accuracy of $h$ on a bag of $N^Q$ instances drawn from $V$ at the class counts $N^Q\pi^{(r)}$, with $\hat d_j$
    the recall of $h$ on $\omega_j$ measured on $V$.

    :param prevalences: draws of $\pi$, of shape ``(R, K)``.
    :param rates: the rate matrix in the ``cap-ml`` layout, row $j$ the column $\hat C_{:j}$ of the paper.
    :param sample_size: $N^Q$, the size of the bags.
    :return: an array of shape ``(R,)``.
    """
    recalls = np.diag(np.asarray(rates, dtype=float))
    return np.asarray(prevalences, dtype=float) @ (recalls * (1 - recalls)) / sample_size


def luck_spreads(prevalence: np.ndarray, rates: np.ndarray) -> tuple[float, float]:
    r"""Return $(S_{\text{rec}},S_{\text{prec}})$ for one prevalence vector $\pi$ and a fixed $\hat C$ (§T.14).

    With $a=\sum_j\pi_j\hat d_j$, $S_{\text{rec}}=\sum_j\pi_j(\hat d_j-a)^2$ is the spread of the recalls weighted by the
    true classes, and $S_{\text{prec}}=\sum_iq_i(\rho_{ii}-a)^2$ the spread of the precisions
    $\rho_{ii}=\hat d_i\pi_i/q_i$ weighted by the predicted shares $q_i=\sum_j\hat C_{ij}\pi_j$. By the law of total
    variance, $a(1-a)=\sum_j\pi_j\hat d_j(1-\hat d_j)+S_{\text{rec}}=\sum_iq_i\rho_{ii}(1-\rho_{ii})+S_{\text{prec}}$. A
    predicted class with $q_i=0$ has weight $0$.

    :param prevalence: $\pi$, of shape ``(K,)``.
    :param rates: the rate matrix in the ``cap-ml`` layout.
    """
    prevalence = np.asarray(prevalence, dtype=float)
    rates = np.asarray(rates, dtype=float)
    recalls = np.diag(rates)
    accuracy = float(prevalence @ recalls)
    s_rec = float(prevalence @ (recalls - accuracy) ** 2)
    # predicted[i] = sum_j pi_j C_ij, the share of the population predicted as class i
    predicted = prevalence @ rates
    precision = np.divide(prevalence * recalls, predicted, out=np.zeros_like(predicted), where=predicted > 0)
    s_prec = float(predicted @ (precision - accuracy) ** 2)
    return s_rec, s_prec


def bag_tables(labels: np.ndarray, predictions: np.ndarray, n_classes: int) -> np.ndarray:
    """The normalised contingency tables (true classes on rows) of bags of equal size, as ``cap-ml``'s
    ``contingency_table`` applied to each bag.

    :param labels: the class indices of the instances of each bag, of shape ``(R, N)``.
    :param predictions: the class indices predicted by $h$, of shape ``(R, N)``.
    """
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    n_bags, size = labels.shape
    cells = labels * n_classes + predictions + (np.arange(n_bags) * n_classes**2)[:, None]
    counts = np.bincount(cells.ravel(), minlength=n_bags * n_classes**2)
    return counts.reshape(n_bags, n_classes, n_classes) / size


class SQSB(CAPContingencyTable, CTCAPWithConfidence):
    r"""SQS-B: one validation bag per posterior draw of a Bayesian quantifier (E3.3; code name ``SQSB``, method
    ``SQS-B`` of ``main``).

    ``fit`` fits the quantifier on $V$ (its rates $\hat C$ of $h$, its input classifier and its densities) and keeps the
    outputs of $h$ on $V$ with their labels, as ``cap-ml``'s ``RQBS``. ``predict_ct_range`` takes the
    ``quantifier.num_samples`` draws $\pi^{(r)}$ of the posterior of $\pi^Q$ given the bag, smooths each as ``RQBS``
    ($\epsilon=10^{-5}$), and draws one bag of ``sample_size`` instances of $V$ per draw with QuaPy's
    ``LabelledCollection.sampling_index`` (class counts $N^Q\pi^{(r)}_j$ rounded, the remainder assigned at random,
    instances drawn with replacement within each class); all the bags are drawn sequentially from the NumPy state
    seeded by ``random_state`` at each call, as in ``RQBS``. It returns the normalised tables of $h$ on these bags (true
    classes on rows, the ``cap-ml`` layout, i.e. the transpose of the paper's tables); ``predict_ct`` returns their
    mean.

    *Estimand:* the realised table of the target sample, as every SQS method: each table is that of $h$ on a simulated
    bag of the target's size, built forwards, from true classes to predictions (§T.14). The predictions of $h$ on the
    target are not used, except for the label-free summary ``sigma_g``. *Uncertainty sources represented:* the
    posterior of $\pi^Q$ ($e_1^U$) and the luck of the bag ($e_3$), through the resampling of $V$; not the error of the
    rates of $h$ measured on $V$ ($e_2$) nor that of the densities of the quantifier ($e_1^V$) (§T.9).

    After each ``predict_ct_range`` call, :meth:`last_nuts_diagnostics` returns the NUTS diagnostics of $\pi$ and
    :meth:`last_posterior_summaries` the ``SQSB_SUMMARIES``, for vanilla accuracy: ``prev_mean`` ($\bar\pi$), ``acc_sd``
    ($\sigma_p$), ``sigma_g`` ($\bar\sigma_g$ of the realised pushforward with the same draws, §T.8), ``val_ece`` (of
    the input classifier of the quantifier), ``sigma_f``
    ($\bar\sigma_f=\sqrt{\frac1R\sum_r\sigma_f^2(\pi^{(r)})}$, :func:`forward_bag_variances`), and ``s_rec``, ``s_prec``
    at $\bar\pi$ (:func:`luck_spreads`). ``last_bag_indices_`` holds the positions in $V$ of the instances of the last
    bags, of shape ``(R, sample_size)``.

    :param acc: a ``cap-ml`` contingency-table performance function.
    :param quantifier: a :class:`QuantifierBayCAP` with a fixed rate matrix (e.g. ``BayCAPKDEy`` with
        ``c_prior="fixed"``); its ``num_samples`` is the number of bags, its ``random_state`` seeds NUTS and its MLP.
    :param sample_size: the size $N^Q$ of each bag; ``qp.environ["SAMPLE_SIZE"]`` if ``None``.
    :param random_state: seed of the bags; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        acc: Callable,
        quantifier: QuantifierBayCAP,
        sample_size: int | None = None,
        random_state: int | None = None,
    ) -> None:
        if not isinstance(quantifier, QuantifierBayCAP):
            raise TypeError(f"quantifier must be a QuantifierBayCAP; found {type(quantifier).__name__}")
        if getattr(quantifier, "c_prior", "fixed") != "fixed":
            raise ValueError(f"quantifier must have a fixed rate matrix; found c_prior={quantifier.c_prior!r}")
        CAPContingencyTable.__init__(self, acc)
        self.quantifier = quantifier
        self.sample_size = qp.environ["SAMPLE_SIZE"] if sample_size is None else sample_size
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state
        if self.sample_size is None:
            raise ValueError(
                'sample_size cannot be None; it must be specified directly or by setting qp.environ["SAMPLE_SIZE"]'
            )

    @property
    def num_samples(self) -> int:
        """The number of bags per target, one per posterior draw."""
        return self.quantifier.num_samples

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        """Fit the quantifier on $V$ and keep the outputs of $h$ on $V$, with the labels as indices of ``val.classes_``."""
        self.quantifier.fit(val, posteriors)
        classes = np.asarray(val.classes_)
        self.classes_ = classes
        position = {class_: index for index, class_ in enumerate(classes.tolist())}
        labels = np.asarray([position[y] for y in np.asarray(val.y).tolist()], dtype=int)
        self.val_post = LabelledCollection(
            instances=np.asarray(posteriors, dtype=float), labels=labels, classes=np.arange(len(classes))
        )
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return the normalised tables of $h$ on one bag of $V$ per posterior draw (true classes on rows).

        ``posteriors`` are the outputs of $h$ on the target: they give the outputs the quantifier reads with
        ``input_classifier="h"`` and $n^t$ for ``sigma_g``; with ``"mlp"`` the quantifier reads its MLP on ``X``.
        """
        if not hasattr(self, "val_post"):
            raise RuntimeError("SQSB must be fitted before prediction")
        q = self.quantifier
        posteriors = np.asarray(posteriors, dtype=float)
        n_classes = len(self.classes_)
        prevalences = q.sample_prevalences(q.outputs(X, posteriors))[:, q._quantifier_to_rate_order]
        _validate_prevalence_samples(prevalences, n_classes)

        # the draws of NUTS are float32: smoothed in double precision, each sums to 1 as sampling_index requires; the
        # summaries below use the draws as returned, as those of BayCAPKDEy
        with _temp_np_seed(self.random_state):
            indices = np.asarray(
                [
                    self.val_post.sampling_index(self.sample_size, *smooth(prevalence, epsilon=SMOOTHING))
                    for prevalence in np.asarray(prevalences, dtype=np.float64)
                ]
            )
        self.last_bag_indices_ = indices
        labels = self.val_post.y[indices]
        predictions = self.val_post.X.argmax(axis=1)[indices]

        rates = q.rate_matrix_
        pred_counts = np.bincount(np.argmax(posteriors, axis=1), minlength=n_classes)
        acc_sd, sigma_g = accuracy_moments(prevalences, rates, pred_counts)
        prev_mean = prevalences.mean(axis=0)
        s_rec, s_prec = luck_spreads(prev_mean, rates)
        self._last_summaries = {
            "prev_mean": prev_mean,
            "acc_sd": acc_sd,
            "sigma_g": sigma_g,
            "val_ece": q.val_ece_,
            "sigma_f": float(np.sqrt(np.mean(forward_bag_variances(prevalences, rates, self.sample_size)))),
            "s_rec": s_rec,
            "s_prec": s_prec,
        }
        return bag_tables(labels, predictions, n_classes)

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return the mean of the tables of :meth:`predict_ct_range`."""
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    def last_nuts_diagnostics(self) -> dict[str, float]:
        """Return the NUTS diagnostics of the posterior of $\\pi$ of the last ``predict_ct_range`` call."""
        return self.quantifier.last_nuts_diagnostics()

    def last_posterior_summaries(self) -> dict:
        """Return the ``SQSB_SUMMARIES`` of the last ``predict_ct_range`` call."""
        if getattr(self, "_last_summaries", None) is None:
            raise RuntimeError("No NUTS run yet: call predict_ct_range first")
        return dict(self._last_summaries)


__all__ = ["SMOOTHING", "SQSB", "SQSB_SUMMARIES", "bag_tables", "forward_bag_variances", "luck_spreads"]
