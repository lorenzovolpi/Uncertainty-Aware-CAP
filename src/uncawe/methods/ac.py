"""Average Confidence (AC) with LasCal and the central Poisson-binomial interval (E8.1 of
``docs/ext8_average_confidence.md``, card WP-29).

AC (Kivimäki et al., JAIR 2025) estimates the accuracy of $h$ on an unlabelled bag of $n$ instances as the mean of its
confidences $s_i$, the largest calibrated posterior of each instance. Here the posteriors are recalibrated per bag by
LasCal (Popordanoska et al., NeurIPS 2024), through the ``lascal`` package called as ``cap-ml``'s wrappers call it
(``cap/calib.py:LasCal``, used by ``CBPE``, and ``cap/calibration/lascal.py``): logits $z=\\log p$, a grid of $100$
temperatures $\\theta$ from $0.1$ to $20$, the $\\theta$ with the smallest label-shift calibration error estimated on
the bag, calibrated posteriors $\\operatorname{softmax}(z/\\theta)$. A temperature keeps the argmax, so $s_i$ is the
calibrated posterior of the prediction of $h$ ($\\hat y=\\arg\\max_kp(\\omega_k\\mid x)$ in ``main``).

The number $K$ of correct predictions is modelled as Poisson-binomial with parameters $s_1,\\dots,s_n$ (independent
Bernoulli trials given the confidences); its probability mass function is computed exactly by the FFT of ``cap-ml``'s
``PoiBin``, as in ``CBPE``. The point estimate is its mean, $\\frac1n\\sum_is_i$, and the interval is the central
$1-\\alpha$ interval of $K/n$ (``CentralPoissonInterval``, §T.3), where ``cap-ml``'s ``PoissonConfidenceInterval``
(used by ``L-CBPE``) is the highest-density interval. On binary data AC has the same confidences, point estimate and
probability mass function as ``CBPE(calib_method="lascal")``, which takes $\\max(\\tilde s,1-\\tilde s)$.

**Estimand.** The realised table of the target bag: the Poisson-binomial distribution is conditional on the confidences
of the bag and describes its realised accuracy, the trace of the realised table. The interval represents only the
luck of the bag given the calibrated confidences (§T.6): the error of the calibration (the temperature and the
importance weights estimated on one bag, the finite $V$, the part of the prior shift a temperature cannot represent)
and the duplicates of the bags drawn with replacement are not represented.

Vanilla accuracy only: for macro-F1 and macro-K the method returns NaN, as ``CBPE``. It has no random component (LasCal
is a deterministic grid search), so it takes no ``random_state``.

**Fallback.** If the ``lascal`` package raises, or selects no temperature (when every loss on the grid is NaN, the
package keeps its initial value $-1$ and returns the logits divided by $-1$, which reverses the order of the classes),
the uncalibrated posteriors are used and ``lascal_fallback`` is set. ``cap/calibration/lascal.py`` falls back silently
on an exception; ``cap/calib.py:LasCal`` (used by ``CBPE``) would instead raise in its fallback, which calls the
module's own ``softmax(preact, temp, biases)`` with ``axis=1``.
"""

import numpy as np
import torch
from cap.models._cbpe import PoiBin
from cap.models.confidence import ConfidenceInterval, DirectCAPWithConfidence
from cap.models.direct import CAPDirect
from lascal import Calibrator
from quapy.data import LabelledCollection

VANILLA_ACCURACY = "vanilla_accuracy"
# relative tolerance on the temperature recovered from the calibrated logits (``lascal_calibrate``)
TEMPERATURE_RTOL = 1e-6


class CentralPoissonInterval(ConfidenceInterval):
    """The central $1-\\alpha$ interval of $K/n$ from the probability mass function ``X`` of $K$ ($n+1$ values,
    ``X[k]``$=P(K=k)$), as in the AC paper (§4.2) with the quantile convention of §T.3:

    $$k_{\\text{low}}=\\min\\{k:F(k)\\ge\\tfrac\\alpha2\\},\\qquad k_{\\text{up}}=\\min\\{k:F(k)\\ge1-\\tfrac\\alpha2\\},$$

    with $F$ the cumulative distribution function of $K$, and the interval $[k_{\\text{low}}/n,k_{\\text{up}}/n]$; it
    holds at least $1-\\alpha$ of the probability. The FFT of ``PoiBin`` adds the machine epsilon to every value and can
    leave values of order $10^{-16}$ below $0$, so $F$ is computed from the values clipped at $0$ and normalised to sum
    to $1$. The point estimate is the mean $\\sum_k(k/n)X_k$ of the stored values, as in ``cap-ml``'s
    ``PoissonConfidenceInterval``. ``samples`` is the probability mass function. A scalar NaN ``X`` (no
    estimate, as stored for macro-F1 and macro-K) gives NaN bounds and point estimate."""

    def __init__(self, X, confidence_level: float = 0.95):
        assert 0 < confidence_level < 1, f"{confidence_level=} must be in range(0,1)"
        self.alpha = 1 - confidence_level
        if np.ndim(X) == 0 and np.isnan(X):
            self._samples = np.nan
            self._mean = np.nan
            self.low, self.high = np.nan, np.nan
            return

        pmf = np.asarray(X, dtype=float)
        if pmf.ndim != 1 or len(pmf) < 2:
            raise ValueError(f"the probability mass function must have n+1>=2 values, got shape {pmf.shape}")
        n = len(pmf) - 1
        self._samples = pmf
        self._mean = float(np.dot(np.arange(n + 1) / n, pmf))
        k_low, k_up = central_bounds(pmf, self.alpha)
        self.low, self.high = k_low / n, k_up / n

    def coverage(self, true_value: float) -> float:
        if not (np.isfinite(self.low) and np.isfinite(true_value)):
            return np.nan
        return super().coverage(true_value)


def central_bounds(pmf: np.ndarray, alpha: float) -> tuple[int, int]:
    """$(k_{\\text{low}},k_{\\text{up}})$ of the central $1-\\alpha$ interval of the distribution ``pmf`` on
    $\\{0,\\dots,n\\}$ (see ``CentralPoissonInterval``)."""
    pmf = np.clip(np.asarray(pmf, dtype=float), 0.0, None)
    cdf = np.cumsum(pmf) / pmf.sum()
    # the last value of cdf is 1 up to rounding: with the guard, the search never returns n+1
    cdf[-1] = 1.0
    k_low = int(np.searchsorted(cdf, alpha / 2, side="left"))
    k_up = int(np.searchsorted(cdf, 1 - alpha / 2, side="left"))
    return k_low, k_up


def lascal_calibrate(
    val_posteriors: np.ndarray, val_y: np.ndarray, posteriors: np.ndarray
) -> tuple[np.ndarray, float, bool]:
    """The posteriors of the bag recalibrated by LasCal, with the selected temperature $\\theta$ and the fallback flag.

    The ``lascal`` package is called with the inputs of ``cap-ml``'s wrappers (logits $\\log p$ as ``float64``
    tensors), so on the same posteriors the calibrated posteriors are those of ``CBPE(calib_method="lascal")``. The
    package returns the logits of the bag divided by $\\theta$ and not $\\theta$ itself, so $\\theta$ is recovered as the
    ratio of the logits to the calibrated logits on the finite non-zero entries (all equal up to rounding). On an
    exception of the package, on a temperature that is not finite and positive, or on ratios that disagree, the
    uncalibrated posteriors are returned, with $\\theta=$ NaN and the flag set."""
    val_logits = torch.log(torch.from_numpy(np.asarray(val_posteriors)))
    logits = torch.log(torch.from_numpy(np.asarray(posteriors)))
    try:
        calibrated_agg = Calibrator(experiment_path=None, verbose=False, covariate=False).calibrate(
            method_name="lascal",
            source_agg={"y_logits": val_logits, "y_true": torch.from_numpy(np.asarray(val_y))},
            target_agg={"y_logits": logits, "y_true": None},
            train_agg=None,
        )
        calibrated_logits = calibrated_agg["target"]["y_logits"]
        temperature = recover_temperature(logits.numpy(), calibrated_logits.numpy())
        if not (np.isfinite(temperature) and temperature > 0):
            raise ValueError(f"LasCal selected no positive temperature ({temperature})")
        return calibrated_logits.softmax(-1).numpy(), temperature, False
    except Exception:  # noqa: BLE001 - any failure of the package falls back, as in cap-ml's wrappers, and is flagged
        return np.asarray(posteriors, dtype=float), np.nan, True


def recover_temperature(logits: np.ndarray, calibrated_logits: np.ndarray) -> float:
    """$\\theta$ from $z$ and $z/\\theta$: the median ratio over the entries where both are finite and $z\\ne0$; NaN if
    there is no such entry or if the ratios differ by more than ``TEMPERATURE_RTOL`` (relative)."""
    logits, calibrated_logits = np.asarray(logits, dtype=float), np.asarray(calibrated_logits, dtype=float)
    valid = np.isfinite(logits) & np.isfinite(calibrated_logits) & (logits != 0) & (calibrated_logits != 0)
    if not valid.any():
        return np.nan
    ratios = logits[valid] / calibrated_logits[valid]
    temperature = float(np.median(ratios))
    if not np.allclose(ratios, temperature, rtol=TEMPERATURE_RTOL, atol=0.0):
        return np.nan
    return temperature


def poisson_binomial_pmf(confidences: np.ndarray) -> np.ndarray:
    """The probability mass function of the number of successes of independent Bernoulli($s_i$) trials, $n+1$ values,
    by the FFT of ``cap-ml``'s ``PoiBin`` (as ``CBPE``)."""
    confidences = np.asarray(confidences, dtype=float)
    if len(confidences) == 0:
        raise ValueError("AC needs at least one instance")
    return np.asarray(PoiBin(confidences).pmf(list(range(len(confidences) + 1))), dtype=float)


class AC(CAPDirect, DirectCAPWithConfidence):
    """Average Confidence with LasCal and the central Poisson-binomial interval (see the module docstring).

    ``fit`` keeps the posteriors and labels of $V$; ``predict_range`` recalibrates the posteriors of each bag with
    LasCal and returns the probability mass function of $K$ ($n+1$ values), from which ``ci_from_accs`` builds the
    central interval and the point estimate. After each call, ``last_lascal_columns()`` gives the per-bag columns
    ``lascal_temperature`` and ``lascal_fallback`` (``None`` for the metrics the method does not estimate)."""

    def __init__(self, acc_name: str = VANILLA_ACCURACY, confidence_level: float = 0.95):
        self.acc_name = acc_name if acc_name == VANILLA_ACCURACY else None
        self.confidence_level = confidence_level
        self._last_lascal = None

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> "AC":
        self.n_classes = val.n_classes
        # kept with their dtype, as CBPE passes them to the lascal package, so that the calibration is the same
        self.val_posteriors = np.asarray(posteriors)
        self.val_y = np.asarray(val.y)
        return self

    def switch_and_fit(self, acc_fn, data, posteriors) -> "AC":
        acc_name = acc_fn.__name__
        if acc_name == "vanilla_acc":
            acc_name = VANILLA_ACCURACY
        self.acc_name = acc_name if acc_name == VANILLA_ACCURACY else None
        return self.fit(data, posteriors)

    def calibrated_confidences(self, posteriors: np.ndarray) -> np.ndarray:
        """The largest calibrated posterior of each instance of the bag; sets the LasCal columns of the bag."""
        calibrated, temperature, fallback = lascal_calibrate(self.val_posteriors, self.val_y, posteriors)
        self._last_lascal = {"lascal_temperature": temperature, "lascal_fallback": fallback}
        # a softmax can exceed 1 by a rounding error, which PoiBin rejects
        return np.clip(calibrated.max(axis=1), 0.0, 1.0)

    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        if self.acc_name is None:
            self._last_lascal = None
            return np.nan
        return poisson_binomial_pmf(self.calibrated_confidences(posteriors))

    def last_lascal_columns(self) -> dict | None:
        return self._last_lascal

    def ci_from_accs(self, accs: np.ndarray, confidence_level: float | None = None) -> CentralPoissonInterval:
        level = self.confidence_level if confidence_level is None else confidence_level
        return CentralPoissonInterval(accs, confidence_level=level)

    def predict(self, X, posteriors, oracle_prev=None) -> float:
        return self.ci_from_accs(self.predict_range(X, posteriors)).point_estimate
