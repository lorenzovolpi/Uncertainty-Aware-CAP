r"""A tempered BayCAP method whose temperature $\tau$ is selected per metric on calibration bags drawn from half of $V$,
inside ``fit`` (work-plan card WP-34; the protocol of E5.3 of ``docs/ext5_baycap_priors_tempering.md``, study
``tau_select``).

``CalibratedTemperature`` wraps a base method built by ``base_factory(temperature=..., bandwidth=...)`` (a tempered
``BayCAP`` or ``BayCAPKDEy``). ``fit(V, posteriors)``:

1. splits $V$ per class into $V_{\text{cal}}$ ($\lfloor n_j/2\rfloor$ instances of class $j$) and $V_{\text{fit}}$ (the
   other $\lceil n_j/2\rceil$), with the permutation seeded by the experiment seed and the dataset (``split_index``);
2. draws ``n_bags`` calibration bags of ``cal_bag_size`` instances from $V_{\text{cal}}$ with QuaPy's UPP, seeded by
   the experiment seed and the dataset (``calibration_bags``); the realised table of a bag comes from the labels of
   $V_{\text{cal}}$ and the predictions of $h$;
3. at each $\tau$ of ``taus``, builds the base at $\tau$, fits it on $V_{\text{fit}}$ and scores its intervals on the
   bags against their realised tables, for every metric of ``metrics`` (Winkler score of the central interval at level
   ``confidence``, as ``tables/main.py:get_acc_winkler``; a bag without draws gives no score);
4. per metric, $\tau_{\text{cal}}$ is the $\tau$ with the lowest mean Winkler score over the bags, ties to the $\tau$
   closest to $1$ on the $\log$ scale, then to the smaller (``best_tau``, as ``tables/tempering.py:best_tau``);
5. fits the base on the full $V$ at each distinct $\tau_{\text{cal}}$ (the deployed fits, ``deployed``).

With a base that selects its bandwidth in the fit (``BayCAP-KDEy-h-inst-cb``, ``bandwidth="logloss"``), $b^\star$ is
selected on $V_{\text{fit}}$ at every $\tau$ (the same value, the selection does not depend on $\tau$), and the deployed
fits reuse it (``bandwidth="reuse"``, the default) or select it again on the full $V$ (``bandwidth="reselect"``).

With ``fixed_tau`` the method selects no temperature: it is the base at that $\tau$ fitted on the full $V$, with the
bandwidth selected on $V_{\text{fit}}$ (``bandwidth="reuse"``); no calibration bag is drawn. These are the methods of the
test grid of ``BayCAP-KDEy-h-inst-cb-tcal-keep`` (E5.2 and E5.3 with the bandwidth of $V_{\text{fit}}$). Where $V$
cannot be split, the bandwidth is selected on the full $V$ (``tau_fallback``, reason ``split``) and $\tau$ stays the
fixed one.

The method has no prediction of its own: the experiments predict with the deployed fit of each metric
(``deployed_for``) and store the per-bag columns of ``metric_columns``. If $V$ cannot be split (a class with fewer than
$2$ instances), or if no $\tau$ gives a finite calibration score for a metric, that metric falls back to $\tau=1$ (the
base itself, with its bandwidth selected on the full $V$), with ``tau_fallback`` set (user's decision of 2026-11-25).

The split and the bags depend on the dataset, which a method does not see: the experiment passes its name with
``set_dataset`` before ``fit``. Every fit is a new base built by ``base_factory``; NUTS starts from
``PRNGKey(random_state)`` on every bag, so the calibration runs are those of ``experiments/tau_select.py`` and the
deployed fit at $\tau$ is the method of E5.2 at $\tau$, for the same seed. The intervals target the realised table of
the target sample; the selection uses only the labels of $V$.
"""

import zlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import time
from typing import Literal

import jax
import numpy as np
import quapy as qp
from cap.models.confidence import ConfidenceInterval
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.protocol import UPP

# the grid of E5.3 (experiments/tau_select.py:TAUS), the reference temperature and the calibration bags
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0)
REFERENCE_TAU = 1.0
N_BAGS = 50
CAL_BAG_SIZE = 1000
CONFIDENCE = 0.95
BANDWIDTH_OPTIONS = ("reuse", "reselect")
# streams of the seed sequences of a dataset (experiments/tau_select.py)
_SPLIT_STREAM = 0
_BAG_STREAM = 1
# the reasons of a fallback to tau=1
FALLBACK_SPLIT = "split"
FALLBACK_NO_SCORE = "no-score"


# --------------------------------------------------------------------------------------------------- seeds


def _seed_words(dataset: str, stream: int, *words: int) -> list[int]:
    """The entropy of a seed sequence: the experiment seed, the ``zlib.crc32`` of the dataset name, the stream and
    ``words``; it depends on nothing else (not on the classifier, the method, $\\tau$ or the global random state)."""
    return [int(qp.environ["_R_SEED"]), zlib.crc32(dataset.encode()), stream, *map(int, words)]


def _seed(dataset: str, stream: int, *words: int) -> int:
    return int(np.random.SeedSequence(_seed_words(dataset, stream, *words)).generate_state(1)[0])


def split_seed(dataset: str) -> int:
    """The seed of the split of $V$ of ``dataset``: a 32-bit integer of the experiment seed and the dataset only."""
    return _seed(dataset, _SPLIT_STREAM)


def calibration_seed(dataset: str) -> int:
    """The seed of the UPP protocol of the calibration bags of ``dataset``: of the experiment seed and the dataset only
    (shared by every classifier, method and $\\tau$)."""
    return _seed(dataset, _BAG_STREAM)


# --------------------------------------------------------------------------------------------------- split of V


def split_index(val_y: np.ndarray, n_classes: int, dataset: str) -> tuple[np.ndarray, np.ndarray]:
    """Indices into $V$ of $V_{\\text{fit}}$ and $V_{\\text{cal}}$, each in increasing order.

    The instances of each class are permuted with one generator seeded by ``split_seed(dataset)`` (class by class, in
    class order); the first $\\lfloor n_j/2\\rfloor$ go to $V_{\\text{cal}}$ and the others to $V_{\\text{fit}}$. Raises
    ``ValueError`` if a class would have no instance in one of the halves.
    """
    val_y = np.asarray(val_y)
    counts = np.bincount(val_y, minlength=n_classes)
    if np.any(counts < 2):
        raise ValueError(
            f"{dataset}: every class needs at least 2 instances in V to be split; counts {counts.tolist()}"
        )
    rng = np.random.default_rng(split_seed(dataset))
    fit, cal = [], []
    for j in range(n_classes):
        permuted = rng.permutation(np.flatnonzero(val_y == j))
        n_cal = counts[j] // 2
        cal.append(permuted[:n_cal])
        fit.append(permuted[n_cal:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(cal))


# --------------------------------------------------------------------------------------------------- calibration bags


@dataclass(frozen=True)
class CalibrationBag:
    bag_index: int
    prevalence: np.ndarray  # of the bag
    index: np.ndarray  # indices into V
    true_ct: np.ndarray  # cap-ml layout, true classes on rows


def calibration_bags(
    dataset: str,
    val_y: np.ndarray,
    val_pred: np.ndarray,
    cal_index: np.ndarray,
    n_classes: int,
    n_bags: int = N_BAGS,
    size: int = CAL_BAG_SIZE,
) -> list[CalibrationBag]:
    """The ``n_bags`` calibration bags of ``dataset``, drawn from $V_{\\text{cal}}$ (``cal_index``, indices into $V$) by
    QuaPy's UPP with ``random_state=calibration_seed(dataset)``, as ``main`` draws its test bags from $U$; the realised
    table comes from the labels and the predictions of $h$ on $V$. The bags depend on the labels of $V_{\\text{cal}}$
    only, not on ``val_pred``."""
    val_y, val_pred, cal_index = np.asarray(val_y), np.asarray(val_pred), np.asarray(cal_index)
    # the instances of the collection are the indices into V, so that each sample carries them
    cal = LabelledCollection(cal_index.reshape(-1, 1), val_y[cal_index], classes=np.arange(n_classes))
    protocol = UPP(
        cal, sample_size=size, repeats=n_bags, random_state=calibration_seed(dataset), return_type="labelled_collection"
    )
    # QuaPy's temp_seed writes the seed of the protocol into qp.environ["_R_SEED"] and does not restore it; the
    # experiment seed must stay as it is (it seeds the next calibration seed and the methods built after the bags)
    experiment_seed = qp.environ["_R_SEED"]
    try:
        samples = list(protocol())
    finally:
        qp.environ["_R_SEED"] = experiment_seed
    bags = []
    for b, sample in enumerate(samples):
        index = np.asarray(sample.X).ravel().astype(int)
        true_ct = contingency_table(val_y[index], val_pred[index], n_classes)
        bags.append(CalibrationBag(b, np.asarray(sample.prevalence(), dtype=float), index, true_ct))
    return bags


# --------------------------------------------------------------------------------------------------- selection


def best_tau(mean_winkler: dict[float, float]) -> float:
    """The $\\tau$ with the lowest value of ``mean_winkler`` ($\\tau$ -> mean Winkler score); ties go to the $\\tau$
    closest to $1$ on the log scale, then to the smaller one; NaN if every value is NaN (as
    ``tables/tempering.py:best_tau``)."""
    finite = {tau: w for tau, w in mean_winkler.items() if np.isfinite(w)}
    if not finite:
        return np.nan
    lowest = min(finite.values())
    return min((tau for tau, w in finite.items() if w == lowest), key=lambda tau: (abs(np.log(tau)), tau))


def bag_winkler(samples, true_acc: float, confidence: float = CONFIDENCE) -> float:
    """The Winkler score of the central interval of ``samples`` at level ``confidence`` for the true value
    ``true_acc``; NaN for a bag without draws (any non-finite sample) or a non-finite true value, as the tables of
    E5.3 (``tables/tempering.py:bag_frame``)."""
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size == 0 or not np.all(np.isfinite(samples)) or not np.isfinite(true_acc):
        return np.nan
    return float(ConfidenceInterval(samples, confidence_level=confidence).winkler(true_acc))


def _key(tau: float) -> float:
    return float(tau)


class CalibratedTemperature:
    """A tempered method with $\\tau$ selected per metric on calibration bags from half of $V$ (see the module).

    :param base_factory: builds the base method: ``base_factory(temperature=tau)``, and, for a base that selects its
        bandwidth in the fit, ``base_factory(temperature=tau, bandwidth=b)`` with a fixed bandwidth.
    :param metrics: ``metrics(multiclass)`` gives the pairs (name, function of the contingency table) of the metrics
        (``experiments/metrics.py:gen_acc_measure``), the functions in the ``cap-ml`` layout.
    :param taus: the grid of $\\tau$.
    :param n_bags: the number of calibration bags.
    :param cal_bag_size: the size of a calibration bag ($1000$, the size of the bags of ``main``; E4.4a passes the size
        of the target bags).
    :param bandwidth: for a base that selects its bandwidth: ``"reuse"`` the $b^\\star$ of $V_{\\text{fit}}$ in the
        deployed fits, or ``"reselect"`` it on the full $V$.
    :param confidence: the level of the intervals scored.
    :param clear_caches: clear the caches of JAX after the calibration runs of each $\\tau$ (each bag compiles its own
        sampler; the results do not change).
    :param fixed_tau: if given, no selection: the base at this $\\tau$ with the bandwidth of $V_{\\text{fit}}$ (see the
        module); ``taus``, ``n_bags`` and ``cal_bag_size`` are then not used.
    """

    def __init__(
        self,
        base_factory: Callable,
        metrics: Callable[[bool], Iterable[tuple[str, Callable]]],
        taus: Iterable[float] = TAUS,
        n_bags: int = N_BAGS,
        cal_bag_size: int = CAL_BAG_SIZE,
        bandwidth: Literal["reuse", "reselect"] = "reuse",
        confidence: float = CONFIDENCE,
        clear_caches: bool = True,
        fixed_tau: float | None = None,
    ):
        if bandwidth not in BANDWIDTH_OPTIONS:
            raise ValueError(f"bandwidth must be one of {BANDWIDTH_OPTIONS}; found {bandwidth!r}")
        if fixed_tau is not None:
            if not float(fixed_tau) > 0:
                raise ValueError(f"fixed_tau must be positive; found {fixed_tau!r}")
            if bandwidth != "reuse":
                raise ValueError('fixed_tau keeps the bandwidth of V_fit: it needs bandwidth="reuse"')
            taus = (float(fixed_tau),)
        taus = tuple(sorted(float(t) for t in taus))
        if not taus or any(not t > 0 for t in taus) or len(set(taus)) != len(taus):
            raise ValueError(f"taus must be distinct positive numbers; found {taus}")
        if int(n_bags) < 1 or int(cal_bag_size) < 1:
            raise ValueError(f"n_bags and cal_bag_size must be positive; found {n_bags}, {cal_bag_size}")
        self.base_factory = base_factory
        self.metrics = metrics
        self.taus = taus
        self.n_bags = int(n_bags)
        self.cal_bag_size = int(cal_bag_size)
        self.bandwidth = bandwidth
        self.confidence = confidence
        self.clear_caches = clear_caches
        self.fixed_tau = None if fixed_tau is None else float(fixed_tau)
        self.dataset = None

    def set_dataset(self, name: str) -> "CalibratedTemperature":
        """The name of the dataset, which seeds the split of $V$ and the calibration bags."""
        self.dataset = str(name)
        return self

    # ----------------------------------------------------------------------------------------------- fit

    @staticmethod
    def _selected_bandwidth(method) -> float:
        """The bandwidth selected in the fit by ``method`` (``bandwidth="logloss"``), NaN if it selects none."""
        quantifier = getattr(method, "quantifier_", None)
        if quantifier is not None and getattr(quantifier, "selects_logloss", False):
            return float(quantifier.logloss_bandwidth_)
        return np.nan

    def _calibrate(self, val: LabelledCollection, posteriors: np.ndarray, metrics: list[tuple[str, Callable]]):
        """Steps 1–4: the calibration scores per metric and $\\tau$ and the bandwidth selected on $V_{\\text{fit}}$;
        ``None`` if $V$ cannot be split."""
        val_y = np.asarray(val.y)
        n_classes = len(val.classes_)
        val_pred = np.argmax(posteriors, axis=-1)
        try:
            fit_index, cal_index = split_index(val_y, n_classes, self.dataset)
        except ValueError:
            return None
        self.v_fit_size_, self.v_cal_size_ = int(fit_index.size), int(cal_index.size)
        v_fit = LabelledCollection(val.X[fit_index], val_y[fit_index], classes=val.classes_)
        bags = calibration_bags(
            self.dataset, val_y, val_pred, cal_index, n_classes, n_bags=self.n_bags, size=self.cal_bag_size
        )
        true_accs = {name: np.array([fn(bag.true_ct) for bag in bags]) for name, fn in metrics}
        scores = {name: {} for name, _ in metrics}
        bandwidths = []
        for tau in self.taus:
            method = self.base_factory(temperature=tau).fit(v_fit, posteriors[fit_index])
            bandwidths.append(self._selected_bandwidth(method))
            cts = [method.predict_ct_range(val.X[bag.index], posteriors[bag.index]) for bag in bags]
            for name, fn in metrics:
                method.switch(fn)
                values = [
                    bag_winkler(method.ci_from_cts(ct).samples, true, self.confidence)
                    for ct, true in zip(cts, true_accs[name])
                ]
                # the mean over the bags with draws, as the tables of E5.3 (pandas' mean skips NaN)
                scores[name][_key(tau)] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else np.nan
            if self.clear_caches:
                jax.clear_caches()
        bandwidths = np.asarray(bandwidths, dtype=float)
        if np.any(np.isfinite(bandwidths)) and not np.allclose(bandwidths, bandwidths[0], rtol=0.0, atol=0.0):
            raise RuntimeError(f"the bandwidth selected on V_fit differs between the temperatures: {bandwidths}")
        return scores, float(bandwidths[0])

    def _bandwidth_of_v_fit(self, val: LabelledCollection, posteriors: np.ndarray):
        """For ``fixed_tau``: the scores (NaN, no calibration bag) and the bandwidth selected by the base on
        $V_{\\text{fit}}$ (a fit without NUTS); ``None`` if $V$ cannot be split."""
        val_y = np.asarray(val.y)
        try:
            fit_index, cal_index = split_index(val_y, len(val.classes_), self.dataset)
        except ValueError:
            return None
        self.v_fit_size_, self.v_cal_size_ = int(fit_index.size), int(cal_index.size)
        v_fit = LabelledCollection(val.X[fit_index], val_y[fit_index], classes=val.classes_)
        method = self.base_factory(temperature=self.fixed_tau).fit(v_fit, posteriors[fit_index])
        scores = {name: {_key(self.fixed_tau): np.nan} for name in self.metric_names_}
        return scores, self._selected_bandwidth(method)

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> "CalibratedTemperature":
        if self.dataset is None:
            raise RuntimeError("set_dataset must be called before fit: the dataset seeds the split of V and the bags")
        started = time()
        metrics = list(self.metrics(len(val.classes_) > 2))
        self.metric_names_ = [name for name, _ in metrics]
        self.v_fit_size_ = self.v_cal_size_ = 0
        if self.fixed_tau is not None:
            calibration = self._bandwidth_of_v_fit(val, posteriors)
        else:
            calibration = self._calibrate(val, posteriors, metrics)
        if calibration is None:
            # the fallback: tau=1 (or the fixed tau), the base with its bandwidth selected on the full V
            fallback_tau = REFERENCE_TAU if self.fixed_tau is None else self.fixed_tau
            self.cal_scores_ = {name: {_key(t): np.nan for t in self.taus} for name in self.metric_names_}
            self.bandwidth_fit_ = np.nan
            self.tau_cal_ = {name: fallback_tau for name in self.metric_names_}
            self.fallback_ = {name: FALLBACK_SPLIT for name in self.metric_names_}
        elif self.fixed_tau is not None:
            self.cal_scores_, self.bandwidth_fit_ = calibration
            self.tau_cal_ = {name: self.fixed_tau for name in self.metric_names_}
            self.fallback_ = {name: "" for name in self.metric_names_}
        else:
            self.cal_scores_, self.bandwidth_fit_ = calibration
            self.tau_cal_, self.fallback_ = {}, {}
            for name in self.metric_names_:
                tau = best_tau(self.cal_scores_[name])
                self.fallback_[name] = "" if np.isfinite(tau) else FALLBACK_NO_SCORE
                self.tau_cal_[name] = float(tau) if np.isfinite(tau) else REFERENCE_TAU
        self.t_calibration_ = time() - started

        # step 5: one deployed fit per distinct tau_cal; a fallback (and a base without a selected bandwidth) is the
        # base itself, fitted on the full V; otherwise the bandwidth of V_fit is reused unless "reselect"
        self.deployed_, self.bandwidth_deployed_ = {}, {}
        for name in self.metric_names_:
            key = self._deployed_key(name)
            if key in self.deployed_:
                continue
            tau, reused = key
            options = {"bandwidth": self.bandwidth_fit_} if reused else {}
            method = self.base_factory(temperature=tau, **options).fit(val, posteriors)
            self.deployed_[key] = method
            selected = self._selected_bandwidth(method)
            self.bandwidth_deployed_[key] = self.bandwidth_fit_ if reused else selected
        return self

    def _deployed_key(self, acc_name: str) -> tuple[float, bool]:
        """The deployed fit of ``acc_name``: its $\\tau_{\\text{cal}}$ and whether it reuses the bandwidth of
        $V_{\\text{fit}}$."""
        reused = self.bandwidth == "reuse" and not self.fallback_[acc_name] and np.isfinite(self.bandwidth_fit_)
        return _key(self.tau_cal_[acc_name]), bool(reused)

    # ----------------------------------------------------------------------------------------------- deployed fits

    def deployed(self) -> dict[tuple[float, bool], object]:
        """The deployed fits, by key (:meth:`deployed_key`)."""
        return dict(self.deployed_)

    def deployed_key(self, acc_name: str) -> tuple[float, bool]:
        """The key of the deployed fit of ``acc_name`` in :meth:`deployed`."""
        if acc_name not in self.tau_cal_:
            raise KeyError(f"no temperature selected for {acc_name}; the metrics are {self.metric_names_}")
        return self._deployed_key(acc_name)

    def deployed_for(self, acc_name: str):
        """The fitted base at $\\tau_{\\text{cal}}$ of ``acc_name``."""
        return self.deployed_[self.deployed_key(acc_name)]

    def metric_columns(self, acc_name: str) -> dict:
        """The per-bag columns of the method for ``acc_name`` (the same on every bag of a pair): $\\tau_{\\text{cal}}$
        of this metric and of each metric, the fallback and its reason, the mean calibration Winkler score of this
        metric at each $\\tau$ (``cal_winkler``, aligned with ``cal_taus``), the sizes of the split and of the
        calibration bags, the seeds, and the bandwidth selected on $V_{\\text{fit}}$ and that of the deployed fit (NaN
        for a base without a selected bandwidth)."""
        key = self.deployed_key(acc_name)
        columns = {
            "tau_cal": self.tau_cal_[acc_name],
            "tau_fallback": bool(self.fallback_[acc_name]),
            "tau_fallback_reason": self.fallback_[acc_name],
            "cal_taus": list(self.taus),
            "cal_winkler": [self.cal_scores_[acc_name][_key(t)] for t in self.taus],
            "cal_n_bags": 0 if self.fixed_tau is not None else self.n_bags,
            "cal_bag_size": 0 if self.fixed_tau is not None else self.cal_bag_size,
            "v_fit_size": self.v_fit_size_,
            "v_cal_size": self.v_cal_size_,
            "split_seed": split_seed(self.dataset),
            "calibration_seed": calibration_seed(self.dataset),
            "bandwidth_fit": self.bandwidth_fit_,
            "bandwidth_deployed": self.bandwidth_deployed_[key],
            "t_calibration": self.t_calibration_,
        }
        columns |= {f"tau_cal_{name}": tau for name, tau in self.tau_cal_.items()}
        return columns
