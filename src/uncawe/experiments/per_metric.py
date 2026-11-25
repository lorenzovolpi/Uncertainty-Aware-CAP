r"""The evaluation of a contingency-table method on the bags of a task, with one fit for every metric or, for the
methods of WP-34 (``methods/tcal.py:CalibratedTemperature``), one deployed fit per temperature selected for a metric.

``main``, ``target_size`` (E4.4) and ``binform`` (E5.5) predict every bag once with the fitted method and compute the
three metrics from the same tables. A ``CalibratedTemperature`` selects $\tau$ per metric, so each metric is scored on
the tables of the base fitted at its own $\tau_{\text{cal}}$: the bags are predicted once per distinct deployed fit
(:func:`bag_runs`) and each metric reads the run of its fit (:func:`metric_run`), with the per-bag columns of the
selection (:func:`metric_columns`). For every other method there is one run, the one of the method itself, so the
results are those of the loop before WP-34.
"""

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from time import time

import numpy as np

from uncawe.methods.tcal import CalibratedTemperature


@dataclass
class BagRun:
    method: object  # the fitted method that predicted the bags
    estim_ct_ranges: list
    estim_cts: list
    diagnostics: list[dict]
    t_test_ave: float

    def diagnostic_columns(self) -> dict[str, list]:
        """The per-bag diagnostics as columns (empty if the method stores none)."""
        if not self.diagnostics:
            return {}
        return {name: [d[name] for d in self.diagnostics] for name in self.diagnostics[0]}


def prepare(method, dataset: str):
    """Pass the name of the dataset to a method that needs it before ``fit`` (``CalibratedTemperature``)."""
    if isinstance(method, CalibratedTemperature):
        method.set_dataset(dataset)
    return method


def predict_bags(
    method, bags: Callable[[], Iterable[tuple]], n_bags: int, diagnostics: Callable[[object], dict | None]
) -> BagRun:
    """Predict the contingency tables of every bag of ``bags()`` (pairs of instances and posteriors) with ``method``;
    ``diagnostics(method)`` gives the per-bag columns after a prediction, or ``None`` if the method stores none. As in
    ``main``, the time of the diagnostics is left out of the test time."""
    started, t_diagnostics = time(), 0.0
    estim_ct_ranges, rows = [], []
    for X, posteriors in bags():
        estim_ct_ranges.append(method.predict_ct_range(X, posteriors))
        started_diagnostics = time()
        row = diagnostics(method)
        if row is not None:
            rows.append(row)
            t_diagnostics += time() - started_diagnostics
    t_test_ave = (time() - started - t_diagnostics) / n_bags
    estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    return BagRun(method, estim_ct_ranges, estim_cts, rows, t_test_ave)


def bag_runs(
    method, bags: Callable[[], Iterable[tuple]], n_bags: int, diagnostics: Callable[[object], dict | None]
) -> dict[Hashable, BagRun]:
    """The runs of the fitted ``method`` on the bags: one per deployed fit of a ``CalibratedTemperature``, one (key
    ``None``) for every other method."""
    if isinstance(method, CalibratedTemperature):
        return {key: predict_bags(fit, bags, n_bags, diagnostics) for key, fit in method.deployed().items()}
    return {None: predict_bags(method, bags, n_bags, diagnostics)}


def metric_run(method, runs: dict[Hashable, BagRun], acc_name: str) -> BagRun:
    """The run on which ``acc_name`` is scored."""
    if isinstance(method, CalibratedTemperature):
        return runs[method.deployed_key(acc_name)]
    return runs[None]


def metric_columns(method, acc_name: str, size: int) -> dict[str, list]:
    """The per-bag columns of the selection of $\\tau$ for ``acc_name`` (none for a method other than
    ``CalibratedTemperature``)."""
    if not isinstance(method, CalibratedTemperature):
        return {}
    return {name: [value] * size for name, value in method.metric_columns(acc_name).items()}


def is_contingency_method(method) -> bool:
    """Whether ``main`` evaluates ``method`` as a contingency-table method (``CalibratedTemperature`` predicts with its
    deployed fits, which are)."""
    return isinstance(method, CalibratedTemperature)
