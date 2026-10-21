"""Provide shared task execution, outcome reporting, persistence, and worker cleanup."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum
from logging import Logger
from time import time
from typing import TypeVar

from cap.utils.commons import parallel
from joblib.externals.loky import get_reusable_executor

from uncawe.pretrain.data import PretrainInfo
from uncawe.results import RDF
from uncawe.util import local_path, timestamp


class OutcomeStatus(IntEnum):
    SUCCESS = 200
    EXISTS = 300
    ERROR = 400


@dataclass
class ExperimentOutcome:
    code: int
    p: PretrainInfo
    acc_name: str
    method_name: str
    df: RDF | None = None
    t_train: float | None = None
    t_test_ave: float | None = None
    err: Exception | None = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return cls(OutcomeStatus.SUCCESS, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return cls(OutcomeStatus.EXISTS, *args, **kwargs)

    @classmethod
    def ERROR(cls, error: Exception, *args, **kwargs):
        return cls(OutcomeStatus.ERROR, *args, err=error, **kwargs)

    @property
    def ok(self) -> bool:
        return self.code == OutcomeStatus.SUCCESS

    @property
    def old(self) -> bool:
        return self.code == OutcomeStatus.EXISTS

    def error(self) -> bool:
        return self.code == OutcomeStatus.ERROR


def save_experiment_results(
    log: Logger,
    results_gen: Iterable[Sequence[ExperimentOutcome]],
    *,
    experiment: str,
) -> None:
    for results in results_gen:
        for result in results:
            label = f"[{result.p.h_info.name}@{result.p.d_info.name}] {result.method_name}"
            if result.ok:
                path = local_path(
                    result.p.domain,
                    result.p.d_info.name,
                    result.p.h_info.full_name,
                    result.method_name,
                    result.acc_name,
                    experiment=experiment,
                )
                result.df.save_result(path)
                log.info(f"{label} on {result.acc_name} done [{timestamp(result.t_train, result.t_test_ave)}]")
            elif result.old:
                log.info(f"{label} on {result.acc_name} exists, skipping")
            elif result.error():
                log.warning(f"{label}: {result.acc_name} gave error '{result.err}' - skipping")


Task = TypeVar("Task")


def log_progress(log: Logger, done: int, total: int, started: float, *, experiment: str) -> None:
    """Log how many tasks of a pool are done, the elapsed time and a linear estimate of the remaining time."""
    elapsed = time() - started
    remaining = elapsed / done * (total - done)
    log.info(
        f"[{experiment}] progress: {done}/{total} tasks done, elapsed {timedelta(seconds=round(elapsed))}, "
        f"remaining ~{timedelta(seconds=round(remaining))}"
    )


def _with_progress(log: Logger, results_gen: Iterable, total: int, *, experiment: str) -> Iterable:
    """Yield the results unchanged, logging progress once each one has been consumed (saved)."""
    started = time()
    for done, results in enumerate(results_gen, start=1):
        yield results
        log_progress(log, done, total, started, experiment=experiment)


def run_tasks(
    log: Logger,
    tasks: list[Task],
    *,
    worker: Callable[[Task], Sequence[ExperimentOutcome]],
    n_jobs: int,
    experiment: str,
) -> None:
    if not tasks:
        return

    # one task is one (pair, method); tasks whose results exist end at once, so early estimates are optimistic
    if n_jobs == 1:
        started = time()
        for done, task in enumerate(tasks, start=1):
            save_experiment_results(log, [worker(task)], experiment=experiment)
            log_progress(log, done, len(tasks), started, experiment=experiment)
        return

    results_gen = parallel(
        func=worker,
        args_list=tasks,
        n_jobs=n_jobs,
        return_as="generator_unordered",
        max_nbytes=None,
    )
    save_experiment_results(
        log, _with_progress(log, results_gen, len(tasks), experiment=experiment), experiment=experiment
    )


def shutdown_parallel_workers(log: Logger) -> None:
    try:
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception as error:
        log.warning(f"Could not shut down joblib workers: {error}")
