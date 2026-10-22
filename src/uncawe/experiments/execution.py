"""Provide shared task execution, outcome reporting, persistence, and worker cleanup."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from logging import Logger
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
    total: int | None = None,
    done: int = 0,
) -> None:
    """Save and log the outcomes; with ``total``, each line ends with the task count ``[<n>/<total>]`` of the pool,
    where ``done`` tasks were already saved before ``results_gen``."""
    for n, results in enumerate(results_gen, start=done + 1):
        count = f" [{n}/{total}]" if total is not None else ""
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
                log.info(f"{label} on {result.acc_name} done [{timestamp(result.t_train, result.t_test_ave)}]{count}")
            elif result.old:
                log.info(f"{label} on {result.acc_name} exists, skipping{count}")
            elif result.error():
                log.warning(f"{label}: {result.acc_name} gave error '{result.err}' - skipping{count}")


Task = TypeVar("Task")


@contextmanager
def worker_environment(env: Mapping[str, str]) -> Iterator[None]:
    """Set ``env`` in ``os.environ`` and restore the previous values on exit.

    Worker processes started inside the context copy the environment, so ``env`` applies to them (e.g. thread limits,
    ``XLA_FLAGS``); joblib keeps the thread-limit variables that are already set instead of its own default.
    """
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_tasks(
    log: Logger,
    tasks: list[Task],
    *,
    worker: Callable[[Task], Sequence[ExperimentOutcome]],
    n_jobs: int,
    experiment: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run ``tasks`` with ``n_jobs`` workers and save their outcomes as they arrive.

    With ``n_jobs > 1`` the workers are started inside :func:`worker_environment` with ``env`` and shut down before and
    after the pool, so that no worker is reused with the environment of another pool. With ``n_jobs == 1`` the tasks
    run in this process and ``env`` is not applied, because thread pools started earlier in this process would not see
    it.
    """
    if not tasks:
        return

    # one task is one (pair, method); its log lines end with the count of tasks of the pool done so far
    if n_jobs == 1:
        for done, task in enumerate(tasks):
            save_experiment_results(log, [worker(task)], experiment=experiment, total=len(tasks), done=done)
        return

    shutdown_parallel_workers(log)
    # the results are consumed inside the context, because joblib may start workers while they are read
    with worker_environment(env or {}):
        results_gen = parallel(
            func=worker,
            args_list=tasks,
            n_jobs=n_jobs,
            return_as="generator_unordered",
            max_nbytes=None,
        )
        save_experiment_results(log, results_gen, experiment=experiment, total=len(tasks))
    shutdown_parallel_workers(log)


def shutdown_parallel_workers(log: Logger) -> None:
    try:
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception as error:
        log.warning(f"Could not shut down joblib workers: {error}")
