"""Order the tasks of a pool longest first, from the times stored with earlier results.

A task of ``main`` is one (pair, method). Its cost is estimated from the result files of the same (domain, dataset,
classifier, method) in reference result folders (``main`` and its backups, e.g. ``main_pre_fix3``): for each metric
file, ``t_train + n_bags * t_test_ave``. Methods that fit once for all the metrics write the same two times in the three
files, so equal pairs of times are counted once; methods that fit per metric write different times, which are summed.

Running the longest tasks first shortens a pool whose tasks differ a lot in length (the longest-processing-time rule):
a long task started last would otherwise run alone at the end. The order changes only when and in which worker a task
runs, as the scheduling of the pool already does; the tasks and their seeded randomness are unchanged.
"""

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from logging import Logger
from typing import NamedTuple, TypeVar

import pandas as pd

from uncawe import env

Task = TypeVar("Task")

TIME_COLUMNS = ["t_train", "t_test_ave"]


class TaskKey(NamedTuple):
    domain: str
    dataset: str
    classifier: str
    method: str


def default_reference_experiments(root_dir: str | None = None, experiment: str = "main") -> list[str]:
    """``experiment`` itself, then every folder of ``root_dir`` named ``<experiment>_*``, sorted by name."""
    root_dir = env.root_dir if root_dir is None else root_dir
    backups = []
    if os.path.isdir(root_dir):
        backups = sorted(
            d
            for d in os.listdir(root_dir)
            if d.startswith(f"{experiment}_") and os.path.isdir(os.path.join(root_dir, d))
        )
    return [experiment] + backups


def stored_cost(
    key: TaskKey, acc_names: Sequence[str], experiments: Sequence[str], root_dir: str | None = None
) -> float | None:
    """Estimated seconds of the task ``key`` from the first reference folder that has a result file of it."""
    root_dir = env.root_dir if root_dir is None else root_dir
    for experiment in experiments:
        times = set()
        for acc_name in acc_names:
            path = os.path.join(
                root_dir, experiment, key.domain, acc_name, key.dataset, key.classifier, f"{key.method}.parquet"
            )
            if not os.path.exists(path):
                continue
            try:
                frame = pd.read_parquet(path, columns=TIME_COLUMNS)
            except Exception:
                continue
            if frame.empty:
                continue
            times.add((float(frame["t_train"].iloc[0]), float(frame["t_test_ave"].mean()), len(frame)))
        if times:
            return sum(t_train + n_bags * t_bag for t_train, t_bag, n_bags in times)
    return None


def estimate_costs(
    keys: Sequence[TaskKey], acc_names: Sequence[str], experiments: Sequence[str], root_dir: str | None = None
) -> tuple[list[float | None], list[str]]:
    """Cost of each key and its source: ``stored`` (its own result files), ``classifiers`` (the mean over the other
    classifiers of the same dataset and method) or ``unknown`` (no estimate, cost ``None``)."""
    costs = [stored_cost(key, acc_names, experiments, root_dir) for key in keys]
    by_dataset_method = defaultdict(list)
    for key, cost in zip(keys, costs):
        if cost is not None:
            by_dataset_method[(key.domain, key.dataset, key.method)].append(cost)
    sources = []
    for i, key in enumerate(keys):
        if costs[i] is not None:
            sources.append("stored")
            continue
        others = by_dataset_method.get((key.domain, key.dataset, key.method))
        if others:
            costs[i] = sum(others) / len(others)
            sources.append("classifiers")
        else:
            sources.append("unknown")
    return costs, sources


def sort_longest_first(
    tasks: Sequence[Task],
    key_of: Callable[[Task], TaskKey],
    acc_names: Sequence[str],
    experiments: Sequence[str],
    log: Logger | None = None,
    root_dir: str | None = None,
) -> list[Task]:
    """``tasks`` sorted by decreasing estimated cost; tasks with no estimate come first, in their original order, since
    they may be long. The sort is stable, so tasks with equal costs keep their order."""
    if not tasks:
        return list(tasks)
    keys = [key_of(task) for task in tasks]
    costs, sources = estimate_costs(keys, acc_names, experiments, root_dir)
    order = sorted(range(len(tasks)), key=lambda i: (costs[i] is not None, -(costs[i] or 0.0)))
    if log is not None:
        known = [c for c in costs if c is not None]
        counts = {s: sources.count(s) for s in ("stored", "classifiers", "unknown")}
        log.info(
            f"longest first: {len(tasks)} tasks, estimates from {list(experiments)}: "
            f"{counts['stored']} stored, {counts['classifiers']} from other classifiers, {counts['unknown']} unknown "
            f"(run first); estimated total {sum(known) / 3600:.1f} h, longest {max(known, default=0.0) / 3600:.1f} h"
        )
        for i in order[:5]:
            cost = "unknown" if costs[i] is None else f"{costs[i] / 3600:.2f} h"
            k = keys[i]
            log.info(f"longest first: [{k.classifier}@{k.dataset}] {k.method}: {cost} ({sources[i]})")
    return [tasks[i] for i in order]
