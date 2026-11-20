r"""The gate of the solver of E7.2a (``docs/ext7_kdey_bandwidth.md``; work-plan card WP-26): may the bandwidth of
``BayCAP-KDEy-mlp-real-cal`` be selected with the standard KDEy, QuaPy's ``KDEyML`` (SLSQP), in place of EM?

Scope: the pilot pairs of E7.2a, the datasets ``mammographic``, ``wine-quality`` and ``poker_hand`` (``PILOT_DATASETS``)
with their five pretrained classifiers, in the classic domain. For each pair, the method ``BayCAP-KDEy-mlp-real-cal`` of
``main`` (``experiments/methods.py:cal_method``, the same seeds) receives $V$ and its posteriors exactly as in ``main``
(``experiments/main.py:_prepare_evaluation``), and :meth:`uncawe.methods.baycap_kdey.BayCAPKDEy.solver_check` runs the
selection of $b_{\text{cal}}$ with both solvers on the same split of $V$, the same $50$ calibration bags and the same
class densities: per bandwidth of the grid and bag, the estimate of $\pi$ and the log-likelihood of the bag at it, for
EM and for SLSQP, the criterion curve of each solver and its $b_{\text{cal}}$, and the wall time of each solver. The
method is then fitted on the full $V$ at the $b_{\text{cal}}$ of EM (the pilot's selection), to time the deployed fit.
No target bag is drawn or scored, no NUTS runs, and only the labels of $V$ are used; the selection does not depend on
the metric.

The time of a solver is the time of its estimates alone; the split, the MLP on $V_{\text{fit}}$, the bags and the class
densities are shared and timed apart. The tasks run in one pool of ``BAYES_N_JOBS`` workers with ``BAYES_ENV`` (one
thread per worker), the conditions of the methods of ``main``, so that the times compare with the test times stored by
the pilot (``tables/kdey_cal_solver.py`` reads them).

Outputs: ``<OUT_DIR>/uncawe/kdey_cal_solver/classic/selection/<dataset>/<classifier>/BayCAP-KDEy-mlp-real-cal.parquet``,
one row per (bandwidth of the grid, calibration bag) with the columns ``ROW_COLUMNS`` and the pair-level columns
``PAIR_COLUMNS`` repeated on every row. A file that exists is skipped.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from logging import Logger
from time import perf_counter
from traceback import print_exception

import jax
import pandas as pd

from uncawe.experiments.execution import ExperimentOutcome, run_tasks, shutdown_parallel_workers
from uncawe.experiments.main import _prepare_evaluation
from uncawe.experiments.methods import BAYES_ENV, BAYES_N_JOBS, CAL_VARIANTS, cal_method
from uncawe.experiments.metrics import gen_acc_measure
from uncawe.methods.baycap_kdey import CAL_SOLVERS
from uncawe.methods.baycap_quantifier import QuantifierBayCAP
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import get_logger, local_path

EXPERIMENT = "kdey_cal_solver"
DOMAIN = "classic"
METHOD = next(iter(CAL_VARIANTS))
# the folder in place of the metric of main: the selection does not depend on the metric
ACC_FOLDER = "selection"
# the pilot pairs of E7.2a
PILOT_DATASETS = ("mammographic", "wine-quality", "poker_hand")
# 1: first version (WP-26, gate of the solver)
SCHEMA_VERSION = 1
# per (bandwidth, bag): the bandwidth and its index in the grid, the bag, its true prevalence, and per solver the
# estimate, whether it converged (EM: within the step cap; SLSQP: SciPy's success flag) and the log-likelihood of the
# bag at it
ROW_COLUMNS = (
    "bandwidth",
    "grid_index",
    "bag",
    "true_prev",
    *(f"{solver}_{name}" for solver in CAL_SOLVERS for name in ("estimate", "converged", "loglik")),
)
# per pair: the recalls of h on V_fit; per solver its criterion at this bandwidth, its b_cal, its criterion at 0.1 and
# its time at this bandwidth (seconds); the time of the class densities at this bandwidth; the time of steps 1-3 of the
# selection and of the deployed fit at the b_cal of EM; |V| and |V_cal|
PAIR_COLUMNS = (
    "dataset",
    "classifier",
    "n_classes",
    "method",
    "recalls",
    *(f"{solver}_{name}" for solver in CAL_SOLVERS for name in ("criterion", "bandwidth_cal", "default", "seconds")),
    "densities_seconds",
    "setup_seconds",
    "refit_seconds",
    "n_val",
    "n_cal",
    "schema_version",
)

EXP = ExperimentOutcome


class SolverFrame(pd.DataFrame):
    """The rows of one pair, saved by ``experiments/execution.py:save_experiment_results`` as a parquet file."""

    @property
    def _constructor(self):
        return SolverFrame

    def save_result(self, path: str) -> None:
        self.to_parquet(path, compression="zstd")


def result_path(p: PretrainInfo) -> str:
    return local_path(p.domain, p.d_info.name, p.h_info.full_name, METHOD, ACC_FOLDER, experiment=EXPERIMENT)


def solver_frame(p: PretrainInfo, check: dict, refit_seconds: float) -> SolverFrame:
    """One row per (bandwidth, bag) of the result of ``BayCAPKDEy.solver_check``, with the pair-level columns."""
    grid = check["grid"]
    rows = []
    for g, bandwidth in enumerate(grid):
        for s, prevalence in enumerate(check["prevalences"]):
            row = {"bandwidth": float(bandwidth), "grid_index": g, "bag": s, "true_prev": prevalence.tolist()}
            for solver in CAL_SOLVERS:
                row |= {
                    f"{solver}_estimate": check[f"{solver}_estimates"][g, s].tolist(),
                    f"{solver}_converged": bool(check[f"{solver}_converged"][g, s]),
                    f"{solver}_loglik": float(check[f"{solver}_loglik"][g, s]),
                    f"{solver}_criterion": float(check[f"{solver}_curve"][g]),
                    f"{solver}_bandwidth_cal": check[f"{solver}_bandwidth"],
                    f"{solver}_default": check[f"{solver}_default"],
                    f"{solver}_seconds": float(check[f"{solver}_seconds"][g]),
                }
            row |= {"densities_seconds": float(check["densities_seconds"][g])}
            rows.append(row)
    frame = SolverFrame(rows)
    frame["dataset"] = p.d_info.name
    frame["classifier"] = p.h_info.name
    frame["n_classes"] = int(p.d_info.n_classes)
    frame["method"] = METHOD
    frame["recalls"] = [check["recalls"].tolist()] * len(frame)
    frame["setup_seconds"] = float(check["setup_seconds"])
    frame["refit_seconds"] = float(refit_seconds)
    frame["n_val"] = check["n_val"]
    frame["n_cal"] = check["n_cal"]
    frame["schema_version"] = SCHEMA_VERSION
    return frame[list(PAIR_COLUMNS[:4]) + list(ROW_COLUMNS) + list(PAIR_COLUMNS[4:])]


def protocol(info_path: str) -> list[ExperimentOutcome]:
    """One task: the gate on the pair of ``info_path``. The caches of JAX are cleared after the task, also when it fails
    (``experiments/main.py:exp_protocol``)."""
    try:
        return _run_task(info_path)
    finally:
        jax.clear_caches()


def _run_task(info_path: str) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(info_path, fast=True)
    if os.path.exists(result_path(p)):
        return [EXP.EXISTS(p, ACC_FOLDER, METHOD, experiment=EXPERIMENT)]
    _, acc_fn = next(gen_acc_measure(p.d_info.n_classes > 2))
    method = cal_method(acc_fn, METHOD)
    try:
        _, validation, posteriors, _ = _prepare_evaluation(p, method)
        started = perf_counter()
        check = method.solver_check(validation, posteriors)
        t_check = perf_counter() - started
        # the deployed fit at the b_cal of EM, as in the pilot (the selection is not repeated)
        method.cal_bandwidth_ = check["em_bandwidth"]
        started = perf_counter()
        QuantifierBayCAP.fit(method, validation, posteriors)
        refit_seconds = perf_counter() - started
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, ACC_FOLDER, METHOD, experiment=EXPERIMENT)]
    frame = solver_frame(p, check, refit_seconds)
    return [EXP.SUCCESS(p, ACC_FOLDER, METHOD, df=frame, t_train=t_check, t_test_ave=0.0, experiment=EXPERIMENT)]


def tasks(datasets=PILOT_DATASETS, classifiers=None, log: Logger | None = None) -> list[str]:
    """The pretrain files of ``datasets`` and ``classifiers`` (default: all) whose result is not stored, by decreasing
    number of classes."""
    found = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in datasets or (classifiers is not None and p.h_info.name not in classifiers):
            continue
        if os.path.exists(result_path(p)):
            if log is not None:
                log.info(f"[{p.h_info.name}@{p.d_info.name}] exists, skipping")
            continue
        found.append((path, p))
    missing = sorted(set(datasets) - {p.d_info.name for _, p in found})
    if missing and log is not None:
        log.info(f"no task for {missing} (stored or no pretrain file)")
    found.sort(key=lambda item: (-item[1].d_info.n_classes, item[1].d_info.name, item[1].h_info.name))
    return [path for path, _ in found]


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="The gate of the solver of E7.2a: EM against QuaPy's KDEyML (WP-26).")
    parser.add_argument("--datasets", nargs="+", default=list(PILOT_DATASETS), help=f"Default: {list(PILOT_DATASETS)}")
    parser.add_argument("--classifiers", nargs="+", default=None, help="Default: all")
    return parser.parse_args(argv)


def main() -> None:
    pargs = parse_args()
    log = get_logger(id=EXPERIMENT)
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        todo = tasks(pargs.datasets, pargs.classifiers, log)
        log.info(f"{len(todo)} tasks to run")
        run_tasks(log, todo, worker=protocol, n_jobs=BAYES_N_JOBS, experiment=EXPERIMENT, env=BAYES_ENV)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
