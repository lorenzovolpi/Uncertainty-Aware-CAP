"""Run the BayCAP methods of ``main`` on the binary datasets of ``quapy`` that ``main`` does not use (work-plan card
WP-7, E2.5 of ``docs/ext2_baycap.md``).

E2.5 tests whether the failures of the empirical-Bayes prior on the recalls found on the binary datasets of ``main``
(E2.4) appear on every binary problem. It runs the four methods of ``experiments/methods.py:BAYCAP_VARIANTS``:
``BayesCAP`` and ``BayesCAP-EB`` target the population table $C\\operatorname{diag}(\\pi^Q)$, ``BayesCAP-real`` and
``BayesCAP-EB-real`` the realised table of the target sample; all are scored downstream against the realised table of
each bag (``true_cts``).

The protocol is that of ``main`` (:func:`uncawe.experiments.main.exp_protocol`): for each pair pretrained by
``pretrain/classic_binary.py`` (domain ``classic_binary``), 100 UPP bags of size 1000 drawn from $U$ with the seed of
``main``, methods fitted on all of $V$, the same records, metrics and NUTS diagnostics. Only the results folder differs:
``<OUT_DIR>/uncawe/eb_binary/classic_binary/<acc_name>/<dataset>/<classifier>/<method>.parquet``, so the grids, tables
and ranks of ``main`` are unchanged. All tasks run in the BayCAP pool of ``main`` (``BAYES_N_JOBS`` workers,
``BAYES_ENV``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from functools import partial
from logging import Logger
from traceback import print_exception

from cap.models.confidence import CAPWithConfidence

from uncawe.experiments.execution import run_tasks, shutdown_parallel_workers
from uncawe.experiments.main import exp_protocol
from uncawe.experiments.methods import BAYCAP_VARIANTS, BAYES_ENV, BAYES_N_JOBS, gen_methods
from uncawe.experiments.metrics import get_acc_names
from uncawe.pretrain.classic_binary import DATASETS, DOMAIN
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import all_results_exist, get_logger

EXPERIMENT = "eb_binary"
METHODS = list(BAYCAP_VARIANTS)


def experiment_args(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[tuple[str, str, CAPWithConfidence]]:
    """Return the (pretrain info path, method name, method) tasks with results still missing; ``datasets``,
    ``classifiers`` and ``methods`` restrict the run (e.g. a pilot) without changing the protocol."""
    method_names = METHODS if methods is None else methods
    unknown = sorted(set(method_names) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; E2.5 runs only {METHODS}")
    paths = []
    for path in sorted(load_info_paths(domain=DOMAIN)):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in DATASETS:
            continue
        if datasets is not None and p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        if all_results_exist(p.domain, p.d_info.name, p.h_info.full_name, method_names, get_acc_names(), EXPERIMENT):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
        else:
            paths.append(path)
    return [(path, name, method) for path in paths for name, method in gen_methods() if name in method_names]


def experiments(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    methods: list[str] | None = None,
) -> None:
    tasks = experiment_args(log, datasets, classifiers, methods)
    if not tasks:
        log.info(f"no pretrained pair of {DOMAIN} needs a run; pretrain first with pretrain/classic_binary.py")
    worker = partial(exp_protocol, experiment=EXPERIMENT)
    run_tasks(log, tasks, worker=worker, n_jobs=BAYES_N_JOBS, experiment=EXPERIMENT, env=BAYES_ENV)


def main() -> None:
    log = get_logger(id=EXPERIMENT)
    parser = ArgumentParser(description="BayCAP methods of main on the binary datasets of quapy outside main (E2.5).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Run only these datasets (default: all)")
    parser.add_argument("--classifiers", nargs="+", default=None, help="Run only these classifiers (default: all)")
    parser.add_argument("--methods", nargs="+", default=None, help=f"Run only these methods (default: {METHODS})")
    pargs = parser.parse_args()

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log, pargs.datasets, pargs.classifiers, pargs.methods)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
