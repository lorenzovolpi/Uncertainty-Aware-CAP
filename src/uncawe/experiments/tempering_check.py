"""Check that the stored untempered runs of E5.2 are reproduced by the current code (work-plan card WP-20).

E5.2 of ``docs/ext5_baycap_priors_tempering.md`` sweeps the temperature $\\tau$ of ``BayesCAP-Hf-real`` and
``BayCAP-KDEy-mlp-real`` on the protocol of ``main``. Its runs at $\\tau=1$ are the stored ``main`` results of the two
methods, which were produced by earlier commits (most of those of ``BayesCAP-Hf-real`` before work-plan card WP-13
changed ``methods/baycap.py``). They are reused only if the current code reproduces them.

``--run`` runs the two untempered methods with the protocol of ``main`` (:func:`uncawe.experiments.main.exp_protocol`:
the same 100 UPP bags of size 1000, seeds, records and NUTS diagnostics) on the selected pairs, in the BayCAP pool of
``main`` (``BAYES_N_JOBS`` workers, ``BAYES_ENV``), and writes them to
``<OUT_DIR>/uncawe/main_e52_check/<domain>/<acc_name>/<dataset>/<classifier>/<method>.parquet``, so that ``main`` is
untouched. Both methods target the realised table of the target sample and are scored against it (``true_cts``).

``--compare`` compares every file of ``main_e52_check`` with the file of ``main`` of the same cell, bag by bag, on the
true tables, the performance samples, the point estimates and the estimated tables, and writes the comparison to
``<OUT_DIR>/uncawe/tables/tempering_check/comparison.csv``. A cell is reproduced when every compared value is equal bit
for bit; the stored runs at $\\tau=1$ are reused only if every checked cell is reproduced.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from functools import partial
from glob import glob
from logging import Logger
from traceback import print_exception

import numpy as np
import pandas as pd
from cap.models.confidence import CAPWithConfidence

from uncawe import env
from uncawe.experiments.datasets import get_filtered_datasets
from uncawe.experiments.execution import run_tasks, shutdown_parallel_workers
from uncawe.experiments.main import EXPERIMENT as MAIN_EXPERIMENT
from uncawe.experiments.main import exp_protocol
from uncawe.experiments.methods import BAYES_ENV, BAYES_N_JOBS, TEMPERED_BASES, gen_methods
from uncawe.experiments.metrics import get_acc_names
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.results import ResultsDataFrame
from uncawe.util import all_results_exist, get_logger

EXPERIMENT = "main_e52_check"
DOMAIN = "classic"
METHODS = list(TEMPERED_BASES)
# the per-bag columns compared with main: the scored table, the performance samples, the point estimate, the estimated
# table
COMPARED = ("true_cts", "estim_acc_ranges", "estim_accs", "estim_cts")
TABLE_DIR = "tempering_check"


def experiment_args(
    log: Logger,
    datasets: list[str],
    classifiers: list[str] | None = None,
    methods: list[str] | None = None,
) -> list[tuple[str, str, CAPWithConfidence]]:
    """Return the (pretrain info path, method name, method) tasks of the selected pairs with results still missing."""
    method_names = METHODS if methods is None else methods
    unknown = sorted(set(method_names) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; the check runs only {METHODS}")
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets() or p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        if all_results_exist(p.domain, p.d_info.name, p.h_info.full_name, method_names, get_acc_names(), EXPERIMENT):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
        else:
            paths.append(path)
    return [(path, name, method) for path in paths for name, method in gen_methods() if name in method_names]


def run(log: Logger, datasets: list[str], classifiers: list[str] | None, methods: list[str] | None) -> None:
    tasks = experiment_args(log, datasets, classifiers, methods)
    worker = partial(exp_protocol, experiment=EXPERIMENT)
    run_tasks(log, tasks, worker=worker, n_jobs=BAYES_N_JOBS, experiment=EXPERIMENT, env=BAYES_ENV)


def _values(column: pd.Series) -> list[np.ndarray]:
    return [np.asarray(value, dtype=float) for value in column]


def compare_frames(checked: pd.DataFrame, stored: pd.DataFrame) -> dict:
    """Compare two result frames of the same cell, bag by bag, on ``COMPARED``: the number of bags, and for each column
    the number of bags that differ and the largest absolute difference (NaN in the same places counts as equal)."""
    row = {"n_bags": len(checked), "n_bags_stored": len(stored)}
    same = len(checked) == len(stored)
    for column in COMPARED:
        if not same or column not in checked or column not in stored:
            row[f"{column}_n_diff"], row[f"{column}_max_abs_diff"] = np.nan, np.nan
            continue
        n_diff, max_diff = 0, 0.0
        for a, b in zip(_values(checked[column]), _values(stored[column])):
            if a.shape != b.shape:
                n_diff, max_diff = n_diff + 1, np.inf
                continue
            if not np.array_equal(a, b, equal_nan=True):
                n_diff += 1
                # NaN in only one of the two is an infinite difference
                one_nan = np.isnan(a) ^ np.isnan(b)
                diff = np.inf if one_nan.any() else float(np.nanmax(np.abs(a - b), initial=0.0))
                max_diff = max(max_diff, diff)
        row[f"{column}_n_diff"], row[f"{column}_max_abs_diff"] = n_diff, max_diff
    row["reproduced"] = same and all(row[f"{column}_n_diff"] == 0 for column in COMPARED)
    return row


def comparison(root: str) -> pd.DataFrame:
    """One row per checked cell (domain, metric, dataset, classifier, method) of ``main_e52_check`` under ``root``, with
    :func:`compare_frames` against the file of ``main`` of the same cell (``stored_exists`` false if there is none)."""
    rows = []
    checked_root = os.path.join(root, EXPERIMENT)
    for path in sorted(glob(os.path.join(checked_root, "*", "*", "*", "*", "*.parquet"))):
        relative = os.path.relpath(path, checked_root)
        domain, acc_name, dataset, classifier, filename = relative.split(os.sep)
        key = {
            "domain": domain,
            "acc_name": acc_name,
            "dataset": dataset,
            "classifier": classifier,
            "method": filename.removesuffix(".parquet"),
        }
        stored_path = os.path.join(root, MAIN_EXPERIMENT, relative)
        if not os.path.exists(stored_path):
            rows.append(key | {"stored_exists": False, "reproduced": False})
            continue
        checked = ResultsDataFrame.load_result(path)
        stored = ResultsDataFrame.load_result(stored_path)
        rows.append(key | {"stored_exists": True} | compare_frames(checked, stored))
    return pd.DataFrame(rows)


def compare(log: Logger) -> pd.DataFrame:
    table = comparison(env.root_dir)
    out_dir = os.path.join(env.root_dir, "tables", TABLE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    if table.empty:
        log.info(f"no result in {EXPERIMENT}; run with --run first")
        return table
    summary = table.groupby("method")["reproduced"].agg(["sum", "count"])
    for method, (n_same, n_cells) in summary.iterrows():
        log.info(f"{method}: {n_same} of {n_cells} cells reproduced")
    log.info("all checked cells reproduced" if table["reproduced"].all() else "some checked cells are NOT reproduced")
    return table


def main() -> None:
    log = get_logger(id=EXPERIMENT)
    parser = ArgumentParser(description="Check that the stored untempered runs of E5.2 are reproduced (WP-20).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help=f"Run {METHODS} into {EXPERIMENT}")
    mode.add_argument("--compare", action="store_true", help=f"Compare {EXPERIMENT} with {MAIN_EXPERIMENT}")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to run (required with --run)")
    parser.add_argument("--classifiers", nargs="+", default=None, help="Run only these classifiers (default: all)")
    parser.add_argument("--methods", nargs="+", default=None, help=f"Run only these methods (default: {METHODS})")
    pargs = parser.parse_args()
    if pargs.run and not pargs.datasets:
        parser.error("--run needs --datasets")

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        if pargs.run:
            run(log, pargs.datasets, pargs.classifiers, pargs.methods)
        else:
            compare(log)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
