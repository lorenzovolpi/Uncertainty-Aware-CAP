"""The rate matrix of $V$ rescaled by a point quantifier, on the protocol of ``main`` (E8.2 of
``docs/ext8_average_confidence.md``, card WP-30).

Under prior probability shift the rates $C_{ij}=P(\\hat Y=\\omega_i\\mid Y=\\omega_j)$ of $h$ do not change, so the table
of the target is $T=C\\operatorname{diag}(\\pi^Q)$ (§T.7). This experiment measures the plainest point estimate built on
that identity, $\\hat T=\\hat C\\operatorname{diag}(\\hat\\pi^Q)$, with $\\hat C$ the rates of $h$ on $V$ and $\\hat\\pi^Q$
the estimate of a point quantifier: `cap-ml`'s ``ContTableTransferCAP``, which normalises the table of $V$ to sum to $1$
and multiplies its true-class rows by $\\hat\\pi^Q/\\pi^V$ (the code stores the transpose of the paper's table, true
classes on rows). No uncertainty, no interval.

Levels (``LEVELS``):

- ``CTT-KDEy``, ``CTT-EMQ``, ``CTT-ACC``, ``CTT-PACC``: QuaPy 0.2.4's ``KDEyML``, ``EMQ``, ``ACC`` and ``PACC`` at their
  default arguments on ``MLP(random_state=qp.environ["_R_SEED"])`` trained on $V$ (the classifier and seed of
  ``experiments/methods.py:kdey``, ``pacc``; ``CTT-ACC`` added on 2026-11-23 by the user's decision); they target the
  population table $C\\operatorname{diag}(\\pi^Q)$;
- ``CTT-oracle`` (**reference**, it uses the target labels, never a method): the table of $V$ rescaled by the true
  prevalence of the bag $\\bar\\pi$, i.e. $\\hat C\\operatorname{diag}(\\bar\\pi)$; it targets the expected realised table
  given the class counts of the bag.

All levels are scored against the realised table of each bag (``true_cts``), as every method of ``main``.

Protocol: that of ``main`` (``experiments/main.py:_prepare_evaluation``): the pairs of ``main``, $V$, the posteriors of
$h$, 100 UPP bags of size 1000 per pair with the seed of ``main``. One task is one (pair, level).

Outputs: ``<OUT_DIR>/uncawe/ctt/<domain>/<acc_name>/<dataset>/<classifier>/<level>.parquet``, one file per pair, level
and metric (vanilla accuracy, macro-F1, macro-K), skipped if present, with the schema of ``main``
(``experiments/records.py``; ``estim_acc_ranges`` is None, ``estim_cts`` the estimated table) and the per-bag columns

- ``test_prev``: $\\hat\\pi^Q$ (for ``CTT-oracle``, $\\bar\\pi$); ``bag_prev``: $\\bar\\pi$, the row sums of ``true_cts``;
- ``q_l1``: the quantification error $\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$ ($0$ for ``CTT-oracle``);
- ``q_failed``: the quantifier raised or gave a non-finite estimate on the bag (estimate NaN);
- vanilla accuracy only: ``e_pi`` and ``e_c``, the quantification part and the rate part of the error $\\hat A-A$
  (§T.8), and ``recall_range``, the range $R$ of the recalls of $h$ on $V$.

A class absent from $V$ ($\\pi^V_j=0$) makes the rescaling divide by zero: the task raises a ``ValueError`` naming the
classes, which the run logs as an error, and writes no file.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from logging import Logger
from time import time
from traceback import print_exception

import numpy as np
import quapy as qp
from cap import env as capenv
from cap.error import vanilla_acc
from cap.models.cont_table import ContTableTransferCAP
from quapy.method.aggregative import ACC, EMQ
from sklearn.neural_network import MLPClassifier as MLP

from uncawe.experiments.datasets import get_filtered_datasets
from uncawe.experiments.execution import ExperimentOutcome, run_tasks, shutdown_parallel_workers
from uncawe.experiments.main import NUM_TEST, _prepare_evaluation
from uncawe.experiments.methods import SAMPLE_SIZE, THREAD_LIMITS, kdey, pacc
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import all_results_exist, get_logger, local_path

EXPERIMENT = "ctt"
DOMAIN = "classic"
ACC_SPLIT = "vanilla_accuracy"

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0

EXP = ExperimentOutcome


def emq():
    """QuaPy's EMQ in its base form (no recalibration, the MLP trained on the whole $V$, the training prevalence that of
    $V$) on the seeded MLP of the quantifiers of ``main``."""
    return EMQ(MLP(random_state=qp.environ["_R_SEED"]))


def acc():
    """QuaPy's ACC in its base form ($5$-fold cross-fitting of the misclassification rates on $V$, solver ``minimize``,
    ``norm="clip"``) on the seeded MLP of the quantifiers of ``main``: PACC with crisp instead of soft predictions."""
    return ACC(MLP(random_state=qp.environ["_R_SEED"]))


QUANTIFIERS = {"CTT-KDEy": kdey, "CTT-EMQ": emq, "CTT-ACC": acc, "CTT-PACC": pacc}
ORACLE = "CTT-oracle"
METHODS = tuple(QUANTIFIERS)
LEVELS = (*METHODS, ORACLE)
# the pilot: one binary and two multiclass datasets of main, LR, every bag
PILOT_DATASETS = ("mammographic", "poker_hand", "letter")
PILOT_CLASSIFIERS = ("LR",)


def absent_classes(validation) -> np.ndarray:
    """The classes with no instance in $V$, where ``ContTableTransferCAP`` would divide by $\\pi^V_j=0$."""
    return np.flatnonzero(np.asarray(validation.counts()) == 0)


def check_validation(validation) -> None:
    absent = absent_classes(validation)
    if len(absent):
        raise ValueError(f"classes {absent.tolist()} are absent from V: the rates of h on V are undefined for them")


def validation_table(validation, validation_posteriors) -> tuple[np.ndarray, np.ndarray]:
    """The table of $h$ on $V$ normalised to sum to $1$, true classes on rows ($\\operatorname{diag}(\\pi^V)\\hat C^\\top$),
    and $\\pi^V$, as ``ContTableTransferCAP`` builds them."""
    check_validation(validation)
    ctt = ContTableTransferCAP(vanilla_acc, None)
    ctt.preprocess_data(validation, validation_posteriors)
    return ctt.cont_table, ctt.train_prev


def val_recalls(table: np.ndarray, val_prev: np.ndarray) -> np.ndarray:
    """The recalls of $h$ on $V$, $\\hat C_{jj}=N^V_{jj}/n^V_j$, from the normalised table (true classes on rows)."""
    return np.diag(table) / val_prev


def bag_prevalence(true_ct: np.ndarray) -> np.ndarray:
    """The prevalence of the bag $\\bar\\pi_j=n_j/n$: the row sums of the normalised true table (true classes on rows),
    i.e. the column sums of the paper's table."""
    return np.asarray(true_ct, dtype=float).sum(axis=1)


def oracle_table(table: np.ndarray, val_prev: np.ndarray, true_ct: np.ndarray) -> np.ndarray:
    """``CTT-oracle``: the table of $V$ with row $j$ multiplied by $\\bar\\pi_j/\\pi^V_j$, i.e. $\\hat
    C\\operatorname{diag}(\\bar\\pi)$ transposed. Uses the labels of the bag: a reference, never a method."""
    return table * (bag_prevalence(true_ct) / val_prev)[:, np.newaxis]


def error_split(recalls: np.ndarray, test_prev: np.ndarray, true_ct: np.ndarray) -> tuple[float, float]:
    """The two parts of $\\hat A-A$ for vanilla accuracy (§T.8): $e_\\pi=\\sum_j(\\hat\\pi^Q_j-\\bar\\pi_j)\\hat C_{jj}$ and
    $e_C=\\sum_j\\bar\\pi_j(\\hat C_{jj}-\\bar C_{jj})=\\sum_j\\bar\\pi_j\\hat C_{jj}-\\sum_jN_{jj}/n$, so that no recall of the
    bag (undefined for a class absent from it) is needed."""
    true_ct = np.asarray(true_ct, dtype=float)
    bag_prev = bag_prevalence(true_ct)
    e_pi = float(np.sum((np.asarray(test_prev) - bag_prev) * recalls))
    e_c = float(np.sum(bag_prev * recalls) - np.trace(true_ct))
    return e_pi, e_c


def predict_tables(level: str, dataset, validation, validation_posteriors, true_cts) -> dict:
    """The estimated table of each bag for ``level`` (true classes on rows), with the flags of the bags where the
    quantifier failed, the table and prevalence of $V$, and the fit and average prediction times."""
    check_validation(validation)
    started = time()
    if level == ORACLE:
        table, val_prev = validation_table(validation, validation_posteriors)
        t_train = time() - started
        started = time()
        tables = [oracle_table(table, val_prev, ct) for ct in true_cts]
        failed = [False] * len(tables)
    else:
        method = ContTableTransferCAP(vanilla_acc, QUANTIFIERS[level]())
        method.fit(validation, validation_posteriors)
        table, val_prev = method.cont_table, method.train_prev
        t_train = time() - started
        started = time()
        tables, failed = [], []
        for sample, posteriors in zip(dataset.test_prot(), dataset.test_prot_posteriors):
            try:
                estimate = np.asarray(method.predict_ct(sample.X, posteriors), dtype=float)
                ok = bool(np.all(np.isfinite(estimate)))
            except Exception as error:
                print_exception(error)
                estimate, ok = None, False
            tables.append(estimate if ok else np.full(table.shape, np.nan))
            failed.append(not ok)
    t_test_ave = (time() - started) / len(tables)
    return {
        "tables": tables,
        "failed": failed,
        "val_table": table,
        "val_prev": np.asarray(val_prev, dtype=float),
        "t_train": t_train,
        "t_test_ave": t_test_ave,
    }


def bag_columns(prediction: dict, true_cts) -> dict[str, list]:
    """The per-bag columns shared by every metric: $\\hat\\pi^Q$ (the row sums of the estimated table, since row $j$ of
    the table of $V$ sums to $\\pi^V_j$), $\\bar\\pi$, $\\lVert\\hat\\pi^Q-\\bar\\pi\\rVert_1$ and the failure flag."""
    test_prev = [t.sum(axis=1) for t in prediction["tables"]]
    bag_prev = [bag_prevalence(ct) for ct in true_cts]
    return {
        "test_prev": [p.tolist() for p in test_prev],
        "bag_prev": [p.tolist() for p in bag_prev],
        "q_l1": [float(np.abs(p - b).sum()) for p, b in zip(test_prev, bag_prev)],
        "q_failed": list(prediction["failed"]),
    }


def split_columns(prediction: dict, true_cts) -> dict[str, list]:
    """The per-bag columns of vanilla accuracy: $e_\\pi$, $e_C$ (NaN where the quantifier failed) and the range $R$ of
    the recalls of $V$. ``vanilla_acc`` divides the trace of the estimated table by its sum, so $\\hat\\pi^Q$ is
    normalised to sum to $1$ here (it differs from $1$ only by the precision of the quantifier, about $10^{-11}$), and
    $e_\\pi+e_C$ equals the stored $\\hat A-A$ to rounding."""
    recalls = val_recalls(prediction["val_table"], prediction["val_prev"])
    parts = [
        error_split(recalls, t.sum(axis=1) / t.sum(), ct) if not failed else (np.nan, np.nan)
        for t, ct, failed in zip(prediction["tables"], true_cts, prediction["failed"])
    ]
    return {
        "e_pi": [e[0] for e in parts],
        "e_c": [e[1] for e in parts],
        "recall_range": [float(recalls.max() - recalls.min())] * len(parts),
    }


def _result_path(p: PretrainInfo, level: str, acc_name: str, experiment: str = EXPERIMENT) -> str:
    return local_path(p.domain, p.d_info.name, p.h_info.full_name, level, acc_name, experiment=experiment)


def evaluate_level(
    p: PretrainInfo,
    level: str,
    dataset,
    validation,
    validation_posteriors,
    context: EvaluationContext,
    experiment: str = EXPERIMENT,
) -> list[ExperimentOutcome]:
    """Fit ``level`` once on $V$, estimate the table of every bag and score it on every metric."""
    try:
        prediction = predict_tables(level, dataset, validation, validation_posteriors, context.true_cts)
        common = bag_columns(prediction, context.true_cts)
        split = split_columns(prediction, context.true_cts)
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, "ct", level)]

    results = []
    for acc_name, acc_fn in gen_acc_measure(p.d_info.n_classes > 2):
        if os.path.exists(_result_path(p, level, acc_name, experiment)):
            results.append(EXP.EXISTS(p, acc_name, level))
            continue
        estim_accs = [np.nan if f else float(acc_fn(t)) for t, f in zip(prediction["tables"], prediction["failed"])]
        frame = build_result_frame(
            context,
            method_name=level,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=[None] * context.size,
            estim_accs=estim_accs,
            estim_cts=list(prediction["tables"]),
            t_train=prediction["t_train"],
            t_test_ave=prediction["t_test_ave"],
            extra_columns=common | (split if acc_name == ACC_SPLIT else {}),
        )
        results.append(
            EXP.SUCCESS(
                p, acc_name, level, df=frame, t_train=prediction["t_train"], t_test_ave=prediction["t_test_ave"]
            )
        )
    return results


def exp_task(args: tuple[str, str], experiment: str = EXPERIMENT) -> list[ExperimentOutcome]:
    """One task: the level ``level`` on the pair of ``info_path``."""
    info_path, level = args
    p = PretrainInfo.load(info_path, fast=True)
    if all(os.path.exists(_result_path(p, level, acc_name, experiment)) for acc_name in get_acc_names()):
        return [EXP.EXISTS(p, "all", level)]
    # the protocol of main; no method of main needs a validation protocol here, so V is the whole validation set
    dataset, validation, validation_posteriors, context = _prepare_evaluation(p, None)
    return evaluate_level(p, level, dataset, validation, validation_posteriors, context, experiment)


def experiment_args(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    levels: list[str] | None = None,
) -> list[tuple[str, str]]:
    """The (pretrain info path, level) tasks with results still missing; ``datasets``, ``classifiers`` and ``levels``
    restrict the run (e.g. the pilot) without changing the protocol."""
    levels = list(LEVELS) if levels is None else levels
    unknown = sorted(set(levels) - set(LEVELS))
    if unknown:
        raise ValueError(f"Unknown levels: {unknown}; E8.2 runs {list(LEVELS)}")
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets():
            continue
        if datasets is not None and p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        if all_results_exist(p.domain, p.d_info.name, p.h_info.full_name, levels, get_acc_names(), EXPERIMENT):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
        else:
            paths.append(path)
    return [(path, level) for path in paths for level in levels]


def experiments(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    levels: list[str] | None = None,
) -> None:
    tasks = experiment_args(log, datasets, classifiers, levels)
    run_tasks(log, tasks, worker=exp_task, n_jobs=capenv["N_JOBS"], experiment=EXPERIMENT, env=THREAD_LIMITS)


def main() -> None:
    log = get_logger(id=EXPERIMENT)
    parser = ArgumentParser(description="E8.2: the rate matrix of V rescaled by a point quantifier (card WP-30).")
    parser.add_argument("--pilot", action="store_true", help=f"The pilot: {list(PILOT_DATASETS)} with LR")
    parser.add_argument("--datasets", nargs="+", default=None, help="Run only these datasets (default: all)")
    parser.add_argument("--classifiers", nargs="+", default=None, help="Run only these classifiers (default: all)")
    parser.add_argument("--levels", nargs="+", default=None, help=f"Run only these levels (default: {list(LEVELS)})")
    pargs = parser.parse_args()
    if pargs.pilot:
        if pargs.datasets is not None or pargs.classifiers is not None:
            parser.error("--pilot sets the datasets and the classifiers; do not pass --datasets or --classifiers")
        pargs.datasets, pargs.classifiers = list(PILOT_DATASETS), list(PILOT_CLASSIFIERS)

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        log.info(f"E8.2 on {NUM_TEST} bags per pair")
        experiments(log, pargs.datasets, pargs.classifiers, pargs.levels)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
