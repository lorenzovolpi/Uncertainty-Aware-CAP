r"""The study ``tau_select``: the temperature $\tau$ of BayCAP-Hf and BayCAP-KDEy chosen per pair on calibration bags
drawn from a held-out half of $V$ (E5.3 of ``docs/ext5_baycap_priors_tempering.md``; work-plan card WP-23).

Scope: the $15$ datasets of ``main`` (``get_filtered_datasets``) and their five pretrained classifiers, in the classic
domain; the methods ``BayesCAP-Hf-real`` (BayCAP with the Hf prior on $C$, both likelihood terms divided by $\tau$) and,
since WP-33, ``BayCAP-KDEy-h-inst`` and ``BayCAP-KDEy-h-inst-cb`` (BayesKDEy with the input classifier $h$, a fixed
$\hat C$ and the per-instance pushforward, the likelihood of the bag divided by $\tau$ and the weights untempered; with
``-cb`` the bandwidth selected in the fit on $V_{\text{fit}}$) (``experiments/methods.py:TEMPERED_BASES``; until WP-33
``BayCAP-KDEy-mlp-real``, with the cell pushforward, whose results are invalidated), all with a realised pushforward, built as in ``main`` and E5.2 (``experiments/methods.py``), at
$\tau\in\{0.5,1,2,4,8\}$ (``TAUS``). Both target the realised table of each calibration bag and are scored against it
(``true_cts``, computed from the labels of $V_{\text{cal}}$). $h$ is never retrained and the split $L/V$ of the pretrain
files is unchanged.

Purpose: the per-bag measures on the calibration bags, from which ``tables/tau_select.py`` chooses $\tau_{\text{cal}}$
per pair, method and metric (the lowest mean Winkler score) and reads the scores of the deployed method, fitted on the
full $V$, on the $100$ test bags of ``main`` from the stored results of ``main`` and E5.2. No test bag is run here, and
no label outside $V$ is used.

- **Split of $V$.** For each dataset, $V$ is split per class into $V_{\text{cal}}$ ($\lfloor n_j/2\rfloor$ instances of
  class $j$) and $V_{\text{fit}}$ (the other $\lceil n_j/2\rceil$), with a permutation of the instances of each class
  drawn from one generator seeded by the experiment seed and the dataset only (``split_seed``, ``split_index``): the
  same split for the five classifiers of the dataset, every method and every $\tau$. Every class must keep at least one
  instance in each half. The methods are fitted on $V_{\text{fit}}$ (BayCAP-Hf on the validation counts of $h$ on
  $V_{\text{fit}}$; BayCAP-KDEy with $\hat C$, the MLP and the kernel densities on $V_{\text{fit}}$).
- **Calibration bags.** ``n_bags`` ($S=50$) bags of ``SAMPLE_SIZE`` instances per pair, drawn from $V_{\text{cal}}$
  with QuaPy's UPP, the protocol with which ``main`` draws its test bags from $U$ (``pretrain/data.py:with_test_data``):
  the prevalences are uniform on the simplex (Kraemer algorithm), the class counts are $\operatorname{round}(|U|\pi_j)$
  with the remainder assigned to classes drawn with probabilities $\pi$, and the instances of each class are drawn
  with replacement (``LabelledCollection.sampling_index``). The protocol is seeded by ``calibration_seed``, of the
  experiment seed and the dataset only, so the bags depend neither on the classifier, nor on the method, nor on
  $\tau$, and every cell of a dataset sees the same instances; the posteriors of $h$ on a bag are the stored
  validation posteriors of its instances.

Outputs: ``<OUT_DIR>/uncawe/tau_select/<domain>/<acc_name>/<dataset>/<classifier>/<method>.parquet``, where ``method`` is
the name of the method at $\tau$ in ``main`` (the base name at $\tau=1$, ``<base>-t<tau>`` otherwise, as
``experiments/methods.py:TEMPERED_VARIANTS``); one row per calibration bag with the schema of ``main`` (the three
metrics, the NUTS diagnostics and posterior summaries that ``main`` stores for the two methods) and the columns
``base_method``, ``tau``, ``split_seed``, ``v_size``, ``v_fit_size``, ``v_cal_size``, ``v_class_counts``,
``v_fit_class_counts``, ``v_recalls`` (the recalls of $h$ on the full $V$, for $\hat\sigma_{e_2}$ of §T.9),
``bag_size``, ``bag_index``, ``calibration_seed``, ``bag_prev`` (the prevalence of the bag), ``config_hash`` and ``schema_version``. A file that
exists is skipped.

Every method is a NumPyro method, so all the tasks run in one pool with ``BAYES_N_JOBS`` workers and ``BAYES_ENV``, and
the caches of JAX are cleared after each task, as in ``main``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hashlib
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception

import jax
import numpy as np
import quapy as qp
from cap.utils.commons import get_shift
from quapy.data import LabelledCollection

from uncawe.experiments.datasets import get_filtered_datasets
from uncawe.experiments.execution import ExperimentOutcome, run_tasks, shutdown_parallel_workers
from uncawe.experiments.methods import (
    BAYES_ENV,
    BAYES_N_JOBS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    SAMPLE_SIZE,
    TEMPERATURE_LEVELS,
    TEMPERED_BASES,
    baycap_method,
    kdey_method,
)
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.methods.tcal import (  # noqa: F401 (re-exported: the split and the bags of the study)
    _BAG_STREAM,
    _SPLIT_STREAM,
    CalibrationBag,
    _seed,
    _seed_words,
    calibration_bags,
    calibration_seed,
    split_index,
    split_seed,
)
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import get_logger, get_plain_prev, local_path

EXPERIMENT = "tau_select"
DOMAIN = "classic"
METHODS = TEMPERED_BASES
REFERENCE_TAU = 1.0
TAUS = tuple(sorted((REFERENCE_TAU, *TEMPERATURE_LEVELS)))
# S, the calibration bags per pair (card WP-23)
N_BAGS = 50
# the pilot of the card
PILOT_DATASETS = ("mammographic", "poker_hand", "letter")
# 1: first version (WP-23)
SCHEMA_VERSION = 1

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0

EXP = ExperimentOutcome


def method_name(base: str, tau: float) -> str:
    """The name of ``base`` at temperature ``tau`` in ``main``: ``base`` itself at $\\tau=1$, ``<base>-t<tau>``
    otherwise (``experiments/methods.py:TEMPERED_VARIANTS``)."""
    if base not in METHODS or tau not in TAUS:
        raise ValueError(f"No method {base} at tau={tau}; the study runs {list(METHODS)} at {list(TAUS)}")
    return base if tau == REFERENCE_TAU else f"{base}-t{tau:g}"


def make_method(base: str, tau: float):
    """``base`` at the temperature ``tau``, built as in ``main`` and E5.2 (same settings and seed)."""
    method_name(base, tau)
    _, acc = next(gen_acc_measure(True))
    if base == "BayesCAP-Hf-real":
        return baycap_method(acc, base, temperature=tau)
    return kdey_method(acc, base, temperature=tau)


# --------------------------------------------------------------------------------------------------- seeds
# split_seed, calibration_seed, split_index, CalibrationBag and calibration_bags live in methods/tcal.py since WP-34
# (the methods with tau selected inside the fit draw the same split and bags), imported above with the same behaviour


def config_hash(n_bags: int, size: int = SAMPLE_SIZE) -> str:
    """A short hash of the settings that define the cells of a file: the experiment seed, $S$, the bag size, the split
    rule and the NUTS settings; files with a different hash are not comparable bag by bag."""
    config = {
        "seed": int(qp.environ["_R_SEED"]),
        "n_bags": int(n_bags),
        "bag_size": int(size),
        "split": "per class, floor(n_j/2) to V_cal",
        "bags": "quapy UPP on V_cal",
        "num_warmup": BAYES_WARMUPS,
        "num_samples": BOOTSTRAP_ITERS,
        "schema_version": SCHEMA_VERSION,
    }
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------- split of V


def recalls(y: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """The recall of each class, the share of its instances predicted as itself (NaN for a class without instances)."""
    y, pred = np.asarray(y), np.asarray(pred)
    counts = np.bincount(y, minlength=n_classes).astype(float)
    correct = np.bincount(y[y == pred], minlength=n_classes).astype(float)
    return np.divide(correct, counts, out=np.full(n_classes, np.nan), where=counts > 0)


# --------------------------------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Task:
    info_path: str
    base: str
    tau: float
    n_bags: int

    @property
    def method_name(self) -> str:
        return method_name(self.base, self.tau)


def _result_path(p: PretrainInfo, method: str, acc_name: str) -> str:
    return local_path(p.domain, p.d_info.name, p.h_info.full_name, method, acc_name, experiment=EXPERIMENT)


def _all_results_exist(p: PretrainInfo, method: str) -> bool:
    return all(os.path.exists(_result_path(p, method, acc_name)) for acc_name in get_acc_names())


def protocol(task: Task) -> list[ExperimentOutcome]:
    """One task: the method of ``task`` fitted on $V_{\\text{fit}}$ and run on every calibration bag of its pair. The
    caches of JAX are cleared after the task, also when it fails (``experiments/main.py:exp_protocol``)."""
    try:
        return _run_task(task)
    finally:
        jax.clear_caches()


def _run_task(task: Task) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(task.info_path, fast=True)
    name = task.method_name
    if _all_results_exist(p, name):
        return [EXP.EXISTS(p, "all", name, experiment=EXPERIMENT)]

    dataset = p.load_dataset_bundle()
    val_posteriors, _ = p.load_posteriors()
    if len(val_posteriors) != len(dataset.V.y):
        raise ValueError(f"{p.d_info.name}: the stored posteriors do not match V")
    n_classes = int(p.d_info.n_classes)
    val_y = np.asarray(dataset.V.y)
    val_pred = np.argmax(val_posteriors, axis=-1)
    try:
        fit_index, cal_index = split_index(val_y, n_classes, p.d_info.name)
    except ValueError as error:
        return [EXP.ERROR(error, p, "ct", name, experiment=EXPERIMENT)]
    validation = LabelledCollection(dataset.V.X[fit_index], val_y[fit_index], classes=dataset.V.classes_)
    bags = calibration_bags(p.d_info.name, val_y, val_pred, cal_index, n_classes, task.n_bags)
    context = EvaluationContext(
        pretrain=p,
        shifts=get_shift(np.array([bag.true_ct.sum(axis=1) for bag in bags]), dataset.L_prevalence).tolist(),
        true_cts=[bag.true_ct for bag in bags],
        train_prev=get_plain_prev(dataset.L_prevalence),
        val_prev=get_plain_prev(validation.prevalence()),
    )

    try:
        method = make_method(task.base, task.tau)
        started = time()
        method = method.fit(validation, val_posteriors[fit_index])
        t_train = time() - started

        # as in main: the time of the NUTS diagnostics and posterior summaries is left out of the test time
        started, t_diagnostics = time(), 0.0
        estim_ct_ranges, diagnostics = [], []
        for bag in bags:
            estim_ct_ranges.append(method.predict_ct_range(dataset.V.X[bag.index], val_posteriors[bag.index]))
            started_diagnostics = time()
            diagnostics.append(method.last_nuts_diagnostics() | method.last_posterior_summaries())
            t_diagnostics += time() - started_diagnostics
        t_test_ave = (time() - started - t_diagnostics) / context.size
        estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, "ct", name, experiment=EXPERIMENT)]

    size = context.size
    extra_columns = {key: [d[key] for d in diagnostics] for key in diagnostics[0]}
    extra_columns |= {
        "base_method": [task.base] * size,
        "tau": [float(task.tau)] * size,
        "split_seed": [split_seed(p.d_info.name)] * size,
        "v_size": [int(val_y.size)] * size,
        "v_fit_size": [int(fit_index.size)] * size,
        "v_cal_size": [int(cal_index.size)] * size,
        "v_class_counts": [np.bincount(val_y, minlength=n_classes).tolist()] * size,
        "v_fit_class_counts": [np.bincount(val_y[fit_index], minlength=n_classes).tolist()] * size,
        "v_recalls": [recalls(val_y, val_pred, n_classes).tolist()] * size,
        "bag_size": [int(bag.index.size) for bag in bags],
        "bag_index": [bag.bag_index for bag in bags],
        "calibration_seed": [calibration_seed(p.d_info.name)] * size,
        "bag_prev": [bag.prevalence.tolist() for bag in bags],
        "config_hash": [config_hash(task.n_bags)] * size,
        "schema_version": [SCHEMA_VERSION] * size,
    }

    results = []
    for acc_name, acc_fn in gen_acc_measure(n_classes > 2):
        if os.path.exists(_result_path(p, name, acc_name)):
            results.append(EXP.EXISTS(p, acc_name, name, experiment=EXPERIMENT))
            continue
        method.switch(acc_fn)
        intervals = [method.ci_from_cts(cts) for cts in estim_ct_ranges]
        frame = build_result_frame(
            context,
            method_name=name,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=[interval.samples for interval in intervals],
            estim_accs=[interval.point_estimate for interval in intervals],
            estim_cts=estim_cts,
            t_train=t_train,
            t_test_ave=t_test_ave,
            extra_columns=extra_columns,
        )
        results.append(
            EXP.SUCCESS(p, acc_name, name, df=frame, t_train=t_train, t_test_ave=t_test_ave, experiment=EXPERIMENT)
        )
    return results


def _pretrains(datasets, classifiers) -> list[tuple[str, PretrainInfo]]:
    """The pretrain files of ``datasets`` (default: those of ``main``) and ``classifiers`` (default: all), ordered by
    decreasing number of classes (the larger $K$, the longer a task), then by dataset and classifier name."""
    allowed = set(get_filtered_datasets() if datasets is None else datasets)
    found = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name in allowed and (classifiers is None or p.h_info.name in classifiers):
            found.append((path, p))
    missing = sorted(allowed - {p.d_info.name for _, p in found})
    if missing:
        raise FileNotFoundError(f"No pretrain file for {missing}")
    found.sort(key=lambda item: (-item[1].d_info.n_classes, item[1].d_info.name, item[1].h_info.name))
    return found


def tasks(
    datasets=None,
    classifiers=None,
    methods=METHODS,
    taus=TAUS,
    n_bags: int = N_BAGS,
    log: Logger | None = None,
) -> list[Task]:
    """The tasks still to run: every (pair, method, $\\tau$) whose three files are not all stored; within a pair,
    BayCAP-Hf first (the longer runs)."""
    unknown = sorted(set(methods) - set(METHODS)) + sorted(float(t) for t in set(taus) - set(TAUS))
    if unknown:
        raise ValueError(f"Unknown methods or temperatures: {unknown}")
    out = []
    for path, p in _pretrains(datasets, classifiers):
        for base in METHODS:
            if base not in methods:
                continue
            for tau in TAUS:
                if tau not in taus:
                    continue
                task = Task(path, base, float(tau), int(n_bags))
                if _all_results_exist(p, task.method_name):
                    if log is not None:
                        log.info(f"[{p.h_info.name}@{p.d_info.name}] {task.method_name} exists, skipping")
                    continue
                out.append(task)
    return out


def run(log: Logger, **kwargs) -> None:
    todo = tasks(log=log, **kwargs)
    log.info(f"{len(todo)} tasks to run")
    run_tasks(log, todo, worker=protocol, n_jobs=BAYES_N_JOBS, experiment=EXPERIMENT, env=BAYES_ENV)


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="The study tau_select: tau chosen per pair on calibration bags (E5.3, WP-23).")
    parser.add_argument("--datasets", nargs="+", default=None, help="Default: the 15 datasets of main")
    parser.add_argument(
        "--pilot", action="store_true", help=f"The pilot datasets {list(PILOT_DATASETS)}; excludes --datasets"
    )
    parser.add_argument("--classifiers", nargs="+", default=None, help="Default: all")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--taus", nargs="+", type=float, default=list(TAUS), choices=TAUS)
    parser.add_argument("--n-bags", type=int, default=N_BAGS, help=f"S, calibration bags per pair (default: {N_BAGS})")
    args = parser.parse_args(argv)
    if args.pilot:
        if args.datasets is not None:
            parser.error("--pilot sets the datasets; do not pass --datasets with it")
        args.datasets = list(PILOT_DATASETS)
    if args.n_bags < 1:
        parser.error(f"--n-bags must be positive: {args.n_bags}")
    return args


def main() -> None:
    pargs = parse_args()
    log = get_logger(id=EXPERIMENT)
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        run(
            log,
            datasets=pargs.datasets,
            classifiers=pargs.classifiers,
            methods=pargs.methods,
            taus=pargs.taus,
            n_bags=pargs.n_bags,
        )
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
