"""Run the implemented binform experiment on prior specification for BayesCAP.

The scope is the three-class ``connect-4`` dataset in the classic domain and the
configured pre-trained classifiers.  It samples 100 test prevalences from a
Dirichlet distribution whose concentration is obtained from validation
prevalence $p$ through $q_i = c(1-p_i)/(K-1)$, then constructs one test bag for
each prevalence.

The experiment compares O-LEAP and RQBS with two BayesCAP variants: a uniform
prior and a prior derived from the transformed validation prevalence.  Its
purpose is to isolate how BayesCAP's prior choice affects point estimates and
confidence intervals under this controlled prevalence-shift regime.  It emits
the same accuracy metrics and result schema as the main experiment so the
variants can be compared with the same error and interval-quality measures.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from collections.abc import Iterable
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Literal

import numpy as np
import quapy as qp
from cap import env as capenv
from cap.models.confidence import ACC_BayesCAP
from cap.utils.commons import contingency_table, get_shift
from quapy.data import LabelledCollection

from uncawe.experiments.execution import (
    ExperimentOutcome,
    run_tasks,
    shutdown_parallel_workers,
)
from uncawe.experiments.execution import (
    save_experiment_results as save_outcomes,
)
from uncawe.experiments.methods import (
    BAYES_ENV,
    BAYES_N_JOBS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    SAMPLE_SIZE,
    THREAD_LIMITS,
    gen_methods,
)
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.pretrain.data import PreTrainedClassifier, PretrainInfo, load_info_paths
from uncawe.util import get_logger, get_plain_prev, local_path

EXPERIMENT = "binform"
DOMAIN = "classic"
DATASET = "connect-4"
COLLECTION = "uci_multiclass"
NUM_TEST = 100
BAYESCAP_PRIOR_C = 3.0
TEST_PRIOR_C = 10.0

CLASSIFIERS = frozenset({"LR", "kNN", "SVM", "MLP", "RF"})
REGULAR_METHOD_NAMES = ("O-LEAP", "RQBS")
METHOD_PRIOR_KINDS = {
    "BayesCAP_uniform": "uniform",
    "BayesCAP_transformed_val": "transformed_val",
}

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0

EXP = ExperimentOutcome


@dataclass(frozen=True)
class TestSample:
    collection: LabelledCollection
    posteriors: np.ndarray
    true_ct: np.ndarray


def _transform_prior(prior: np.ndarray, c: float) -> np.ndarray:
    p = np.asarray(prior, dtype=float)
    return ((1 - p) / float(p.shape[0] - 1.0)) * c


def _method_names() -> list[str]:
    return list(REGULAR_METHOD_NAMES) + list(METHOD_PRIOR_KINDS)


def _result_path(p: PretrainInfo, method_name: str, acc_name: str) -> str:
    return local_path(
        p.domain,
        p.d_info.name,
        p.h_info.full_name,
        method_name,
        acc_name,
        experiment=EXPERIMENT,
    )


def _all_method_results_exist(p: PretrainInfo, method_name: str) -> bool:
    return all(os.path.exists(_result_path(p, method_name, acc_name)) for acc_name in get_acc_names())


def _is_binform_pretrain(p: PretrainInfo) -> bool:
    return (
        p.domain == DOMAIN
        and p.d_info.collection == COLLECTION
        and p.d_info.name == DATASET
        and p.h_info.class_name in CLASSIFIERS
    )


def _bayescap_prior(kind: Literal["uniform", "transformed_val"], val_prev: np.ndarray) -> str | np.ndarray:
    if kind == "uniform":
        return "uniform"
    if kind == "transformed_val":
        return _transform_prior(val_prev, BAYESCAP_PRIOR_C)
    raise ValueError(f"Unknown BayesCAP prior kind: {kind}")


def _make_test_samples(
    unlabelled: LabelledCollection,
    classifier: PreTrainedClassifier,
    requested_prevs: np.ndarray,
    seed: int,
) -> list[TestSample]:
    samples = []
    for sample_id, requested_prev in enumerate(requested_prevs):
        index = unlabelled.sampling_index(SAMPLE_SIZE, *requested_prev, random_state=seed + sample_id)
        collection = unlabelled.sampling_from_index(index)
        posteriors = classifier.predict_proba(collection.X)
        y_hat = np.argmax(posteriors, axis=-1)
        true_ct = contingency_table(collection.y, y_hat, collection.n_classes)
        samples.append(TestSample(collection, posteriors, true_ct))
    return samples


def _requested_test_prevs(val_prev: np.ndarray) -> np.ndarray:
    alpha = _transform_prior(val_prev, TEST_PRIOR_C)
    rng = np.random.RandomState(qp.environ["_R_SEED"])
    return rng.dirichlet(alpha, size=NUM_TEST)


def _make_method(method_name: str, val_prev: np.ndarray):
    for configured_name, method in gen_methods():
        if configured_name == method_name:
            return method

    _, acc = next(gen_acc_measure(True))
    prior = _bayescap_prior(METHOD_PRIOR_KINDS[method_name], val_prev)
    return ACC_BayesCAP(
        acc,
        num_samples=BOOTSTRAP_ITERS,
        num_warmup=BAYES_WARMUPS,
        prior=prior,
    )


def exp_protocol(args: tuple[str, str]) -> list[ExperimentOutcome]:
    info_path, method_name = args
    p = PretrainInfo.load(info_path, fast=True)
    if _all_method_results_exist(p, method_name):
        return [EXP.EXISTS(p, "all", method_name)]

    dataset = p.load_dataset_bundle()
    classifier = p.load_pretrained_classifier(dataset)
    validation = dataset.V
    val_prev_raw = validation.prevalence()
    validation_posteriors = classifier.predict_proba(validation.X)
    requested_prevs = _requested_test_prevs(val_prev_raw)
    test_samples = _make_test_samples(dataset.U, classifier, requested_prevs, qp.environ["_R_SEED"])
    context = EvaluationContext(
        pretrain=p,
        shifts=get_shift(
            np.array([sample.collection.prevalence() for sample in test_samples]),
            dataset.L_prevalence,
        ).tolist(),
        true_cts=[sample.true_ct for sample in test_samples],
        train_prev=get_plain_prev(dataset.L_prevalence),
        val_prev=get_plain_prev(val_prev_raw),
    )

    try:
        method = _make_method(method_name, val_prev_raw)
        started = time()
        method = method.fit(validation, validation_posteriors)
        t_train = time() - started

        started = time()
        estim_ct_ranges = [method.predict_ct_range(sample.collection.X, sample.posteriors) for sample in test_samples]
        t_test_ave = (time() - started) / context.size
        estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, "ct", method_name)]

    results = []
    for acc_name, acc_fn in gen_acc_measure(p.d_info.n_classes > 2):
        if os.path.exists(_result_path(p, method_name, acc_name)):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        method.switch(acc_fn)
        intervals = [method.ci_from_cts(cts) for cts in estim_ct_ranges]
        estim_acc_ranges = [interval.samples for interval in intervals]
        estim_accs = [interval.point_estimate for interval in intervals]
        frame = build_result_frame(
            context,
            method_name=method_name,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=estim_acc_ranges,
            estim_accs=estim_accs,
            estim_cts=estim_cts,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                method_name,
                df=frame,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )
    return results


def save_experiment_results(log: Logger, results_gen: Iterable[list[ExperimentOutcome]]) -> None:
    save_outcomes(log, results_gen, experiment=EXPERIMENT)


def run_experiment_args(
    log: Logger,
    experiment_args: list[tuple[str, str]],
    n_jobs: int,
    env: dict[str, str] | None = None,
) -> None:
    run_tasks(log, experiment_args, worker=exp_protocol, n_jobs=n_jobs, experiment=EXPERIMENT, env=env)


def experiments(log: Logger) -> None:
    info_paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if not _is_binform_pretrain(p):
            continue
        if all(_all_method_results_exist(p, method_name) for method_name in _method_names()):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
        else:
            info_paths.append(path)

    regular_args = [(path, name) for path in info_paths for name in REGULAR_METHOD_NAMES]
    bayes_args = [(path, name) for path in info_paths for name in METHOD_PRIOR_KINDS]
    run_experiment_args(log, regular_args, n_jobs=capenv["N_JOBS"], env=THREAD_LIMITS)
    run_experiment_args(log, bayes_args, n_jobs=BAYES_N_JOBS, env=BAYES_ENV)


def main() -> None:
    log = get_logger(id=EXPERIMENT)
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)
