"""Run the main implemented benchmark for accuracy prediction under prior shift.

The experiment evaluates every configured CAP method on the selected pre-trained
dataset/classifier pairs.  For each pair, it draws 100 test bags with the UPP
protocol, exposes only examples and classifier posteriors to the method, and
retains the true contingency tables solely for evaluation.

Its purpose is to compare point accuracy estimates and uncertainty samples for
vanilla accuracy, macro-F1, and macro-K across methods and shift conditions.
The saved records contain absolute errors, predicted intervals or samples,
estimated contingency tables when available, prevalences, and train/test times;
downstream code derives coverage, interval amplitude, and Winkler scores from
these records.  Classifier pre-training is outside this experiment's scope.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from collections.abc import Iterable
from logging import Logger
from time import time
from traceback import print_exception

import numpy as np
import quapy as qp
from cap import env as capenv
from cap.models.base import NeedsValidationProtocol
from cap.models.confidence import BootstrapCAP, CAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from cap.models.direct import CAPDirect
from cap.utils.commons import get_shift

from uncawe.experiments.datasets import get_filtered_datasets
from uncawe.experiments.execution import (
    ExperimentOutcome,
    run_tasks,
    save_experiment_results as save_outcomes,
    shutdown_parallel_workers,
)
from uncawe.experiments.methods import BAYES_N_JOBS, SAMPLE_SIZE, gen_methods, get_method_names
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import all_results_exist, get_logger, get_plain_prev, local_path

EXPERIMENT = "main"
DOMAIN = "classic"
NUM_TEST = 100

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0

EXP = ExperimentOutcome


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


def _prepare_evaluation(p: PretrainInfo, method: CAPWithConfidence):
    dataset = p.load_dataset_bundle()
    classifier = p.load_pretrained_classifier(dataset)
    dataset.with_test_data(NUM_TEST).with_posteriors(classifier)

    if isinstance(method, BootstrapCAP) and isinstance(method.base_method, NeedsValidationProtocol):
        validation = dataset.V1
        validation_posteriors = dataset.V1_posteriors
        method.base_method.set_validation_protocol(dataset.V2_prot, dataset.V2_prot_posteriors)
    else:
        validation = dataset.V
        validation_posteriors = dataset.V_posteriors

    context = EvaluationContext(
        pretrain=p,
        shifts=get_shift(
            np.array([sample.prevalence() for sample in dataset.test_prot()]),
            dataset.L_prevalence,
        ).tolist(),
        true_cts=dataset.test_prot_true_cts,
        train_prev=get_plain_prev(dataset.L_prevalence),
        val_prev=get_plain_prev(validation.prevalence()),
    )
    return dataset, validation, validation_posteriors, context


def _evaluate_contingency_method(
    p: PretrainInfo,
    method_name: str,
    method: CAPContingencyTable,
    dataset,
    validation,
    validation_posteriors,
    context: EvaluationContext,
) -> list[ExperimentOutcome]:
    try:
        started = time()
        method = method.fit(validation, validation_posteriors)
        t_train = time() - started

        started = time()
        estim_ct_ranges = [
            method.predict_ct_range(sample.X, posteriors)
            for sample, posteriors in zip(dataset.test_prot(), dataset.test_prot_posteriors)
        ]
        t_test_ave = (time() - started) / dataset.test_prot.total()
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


def _evaluate_direct_method(
    p: PretrainInfo,
    method_name: str,
    method: CAPDirect,
    dataset,
    validation,
    validation_posteriors,
    context: EvaluationContext,
) -> list[ExperimentOutcome]:
    results = []
    for acc_name, acc_fn in gen_acc_measure(p.d_info.n_classes > 2):
        if os.path.exists(_result_path(p, method_name, acc_name)):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        try:
            started = time()
            method = method.switch_and_fit(acc_fn, validation, validation_posteriors)
            t_train = time() - started

            started = time()
            estim_acc_ranges = [
                method.predict_range(sample.X, posteriors)
                for sample, posteriors in zip(dataset.test_prot(), dataset.test_prot_posteriors)
            ]
            t_test_ave = (time() - started) / dataset.test_prot.total()
        except Exception as error:
            print_exception(error)
            results.append(EXP.ERROR(error, p, acc_name, method_name))
            continue

        intervals = [method.ci_from_accs(acc_range) for acc_range in estim_acc_ranges]
        estim_accs = [interval.point_estimate for interval in intervals]
        frame = build_result_frame(
            context,
            method_name=method_name,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=estim_acc_ranges,
            estim_accs=estim_accs,
            estim_cts=[None] * context.size,
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


def exp_protocol(args: tuple[str, str, CAPWithConfidence]) -> list[ExperimentOutcome]:
    info_path, method_name, method = args
    p = PretrainInfo.load(info_path, fast=True)
    if _all_method_results_exist(p, method_name):
        return [EXP.EXISTS(p, "all", method_name)]

    dataset, validation, validation_posteriors, context = _prepare_evaluation(p, method)
    if isinstance(method, CAPContingencyTable):
        return _evaluate_contingency_method(
            p,
            method_name,
            method,
            dataset,
            validation,
            validation_posteriors,
            context,
        )
    if isinstance(method, CAPDirect):
        return _evaluate_direct_method(
            p,
            method_name,
            method,
            dataset,
            validation,
            validation_posteriors,
            context,
        )
    return []


def save_experiment_results(log: Logger, results_gen: Iterable[list[ExperimentOutcome]]) -> None:
    save_outcomes(log, results_gen, experiment=EXPERIMENT)


def run_experiment_args(
    log: Logger,
    experiment_args: list[tuple[str, str, CAPWithConfidence]],
    n_jobs: int,
) -> None:
    run_tasks(log, experiment_args, worker=exp_protocol, n_jobs=n_jobs, experiment=EXPERIMENT)


def experiments(log: Logger, domain: str) -> None:
    filtered_paths = []
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets():
            continue
        if all_results_exist(
            p.domain,
            p.d_info.name,
            p.h_info.full_name,
            get_method_names(),
            get_acc_names(),
            EXPERIMENT,
        ):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
        else:
            filtered_paths.append(path)

    regular_args = []
    bayes_args = []
    for info_path in filtered_paths:
        for method_name, method in gen_methods():
            args = (info_path, method_name, method)
            (bayes_args if "BayesCAP" in method_name else regular_args).append(args)

    regular_n_jobs = capenv["N_JOBS"]
    run_experiment_args(log, regular_args, n_jobs=regular_n_jobs)
    if regular_n_jobs != 1:
        shutdown_parallel_workers(log)
    run_experiment_args(log, bayes_args, n_jobs=BAYES_N_JOBS)


def main() -> None:
    log = get_logger(id=EXPERIMENT)
    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log, pargs.domain)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        log.info("-" * 32 + "  end  " + "-" * 32)
