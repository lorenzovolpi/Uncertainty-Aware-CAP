r"""Run the implemented binform experiment on prior specification for BayesCAP.

The scope is the three-class ``connect-4`` dataset in the classic domain and the
configured pre-trained classifiers.  It samples 100 test prevalences from a
Dirichlet distribution whose concentration is obtained from validation
prevalence $p$ through $q_i = c(1-p_i)/(K-1)$, then constructs one test bag for
each prevalence.

The experiment compares O-LEAP and SQS-Hbs (called RQBS until 2026-11-10) with two BayesCAP variants: a uniform
prior and a prior derived from the transformed validation prevalence.  Its
purpose is to isolate how BayesCAP's prior choice affects point estimates and
confidence intervals under this controlled prevalence-shift regime.  It emits
the same accuracy metrics and result schema as the main experiment so the
variants can be compared with the same error and interval-quality measures.

With ``--e51`` it runs instead E5.1 of ``docs/ext5_baycap_priors_tempering.md`` (work-plan card WP-19): the prior on
the target prevalence $\pi^Q$ of ``BayesCAP-Hf-real`` (BayCAP-Hf) and ``BayCAP-KDEy-mlp-real`` (BayCAP-KDEy with an
MLP input and a fixed $\hat C$), both with the realised pushforward and scored against the realised table of each bag
(``true_cts``). For each dataset of ``main`` two means $\mu^\star$ and $\mu^\dagger$ are drawn from
$\operatorname{Dir}(\mathbf 1)$ with a seed fixed per dataset (shared by its five classifiers), and drawn again until
each is at $L_1$ distance at least $0.5$ from the centre $\mathbf 1/K$ and they are at $L_1$ distance at least $0.5$
from each other (``e51_means``). The 100 requested prevalences of the bags are drawn from
$G=\operatorname{Dir}(\lambda K\mu^\star)$, $\lambda=3$, with a seed fixed per dataset (``e51_requested_prevs``), and
the bags of each size $|U|\in\{100,1000\}$ are drawn from the test pool $U$ as above (seed ``_R_SEED + bag index``),
so they are the same for every prior and method. Each method is run with three priors on $\pi^Q$: uniform
(``_uniform``), well specified $\operatorname{Dir}(\lambda K\mu^\star)=G$ (``_well``) and misspecified
$\operatorname{Dir}(\lambda K\mu^\dagger)$ (``_wrong``); everything else is as in ``main``. Results go to
``<OUT_DIR>/uncawe/binform_e51/u<|U|>/``, with the schema of ``main``, the NUTS diagnostics and posterior summaries that
``main`` stores for the two methods, and the columns ``bag_size``, ``requested_prev``, ``mu_star``, ``mu_dagger`` and
``prior_alpha``. The default run (without ``--e51``) is unchanged.

With ``--e54`` it runs E5.4 of the same document (work-plan card WP-24): the same two methods, but the requested
prevalences of the bags come from one of two symmetric distributions with mean $\mathbf 1/K$, concentrated
($G_c=\operatorname{Dir}(3\cdot\mathbf 1)$, ``centre``) or sparse ($G_e=\operatorname{Dir}(\tfrac13\mathbf 1)$,
``edge``), $100$ per dataset and distribution, drawn with a seed fixed per dataset and distribution
(``e54_requested_prevs``; streams distinct from those of E5.1) and shared by the five classifiers and both bag sizes.
Each method is run on each set of bags with three priors on $\pi^Q$: uniform (``_uniform``),
$\operatorname{Dir}(3\cdot\mathbf 1)$ (``_centre``) and $\operatorname{Dir}(\tfrac13\mathbf 1)$ (``_edge``); the prior
equal to the distribution of the bags is the well-informed one, the other the ill-informed one. Results go to
``<OUT_DIR>/uncawe/binform_e54/<distribution>/u<|U|>/``, with the columns of E5.1 except the means, and
``distribution``. Both methods target the realised table of each bag and are scored against it (``true_cts``).

With ``--e55`` it runs E5.5 of the same document (work-plan card WP-27): bags of $|U|=1000$ instances only, at
requested prevalences from distributions more extreme than those of E5.4, concentrated
($G_c=\operatorname{Dir}(10\cdot\mathbf 1)$, ``centre``) or sparse ($G_e=\operatorname{Dir}(\tfrac1{10}\mathbf 1)$,
``edge``), $100$ per dataset and distribution (``e55_requested_prevs``; streams distinct from those of E5.1 and E5.4),
shared by the five classifiers. The priors on $\pi^Q$ stay those of E5.4 (``_uniform``, ``_centre``
$=\operatorname{Dir}(3\cdot\mathbf 1)$, ``_edge`` $=\operatorname{Dir}(\tfrac13\mathbf 1)$), so the parameters of the
distributions of the bags (``E55_BAG_ALPHAS``) are set apart from those of the priors (``E54_ALPHAS``); on the bags of a
distribution the prior on its side is the right-side one, the other the wrong-side one. The methods with a prior are the
two of E5.4 and ``SQS-B`` (``experiments/methods.py:sqs_b`` with ``prior``); ``SQS-H``, ``SQS-Hbs`` and ``PQ`` (SQS-S)
take no prior on $\pi^Q$, are built as in ``main`` and run once per distribution (method names without a suffix), in the
pool of the regular methods of ``main``. Results go to ``<OUT_DIR>/uncawe/binform_e55/<distribution>/``, with the
columns ``bag_size``, ``requested_prev``, ``distribution``, ``bag_alpha`` and, for the methods with a prior,
``prior_alpha`` and the NUTS diagnostics and summaries. Every method targets the realised table of each bag and is
scored against it (``true_cts``).
"""

import os
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from collections.abc import Iterable
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Literal

import jax
import numpy as np
import quapy as qp
from cap import env as capenv
from cap.models.confidence import ACC_BayesCAP
from cap.utils.commons import contingency_table, get_shift
from quapy.data import LabelledCollection

from uncawe.experiments.datasets import get_filtered_datasets
from uncawe.experiments.execution import (
    ExperimentOutcome,
    run_tasks,
    shutdown_parallel_workers,
)
from uncawe.experiments.execution import (
    save_experiment_results as save_outcomes,
)
from uncawe.experiments.methods import (
    BAYCAP_OPTIONS,
    BAYCAP_VARIANTS,
    BAYES_ENV,
    BAYES_N_JOBS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    SAMPLE_SIZE,
    SQS_B,
    THREAD_LIMITS,
    fixed_c_method,
    gen_methods,
    sqs_b,
)
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.methods.baycap import BayCAP
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
REGULAR_METHOD_NAMES = ("O-LEAP", "SQS-Hbs")
METHOD_PRIOR_KINDS = {
    "BayesCAP_uniform": "uniform",
    "BayesCAP_transformed_val": "transformed_val",
}

# E5.1 (docs/ext5_baycap_priors_tempering.md, card WP-19): results folder, bag sizes, concentration factor lambda of
# G = Dir(lambda K mu*), smallest L1 distance of the two means from the centre and from each other, the two methods (as
# in main) and the three priors on pi^Q
E51_EXPERIMENT = "binform_e51"
E51_SIZES = (100, 1000)
E51_LAMBDA = 3.0
E51_MIN_DISTANCE = 0.5
E51_MAX_DRAWS = 100_000
E51_BASE_METHODS = ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real")
E51_PRIORS = ("uniform", "well", "wrong")
E51_METHODS = {f"{base}_{prior}": (base, prior) for base in E51_BASE_METHODS for prior in E51_PRIORS}
# streams of the seed sequence of a dataset: the two means and the requested prevalences
_E51_MEANS_STREAM = 0
_E51_PREVS_STREAM = 1

# E5.4 (docs/ext5_baycap_priors_tempering.md, card WP-24): results folder, the two distributions of the requested
# prevalences, symmetric Dir(alpha 1) (concentrated: alpha = 3; sparse: alpha = 1/3), the same bag sizes and methods as
# E5.1, the three priors on pi^Q (uniform and the two distributions) and the streams of the requested prevalences
E54_EXPERIMENT = "binform_e54"
E54_SIZES = E51_SIZES
E54_ALPHAS = {"centre": 3.0, "edge": 1.0 / 3.0}
E54_DISTRIBUTIONS = tuple(E54_ALPHAS)
E54_PRIORS = ("uniform", *E54_DISTRIBUTIONS)
E54_METHODS = {f"{base}_{prior}": (base, prior) for base in E51_BASE_METHODS for prior in E54_PRIORS}
_E54_PREVS_STREAMS = {"centre": 2, "edge": 3}

# E5.5 (docs/ext5_baycap_priors_tempering.md, card WP-27): results folder, the bag size, the two distributions of the
# requested prevalences, symmetric Dir(alpha 1) more extreme than those of E5.4 (concentrated: alpha = 10; sparse:
# alpha = 1/10), the priors on pi^Q of E5.4 (E54_ALPHAS), the methods with a prior (the two of E5.4 and SQS-B), the SQS
# methods without a prior on pi^Q (built as in main, run once per distribution) and the streams of the requested
# prevalences
E55_EXPERIMENT = "binform_e55"
E55_SIZE = 1000
E55_BAG_ALPHAS = {"centre": 10.0, "edge": 0.1}
E55_DISTRIBUTIONS = tuple(E55_BAG_ALPHAS)
E55_PRIORS = E54_PRIORS
E55_PRIOR_BASES = (*E51_BASE_METHODS, SQS_B)
E55_REFERENCES = ("SQS-H", "SQS-Hbs", "PQ")
# name -> (base method, prior); the prior is None for the methods without a prior on pi^Q
E55_METHODS = {f"{base}_{prior}": (base, prior) for base in E55_PRIOR_BASES for prior in E55_PRIORS} | {
    name: (name, None) for name in E55_REFERENCES
}
_E55_PREVS_STREAMS = {"centre": 4, "edge": 5}

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


# ----------------------------------------------------------------------------------------------------------- E5.1


def _e51_rng(dataset: str, stream: int, seed: int | None = None) -> np.random.Generator:
    """A generator that depends only on the experiment seed, the dataset name and the stream (``zlib.crc32`` of the
    name, stable across runs and machines, unlike ``hash``)."""
    seed = qp.environ["_R_SEED"] if seed is None else seed
    return np.random.default_rng([seed, zlib.crc32(dataset.encode()), stream])


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def e51_means(
    dataset: str,
    n_classes: int,
    seed: int | None = None,
    min_distance: float = E51_MIN_DISTANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """The two means $(\\mu^\\star,\\mu^\\dagger)$ of E5.1 for ``dataset``: $\\mu^\\star$ is drawn from
    $\\operatorname{Dir}(\\mathbf 1)$ until it is at $L_1$ distance at least ``min_distance`` from the centre
    $\\mathbf 1/K$; then $\\mu^\\dagger$ is drawn from $\\operatorname{Dir}(\\mathbf 1)$ until it is at $L_1$ distance at
    least ``min_distance`` from the centre and from $\\mu^\\star$. The generator depends only on the seed and the dataset,
    so the five classifiers of a dataset share the means."""
    rng = _e51_rng(dataset, _E51_MEANS_STREAM, seed)
    centre = np.full(n_classes, 1.0 / n_classes)
    ones = np.ones(n_classes)

    def draw(far_from: list[np.ndarray]) -> np.ndarray:
        for _ in range(E51_MAX_DRAWS):
            mu = rng.dirichlet(ones)
            if all(_l1(mu, point) >= min_distance for point in far_from):
                return mu
        raise RuntimeError(f"No mean at L1 distance >= {min_distance} found for {dataset} in {E51_MAX_DRAWS} draws")

    mu_star = draw([centre])
    mu_dagger = draw([centre, mu_star])
    return mu_star, mu_dagger


def e51_concentration(mu: np.ndarray, lam: float = E51_LAMBDA) -> np.ndarray:
    """The Dirichlet parameter $\\lambda K\\mu$ of E5.1."""
    mu = np.asarray(mu, dtype=float)
    return lam * mu.shape[0] * mu


def e51_prior(kind: str, mu_star: np.ndarray, mu_dagger: np.ndarray) -> str | np.ndarray:
    """The prior on $\\pi^Q$ of ``kind``: ``"uniform"``, ``"well"`` ($\\operatorname{Dir}(\\lambda K\\mu^\\star)=G$) or
    ``"wrong"`` ($\\operatorname{Dir}(\\lambda K\\mu^\\dagger)$)."""
    if kind == "uniform":
        return "uniform"
    if kind == "well":
        return e51_concentration(mu_star)
    if kind == "wrong":
        return e51_concentration(mu_dagger)
    raise ValueError(f"Unknown E5.1 prior: {kind!r}")


def e51_requested_prevs(dataset: str, mu_star: np.ndarray, seed: int | None = None) -> np.ndarray:
    """The ``NUM_TEST`` requested prevalences of the bags of ``dataset``, drawn from $G=\\operatorname{Dir}(\\lambda
    K\\mu^\\star)$ with a generator that depends only on the seed and the dataset (the same for every classifier, bag
    size, prior and method). They are renormalised, since a draw with parameters below $1$ may round to a sum other
    than $1$."""
    prevs = _e51_rng(dataset, _E51_PREVS_STREAM, seed).dirichlet(e51_concentration(mu_star), size=NUM_TEST)
    return prevs / prevs.sum(axis=1, keepdims=True)


def _make_sized_test_samples(
    unlabelled: LabelledCollection,
    classifier: PreTrainedClassifier,
    requested_prevs: np.ndarray,
    seed: int,
    size: int,
) -> list[TestSample]:
    """As ``_make_test_samples``, with bags of ``size`` instances."""
    samples = []
    for sample_id, requested_prev in enumerate(requested_prevs):
        index = unlabelled.sampling_index(size, *requested_prev, random_state=seed + sample_id)
        collection = unlabelled.sampling_from_index(index)
        posteriors = classifier.predict_proba(collection.X)
        y_hat = np.argmax(posteriors, axis=-1)
        true_ct = contingency_table(collection.y, y_hat, collection.n_classes)
        samples.append(TestSample(collection, posteriors, true_ct))
    return samples


def e51_method(method_name: str, mu_star: np.ndarray, mu_dagger: np.ndarray):
    """The method ``method_name`` of ``E51_METHODS``: its base method built as in ``main`` (same settings and seed),
    with the prior on $\\pi^Q$ of the suffix."""
    base, kind = E51_METHODS[method_name]
    return _prior_method(base, e51_prior(kind, mu_star, mu_dagger))


def _prior_method(base: str, prior: str | np.ndarray):
    """The base method ``base`` of ``E51_BASE_METHODS`` (or ``SQS-B``, E5.5) built as in ``main`` (same settings and
    seed), with the prior ``prior`` on $\\pi^Q$ (E5.1, E5.4 and E5.5)."""
    _, acc = next(gen_acc_measure(True))
    if base == SQS_B:
        return sqs_b(acc, prior=prior)
    if base in BAYCAP_VARIANTS:
        estimand, c_prior = BAYCAP_VARIANTS[base]
        return BayCAP(
            acc,
            estimand=estimand,
            c_prior=c_prior,
            num_samples=BOOTSTRAP_ITERS,
            num_warmup=BAYES_WARMUPS,
            prior=prior,
            **BAYCAP_OPTIONS.get(base, {}),
        )
    return fixed_c_method(acc, base, prior=prior)


def e51_experiment(size: int) -> str:
    """The results folder of E5.1 for bags of ``size`` instances, relative to ``<OUT_DIR>/uncawe``."""
    return os.path.join(E51_EXPERIMENT, f"u{size}")


def _e51_result_path(p: PretrainInfo, method_name: str, acc_name: str, size: int) -> str:
    return local_path(
        p.domain,
        p.d_info.name,
        p.h_info.full_name,
        method_name,
        acc_name,
        experiment=e51_experiment(size),
    )


def _e51_all_results_exist(p: PretrainInfo, method_name: str, size: int) -> bool:
    return all(os.path.exists(_e51_result_path(p, method_name, acc_name, size)) for acc_name in get_acc_names())


def e51_protocol(args: tuple[str, str, int]) -> list[ExperimentOutcome]:
    """One task of E5.1: the method ``method_name`` on the pair of ``info_path`` with bags of ``size`` instances.

    Every bag builds and compiles its own NUTS sampler, and a worker keeps the compiled code in the caches of JAX, so a
    worker of a long pool grows by some GB per task until a compilation fails for lack of memory (both crashes of the
    first full run). The caches are cleared after each task: nothing is reused across bags anyway, so the results do
    not change."""
    try:
        return _e51_task(*args)
    finally:
        jax.clear_caches()


def _e51_task(info_path: str, method_name: str, size: int) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(info_path, fast=True)
    if _e51_all_results_exist(p, method_name, size):
        return [EXP.EXISTS(p, "all", method_name)]

    n_classes = p.d_info.n_classes
    mu_star, mu_dagger = e51_means(p.d_info.name, n_classes)
    # the Dirichlet parameter of the prior, a vector also for the uniform prior (all ones)
    prior = e51_prior(E51_METHODS[method_name][1], mu_star, mu_dagger)
    prior_alpha = np.ones(n_classes) if isinstance(prior, str) else prior
    return _prior_task(
        p,
        method_name,
        build_method=lambda: e51_method(method_name, mu_star, mu_dagger),
        requested_prevs=e51_requested_prevs(p.d_info.name, mu_star),
        size=size,
        result_path=lambda acc_name: _e51_result_path(p, method_name, acc_name, size),
        static_columns={
            "mu_star": mu_star.tolist(),
            "mu_dagger": mu_dagger.tolist(),
            "prior_alpha": prior_alpha.tolist(),
        },
    )


def _prior_task(
    p: PretrainInfo,
    method_name: str,
    build_method,
    requested_prevs: np.ndarray,
    size: int,
    result_path,
    static_columns: dict,
) -> list[ExperimentOutcome]:
    """The body of a task of E5.1, E5.4 or E5.5: the bags of ``size`` instances at ``requested_prevs`` drawn from the
    test pool (seed ``_R_SEED + bag index``), the method of ``build_method()`` fitted on $V$ and run on every bag, with
    the NUTS diagnostics and posterior summaries of each bag for the methods that sample with NUTS (all those of E5.1
    and E5.4; not ``SQS-H``, ``SQS-Hbs`` and ``PQ`` of E5.5); one result file per metric at ``result_path(acc_name)``,
    with the columns ``bag_size``, ``requested_prev`` and, repeated on every row, ``static_columns``."""
    dataset = p.load_dataset_bundle()
    classifier = p.load_pretrained_classifier(dataset)
    validation = dataset.V
    val_prev_raw = validation.prevalence()
    validation_posteriors = classifier.predict_proba(validation.X)
    n_classes = p.d_info.n_classes
    test_samples = _make_sized_test_samples(dataset.U, classifier, requested_prevs, qp.environ["_R_SEED"], size)
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
        method = build_method()
        nuts = hasattr(method, "last_nuts_diagnostics")
        started = time()
        method = method.fit(validation, validation_posteriors)
        t_train = time() - started

        # as in main: the time of the NUTS diagnostics and posterior summaries is left out of the test time
        started, t_diagnostics = time(), 0.0
        estim_ct_ranges, diagnostics = [], []
        for sample in test_samples:
            estim_ct_ranges.append(method.predict_ct_range(sample.collection.X, sample.posteriors))
            if not nuts:
                continue
            started_diagnostics = time()
            diagnostics.append(method.last_nuts_diagnostics() | method.last_posterior_summaries())
            t_diagnostics += time() - started_diagnostics
        t_test_ave = (time() - started - t_diagnostics) / context.size
        estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, "ct", method_name)]

    extra_columns = {name: [d[name] for d in diagnostics] for name in diagnostics[0]} if diagnostics else {}
    extra_columns |= {
        "bag_size": [size] * context.size,
        "requested_prev": [prev.tolist() for prev in requested_prevs],
    }
    extra_columns |= {name: [value] * context.size for name, value in static_columns.items()}

    results = []
    for acc_name, acc_fn in gen_acc_measure(n_classes > 2):
        if os.path.exists(result_path(acc_name)):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        method.switch(acc_fn)
        intervals = [method.ci_from_cts(cts) for cts in estim_ct_ranges]
        frame = build_result_frame(
            context,
            method_name=method_name,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=[interval.samples for interval in intervals],
            estim_accs=[interval.point_estimate for interval in intervals],
            estim_cts=estim_cts,
            t_train=t_train,
            t_test_ave=t_test_ave,
            extra_columns=extra_columns,
        )
        results.append(EXP.SUCCESS(p, acc_name, method_name, df=frame, t_train=t_train, t_test_ave=t_test_ave))
    return results


def e51_tasks(
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    sizes: list[int] | None = None,
    methods: list[str] | None = None,
    log: Logger | None = None,
) -> list[tuple[str, str, int]]:
    """The tasks of E5.1 still to run, ``(info_path, method_name, size)``: the pairs of ``main`` (its datasets, every
    classifier), restricted by the arguments (e.g. a pilot), BayCAP-Hf first and the larger bags first."""
    sizes = list(E51_SIZES) if sizes is None else sizes
    methods = list(E51_METHODS) if methods is None else methods
    unknown = sorted(set(methods) - set(E51_METHODS))
    if unknown:
        raise ValueError(f"Unknown E5.1 methods: {unknown}")
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets():
            continue
        if datasets is not None and p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        paths.append((path, p))

    tasks = []
    for size in sorted(sizes, reverse=True):
        for method_name in sorted(methods, key=lambda name: E51_BASE_METHODS.index(E51_METHODS[name][0])):
            for path, p in paths:
                if _e51_all_results_exist(p, method_name, size):
                    if log is not None:
                        log.info(f"[{p.h_info.name}@{p.d_info.name}] {method_name} |U|={size} exists, skipping")
                    continue
                tasks.append((path, method_name, size))
    return tasks


def e51_experiments(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    sizes: list[int] | None = None,
    methods: list[str] | None = None,
) -> None:
    """Run E5.1; every method is a NumPyro method, so all tasks run in the BayCAP pool, one pool per bag size (the
    outcomes of a pool are saved in the folder of its size)."""
    tasks = e51_tasks(datasets, classifiers, sizes, methods, log)
    for size in sorted({task[2] for task in tasks}, reverse=True):
        sized = [task for task in tasks if task[2] == size]
        run_tasks(log, sized, worker=e51_protocol, n_jobs=BAYES_N_JOBS, experiment=e51_experiment(size), env=BAYES_ENV)


# ----------------------------------------------------------------------------------------------------------- E5.4


def e54_alpha(kind: str, n_classes: int) -> np.ndarray:
    """The Dirichlet parameter $\\alpha\\mathbf 1$ of ``kind``: ``"uniform"`` ($\\alpha=1$), ``"centre"``
    ($\\alpha=3$) or ``"edge"`` ($\\alpha=1/3$)."""
    if kind == "uniform":
        return np.ones(n_classes)
    if kind in E54_ALPHAS:
        return np.full(n_classes, E54_ALPHAS[kind])
    raise ValueError(f"Unknown E5.4 distribution or prior: {kind!r}")


def e54_prior(kind: str, n_classes: int) -> str | np.ndarray:
    """The prior on $\\pi^Q$ of ``kind``: ``"uniform"`` (the default of both methods), or the distribution ``"centre"``
    or ``"edge"``."""
    return "uniform" if kind == "uniform" else e54_alpha(kind, n_classes)


def e54_requested_prevs(dataset: str, n_classes: int, distribution: str, seed: int | None = None) -> np.ndarray:
    """The ``NUM_TEST`` requested prevalences of the bags of ``dataset`` drawn from ``distribution`` (``"centre"`` or
    ``"edge"``), with a generator that depends only on the seed, the dataset and the distribution (the same for every
    classifier, bag size, prior and method; streams distinct from those of E5.1). Renormalised as in E5.1."""
    if distribution not in E54_ALPHAS:
        raise ValueError(f"Unknown E5.4 distribution: {distribution!r}")
    rng = _e51_rng(dataset, _E54_PREVS_STREAMS[distribution], seed)
    prevs = rng.dirichlet(e54_alpha(distribution, n_classes), size=NUM_TEST)
    return prevs / prevs.sum(axis=1, keepdims=True)


def e54_method(method_name: str, n_classes: int):
    """The method ``method_name`` of ``E54_METHODS``: its base method built as in ``main``, with the prior on
    $\\pi^Q$ of the suffix."""
    base, kind = E54_METHODS[method_name]
    return _prior_method(base, e54_prior(kind, n_classes))


def e54_experiment(distribution: str, size: int) -> str:
    """The results folder of E5.4 for bags drawn from ``distribution`` with ``size`` instances, relative to
    ``<OUT_DIR>/uncawe``."""
    return os.path.join(E54_EXPERIMENT, distribution, f"u{size}")


def _e54_result_path(p: PretrainInfo, method_name: str, acc_name: str, distribution: str, size: int) -> str:
    return local_path(
        p.domain,
        p.d_info.name,
        p.h_info.full_name,
        method_name,
        acc_name,
        experiment=e54_experiment(distribution, size),
    )


def _e54_all_results_exist(p: PretrainInfo, method_name: str, distribution: str, size: int) -> bool:
    return all(
        os.path.exists(_e54_result_path(p, method_name, acc_name, distribution, size)) for acc_name in get_acc_names()
    )


def e54_protocol(args: tuple[str, str, str, int]) -> list[ExperimentOutcome]:
    """One task of E5.4: the method ``method_name`` on the pair of ``info_path`` with the bags of ``size`` instances
    drawn from ``distribution``; the caches of JAX are cleared after each task, as in ``e51_protocol``."""
    try:
        return _e54_task(*args)
    finally:
        jax.clear_caches()


def _e54_task(info_path: str, method_name: str, distribution: str, size: int) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(info_path, fast=True)
    if _e54_all_results_exist(p, method_name, distribution, size):
        return [EXP.EXISTS(p, "all", method_name)]

    n_classes = p.d_info.n_classes
    prior_alpha = e54_alpha(E54_METHODS[method_name][1], n_classes)
    return _prior_task(
        p,
        method_name,
        build_method=lambda: e54_method(method_name, n_classes),
        requested_prevs=e54_requested_prevs(p.d_info.name, n_classes, distribution),
        size=size,
        result_path=lambda acc_name: _e54_result_path(p, method_name, acc_name, distribution, size),
        static_columns={"distribution": distribution, "prior_alpha": prior_alpha.tolist()},
    )


def e54_tasks(
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    distributions: list[str] | None = None,
    sizes: list[int] | None = None,
    methods: list[str] | None = None,
    log: Logger | None = None,
) -> list[tuple[str, str, str, int]]:
    """The tasks of E5.4 still to run, ``(info_path, method_name, distribution, size)``: the pairs of ``main``
    restricted by the arguments (e.g. the pilot), the larger bags first, then by distribution, BayCAP-Hf first."""
    distributions = list(E54_DISTRIBUTIONS) if distributions is None else distributions
    sizes = list(E54_SIZES) if sizes is None else sizes
    methods = list(E54_METHODS) if methods is None else methods
    unknown = sorted(set(methods) - set(E54_METHODS)) + sorted(set(distributions) - set(E54_DISTRIBUTIONS))
    if unknown:
        raise ValueError(f"Unknown E5.4 methods or distributions: {unknown}")
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets():
            continue
        if datasets is not None and p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        paths.append((path, p))

    tasks = []
    for size in sorted(sizes, reverse=True):
        for distribution in sorted(distributions, key=E54_DISTRIBUTIONS.index):
            for method_name in sorted(methods, key=lambda name: E51_BASE_METHODS.index(E54_METHODS[name][0])):
                for path, p in paths:
                    if _e54_all_results_exist(p, method_name, distribution, size):
                        if log is not None:
                            log.info(
                                f"[{p.h_info.name}@{p.d_info.name}] {method_name} {distribution} |U|={size} exists, "
                                "skipping"
                            )
                        continue
                    tasks.append((path, method_name, distribution, size))
    return tasks


def e54_experiments(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    distributions: list[str] | None = None,
    sizes: list[int] | None = None,
    methods: list[str] | None = None,
) -> None:
    """Run E5.4 in the BayCAP pool, one pool per (bag size, distribution), whose outcomes are saved in its folder."""
    tasks = e54_tasks(datasets, classifiers, distributions, sizes, methods, log)
    cells = sorted({(task[3], task[2]) for task in tasks}, key=lambda c: (-c[0], E54_DISTRIBUTIONS.index(c[1])))
    for size, distribution in cells:
        part = [task for task in tasks if (task[3], task[2]) == (size, distribution)]
        run_tasks(
            log,
            part,
            worker=e54_protocol,
            n_jobs=BAYES_N_JOBS,
            experiment=e54_experiment(distribution, size),
            env=BAYES_ENV,
        )


# ----------------------------------------------------------------------------------------------------------- E5.5


def e55_bag_alpha(distribution: str, n_classes: int) -> np.ndarray:
    """The Dirichlet parameter $\\alpha\\mathbf 1$ of the distribution of the bags ``distribution``: ``"centre"``
    ($\\alpha=10$) or ``"edge"`` ($\\alpha=1/10$). The priors are those of E5.4 (``e54_prior``)."""
    if distribution not in E55_BAG_ALPHAS:
        raise ValueError(f"Unknown E5.5 distribution: {distribution!r}")
    return np.full(n_classes, E55_BAG_ALPHAS[distribution])


def e55_requested_prevs(dataset: str, n_classes: int, distribution: str, seed: int | None = None) -> np.ndarray:
    """The ``NUM_TEST`` requested prevalences of the bags of ``dataset`` drawn from ``distribution`` (``"centre"`` or
    ``"edge"``), with a generator that depends only on the seed, the dataset and the distribution (the same for every
    classifier, prior and method; streams distinct from those of E5.1 and E5.4). Renormalised as in E5.1."""
    alpha = e55_bag_alpha(distribution, n_classes)
    prevs = _e51_rng(dataset, _E55_PREVS_STREAMS[distribution], seed).dirichlet(alpha, size=NUM_TEST)
    return prevs / prevs.sum(axis=1, keepdims=True)


def e55_method(method_name: str, n_classes: int):
    """The method ``method_name`` of ``E55_METHODS``: a method with a prior is built as in ``main`` with the prior on
    $\\pi^Q$ of its suffix (``e54_prior``); a method without a prior is the method of ``main`` itself."""
    base, kind = E55_METHODS[method_name]
    if kind is None:
        return dict(gen_methods())[base]
    return _prior_method(base, e54_prior(kind, n_classes))


def e55_experiment(distribution: str) -> str:
    """The results folder of E5.5 for the bags drawn from ``distribution``, relative to ``<OUT_DIR>/uncawe``."""
    return os.path.join(E55_EXPERIMENT, distribution)


def _e55_result_path(p: PretrainInfo, method_name: str, acc_name: str, distribution: str) -> str:
    return local_path(
        p.domain,
        p.d_info.name,
        p.h_info.full_name,
        method_name,
        acc_name,
        experiment=e55_experiment(distribution),
    )


def _e55_all_results_exist(p: PretrainInfo, method_name: str, distribution: str) -> bool:
    return all(os.path.exists(_e55_result_path(p, method_name, acc_name, distribution)) for acc_name in get_acc_names())


def e55_protocol(args: tuple[str, str, str]) -> list[ExperimentOutcome]:
    """One task of E5.5: the method ``method_name`` on the pair of ``info_path`` with the bags drawn from
    ``distribution``; the caches of JAX are cleared after each task, as in ``e51_protocol``."""
    try:
        return _e55_task(*args)
    finally:
        jax.clear_caches()


def _e55_task(info_path: str, method_name: str, distribution: str) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(info_path, fast=True)
    if _e55_all_results_exist(p, method_name, distribution):
        return [EXP.EXISTS(p, "all", method_name)]

    n_classes = p.d_info.n_classes
    kind = E55_METHODS[method_name][1]
    static_columns = {"distribution": distribution, "bag_alpha": E55_BAG_ALPHAS[distribution]}
    if kind is not None:
        static_columns["prior_alpha"] = e54_alpha(kind, n_classes).tolist()
    return _prior_task(
        p,
        method_name,
        build_method=lambda: e55_method(method_name, n_classes),
        requested_prevs=e55_requested_prevs(p.d_info.name, n_classes, distribution),
        size=E55_SIZE,
        result_path=lambda acc_name: _e55_result_path(p, method_name, acc_name, distribution),
        static_columns=static_columns,
    )


def _e55_method_order(method_name: str) -> int:
    """BayCAP-Hf, BayCAP-KDEy and SQS-B, then the methods without a prior."""
    base = E55_METHODS[method_name][0]
    return (*E55_PRIOR_BASES, *E55_REFERENCES).index(base)


def e55_tasks(
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    distributions: list[str] | None = None,
    methods: list[str] | None = None,
    log: Logger | None = None,
) -> list[tuple[str, str, str]]:
    """The tasks of E5.5 still to run, ``(info_path, method_name, distribution)``: the pairs of ``main`` restricted by
    the arguments (e.g. the pilot), by distribution, then BayCAP-Hf, BayCAP-KDEy, SQS-B and the methods without a
    prior."""
    distributions = list(E55_DISTRIBUTIONS) if distributions is None else distributions
    methods = list(E55_METHODS) if methods is None else methods
    unknown = sorted(set(methods) - set(E55_METHODS)) + sorted(set(distributions) - set(E55_DISTRIBUTIONS))
    if unknown:
        raise ValueError(f"Unknown E5.5 methods or distributions: {unknown}")
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name not in get_filtered_datasets():
            continue
        if datasets is not None and p.d_info.name not in datasets:
            continue
        if classifiers is not None and p.h_info.name not in classifiers:
            continue
        paths.append((path, p))

    tasks = []
    for distribution in sorted(distributions, key=E55_DISTRIBUTIONS.index):
        for method_name in sorted(methods, key=_e55_method_order):
            for path, p in paths:
                if _e55_all_results_exist(p, method_name, distribution):
                    if log is not None:
                        log.info(f"[{p.h_info.name}@{p.d_info.name}] {method_name} {distribution} exists, skipping")
                    continue
                tasks.append((path, method_name, distribution))
    return tasks


def e55_experiments(
    log: Logger,
    datasets: list[str] | None = None,
    classifiers: list[str] | None = None,
    distributions: list[str] | None = None,
    methods: list[str] | None = None,
) -> None:
    """Run E5.5, per distribution (outcomes saved in its folder): the methods without a prior in the pool of the
    regular methods of ``main`` (``N_JOBS`` workers, ``THREAD_LIMITS``), then the methods with a prior, which sample
    with NUTS, in the BayCAP pool, as in ``main``."""
    tasks = e55_tasks(datasets, classifiers, distributions, methods, log)
    for distribution in sorted({task[2] for task in tasks}, key=E55_DISTRIBUTIONS.index):
        part = [task for task in tasks if task[2] == distribution]
        regular = [task for task in part if E55_METHODS[task[1]][1] is None]
        bayes = [task for task in part if E55_METHODS[task[1]][1] is not None]
        experiment = e55_experiment(distribution)
        run_tasks(log, regular, worker=e55_protocol, n_jobs=capenv["N_JOBS"], experiment=experiment, env=THREAD_LIMITS)
        run_tasks(log, bayes, worker=e55_protocol, n_jobs=BAYES_N_JOBS, experiment=experiment, env=BAYES_ENV)


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


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="The binform experiment; with --e51, E5.1 (priors on pi^Q, card WP-19); with --e54, E5.4 "
        "(concentrated and sparse prevalence distributions, card WP-24); with --e55, E5.5 (right-side and wrong-side "
        "priors under strong shifts, with SQS references, card WP-27)."
    )
    study = parser.add_mutually_exclusive_group()
    study.add_argument("--e51", action="store_true", help="Run E5.1 instead of the default configuration")
    study.add_argument("--e54", action="store_true", help="Run E5.4 instead of the default configuration")
    study.add_argument("--e55", action="store_true", help="Run E5.5 instead of the default configuration")
    parser.add_argument(
        "--datasets", nargs="+", default=None, help="E5.1/E5.4/E5.5: only these datasets (default: those of main)"
    )
    parser.add_argument(
        "--classifiers", nargs="+", default=None, help="E5.1/E5.4/E5.5: only these classifiers (default: all)"
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="E5.1/E5.4: bag sizes (default: 100 1000)")
    parser.add_argument(
        "--methods", nargs="+", default=None, help="E5.1/E5.4/E5.5: only these methods (default: all of the study)"
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=E54_DISTRIBUTIONS,
        default=None,
        help="E5.4/E5.5: only these distributions of the bags (default: centre edge)",
    )
    pargs = parser.parse_args(argv)
    study_args = pargs.e51 or pargs.e54 or pargs.e55
    restricted = any(getattr(pargs, name) is not None for name in ("datasets", "classifiers", "sizes", "methods"))
    if restricted and not study_args:
        parser.error("--datasets, --classifiers, --sizes and --methods need --e51, --e54 or --e55")
    if pargs.distributions is not None and not (pargs.e54 or pargs.e55):
        parser.error("--distributions needs --e54 or --e55")
    if pargs.sizes is not None and pargs.e55:
        parser.error("--sizes does not apply to --e55 (|U|=1000 only)")
    return pargs


def main() -> None:
    pargs = parse_args()
    study = {E51_EXPERIMENT: pargs.e51, E54_EXPERIMENT: pargs.e54, E55_EXPERIMENT: pargs.e55}
    log = get_logger(id=next((name for name, chosen in study.items() if chosen), EXPERIMENT))
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        if pargs.e51:
            e51_experiments(log, pargs.datasets, pargs.classifiers, pargs.sizes, pargs.methods)
        elif pargs.e54:
            e54_experiments(log, pargs.datasets, pargs.classifiers, pargs.distributions, pargs.sizes, pargs.methods)
        elif pargs.e55:
            e55_experiments(log, pargs.datasets, pargs.classifiers, pargs.distributions, pargs.methods)
        else:
            experiments(log)
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)
