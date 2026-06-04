import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Iterable, Literal

import cap
import numpy as np
import quapy as qp
from cap import env as capenv
from cap.models.confidence import ACC_BayesCAP
from cap.utils.commons import contingency_table, get_shift, parallel
from joblib.externals.loky import get_reusable_executor
from quapy.data import LabelledCollection

from uncawe.config import (
    BAYES_N_JOBS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    SAMPLE_SIZE,
    gen_acc_measure,
    gen_methods,
    get_acc_names,
)
from uncawe.pretrain.data import PreTrainedClassifier, PretrainInfo, load_info_paths
from uncawe.results import RDF
from uncawe.util import get_logger, get_plain_prev, local_path, timestamp

EXPERIMENT = "binform"
DOMAIN = "classic"
DATASET = "connect-4"
COLLECTION = "uci_multiclass"

NUM_TEST = 100
BAYESCAP_PRIOR_C = 20.0
TEST_PRIOR_C = 10.0

CLASSIFIERS = frozenset({"LR", "kNN", "SVM", "MLP", "RF"})
REGULAR_METHOD_NAMES = ("O-LEAP", "RQBS")
METHOD_PRIOR_KINDS = {
    "BayesCAP_uniform": "uniform",
    "BayesCAP_transformed_val": "transformed_val",
}

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0


@dataclass()
class EXP:
    code: int
    p: PretrainInfo
    acc_name: str
    method_name: str
    df: RDF = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


@dataclass(frozen=True)
class TestSample:
    collection: LabelledCollection
    posteriors: np.ndarray
    true_ct: np.ndarray


def _transform_prior(prior: np.ndarray, c: float) -> np.ndarray:
    p = np.asarray(prior, dtype=float)
    q = (1 - p) / float(p.shape[0] - 1.0)
    return q * c


def _method_names() -> list[str]:
    return list(REGULAR_METHOD_NAMES) + list(METHOD_PRIOR_KINDS.keys())


def _all_method_results_exist(p: PretrainInfo, method_name: str) -> bool:
    return all(
        os.path.exists(
            local_path(
                p.domain,
                p.d_info.name,
                p.h_info.full_name,
                method_name,
                acc_name,
                experiment=EXPERIMENT,
            )
        )
        for acc_name in get_acc_names()
    )


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
    U: LabelledCollection,
    h: PreTrainedClassifier,
    requested_prevs: np.ndarray,
    seed: int,
) -> list[TestSample]:
    samples = []
    for sample_id, requested_prev in enumerate(requested_prevs):
        index = U.sampling_index(SAMPLE_SIZE, *requested_prev, random_state=seed + sample_id)
        collection = U.sampling_from_index(index)
        posteriors = h.predict_proba(collection.X)
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
    prior_kind = METHOD_PRIOR_KINDS[method_name]
    prior = _bayescap_prior(prior_kind, val_prev)
    return ACC_BayesCAP(
        acc,
        num_samples=BOOTSTRAP_ITERS,
        num_warmup=BAYES_WARMUPS,
        prior=prior,
    )


def exp_protocol(args: tuple[str, str]) -> list[EXP]:
    info_path, method_name = args
    results = []

    p = PretrainInfo.load(info_path, fast=True)
    d_info, h_info = p.d_info, p.h_info

    if _all_method_results_exist(p, method_name):
        results.append(EXP.EXISTS(p, "all", method_name))
        return results

    D = p.load_dataset_bundle()
    h = p.load_pretrained_classifier(D)

    val = D.V
    val_prev_raw = val.prevalence()
    val_posteriors = h.predict_proba(val.X)
    requested_prevs = _requested_test_prevs(val_prev_raw)
    test_samples = _make_test_samples(D.U, h, requested_prevs, qp.environ["_R_SEED"])

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val_prev_raw)
    df_len = len(test_samples)
    test_shift = get_shift(
        np.array([sample.collection.prevalence() for sample in test_samples]),
        D.L_prevalence,
    ).tolist()
    tp_true_cts = [sample.true_ct for sample in test_samples]

    try:
        method = _make_method(method_name, val_prev_raw)

        tinit = time()
        method = method.fit(val, val_posteriors)
        t_train = time() - tinit

        tinit = time()
        estim_ct_ranges = [method.predict_ct_range(sample.collection.X, sample.posteriors) for sample in test_samples]
        t_test_ave = (time() - tinit) / df_len
        estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    except Exception as e:
        print_exception(e)
        results.append(EXP.ERROR(e, p, "ct", method_name))
        return results

    for acc_name, acc_fn in gen_acc_measure(d_info.n_classes > 2):
        path = local_path(
            p.domain,
            d_info.name,
            h_info.full_name,
            method_name,
            acc_name,
            experiment=EXPERIMENT,
        )
        if os.path.exists(path):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        method.switch(acc_fn)
        estim_acc_cis = [method.ci_from_cts(cts) for cts in estim_ct_ranges]
        estim_acc_ranges = [ci.samples for ci in estim_acc_cis]
        estim_accs = [ci.point_estimate for ci in estim_acc_cis]
        true_accs = [acc_fn(ct) for ct in tp_true_cts]
        acc_err = cap.error.ae(np.array(true_accs), np.array(estim_accs)).tolist()

        method_df = RDF.from_records(
            df_len,
            shifts=test_shift,
            true_cts=tp_true_cts,
            estim_accs=estim_accs,
            estim_acc_ranges=estim_acc_ranges,
            acc_err=acc_err,
            estim_cts=estim_cts,
            classifier=h_info.name,
            method=method_name,
            dataset=d_info.name,
            collection=d_info.collection,
            n_classes=d_info.n_classes,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def save_experiment_results(log: Logger, results_gen: Iterable[list[EXP]]):
    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(
                    r.p.domain,
                    r.p.d_info.name,
                    r.p.h_info.full_name,
                    r.method_name,
                    r.acc_name,
                    experiment=EXPERIMENT,
                )
                r.df.save_result(path)
                log.info(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


def run_experiment_args(log: Logger, experiment_args: list[tuple[str, str]], n_jobs: int):
    if not experiment_args:
        return

    if n_jobs == 1:
        for args in experiment_args:
            results = exp_protocol(args)
            save_experiment_results(log, [results])
    else:
        results_gen: Iterable[list[EXP]] = parallel(
            func=exp_protocol,
            args_list=experiment_args,
            n_jobs=n_jobs,
            return_as="generator_unordered",
            max_nbytes=None,
        )
        save_experiment_results(log, results_gen)


def shutdown_parallel_workers(log: Logger):
    try:
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception as e:
        log.warning(f"Could not shut down joblib workers: {e}")


def experiments(log: Logger):
    info_paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if not _is_binform_pretrain(p):
            continue
        if all(_all_method_results_exist(p, method_name) for method_name in _method_names()):
            log.info(f"[{p.h_info.name}@{p.d_info.name}] all results exist, skipping")
            continue
        info_paths.append(path)

    regular_args = [(info_path, method_name) for info_path in info_paths for method_name in REGULAR_METHOD_NAMES]
    bayes_args = [(info_path, method_name) for info_path in info_paths for method_name in METHOD_PRIOR_KINDS]

    regular_n_jobs = capenv["N_JOBS"]
    run_experiment_args(log, regular_args, n_jobs=regular_n_jobs)
    if regular_n_jobs != 1:
        shutdown_parallel_workers(log)
    run_experiment_args(log, bayes_args, n_jobs=BAYES_N_JOBS)


def main():
    log = get_logger(id=f"{EXPERIMENT}")

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log)
    except Exception as e:
        log.error(e)
        print_exception(e)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
