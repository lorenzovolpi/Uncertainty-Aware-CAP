import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Iterable

import cap
import numpy as np
import quapy as qp
from cap import env as capenv
from cap.models.base import NeedsValidationProtocol
from cap.models.confidence import BootstrapCAP, CAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from cap.models.direct import CAPDirect
from cap.utils.commons import get_shift, parallel
from joblib.externals.loky import get_reusable_executor

from uncawe.config import (
    BAYES_N_JOBS,
    SAMPLE_SIZE,
    gen_acc_measure,
    gen_methods,
    get_acc_names,
    get_filtered_datasets,
    get_method_names,
)
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.results import RDF
from uncawe.util import (
    all_results_exist,
    get_logger,
    get_plain_prev,
    local_path,
    timestamp,
)

EXPERIMENT = "main"
DOMAIN = "classic"

NUM_TEST = 100

# jax.config.update("jax_compilation_cache_dir", JAX_COMPILATION_CACHE_DIR)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

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


def exp_protocol(args: tuple[str, str, CAPWithConfidence]) -> EXP:
    # bundle_path, method_name, method, acc_name, acc_fn = args
    # clsf, D, method_name, method, val, val_posteriors = args
    info_path, method_name, method = args
    results = []

    p = PretrainInfo.load(info_path, fast=True)
    d_info, h_info = p.d_info, p.h_info

    all_exist = True
    for acc in get_acc_names():
        all_exist = all_exist and os.path.exists(
            local_path(
                p.domain,
                d_info.name,
                h_info.full_name,
                method_name,
                acc,
                experiment=EXPERIMENT,
            )
        )
    if all_exist:
        results.append(EXP.EXISTS(p, "all", method_name))
        return results

    D = p.load_dataset_bundle()
    h = p.load_pretrained_classifier(D)

    D.with_test_data(NUM_TEST).with_posteriors(h)
    if isinstance(method, BootstrapCAP) and isinstance(method.base_method, NeedsValidationProtocol):
        val, val_posteriors = D.V1, D.V1_posteriors
        method.base_method.set_validation_protocol(D.V2_prot, D.V2_prot_posteriors)
    else:
        val, val_posteriors = D.V, D.V_posteriors

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    df_len = D.test_prot.total()
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
    tp_true_cts = D.test_prot_true_cts

    if isinstance(method, CAPContingencyTable):
        try:
            tinit = time()
            method = method.fit(val, val_posteriors)
            t_train = time() - tinit

            tinit = time()
            estim_ct_ranges = [
                method.predict_ct_range(Ui.X, Ui_P) for Ui, Ui_P in zip(D.test_prot(), D.test_prot_posteriors)
            ]
            t_test_ave = (time() - tinit) / D.test_prot.total()
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
                # uids=np.arange(df_len).tolist(),
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

    elif isinstance(method, CAPDirect):
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

            try:
                tinit = time()
                method = method.switch_and_fit(acc_fn, val, val_posteriors)
                t_train = time() - tinit

                tinit = time()
                estim_acc_ranges = [
                    method.predict_range(Ui.X, Ui_P) for Ui, Ui_P in zip(D.test_prot(), D.test_prot_posteriors)
                ]
                t_test_ave = (time() - tinit) / D.test_prot.total()
            except Exception as e:
                print_exception(e)
                results.append(EXP.ERROR(e, p, acc_name, method_name))
                continue

            estim_acc_cis = [method.ci_from_accs(acc_range) for acc_range in estim_acc_ranges]
            estim_accs = [ci.point_estimate for ci in estim_acc_cis]
            true_accs = [acc_fn(ct) for ct in tp_true_cts]
            acc_err = cap.error.ae(np.array(true_accs), np.array(estim_accs)).tolist()

            # df_len = len(estim_accs)
            method_df = RDF.from_records(
                df_len,
                # uids=np.arange(df_len).tolist(),
                shifts=test_shift,
                true_cts=tp_true_cts,
                estim_accs=estim_accs,
                estim_acc_ranges=estim_acc_ranges,
                acc_err=acc_err,
                estim_cts=[None] * df_len,
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


def run_experiment_args(log: Logger, experiment_args: list[tuple[str, str, CAPWithConfidence]], n_jobs: int):
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


def experiments(log: Logger, domain: str):
    regular_args = []
    bayes_args = []
    info_paths = load_info_paths(domain=domain)
    filtered_paths = []
    for path in info_paths:
        p = PretrainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info

        if d_info.name not in get_filtered_datasets():
            continue

        if not all_results_exist(
            p.domain,
            d_info.name,
            h_info.full_name,
            get_method_names(),
            get_acc_names(),
            EXPERIMENT,
        ):
            filtered_paths.append(path)
        else:
            log.info(f"[{h_info.name}@{d_info.name}] all results exist, skipping")

    for info_path in filtered_paths:
        for method_name, method in gen_methods():
            args = (info_path, method_name, method)
            if "BayesCAP" in method_name:
                bayes_args.append(args)
            else:
                regular_args.append(args)

    regular_n_jobs = capenv["N_JOBS"]
    run_experiment_args(log, regular_args, n_jobs=regular_n_jobs)
    if regular_n_jobs != 1:
        shutdown_parallel_workers(log)
    run_experiment_args(log, bayes_args, n_jobs=BAYES_N_JOBS)


def main():
    log = get_logger(id=f"{EXPERIMENT}")

    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log, pargs.domain)
    except Exception as e:
        log.error(e)
        print_exception(e)
    finally:
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
