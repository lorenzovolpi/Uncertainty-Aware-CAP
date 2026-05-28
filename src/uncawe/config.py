from collections import defaultdict
from functools import lru_cache
from typing import Callable, Iterable, Literal, Tuple

import numpy as np
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import f1, f1_macro, k_bin, k_macro, smooth, vanilla_acc
from cap.models.confidence import (
    CBPE,
    RQBS,
    ACC_BayesCAP,
    BootstrapCTCAP,
    BootstrapDirectCAP,
    CAPWithConfidence,
    PrediQuant,
)
from cap.models.cont_table import O_LEAP, QuAccNxN
from cap.models.direct import DoC
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import PACC, KDEyML
from sklearn.neural_network import MLPClassifier as MLP

from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.pretrain.dataset import sort_datasets_by_size
from uncawe.util import all_results_exist

BAYES_WARMUPS = 500
BAYES_N_JOBS = 1
BOOTSTRAP_ITERS = 1000
SAMPLE_SIZE = 1000
JAX_COMPILATION_CACHE_DIR = "./output/jax_cache"
CONFIDENCE_LEVEL = 0.95
METHOD_CLASS = {
    "DoC": DoC,
    "QuAcc_NxN": QuAccNxN,
    "O-LEAP": O_LEAP,
    "RQBS": RQBS,
    "PQ": PrediQuant,
    "BayesCAP": ACC_BayesCAP,
    "CBPE": CBPE,
}


def kdey():
    return KDEyML(MLP())


def pacc():
    return PACC(MLP())


def get_acc_names():
    return ["vanilla_accuracy", "macro-F1", "macro-K"]


def get_selection_acc(name: str, multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (smooth(f1_macro) if multiclass else smooth(f1)),
        "macro-K": (k_macro if multiclass else k_bin),
    }[name]


def get_evaluation_acc(name: str, is_multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (f1_macro if is_multiclass else f1),
        "macro-K": (k_macro if is_multiclass else k_bin),
    }[name]


def acc_from_ct(
    acc_name: str,
    ct: np.ndarray,
    type: Literal["selection", "evaluation"] = "selection",
) -> float:
    n_classes = ct.shape[0]
    if type == "selection":
        return get_selection_acc(acc_name, n_classes > 2)(ct)
    elif type == "evaluation":
        return get_evaluation_acc(acc_name, n_classes > 2)(ct)


def gen_acc_measure(is_multiclass: bool):
    for acc in get_acc_names():
        yield acc, get_selection_acc(acc, is_multiclass)


def gen_methods() -> Iterable[Tuple[str, CAPWithConfidence]]:
    acc_name, acc = next(gen_acc_measure(True))
    yield "DoC", BootstrapDirectCAP(DoC(acc), num_samples=BOOTSTRAP_ITERS)
    yield (
        "QuAcc_NxN",
        BootstrapCTCAP(
            QuAccNxN(acc, kdey(), add_maxconf=True, add_negentropy=True, add_maxinfsoft=True),
            num_samples=BOOTSTRAP_ITERS,
        ),
    )
    yield "O-LEAP", BootstrapCTCAP(O_LEAP(acc, kdey()), num_samples=BOOTSTRAP_ITERS)
    yield "CBPE", CBPE(acc_name)
    #
    yield "RQBS", RQBS(acc, pacc(), num_samples=BOOTSTRAP_ITERS, sample_size=SAMPLE_SIZE)
    yield (
        "PQ",
        PrediQuant(
            acc,
            kdey(),
            num_samples=BOOTSTRAP_ITERS,
            sample_size=SAMPLE_SIZE,
            alpha=0.1,
            distance="l1",
            predict_train_prev=True,
        ),
    )
    yield "BayesCAP", ACC_BayesCAP(acc, num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS)
    # yield "HD_BayesCAP", HD_BayesCAP(acc, num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS)


def get_existing_dataset_names(experiment: str, domain: str, sort=True):
    info_paths = load_info_paths(domain=domain)
    dataset_h_map = defaultdict(lambda: True)
    for path in info_paths:
        p = PretrainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        # if not dataset_h_map[D.name]:
        #     continue
        # problem = "multiclass" if d_info.n_classes > 2 else "binary"
        _key = (d_info.name, d_info.collection)
        dataset_h_map[_key] = dataset_h_map[_key] and all_results_exist(
            p.domain,
            d_info.name,
            h_info.full_name,
            get_method_names(),
            get_acc_names(),
            experiment,
        )

    datasets = [dc for dc, all_exist in dataset_h_map.items() if all_exist]
    dataset_names, dataset_colls = tuple(map(lambda x: list(x), zip(*datasets)))
    if sort:
        return sort_datasets_by_size(dataset_colls, dataset_names)
    else:
        return dataset_names


def get_method_names():
    return list(METHOD_CLASS.keys())


@lru_cache(maxsize=1)
def get_filtered_datasets() -> list[str]:
    _uci_bin_native = [
        "breast-cancer",
        "german",
        "haberman",
        "ionosphere",
        "mammographic",
        "semeion",
        "sonar",
        "spambase",
        "spectf",
        "tictactoe",
        "transfusion",
        "wdbc",
    ]
    _uci_bin_names = [d for d in UCI_BINARY_DATASETS if d in _uci_bin_native]
    coll = "uci_binary"
    _sorted_bin_names = sort_datasets_by_size(coll, _uci_bin_names, fetch_UCIBinaryDataset)
    bin_names = _sorted_bin_names[:5]
    _uci_mul_names = UCI_MULTICLASS_DATASETS
    coll = "uci_multiclass"
    _sorted_mul_names = sort_datasets_by_size(coll, _uci_mul_names, fetch_UCIMulticlassDataset)
    mul_names = _sorted_mul_names[:10]
    return bin_names + mul_names
