"""Select experiment datasets and identify datasets with complete result grids."""

from collections import defaultdict
from functools import lru_cache

from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS

from uncawe.experiments.methods import get_method_names
from uncawe.experiments.metrics import get_acc_names
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.pretrain.dataset import sort_datasets_by_size
from uncawe.util import all_results_exist


def get_existing_dataset_names(experiment: str, domain: str, sort: bool = True):
    dataset_h_map = defaultdict(lambda: True)
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        key = (p.d_info.name, p.d_info.collection)
        dataset_h_map[key] = dataset_h_map[key] and all_results_exist(
            p.domain,
            p.d_info.name,
            p.h_info.full_name,
            get_method_names(),
            get_acc_names(),
            experiment,
        )

    datasets = [dataset for dataset, complete in dataset_h_map.items() if complete]
    dataset_names, dataset_colls = tuple(map(lambda values: list(values), zip(*datasets)))
    return sort_datasets_by_size(dataset_colls, dataset_names) if sort else dataset_names


@lru_cache(maxsize=1)
def get_filtered_datasets() -> list[str]:
    uci_binary_native = [
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
    binary_names = [name for name in UCI_BINARY_DATASETS if name in uci_binary_native]
    sorted_binary = sort_datasets_by_size("uci_binary", binary_names, fetch_UCIBinaryDataset)
    sorted_multiclass = sort_datasets_by_size(
        "uci_multiclass",
        UCI_MULTICLASS_DATASETS,
        fetch_UCIMulticlassDataset,
    )
    return sorted_binary[:5] + sorted_multiclass[:10]
