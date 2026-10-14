"""Select compatible pretrains and construct deterministic PrediQuant test samples."""

from dataclasses import dataclass

import numpy as np
from quapy.data import LabelledCollection

from uncawe.experiments.predist_config import LR_ALIASES, THREE_CLASS_DATASETS
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.pretrain.dataset import sort_datasets_by_size


@dataclass(frozen=True)
class TestSample:
    sample_id: int
    requested_prev: np.ndarray
    actual_prev: np.ndarray
    collection: LabelledCollection
    posteriors: np.ndarray


def _normalise_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def is_lr_classifier(p: PretrainInfo, requested: str) -> bool:
    requested_norm = _normalise_name(requested)
    accepted = {_normalise_name(alias) for alias in LR_ALIASES | {requested}}
    candidates = [p.h_info.name, p.h_info.full_name, p.h_info.class_name]
    return requested_norm in accepted and any(_normalise_name(str(candidate)) in accepted for candidate in candidates)


def three_class_dataset_names(requested: list[str] | None = None) -> list[str]:
    return sort_datasets_by_size("uci_multiclass", THREE_CLASS_DATASETS if requested is None else requested)


def load_filtered_pretrains(domain: str, classifier: str, datasets: list[str]) -> list[PretrainInfo]:
    wanted = set(datasets)
    pretrains = []
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.collection != "uci_multiclass" or p.d_info.n_classes != 3:
            continue
        if p.d_info.name not in wanted or not is_lr_classifier(p, classifier):
            continue
        pretrains.append(p)

    order = {name: index for index, name in enumerate(datasets)}
    pretrains.sort(key=lambda p: (order[p.d_info.name], p.h_info.full_name))
    return pretrains


def simplex_points(rng: np.random.RandomState, n_points: int, eps: float = 0.01) -> list[np.ndarray]:
    if not 3 <= n_points <= 5:
        raise ValueError("--num-test must be between 3 and 5 for the ternary predist experiment")

    candidates = [np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)]
    for cls in range(3):
        prev = np.full(3, eps / 2, dtype=float)
        prev[cls] = 1 - eps
        candidates.append(prev)
    if n_points <= len(candidates):
        return candidates[:n_points]

    random_pool = rng.dirichlet(np.ones(3), size=512)
    chosen = candidates.copy()
    while len(chosen) < n_points:
        distances = np.stack([np.linalg.norm(random_pool - prev, axis=1) for prev in chosen], axis=1)
        index = int(np.argmax(distances.min(axis=1)))
        chosen.append(random_pool[index])
        random_pool = np.delete(random_pool, index, axis=0)
    return chosen


def make_test_samples(dataset, classifier, requested_prevs, sample_size: int, seed: int) -> list[TestSample]:
    samples = []
    for sample_id, requested_prev in enumerate(requested_prevs):
        index = dataset.U.sampling_index(sample_size, *requested_prev, random_state=seed + sample_id)
        collection = dataset.U.sampling_from_index(index)
        posteriors = classifier.predict_proba(collection.X)
        samples.append(
            TestSample(
                sample_id=sample_id,
                requested_prev=requested_prev,
                actual_prev=collection.prevalence(),
                collection=collection,
                posteriors=posteriors,
            )
        )
    return samples
