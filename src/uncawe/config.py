"""Compatibility facade for experiment definitions.

New code should import from the focused modules under :mod:`uncawe.experiments`.
"""

from uncawe.experiments.datasets import get_existing_dataset_names, get_filtered_datasets
from uncawe.experiments.methods import (
    BAYES_N_JOBS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    CONFIDENCE_LEVEL,
    JAX_COMPILATION_CACHE_DIR,
    METHOD_CLASS,
    SAMPLE_SIZE,
    gen_methods,
    get_method_names,
    kdey,
    pacc,
)
from uncawe.experiments.metrics import (
    acc_from_ct,
    gen_acc_measure,
    get_acc_names,
    get_evaluation_acc,
    get_selection_acc,
)

__all__ = [
    "BAYES_N_JOBS",
    "BAYES_WARMUPS",
    "BOOTSTRAP_ITERS",
    "CONFIDENCE_LEVEL",
    "JAX_COMPILATION_CACHE_DIR",
    "METHOD_CLASS",
    "SAMPLE_SIZE",
    "acc_from_ct",
    "gen_acc_measure",
    "gen_methods",
    "get_acc_names",
    "get_evaluation_acc",
    "get_existing_dataset_names",
    "get_filtered_datasets",
    "get_method_names",
    "get_selection_acc",
    "kdey",
    "pacc",
]
