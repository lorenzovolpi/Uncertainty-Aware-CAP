"""Compatibility entry point for the binform experiment."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from uncawe.experiments.binform import (
    BAYESCAP_PRIOR_C,
    CLASSIFIERS,
    COLLECTION,
    DATASET,
    DOMAIN,
    EXP,
    EXPERIMENT,
    METHOD_PRIOR_KINDS,
    NUM_TEST,
    REGULAR_METHOD_NAMES,
    TEST_PRIOR_C,
    TestSample,
    _method_names,
    _transform_prior,
    exp_protocol,
    experiments,
    main,
    run_experiment_args,
    save_experiment_results,
    shutdown_parallel_workers,
)

__all__ = [
    "BAYESCAP_PRIOR_C",
    "CLASSIFIERS",
    "COLLECTION",
    "DATASET",
    "DOMAIN",
    "EXP",
    "EXPERIMENT",
    "METHOD_PRIOR_KINDS",
    "NUM_TEST",
    "REGULAR_METHOD_NAMES",
    "TEST_PRIOR_C",
    "TestSample",
    "_method_names",
    "_transform_prior",
    "exp_protocol",
    "experiments",
    "main",
    "run_experiment_args",
    "save_experiment_results",
    "shutdown_parallel_workers",
]


if __name__ == "__main__":
    main()
