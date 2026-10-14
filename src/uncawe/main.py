"""Compatibility entry point for the main experiment."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from uncawe.experiments.main import (
    DOMAIN,
    EXP,
    EXPERIMENT,
    NUM_TEST,
    exp_protocol,
    experiments,
    main,
    run_experiment_args,
    save_experiment_results,
    shutdown_parallel_workers,
)

__all__ = [
    "DOMAIN",
    "EXP",
    "EXPERIMENT",
    "NUM_TEST",
    "exp_protocol",
    "experiments",
    "main",
    "run_experiment_args",
    "save_experiment_results",
    "shutdown_parallel_workers",
]


if __name__ == "__main__":
    main()
