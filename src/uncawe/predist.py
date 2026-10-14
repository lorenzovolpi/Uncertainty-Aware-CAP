"""Compatibility entry point for the PrediQuant distance experiment."""

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

from uncawe.experiments.predist import (
    DEFAULT_ACC_NAME,
    DEFAULT_CLASSIFIER,
    DEFAULT_DOMAIN,
    DEFAULT_N_JOBS,
    DEFAULT_NUM_TEST,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    EXPERIMENT,
    LR_ALIASES,
    METHOD_NAME,
    PREDIQUANT_DISTANCE_FNS,
    RESULT_SCHEMA_VERSION,
    THREE_CLASS_DATASETS,
    TestSample,
    main,
    parse_args,
    run,
)

__all__ = [
    "DEFAULT_ACC_NAME",
    "DEFAULT_CLASSIFIER",
    "DEFAULT_DOMAIN",
    "DEFAULT_N_JOBS",
    "DEFAULT_NUM_TEST",
    "DEFAULT_SAMPLE_SIZE",
    "DEFAULT_SEED",
    "EXPERIMENT",
    "LR_ALIASES",
    "METHOD_NAME",
    "PREDIQUANT_DISTANCE_FNS",
    "RESULT_SCHEMA_VERSION",
    "THREE_CLASS_DATASETS",
    "TestSample",
    "main",
    "parse_args",
    "run",
]


if __name__ == "__main__":
    main()
