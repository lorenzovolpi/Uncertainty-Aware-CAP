"""Centralize constants and distance functions for the PrediQuant distance experiment."""

from cap.models import utils as cap_utils

EXPERIMENT = "predist"
METHOD_NAME = "PrediQuant"
DEFAULT_DOMAIN = "classic"
DEFAULT_ACC_NAME = "vanilla_accuracy"
DEFAULT_CLASSIFIER = "LR"
DEFAULT_NUM_TEST = 5
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 0
DEFAULT_N_JOBS = 1
RESULT_SCHEMA_VERSION = 3
THREE_CLASS_DATASETS = [
    "connect-4",
    "page_block",
    "waveform-v1",
    "academic-success",
    "molecular",
    "cmc",
    "phishing",
    "mhr",
]
LR_ALIASES = {"lr", "logisticregression", "logistic_regression", "logistic regression"}
PREDIQUANT_DISTANCE_FNS = {
    "l1": cap_utils.l1_dist,
    "hellinger": cap_utils.hellinger_dist,
    "jensen-shannon": cap_utils.jensen_shannon_dist,
}
