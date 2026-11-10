"""Centralize constants and distance functions for the PrediQuant distance experiment."""

from cap.models import utils as cap_utils

EXPERIMENT = "predist"
METHOD_NAME = "PrediQuant"
DEFAULT_DOMAIN = "classic"
DEFAULT_ACC_NAME = "vanilla_accuracy"
# the five classifiers of main (pretrain/classic.py), as named by ClassifierInfo.name (E6.1, WP-16)
DEFAULT_CLASSIFIERS = ("LR", "kNN", "SVM", "MLP", "RF")
DEFAULT_NUM_TEST = 5
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 0
DEFAULT_N_JOBS = 1
# 4 (WP-16): PACC(MLP) as the quantifier of PrediQuant (before: KDEy(MLP)), several classifiers, the new columns
# quantifier, region and true_dist_l1_mean
RESULT_SCHEMA_VERSION = 4
# the surrogate quantifier of PrediQuant, as in PQ of main (experiments/methods.py:pacc)
QUANTIFIER_NAME = "PACC(MLP)"
# the settings of PrediQuant shared with PQ of main: m = num_samples selected bags out of num_samples / alpha candidates
PREDIQUANT_ALPHA = 0.1
DEFAULT_NUM_SAMPLES = 1000
# position on the simplex of the test bags of simplex_points (predist_sampling.py), by sample_id
SAMPLE_REGIONS = {0: "centre", 1: "vertex", 2: "vertex", 3: "vertex", 4: "random"}
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
PREDIQUANT_DISTANCE_FNS = {
    "l1": cap_utils.l1_dist,
    "hellinger": cap_utils.hellinger_dist,
    "jensen-shannon": cap_utils.jensen_shannon_dist,
}
