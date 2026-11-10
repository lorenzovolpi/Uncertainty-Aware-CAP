"""UCI multiclass datasets with a class filter based on the expected size of V (FIX-8).

QuaPy drops the classes with fewer than ``QUAPY_MIN_CLASS_SUPPORT`` instances of
the full collection, caps the training part at ``MAX_TRAIN_INSTANCES`` and cap-ml's
``split_train`` halves it into T and V. A class with ``s_j`` instances of the ``n``
of the collection has then about ``s_j * n_train * TRAIN_VAL_SPLIT / n`` instances
in V. ``fetch_uci_multiclass`` keeps a class only if this expected count is at
least ``MIN_EXPECTED_V_SUPPORT`` (and the class has at least the QuaPy threshold).
"""

import functools
import math
from fractions import Fraction

from cap.data.datasets import fetch_UCIMulticlassDataset
from cap.environment import env
from quapy.data.datasets import fetch_UCIMulticlassLabelledCollection

# QuaPy's defaults in fetch_UCIMulticlassDataset, and cap-ml's split of the training part.
QUAPY_MIN_CLASS_SUPPORT = 100
MIN_TEST_SPLIT = 0.3
MAX_TRAIN_INSTANCES = 25000
TRAIN_VAL_SPLIT = 0.5

# k of the rule: minimum expected number of instances of a class in V.
MIN_EXPECTED_V_SUPPORT = 10


def n_train_instances(n: int) -> int:
    """Size of the training part (T and V) of a collection of ``n`` instances, as in QuaPy."""
    n_train = int(n * (1.0 - MIN_TEST_SPLIT))
    return min(n_train, MAX_TRAIN_INSTANCES)


def class_support_threshold(n: int, k: int = MIN_EXPECTED_V_SUPPORT) -> int:
    """Minimum number of instances of a class in the collection, so that it has at least ``k``
    expected instances in V: ``max(100, ceil(k * n / (n_train * TRAIN_VAL_SPLIT)))``."""
    expected_v_share = Fraction(n_train_instances(n)) * Fraction(TRAIN_VAL_SPLIT) / n
    return max(QUAPY_MIN_CLASS_SUPPORT, math.ceil(k / expected_v_share))


@functools.lru_cache(maxsize=None)
def min_class_support(dataset_name: str, k: int = MIN_EXPECTED_V_SUPPORT) -> int:
    """Threshold of the rule for a dataset; ``n`` is the size of the collection after QuaPy's
    default filter (the collection on which the threshold is applied)."""
    coll = fetch_UCIMulticlassLabelledCollection(
        dataset_name,
        data_home=env["QUAPY_DATA"],
        min_class_support=QUAPY_MIN_CLASS_SUPPORT,
        standardize=False,
    )
    return class_support_threshold(len(coll), k)


def fetch_uci_multiclass(dataset_name: str, k: int = MIN_EXPECTED_V_SUPPORT):
    """cap-ml's ``fetch_UCIMulticlassDataset`` with the threshold of the rule; returns (T, V, U)."""
    return fetch_UCIMulticlassDataset(dataset_name, min_class_support=min_class_support(dataset_name, k))
