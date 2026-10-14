"""Define accuracy metrics used for method selection and result evaluation."""

from collections.abc import Callable, Iterator
from typing import Literal

import numpy as np
from cap.error import f1, f1_macro, k_bin, k_macro, smooth, vanilla_acc

ACCURACY_NAMES = ("vanilla_accuracy", "macro-F1", "macro-K")


def get_acc_names() -> list[str]:
    return list(ACCURACY_NAMES)


def get_selection_acc(name: str, multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": smooth(f1_macro) if multiclass else smooth(f1),
        "macro-K": k_macro if multiclass else k_bin,
    }[name]


def get_evaluation_acc(name: str, is_multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": f1_macro if is_multiclass else f1,
        "macro-K": k_macro if is_multiclass else k_bin,
    }[name]


def acc_from_ct(
    acc_name: str,
    ct: np.ndarray,
    type: Literal["selection", "evaluation"] = "selection",
) -> float | None:
    multiclass = ct.shape[0] > 2
    if type == "selection":
        return get_selection_acc(acc_name, multiclass)(ct)
    if type == "evaluation":
        return get_evaluation_acc(acc_name, multiclass)(ct)
    return None


def gen_acc_measure(is_multiclass: bool) -> Iterator[tuple[str, Callable]]:
    for acc_name in ACCURACY_NAMES:
        yield acc_name, get_selection_acc(acc_name, is_multiclass)
