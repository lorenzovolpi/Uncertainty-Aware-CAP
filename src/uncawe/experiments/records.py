"""Build the shared tabular result schema emitted by main and binform experiments."""

from dataclasses import dataclass

import cap
import numpy as np

from uncawe.pretrain.data import PretrainInfo
from uncawe.results import RDF


@dataclass(frozen=True)
class EvaluationContext:
    pretrain: PretrainInfo
    shifts: list[float]
    true_cts: list[np.ndarray]
    train_prev: float | list[float]
    val_prev: float | list[float]

    @property
    def size(self) -> int:
        return len(self.true_cts)


def build_result_frame(
    context: EvaluationContext,
    *,
    method_name: str,
    acc_name: str,
    acc_fn,
    estim_acc_ranges: list,
    estim_accs: list,
    estim_cts: list,
    t_train: float,
    t_test_ave: float,
) -> RDF:
    p = context.pretrain
    true_accs = [acc_fn(ct) for ct in context.true_cts]
    acc_err = cap.error.ae(np.asarray(true_accs), np.asarray(estim_accs)).tolist()
    return RDF.from_records(
        context.size,
        shifts=context.shifts,
        true_cts=context.true_cts,
        estim_accs=estim_accs,
        estim_acc_ranges=estim_acc_ranges,
        acc_err=acc_err,
        estim_cts=estim_cts,
        classifier=p.h_info.name,
        method=method_name,
        dataset=p.d_info.name,
        collection=p.d_info.collection,
        n_classes=p.d_info.n_classes,
        acc_name=acc_name,
        train_prev=[context.train_prev] * context.size,
        val_prev=[context.val_prev] * context.size,
        t_train=t_train,
        t_test_ave=t_test_ave,
    )
