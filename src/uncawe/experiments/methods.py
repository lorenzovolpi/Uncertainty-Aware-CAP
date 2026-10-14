"""Define experiment method factories and their shared hyperparameters."""

from collections.abc import Iterator

from cap.models.confidence import (
    ACC_BayesCAP,
    BootstrapCTCAP,
    BootstrapDirectCAP,
    CAPWithConfidence,
    CBPE,
    PrediQuant,
    RQBS,
)
from cap.models.cont_table import O_LEAP, QuAccNxN
from cap.models.direct import DoC
from quapy.method.aggregative import KDEyML, PACC
from sklearn.neural_network import MLPClassifier as MLP

from uncawe.experiments.metrics import gen_acc_measure

BAYES_WARMUPS = 500
BAYES_N_JOBS = 1
BOOTSTRAP_ITERS = 1000
SAMPLE_SIZE = 1000
JAX_COMPILATION_CACHE_DIR = "./output/jax_cache"
CONFIDENCE_LEVEL = 0.95

METHOD_CLASS = {
    "DoC": DoC,
    "QuAcc_NxN": QuAccNxN,
    "O-LEAP": O_LEAP,
    "L-CBPE": CBPE,
    "RQBS": RQBS,
    "PQ": PrediQuant,
    "BayesCAP": ACC_BayesCAP,
}


def kdey():
    return KDEyML(MLP())


def pacc():
    return PACC(MLP())


def gen_methods() -> Iterator[tuple[str, CAPWithConfidence]]:
    acc_name, acc = next(gen_acc_measure(True))
    yield "DoC", BootstrapDirectCAP(DoC(acc), num_samples=BOOTSTRAP_ITERS)
    yield (
        "QuAcc_NxN",
        BootstrapCTCAP(
            QuAccNxN(acc, kdey(), add_maxconf=True, add_negentropy=True, add_maxinfsoft=True),
            num_samples=BOOTSTRAP_ITERS,
        ),
    )
    yield "O-LEAP", BootstrapCTCAP(O_LEAP(acc, kdey()), num_samples=BOOTSTRAP_ITERS)
    yield "L-CBPE", CBPE(acc_name, calib_method="lascal")
    yield "RQBS", RQBS(acc, pacc(), num_samples=BOOTSTRAP_ITERS, sample_size=SAMPLE_SIZE)
    yield (
        "PQ",
        PrediQuant(
            acc,
            pacc(),
            num_samples=BOOTSTRAP_ITERS,
            sample_size=SAMPLE_SIZE,
            alpha=0.1,
            distance="l1",
            predict_train_prev=True,
        ),
    )
    yield "BayesCAP", ACC_BayesCAP(acc, num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS)


def get_method_names() -> list[str]:
    return list(METHOD_CLASS)
