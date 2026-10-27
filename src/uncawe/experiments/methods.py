"""Define experiment method factories and their shared hyperparameters."""

from collections.abc import Iterator

from cap.models.confidence import (
    CBPE,
    RQBS,
    BootstrapCTCAP,
    BootstrapDirectCAP,
    CAPWithConfidence,
    PrediQuant,
)
from cap.models.cont_table import O_LEAP, QuAccNxN
from cap.models.direct import DoC
from quapy.method.aggregative import PACC, KDEyML
from sklearn.neural_network import MLPClassifier as MLP

from uncawe.experiments.metrics import gen_acc_measure
from uncawe.methods.baycap import BayCAP

BAYES_WARMUPS = 500
BAYES_N_JOBS = 16
# environment of the worker processes (experiments/execution.py:run_tasks): one thread in the thread pools of the
# numerical libraries (the variables joblib sets in its workers), so that each worker keeps about one core busy
THREAD_LIMITS = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}
# BayCAP methods run on CPU with single-threaded XLA (docs/work_plan.md, Part V.1)
BAYES_ENV = THREAD_LIMITS | {
    "JAX_PLATFORMS": "cpu",
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
}
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
    "BayesCAP": BayCAP,
    "BayesCAP-real": BayCAP,
    "BayesCAP-EB": BayCAP,
    "BayesCAP-EB-real": BayCAP,
}
# estimand and prior on the columns of C of the BayCAP methods of main (methods/baycap.py:BayCAP): BayesCAP-real is
# E2.3, the EB methods are E2.4 of docs/ext2_baycap.md; all share the priors on pi, NUTS settings and default seed
BAYCAP_VARIANTS = {
    "BayesCAP": ("population", "uniform"),
    "BayesCAP-real": ("realised", "uniform"),
    "BayesCAP-EB": ("population", "empirical-bayes"),
    "BayesCAP-EB-real": ("realised", "empirical-bayes"),
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
    # BayesCAP is built from BayCAP, with the draws of cap-ml's ACC_BayesCAP, so that it stores the NUTS diagnostics
    for name, (estimand, c_prior) in BAYCAP_VARIANTS.items():
        yield (
            name,
            BayCAP(acc, estimand=estimand, c_prior=c_prior, num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS),
        )


def get_method_names() -> list[str]:
    return list(METHOD_CLASS)
