"""Define experiment method factories and their shared hyperparameters."""

from collections.abc import Iterator

import quapy as qp
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
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS

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
    "SQS-H": RQBS,
    "SQS-Hbs": RQBS,
    "PQ": PrediQuant,
    "BayesCAP": BayCAP,
    "BayesCAP-real": BayCAP,
    "BayesCAP-EB": BayCAP,
    "BayesCAP-EB-real": BayCAP,
    "BayesCAP-H-real": BayCAP,
    "BayesCAP-Hf-real": BayCAP,
    "BayesCAP-Hfn-real": BayCAP,
    "BayCAP-KDEy-h": BayCAPKDEy,
    "BayCAP-KDEy-h-real": BayCAPKDEy,
    "BayCAP-KDEy-mlp": BayCAPKDEy,
    "BayCAP-KDEy-mlp-real": BayCAPKDEy,
    "BayCAP-EMQ-h": BayCAPMAPLS,
    "BayCAP-EMQ-h-real": BayCAPMAPLS,
    "BayCAP-EMQ-mlp": BayCAPMAPLS,
    "BayCAP-EMQ-mlp-real": BayCAPMAPLS,
    "BayCAP-KDEy-Hf-h-real": BayCAPKDEy,
    "BayCAP-KDEy-Hf-mlp-real": BayCAPKDEy,
}
# estimand and prior on the columns of C of the BayCAP methods of main (methods/baycap.py:BayCAP): BayesCAP-real is
# E2.3, the EB methods are E2.4, BayesCAP-H-real is E2.7, BayesCAP-Hf-real is E2.8 and BayesCAP-Hfn-real is E2.9 of
# docs/ext2_baycap.md; all share the priors on pi, NUTS settings and default seed
BAYCAP_VARIANTS = {
    "BayesCAP": ("population", "uniform"),
    "BayesCAP-real": ("realised", "uniform"),
    "BayesCAP-EB": ("population", "empirical-bayes"),
    "BayesCAP-EB-real": ("realised", "empirical-bayes"),
    "BayesCAP-H-real": ("realised", "hierarchical"),
    "BayesCAP-Hf-real": ("realised", "hierarchical-f"),
    "BayesCAP-Hfn-real": ("realised", "hierarchical-f"),
}
# the other hyperparameters of BayCAP that differ from its defaults: BayesCAP-Hfn-real samples the model of
# BayesCAP-Hf-real with non-centred shares of the mistakes (E2.9, §T.17)
BAYCAP_OPTIONS = {"BayesCAP-Hfn-real": {"shares": "non-centred"}}
# quantifier, input classifier and estimand of the methods with a fixed rate matrix of E3.1 of
# docs/ext3_posterior_propagation.md (methods/baycap_kdey.py:BayCAPKDEy with c_prior="fixed" for KDEy,
# methods/baycap_mapls.py:BayCAPMAPLS for EMQ; before WP-13 both were methods/baycap_fixed.py:BayCAPFixed, with the same
# draws); BayesKDEy with the Gaussian kernel, bandwidth 0.1 and no shrinkage, BayesianMAPLS as in QuaPy; NUTS settings
# and default seed of the BayCAP methods. The population and realised method of a cell share their prevalence draws
FIXED_C_VARIANTS = {
    f"BayCAP-{label}-{source}{suffix}": (quantifier, source, estimand)
    for label, quantifier in (("KDEy", "kdey"), ("EMQ", "emq"))
    for source in ("h", "mlp")
    for suffix, estimand in (("", "population"), ("-real", "realised"))
}
# input classifier of the methods of E3.2 (methods/baycap_kdey.py:BayCAPKDEy with c_prior="hierarchical-f"): BayesKDEy
# with a random C, the Hf prior with centred shares and the kernel of E3.1; realised pushforward only; NUTS settings and
# default seed of the BayCAP methods
KDEY_HF_VARIANTS = {f"BayCAP-KDEy-Hf-{source}-real": source for source in ("h", "mlp")}
# the settings of BayesKDEy in E3.1 and E3.2
KDEY_SETTINGS = {"kernel": "gaussian", "bandwidth": 0.1, "shrinkage": 0.0}
# the methods run by main in the BayCAP pool, with BAYES_N_JOBS workers and BAYES_ENV
BAYES_POOL_METHODS = frozenset(BAYCAP_VARIANTS) | frozenset(FIXED_C_VARIANTS) | frozenset(KDEY_HF_VARIANTS)


# the MLPs of the quantifiers are seeded with the experiment seed, so that every fit trains the same MLP (FIX-3)
def kdey():
    return KDEyML(MLP(random_state=qp.environ["_R_SEED"]))


def pacc():
    return PACC(MLP(random_state=qp.environ["_R_SEED"]))


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
    # SQS-H as in the paper (one estimate of the target prevalence) and with a bootstrap of the prevalence (one bag per
    # bootstrap estimate; the method called RQBS until 2026-11-10), cap-ml's RQBS with bootstrap=False and True (FIX-2)
    yield "SQS-H", RQBS(acc, pacc(), num_samples=BOOTSTRAP_ITERS, sample_size=SAMPLE_SIZE, bootstrap=False)
    yield "SQS-Hbs", RQBS(acc, pacc(), num_samples=BOOTSTRAP_ITERS, sample_size=SAMPLE_SIZE, bootstrap=True)
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
            BayCAP(
                acc,
                estimand=estimand,
                c_prior=c_prior,
                num_samples=BOOTSTRAP_ITERS,
                num_warmup=BAYES_WARMUPS,
                **BAYCAP_OPTIONS.get(name, {}),
            ),
        )
    for name, (quantifier, source, estimand) in FIXED_C_VARIANTS.items():
        common = {
            "input_classifier": source,
            "estimand": estimand,
            "temperature": 1.0,
            "num_samples": BOOTSTRAP_ITERS,
            "num_warmup": BAYES_WARMUPS,
        }
        if quantifier == "kdey":
            yield name, BayCAPKDEy(acc, c_prior="fixed", **KDEY_SETTINGS, **common)
        else:
            yield name, BayCAPMAPLS(acc, **common)
    for name, source in KDEY_HF_VARIANTS.items():
        yield (
            name,
            BayCAPKDEy(
                acc,
                input_classifier=source,
                estimand="realised",
                c_prior="hierarchical-f",
                **KDEY_SETTINGS,
                temperature=1.0,
                num_samples=BOOTSTRAP_ITERS,
                num_warmup=BAYES_WARMUPS,
            ),
        )


def get_method_names() -> list[str]:
    return list(METHOD_CLASS)
