"""Define experiment method factories and their shared hyperparameters."""

from collections.abc import Iterator
from functools import partial

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
from uncawe.methods.ac import AC
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import CAL, INSTANCE, BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS
from uncawe.methods.bayes_kdey import LOGLOSS, LOO
from uncawe.methods.sqs_b import SQSB
from uncawe.methods.tcal import CAL_BAG_SIZE, CalibratedTemperature

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
    # AC of E8.1 of docs/ext8_average_confidence.md (card WP-29), see AC_METHOD below
    "AC": AC,
    "SQS-H": RQBS,
    "SQS-Hbs": RQBS,
    # SQS-B of E3.3 (card WP-18), see SQS_B below
    "SQS-B": SQSB,
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
    # E3.4 (card WP-31), see INSTANCE_VARIANTS below
    "BayCAP-KDEy-h-inst": BayCAPKDEy,
    # E3.5 (card WP-33), see LOGLOSS_VARIANTS below
    "BayCAP-KDEy-h-inst-cb": BayCAPKDEy,
    "BayCAP-KDEy-mlp": BayCAPKDEy,
    "BayCAP-KDEy-mlp-real": BayCAPKDEy,
    "BayCAP-EMQ-h": BayCAPMAPLS,
    "BayCAP-EMQ-h-real": BayCAPMAPLS,
    "BayCAP-EMQ-mlp": BayCAPMAPLS,
    "BayCAP-EMQ-mlp-real": BayCAPMAPLS,
    "BayCAP-KDEy-Hf-h-real": BayCAPKDEy,
    "BayCAP-KDEy-Hf-mlp-real": BayCAPKDEy,
}
# the allocations of SQS-S of E6.2 of docs/ext6_sqs_s_design.md (card WP-17): PQ with cap-ml's nested split at the share
# rho of V that trains the quantifier (PrediQuant(rho=...)); the level 0.5 is run too, since the nested split differs
# from the train_test_split of PQ (rho=None). Outside the main tables, ranks and default plots (tables/main.py)
PQ_RHO_LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
PQ_RHO_VARIANTS = {f"PQ-rho{rho}": rho for rho in PQ_RHO_LEVELS}
METHOD_CLASS |= {name: PrediQuant for name in PQ_RHO_VARIANTS}
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
# the method of E3.4 of docs/ext3_posterior_propagation.md (card WP-31, §T.15): BayCAP-KDEy-h-real with the per-instance
# pushforward (methods/baycap_kdey.py:BayCAPKDEy, estimand="realised-instance"), everything else, seeds included, as
# the stored method, so that the two have the same prevalence draws; name -> stored method. The BayCAP-KDEy method of
# the project since 2026-11-24 (user's decision), in the main tables in place of BayCAP-KDEy-mlp-real
INSTANCE_VARIANTS = {"BayCAP-KDEy-h-inst": "BayCAP-KDEy-h-real"}
# the method of E3.5 of docs/ext3_posterior_propagation.md (card WP-33, §T.16): BayCAP-KDEy-h-inst with BayesKDEy at
# bandwidth="logloss" (one bandwidth shared by the classes, selected per pair over methods/bayes_kdey.py:LOO_GRID by the
# class-balanced leave-one-out log-loss of the kernel class probabilities on V, ties towards 0.1, and used in both the
# likelihood and the per-instance weights), everything else, seeds included, as the method of E3.4; name -> method of
# E3.4. Outside the main tables, ranks and default plots until it is analysed (user's decision of 2026-11-24)
LOGLOSS_VARIANTS = {"BayCAP-KDEy-h-inst-cb": "BayCAP-KDEy-h-inst"}
# input classifier of the methods of E3.2 (methods/baycap_kdey.py:BayCAPKDEy with c_prior="hierarchical-f"): BayesKDEy
# with a random C, the Hf prior with centred shares and the kernel of E3.1; realised pushforward only; NUTS settings and
# default seed of the BayCAP methods
KDEY_HF_VARIANTS = {f"BayCAP-KDEy-Hf-{source}-real": source for source in ("h", "mlp")}
# the settings of BayesKDEy in E3.1 and E3.2
KDEY_SETTINGS = {"kernel": "gaussian", "bandwidth": 0.1, "shrinkage": 0.0}
# SQS-B of E3.3 (methods/sqs_b.py:SQSB, card WP-18): one bag of SAMPLE_SIZE instances of V per posterior draw of
# BayesKDEy with the configuration and seed of the quantifier of BayCAP-KDEy-mlp-real (MLP input, same settings, so the
# same prevalence draws); the bags are seeded with the experiment seed, as SQS-H(bs). Since WP-25, SQSB builds its own
# BayesKDEy (methods/bayes_kdey.py) from these settings, with the same draws
SQS_B = "SQS-B"
SQS_B_QUANTIFIER = "BayCAP-KDEy-mlp-real"
# the tempered methods of E5.2 of docs/ext5_baycap_priors_tempering.md (card WP-20): BayesCAP-Hf-real with its two
# likelihood terms divided by tau (methods/baycap.py:BayCAP, temperature) and BayCAP-KDEy-mlp-real with the likelihood
# of the bag divided by tau (methods/baycap_kdey.py:BayCAPKDEy, temperature), everything else as the untempered method,
# so tau=1 is the stored method itself; name -> (untempered method, tau). Outside the main tables, ranks and default
# plots (tables/main.py). Since WP-33 (user's decision of 2026-11-24) E5.2, E5.3 and their tables use the bases
# TEMPERED_BASES: BayesCAP-Hf-real and the two per-instance BayCAP-KDEy methods of E3.4 and E3.5 (the likelihood of the
# bag divided by tau, the per-instance weights untempered, with -cb the bandwidth selected in the fit as untempered);
# BayCAP-KDEy-mlp-real, whose results are invalidated (2026-11-24), keeps its tempered methods in the factory
# (INVALIDATED_TEMPERED_BASES), so that its stored results keep their names, but no study runs or tabulates them
TEMPERATURE_LEVELS = (0.5, 2.0, 4.0, 8.0)
# the grid of E5.3 (experiments/tau_select.py:TAUS) and of the methods of WP-34
TAU_GRID = tuple(sorted((1.0, *TEMPERATURE_LEVELS)))
TEMPERED_BASES = ("BayesCAP-Hf-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb")
INVALIDATED_TEMPERED_BASES = ("BayCAP-KDEy-mlp-real",)
# in the order of WP-20 first, then the bases added by WP-33
ALL_TEMPERED_BASES = ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb")
TEMPERED_VARIANTS = {f"{base}-t{tau:g}": (base, tau) for base in ALL_TEMPERED_BASES for tau in TEMPERATURE_LEVELS}
METHOD_CLASS |= {name: METHOD_CLASS[base] for name, (base, _) in TEMPERED_VARIANTS.items()}
# the methods of card WP-35: where the temperature acts with the realised step (methods/baycap.py:TEMPERINGS), for
# BayesCAP-real, BayesCAP-Hf-real and BayCAP-KDEy-h-inst-cb (user's choice of 2026-11-25), at the temperatures of E5.2:
# <base>-t<tau>-complete (the per-instance complete-data likelihood of the bag, with the latent true classes, raised to
# 1/tau) and <base>-t<tau>-obs (its term of the observation only), and BayesCAP-real-t<tau> (tempering="marginal", as
# the methods of E5.2, absent so far); everything else, seeds included, as the untempered method. Apart from
# TEMPERED_VARIANTS, so that E5.2, E5.3 and their tables are unchanged; outside the main tables, ranks and default plots
# (tables/main.py); name -> (untempered method, tau, tempering)
PLACEMENT_BASES = ("BayesCAP-real", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst-cb")
PLACEMENT_SUFFIXES = {"complete": "complete", "observation": "obs"}
PLACEMENT_VARIANTS = {f"BayesCAP-real-t{tau:g}": ("BayesCAP-real", tau, "marginal") for tau in TEMPERATURE_LEVELS} | {
    f"{base}-t{tau:g}-{suffix}": (base, tau, tempering)
    for base in PLACEMENT_BASES
    for tau in TEMPERATURE_LEVELS
    for tempering, suffix in PLACEMENT_SUFFIXES.items()
}
METHOD_CLASS |= {name: METHOD_CLASS[base] for name, (base, _, _) in PLACEMENT_VARIANTS.items()}
# the methods of E7.1 of docs/ext7_kdey_bandwidth.md (card WP-25): BayCAP-KDEy-mlp-real and SQS-B with BayesKDEy at
# bandwidth="loo" (one bandwidth per class, selected by leave-one-out likelihood on the cross-fitted validation outputs
# of the MLP over methods/bayes_kdey.py:LOO_GRID, densities normalised on the simplex), everything else, seeds included,
# as the stored method; name -> stored method. Outside the main tables, ranks and default plots (tables/main.py)
LOO_VARIANTS = {"BayCAP-KDEy-mlp-real-loo": SQS_B_QUANTIFIER, "SQS-B-loo": SQS_B}
METHOD_CLASS |= {name: METHOD_CLASS[base] for name, base in LOO_VARIANTS.items()}
# the method of E7.2 of docs/ext7_kdey_bandwidth.md (card WP-26): BayCAP-KDEy-mlp-real with BayesKDEy at
# bandwidth="cal" (one bandwidth shared by the classes, selected per pair over methods/bayes_kdey.py:LOO_GRID on 50
# calibration bags of 1000 instances drawn from half of V, by the accuracy error of the EM estimate, then the method
# fitted on the full V at the selected bandwidth), everything else, seeds included, as the stored method (E7.2a); the
# method -tcal also selects the temperature on the same calibration bags after the bandwidth, over
# methods/baycap_kdey.py:TAU_GRID by the mean Winkler score on vanilla accuracy (temperature="cal", E7.2b); name ->
# stored method. Outside the main tables, ranks and default plots (tables/main.py)
CAL_VARIANTS = {"BayCAP-KDEy-mlp-real-cal": SQS_B_QUANTIFIER, "BayCAP-KDEy-mlp-real-cal-tcal": SQS_B_QUANTIFIER}
# the methods of CAL_VARIANTS that also select the temperature (E7.2b)
CAL_TEMPERATURE_VARIANTS = frozenset({"BayCAP-KDEy-mlp-real-cal-tcal"})
METHOD_CLASS |= {name: METHOD_CLASS[base] for name, base in CAL_VARIANTS.items()}
# the methods of card WP-34: the bases of E5.3 with the temperature selected per pair and metric inside the fit, on 50
# calibration bags drawn from half of V, over the grid of E5.2 and E5.3 (methods/tcal.py:CalibratedTemperature), every
# fit built as the tempered method of E5.2 (same settings and seeds); for BayCAP-KDEy-h-inst-cb the deployed fit reuses
# the bandwidth selected on V_fit (user's decision of 2026-11-25); name -> base. Outside the main tables, ranks and
# default plots (tables/main.py); before WP-34, the main and rank tables assembled columns with these names from the
# stored runs of E5.2 and E5.3 (tables/tcal_methods.py, now <name>-asm)
TCAL_VARIANTS = {f"{base}-tcal": base for base in ("BayesCAP-Hf-real", "BayCAP-KDEy-h-inst-cb")}
METHOD_CLASS |= {name: CalibratedTemperature for name in TCAL_VARIANTS}
# Average Confidence of E8.1 of docs/ext8_average_confidence.md (card WP-29): the largest posterior recalibrated per bag
# by LasCal, the Poisson-binomial distribution of the correct predictions and its central interval
# (methods/ac.py:AC), on every dataset, vanilla accuracy only. Outside the main tables, ranks and default plots
# (tables/main.py, tables/rank.py), compared with L-CBPE in tables/average_confidence.py
AC_METHOD = "AC"
# the methods run by main in the BayCAP pool, with BAYES_N_JOBS workers and BAYES_ENV
BAYES_POOL_METHODS = (
    frozenset(BAYCAP_VARIANTS)
    | frozenset(FIXED_C_VARIANTS)
    | frozenset(INSTANCE_VARIANTS)
    | frozenset(LOGLOSS_VARIANTS)
    | frozenset(KDEY_HF_VARIANTS)
    | frozenset({SQS_B})
    | frozenset(TEMPERED_VARIANTS)
    | frozenset(PLACEMENT_VARIANTS)
    | frozenset(LOO_VARIANTS)
    | frozenset(CAL_VARIANTS)
    | frozenset(TCAL_VARIANTS)
)


# the MLPs of the quantifiers are seeded with the experiment seed, so that every fit trains the same MLP (FIX-3)
def kdey():
    return KDEyML(MLP(random_state=qp.environ["_R_SEED"]))


def pacc():
    return PACC(MLP(random_state=qp.environ["_R_SEED"]))


def sqs_h(acc, bootstrap: bool, sample_size: int = SAMPLE_SIZE) -> RQBS:
    """SQS-H (``bootstrap=False``, one estimate of the target prevalence) or SQS-Hbs (``bootstrap=True``, one bag per
    bootstrap estimate of the prevalence): cap-ml's RQBS with PACC(MLP), ``BOOTSTRAP_ITERS`` bags of ``sample_size``
    instances of $V$ ($1000$ in ``main``; the size of the target bag in E4.4, ``experiments/target_size.py``)."""
    return RQBS(acc, pacc(), num_samples=BOOTSTRAP_ITERS, sample_size=sample_size, bootstrap=bootstrap)


def sqs_s(acc, rho: float | None = None, sample_size: int = SAMPLE_SIZE) -> PrediQuant:
    """SQS-S as in ``PQ`` (PACC(MLP), 10,000 candidates, gamma=0.1, L1, m=1000), with the allocation ``rho``;
    ``sample_size`` is the size of the candidate bags ($1000$ in ``main``; the size of the target bag in E4.4)."""
    return PrediQuant(
        acc,
        pacc(),
        num_samples=BOOTSTRAP_ITERS,
        sample_size=sample_size,
        alpha=0.1,
        distance="l1",
        predict_train_prev=True,
        rho=rho,
    )


def fixed_c_method(
    acc,
    name: str,
    prior="uniform",
    temperature: float | str = 1.0,
    bandwidth=KDEY_SETTINGS["bandwidth"],
    estimand: str | None = None,
    **kdey_options,
) -> BayCAPKDEy | BayCAPMAPLS:
    """The method ``name`` of E3.1 (``FIXED_C_VARIANTS``), with the NUTS settings and default seed of BayCAP; ``prior``
    is the Dirichlet parameter of the prior on $\\pi$ (uniform in ``main``; E5.1, ``experiments/binform.py``, varies
    it), ``temperature`` the posterior temperature ($1$ in E3.1 and E5.1; the other values are those of E5.2; ``"cal"``
    in E7.2b), and
    ``bandwidth`` that of BayesKDEy ($0.1$; ``"loo"`` in E7.1, ``"cal"`` in E7.2, ``"logloss"`` in E3.5); ``kdey_options`` are further
    arguments of ``BayCAPKDEy`` (e.g. ``cal_nuts_check``); ``estimand``, if given, replaces that of ``name``
    (``"realised-instance"`` in E3.4)."""
    quantifier, source, stored_estimand = FIXED_C_VARIANTS[name]
    estimand = stored_estimand if estimand is None else estimand
    common = {
        "input_classifier": source,
        "estimand": estimand,
        "temperature": temperature,
        "prior": prior,
        "num_samples": BOOTSTRAP_ITERS,
        "num_warmup": BAYES_WARMUPS,
    }
    if quantifier == "kdey":
        return BayCAPKDEy(acc, c_prior="fixed", **(KDEY_SETTINGS | {"bandwidth": bandwidth}), **common, **kdey_options)
    if kdey_options:
        raise ValueError(f"{name} takes no option of BayCAPKDEy; found {sorted(kdey_options)}")
    return BayCAPMAPLS(acc, **common)


def instance_method(acc, name: str, temperature: float = 1.0, prior="uniform") -> BayCAPKDEy:
    """The method ``name`` of E3.4 (``INSTANCE_VARIANTS``): its stored method of E3.1 with the per-instance
    pushforward (``estimand="realised-instance"``), everything else as the stored method; ``temperature`` divides the
    likelihood of the bag ($1$ except in E5.2 and E5.3), the per-instance weights stay untempered; ``prior`` is the
    Dirichlet parameter of the prior on $\\pi$ (uniform except in E5.5)."""
    return fixed_c_method(acc, INSTANCE_VARIANTS[name], estimand=INSTANCE, temperature=temperature, prior=prior)


def logloss_method(acc, name: str, temperature: float = 1.0, prior="uniform", **kdey_options) -> BayCAPKDEy:
    """The method ``name`` of E3.5 (``LOGLOSS_VARIANTS``): its method of E3.4 with ``bandwidth="logloss"`` (grid
    ``LOO_GRID``), everything else as the method of E3.4; ``temperature`` and ``prior`` as in :func:`instance_method`
    (the selection of the bandwidth depends on neither); ``kdey_options`` are further arguments of ``BayCAPKDEy``
    (``tempering`` in WP-35)."""
    return fixed_c_method(
        acc,
        INSTANCE_VARIANTS[LOGLOSS_VARIANTS[name]],
        estimand=INSTANCE,
        bandwidth=LOGLOSS,
        temperature=temperature,
        prior=prior,
        **kdey_options,
    )


def kdey_method(acc, name: str, temperature: float = 1.0, prior="uniform") -> BayCAPKDEy | BayCAPMAPLS:
    """A method with a fixed $\\hat C$ by name (``FIXED_C_VARIANTS``, ``INSTANCE_VARIANTS`` or ``LOGLOSS_VARIANTS``) at
    the temperature ``temperature`` and with the prior ``prior`` on $\\pi$."""
    if name in LOGLOSS_VARIANTS:
        return logloss_method(acc, name, temperature=temperature, prior=prior)
    if name in INSTANCE_VARIANTS:
        return instance_method(acc, name, temperature=temperature, prior=prior)
    return fixed_c_method(acc, name, temperature=temperature, prior=prior)


def sqs_b(acc, bandwidth=KDEY_SETTINGS["bandwidth"], prior="uniform", sample_size: int = SAMPLE_SIZE) -> SQSB:
    """SQS-B of E3.3 with BayesKDEy as the quantifier of ``SQS_B_QUANTIFIER`` (input classifier, kernel, prior,
    temperature, NUTS settings and default seed), bags of ``sample_size`` instances (``SAMPLE_SIZE`` in ``main``; the
    size of the target bag in E4.4), and the bandwidth ``bandwidth`` ($0.1$; ``"loo"`` in E7.1); ``prior`` is the
    Dirichlet parameter of the prior on $\\pi$ (uniform in ``main``; E5.5, ``experiments/binform.py``, varies it)."""
    _, source, _ = FIXED_C_VARIANTS[SQS_B_QUANTIFIER]
    return SQSB(
        acc,
        input_classifier=source,
        **(KDEY_SETTINGS | {"bandwidth": bandwidth}),
        num_samples=BOOTSTRAP_ITERS,
        num_warmup=BAYES_WARMUPS,
        sample_size=sample_size,
        prior=prior,
    )


def loo_method(acc, name: str) -> BayCAPKDEy | SQSB:
    """The method ``name`` of E7.1 (``LOO_VARIANTS``): its stored method with ``bandwidth="loo"``."""
    base = LOO_VARIANTS[name]
    if base == SQS_B:
        return sqs_b(acc, bandwidth=LOO)
    return fixed_c_method(acc, base, bandwidth=LOO)


# the solver of the estimate of pi on a calibration bag in the methods of E7.2: QuaPy's KDEyML (SLSQP), as the
# bandwidth selection of the BayesKDEy paper (user's decision of 2026-11-20, after the gate of the solver; the pilot of
# E7.2a used EM, the default of BayCAPKDEy)
CAL_SOLVER = "slsqp"


def cal_method(acc, name: str, nuts_check: bool = False) -> BayCAPKDEy:
    """The method ``name`` of E7.2 (``CAL_VARIANTS``): its stored method with ``bandwidth="cal"`` (grid ``LOO_GRID``,
    $50$ calibration bags of $1000$ instances, the estimate of QuaPy's ``KDEyML``, ``CAL_SOLVER``), and for the methods
    of ``CAL_TEMPERATURE_VARIANTS`` with ``temperature="cal"`` (grid ``TAU_GRID``); ``nuts_check`` adds the pilot check
    of the estimate against the NUTS posterior mean (``cal_nuts_check``)."""
    temperature = CAL if name in CAL_TEMPERATURE_VARIANTS else 1.0
    return fixed_c_method(
        acc,
        CAL_VARIANTS[name],
        temperature=temperature,
        bandwidth=CAL,
        cal_nuts_check=nuts_check,
        cal_solver=CAL_SOLVER,
    )


def baycap_method(acc, name: str, temperature: float = 1.0, tempering: str = "marginal", prior="uniform") -> BayCAP:
    """The BayCAP method ``name`` of ``BAYCAP_VARIANTS``, with the NUTS settings and default seed of BayCAP, its
    options (``BAYCAP_OPTIONS``), the posterior temperature ``temperature`` ($1$ except in E5.2 and WP-35), the
    placement ``tempering`` of the temperature (``"marginal"`` except in WP-35) and the prior ``prior`` on $\\pi$
    (uniform except in E5.1, E5.4 and E5.5)."""
    estimand, c_prior = BAYCAP_VARIANTS[name]
    return BayCAP(
        acc,
        estimand=estimand,
        c_prior=c_prior,
        num_samples=BOOTSTRAP_ITERS,
        num_warmup=BAYES_WARMUPS,
        temperature=temperature,
        tempering=tempering,
        prior=prior,
        **BAYCAP_OPTIONS.get(name, {}),
    )


def tcal_base(acc, base: str, temperature: float = 1.0, bandwidth: float | None = None, prior="uniform"):
    """The base ``base`` of ``TCAL_VARIANTS`` at the temperature ``temperature``, built as the tempered method of E5.2
    (``tempered_method``) with the prior ``prior`` on $\\pi$; for ``BayCAP-KDEy-h-inst-cb``, ``bandwidth`` fixes the
    bandwidth of BayesKDEy in place of its selection in the fit (``None``: selected, ``bandwidth="logloss"``)."""
    if base in BAYCAP_VARIANTS:
        if bandwidth is not None:
            raise ValueError(f"{base} has no bandwidth")
        return baycap_method(acc, base, temperature=temperature, prior=prior)
    if base not in LOGLOSS_VARIANTS:
        raise ValueError(f"No base {base} of the -tcal methods; the bases are {sorted(set(TCAL_VARIANTS.values()))}")
    if bandwidth is None:
        return logloss_method(acc, base, temperature=temperature, prior=prior)
    return fixed_c_method(
        acc,
        INSTANCE_VARIANTS[LOGLOSS_VARIANTS[base]],
        estimand=INSTANCE,
        bandwidth=float(bandwidth),
        temperature=temperature,
        prior=prior,
    )


def tcal_method(acc, name: str, prior="uniform", cal_bag_size: int = CAL_BAG_SIZE) -> CalibratedTemperature:
    """The method ``name`` of WP-34 (``TCAL_VARIANTS``): its base with $\\tau$ selected per metric inside the fit over
    ``TAU_GRID`` (the grid of E5.3), on $50$ calibration bags of ``cal_bag_size`` instances ($1000$; the size of the
    target bags in E4.4a), with the prior ``prior`` on $\\pi$ (uniform except in E5.5); the deployed fit of the KDEy
    base reuses the bandwidth selected on $V_{\\text{fit}}$."""
    return CalibratedTemperature(
        partial(tcal_base, acc, TCAL_VARIANTS[name], prior=prior),
        metrics=gen_acc_measure,
        taus=TAU_GRID,
        cal_bag_size=cal_bag_size,
        bandwidth="reuse",
    )


def tempered_method(acc, name: str) -> BayCAP | BayCAPKDEy:
    """The tempered method ``name`` of E5.2 (``TEMPERED_VARIANTS``): its untempered method with the temperature of
    ``name``."""
    base, tau = TEMPERED_VARIANTS[name]
    if base in BAYCAP_VARIANTS:
        return baycap_method(acc, base, temperature=tau)
    return kdey_method(acc, base, temperature=tau)


def placement_method(acc, name: str) -> BayCAP | BayCAPKDEy:
    """The method ``name`` of WP-35 (``PLACEMENT_VARIANTS``): its untempered method with the temperature and the
    placement ``tempering`` of ``name``."""
    base, tau, tempering = PLACEMENT_VARIANTS[name]
    if base in BAYCAP_VARIANTS:
        return baycap_method(acc, base, temperature=tau, tempering=tempering)
    return logloss_method(acc, base, temperature=tau, tempering=tempering)


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
    yield AC_METHOD, AC(acc_name, confidence_level=CONFIDENCE_LEVEL)
    # SQS-H as in the paper (one estimate of the target prevalence) and with a bootstrap of the prevalence (one bag per
    # bootstrap estimate; the method called RQBS until 2026-11-10), cap-ml's RQBS with bootstrap=False and True (FIX-2)
    yield "SQS-H", sqs_h(acc, bootstrap=False)
    yield "SQS-Hbs", sqs_h(acc, bootstrap=True)
    yield SQS_B, sqs_b(acc)
    yield "PQ", sqs_s(acc)
    # BayesCAP is built from BayCAP, with the draws of cap-ml's ACC_BayesCAP, so that it stores the NUTS diagnostics
    for name in BAYCAP_VARIANTS:
        yield name, baycap_method(acc, name)
    for name in FIXED_C_VARIANTS:
        yield name, fixed_c_method(acc, name)
        # the method of E3.4 right after its stored method, in the order of METHOD_CLASS
        for variant in (v for v, base in INSTANCE_VARIANTS.items() if base == name):
            yield variant, instance_method(acc, variant)
            # the method of E3.5 right after its method of E3.4, in the order of METHOD_CLASS
            for cb in (v for v, inst in LOGLOSS_VARIANTS.items() if inst == variant):
                yield cb, logloss_method(acc, cb)
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

    for name, rho in PQ_RHO_VARIANTS.items():
        yield name, sqs_s(acc, rho)
    for name in TEMPERED_VARIANTS:
        yield name, tempered_method(acc, name)
    for name in PLACEMENT_VARIANTS:
        yield name, placement_method(acc, name)
    for name in LOO_VARIANTS:
        yield name, loo_method(acc, name)
    for name in CAL_VARIANTS:
        yield name, cal_method(acc, name)
    for name in TCAL_VARIANTS:
        yield name, tcal_method(acc, name)


def get_method_names() -> list[str]:
    return list(METHOD_CLASS)
