r"""The study ``target_size``: the error split of two Bayesian CAP methods across the size $|V|$ of the validation set
and the size $|U|$ of the target bag (E4.2 of ``docs/ext4_uncertainty_budget.md``; work-plan card WP-21 for E4.2b,
the grid over $|V|$ at $|U|=1000$, and WP-22 for E4.2a, the grid over $|U|$ with the full $V$, ``--u-grid``), and the
accuracy and intervals of six CAP methods across the two sizes (E4.4, card WP-28: E4.4a ``--e44a``, E4.4b ``--e44b``).

Scope: the datasets ``DATASETS`` of ``main`` and their pretrained classifiers LR and RF, in the classic domain; the
methods ``BayesCAP-Hf-real`` (BayCAP with the Hf prior on $C$) and ``BayCAP-KDEy-mlp-real`` (BayesKDEy with an MLP
input and a fixed $\hat C$), both with the realised pushforward, built as in ``main``
(``experiments/methods.py``). Both target the realised table of the target bag and are scored against it
(``true_cts``). $h$ is never retrained and the split $L/V$ of the pretrain files is unchanged.

**E4.4 (WP-28).** The methods ``E44_METHODS``: the two above and ``SQS-H``, ``SQS-Hbs``, ``PQ`` (SQS-S) and ``SQS-B``,
built as in ``main`` except that the SQS methods draw bags of ``sample_size`` $=|U|$ instances from $V$
(``make_method``); on the $15$ datasets of ``main`` (``main_datasets``) with LR only. E4.4a (``--e44a``): the sizes
``U_SIZES`` with the full $V$. E4.4b (``--e44b``): $|U|=1000$ and the levels ``E44_LEVELS`` (full, $1/\sqrt2$, $1/2$,
$1/(2\sqrt2)$, $1/4$), with no threshold on the instances per class (``--no-class-threshold``, user's decision of
2026-11-19); the cells below ``MIN_CLASS_COUNT`` are flagged afterwards from ``v_class_counts``
(``tables/target_size_scores.py``). The scores (AE, Winkler score, coverage, amplitude) are computed by
``tables/target_size_scores.py``. Both target the realised table of each bag and are scored against it.

Purpose: the per-bag error split $e_1+e_2+e_3$ (``uncawe.error_split``) of the methods' mean tables, and its fixed and
bag parts within a cell (§T.4), computed afterwards by ``tables/target_size.py``. A cell is (dataset, classifier,
method, prevalence, $|U|$, $|V|$ level, $V$ seed), with ``n_bags`` ($R$) bags.

- **Grids.** E4.2b (WP-21): every $|V|$ level at $|U|=1000$ (the default, ``--sizes`` and ``--levels``). E4.2a
  (WP-22, ``--u-grid``): the full $V$ only, at the bag sizes ``U_SIZES``
  ($|U|\in\{100,300,1000,3000,10000\}$); the cells at $|U|=1000$ are those of E4.2b and, being stored, are skipped.
  Bags larger than the test pool repeat instances (they are drawn with replacement at every size).
- **Validation sets.** The full $V$ (level ``full``, seed $0$ only) and nested, seeded, stratified subsamples of
  $1/4$ and $1/16$ of $V$ (levels ``1/4`` and ``1/16``; E4.4b adds $1/\sqrt2$, $1/2$, $1/(2\sqrt2)$, ``ALL_LEVELS``),
  $S_V$ seeds each (``--n-v-seeds``). For seed $s$, the
  instances of each class of $V$ are permuted with a generator that depends only on the experiment seed, the dataset
  and $s$ (``subsample_index``), and the level of fraction $f$ keeps the first $\operatorname{round}(fn_j)$ of class
  $j$, so $V_{1/16}\subset V_{1/4}\subset V$ for each seed and every class keeps its share. A subsampled level runs on
  a dataset only if every class keeps at least ``MIN_CLASS_COUNT`` instances (user's decision of 2026-11-15, both
  levels), unless ``--no-class-threshold`` (E4.4b); every class keeps at least one instance; ``--counts`` reports the counts without running any method. The methods are refitted on each subsample; on
  the full $V$ they are fitted exactly as in ``main`` (same instances in the same order).
- **Prevalences.** Three fixed vectors per dataset (``target_prevalences``): uniform $\mathbf 1/K$; ``skewed``, a draw
  from $\operatorname{Dir}(\mathbf 1)$ with a generator that depends only on the experiment seed and the dataset;
  ``vertex``, $0.9$ on the rarest class of the full $V$ (the first one in case of ties), the rest uniform.
- **Bags.** For each prevalence and size, the class counts $n_j$ are $|U|\pi_j$ rounded by largest remainder
  (``class_counts``), the same for every bag; the $n_j$ instances of class $j$ are drawn with replacement from those of
  the test pool. Bag $b$ uses a generator seeded from the experiment seed, the dataset, $|U|$, the prevalence and $b$
  (``bag_seed``): the bags depend neither on the classifier, nor on the method, nor on the $|V|$ level or the $V$ seed
  (user's decision of 2026-11-15, a deviation from the card, which also seeded them by level and seed), so every cell
  of a given prevalence and $|U|$ sees the same bags. The posteriors of $h$ on a bag are the stored test posteriors of
  its instances.

Outputs: ``<OUT_DIR>/uncawe/target_size/u<|U|>/<level>/s<seed>/<domain>/<acc_name>/<dataset>/<classifier>/<method>.parquet``
(``level`` is one of ``LEVEL_DIRS``: ``v1``, ``v1_sqrt2``, ``v1_2``, ``v1_2sqrt2``, ``v1_4``, ``v1_16``), one file per
cell group (all prevalences and bags of a task), with the schema of ``main`` (the three metrics, and the NUTS
diagnostics and posterior summaries that ``main`` stores for the methods that sample with NUTS) and the columns ``bag_size``, ``v_level``, ``v_fraction``, ``v_seed``, ``v_size``, ``v_class_counts``,
``prevalence_kind``, ``requested_prev``, ``bag_index``, ``bag_seed``, ``config_hash`` and ``schema_version``. A file
that exists is skipped. ``--counts`` writes ``<OUT_DIR>/uncawe/tables/target_size/v_counts.csv`` (with ``--e44b``,
``v_counts_e44b.csv``, on the levels of E4.4b).

As in ``main``, the methods that sample with NumPyro (``BAYES_POOL_METHODS``) run in one pool with ``BAYES_N_JOBS``
workers and ``BAYES_ENV``, the others (``SQS-H``, ``SQS-Hbs``, ``PQ``) in a pool with ``N_JOBS`` of ``cap.env`` and
``THREAD_LIMITS``; the caches of JAX are cleared after each task.
"""

import os
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hashlib
import json
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception

import jax
import numpy as np
import pandas as pd
import quapy as qp
from cap import env as capenv
from cap.utils.commons import contingency_table, get_shift
from quapy.data import LabelledCollection

from uncawe import env
from uncawe.experiments.execution import ExperimentOutcome, run_tasks, shutdown_parallel_workers
from uncawe.experiments.methods import (
    BAYES_ENV,
    BAYES_N_JOBS,
    BAYES_POOL_METHODS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    SAMPLE_SIZE,
    THREAD_LIMITS,
    baycap_method,
    fixed_c_method,
    sqs_b,
    sqs_h,
    sqs_s,
)
from uncawe.experiments.metrics import gen_acc_measure, get_acc_names
from uncawe.experiments.records import EvaluationContext, build_result_frame
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_quantifier import QuantifierBayCAP
from uncawe.methods.sqs_b import SQSB
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.util import get_logger, get_plain_prev, local_path

EXPERIMENT = "target_size"
DOMAIN = "classic"
# E4.2 (user's decision of 2026-11-11)
DATASETS = ("mammographic", "german", "spambase", "wine-quality", "dry-bean", "letter")
CLASSIFIERS = ("LR", "RF")
METHODS = ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real")
# E4.4 (WP-28, user's decision of 2026-11-19): the two methods of E4.2 and the SQS methods of main; O-LEAP left out
E44_METHODS = METHODS + ("SQS-H", "SQS-Hbs", "PQ", "SQS-B")
E44_CLASSIFIERS = ("LR",)
# the SQS methods, whose bags of V get the size of the target bag in this study (make_method)
SQS_METHODS = ("SQS-H", "SQS-Hbs", "PQ", "SQS-B")
# |V| levels of E4.2: name -> fraction of V. Also part of ``config_hash``, so this dictionary stays as it is: the hash of
# the stored cells of E4.2 must not change (the fraction of each cell is stored in ``v_fraction``)
LEVELS = {"full": 1.0, "1/4": 1 / 4, "1/16": 1 / 16}
# |V| levels of E4.4b, a geometric grid from V to V/4 (nested with those of E4.2)
E44_LEVELS = {"full": 1.0, "1/sqrt2": 2**-0.5, "1/2": 1 / 2, "1/(2sqrt2)": 2**-1.5, "1/4": 1 / 4}
# every level of the study, by decreasing fraction, and the folder of each level
ALL_LEVELS = dict(sorted((LEVELS | E44_LEVELS).items(), key=lambda item: -item[1]))
LEVEL_DIRS = {
    "full": "v1",
    "1/sqrt2": "v1_sqrt2",
    "1/2": "v1_2",
    "1/(2sqrt2)": "v1_2sqrt2",
    "1/4": "v1_4",
    "1/16": "v1_16",
}
MIN_CLASS_COUNT = 10
PREVALENCE_KINDS = ("uniform", "skewed", "vertex")
VERTEX_MASS = 0.9
# R and S_V: proposals of the card, to be fixed by the pilot
N_BAGS = 30
N_V_SEEDS = 3
BAG_SIZES = (SAMPLE_SIZE,)
# the bag sizes of E4.2a (WP-22), run with the full V only (``--u-grid``)
U_SIZES = (100, 300, SAMPLE_SIZE, 3000, 10000)
# 1: first version (WP-21)
SCHEMA_VERSION = 1
# streams of the seed sequences of a dataset
_PREVALENCE_STREAM = 0
_SUBSAMPLE_STREAM = 1
_BAG_STREAM = 2

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = 0

EXP = ExperimentOutcome


# --------------------------------------------------------------------------------------------------- seeds


def _seed_words(dataset: str, stream: int, *words: int) -> list[int]:
    """The entropy of a seed sequence: the experiment seed, the ``zlib.crc32`` of the dataset name, the stream and
    ``words``; it depends on nothing else (not on the classifier, the method or the global random state)."""
    return [int(qp.environ["_R_SEED"]), zlib.crc32(dataset.encode()), stream, *map(int, words)]


def _rng(dataset: str, stream: int, *words: int) -> np.random.Generator:
    return np.random.default_rng(_seed_words(dataset, stream, *words))


def bag_seed(dataset: str, size: int, prevalence_index: int, bag_index: int) -> int:
    """The seed of bag ``bag_index`` of the prevalence ``PREVALENCE_KINDS[prevalence_index]`` and size ``size``: a
    32-bit integer drawn from a seed sequence of the experiment seed, the dataset, the size, the prevalence and the
    bag index only (shared by every classifier, method, $|V|$ level and $V$ seed)."""
    words = _seed_words(dataset, _BAG_STREAM, size, prevalence_index, bag_index)
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def config_hash(n_bags: int) -> str:
    """A short hash of the settings that define the cells of a file: the experiment seed, $R$, the prevalences, the
    subsampling rule and the NUTS settings; files with a different hash are not comparable cell by cell.

    The hash does not depend on the level of a cell nor on the threshold on the classes, which do not change the
    content of a cell: ``levels`` is the dictionary of E4.2 (``LEVELS``) and ``min_class_count`` its threshold, kept as
    they were so that the cells of E4.2 keep their hash (``25fc61de00cb`` with $R=30$) and are reused by E4.4."""
    config = {
        "seed": int(qp.environ["_R_SEED"]),
        "n_bags": int(n_bags),
        "prevalence_kinds": list(PREVALENCE_KINDS),
        "vertex_mass": VERTEX_MASS,
        "levels": LEVELS,
        "min_class_count": MIN_CLASS_COUNT,
        "num_warmup": BAYES_WARMUPS,
        "num_samples": BOOTSTRAP_ITERS,
        "schema_version": SCHEMA_VERSION,
    }
    return hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


# --------------------------------------------------------------------------------------------------- prevalences and bags


def target_prevalences(dataset: str, val_counts: np.ndarray) -> dict[str, np.ndarray]:
    """The three prevalence vectors of ``dataset`` (``PREVALENCE_KINDS``); ``val_counts`` are the class counts of the
    full $V$, which give the rarest class of ``vertex``."""
    val_counts = np.asarray(val_counts)
    k = len(val_counts)
    skewed = _rng(dataset, _PREVALENCE_STREAM).dirichlet(np.ones(k))
    vertex = np.full(k, (1 - VERTEX_MASS) / (k - 1))
    vertex[int(np.argmin(val_counts))] = VERTEX_MASS
    return {"uniform": np.full(k, 1 / k), "skewed": skewed / skewed.sum(), "vertex": vertex}


def class_counts(prevalence: np.ndarray, size: int) -> np.ndarray:
    """The class counts of a bag of ``size`` instances with ``prevalence``: $\\lfloor|U|\\pi_j\\rfloor$, plus one for
    the classes with the largest remainders (the first ones in case of ties) until the counts sum to ``size``."""
    prevalence = np.asarray(prevalence, dtype=float)
    if np.any(prevalence < 0) or not np.isclose(prevalence.sum(), 1):
        raise ValueError(f"Not a prevalence vector: {prevalence}")
    exact = prevalence * size
    counts = np.floor(exact).astype(int)
    missing = size - counts.sum()
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:missing]] += 1
    return counts


def sample_bag_index(pool_y: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    """Indices into the pool of a bag with ``counts[j]`` instances of class $j$, each drawn with replacement among the
    instances of class $j$ of the pool, with ``np.random.default_rng(seed)``; class by class, in class order."""
    rng = np.random.default_rng(seed)
    pool_y = np.asarray(pool_y)
    index = []
    for j, n_j in enumerate(counts):
        if n_j == 0:
            continue
        candidates = np.flatnonzero(pool_y == j)
        if candidates.size == 0:
            raise ValueError(f"Class {j} has no instance in the pool")
        index.append(candidates[rng.integers(0, candidates.size, size=n_j)])
    return np.concatenate(index)


@dataclass(frozen=True)
class Bag:
    prevalence_kind: str
    requested_prev: np.ndarray
    bag_index: int
    seed: int
    index: np.ndarray
    true_ct: np.ndarray


def make_bags(
    dataset: str, pool_y: np.ndarray, pool_pred: np.ndarray, val_counts: np.ndarray, size: int, n_bags: int
) -> list[Bag]:
    """The bags of every prevalence of ``dataset`` at ``size``: ``n_bags`` per prevalence, with their realised tables
    (true classes on rows) from the labels and the predictions of $h$ on the pool."""
    n_classes = len(val_counts)
    bags = []
    for p_index, (kind, prev) in enumerate(target_prevalences(dataset, val_counts).items()):
        counts = class_counts(prev, size)
        for b in range(n_bags):
            seed = bag_seed(dataset, size, p_index, b)
            index = sample_bag_index(pool_y, counts, seed)
            true_ct = contingency_table(pool_y[index], pool_pred[index], n_classes)
            bags.append(Bag(kind, prev, b, seed, index, true_ct))
    return bags


# --------------------------------------------------------------------------------------------------- subsamples of V


def subsample_counts(val_counts: np.ndarray, fraction: float) -> np.ndarray:
    """The class counts of the subsample of fraction ``fraction`` of a validation set with ``val_counts``:
    $\\operatorname{round}(fn_j)$ (halves rounded up), at least $1$; the counts themselves for ``fraction == 1``."""
    val_counts = np.asarray(val_counts, dtype=int)
    if fraction >= 1:
        return val_counts.copy()
    return np.maximum(np.floor(fraction * val_counts + 0.5).astype(int), 1)


def level_available(val_counts: np.ndarray, level: str, min_count: int | None = MIN_CLASS_COUNT) -> bool:
    """Whether the level runs on a validation set with ``val_counts``: always for ``full`` and with ``min_count=None``
    (no threshold, E4.4b); for a subsample, if every class keeps at least ``min_count`` instances."""
    if level == "full" or min_count is None:
        return True
    return int(subsample_counts(val_counts, ALL_LEVELS[level]).min()) >= min_count


def subsample_index(val_y: np.ndarray, fraction: float, dataset: str, v_seed: int) -> np.ndarray:
    """Indices into $V$ of the subsample of fraction ``fraction`` and seed ``v_seed``, in increasing order.

    The instances of each class are permuted with one generator of the experiment seed, the dataset and ``v_seed``
    (class by class, in class order), and the first ``subsample_counts`` of each class are kept, so subsamples of the
    same seed are nested and every class keeps its share. ``fraction == 1`` returns all the indices.
    """
    val_y = np.asarray(val_y)
    if fraction >= 1:
        return np.arange(val_y.size)
    n_classes = int(val_y.max()) + 1
    counts = subsample_counts(np.bincount(val_y, minlength=n_classes), fraction)
    rng = _rng(dataset, _SUBSAMPLE_STREAM, v_seed)
    kept = [rng.permutation(np.flatnonzero(val_y == j))[: counts[j]] for j in range(n_classes)]
    return np.sort(np.concatenate(kept))


# --------------------------------------------------------------------------------------------------- tasks


@dataclass(frozen=True)
class Task:
    info_path: str
    method_name: str
    size: int
    level: str
    v_seed: int
    n_bags: int
    # the threshold on the instances per class of a subsampled level; None: no threshold (E4.4b)
    min_class_count: int | None = MIN_CLASS_COUNT

    @property
    def folder(self) -> str:
        """The results folder of the task, relative to ``<OUT_DIR>/uncawe``."""
        return result_folder(self.size, self.level, self.v_seed)


def result_folder(size: int, level: str, v_seed: int) -> str:
    return os.path.join(EXPERIMENT, f"u{size}", LEVEL_DIRS[level], f"s{v_seed}")


def _result_path(p: PretrainInfo, task: Task, acc_name: str) -> str:
    return local_path(p.domain, p.d_info.name, p.h_info.full_name, task.method_name, acc_name, experiment=task.folder)


def _all_results_exist(p: PretrainInfo, task: Task) -> bool:
    return all(os.path.exists(_result_path(p, task, acc_name)) for acc_name in get_acc_names())


def make_method(method_name: str, size: int = SAMPLE_SIZE):
    """The method ``method_name`` of ``E44_METHODS``, built as in ``main`` (same settings and seed), except that the
    SQS methods draw bags of ``sample_size`` $=$ ``size`` instances from $V$, the size of the target bags of the task
    (E4.4; at ``size`` $=1000$ they are the methods of ``main``)."""
    _, acc = next(gen_acc_measure(True))
    if method_name == "BayesCAP-Hf-real":
        return baycap_method(acc, method_name)
    if method_name == "BayCAP-KDEy-mlp-real":
        return fixed_c_method(acc, method_name)
    if method_name == "SQS-H":
        return sqs_h(acc, bootstrap=False, sample_size=size)
    if method_name == "SQS-Hbs":
        return sqs_h(acc, bootstrap=True, sample_size=size)
    if method_name == "PQ":
        return sqs_s(acc, sample_size=size)
    if method_name == "SQS-B":
        return sqs_b(acc, sample_size=size)
    raise ValueError(f"Unknown method {method_name}; the study runs {list(E44_METHODS)}")


def _samples_with_nuts(method) -> bool:
    """The methods that store the NUTS diagnostics and posterior summaries of each bag, as in ``main``."""
    return isinstance(method, (BayCAP, QuantifierBayCAP, SQSB))


def protocol(task: Task) -> list[ExperimentOutcome]:
    """One task: the method of ``task`` fitted on the subsample of ``task`` and run on every bag of its size. The caches
    of JAX are cleared after the task, also when it fails (``experiments/main.py:exp_protocol``)."""
    try:
        return _run_task(task)
    finally:
        jax.clear_caches()


def _run_task(task: Task) -> list[ExperimentOutcome]:
    p = PretrainInfo.load(task.info_path, fast=True)
    if _all_results_exist(p, task):
        return [EXP.EXISTS(p, "all", task.method_name, experiment=task.folder)]

    dataset = p.load_dataset_bundle()
    val_posteriors_all, pool_posteriors = p.load_posteriors()
    if len(pool_posteriors) != len(dataset.U.y) or len(val_posteriors_all) != len(dataset.V.y):
        raise ValueError(f"{p.d_info.name}: the stored posteriors do not match V and U")
    n_classes = int(p.d_info.n_classes)
    val_y = np.asarray(dataset.V.y)
    val_counts = np.bincount(val_y, minlength=n_classes)
    if not level_available(val_counts, task.level, task.min_class_count):
        error = ValueError(f"{p.d_info.name}: level {task.level} keeps fewer than {task.min_class_count} of a class")
        return [EXP.ERROR(error, p, "ct", task.method_name, experiment=task.folder)]

    v_index = subsample_index(val_y, ALL_LEVELS[task.level], p.d_info.name, task.v_seed)
    if task.level == "full":
        validation = dataset.V
    else:
        validation = LabelledCollection(dataset.V.X[v_index], val_y[v_index], classes=dataset.V.classes_)
    validation_posteriors = val_posteriors_all[v_index]
    v_class_counts = np.bincount(val_y[v_index], minlength=n_classes)

    pool_y = np.asarray(dataset.U.y)
    pool_pred = np.argmax(pool_posteriors, axis=-1)
    bags = make_bags(p.d_info.name, pool_y, pool_pred, val_counts, task.size, task.n_bags)
    context = EvaluationContext(
        pretrain=p,
        shifts=get_shift(np.array([bag.true_ct.sum(axis=1) for bag in bags]), dataset.L_prevalence).tolist(),
        true_cts=[bag.true_ct for bag in bags],
        train_prev=get_plain_prev(dataset.L_prevalence),
        val_prev=get_plain_prev(validation.prevalence()),
    )

    try:
        method = make_method(task.method_name, task.size)
        started = time()
        method = method.fit(validation, validation_posteriors)
        t_train = time() - started

        # as in main: the time of the NUTS diagnostics and posterior summaries is left out of the test time
        started, t_diagnostics = time(), 0.0
        estim_ct_ranges, diagnostics = [], []
        for bag in bags:
            estim_ct_ranges.append(method.predict_ct_range(dataset.U.X[bag.index], pool_posteriors[bag.index]))
            if _samples_with_nuts(method):
                started_diagnostics = time()
                diagnostics.append(method.last_nuts_diagnostics() | method.last_posterior_summaries())
                t_diagnostics += time() - started_diagnostics
        t_test_ave = (time() - started - t_diagnostics) / context.size
        estim_cts = [np.nan if cts is np.nan else cts.mean(axis=0) for cts in estim_ct_ranges]
    except Exception as error:
        print_exception(error)
        return [EXP.ERROR(error, p, "ct", task.method_name, experiment=task.folder)]

    size = context.size
    extra_columns = {name: [d[name] for d in diagnostics] for name in diagnostics[0]} if diagnostics else {}
    extra_columns |= {
        "bag_size": [task.size] * size,
        "v_level": [task.level] * size,
        "v_fraction": [ALL_LEVELS[task.level]] * size,
        "v_seed": [task.v_seed] * size,
        "v_size": [int(v_index.size)] * size,
        "v_class_counts": [v_class_counts.tolist()] * size,
        "prevalence_kind": [bag.prevalence_kind for bag in bags],
        "requested_prev": [bag.requested_prev.tolist() for bag in bags],
        "bag_index": [bag.bag_index for bag in bags],
        "bag_seed": [bag.seed for bag in bags],
        "config_hash": [config_hash(task.n_bags)] * size,
        "schema_version": [SCHEMA_VERSION] * size,
    }

    results = []
    for acc_name, acc_fn in gen_acc_measure(n_classes > 2):
        if os.path.exists(_result_path(p, task, acc_name)):
            results.append(EXP.EXISTS(p, acc_name, task.method_name, experiment=task.folder))
            continue
        method.switch(acc_fn)
        intervals = [method.ci_from_cts(cts) for cts in estim_ct_ranges]
        frame = build_result_frame(
            context,
            method_name=task.method_name,
            acc_name=acc_name,
            acc_fn=acc_fn,
            estim_acc_ranges=[interval.samples for interval in intervals],
            estim_accs=[interval.point_estimate for interval in intervals],
            estim_cts=estim_cts,
            t_train=t_train,
            t_test_ave=t_test_ave,
            extra_columns=extra_columns,
        )
        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                task.method_name,
                df=frame,
                t_train=t_train,
                t_test_ave=t_test_ave,
                experiment=task.folder,
            )
        )
    return results


def _pretrains(datasets, classifiers) -> list[tuple[str, PretrainInfo]]:
    """The pretrain files of ``datasets`` and ``classifiers``, ordered by decreasing number of classes, then by dataset
    and classifier as listed (the larger $K$, the longer a task)."""
    found = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name in datasets and p.h_info.name in classifiers:
            found.append((path, p))
    missing = sorted(
        {(d, c) for d in datasets for c in classifiers} - {(p.d_info.name, p.h_info.name) for _, p in found}
    )
    if missing:
        raise FileNotFoundError(f"No pretrain file for {missing}")
    found.sort(
        key=lambda item: (
            -item[1].d_info.n_classes,
            datasets.index(item[1].d_info.name),
            classifiers.index(item[1].h_info.name),
        )
    )
    return found


def _val_counts(p: PretrainInfo) -> np.ndarray:
    y = np.asarray(p.load_dataset_bundle().V.y)
    return np.bincount(y, minlength=int(p.d_info.n_classes))


def tasks(
    datasets=DATASETS,
    classifiers=CLASSIFIERS,
    methods=METHODS,
    sizes=BAG_SIZES,
    levels=tuple(LEVELS),
    n_v_seeds: int = N_V_SEEDS,
    n_bags: int = N_BAGS,
    min_class_count: int | None = MIN_CLASS_COUNT,
    log: Logger | None = None,
) -> list[Task]:
    """The tasks still to run: every (pair, method, size, available level, $V$ seed), the full $V$ with seed $0$ only;
    a subsampled level is left out on the datasets where a class would keep fewer than ``min_class_count`` (never
    with ``min_class_count=None``)."""
    unknown = sorted(set(methods) - set(E44_METHODS)) + sorted(set(levels) - set(ALL_LEVELS))
    if unknown:
        raise ValueError(f"Unknown methods or levels: {unknown}")
    datasets, classifiers = list(datasets), list(classifiers)
    counts = {}
    out = []
    for path, p in _pretrains(datasets, classifiers):
        name = p.d_info.name
        if name not in counts:
            counts[name] = _val_counts(p)
        for level in levels:
            if not level_available(counts[name], level, min_class_count):
                if log is not None:
                    log.info(f"[{p.h_info.name}@{name}] level {level} skipped: a class keeps < {min_class_count}")
                continue
            seeds = [0] if level == "full" else range(n_v_seeds)
            for size in sizes:
                for v_seed in seeds:
                    for method_name in methods:
                        task = Task(path, method_name, int(size), level, int(v_seed), int(n_bags), min_class_count)
                        if _all_results_exist(p, task):
                            if log is not None:
                                log.info(f"[{p.h_info.name}@{name}] {method_name} {task.folder} exists, skipping")
                            continue
                        out.append(task)
    return out


def split_pools(todo: list[Task]) -> tuple[list[Task], list[Task]]:
    """The tasks of the methods that sample with NumPyro (``BAYES_POOL_METHODS``) and those of the other methods."""
    bayes = [task for task in todo if task.method_name in BAYES_POOL_METHODS]
    regular = [task for task in todo if task.method_name not in BAYES_POOL_METHODS]
    return bayes, regular


def run(log: Logger, **kwargs) -> None:
    todo = tasks(log=log, **kwargs)
    bayes, regular = split_pools(todo)
    log.info(f"{len(todo)} tasks to run ({len(bayes)} NumPyro, {len(regular)} other)")
    run_tasks(log, regular, worker=protocol, n_jobs=capenv["N_JOBS"], experiment=EXPERIMENT, env=THREAD_LIMITS)
    run_tasks(log, bayes, worker=protocol, n_jobs=BAYES_N_JOBS, experiment=EXPERIMENT, env=BAYES_ENV)


def count_table(
    datasets=DATASETS,
    n_v_seeds: int = N_V_SEEDS,
    levels=tuple(LEVELS),
    classifiers=CLASSIFIERS,
    min_class_count: int | None = MIN_CLASS_COUNT,
) -> pd.DataFrame:
    """Per dataset, level and $V$ seed: the size of the subsample, its smallest class count, whether the level runs
    (``level_available`` with ``min_class_count``) and whether a class keeps fewer than ``MIN_CLASS_COUNT`` instances
    (``below_threshold``, the flag of E4.4b), with the counts per class; the subsample itself is drawn, so the counts
    are checked."""
    rows = []
    seen = set()
    for _, p in _pretrains(list(datasets), list(classifiers)):
        name = p.d_info.name
        if name in seen:
            continue
        seen.add(name)
        val_y = np.asarray(p.load_dataset_bundle().V.y)
        val_counts = np.bincount(val_y, minlength=int(p.d_info.n_classes))
        for level in levels:
            fraction = ALL_LEVELS[level]
            for v_seed in [0] if level == "full" else range(n_v_seeds):
                index = subsample_index(val_y, fraction, name, v_seed)
                counts = np.bincount(val_y[index], minlength=len(val_counts))
                rows.append(
                    {
                        "dataset": name,
                        "n_classes": len(val_counts),
                        "level": level,
                        "v_seed": v_seed,
                        "v_size": int(index.size),
                        "min_class_count": int(counts.min()),
                        "runs": level_available(val_counts, level, min_class_count),
                        "below_threshold": bool(counts.min() < MIN_CLASS_COUNT),
                        "class_counts": counts.tolist(),
                    }
                )
    return pd.DataFrame(rows)


def write_counts(
    log: Logger, datasets=DATASETS, n_v_seeds: int = N_V_SEEDS, suffix: str = "", **kwargs
) -> pd.DataFrame:
    """Write ``count_table`` (``kwargs``: its levels, classifiers and threshold) to ``v_counts<suffix>.csv``."""
    table = count_table(datasets, n_v_seeds, **kwargs)
    out_dir = os.path.join(env.root_dir, "tables", EXPERIMENT)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"v_counts{suffix}.csv")
    table.to_csv(path, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 80):
        print(table.to_string(index=False))
    log.info(f"subsample counts written to {path}")
    return table


def main_datasets() -> list[str]:
    """The $15$ datasets of ``main`` (``experiments/datasets.py:get_filtered_datasets``), those of E4.4."""
    from uncawe.experiments.datasets import get_filtered_datasets

    return list(get_filtered_datasets())


# the grids of E4.4 (WP-28): the defaults that ``--e44a`` and ``--e44b`` give to the options not passed
E44_PRESETS = {
    "e44a": {"sizes": list(U_SIZES), "levels": ["full"], "class_threshold": True},
    "e44b": {"sizes": [SAMPLE_SIZE], "levels": list(E44_LEVELS), "class_threshold": False},
}


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(
        description="The study target_size: the error split across |V| and |U| (E4.2, WP-21, WP-22) and the scores "
        "of six methods across |U| and |V| (E4.4, WP-28)."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Run the methods on the grid")
    mode.add_argument("--counts", action="store_true", help="Write the class counts of the subsamples of V only")
    grid = parser.add_mutually_exclusive_group()
    grid.add_argument(
        "--u-grid",
        action="store_true",
        help=f"E4.2a (WP-22): the sizes {list(U_SIZES)} with the full V only; excludes --sizes and --levels",
    )
    grid.add_argument(
        "--e44a",
        action="store_true",
        help=f"E4.4a (WP-28): the methods {list(E44_METHODS)}, the datasets of main, LR, the sizes {list(U_SIZES)} "
        "with the full V; --datasets, --classifiers and --methods restrict it (e.g. the pilot)",
    )
    grid.add_argument(
        "--e44b",
        action="store_true",
        help=f"E4.4b (WP-28): as --e44a, at |U|={SAMPLE_SIZE} with the levels {list(E44_LEVELS)} and no threshold on "
        "the classes",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help=f"Datasets (default: {list(DATASETS)})")
    parser.add_argument("--classifiers", nargs="+", default=None, choices=CLASSIFIERS)
    parser.add_argument("--methods", nargs="+", default=None, choices=E44_METHODS)
    parser.add_argument("--sizes", nargs="+", type=int, default=None, help="Bag sizes |U| (default: 1000)")
    parser.add_argument(
        "--levels", nargs="+", default=None, choices=tuple(ALL_LEVELS), help="|V| levels (default: those of E4.2)"
    )
    parser.add_argument(
        "--no-class-threshold",
        action="store_true",
        help=f"Run the subsampled levels also where a class keeps fewer than {MIN_CLASS_COUNT} instances (E4.4b)",
    )
    parser.add_argument("--n-v-seeds", type=int, default=N_V_SEEDS, help=f"S_V (default: {N_V_SEEDS})")
    parser.add_argument("--n-bags", type=int, default=N_BAGS, help=f"R, bags per cell (default: {N_BAGS})")
    args = parser.parse_args(argv)
    args.preset = "e44a" if args.e44a else "e44b" if args.e44b else None
    if (args.u_grid or args.preset is not None) and (args.sizes is not None or args.levels is not None):
        parser.error("--u-grid, --e44a and --e44b set the sizes and the levels; do not pass --sizes or --levels")
    if args.u_grid:
        args.sizes, args.levels = list(U_SIZES), ["full"]
    if args.preset is not None:
        preset = E44_PRESETS[args.preset]
        args.sizes, args.levels = list(preset["sizes"]), list(preset["levels"])
        args.no_class_threshold = args.no_class_threshold or not preset["class_threshold"]
        args.datasets = main_datasets() if args.datasets is None else args.datasets
        args.classifiers = list(E44_CLASSIFIERS) if args.classifiers is None else args.classifiers
        args.methods = list(E44_METHODS) if args.methods is None else args.methods
    args.datasets = list(DATASETS) if args.datasets is None else args.datasets
    args.classifiers = list(CLASSIFIERS) if args.classifiers is None else args.classifiers
    args.methods = list(METHODS) if args.methods is None else args.methods
    args.sizes = list(BAG_SIZES) if args.sizes is None else args.sizes
    args.levels = list(LEVELS) if args.levels is None else args.levels
    args.min_class_count = None if args.no_class_threshold else MIN_CLASS_COUNT
    if any(size < 1 for size in args.sizes):
        parser.error(f"Bag sizes must be positive: {args.sizes}")
    return args


def main() -> None:
    pargs = parse_args()
    log = get_logger(id=EXPERIMENT)
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        if pargs.counts:
            write_counts(
                log,
                pargs.datasets,
                pargs.n_v_seeds,
                suffix="" if pargs.preset is None else f"_{pargs.preset}",
                levels=pargs.levels,
                classifiers=pargs.classifiers,
                min_class_count=pargs.min_class_count,
            )
        else:
            run(
                log,
                datasets=pargs.datasets,
                classifiers=pargs.classifiers,
                methods=pargs.methods,
                sizes=pargs.sizes,
                levels=pargs.levels,
                n_v_seeds=pargs.n_v_seeds,
                n_bags=pargs.n_bags,
                min_class_count=pargs.min_class_count,
            )
    except Exception as error:
        log.error(error)
        print_exception(error)
    finally:
        shutdown_parallel_workers(log)
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
