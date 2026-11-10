"""Run the implemented diagnostic experiment for PrediQuant distance choices (E6.1 of
``docs/ext6_sqs_s_design.md``, card WP-16).

The experiment covers the three-class UCI datasets of ``THREE_CLASS_DATASETS``
and the requested pre-trained classifiers (the five of ``main`` by default), with
all (dataset, classifier) pairs in one parallel pool.  It places three to five
test prevalences at informative locations of the ternary simplex (the centre,
the three vertices at $1-0.01$, the random point farthest from them), fits
PrediQuant once per pair on the validation data with PACC(MLP) and the other
settings of ``PQ`` in ``main``, and evaluates each configured distance on the
same deterministic test bags.  Every score is computed against the realised
table of the test bag.

Its purpose is to study how the distance used by PrediQuant changes the chosen
validation-prevalence neighbourhood and, consequently, its point estimate and
confidence interval.  Each record therefore retains requested, actual, and
predicted test prevalences; selected predicted and true validation prevalences;
distance summaries, among them the mean $L_1$ distance between the true
prevalences of the selected bags and the true prevalence of the test bag; accuracy error; interval coverage, amplitude, and Winkler
score; and runtime.  This is a diagnostic of one method's selection mechanism,
not a general comparison among CAP methods.
"""

from argparse import ArgumentParser
from time import time
from traceback import print_exception
from typing import Any

import numpy as np
import pandas as pd
import quapy as qp
from cap.models.confidence import PrediQuant
from cap.utils.commons import contingency_table, parallel

from uncawe.experiments.execution import shutdown_parallel_workers, worker_environment
from uncawe.experiments.methods import CONFIDENCE_LEVEL, THREAD_LIMITS, pacc
from uncawe.experiments.metrics import acc_from_ct
from uncawe.experiments.predist_config import (
    DEFAULT_ACC_NAME,
    DEFAULT_CLASSIFIERS,
    DEFAULT_DOMAIN,
    DEFAULT_N_JOBS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_NUM_TEST,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    EXPERIMENT,
    METHOD_NAME,
    PREDIQUANT_ALPHA,
    PREDIQUANT_DISTANCE_FNS,
    QUANTIFIER_NAME,
    RESULT_SCHEMA_VERSION,
    SAMPLE_REGIONS,
    THREE_CLASS_DATASETS,
)
from uncawe.experiments.predist_io import (
    atomic_save_parquet as _atomic_save_parquet,
)
from uncawe.experiments.predist_io import (
    completed_keys as _completed_keys,
)
from uncawe.experiments.predist_io import (
    load_existing as _load_existing,
)
from uncawe.experiments.predist_io import (
    result_path as _result_path,
)
from uncawe.experiments.predist_sampling import (
    TestSample,
)
from uncawe.experiments.predist_sampling import (
    load_filtered_pretrains as _load_filtered_pretrains,
)
from uncawe.experiments.predist_sampling import (
    make_test_samples as _make_test_samples,
)
from uncawe.experiments.predist_sampling import (
    parse_classifiers as _parse_classifiers,
)
from uncawe.experiments.predist_sampling import (
    simplex_points as _simplex_points,
)
from uncawe.experiments.predist_sampling import (
    three_class_dataset_names as _three_class_dataset_names,
)
from uncawe.pretrain.data import PretrainInfo
from uncawe.util import get_logger


def _plain_prev(prev: np.ndarray) -> list[float]:
    return np.around(np.asarray(prev, dtype=float), decimals=6).tolist()


def _parse_distances(raw_distances: list[str] | None) -> list[str]:
    if raw_distances:
        distances = [part.strip() for item in raw_distances for part in item.split(",") if part.strip()]
    else:
        distances = list(PrediQuant.DISTANCES)
    invalid = sorted(set(distances).difference(PrediQuant.DISTANCES))
    if invalid:
        raise ValueError(f"Invalid PrediQuant distances {invalid}; valid values are {list(PrediQuant.DISTANCES)}")
    return distances


def _set_prediquant_distance(method: PrediQuant, distance: str) -> None:
    method.error = PREDIQUANT_DISTANCE_FNS[distance]


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _prediction_summary(prediction: Any) -> tuple[float | None, float | None, float | None]:
    if hasattr(prediction, "point_estimate"):
        return (
            float(np.asarray(prediction.point_estimate).mean()),
            _maybe_float(getattr(prediction, "I_low", None)),
            _maybe_float(getattr(prediction, "I_high", None)),
        )
    array = np.asarray(prediction)
    if array.size == 0:
        return None, None, None
    if array.ndim == 0:
        return float(array), None, None
    alpha = 1 - CONFIDENCE_LEVEL
    low, high = np.nanpercentile(array, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.nanmean(array)), float(low), float(high)


def _interval_metrics(
    true_value: float,
    low: float | None,
    high: float | None,
) -> tuple[float | None, float | None, float | None]:
    if low is None or high is None:
        return None, None, None
    coverage = float(low <= true_value <= high)
    amplitude = float(high - low)
    alpha = 1 - CONFIDENCE_LEVEL
    winkler = amplitude
    if true_value < low:
        winkler += (2 / alpha) * (low - true_value)
    elif true_value > high:
        winkler += (2 / alpha) * (true_value - high)
    return coverage, amplitude, float(winkler)


def build_prediquant(acc_fn, num_samples: int, sample_size: int, distance: str, seed: int) -> PrediQuant:
    """PrediQuant with the settings of ``PQ`` in ``main`` (``experiments/methods.py:gen_methods``): PACC(MLP) with the
    MLP seeded by ``qp.environ["_R_SEED"]``, $\\alpha=0.1$ (``num_samples / alpha`` candidate bags), estimated
    prevalences of the candidates; the split of $V$ and the candidate bags are seeded by ``seed``."""
    return PrediQuant(
        acc_fn,
        pacc(),
        alpha=PREDIQUANT_ALPHA,
        num_samples=num_samples,
        sample_size=sample_size,
        distance=distance,
        predict_train_prev=True,
        random_state=seed,
    )


def _prediquant_predict_with_val_prevs(method: PrediQuant, sample: TestSample):
    accs = method.predict_range(sample.collection.X, sample.posteriors)
    closest_idx = method.last_selected_bags
    val_prevs = np.asarray(method.sigma_pred_prevs, dtype=float)[closest_idx]
    val_true_prevs = np.asarray(method.sigma_true_prevs, dtype=float)[closest_idx]
    return accs, method.last_test_prev, val_prevs, val_true_prevs


def true_space_l1(val_true_prev: np.ndarray, actual_prev: np.ndarray) -> float:
    """Mean $L_1$ distance between the true prevalences of the selected bags and the true prevalence of the test bag
    (how well the selection works in true-prevalence space, §T.2 of ``docs/ext6_sqs_s_design.md``)."""
    val_true_prev = np.asarray(val_true_prev, dtype=float)
    return float(np.abs(val_true_prev - np.asarray(actual_prev, dtype=float).reshape(1, -1)).sum(axis=1).mean())


def _record(
    p: PretrainInfo,
    train_prev: np.ndarray,
    val_global_prev: np.ndarray,
    acc_name: str,
    distance: str,
    sample: TestSample,
    predicted_test_prev: np.ndarray,
    val_prev: np.ndarray,
    val_true_prev: np.ndarray,
    prediction: Any,
    elapsed: float,
) -> dict[str, Any]:
    distances = np.linalg.norm(val_prev - sample.actual_prev.reshape(1, 3), axis=1)
    y_hat = np.argmax(sample.posteriors, axis=-1)
    ct = contingency_table(sample.collection.y, y_hat, sample.collection.n_classes)
    true_acc = acc_from_ct(acc_name, ct, type="evaluation")
    pred_acc, interval_low, interval_high = _prediction_summary(prediction)
    ci_coverage, ci_amplitude, ci_winkler = _interval_metrics(true_acc, interval_low, interval_high)
    return {
        "dataset": p.d_info.name,
        "collection": p.d_info.collection,
        "classifier": p.h_info.name,
        "classifier_full_name": p.h_info.full_name,
        "acc_name": acc_name,
        "method": METHOD_NAME,
        "quantifier": QUANTIFIER_NAME,
        "schema_version": RESULT_SCHEMA_VERSION,
        "distance": distance,
        "sample_id": sample.sample_id,
        "region": SAMPLE_REGIONS[sample.sample_id],
        "requested_test_prev": _plain_prev(sample.requested_prev),
        "actual_test_prev": _plain_prev(sample.actual_prev),
        "predicted_test_prev": _plain_prev(predicted_test_prev),
        "train_prev": _plain_prev(train_prev),
        "val_global_prev": _plain_prev(val_global_prev),
        "selected_val_prevs": [_plain_prev(prev) for prev in val_prev],
        "selected_val_true_prevs": [_plain_prev(prev) for prev in val_true_prev],
        "dist_min": float(np.min(distances)),
        "dist_mean": float(np.mean(distances)),
        "dist_median": float(np.median(distances)),
        "dist_max": float(np.max(distances)),
        "dist_p95": float(np.percentile(distances, 95)),
        "true_dist_l1_mean": true_space_l1(val_true_prev, sample.actual_prev),
        "true_acc": float(true_acc),
        "pred_acc": pred_acc,
        "acc_abs_error": None if pred_acc is None else float(abs(true_acc - pred_acc)),
        "interval_low": interval_low,
        "interval_high": interval_high,
        "ci_coverage": ci_coverage,
        "ci_amplitude": ci_amplitude,
        "ci_winkler": ci_winkler,
        "runtime": elapsed,
        "confidence_level": CONFIDENCE_LEVEL,
    }


def _generate_plots(df: pd.DataFrame, p: PretrainInfo, acc_name: str) -> None:
    from uncawe.plots.predist import plot_grouped_sample

    for sample_id in sorted(df["sample_id"].unique()):
        plot_grouped_sample(df, p.d_info.name, p.h_info.full_name, acc_name, int(sample_id))


def _run_one_pretrain(
    p: PretrainInfo,
    distances: list[str],
    num_test: int,
    sample_size: int,
    num_samples: int,
    seed: int,
    acc_name: str,
    plot: bool,
    overwrite: bool,
    log,
) -> None:
    result_path = _result_path(p.domain, p.d_info.name, p.h_info.full_name, acc_name)
    existing = pd.DataFrame() if overwrite else _load_existing(result_path)
    done = set() if overwrite else _completed_keys(existing)

    dataset = p.load_dataset_bundle()
    classifier = p.load_pretrained_classifier(dataset)
    val_posteriors = classifier.predict_proba(dataset.V.X)
    acc_fn = lambda ct: acc_from_ct(acc_name, ct, type="selection")
    samples = _make_test_samples(
        dataset,
        classifier,
        _simplex_points(np.random.RandomState(seed), num_test),
        sample_size,
        seed,
    )
    method = build_prediquant(acc_fn, num_samples, sample_size, distances[0], seed)
    method.fit(dataset.V, val_posteriors)

    rows = []
    for distance in distances:
        _set_prediquant_distance(method, distance)
        distance_rows = []
        for sample in samples:
            if (distance, sample.sample_id) in done:
                log.info(f"[{p.h_info.name}@{p.d_info.name}] {distance} sample {sample.sample_id} exists, skipping")
                continue
            started = time()
            prediction, predicted_test_prev, val_prev, val_true_prev = _prediquant_predict_with_val_prevs(
                method, sample
            )
            distance_rows.append(
                _record(
                    p,
                    dataset.L_prevalence,
                    dataset.V.prevalence(),
                    acc_name,
                    distance,
                    sample,
                    predicted_test_prev,
                    val_prev,
                    val_true_prev,
                    prediction,
                    time() - started,
                )
            )
        if distance_rows:
            rows.extend(distance_rows)
            _atomic_save_parquet(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True), result_path)
            log.info(f"[{p.h_info.name}@{p.d_info.name}] {distance} saved to {result_path}")

    if plot:
        final_df = _load_existing(result_path)
        if not final_df.empty:
            _generate_plots(final_df, p, acc_name)


def _run_one_pretrain_args(args):
    p, distances, num_test, sample_size, num_samples, seed, acc_name, plot, overwrite = args
    qp.environ["SAMPLE_SIZE"] = sample_size
    qp.environ["_R_SEED"] = seed
    log = get_logger(EXPERIMENT)
    try:
        _run_one_pretrain(
            p,
            distances,
            num_test,
            sample_size,
            num_samples,
            seed,
            acc_name,
            plot,
            overwrite,
            log,
        )
        return True, p.d_info.name, p.h_info.name, None
    except Exception as error:
        log.warning(f"[{p.h_info.name}@{p.d_info.name}] failed: {error}")
        print_exception(error)
        return False, p.d_info.name, p.h_info.name, repr(error)


def run(args) -> None:
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be >= 1")
    qp.environ["SAMPLE_SIZE"] = args.sample_size
    qp.environ["_R_SEED"] = args.seed

    log = get_logger(EXPERIMENT)
    distances = _parse_distances(args.distances)
    datasets = _three_class_dataset_names(args.datasets)
    classifiers = _parse_classifiers(args.classifiers)
    log.info(
        f"starting {EXPERIMENT}: domain={args.domain}, classifiers={classifiers}, quantifier={QUANTIFIER_NAME}, "
        f"acc={args.acc_name}, "
        f"datasets={datasets}, distances={distances}, num_test={args.num_test}, sample_size={args.sample_size}, "
        f"num_samples={args.num_samples}, n_jobs={args.n_jobs}, plot={not args.no_plot}, overwrite={args.overwrite}"
    )
    pretrains = _load_filtered_pretrains(args.domain, classifiers, datasets)
    missing = sorted(
        {(d, h) for d in datasets for h in classifiers}.difference((p.d_info.name, p.h_info.name) for p in pretrains)
    )
    if missing:
        log.warning(f"no pretrain artifact for the pairs {missing}")
    if not pretrains:
        raise RuntimeError(
            f"No pretrain artifacts found for domain={args.domain!r}, classifiers={classifiers}, "
            f"datasets={datasets}. Run the pretrain pipeline first or check the classifier name."
        )

    experiment_args = [
        (
            p,
            distances,
            args.num_test,
            args.sample_size,
            args.num_samples,
            args.seed,
            args.acc_name,
            not args.no_plot,
            args.overwrite,
        )
        for p in pretrains
    ]
    if args.n_jobs == 1:
        for experiment_args_item in experiment_args:
            _run_one_pretrain_args(experiment_args_item)
    else:
        # as experiments/execution.py:run_tasks: fresh workers with one thread per numerical library
        shutdown_parallel_workers(log)
        with worker_environment(THREAD_LIMITS):
            results = parallel(
                func=_run_one_pretrain_args,
                args_list=experiment_args,
                n_jobs=args.n_jobs,
                return_as="generator_unordered",
                max_nbytes=None,
            )
            for ok, dataset, classifier, error in results:
                if ok:
                    log.info(f"[{classifier}@{dataset}] completed")
                else:
                    log.warning(f"[{classifier}@{dataset}] failed: {error}")
        shutdown_parallel_workers(log)
    log.info(f"finished {EXPERIMENT}: processed {len(experiment_args)} pretrain artifact(s)")


def parse_args():
    parser = ArgumentParser(description="PrediQuant simplex-distance experiment on 3-class UCI datasets.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument(
        "--classifiers",
        nargs="*",
        default=None,
        help=f"Classifiers (names or comma-separated lists; default: {', '.join(DEFAULT_CLASSIFIERS)})",
    )
    parser.add_argument("--acc-name", default=DEFAULT_ACC_NAME)
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Selected bags m (num_samples / {PREDIQUANT_ALPHA} candidates)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--distances", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())
