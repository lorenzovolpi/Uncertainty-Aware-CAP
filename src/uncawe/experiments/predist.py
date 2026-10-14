"""Run the implemented diagnostic experiment for PrediQuant distance choices.

The experiment covers selected three-class UCI datasets and one requested
pre-trained classifier (LR by default).  It places three to five test
prevalences at informative locations of the ternary simplex, fits PrediQuant
with KDEy on the validation data, and evaluates each configured distance on the
same deterministic test bags.

Its purpose is to study how the distance used by PrediQuant changes the chosen
validation-prevalence neighbourhood and, consequently, its point estimate and
confidence interval.  Each record therefore retains requested, actual, and
predicted test prevalences; selected predicted and true validation prevalences;
distance summaries; accuracy error; interval coverage, amplitude, and Winkler
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

from uncawe.experiments.methods import CONFIDENCE_LEVEL, kdey
from uncawe.experiments.metrics import acc_from_ct
from uncawe.experiments.predist_config import (
    DEFAULT_ACC_NAME,
    DEFAULT_CLASSIFIER,
    DEFAULT_DOMAIN,
    DEFAULT_N_JOBS,
    DEFAULT_NUM_TEST,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    EXPERIMENT,
    LR_ALIASES,
    METHOD_NAME,
    PREDIQUANT_DISTANCE_FNS,
    RESULT_SCHEMA_VERSION,
    THREE_CLASS_DATASETS,
)
from uncawe.experiments.predist_io import (
    atomic_save_parquet as _atomic_save_parquet,
    completed_keys as _completed_keys,
    load_existing as _load_existing,
    result_path as _result_path,
)
from uncawe.experiments.predist_sampling import (
    TestSample,
    load_filtered_pretrains as _load_filtered_pretrains,
    make_test_samples as _make_test_samples,
    simplex_points as _simplex_points,
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


def _prediquant_predict_with_val_prevs(method: PrediQuant, sample: TestSample):
    test_prior = method._predict_test_prior(sample.collection.X)
    closest_idx = method._predict_closest_idx(test_prior)[: method.num_samples]
    val_prevs = np.asarray(method.sigma_pred_prevs, dtype=float)[closest_idx]
    val_true_prevs = np.asarray(method.sigma_true_prevs, dtype=float)[closest_idx]
    accs = np.array([method.acc(ct) for ct in method.sigma_ct])[closest_idx]
    return accs, test_prior, val_prevs, val_true_prevs


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
        "schema_version": RESULT_SCHEMA_VERSION,
        "distance": distance,
        "sample_id": sample.sample_id,
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
    method = PrediQuant(
        acc_fn,
        kdey(),
        num_samples=num_samples,
        sample_size=sample_size,
        distance=distances[0],
        random_state=seed,
    )
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
    log.info(
        f"starting {EXPERIMENT}: domain={args.domain}, classifier={args.classifier}, acc={args.acc_name}, "
        f"datasets={datasets}, distances={distances}, num_test={args.num_test}, sample_size={args.sample_size}, "
        f"num_samples={args.num_samples}, n_jobs={args.n_jobs}, plot={not args.no_plot}, overwrite={args.overwrite}"
    )
    pretrains = _load_filtered_pretrains(args.domain, args.classifier, datasets)
    if not pretrains:
        raise RuntimeError(
            f"No pretrain artifacts found for domain={args.domain!r}, classifier={args.classifier!r}, "
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
    log.info(f"finished {EXPERIMENT}: processed {len(experiment_args)} pretrain artifact(s)")


def parse_args():
    parser = ArgumentParser(description="PrediQuant simplex-distance experiment on 3-class UCI datasets.")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--classifier", default=DEFAULT_CLASSIFIER)
    parser.add_argument("--acc-name", default=DEFAULT_ACC_NAME)
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--distances", nargs="*", default=None)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())
