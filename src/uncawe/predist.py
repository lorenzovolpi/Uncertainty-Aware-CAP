import os
import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass
from time import time
from traceback import print_exception
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp
from cap.models.confidence import PrediQuant
from cap.utils.commons import contingency_table, parallel
from matplotlib.patches import Polygon
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC

import uncawe.env as env
from uncawe.config import CONFIDENCE_LEVEL, acc_from_ct, kdey
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.pretrain.dataset import sort_datasets_by_size
from uncawe.util import get_logger

EXPERIMENT = "predist"
METHOD_NAME = "PrediQuant"
DEFAULT_DOMAIN = "classic"
DEFAULT_ACC_NAME = "vanilla_accuracy"
DEFAULT_CLASSIFIER = "LR"
DEFAULT_NUM_TEST = 5
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_SEED = 0
DEFAULT_N_JOBS = 1
RESULT_SCHEMA_VERSION = 2
THREE_CLASS_DATASETS = [
    "connect-4",
    "page_block",
    "waveform-v1",
    "academic-success",
    "molecular",
    "cmc",
    "phishing",
    "mhr",
]
LR_ALIASES = {"lr", "logisticregression", "logistic_regression", "logistic regression"}


@dataclass(frozen=True)
class TestSample:
    sample_id: int
    requested_prev: np.ndarray
    actual_prev: np.ndarray
    collection: LabelledCollection
    posteriors: np.ndarray


def _plain_prev(prev: np.ndarray) -> list[float]:
    return np.around(np.asarray(prev, dtype=float), decimals=6).tolist()


def _normalise_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "").replace(" ", "")


def _is_lr_classifier(p: PretrainInfo, requested: str) -> bool:
    requested_norm = _normalise_name(requested)
    accepted = {_normalise_name(a) for a in LR_ALIASES | {requested}}
    h = p.h_info
    candidates = [h.name, h.full_name, h.class_name]
    return requested_norm in accepted and any(_normalise_name(str(c)) in accepted for c in candidates)


def _predist_base_dir() -> str:
    return os.path.join(env.root_dir, EXPERIMENT)


def _result_path(domain: str, dataset: str, classifier: str, acc_name: str) -> str:
    parent = os.path.join(_predist_base_dir(), domain, acc_name, dataset, classifier)
    os.makedirs(parent, exist_ok=True)
    return os.path.join(parent, f"{METHOD_NAME}.parquet")


def _safe_name(name: str) -> str:
    return str(name).replace(os.sep, "__").replace(" ", "_")


def _plot_stem(dataset: str, classifier: str, acc_name: str, sample_id: int) -> str:
    parent = os.path.join(env.cap.env["OUT_DIR"], "plots", EXPERIMENT, dataset)
    os.makedirs(parent, exist_ok=True)
    filename = f"{_safe_name(acc_name)}_{_safe_name(classifier)}_{METHOD_NAME}_{sample_id:02d}"
    return os.path.join(parent, filename)


def _atomic_save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".parquet", dir=os.path.dirname(path))
    os.close(fd)
    try:
        df.to_parquet(tmp_path, compression="zstd", index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_existing(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "schema_version" not in df.columns:
        return pd.DataFrame()
    if not (df["schema_version"] == RESULT_SCHEMA_VERSION).all():
        return pd.DataFrame()
    return df


def _completed_keys(df: pd.DataFrame) -> set[tuple[str, int]]:
    if df.empty:
        return set()
    return set(zip(df["distance"].astype(str), df["sample_id"].astype(int)))


def _three_class_dataset_names(requested: list[str] | None = None) -> list[str]:
    names = THREE_CLASS_DATASETS if requested is None else requested
    return sort_datasets_by_size("uci_multiclass", names)


def _load_filtered_pretrains(domain: str, classifier: str, datasets: list[str]) -> list[PretrainInfo]:
    wanted = set(datasets)
    out = []
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.collection != "uci_multiclass":
            continue
        if p.d_info.n_classes != 3:
            continue
        if p.d_info.name not in wanted:
            continue
        if not _is_lr_classifier(p, classifier):
            continue
        out.append(p)

    order = {name: i for i, name in enumerate(datasets)}
    out.sort(key=lambda p: (order[p.d_info.name], p.h_info.full_name))
    return out


def _simplex_points(rng: np.random.RandomState, n_points: int, eps: float = 0.01) -> list[np.ndarray]:
    if not 3 <= n_points <= 5:
        raise ValueError("--num-test must be between 3 and 5 for the ternary predist experiment")

    candidates = [np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)]
    for cls in range(3):
        prev = np.full(3, eps / 2, dtype=float)
        prev[cls] = 1 - eps
        candidates.append(prev)

    if n_points <= len(candidates):
        return candidates[:n_points]

    random_pool = rng.dirichlet(np.ones(3), size=512)
    chosen = candidates.copy()
    while len(chosen) < n_points:
        distances = np.stack([np.linalg.norm(random_pool - prev, axis=1) for prev in chosen], axis=1)
        idx = int(np.argmax(distances.min(axis=1)))
        chosen.append(random_pool[idx])
        random_pool = np.delete(random_pool, idx, axis=0)
    return chosen


def _make_test_samples(D, h, requested_prevs: list[np.ndarray], sample_size: int, seed: int) -> list[TestSample]:
    samples = []
    for sample_id, requested_prev in enumerate(requested_prevs):
        index = D.U.sampling_index(sample_size, *requested_prev, random_state=seed + sample_id)
        collection = D.U.sampling_from_index(index)
        posteriors = h.predict_proba(collection.X)
        samples.append(
            TestSample(
                sample_id=sample_id,
                requested_prev=requested_prev,
                actual_prev=collection.prevalence(),
                collection=collection,
                posteriors=posteriors,
            )
        )
    return samples


def _parse_distances(raw_distances: list[str] | None) -> list[str]:
    if raw_distances:
        distances = []
        for item in raw_distances:
            distances.extend(part.strip() for part in item.split(",") if part.strip())
    else:
        distances = list(PrediQuant.DISTANCES)

    invalid = sorted(set(distances).difference(PrediQuant.DISTANCES))
    if invalid:
        raise ValueError(f"Invalid PrediQuant distances {invalid}; valid values are {list(PrediQuant.DISTANCES)}")

    return distances


def _prediction_summary(prediction: Any) -> tuple[float | None, float | None, float | None]:
    if hasattr(prediction, "point_estimate"):
        point = prediction.point_estimate
        low = getattr(prediction, "I_low", None)
        high = getattr(prediction, "I_high", None)
        return float(np.asarray(point).mean()), _maybe_float(low), _maybe_float(high)

    arr = np.asarray(prediction)
    if arr.size == 0:
        return None, None, None
    if arr.ndim == 0:
        return float(arr), None, None
    return float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except TypeError:
        pass
    return float(value)


def _prediquant_predict_with_val_prevs(method: PrediQuant, sample: TestSample):
    test_prior = method._predict_test_prior(sample.collection.X)
    closest_idx = method._predict_closest_idx(test_prior)[: method.num_samples]
    val_prevs = np.asarray(method.sigma_pred_prevs, dtype=float)[closest_idx]
    val_true_prevs = np.asarray(method.sigma_true_prevs, dtype=float)[closest_idx]
    accs = np.array([method.acc(ct) for ct in method.sigma_ct])[closest_idx]

    return accs, test_prior, val_prevs, val_true_prevs


def _ternary_xy(prevs: np.ndarray) -> np.ndarray:
    prevs = np.asarray(prevs, dtype=float)
    x = prevs[..., 1] + 0.5 * prevs[..., 2]
    y = (np.sqrt(3) / 2) * prevs[..., 2]
    return np.stack([x, y], axis=-1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    return np.asarray(lower[:-1] + upper[:-1])


def _draw_simplex_panel(
    ax,
    distance: str,
    true_test_prev: np.ndarray,
    predicted_test_prev: np.ndarray,
    predicted_val_prevs: np.ndarray,
    true_val_prevs: np.ndarray,
    dist_max: float,
):
    triangle = _ternary_xy(np.eye(3))
    true_test_xy = _ternary_xy(true_test_prev.reshape(1, 3))[0]
    predicted_test_xy = _ternary_xy(predicted_test_prev.reshape(1, 3))[0]
    predicted_val_xy = _ternary_xy(predicted_val_prevs)
    true_val_xy = _ternary_xy(true_val_prevs)

    ax.add_patch(Polygon(triangle, closed=True, facecolor="white", edgecolor="black", linewidth=1.2))

    for points_xy, face_color, edge_color, label in (
        (true_val_xy, "#54A24B", "#347433", "true val"),
        (predicted_val_xy, "#4C78A8", "#2F5F93", "pred val"),
    ):
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=13, color=edge_color, alpha=0.50, label=label)
        hull = _convex_hull(points_xy)
        if len(hull) >= 3:
            ax.add_patch(
                Polygon(hull, closed=True, facecolor=face_color, edgecolor=edge_color, alpha=0.16, linewidth=1.0)
            )
        elif len(hull) == 2:
            ax.plot(hull[:, 0], hull[:, 1], color=edge_color, linewidth=1.0, alpha=0.8)

    ax.scatter(
        [true_test_xy[0]],
        [true_test_xy[1]],
        s=82,
        color="#E45756",
        edgecolor="black",
        linewidth=0.6,
        label="true test",
        zorder=5,
    )
    ax.scatter(
        [predicted_test_xy[0]],
        [predicted_test_xy[1]],
        s=82,
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
        marker="D",
        label="pred test",
        zorder=6,
    )
    labels = [r"$p_0$", r"$p_1$", r"$p_2$"]
    offsets = [(-0.04, -0.04), (0.025, -0.04), (0.0, 0.035)]
    for xy, label, offset in zip(triangle, labels, offsets):
        ax.text(xy[0] + offset[0], xy[1] + offset[1], label, fontsize=11)

    ax.set_title(f"{distance} | max d={dist_max:.3f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.axis("off")


def _prev_array(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        arr = np.stack([np.asarray(v, dtype=float) for v in arr])
    else:
        arr = arr.astype(float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _plot_grouped_sample(df: pd.DataFrame, dataset: str, classifier: str, acc_name: str, sample_id: int):
    rows = df.loc[df["sample_id"] == sample_id, :].copy()
    if rows.empty:
        return
    rows["distance"] = pd.Categorical(rows["distance"], categories=list(PrediQuant.DISTANCES), ordered=True)
    rows = rows.sort_values("distance")

    fig, axes = plt.subplots(1, len(rows), figsize=(5.1 * len(rows), 4.8), squeeze=False)
    axes = axes[0]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        _draw_simplex_panel(
            ax,
            row["distance"],
            _prev_array(row["actual_test_prev"])[0],
            _prev_array(row["predicted_test_prev"])[0],
            _prev_array(row["selected_val_prevs"]),
            _prev_array(row["selected_val_true_prevs"]),
            row["dist_max"],
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{dataset} | {acc_name} | {classifier} | sample {sample_id:02d}", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, top=0.86)

    stem = _plot_stem(dataset, classifier, acc_name, sample_id)
    fig.savefig(f"{stem}.png", dpi=180)
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


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
        "runtime": elapsed,
        "confidence_level": CONFIDENCE_LEVEL,
    }


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
):
    result_path = _result_path(p.domain, p.d_info.name, p.h_info.full_name, acc_name)
    existing = pd.DataFrame() if overwrite else _load_existing(result_path)
    done = set() if overwrite else _completed_keys(existing)

    D = p.load_dataset_bundle()
    h = p.load_pretrained_classifier(D)
    val_posteriors = h.predict_proba(D.V.X)
    acc_fn = lambda ct: acc_from_ct(acc_name, ct, type="selection")
    requested_prevs = _simplex_points(np.random.RandomState(seed), num_test)
    samples = _make_test_samples(D, h, requested_prevs, sample_size, seed)

    rows = []
    for distance in distances:
        method = PrediQuant(
            acc_fn,
            # PACC(h),
            kdey(),
            num_samples=num_samples,
            sample_size=sample_size,
            distance=distance,
            # reuse_h=h,
            random_state=seed,
        )
        method.fit(D.V, val_posteriors)
        distance_rows = []
        for sample in samples:
            key = (distance, sample.sample_id)
            if key in done:
                log.info(f"[{p.h_info.name}@{p.d_info.name}] {distance} sample {sample.sample_id} exists, skipping")
                continue

            tinit = time()
            prediction, predicted_test_prev, val_prev, val_true_prev = _prediquant_predict_with_val_prevs(
                method, sample
            )
            # prediction = method.predict_range(sample.collection.X, sample.posteriors)
            # val_prev = _extract_prediquant_validation_prevs(method, sample)
            elapsed = time() - tinit
            rec = _record(
                p,
                D.L_prevalence,
                D.V.prevalence(),
                acc_name,
                distance,
                sample,
                predicted_test_prev,
                val_prev,
                val_true_prev,
                prediction,
                elapsed,
            )
            distance_rows.append(rec)

        if distance_rows:
            rows.extend(distance_rows)
            combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
            _atomic_save_parquet(combined, result_path)
            log.info(f"[{p.h_info.name}@{p.d_info.name}] {distance} saved to {result_path}")

    if plot:
        final_df = _load_existing(result_path)
        if not final_df.empty:
            for sample_id in sorted(final_df["sample_id"].unique()):
                _plot_grouped_sample(final_df, p.d_info.name, p.h_info.full_name, acc_name, int(sample_id))


def _run_one_pretrain_args(args):
    (
        p,
        distances,
        num_test,
        sample_size,
        num_samples,
        seed,
        acc_name,
        plot,
        overwrite,
    ) = args
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
    except Exception as e:
        log.warning(f"[{p.h_info.name}@{p.d_info.name}] failed: {e}")
        print_exception(e)
        return False, p.d_info.name, p.h_info.name, repr(e)


def run(args):
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
        for exp_args in experiment_args:
            _run_one_pretrain_args(exp_args)
    else:
        results = parallel(
            func=_run_one_pretrain_args,
            args_list=experiment_args,
            n_jobs=args.n_jobs,
            return_as="generator_unordered",
            max_nbytes=None,
        )
        for ok, dataset, classifier, err in results:
            if ok:
                log.info(f"[{classifier}@{dataset}] completed")
            else:
                log.warning(f"[{classifier}@{dataset}] failed: {err}")


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


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
