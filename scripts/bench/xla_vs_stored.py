"""Compare BayCAP results recomputed with single-threaded XLA on CPU against the stored ``main`` results.

The stored ``main`` results of the BayCAP methods were computed with the default XLA threading (``BayesCAP-real``: CPU,
run of 2026-10-21; ``BayesCAP``: earlier runs, probably on GPU). Design A of the BayCAP pool runs every process with
one computing thread, so this script checks whether that changes the results. For each (dataset, classifier, method)
with stored results it repeats the ``main`` task as ``main`` does (pretrain load, the 100 test bags of ``main``, ``fit``,
``predict_ct_range`` on every bag, ``ci_from_cts`` for vanilla accuracy) and saves nothing in the results tree.

Every process computes on one thread: ``JAX_PLATFORMS=cpu``,
``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`` and one BLAS/OpenMP thread, set before
anything is imported, so the ``joblib`` workers inherit them. The tasks run with the ``joblib`` call of ``run_tasks``
(``--n-jobs`` workers, default 16, at most ``cap.env["N_JOBS"]``).

Reported per (pair, method), vanilla accuracy: whether the 100 bags are the same as the stored ones (equal realised
tables ``true_cts``, without which the comparison is void), whether the results are identical bit for bit, the largest
absolute difference of the mean estimated tables (``estim_cts``) and of the accuracy samples (``estim_acc_ranges``),
and coverage, amplitude and Winkler score (means over the bags, against the realised table) of the stored and the new
results. Methods target as in ``main``: ``BayesCAP`` the population table, ``BayesCAP-real`` the realised table; both
are scored against the realised table.

Usage (from the repository root): uv run python scripts/bench/xla_vs_stored.py [--n-jobs 16]
    [--datasets german spambase dry-bean wine-quality letter] [--classifiers LR] [--methods BayesCAP ...]
Output: ``<OUT_DIR>/uncawe/bench/xla_vs_stored/comparison.csv``.
"""

import os

# one computing thread per process, set before JAX, NumPy or joblib are imported (workers inherit the environment)
os.environ.update(
    {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
)

from argparse import ArgumentParser

import numpy as np
import pandas as pd
from cap import env as capenv
from cap.utils.commons import parallel

from uncawe import env
from uncawe.experiments import main as main_exp
from uncawe.experiments.methods import gen_methods, get_method_names
from uncawe.experiments.metrics import acc_from_ct, gen_acc_measure
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.results import RDF
from uncawe.tables import main as main_tables

ACC_NAME = "vanilla_accuracy"
DEFAULT_DATASETS = ["german", "spambase", "dry-bean", "wine-quality", "letter"]
SCORES = ("coverage", "amplitude", "winkler")


def stored_path(p: PretrainInfo, method_name: str) -> str:
    """Path of the stored result, as ``util.local_path`` builds it for ``main``, without creating folders."""
    return os.path.join(
        env.root_dir,
        main_exp.EXPERIMENT,
        p.domain,
        ACC_NAME,
        p.d_info.name,
        p.h_info.full_name,
        f"{method_name}.parquet",
    )


def recompute(args) -> dict:
    """Repeat the ``main`` task of one (pair, method) for vanilla accuracy and return the per-bag outputs."""
    info_path, method_name, method = args
    p = PretrainInfo.load(info_path, fast=True)
    dataset, validation, validation_posteriors, context = main_exp._prepare_evaluation(p, method)
    method = method.fit(validation, validation_posteriors)
    estim_ct_ranges = [
        method.predict_ct_range(sample.X, posteriors)
        for sample, posteriors in zip(dataset.test_prot(), dataset.test_prot_posteriors)
    ]
    acc_fn = dict(gen_acc_measure(p.d_info.n_classes > 2))[ACC_NAME]
    method.switch(acc_fn)
    return {
        "info_path": info_path,
        "method": method_name,
        "true_cts": context.true_cts,
        "estim_cts": [cts.mean(axis=0) for cts in estim_ct_ranges],
        "estim_acc_ranges": [method.ci_from_cts(cts).samples for cts in estim_ct_ranges],
    }


def mean_scores(true_cts, estim_acc_ranges) -> dict:
    """Coverage, amplitude and Winkler score of the accuracy samples, means over the bags (as the ``main`` tables)."""
    bags = []
    for true_ct, samples in zip(true_cts, estim_acc_ranges):
        true_acc = acc_from_ct(ACC_NAME, np.asarray(true_ct, dtype=float))
        samples = np.asarray(samples, dtype=float)
        bags.append(
            (
                main_tables.get_acc_coverage("", true_acc, samples),
                main_tables.get_acc_amplitude("", samples),
                main_tables.get_acc_winkler("", true_acc, samples),
            )
        )
    return dict(zip(SCORES, np.mean(bags, axis=0).tolist()))


def compare(new: dict, stored: pd.DataFrame) -> dict:
    stored_true = np.stack([np.asarray(ct, dtype=float) for ct in stored["true_cts"]])
    new_true = np.stack([np.asarray(ct, dtype=float) for ct in new["true_cts"]])
    same_bags = stored_true.shape == new_true.shape and np.array_equal(stored_true, new_true)
    stored_cts = np.stack([np.asarray(ct, dtype=float) for ct in stored["estim_cts"]])
    new_cts = np.stack([np.asarray(ct, dtype=float) for ct in new["estim_cts"]])
    stored_acc = np.stack([np.asarray(s, dtype=float) for s in stored["estim_acc_ranges"]])
    new_acc = np.stack([np.asarray(s, dtype=float) for s in new["estim_acc_ranges"]])
    comparable = same_bags and stored_cts.shape == new_cts.shape and stored_acc.shape == new_acc.shape
    row = {
        "same_bags": same_bags,
        "identical": comparable and np.array_equal(stored_cts, new_cts) and np.array_equal(stored_acc, new_acc),
        "max_diff_cts": float(np.abs(stored_cts - new_cts).max()) if comparable else np.nan,
        "max_diff_acc_samples": float(np.abs(stored_acc - new_acc).max()) if comparable else np.nan,
    }
    stored_scores = mean_scores(stored["true_cts"], stored["estim_acc_ranges"])
    new_scores = mean_scores(new["true_cts"], new["estim_acc_ranges"])
    for score in SCORES:
        row[f"stored_{score}"] = stored_scores[score]
        row[f"new_{score}"] = new_scores[score]
        row[f"diff_{score}"] = new_scores[score] - stored_scores[score]
    return row


def main():
    parser = ArgumentParser(description="Compare single-threaded XLA BayCAP results with the stored main results.")
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--classifiers", nargs="+", default=["LR"])
    parser.add_argument("--methods", nargs="+", default=None, help="Default: the BayCAP methods of main")
    args = parser.parse_args()
    if args.n_jobs > capenv["N_JOBS"]:
        raise ValueError(f"--n-jobs {args.n_jobs} exceeds cap.env['N_JOBS'] = {capenv['N_JOBS']}.")

    methods = args.methods or [name for name in get_method_names() if "BayesCAP" in name]
    tasks, stored = [], {}
    for info_path in load_info_paths(domain=main_exp.DOMAIN):
        p = PretrainInfo.load(info_path, fast=True)
        if p.d_info.name not in args.datasets or p.h_info.name not in args.classifiers:
            continue
        for method_name, method in gen_methods():
            if method_name not in methods:
                continue
            path = stored_path(p, method_name)
            if not os.path.exists(path):
                print(f"[{p.h_info.name}@{p.d_info.name}] {method_name}: no stored result, skipped", flush=True)
                continue
            stored[(info_path, method_name)] = RDF.load_result(path)
            tasks.append((info_path, method_name, method))
    if not tasks:
        raise ValueError("No stored result matches --datasets, --classifiers and --methods.")
    print(f"{len(tasks)} tasks, {min(args.n_jobs, len(tasks))} workers", flush=True)

    if args.n_jobs == 1:
        outputs = (recompute(task) for task in tasks)
    else:
        outputs = parallel(
            func=recompute,
            args_list=tasks,
            n_jobs=args.n_jobs,
            return_as="generator_unordered",
            max_nbytes=None,
        )
    rows = []
    for new in outputs:
        frame = stored[(new["info_path"], new["method"])]
        row = {
            "dataset": frame["dataset"].iloc[0],
            "classifier": frame["classifier"].iloc[0],
            "method": new["method"],
            "n_classes": int(frame["n_classes"].iloc[0]),
        } | compare(new, frame)
        rows.append(row)
        print(f"[{row['classifier']}@{row['dataset']}] {row['method']}: identical={row['identical']}", flush=True)

    table = pd.DataFrame(rows).sort_values(["n_classes", "dataset", "classifier", "method"])
    out_dir = os.path.join(env.root_dir, "bench", "xla_vs_stored")
    os.makedirs(out_dir, exist_ok=True)
    table.to_csv(os.path.join(out_dir, "comparison.csv"), index=False)
    with pd.option_context("display.width", 250, "display.precision", 4, "display.max_columns", 30):
        print(table.to_string(index=False))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
