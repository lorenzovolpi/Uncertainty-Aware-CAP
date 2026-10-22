"""Benchmark the number of parallel workers of the BayCAP pool of ``main`` on CPU (``BAYES_N_JOBS``).

``main`` runs the NumPyro methods (method names containing "BayesCAP") in their own pool with ``BAYES_N_JOBS = 1``.
With ``JAX_PLATFORMS=cpu`` each task is one sequential NUTS chain per bag, so several workers may run side by side.
This script measures whether they do, on the real workload of a ``main`` task: for each (dataset, classifier, method)
it loads the pretrain data, draws the test bags as ``main`` does (``--bags`` of them instead of 100), fits the method
on $V$ and calls ``predict_ct_range`` on every bag. It does not run ``ci_from_cts`` (cheap) and writes no result.

Each setting of ``n_jobs`` runs in a fresh subprocess, so every setting pays the JIT compilation of a cold start, as a
run of ``main`` does; ``n_jobs=1`` runs in the subprocess itself, as ``run_tasks`` does, and ``n_jobs>1`` uses the same
``joblib`` call as ``run_tasks``. Every process computes on one thread: XLA runs single-threaded
(``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1``) and the BLAS/OpenMP pools have one
thread, so ``n_jobs`` processes keep at most about ``n_jobs`` cores busy; the other OS threads of a process (compilation,
dispatch, Python) stay mostly idle and the OS may place them on any core. ``n_jobs`` above ``cap.env["N_JOBS"]`` is
refused.

Reported per setting: wall time, tasks per hour, speed-up against ``n_jobs=1``, the average number of busy cores (CPU
time of the subprocess and its workers over the wall time), and the largest absolute difference of the posterior mean
tables from the first setting (zero means the parallel run gives the same draws). The load average is recorded because
``bcuda`` is shared. Each task also records the number of OS threads of its process when it ends (entries of
``/proc/self/task``, all threads, idle ones included); the summary reports the largest of them per setting.

Usage (from the repository root, on the machine to measure):
    uv run python scripts/bench/bayes_n_jobs.py [--n-jobs 1 2 4 8 16] [--bags 10]
        [--datasets ...] [--classifiers LR] [--methods BayesCAP ...]
Output: ``<OUT_DIR>/uncawe/bench/bayes_n_jobs/`` (``tasks.csv``, ``summary.csv``).
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
from argparse import SUPPRESS, ArgumentParser
from time import time

# one computing thread per process: single-threaded XLA, and one thread in the BLAS/OpenMP pools
SINGLE_THREAD_ENV = {
    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def bench_task(args):
    """Run the NUTS part of one ``main`` task and return its timings and a fingerprint of the draws."""
    import numpy as np

    from uncawe.experiments import main as main_exp
    from uncawe.pretrain.data import PretrainInfo

    info_path, method_name, method, bags = args
    main_exp.NUM_TEST = bags
    started = time()
    p = PretrainInfo.load(info_path, fast=True)
    dataset, validation, validation_posteriors, _ = main_exp._prepare_evaluation(p, method)
    t_load = time() - started

    started = time()
    method = method.fit(validation, validation_posteriors)
    ct_ranges = [
        method.predict_ct_range(sample.X, posteriors)
        for sample, posteriors in zip(dataset.test_prot(), dataset.test_prot_posteriors)
    ]
    t_nuts = time() - started
    return {
        "dataset": p.d_info.name,
        "classifier": p.h_info.name,
        "method": method_name,
        "n_classes": int(p.d_info.n_classes),
        "pid": os.getpid(),
        "threads": len(os.listdir("/proc/self/task")),
        "t_load": t_load,
        "t_nuts": t_nuts,
        "fingerprint": np.mean([np.asarray(cts, dtype=float).mean(axis=0) for cts in ct_ranges], axis=0).tolist(),
    }


def _cpu_time() -> float:
    """CPU time (user and system) of this process and of its joined children."""
    usage = [resource.getrusage(who) for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(u.ru_utime + u.ru_stime for u in usage)


def run_setting(config: dict) -> dict:
    """Body of the subprocess: build the tasks and run them with ``config["n_jobs"]`` workers."""
    from cap.utils.commons import parallel
    from joblib.externals.loky import get_reusable_executor

    # importing main sets qp.environ as a run of main does (seed 0, sample size), which the methods read when built
    import uncawe.experiments.main  # noqa: F401
    from uncawe.experiments.methods import gen_methods

    tasks = []
    for info_path in config["info_paths"]:
        for method_name, method in gen_methods():
            if method_name in config["methods"]:
                tasks.append((info_path, method_name, method, config["bags"]))

    cpu_before = _cpu_time()
    started = time()
    if config["n_jobs"] == 1:
        rows = [bench_task(task) for task in tasks]
    else:
        rows = list(
            parallel(
                func=bench_task,
                args_list=tasks,
                n_jobs=config["n_jobs"],
                return_as="generator_unordered",
                max_nbytes=None,
            )
        )
        # join the workers, so that their CPU time is counted in RUSAGE_CHILDREN
        get_reusable_executor().shutdown(wait=True, kill_workers=False)
    return {"wall": time() - started, "cpu": _cpu_time() - cpu_before, "rows": rows}


def select_pairs(datasets, classifiers) -> list[str]:
    from uncawe.experiments.datasets import get_filtered_datasets
    from uncawe.experiments.main import DOMAIN
    from uncawe.pretrain.data import PretrainInfo, load_info_paths

    datasets = get_filtered_datasets() if datasets is None else datasets
    paths = []
    for path in load_info_paths(domain=DOMAIN):
        p = PretrainInfo.load(path, fast=True)
        if p.d_info.name in datasets and p.h_info.name in classifiers:
            paths.append(path)
    return paths


def spawn_setting(config: dict) -> dict:
    env = os.environ | {"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""} | SINGLE_THREAD_ENV
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path, out_path = os.path.join(tmp, "config.json"), os.path.join(tmp, "out.json")
        with open(cfg_path, "w") as f:
            json.dump(config, f)
        subprocess.run([sys.executable, __file__, "--child", cfg_path, out_path], env=env, check=True)
        with open(out_path) as f:
            return json.load(f)


def main():
    parser = ArgumentParser(description="Benchmark the workers of the BayCAP pool of main on CPU.")
    parser.add_argument("--n-jobs", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--bags", type=int, default=10, help="Test bags per task (main uses 100)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Default: the datasets of main")
    parser.add_argument("--classifiers", nargs="+", default=["LR"])
    parser.add_argument("--methods", nargs="+", default=None, help="Default: the BayCAP methods of main")
    parser.add_argument("--child", nargs=2, help=SUPPRESS)  # internal: config and output paths of a setting
    args = parser.parse_args()

    if args.child is not None:
        cfg_path, out_path = args.child
        with open(cfg_path) as f:
            out = run_setting(json.load(f))
        with open(out_path, "w") as f:
            json.dump(out, f)
        return

    import numpy as np
    import pandas as pd
    from cap import env as capenv

    from uncawe import env
    from uncawe.experiments.methods import get_method_names

    too_many = [n for n in args.n_jobs if n > capenv["N_JOBS"]]
    if too_many:
        raise ValueError(f"--n-jobs {too_many} exceed cap.env['N_JOBS'] = {capenv['N_JOBS']}.")

    methods = args.methods or [name for name in get_method_names() if "BayesCAP" in name]
    info_paths = select_pairs(args.datasets, args.classifiers)
    if not info_paths:
        raise ValueError("No pretrained pair matches --datasets and --classifiers.")
    print(
        f"{len(info_paths)} pairs x {len(methods)} methods ({', '.join(methods)}), {args.bags} bags per task, "
        f"{os.cpu_count()} CPUs on the machine",
        flush=True,
    )

    out_dir = os.path.join(env.root_dir, "bench", "bayes_n_jobs")
    os.makedirs(out_dir, exist_ok=True)
    task_rows, summary, reference = [], [], None
    for n_jobs in args.n_jobs:
        load = os.getloadavg()[0]
        config = {"info_paths": info_paths, "methods": methods, "bags": args.bags, "n_jobs": n_jobs}
        out = spawn_setting(config)
        rows = sorted(out["rows"], key=lambda r: (r["dataset"], r["classifier"], r["method"]))
        prints = np.concatenate([np.ravel(r["fingerprint"]) for r in rows])
        reference = prints if reference is None else reference
        for r in rows:
            task_rows.append({"n_jobs": n_jobs} | {k: v for k, v in r.items() if k != "fingerprint"})
        summary.append(
            {
                "n_jobs": n_jobs,
                "tasks": len(rows),
                "wall_s": out["wall"],
                "tasks_per_h": 3600 * len(rows) / out["wall"],
                "mean_task_s": float(np.mean([r["t_load"] + r["t_nuts"] for r in rows])),
                "busy_cores": out["cpu"] / out["wall"],
                "workers": len({r["pid"] for r in rows}),
                "max_threads": max(r["threads"] for r in rows),
                "max_abs_diff": float(np.abs(prints - reference).max()),
                "load_before": load,
            }
        )
        print(pd.DataFrame(summary[-1:]).to_string(index=False, header=len(summary) == 1), flush=True)

    summary = pd.DataFrame(summary)
    base = summary.loc[summary["n_jobs"] == 1, "wall_s"]
    summary["speedup"] = base.iloc[0] / summary["wall_s"] if not base.empty else np.nan
    pd.DataFrame(task_rows).to_csv(os.path.join(out_dir, "tasks.csv"), index=False)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary.to_string(index=False))
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
