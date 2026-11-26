"""WP-34 check: the stored -tcal methods of main against the columns assembled from the stored runs (the tau_select
calibration runs and the main test runs at tau_cal; for -keep, the -keep grid of main). Reads stored results only.

Run on bcuda from the pushed tree: ssh bcuda "cd /home/volpi/uncawe-devel && UV_PYTHON=<REMOTE_PY> <REMOTE_UV> run --locked
python -" < scripts/checks/wp34_tcal_check.py"""

import os

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.config import get_acc_names
from uncawe.results import Results
from uncawe.tables import tau_select as ts
from uncawe.tables import tempering as tt
from uncawe.tables.main import CLASSIFIER_FILTER

BASE_DIR = os.path.join(env.root_dir, "main")
DOMAIN = "classic"
CLASSIFIERS = list(CLASSIFIER_FILTER)
# stored -tcal method -> (calibration base of tau_select, method whose main runs at tau_cal make the assembled column)
CHECKS = {
    "BayesCAP-tcal": ("BayesCAP", "BayesCAP"),
    "BayesCAP-real-tcal": ("BayesCAP-real", "BayesCAP-real"),
    "BayesCAP-Hf-real-tcal": ("BayesCAP-Hf-real", "BayesCAP-Hf-real"),
    "BayCAP-KDEy-h-inst-cb-tcal-resel": ("BayCAP-KDEy-h-inst-cb", "BayCAP-KDEy-h-inst-cb"),
    "BayCAP-KDEy-h-inst-cb-tcal-keep": ("BayCAP-KDEy-h-inst-cb-keep", "BayCAP-KDEy-h-inst-cb-keep"),
}
KEYS = ["dataset", "classifier"]


def load(acc, methods):
    return Results.load(
        base_dir=BASE_DIR, acc_name=acc, domain=DOMAIN, filter_classifiers=CLASSIFIERS, filter_methods=methods
    ).df


def ranges(df):
    return np.stack([np.asarray(r, dtype=float).ravel() for r in df["estim_acc_ranges"]])


rows = []
for acc in get_acc_names():
    cal = ts.calibration_table([acc], None, tuple(dict.fromkeys(b for b, _ in CHECKS.values())))
    cal = cal.assign(tau_cal=ts.cal_taus(cal))
    for stored_name, (cal_base, test_base) in CHECKS.items():
        stored = load(acc, [stored_name])
        grid = load(acc, [tt.method_name(test_base, t) for t in tt.TAUS])
        cb = cal[cal["method"] == cal_base].set_index(KEYS)
        for (dataset, classifier), part in stored.groupby(KEYS, sort=True):
            r = {"metric": acc, "method": stored_name, "dataset": dataset, "classifier": classifier}
            tau_stored = float(part["tau_cal"].iloc[0])
            r["tau_stored"] = tau_stored
            r["fallback"] = bool(part["tau_fallback"].iloc[0])
            if (dataset, classifier) not in cb.index:
                r["status"] = "no calibration runs"
                rows.append(r)
                continue
            c = cb.loc[(dataset, classifier)]
            tau_asm = float(c["tau_cal"])
            r["tau_asm"] = tau_asm
            r["tau_equal"] = np.isclose(tau_stored, tau_asm)
            # mean calibration Winkler score at each tau: stored vs from the tau_select runs
            stored_cal = dict(zip(part["cal_taus"].iloc[0], part["cal_winkler"].iloc[0]))
            asm_cal = {t: float(c[f"t{tt.tau_label(t)}_winkler"]) for t in tt.TAUS}
            common = [t for t in stored_cal if t in asm_cal]
            r["cal_max_abs_diff"] = max(abs(stored_cal[t] - asm_cal[t]) for t in common) if common else np.nan
            # test bags: stored method vs the grid run at the stored tau_cal
            g = grid[
                (grid["method"] == tt.method_name(test_base, tau_stored))
                & (grid["dataset"] == dataset)
                & (grid["classifier"] == classifier)
            ]
            if len(g) != len(part):
                r["status"] = f"bags {len(part)} vs grid {len(g)}"
                rows.append(r)
                continue
            r["n_bags"] = len(part)
            r["true_cts_equal"] = all(
                np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(part["true_cts"], g["true_cts"])
            )
            r["acc_err_max_abs_diff"] = float(np.max(np.abs(part["acc_err"].to_numpy() - g["acc_err"].to_numpy())))
            r["ranges_max_abs_diff"] = float(np.nanmax(np.abs(ranges(part) - ranges(g))))
            r["status"] = "ok"
            rows.append(r)

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
print(
    out.groupby(["method", "metric"]).agg(
        pairs=("dataset", "size"),
        ok=("status", lambda s: int((s == "ok").sum())),
        fallback=("fallback", "sum"),
        tau_equal=("tau_equal", "sum"),
        cal_max=("cal_max_abs_diff", "max"),
        true_cts_equal=("true_cts_equal", "sum"),
        acc_err_max=("acc_err_max_abs_diff", "max"),
        ranges_max=("ranges_max_abs_diff", "max"),
    )
)
bad = out[(out["status"] != "ok") | (~out["tau_equal"].fillna(False).astype(bool))]
print("\nrows not ok or with different tau:")
print(bad.to_string() if len(bad) else "none")
print("\nlargest per-bag differences:")
print(out.sort_values("ranges_max_abs_diff", ascending=False).head(10).to_string())
