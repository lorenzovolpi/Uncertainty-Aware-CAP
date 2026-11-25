r"""The methods of E5.3 with the temperature chosen per pair on calibration bags, as columns of the main and rank tables
(work-plan card WP-33).

Since WP-34, ``BayesCAP-Hf-real-tcal`` and ``BayCAP-KDEy-h-inst-cb-tcal-resel`` are stored methods of ``main`` (the
selection inside the fit, ``methods/tcal.py``); their assembled columns here are ``<base>-tcal-asm``, kept to check the
stored methods against them until they match (user's decision of 2026-11-25), and ``BayCAP-KDEy-h-inst-tcal`` keeps its
name.

``<base>-tcal`` or ``<base>-tcal-asm`` (``tables/common.py:TCAL_METHODS``, one per base of ``TEMPERED_BASES``) is not
a stored method: on each pair and metric it is the stored test run of ``main`` (E5.2) of ``<base>`` at $\tau_{\text{cal}}$, the temperature with
the lowest mean Winkler score of that metric on the $50$ calibration bags drawn from $V_{\text{cal}}$ by
``experiments/tau_select.py`` (``tables/tau_select.py:cal_taus``, ties to the $\tau$ closest to $1$). It uses only the
labels of $V$, so it is a method, not a reference. The fits on the calibration bags are on $V_{\text{fit}}$, those of
the test runs on the full $V$, as in E5.3. The intervals target the realised table of the target bag and are scored
against it (``true_cts``), as every method of ``main``. A pair without calibration runs, or without the test run at its
$\tau_{\text{cal}}$, is left out for that method.
"""

import numpy as np
import pandas as pd

from uncawe.results import Results
from uncawe.tables import tau_select as ts
from uncawe.tables import tempering as tt
from uncawe.tables.common import TCAL_METHODS

KEYS = tt.KEYS


def tcal_taus(acc_name: str, methods, root: str | None = None) -> pd.DataFrame:
    """Per ``-tcal`` method of ``methods`` and pair with calibration runs: ``method``, ``dataset``, ``classifier`` and
    ``tau_cal`` for ``acc_name`` (empty if none is stored)."""
    bases = tuple(dict.fromkeys(TCAL_METHODS[m] for m in methods if m in TCAL_METHODS))
    columns = ["method", *KEYS, "tau_cal"]
    if not bases:
        return pd.DataFrame(columns=columns)
    try:
        cal = ts.calibration_table([acc_name], root, bases)
    except RuntimeError:
        return pd.DataFrame(columns=columns)
    cal = cal.assign(tau_cal=ts.cal_taus(cal))
    by_base = {base: name for name, base in TCAL_METHODS.items() if name in methods}
    cal["method"] = cal["method"].map(by_base)
    return cal[columns].reset_index(drop=True)


def tcal_rows(
    base_dir: str, acc_name: str, domain: str, classifiers: list[str], methods, root: str | None = None
) -> pd.DataFrame:
    """The stored rows of ``main`` that make up the ``-tcal`` methods of ``methods``: per pair, those of the base at
    $\\tau_{\\text{cal}}$, with ``method`` renamed to the ``-tcal`` method (empty if there is none)."""
    taus = tcal_taus(acc_name, methods, root)
    if taus.empty:
        return pd.DataFrame()
    stored = [tt.method_name(TCAL_METHODS[name], tau) for name in taus["method"].unique() for tau in tt.TAUS]
    try:
        df = Results.load(
            base_dir=base_dir,
            acc_name=acc_name,
            domain=domain,
            filter_classifiers=classifiers,
            filter_methods=stored,
        ).df
    except ValueError:  # no stored file
        return pd.DataFrame()
    parts = []
    for row in taus.to_dict("records"):
        name = tt.method_name(TCAL_METHODS[row["method"]], float(row["tau_cal"]))
        match = (df["method"] == name) & (df["dataset"] == row["dataset"]) & (df["classifier"] == row["classifier"])
        if np.any(match):
            parts.append(df.loc[match].assign(method=row["method"]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def with_tcal(results: Results, base_dir: str, acc_name: str, domain: str, classifiers: list[str], methods) -> Results:
    """``results`` with the rows of the ``-tcal`` methods of ``methods`` appended."""
    if not any(m in TCAL_METHODS for m in methods):
        return results
    rows = tcal_rows(base_dir, acc_name, domain, classifiers, methods)
    if rows.empty:
        return results
    return Results(pd.concat([results.df, rows], ignore_index=True))
