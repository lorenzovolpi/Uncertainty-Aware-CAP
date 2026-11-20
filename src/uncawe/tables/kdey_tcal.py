r"""The ablation of E7.2 of ``docs/ext7_kdey_bandwidth.md`` (work-plan card WP-26): bandwidth and temperature of BayesKDEy
selected per pair on calibration bags drawn from half of $V$, each step alone and both (E7.2b).

The four cells, all ``BayCAP-KDEy-mlp-real`` (``experiments/methods.py``) with the seeds of ``main``, fitted on the full
$V$ and run on the $100$ target bags of ``main``; all target the realised table of each bag and are scored against it
(``true_cts``); each uses only the labels of $V$, so each is a method:

- ``stored``: $b=0.1$, $\tau=1$, ``BayCAP-KDEy-mlp-real``;
- ``e53``: $b=0.1$, $\tau=\tau_{\text{cal}}$ of E5.3 of ``docs/ext5_baycap_priors_tempering.md``, the stored
  ``BayCAP-KDEy-mlp-real-t<τ>`` of E5.2 (or the stored method at $\tau=1$) at the $\tau$ that E5.3 selects for the pair on
  vanilla accuracy, computed from the stored calibration runs of ``experiments/tau_select.py`` as
  ``tables/tau_select.py`` does (``tau_select.cal_taus``); no new run, and NaN where those runs are not stored;
- ``cal``: $b=b_{\text{cal}}$, $\tau=1$, ``BayCAP-KDEy-mlp-real-cal`` (E7.2a);
- ``tcal``: $b=b_{\text{cal}}$, $\tau=\tau_{\text{cal}}$, ``BayCAP-KDEy-mlp-real-cal-tcal`` (E7.2b), $\tau$ selected on
  the calibration bags of $b_{\text{cal}}$ after it.

Per metric and pair (``pair_table``, the pairs where ``stored``, ``cal`` and ``tcal`` are stored in ``main``):

- AE, coverage, amplitude and Winkler score at $\alpha=0.05$ of the four cells, means over the bags;
- for AE and the Winkler score, ``tcal`` against each other cell: the mean per-bag difference (``tcal`` minus the cell),
  its paired bootstrap interval and $p$-value (``tables/pushforward.py:bootstrap_mean_test``, $10{,}000$ resamples),
  Holm-adjusted over the pairs, per metric, cell and score;
- label-free, from the summaries stored by ``tcal``: $b_{\text{cal}}$ (checked equal to that of ``cal``),
  $\tau_{\text{cal}}$ at $b_{\text{cal}}$ and at $b=0.1$ (E5.3), and per $\tau$ of the grid the mean Winkler score and the
  coverage on the calibration bags and the NUTS diagnostics of the calibration runs (runs with a divergent transition,
  largest split-$\hat R$, smallest effective sample size).

``summary_table``: per metric, group of $K$ and cell, the means over the pairs of the four scores; for ``tcal``, against
each other cell, the pairs where its AE and its Winkler score are significantly (Holm) lower and higher and the
two-sided Wilcoxon signed-rank $p$-values over the per-pair means. ``crosstab``: $\tau_{\text{cal}}$ at
$b_{\text{cal}}$ against $\tau_{\text{cal}}$ at $b=0.1$ (E5.3), counts of pairs. ``nuts_table``: per $\tau$, the NUTS
diagnostics of the calibration runs summed over the pairs.

Outputs, under ``<OUT_DIR>/uncawe/tables/kdey_tcal/``: the CSVs (read by ``plots/kdey_tcal.py``) and, per metric, the
summary and the per-pair table, and the cross-tabulation, each as a LaTeX tabular fragment, a longtable and a
standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import CAL_TEMPERATURE_VARIANTS, CAL_VARIANTS
from uncawe.experiments.metrics import get_acc_names
from uncawe.methods.baycap_kdey import CAL_BAGS, TAU_GRID
from uncawe.tables import baycap_h, eb_prior, pushforward, tempering
from uncawe.tables import tau_select as ts
from uncawe.tables.displacement import ordered
from uncawe.tables.estimand_gap import SEED, booktabs_clines, pair_rng
from uncawe.tables.kdey_loo import two_sided_wilcoxon

TABLE_NAME = "kdey_tcal"
KEYS = ["dataset", "classifier"]
SCORES = ("ae", "coverage", "amplitude", "winkler")
TESTED = ("ae", "winkler")
# the score groups of the per-pair LaTeX table
PAIR_SCORES = ("ae", "winkler", "coverage")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
TCAL_METHOD = next(iter(CAL_TEMPERATURE_VARIANTS))
STORED_METHOD = CAL_VARIANTS[TCAL_METHOD]
CAL_METHOD = next(name for name, base in CAL_VARIANTS.items() if base == STORED_METHOD and name != TCAL_METHOD)
# the metric on which tau_cal is selected, in E5.3 and in E7.2b
SELECTION_METRIC = "vanilla_accuracy"
# the cells of the ablation; the methods stored in main for the three that are not read at a per-pair tau
CELLS = ("stored", "e53", "cal", "tcal")
CELL_METHODS = {"stored": STORED_METHOD, "cal": CAL_METHOD, "tcal": TCAL_METHOD}
# the cells against which tcal is tested
COMPARED = ("stored", "e53", "cal")
CELL_LABELS = {
    "stored": "$b{=}0.1$, $\\tau{=}1$",
    "e53": "$b{=}0.1$, $\\tau_{\\mathrm{cal}}$",
    "cal": "$b_{\\mathrm{cal}}$, $\\tau{=}1$",
    "tcal": "$b_{\\mathrm{cal}}$, $\\tau_{\\mathrm{cal}}$",
}
SHORT_LABELS = {"stored": "KDEy", "e53": "KDEy tcal (E5.3)", "cal": "KDEy cal", "tcal": "KDEy cal-tcal"}
TAUS = tuple(float(tau) for tau in TAU_GRID)


def tau_label(tau: float) -> str:
    return tempering.tau_label(tau)


# --------------------------------------------------------------------------------------------------- loading


def e53_taus(root: str | None = None) -> pd.DataFrame:
    """Per pair, $\\tau_{\\text{cal}}$ of E5.3 for ``BayCAP-KDEy-mlp-real`` on vanilla accuracy (``tau_cal_e53``), from
    the stored calibration runs of ``tau_select`` (``tables/tau_select.py:cal_taus``); empty if none is stored."""
    try:
        cal = ts.calibration_table([SELECTION_METRIC], root)
    except RuntimeError:
        return pd.DataFrame(columns=[*KEYS, "tau_cal_e53"])
    part = cal.loc[cal["method"] == STORED_METHOD].reset_index(drop=True)
    return part[KEYS].assign(tau_cal_e53=ts.cal_taus(part))


def load_cells(acc_name: str, root: str | None = None) -> tuple[dict[str, pd.DataFrame], dict[float, pd.DataFrame]]:
    """The stored rows of ``main`` for ``acc_name``: of the three methods of ``CELL_METHODS``, restricted to the pairs
    stored for all three (empty dict if there is none), and of the stored method at each $\\tau$ of the grid (``e53``)."""
    frames = {cell: tempering.load_method(method, acc_name, root) for cell, method in CELL_METHODS.items()}
    tempered = {tau: tempering.load_method(tempering.method_name(STORED_METHOD, tau), acc_name, root) for tau in TAUS}
    if any(df.empty for df in frames.values()):
        return {}, tempered
    pairs = eb_prior.common_pairs(frames)
    if not pairs:
        return {}, tempered
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    frames = {
        cell: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for cell, df in frames.items()
    }
    return frames, tempered


# --------------------------------------------------------------------------------------------------- per pair


def _first(df: pd.DataFrame, name: str):
    """The summary ``name`` stored with the first bag (fixed by the fit); NaN if it is not stored."""
    if name not in df:
        return np.nan
    return df[name].iloc[0]


def _vector(df: pd.DataFrame, name: str) -> np.ndarray:
    value = _first(df, name)
    if np.ndim(value) != 1 or len(value) != len(TAUS):
        return np.full(len(TAUS), np.nan)
    return np.asarray(value, dtype=float)


def selection_columns(tcal: pd.DataFrame, cal: pd.DataFrame) -> dict:
    """The label-free summary of the two selections of one pair, from the rows of ``tcal`` (and ``cal`` for the check
    that $b_{\\text{cal}}$ is the same)."""
    b_cal = float(_first(tcal, "cal_bandwidth"))
    out = {
        "b_cal": b_cal,
        "b_cal_same": bool(np.isclose(b_cal, float(_first(cal, "cal_bandwidth")))),
        "tau_cal": float(_first(tcal, "cal_temperature")),
        "n_val": int(_first(tcal, "cal_n_val")),
        "n_cal": int(_first(tcal, "cal_n_cal")),
    }
    columns = {
        "calw": "cal_tau_winkler",
        "calc": "cal_tau_coverage",
        "div": "cal_tau_divergent",
        "rhat": "cal_tau_max_rhat",
        "ess": "cal_tau_min_ess",
    }
    for prefix, name in columns.items():
        values = _vector(tcal, name)
        out |= {f"{prefix}_t{tau_label(tau)}": float(value) for tau, value in zip(TAUS, values, strict=True)}
    return out


def _check_paired(bags: dict[str, pd.DataFrame], key) -> None:
    reference = bags["tcal"]
    for cell, other in bags.items():
        if other is not None and (
            len(other) != len(reference) or not np.allclose(other["true_acc"], reference["true_acc"])
        ):
            raise ValueError(f"{CELL_METHODS.get(cell, cell)} and {TCAL_METHOD} do not have the same bags on {key}")


def pair_rows(
    frames: dict[str, pd.DataFrame], tempered: dict[float, pd.DataFrame], taus_e53: dict, acc_name: str
) -> list[dict]:
    """The rows of :func:`pair_table` of one metric; ``taus_e53``: (dataset, classifier) -> $\\tau_{\\text{cal}}$ of
    E5.3."""
    groups = {cell: dict(list(df.groupby(KEYS, sort=False))) for cell, df in frames.items()}
    tempered_groups = {
        tau: dict(list(df.groupby(KEYS, sort=False))) if not df.empty else {} for tau, df in tempered.items()
    }
    rows = []
    for key, first in groups["tcal"].items():
        bags = {cell: tempering.bag_frame(groups[cell][key], acc_name) for cell in CELL_METHODS}
        tau_e53 = float(taus_e53.get(key, np.nan))
        e53 = tempered_groups.get(tau_e53, {}).get(key) if np.isfinite(tau_e53) else None
        bags["e53"] = tempering.bag_frame(e53, acc_name) if e53 is not None else None
        _check_paired(bags, key)
        row = {
            "acc_name": acc_name,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
            "tau_cal_e53": tau_e53,
        }
        for cell in CELLS:
            for score in SCORES:
                row[f"{cell}_{score}"] = np.nan if bags[cell] is None else float(bags[cell][score].mean())
        for c, cell in enumerate(COMPARED):
            for s, score in enumerate(TESTED):
                name = f"{score}_vs_{cell}"
                diff = np.array([]) if bags[cell] is None else (bags["tcal"][score] - bags[cell][score]).to_numpy()
                diff = diff[np.isfinite(diff)]
                if len(diff) == 0:
                    mean = low = high = p_value = np.nan
                else:
                    rng = pair_rng(*key, seed=SEED + 1000 * s + 100 * c)
                    low, high, p_value = pushforward.bootstrap_mean_test(diff, rng)
                    mean = float(diff.mean())
                row |= {f"{name}_diff": mean, f"{name}_ci_low": low, f"{name}_ci_high": high, f"{name}_p": p_value}
        row["share_divergent"] = float(bags["tcal"]["divergent"].mean())
        row |= selection_columns(first, groups["cal"][key])
        rows.append(row)
    return rows


def pair_table(acc_names=None, root: str | None = None) -> pd.DataFrame:
    """One row per metric and pair where ``stored``, ``cal`` and ``tcal`` are stored, ordered as the tables of E3.1."""
    taus = e53_taus(root)
    taus_e53 = {(row.dataset, row.classifier): row.tau_cal_e53 for row in taus.itertuples(index=False)}
    tables = []
    for acc_name in acc_names or get_acc_names():
        frames, tempered = load_cells(acc_name, root)
        if not frames:
            continue
        table = pd.DataFrame(pair_rows(frames, tempered, taus_e53, acc_name))
        for cell in COMPARED:
            for score in TESTED:
                table[f"{score}_vs_{cell}_p_holm"] = pushforward.holm(table[f"{score}_vs_{cell}_p"].to_numpy())
        tables.append(ordered(table))
    if not tables:
        raise RuntimeError(f"No pair of main has stored results for all of {list(CELL_METHODS.values())}.")
    return pd.concat(tables, ignore_index=True)


# --------------------------------------------------------------------------------------------------- summaries


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric, group of $K$ and cell: the number of pairs where the cell is available, the means over them of its
    scores, and for ``tcal`` against each other cell the pairs significantly (Holm) lower and higher and the Wilcoxon
    $p$-values, for AE and the Winkler score."""
    rows = []
    for acc_name, part in table.groupby("acc_name", sort=False):
        for k_group, sub in tempering.k_groups(part):
            for cell in CELLS:
                available = sub[f"{cell}_winkler"].notna()
                row = {"acc_name": acc_name, "K": k_group, "cell": cell, "pairs": int(available.sum())}
                row |= {score: float(sub.loc[available, f"{cell}_{score}"].mean()) for score in SCORES}
                if cell == "tcal":
                    for other in COMPARED:
                        for score in TESTED:
                            name = f"{score}_vs_{other}"
                            significant = sub[f"{name}_p_holm"] < TEST_LEVEL
                            lower = sub[f"{name}_diff"] < 0
                            row[f"{name}_lower"] = int((lower & significant).sum())
                            row[f"{name}_higher"] = int((~lower & significant).sum())
                            row[f"{name}_wilcoxon_p"] = two_sided_wilcoxon(
                                sub[f"tcal_{score}"], sub[f"{other}_{score}"]
                            )
                rows.append(row)
    return pd.DataFrame(rows)


def _selection_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The rows of one metric (the selections do not depend on it)."""
    return table.loc[table["acc_name"] == table["acc_name"].iloc[0]]


def crosstab(table: pd.DataFrame) -> pd.DataFrame:
    """Counts of pairs: $\\tau_{\\text{cal}}$ at $b_{\\text{cal}}$ (rows) against $\\tau_{\\text{cal}}$ at $b=0.1$ of
    E5.3 (columns, ``NaN`` where E5.3 is not stored)."""
    part = _selection_rows(table)
    columns = [*TAUS, np.nan]
    out = pd.DataFrame(0, index=pd.Index(TAUS, name="tau_cal"), columns=pd.Index(columns, name="tau_cal_e53"))
    for tau, tau_e53 in zip(part["tau_cal"], part["tau_cal_e53"], strict=True):
        column = tau_e53 if np.isfinite(tau_e53) else np.nan
        match = [c for c in columns if (np.isnan(c) and np.isnan(column)) or c == column]
        out.loc[tau, match[0]] += 1
    return out


def nuts_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per $\\tau$ of the grid, over the pairs: the calibration runs with a divergent transition (and their share), the
    largest split-$\\hat R$, the smallest effective sample size, and the pairs whose $\\tau_{\\text{cal}}$ is $\\tau$."""
    part = _selection_rows(table)
    runs = len(part) * CAL_BAGS
    rows = []
    for tau in TAUS:
        t = tau_label(tau)
        divergent = float(part[f"div_t{t}"].sum())
        rows.append(
            {
                "tau": tau,
                "runs": runs,
                "runs_divergent": divergent,
                "share_divergent": divergent / runs if runs else np.nan,
                "max_rhat": float(part[f"rhat_t{t}"].max()),
                "min_ess": float(part[f"ess_t{t}"].min()),
                "pairs_selected": int(np.sum(part["tau_cal"] == tau)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX


def _fmt(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _tau_text(value: float) -> str:
    return "--" if not np.isfinite(value) else f"{value:g}"


def _summary_styler(summary: pd.DataFrame):
    """One row per group of $K$ and cell: the four scores; for ``tcal``, against each other cell, the pairs lower and
    higher and the Wilcoxon $p$-value of the Winkler score and of AE."""
    columns = {(tempering.SCORE_LABELS[score], ""): summary[score].map(_fmt).to_numpy() for score in SCORES}
    for other in COMPARED:
        for score in TESTED:
            name = f"{score}_vs_{other}"
            label = f"vs {SHORT_LABELS[other]}: $\\Delta${tempering.SCORE_LABELS[score]}"
            is_tcal = summary["cell"] == "tcal"
            columns[(label, "lower")] = np.where(is_tcal, summary.get(f"{name}_lower", np.nan), np.nan)
            columns[(label, "higher")] = np.where(is_tcal, summary.get(f"{name}_higher", np.nan), np.nan)
            columns[(label, "$p$")] = np.where(is_tcal, summary.get(f"{name}_wilcoxon_p", np.nan), np.nan)
    out = pd.DataFrame(columns)
    for column in out.columns:
        if column[1] in ("lower", "higher"):
            out[column] = out[column].map(lambda v: "--" if pd.isna(v) else f"{int(v)}")
        elif column[1] == "$p$":
            out[column] = out[column].map(lambda v: "--" if pd.isna(v) else _fmt(float(v)))
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["K"].map(tempering.K_LABELS).to_numpy(),
            summary["pairs"].astype(int).to_numpy(),
            summary["cell"].map(CELL_LABELS).to_numpy(),
        ],
        names=["$K$", "pairs", "cell"],
    )
    return out.style


def _pairs_styler(table: pd.DataFrame):
    """One row per pair: $b_{\\text{cal}}$, $\\tau_{\\text{cal}}$ at $b=0.1$ (E5.3) and at $b_{\\text{cal}}$, and the
    AE, Winkler score and coverage of the four cells, each coloured per row from the worst to the best."""
    columns = {
        ("selection", "$b_{\\mathrm{cal}}$"): table["b_cal"].map(_fmt).to_numpy(),
        ("selection", "$\\tau_{\\mathrm{cal}}$ ($b{=}0.1$)"): table["tau_cal_e53"].map(_tau_text).to_numpy(),
        ("selection", "$\\tau_{\\mathrm{cal}}$ ($b_{\\mathrm{cal}}$)"): table["tau_cal"].map(_tau_text).to_numpy(),
    }
    for score in PAIR_SCORES:
        for cell in CELLS:
            columns[(tempering.SCORE_LABELS[score], CELL_LABELS[cell])] = table[f"{cell}_{score}"].to_numpy(dtype=float)
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = tempering.score_styles(out, {score: tempering.SCORE_LABELS[score] for score in PAIR_SCORES})
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


def _crosstab_styler(cross: pd.DataFrame):
    out = cross.copy()
    out.index = [f"${_tau_text(t)}$" for t in out.index]
    out.columns = [("--" if np.isnan(c) else f"${_tau_text(c)}$") for c in out.columns]
    out.index.name = "$\\tau_{\\mathrm{cal}}$ at $b_{\\mathrm{cal}}$"
    out.columns.name = "$\\tau_{\\mathrm{cal}}$ at $b{=}0.1$ (E5.3)"
    return out.style


def caption(acc_name: str, name: str) -> str:
    common = (
        "E7.2, ablation of the bandwidth and the temperature of BayesKDEy selected per pair on calibration bags drawn "
        f"from half of $V$, on \\texttt{{main}} ({tempering.ACC_LABELS[acc_name]}); all cells are \\bcapkdey{{}}: "
        "$b=0.1$ or $b_{\\mathrm{cal}}$ (E7.2a), $\\tau=1$ or $\\tau_{\\mathrm{cal}}$ (lowest mean Winkler score on "
        "vanilla accuracy over the calibration bags; at $b=0.1$ from E5.3, at $b_{\\mathrm{cal}}$ from E7.2b). "
    )
    if name == "summary":
        return common + (
            "Scores against the realised table, means over the 100 bags and over the pairs where the cell is stored; "
            "lower, higher: pairs where the mean AE or Winkler score of $(b_{\\mathrm{cal}},\\tau_{\\mathrm{cal}})$ is "
            "significantly lower or higher than that of the cell (paired bootstrap, Holm over the pairs, $p<0.05$); "
            "$p$: two-sided Wilcoxon signed-rank test over the per-pair means."
        )
    if name == "crosstab":
        return common + "Pairs by $\\tau_{\\mathrm{cal}}$ at $b_{\\mathrm{cal}}$ and at $b=0.1$ (--: E5.3 not stored)."
    return common + (
        "Selected $b$ and $\\tau$ (label-free on the target); AE, Winkler score and coverage against the realised table, "
        "means over the 100 bags, coloured per row from the worst (red) to the best (green, in bold; coverage: closest "
        "to $0.95$)."
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and getattr(pargs, "table_dir", None) is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str = "pairs") -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_{name}.csv")


def _stem(acc_name: str, name: str) -> str:
    return f"{TABLE_NAME}_{name}_{acc_name}"


def _write(out_dir: str, stem: str, styler, caption_text: str, label: str) -> None:
    tabular_dir = os.path.join(out_dir, "tabular")
    with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
        f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
    with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
        f.write(eb_prior._latex_longtable(styler, caption_text, label))
    with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
        f.write(tempering._latex_document(f"tabular/{stem}_long.tex"))


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    summary = summary_table(table)
    cross = crosstab(table)
    nuts = nuts_table(table)
    out_dir = table_dir(pargs)
    os.makedirs(os.path.join(out_dir, "tabular"), exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    summary.to_csv(csv_path(out_dir, "summary"), index=False)
    cross.to_csv(csv_path(out_dir, "crosstab"))
    nuts.to_csv(csv_path(out_dir, "nuts"), index=False)

    for acc_name in table["acc_name"].unique():
        stylers = {
            "summary": _summary_styler(summary.loc[summary["acc_name"] == acc_name].reset_index(drop=True)),
            "pairs": _pairs_styler(table.loc[table["acc_name"] == acc_name].reset_index(drop=True)),
        }
        for name, styler in stylers.items():
            _write(
                out_dir, _stem(acc_name, name), styler, caption(acc_name, name), f"tab:{TABLE_NAME}-{name}-{acc_name}"
            )
    _write(
        out_dir,
        f"{TABLE_NAME}_crosstab",
        _crosstab_styler(cross),
        caption(SELECTION_METRIC, "crosstab"),
        f"tab:{TABLE_NAME}-crosstab",
    )

    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all", ["acc_name", "cell", "pairs", *SCORES]].to_string(index=False))
        print(cross.to_string())
        print(nuts.to_string(index=False))
        first = _selection_rows(table)
        print(f"pairs: {len(first)}; b_cal differs from the cal method on {int((~first['b_cal_same']).sum())}")
    print(f"rows: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    stems = [_stem(acc_name, name) for acc_name in pargs.metrics or get_acc_names() for name in ("summary", "pairs")]
    for stem in [*stems, f"{TABLE_NAME}_crosstab"]:
        tex_filename = f"{stem}_doc.tex"
        if not os.path.exists(os.path.join(out_dir, tex_filename)):
            continue
        # longtable needs a second pass to settle the column widths
        for _ in range(2):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                cwd=out_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Ablation of the bandwidth and temperature selected on calibration bags (E7.2)."
    )
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=None, help="Default: all three")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSVs, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
