"""Per-class leave-one-out bandwidths of BayesKDEy on the pairs of ``main`` (work-plan card WP-25, E7.1 of
``docs/ext7_kdey_bandwidth.md``).

``BayCAP-KDEy-mlp-real-loo`` and ``SQS-B-loo`` (``experiments/methods.py:LOO_VARIANTS``) are ``BayCAP-KDEy-mlp-real``
and ``SQS-B`` with BayesKDEy at ``bandwidth="loo"``: each class gets the bandwidth of ``methods/bayes_kdey.py:LOO_GRID``
that maximises the leave-one-out log-likelihood of its density on the cross-fitted validation outputs of the MLP, with
the densities normalised on the simplex; everything else, seeds included, is the stored method with $b=0.1$. The four
methods target the realised table of the target bag and are scored against it (``true_cts``); the selection uses no
target label. All three metrics of ``main``; the pairs where the four methods have stored results (so it also works on
the pilot).

Per metric and pair (``pair_table``, one row per metric and pair, every column in the pairs CSV):

- AE, coverage, amplitude and Winkler score of the four methods, means over the bags;
- for each LOO method against its stored method, the mean per-bag Winkler difference (LOO minus stored), its paired
  bootstrap interval and p-value (``tables/pushforward.py:bootstrap_mean_test``, $10{,}000$ resamples), Holm-adjusted
  over the pairs, per metric and comparison;
- label-free, from the summaries stored by ``BayCAP-KDEy-mlp-real-loo`` (the same selection as ``SQS-B-loo``): the
  median, smallest and largest $\\hat b_j$ over the classes, the median of $\\log_{10}(\\hat b_j/0.1)$, the number of
  classes at the lower and at the upper edge of the grid, and the mean over the classes of the gain of the
  leave-one-out log-likelihood per instance at $\\hat b_j$ over $b=0.1$; the share of bags with divergent transitions
  of each LOO method.

Per class (``class_table``, the classes CSV): $n_j$, $\\hat b_j$, the edge flag and the leave-one-out log-likelihood
per instance at $\\hat b_j$ and at $0.1$, read by ``plots/kdey_loo.py``. ``summary_table``: per metric, the means over
the pairs of the four methods' scores, and for each LOO method the pairs where its Winkler score is significantly
(Holm) lower and higher than the stored one and the two-sided Wilcoxon signed-rank $p$-value over the per-pair means.

Outputs, under ``<OUT_DIR>/uncawe/tables/kdey_loo/``: the CSVs and, per metric, a summary and a per-pair table, each as
a LaTeX tabular fragment, a longtable and a standalone document. Tables follow the ``cap-ml`` layout (true classes on
rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncawe import env
from uncawe.experiments.methods import LOO_VARIANTS
from uncawe.experiments.metrics import get_acc_names
from uncawe.methods.bayes_kdey import LOO_DEFAULT_BANDWIDTH
from uncawe.tables import baycap_h, eb_prior, pushforward, tempering
from uncawe.tables.displacement import ordered
from uncawe.tables.estimand_gap import SEED, booktabs_clines, pair_rng

TABLE_NAME = "kdey_loo"
KEYS = ["dataset", "classifier"]
SCORES = ("ae", "coverage", "amplitude", "winkler")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
# LOO method -> stored method, and the four methods in the order of the tables
COMPARISONS = dict(LOO_VARIANTS)
METHODS = [m for pair in COMPARISONS.items() for m in reversed(pair)]
# the LOO method whose stored summaries give the selected bandwidths (SQS-B-loo stores the same selection)
SELECTION_METHOD = next(iter(COMPARISONS))
SHORT_LABELS = {
    "BayCAP-KDEy-mlp-real": "KDEy",
    "BayCAP-KDEy-mlp-real-loo": "KDEy loo",
    "SQS-B": "SQS-B",
    "SQS-B-loo": "SQS-B loo",
}


def load_all(acc_name: str) -> dict[str, pd.DataFrame]:
    """The stored ``main`` rows of the four methods for ``acc_name``, restricted to the pairs stored for all of them;
    empty if there is none."""
    frames = {method: tempering.load_method(method, acc_name) for method in METHODS}
    if any(df.empty for df in frames.values()):
        return {}
    pairs = eb_prior.common_pairs(frames)
    if not pairs:
        return {}
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    return {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in frames.items()
    }


def _check_paired(a: pd.DataFrame, b: pd.DataFrame, key, names) -> None:
    if len(a) != len(b) or not np.allclose(a["true_acc"], b["true_acc"]):
        raise ValueError(f"{names[0]} and {names[1]} do not have the same bags on {key}")


def _vectors(df: pd.DataFrame, name: str) -> np.ndarray:
    """The vector summary ``name`` stored with the first bag (the selection is fixed by the fit, so the same for every
    bag)."""
    return np.asarray(df[name].iloc[0], dtype=float)


def selection_columns(df: pd.DataFrame) -> dict:
    """The label-free summary of the selected bandwidths of one pair, from the rows of ``SELECTION_METHOD``."""
    bandwidths = _vectors(df, "bandwidths")
    edges = _vectors(df, "bandwidth_edge")
    gain = _vectors(df, "loo_selected") - _vectors(df, "loo_default")
    return {
        "n_val": int(_vectors(df, "class_sizes").sum()),
        "b_median": float(np.median(bandwidths)),
        "b_min": float(bandwidths.min()),
        "b_max": float(bandwidths.max()),
        "log_b_ratio_median": float(np.median(np.log10(bandwidths / LOO_DEFAULT_BANDWIDTH))),
        "n_lower_edge": int(np.sum(edges < 0)),
        "n_upper_edge": int(np.sum(edges > 0)),
        "loo_gain_mean": float(np.nanmean(gain)) if np.isfinite(gain).any() else np.nan,
    }


def pair_rows(frames: dict[str, pd.DataFrame], acc_name: str) -> list[dict]:
    rows = []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in frames.items()}
    for key, first in groups[SELECTION_METHOD].items():
        bags = {method: tempering.bag_frame(groups[method][key], acc_name) for method in METHODS}
        row = {
            "acc_name": acc_name,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
        for index, (loo, stored) in enumerate(COMPARISONS.items()):
            _check_paired(bags[loo], bags[stored], key, (loo, stored))
            diff = (bags[loo]["winkler"] - bags[stored]["winkler"]).to_numpy()
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0:
                low = high = p_value = mean = np.nan
            else:
                low, high, p_value = pushforward.bootstrap_mean_test(diff, pair_rng(*key, seed=SEED + 1000 * index))
                mean = float(diff.mean())
            row |= {f"{loo}_diff_mean": mean, f"{loo}_ci_low": low, f"{loo}_ci_high": high, f"{loo}_p": p_value}
            row[f"{loo}_share_divergent"] = float(bags[loo]["divergent"].mean())
        row |= selection_columns(first)
        rows.append(row)
    return rows


def pair_table(acc_names=None) -> pd.DataFrame:
    """One row per metric and pair with stored results of the four methods, ordered as the tables of E3.1."""
    tables = []
    for acc_name in acc_names or get_acc_names():
        frames = load_all(acc_name)
        if not frames:
            continue
        table = pd.DataFrame(pair_rows(frames, acc_name))
        for loo in COMPARISONS:
            table[f"{loo}_p_holm"] = pushforward.holm(table[f"{loo}_p"].to_numpy())
        tables.append(ordered(table))
    if not tables:
        raise RuntimeError(f"No pair of main has stored results for all of {METHODS}.")
    return pd.concat(tables, ignore_index=True)


def class_table(acc_name: str = "vanilla_accuracy") -> pd.DataFrame:
    """One row per pair and class: $n_j$, $\\hat b_j$, the edge flag and the leave-one-out log-likelihood per instance
    at $\\hat b_j$ and at $b=0.1$ (from the rows of ``SELECTION_METHOD`` for ``acc_name``; the selection does not depend
    on the metric)."""
    frames = load_all(acc_name)
    if not frames:
        raise RuntimeError(f"No pair of main has stored results for all of {METHODS}.")
    rows = []
    for key, df in frames[SELECTION_METHOD].groupby(KEYS, sort=False):
        columns = {name: _vectors(df, name) for name in ("bandwidths", "class_sizes", "bandwidth_edge")}
        loo = {name: _vectors(df, name) for name in ("loo_selected", "loo_default")}
        for j in range(len(columns["bandwidths"])):
            rows.append(
                {
                    "dataset": key[0],
                    "classifier": key[1],
                    "n_classes": int(df["n_classes"].iloc[0]),
                    "class_index": j,
                    "n_j": int(columns["class_sizes"][j]),
                    "bandwidth": float(columns["bandwidths"][j]),
                    "edge": int(columns["bandwidth_edge"][j]),
                    "loo_selected": float(loo["loo_selected"][j]),
                    "loo_default": float(loo["loo_default"][j]),
                }
            )
    return pd.DataFrame(rows)


def two_sided_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """$p$-value of the two-sided paired Wilcoxon signed-rank test; NaN with fewer than two finite nonzero
    differences."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    if np.count_nonzero(diff) < 2:
        return np.nan
    return float(wilcoxon(diff).pvalue)


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric and group of $K$, one row per method: the means over the pairs of its scores and, for a LOO method,
    the pairs where its Winkler score is significantly (Holm) lower and higher than the stored one, and the Wilcoxon
    $p$-value over the per-pair means."""
    rows = []
    for acc_name, part in table.groupby("acc_name", sort=False):
        for k_group, sub in tempering.k_groups(part):
            for method in METHODS:
                row = {"acc_name": acc_name, "K": k_group, "pairs": len(sub), "method": method}
                row |= {score: float(sub[f"{method}_{score}"].mean()) for score in SCORES}
                if method in COMPARISONS:
                    significant = sub[f"{method}_p_holm"] < TEST_LEVEL
                    lower = sub[f"{method}_diff_mean"] < 0
                    row["pairs_lower"] = int((lower & significant).sum())
                    row["pairs_higher"] = int((~lower & significant).sum())
                    row["wilcoxon_p"] = two_sided_wilcoxon(
                        sub[f"{method}_winkler"], sub[f"{COMPARISONS[method]}_winkler"]
                    )
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

SCORE_LABELS = {
    "ae": "AE",
    "coverage": "$\\mathcal{C}$",
    "amplitude": "$\\mathcal{A}$",
    "winkler": "$\\mathcal{W}$",
}


def _fmt(value: float, digits: int = 3) -> str:
    return "--" if not np.isfinite(value) else f"{value:.{digits}f}"


def _summary_styler(summary: pd.DataFrame):
    """One row per group of $K$ and method: the four scores, and for the LOO methods the pairs lower/higher and the
    Wilcoxon $p$-value."""
    out = pd.DataFrame(
        {
            **{SCORE_LABELS[score]: summary[score].map(_fmt).to_numpy() for score in SCORES},
            "lower": summary["pairs_lower"].map(lambda v: "--" if pd.isna(v) else f"{int(v)}").to_numpy(),
            "higher": summary["pairs_higher"].map(lambda v: "--" if pd.isna(v) else f"{int(v)}").to_numpy(),
            "$p$": summary["wilcoxon_p"].map(_fmt).to_numpy(),
        }
    )
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["K"].map(tempering.K_LABELS).to_numpy(),
            summary["pairs"].astype(int).to_numpy(),
            summary["method"].map(SHORT_LABELS).to_numpy(),
        ],
        names=["$K$", "pairs", "method"],
    )
    return out.style


def _pairs_styler(table: pd.DataFrame):
    """One row per pair: the Winkler score and the coverage of the four methods (coloured per row, as in E3.1), the
    Winkler difference of each LOO method (bold where Holm $p<0.05$) and the median $\\hat b_j$."""
    columns = {}
    for score in ("winkler", "coverage"):
        for method in METHODS:
            columns[(SCORE_LABELS[score], SHORT_LABELS[method])] = table[f"{method}_{score}"].to_numpy(dtype=float)
    for loo in COMPARISONS:
        columns[("$\\Delta\\mathcal{W}$", SHORT_LABELS[loo])] = [
            "--" if not np.isfinite(d) else (f"\\textbf{{{d:.3f}}}" if p < TEST_LEVEL else f"{d:.3f}")
            for d, p in zip(table[f"{loo}_diff_mean"], table[f"{loo}_p_holm"])
        ]
    columns[("$\\hat b_j$", "median")] = table["b_median"].to_numpy(dtype=float)
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = tempering.score_styles(out, {score: SCORE_LABELS[score] for score in ("winkler", "coverage")})
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


def caption(acc_name: str, name: str) -> str:
    common = (
        f"E7.1, per-class leave-one-out bandwidths of BayesKDEy on \\texttt{{main}} ({tempering.ACC_LABELS[acc_name]}); "
        "KDEy: \\bcap{}-KDEy-mlp-real ($b=0.1$); SQS-B: \\sqsb{} ($b=0.1$); loo: the same method with one bandwidth per class "
        "selected by leave-one-out likelihood on $V$. Scores against the realised table, means over the 100 bags. "
    )
    if name == "summary":
        return common + (
            "Means over the pairs; lower, higher: pairs where the mean Winkler score of the loo method is significantly "
            "lower or higher than that of the stored method (paired bootstrap, Holm over the pairs, $p<0.05$); $p$: "
            "two-sided Wilcoxon signed-rank test over the per-pair means."
        )
    return common + (
        "Each score coloured per row from the worst (red) to the best (green, in bold; coverage: closest to $0.95$); "
        "$\\Delta\\mathcal{W}$: mean per-bag Winkler difference, loo minus stored (bold: Holm $p<0.05$); median of the "
        "selected bandwidths over the classes."
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str = "pairs") -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_{name}.csv")


def _stem(acc_name: str, name: str) -> str:
    return f"{TABLE_NAME}_{name}_{acc_name}"


TABLES = ("summary", "pairs")


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    summary = summary_table(table)
    classes = class_table()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    summary.to_csv(csv_path(out_dir, "summary"), index=False)
    classes.to_csv(csv_path(out_dir, "classes"), index=False)

    for acc_name in table["acc_name"].unique():
        stylers = {
            "summary": _summary_styler(summary.loc[summary["acc_name"] == acc_name].reset_index(drop=True)),
            "pairs": _pairs_styler(table.loc[table["acc_name"] == acc_name].reset_index(drop=True)),
        }
        for name, styler in stylers.items():
            stem = _stem(acc_name, name)
            with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
                f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
            with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
                f.write(
                    eb_prior._latex_longtable(styler, caption(acc_name, name), f"tab:{TABLE_NAME}-{name}-{acc_name}")
                )
            with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
                f.write(tempering._latex_document(f"tabular/{stem}_long.tex"))

    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
        edges = classes["edge"]
        print(
            f"classes: {len(classes)}; at the lower edge: {int((edges < 0).sum())}; at the upper edge: "
            f"{int((edges > 0).sum())}; median bandwidth: {classes['bandwidth'].median():.3f}"
        )
    print(f"rows: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for acc_name in pargs.metrics or get_acc_names():
        for name in TABLES:
            tex_filename = f"{_stem(acc_name, name)}_doc.tex"
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
    parser = ArgumentParser(description="Per-class leave-one-out bandwidths of BayesKDEy on the pairs of main (E7.1).")
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
