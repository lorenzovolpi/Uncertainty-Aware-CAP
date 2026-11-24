"""BayCAP-KDEy-h-inst with a bandwidth selected for calibrated class probabilities on the pairs of ``main`` (work-plan
card WP-33, E3.5 of ``docs/ext3_posterior_propagation.md``, §T.16).

``BayCAP-KDEy-h-inst-cb`` (``methods/baycap_kdey.py:BayCAPKDEy``, ``bandwidth="logloss"``,
``estimand="realised-instance"``) is ``BayCAP-KDEy-h-inst`` with one bandwidth $b^\\star$ shared by the classes,
selected per pair from $V$ alone by the class-balanced leave-one-out log-loss $\\mathcal L(b)$ of the kernel class
probabilities over ``methods/bayes_kdey.py:LOO_GRID``, and used in both the likelihood of the bag and the per-instance
weights. It is compared bag by bag with ``BayCAP-KDEy-h-inst`` ($b=0.1$, E3.4), ``BayCAP-KDEy-h-real`` (the cell
pushforward at $b=0.1$, E3.1) and ``BayesCAP-Hf-real`` (the reference for coverage). All four target the realised
table of the target bag and are scored against it (``true_cts``). Vanilla accuracy, $|U|=1000$, the pairs where the
four methods have stored results (so it also works on the pilot).

Per pair (``pair_table``, one row per pair, every column in the CSV):

- AE, coverage, amplitude and Winkler score of the four methods, means over the bags; the bags where
  ``BayesCAP-Hf-real`` produced no draws (``nuts_failed``) are left out of its scores; the mean signed error
  (point estimate minus realised accuracy) of the three BayCAP-KDEy methods (P24);
- the selection, label-free with respect to the target and the same for every bag of the pair
  (``methods/bayes_kdey.py:LOGLOSS_SUMMARIES``): $b^\\star$, $\\log_{10}(b^\\star/0.1)$, the edge flag, the number of
  classes in $\\mathcal L$, $\\mathcal L(0.1)$ and $\\mathcal L(b^\\star)$, and the ECE of the leave-one-out confidences
  on $V$ at $0.1$ and at $b^\\star$ (P23);
- label-free, the amplitude ratio of ``BayCAP-KDEy-h-inst-cb`` to ``BayCAP-KDEy-h-inst`` (median over the bags of the
  per-bag ratio), the mean over the bags of the difference of their point estimates (the centre, P24), and, for both
  methods, the means over the bags of $\\sigma_p$ (``acc_sd``) and of the summaries of E3.4 ($\\bar\\sigma_w$,
  $\\bar{\\tilde\\sigma}_g$, the within-class term, $\\operatorname{sd}(m_w)$, $\\operatorname{sd}(m)$,
  $\\bar m_w-\\bar m$), to see whether a wider interval comes from the luck of the bag or from the posterior of $\\pi$
  (P25); the NUTS diagnostics of ``BayCAP-KDEy-h-inst-cb``;
- two paired bootstrap tests of the mean per-bag Winkler difference (``tables/pushforward.py:bootstrap_mean_test``,
  $10{,}000$ resamples), each Holm-adjusted over the pairs on its own: ``BayCAP-KDEy-h-inst-cb`` minus
  ``BayCAP-KDEy-h-inst`` and ``BayCAP-KDEy-h-inst-cb`` minus ``BayCAP-KDEy-h-real``.

The curves $\\mathcal L(b)$ go to a second CSV, one row per pair and bandwidth, which ``plots/baycap_kdey_cb.py`` reads
with the per-pair CSV. The scores, the signed errors and the tests use the true tables, so they are analyses; the
selection, the ratios, the centre difference, the summaries and the diagnostics use no target labels. Outputs, under
``<OUT_DIR>/uncawe/tables/baycap_kdey_cb/``: the two CSVs and three tables (comparison; selection and widths; tests),
each as a LaTeX tabular fragment, a longtable and a standalone landscape document; the comparison table of the
Representation of E3.5 is split into these three, as that of E3.4, since it does not fit one page. Tables follow the
``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import INSTANCE_VARIANTS, LOGLOSS_VARIANTS
from uncawe.methods.bayes_kdey import LOO_DEFAULT_BANDWIDTH, LOO_GRID
from uncawe.tables import baycap_fixed, baycap_h, baycap_kdey_hf, eb_prior, pushforward
from uncawe.tables.baycap_kdey_hf import bag_frame
from uncawe.tables.displacement import load_validation_counts, ordered
from uncawe.tables.estimand_gap import ACC_NAME, DOMAIN, SEED, booktabs_clines, pair_rng

TABLE_NAME = "baycap_kdey_cb"
KEYS = eb_prior.KEYS
CB = "BayCAP-KDEy-h-inst-cb"
INST = LOGLOSS_VARIANTS[CB]
H_REAL = INSTANCE_VARIANTS[INST]
HF = baycap_fixed.HF
# the methods, in the order of the tables
METHODS = [CB, INST, H_REAL, HF]
# the methods whose signed error is reported (P24)
BIAS_METHODS = (CB, INST, H_REAL)
SCORES = ("ae", "coverage", "amplitude", "winkler")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
# the tests, name -> (method, reference): the per-bag difference is method minus reference
TESTS = {"vs_inst": (CB, INST), "vs_h_real": (CB, H_REAL)}
# the summaries of the selection stored by BayCAP-KDEy-h-inst-cb (methods/bayes_kdey.py:LOGLOSS_SUMMARIES, without the
# curve), the same for every bag of a pair -> column of the table
SELECTION = {
    "logloss_bandwidth": "bandwidth",
    "logloss_edge": "edge",
    "logloss_classes": "n_loss_classes",
    "logloss_default": "logloss_default",
    "logloss_selected": "logloss_selected",
    "loo_ece_default": "ece_default",
    "loo_ece_selected": "ece_selected",
}
# the label-free width summaries averaged over the bags, for both BayCAP-KDEy-h-inst-cb and BayCAP-KDEy-h-inst: sigma_p
# (FIXED_SUMMARIES) and the summaries of E3.4 (methods/baycap_kdey.py:INSTANCE_SUMMARIES)
SUMMARIES = ("acc_sd", "sigma_w", "sigma_g_tilde", "within_class", "m_w_sd", "m_sd", "centre_shift")
# the methods whose width summaries are reported
SUMMARY_METHODS = (CB, INST)


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def _constant(df: pd.DataFrame, column: str, key: tuple[str, str]) -> float:
    """The value of ``column`` on the bags of one pair, which the fit fixes; ``ValueError`` if it varies."""
    values = _numeric(df, column)
    if not np.allclose(values, values[0], rtol=1e-12, atol=0.0, equal_nan=True):
        raise ValueError(f"{column} of {CB} differs between the bags of {key}")
    return float(values[0])


def selection_columns(df: pd.DataFrame, key: tuple[str, str]) -> dict:
    """The selection of $b^\\star$ on one pair (the stored summaries of ``BayCAP-KDEy-h-inst-cb``)."""
    out = {column: _constant(df, name, key) for name, column in SELECTION.items()}
    out["log10_ratio"] = float(np.log10(out["bandwidth"] / LOO_DEFAULT_BANDWIDTH))
    return {f"{CB}_{name}": value for name, value in out.items()}


def curve_rows(df: pd.DataFrame, key: tuple[str, str], n_classes: int) -> list[dict]:
    """The curve $\\mathcal L(b)$ of one pair over ``LOO_GRID``, one row per bandwidth; ``ValueError`` if the bags store
    different curves."""
    curves = np.stack([np.asarray(curve, dtype=float) for curve in df["logloss_curve"]])
    if curves.shape[1] != len(LOO_GRID):
        raise ValueError(f"the curve of {key} has {curves.shape[1]} values; LOO_GRID has {len(LOO_GRID)}")
    if not np.allclose(curves, curves[0], rtol=1e-12, atol=0.0):
        raise ValueError(f"the curve of {CB} differs between the bags of {key}")
    return [
        {"dataset": key[0], "classifier": key[1], "n_classes": n_classes, "bandwidth": b, "logloss": float(value)}
        for b, value in zip(LOO_GRID, curves[0])
    ]


def width_columns(frames: dict[str, pd.DataFrame], bags: dict[str, pd.DataFrame]) -> dict:
    """The label-free comparison of ``BayCAP-KDEy-h-inst-cb`` and ``BayCAP-KDEy-h-inst`` on one pair (``frames``: the
    stored rows of each method, in the order of the bags)."""
    amplitude, reference = bags[CB]["amplitude"].to_numpy(), bags[INST]["amplitude"].to_numpy()
    ratio = np.divide(amplitude, reference, out=np.full_like(amplitude, np.nan), where=reference > 0)
    centre = _numeric(frames[CB], "estim_accs") - _numeric(frames[INST], "estim_accs")
    df = frames[CB]
    out = {
        f"{CB}_ratio_vs_inst": float(np.nanmedian(ratio)),
        f"{CB}_centre_vs_inst": float(np.nanmean(centre)),
        f"{CB}_share_divergent": float(np.mean(_numeric(df, "nuts_divergences") > 0)),
        f"{CB}_max_rhat": float(np.max(_numeric(df, "nuts_max_rhat"))),
        f"{CB}_min_ess": float(np.min(_numeric(df, "nuts_min_ess"))),
    }
    for method in SUMMARY_METHODS:
        for name in SUMMARIES:
            out[f"{method}_{name}"] = float(np.mean(_numeric(frames[method], name)))
    return out


def pair_tests(bags: dict[str, pd.DataFrame], key: tuple[str, str]) -> dict:
    """The two tests of one pair, without the Holm adjustment; the generator of each test depends only on the seed, the
    pair and the test."""
    row = {}
    for index, (name, (method, reference)) in enumerate(TESTS.items()):
        a, b = bags[method], bags[reference]
        baycap_fixed._check_paired(a, b, key, (method, reference))
        ok = ~(a["failed"].to_numpy() | b["failed"].to_numpy())
        diff = (a["winkler"] - b["winkler"]).to_numpy()[ok]
        if len(diff) == 0:
            low = high = p_value = mean = np.nan
        else:
            low, high, p_value = pushforward.bootstrap_mean_test(diff, pair_rng(*key, seed=SEED + 1000 * index))
            mean = float(diff.mean())
        row |= {f"{name}_diff_mean": mean, f"{name}_ci_low": low, f"{name}_ci_high": high, f"{name}_p": p_value}
    return row


def pair_tables(results: dict[str, pd.DataFrame], val_counts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per pair of ``results`` (method -> stored results, the same pairs and bags for every method), ordered as
    the tables of E3.1 with the validation counts ``val_counts`` (pair -> counts in the ``cap-ml`` layout), and the
    curves $\\mathcal L(b)$, one row per pair and bandwidth, in the same order of the pairs."""
    rows, curves = [], []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
    for key, first in groups[CB].items():
        frames = {method: groups[method][key] for method in METHODS}
        bags = {method: bag_frame(frames[method], method) for method in METHODS}
        for method in METHODS[1:]:
            baycap_fixed._check_paired(bags[CB], bags[method], key, (CB, method))
        n_classes = int(first["n_classes"].iloc[0])
        row = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": n_classes,
            "n_bags": len(first),
            "n_val": int(val_counts[key].sum()),
            "min_val_per_class": int(val_counts[key].sum(axis=1).min()),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
        for method in BIAS_METHODS:
            ok = ~bags[method]["failed"].to_numpy()
            signed = _numeric(frames[method], "estim_accs") - bags[method]["true_acc"].to_numpy()
            row[f"{method}_bias"] = float(np.mean(signed[ok])) if ok.any() else np.nan
        row[f"{HF}_share_failed"] = float(bags[HF]["failed"].mean())
        row |= selection_columns(first, key)
        row |= width_columns(frames, bags)
        row |= pair_tests(bags, key)
        rows.append(row)
        curves += curve_rows(first, key, n_classes)
    table = pd.DataFrame(rows)
    for name in TESTS:
        table[f"{name}_p_holm"] = pushforward.holm(table[f"{name}_p"].to_numpy())
    table = ordered(table)
    order = {(d, c): i for i, (d, c) in enumerate(zip(table["dataset"], table["classifier"]))}
    curve_table = pd.DataFrame(curves)
    rank = [order[(d, c)] for d, c in zip(curve_table["dataset"], curve_table["classifier"])]
    curve_table = curve_table.assign(_rank=rank).sort_values(["_rank", "bandwidth"], kind="stable")
    return table, curve_table.drop(columns="_rank").reset_index(drop=True)


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """The per-pair table of :func:`pair_tables`."""
    return pair_tables(results, val_counts)[0]


def load_all(acc_name: str = ACC_NAME) -> tuple[dict[str, pd.DataFrame], list]:
    """The stored results of ``METHODS`` on the pairs stored for all of them."""
    return baycap_fixed.load_all(acc_name, methods=METHODS)


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E3.5, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage, amplitude, Winkler score of the four methods and mean signed error of the three
      BayCAP-KDEy methods;
    - ``selection``: the pairs with $b^\\star$ above, at and below $0.1$ and at an edge of the grid, the median
      $b^\\star$, the pairs where the ECE at $b^\\star$ is lower than at $0.1$, and the median ECE at both (P23);
    - ``widths``: the median amplitude ratio to ``BayCAP-KDEy-h-inst``, the mean centre difference, and the medians of
      $\\bar\\sigma_w$, $\\operatorname{sd}(m_w)$ and $\\sigma_p$ of both methods (P24, P25);
    - ``tests``: per test, the pairs where ``BayCAP-KDEy-h-inst-cb`` is significantly (Holm) lower and higher than its
      reference.
    """
    out = {"scores": [], "selection": [], "widths": [], "tests": []}
    for k_group, sub in baycap_h._k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        scores = {f"{m}_{s}": float(sub[f"{m}_{s}"].mean()) for m in METHODS for s in SCORES}
        out["scores"].append(key | scores | {f"{m}_bias": float(sub[f"{m}_bias"].mean()) for m in BIAS_METHODS})
        bandwidth = sub[f"{CB}_bandwidth"]
        at_default = np.isclose(bandwidth, LOO_DEFAULT_BANDWIDTH)
        out["selection"].append(
            key
            | {
                "pairs_above": int(((bandwidth > LOO_DEFAULT_BANDWIDTH) & ~at_default).sum()),
                "pairs_at": int(at_default.sum()),
                "pairs_below": int(((bandwidth < LOO_DEFAULT_BANDWIDTH) & ~at_default).sum()),
                "pairs_edge": int((sub[f"{CB}_edge"] != 0).sum()),
                "bandwidth_median": float(bandwidth.median()),
                "pairs_ece_lower": int((sub[f"{CB}_ece_selected"] < sub[f"{CB}_ece_default"]).sum()),
                "ece_default_median": float(sub[f"{CB}_ece_default"].median()),
                "ece_selected_median": float(sub[f"{CB}_ece_selected"].median()),
            }
        )
        widths = {
            "ratio_vs_inst": float(sub[f"{CB}_ratio_vs_inst"].median()),
            "centre_vs_inst": float(sub[f"{CB}_centre_vs_inst"].mean()),
        }
        for method, label in ((CB, "cb"), (INST, "inst")):
            for name in ("sigma_w", "m_w_sd", "acc_sd"):
                widths[f"{name}_{label}"] = float(sub[f"{method}_{name}"].median())
        out["widths"].append(key | widths)
        tests = dict(key)
        for name in TESTS:
            significant = sub[f"{name}_p_holm"] < TEST_LEVEL
            lower = sub[f"{name}_diff_mean"] < 0
            tests[f"{name}_lower"] = int((lower & significant).sum())
            tests[f"{name}_higher"] = int((~lower & significant).sum())
        out["tests"].append(tests)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


SHORT_LABELS = {CB: "KDEy h inst cb", INST: "KDEy h inst", H_REAL: "KDEy h cell", HF: "Hf"}


def _rotated(label: str) -> str:
    return baycap_kdey_hf._rotated(label)


def _comparison_styler(table: pd.DataFrame):
    """Columns grouped by score (AE, amplitude, coverage, Winkler, as in E3.1), then by method; each row and score
    group coloured as in E3.1 (``tables/baycap_fixed.py:_score_cell_styles``)."""
    labels = {score: baycap_fixed.SCORE_LABELS[score] for score in baycap_fixed.TABLE_SCORES}
    cols = {}
    for score in baycap_fixed.TABLE_SCORES:
        for method in METHODS:
            cols[(labels[score], _rotated(SHORT_LABELS[method]))] = table[f"{method}_{score}"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = baycap_fixed._score_cell_styles(out, labels)
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


# the columns of the selection and widths table: (group, header) -> column of the per-pair table
SELECTION_COLUMNS = {
    ("selection", "$b^\\star$"): f"{CB}_bandwidth",
    ("selection", "edge"): f"{CB}_edge",
    ("selection", "$\\mathcal L(0.1)$"): f"{CB}_logloss_default",
    ("selection", "$\\mathcal L(b^\\star)$"): f"{CB}_logloss_selected",
    ("ECE on $V$", "$0.1$"): f"{CB}_ece_default",
    ("ECE on $V$", "$b^\\star$"): f"{CB}_ece_selected",
    ("vs inst", "ratio"): f"{CB}_ratio_vs_inst",
    ("vs inst", "centre"): f"{CB}_centre_vs_inst",
    ("$\\bar\\sigma_w$", "cb"): f"{CB}_sigma_w",
    ("$\\bar\\sigma_w$", "inst"): f"{INST}_sigma_w",
    ("sd$(m_w)$", "cb"): f"{CB}_m_w_sd",
    ("sd$(m_w)$", "inst"): f"{INST}_m_w_sd",
    ("$\\sigma_p$", "cb"): f"{CB}_acc_sd",
    ("$\\sigma_p$", "inst"): f"{INST}_acc_sd",
    ("NUTS", "div."): f"{CB}_share_divergent",
    ("NUTS", "$\\hat R$"): f"{CB}_max_rhat",
}
# the columns of SELECTION_COLUMNS printed with four decimals
SMALL_GROUPS = ("$\\bar\\sigma_w$", "sd$(m_w)$", "$\\sigma_p$")


def _selection_styler(table: pd.DataFrame):
    """The selection of $b^\\star$, the ECE on $V$, and the label-free comparison of the widths with
    BayCAP-KDEy-h-inst."""
    out = pd.DataFrame({column: table[source].to_numpy() for column, source in SELECTION_COLUMNS.items()})
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    small = [column for column in out.columns if column[0] in SMALL_GROUPS] + [("vs inst", "centre")]
    return (
        out.style.format(precision=3, na_rep="--")
        .format("{:.4f}", subset=small, na_rep="--")
        .format("{:.0f}", subset=[("selection", "edge")], na_rep="--")
    )


TEST_LABELS = {"vs_inst": ("KDEy h inst cb", "$-$ KDEy h inst"), "vs_h_real": ("KDEy h inst cb", "$-$ KDEy h cell")}


def _tests_styler(table: pd.DataFrame):
    """Per test: the mean Winkler difference (bold where the Holm-adjusted p-value is below 0.05), its $95\\%$ bootstrap
    interval and the Holm-adjusted p-value."""
    cols = {}
    for name in TESTS:
        group = TEST_LABELS[name]
        cols[(*group, "diff.")] = [
            "--" if np.isnan(d) else (f"\\textbf{{{d:.3f}}}" if p < TEST_LEVEL else f"{d:.3f}")
            for d, p in zip(table[f"{name}_diff_mean"], table[f"{name}_p_holm"])
        ]
        cols[(*group, "95\\% CI")] = [
            "--" if np.isnan(lo) else f"[{lo:.3f}, {hi:.3f}]"
            for lo, hi in zip(table[f"{name}_ci_low"], table[f"{name}_ci_high"])
        ]
        cols[(*group, "$p_{\\mathrm{Holm}}$")] = ["--" if np.isnan(p) else f"{p:.3f}" for p in table[f"{name}_p_holm"]]
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style


COMMON_CAPTION = (
    "\\bcap{}-KDEy-h-inst with the bandwidth selected for calibrated class probabilities on \\texttt{main} (E3.5; "
    "vanilla accuracy, $|U|=1000$); means over the 100 bags, scored against the realised table; KDEy h inst cb: "
    "\\bcap{}-KDEy-h-inst-cb ($b^\\star$ in the likelihood and the weights); KDEy h inst: \\bcap{}-KDEy-h-inst "
    "($b=0.1$); KDEy h cell: \\bcap{}-KDEy-h-real (cell pushforward, $b=0.1$); Hf: \\bcap{}-Hf-real. "
)
CAPTIONS = {
    "comparison": COMMON_CAPTION
    + "Absolute error AE, amplitude $\\mathcal{A}$, coverage $\\mathcal{C}$, Winkler score $\\mathcal{W}$. "
    + baycap_fixed.SCORE_HIGHLIGHT,
    "selection": COMMON_CAPTION
    + "Label-free: the selected bandwidth $b^\\star$, $-1$/$0$/$+1$ at the lower edge, inside or at the upper edge "
    "of the grid, the class-balanced leave-one-out log-loss on $V$ at $0.1$ and at $b^\\star$, the ECE of the "
    "leave-one-out confidences on $V$ at $0.1$ and at $b^\\star$; amplitude ratio KDEy h inst cb / KDEy h inst "
    "(median over the bags) and mean difference of their point estimates (centre); means over the bags of "
    "$\\bar\\sigma_w$, of the standard deviation of the centres $m_w$ over the draws and of the standard deviation "
    "$\\sigma_p$ of the population accuracies, for both methods; NUTS of KDEy h inst cb: share of bags with "
    "divergent transitions and largest split-$\\hat R$.",
    "tests": COMMON_CAPTION
    + "Mean per-bag Winkler difference of KDEy h inst cb minus its reference, $95\\%$ paired bootstrap interval and "
    "Holm-adjusted p-value over the pairs; bold: $p<0.05$.",
}


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}.csv")


def curves_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_curves.csv")


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def gen_tables(pargs):
    results, pairs = load_all()
    table, curves = pair_tables(results, load_validation_counts(pairs, domain=DOMAIN))
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    curves.to_csv(curves_path(out_dir), index=False)

    stylers = {
        "comparison": _comparison_styler(table),
        "selection": _selection_styler(table),
        "tests": _tests_styler(table),
    }
    for name, styler in stylers.items():
        label = f"tab:{TABLE_NAME}-{name}"
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, CAPTIONS[name], label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(baycap_kdey_hf._latex_document(f"tabular/{_stem(name)}_long.tex"))

    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in analysis_summary(table).items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")
    print(f"pairs: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for name in CAPTIONS:
        tex_filename = f"{_stem(name)}_doc.tex"
        if not os.path.exists(os.path.join(out_dir, tex_filename)):
            raise FileNotFoundError(f"{tex_filename} does not exist in {out_dir}. Generate tables first.")
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
        description="BayCAP-KDEy-h-inst with a bandwidth selected for calibrated class probabilities (E3.5)."
    )
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
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
