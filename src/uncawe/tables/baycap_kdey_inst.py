"""BayCAP-KDEy with the per-instance pushforward on the pairs of ``main`` (work-plan card WP-31, E3.4 of
``docs/ext3_posterior_propagation.md``, §T.15).

``BayCAP-KDEy-h-inst`` (``methods/baycap_kdey.py:BayCAPKDEy``, ``estimand="realised-instance"``) draws the true class of
each instance of the bag from its posterior class probabilities under each draw of $\\pi$ of BayesKDEy with the input
classifier $h$. It is compared bag by bag with ``BayCAP-KDEy-h-real`` (the cell pushforward with the same prevalence
draws: only the bridge from the prevalence to the table differs), ``BayCAP-KDEy-mlp-real`` (the former BayCAP-KDEy of
the project: input classifier and pushforward differ) and ``BayesCAP-Hf-real`` (the reference for coverage). All four
target the realised table of the target bag and are scored against it (``true_cts``). Vanilla accuracy, $|U|=1000$,
the pairs where the four methods have stored results (so it also works on the pilot).

Per pair (``pair_table``, one row per pair, every column in the CSV):

- AE, coverage, amplitude and Winkler score of the four methods, means over the bags; the bags where
  ``BayesCAP-Hf-real`` produced no draws (``nuts_failed``) are left out of its scores;
- the amplitude ratio of ``BayCAP-KDEy-h-inst`` to ``BayCAP-KDEy-h-real``, observed (median over the bags of the
  per-bag ratio) and predicted (median over the bags of
  $\\sqrt{(\\operatorname{sd}(m_w)^2+\\bar\\sigma_w^2)/(\\operatorname{sd}(m)^2+\\bar\\sigma_g^2)}$, from the summaries
  stored by ``BayCAP-KDEy-h-inst``), and the share of bags where the two are within $10\\%$ (P20);
- the means over the bags of the label-free summaries of ``BayCAP-KDEy-h-inst``: $\\bar\\sigma_w$, $\\bar\\sigma_g$,
  $\\bar{\\tilde\\sigma}_g$, the within-class term, the share of $\\bar{\\tilde\\sigma}^2_g$ that it removes
  (``within_share``, the mean over the bags of the within-class term over $\\bar{\\tilde\\sigma}^2_g$),
  $\\operatorname{sd}(m_w)$, $\\operatorname{sd}(m)$ and the centre difference $\\bar m_w-\\bar m$ (P21); the share of
  bags with $\\bar\\sigma_w\\le\\bar{\\tilde\\sigma}_g$ (P20); the NUTS diagnostics (share of bags with divergent
  transitions, largest split-$\\hat R$, smallest effective sample size); and the largest difference between the
  posterior means of $\\pi$ stored by ``BayCAP-KDEy-h-inst`` and by ``BayCAP-KDEy-h-real``, which is $0$ up to rounding
  when the two methods have the same prevalence draws (P19);
- two paired bootstrap tests of the mean per-bag Winkler difference (``tables/pushforward.py:bootstrap_mean_test``,
  $10{,}000$ resamples), each Holm-adjusted over the pairs on its own: ``BayCAP-KDEy-h-inst`` minus
  ``BayCAP-KDEy-h-real`` and ``BayCAP-KDEy-h-inst`` minus ``BayCAP-KDEy-mlp-real``.

The scores and the tests use the true tables, so they are analyses; the ratios, the summaries and the diagnostics use no
target labels. Outputs, under ``<OUT_DIR>/uncawe/tables/baycap_kdey_inst/``: the per-pair CSV, which
``plots/baycap_kdey_inst.py`` reads, and three tables (comparison; widths; tests), each as a LaTeX tabular fragment, a
longtable and a standalone landscape document; the comparison table of the Representation of E3.4 is split into these
three, as those of E3.2 and E3.3, since it does not fit one page. Tables follow the ``cap-ml`` layout (true classes on
rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import INSTANCE_VARIANTS
from uncawe.tables import baycap_fixed, baycap_h, baycap_kdey_hf, eb_prior, pushforward
from uncawe.tables.baycap_kdey_hf import bag_frame
from uncawe.tables.displacement import load_validation_counts, ordered
from uncawe.tables.estimand_gap import ACC_NAME, DOMAIN, SEED, booktabs_clines, pair_rng

TABLE_NAME = "baycap_kdey_inst"
KEYS = eb_prior.KEYS
INST = "BayCAP-KDEy-h-inst"
H_REAL = INSTANCE_VARIANTS[INST]
MLP_REAL = "BayCAP-KDEy-mlp-real"
HF = baycap_fixed.HF
# the methods, in the order of the tables
METHODS = [INST, H_REAL, MLP_REAL, HF]
SCORES = ("ae", "coverage", "amplitude", "winkler")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
# the tests, name -> (method, reference): the per-bag difference is method minus reference
TESTS = {"vs_h_real": (INST, H_REAL), "vs_mlp_real": (INST, MLP_REAL)}
# the label-free summaries of BayCAP-KDEy-h-inst averaged over the bags (methods/baycap_kdey.py:INSTANCE_SUMMARIES and
# the sigma_g of FIXED_SUMMARIES)
SUMMARIES = ("sigma_w", "sigma_g", "sigma_g_tilde", "within_class", "m_w_sd", "m_sd", "centre_shift")
# a predicted ratio is "close" to the observed one within this relative tolerance (P20), as in E3.3
RATIO_TOLERANCE = 0.10


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def predicted_ratio(m_w_sd, sigma_w, m_sd, sigma_g) -> np.ndarray:
    """$\\sqrt{(\\operatorname{sd}(m_w)^2+\\bar\\sigma_w^2)/(\\operatorname{sd}(m)^2+\\bar\\sigma_g^2)}$ per bag
    (§T.15, P20); NaN where the denominator is $0$."""
    m_w_sd, sigma_w, m_sd, sigma_g = (np.asarray(x, dtype=float) for x in (m_w_sd, sigma_w, m_sd, sigma_g))
    num, den = m_w_sd**2 + sigma_w**2, m_sd**2 + sigma_g**2
    return np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))


def width_columns(df: pd.DataFrame, bags: dict[str, pd.DataFrame], reference_df: pd.DataFrame) -> dict:
    """The ratios of the amplitudes, the label-free summaries and the NUTS diagnostics of ``BayCAP-KDEy-h-inst`` on one
    pair, and the largest difference of the stored posterior means of $\\pi$ of ``BayCAP-KDEy-h-inst`` and
    ``BayCAP-KDEy-h-real`` (``reference_df``)."""
    amplitude, reference = bags[INST]["amplitude"].to_numpy(), bags[H_REAL]["amplitude"].to_numpy()
    observed = np.divide(amplitude, reference, out=np.full_like(amplitude, np.nan), where=reference > 0)
    sigma_w, sigma_g_tilde, within = (
        _numeric(df, "sigma_w"),
        _numeric(df, "sigma_g_tilde"),
        _numeric(df, "within_class"),
    )
    predicted = predicted_ratio(_numeric(df, "m_w_sd"), sigma_w, _numeric(df, "m_sd"), _numeric(df, "sigma_g"))
    close = np.abs(observed - predicted) <= RATIO_TOLERANCE * predicted
    tilde2 = sigma_g_tilde**2
    prev = np.stack([np.asarray(p, dtype=float) for p in df["prev_mean"]])
    prev_reference = np.stack([np.asarray(p, dtype=float) for p in reference_df["prev_mean"]])
    out = {
        "ratio_observed": float(np.nanmedian(observed)),
        "ratio_predicted": float(np.nanmedian(predicted)),
        "share_ratio_close": float(np.mean(close)),
        "within_share": float(
            np.nanmean(np.divide(within, tilde2, out=np.full_like(within, np.nan), where=tilde2 > 0))
        ),
        # sigma_w <= sigma_g_tilde holds on every draw (§T.15), so on the root means too; the tolerance is rounding
        "share_sigma_w_below": float(np.mean(sigma_w <= sigma_g_tilde * (1 + 1e-9))),
        "prev_mean_max_diff": float(np.max(np.abs(prev - prev_reference))),
        "share_divergent": float(np.mean(_numeric(df, "nuts_divergences") > 0)),
        "max_rhat": float(np.max(_numeric(df, "nuts_max_rhat"))),
        "min_ess": float(np.min(_numeric(df, "nuts_min_ess"))),
    }
    for name in SUMMARIES:
        out[name] = float(np.mean(_numeric(df, name)))
    return {f"{INST}_{name}": value for name, value in out.items()}


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


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (method -> stored results, the same pairs and bags for every method), ordered as
    the tables of E3.1 with the validation counts ``val_counts`` (pair -> counts in the ``cap-ml`` layout)."""
    rows = []
    groups = {method: dict(list(df.groupby(KEYS, sort=False))) for method, df in results.items()}
    for key, first in groups[INST].items():
        bags = {method: bag_frame(groups[method][key], method) for method in METHODS}
        for method in METHODS[1:]:
            baycap_fixed._check_paired(bags[INST], bags[method], key, (INST, method))
        row = {
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
            "n_val": int(val_counts[key].sum()),
            "min_val_per_class": int(val_counts[key].sum(axis=1).min()),
        }
        for method in METHODS:
            for score in SCORES:
                row[f"{method}_{score}"] = float(bags[method][score].mean())
        row[f"{HF}_share_failed"] = float(bags[HF]["failed"].mean())
        row |= width_columns(first, bags, groups[H_REAL][key])
        row |= pair_tests(bags, key)
        rows.append(row)
    table = pd.DataFrame(rows)
    for name in TESTS:
        table[f"{name}_p_holm"] = pushforward.holm(table[f"{name}_p"].to_numpy())
    return ordered(table)


def load_all(acc_name: str = ACC_NAME) -> tuple[dict[str, pd.DataFrame], list]:
    """The stored results of ``METHODS`` on the pairs stored for all of them."""
    return baycap_fixed.load_all(acc_name, methods=METHODS)


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E3.4, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage, amplitude and Winkler score of the four methods;
    - ``tests``: per test, the pairs where ``BayCAP-KDEy-h-inst`` is significantly (Holm) lower and higher than its
      reference;
    - ``widths``: the median over the pairs of the observed and predicted ratio and of their quotient, the pairs where
      the observed ratio is within ``RATIO_TOLERANCE`` of the predicted one, the pairs with an observed ratio below $1$
      (P20); the median within share and the mean centre difference (P21); the largest difference of the posterior
      means of $\\pi$ (P19).
    """
    out = {"scores": [], "tests": [], "widths": []}
    for k_group, sub in baycap_h._k_groups(table):
        key = {"K": k_group, "pairs": len(sub)}
        out["scores"].append(key | {f"{m}_{s}": float(sub[f"{m}_{s}"].mean()) for m in METHODS for s in SCORES})
        tests = dict(key)
        for name in TESTS:
            significant = sub[f"{name}_p_holm"] < TEST_LEVEL
            lower = sub[f"{name}_diff_mean"] < 0
            tests[f"{name}_lower"] = int((lower & significant).sum())
            tests[f"{name}_higher"] = int((~lower & significant).sum())
        out["tests"].append(tests)
        observed, predicted = sub[f"{INST}_ratio_observed"], sub[f"{INST}_ratio_predicted"]
        out["widths"].append(
            key
            | {
                "ratio_observed": float(observed.median()),
                "ratio_predicted": float(predicted.median()),
                "observed_over_predicted": float((observed / predicted).median()),
                "pairs_close": int((np.abs(observed - predicted) <= RATIO_TOLERANCE * predicted).sum()),
                "pairs_ratio_below_1": int((observed < 1).sum()),
                "within_share": float(sub[f"{INST}_within_share"].median()),
                "centre_shift": float(sub[f"{INST}_centre_shift"].mean()),
                "prev_mean_max_diff": float(sub[f"{INST}_prev_mean_max_diff"].max()),
            }
        )
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


SHORT_LABELS = {INST: "KDEy h inst", H_REAL: "KDEy h cell", MLP_REAL: "KDEy mlp cell", HF: "Hf"}


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


WIDTH_COLUMNS = {
    "sigma_w": "$\\bar\\sigma_w$",
    "sigma_g": "$\\bar\\sigma_g$",
    "sigma_g_tilde": "$\\bar{\\tilde\\sigma}_g$",
    "within_share": "within",
    "m_w_sd": "sd$(m_w)$",
    "m_sd": "sd$(m)$",
    "centre_shift": "$\\bar m_w-\\bar m$",
    "ratio_observed": "obs.",
    "ratio_predicted": "pred.",
    "share_ratio_close": "close",
    "share_divergent": "div.",
    "max_rhat": "$\\hat R$",
    "min_ess": "ESS",
}


def _widths_styler(table: pd.DataFrame):
    """The label-free summaries of BayCAP-KDEy-h-inst, the observed and predicted amplitude ratios and the NUTS
    diagnostics."""
    out = pd.DataFrame({header: table[f"{INST}_{column}"].to_numpy() for column, header in WIDTH_COLUMNS.items()})
    out.index = baycap_h._index(table)
    small = [WIDTH_COLUMNS[name] for name in ("sigma_w", "sigma_g", "sigma_g_tilde", "m_w_sd", "m_sd", "centre_shift")]
    return (
        out.style.format(precision=3, na_rep="--")
        .format("{:.4f}", subset=small, na_rep="--")
        .format("{:.0f}", subset=["ESS"], na_rep="--")
    )


TEST_LABELS = {"vs_h_real": ("KDEy h inst", "$-$ KDEy h cell"), "vs_mlp_real": ("KDEy h inst", "$-$ KDEy mlp cell")}


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
    "\\bcap{}-KDEy with the per-instance pushforward on \\texttt{main} (E3.4; vanilla accuracy, $|U|=1000$); means "
    "over the 100 bags, scored against the realised table; KDEy h inst: \\bcap{}-KDEy-h-inst; KDEy h cell: "
    "\\bcap{}-KDEy-h-real (cell pushforward, the same prevalence draws); KDEy mlp cell: \\bcap{}-KDEy-mlp-real; "
    "Hf: \\bcap{}-Hf-real. "
)
CAPTIONS = {
    "comparison": COMMON_CAPTION
    + "Absolute error AE, amplitude $\\mathcal{A}$, coverage $\\mathcal{C}$, Winkler score $\\mathcal{W}$. "
    + baycap_fixed.SCORE_HIGHLIGHT,
    "widths": COMMON_CAPTION
    + "Label-free, for KDEy h inst: means over the bags of $\\bar\\sigma_w$, $\\bar\\sigma_g$, "
    "$\\bar{\\tilde\\sigma}_g$, of the share of $\\bar{\\tilde\\sigma}^2_g$ removed by the within-class spread of the "
    "confidences (within), of the standard deviations of the centres $m_w$ and $m$ over the draws and of the centre "
    "difference $\\bar m_w-\\bar m$; amplitude ratio KDEy h inst / KDEy h cell, observed and predicted "
    "($\\sqrt{(\\mathrm{sd}(m_w)^2+\\bar\\sigma_w^2)/(\\mathrm{sd}(m)^2+\\bar\\sigma_g^2)}$), medians over the bags, and "
    "the share of bags where they agree within $10\\%$; NUTS: share of bags with divergent transitions, largest "
    "split-$\\hat R$, smallest effective sample size.",
    "tests": COMMON_CAPTION
    + "Mean per-bag Winkler difference of KDEy h inst minus its reference, $95\\%$ paired bootstrap interval and "
    "Holm-adjusted p-value over the pairs; bold: $p<0.05$.",
}


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}.csv")


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def gen_tables(pargs):
    results, pairs = load_all()
    table = pair_table(results, load_validation_counts(pairs, domain=DOMAIN))
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)

    stylers = {
        "comparison": _comparison_styler(table),
        "widths": _widths_styler(table),
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
    parser = ArgumentParser(description="BayCAP-KDEy with the per-instance pushforward on the pairs of main (E3.4).")
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument(
        "--table-dir",
        "--output-dir",
        dest="table_dir",
        default=None,
        help="Directory where the CSV, table documents and tabular fragments are saved",
    )
    pargs = parser.parse_args()

    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
