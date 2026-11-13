"""SQS-B on the pairs of ``main`` (work-plan card WP-18, E3.3 of ``docs/ext3_posterior_propagation.md``).

``SQS-B`` (``methods/sqs_b.py:SQSB``) draws one bag of $N^Q=1000$ instances of $V$ at each posterior draw of BayesKDEy
with an MLP input and returns the tables of $h$ on these bags. It is compared bag by bag with ``BayCAP-KDEy-mlp-real``
(the same prevalence draws, only the bridge from the prevalence to the table differs), ``SQS-Hbs`` (the same
construction of the bags; PACC with a bootstrap of the target instead of the posterior), ``SQS-H`` and
``BayesCAP-Hf-real`` (the reference for coverage). All five target the realised table of the target bag and are scored
against it (``true_cts``); the population method ``BayCAP-KDEy-mlp`` is loaded only for its AE, the AE of the centre
$\\hat d^\\top\\bar\\pi$ that SQS-B shares (P15, §T.14). Vanilla accuracy, $|U|=1000$, the pairs where the six methods
have stored results (so it also works on the pilot).

Per pair (``pair_table``, one row per pair, every column in the CSV):

- AE, coverage, amplitude and Winkler score of the six methods, means over the bags; the bags where
  ``BayesCAP-Hf-real`` produced no draws (``nuts_failed``) are left out of its scores;
- the amplitude ratio of ``SQS-B`` to ``BayCAP-KDEy-mlp-real``, observed (median over the bags of the per-bag ratio)
  and predicted (median over the bags of $\\sqrt{(\\sigma_p^2+\\bar\\sigma_f^2)/(\\sigma_p^2+\\bar\\sigma_g^2)}$, from the
  summaries stored by ``SQS-B``), and the share of bags where the two are within $10\\%$ (P16);
- the means over the bags of the label-free summaries of ``SQS-B``: $\\sigma_p$, $\\bar\\sigma_f$, $\\bar\\sigma_g$,
  $S_{\\text{rec}}$, $S_{\\text{prec}}$ and $S_{\\text{prec}}-S_{\\text{rec}}$; the NUTS diagnostics (share of bags with
  divergent transitions, largest split-$\\hat R$, smallest effective sample size); and the largest difference between
  the posterior means of $\\pi$ stored by ``SQS-B`` and by ``BayCAP-KDEy-mlp-real``, which is $0$ up to rounding when
  the two methods have the same prevalence draws;
- two paired bootstrap tests of the mean per-bag Winkler difference (``tables/pushforward.py:bootstrap_mean_test``,
  $10{,}000$ resamples), each Holm-adjusted over the pairs on its own: ``SQS-B`` minus ``BayCAP-KDEy-mlp-real`` and
  ``SQS-B`` minus ``SQS-Hbs``.

The scores and the tests use the true tables, so they are analyses; the ratios, the summaries and the diagnostics use no
target labels. Outputs, under ``<OUT_DIR>/uncawe/tables/sqs_b/``: the per-pair CSV, which ``plots/sqs_b.py`` reads, and
three tables (comparison; widths; tests), each as a LaTeX tabular fragment, a longtable and a standalone landscape
document. Tables follow the ``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import SQS_B, SQS_B_QUANTIFIER
from uncawe.tables import baycap_fixed, baycap_h, baycap_kdey_hf, eb_prior, pushforward
from uncawe.tables.baycap_kdey_hf import bag_frame
from uncawe.tables.displacement import load_validation_counts, ordered
from uncawe.tables.estimand_gap import ACC_NAME, DOMAIN, SEED, booktabs_clines, pair_rng

TABLE_NAME = "sqs_b"
KEYS = eb_prior.KEYS
KDEY_REAL = SQS_B_QUANTIFIER
KDEY_POPULATION = "BayCAP-KDEy-mlp"
SQS_HBS = baycap_fixed.SQS_HBS
SQS_H = "SQS-H"
HF = baycap_fixed.HF
# the methods with an interval, in the order of the tables, and the population method, loaded for its AE only
INTERVAL_METHODS = [SQS_B, KDEY_REAL, SQS_HBS, SQS_H, HF]
METHODS = [*INTERVAL_METHODS, KDEY_POPULATION]
SCORES = ("ae", "coverage", "amplitude", "winkler")
TEST_LEVEL = pushforward.WINKLER_TEST_LEVEL
# the tests, name -> (method, reference): the per-bag difference is method minus reference
TESTS = {"vs_kdey": (SQS_B, KDEY_REAL), "vs_sqs_hbs": (SQS_B, SQS_HBS)}
# the label-free summaries of SQS-B averaged over the bags
SUMMARIES = ("acc_sd", "sigma_f", "sigma_g", "s_rec", "s_prec")
# a predicted ratio is "close" to the observed one within this relative tolerance (P16)
RATIO_TOLERANCE = 0.10


def _numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def predicted_ratio(acc_sd: np.ndarray, sigma_f: np.ndarray, sigma_g: np.ndarray) -> np.ndarray:
    """$\\sqrt{(\\sigma_p^2+\\bar\\sigma_f^2)/(\\sigma_p^2+\\bar\\sigma_g^2)}$ per bag (§T.14); NaN where the denominator
    is $0$."""
    acc_sd, sigma_f, sigma_g = (np.asarray(x, dtype=float) for x in (acc_sd, sigma_f, sigma_g))
    num, den = acc_sd**2 + sigma_f**2, acc_sd**2 + sigma_g**2
    return np.sqrt(np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0))


def width_columns(df: pd.DataFrame, bags: dict[str, pd.DataFrame], kdey_df: pd.DataFrame) -> dict:
    """The ratios of the amplitudes, the label-free summaries, the NUTS diagnostics of ``SQS-B`` on one pair, and the
    largest difference of the stored posterior means of $\\pi$ of ``SQS-B`` and ``BayCAP-KDEy-mlp-real``."""
    amplitude, reference = bags[SQS_B]["amplitude"].to_numpy(), bags[KDEY_REAL]["amplitude"].to_numpy()
    observed = np.divide(amplitude, reference, out=np.full_like(amplitude, np.nan), where=reference > 0)
    predicted = predicted_ratio(_numeric(df, "acc_sd"), _numeric(df, "sigma_f"), _numeric(df, "sigma_g"))
    close = np.abs(observed - predicted) <= RATIO_TOLERANCE * predicted
    prev = np.stack([np.asarray(p, dtype=float) for p in df["prev_mean"]])
    prev_kdey = np.stack([np.asarray(p, dtype=float) for p in kdey_df["prev_mean"]])
    out = {
        "ratio_observed": float(np.nanmedian(observed)),
        "ratio_predicted": float(np.nanmedian(predicted)),
        "share_ratio_close": float(np.mean(close)),
        "prev_mean_max_diff": float(np.max(np.abs(prev - prev_kdey))),
        "share_divergent": float(np.mean(_numeric(df, "nuts_divergences") > 0)),
        "max_rhat": float(np.max(_numeric(df, "nuts_max_rhat"))),
        "min_ess": float(np.min(_numeric(df, "nuts_min_ess"))),
    }
    for name in SUMMARIES:
        out[name] = float(np.mean(_numeric(df, name)))
    out["s_diff"] = out["s_prec"] - out["s_rec"]
    return {f"{SQS_B}_{name}": value for name, value in out.items()}


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
    for key, first in groups[SQS_B].items():
        bags = {method: bag_frame(groups[method][key], method) for method in METHODS}
        for method in METHODS[1:]:
            baycap_fixed._check_paired(bags[SQS_B], bags[method], key, (SQS_B, method))
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
        row |= width_columns(first, bags, groups[KDEY_REAL][key])
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
    """Reduce the per-pair table to the summaries of the Analysis of E3.3, over all pairs and per group of $K$.

    - ``scores``: mean AE, coverage, amplitude and Winkler score of the six methods;
    - ``tests``: per test, the pairs where ``SQS-B`` is significantly (Holm) lower and higher than its reference;
    - ``widths``: the median over the pairs of the observed and predicted ratio and of their quotient, the pairs where
      the observed ratio is within ``RATIO_TOLERANCE`` of the predicted one, the pairs with $S_{\\text{prec}}>
      S_{\\text{rec}}$ and, among them, those with an observed ratio above $1$; the largest difference of the posterior
      means of $\\pi$ (P16, and the check of the shared draws).
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
        observed, predicted = sub[f"{SQS_B}_ratio_observed"], sub[f"{SQS_B}_ratio_predicted"]
        prec_larger = sub[f"{SQS_B}_s_diff"] > 0
        out["widths"].append(
            key
            | {
                "ratio_observed": float(observed.median()),
                "ratio_predicted": float(predicted.median()),
                "observed_over_predicted": float((observed / predicted).median()),
                "pairs_close": int((np.abs(observed - predicted) <= RATIO_TOLERANCE * predicted).sum()),
                "pairs_s_prec_larger": int(prec_larger.sum()),
                "pairs_s_prec_larger_ratio_above_1": int((prec_larger & (observed > 1)).sum()),
                "prev_mean_max_diff": float(sub[f"{SQS_B}_prev_mean_max_diff"].max()),
            }
        )
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


SHORT_LABELS = {
    SQS_B: "SQS-B",
    KDEY_REAL: "KDEy mlp",
    KDEY_POPULATION: "KDEy mlp pop.",
    SQS_HBS: "SQS-H\\textsubscript{bs}",
    SQS_H: "SQS-H",
    HF: "Hf",
}
# the methods of each score group of the comparison table: the AE of SQS-B and of the population method whose centre it
# shares, then the amplitude, coverage and Winkler score of the five methods with an interval
SCORE_METHODS = {
    "ae": [SQS_B, KDEY_POPULATION, KDEY_REAL],
    **dict.fromkeys(("amplitude", "coverage", "winkler"), INTERVAL_METHODS),
}


def _rotated(label: str) -> str:
    return baycap_kdey_hf._rotated(label)


def _comparison_styler(table: pd.DataFrame):
    """Columns grouped by score (AE, amplitude, coverage, Winkler, as in E3.1), then by method; each row and score
    group coloured as in E3.1 (``tables/baycap_fixed.py:_score_cell_styles``)."""
    labels = {score: baycap_fixed.SCORE_LABELS[score] for score in baycap_fixed.TABLE_SCORES}
    cols = {}
    for score in baycap_fixed.TABLE_SCORES:
        for method in SCORE_METHODS[score]:
            cols[(labels[score], _rotated(SHORT_LABELS[method]))] = table[f"{method}_{score}"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styles = baycap_fixed._score_cell_styles(out, labels)
    return out.style.format(precision=3, na_rep="--").apply(lambda _: styles, axis=None)


WIDTH_COLUMNS = {
    "acc_sd": "$\\sigma_p$",
    "sigma_f": "$\\bar\\sigma_f$",
    "sigma_g": "$\\bar\\sigma_g$",
    "s_rec": "$S_{\\text{rec}}$",
    "s_prec": "$S_{\\text{prec}}$",
    "ratio_observed": "obs.",
    "ratio_predicted": "pred.",
    "share_ratio_close": "close",
    "share_divergent": "div.",
    "max_rhat": "$\\hat R$",
    "min_ess": "ESS",
}


def _widths_styler(table: pd.DataFrame):
    """The label-free summaries of SQS-B, the observed and predicted amplitude ratios and the NUTS diagnostics."""
    out = pd.DataFrame({header: table[f"{SQS_B}_{column}"].to_numpy() for column, header in WIDTH_COLUMNS.items()})
    out.index = baycap_h._index(table)
    small = ["$\\sigma_p$", "$\\bar\\sigma_f$", "$\\bar\\sigma_g$", "$S_{\\text{rec}}$", "$S_{\\text{prec}}$"]
    return (
        out.style.format(precision=3, na_rep="--")
        .format("{:.4f}", subset=small, na_rep="--")
        .format("{:.0f}", subset=["ESS"], na_rep="--")
    )


TEST_LABELS = {"vs_kdey": ("SQS-B", "$-$ KDEy mlp"), "vs_sqs_hbs": ("SQS-B", "$-$ SQS-H\\textsubscript{bs}")}


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
    "SQS-B on \\texttt{main} (E3.3; vanilla accuracy, $|U|=1000$); means over the 100 bags, scored against the "
    "realised table; KDEy mlp: \\bcap{}-KDEy-mlp-real (the same prevalence draws); KDEy mlp pop.: \\bcap{}-KDEy-mlp "
    "(population pushforward, AE only); Hf: \\bcap{}-Hf-real. "
)
CAPTIONS = {
    "comparison": COMMON_CAPTION
    + "Absolute error AE, amplitude $\\mathcal{A}$, coverage $\\mathcal{C}$, Winkler score $\\mathcal{W}$. "
    + baycap_fixed.SCORE_HIGHLIGHT,
    "widths": COMMON_CAPTION
    + "Label-free, for SQS-B: means over the bags of $\\sigma_p$, $\\bar\\sigma_f$, $\\bar\\sigma_g$, "
    "$S_{\\text{rec}}$ and $S_{\\text{prec}}$; amplitude ratio SQS-B / KDEy mlp, observed and predicted "
    "($\\sqrt{(\\sigma_p^2+\\bar\\sigma_f^2)/(\\sigma_p^2+\\bar\\sigma_g^2)}$), medians over the bags, and the share of "
    "bags where they agree within $10\\%$; NUTS: share of bags with divergent transitions, largest split-$\\hat R$, "
    "smallest effective sample size.",
    "tests": COMMON_CAPTION
    + "Mean per-bag Winkler difference of SQS-B minus its reference, $95\\%$ paired bootstrap interval and "
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
    parser = ArgumentParser(description="SQS-B on the pairs of main (E3.3).")
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
