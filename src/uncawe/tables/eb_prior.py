"""BayCAP with an empirical-Bayes prior on the recalls (work-plan card WP-6, E2.4 of ``docs/ext2_baycap.md``).

Four methods of ``main``, all ``methods/baycap.py:BayCAP`` with the prior on $\\pi^Q$, the NUTS settings and the seed of
``BayesCAP``: ``BayesCAP`` (population pushforward, $\\operatorname{Dir}(\\mathbf 1)$ prior on the columns of $C$),
``BayesCAP-real`` (realised pushforward, $\\operatorname{Dir}(\\mathbf 1)$), ``BayesCAP-EB`` (population, empirical
Bayes) and ``BayesCAP-EB-real`` (realised, empirical Bayes). The population methods target the population table
$C\\operatorname{diag}(\\pi^Q)$, the realised ones the realised table of the target sample; all four are scored against
the realised table of each bag (``true_cts``). Vanilla accuracy, $|U|=1000$, the pairs where all four have stored results.

The check predicts the effect of the prior without runs. From $V$ alone it computes three estimates $\\tilde r_j$ of each
recall: the plain recall $f_{jj}/n^s_j$, the posterior mean under $\\operatorname{Dir}(\\mathbf 1)$,
$\\frac{f_{jj}+1}{n^s_j+K}$, and the posterior mean under empirical Bayes, $\\frac{f_{jj}+\\hat\\nu\\hat m}{n^s_j+\\hat\\nu}$,
with $(\\hat m,\\hat\\nu)$ fitted as ``BayCAP.fit`` does. For each estimate and each bag $b$ the recall error is
$\\Delta^{(b)}=\\sum_j\\pi^{(b)}_j(\\tilde r_j-r^{(b)}_j)$, with $\\pi^{(b)}$ and $r^{(b)}$ the true class proportions and
realised recalls of the bag; per pair the table reports its mean (the displacement it predicts) and the mean of
$|\\Delta^{(b)}|$. The check uses target labels, so it is an analysis, not a method. It also reports $\\hat m$, $\\hat\\nu$,
whether the fit reached a bound, and the standard deviation that empirical Bayes leaves out of the interval,
$\\sum_j\\pi_j(1-w_j)\\operatorname{sd}(\\hat m)$ with $w_j=n^s_j/(n^s_j+\\hat\\nu)$ and $\\pi$ uniform.

For the four methods, the table reports coverage, amplitude, Winkler score, the error terms of E2.2
(:mod:`uncawe.tables.displacement`; the prior part $R_0$ uses each method's own prior) and, per pair, the NUTS
diagnostics stored by ``main``: the share of bags with divergent transitions, the largest split-$\\hat R$ and the smallest
effective sample size (NaN for results stored without them). Within each pushforward, the mean per-bag Winkler difference,
empirical Bayes minus $\\operatorname{Dir}(\\mathbf 1)$, is tested with the paired bootstrap of E2.3
(:func:`uncawe.tables.pushforward.winkler_tests`), Holm-adjusted over the pairs. The change of the recall term, $R$ of
``BayesCAP-EB`` minus $R$ of ``BayesCAP``, is compared with the change the check predicts, the difference between the
mean errors of the empirical-Bayes and the $\\operatorname{Dir}(\\mathbf 1)$ estimates.

Outputs, under ``<OUT_DIR>/uncawe/tables/eb_prior/``: a per-pair CSV with every column, and two tables, each as a LaTeX
tabular fragment, a longtable and a standalone landscape document: the comparison table (interval scores of the four
methods and the Winkler tests) and the prior table (the fit, the check, $R$, $R_0$, $P$ and the NUTS diagnostics).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.main import EXPERIMENT
from uncawe.methods.baycap import BetaBinomialFit, fit_beta_binomial
from uncawe.tables import displacement, estimand_gap, pushforward
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import METHOD_LABELS, load_validation_counts, ordered, plain_recalls, smoothed_recalls
from uncawe.tables.estimand_gap import booktabs_clines, load_results
from uncawe.util import decorate_dataset

TABLE_NAME = "eb_prior"
KEYS = ["dataset", "classifier"]
# the Dir(1) method and the empirical-Bayes method of each pushforward
PUSHFORWARDS = {"population": ("BayesCAP", "BayesCAP-EB"), "realised": ("BayesCAP-real", "BayesCAP-EB-real")}
METHODS = [method for pair in PUSHFORWARDS.values() for method in pair]
ESTIMATES = ("plain", "uniform", "eb")
ESTIMATE_LABELS = {"plain": "plain", "uniform": "$\\operatorname{Dir}(\\mathbf 1)$", "eb": "EB"}
# per-method columns of the E2.2 table (tables/displacement.py:pair_table)
METHOD_COLUMNS = ("coverage", "amplitude", "winkler", "predicted_coverage") + displacement.TERMS
DIAGNOSTICS = ("share_divergent", "max_rhat", "min_ess")
TEST_COLUMNS = ("winkler_diff_mean", "winkler_ci_low", "winkler_ci_high", "winkler_diff_median", "winkler_p")


def recall_estimates(val_counts: np.ndarray) -> tuple[dict[str, np.ndarray], BetaBinomialFit]:
    """Return the three estimates of the recalls from $V$ (``ESTIMATES``) and the empirical-Bayes fit.

    ``val_counts`` is the validation confusion table in the ``cap-ml`` layout (true classes on rows).
    """
    val_counts = np.asarray(val_counts, dtype=float)
    f_diag, n_s = np.diag(val_counts), val_counts.sum(axis=1)
    fit = fit_beta_binomial(f_diag, n_s)
    estimates = {
        "plain": plain_recalls(val_counts),
        "uniform": smoothed_recalls(val_counts, "uniform"),
        "eb": (f_diag + fit.nu * fit.m) / (n_s + fit.nu),
    }
    return estimates, fit


def recall_error(recalls: np.ndarray, true_ct: np.ndarray) -> float:
    """Return $\\Delta=\\sum_j\\pi_j(\\tilde r_j-r_j)$ for one bag, written as $\\sum_j(\\pi_j\\tilde r_j-T_{jj})$.

    ``true_ct`` is the realised table of the bag in the ``cap-ml`` layout; classes absent from the bag do not count.
    """
    true_ct = np.asarray(true_ct, dtype=float)
    props = true_ct.sum(axis=1)
    present = props > 0
    return float(np.sum(props[present] * np.asarray(recalls, dtype=float)[present] - np.diag(true_ct)[present]))


def left_out_sd(n_s: np.ndarray, fit: BetaBinomialFit) -> float:
    """Return $\\frac1K\\sum_j(1-w_j)\\operatorname{sd}(\\hat m)$, $w_j=n^s_j/(n^s_j+\\hat\\nu)$ (NaN if the sd is)."""
    n_s = np.asarray(n_s, dtype=float)
    return float(np.mean(1 - n_s / (n_s + fit.nu)) * fit.sd_m)


def check_table(df: pd.DataFrame, val_counts: dict) -> pd.DataFrame:
    """Return, per (dataset, classifier) of ``df``, the empirical-Bayes fit and the mean recall errors of the check.

    The bags are the rows of ``df`` (their ``true_cts``); any method of ``main`` gives the same bags.
    """
    rows = []
    for (dataset, classifier), _df in df.groupby(KEYS, sort=False):
        counts = val_counts[(dataset, classifier)]
        estimates, fit = recall_estimates(counts)
        n_s = counts.sum(axis=1)
        record = {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": int(_df["n_classes"].iloc[0]),
            "n_val": int(n_s.sum()),
            "min_val_per_class": int(n_s.min()),
            "median_val_per_class": float(np.median(n_s)),
            "eb_m": fit.m,
            "eb_nu": fit.nu,
            "eb_m_at_bound": fit.m_at_bound,
            "eb_nu_at_bound": fit.nu_at_bound,
            "eb_sd_m": fit.sd_m,
            "left_out_sd": left_out_sd(n_s, fit),
        }
        for name, recalls in estimates.items():
            errors = np.array([recall_error(recalls, ct) for ct in _df["true_cts"]])
            record[f"check_{name}_mean"] = float(errors.mean())
            record[f"check_{name}_abs"] = float(np.abs(errors).mean())
        rows.append(record)
    return pd.DataFrame(rows)


def diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return, per (dataset, classifier), the share of bags with divergences, the largest split-$\\hat R$ and the
    smallest effective sample size; NaN where the results were stored without the NUTS diagnostics."""
    df = df.copy()
    for column in ("nuts_divergences", "nuts_max_rhat", "nuts_min_ess"):
        if column not in df:
            df[column] = np.nan
    divergences = pd.to_numeric(df["nuts_divergences"], errors="coerce")
    df["divergent"] = (divergences > 0).astype(float).where(divergences.notna())
    grouped = df.groupby(KEYS, sort=False)
    return pd.DataFrame(
        {
            "share_divergent": grouped["divergent"].mean(),
            "max_rhat": grouped["nuts_max_rhat"].max(),
            "min_ess": grouped["nuts_min_ess"].min(),
        }
    ).reset_index()


def method_table(df: pd.DataFrame, val_counts: dict, method: str) -> pd.DataFrame:
    """Return the E2.2 scores and terms and the NUTS diagnostics of ``method``, with columns ``<method>_<column>``."""
    scores = displacement.pair_table(df, val_counts, method)[KEYS + list(METHOD_COLUMNS)]
    table = scores.merge(diagnostics_table(df), on=KEYS, how="left")
    return table.rename(columns={col: f"{method}_{col}" for col in METHOD_COLUMNS + DIAGNOSTICS})


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """Join the check, the four methods and the Winkler tests, one row per pair, in the order of E2.2.

    ``results`` maps each method of ``METHODS`` to its stored ``main`` results, restricted to the same pairs.
    """
    table = check_table(results["BayesCAP"], val_counts)
    for method in METHODS:
        table = table.merge(method_table(results[method], val_counts, method), on=KEYS, how="inner")
    for name, (uniform, eb) in PUSHFORWARDS.items():
        # empirical Bayes minus Dir(1): winkler_tests subtracts its first argument from its second
        tests = pushforward.winkler_tests(
            pushforward.bag_scores(results[uniform], uniform), pushforward.bag_scores(results[eb], eb)
        )
        columns = list(TEST_COLUMNS) + ["winkler_p_holm"]
        table = table.merge(tests.rename(columns={col: f"{name}_{col}" for col in columns}), on=KEYS, how="left")
    table["predicted_recall_change"] = table["check_eb_mean"] - table["check_uniform_mean"]
    table["observed_recall_change"] = table["BayesCAP-EB_recall_term"] - table["BayesCAP_recall_term"]
    return ordered(table)


def common_pairs(results: dict[str, pd.DataFrame]) -> list[tuple[str, str]]:
    """Return the (dataset, classifier) pairs with stored results for every method, in the order of the first."""
    pair_sets = [list(df.groupby(KEYS, sort=False).groups) for df in results.values()]
    first, *others = pair_sets
    return [pair for pair in first if all(pair in set(other) for other in others)]


def compute_pair_table() -> pd.DataFrame:
    """Load the stored ``main`` results of the four methods and the validation counts, and return the ordered table."""
    results = {}
    for method in METHODS:
        df = load_results(method)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT}. Check that experiment results exist.")
        results[method] = df
    pairs = common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair has stored results for all of {METHODS}.")
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    results = {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in results.items()
    }
    return pair_table(results, load_validation_counts(pairs))


COMPARISON_CAPTION = (
    "BayCAP with the $\\operatorname{Dir}(\\mathbf 1)$ and the empirical-Bayes (EB) prior on the columns of $C$, "
    "population and realised pushforwards, on \\texttt{main} (vanilla accuracy, $|U|=1000$). Coverage "
    "$\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler score $\\mathcal{W}$ against the realised table, means over "
    "the 100 bags. Tests: 95\\% paired-bootstrap interval of the mean per-bag Winkler difference, EB minus "
    "$\\operatorname{Dir}(\\mathbf 1)$, within each pushforward, and its two-sided p-value, Holm-adjusted over the pairs."
)
PRIOR_CAPTION = (
    "The empirical-Bayes prior on \\texttt{main} (vanilla accuracy, $|U|=1000$). $n^s_j$: validation instances of class "
    "$\\omega_j$. Fit: $\\hat m$, $\\hat\\nu$ ($^*$ at a bound of the fit) and $\\sigma_{\\mathrm{lo}}$, the standard "
    "deviation that empirical Bayes leaves out of the interval. Check (no runs): mean and mean absolute recall error "
    "$\\Delta$ over the bags for the plain, the $\\operatorname{Dir}(\\mathbf 1)$ and the EB estimates of the recalls. "
    "$R$, $R_0$, $P$: recall term, its prior part and prevalence term (means over the bags) of BayCAP and BayCAP-EB. "
    "NUTS: share of bags with divergent transitions, largest split-$\\hat R$ and smallest effective sample size."
)
COMPARISON_LABEL = "tab:eb-prior-comparison"
PRIOR_LABEL = "tab:eb-prior"
PUSHFORWARD_LABELS = {"population": "test, population", "realised": "test, realised"}
TERM_LABELS = {"recall_term": "$R$", "prior_part": "$R_0$", "prevalence_term": "$P$"}
DIAGNOSTIC_LABELS = {"share_divergent": "div.", "max_rhat": "$\\hat R$", "min_ess": "ESS"}


def _index(table: pd.DataFrame, with_val_counts: bool) -> pd.MultiIndex:
    arrays = [
        table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d))),
        table["n_classes"].astype(int),
    ]
    names = ["dataset", "$K$"]
    if with_val_counts:
        arrays += [table["min_val_per_class"].astype(int), table["median_val_per_class"].round().astype(int)]
        names += ["$\\min n^s_j$", "$\\tilde n^s_j$"]
    return pd.MultiIndex.from_arrays(arrays + [table["classifier"].astype(str)], names=names + ["classifier"])


def _fmt_ci(low: float, high: float) -> str:
    return "--" if np.isnan(low) else f"[{low:.3f}, {high:.3f}]"


def _fmt_fit(value: float, at_bound: bool, digits: int) -> str:
    return f"{value:.{digits}f}" + ("$^*$" if at_bound else "")


def _comparison_styler(table: pd.DataFrame):
    cols = {}
    for method in METHODS:
        for score in pushforward.SCORES:
            cols[(METHOD_LABELS[method], main_tables.metrics_map[score])] = table[f"{method}_{score}"].to_numpy()
    for name, label in PUSHFORWARD_LABELS.items():
        cols[(label, "$\\Delta\\mathcal{W}$ 95\\%")] = [
            _fmt_ci(low, high) for low, high in zip(table[f"{name}_winkler_ci_low"], table[f"{name}_winkler_ci_high"])
        ]
        cols[(label, "$p_{\\mathcal{W}}$")] = table[f"{name}_winkler_p_holm"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table, with_val_counts=False)
    return out.style.format(precision=3, na_rep="--")


def _prior_styler(table: pd.DataFrame):
    cols = {
        ("fit", "$\\hat m$"): [_fmt_fit(v, b, 3) for v, b in zip(table["eb_m"], table["eb_m_at_bound"])],
        ("fit", "$\\hat\\nu$"): [_fmt_fit(v, b, 1) for v, b in zip(table["eb_nu"], table["eb_nu_at_bound"])],
        ("fit", "$\\sigma_{\\mathrm{lo}}$"): table["left_out_sd"].to_numpy(),
    }
    for stat, label in (("mean", "check, mean $\\Delta$"), ("abs", "check, mean $|\\Delta|$")):
        for name in ESTIMATES:
            cols[(label, ESTIMATE_LABELS[name])] = table[f"check_{name}_{stat}"].to_numpy()
    for method in PUSHFORWARDS["population"]:
        for term, label in TERM_LABELS.items():
            cols[(METHOD_LABELS[method], label)] = table[f"{method}_{term}"].to_numpy()
    formats = {}
    for method in METHODS:
        for diagnostic, label in DIAGNOSTIC_LABELS.items():
            key = (f"NUTS, {METHOD_LABELS[method]}", label)
            cols[key] = table[f"{method}_{diagnostic}"].to_numpy()
            formats[key] = {"share_divergent": "{:.2f}", "max_rhat": "{:.3f}", "min_ess": "{:.0f}"}[diagnostic]
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table, with_val_counts=True)
    styler = out.style.format(precision=3, na_rep="--")
    for key, fmt in formats.items():
        styler = styler.format(fmt, subset=[key], na_rep="--")
    return styler


TABLES = {
    "comparison": (_comparison_styler, COMPARISON_CAPTION, COMPARISON_LABEL),
    "prior": (_prior_styler, PRIOR_CAPTION, PRIOR_LABEL),
}


def _latex_table(styler) -> str:
    return booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data"))


def _latex_longtable(styler, caption: str, label: str) -> str:
    latex = styler.to_latex(
        environment="longtable",
        hrules=True,
        multicol_align="c",
        multirow_align="naive",
        clines="skip-last;data",
        caption=caption,
        label=label,
    )
    return booktabs_clines(latex)


def _latex_document(longtable_path: str) -> str:
    return "\n".join(
        [
            "\\documentclass{article}",
            "\\usepackage[a4paper,landscape,hmargin=1cm,vmargin=0.8cm]{geometry}",
            "\\usepackage{amsmath,booktabs,longtable,multirow}",
            "\\begin{document}",
            "\\pagestyle{empty}",
            "\\tiny",
            "\\setlength{\\LTcapwidth}{\\textwidth}",
            "\\setlength{\\tabcolsep}{3pt}",
            f"\\input{{{longtable_path}}}",
            "\\end{document}",
            "",
        ]
    )


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", TABLE_NAME)


def _print_summary(table: pd.DataFrame):
    print(f"pairs: {len(table)}")
    print(
        f"EB fit: m at a bound on {int(table['eb_m_at_bound'].sum())} pairs, nu at a bound on "
        f"{int(table['eb_nu_at_bound'].sum())}; largest left-out sd: {table['left_out_sd'].max():.4f}"
    )
    print(
        f"check: correlation of the Dir(1) check with R of BayesCAP "
        f"{table['check_uniform_mean'].corr(table['BayesCAP_recall_term']):.3f}, largest gap "
        f"{(table['check_uniform_mean'] - table['BayesCAP_recall_term']).abs().max():.4f}; correlation of the predicted "
        f"and the observed change of R {table['predicted_recall_change'].corr(table['observed_recall_change']):.3f}, "
        f"largest gap {(table['predicted_recall_change'] - table['observed_recall_change']).abs().max():.4f}"
    )
    for method in METHODS:
        coverage = table[f"{method}_coverage"]
        print(
            f"{method}: mean coverage {coverage.mean():.3f}, pairs at or above 0.95: {int((coverage >= 0.95).sum())}, "
            f"under {estimand_gap.UNDER_COVERAGE}: {int((coverage < estimand_gap.UNDER_COVERAGE).sum())}; "
            f"pairs with divergences: {int((table[f'{method}_share_divergent'] > 0).sum())}, without diagnostics: "
            f"{int(table[f'{method}_share_divergent'].isna().sum())}"
        )
    for name, (uniform, eb) in PUSHFORWARDS.items():
        significant = table[f"{name}_winkler_p_holm"] < pushforward.WINKLER_TEST_LEVEL
        lower = table[f"{eb}_winkler"] < table[f"{uniform}_winkler"]
        print(
            f"Winkler, {eb} against {uniform} (Holm p < {pushforward.WINKLER_TEST_LEVEL}): lower on "
            f"{int((lower & significant).sum())} pairs, higher on {int((~lower & significant).sum())}, not significant "
            f"on {int((~significant).sum())}"
        )


def gen_tables(pargs):
    table = compute_pair_table()
    out_dir = _table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    table.to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)
    for name, (styler_fn, caption, label) in TABLES.items():
        styler = styler_fn(table)
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(_latex_table(styler))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(_latex_longtable(styler, caption, label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{_stem(name)}_long.tex"))

    _print_summary(table)
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = _table_dir(pargs)
    for name in TABLES:
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
    parser = ArgumentParser(description="BayCAP with an empirical-Bayes prior on the recalls on main (E2.4).")
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
