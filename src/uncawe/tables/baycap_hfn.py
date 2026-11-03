"""BayCAP-Hfn, BayCAP-Hf with non-centred shares of the mistakes, on the pairs of ``main`` (work-plan card WP-12,
E2.9 of ``docs/ext2_baycap.md``).

``BayesCAP-Hfn-real`` is ``methods/baycap.py:BayCAP`` with ``c_prior="hierarchical-f"`` and ``shares="non-centred"``
(§T.17): the model of ``BayesCAP-Hf-real`` (E2.8), with the shares of the mistakes of each true class built from
$G'_{ij}\\sim\\operatorname{Gamma}(\\kappa+1,1)$ and $e_{ij}\\sim\\operatorname{Exp}(1)$, $\\ln g_{ij}=\\ln G'_{ij}-e_{ij}/\\kappa$,
normalised. It is compared bag by bag with the stored ``BayesCAP-Hf-real``, and its scores are shown next to those of
``BayesCAP-H-real``, ``BayesCAP-EB-real`` and ``BayesCAP-real``. All target the realised table of the target sample and
are scored against it (``true_cts``). Vanilla accuracy, $|U|=1000$, the pairs where the five have stored results (so it
also works on the pilot).

Per pair:

- the comparison: AE, coverage, amplitude and Winkler score of the five methods (means over the bags; for a
  hierarchical method, over its bags with draws), and the paired bootstrap tests of the mean per-bag difference of AE
  and of the Winkler score, BayCAP-Hfn minus BayCAP-Hf (``tables/pushforward.py:bootstrap_mean_test``), on the bags
  where neither failed, each Holm-adjusted over the pairs;
- the sampler of BayCAP-Hfn and BayCAP-Hf: the share of bags with divergent transitions, the mean number of divergent
  transitions per bag and the share of divergent draws (``tables/baycap_hf.py:divergence_load``), the largest
  split-$\\hat R$, the smallest effective sample size, the failed bags, the mean time per bag (``t_test_ave``), and the
  failure class of E2.8 (``tables/baycap_hf.py:failure_class``) of each;
- the posterior of the runs: means over the bags of the posterior mean of $m$, of the median of $\\ln\\nu$, of
  $P(\\nu>\\min_jn_j)$, of the median of $\\ln\\kappa$ and of $P(\\kappa<1)$, for both;
- from $V$, the indicators of the check of E2.9: the share of empty off-diagonal cells, the number of columns with no
  observed mistake ($f_{jj}=n_j$), the median over the columns of the observed mistakes per off-diagonal cell,
  $(n_j-f_{jj})/(K-1)$, and the classes perfect and never right.

The check of the hypothesis of E2.8 (``--check``) needs no run of BayCAP-Hfn: it reads the CSV of
``tables/baycap_hf.py`` and the validation counts, and gives, for the pairs with $K\\ge3$, the indicators from $V$ next
to the share of divergent draws of BayCAP-Hf, with their Spearman correlations.

The scores use the true tables, so they are analyses. Outputs, under ``<OUT_DIR>/uncawe/tables/baycap_hfn/``: a
per-pair CSV with every column, which ``plots/baycap_hfn.py`` reads, the CSV of the check, and two tables (comparison;
sampler), each as a LaTeX tabular fragment, a longtable and a standalone landscape document. Validation counts and
tables follow the ``cap-ml`` layout (true classes on rows).
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.main import EXPERIMENT
from uncawe.tables import baycap_h, eb_prior, estimand_gap, pushforward
from uncawe.tables import baycap_hf as hf
from uncawe.tables import hierarchical as h
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import METHOD_LABELS, load_validation_counts
from uncawe.tables.estimand_gap import SEED, booktabs_clines, load_results, pair_rng

TABLE_NAME = "baycap_hfn"
KEYS = eb_prior.KEYS
DOMAIN = estimand_gap.DOMAIN
METHOD = "BayesCAP-Hfn-real"
HF_METHOD = hf.METHOD
METHODS = [hf.UNIFORM_METHOD, hf.EB_METHOD, hf.H_METHOD, HF_METHOD, METHOD]
# the two samplers of the model of E2.8, with their column prefixes
SAMPLERS = {METHOD: "hfn", HF_METHOD: "hf"}
SCORES = hf.SCORES
TESTED_SCORES = hf.TESTED_SCORES
POSTERIOR = ("h_m_mean", "h_ln_nu_median", "h_p_nu_above_min_n", "h_ln_kappa_median", "h_p_kappa_below_1")
CHECK_INDICATORS = ("empty_off_diagonal_share", "n_no_mistake_columns", "median_mistakes_per_cell")


def test_column(score: str, stat: str) -> str:
    """Name of a test column, BayCAP-Hfn minus BayCAP-Hf: ``hf_<score>_<stat>``, e.g. ``hf_winkler_p_holm``."""
    return f"hf_{score}_{stat}"


def mistake_indicators(val_counts: np.ndarray) -> dict:
    """The indicators of the check of E2.9 from a validation table (true classes on rows): the number of columns of
    $C$ with no observed mistake ($f_{jj}=n_j$) and the median over the columns of $(n_j-f_{jj})/(K-1)$."""
    f, n = h.diagonal_counts(val_counts)
    mistakes = n - f
    return {
        "n_no_mistake_columns": int((mistakes == 0).sum()),
        "median_mistakes_per_cell": float(np.median(mistakes / (len(n) - 1))),
    }


def is_unreliable(table: pd.DataFrame, prefix: str) -> pd.Series:
    """The pairs where the sampler with column prefix ``prefix`` has a failure class (empty in the table, NaN once read
    back from the CSV)."""
    return table[f"{prefix}_failure_class"].fillna("").astype(str) != ""


def _mean(df: pd.DataFrame, column: str, ok: np.ndarray) -> float:
    if column not in df or not ok.any():
        return np.nan
    return float(pd.to_numeric(df[column], errors="coerce")[ok].mean())


def pair_tests(bags: dict[str, pd.DataFrame], ok: np.ndarray, key: tuple[str, str]) -> dict:
    """The paired bootstrap tests of one pair, BayCAP-Hfn minus BayCAP-Hf, on the bags in ``ok``; without Holm."""
    new, old = bags[METHOD], bags[HF_METHOD]
    if len(new) != len(old) or not np.allclose(new["true_acc"].to_numpy(), old["true_acc"].to_numpy()):
        raise ValueError(f"The bags of {key} differ between {METHOD} and {HF_METHOD}")
    row = {}
    for score in TESTED_SCORES:
        diff = (new[score] - old[score]).to_numpy()[ok]
        if len(diff) == 0:
            low = high = p_value = mean = np.nan
        else:
            # the generator depends only on the seed, the pair and the score
            rng = pair_rng(*key, seed=SEED + 2000 + TESTED_SCORES.index(score))
            low, high, p_value = pushforward.bootstrap_mean_test(diff, rng)
            mean = float(diff.mean())
        row.update(
            {
                test_column(score, "diff_mean"): mean,
                test_column(score, "ci_low"): low,
                test_column(score, "ci_high"): high,
                test_column(score, "p"): p_value,
            }
        )
    return row


def sampler_columns(df: pd.DataFrame, failed: np.ndarray, prefix: str, val_counts: np.ndarray) -> dict:
    """The sampler, the posterior of the run and the failure class of one sampler on one pair."""
    ok = ~failed
    row = {f"{prefix}_share_failed": float(failed.mean())}
    diagnostics = eb_prior.diagnostics_table(df).iloc[0]
    row.update({f"{prefix}_{name}": float(diagnostics[name]) for name in eb_prior.DIAGNOSTICS})
    row.update({f"{prefix}_{name}": value for name, value in hf.divergence_load(df, ok).items()})
    row[f"{prefix}_time_per_bag"] = _mean(df, "t_test_ave", np.ones(len(df), dtype=bool))
    row.update({f"{prefix}_{name.removeprefix('h_')}": _mean(df, name, ok) for name in POSTERIOR})
    f, n = h.diagonal_counts(val_counts)
    row[f"{prefix}_failure_class"] = hf.failure_class(
        row[f"{prefix}_share_divergent_draws"],
        row[f"{prefix}_max_rhat"],
        row[f"{prefix}_p_nu_above_min_n"],
        int((f == n).sum() + (f == 0).sum()),
        row[f"{prefix}_ln_kappa_median"],
        hf.empty_off_diagonal_share(val_counts),
    )
    return row


def pair_row(results: dict[str, pd.DataFrame], val_counts: np.ndarray, key: tuple[str, str]) -> dict:
    """The per-pair columns: scores, tests (without Holm), sampler and posterior of both samplers, $V$.

    ``results`` maps each method of ``METHODS`` to its stored results on the pair, in the stored order of the bags.
    """
    f, n = h.diagonal_counts(val_counts)
    failed = {method: baycap_h.failed_bags(results[method]) for method in (*SAMPLERS, hf.H_METHOD)}
    row = {
        "n_classes": int(val_counts.shape[0]),
        "min_val_per_class": int(n.min()),
        "n_bags": len(results[METHOD]),
        "empty_off_diagonal_share": hf.empty_off_diagonal_share(val_counts),
        **mistake_indicators(val_counts),
        "n_perfect_classes": int((f == n).sum()),
        "n_never_right_classes": int((f == 0).sum()),
    }
    bags = {}
    for method in METHODS:
        df = results[method]
        ok = ~failed[method] if method in failed else np.ones(len(df), dtype=bool)
        bags[method] = hf.bag_frame(df, method)
        kept = bags[method].loc[ok]
        for score in SCORES:
            row[f"{method}_{score}"] = float(kept[score].mean()) if len(kept) else np.nan
    for method, prefix in SAMPLERS.items():
        row.update(sampler_columns(results[method], failed[method], prefix, val_counts))
    row.update(pair_tests(bags, ~failed[METHOD] & ~failed[HF_METHOD], key))
    return row


def pair_table(results: dict[str, pd.DataFrame], val_counts: dict) -> pd.DataFrame:
    """One row per pair of ``results`` (each method restricted to the same pairs), ordered by $K$; the p-values of
    each test are Holm-adjusted over the pairs."""
    rows = []
    for (dataset, classifier), _ in results[METHOD].groupby(KEYS, sort=False):
        pair_results = {
            method: df.loc[(df["dataset"] == dataset) & (df["classifier"] == classifier)].reset_index(drop=True)
            for method, df in results.items()
        }
        key = (dataset, classifier)
        rows.append({"dataset": dataset, "classifier": classifier, **pair_row(pair_results, val_counts[key], key)})
    table = pd.DataFrame(rows)
    for score in TESTED_SCORES:
        table[test_column(score, "p_holm")] = pushforward.holm(table[test_column(score, "p")].to_numpy())
    return h.ordered(table)


def compute_table() -> pd.DataFrame:
    """Load the stored results of the five methods and the validation counts, and return the ordered table."""
    results = {}
    for method in METHODS:
        df = load_results(method, experiment=EXPERIMENT, domain=DOMAIN)
        if df.empty:
            raise RuntimeError(f"No {method} results found for {EXPERIMENT}.")
        results[method] = df
    pairs = eb_prior.common_pairs(results)
    if not pairs:
        raise RuntimeError(f"No pair of {EXPERIMENT} has stored results for all of {METHODS}.")
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    results = {
        method: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
        for method, df in results.items()
    }
    return pair_table(results, load_validation_counts(pairs, domain=DOMAIN))


def check_table(hf_table: pd.DataFrame, val_counts: dict) -> pd.DataFrame:
    """The check of the hypothesis of E2.8 (E2.9, no run): for the pairs of the E2.8 CSV with $K\\ge3$, the indicators
    from $V$ next to the share of divergent draws and the failure class of BayCAP-Hf."""
    multi = hf_table.loc[hf_table["n_classes"] > 2]
    rows = []
    for _, pair in multi.iterrows():
        key = (pair["dataset"], pair["classifier"])
        rows.append(
            {
                "dataset": key[0],
                "classifier": key[1],
                "n_classes": int(pair["n_classes"]),
                "min_val_per_class": int(pair["min_val_per_class"]),
                "empty_off_diagonal_share": hf.empty_off_diagonal_share(val_counts[key]),
                **mistake_indicators(val_counts[key]),
                "n_perfect_classes": int(pair["n_perfect_classes"]),
                "n_never_right_classes": int(pair["n_never_right_classes"]),
                "hf_share_divergent_draws": float(pair["hf_share_divergent_draws"]),
                "hf_failure_class": pair["failure_class"] if isinstance(pair["failure_class"], str) else "",
            }
        )
    return h.ordered(pd.DataFrame(rows))


def check_summary(check: pd.DataFrame) -> pd.DataFrame:
    """The Spearman correlation across pairs of each indicator of ``CHECK_INDICATORS`` with the share of divergent
    draws of BayCAP-Hf, and the means of the indicators over the pairs with and without a boundary class."""
    boundary = check["hf_failure_class"] == "boundary"
    rows = []
    for name in CHECK_INDICATORS:
        rows.append(
            {
                "indicator": name,
                "pairs": int(check[name].notna().sum()),
                "spearman_with_divergent_draws": float(
                    check[name].corr(check["hf_share_divergent_draws"], method="spearman")
                ),
                "mean_boundary": float(check.loc[boundary, name].mean()),
                "mean_other": float(check.loc[~boundary, name].mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_check() -> pd.DataFrame:
    """Read the CSV of ``tables/baycap_hf.py`` and the validation counts of its pairs, and return the check table."""
    path = hf.csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/baycap_hf.py --tables first.")
    hf_table = pd.read_csv(path)
    pairs = list(zip(hf_table["dataset"], hf_table["classifier"]))
    return check_table(hf_table, load_validation_counts(pairs, domain=DOMAIN))


def _sampler_counts(sub: pd.DataFrame, prefix: str) -> dict:
    return {
        f"{prefix}_not_mixing": int((sub[f"{prefix}_max_rhat"] > hf.RHAT_HIGH).sum()),
        f"{prefix}_above_draws": int((sub[f"{prefix}_share_divergent_draws"] > hf.DIVERGENT_DRAWS).sum()),
        f"{prefix}_unreliable": int(is_unreliable(sub, prefix).sum()),
        f"{prefix}_with_failed_bags": int((sub[f"{prefix}_share_failed"] > 0).sum()),
    }


def _scores_summary(sub: pd.DataFrame, group: str) -> dict:
    scores = {"pairs": group, "n": len(sub)}
    for method in METHODS:
        for score, label in (("ae", "AE"), ("coverage", "cov"), ("winkler", "W")):
            scores[f"{method}_{label}"] = float(sub[f"{method}_{score}"].mean())
    for score in TESTED_SCORES:
        significant = sub[test_column(score, "p_holm")] < pushforward.WINKLER_TEST_LEVEL
        lower = sub[test_column(score, "diff_mean")] < 0
        scores[f"{score}_lower_than_hf"] = int((lower & significant).sum())
        scores[f"{score}_higher_than_hf"] = int((~lower & significant).sum())
    return scores


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Reduce the per-pair table to the summaries of the Analysis of E2.9, by the class of BayCAP-Hf (E2.8).

    - ``scores``: mean AE, coverage and Winkler score of the five methods and the pairs where BayCAP-Hfn is
      significantly (Holm) lower and higher than BayCAP-Hf, over all the pairs, the pairs reliable under BayCAP-Hf and
      those with a boundary or a funnel class;
    - ``sampler``: in the same groups, for both samplers, the pairs that do not mix, with more than
      ``DIVERGENT_DRAWS`` of the draws divergent, unreliable, with failed bags; the pairs reliable under BayCAP-Hf that
      become unreliable, and the unreliable ones that become reliable; the mean time per bag of both;
    - ``posterior``: on the pairs reliable under both, the mean and the largest absolute difference between the two
      samplers of the medians of $\\ln\\kappa$ and of $\\ln\\nu$.
    """
    groups = [("all", table), ("reliable_hf", table.loc[~is_unreliable(table, "hf")])]
    for cls in ("boundary", "funnel"):
        groups.append((f"{cls}_hf", table.loc[table["hf_failure_class"].fillna("") == cls]))
    out = {"scores": [], "sampler": [], "posterior": []}
    for name, sub in groups:
        out["scores"].append(_scores_summary(sub, name))
        sampler = {"pairs": name, "n": len(sub)}
        for prefix in SAMPLERS.values():
            sampler.update(_sampler_counts(sub, prefix))
            sampler[f"{prefix}_time_per_bag"] = float(sub[f"{prefix}_time_per_bag"].mean())
        old, new = is_unreliable(sub, "hf"), is_unreliable(sub, "hfn")
        sampler["became_unreliable"] = int((~old & new).sum())
        sampler["became_reliable"] = int((old & ~new).sum())
        out["sampler"].append(sampler)
    both = table.loc[~is_unreliable(table, "hf") & ~is_unreliable(table, "hfn")]
    posterior = {"pairs": "reliable_both", "n": len(both)}
    for name in ("ln_kappa_median", "ln_nu_median"):
        diff = (both[f"hfn_{name}"] - both[f"hf_{name}"]).abs()
        posterior[f"{name}_mean_abs_diff"] = float(diff.mean()) if diff.notna().any() else np.nan
        posterior[f"{name}_max_abs_diff"] = float(diff.max()) if diff.notna().any() else np.nan
    out["posterior"].append(posterior)
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


COMPARISON_CAPTION = (
    "BayCAP-Hfn-real (BayCAP-Hf with non-centred shares of the mistakes, the same model) against the stored "
    "BayCAP-Hf-real, BayCAP-H-real, BayCAP-EB-real and BayCAP-real, realised pushforward, on \\texttt{main} (E2.9, "
    "vanilla accuracy, $|U|=1000$). AE, coverage $\\mathcal{C}$, amplitude $\\mathcal{A}$ and Winkler score "
    "$\\mathcal{W}$ against the realised table, means over the bags (for the hierarchical methods, over the bags where "
    "NUTS produced draws). Tests: mean per-bag difference of AE and of $\\mathcal{W}$, BayCAP-Hfn minus BayCAP-Hf, and "
    "its two-sided paired-bootstrap p-value, Holm-adjusted over the pairs."
)
SAMPLER_CAPTION = (
    "Sampler of BayCAP-Hfn-real and BayCAP-Hf-real on \\texttt{main} (E2.9, vanilla accuracy, $|U|=1000$): share of "
    f"divergent draws (of the {hf.NUTS_DRAWS} draws of a bag, over the bags with draws), largest split-$\\hat R$, "
    "smallest effective sample size, mean time per bag (s), failure class of E2.8 (more than 1\\% of the draws "
    "divergent or $\\hat R$ above 1.1); means over the bags of the median of $\\ln\\kappa$ and of $\\ln\\nu$. $V$: share "
    "of empty off-diagonal cells, columns with no observed mistake, median observed mistakes per off-diagonal cell."
)
COMPARISON_LABEL = "tab:baycap-hfn-comparison"
SAMPLER_LABEL = "tab:baycap-hfn-sampler"
SCORE_LABELS = {"ae": "AE", **{score: main_tables.metrics_map[score] for score in pushforward.SCORES}}
SAMPLER_LABELS = {"hfn": "Hfn", "hf": "Hf"}


def _comparison_styler(table: pd.DataFrame):
    cols = {}
    for method in METHODS:
        for score in SCORES:
            cols[(METHOD_LABELS[method], SCORE_LABELS[score])] = table[f"{method}_{score}"].to_numpy()
    for score in TESTED_SCORES:
        label = SCORE_LABELS[score]
        cols[("Hfn $-$ Hf", f"$\\Delta${label}")] = table[test_column(score, "diff_mean")].to_numpy()
        cols[("Hfn $-$ Hf", f"$p_{{{label.strip('$')}}}$")] = table[test_column(score, "p_holm")].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    return out.style.format(precision=3, na_rep="--")


def _sampler_styler(table: pd.DataFrame):
    cols = {}
    for prefix, label in SAMPLER_LABELS.items():
        group = f"NUTS {label}"
        cols[(group, "draws")] = table[f"{prefix}_share_divergent_draws"].to_numpy()
        cols[(group, "$\\hat R$")] = table[f"{prefix}_max_rhat"].to_numpy()
        cols[(group, "ESS")] = table[f"{prefix}_min_ess"].to_numpy()
        cols[(group, "time")] = table[f"{prefix}_time_per_bag"].to_numpy()
        cols[(group, "class")] = table[f"{prefix}_failure_class"].fillna("").to_numpy()
    for prefix, label in SAMPLER_LABELS.items():
        cols[(f"run {label}", "$\\ln\\kappa$")] = table[f"{prefix}_ln_kappa_median"].to_numpy()
        cols[(f"run {label}", "$\\ln\\nu$")] = table[f"{prefix}_ln_nu_median"].to_numpy()
    cols[("$V$", "empty")] = table["empty_off_diagonal_share"].to_numpy()
    cols[("$V$", "no mist.")] = table["n_no_mistake_columns"].to_numpy()
    cols[("$V$", "mist./cell")] = table["median_mistakes_per_cell"].to_numpy()
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = baycap_h._index(table)
    styler = out.style.format(precision=2, na_rep="--")
    for label in SAMPLER_LABELS.values():
        styler = styler.format("{:.3f}", subset=[(f"NUTS {label}", "draws")], na_rep="--")
        styler = styler.format("{:.3f}", subset=[(f"NUTS {label}", "$\\hat R$")], na_rep="--")
        styler = styler.format("{:.0f}", subset=[(f"NUTS {label}", "ESS")], na_rep="--")
    return styler.format("{:.0f}", subset=[("$V$", "no mist.")], na_rep="--")


TABLES = {
    "comparison": (_comparison_styler, COMPARISON_CAPTION, COMPARISON_LABEL),
    "sampler": (_sampler_styler, SAMPLER_CAPTION, SAMPLER_LABEL),
}


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}.csv")


def check_csv_path(directory: str | None = None) -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_check.csv")


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def _print_summaries(summaries: dict[str, pd.DataFrame]):
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in summaries.items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")


def gen_check(pargs):
    check = compute_check()
    out_dir = table_dir(pargs)
    os.makedirs(out_dir, exist_ok=True)
    check.to_csv(check_csv_path(out_dir), index=False)
    _print_summaries({"check": check_summary(check)})
    print(f"check written to {check_csv_path(out_dir)}")


def gen_tables(pargs):
    table = compute_table()
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    for name, (styler_fn, caption, label) in TABLES.items():
        styler = styler_fn(table)
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, caption, label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(eb_prior._latex_document(f"tabular/{_stem(name)}_long.tex"))
    _print_summaries(analysis_summary(table))
    print(f"tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
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
    parser = ArgumentParser(description="BayCAP-Hfn, the non-centred shares of BayCAP-Hf, on main (E2.9).")
    parser.add_argument("-c", "--check", action="store_true", help="The check of E2.9 on the E2.8 CSV (no run)")
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

    if pargs.check:
        gen_check(pargs)
    if pargs.tables or pargs.all:
        gen_tables(pargs)
    if pargs.pdf or pargs.all:
        gen_pdf(pargs)
