"""The empirical-Bayes prior of BayCAP on all the binary datasets of ``quapy`` (work-plan card WP-7, E2.5 of
``docs/ext2_baycap.md``).

E2.5 reads the results of ``experiments/eb_binary.py`` (the $25$ binary datasets of ``quapy`` outside ``main``) together
with the $25$ binary pairs of ``main`` analysed by E2.4. On both it uses the four BayCAP methods of E2.4: ``BayesCAP``
and ``BayesCAP-EB`` target the population table $C\\operatorname{diag}(\\pi^Q)$, ``BayesCAP-real`` and
``BayesCAP-EB-real`` the realised table of the target sample; all four are scored against the realised table of each
bag. Vanilla accuracy, $|U|=1000$.

Each row is the per-pair table of :mod:`uncawe.tables.eb_prior` (scores, error terms, check, fit, NUTS diagnostics and
Winkler tests), with the column ``source`` (``E2.5`` or ``E2.4``) and the fit diagnostics of §T.10, computed from the
validation counts alone: $n^s_1$, $n^s_2$, the plain recalls $\\hat r_1$, $\\hat r_2$, their difference and the gap
statistic $z=(\\hat r_1-\\hat r_2)/\\sqrt{\\hat r_1(1-\\hat r_1)/n^s_1+\\hat r_2(1-\\hat r_2)/n^s_2}$. It also has the
predicted pull of §T.10, the mean over the bags of $\\sum_j\\pi_j(1-w_j)(\\hat m-\\hat r_j)$ with $w_j=n^s_j/(n^s_j+\\hat
\\nu)$, and the change of coverage from the $\\operatorname{Dir}(\\mathbf 1)$ to the empirical-Bayes prior within each
pushforward. Since $\\frac{f_{jj}+\\hat\\nu\\hat m}{n^s_j+\\hat\\nu}-\\hat r_j=(1-w_j)(\\hat m-\\hat r_j)$, the pull is
the prior part $R_0$ of the empirical-Bayes methods. The check, the error terms and the pull use the true class
proportions and recalls of the bags, so they are an analysis, not a method.

Each row also has the size of the test pool $U$ from which the bags are drawn (``n_test``), the smaller of the two
Beta parameters of the empirical-Bayes prior, $\\min(\\hat\nu\\hat m,\\hat\nu(1-\\hat m))$ (``eb_alpha_min``), and the
ratio of the realised amplitudes, empirical Bayes over $\\operatorname{Dir}(\\mathbf 1)$ (``realised_amplitude_ratio``).
:func:`analysis_summary` reduces the table to the summaries of the Analysis of E2.5, per source and per pool size
($|U|$ below or above ``SMALL_POOL``), which ``--tables`` prints.

The Winkler tests are Holm-adjusted within each source: over the pairs of E2.5 for the new rows, over the $75$ pairs of
E2.4 for the rows of ``main``, which therefore keep the p-values of the E2.4 table.

Outputs, under ``<OUT_DIR>/uncawe/tables/eb_binary/``: a per-pair CSV with every column, and two tables, each as a
LaTeX tabular fragment, a longtable and a standalone landscape document: the comparison table (as in E2.4) and the fit
table.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.eb_binary import EXPERIMENT
from uncawe.methods.baycap import BetaBinomialFit, fit_beta_binomial
from uncawe.pretrain.classic_binary import DOMAIN
from uncawe.pretrain.data import PretrainInfo, load_info_paths
from uncawe.tables import eb_prior, estimand_gap, pushforward
from uncawe.tables import main as main_tables
from uncawe.tables.displacement import load_validation_counts, plain_recalls
from uncawe.tables.estimand_gap import CLASSIFIER_ORDER, booktabs_clines
from uncawe.util import decorate_dataset

TABLE_NAME = "eb_binary"
KEYS = eb_prior.KEYS
SOURCES = ("E2.5", "E2.4")
E24_MARK = "$^\\ddagger$"
SMALL_POOL = 100  # test pools with fewer instances are read apart (cause 4, §T.13)
Z_BANDS = ((0, 1), (1, 2), (2, np.inf))


def gap_statistic(recalls: np.ndarray, n_s: np.ndarray) -> float:
    """Return $z=(\\hat r_1-\\hat r_2)/\\sqrt{\\sum_j\\hat r_j(1-\\hat r_j)/n^s_j}$ for two classes.

    With both recalls in $\\{0,1\\}$ the standard error is zero: $z$ is NaN if the recalls are equal and $\\pm\\infty$
    otherwise.
    """
    recalls, n_s = np.asarray(recalls, dtype=float), np.asarray(n_s, dtype=float)
    diff = recalls[0] - recalls[1]
    se = np.sqrt(np.sum(recalls * (1 - recalls) / n_s))
    if se == 0:
        return np.nan if diff == 0 else float(np.sign(diff) * np.inf)
    return float(diff / se)


def pull(val_counts: np.ndarray, true_props: np.ndarray, fit: BetaBinomialFit | None = None) -> float:
    """Return $\\sum_j\\pi_j(1-w_j)(\\hat m-\\hat r_j)$ for one bag, with the fit of ``BayCAP.fit`` on ``val_counts``.

    ``val_counts`` is the validation confusion table in the ``cap-ml`` layout (true classes on rows), ``true_props`` the
    true class proportions $\\pi$ of the bag; ``fit`` is the fit of ``val_counts``, computed if not given.
    """
    val_counts = np.asarray(val_counts, dtype=float)
    n_s = val_counts.sum(axis=1)
    fit = fit_beta_binomial(np.diag(val_counts), n_s) if fit is None else fit
    shrinkage = 1 - n_s / (n_s + fit.nu)
    return float(np.sum(np.asarray(true_props, dtype=float) * shrinkage * (fit.m - plain_recalls(val_counts))))


def fit_diagnostics(val_counts: np.ndarray) -> dict:
    """Return $n^s_1$, $n^s_2$, $\\hat r_1$, $\\hat r_2$, $|\\hat r_1-\\hat r_2|$ and $z$ of a binary validation table."""
    val_counts = np.asarray(val_counts, dtype=float)
    if val_counts.shape != (2, 2):
        raise ValueError(f"E2.5 needs binary tables, got shape {val_counts.shape}")
    n_s, recalls = val_counts.sum(axis=1), plain_recalls(val_counts)
    return {
        "n_val_1": int(n_s[0]),
        "n_val_2": int(n_s[1]),
        "recall_1": float(recalls[0]),
        "recall_2": float(recalls[1]),
        "recall_gap": float(abs(recalls[0] - recalls[1])),
        "gap_z": gap_statistic(recalls, n_s),
    }


def binary_columns(results: pd.DataFrame, val_counts: dict) -> pd.DataFrame:
    """Return, per (dataset, classifier) of ``results`` (any of the four methods: the bags are the same), the fit
    diagnostics and the predicted pull averaged over the bags."""
    rows = []
    for (dataset, classifier), _df in results.groupby(KEYS, sort=False):
        counts = val_counts[(dataset, classifier)]
        fit = fit_beta_binomial(np.diag(counts), counts.sum(axis=1))
        pulls = [pull(counts, np.asarray(ct, dtype=float).sum(axis=1), fit) for ct in _df["true_cts"]]
        rows.append({"dataset": dataset, "classifier": classifier, **fit_diagnostics(counts), "pull": np.mean(pulls)})
    return pd.DataFrame(rows)


def load_pool_sizes(pairs, domain: str) -> dict:
    """Return the size of the test pool $U$ of the given (dataset, classifier) pairs of ``domain``."""
    infos = {}
    for path in load_info_paths(domain=domain):
        p = PretrainInfo.load(path, fast=True)
        infos[(p.d_info.name, p.h_info.name)] = p
    missing = [pair for pair in pairs if pair not in infos]
    if missing:
        raise FileNotFoundError(f"No pretrain info found for {missing}")
    return {pair: len(infos[pair].load_dataset_bundle().U) for pair in pairs}


def extend(
    table: pd.DataFrame, results: pd.DataFrame, val_counts: dict, source: str, pool_sizes: dict | None = None
) -> pd.DataFrame:
    """Add to an E2.4 per-pair table the columns of E2.5: ``source``, the fit diagnostics, the pull, the change of
    coverage within each pushforward (empirical Bayes minus $\\operatorname{Dir}(\\mathbf 1)$), the smaller Beta
    parameter of the empirical-Bayes prior, the ratio of the realised amplitudes and, if given, the pool size."""
    table = table.merge(binary_columns(results, val_counts), on=KEYS, how="left")
    table.insert(2, "source", source)
    for name, (uniform, eb) in eb_prior.PUSHFORWARDS.items():
        table[f"{name}_coverage_change"] = table[f"{eb}_coverage"] - table[f"{uniform}_coverage"]
    table["eb_alpha_min"] = table["eb_nu"] * np.minimum(table["eb_m"], 1 - table["eb_m"])
    table["realised_amplitude_ratio"] = table["BayesCAP-EB-real_amplitude"] / table["BayesCAP-real_amplitude"]
    pairs = zip(table["dataset"].astype(str), table["classifier"].astype(str))
    table["n_test"] = [np.nan if pool_sizes is None else pool_sizes[pair] for pair in pairs]
    return table


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Order the pairs of E2.5 before those of E2.4, datasets by decreasing $|V|$, classifiers as in ``main``."""
    table = table.copy()
    source_rank = table["source"].map({source: i for i, source in enumerate(SOURCES)})
    keys = table.assign(source_rank=source_rank).drop_duplicates("dataset")
    order = keys.sort_values(["source_rank", "n_val"], ascending=[True, False])["dataset"].tolist()
    table["dataset"] = pd.Categorical(table["dataset"], categories=order, ordered=True)
    table["classifier"] = pd.Categorical(table["classifier"], categories=CLASSIFIER_ORDER, ordered=True)
    return table.sort_values(["dataset", "classifier"]).reset_index(drop=True)


def _source_table(experiment: str, domain: str, source: str, binary_only: bool) -> pd.DataFrame:
    table = eb_prior.compute_pair_table(experiment=experiment, domain=domain)
    if binary_only:
        table = table.loc[table["n_classes"] == 2].reset_index(drop=True)
    pairs = list(zip(table["dataset"].astype(str), table["classifier"].astype(str)))
    if not pairs:
        raise RuntimeError(f"No binary pair with stored results for all of {eb_prior.METHODS} in {experiment}.")
    results = eb_prior.load_results("BayesCAP", experiment=experiment, domain=domain)
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    results = results.loc[pd.MultiIndex.from_frame(results[KEYS]).isin(index)].reset_index(drop=True)
    table = table.assign(dataset=table["dataset"].astype(str), classifier=table["classifier"].astype(str))
    return extend(
        table, results, load_validation_counts(pairs, domain=domain), source, load_pool_sizes(pairs, domain=domain)
    )


def compute_table() -> pd.DataFrame:
    """Return the ordered per-pair table of E2.5 and of the binary pairs of E2.4."""
    new = _source_table(EXPERIMENT, DOMAIN, "E2.5", binary_only=False)
    main = _source_table(eb_prior.EXPERIMENT, estimand_gap.DOMAIN, "E2.4", binary_only=True)
    return ordered(pd.concat([new, main], ignore_index=True))


def _winkler_tests(sub: pd.DataFrame) -> tuple[int, int]:
    """Number of pairs where the realised Winkler score of empirical Bayes is significantly lower and higher."""
    significant = sub["realised_winkler_p_holm"] < pushforward.WINKLER_TEST_LEVEL
    lower = sub["realised_winkler_diff_mean"] < 0
    return int((significant & lower).sum()), int((significant & ~lower).sum())


def _groups(table: pd.DataFrame):
    """Yield (label, rows): each source, and the pairs of E2.5 split by pool size."""
    for source in SOURCES:
        sub = table.loc[table["source"] == source]
        if sub.empty:
            continue
        yield source, sub
        if source == "E2.5" and sub["n_test"].notna().all():
            yield f"E2.5, |U| >= {SMALL_POOL}", sub.loc[sub["n_test"] >= SMALL_POOL]
            yield f"E2.5, |U| < {SMALL_POOL}", sub.loc[sub["n_test"] < SMALL_POOL]


def analysis_summary(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the summaries of the Analysis of E2.5, one row per group of :func:`_groups`.

    - ``scores``: realised coverage of the two priors (mean, pairs below 0.9 and 0.5), mean realised Winkler scores,
      mean change of realised coverage, and the Winkler tests;
    - ``uniform_failures``: on the pairs where ``BayesCAP-real`` covers less than 0.9, how many underestimate accuracy
      (negative shift), and the means of the shift, the recall term $R$, its prior part $R_0$ and the prevalence term;
    - ``bound``: per band of $|z|$ (and ``z undefined``), the pairs, those with $\\hat\\nu$ at a bound, the mean
      change of realised coverage and the Winkler tests;
    - ``finite_nu``: on the pairs with $\\hat\\nu$ not at a bound, the correlation and slope of the observed against
      the predicted change of $R$, the Spearman correlation of $|\\Delta R|$ with $|\\hat r_1-\\hat r_2|$, the
      mean change of realised coverage, the Winkler tests and the median amplitude ratio, also on the pairs where
      empirical Bayes is significantly worse;
    - ``divergences``: the mean share of bags with divergent transitions of the two realised methods, the pairs where
      it exceeds 0.1 for ``BayesCAP-EB-real`` and how many of these have ``eb_alpha_min`` below 1, and the pairs with
      ``eb_alpha_min`` below 1.
    """
    out = {name: [] for name in ("scores", "uniform_failures", "bound", "finite_nu", "divergences")}
    for label, sub in _groups(table):
        uniform, eb = sub["BayesCAP-real_coverage"], sub["BayesCAP-EB-real_coverage"]
        change = sub["realised_coverage_change"]
        lower, higher = _winkler_tests(sub)
        out["scores"].append(
            {
                "group": label,
                "pairs": len(sub),
                "uniform_cov": uniform.mean(),
                "uniform_below_0.9": int((uniform < 0.9).sum()),
                "uniform_below_0.5": int((uniform < 0.5).sum()),
                "eb_cov": eb.mean(),
                "eb_below_0.5": int((eb < 0.5).sum()),
                "uniform_W": sub["BayesCAP-real_winkler"].mean(),
                "eb_W": sub["BayesCAP-EB-real_winkler"].mean(),
                "cov_change": change.mean(),
                "eb_W_lower": lower,
                "eb_W_higher": higher,
            }
        )
        failing = sub.loc[uniform < 0.9]
        out["uniform_failures"].append(
            {
                "group": label,
                "pairs": len(failing),
                "negative_shift": int((failing["BayesCAP-real_shift"] < 0).sum()),
                "shift": failing["BayesCAP-real_shift"].mean(),
                "R": failing["BayesCAP-real_recall_term"].mean(),
                "R0": failing["BayesCAP-real_prior_part"].mean(),
                "P": failing["BayesCAP-real_prevalence_term"].mean(),
            }
        )
        abs_z, at_bound = sub["gap_z"].abs(), sub["eb_nu_at_bound"].astype(bool)
        bands = [(f"{lo} <= |z| < {hi}", (abs_z >= lo) & (abs_z < hi)) for lo, hi in Z_BANDS]
        for band, mask in bands + [("z undefined", abs_z.isna())]:
            lower, higher = _winkler_tests(sub.loc[mask])
            out["bound"].append(
                {
                    "group": label,
                    "band": band,
                    "pairs": int(mask.sum()),
                    "at_bound": int((mask & at_bound).sum()),
                    "cov_change": change[mask].mean(),
                    "eb_W_lower": lower,
                    "eb_W_higher": higher,
                }
            )
        finite = sub.loc[~at_bound]
        predicted, observed = finite["predicted_recall_change"], finite["observed_recall_change"]
        lower, higher = _winkler_tests(finite)
        enough = len(finite) > 2  # correlations and slopes need at least three pairs
        worse = finite.loc[
            (finite["realised_winkler_p_holm"] < pushforward.WINKLER_TEST_LEVEL)
            & (finite["realised_winkler_diff_mean"] > 0)
        ]
        out["finite_nu"].append(
            {
                "group": label,
                "pairs": len(finite),
                "corr_dR": predicted.corr(observed) if enough else np.nan,
                "slope_dR": np.polyfit(predicted, observed, 1)[0] if enough else np.nan,
                "spearman_gap_dR": finite["recall_gap"].corr(observed.abs(), method="spearman") if enough else np.nan,
                "cov_change": finite["realised_coverage_change"].mean(),
                "eb_W_lower": lower,
                "eb_W_higher": higher,
                "amplitude_ratio": finite["realised_amplitude_ratio"].median(),
                "amplitude_ratio_worse": worse["realised_amplitude_ratio"].median(),
            }
        )
        heavy = sub["BayesCAP-EB-real_share_divergent"] > 0.1
        small_alpha = sub["eb_alpha_min"] < 1
        out["divergences"].append(
            {
                "group": label,
                "uniform_share": sub["BayesCAP-real_share_divergent"].mean(),
                "eb_share": sub["BayesCAP-EB-real_share_divergent"].mean(),
                "eb_above_0.1": int(heavy.sum()),
                "eb_above_0.1_alpha_below_1": int((heavy & small_alpha).sum()),
                "alpha_below_1": int(small_alpha.sum()),
            }
        )
    return {name: pd.DataFrame(rows) for name, rows in out.items()}


COMPARISON_CAPTION = (
    "BayCAP with the $\\operatorname{Dir}(\\mathbf 1)$ and the empirical-Bayes (EB) prior on the columns of $C$, "
    "population and realised pushforwards, on the binary datasets of \\texttt{quapy} (vanilla accuracy, $|U|=1000$); "
    f"{E24_MARK}: binary datasets of \\texttt{{main}} (E2.4). Coverage $\\mathcal{{C}}$, amplitude $\\mathcal{{A}}$ and "
    "Winkler score $\\mathcal{W}$ against the realised table, means over the 100 bags. Tests: 95\\% paired-bootstrap "
    "interval of the mean per-bag Winkler difference, EB minus $\\operatorname{Dir}(\\mathbf 1)$, within each "
    "pushforward, and its two-sided p-value, Holm-adjusted over the pairs of E2.5 (over the 75 pairs of E2.4 for "
    f"{E24_MARK})."
)
FIT_CAPTION = (
    "The empirical-Bayes fit on the binary datasets of \\texttt{quapy} (vanilla accuracy, $|U|=1000$); "
    f"{E24_MARK}: binary datasets of \\texttt{{main}} (E2.4). From $V$ alone: $n^s_j$ and plain recalls $\\hat r_j$, "
    "gap statistic $z$, fit $\\hat m$, $\\hat\\nu$ ($^*$ at a bound), $\\operatorname{sd}(\\hat m)$ and "
    "$\\sigma_{\\mathrm{lo}}$, the standard deviation that EB leaves out of the interval. Means over the bags: $R_0$ "
    "of BayCAP ($\\operatorname{Dir}(\\mathbf 1)$), the pull of EB (equal to $R_0$ of BayCAP-EB), the change of $R$ "
    "predicted by the check and observed (BayCAP-EB minus BayCAP), and the change of realised coverage (BayCAP-EB-real "
    "minus BayCAP-real). Share of bags with divergent transitions for the two EB methods."
)
COMPARISON_LABEL = "tab:eb-binary-comparison"
FIT_LABEL = "tab:eb-binary-fit"


def _dataset_labels(table: pd.DataFrame) -> pd.Series:
    names = table["dataset"].astype(str).map(lambda d: decorate_dataset(main_tables.dataset_map.get(d, d)))
    return names + np.where(table["source"] == "E2.4", E24_MARK, "")


def _index(table: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [_dataset_labels(table), table["classifier"].astype(str)], names=["dataset", "classifier"]
    )


def _fmt_z(value: float) -> str:
    return "--" if np.isnan(value) else ("$\\pm\\infty$" if np.isinf(value) else f"{value:.2f}")


def _comparison_styler(table: pd.DataFrame):
    out = eb_prior._comparison_styler(table).data.copy()
    out.index = _index(table)
    return out.style.format(precision=3, na_rep="--")


def _fit_styler(table: pd.DataFrame):
    cols = {
        ("$V$", "$n^s_1$"): table["n_val_1"].astype(int).to_numpy(),
        ("$V$", "$n^s_2$"): table["n_val_2"].astype(int).to_numpy(),
        ("$V$", "$\\hat r_1$"): table["recall_1"].to_numpy(),
        ("$V$", "$\\hat r_2$"): table["recall_2"].to_numpy(),
        ("$V$", "$z$"): [_fmt_z(v) for v in table["gap_z"]],
        ("fit", "$\\hat m$"): [eb_prior._fmt_fit(v, b, 3) for v, b in zip(table["eb_m"], table["eb_m_at_bound"])],
        ("fit", "$\\hat\\nu$"): [eb_prior._fmt_fit(v, b, 1) for v, b in zip(table["eb_nu"], table["eb_nu_at_bound"])],
        ("fit", "$\\operatorname{sd}(\\hat m)$"): table["eb_sd_m"].to_numpy(),
        ("fit", "$\\sigma_{\\mathrm{lo}}$"): table["left_out_sd"].to_numpy(),
        ("bags", "$R_0$"): table["BayesCAP_prior_part"].to_numpy(),
        ("bags", "pull"): table["pull"].to_numpy(),
        ("bags", "$\\Delta R$ pred."): table["predicted_recall_change"].to_numpy(),
        ("bags", "$\\Delta R$ obs."): table["observed_recall_change"].to_numpy(),
        ("bags", "$\\Delta\\mathcal{C}$ real."): table["realised_coverage_change"].to_numpy(),
    }
    formats = {}
    for method in ("BayesCAP-EB", "BayesCAP-EB-real"):
        key = ("div.", eb_prior.METHOD_LABELS[method])
        cols[key] = table[f"{method}_share_divergent"].to_numpy()
        formats[key] = "{:.2f}"
    out = pd.DataFrame(cols)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index = _index(table)
    styler = out.style.format(precision=3, na_rep="--")
    for key, fmt in formats.items():
        styler = styler.format(fmt, subset=[key], na_rep="--")
    return styler


TABLES = {
    "comparison": (_comparison_styler, COMPARISON_CAPTION, COMPARISON_LABEL),
    "fit": (_fit_styler, FIT_CAPTION, FIT_LABEL),
}


def _stem(name: str) -> str:
    return f"{TABLE_NAME}_{name}"


def _table_dir(pargs) -> str:
    return pargs.table_dir if pargs.table_dir is not None else os.path.join(env.root_dir, "tables", TABLE_NAME)


def _print_summary(table: pd.DataFrame):
    for source in SOURCES:
        sub = table.loc[table["source"] == source]
        if sub.empty:
            continue
        at_bound = sub["eb_nu_at_bound"].astype(bool)
        abs_z = sub["gap_z"].abs()
        change = sub["realised_coverage_change"]
        significant = sub["realised_winkler_p_holm"] < pushforward.WINKLER_TEST_LEVEL
        lower = sub["BayesCAP-EB-real_winkler"] < sub["BayesCAP-real_winkler"]
        small_z = abs_z < 1
        print(
            f"{source}: {len(sub)} pairs; nu at the bound on {int((at_bound & small_z).sum())} of the "
            f"{int(small_z.sum())} pairs with |z| < 1 and on {int((at_bound & ~small_z).sum())} of the "
            f"{int((~small_z).sum())} others; realised coverage change: mean {change.mean():+.3f}, below -0.05 on "
            f"{int((change < -0.05).sum())}, above +0.05 on {int((change > 0.05).sum())}; realised Winkler (Holm p < "
            f"{pushforward.WINKLER_TEST_LEVEL}): lower on {int((lower & significant).sum())}, higher on "
            f"{int((~lower & significant).sum())}; correlation of the predicted and the observed change of R "
            f"{sub['predicted_recall_change'].corr(sub['observed_recall_change']):.3f}; pairs with divergences (EB, "
            f"EB-real): {int((sub['BayesCAP-EB_share_divergent'] > 0).sum())}, "
            f"{int((sub['BayesCAP-EB-real_share_divergent'] > 0).sum())}"
        )


def gen_tables(pargs):
    table = compute_table()
    out_dir = _table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)

    table.to_csv(os.path.join(out_dir, f"{TABLE_NAME}.csv"), index=False)
    for name, (styler_fn, caption, label) in TABLES.items():
        styler = styler_fn(table)
        with open(os.path.join(tabular_dir, f"{_stem(name)}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{_stem(name)}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, caption, label))
        with open(os.path.join(out_dir, f"{_stem(name)}_doc.tex"), "w") as f:
            f.write(eb_prior._latex_document(f"tabular/{_stem(name)}_long.tex"))

    _print_summary(table)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        for name, summary in analysis_summary(table).items():
            print(f"\n{name}:\n{summary.to_string(index=False)}")
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
    parser = ArgumentParser(description="The empirical-Bayes prior of BayCAP on the binary datasets of quapy (E2.5).")
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
