r"""The prior on $\pi^Q$ of BayCAP-Hf and BayCAP-KDEy (work-plan card WP-19, E5.1 of
``docs/ext5_baycap_priors_tempering.md``).

E5.1 (``experiments/binform.py --e51``) draws the bags of each pair of ``main`` at prevalences from
$G=\operatorname{Dir}(3K\mu^\star)$ and runs ``BayesCAP-Hf-real`` (BayCAP-Hf) and ``BayCAP-KDEy-mlp-real`` (BayCAP-KDEy
with an MLP input and a fixed $\hat C$) with three priors on $\pi^Q$: uniform, well specified ($=G$) and misspecified
($\operatorname{Dir}(3K\mu^\dagger)$). Both methods target the realised table of each bag and are scored against it
(``true_cts``); the scores use the target labels, so they are analyses. The pairs are those where all three priors of a
method are stored, for each bag size, so the table also works on the pilot.

Per pair (``pair_table``, one row per metric, bag size, method and pair, every column in the CSV): the means over the
bags of AE, coverage, amplitude and Winkler score at $\alpha=0.05$ for each prior; the value of the knowledge
$\Delta W_{\text{know}}=W_{\text{uniform}}-W_{\text{well}}$ and the cost of wrong knowledge
$\Delta W_{\text{wrong}}=W_{\text{wrong}}-W_{\text{uniform}}$ (both positive when they mean what their names say); the
ratio of the mean amplitudes of the well-specified and the uniform prior; per prior, the share of bags with divergent
transitions and of bags where NUTS produced no draws (left out of the scores). The bags of the three priors are checked
to be the same (equal stored realised tables).

Summary (``summary_table``, per metric, bag size, method, group of $K$ — all, $K=2$, $3\le K\le7$, $K\ge8$ — and prior):
the number of pairs, the means over the pairs of the four scores, the mean of the difference of the Winkler score from
that of the uniform prior ($-\Delta W_{\text{know}}$ for the well-specified prior, $\Delta W_{\text{wrong}}$ for the
misspecified one), the number of pairs where the prior has a lower Winkler score than the uniform one, the $p$-value of
the paired Wilcoxon signed-rank test (two-sided) over the per-pair means, and the median amplitude ratio to the uniform
prior. E5.1 has no decision rule.

Outputs, under ``<OUT_DIR>/uncawe/tables/binform_prior/``: the per-pair CSV (read by ``plots/binform_prior.py``), the
summary CSV and, per metric, the summary as a LaTeX tabular fragment, a longtable and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from uncawe import env
from uncawe.experiments.binform import E51_BASE_METHODS, E51_METHODS, E51_PRIORS, E51_SIZES, e51_experiment
from uncawe.experiments.metrics import acc_from_ct, get_acc_names
from uncawe.results import Results
from uncawe.tables import eb_prior, hierarchical
from uncawe.tables import main as main_tables
from uncawe.tables.estimand_gap import CLASSIFIER_ORDER, booktabs_clines

TABLE_NAME = "binform_prior"
DOMAIN = "classic"
KEYS = ["dataset", "classifier"]
SCORES = ("ae", "coverage", "amplitude", "winkler")
REFERENCE_PRIOR = "uniform"
OTHER_PRIORS = tuple(prior for prior in E51_PRIORS if prior != REFERENCE_PRIOR)
K_GROUPS = hierarchical.K_GROUPS


def method_name(base: str, prior: str) -> str:
    return f"{base}_{prior}"


assert all(method_name(base, prior) in E51_METHODS for base in E51_BASE_METHODS for prior in E51_PRIORS)


# --------------------------------------------------------------------------------------------------- loading


def load_method(method: str, acc_name: str, size: int, root: str | None = None) -> pd.DataFrame:
    """The stored rows of the E5.1 method ``method`` for ``acc_name`` and bags of ``size`` instances (empty if none)."""
    base_dir = os.path.join(env.root_dir if root is None else root, e51_experiment(size))
    try:
        return Results.load(
            base_dir=base_dir,
            domain=DOMAIN,
            acc_name=acc_name,
            filter_classifiers=list(CLASSIFIER_ORDER),
            filter_methods=[method],
        ).df
    except ValueError:  # no file: pd.concat of an empty list
        return pd.DataFrame()


def load_all(acc_name: str, root: str | None = None) -> dict[tuple[int, str], dict[str, pd.DataFrame]]:
    """``(size, base method) -> {prior: rows}``, restricted to the pairs stored for the three priors; the (size, base
    method) without any such pair are left out."""
    out = {}
    for size in E51_SIZES:
        for base in E51_BASE_METHODS:
            frames = {prior: load_method(method_name(base, prior), acc_name, size, root) for prior in E51_PRIORS}
            if any(df.empty for df in frames.values()):
                continue
            pairs = eb_prior.common_pairs(frames)
            if not pairs:
                continue
            index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
            out[(size, base)] = {
                prior: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
                for prior, df in frames.items()
            }
    return out


# --------------------------------------------------------------------------------------------------- scores


def bag_frame(df: pd.DataFrame, acc_name: str) -> pd.DataFrame:
    """One row per bag of one pair (stored order): the true value of the metric on the realised table, AE, coverage,
    amplitude and Winkler score, whether NUTS produced no draws (``failed``, scores NaN) and whether it had divergent
    transitions."""
    rows = []
    divergences = df["nuts_divergences"] if "nuts_divergences" in df else pd.Series(np.nan, index=df.index)
    for row, n_div in zip(df.itertuples(index=False), divergences):
        true_acc = acc_from_ct(acc_name, np.asarray(row.true_cts, dtype=float))
        samples = np.asarray(row.estim_acc_ranges, dtype=float)
        failed = not np.all(np.isfinite(samples))
        rows.append(
            {
                "true_acc": true_acc,
                "failed": failed,
                "divergent": bool(np.nan_to_num(float(n_div)) > 0),
                "ae": np.nan if failed else float(row.acc_err),
                "coverage": np.nan if failed else main_tables.get_acc_coverage("", true_acc, samples),
                "amplitude": np.nan if failed else main_tables.get_acc_amplitude("", samples),
                "winkler": np.nan if failed else main_tables.get_acc_winkler("", true_acc, samples),
            }
        )
    return pd.DataFrame(rows)


def _check_same_bags(frames: dict[str, pd.DataFrame], key) -> None:
    """The stored realised tables of the bags of one pair are the same for every prior."""
    reference = np.stack([np.ravel(ct) for ct in frames[REFERENCE_PRIOR]["true_cts"]])
    for prior, frame in frames.items():
        tables = np.stack([np.ravel(ct) for ct in frame["true_cts"]])
        if tables.shape != reference.shape or not np.allclose(tables, reference):
            raise ValueError(f"{key}: the bags of the {prior} prior differ from those of the uniform prior")


def pair_rows(frames: dict[str, pd.DataFrame], acc_name: str, size: int, base: str) -> list[dict]:
    """The rows of ``pair_table`` of one metric, bag size and base method (``frames``: prior -> stored rows)."""
    groups = {prior: dict(list(df.groupby(KEYS, sort=False))) for prior, df in frames.items()}
    rows = []
    for key, first in groups[REFERENCE_PRIOR].items():
        _check_same_bags({prior: groups[prior][key] for prior in E51_PRIORS}, key)
        bags = {prior: bag_frame(groups[prior][key], acc_name) for prior in E51_PRIORS}
        row = {
            "acc_name": acc_name,
            "size": size,
            "method": base,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": int(first["n_classes"].iloc[0]),
            "n_bags": len(first),
        }
        for prior in E51_PRIORS:
            for score in SCORES:
                row[f"{prior}_{score}"] = float(bags[prior][score].mean())
            row[f"{prior}_share_failed"] = float(bags[prior]["failed"].mean())
            row[f"{prior}_share_divergent"] = float(bags[prior]["divergent"].mean())
        row["d_w_know"] = row["uniform_winkler"] - row["well_winkler"]
        row["d_w_wrong"] = row["wrong_winkler"] - row["uniform_winkler"]
        for prior in OTHER_PRIORS:
            reference = row["uniform_amplitude"]
            row[f"{prior}_amplitude_ratio"] = row[f"{prior}_amplitude"] / reference if reference > 0 else np.nan
        rows.append(row)
    return rows


def pair_table(acc_names: list[str] | None = None, root: str | None = None) -> pd.DataFrame:
    """One row per metric, bag size, base method and pair, ordered by those keys, then by $K$, dataset and classifier
    (as in ``CLASSIFIER_ORDER``)."""
    rows = []
    for acc_name in acc_names or get_acc_names():
        for (size, base), frames in load_all(acc_name, root).items():
            rows.extend(pair_rows(frames, acc_name, size, base))
    if not rows:
        raise RuntimeError("No pair of E5.1 has stored results for the three priors of a method.")
    return ordered(pd.DataFrame(rows))


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["_acc"] = table["acc_name"].map(get_acc_names().index)
    table["_method"] = table["method"].map(E51_BASE_METHODS.index)
    table["_clf"] = table["classifier"].map(lambda c: CLASSIFIER_ORDER.index(c) if c in CLASSIFIER_ORDER else 99)
    table = table.sort_values(["_acc", "size", "_method", "n_classes", "dataset", "_clf"])
    return table.drop(columns=["_acc", "_method", "_clf"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- summary


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided $p$-value of the Wilcoxon signed-rank test of the pairs $(x_i,y_i)$; NaN with fewer than two finite
    nonzero differences."""
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    diff = diff[np.isfinite(diff)]
    if np.count_nonzero(diff) < 2:
        return np.nan
    return float(wilcoxon(diff).pvalue)


def k_groups(table: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("all", table)]
    for name in K_GROUPS:
        part = table.loc[table["n_classes"].map(hierarchical._k_group) == name]
        if not part.empty:
            groups.append((name, part))
    return groups


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric, bag size, base method, group of $K$ and prior: the summaries of the module docstring."""
    rows = []
    for (acc_name, size, base), sub in table.groupby(["acc_name", "size", "method"], sort=False):
        for k_group, part in k_groups(sub):
            uniform_w = part["uniform_winkler"].to_numpy(dtype=float)
            for prior in E51_PRIORS:
                row = {"acc_name": acc_name, "size": size, "method": base, "K": k_group, "prior": prior}
                row["pairs"] = len(part)
                for score in SCORES:
                    row[score] = float(part[f"{prior}_{score}"].mean())
                row["share_divergent"] = float(part[f"{prior}_share_divergent"].mean())
                row["share_failed"] = float(part[f"{prior}_share_failed"].mean())
                if prior == REFERENCE_PRIOR:
                    row |= {"d_winkler": np.nan, "pairs_lower": np.nan, "p_value": np.nan, "amplitude_ratio": np.nan}
                else:
                    w = part[f"{prior}_winkler"].to_numpy(dtype=float)
                    row["d_winkler"] = float(np.nanmean(w - uniform_w))
                    row["pairs_lower"] = int(np.sum(w < uniform_w))
                    row["p_value"] = paired_wilcoxon(w, uniform_w)
                    row["amplitude_ratio"] = float(part[f"{prior}_amplitude_ratio"].median())
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

METHOD_LABELS = {"BayesCAP-Hf-real": "\\bcap{}-Hf", "BayCAP-KDEy-mlp-real": "\\bcap{}-KDEy-mlp"}
PRIOR_LABELS = {"uniform": "uniform", "well": "well specified", "wrong": "misspecified"}
K_LABELS = {"all": "all", "K=2": "$K=2$", "3<=K<=7": "$3\\le K\\le7$", "K>=8": "$K\\ge8$"}
COLUMN_LABELS = {
    "pairs": "pairs",
    "ae": "AE",
    "coverage": main_tables.metrics_map["coverage"],
    "amplitude": main_tables.metrics_map["amplitude"],
    "winkler": main_tables.metrics_map["winkler"],
    "d_winkler": "$\\Delta\\mathcal{W}$",
    "pairs_lower": "lower",
    "p_value": "$p$",
    "amplitude_ratio": "$\\mathcal{A}/\\mathcal{A}_{\\mathrm{unif}}$",
    "share_divergent": "div.",
}


def _summary_styler(summary: pd.DataFrame):
    """One row per bag size, method, group of $K$ and prior; the columns of ``COLUMN_LABELS``."""
    out = pd.DataFrame({label: summary[column].to_numpy() for column, label in COLUMN_LABELS.items()})
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["size"].map(lambda s: f"$|U|={s}$"),
            summary["method"].map(METHOD_LABELS),
            summary["K"].map(K_LABELS),
            summary["prior"].map(PRIOR_LABELS),
        ],
        names=["bags", "method", "$K$", "prior"],
    )
    integers = [COLUMN_LABELS["pairs"], COLUMN_LABELS["pairs_lower"]]
    return out.style.format(precision=3, na_rep="--").format("{:.0f}", subset=integers, na_rep="--")


ACC_LABELS = {"vanilla_accuracy": "vanilla accuracy", "macro-F1": "macro-$F_1$", "macro-K": "macro-$K$"}


def caption(acc_name: str) -> str:
    metric = ACC_LABELS[acc_name]
    return (
        f"E5.1, the prior on $\\pi^Q$ ({metric}); bags at prevalences from $G=\\operatorname{{Dir}}(3K\\mu^\\star)$; "
        "means over the pairs of the per-pair means over the 100 bags, scored against the realised table. "
        "Uniform: $\\operatorname{Dir}(\\mathbf 1)$; well specified: $G$; misspecified: "
        "$\\operatorname{Dir}(3K\\mu^\\dagger)$. $\\Delta\\mathcal{W}$: mean over the pairs of the Winkler score minus "
        "that of the uniform prior; lower: pairs where it is negative; $p$: paired Wilcoxon signed-rank test over the "
        "pairs; $\\mathcal{A}/\\mathcal{A}_{\\mathrm{unif}}$: median over the pairs of the ratio of the mean "
        "amplitudes; div.: share of bags with divergent transitions."
    )


def _latex_document(longtable_path: str) -> str:
    return eb_prior._latex_document(longtable_path).replace(
        "\\begin{document}", "\\newcommand{\\bcap}{BayCAP}\n\\begin{document}", 1
    )


# --------------------------------------------------------------------------------------------------- CLI


def table_dir(pargs=None) -> str:
    if pargs is not None and pargs.table_dir is not None:
        return pargs.table_dir
    return os.path.join(env.root_dir, "tables", TABLE_NAME)


def csv_path(directory: str | None = None, name: str = "pairs") -> str:
    return os.path.join(directory or table_dir(), f"{TABLE_NAME}_{name}.csv")


def _stem(acc_name: str) -> str:
    return f"{TABLE_NAME}_{acc_name}"


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    summary = summary_table(table)
    out_dir = table_dir(pargs)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    summary.to_csv(csv_path(out_dir, "summary"), index=False)

    for acc_name, part in summary.groupby("acc_name", sort=False):
        styler = _summary_styler(part)
        stem = _stem(acc_name)
        with open(os.path.join(tabular_dir, f"{stem}.tex"), "w") as f:
            f.write(booktabs_clines(styler.to_latex(hrules=True, multicol_align="c", clines="skip-last;data")))
        with open(os.path.join(tabular_dir, f"{stem}_long.tex"), "w") as f:
            f.write(eb_prior._latex_longtable(styler, caption(acc_name), f"tab:{TABLE_NAME}-{acc_name}"))
        with open(os.path.join(out_dir, f"{stem}_doc.tex"), "w") as f:
            f.write(_latex_document(f"tabular/{stem}_long.tex"))

    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
    print(f"rows: {len(table)}; tables written to {out_dir}")


def gen_pdf(pargs):
    out_dir = table_dir(pargs)
    for acc_name in pargs.metrics or get_acc_names():
        tex_filename = f"{_stem(acc_name)}_doc.tex"
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
    parser = ArgumentParser(description="The prior on pi^Q of BayCAP-Hf and BayCAP-KDEy (E5.1).")
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
