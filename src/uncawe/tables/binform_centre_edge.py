r"""Concentrated and sparse prevalence distributions for BayCAP-Hf and BayCAP-KDEy (work-plan card WP-24, E5.4 of
``docs/ext5_baycap_priors_tempering.md``).

E5.4 (``experiments/binform.py --e54``) draws the bags of each pair of ``main`` at prevalences from one of two symmetric
distributions with mean $\mathbf 1/K$, concentrated ($G_c=\operatorname{Dir}(3\cdot\mathbf 1)$, ``centre``) or sparse
($G_e=\operatorname{Dir}(\tfrac13\mathbf 1)$, ``edge``), and runs ``BayesCAP-Hf-real`` (BayCAP-Hf) and
``BayCAP-KDEy-mlp-real`` (BayCAP-KDEy with an MLP input and a fixed $\hat C$) on each set of bags with three priors on
$\pi^Q$: uniform, ``centre`` and ``edge``. For the bags of a distribution, the prior equal to it is the well-informed
one (``well``) and the other the ill-informed one (``ill``). Both methods target the realised table of each bag and are
scored against it (``true_cts``); the scores use the target labels, so they are analyses. The pairs are those where all
three priors of a method are stored, for each distribution and bag size, so the table also works on the pilot.

Per pair (``pair_table``, one row per metric, distribution, bag size, method and pair, every column in the CSV): the
means over the bags of AE, coverage, amplitude and Winkler score at $\alpha=0.05$ for the uniform, well-informed and
ill-informed prior; $\Delta W_{\text{know}}=W_{\text{uniform}}-W_{\text{well}}$ and
$\Delta W_{\text{wrong}}=W_{\text{ill}}-W_{\text{uniform}}$ (both positive when they mean what their names say); the
ratio of the mean amplitude of each informative prior to that of the uniform one; per prior, the share of bags with
divergent transitions and of bags where NUTS produced no draws (left out of the scores); the mean number of classes
absent from the realised table of the bags. The bags of the three priors are checked to be the same.

Summary (``summary_table``, per metric, distribution, bag size, method, group of $K$ and prior): the number of pairs,
the means over the pairs of the four scores, the mean of the difference of the Winkler score from that of the uniform
prior, the number of pairs and of datasets (means over the classifiers of a dataset) where the prior has a lower
Winkler score than the uniform one, the $p$-values of the paired Wilcoxon signed-rank tests (two-sided) over the pairs
and over the datasets, the median amplitude ratio, the shares of divergent bags and the mean number of absent classes.
E5.4 has no decision rule.

Note on macro-$F_1$ (Part IV, FIX-6 of ``docs/work_plan.md``): the true value is computed with the smoothed metric
(``acc_from_ct(..., type="selection")``, as everywhere), so a class absent from the bag and never predicted scores an
$F_1$ of about $1/K$ instead of $1$; the bags of $G_e$ have many such classes.

Outputs, under ``<OUT_DIR>/uncawe/tables/binform_centre_edge/``: the per-pair CSV (read by
``plots/binform_centre_edge.py``), the summary CSV and, per metric, the summary as a LaTeX tabular fragment, a longtable
and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.binform import (
    E51_BASE_METHODS,
    E54_DISTRIBUTIONS,
    E54_METHODS,
    E54_PRIORS,
    E54_SIZES,
    e54_experiment,
)
from uncawe.experiments.metrics import get_acc_names
from uncawe.results import Results
from uncawe.tables import binform_prior, eb_prior
from uncawe.tables import main as main_tables
from uncawe.tables.binform_prior import (
    ACC_LABELS,
    K_LABELS,
    METHOD_LABELS,
    SCORES,
    bag_frame,
    k_groups,
    paired_wilcoxon,
)
from uncawe.tables.estimand_gap import CLASSIFIER_ORDER, booktabs_clines

TABLE_NAME = "binform_centre_edge"
DOMAIN = "classic"
KEYS = ["dataset", "classifier"]
ROLES = ("uniform", "well", "ill")
INFORMATIVE_ROLES = ("well", "ill")


def method_name(base: str, prior: str) -> str:
    return f"{base}_{prior}"


assert all(method_name(base, prior) in E54_METHODS for base in E51_BASE_METHODS for prior in E54_PRIORS)


def role_priors(distribution: str) -> dict[str, str]:
    """``role -> prior`` for the bags of ``distribution``: the well-informed prior is the distribution itself, the
    ill-informed one the other distribution."""
    (other,) = (d for d in E54_DISTRIBUTIONS if d != distribution)
    return {"uniform": "uniform", "well": distribution, "ill": other}


# --------------------------------------------------------------------------------------------------- loading


def load_method(method: str, acc_name: str, distribution: str, size: int, root: str | None = None) -> pd.DataFrame:
    """The stored rows of the E5.4 method ``method`` for ``acc_name`` and the bags of ``size`` instances drawn from
    ``distribution`` (empty if none)."""
    base_dir = os.path.join(env.root_dir if root is None else root, e54_experiment(distribution, size))
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


def load_all(acc_name: str, root: str | None = None) -> dict[tuple[str, int, str], dict[str, pd.DataFrame]]:
    """``(distribution, size, base method) -> {role: rows}``, restricted to the pairs stored for the three priors; the
    cells without any such pair are left out."""
    out = {}
    for distribution in E54_DISTRIBUTIONS:
        roles = role_priors(distribution)
        for size in E54_SIZES:
            for base in E51_BASE_METHODS:
                frames = {
                    role: load_method(method_name(base, prior), acc_name, distribution, size, root)
                    for role, prior in roles.items()
                }
                if any(df.empty for df in frames.values()):
                    continue
                pairs = eb_prior.common_pairs(frames)
                if not pairs:
                    continue
                index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
                out[(distribution, size, base)] = {
                    role: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
                    for role, df in frames.items()
                }
    return out


# --------------------------------------------------------------------------------------------------- scores


def absent_classes(true_ct, n_classes: int) -> int:
    """The number of classes with no instance in the bag: the zero rows of the realised table (cap-ml layout, true
    classes on rows)."""
    ct = np.asarray(true_ct, dtype=float).reshape(n_classes, n_classes)
    return int(np.sum(ct.sum(axis=1) == 0))


def pair_rows(frames: dict[str, pd.DataFrame], acc_name: str, distribution: str, size: int, base: str) -> list[dict]:
    """The rows of ``pair_table`` of one metric, distribution, bag size and base method (``frames``: role -> stored
    rows)."""
    groups = {role: dict(list(df.groupby(KEYS, sort=False))) for role, df in frames.items()}
    roles = role_priors(distribution)
    rows = []
    for key, first in groups["uniform"].items():
        # the check compares every prior with the uniform one (binform_prior.REFERENCE_PRIOR)
        binform_prior._check_same_bags({role: groups[role][key] for role in ROLES}, key)
        bags = {role: bag_frame(groups[role][key], acc_name) for role in ROLES}
        n_classes = int(first["n_classes"].iloc[0])
        row = {
            "acc_name": acc_name,
            "distribution": distribution,
            "size": size,
            "method": base,
            "dataset": key[0],
            "classifier": key[1],
            "n_classes": n_classes,
            "n_bags": len(first),
            "well_prior": roles["well"],
            "ill_prior": roles["ill"],
            "absent_classes": float(np.mean([absent_classes(ct, n_classes) for ct in first["true_cts"]])),
        }
        for role in ROLES:
            for score in SCORES:
                row[f"{role}_{score}"] = float(bags[role][score].mean())
            row[f"{role}_share_failed"] = float(bags[role]["failed"].mean())
            row[f"{role}_share_divergent"] = float(bags[role]["divergent"].mean())
        row["d_w_know"] = row["uniform_winkler"] - row["well_winkler"]
        row["d_w_wrong"] = row["ill_winkler"] - row["uniform_winkler"]
        for role in INFORMATIVE_ROLES:
            reference = row["uniform_amplitude"]
            row[f"{role}_amplitude_ratio"] = row[f"{role}_amplitude"] / reference if reference > 0 else np.nan
        rows.append(row)
    return rows


def pair_table(acc_names: list[str] | None = None, root: str | None = None) -> pd.DataFrame:
    """One row per metric, distribution, bag size, base method and pair, ordered by those keys, then by $K$, dataset
    and classifier (as in ``CLASSIFIER_ORDER``)."""
    rows = []
    for acc_name in acc_names or get_acc_names():
        for (distribution, size, base), frames in load_all(acc_name, root).items():
            rows.extend(pair_rows(frames, acc_name, distribution, size, base))
    if not rows:
        raise RuntimeError("No pair of E5.4 has stored results for the three priors of a method.")
    return ordered(pd.DataFrame(rows))


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["_acc"] = table["acc_name"].map(get_acc_names().index)
    table["_dist"] = table["distribution"].map(E54_DISTRIBUTIONS.index)
    table["_method"] = table["method"].map(E51_BASE_METHODS.index)
    table["_clf"] = table["classifier"].map(lambda c: CLASSIFIER_ORDER.index(c) if c in CLASSIFIER_ORDER else 99)
    table = table.sort_values(["_acc", "_dist", "size", "_method", "n_classes", "dataset", "_clf"])
    return table.drop(columns=["_acc", "_dist", "_method", "_clf"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- summary


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric, distribution, bag size, base method, group of $K$ and role of the prior: the summaries of the module
    docstring."""
    rows = []
    for (acc_name, distribution, size, base), sub in table.groupby(
        ["acc_name", "distribution", "size", "method"], sort=False
    ):
        roles = role_priors(distribution)
        for k_group, part in k_groups(sub):
            uniform_w = part["uniform_winkler"].to_numpy(dtype=float)
            datasets = part.groupby("dataset", sort=False)
            for role in ROLES:
                row = {
                    "acc_name": acc_name,
                    "distribution": distribution,
                    "size": size,
                    "method": base,
                    "K": k_group,
                    "role": role,
                    "prior": roles[role],
                    "pairs": len(part),
                    "datasets": datasets.ngroups,
                }
                for score in SCORES:
                    row[score] = float(part[f"{role}_{score}"].mean())
                row["share_divergent"] = float(part[f"{role}_share_divergent"].mean())
                row["share_failed"] = float(part[f"{role}_share_failed"].mean())
                row["absent_classes"] = float(part["absent_classes"].mean())
                if role == "uniform":
                    row |= {
                        "d_winkler": np.nan,
                        "pairs_lower": np.nan,
                        "p_value": np.nan,
                        "datasets_lower": np.nan,
                        "p_value_datasets": np.nan,
                        "amplitude_ratio": np.nan,
                    }
                else:
                    w = part[f"{role}_winkler"].to_numpy(dtype=float)
                    row["d_winkler"] = float(np.nanmean(w - uniform_w))
                    row["pairs_lower"] = int(np.sum(w < uniform_w))
                    row["p_value"] = paired_wilcoxon(w, uniform_w)
                    means = datasets[[f"{role}_winkler", "uniform_winkler"]].mean()
                    w_d, uniform_d = means[f"{role}_winkler"].to_numpy(), means["uniform_winkler"].to_numpy()
                    row["datasets_lower"] = int(np.sum(w_d < uniform_d))
                    row["p_value_datasets"] = paired_wilcoxon(w_d, uniform_d)
                    row["amplitude_ratio"] = float(part[f"{role}_amplitude_ratio"].median())
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

DISTRIBUTION_LABELS = {
    "centre": "$G_c=\\operatorname{Dir}(3\\cdot\\mathbf 1)$",
    "edge": "$G_e=\\operatorname{Dir}(\\tfrac13\\mathbf 1)$",
}
ROLE_LABELS = {"uniform": "uniform", "well": "well informed", "ill": "ill informed"}
COLUMN_LABELS = {
    "pairs": "pairs",
    "ae": "AE",
    "coverage": main_tables.metrics_map["coverage"],
    "amplitude": main_tables.metrics_map["amplitude"],
    "winkler": main_tables.metrics_map["winkler"],
    "d_winkler": "$\\Delta\\mathcal{W}$",
    "pairs_lower": "lower",
    "p_value": "$p$",
    "datasets_lower": "lower (d.)",
    "p_value_datasets": "$p$ (d.)",
    "amplitude_ratio": "$\\mathcal{A}/\\mathcal{A}_{\\mathrm{unif}}$",
    "share_divergent": "div.",
    "absent_classes": "absent",
}


def _summary_styler(summary: pd.DataFrame):
    """One row per bag size, method, distribution, group of $K$ and role of the prior; the columns of
    ``COLUMN_LABELS``."""
    out = pd.DataFrame({label: summary[column].to_numpy() for column, label in COLUMN_LABELS.items()})
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["size"].map(lambda s: f"$|U|={s}$"),
            summary["method"].map(METHOD_LABELS),
            summary["distribution"].map(DISTRIBUTION_LABELS),
            summary["K"].map(K_LABELS),
            summary["role"].map(ROLE_LABELS),
        ],
        names=["bags", "method", "$G$", "$K$", "prior"],
    )
    integers = [COLUMN_LABELS[c] for c in ("pairs", "pairs_lower", "datasets_lower")]
    return out.style.format(precision=3, na_rep="--").format("{:.0f}", subset=integers, na_rep="--")


def caption(acc_name: str) -> str:
    metric = ACC_LABELS[acc_name]
    return (
        f"E5.4, concentrated and sparse prevalence distributions ({metric}); bags at prevalences from "
        "$G_c=\\operatorname{Dir}(3\\cdot\\mathbf 1)$ or $G_e=\\operatorname{Dir}(\\tfrac13\\mathbf 1)$; means over the "
        "pairs of the per-pair means over the 100 bags, scored against the realised table. Well informed: the prior "
        "equal to $G$; ill informed: the other distribution. $\\Delta\\mathcal{W}$: mean over the pairs of the Winkler "
        "score minus that of the uniform prior; lower: pairs (d.: datasets, means over their classifiers) where it is "
        "negative; $p$: paired Wilcoxon signed-rank test over the pairs (d.: over the datasets); "
        "$\\mathcal{A}/\\mathcal{A}_{\\mathrm{unif}}$: median over the pairs of the ratio of the mean amplitudes; div.: "
        "share of bags with divergent transitions; absent: mean number of classes absent from a bag."
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


def write_tables(table: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Write the CSVs and, per metric, the LaTeX fragment, longtable and document of the summary; return it."""
    summary = summary_table(table)
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
            f.write(binform_prior._latex_document(f"tabular/{stem}_long.tex"))
    return summary


def gen_tables(pargs):
    table = pair_table(pargs.metrics)
    out_dir = table_dir(pargs)
    summary = write_tables(table, out_dir)
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
    parser = ArgumentParser(description="Concentrated and sparse prevalence distributions (E5.4).")
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
