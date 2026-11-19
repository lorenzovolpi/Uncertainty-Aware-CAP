r"""Right-side and wrong-side priors under strong shifts, with SQS references (work-plan card WP-27, E5.5 of
``docs/ext5_baycap_priors_tempering.md``).

E5.5 (``experiments/binform.py --e55``) draws the bags of $|U|=1000$ instances of each pair of ``main`` at prevalences
from one of two symmetric distributions with mean $\mathbf 1/K$, concentrated ($G_c=\operatorname{Dir}(10\cdot\mathbf
1)$, ``centre``) or sparse ($G_e=\operatorname{Dir}(\tfrac1{10}\mathbf 1)$, ``edge``). ``BayesCAP-Hf-real``
(BayCAP-Hf), ``BayCAP-KDEy-mlp-real`` (BayCAP-KDEy) and ``SQS-B`` run on each set of bags with three priors on $\pi^Q$:
uniform, ``centre`` ($\operatorname{Dir}(3\cdot\mathbf 1)$) and ``edge`` ($\operatorname{Dir}(\tfrac13\mathbf 1)$). For
the bags of a distribution, the prior on its side is the right-side one (``right``) and the other the wrong-side one
(``wrong``); neither equals the distribution. ``SQS-H``, ``SQS-Hbs`` and ``PQ`` (SQS-S) take no prior on $\pi^Q$ and run
once per distribution (role ``none``). Every method targets the realised table of each bag and is scored against it
(``true_cts``); the scores use the target labels, so they are analyses. For each distribution, the pairs are those
stored for every method that has stored results, and a method with a prior is kept only if its three priors are stored,
so the tables also work on the pilot.

Per bag (``bag_table``, one row per metric, distribution, method, role, pair and bag): AE, coverage, amplitude and
Winkler score at $\alpha=0.05$, whether the method produced no draws (``failed``, scores NaN) and whether NUTS had
divergent transitions (methods with a prior only); read by the violins of ``plots/binform_side_prior.py``.

Per pair (``pair_table``, one row per metric, distribution, method and pair): the means over the bags of the four scores
for each role of the method (``uniform_*``, ``right_*``, ``wrong_*`` for the methods with a prior, ``none_*`` for the
others, NaN elsewhere); for the methods with a prior, $\Delta W_{\text{know}}=W_{\text{uniform}}-W_{\text{right}}$ and
$\Delta W_{\text{wrong}}=W_{\text{wrong}}-W_{\text{uniform}}$ (both positive when they mean what their names say) and
the ratio of the mean amplitude of each informative prior to that of the uniform one; per role, the share of bags with
divergent transitions and of failed bags; the mean number of classes absent from the realised table of the bags. The
bags of every method and prior are checked to be the same.

Summary (``summary_table``, per metric, distribution, group of $K$, method and role): as in E5.4
(``tables/binform_centre_edge.py``): the number of pairs and of datasets, the means over the pairs of the four scores,
and for the informative priors the mean difference of the Winkler score from that of the uniform prior, the number of
pairs and of datasets (means over the classifiers of a dataset) where it is lower, the $p$-values of the paired Wilcoxon
signed-rank tests (two-sided) over the pairs and over the datasets and the median amplitude ratio; the shares of
divergent bags and the mean number of absent classes. E5.5 has no decision rule.

Macro-$F_1$ is scored with the smoothed metric, as in E5.4 (Part IV, FIX-6 of ``docs/work_plan.md``).

Outputs, under ``<OUT_DIR>/uncawe/tables/binform_side_prior/``: the per-bag, per-pair and summary CSVs and, per metric,
the summary as a LaTeX tabular fragment, a longtable and a standalone document.
"""

import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from uncawe import env
from uncawe.experiments.binform import (
    E55_BAG_ALPHAS,
    E55_DISTRIBUTIONS,
    E55_METHODS,
    E55_PRIOR_BASES,
    E55_PRIORS,
    E55_REFERENCES,
    E55_SIZE,
    e55_experiment,
)
from uncawe.experiments.metrics import get_acc_names
from uncawe.results import Results
from uncawe.tables import binform_prior, eb_prior
from uncawe.tables import main as main_tables
from uncawe.tables.binform_prior import ACC_LABELS, K_LABELS, SCORES, bag_frame, k_groups, paired_wilcoxon
from uncawe.tables.common import METHOD_LATEX_LABELS
from uncawe.tables.estimand_gap import CLASSIFIER_ORDER, booktabs_clines

TABLE_NAME = "binform_side_prior"
DOMAIN = "classic"
KEYS = ["dataset", "classifier"]
ROLES = ("uniform", "right", "wrong")
INFORMATIVE_ROLES = ("right", "wrong")
# the role of the methods without a prior on pi^Q
NO_PRIOR = "none"
ALL_ROLES = (*ROLES, NO_PRIOR)
METHODS = (*E55_PRIOR_BASES, *E55_REFERENCES)


def method_name(base: str, prior: str | None) -> str:
    """The stored name of ``base`` with ``prior`` (``None`` for the methods without a prior)."""
    return base if prior is None else f"{base}_{prior}"


assert all(method_name(base, prior) in E55_METHODS for base in E55_PRIOR_BASES for prior in E55_PRIORS)
assert all(method_name(base, None) in E55_METHODS for base in E55_REFERENCES)


def role_priors(distribution: str) -> dict[str, str]:
    """``role -> prior`` for the bags of ``distribution``: the right-side prior is the prior named after the
    distribution (on the same side, less extreme), the wrong-side one the other."""
    (other,) = (d for d in E55_DISTRIBUTIONS if d != distribution)
    return {"uniform": "uniform", "right": distribution, "wrong": other}


def method_roles(base: str, distribution: str) -> dict[str, str]:
    """``role -> stored method name`` of ``base`` on the bags of ``distribution``."""
    if base in E55_REFERENCES:
        return {NO_PRIOR: method_name(base, None)}
    return {role: method_name(base, prior) for role, prior in role_priors(distribution).items()}


# --------------------------------------------------------------------------------------------------- loading


def load_method(method: str, acc_name: str, distribution: str, root: str | None = None) -> pd.DataFrame:
    """The stored rows of the E5.5 method ``method`` for ``acc_name`` and the bags drawn from ``distribution`` (empty
    if none)."""
    base_dir = os.path.join(env.root_dir if root is None else root, e55_experiment(distribution))
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


def restrict(frames: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, pd.DataFrame]]:
    """Keep the methods with every role stored, and their rows on the pairs stored for all of them (``frames``: base
    method -> role -> rows; a method or role with no row is an empty frame)."""
    complete = {base: roles for base, roles in frames.items() if all(not df.empty for df in roles.values())}
    if not complete:
        return {}
    pairs = eb_prior.common_pairs({f"{b}/{r}": df for b, roles in complete.items() for r, df in roles.items()})
    if not pairs:
        return {}
    index = pd.MultiIndex.from_tuples(pairs, names=KEYS)
    return {
        base: {
            role: df.loc[pd.MultiIndex.from_frame(df[KEYS]).isin(index)].reset_index(drop=True)
            for role, df in roles.items()
        }
        for base, roles in complete.items()
    }


def load_all(acc_name: str, root: str | None = None) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    """``distribution -> base method -> role -> rows``, restricted as in ``restrict``; the distributions without any
    stored pair are left out."""
    out = {}
    for distribution in E55_DISTRIBUTIONS:
        frames = {
            base: {
                role: load_method(name, acc_name, distribution, root)
                for role, name in method_roles(base, distribution).items()
            }
            for base in METHODS
        }
        kept = restrict(frames)
        if kept:
            out[distribution] = kept
    return out


# --------------------------------------------------------------------------------------------------- scores


def absent_classes(true_ct, n_classes: int) -> int:
    """The number of classes with no instance in the bag: the zero rows of the realised table (cap-ml layout, true
    classes on rows)."""
    ct = np.asarray(true_ct, dtype=float).reshape(n_classes, n_classes)
    return int(np.sum(ct.sum(axis=1) == 0))


def _check_same_bags(frames: dict[str, pd.DataFrame], key) -> None:
    """The stored realised tables of the bags of one pair are the same for every method and prior."""
    (first_name, first), *others = frames.items()
    reference = np.stack([np.ravel(ct) for ct in first["true_cts"]])
    for name, frame in others:
        tables = np.stack([np.ravel(ct) for ct in frame["true_cts"]])
        if tables.shape != reference.shape or not np.allclose(tables, reference):
            raise ValueError(f"{key}: the bags of {name} differ from those of {first_name}")


def distribution_rows(
    frames: dict[str, dict[str, pd.DataFrame]], acc_name: str, distribution: str
) -> tuple[list[dict], list[dict]]:
    """The rows of ``pair_table`` and of ``bag_table`` of one metric and distribution (``frames``: base method -> role
    -> stored rows, on the same pairs)."""
    groups = {
        (base, role): dict(list(df.groupby(KEYS, sort=False)))
        for base, roles in frames.items()
        for role, df in roles.items()
    }
    priors = role_priors(distribution)
    pair_rows, bag_rows = [], []
    first_cell = next(iter(groups))
    for key, first in groups[first_cell].items():
        _check_same_bags({f"{base}/{role}": groups[(base, role)][key] for base, role in groups}, key)
        n_classes = int(first["n_classes"].iloc[0])
        absent = float(np.mean([absent_classes(ct, n_classes) for ct in first["true_cts"]]))
        for base, roles in frames.items():
            row = {
                "acc_name": acc_name,
                "distribution": distribution,
                "size": E55_SIZE,
                "method": base,
                "dataset": key[0],
                "classifier": key[1],
                "n_classes": n_classes,
                "n_bags": len(first),
                "right_prior": priors["right"] if base in E55_PRIOR_BASES else None,
                "wrong_prior": priors["wrong"] if base in E55_PRIOR_BASES else None,
                "absent_classes": absent,
            }
            for role in ALL_ROLES:
                if role not in roles:
                    for score in SCORES:
                        row[f"{role}_{score}"] = np.nan
                    row[f"{role}_share_failed"] = row[f"{role}_share_divergent"] = np.nan
                    continue
                bags = bag_frame(groups[(base, role)][key], acc_name)
                for score in SCORES:
                    row[f"{role}_{score}"] = float(bags[score].mean())
                row[f"{role}_share_failed"] = float(bags["failed"].mean())
                row[f"{role}_share_divergent"] = float(bags["divergent"].mean())
                prior = priors.get(role)
                for bag, values in enumerate(bags.itertuples(index=False)):
                    bag_rows.append(
                        {
                            "acc_name": acc_name,
                            "distribution": distribution,
                            "method": base,
                            "role": role,
                            "prior": prior,
                            "dataset": key[0],
                            "classifier": key[1],
                            "n_classes": n_classes,
                            "bag": bag,
                        }
                        | {name: getattr(values, name) for name in (*SCORES, "failed", "divergent")}
                    )
            if base in E55_PRIOR_BASES:
                row["d_w_know"] = row["uniform_winkler"] - row["right_winkler"]
                row["d_w_wrong"] = row["wrong_winkler"] - row["uniform_winkler"]
                reference = row["uniform_amplitude"]
                for role in INFORMATIVE_ROLES:
                    row[f"{role}_amplitude_ratio"] = row[f"{role}_amplitude"] / reference if reference > 0 else np.nan
            else:
                row["d_w_know"] = row["d_w_wrong"] = np.nan
                row["right_amplitude_ratio"] = row["wrong_amplitude_ratio"] = np.nan
            pair_rows.append(row)
    return pair_rows, bag_rows


def tables(acc_names: list[str] | None = None, root: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The per-pair and per-bag tables, ordered by metric, distribution, method, $K$, dataset and classifier."""
    pair_rows, bag_rows = [], []
    for acc_name in acc_names or get_acc_names():
        for distribution, frames in load_all(acc_name, root).items():
            pairs, bags = distribution_rows(frames, acc_name, distribution)
            pair_rows.extend(pairs)
            bag_rows.extend(bags)
    if not pair_rows:
        raise RuntimeError("No pair of E5.5 has stored results for a method.")
    return ordered(pd.DataFrame(pair_rows)), ordered(pd.DataFrame(bag_rows))


def ordered(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    keys = {
        "_acc": table["acc_name"].map(get_acc_names().index),
        "_dist": table["distribution"].map(E55_DISTRIBUTIONS.index),
        "_method": table["method"].map(METHODS.index),
        "_clf": table["classifier"].map(lambda c: CLASSIFIER_ORDER.index(c) if c in CLASSIFIER_ORDER else 99),
    }
    order = ["_acc", "_dist", "_method", "n_classes", "dataset", "_clf"]
    if "role" in table:
        keys["_role"] = table["role"].map(ALL_ROLES.index)
        order = ["_acc", "_dist", "_method", "_role", "n_classes", "dataset", "_clf", "bag"]
    table = table.assign(**keys).sort_values(order, kind="stable")
    return table.drop(columns=list(keys)).reset_index(drop=True)


# --------------------------------------------------------------------------------------------------- summary


def summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Per metric, distribution, group of $K$, method and role: the summaries of the module docstring."""
    rows = []
    for (acc_name, distribution), sub in table.groupby(["acc_name", "distribution"], sort=False):
        priors = role_priors(distribution)
        for k_group, part_k in k_groups(sub):
            for base, part in part_k.groupby("method", sort=False):
                uniform_w = part["uniform_winkler"].to_numpy(dtype=float)
                datasets = part.groupby("dataset", sort=False)
                for role in ROLES if base in E55_PRIOR_BASES else (NO_PRIOR,):
                    row = {
                        "acc_name": acc_name,
                        "distribution": distribution,
                        "K": k_group,
                        "method": base,
                        "role": role,
                        "prior": priors.get(role),
                        "pairs": len(part),
                        "datasets": datasets.ngroups,
                    }
                    for score in SCORES:
                        row[score] = float(part[f"{role}_{score}"].mean())
                    row["share_divergent"] = float(part[f"{role}_share_divergent"].mean())
                    row["share_failed"] = float(part[f"{role}_share_failed"].mean())
                    row["absent_classes"] = float(part["absent_classes"].mean())
                    if role in INFORMATIVE_ROLES:
                        w = part[f"{role}_winkler"].to_numpy(dtype=float)
                        row["d_winkler"] = float(np.nanmean(w - uniform_w))
                        row["pairs_lower"] = int(np.sum(w < uniform_w))
                        row["p_value"] = paired_wilcoxon(w, uniform_w)
                        means = datasets[[f"{role}_winkler", "uniform_winkler"]].mean()
                        w_d, uniform_d = means[f"{role}_winkler"].to_numpy(), means["uniform_winkler"].to_numpy()
                        row["datasets_lower"] = int(np.sum(w_d < uniform_d))
                        row["p_value_datasets"] = paired_wilcoxon(w_d, uniform_d)
                        row["amplitude_ratio"] = float(part[f"{role}_amplitude_ratio"].median())
                    else:
                        row |= dict.fromkeys(
                            (
                                "d_winkler",
                                "pairs_lower",
                                "p_value",
                                "datasets_lower",
                                "p_value_datasets",
                                "amplitude_ratio",
                            ),
                            np.nan,
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- LaTeX

DISTRIBUTION_LABELS = {
    "centre": "$G_c=\\operatorname{Dir}(10\\cdot\\mathbf 1)$",
    "edge": "$G_e=\\operatorname{Dir}(\\tfrac1{10}\\mathbf 1)$",
}
assert set(DISTRIBUTION_LABELS) == set(E55_BAG_ALPHAS)
ROLE_LABELS = {"uniform": "uniform", "right": "right side", "wrong": "wrong side", NO_PRIOR: "no prior"}
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
    """One row per distribution, group of $K$, method and role of the prior; the columns of ``COLUMN_LABELS``."""
    out = pd.DataFrame({label: summary[column].to_numpy() for column, label in COLUMN_LABELS.items()})
    out.index = pd.MultiIndex.from_arrays(
        [
            summary["distribution"].map(DISTRIBUTION_LABELS),
            summary["K"].map(K_LABELS),
            summary["method"].map(METHOD_LATEX_LABELS),
            summary["role"].map(ROLE_LABELS),
        ],
        names=["$G$", "$K$", "method", "prior"],
    )
    integers = [COLUMN_LABELS[c] for c in ("pairs", "pairs_lower", "datasets_lower")]
    return out.style.format(precision=3, na_rep="--").format("{:.0f}", subset=integers, na_rep="--")


def caption(acc_name: str) -> str:
    metric = ACC_LABELS[acc_name]
    return (
        f"E5.5, right-side and wrong-side priors under strong shifts ({metric}); bags of 1000 instances at prevalences "
        "from $G_c=\\operatorname{Dir}(10\\cdot\\mathbf 1)$ or $G_e=\\operatorname{Dir}(\\tfrac1{10}\\mathbf 1)$; means "
        "over the pairs of the per-pair means over the 100 bags, scored against the realised table. Right side: the "
        "prior on the side of $G$ ($\\operatorname{Dir}(3\\cdot\\mathbf 1)$ under $G_c$, "
        "$\\operatorname{Dir}(\\tfrac13\\mathbf 1)$ under $G_e$); wrong side: the other; no prior: methods without a "
        "prior on $\\pi^Q$. $\\Delta\\mathcal{W}$: mean over the pairs of the Winkler score minus that of the uniform "
        "prior; lower: pairs (d.: datasets, means over their classifiers) where it is negative; $p$: paired Wilcoxon "
        "signed-rank test over the pairs (d.: over the datasets); $\\mathcal{A}/\\mathcal{A}_{\\mathrm{unif}}$: median "
        "over the pairs of the ratio of the mean amplitudes; div.: share of bags with divergent transitions; absent: "
        "mean number of classes absent from a bag."
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


def write_tables(table: pd.DataFrame, bags: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    """Write the CSVs and, per metric, the LaTeX fragment, longtable and document of the summary; return it."""
    summary = summary_table(table)
    tabular_dir = os.path.join(out_dir, "tabular")
    os.makedirs(tabular_dir, exist_ok=True)
    table.to_csv(csv_path(out_dir), index=False)
    bags.to_csv(csv_path(out_dir, "bags"), index=False)
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
    table, bags = tables(pargs.metrics)
    out_dir = table_dir(pargs)
    summary = write_tables(table, bags, out_dir)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.precision", 3):
        print(summary.loc[summary["K"] == "all"].to_string(index=False))
    print(f"rows: {len(table)} pairs, {len(bags)} bags; tables written to {out_dir}")


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
    parser = ArgumentParser(description="Right-side and wrong-side priors under strong shifts (E5.5).")
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
