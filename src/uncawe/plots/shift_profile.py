r"""Shift profile of the methods of ``main``: how the AE and the Winkler score of each method change from the least to
the most shifted target bags, over the (dataset, classifier) pairs. The design choices and their reasons are in §II.6.1
of ``docs/work_plan.md``.

Every value is scored against the realised table of each target bag (``true_cts``), as in the main tables.

- **Shift level.** The stored shift of a bag is $s=\frac1K\sum_k\lvert\pi^U_k-\pi^L_k\rvert$ (computed from the true
  labels of the bag, used only to group the bags). Its scale depends on $K$ and on $\pi^L$, so the bags are not
  grouped by value: within each pair they are sorted by $s$ (stable sort, ties broken by the bag index) and cut into
  ``--bins`` groups of equal size (deciles by default), D1 the least and D10 the most shifted. The grouping depends
  only on the bags, so it is the same for every method.
- **Value.** For each pair, method and decile, the mean of the metric over the bags of the decile, averaged over the
  classifiers of the dataset: $y_{d,j}$ for dataset $d$ and condition $j$ (a method and a decile). The point of a
  condition is the geometric mean over the datasets, $\exp(\bar\ell_j)$ with $\ell_{d,j}=\log y_{d,j}$, on a
  logarithmic axis (user's decision of 2026-11-25; before, a value relative to the mean of the methods on the pair,
  on a linear axis, without bars).
- **Bars.** $\pm1$ standard error over the datasets after removing each dataset's average level:
  $w_{d,j}=\ell_{d,j}-\bar\ell_{d\cdot}$, with $\bar\ell_{d\cdot}$ the mean of $\ell_{d,j}$ over the plotted methods and
  the deciles, $\mathrm{SE}_j=\operatorname{sd}_d(w_{d,j})/\sqrt n$, and the bar from $\exp(\bar\ell_j-\mathrm{SE}_j)$ to
  $\exp(\bar\ell_j+\mathrm{SE}_j)$. The bars depend on the set of plotted methods (``--methods`` without
  ``--exclude-methods``; the paper leaves out SQS-B), the points do not. The methods are shifted sideways around each
  decile by ``DODGE`` times the spacing between deciles, so that their bars do not overlap.

Figures (``--figures``):

- ``pooled`` (default): 1x2, AE (left) and Winkler score (right) over all datasets.
- ``by_k``: 3x2 grid, the same per group of $K$ (``K_GROUPS``: $K=2$, $3\leq K\leq 9$, $K\geq 10$; the row label gives
  the range of $K$ present and the number of datasets); the bars remove the dataset levels within each group.

Each figure is written as png (and svg unless ``--no-svg``, pdf unless ``--no-pdf``) with a CSV of the plotted values
and bar ends, to ``<OUT_DIR>/uncawe/plots/main/shift_profile/``. Method names are the labels of the main tables with
the LaTeX macros resolved (``plots/common.py:method_table_label``).
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from uncawe import env
from uncawe.config import acc_from_ct, get_acc_names
from uncawe.plots.common import DEFAULT_DOMAIN, EXPERIMENT, font_size, method_table_label, plot_dir, safe_name
from uncawe.results import Results
from uncawe.tables.main import CLASSIFIER_FILTER, get_acc_winkler

PLOT_NAME = "shift_profile"
FIGURES = ("by_k", "pooled")
# the pooled figure is the default since 2026-11-25 (user's decision); before, both
DEFAULT_FIGURES = ("pooled",)
METRICS = (("acc_err", "AE"), ("winkler", "W"))
# BayCAP-KDEy-h-inst in place of BayCAP-KDEy-mlp-real since E3.4 (card WP-31, user's decision of 2026-11-24), with its
# colour
DEFAULT_METHODS = ("O-LEAP", "SQS-Hbs", "SQS-B", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst")
# colours of this figure only (user's decision of 2026-11-21): the shared METHOD_COLORS of BayesCAP-Hf-real and
# BayCAP-KDEy-mlp-real are too faint next to the other three; methods not listed fall back to grey
SHIFT_PROFILE_COLORS = {
    "O-LEAP": "#4C78A8",
    "SQS-Hbs": "#B279A2",
    "SQS-B": "#F58518",
    "BayesCAP-Hf-real": "#E45756",
    "BayCAP-KDEy-h-inst": "#54A24B",
}
# groups of the number of classes K, as (lower, upper) bounds, upper None for no bound
K_GROUPS = ((2, 2), (3, 9), (10, None))
DEFAULT_BINS = 10
# sideways shift between consecutive methods, as a fraction of the spacing between deciles
DODGE = 0.12
Y_TICKS = (0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)


def load_scores(domain: str, acc_name: str, methods: list[str]) -> pd.DataFrame:
    """Per-bag AE and Winkler score of the realised accuracy (from ``true_cts``) of ``methods`` in ``main``, with the
    stored shift and the number of classes."""
    results = Results.load(
        base_dir=os.path.join(env.root_dir, EXPERIMENT),
        acc_name=acc_name,
        domain=domain,
        filter_classifiers=list(CLASSIFIER_FILTER),
        filter_methods=list(methods),
    )
    df = (
        results.apply_to_column("true_cts", lambda ct: acc_from_ct(acc_name, ct), new_col="true_accs")
        .apply_to_column(["method", "true_accs", "estim_acc_ranges"], get_acc_winkler, new_col="winkler")
        .select_columns(["dataset", "classifier", "method", "n_classes", "shifts", "acc_err", "winkler"])
        .df
    )
    return df.reset_index(drop=True)


def add_bag_index(df: pd.DataFrame) -> pd.DataFrame:
    """The index of each bag within its (method, dataset, classifier) cell, in stored order; checks that every method
    of a pair has the same bags (same count and same shift at each index)."""
    df = df.copy()
    df["bag"] = df.groupby(["method", "dataset", "classifier"], sort=False).cumcount()
    spread = df.groupby(["dataset", "classifier", "bag"])["shifts"].agg(["min", "max", "size"])
    n_methods = df.groupby(["dataset", "classifier"])["method"].nunique()
    sizes = spread["size"].groupby(level=["dataset", "classifier"]).agg(["min", "max"])
    if not np.allclose(spread["min"], spread["max"]):
        raise ValueError("The methods of a pair have different shifts at the same bag index.")
    if not ((sizes["min"] == n_methods) & (sizes["max"] == n_methods)).all():
        raise ValueError("The methods of a pair have different numbers of bags.")
    return df


def shift_levels(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Adds ``level`` in ``0..n_bins-1``: within each pair, the bags sorted by shift (stable, ties by bag index) and cut
    into ``n_bins`` groups of (nearly) equal size, 0 the least shifted. The same for every method of the pair."""
    bags = df.drop_duplicates(["dataset", "classifier", "bag"])[["dataset", "classifier", "bag", "shifts"]]
    bags = bags.sort_values(["dataset", "classifier", "shifts", "bag"], kind="stable")
    rank = bags.groupby(["dataset", "classifier"]).cumcount()
    size = bags.groupby(["dataset", "classifier"])["bag"].transform("size")
    bags = bags.assign(level=(rank * n_bins // size).astype(int))
    return df.merge(bags[["dataset", "classifier", "bag", "level"]], on=["dataset", "classifier", "bag"])


def dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per method, dataset and level, the AE and Winkler score: the mean over the bags of each pair and level, then
    averaged over the classifiers of the dataset."""
    metrics = [metric for metric, _ in METRICS]
    pair = df.groupby(["method", "dataset", "classifier", "n_classes", "level"])[metrics].mean().reset_index()
    return pair.groupby(["method", "dataset", "n_classes", "level"])[metrics].mean().reset_index()


def geometric_summary(profile: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    r"""Per method and level of ``methods``: the geometric mean of ``metric`` over the datasets (``value``) and the ends
    of the bar of $\pm1$ standard error over the datasets of the logarithms after removing each dataset's mean over
    the methods and levels (``lower``, ``upper``). Every dataset must have every method and level, with positive
    values."""
    rows = profile.loc[profile["method"].isin(methods)]
    logs = rows.pivot_table(index="dataset", columns=["method", "level"], values=metric, aggfunc="first")
    if logs.isna().any().any():
        raise ValueError(f"Some datasets lack a method or a level of {metric}.")
    if (logs <= 0).any().any():
        raise ValueError(f"Non-positive {metric}: the geometric mean is not defined.")
    logs = np.log(logs)
    n = len(logs)
    centred = logs.sub(logs.mean(axis=1), axis=0)
    mean = logs.mean()
    se = centred.std(ddof=1) / np.sqrt(n) if n > 1 else centred.std() * 0.0
    out = pd.DataFrame({"value": np.exp(mean), "lower": np.exp(mean - se), "upper": np.exp(mean + se)})
    out["n_datasets"] = n
    return out.reset_index()


def k_group_label(k_values) -> str:
    k_min, k_max = int(min(k_values)), int(max(k_values))
    if k_min == k_max:
        return f"$K={k_min}$"
    return f"${k_min}\\leq K\\leq {k_max}$"


def k_groups(profile: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = []
    for lower, upper in K_GROUPS:
        k = profile["n_classes"]
        part = profile.loc[(k >= lower) & (k <= upper if upper is not None else True)]
        if part.empty:
            continue
        label = f"$K\\geq {lower}$" if upper is None else k_group_label(part["n_classes"].unique())
        groups.append((label, part))
    return groups


def plot_profile(rows: list[tuple[str, pd.DataFrame]], methods: list[str], n_bins: int):
    """One row per group, AE and Winkler score in the two columns, on logarithmic axes: per method the geometric mean
    over the datasets with bars of ``geometric_summary``; returns the figure and the plotted values."""
    x = np.arange(n_bins)
    fig, axes = plt.subplots(
        len(rows),
        2,
        figsize=(9.2, 0.7 + 3.1 * len(rows)),
        sharex=True,
        sharey="col",
        squeeze=False,
        layout="constrained",
    )
    values = []
    for r, (label, part) in enumerate(rows):
        present = [method for method in methods if method in set(part["method"])]
        for c, (metric, name) in enumerate(METRICS):
            ax = axes[r, c]
            summary = geometric_summary(part, metric, present)
            n_datasets = int(summary["n_datasets"].iloc[0])
            for i, method in enumerate(present):
                line = summary.loc[summary["method"] == method].set_index("level").reindex(x)
                shift = DODGE * (i - (len(present) - 1) / 2)
                y = line["value"].to_numpy()
                ax.errorbar(
                    x + 1 + shift,
                    y,
                    yerr=[y - line["lower"].to_numpy(), line["upper"].to_numpy() - y],
                    color=SHIFT_PROFILE_COLORS.get(method, "#8c8c8c"),
                    linewidth=1.4,
                    marker="o",
                    markersize=3.5,
                    elinewidth=0.9,
                    capsize=2.5,
                    capthick=0.9,
                    label=method_table_label(method),
                )
                values.extend(
                    {
                        "group": label,
                        "n_datasets": n_datasets,
                        "metric": metric,
                        "method": method,
                        "level": level + 1,
                        "value": row["value"],
                        "lower": row["lower"],
                        "upper": row["upper"],
                    }
                    for level, row in line.iterrows()
                )
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(which="major", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(8))
            if r == 0:
                ax.set_title(name, fontsize=font_size(11))
            if c == 0:
                ax.set_ylabel(f"{label} ({n_datasets} datasets)", fontsize=font_size(10))
            if r == len(rows) - 1:
                ax.set_xticks(x + 1, [f"D{i + 1}" for i in x])
                ax.set_xlabel(
                    "within-pair shift decile" if n_bins == 10 else "within-pair shift level", fontsize=font_size(10)
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=font_size(9))
    return fig, pd.DataFrame(values)


def generate_plots(args):
    out_dir = plot_dir(PLOT_NAME)
    for acc_name in args.metrics:
        df = load_scores(args.domain, acc_name, args.methods)
        df = shift_levels(add_bag_index(df), args.bins)
        profile = dataset_profile(df)
        figures = []
        if "by_k" in args.figures:
            figures.append(("by_k", k_groups(profile)))
        if "pooled" in args.figures:
            figures.append(("pooled", [("all $K$", profile)]))
        for name, rows in figures:
            fig, values = plot_profile(rows, args.methods, args.bins)
            stem = os.path.join(out_dir, f"{PLOT_NAME}_{name}_{safe_name(args.domain)}_{safe_name(acc_name)}")
            if args.add_stem is not None:
                stem = f"{stem}_{args.add_stem}"
            for fmt in args.formats:
                fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
            values.to_csv(f"{stem}.csv", index=False)
            plt.close(fig)
            print(f"plot generated: {stem}")


def parse_args(argv=None):
    parser = ArgumentParser(description="Shift profile of the methods of main (AE and Winkler score by shift decile).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(DEFAULT_FIGURES))
    parser.add_argument("--domain", default=DEFAULT_DOMAIN)
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=["vanilla_accuracy"])
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    # the paper leaves SQS-B out (uncawe/paper.py, user's decision of 2026-11-25); the bars remove each dataset's mean
    # over the remaining methods
    parser.add_argument("--exclude-methods", nargs="*", default=[], help="Methods removed from --methods")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    # also pdf since 2026-11-24, for the paper folder (uncawe/paper.py)
    parser.add_argument("--pdf", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args(argv)
    args.methods = [method for method in args.methods if method not in args.exclude_methods]
    args.formats = ["png"] + (["svg"] if args.svg else []) + (["pdf"] if args.pdf else [])
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
