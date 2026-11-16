r"""Plots of the error split across $|V|$ and $|U|$ in the study ``target_size`` (work-plan cards WP-21, E4.2b of
``docs/ext4_uncertainty_budget.md``, and WP-22, E4.2a).

Reads the per-cell and per-level tables of :mod:`uncawe.tables.target_size` (vanilla accuracy, every method scored against the realised
table of each bag; the split is an analysis that uses the true prevalence of the bag and the labels of the pool).

Figures (``--figures``):

- ``v-size``: the RMS of the fixed part of each part ($e_1$, $e_2$, $e_3$, one row each) against $|V|$, log–log, one
  column per method, one colour per dataset, for one classifier (``--classifier``, default LR) and one bag size
  (``--size``, default $1000$). For each $V$ seed a point gives the RMS over the three prevalences of the fixed parts
  of that seed (the full $V$ has one seed), and the line joins, level by level, the means of these points over the
  seeds.
- ``u-size`` (E4.2a): with the full $V$, the RMS of each part ($e_1$, $e_2$, $e_3$, one row each) against $|U|$,
  log–log, one column per method, one colour per dataset, for one classifier (``--classifier``): a solid line for the
  fixed part (RMS over the three prevalences of the mean of the part over the bags of a cell) and a dashed line for the
  bag part (RMS over the bags of the deviation of each bag from the mean of its cell); ``--size`` is not used.
- ``v-total`` (E4.2b) and ``u-total`` (E4.2a): the total error $\hat A-A(h,U)=e_1+e_2+e_3$ aggregated over the pairs
  (both classifiers), against the fraction of $V$ (at ``--size``, default $|U|=1000$) and against $|U|$ (full $V$),
  one column for the binary and one for the multiclass pairs, both methods on the same axes, one colour each. Top row: per pair, the AE of the
  total (mean of $|\hat A-A(h,U)|$ over the bags of the pair at that size, i.e. over the prevalences and the $V$
  seeds); the line is the mean over the pairs of the group, the band the 25th–75th percentile over them. Bottom row:
  the RMS of the total (thick) and of its fixed (dash-dot) and bag (dashed) parts, each the square root of the mean
  over the pairs of the per-pair mean square over the bags, so that $\text{MSE}_\text{total}=\text{MSE}_\text{fixed}+
  \text{MSE}_\text{bag}$ holds on the lines too. A level of $V$ present only on some pairs of a group (the $1/16$
  level) is drawn with an open marker, a dotted segment, an interquartile bar instead of the band, and the number of
  its pairs; ``--classifier`` is not used.

Output: ``<OUT_DIR>/uncawe/plots/target_size/<figure>/`` (``v_size``, ``u_size``, ``v_total``, ``u_total``).
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from uncawe import env
from uncawe import error_split as es
from uncawe.experiments.target_size import EXPERIMENT, LEVELS, METHODS
from uncawe.plots.common import font_size, method_plot_label
from uncawe.tables import main as main_tables
from uncawe.tables.target_size import PARTS, compute

FIGURES = ("v-size", "u-size", "v-total", "u-total")
PART_NAMES = {"e1": "$e_1$ prevalence", "e2": "$e_2$ recalls", "e3": "$e_3$ luck of the bag"}
DATASET_CMAP = "tab10"
GROUPS = ("binary", "multiclass")
METHOD_COLORS = {"BayesCAP-Hf-real": "#1f77b4", "BayCAP-KDEy-mlp-real": "#d62728"}
PAIR_KEYS = ["method", "dataset", "classifier", "n_classes", "bag_size", "v_level"]


def seed_points(cells: pd.DataFrame, classifier: str, size: int) -> pd.DataFrame:
    """Per method, dataset, level and $V$ seed: $|V|$ and, per part, the RMS over the prevalences of the fixed parts."""
    cells = cells.loc[(cells["classifier"] == classifier) & (cells["bag_size"] == size)]
    keys = ["method", "dataset", "n_classes", "v_level", "v_seed"]
    rows = []
    for key, group in cells.groupby(keys, sort=False):
        row = dict(zip(keys, key)) | {"v_size": int(group["v_size"].iloc[0])}
        row |= {name: es.rms(group[f"{name}_fixed"]) for name in PARTS}
        rows.append(row)
    return pd.DataFrame(rows)


def plot_v_size(cells: pd.DataFrame, classifier: str = "LR", size: int = 1000):
    points = seed_points(cells, classifier, size)
    if points.empty:
        raise RuntimeError(f"No cell with classifier {classifier} and |U|={size}.")
    datasets = points.drop_duplicates("dataset").sort_values("n_classes", kind="stable")["dataset"].tolist()
    colors = dict(zip(datasets, plt.get_cmap(DATASET_CMAP)(np.arange(len(datasets)) % 10)))
    methods = [m for m in METHODS if m in set(points["method"])]
    fig, axes = plt.subplots(len(PARTS), len(methods), figsize=(4.4 * len(methods), 3.2 * len(PARTS)), squeeze=False)
    for col, method in enumerate(methods):
        for row, name in enumerate(PARTS):
            ax = axes[row, col]
            for dataset in datasets:
                part = points.loc[(points["method"] == method) & (points["dataset"] == dataset)]
                if part.empty:
                    continue
                ax.scatter(part["v_size"], part[name], color=colors[dataset], s=14, alpha=0.6, linewidths=0)
                means = (
                    part.assign(order=part["v_level"].map(list(LEVELS).index))
                    .groupby(["order", "v_size"], as_index=False)[name]
                    .mean()
                    .sort_values("v_size")
                )
                ax.plot(means["v_size"], means[name], color=colors[dataset], linewidth=1.2, marker="o", markersize=3)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="both", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(7))
            if row == 0:
                ax.set_title(method_plot_label(method).replace("-real", ""), fontsize=font_size(9))
            if col == 0:
                ax.set_ylabel(f"RMS of the fixed part\n{PART_NAMES[name]}", fontsize=font_size(8))
            if row == len(PARTS) - 1:
                ax.set_xlabel("$|V|$", fontsize=font_size(8))
    handles = [
        Line2D([], [], color=colors[d], marker="o", markersize=3, label=main_tables.dataset_map.get(d, d))
        for d in datasets
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(8), frameon=False)
    fig.suptitle(f"{classifier}, $|U|={size}$", fontsize=font_size(9))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_u_size(levels: pd.DataFrame, classifier: str = "LR", size: int | None = None):
    """E4.2a: the RMS of the fixed (solid) and of the bag (dashed) part of each part against $|U|$, full $V$.
    ``size`` is accepted for a common signature with ``plot_v_size`` and not used."""
    rows = levels.loc[(levels["classifier"] == classifier) & (levels["v_level"] == "full")]
    if rows["bag_size"].nunique() < 2:
        raise RuntimeError(f"No results with classifier {classifier} and the full V at two or more bag sizes.")
    datasets = rows.drop_duplicates("dataset").sort_values("n_classes", kind="stable")["dataset"].tolist()
    colors = dict(zip(datasets, plt.get_cmap(DATASET_CMAP)(np.arange(len(datasets)) % 10)))
    methods = [m for m in METHODS if m in set(rows["method"])]
    fig, axes = plt.subplots(len(PARTS), len(methods), figsize=(4.4 * len(methods), 3.2 * len(PARTS)), squeeze=False)
    for col, method in enumerate(methods):
        for row, name in enumerate(PARTS):
            ax = axes[row, col]
            for dataset in datasets:
                part = rows.loc[(rows["method"] == method) & (rows["dataset"] == dataset)].sort_values("bag_size")
                if part.empty:
                    continue
                for kind, style in (("fixed", "-"), ("bag", "--")):
                    ax.plot(
                        part["bag_size"],
                        part[f"{name}_{kind}_rms"],
                        color=colors[dataset],
                        linestyle=style,
                        linewidth=1.2,
                        marker="o",
                        markersize=3,
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="both", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(7))
            if row == 0:
                ax.set_title(method_plot_label(method).replace("-real", ""), fontsize=font_size(9))
            if col == 0:
                ax.set_ylabel(f"RMS\n{PART_NAMES[name]}", fontsize=font_size(8))
            if row == len(PARTS) - 1:
                ax.set_xlabel("$|U|$", fontsize=font_size(8))
    handles = [
        Line2D([], [], color=colors[d], marker="o", markersize=3, label=main_tables.dataset_map.get(d, d))
        for d in datasets
    ]
    handles += [
        Line2D([], [], color="black", linestyle="-", label="fixed part"),
        Line2D([], [], color="black", linestyle="--", label="bag part"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(8), frameon=False)
    fig.suptitle(f"{classifier}, full $V$", fontsize=font_size(9))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def pair_totals(bags: pd.DataFrame) -> pd.DataFrame:
    """Per (method, pair, $|U|$, level of $V$): the AE of the total and the mean squares over the bags of the total and
    of its fixed and bag parts. Pooled over the cells, the mean square of the total is still the sum of the other two,
    since within each cell the bag parts sum to zero."""
    frame = bags.assign(
        ae=bags["total"].abs(),
        mse_total=bags["total"] ** 2,
        mse_fixed=bags["total_fixed"] ** 2,
        mse_bag=bags["total_bag"] ** 2,
    )
    return frame.groupby(PAIR_KEYS, as_index=False, sort=False)[["ae", "mse_total", "mse_fixed", "mse_bag"]].mean()


def group_totals(pairs: pd.DataFrame, x: str) -> pd.DataFrame:
    """Per method, group (binary or multiclass) and value of ``x``: the mean and the 25th and 75th percentiles over the
    pairs of the AE, the root of the mean over the pairs of each mean square, the number of pairs, and whether every
    pair of the group (method and group) is present at that value."""
    pairs = pairs.assign(group=np.where(pairs["n_classes"] == 2, "binary", "multiclass"))
    pairs["pair"] = pairs["dataset"] + "|" + pairs["classifier"]
    n_group = pairs.groupby(["method", "group"])["pair"].nunique().rename("n_group")
    rows = []
    for (method, group, value), sub in pairs.groupby(["method", "group", x], sort=False):
        row = {"method": method, "group": group, x: value, "n_pairs": sub["pair"].nunique()}
        row |= {"ae_mean": sub["ae"].mean(), "ae_q25": sub["ae"].quantile(0.25), "ae_q75": sub["ae"].quantile(0.75)}
        row |= {f"rms_{k}": float(np.sqrt(sub[f"mse_{k}"].mean())) for k in ("total", "fixed", "bag")}
        rows.append(row)
    out = pd.DataFrame(rows).join(n_group, on=["method", "group"])
    out["complete"] = out["n_pairs"] == out["n_group"]
    return out.sort_values(["method", "group", x], kind="stable").reset_index(drop=True)


def _plot_totals(groups: pd.DataFrame, x: str, xlabel: str, title: str, ticks: dict | None = None):
    """One column per group of pairs (binary, multiclass), both methods on the same axes, one colour each."""
    methods = [m for m in METHODS if m in set(groups["method"])]
    names = [g for g in GROUPS if g in set(groups["group"])]
    fig, axes = plt.subplots(2, len(names), figsize=(4.4 * len(names), 6.4), squeeze=False)
    for col, group in enumerate(names):
        ax_ae, ax_rms = axes[0, col], axes[1, col]
        for method in methods:
            color = METHOD_COLORS[method]
            rows = groups.loc[(groups["method"] == method) & (groups["group"] == group)].sort_values(x)
            if rows.empty:
                continue
            full, part = rows.loc[rows["complete"]], rows.loc[~rows["complete"]]
            ax_ae.plot(full[x], full["ae_mean"], color=color, linewidth=1.4, marker="o", markersize=3.5)
            ax_ae.fill_between(full[x], full["ae_q25"], full["ae_q75"], color=color, alpha=0.15, linewidth=0)
            ax_rms.plot(full[x], full["rms_total"], color=color, linewidth=1.6, marker="o", markersize=3.5)
            for kind, style in (("fixed", "-."), ("bag", "--")):
                ax_rms.plot(rows[x], rows[f"rms_{kind}"], color=color, linewidth=0.9, linestyle=style)
            for _, row in part.iterrows():
                # the nearest value of x drawn with every pair of the group, joined by a dotted segment (the dashes
                # are the bag part)
                anchor = full.iloc[(np.log(full[x].astype(float)) - np.log(float(row[x]))).abs().argmin()]
                for ax, name in ((ax_ae, "ae_mean"), (ax_rms, "rms_total")):
                    ax.plot([anchor[x], row[x]], [anchor[name], row[name]], color=color, linestyle=":", linewidth=1.2)
                    ax.plot(row[x], row[name], marker="o", markersize=5, markerfacecolor="white", color=color)
                ax_ae.vlines(row[x], row["ae_q25"], row["ae_q75"], color=color, linewidth=1.0, alpha=0.6)
                ax_ae.annotate(
                    f"{row['n_pairs']} pair" + ("s" if row["n_pairs"] != 1 else ""),
                    (row[x], row["ae_mean"]),
                    textcoords="offset points",
                    xytext=(5, -9),
                    fontsize=font_size(6),
                    color=color,
                )
        for row, ax in enumerate((ax_ae, ax_rms)):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(which="both", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(7))
            if ticks is not None:
                ax.set_xticks(list(ticks.values()), list(ticks.keys()))
                ax.xaxis.set_minor_locator(plt.NullLocator())
            if row == 1:
                ax.set_xlabel(xlabel, fontsize=font_size(8))
        n_group = groups.loc[groups["group"] == group, "n_group"].max()
        ax_ae.set_title(f"{group} ({n_group} pairs)", fontsize=font_size(9))
        if col == 0:
            ax_ae.set_ylabel("AE of the total\n(mean, 25th–75th percentile over the pairs)", fontsize=font_size(8))
            ax_rms.set_ylabel("RMS of the total", fontsize=font_size(8))
    handles = [
        Line2D(
            [], [], color=METHOD_COLORS[m], marker="o", markersize=3.5, label=method_plot_label(m).replace("-real", "")
        )
        for m in methods
    ]
    handles += [
        Line2D([], [], color="black", linewidth=1.6, label="total"),
        Line2D([], [], color="black", linewidth=0.9, linestyle="-.", label="fixed part"),
        Line2D([], [], color="black", linewidth=0.9, linestyle="--", label="bag part"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(8), frameon=False)
    fig.suptitle(title, fontsize=font_size(9))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def plot_v_total(bags: pd.DataFrame, classifier: str | None = None, size: int = 1000):
    """E4.2b: the total error aggregated over the pairs against the fraction of $V$, at $|U|=$ ``size``.
    ``classifier`` is accepted for a common signature and not used (both classifiers are aggregated)."""
    pairs = pair_totals(bags.loc[bags["bag_size"] == size])
    if pairs["v_level"].nunique() < 2:
        raise RuntimeError(f"No results at |U|={size} with two or more levels of V.")
    pairs["fraction"] = pairs["v_level"].map(LEVELS)
    ticks = {name: LEVELS[name] for name in sorted(set(pairs["v_level"]), key=LEVELS.get)}
    return _plot_totals(group_totals(pairs, "fraction"), "fraction", "fraction of $V$", f"$|U|={size}$", ticks)


def plot_u_total(bags: pd.DataFrame, classifier: str | None = None, size: int | None = None):
    """E4.2a: the total error aggregated over the pairs against $|U|$, full $V$. ``classifier`` and ``size`` are
    accepted for a common signature and not used."""
    pairs = pair_totals(bags.loc[bags["v_level"] == "full"])
    if pairs["bag_size"].nunique() < 2:
        raise RuntimeError("No results with the full V at two or more bag sizes.")
    return _plot_totals(group_totals(pairs, "bag_size"), "bag_size", "$|U|$", "full $V$")


# figure -> (plotter, table of ``tables.target_size.compute`` it reads, whether the file name carries ``--size``,
# whether it carries ``--classifier``)
PLOTTERS = {
    "v-size": (plot_v_size, "cells", True, True),
    "u-size": (plot_u_size, "levels", False, True),
    "v-total": (plot_v_total, "bags", True, False),
    "u-total": (plot_u_total, "bags", False, False),
}


def out_dir(figure: str) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, figure.replace("-", "_"))
    os.makedirs(path, exist_ok=True)
    return path


def generate_plots(args):
    out = compute()
    for name in args.figures:
        plotter, table, by_size, by_classifier = PLOTTERS[name]
        try:
            fig = plotter(out[table], classifier=args.classifier, size=args.size)
        except RuntimeError as error:
            print(f"{name} not drawn: {error}")
            continue
        stem = os.path.join(out_dir(name), f"{EXPERIMENT}_{name}")
        if by_classifier:
            stem = f"{stem}_{args.classifier}"
        if by_size:
            stem = f"{stem}_u{args.size}"
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Plots of the error split across |V| and |U| (E4.2, WP-21, WP-22).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--classifier", default="LR", help="Classifier of the pairs of v-size and u-size (default: LR)")
    parser.add_argument("--size", type=int, default=1000, help="Bag size |U| of v-size and v-total (default: 1000)")
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args(argv)
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
