"""Line plots of the allocation $\\rho$ of SQS-S on the pairs of ``main`` (E6.2 of ``docs/ext6_sqs_s_design.md``, card
WP-17).

The figures read the per-pair CSV of :mod:`uncawe.tables.pq_rho` (run ``tables/pq_rho.py --tables`` first): one figure
per metric (vanilla accuracy, macro-F1, macro-K), one panel per measure (quantification error, absolute error,
coverage, amplitude, Winkler score, minimum class support), each measure against $\\rho$: each pair as a faint line
(over the levels it has), and in bold the mean over the pairs that have every level, as in the summary table. The
coverage panel has a reference line at the nominal level $0.95$. A second figure per metric has two panels only,
the absolute error (AE, left) and the Winkler score (W, right), on a log scale, with the mean and the median over the
pairs that have every level and no title. Every score is against the realised table of each target bag.

Output: ``<OUT_DIR>/uncawe/plots/pq_rho/pq_rho_<metric>.{png,svg}`` and ``pq_rho_ae_w_<metric>.{png,svg}``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from uncawe import env
from uncawe.experiments.methods import CONFIDENCE_LEVEL
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.common import font_size
from uncawe.tables.pq_rho import KEYS, MEASURES, METRIC_LABELS, pairs_csv_path

PLOT_NAME = "pq_rho"
# measures that are shares in [0, 1], drawn on a fixed range
SHARE_MEASURES = ("coverage", "class_support")

PANEL_LABELS = {
    "quant_error": "quantification error",
    "acc_err": "absolute error",
    "coverage": "coverage",
    "amplitude": "amplitude",
    "winkler": "Winkler score",
    "class_support": "min. class support",
}
PAIR_LINE = {"color": "#4C78A8", "alpha": 0.15, "linewidth": 0.7}
MEAN_LINE = {"color": "#1f1f1f", "linewidth": 2.2, "marker": "o"}
REFERENCE_LINE = {"color": "#8c8c8c", "linewidth": 0.8, "linestyle": "--"}
MEDIAN_LINE = {"color": "#d62728", "linewidth": 1.6, "marker": "s", "linestyle": "--"}
# the two-panel figure: absolute error on the left, Winkler score on the right, log scale
AE_W_PANELS = (("acc_err", "AE"), ("winkler", "W"))


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or pairs_csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/pq_rho.py --tables first.")
    return pd.read_csv(path)


def plot_metric(table: pd.DataFrame, acc_name: str):
    """The six panels of one metric; ``table`` has the per-pair rows of that metric."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    common = table.loc[table["common"].astype(bool)]
    levels = sorted(table["rho"].unique())
    n_pairs = common[KEYS].drop_duplicates().shape[0]
    for ax, measure in zip(axes.ravel(), MEASURES):
        for _, pair in table.groupby(KEYS, sort=False):
            pair = pair.sort_values("rho")
            ax.plot(pair["rho"], pair[measure], **PAIR_LINE)
        if not common.empty:
            mean = common.groupby("rho")[measure].mean()
            ax.plot(mean.index, mean.to_numpy(), **MEAN_LINE)
        if measure == "coverage":
            ax.axhline(CONFIDENCE_LEVEL, **REFERENCE_LINE)
        if measure in SHARE_MEASURES:
            # shares in [0, 1]; without a fixed range and offset, values within 1e-5 of 1
            # are drawn as an offset axis that reads as a wide spread
            ax.set_ylim(-0.05, 1.05)
            ax.ticklabel_format(axis="y", useOffset=False)
        ax.set_title(PANEL_LABELS[measure], fontsize=font_size(11))
        ax.set_xticks(levels)
        ax.grid(True, color="#d9d9d9", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("$\\rho=|V_1|/|V|$", fontsize=font_size(11))
    fig.suptitle(
        f"SQS-S against the allocation $\\rho$ (E6.2), {METRIC_LABELS.get(acc_name, acc_name)}; "
        f"bold: mean over the {n_pairs} pairs with every level",
        fontsize=font_size(12),
    )
    fig.tight_layout()
    return fig


def plot_ae_w(table: pd.DataFrame):
    """AE (left) and W (right) against $\\rho$ on a log scale, without a title; ``table`` has the per-pair rows of
    one metric. Per-pair values span over two orders of magnitude, so a linear axis hides all but the worst pairs."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    common = table.loc[table["common"].astype(bool)]
    levels = sorted(table["rho"].unique())
    for ax, (measure, label) in zip(axes, AE_W_PANELS):
        for _, pair in table.groupby(KEYS, sort=False):
            pair = pair.sort_values("rho")
            ax.plot(pair["rho"], pair[measure], **PAIR_LINE)
        if not common.empty:
            by_level = common.groupby("rho")[measure]
            mean, median = by_level.mean(), by_level.median()
            ax.plot(mean.index, mean.to_numpy(), label="mean", **MEAN_LINE)
            ax.plot(median.index, median.to_numpy(), label="median", **MEDIAN_LINE)
        ax.set_yscale("log")
        ax.set_title(label, fontsize=font_size(11))
        ax.set_xticks(levels)
        ax.set_xlabel("$\\rho=|V_1|/|V|$", fontsize=font_size(11))
        ax.grid(True, which="both", color="#e5e5e5", linewidth=0.5)
    if not common.empty:
        axes[0].legend(frameon=False)
    fig.tight_layout()
    return fig


def save(fig, stem: str, args):
    if args.add_stem is not None:
        stem = f"{stem}_{args.add_stem}"
    for fmt in args.formats:
        fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"plot generated: {stem}")


def generate_plots(args):
    table = load_table(args.csv)
    out_dir = plot_dir()
    for acc_name in args.metrics:
        sub = table.loc[table["acc_name"] == acc_name]
        if sub.empty:
            print(f"{acc_name}: no rows in the per-pair CSV, skipped")
            continue
        save(plot_metric(sub, acc_name), os.path.join(out_dir, f"{PLOT_NAME}_{acc_name}"), args)
        save(plot_ae_w(sub), os.path.join(out_dir, f"{PLOT_NAME}_ae_w_{acc_name}"), args)


def parse_args(argv=None):
    parser = ArgumentParser(description="Line plots of the allocation rho of SQS-S on the pairs of main (E6.2).")
    parser.add_argument("--metrics", "--figures", nargs="+", choices=get_acc_names(), default=get_acc_names())
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/pq_rho.py (default: its output)")
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
