"""Line plots of the allocation $\\rho$ of SQS-S on the pairs of ``main`` (E6.2 of ``docs/ext6_sqs_s_design.md``, card
WP-17).

The figures read the per-pair CSV of :mod:`uncawe.tables.pq_rho` (run ``tables/pq_rho.py --tables`` first): one figure
per metric (vanilla accuracy, macro-F1, macro-K), one panel per measure (quantification error, absolute error,
coverage, amplitude, Winkler score, minimum class support), each measure against $\\rho$: each pair as a faint line
(over the levels it has), and in bold the mean over the pairs that have every level, as in the summary table. The
coverage panel has a reference line at the nominal level $0.95$. Every score is against the realised table of each
target bag.

Output: ``<OUT_DIR>/uncawe/plots/pq_rho/pq_rho_<metric>.{png,svg}``.
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
        ax.set_title(PANEL_LABELS[measure], fontsize=font_size(11))
        ax.set_xticks(levels)
        ax.grid(True, color="#d9d9d9", linewidth=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("$\\rho=|V_q|/|V|$", fontsize=font_size(11))
    fig.suptitle(
        f"SQS-S against the allocation $\\rho$ (E6.2), {METRIC_LABELS.get(acc_name, acc_name)}; "
        f"bold: mean over the {n_pairs} pairs with every level",
        fontsize=font_size(12),
    )
    fig.tight_layout()
    return fig


def generate_plots(args):
    table = load_table(args.csv)
    out_dir = plot_dir()
    for acc_name in args.metrics:
        sub = table.loc[table["acc_name"] == acc_name]
        if sub.empty:
            print(f"{acc_name}: no rows in the per-pair CSV, skipped")
            continue
        fig = plot_metric(sub, acc_name)
        stem = os.path.join(out_dir, f"{PLOT_NAME}_{acc_name}")
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


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
