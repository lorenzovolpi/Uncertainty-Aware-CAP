r"""Violins and scatters of E5.5, right-side and wrong-side priors under strong shifts, with SQS references (work-plan
card WP-27, ``docs/ext5_baycap_priors_tempering.md``).

The figures read the per-bag and per-pair CSVs of :mod:`uncawe.tables.binform_side_prior` (run
``tables/binform_side_prior.py --tables`` first). Every method targets the realised table of each bag and is scored
against it.

- **Pair violins** (``pair-violins``): one figure per distribution of the bags ($G_c=\operatorname{Dir}(10\cdot\mathbf
  1)$, ``centre``; $G_e=\operatorname{Dir}(\tfrac1{10}\mathbf 1)$, ``edge``), group of pairs (all, binary $K=2$,
  multiclass $K\ge3$; ``PAIR_GROUPS``) and metric (vanilla accuracy by default), with two panels side by side: the AE
  and the Winkler score of the per-pair means over the $100$ bags (``pair_means``), the pairs drawn as points, on a
  logarithmic vertical axis. Methods left to right in ``METHOD_ORDER`` (SQS-S, SQS-H, SQS-Hbs, SQS-B, BayCAP-Hf,
  BayCAP-KDEy, BayCAP-KDEy-cb; the panels widen with the number of violins, ``PANEL_WIDTH``); a method without a prior has one violin, a method with a prior three, the wrong-side prior on the
  left, the uniform prior in the middle and the right-side prior on the right. The uniform prior and the methods
  without a prior are drawn light and hatched, the informative priors solid. A white dot marks the median and a black
  diamond the arithmetic mean of the values (not floored, drawn at $\max(\text{mean},10^{-5})$); the mean of a violin is
  the value of the summary table (mean over the pairs of the per-pair means). The Winkler score is skewed by the
  penalties of the intervals that miss, so the median follows the typical pair and the mean follows the tables.
- **Bag violins** (``bag-violins``, not drawn by default): the same figures with, per violin, the AE and the Winkler score of every bag (pairs
  $\times$ $100$ bags), without the points.

The violins are drawn on $\log_{10}$ of the values, floored at $10^{-5}$ (``FLOOR``; an AE or a Winkler score of $0$, from
an interval of zero width that covers, would have no logarithm); values that are not finite are left out. The density
is a Gaussian kernel estimate evaluated up to ``CUT`` bandwidths beyond the smallest and the largest value (not below
the floor), so the outline closes smoothly instead of being cut flat at the extreme values.

- **Scatter** (``winkler``): per method with a prior, distribution and metric, the per-pair Winkler score with the uniform
  prior against that with the right-side (top) and the wrong-side prior (bottom), as in E5.4
  (``plots/binform_prior.py:plot_winkler``).

Output: ``<OUT_DIR>/uncawe/plots/binform_side_prior/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator
from scipy.stats import gaussian_kde

from uncawe import env
from uncawe.experiments.binform import E55_DISTRIBUTIONS, E55_PRIOR_BASES, SQS_B
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.binform_prior import plot_winkler
from uncawe.plots.common import ACC_NAME, METHOD_COLORS, font_size, method_plot_label
from uncawe.tables.binform_prior import ACC_LABELS
from uncawe.tables.binform_side_prior import INFORMATIVE_ROLES, NO_PRIOR, ROLE_LABELS, csv_path

PLOT_NAME = "binform_side_prior"
FIGURES = ("pair-violins", "bag-violins", "winkler")
DEFAULT_FIGURES = ("pair-violins", "winkler")
LEVELS = ("bags", "pairs")
VIOLIN_FIGURES = {"pair-violins": "pairs", "bag-violins": "bags"}
PAIR_KEYS = ["acc_name", "distribution", "method", "role", "dataset", "classifier", "n_classes"]
PAIR_GROUPS = {"all": "all pairs", "binary": "binary pairs ($K=2$)", "multiclass": "multiclass pairs ($K\\geq 3$)"}
FLOOR = 1e-5
CUT = 2.0
BANDWIDTHS = {"bags": 0.15, "pairs": 0.3}
PANELS = {"ae": "AE", "winkler": "Winkler score"}
# the per-instance BayCAP-KDEy methods in place of BayCAP-KDEy-mlp-real since card WP-33
METHOD_ORDER = ("PQ", "SQS-H", "SQS-Hbs", SQS_B, "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb")
SIDE_ROLES = ("wrong", "uniform", "right")
ROLE_TICKS = {"wrong": "wrong", "uniform": "uniform", "right": "right", NO_PRIOR: "no prior"}
GROUP_GAP = 0.9
# the width of a panel: 7.5 inches for the 12 violins of WP-27, the same width per violin for more (card WP-33)
PANEL_WIDTH = 7.5
PANEL_VIOLINS = 12
DISTRIBUTION_TITLES = {
    "centre": "bags from $\\operatorname{Dir}(10\\cdot\\mathbf{1})$",
    "edge": "bags from $\\operatorname{Dir}(\\frac{1}{10}\\mathbf{1})$",
}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(name: str = "pairs", path: str | None = None) -> pd.DataFrame:
    path = path or csv_path(name=name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/binform_side_prior.py --tables first.")
    return pd.read_csv(path, low_memory=False)


def short_label(method: str) -> str:
    """The plot label of ``method`` without the input and estimand suffixes (``BayCAP-KDEy``, ``BayCAP-Hf``)."""
    return method_plot_label(method).removesuffix("-real").removesuffix("-mlp")


def violin_cells() -> list[tuple[str, str, float]]:
    """The ``(method, role, position)`` of the violins of a figure, left to right in ``METHOD_ORDER``: three violins
    (wrong side, uniform, right side) for a method with a prior, one (no prior) for the others; the groups of a method
    are separated by ``GROUP_GAP``."""
    cells, position = [], 0.0
    for method in METHOD_ORDER:
        roles = SIDE_ROLES if method in E55_PRIOR_BASES else (NO_PRIOR,)
        for role in roles:
            cells.append((method, role, position))
            position += 1.0
        position += GROUP_GAP
    return cells


def log_values(bags: pd.DataFrame, score: str) -> np.ndarray:
    """$\\log_{10}$ of the finite values of ``score``, floored at ``FLOOR``."""
    values = bags[score].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return np.log10(np.maximum(values, FLOOR))


def finite_mean(bags: pd.DataFrame, score: str) -> float:
    """The arithmetic mean of the finite values of ``score`` (not floored); NaN without values."""
    values = bags[score].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else np.nan


def pair_means(bags: pd.DataFrame) -> pd.DataFrame:
    """Per pair (and metric, distribution, method, role), the means over the bags of the scores of ``PANELS``, NaN
    values left out: the per-pair values that the summary table averages."""
    return bags.groupby(PAIR_KEYS, as_index=False, dropna=False)[list(PANELS)].mean()


def pair_group(rows: pd.DataFrame, group: str) -> pd.DataFrame:
    """The rows of the pairs of ``group`` (``PAIR_GROUPS``)."""
    if group == "binary":
        return rows.loc[rows["n_classes"] == 2]
    if group == "multiclass":
        return rows.loc[rows["n_classes"] > 2]
    if group == "all":
        return rows
    raise ValueError(f"unknown group {group!r}, expected one of {tuple(PAIR_GROUPS)}")


def violin_stats(values: np.ndarray, bandwidth: float, points: int = 200) -> dict:
    """The statistics that ``Axes.violin`` draws, with the Gaussian kernel density of ``values`` (on the $\\log_{10}$
    scale) evaluated from ``CUT`` bandwidths below the smallest value (not below $\\log_{10}$ ``FLOOR``) to ``CUT``
    bandwidths above the largest, so that the outline closes instead of being cut at the extreme values."""
    low, high = float(values.min()), float(values.max())
    floor = np.log10(FLOOR)
    if values.size < 2 or np.ptp(values) == 0:
        coords = np.linspace(max(low - 0.05, floor), high + 0.05, points)
        vals = np.exp(-0.5 * ((coords - low) / 0.02) ** 2)
    else:
        kde = gaussian_kde(values, bw_method=bandwidth)
        spread = CUT * float(np.sqrt(kde.covariance[0, 0]))
        coords = np.linspace(max(low - spread, floor), high + spread, points)
        vals = kde(coords)
    return {
        "coords": coords,
        "vals": vals,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": low,
        "max": high,
        "quantiles": np.array([]),
    }


def plot_violins(
    bags: pd.DataFrame, distribution: str, acc_name: str = ACC_NAME, level: str = "pairs", group: str = "all"
):
    """The violin figure of ``distribution``, over the per-pair means of ``pair_means`` (``level="pairs"``) or over the
    bags, for the pairs of ``group``; ``None`` without rows."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")
    rows = bags.loc[(bags["distribution"] == distribution) & (bags["acc_name"] == acc_name)]
    rows = pair_group(rows, group)
    if rows.empty:
        return None
    if level == "pairs":
        rows = pair_means(rows)
    cells = [
        (method, role, position, part)
        for method, role, position in violin_cells()
        if not (part := rows.loc[(rows["method"] == method) & (rows["role"] == role)]).empty
    ]
    width = PANEL_WIDTH * max(1.0, len(cells) / PANEL_VIOLINS)
    fig, axes = plt.subplots(1, len(PANELS), figsize=(width * len(PANELS), 5.2), layout="constrained")
    for ax, (score, label) in zip(axes, PANELS.items()):
        data = [log_values(part, score) for *_, part in cells]
        keep = [i for i, d in enumerate(data) if d.size > 0]
        positions = np.array([cells[i][2] for i in keep])
        stats = [violin_stats(data[i], BANDWIDTHS[level]) for i in keep]
        parts = ax.violin(stats, positions=positions, widths=0.8, showextrema=False)
        for body, i in zip(parts["bodies"], keep):
            method, role, *_ = cells[i]
            light = role in ("uniform", NO_PRIOR)
            body.set_facecolor(METHOD_COLORS.get(method, "#888888"))
            body.set_edgecolor("#1f1f1f")
            body.set_linewidth(0.8)
            body.set_alpha(0.45 if light else 0.9)
            if light:
                body.set_hatch("///")
        if level == "pairs":
            rng = np.random.default_rng(0)
            for position, i in zip(positions, keep):
                jitter = rng.uniform(-0.12, 0.12, data[i].size)
                ax.scatter(position + jitter, data[i], s=5, color="#1f1f1f", alpha=0.35, linewidth=0, zorder=2)
        medians = [np.median(data[i]) for i in keep]
        means = [np.log10(max(finite_mean(cells[i][3], score), FLOOR)) for i in keep]
        ax.scatter(positions, medians, s=22, color="white", edgecolor="#1f1f1f", linewidth=0.8, zorder=3)
        ax.scatter(positions, means, s=30, marker="D", color="#1f1f1f", edgecolor="white", linewidth=0.6, zorder=4)
        ax.set_xticks([position for _, _, position, _ in cells])
        ax.set_xticklabels(
            [ROLE_TICKS[role] for _, role, _, _ in cells], rotation=90, fontsize=font_size(7.5), color="#444444"
        )
        methods = list(dict.fromkeys(method for method, *_ in cells))
        centres = [np.mean([position for m, _, position, _ in cells if m == method]) for method in methods]
        method_axis = ax.secondary_xaxis(-0.2)
        method_axis.set_xticks(centres)
        method_axis.set_xticklabels([short_label(method) for method in methods], fontsize=font_size(9))
        method_axis.tick_params(length=0)
        method_axis.spines["bottom"].set_visible(False)
        low, high = ax.get_ylim()
        ticks = np.arange(np.floor(low), np.ceil(high) + 1)
        ax.yaxis.set_major_locator(FixedLocator(ticks))
        ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks], fontsize=font_size(9))
        ax.set_ylim(low, high)
        ax.set_ylabel(f"{label} (log scale)", fontsize=font_size(10))
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.set_title(label, fontsize=font_size(11))
    markers = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=5, color="white", markeredgecolor="#1f1f1f"),
        plt.Line2D([], [], marker="D", linestyle="", markersize=5, color="#1f1f1f", markeredgecolor="white"),
    ]
    fig.legend(markers, ["median", "mean"], loc="outside lower center", ncols=2, fontsize=font_size(8), frameon=False)
    unit = "every bag" if level == "bags" else "per-pair means over the bags"
    fig.suptitle(
        f"E5.5: {DISTRIBUTION_TITLES[distribution]}, {PAIR_GROUPS[group]}, {unit} ({ACC_LABELS[acc_name]})",
        fontsize=font_size(12),
    )
    return fig


def plot_scatter(table: pd.DataFrame, method: str, distribution: str, acc_name: str = ACC_NAME):
    """The scatter of E5.4 for E5.5 (uniform against right-side and wrong-side prior); ``None`` without rows or for a
    method without a prior."""
    rows = table.loc[(table["distribution"] == distribution) & (table["method"] == method)]
    if rows.empty or method not in E55_PRIOR_BASES:
        return None
    title = f"E5.5, {method_plot_label(method)}: {DISTRIBUTION_TITLES[distribution]} ({ACC_LABELS[acc_name]})"
    return plot_winkler(rows, method, acc_name, priors=INFORMATIVE_ROLES, labels=ROLE_LABELS, title=title)


def _save(fig, stem: str, args) -> None:
    if args.add_stem is not None:
        stem = f"{stem}_{args.add_stem}"
    for fmt in args.formats:
        fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"plot generated: {stem}")


def generate_plots(args):
    out_dir = plot_dir()
    if any(figure in args.figures for figure in VIOLIN_FIGURES):
        bags = load_table("bags", args.bags_csv)
    for figure, level in VIOLIN_FIGURES.items():
        if figure not in args.figures:
            continue
        prefix = f"{PLOT_NAME}_{figure.replace('-', '_')}"
        for acc_name in args.metrics:
            for distribution in args.distributions:
                for group in args.groups:
                    fig = plot_violins(bags, distribution, acc_name, level, group)
                    if fig is None:
                        print(f"no bags for {distribution}, {group} ({acc_name}), skipped")
                        continue
                    _save(fig, os.path.join(out_dir, f"{prefix}_{distribution}_{group}_{acc_name}"), args)
    if "winkler" in args.figures:
        table = load_table("pairs", args.csv)
        for acc_name in args.metrics:
            for method in args.methods:
                for distribution in args.distributions:
                    fig = plot_scatter(table, method, distribution, acc_name)
                    if fig is None:
                        print(f"no rows for {method}, {distribution} ({acc_name}), skipped")
                        continue
                    _save(fig, os.path.join(out_dir, f"{PLOT_NAME}_winkler_{method}_{distribution}_{acc_name}"), args)


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Plots of E5.5, right-side and wrong-side priors under strong shifts.")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(DEFAULT_FIGURES))
    parser.add_argument("--methods", nargs="+", choices=E55_PRIOR_BASES, default=list(E55_PRIOR_BASES))
    parser.add_argument("--distributions", nargs="+", choices=E55_DISTRIBUTIONS, default=list(E55_DISTRIBUTIONS))
    parser.add_argument("--groups", nargs="+", choices=PAIR_GROUPS, default=list(PAIR_GROUPS))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument(
        "--csv", default=None, help="Per-pair CSV of tables/binform_side_prior.py (default: its output)"
    )
    parser.add_argument(
        "--bags-csv", default=None, help="Per-bag CSV of tables/binform_side_prior.py (default: its output)"
    )
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    # also pdf since 2026-11-24, for the paper folder (uncawe/paper.py)
    parser.add_argument("--pdf", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args(argv)
    args.formats = ["png"] + (["svg"] if args.svg else []) + (["pdf"] if args.pdf else [])
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
