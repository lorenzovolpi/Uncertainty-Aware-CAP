r"""Violins and scatters of E5.5, right-side and wrong-side priors under strong shifts, with SQS references (work-plan
card WP-27, ``docs/ext5_baycap_priors_tempering.md``).

The figures read the per-bag and per-pair CSVs of :mod:`uncawe.tables.binform_side_prior` (run
``tables/binform_side_prior.py --tables`` first). Every method targets the realised table of each bag and is scored
against it.

- **Violins** (``violins``): one figure per distribution of the bags ($G_c=\operatorname{Dir}(10\cdot\mathbf 1)$,
  ``centre``; $G_e=\operatorname{Dir}(\tfrac1{10}\mathbf 1)$, ``edge``), role of the informative prior (right side,
  wrong side) and metric (vanilla accuracy by default), with two panels side by side: the AE and the Winkler score of
  every bag (all pairs $\times$ $100$ bags per violin) on a logarithmic vertical axis. The violins are drawn on
  $\log_{10}$ of the values, floored at $10^{-5}$ (``FLOOR``; an AE of $0$ or a Winkler score of $0$, from an interval
  of zero width that covers, would have no logarithm); bags where the method produced no draws are left out. For
  BayCAP-Hf, BayCAP-KDEy and SQS-B two violins, uniform prior (light, hatched) and the informative prior of the figure;
  for SQS-H, SQS-Hbs and SQS-S one violin, the same in the two figures of a distribution. A white dot marks the median
  and a black diamond the arithmetic mean of the values (not floored, drawn at $\max(\text{mean},10^{-5})$): the
  Winkler score is skewed, so the median follows the typical bag while the mean includes the large penalties of the
  intervals that miss.
- **Pair violins** (``pair-violins``): the same figures with, per violin, the $75$ per-pair means over the $100$ bags
  (``pair_means``) instead of the bags, with the pairs drawn as points. The mean of a violin is then the value of the
  summary table (mean over the pairs of the per-pair means), so the diamonds follow the ranking of the tables.
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

from uncawe import env
from uncawe.experiments.binform import E55_DISTRIBUTIONS, E55_PRIOR_BASES, E55_REFERENCES
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.binform_prior import plot_winkler
from uncawe.plots.common import ACC_NAME, METHOD_COLORS, font_size, method_plot_label
from uncawe.tables.binform_prior import ACC_LABELS
from uncawe.tables.binform_side_prior import INFORMATIVE_ROLES, NO_PRIOR, ROLE_LABELS, csv_path

PLOT_NAME = "binform_side_prior"
FIGURES = ("violins", "pair-violins", "winkler")
LEVELS = ("bags", "pairs")
PAIR_KEYS = ["acc_name", "distribution", "method", "role", "dataset", "classifier"]
FLOOR = 1e-5
PANELS = {"ae": "AE", "winkler": "Winkler score"}
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
    return pd.read_csv(path)


def violin_cells(role: str) -> list[tuple[str, str]]:
    """The ``(method, role)`` of the violins of a figure, left to right: for each method with a prior its uniform
    prior and ``role``, then the methods without a prior."""
    return [(base, r) for base in E55_PRIOR_BASES for r in ("uniform", role)] + [
        (base, NO_PRIOR) for base in E55_REFERENCES
    ]


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


def plot_violins(bags: pd.DataFrame, distribution: str, role: str, acc_name: str = ACC_NAME, level: str = "bags"):
    """The violin figure of ``distribution`` and the informative prior ``role``, over the bags or (``level="pairs"``)
    over the per-pair means of ``pair_means``; ``None`` without rows."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")
    rows = bags.loc[(bags["distribution"] == distribution) & (bags["acc_name"] == acc_name)]
    if rows.empty:
        return None
    if level == "pairs":
        rows = pair_means(rows)
    cells = [
        (base, r, part)
        for base, r in violin_cells(role)
        if not (part := rows.loc[(rows["method"] == base) & (rows["role"] == r)]).empty
    ]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(6.5 * len(PANELS), 5.2), layout="constrained")
    for ax, (score, label) in zip(axes, PANELS.items()):
        data = [log_values(part, score) for _, _, part in cells]
        positions = np.arange(len(cells))
        keep = [i for i, d in enumerate(data) if d.size > 0]
        parts = ax.violinplot(
            [data[i] for i in keep],
            positions=positions[keep],
            widths=0.8,
            showextrema=False,
            bw_method=0.15 if level == "bags" else 0.3,
        )
        for body, i in zip(parts["bodies"], keep):
            base, r, _ = cells[i]
            uniform = r == "uniform"
            body.set_facecolor(METHOD_COLORS.get(base, "#888888"))
            body.set_edgecolor("#1f1f1f")
            body.set_linewidth(0.8)
            body.set_alpha(0.45 if uniform else 0.9)
            if uniform:
                body.set_hatch("///")
        if level == "pairs":
            rng = np.random.default_rng(0)
            for i in keep:
                jitter = rng.uniform(-0.12, 0.12, data[i].size)
                ax.scatter(positions[i] + jitter, data[i], s=5, color="#1f1f1f", alpha=0.35, linewidth=0, zorder=2)
        medians = [np.median(data[i]) for i in keep]
        means = [np.log10(max(finite_mean(cells[i][2], score), FLOOR)) for i in keep]
        ax.scatter(positions[keep], medians, s=22, color="white", edgecolor="#1f1f1f", linewidth=0.8, zorder=3)
        ax.scatter(
            positions[keep], means, s=30, marker="D", color="#1f1f1f", edgecolor="white", linewidth=0.6, zorder=4
        )
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{method_plot_label(base)}\n{ROLE_LABELS[r]}" for base, r, _ in cells],
            rotation=60,
            ha="right",
            fontsize=font_size(8),
        )
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
        f"E5.5: {DISTRIBUTION_TITLES[distribution]}, {ROLE_LABELS[role]} prior, {unit} ({ACC_LABELS[acc_name]})",
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
    levels = {"violins": "bags", "pair-violins": "pairs"}
    if any(figure in args.figures for figure in levels):
        bags = load_table("bags", args.bags_csv)
    for figure, level in levels.items():
        if figure not in args.figures:
            continue
        prefix = f"{PLOT_NAME}_{figure.replace('-', '_')}"
        for acc_name in args.metrics:
            for distribution in args.distributions:
                for role in INFORMATIVE_ROLES:
                    fig = plot_violins(bags, distribution, role, acc_name, level)
                    if fig is None:
                        print(f"no bags for {distribution} ({acc_name}), skipped")
                        continue
                    _save(fig, os.path.join(out_dir, f"{prefix}_{distribution}_{role}_{acc_name}"), args)
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
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--methods", nargs="+", choices=E55_PRIOR_BASES, default=list(E55_PRIOR_BASES))
    parser.add_argument("--distributions", nargs="+", choices=E55_DISTRIBUTIONS, default=list(E55_DISTRIBUTIONS))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument(
        "--csv", default=None, help="Per-pair CSV of tables/binform_side_prior.py (default: its output)"
    )
    parser.add_argument(
        "--bags-csv", default=None, help="Per-bag CSV of tables/binform_side_prior.py (default: its output)"
    )
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
