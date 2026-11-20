r"""Plots of the accuracy and the intervals of six CAP methods across $|U|$ and $|V|$ in the study ``target_size``
(work-plan card WP-28, E4.4 of ``docs/ext4_uncertainty_budget.md``).

Reads ``summary`` of :mod:`uncawe.tables.target_size_scores` (LR, vanilla accuracy, every method scored against the
realised accuracy of each bag).

Figures (``--figures``), one per part, each with two panels: on the left the absolute error (AE) of the point
estimate, on the right the Winkler score $W$ at $1-\alpha=0.95$ (logarithmic axis); one line per method, the median
over the pairs of the per-pair means (over the bags, and over the $V$ seeds in E4.4b), with the band from the 25th to
the 75th percentile over the pairs.

- ``u-scores`` (E4.4a): against $|U|$ (logarithmic), full $V$.
- ``v-scores`` (E4.4b): against the fraction of $V$ (logarithmic), $|U|=1000$; a level where a pair has a class with
  fewer than ``MIN_CLASS_COUNT`` validation instances is marked by an open marker and a grey tick label with $^*$.

``--without poker_hand`` draws the medians over the pairs without ``poker_hand`` (the check of the Analysis of E4.4b).

Output: ``<OUT_DIR>/uncawe/plots/target_size/<figure>/`` (``u_scores``, ``v_scores``).
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from uncawe import env
from uncawe.experiments.target_size import E44_LEVELS, E44_METHODS, EXPERIMENT, MIN_CLASS_COUNT
from uncawe.plots.common import font_size
from uncawe.tables.target_size_scores import CHECK_DATASET, SHORT_LABELS, compute, load_results

FIGURES = {"u-scores": "a", "v-scores": "b"}
METHOD_COLORS = {
    "BayesCAP-Hf-real": "#1f77b4",
    "BayCAP-KDEy-mlp-real": "#d62728",
    "SQS-H": "#2ca02c",
    "SQS-Hbs": "#ff7f0e",
    "PQ": "#9467bd",
    "SQS-B": "#8c564b",
}
LEVEL_TICKS = {
    "full": "1",
    "1/sqrt2": "$1/\\sqrt{2}$",
    "1/2": "$1/2$",
    "1/(2sqrt2)": "$1/(2\\sqrt{2})$",
    "1/4": "$1/4$",
}
PANELS = (("ae", "AE", "linear"), ("winkler", "Winkler score $W$", "log"))


def subset_name(without: str | None) -> str:
    return "all" if without is None else f"without {without}"


def plot_scores(summary: pd.DataFrame, part: str, without: str | None = None):
    """One figure of a part of E4.4: AE on the left and $W$ on the right, against $|U|$ (part ``a``) or the fraction of
    $V$ (part ``b``), one line per method (median over the pairs) with the interquartile band."""
    rows = summary.loc[(summary["part"] == part) & (summary["subset"] == subset_name(without))]
    if rows.empty or rows["x"].nunique() < 2:
        raise RuntimeError(f"No results of E4.4{part} at two or more sizes.")
    methods = [m for m in E44_METHODS if m in set(rows["method"])]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.6 * len(PANELS), 3.6), squeeze=False)
    flagged = set(rows.loc[rows["n_flagged"] > 0, "x"])
    for ax, (name, label, scale) in zip(axes[0], PANELS):
        for method in methods:
            color = METHOD_COLORS.get(method, "black")
            line = rows.loc[rows["method"] == method].sort_values("x")
            ax.plot(line["x"], line[f"{name}_median"], color=color, linewidth=1.4, marker="o", markersize=3.5)
            ax.fill_between(line["x"], line[f"{name}_q25"], line[f"{name}_q75"], color=color, alpha=0.12, linewidth=0)
            marked = line.loc[line["x"].isin(flagged)]
            ax.plot(
                marked["x"],
                marked[f"{name}_median"],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor="white",
                color=color,
            )
        ax.set_xscale("log")
        ax.set_yscale(scale)
        ax.grid(which="both", color="#e5e5e5", linewidth=0.5)
        ax.tick_params(labelsize=font_size(7))
        ax.set_ylabel(label, fontsize=font_size(8))
        if part == "a":
            ax.set_xlabel("$|U|$", fontsize=font_size(8))
        else:
            ticks = {
                LEVEL_TICKS[level]: fraction for level, fraction in E44_LEVELS.items() if fraction in set(rows["x"])
            }
            labels = [text + ("$^*$" if fraction in flagged else "") for text, fraction in ticks.items()]
            ax.set_xticks(list(ticks.values()), labels)
            ax.xaxis.set_minor_locator(plt.NullLocator())
            for tick, fraction in zip(ax.get_xticklabels(), ticks.values()):
                if fraction in flagged:
                    tick.set_color("grey")
            ax.set_xlabel("fraction of $V$", fontsize=font_size(8))
    handles = [
        Line2D([], [], color=METHOD_COLORS.get(m, "black"), marker="o", markersize=3.5, label=SHORT_LABELS.get(m, m))
        for m in methods
    ]
    if flagged:
        handles.append(
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=6,
                markerfacecolor="white",
                color="black",
                label=f"$^*$ a pair with a class below {MIN_CLASS_COUNT} in $V$",
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 4), fontsize=font_size(8), frameon=False)
    n_pairs = int(rows["n_pairs"].max())
    where = "full $V$" if part == "a" else "$|U|=1000$"
    extra = "" if without is None else f", without {without}"
    fig.suptitle(
        f"E4.4{part}: LR, {where}, {n_pairs} pairs{extra}; median over the pairs, band 25th–75th percentile",
        fontsize=font_size(9),
    )
    fig.tight_layout(rect=(0, 0.12 if len(handles) > 4 else 0.08, 1, 1))
    return fig


def out_dir(figure: str) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, figure.replace("-", "_"))
    os.makedirs(path, exist_ok=True)
    return path


def generate_plots(args, summary: pd.DataFrame | None = None):
    if summary is None:
        summary = compute(load_results(datasets=args.datasets), split=False)["summary"]
    for name in args.figures:
        try:
            fig = plot_scores(summary, FIGURES[name], without=args.without)
        except RuntimeError as error:
            print(f"{name} not drawn: {error}")
            continue
        stem = os.path.join(out_dir(name), f"{EXPERIMENT}_{name}")
        if args.without is not None:
            stem = f"{stem}_without_{args.without}"
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Plots of the accuracy and intervals across |U| and |V| (E4.4, WP-28).")
    parser.add_argument("--figures", nargs="+", choices=tuple(FIGURES), default=list(FIGURES))
    parser.add_argument(
        "--without",
        nargs="?",
        const=CHECK_DATASET,
        default=None,
        choices=(CHECK_DATASET,),
        help=f"Medians over the pairs without {CHECK_DATASET} (the check of E4.4b)",
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict the datasets (default: every one found)")
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
