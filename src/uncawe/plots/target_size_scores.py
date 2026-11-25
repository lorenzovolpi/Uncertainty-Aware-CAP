r"""Plots of the accuracy and the intervals of the methods of E4.4 across $|U|$ and $|V|$ in the study ``target_size``
(work-plan card WP-28, E4.4 of ``docs/ext4_uncertainty_budget.md``).

Reads ``pairs`` of :mod:`uncawe.tables.target_size_scores` (LR, vanilla accuracy, every method scored against the
realised accuracy of each bag): per pair, method and size, the means over the bags (and over the $V$ seeds in E4.4b).

Figures (``--figures``), one per part, each with four panels (user's decision of 2026-11-25): in the first row the
absolute error (AE) of the point estimate and the Winkler score $W$ at $1-\alpha=0.95$, in the second row the amplitude
and the coverage of the interval; one line per method. The values and the bars follow the shift profile
(``plots/common.py:dataset_bars``): the classifiers of a dataset are averaged first (one, LR, in E4.4), then

- AE, $W$ and amplitude: the geometric mean over the datasets, with bars of $\pm1$ standard error over the datasets of
  the logarithms after removing each dataset's mean over the plotted methods and sizes (so the bars are slightly
  longer above the point than below it);
- coverage (which can be $0$): the arithmetic mean over the datasets on a linear axis, with bars of $\pm1$ standard
  error over the datasets of the values after removing each dataset's mean, and a dashed line at the nominal $0.95$.

Every vertical axis is linear (user's decision of 2026-11-25), with the labels AE, W, Amplitude and Coverage.

The bars depend on the set of plotted methods, the points do not. Before 2026-11-25: two panels (AE, $W$), arithmetic
means over the pairs with bars of $\pm1$ standard error over the pairs. The methods are shifted sideways around each
size by ``DODGE`` times the spacing (in decades) between consecutive sizes, so that their bars do not overlap.

- ``u-scores`` (E4.4a): against $|U|$ (logarithmic), full $V$.
- ``v-scores`` (E4.4b): against the fraction of $V$ (logarithmic), $|U|=1000$; a level where a pair has a class with
  fewer than ``MIN_CLASS_COUNT`` validation instances has a grey tick label with $^*$.

``--without poker_hand`` draws the means over the datasets without ``poker_hand`` (the check of the Analysis of E4.4b).
``--exclude-methods`` leaves methods out of the figures (the paper leaves out SQS-B).

Output: ``<OUT_DIR>/uncawe/plots/target_size/<figure>/`` (``u_scores``, ``v_scores``).
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
from uncawe.experiments.target_size import E44_LEVELS, E44_METHODS, EXPERIMENT
from uncawe.plots.common import dataset_bars, font_size
from uncawe.tables.target_size_scores import CHECK_DATASET, SHORT_LABELS, compute, load_results, part_rows

FIGURES = {"u-scores": "a", "v-scores": "b"}
METHOD_COLORS = {
    "BayesCAP-Hf-real": "#1f77b4",
    "BayCAP-KDEy-h-inst": "#d62728",
    "BayCAP-KDEy-h-inst-cb": "#e377c2",
    # the former BayCAP-KDEy (invalidated), no longer among E44_METHODS (card WP-33)
    "BayCAP-KDEy-mlp-real": "#7f7f7f",
    # WP-34
    "BayesCAP-Hf-real-tcal": "#17becf",
    "BayCAP-KDEy-h-inst-cb-tcal-keep": "#bcbd22",
    "BayCAP-KDEy-h-inst-cb-tcal-resel": "#dbdb8d",
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
# rows of panels: (score, label, geometric mean with bars from the logarithms); every vertical axis is linear
PANELS = (
    (("ae", "AE", True), ("winkler", "W", True)),
    (("amplitude", "Amplitude", True), ("coverage", "Coverage", False)),
)
NOMINAL_COVERAGE = 0.95
NOMINAL_LINE = {"color": "0.4", "linewidth": 0.8, "linestyle": "--", "zorder": 1}
# sideways shift between consecutive methods, as a fraction of the spacing (in decades) between consecutive sizes
DODGE = 0.05
U_TICKS = (100, 300, 1000, 3000, 10000)


def subset_name(without: str | None) -> str:
    return "all" if without is None else f"without {without}"


def dataset_rows(pairs: pd.DataFrame, part: str, without: str | None = None) -> pd.DataFrame:
    """The rows of a part of E4.4 (``x`` the size that varies), without the dataset ``without``, with the scores
    averaged over the classifiers of each dataset, and whether a pair of the dataset is flagged at that size."""
    rows = part_rows(pairs, part)
    if without is not None:
        rows = rows.loc[rows["dataset"] != without]
    scores = [score for row in PANELS for score, _, _ in row]
    return (
        rows.groupby(["method", "dataset", "x"], as_index=False)
        .agg(**{score: (score, "mean") for score in scores}, below_threshold=("below_threshold", "any"))
        .reset_index(drop=True)
    )


def plot_scores(pairs: pd.DataFrame, part: str, without: str | None = None, exclude=()):
    """One figure of a part of E4.4: AE and $W$ in the first row, amplitude and coverage in the second, against $|U|$
    (part ``a``) or the fraction of $V$ (part ``b``), one line per method with the means and bars of
    ``plots/common.py:dataset_bars``; the methods of ``exclude`` are not drawn."""
    rows = dataset_rows(pairs, part, without)
    if rows.empty or rows["x"].nunique() < 2:
        raise RuntimeError(f"No results of E4.4{part} at two or more sizes.")
    methods = [m for m in E44_METHODS if m in set(rows["method"]) and m not in exclude]
    sizes = np.sort(rows["x"].unique())
    step = float(np.median(np.diff(np.log10(sizes))))
    flagged = set(rows.loc[rows["below_threshold"], "x"])
    n_rows, n_cols = len(PANELS), len(PANELS[0])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.4 * n_rows), squeeze=False)
    n_datasets = 0
    for r, panel_row in enumerate(PANELS):
        for ax, (name, label, geometric) in zip(axes[r], panel_row):
            bars = dataset_bars(rows, name, "x", methods, log=geometric)
            n_datasets = int(bars["n_datasets"].iloc[0])
            for i, method in enumerate(methods):
                line = bars.loc[bars["method"] == method].sort_values("x")
                shift = 10 ** (DODGE * step * (i - (len(methods) - 1) / 2))
                y = line["value"].to_numpy()
                ax.errorbar(
                    line["x"] * shift,
                    y,
                    yerr=[y - line["lower"].to_numpy(), line["upper"].to_numpy() - y],
                    color=METHOD_COLORS.get(method, "black"),
                    linewidth=1.4,
                    marker="o",
                    markersize=3.5,
                    elinewidth=0.9,
                    capsize=2.5,
                    capthick=0.9,
                )
            ax.set_xscale("log")
            if name == "coverage":
                ax.axhline(NOMINAL_COVERAGE, **NOMINAL_LINE)
            ax.grid(which="major", color="#e5e5e5", linewidth=0.5)
            ax.tick_params(labelsize=font_size(7))
            ax.set_ylabel(label, fontsize=font_size(8))
            if part == "a":
                ticks = [size for size in U_TICKS if size in set(sizes)]
                ax.set_xticks(ticks, [str(size) for size in ticks])
                ax.xaxis.set_minor_locator(plt.NullLocator())
                ax.set_xlabel("$|U|$", fontsize=font_size(8))
            else:
                ticks = {
                    LEVEL_TICKS[level]: fraction for level, fraction in E44_LEVELS.items() if fraction in set(sizes)
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
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(8), frameon=False)
    where = "full $V$" if part == "a" else "$|U|=1000$"
    extra = "" if without is None else f", without {without}"
    fig.suptitle(
        f"E4.4{part}: LR, {where}, {n_datasets} datasets{extra}; geometric mean (coverage: mean) over the datasets, "
        "bars $\\pm1$ standard error after removing each dataset's level",
        fontsize=font_size(9),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def out_dir(figure: str) -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, figure.replace("-", "_"))
    os.makedirs(path, exist_ok=True)
    return path


def generate_plots(args, pairs: pd.DataFrame | None = None):
    if pairs is None:
        pairs = compute(load_results(datasets=args.datasets), split=False)["pairs"]
    for name in args.figures:
        try:
            fig = plot_scores(pairs, FIGURES[name], without=args.without, exclude=args.exclude_methods)
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
        help=f"Means over the pairs without {CHECK_DATASET} (the check of E4.4b)",
    )
    # the paper leaves SQS-B out (uncawe/paper.py, user's decision of 2026-11-25)
    parser.add_argument("--exclude-methods", nargs="*", choices=E44_METHODS, default=[], help="Methods not drawn")
    parser.add_argument("--datasets", nargs="+", default=None, help="Restrict the datasets (default: every one found)")
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
