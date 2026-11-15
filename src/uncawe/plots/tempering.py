r"""Plots of the tempering of BayCAP-Hf and BayCAP-KDEy on ``main`` (work-plan card WP-20, E5.2 of
``docs/ext5_baycap_priors_tempering.md``).

The figures read the per-pair CSV of :mod:`uncawe.tables.tempering` (run ``tables/tempering.py --tables`` first). Both
methods target the realised table of each bag and are scored against it.

- ``lines``: per method (``BayesCAP-Hf-real``, ``BayCAP-KDEy-mlp-real``) and metric (vanilla accuracy by default), the
  per-pair mean Winkler score, coverage and amplitude against $\tau$ (log scale), one panel per measure, each pair as a
  faint line coloured by $K$ and the mean over the pairs in bold; coverage with the nominal level $0.95$.
- ``lines_k``: the same with one row of panels per group of $K$ ($K=2$, $3\le K\le7$, $K\ge8$).
- ``widening``: for ``BayCAP-KDEy-mlp-real`` and vanilla accuracy, the observed widening of the amplitude with respect
  to $\tau=1$ (median over the bags of the per-bag ratio) against the predicted one, $\sqrt{(\tau+r^2)/(1+r^2)}$ with
  $r=\bar\sigma_g/\sigma_p$ (§T.6), one point per pair and $\tau\neq1$, coloured by $K$, markers by classifier, with the
  diagonal.

The figures belong to E5.2, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/tempering/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from uncawe import env
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.common import (
    ACC_NAME,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.tempering import (
    REFERENCE_TAU,
    TAUS,
    TEMPERED_BASES,
    WIDENING_METHOD,
    WIDENING_METRIC,
    csv_path,
    k_groups,
    tau_label,
)

PLOT_NAME = "tempering"
FIGURES = ("lines", "lines_k", "widening")
MEASURES = {"winkler": "Winkler score", "coverage": "coverage", "amplitude": "amplitude"}
NOMINAL = 0.95
K_TITLES = {"all": "all pairs", "K=2": "$K=2$", "3<=K<=7": "$3\\leq K\\leq7$", "K>=8": "$K\\geq8$"}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}
NOMINAL_LINE = {"color": "#595959", "linewidth": 1.0, "linestyle": ":", "zorder": 1}
PAIR_LINE = {"linewidth": 0.8, "alpha": 0.35, "zorder": 2}
MEAN_LINE = {"color": "black", "linewidth": 2.6, "marker": "o", "zorder": 3}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/tempering.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _draw_lines(ax, part: pd.DataFrame, measure: str, norm: LogNorm) -> None:
    """Each pair of ``part`` as a faint line against $\\tau$, coloured by $K$, and the mean over the pairs in bold."""
    taus = np.array(TAUS)
    values = part[[f"t{tau_label(tau)}_{measure}" for tau in TAUS]].to_numpy(dtype=float)
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    for row, k in zip(values, part["n_classes"]):
        ax.plot(taus, row, color=cmap(norm(k)), **PAIR_LINE)
    ax.plot(taus, np.nanmean(values, axis=0), **MEAN_LINE)
    if measure == "coverage":
        ax.axhline(NOMINAL, **NOMINAL_LINE)
    ax.axvline(REFERENCE_TAU, **NOMINAL_LINE)
    ax.set_xscale("log", base=2)
    ax.set_xticks(taus, [tau_label(tau) for tau in TAUS])
    ax.minorticks_off()
    ax.set_xlabel("$\\tau$", fontsize=font_size(11))
    ax.set_ylabel(MEASURES[measure], fontsize=font_size(11))
    ax.grid(alpha=0.3)


def plot_lines(table: pd.DataFrame, method: str, acc_name: str = ACC_NAME, by_k: bool = False):
    """The measures against $\\tau$, one column per measure, one row (``by_k``: one row per group of $K$)."""
    rows = table.loc[(table["method"] == method) & (table["acc_name"] == acc_name)]
    if rows.empty:
        return None
    groups = k_groups(rows) if by_k else [("all", rows)]
    fig, axes = plt.subplots(
        len(groups),
        len(MEASURES),
        figsize=(4.6 * len(MEASURES) + 1.2, 3.6 * len(groups)),
        layout="constrained",
        squeeze=False,
    )
    norm = _k_norm(table)
    for r, (k_group, part) in enumerate(groups):
        for c, measure in enumerate(MEASURES):
            ax = axes[r, c]
            _draw_lines(ax, part, measure, norm)
            ax.set_title(f"{K_TITLES[k_group]} ({len(part)} pairs)", fontsize=font_size(11))
    fig.colorbar(ScalarMappable(norm=norm, cmap=SEQUENTIAL_CMAP), ax=axes).set_label("$K$", fontsize=font_size(12))
    fig.suptitle(f"E5.2, {method_plot_label(method)}: tempering ({acc_name})", fontsize=font_size(12))
    return fig


def widening_points(table: pd.DataFrame) -> pd.DataFrame:
    """One row per pair and $\\tau\\neq1$ of ``WIDENING_METHOD`` and ``WIDENING_METRIC``: observed and predicted
    widening."""
    rows = table.loc[(table["method"] == WIDENING_METHOD) & (table["acc_name"] == WIDENING_METRIC)]
    parts = []
    for tau in TAUS:
        if tau == REFERENCE_TAU:
            continue
        t = tau_label(tau)
        part = rows[["dataset", "classifier", "n_classes"]].copy()
        part["tau"] = tau
        part["observed"] = rows[f"t{t}_widening"].to_numpy(dtype=float)
        part["predicted"] = rows[f"t{t}_widening_predicted"].to_numpy(dtype=float)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def plot_widening(table: pd.DataFrame):
    """Observed against predicted widening of the amplitude of BayCAP-KDEy (§T.6)."""
    points = widening_points(table)
    if points.empty or not np.isfinite(points["predicted"]).any():
        return None
    fig, ax = plt.subplots(figsize=(6.2, 5.0), layout="constrained")
    norm = _k_norm(table)
    sc = scatter_by_classifier(ax, points, "predicted", "observed", "n_classes", SEQUENTIAL_CMAP, norm)
    values = points[["predicted", "observed"]].to_numpy(dtype=float)
    low, high = np.nanmin(values), np.nanmax(values)
    pad = 0.05 * max(high - low, 1e-3)
    low, high = max(low - pad, 0.0), high + pad
    ax.plot((low, high), (low, high), **IDENTITY_LINE)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    style_scatter_axes(ax, "predicted widening $\\sqrt{(\\tau+r^2)/(1+r^2)}$", "observed widening")
    fig.colorbar(sc, ax=ax).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, points, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle(
        f"E5.2, {method_plot_label(WIDENING_METHOD)}: amplitude at $\\tau$ / amplitude at $\\tau=1$",
        fontsize=font_size(12),
    )
    return fig


def _save(fig, name: str, args) -> None:
    stem = os.path.join(plot_dir(), f"{PLOT_NAME}_{name}")
    if args.add_stem is not None:
        stem = f"{stem}_{args.add_stem}"
    for fmt in args.formats:
        fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"plot generated: {stem}")


def generate_plots(args):
    table = load_table(path=args.csv)
    for figure in args.figures:
        if figure == "widening":
            fig = plot_widening(table)
            if fig is None:
                print("no predicted widening in the table, skipped")
                continue
            _save(fig, "widening", args)
            continue
        for acc_name in args.metrics:
            for method in args.methods:
                fig = plot_lines(table, method, acc_name, by_k=figure == "lines_k")
                if fig is None:
                    print(f"no rows for {method} ({acc_name}), skipped")
                    continue
                _save(fig, f"{figure}_{method}_{acc_name}", args)


def parse_args():
    parser = ArgumentParser(description="Plots of the tempering of BayCAP-Hf and BayCAP-KDEy on main (E5.2).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--methods", nargs="+", choices=TEMPERED_BASES, default=list(TEMPERED_BASES))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument("--csv", default=None, help="Per-pair CSV of tables/tempering.py (default: its output)")
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args()
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


def main():
    generate_plots(parse_args())


if __name__ == "__main__":
    main()
