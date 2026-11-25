r"""Plots of the temperature chosen per pair on calibration bags for BayCAP-Hf and BayCAP-KDEy (work-plan card WP-23,
E5.3 of ``docs/ext5_baycap_priors_tempering.md``).

The figures read the pairs CSV of :mod:`uncawe.tables.tau_select` (run ``tables/tau_select.py --tables`` first). Both
methods target the realised table of each bag and are scored against it; the test scores are those of the methods
fitted on the full $V$ (``main`` and E5.2).

- ``sigma``: per method (one panel each) and metric (vanilla accuracy by default), $\tau_{\text{cal}}$ against the
  expected size of the error from $V$, $\hat\sigma_{e_2}$ (§T.9), both on the log scale, one point per pair coloured by
  $K$, markers by classifier; the points of a $\tau$ are spread vertically by classifier, so that they do not overlap.
- ``gain``: per method, the gain of $\tau_{\text{cal}}$ over $\tau=1$ on the calibration bags,
  $W^{\text{cal}}(1)-W^{\text{cal}}(\tau_{\text{cal}})$, against the gain on the test bags, $W(1)-W(\tau_{\text{cal}})$,
  one point per pair coloured by $K$, with the zero lines and the diagonal.

The figures belong to E5.3, so they live in one module (Part V.5 of ``docs/work_plan.md``). Output:
``<OUT_DIR>/uncawe/plots/tau_select/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from uncawe import env
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.common import (
    ACC_NAME,
    SEQUENTIAL_CMAP,
    classifier_legend,
    font_size,
    method_plot_label,
    ordered_classifiers,
    scatter_by_classifier,
    style_scatter_axes,
)
from uncawe.tables.tau_select import TAUS, csv_path
from uncawe.tables.tempering import TABLE_TEMPERED_BASES as TEMPERED_BASES
from uncawe.tables.tempering import tau_label

PLOT_NAME = "tau_select"
FIGURES = ("sigma", "gain")
REFERENCE_LINE = {"color": "#595959", "linewidth": 1.0, "linestyle": ":", "zorder": 1}
IDENTITY_LINE = {"color": "#595959", "linewidth": 1.2, "linestyle": "--", "zorder": 1}
# vertical spread of the points of one tau, in octaves, from the first to the last classifier
JITTER_SPAN = 0.4


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/tau_select.py --tables first.")
    return pd.read_csv(path)


def _k_norm(table: pd.DataFrame) -> LogNorm:
    return LogNorm(2, max(float(table["n_classes"].max()), 3.0))


def _rows(table: pd.DataFrame, acc_name: str, methods) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for method in methods:
        part = table.loc[(table["method"] == method) & (table["acc_name"] == acc_name)]
        if not part.empty:
            out.append((method, part))
    return out


def jittered_tau(part: pd.DataFrame) -> np.ndarray:
    """$\\tau_{\\text{cal}}$ multiplied by $2^{o}$, with an offset $o$ per classifier evenly spread over
    ``JITTER_SPAN`` octaves and centred on $0$."""
    classifiers = ordered_classifiers(part)
    n = len(classifiers)
    offsets = {c: (i - (n - 1) / 2) * (JITTER_SPAN / max(n - 1, 1)) for i, c in enumerate(classifiers)}
    return part["tau_cal"].to_numpy(dtype=float) * np.exp2(part["classifier"].astype(str).map(offsets).to_numpy())


def _panels(n: int):
    fig, axes = plt.subplots(1, n, figsize=(5.6 * n + 1.4, 4.8), layout="constrained", squeeze=False)
    return fig, axes[0]


def plot_sigma(table: pd.DataFrame, acc_name: str = ACC_NAME, methods=TEMPERED_BASES):
    """$\\tau_{\\text{cal}}$ against $\\hat\\sigma_{e_2}$, one panel per method."""
    rows = _rows(table, acc_name, methods)
    if not rows:
        return None
    fig, axes = _panels(len(rows))
    norm = _k_norm(table)
    sc = None
    for ax, (method, part) in zip(axes, rows):
        part = part.assign(tau_plot=jittered_tau(part))
        sc = scatter_by_classifier(ax, part, "sigma_e2", "tau_plot", "n_classes", SEQUENTIAL_CMAP, norm)
        ax.set_xscale("log")
        ax.set_yscale("log", base=2)
        ax.set_yticks(TAUS, [tau_label(tau) for tau in TAUS])
        ax.minorticks_off()
        ax.axhline(1.0, **REFERENCE_LINE)
        style_scatter_axes(ax, "$\\hat\\sigma_{e_2}$", "$\\tau_{\\mathrm{cal}}$")
        ax.set_title(f"{method_plot_label(method)} ({len(part)} pairs)", fontsize=font_size(11))
    fig.colorbar(sc, ax=list(axes)).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="outside right center")
    fig.suptitle(f"E5.3: $\\tau$ chosen on calibration bags against the expected error from $V$ ({acc_name})")
    return fig


def plot_gain(table: pd.DataFrame, acc_name: str = ACC_NAME, methods=TEMPERED_BASES):
    """The gain on the calibration bags against the gain on the test bags, one panel per method."""
    rows = _rows(table, acc_name, methods)
    if not rows:
        return None
    fig, axes = _panels(len(rows))
    norm = _k_norm(table)
    sc = None
    for ax, (method, part) in zip(axes, rows):
        sc = scatter_by_classifier(ax, part, "gain_calib", "gain_test", "n_classes", SEQUENTIAL_CMAP, norm)
        values = part[["gain_calib", "gain_test"]].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = min(finite.min(), 0.0), max(finite.max(), 0.0)
            pad = 0.05 * max(high - low, 1e-3)
            ax.plot((low - pad, high + pad), (low - pad, high + pad), **IDENTITY_LINE)
        ax.axhline(0.0, **REFERENCE_LINE)
        ax.axvline(0.0, **REFERENCE_LINE)
        style_scatter_axes(
            ax,
            "gain on the calibration bags, $W^{\\mathrm{cal}}(1)-W^{\\mathrm{cal}}(\\tau_{\\mathrm{cal}})$",
            "gain on the test bags, $W(1)-W(\\tau_{\\mathrm{cal}})$",
        )
        ax.set_title(f"{method_plot_label(method)} ({len(part)} pairs)", fontsize=font_size(11))
    fig.colorbar(sc, ax=list(axes)).set_label("$K$", fontsize=font_size(12))
    classifier_legend(fig, table, loc="outside right center")
    fig.suptitle(f"E5.3: gain of $\\tau_{{\\mathrm{{cal}}}}$ over $\\tau=1$ in Winkler score ({acc_name})")
    return fig


PLOTTERS = {"sigma": plot_sigma, "gain": plot_gain}


def _save(fig, name: str, args, out_dir: str | None = None) -> None:
    stem = os.path.join(out_dir or plot_dir(), f"{PLOT_NAME}_{name}")
    if args.add_stem is not None:
        stem = f"{stem}_{args.add_stem}"
    for fmt in args.formats:
        fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"plot generated: {stem}")


def generate_plots(args, out_dir: str | None = None):
    table = load_table(path=args.csv)
    for figure in args.figures:
        for acc_name in args.metrics:
            fig = PLOTTERS[figure](table, acc_name, args.methods)
            if fig is None:
                print(f"no rows for {acc_name}, {figure} skipped")
                continue
            _save(fig, f"{figure}_{acc_name}", args, out_dir)


def parse_args(argv: list[str] | None = None):
    parser = ArgumentParser(description="Plots of tau chosen per pair on calibration bags (E5.3).")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--methods", nargs="+", choices=TEMPERED_BASES, default=list(TEMPERED_BASES))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument("--csv", default=None, help="Pairs CSV of tables/tau_select.py (default: its output)")
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
