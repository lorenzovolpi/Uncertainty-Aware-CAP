r"""Plot of E8.3 of ``docs/ext8_average_confidence.md`` (card WP-32): the signed error of vanilla accuracy against the
shift of the bag, for ``AC`` (LasCal) and ``AC-LSB`` (label-shift binning).

Reads the stored ``main`` results of the two methods (:func:`uncawe.tables.ac_lsb.load_results`). One panel per method;
one point per bag at the shift $\lVert\bar\pi-\pi^V\rVert_1$ of the bag (x; $\bar\pi$ from the stored ``true_cts``,
$\pi^V$ from the rounded ``val_prev``, :func:`uncawe.tables.ac_lsb.bag_shifts`) and the signed error $\hat A-A$ of the
method (y), with $A$ the realised accuracy of the bag, the trace of its realised table; binary and multiclass pairs in
different colours. The hypothesis of §T.4 is that the error of ``AC`` grows with the shift and that label-shift binning
removes most of it.

Figures (``--figures``): ``signed_error_vs_shift``.

Output: ``<OUT_DIR>/uncawe/plots/ac_lsb/signed_error_vs_shift/``.
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
from uncawe.config import acc_from_ct
from uncawe.experiments.methods import AC_LSB_METHOD, AC_METHOD
from uncawe.plots.common import font_size, method_plot_label
from uncawe.tables.ac_lsb import ACC_NAME, KEYS, TABLE_NAME, bag_shifts, load_results

PLOT_NAME = "signed_error_vs_shift"
FIGURES = ("signed_error_vs_shift",)
PANEL_METHODS = (AC_METHOD, AC_LSB_METHOD)
KIND_COLORS = {"binary": "#4C78A8", "multiclass": "#F58518"}


def plot_dir(plot_name: str = PLOT_NAME) -> str:
    path = os.path.join(env.root_dir, "plots", TABLE_NAME, plot_name)
    os.makedirs(path, exist_ok=True)
    return path


def points(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per bag and method of ``PANEL_METHODS``: the pair, whether it is binary, the shift of the bag and the
    signed error of the method; bags without an estimate are dropped."""
    rows = []
    for method in PANEL_METHODS:
        df = results.get(method, pd.DataFrame())
        if df.empty:
            continue
        for key, part in df.groupby(KEYS, sort=False):
            shifts = bag_shifts(part)
            true_accs = np.array([acc_from_ct(ACC_NAME, np.asarray(ct, dtype=float)) for ct in part["true_cts"]])
            estimates = pd.to_numeric(part["estim_accs"], errors="coerce").to_numpy(dtype=float)
            binary = int(part["n_classes"].iloc[0]) == 2
            for shift, estimate, true_acc in zip(shifts, estimates, true_accs):
                if np.isfinite(estimate):
                    rows.append(
                        {
                            "method": method,
                            "dataset": key[0],
                            "classifier": key[1],
                            "binary": binary,
                            "shift": float(shift),
                            "signed_error": float(estimate - true_acc),
                        }
                    )
    return pd.DataFrame(rows, columns=["method", "dataset", "classifier", "binary", "shift", "signed_error"])


def plot_signed_error_vs_shift(results: dict[str, pd.DataFrame]):
    data = points(results)
    if data.empty:
        raise RuntimeError(f"No results of {PANEL_METHODS} found. Check that experiment results exist.")
    fig, axes = plt.subplots(1, len(PANEL_METHODS), figsize=(4.6 * len(PANEL_METHODS), 4.0), sharey=True, squeeze=False)
    for ax, method in zip(axes.flat, PANEL_METHODS):
        part = data[data["method"] == method]
        for kind, color in KIND_COLORS.items():
            sub = part[part["binary"] == (kind == "binary")]
            ax.scatter(sub["shift"], sub["signed_error"], color=color, s=5, alpha=0.35, linewidths=0, zorder=2)
        ax.axhline(0.0, color="#595959", linewidth=0.8, zorder=1)
        ax.set_title(method_plot_label(method), fontsize=font_size(10))
        ax.set_xlabel("shift of the bag $\\|\\bar\\pi-\\pi^V\\|_1$", fontsize=font_size(9))
        ax.tick_params(labelsize=font_size(8))
        ax.grid(color="#e0e0e0", linewidth=0.6)
        ax.set_axisbelow(True)
    axes.flat[0].set_ylabel("signed error $\\hat A-A$, vanilla accuracy", fontsize=font_size(9))
    handles = [
        Line2D([], [], marker="o", color=color, linestyle="", label=f"{kind} pair")
        for kind, color in KIND_COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=font_size(9), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


PLOTTERS = {"signed_error_vs_shift": plot_signed_error_vs_shift}


def generate_plots(args):
    results = load_results()
    for name in args.figures:
        fig = PLOTTERS[name](results)
        stem = os.path.join(plot_dir(name), name)
        if args.add_stem is not None:
            stem = f"{stem}_{args.add_stem}"
        for fmt in args.formats:
            fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
        plt.close(fig)
        print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="E8.3: signed error of AC and AC-LSB against the shift of the bag.")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
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
