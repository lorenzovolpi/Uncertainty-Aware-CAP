r"""Scatter plots of E5.4, concentrated and sparse prevalence distributions for BayCAP-Hf and BayCAP-KDEy (work-plan card
WP-24, ``docs/ext5_baycap_priors_tempering.md``).

The figures read the per-pair CSV of :mod:`uncawe.tables.binform_centre_edge` (run ``tables/binform_centre_edge.py
--tables`` first). One figure per method (``BayesCAP-Hf-real``, ``BayCAP-KDEy-mlp-real``), distribution of the bags
($G_c=\operatorname{Dir}(3\cdot\mathbf 1)$, ``centre``; $G_e=\operatorname{Dir}(\tfrac13\mathbf 1)$, ``edge``) and metric
(vanilla accuracy by default): the mean Winkler score with the uniform prior (horizontal) against that with the
well-informed prior (top row, the prior equal to the distribution) and with the ill-informed prior (bottom row, the
other distribution), one point per (dataset, classifier) pair, markers by classifier, coloured by $K$, one column per
bag size, with the diagonal $y=x$ (points below it: the informative prior has a lower Winkler score). Both methods target
the realised table of each bag and are scored against it.

The panels are drawn by ``plots/binform_prior.py:plot_winkler``, the scatter of E5.1. Output:
``<OUT_DIR>/uncawe/plots/binform_centre_edge/``.
"""

import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd

from uncawe import env
from uncawe.experiments.binform import E51_BASE_METHODS, E54_DISTRIBUTIONS
from uncawe.experiments.metrics import get_acc_names
from uncawe.plots.binform_prior import plot_winkler
from uncawe.plots.common import ACC_NAME, method_plot_label
from uncawe.tables.binform_centre_edge import INFORMATIVE_ROLES, ROLE_LABELS, csv_path
from uncawe.tables.binform_prior import ACC_LABELS

PLOT_NAME = "binform_centre_edge"
FIGURES = ("winkler",)
DISTRIBUTION_TITLES = {
    "centre": "bags from $\\operatorname{Dir}(3\\cdot\\mathbf{1})$",
    "edge": "bags from $\\operatorname{Dir}(\\frac{1}{3}\\mathbf{1})$",
}


def plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def load_table(path: str | None = None) -> pd.DataFrame:
    path = path or csv_path()
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run tables/binform_centre_edge.py --tables first.")
    return pd.read_csv(path)


def plot_distribution(table: pd.DataFrame, method: str, distribution: str, acc_name: str = ACC_NAME):
    """The scatter of E5.4 for the bags of ``distribution``; ``None`` without rows."""
    rows = table.loc[table["distribution"] == distribution]
    if rows.empty:
        return None
    title = f"E5.4, {method_plot_label(method)}: {DISTRIBUTION_TITLES[distribution]} ({ACC_LABELS[acc_name]})"
    return plot_winkler(rows, method, acc_name, priors=INFORMATIVE_ROLES, labels=ROLE_LABELS, title=title)


def generate_plots(args):
    table = load_table(path=args.csv)
    out_dir = plot_dir()
    if "winkler" not in args.figures:
        return
    for acc_name in args.metrics:
        for method in args.methods:
            for distribution in args.distributions:
                fig = plot_distribution(table, method, distribution, acc_name)
                if fig is None:
                    print(f"no rows for {method}, {distribution} ({acc_name}), skipped")
                    continue
                stem = os.path.join(out_dir, f"{PLOT_NAME}_{method}_{distribution}_{acc_name}")
                if args.add_stem is not None:
                    stem = f"{stem}_{args.add_stem}"
                for fmt in args.formats:
                    fig.savefig(f"{stem}.{fmt}", dpi=args.dpi if fmt == "png" else None, bbox_inches="tight")
                plt.close(fig)
                print(f"plot generated: {stem}")


def parse_args():
    parser = ArgumentParser(description="Plots of E5.4, concentrated and sparse prevalence distributions.")
    parser.add_argument("--figures", nargs="+", choices=FIGURES, default=list(FIGURES))
    parser.add_argument("--methods", nargs="+", choices=E51_BASE_METHODS, default=list(E51_BASE_METHODS))
    parser.add_argument("--distributions", nargs="+", choices=E54_DISTRIBUTIONS, default=list(E54_DISTRIBUTIONS))
    parser.add_argument("--metrics", nargs="+", choices=get_acc_names(), default=[ACC_NAME])
    parser.add_argument(
        "--csv", default=None, help="Per-pair CSV of tables/binform_centre_edge.py (default: its output)"
    )
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
