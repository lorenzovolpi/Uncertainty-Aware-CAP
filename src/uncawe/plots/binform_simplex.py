import os
import tempfile
from argparse import ArgumentParser, BooleanOptionalAction
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import quapy as qp
from cap.data.datasets import fetch_UCIMulticlassDataset
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

import uncawe.env as env
from uncawe.binform import (
    BAYESCAP_PRIOR_C,
    DATASET,
    EXPERIMENT,
    TEST_PRIOR_C,
    _transform_prior,
)
from uncawe.config import SAMPLE_SIZE

PLOT_NAME = "simplex_prior"
SEED = 0
TEST_DRAWS = 100
BAYESCAP_DRAWS = 1000

COLORS = {
    "validation": "#4C78A8",
    "c1": "#F58518",
    "test_samples": "#54A24B",
    "bayescap_samples": "#B279A2",
}

qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
qp.environ["_R_SEED"] = SEED


def _plot_dir() -> str:
    path = os.path.join(env.root_dir, "plots", EXPERIMENT, PLOT_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _plot_stem(add_stem: str | None = None) -> str:
    stem = os.path.join(_plot_dir(), DATASET)
    if add_stem:
        stem = f"{stem}_{add_stem}"
    return stem


def _ternary_xy(prevs: np.ndarray) -> np.ndarray:
    prevs = np.asarray(prevs, dtype=float)
    x = prevs[..., 1] + 0.5 * prevs[..., 2]
    y = (np.sqrt(3) / 2) * prevs[..., 2]
    return np.stack([x, y], axis=-1)


def _normalize_prev(prev: np.ndarray) -> np.ndarray:
    prev = np.asarray(prev, dtype=float)
    total = prev.sum()
    if not np.isfinite(total) or np.isclose(total, 0):
        raise ValueError(f"Cannot normalize prevalence vector with sum={total}: {prev}")
    return prev / total


def _validation_prior() -> np.ndarray:
    _, V, _ = fetch_UCIMulticlassDataset(DATASET)
    return V.prevalence()


def _dirichlet_samples(alpha: np.ndarray, size: int) -> np.ndarray:
    rng = np.random.RandomState(SEED)
    return rng.dirichlet(alpha, size=size)


def _draw_simplex(ax):
    triangle = _ternary_xy(np.eye(3))
    patch = Polygon(
        triangle,
        closed=True,
        facecolor="white",
        edgecolor="#222222",
        linewidth=1.2,
        zorder=1,
    )
    ax.add_patch(patch)

    labels = [r"$p_0$", r"$p_1$", r"$p_2$"]
    offsets = [(-0.045, -0.045), (0.025, -0.045), (0.0, 0.035)]
    for xy, label, offset in zip(triangle, labels, offsets):
        ax.text(xy[0] + offset[0], xy[1] + offset[1], label, fontsize=12)

    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.axis("off")
    return patch


def _distances_from_center(prevs: np.ndarray, center_prev: np.ndarray) -> np.ndarray:
    xy = _ternary_xy(prevs)
    center_xy = _ternary_xy(center_prev.reshape(1, 3))[0]
    return np.linalg.norm(xy - center_xy, axis=1)


def _draw_samples_and_rings(ax, prevs: np.ndarray, center_prev: np.ndarray, color: str, clip_path: Polygon):
    xy = _ternary_xy(prevs)
    center_xy = _ternary_xy(center_prev.reshape(1, 3))[0]
    distances = _distances_from_center(prevs, center_prev)

    ax.scatter(xy[:, 0], xy[:, 1], s=9, color=color, alpha=0.24, edgecolor="none", zorder=2)

    for quantile, linestyle, linewidth, alpha in (
        (0.50, "--", 1.4, 0.90),
        (0.90, "-", 2.0, 0.95),
    ):
        radius = float(np.quantile(distances, quantile))
        circle = Circle(
            center_xy,
            radius,
            fill=False,
            edgecolor=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=4,
        )
        circle.set_clip_path(clip_path)
        ax.add_patch(circle)


def _draw_point(ax, prev: np.ndarray, color: str, label: str, marker: str):
    xy = _ternary_xy(np.asarray(prev, dtype=float).reshape(1, 3))[0]
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=95,
        color=color,
        marker=marker,
        edgecolor="#222222",
        linewidth=0.7,
        label=label,
        zorder=6,
    )


def _legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["validation"],
            markeredgecolor="#222222",
            markersize=9,
            label="Validation prior",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor=COLORS["c1"],
            markeredgecolor="#222222",
            markersize=8,
            label=r"Transformed prior ($c=1$)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["test_samples"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=rf"Test Dirichlet samples, $c={TEST_PRIOR_C:g}$, $n={TEST_DRAWS}$",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["bayescap_samples"],
            marker="o",
            linewidth=2,
            markersize=5,
            label=rf"BayesCAP Dirichlet samples, $c={BAYESCAP_PRIOR_C:g}$, $n={BAYESCAP_DRAWS}$",
        ),
        Line2D([0], [0], color="#555555", linestyle="--", linewidth=1.4, label="50% distance ring"),
        Line2D([0], [0], color="#555555", linestyle="-", linewidth=2.0, label="90% distance ring"),
    ]


def _plot(formats: Iterable[str], dpi: int, add_stem: str | None):
    val_prior = _validation_prior()
    transformed_c1 = _transform_prior(val_prior, 1)
    ring_center = _normalize_prev(transformed_c1)
    test_alpha = _transform_prior(val_prior, TEST_PRIOR_C)
    bayescap_alpha = _transform_prior(val_prior, BAYESCAP_PRIOR_C)
    test_samples = _dirichlet_samples(test_alpha, TEST_DRAWS)
    bayescap_samples = _dirichlet_samples(bayescap_alpha, BAYESCAP_DRAWS)

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    simplex_clip = _draw_simplex(ax)
    _draw_samples_and_rings(ax, test_samples, ring_center, COLORS["test_samples"], simplex_clip)
    _draw_samples_and_rings(ax, bayescap_samples, ring_center, COLORS["bayescap_samples"], simplex_clip)
    _draw_point(ax, val_prior, COLORS["validation"], "validation prior", marker="o")
    _draw_point(ax, transformed_c1, COLORS["c1"], "transformed prior", marker="D")

    ax.legend(
        handles=_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    stem = _plot_stem(add_stem)
    for fmt in formats:
        fig.savefig(f"{stem}.{fmt}", dpi=dpi if fmt.lower() == "png" else None)
    plt.close(fig)


def generate_plot(args):
    _plot(args.formats, args.dpi, args.add_stem)
    print(f"plot generated: {PLOT_NAME}/{DATASET}")


def parse_args():
    parser = ArgumentParser(description="Generate the binform prior simplex plot for connect-4.")
    parser.add_argument("--svg", action=BooleanOptionalAction, default=True)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--add-stem", default=None)
    args = parser.parse_args()
    args.formats = ["png", "svg"] if args.svg else ["png"]
    return args


if __name__ == "__main__":
    generate_plot(parse_args())
