import os
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cap.models.confidence import PrediQuant
from matplotlib.patches import Polygon

import uncawe.env as env
from uncawe.experiments.predist import _interval_metrics
from uncawe.experiments.predist_config import EXPERIMENT, METHOD_NAME


def _safe_name(name: str) -> str:
    return str(name).replace(os.sep, "__").replace(" ", "_")


def _plot_stem(dataset: str, classifier: str, acc_name: str, sample_id: int) -> str:
    parent = os.path.join(env.root_dir, "plots", EXPERIMENT, dataset)
    os.makedirs(parent, exist_ok=True)
    filename = f"{_safe_name(acc_name)}_{_safe_name(classifier)}_{METHOD_NAME}_{sample_id:02d}"
    return os.path.join(parent, filename)


def _ternary_xy(prevs: np.ndarray) -> np.ndarray:
    prevs = np.asarray(prevs, dtype=float)
    x = prevs[..., 1] + 0.5 * prevs[..., 2]
    y = (np.sqrt(3) / 2) * prevs[..., 2]
    return np.stack([x, y], axis=-1)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(origin, left, right):
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(tuple(point))
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(tuple(point))
    return np.asarray(lower[:-1] + upper[:-1])


def _draw_simplex_panel(
    ax,
    distance: str,
    true_test_prev: np.ndarray,
    predicted_test_prev: np.ndarray,
    predicted_val_prevs: np.ndarray,
    true_val_prevs: np.ndarray,
    dist_max: float,
    acc_abs_error: float,
    ci_coverage: float,
    ci_amplitude: float,
    ci_winkler: float,
) -> None:
    triangle = _ternary_xy(np.eye(3))
    true_test_xy = _ternary_xy(true_test_prev.reshape(1, 3))[0]
    predicted_test_xy = _ternary_xy(predicted_test_prev.reshape(1, 3))[0]
    predicted_val_xy = _ternary_xy(predicted_val_prevs)
    true_val_xy = _ternary_xy(true_val_prevs)
    ax.add_patch(Polygon(triangle, closed=True, facecolor="white", edgecolor="black", linewidth=1.2))

    for points_xy, face_color, edge_color, label in (
        (true_val_xy, "#54A24B", "#347433", "true val"),
        (predicted_val_xy, "#4C78A8", "#2F5F93", "pred val"),
    ):
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=13, color=edge_color, alpha=0.50, label=label)
        hull = _convex_hull(points_xy)
        if len(hull) >= 3:
            ax.add_patch(
                Polygon(hull, closed=True, facecolor=face_color, edgecolor=edge_color, alpha=0.16, linewidth=1.0)
            )
        elif len(hull) == 2:
            ax.plot(hull[:, 0], hull[:, 1], color=edge_color, linewidth=1.0, alpha=0.8)

    ax.scatter(
        [true_test_xy[0]],
        [true_test_xy[1]],
        s=82,
        color="#E45756",
        edgecolor="black",
        linewidth=0.6,
        label="true test",
        zorder=5,
    )
    ax.scatter(
        [predicted_test_xy[0]],
        [predicted_test_xy[1]],
        s=82,
        color="#F58518",
        edgecolor="black",
        linewidth=0.6,
        marker="D",
        label="pred test",
        zorder=6,
    )
    labels = [r"$p_0$", r"$p_1$", r"$p_2$"]
    offsets = [(-0.04, -0.04), (0.025, -0.04), (0.0, 0.035)]
    for xy, label, offset in zip(triangle, labels, offsets):
        ax.text(xy[0] + offset[0], xy[1] + offset[1], label, fontsize=11)

    ax.set_title(f"{distance} | max d={dist_max:.3f}", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
    ax.text(
        0.5,
        -0.11,
        f"AE={acc_abs_error:.3f}  cov={ci_coverage:.0f}  amp={ci_amplitude:.3f}  W={ci_winkler:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.axis("off")


def _prev_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.stack([np.asarray(item, dtype=float) for item in array])
    else:
        array = array.astype(float)
    return array.reshape(1, -1) if array.ndim == 1 else array


def _ensure_plot_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "acc_abs_error" not in df.columns:
        df["acc_abs_error"] = np.abs(df["true_acc"] - df["pred_acc"])
    if {"ci_coverage", "ci_amplitude", "ci_winkler"}.issubset(df.columns):
        return df
    metrics = [
        _interval_metrics(row.true_acc, row.interval_low, row.interval_high)
        for row in df.itertuples(index=False)
    ]
    df["ci_coverage"] = [coverage for coverage, _, _ in metrics]
    df["ci_amplitude"] = [amplitude for _, amplitude, _ in metrics]
    df["ci_winkler"] = [winkler for _, _, winkler in metrics]
    return df


def plot_grouped_sample(
    df: pd.DataFrame,
    dataset: str,
    classifier: str,
    acc_name: str,
    sample_id: int,
) -> None:
    plot_df = _ensure_plot_metrics(df)
    rows = plot_df.loc[plot_df["sample_id"] == sample_id, :].copy()
    if rows.empty:
        return
    rows["distance"] = pd.Categorical(rows["distance"], categories=list(PrediQuant.DISTANCES), ordered=True)
    rows = rows.sort_values("distance")
    fig, axes = plt.subplots(1, len(rows), figsize=(5.1 * len(rows), 4.8), squeeze=False)
    axes = axes[0]
    for ax, (_, row) in zip(axes, rows.iterrows()):
        _draw_simplex_panel(
            ax,
            row["distance"],
            _prev_array(row["actual_test_prev"])[0],
            _prev_array(row["predicted_test_prev"])[0],
            _prev_array(row["selected_val_prevs"]),
            _prev_array(row["selected_val_true_prevs"]),
            row["dist_max"],
            row["acc_abs_error"],
            row["ci_coverage"],
            row["ci_amplitude"],
            row["ci_winkler"],
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(f"{dataset} | {acc_name} | {classifier} | sample {sample_id:02d}", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20, top=0.86)
    stem = _plot_stem(dataset, classifier, acc_name, sample_id)
    fig.savefig(f"{stem}.png", dpi=180)
    fig.savefig(f"{stem}.svg")
    plt.close(fig)


# Compatibility with the former private helper name.
_plot_grouped_sample = plot_grouped_sample
