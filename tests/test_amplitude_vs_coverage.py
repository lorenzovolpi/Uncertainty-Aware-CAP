"""The options of the coverage-vs-amplitude plot (plots/amplitude_vs_coverage.py): KDE regions by mass, excluded pairs,
summary box and zoomed axes, on synthetic per-pair means."""

import os
import tempfile
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import amplitude_vs_coverage as avc

METHODS = ["DoC", "PQ"]
DATASETS = [f"d{i}" for i in range(8)]
CLASSIFIERS = ["LR", "kNN"]


def synthetic_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for method in METHODS:
        for dataset in DATASETS:
            for classifier in CLASSIFIERS:
                rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "method": method,
                        "mean_coverage": float(rng.uniform(0.5, 1.0)),
                        "mean_amplitude": float(rng.uniform(0.01, 0.3)),
                        "mean_winkler": float(rng.uniform(0.1, 1.0)),
                    }
                )
    return pd.DataFrame(rows)


class KdeLevelsTest(unittest.TestCase):
    def test_masses_become_iso_proportions_from_the_lowest(self):
        np.testing.assert_allclose(avc.kde_levels([0.5, 0.7, 0.99]), [0.01, 0.3, 0.5, 1.0])

    def test_no_masses_are_rejected(self):
        with self.assertRaises(ValueError):
            avc.kde_levels([])

    def test_masses_outside_the_unit_interval_are_rejected(self):
        with self.assertRaises(ValueError):
            avc.kde_levels([0.5, 1.0])


class ExcludePairsTest(unittest.TestCase):
    def test_drops_the_pair_for_every_method(self):
        df = synthetic_pairs()
        out = avc.exclude_pairs(df, ["d0:LR"])
        self.assertEqual(len(out), len(df) - len(METHODS))
        self.assertFalse(((out["dataset"] == "d0") & (out["classifier"] == "LR")).any())
        self.assertTrue(((out["dataset"] == "d0") & (out["classifier"] == "kNN")).any())

    def test_malformed_pair_is_rejected(self):
        with self.assertRaises(ValueError):
            avc.exclude_pairs(synthetic_pairs(), ["d0"])


class ZoomTest(unittest.TestCase):
    def test_zoom_is_monotone_and_puts_full_coverage_at_the_top(self):
        self.assertEqual(float(avc.zoom_x(0)), 0.0)
        self.assertTrue(np.all(np.diff(avc.zoom_x([0, 0.01, 0.05, 0.5])) > 0))
        self.assertEqual(float(avc.zoom_y(1)), 0.0)
        self.assertTrue(np.all(np.diff(avc.zoom_y([0, 0.5, 0.95, 1])) > 0))

    def test_nominal_values_fall_in_the_enlarged_part_of_the_axes(self):
        # amplitude 0.05 at about 40% of the x range, coverage 0.95 within the top third of the y range
        x_share = float(avc.zoom_x(0.05) / avc.zoom_x(avc.ZOOM_X_MAX))
        y_share = float(avc.zoom_y(0.95) / avc.zoom_y(0))
        self.assertTrue(0.3 < x_share < 0.5)
        self.assertTrue(y_share < 1 / 3)


class SummaryBoxTest(unittest.TestCase):
    def test_box_has_the_means_over_the_pairs(self):
        df = synthetic_pairs()
        sub = df[df["method"] == "PQ"]
        text = avc.summary_text(sub)
        self.assertEqual(
            text.splitlines(),
            [
                f"W = {sub['mean_winkler'].mean():.3f}",
                f"A = {sub['mean_amplitude'].mean():.3f}",
                f"C = {sub['mean_coverage'].mean():.3f}",
            ],
        )

    def test_plot_with_box_and_zoom_writes_the_figure(self):
        df = synthetic_pairs()
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(avc, "_plot_dir", return_value=tmp):
            captured = {}
            savefig = plt.Figure.savefig

            def keep(fig, *args, **kwargs):
                captured["fig"] = fig
                return savefig(fig, *args, **kwargs)

            with mock.patch.object(plt.Figure, "savefig", keep):
                avc._plot(
                    df,
                    "classic",
                    METHODS,
                    0.6,
                    "linear",
                    ["png"],
                    50,
                    "test",
                    kde_kws={"bw_adjust": 0.75, "levels": avc.kde_levels([0.5, 0.7, 0.99])},
                    summary_box=True,
                    zoom=True,
                )
            self.assertTrue(os.path.exists(os.path.join(tmp, "classic_vanilla_accuracy_test_zoom.png")))
            axes = [ax for ax in captured["fig"].axes if ax.get_title()]
            self.assertEqual(len(axes), len(METHODS))
            for ax, method in zip(axes, METHODS):
                texts = [t.get_text() for t in ax.texts]
                self.assertIn(avc.summary_text(df[df["method"] == method]), texts)
                np.testing.assert_allclose(ax.get_xticks(), avc.zoom_x(avc.ZOOM_X_TICKS))


class DefaultsTest(unittest.TestCase):
    def test_defaults_are_the_paper_figures(self):
        args = avc.parse_args([])
        self.assertEqual(args.kde_mass, [0.5, 0.7, 0.99])
        self.assertEqual(args.kde_bw_adjust, 0.75)
        self.assertTrue(args.summary_box)
        self.assertEqual(args.exclude, ["poker_hand:LR"])
        self.assertEqual(args.views, ["linear", "zoom"])

    def test_empty_exclude_keeps_every_pair(self):
        args = avc.parse_args(["--exclude", "--no-summary-box", "--views", "linear"])
        self.assertEqual(args.exclude, [])
        self.assertFalse(args.summary_box)
        df = synthetic_pairs()
        self.assertEqual(len(avc.exclude_pairs(df, args.exclude)), len(df))

    def test_zoomed_figure_has_its_own_stem(self):
        with mock.patch.object(avc, "_plot_dir", return_value="d"):
            self.assertEqual(avc._plot_stem("classic", "vanilla_accuracy", None), "d/classic_vanilla_accuracy")
            self.assertEqual(
                avc._plot_stem("classic", "vanilla_accuracy", None, zoom=True), "d/classic_vanilla_accuracy_zoom"
            )


if __name__ == "__main__":
    unittest.main()
