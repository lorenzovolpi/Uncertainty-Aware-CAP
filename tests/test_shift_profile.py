import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots.common import method_table_label
from uncawe.plots.shift_profile import (
    DEFAULT_METHODS,
    add_bag_index,
    dataset_profile,
    geometric_summary,
    k_groups,
    parse_args,
    plot_profile,
    shift_levels,
)


def _scores(shifts: dict, methods=("A", "B"), scale=None) -> pd.DataFrame:
    """Synthetic per-bag scores: ``shifts`` maps (dataset, classifier, K) to the shifts of its bags, in stored order;
    the AE and Winkler score of a bag are ``scale[method] * (1 + shift)``."""
    scale = scale or {method: i + 1.0 for i, method in enumerate(methods)}
    rows = []
    for (dataset, classifier, k), bag_shifts in shifts.items():
        for method in methods:
            for s in bag_shifts:
                value = scale[method] * (1.0 + s)
                rows.append(
                    {
                        "dataset": dataset,
                        "classifier": classifier,
                        "method": method,
                        "n_classes": k,
                        "shifts": s,
                        "acc_err": value,
                        "winkler": value,
                    }
                )
    return pd.DataFrame(rows)


class TestShiftLevels(unittest.TestCase):
    def test_levels_are_within_pair_ranks_with_ties_by_bag_index(self):
        df = add_bag_index(_scores({("d", "LR", 2): [0.5, 0.1, 0.3, 0.1, 0.9, 0.7]}))
        out = shift_levels(df, n_bins=3)
        levels = out.loc[out["method"] == "A"].sort_values("bag")["level"].tolist()
        # sorted: bag1 (0.1), bag3 (0.1), bag2 (0.3), bag0 (0.5), bag5 (0.7), bag4 (0.9)
        self.assertEqual(levels, [1, 0, 1, 0, 2, 2])

    def test_levels_are_equal_sized_and_shared_by_methods(self):
        rng = np.random.default_rng(0)
        df = add_bag_index(_scores({("d", "LR", 3): rng.random(100).round(2).tolist()}))
        out = shift_levels(df, n_bins=10)
        counts = out.loc[out["method"] == "A"].groupby("level").size()
        self.assertEqual(counts.tolist(), [10] * 10)
        by_method = out.pivot_table(index="bag", columns="method", values="level")
        self.assertTrue((by_method["A"] == by_method["B"]).all())

    def test_mismatched_bags_raise(self):
        df = _scores({("d", "LR", 2): [0.1, 0.2]})
        df.loc[(df["method"] == "B") & (df["shifts"] == 0.2), "shifts"] = 0.3
        with self.assertRaises(ValueError):
            add_bag_index(df)
        with self.assertRaises(ValueError):
            add_bag_index(_scores({("d", "LR", 2): [0.1, 0.2]}).iloc[:-1])


class TestProfile(unittest.TestCase):
    def test_dataset_profile_averages_the_classifiers_of_a_dataset(self):
        easy = _scores({("d", "LR", 2): [0.0, 1.0]})
        hard = _scores({("d", "kNN", 2): [0.0, 1.0]}, scale={"A": 10.0, "B": 20.0})
        df = shift_levels(add_bag_index(pd.concat([easy, hard], ignore_index=True)), n_bins=2)
        profile = dataset_profile(df).set_index(["method", "level"])
        # A: (1 + 10) / 2 at level 0, (2 + 20) / 2 at level 1
        self.assertAlmostEqual(profile.loc[("A", 0), "acc_err"], 5.5)
        self.assertAlmostEqual(profile.loc[("A", 1), "winkler"], 11.0)

    def test_geometric_mean_and_bars_without_the_dataset_level(self):
        # dataset e: A=1, B=2 at both levels; dataset h: the same times 100, so the dataset levels differ a lot
        rows = [
            {"method": m, "dataset": d, "n_classes": 2, "level": lvl, "acc_err": v * f, "winkler": v * f}
            for d, f in (("e", 1.0), ("h", 100.0))
            for m, v in (("A", 1.0), ("B", 2.0))
            for lvl in (0, 1)
        ]
        summary = geometric_summary(pd.DataFrame(rows), "acc_err", ["A", "B"]).set_index(["method", "level"])
        # geometric mean of 1 and 100 is 10; B is twice A
        self.assertAlmostEqual(summary.loc[("A", 0), "value"], 10.0)
        self.assertAlmostEqual(summary.loc[("B", 1), "value"], 20.0)
        # the same pattern on both datasets once their levels are removed: zero-width bars
        np.testing.assert_allclose(summary["lower"], summary["value"])
        np.testing.assert_allclose(summary["upper"], summary["value"])
        self.assertEqual(summary["n_datasets"].unique().tolist(), [2])

    def test_bars_are_the_standard_error_of_the_centred_logs(self):
        rows = [
            {"method": m, "dataset": d, "n_classes": 2, "level": 0, "acc_err": v, "winkler": v}
            for d, values in (("e", (1.0, 4.0)), ("h", (1.0, 1.0)))
            for m, v in zip(("A", "B"), values)
        ]
        summary = geometric_summary(pd.DataFrame(rows), "acc_err", ["A", "B"]).set_index("method")
        # centred logs of A: -log 2 (e) and 0 (h); sd = log 2 / sqrt 2, SE = sd / sqrt 2 = log 2 / 2
        self.assertAlmostEqual(summary.loc["A", "value"], np.sqrt(1.0))
        self.assertAlmostEqual(summary.loc["A", "upper"], np.exp(np.log(2) / 2))
        self.assertAlmostEqual(summary.loc["A", "lower"], np.exp(-np.log(2) / 2))

    def test_missing_or_non_positive_values_raise(self):
        rows = pd.DataFrame(
            [
                {"method": "A", "dataset": "e", "n_classes": 2, "level": 0, "acc_err": 1.0, "winkler": 1.0},
                {"method": "B", "dataset": "e", "n_classes": 2, "level": 0, "acc_err": 0.0, "winkler": 1.0},
                {"method": "A", "dataset": "h", "n_classes": 2, "level": 0, "acc_err": 1.0, "winkler": 1.0},
            ]
        )
        with self.assertRaises(ValueError):
            geometric_summary(rows, "winkler", ["A", "B"])
        with self.assertRaises(ValueError):
            geometric_summary(rows.iloc[:2], "acc_err", ["A", "B"])

    def test_plot_has_log_axes_dodged_bars_and_csv_values(self):
        rng = np.random.default_rng(0)
        shifts = {(f"d{i}", "LR", 2): rng.random(20).round(2).tolist() for i in range(3)}
        df = shift_levels(add_bag_index(_scores(shifts)), n_bins=4)
        fig, values = plot_profile([("all $K$", dataset_profile(df))], ["A", "B"], 4)
        left, right = fig.axes[:2]
        self.assertEqual((left.get_yscale(), right.get_yscale()), ("log", "log"))
        self.assertEqual([left.get_title(), right.get_title()], ["AE", "W"])
        xs = [np.asarray(container.lines[0].get_xdata(), dtype=float) for container in left.containers]
        np.testing.assert_allclose((xs[0] + xs[1]) / 2, np.arange(1, 5))
        self.assertTrue((xs[0] < xs[1]).all())
        self.assertEqual(len(values), 2 * 2 * 4)
        self.assertTrue((values["lower"] <= values["value"]).all() and (values["value"] <= values["upper"]).all())
        plt.close(fig)

    def test_pooled_is_the_default_figure(self):
        self.assertEqual(parse_args([]).figures, ["pooled"])

    def test_k_groups_label_the_range_present(self):
        df = _scores(
            {
                ("b", "LR", 2): [0.1, 0.2],
                ("m3", "LR", 3): [0.1, 0.2],
                ("m7", "LR", 7): [0.1, 0.2],
                ("l", "LR", 26): [0.1, 0.2],
            }
        )
        profile = dataset_profile(shift_levels(add_bag_index(df), n_bins=2))
        groups = k_groups(profile)
        self.assertEqual([label for label, _ in groups], ["$K=2$", "$3\\leq K\\leq 7$", "$K\\geq 10$"])
        self.assertEqual([part["dataset"].nunique() for _, part in groups], [1, 2, 1])


class TestExcludeMethods(unittest.TestCase):
    def test_excluded_methods_leave_the_plotted_set(self):
        self.assertEqual(parse_args([]).methods, list(DEFAULT_METHODS))
        methods = parse_args(["--exclude-methods", "SQS-B"]).methods
        self.assertEqual(methods, [m for m in DEFAULT_METHODS if m != "SQS-B"])


class TestMethodTableLabel(unittest.TestCase):
    def test_macros_are_resolved(self):
        self.assertEqual(method_table_label("O-LEAP"), "LEAP")
        self.assertEqual(method_table_label("SQS-Hbs-KDEy"), "SQS-H$_{\\mathrm{bs}}$")
        self.assertEqual(method_table_label("SQS-B"), "SQS-H$_{\\mathrm{bay}}$")
        self.assertEqual(method_table_label("BayesCAP-Hf-real"), "BayCAP-Hf")
        self.assertEqual(method_table_label("BayCAP-KDEy-h-inst"), "BayCAP-KDEy")
        self.assertEqual(method_table_label("BayCAP-KDEy-mlp-real"), "BayCAP-KDEy-mlp-real")
        self.assertEqual(method_table_label("BayesCAP-real"), "BayCAP-sample")
        self.assertEqual(method_table_label("PQ-KDEy"), "SQS-S")
        self.assertEqual(method_table_label("DoC"), "DoC")


if __name__ == "__main__":
    unittest.main()
