import unittest

import numpy as np
import pandas as pd

from uncawe.plots.common import method_table_label
from uncawe.plots.shift_profile import add_bag_index, k_groups, relative_profile, shift_levels


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


class TestRelativeProfile(unittest.TestCase):
    def test_common_reference_cancels_pair_difficulty_and_keeps_gaps(self):
        # two pairs of the same dataset, the second 10 times harder; B is twice A in every bag
        easy = _scores({("d", "LR", 2): [0.0, 1.0]})
        hard = _scores({("d", "kNN", 2): [0.0, 1.0]}, scale={"A": 10.0, "B": 20.0})
        df = shift_levels(add_bag_index(pd.concat([easy, hard], ignore_index=True)), n_bins=2)
        profile = relative_profile(df).set_index(["method", "level"])
        # reference of each pair: mean of A and B over the levels = 1.5 * 1.5 = 2.25 (times 10 for the hard pair)
        self.assertAlmostEqual(profile.loc[("A", 0), "acc_err"], 1.0 / 2.25)
        self.assertAlmostEqual(profile.loc[("B", 1), "winkler"], 4.0 / 2.25)
        self.assertAlmostEqual(profile.loc[("B", 0), "acc_err"] / profile.loc[("A", 0), "acc_err"], 2.0)

    def test_k_groups_label_the_range_present(self):
        df = _scores(
            {
                ("b", "LR", 2): [0.1, 0.2],
                ("m3", "LR", 3): [0.1, 0.2],
                ("m7", "LR", 7): [0.1, 0.2],
                ("l", "LR", 26): [0.1, 0.2],
            }
        )
        profile = relative_profile(shift_levels(add_bag_index(df), n_bins=2))
        groups = k_groups(profile)
        self.assertEqual([label for label, _ in groups], ["$K=2$", "$3\\leq K\\leq 7$", "$K\\geq 10$"])
        self.assertEqual([part["dataset"].nunique() for _, part in groups], [1, 2, 1])


class TestMethodTableLabel(unittest.TestCase):
    def test_macros_are_resolved(self):
        self.assertEqual(method_table_label("O-LEAP"), "LEAP")
        self.assertEqual(method_table_label("SQS-Hbs"), "SQS-H$_{\\mathrm{bs}}$")
        self.assertEqual(method_table_label("SQS-B"), "SQS-B")
        self.assertEqual(method_table_label("BayesCAP-Hf-real"), "BayCAP-Hf")
        self.assertEqual(method_table_label("BayCAP-KDEy-mlp-real"), "BayCAP-KDEy")
        self.assertEqual(method_table_label("BayesCAP-real"), "BayCAP-sample")
        self.assertEqual(method_table_label("PQ"), "SQS-S")
        self.assertEqual(method_table_label("DoC"), "DoC")


if __name__ == "__main__":
    unittest.main()
