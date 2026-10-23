import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import pushforward as pushforward_plots
from uncawe.tables.pushforward import (
    _latex_longtable,
    _styled_table,
    bag_scores,
    comparison_table,
    holm,
    interval_scores,
    winkler_tests,
)


def bags(dataset, classifier, n_classes, true_acc, centre, n_bags=2):
    """Bags whose realised table has accuracy ``true_acc`` and whose accuracy samples are centred on ``centre``."""
    true_ct = np.full((n_classes, n_classes), (1 - true_acc) / (n_classes * (n_classes - 1)))
    np.fill_diagonal(true_ct, true_acc / n_classes)
    return [
        {
            "dataset": dataset,
            "classifier": classifier,
            "n_classes": n_classes,
            "true_cts": true_ct,
            "estim_acc_ranges": centre + np.linspace(-0.01, 0.01, 101),
        }
    ] * n_bags


def e21_table(pairs):
    return pd.DataFrame(
        [
            {
                "dataset": d,
                "classifier": c,
                "n_classes": k,
                "n_val": n_val,
                "ratio": ratio,
                "population_coverage": pop,
                "predictive_coverage": approx,
            }
            for d, c, k, n_val, ratio, pop, approx in pairs
        ]
    )


class PushforwardTest(unittest.TestCase):
    def setUp(self):
        population = pd.DataFrame(
            bags("binary", "LR", 2, 0.8, 0.8) + bags("ternary", "LR", 3, 0.6, 0.7) + bags("ternary", "kNN", 3, 0.6, 0.6)
        )
        # pilot-like: the realised pushforward is stored only on two of the three pairs
        realised = pd.DataFrame(bags("ternary", "LR", 3, 0.6, 0.6) + bags("binary", "LR", 2, 0.8, 0.9))
        self.population_bags = bag_scores(population, "BayesCAP")
        self.realised_bags = bag_scores(realised, "BayesCAP-real")
        self.population = interval_scores(self.population_bags)
        self.realised = interval_scores(self.realised_bags)
        self.tests = winkler_tests(self.population_bags, self.realised_bags)
        self.approx = e21_table(
            [
                ("ternary", "LR", 3, 500, 0.5, 0.0, 0.2),
                ("ternary", "kNN", 3, 500, 0.4, 1.0, 1.0),
                ("binary", "LR", 2, 300, 0.1, 1.0, 1.0),
            ]
        )

    def test_interval_scores(self):
        row = self.population.set_index(["dataset", "classifier"]).loc[("ternary", "LR")]
        self.assertEqual(row["n_bags"], 2)
        self.assertAlmostEqual(row["coverage"], 0.0)
        self.assertAlmostEqual(row["amplitude"], 0.019, places=3)
        self.assertGreater(row["winkler"], row["amplitude"])
        self.assertAlmostEqual(
            self.population.set_index(["dataset", "classifier"]).loc[("binary", "LR"), "coverage"], 1.0
        )

    def test_comparison_keeps_pairs_of_both_methods_ordered_by_k(self):
        table = comparison_table(self.population, self.realised, self.approx, self.tests)
        self.assertEqual(list(zip(table["dataset"], table["classifier"])), [("binary", "LR"), ("ternary", "LR")])
        ternary = table.iloc[1]
        self.assertAlmostEqual(ternary["population_coverage"], 0.0)
        self.assertAlmostEqual(ternary["realised_coverage"], 1.0)
        self.assertAlmostEqual(ternary["approx_coverage"], 0.2)
        self.assertAlmostEqual(ternary["ratio"], 0.5)
        self.assertLess(table["e21_population_gap"].max(), 1e-12)
        binary = table.iloc[0]
        self.assertAlmostEqual(binary["realised_coverage"], 0.0)

    def test_comparison_requires_every_pair_in_e21(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            comparison_table(self.population, self.realised, self.approx.iloc[:2], self.tests)

    def test_table_layout(self):
        table = comparison_table(self.population, self.realised, self.approx, self.tests)
        styled = _styled_table(table)
        self.assertEqual(
            list(styled.data.columns.get_level_values(0).unique()), ["BayCAP", "BayCAP-real", "test", "E2.1"]
        )
        self.assertEqual(styled.data.shape, (2, 9))
        self.assertEqual(styled.data.index.nlevels, 3)
        self.assertIn("\\label{tab:pushforward}", _latex_longtable(table))

    def test_winkler_tests_pair_the_bags(self):
        # pairs present in both pushforwards only
        self.assertEqual(
            sorted(zip(self.tests["dataset"], self.tests["classifier"])), [("binary", "LR"), ("ternary", "LR")]
        )
        rng = np.random.default_rng(0)
        true_acc = rng.uniform(0.5, 0.9, 30)
        pop = pd.DataFrame({"dataset": "d", "classifier": "LR", "bag": range(30), "true_acc": true_acc})
        real = pop.copy()
        winkler = rng.uniform(0.1, 0.2, 30)
        cases = {
            "lower": (winkler, winkler - 0.05),
            "higher": (winkler, winkler + 0.05),
            "noise": (winkler, winkler + rng.normal(0, 1e-3, 30) * rng.choice([-1, 1], 30)),
            "equal": (winkler, winkler),
        }
        for name, (w_pop, w_real) in cases.items():
            tests = winkler_tests(pop.assign(winkler=w_pop), real.assign(winkler=w_real))
            p_value = tests["winkler_p"].iloc[0]
            if name in ("lower", "higher"):
                self.assertLess(p_value, 1e-6, name)
                self.assertEqual(np.sign(tests["winkler_diff_median"].iloc[0]), -1 if name == "lower" else 1)
            elif name == "equal":
                self.assertEqual(p_value, 1.0)
            else:
                self.assertGreater(p_value, 0.01, name)
        with self.assertRaisesRegex(ValueError, "differ"):
            winkler_tests(pop.assign(winkler=winkler), real.assign(winkler=winkler, true_acc=true_acc[::-1]))

    def test_holm(self):
        adjusted = holm(np.array([0.01, 0.04, np.nan, 0.03]))
        np.testing.assert_allclose(adjusted[[0, 1, 3]], [0.03, 0.06, 0.06])
        self.assertTrue(np.isnan(adjusted[2]))
        np.testing.assert_allclose(holm(np.array([0.5, 0.9])), [1.0, 1.0])

    def test_figures_build(self):
        table = comparison_table(self.population, self.realised, self.approx, self.tests)
        for plotter in pushforward_plots.PLOTTERS.values():
            fig = plotter(table)
            ax = fig.axes[0]
            self.assertEqual(len(ax.collections), table["classifier"].nunique())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
