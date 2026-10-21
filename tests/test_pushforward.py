import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe.plots import pushforward as pushforward_plots
from uncawe.tables.pushforward import _latex_longtable, _styled_table, comparison_table, interval_scores


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
        self.population = interval_scores(population, "BayesCAP")
        self.realised = interval_scores(realised, "BayesCAP-real")
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
        table = comparison_table(self.population, self.realised, self.approx)
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
            comparison_table(self.population, self.realised, self.approx.iloc[:2])

    def test_table_layout(self):
        table = comparison_table(self.population, self.realised, self.approx)
        styled = _styled_table(table)
        self.assertEqual(list(styled.data.columns.get_level_values(0).unique()), ["BayCAP", "BayCAP-real", "E2.1"])
        self.assertEqual(styled.data.shape, (2, 8))
        self.assertEqual(styled.data.index.nlevels, 3)
        self.assertIn("\\label{tab:pushforward}", _latex_longtable(table))

    def test_figures_build(self):
        table = comparison_table(self.population, self.realised, self.approx)
        for plotter in pushforward_plots.PLOTTERS.values():
            fig = plotter(table)
            ax = fig.axes[0]
            self.assertEqual(len(ax.collections), table["classifier"].nunique())
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
