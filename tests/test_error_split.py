import os
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe import error_split as es
from uncawe.plots import error_split as plots
from uncawe.tables import error_split as tables


def _random_tables(rng, n_bags, n_classes, zero_rows=False):
    """Random tables in the cap-ml layout (true classes on rows), each summing to 1."""
    cts = rng.dirichlet(np.ones(n_classes * n_classes), size=n_bags).reshape(n_bags, n_classes, n_classes)
    if zero_rows:
        cts[0, 0, :] = 0.0
        cts /= cts.sum(axis=(1, 2), keepdims=True)
    return cts


def _table(props, recalls):
    """A cap-ml table with the given prevalences and recalls, mistakes spread evenly on the other predicted classes."""
    props, recalls = np.asarray(props, dtype=float), np.asarray(recalls, dtype=float)
    k = len(props)
    ct = np.outer(props * (1 - recalls) / (k - 1), np.ones(k))
    np.fill_diagonal(ct, props * recalls)
    return ct


class TestSplit(unittest.TestCase):
    def test_identity_on_random_tables(self):
        rng = np.random.default_rng(0)
        for k in (2, 3, 7):
            est, true = _random_tables(rng, 50, k), _random_tables(rng, 50, k)
            d_pool = rng.uniform(0.3, 0.95, size=k)
            out = es.split(est, true, d_pool)
            total = np.trace(est, axis1=1, axis2=2) - np.trace(true, axis1=1, axis2=2)
            np.testing.assert_allclose(out["e1"] + out["e2"] + out["e3"], total, atol=1e-12)
            self.assertLess(out["identity_gap"].max(), 1e-12)
            np.testing.assert_allclose(out["acc_est_est"], out["estimate"], atol=1e-12)

    def test_reading_a_transposed_table(self):
        rng = np.random.default_rng(1)
        k = 4
        rates = rng.dirichlet(np.ones(k), size=k).T  # paper layout: C[i, j] = P(pred i | true j), columns sum to 1
        pi = rng.dirichlet(np.ones(k))
        paper = rates @ np.diag(pi)
        props, recalls, empty = es.props_and_recalls(paper.T)
        np.testing.assert_allclose(props[0], paper.sum(axis=0))
        np.testing.assert_allclose(recalls[0], np.diag(rates))
        self.assertFalse(empty.any())

    def test_unnormalised_tables_are_normalised(self):
        ct = _table([0.3, 0.7], [0.9, 0.6])
        a = es.split(ct * 1000, ct * 1000, [0.8, 0.7])
        b = es.split(ct, ct, [0.8, 0.7])
        for key in es.SPLIT_KEYS:
            np.testing.assert_allclose(a[key], b[key])

    def test_right_components_leave_only_the_luck_of_the_bag(self):
        d_pool = np.array([0.9, 0.7, 0.5])
        true = _table([0.2, 0.5, 0.3], [0.85, 0.72, 0.4])
        est = _table([0.2, 0.5, 0.3], d_pool)
        out = es.split(est, true, d_pool)
        self.assertAlmostEqual(out["e1"][0], 0.0, places=12)
        self.assertAlmostEqual(out["e2"][0], 0.0, places=12)
        self.assertAlmostEqual(out["e3"][0], out["total"][0], places=12)

    def test_worked_example_of_the_document(self):
        # docs/ext4_uncertainty_budget.md §T.2
        out = es.split(_table([0.45, 0.55], [0.88, 0.73]), _table([0.5, 0.5], [0.89, 0.69]), [0.90, 0.70])
        self.assertAlmostEqual(out["total"][0], 0.0075, places=12)
        self.assertAlmostEqual(out["e1"][0], -0.0075, places=12)
        self.assertAlmostEqual(out["e2"][0], 0.005, places=12)
        self.assertAlmostEqual(out["e3"][0], 0.01, places=12)

    def test_oracle_cells(self):
        rng = np.random.default_rng(2)
        est, true = _random_tables(rng, 40, 3), _random_tables(rng, 40, 3)
        out = es.split(est, true, rng.uniform(0.4, 0.9, size=3))
        a = out["true_acc"]
        e1, e2, e3, x = out["e1"], out["e2"], out["e3"], out["x"]
        np.testing.assert_allclose(out["acc_est_est"] - a, e1 + e2 + e3, atol=1e-12)
        np.testing.assert_allclose(out["acc_oracle_est"] - a, e2 + e3, atol=1e-12)
        np.testing.assert_allclose(out["acc_est_oracle"] - a, e1 - x + e3, atol=1e-12)
        np.testing.assert_allclose(out["acc_oracle_oracle"] - a, e3, atol=1e-12)
        np.testing.assert_allclose(out["acc_oracle_oracle"], out["pool_acc"])

    def test_class_absent_from_the_bag(self):
        d_pool = np.array([0.9, 0.6, 0.8])
        true = _table([0.4, 0.6, 0.0], [0.85, 0.65, 0.5])
        est = _table([0.35, 0.55, 0.10], [0.88, 0.6, 0.7])
        out = es.split(est, true, d_pool)
        expected_e3 = 0.4 * (0.9 - 0.85) + 0.6 * (0.6 - 0.65)
        self.assertAlmostEqual(out["e3"][0], expected_e3, places=12)
        self.assertLess(out["identity_gap"][0], 1e-12)
        self.assertEqual(out["n_empty"][0], 0)

    def test_empty_estimated_column(self):
        d_pool = np.array([0.9, 0.6, 0.8])
        true = _table([0.3, 0.5, 0.2], [0.85, 0.65, 0.75])
        est = _table([0.4, 0.6, 0.0], [0.88, 0.6, 0.7])
        out = es.split(est, true, d_pool)
        _, d_t, empty = es.props_and_recalls(est, fill=d_pool)
        self.assertEqual(d_t[0, 2], d_pool[2])
        self.assertTrue(empty[0, 2])
        self.assertEqual(out["n_empty"][0], 1)
        # class 3: no recall error, its prevalence error (0 - 0.2) weighted by d_pool goes to e1
        e1 = 0.1 * 0.88 + 0.1 * 0.6 - 0.2 * 0.8
        e2 = 0.3 * (0.88 - 0.9) + 0.5 * (0.6 - 0.6)
        self.assertAlmostEqual(out["e1"][0], e1, places=12)
        self.assertAlmostEqual(out["e2"][0], e2, places=12)
        self.assertLess(out["identity_gap"][0], 1e-12)

    def test_pool_recalls(self):
        y = np.array([0, 0, 0, 0, 1, 1, 2, 2, 2, 2])
        y_hat = np.array([0, 0, 1, 0, 1, 0, 2, 2, 2, 0])
        np.testing.assert_allclose(es.pool_recalls(y, y_hat, 3), [0.75, 0.5, 0.75])
        with self.assertRaises(ValueError):
            es.pool_recalls(y, y_hat, 4)

    def test_rms_and_largest_part(self):
        self.assertAlmostEqual(es.rms([3.0, -4.0]), np.sqrt(12.5))
        self.assertTrue(np.isnan(es.rms([])))
        self.assertEqual(es.largest_part([0.1, 0.3, 0.2]), 2)
        self.assertEqual(es.largest_part([np.nan, np.nan, np.nan]), 0)


N_BAGS = 10
PAIRS = (("spam", "LR", 2), ("spam", "kNN", 2), ("wine", "LR", 3))
# a prevalence and a recall error per method, so that the methods differ
METHOD_ERRORS = {m: (0.01 * (i + 1), 0.005 * (i % 3)) for i, m in enumerate(tables.METHODS)}


def _synthetic():
    rng = np.random.default_rng(3)
    results = {m: [] for m in tables.METHODS}
    pool = {}
    for dataset, classifier, k in PAIRS:
        d_pool = rng.uniform(0.6, 0.9, size=k)
        pool[(dataset, classifier)] = {"d_pool": d_pool, "n_val": 500 if dataset == "spam" else 300, "n_pool": 1000}
        bags = []
        for _ in range(N_BAGS):
            props = rng.dirichlet(np.ones(k))
            bags.append((props, _table(props, np.clip(d_pool + rng.normal(0, 0.02, k), 0, 1))))
        for m, (prev_err, rec_err) in METHOD_ERRORS.items():
            for b, (props, true_ct) in enumerate(bags):
                est_props = np.abs(props + prev_err * rng.normal(size=k))
                est_props /= est_props.sum()
                est = _table(est_props, np.clip(d_pool + rec_err, 0, 1))
                failed = m == "BayesCAP-Hf-real" and dataset == "wine" and b == 0
                if failed:
                    est = np.full((k, k), np.nan)
                row = {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": k,
                    "method": m,
                    "true_cts": true_ct,
                    "estim_cts": est,
                    "estim_accs": np.nan if failed else float(np.trace(est)),
                }
                if m == "BayesCAP-Hf-real":
                    row["nuts_failed"] = int(failed)
                results[m].append(row)
    return {m: pd.DataFrame(rows) for m, rows in results.items()}, pool


class TestTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        results, pool = _synthetic()
        cls.bags, cls.table = tables.compute(results, pool)

    def test_pair_table(self):
        self.assertEqual(len(self.table), len(PAIRS))
        self.assertEqual(list(self.table["dataset"].astype(str)), ["spam", "spam", "wine"])
        self.assertLess(max(self.table[f"{m}_max_identity_gap"].max() for m in tables.METHODS), tables.IDENTITY_TOL)
        wine = self.table.loc[self.table["dataset"] == "wine"].iloc[0]
        self.assertEqual(wine["BayesCAP-Hf-real_n_dropped"], 1)
        self.assertEqual(wine["PQ_n_dropped"], 0)
        self.assertEqual(len(self.bags), len(PAIRS) * N_BAGS * len(tables.METHODS) - 1)
        self.assertEqual(wine["oracle_oracle_ae_pool"], 0.0)
        # the oracle cell (pi^Q, d^pool) of every method equals the e3 of the bag
        for m in tables.METHODS:
            np.testing.assert_allclose(self.table[f"{m}_oracle_oracle_ae_pool"], 0.0, atol=1e-12)

    def test_check_summary(self):
        checks = tables.check_summary(self.table).set_index("method")
        self.assertEqual(checks.loc["BayesCAP-Hf-real", "dropped_bags"], 1)
        self.assertEqual(checks["pairs_above_tol"].sum(), 0)

    def test_rms_styler(self):
        styler = tables.rms_styler(self.table)
        latex = styler.to_latex(hrules=True, multicol_align="c")
        self.assertIn("cellcolor", latex)
        self.assertIn("\\bfseries", latex)
        self.assertRegex(latex, r"\$\^\{[123]\}\$")
        self.assertIn("largest part, binary", latex)
        self.assertIn("mean, multiclass", latex)
        data = styler.data
        styles = styler._compute().ctx  # the styles of the original styler (to_latex renders a copy)
        e3_col = data.columns.get_loc((tables.PART_LABELS["e3"], ""))
        self.assertFalse(any(styles.get((i, e3_col)) for i in range(len(data))))
        # one bold cell per row in each coloured group of a pair row
        group = [i for i, c in enumerate(data.columns) if c[0] == tables.PART_LABELS["e1"]]
        bold = [any("bfseries" in prop for prop, _ in styles.get((0, j), [])) for j in group]
        self.assertGreaterEqual(sum(bold), 1)
        # the method with the smallest prevalence error has the smallest RMS of e1 on the first row
        values = data.iloc[0, group].astype(float).to_numpy()
        self.assertEqual(int(np.argmin(values)), 0)
        self.assertTrue(bold[0])

    def test_share_row(self):
        styler = tables.rms_styler(self.table)
        data = styler.data
        share_rows = [i for i, label in enumerate(data.index.get_level_values(0)) if "largest part" in label]
        self.assertEqual(len(share_rows), 2)
        for i in share_rows:
            for m in tables.METHODS:
                cells = [data.iloc[i][(tables.PART_LABELS[p], tables.SHORT_LABELS[m])] for p in ("e1", "e2", "total")]
                shares = [int(c.replace("\\%", "")) for c in cells]
                self.assertTrue(98 <= sum(shares) <= 102)

    def test_other_stylers(self):
        for name, styler in tables.stylers(self.table, tables.table_names(tables.TABLE_SETS)).items():
            latex = styler.to_latex(hrules=True, multicol_align="c")
            self.assertIn("mean, binary", latex, name)
            if name == "signed":
                self.assertNotIn("cellcolor", latex)
            else:
                self.assertIn("cellcolor", latex)

    def test_bar_plot(self):
        fig = plots.plot_bars(self.table)
        self.assertEqual(sum(ax.get_visible() for ax in fig.axes), 2)
        # QuAcc_NxN is left out of the main figure: one bar per other method
        self.assertNotIn("QuAcc_NxN", plots.MAIN_METHODS)
        self.assertEqual(len(plots.MAIN_METHODS), len(tables.METHODS) - 1)
        self.assertEqual(len(fig.axes[0].get_xticklabels()), len(plots.MAIN_METHODS))
        ms = plots.mean_squares(self.table)
        self.assertEqual(len(ms), 2 * len(tables.METHODS))
        plt.close(fig)

    def test_separate_bar_plot(self):
        fig = plots.plot_bars_single(self.table)
        self.assertEqual(len(fig.axes), 1)
        # one bar per dataset, stacking the three parts
        self.assertEqual(len(fig.axes[0].get_xticklabels()), 2)
        self.assertEqual(len(fig.axes[0].patches), 2 * len(plots.PART_COLORS))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
