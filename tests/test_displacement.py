import unittest

import numpy as np
import pandas as pd

from uncawe.tables.displacement import (
    COLUMNS,
    _latex_longtable,
    _styled_table,
    class_props_and_recalls,
    conditioning,
    error_terms,
    ordered,
    pair_table,
    prior_part,
    rate_matrix,
    sum_zero_basis,
)


class DisplacementTest(unittest.TestCase):
    def setUp(self):
        # cap-ml layout: true classes on rows, predicted classes on columns
        self.true_ct = np.array([[0.45, 0.05], [0.05, 0.45]])
        self.estim_ct = np.array([[0.4, 0.2], [0.1, 0.3]])
        self.val_counts = np.array([[8, 2], [3, 7]])

    def test_props_and_recalls(self):
        props, recalls = class_props_and_recalls(self.estim_ct)
        np.testing.assert_allclose(props, [0.6, 0.4])
        np.testing.assert_allclose(recalls, [0.4 / 0.6, 0.3 / 0.4])
        _, recalls = class_props_and_recalls(np.array([[0.5, 0.5], [0.0, 0.0]]))
        np.testing.assert_allclose(recalls, [0.5, 0.0])

    def test_error_terms_hand_computed(self):
        terms = error_terms(self.estim_ct, self.true_ct)
        self.assertAlmostEqual(terms["shift"], 0.7 - 0.9)
        # 0.5 * (2/3 - 0.9) + 0.5 * (0.75 - 0.9)
        self.assertAlmostEqual(terms["recall_term"], 0.5 * (2 / 3 - 0.9) + 0.5 * (0.75 - 0.9))
        # 0.1 * 2/3 - 0.1 * 0.75
        self.assertAlmostEqual(terms["prevalence_term"], 0.1 * 2 / 3 - 0.1 * 0.75)
        self.assertAlmostEqual(terms["prev_l1"], 0.2)
        self.assertAlmostEqual(terms["recall_term"] + terms["prevalence_term"], terms["shift"])

    def test_error_terms_with_absent_class(self):
        true_ct = np.array([[0.5, 0.1, 0.0], [0.1, 0.3, 0.0], [0.0, 0.0, 0.0]])
        estim_ct = np.array([[0.3, 0.1, 0.1], [0.1, 0.2, 0.05], [0.05, 0.0, 0.1]])
        terms = error_terms(estim_ct, true_ct)
        self.assertAlmostEqual(terms["recall_term"] + terms["prevalence_term"], terms["shift"])
        props, recalls = class_props_and_recalls(true_ct)
        _, est_recalls = class_props_and_recalls(estim_ct)
        present = props > 0
        expected = np.sum(props[present] * (est_recalls[present] - recalls[present]))
        self.assertAlmostEqual(terms["recall_term"], expected)

    def test_prior_part_hand_computed(self):
        # class 0: 9/12 - 8/10; class 1: 8/12 - 7/10
        expected = 0.5 * (9 / 12 - 0.8) + 0.5 * (8 / 12 - 0.7)
        self.assertAlmostEqual(prior_part(self.val_counts, [0.5, 0.5]), expected)
        self.assertAlmostEqual(prior_part(self.val_counts, [1.0, 0.0]), 9 / 12 - 0.8)

    def test_rate_matrix_is_paper_layout(self):
        rates = rate_matrix(self.val_counts)
        # C_ij = P(Yhat = i | Y = j): column j is the prediction distribution of true class j
        np.testing.assert_allclose(rates, [[0.8, 0.3], [0.2, 0.7]])
        np.testing.assert_allclose(rates.sum(axis=0), [1.0, 1.0])

    def test_sum_zero_basis(self):
        for k in (2, 3, 7):
            basis = sum_zero_basis(k)
            self.assertEqual(basis.shape, (k, k - 1))
            np.testing.assert_allclose(basis.T @ basis, np.eye(k - 1), atol=1e-12)
            np.testing.assert_allclose(basis.sum(axis=0), 0.0, atol=1e-12)

    def test_conditioning_binary(self):
        # Q^T C Q = TPR - FPR = C_11 - C_10 = 0.7 - 0.2
        kappa0, inv_sigma_min = conditioning(rate_matrix(self.val_counts))
        self.assertAlmostEqual(kappa0, 1.0)
        self.assertAlmostEqual(inv_sigma_min, 1 / 0.5)
        self.assertEqual(conditioning(np.eye(4)), (1.0, 1.0))

    def test_conditioning_three_classes(self):
        # symmetric with columns summing to one: the sum-zero vectors are invariant, and the singular values there are
        # the other two eigenvalues, roots of x^2 - 1.2 x + 0.29 (trace 2.2, determinant 0.29, one eigenvalue 1)
        rates = np.array([[0.9, 0.1, 0.0], [0.1, 0.6, 0.3], [0.0, 0.3, 0.7]])
        hi, lo = 0.6 + np.sqrt(0.07), 0.6 - np.sqrt(0.07)
        kappa0, inv_sigma_min = conditioning(rates)
        self.assertAlmostEqual(kappa0, hi / lo)
        self.assertAlmostEqual(inv_sigma_min, 1 / lo)

    def test_conditioning_singular(self):
        self.assertEqual(conditioning(np.full((2, 2), 0.5)), (np.inf, np.inf))
        rates = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual(conditioning(rates), (np.inf, np.inf))

    def test_pair_table_and_order(self):
        def bags(dataset, classifier, n_classes, true_ct, estim_ct):
            return [
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": n_classes,
                    "true_cts": true_ct,
                    "estim_cts": estim_ct,
                    "estim_acc_ranges": np.trace(estim_ct) + np.linspace(-0.01, 0.01, 101),
                }
            ] * 2

        three_true = np.diag([0.3, 0.3, 0.4])
        df = pd.DataFrame(
            bags("ternary", "LR", 3, three_true, three_true)
            + bags("binary", "kNN", 2, self.true_ct, self.estim_ct)
            + bags("binary", "LR", 2, self.true_ct, self.estim_ct)
        )
        val_counts = {
            ("ternary", "LR"): np.diag([10, 20, 30]),
            ("binary", "kNN"): self.val_counts,
            ("binary", "LR"): self.val_counts,
        }
        table = ordered(pair_table(df, val_counts))
        self.assertEqual(list(zip(table["dataset"], table["classifier"])), list(val_counts)[::-1])
        row = table.iloc[0]
        self.assertAlmostEqual(row["shift"], -0.2)
        self.assertAlmostEqual(row["recall_term"] + row["prevalence_term"], row["shift"])
        self.assertAlmostEqual(row["coverage"], 0.0)
        self.assertAlmostEqual(row["val_accuracy"], 0.75)
        self.assertEqual(row["min_val_per_class"], 10)
        self.assertLess(row["max_point_gap"], 1e-12)
        ternary = table.iloc[2]
        self.assertAlmostEqual(ternary["shift"], 0.0)
        self.assertAlmostEqual(ternary["coverage"], 1.0)
        self.assertEqual((ternary["kappa0"], ternary["inv_sigma_min"]), (1.0, 1.0))
        self.assertEqual(ternary["min_val_per_class"], 10)
        self.assertEqual(ternary["median_val_per_class"], 20)

    def test_single_table_layout(self):
        table = pd.DataFrame(
            {
                "dataset": ["letter", "letter"],
                "classifier": ["LR", "kNN"],
                "n_classes": [26, 26],
                "min_val_per_class": [257, 257],
                "median_val_per_class": [268.5, 268.5],
                **{col: [0.5, 0.25] for col in COLUMNS},
            }
        )
        table["kappa0"] = [np.inf, 2.0]
        styled = _styled_table(table)
        self.assertEqual(list(styled.data.columns), list(COLUMNS.values()))
        self.assertEqual(styled.data.index.nlevels, 5)
        latex = _latex_longtable(table)
        for group in ("diagnostics", "interval", "error"):
            self.assertIn(group, latex)
        self.assertIn("$\\infty$", latex)
        self.assertIn("\\label{tab:displacement}", latex)


if __name__ == "__main__":
    unittest.main()
