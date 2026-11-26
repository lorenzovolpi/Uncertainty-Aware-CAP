"""Tests of the SQS methods with KDEy as quantifier (work-plan card WP-36): SQS-H-KDEy, SQS-Hbs-KDEy and PQ-KDEy."""

import unittest

import quapy as qp
from quapy.method.aggregative import PACC, KDEyML

from uncawe.experiments import methods as em
from uncawe.experiments import target_size as ts
from uncawe.experiments.metrics import gen_acc_measure
from uncawe.plots import common as plots_common
from uncawe.plots import shift_profile
from uncawe.plots import target_size_scores as ts_plots
from uncawe.tables import error_split
from uncawe.tables import main as main_tables
from uncawe.tables import target_size_scores as ts_tables
from uncawe.tables.common import METHOD_LATEX_LABELS

KDEY = ("SQS-H-KDEy", "SQS-Hbs-KDEy", "PQ-KDEy")
PACC_METHODS = ("SQS-H", "SQS-Hbs", "PQ")


def _settings(method) -> dict:
    return {k: v for k, v in vars(method).items() if k not in ("q", "acc_fn", "error")}


class SqsKdeyMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qp.environ["_R_SEED"] = 0
        cls.methods = dict(em.gen_methods())

    def test_variants_and_order(self):
        self.assertEqual(em.SQS_KDEY_VARIANTS, dict(zip(KDEY, PACC_METHODS)))
        names = em.get_method_names()
        self.assertEqual(list(self.methods), names)
        # each KDEy method right after its PACC version
        for kdey, pacc in em.SQS_KDEY_VARIANTS.items():
            self.assertEqual(names.index(kdey), names.index(pacc) + 1)
            self.assertIs(em.METHOD_CLASS[kdey], em.METHOD_CLASS[pacc])

    def test_quantifier_is_the_only_difference(self):
        for kdey, pacc in em.SQS_KDEY_VARIANTS.items():
            with self.subTest(method=kdey):
                new, old = self.methods[kdey], self.methods[pacc]
                self.assertIsInstance(old.q, PACC)
                self.assertIsInstance(new.q, KDEyML)
                # the kdey() of LEAP and QuAcc: QuaPy's default bandwidth, the MLP seeded with the experiment seed
                self.assertEqual(new.q.bandwidth, 0.1)
                self.assertEqual(new.q.classifier.random_state, qp.environ["_R_SEED"])
                self.assertEqual(old.q.classifier.random_state, qp.environ["_R_SEED"])
                self.assertEqual(_settings(new), _settings(old))
        # outside the NumPyro pool, as the PACC versions
        self.assertFalse(set(KDEY) & set(em.BAYES_POOL_METHODS))

    def test_defaults_unchanged_elsewhere(self):
        _, acc = next(gen_acc_measure(True))
        self.assertIsInstance(em.sqs_h(acc, bootstrap=True).q, PACC)
        self.assertIsInstance(em.sqs_s(acc).q, PACC)
        # the allocations of E6.2 keep PACC
        for name in em.PQ_RHO_VARIANTS:
            self.assertIsInstance(self.methods[name].q, PACC)

    def test_e44(self):
        for name in KDEY:
            self.assertIn(name, ts.E44_METHODS)
            self.assertIn(name, ts.SQS_METHODS)
            self.assertEqual(ts.E44_METHODS.index(name), ts.E44_METHODS.index(em.SQS_KDEY_VARIANTS[name]) + 1)
        for kdey, pacc in em.SQS_KDEY_VARIANTS.items():
            with self.subTest(method=kdey):
                new, old = ts.make_method(kdey, size=300), ts.make_method(pacc, size=300)
                self.assertIsInstance(new.q, KDEyML)
                self.assertIsInstance(old.q, PACC)
                self.assertEqual(new.sample_size, 300)
                self.assertEqual(_settings(new), _settings(old))


class SqsKdeyTablesTest(unittest.TestCase):
    def test_defaults_of_tables_and_plots(self):
        for kdey, pacc in em.SQS_KDEY_VARIANTS.items():
            with self.subTest(method=kdey):
                self.assertIn(kdey, main_tables.MAIN_METHODS)
                self.assertNotIn(pacc, main_tables.MAIN_METHODS)
                self.assertIn(pacc, main_tables.EXCLUDED_METHODS)
                self.assertIn(kdey, plots_common.DEFAULT_METHODS)
                self.assertNotIn(pacc, plots_common.DEFAULT_METHODS)
                self.assertIn(kdey, error_split.METHODS)
        self.assertIn("SQS-Hbs-KDEy", shift_profile.DEFAULT_METHODS)
        self.assertNotIn("SQS-Hbs", shift_profile.DEFAULT_METHODS)

    def test_labels_mark_the_default(self):
        # the default (KDEy) versions get the plain labels, the PACC versions name their quantifier
        self.assertEqual(METHOD_LATEX_LABELS["SQS-H-KDEy"], "\\sqsh")
        self.assertEqual(METHOD_LATEX_LABELS["SQS-Hbs-KDEy"], "\\sqshbs")
        self.assertEqual(METHOD_LATEX_LABELS["PQ-KDEy"], "\\pq")
        for pacc in PACC_METHODS:
            self.assertTrue(METHOD_LATEX_LABELS[pacc].endswith("-PACC"))
            self.assertTrue(plots_common.METHOD_PLOT_LABELS[pacc].endswith("-PACC"))
            self.assertTrue(error_split.SHORT_LABELS[pacc].endswith("-PACC"))
            self.assertTrue(ts_tables.SHORT_LABELS[pacc].endswith("-PACC"))
        self.assertEqual(plots_common.METHOD_PLOT_LABELS["PQ-KDEy"], "SQS-S")
        self.assertEqual(plots_common.METHOD_PLOT_LABELS["SQS-H-KDEy"], "SQS-H")
        for name in (*KDEY, *PACC_METHODS):
            with self.subTest(method=name):
                self.assertIn(name, plots_common.METHOD_COLORS)
                self.assertIn(name, ts_plots.METHOD_COLORS)
                self.assertIn(name, error_split.SHORT_LABELS)
                self.assertIn(name, ts_tables.SHORT_LABELS)
        for name in shift_profile.DEFAULT_METHODS:
            self.assertIn(name, shift_profile.SHIFT_PROFILE_COLORS)
        colors = [plots_common.METHOD_COLORS[m] for m in (*KDEY, *PACC_METHODS)]
        self.assertEqual(len(set(colors)), len(colors))


if __name__ == "__main__":
    unittest.main()
