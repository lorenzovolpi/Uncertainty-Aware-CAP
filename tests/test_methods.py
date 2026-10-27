import unittest

from cap.models.confidence import ACC_BayesCAP

from uncawe.experiments.methods import (
    BAYCAP_VARIANTS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    METHOD_CLASS,
    gen_methods,
    get_method_names,
)
from uncawe.methods.baycap import BayCAP
from uncawe.plots.amplitude_vs_coverage import PLOT_METHODS as AVC_METHODS
from uncawe.plots.common import DEFAULT_METHODS, METHOD_COLORS, METHOD_PLOT_LABELS
from uncawe.tables import main as main_tables
from uncawe.tables import rank
from uncawe.tables.common import METHOD_LATEX_LABELS
from uncawe.tables.displacement import METHOD_LABELS

BAYCAP_METHODS = ["BayesCAP", "BayesCAP-real", "BayesCAP-EB", "BayesCAP-EB-real"]
E2_METHODS = ["BayesCAP-real", "BayesCAP-EB", "BayesCAP-EB-real"]


class BayCAPFactoryTest(unittest.TestCase):
    def setUp(self):
        self.methods = dict(gen_methods())

    def test_factory_builds_every_baycap_method_with_main_settings(self):
        reference = ACC_BayesCAP(None, num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS)
        expected = {
            "BayesCAP": ("population", "uniform"),
            "BayesCAP-real": ("realised", "uniform"),
            "BayesCAP-EB": ("population", "empirical-bayes"),
            "BayesCAP-EB-real": ("realised", "empirical-bayes"),
        }
        self.assertEqual(BAYCAP_VARIANTS, expected)
        for name, (estimand, c_prior) in expected.items():
            with self.subTest(method=name):
                method = self.methods[name]
                self.assertIs(METHOD_CLASS[name], BayCAP)
                self.assertIsInstance(method, BayCAP)
                self.assertEqual((method.estimand, method.c_prior), (estimand, c_prior))
                self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
                # ACC_BayesCAP stores its seed as ``randm_state``; equal seeds give equal NUTS draws
                self.assertEqual(method.random_state, reference.randm_state)
                self.assertEqual(method.prior, reference.prior)

    def test_factory_names_match_method_class(self):
        self.assertEqual(list(self.methods), get_method_names())

    def test_main_tables_ranks_and_plots_exclude_the_e2_methods(self):
        for name in ("BayesCAP", "BayesCAP-EB-real"):
            self.assertIn(name, main_tables.table_method_names())
            self.assertIn(name, rank._methods())
        self.assertNotIn("L-CBPE", rank._methods())
        for name in E2_METHODS:
            with self.subTest(method=name):
                if name != "BayesCAP-EB-real":
                    self.assertNotIn(name, main_tables.table_method_names())
                    self.assertNotIn(name, rank._methods())
                self.assertNotIn(name, DEFAULT_METHODS)

    def test_coverage_vs_amplitude_plot_shows_every_baycap_method(self):
        self.assertEqual(AVC_METHODS[: len(DEFAULT_METHODS)], DEFAULT_METHODS)
        for name in BAYCAP_METHODS:
            with self.subTest(method=name):
                self.assertIn(name, AVC_METHODS)

    def test_every_baycap_method_has_labels_and_a_colour(self):
        for name in BAYCAP_METHODS:
            with self.subTest(method=name):
                self.assertIn(name, METHOD_COLORS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_LATEX_LABELS)
                self.assertIn(name, METHOD_LABELS)


if __name__ == "__main__":
    unittest.main()
