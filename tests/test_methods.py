import unittest

from cap.models.confidence import ACC_BayesCAP

from uncawe.experiments.methods import (
    BAYCAP_OPTIONS,
    BAYCAP_VARIANTS,
    BAYES_POOL_METHODS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    FIXED_C_VARIANTS,
    METHOD_CLASS,
    gen_methods,
    get_method_names,
)
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_fixed import BayCAPFixed
from uncawe.plots.amplitude_vs_coverage import PLOT_METHODS as AVC_METHODS
from uncawe.plots.common import DEFAULT_METHODS, METHOD_COLORS, METHOD_PLOT_LABELS
from uncawe.tables import main as main_tables
from uncawe.tables import rank
from uncawe.tables.common import METHOD_LATEX_LABELS
from uncawe.tables.displacement import METHOD_LABELS

E2_METHODS = [
    "BayesCAP-real",
    "BayesCAP-EB",
    "BayesCAP-EB-real",
    "BayesCAP-H-real",
    "BayesCAP-Hf-real",
    "BayesCAP-Hfn-real",
]
BAYCAP_METHODS = ["BayesCAP", *E2_METHODS]
# E2 methods in the tables of main (user's requests)
IN_MAIN_TABLES = ["BayesCAP-EB-real", "BayesCAP-H-real"]
# E2 methods in the ranks of main (user's request of 2026-11-05)
IN_RANKS = ["BayesCAP-Hf-real"]
# BayCAP methods in the coverage vs amplitude plot (user's requests); BayesCAP-Hf-real (E2.8) and BayesCAP-Hfn-real
# (E2.9) wait for the user
IN_AVC = ["BayesCAP", "BayesCAP-real", "BayesCAP-EB", "BayesCAP-EB-real", "BayesCAP-H-real"]
# the eight methods with a fixed rate matrix of E3.1 (docs/ext3_posterior_propagation.md)
E31_METHODS = [f"BayCAP-{q}-{s}{x}" for q in ("KDEy", "EMQ") for s in ("h", "mlp") for x in ("", "-real")]


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
            "BayesCAP-H-real": ("realised", "hierarchical"),
            "BayesCAP-Hf-real": ("realised", "hierarchical-f"),
            "BayesCAP-Hfn-real": ("realised", "hierarchical-f"),
        }
        self.assertEqual(BAYCAP_VARIANTS, expected)
        # only BayesCAP-Hfn-real changes another hyperparameter: the coordinates of the shares (E2.9)
        self.assertEqual(BAYCAP_OPTIONS, {"BayesCAP-Hfn-real": {"shares": "non-centred"}})
        for name, (estimand, c_prior) in expected.items():
            with self.subTest(method=name):
                method = self.methods[name]
                self.assertIs(METHOD_CLASS[name], BayCAP)
                self.assertIsInstance(method, BayCAP)
                self.assertEqual((method.estimand, method.c_prior), (estimand, c_prior))
                self.assertEqual(method.shares, BAYCAP_OPTIONS.get(name, {}).get("shares", "centred"))
                self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
                # ACC_BayesCAP stores its seed as ``randm_state``; equal seeds give equal NUTS draws
                self.assertEqual(method.random_state, reference.randm_state)
                self.assertEqual(method.prior, reference.prior)

    def test_factory_builds_the_e31_methods_with_their_configuration(self):
        self.assertEqual(sorted(FIXED_C_VARIANTS), sorted(E31_METHODS))
        reference = BayCAP(None)
        for name in E31_METHODS:
            with self.subTest(method=name):
                method = self.methods[name]
                quantifier, source, estimand = FIXED_C_VARIANTS[name]
                self.assertIs(METHOD_CLASS[name], BayCAPFixed)
                self.assertIsInstance(method, BayCAPFixed)
                self.assertEqual(quantifier, "kdey" if "-KDEy-" in name else "emq")
                self.assertEqual(source, "h" if "-h" in name.replace("-real", "") else "mlp")
                self.assertEqual(estimand, "realised" if name.endswith("-real") else "population")
                self.assertEqual(
                    (method.quantifier, method.input_classifier, method.estimand), (quantifier, source, estimand)
                )
                self.assertEqual((method.kernel, method.bandwidth, method.shrinkage), ("gaussian", 0.1, 0.0))
                self.assertEqual((method.temperature, method.prior), (1.0, "uniform"))
                self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
                self.assertEqual(method.random_state, reference.random_state)

    def test_bayes_pool_holds_every_baycap_method(self):
        self.assertEqual(BAYES_POOL_METHODS, set(BAYCAP_METHODS) | set(E31_METHODS))
        for name in get_method_names():
            self.assertEqual(name in BAYES_POOL_METHODS, isinstance(self.methods[name], (BayCAP, BayCAPFixed)))

    def test_factory_names_match_method_class(self):
        self.assertEqual(list(self.methods), get_method_names())

    def test_main_tables_ranks_and_plots_exclude_the_e2_methods(self):
        for name in ("BayesCAP", *IN_MAIN_TABLES):
            self.assertIn(name, main_tables.table_method_names())
        for name in ("BayesCAP", *IN_RANKS):
            self.assertIn(name, rank._methods())
        self.assertNotIn("L-CBPE", rank._methods())
        for name in E2_METHODS:
            with self.subTest(method=name):
                if name not in IN_MAIN_TABLES:
                    self.assertNotIn(name, main_tables.table_method_names())
                if name not in IN_RANKS:
                    self.assertNotIn(name, rank._methods())
                self.assertNotIn(name, DEFAULT_METHODS)
        for name in E31_METHODS:
            with self.subTest(method=name):
                self.assertNotIn(name, main_tables.table_method_names())
                self.assertNotIn(name, rank._methods())
                self.assertNotIn(name, DEFAULT_METHODS)
                self.assertNotIn(name, AVC_METHODS)

    def test_coverage_vs_amplitude_plot_shows_the_requested_baycap_methods(self):
        self.assertEqual(AVC_METHODS[: len(DEFAULT_METHODS)], DEFAULT_METHODS)
        for name in BAYCAP_METHODS:
            with self.subTest(method=name):
                if name in IN_AVC:
                    self.assertIn(name, AVC_METHODS)
                else:
                    self.assertNotIn(name, AVC_METHODS)

    def test_every_baycap_method_has_labels_and_a_colour(self):
        for name in BAYCAP_METHODS + E31_METHODS:
            with self.subTest(method=name):
                self.assertIn(name, METHOD_COLORS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_LATEX_LABELS)
                self.assertIn(name, METHOD_LABELS)


if __name__ == "__main__":
    unittest.main()
