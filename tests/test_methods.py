import unittest

import quapy as qp
from cap.models.confidence import ACC_BayesCAP

from uncawe.experiments.methods import (
    BAYCAP_OPTIONS,
    BAYCAP_VARIANTS,
    BAYES_POOL_METHODS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    FIXED_C_VARIANTS,
    KDEY_HF_VARIANTS,
    METHOD_CLASS,
    gen_methods,
    get_method_names,
)
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS
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
# the methods of the main tables, ranks and coverage vs amplitude plot (user's decision of 2026-11-09)
MAIN_SET = [
    "DoC",
    "QuAcc_NxN",
    "O-LEAP",
    "RQBS",
    "PQ",
    "BayesCAP-real",
    "BayesCAP-Hf-real",
    "BayCAP-KDEy-mlp-real",
]
# the eight methods with a fixed rate matrix of E3.1 (docs/ext3_posterior_propagation.md)
E31_METHODS = [f"BayCAP-{q}-{s}{x}" for q in ("KDEy", "EMQ") for s in ("h", "mlp") for x in ("", "-real")]
# the two methods with a random C of E3.2
E32_METHODS = ["BayCAP-KDEy-Hf-h-real", "BayCAP-KDEy-Hf-mlp-real"]


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
                cls = BayCAPKDEy if quantifier == "kdey" else BayCAPMAPLS
                self.assertIs(METHOD_CLASS[name], cls)
                self.assertIs(type(method), cls)
                self.assertEqual(quantifier, "kdey" if "-KDEy-" in name else "emq")
                self.assertEqual(source, "h" if "-h" in name.replace("-real", "") else "mlp")
                self.assertEqual(estimand, "realised" if name.endswith("-real") else "population")
                self.assertEqual((method.input_classifier, method.estimand), (source, estimand))
                if quantifier == "kdey":
                    self.assertEqual(method.c_prior, "fixed")
                    self.assertEqual((method.kernel, method.bandwidth, method.shrinkage), ("gaussian", 0.1, 0.0))
                self.assertEqual((method.temperature, method.prior), (1.0, "uniform"))
                self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
                self.assertEqual(method.random_state, reference.random_state)

    def test_factory_builds_the_e32_methods_with_their_configuration(self):
        self.assertEqual(KDEY_HF_VARIANTS, {name: name.split("-")[3] for name in E32_METHODS})
        reference = BayCAP(None)
        for name in E32_METHODS:
            with self.subTest(method=name):
                method = self.methods[name]
                self.assertIs(METHOD_CLASS[name], BayCAPKDEy)
                self.assertIsInstance(method, BayCAPKDEy)
                self.assertEqual(
                    (method.input_classifier, method.estimand, method.c_prior),
                    (KDEY_HF_VARIANTS[name], "realised", "hierarchical-f"),
                )
                self.assertEqual((method.kernel, method.bandwidth, method.shrinkage), ("gaussian", 0.1, 0.0))
                self.assertEqual((method.temperature, method.prior), (1.0, "uniform"))
                self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
                self.assertEqual(method.random_state, reference.random_state)

    def test_bayes_pool_holds_every_baycap_method(self):
        self.assertEqual(BAYES_POOL_METHODS, set(BAYCAP_METHODS) | set(E31_METHODS) | set(E32_METHODS))
        for name in get_method_names():
            self.assertEqual(
                name in BAYES_POOL_METHODS, isinstance(self.methods[name], (BayCAP, BayCAPKDEy, BayCAPMAPLS))
            )

    def test_factory_names_match_method_class(self):
        self.assertEqual(list(self.methods), get_method_names())

    def test_main_tables_ranks_and_plots_show_the_main_set(self):
        self.assertEqual(main_tables.table_method_names(), MAIN_SET)
        self.assertEqual(rank._methods(), MAIN_SET)
        self.assertEqual(AVC_METHODS, MAIN_SET)
        for name in ("BayesCAP", "L-CBPE", *E2_METHODS, *E31_METHODS, *E32_METHODS):
            with self.subTest(method=name):
                if name not in MAIN_SET:
                    self.assertIn(name, main_tables.EXCLUDED_METHODS)
                    self.assertIn(name, rank.EXCLUDED_METHODS)
        for name in (*E2_METHODS, *E31_METHODS, *E32_METHODS):
            with self.subTest(method=name):
                self.assertNotIn(name, DEFAULT_METHODS)

    def test_every_baycap_method_has_labels_and_a_colour(self):
        for name in BAYCAP_METHODS + E31_METHODS + E32_METHODS:
            with self.subTest(method=name):
                self.assertIn(name, METHOD_COLORS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_LATEX_LABELS)
                self.assertIn(name, METHOD_LABELS)


class SeededQuantifierTest(unittest.TestCase):
    """The MLPs of the quantifiers of the non-Bayesian methods are seeded with the experiment seed (FIX-3)."""

    def test_quantifier_mlps_are_seeded(self):
        methods = dict(gen_methods())
        quantifiers = {
            "QuAcc_NxN": methods["QuAcc_NxN"].base_method.q_class,
            "O-LEAP": methods["O-LEAP"].base_method.q_class,
            "RQBS": methods["RQBS"].q,
            "PQ": methods["PQ"].q,
        }
        for name, quantifier in quantifiers.items():
            with self.subTest(method=name):
                self.assertEqual(quantifier.classifier.random_state, qp.environ["_R_SEED"])

    def test_rqbs_and_pq_use_the_experiment_seed(self):
        methods = dict(gen_methods())
        for name in ("RQBS", "PQ"):
            with self.subTest(method=name):
                self.assertEqual(methods[name].random_state, qp.environ["_R_SEED"])


if __name__ == "__main__":
    unittest.main()
