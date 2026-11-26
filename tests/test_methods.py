import unittest

import quapy as qp
from cap.models.confidence import ACC_BayesCAP, PrediQuant

from uncawe.experiments.methods import (
    BAYCAP_OPTIONS,
    BAYCAP_VARIANTS,
    BAYES_POOL_METHODS,
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    CAL_VARIANTS,
    FIXED_C_VARIANTS,
    KDEY_HF_VARIANTS,
    KEEP_VARIANTS,
    LOO_VARIANTS,
    METHOD_CLASS,
    PLACEMENT_VARIANTS,
    PQ_RHO_VARIANTS,
    SAMPLE_SIZE,
    SQS_B,
    SQS_B_QUANTIFIER,
    TCAL_VARIANTS,
    TEMPERED_VARIANTS,
    cal_method,
    gen_methods,
    get_method_names,
)
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS
from uncawe.methods.sqs_b import SQSB
from uncawe.methods.tcal import CalibratedTemperature
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
# the methods of the main tables, ranks and coverage vs amplitude plot (user's decision of 2026-11-09); since
# 2026-11-26 (user's decision) the BayCAP methods with the temperature calibrated per pair (WP-34)
MAIN_SET = [
    "DoC",
    "QuAcc_NxN",
    "O-LEAP",
    "AC-LSB",
    "SQS-H-KDEy",
    "SQS-Hbs-KDEy",
    "PQ-KDEy",
    "BayesCAP-tcal",
    "BayesCAP-real-tcal",
    "BayesCAP-Hf-real-tcal",
    "BayCAP-KDEy-h-inst-cb-tcal-keep",
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
        self.assertEqual(
            BAYES_POOL_METHODS,
            set(BAYCAP_METHODS)
            | set(E31_METHODS)
            | set(E32_METHODS)
            | {"BayCAP-KDEy-h-inst"}
            | {"BayCAP-KDEy-h-inst-cb"}
            | {"SQS-B"}
            | set(TEMPERED_VARIANTS)
            | set(PLACEMENT_VARIANTS)
            | set(LOO_VARIANTS)
            | set(CAL_VARIANTS)
            | set(TCAL_VARIANTS)
            | set(KEEP_VARIANTS),
        )
        for name in get_method_names():
            self.assertEqual(
                name in BAYES_POOL_METHODS,
                isinstance(self.methods[name], (BayCAP, BayCAPKDEy, BayCAPMAPLS, SQSB, CalibratedTemperature)),
            )

    def test_factory_names_match_method_class(self):
        self.assertEqual(list(self.methods), get_method_names())

    def test_main_tables_ranks_and_plots_show_the_main_set(self):
        self.assertEqual(main_tables.table_method_names(), MAIN_SET)
        self.assertEqual(rank._methods(), MAIN_SET)
        self.assertEqual(AVC_METHODS, [m for m in MAIN_SET if m != "BayesCAP-tcal"])
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
        for labels in (METHOD_COLORS, METHOD_PLOT_LABELS, METHOD_LATEX_LABELS):
            self.assertIn("SQS-B", labels)

    def test_factory_builds_sqs_b_with_the_quantifier_settings_of_baycap_kdey_mlp_real(self):
        # E3.3 (card WP-18): BayesKDEy with the configuration and seeds of BayCAP-KDEy-mlp-real, so the same draws;
        # since WP-25 SQSB builds its own BayesKDEy from these settings
        self.assertEqual((SQS_B, SQS_B_QUANTIFIER), ("SQS-B", "BayCAP-KDEy-mlp-real"))
        method = self.methods["SQS-B"]
        self.assertIs(METHOD_CLASS["SQS-B"], SQSB)
        self.assertIsInstance(method, SQSB)
        reference = self.methods[SQS_B_QUANTIFIER]
        for attribute in (
            "input_classifier",
            "kernel",
            "bandwidth",
            "bandwidth_grid",
            "shrinkage",
            "temperature",
            "prior",
            "num_samples",
            "num_warmup",
            "random_state",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(method, attribute), getattr(reference, attribute))
        self.assertEqual((reference.input_classifier, reference.c_prior), ("mlp", "fixed"))
        self.assertEqual((method.num_samples, method.sample_size), (BOOTSTRAP_ITERS, SAMPLE_SIZE))
        self.assertEqual(method.random_state, qp.environ["_R_SEED"])


class LOOFactoryTest(unittest.TestCase):
    """The two methods of E7.1 (card WP-25): the stored method with bandwidth="loo", outside the main set."""

    def setUp(self):
        self.methods = dict(gen_methods())

    def test_names_and_classes(self):
        self.assertEqual(LOO_VARIANTS, {"BayCAP-KDEy-mlp-real-loo": "BayCAP-KDEy-mlp-real", "SQS-B-loo": "SQS-B"})
        for name, base in LOO_VARIANTS.items():
            with self.subTest(method=name):
                self.assertIs(METHOD_CLASS[name], METHOD_CLASS[base])
                self.assertIs(type(self.methods[name]), type(self.methods[base]))

    def test_same_configuration_as_the_stored_method_except_the_bandwidth(self):
        for name, base in LOO_VARIANTS.items():
            method, reference = self.methods[name], self.methods[base]
            self.assertEqual((reference.bandwidth, method.bandwidth), (0.1, "loo"))
            self.assertIsNone(method.bandwidth_grid)
            for attribute in (
                "input_classifier",
                "kernel",
                "shrinkage",
                "temperature",
                "prior",
                "num_samples",
                "num_warmup",
                "random_state",
            ):
                with self.subTest(method=name, attribute=attribute):
                    self.assertEqual(getattr(method, attribute), getattr(reference, attribute))
        self.assertEqual(self.methods["SQS-B-loo"].sample_size, self.methods["SQS-B"].sample_size)
        kdey = self.methods["BayCAP-KDEy-mlp-real-loo"]
        self.assertEqual((kdey.estimand, kdey.c_prior), ("realised", "fixed"))

    def test_excluded_from_main_tables_ranks_and_default_plots_and_run_in_the_bayes_pool(self):
        for name in LOO_VARIANTS:
            with self.subTest(method=name):
                self.assertNotIn(name, MAIN_SET)
                self.assertIn(name, main_tables.EXCLUDED_METHODS)
                self.assertIn(name, rank.EXCLUDED_METHODS)
                self.assertNotIn(name, DEFAULT_METHODS)
                self.assertNotIn(name, AVC_METHODS)
                self.assertIn(name, BAYES_POOL_METHODS)
                self.assertIn(name, METHOD_COLORS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_LATEX_LABELS)


class SeededQuantifierTest(unittest.TestCase):
    """The quantifiers of the non-Bayesian methods are seeded with the experiment seed (FIX-3); the two SQS-H (FIX-2)."""

    def test_quantifier_mlps_are_seeded(self):
        methods = dict(gen_methods())
        quantifiers = {
            "QuAcc_NxN": methods["QuAcc_NxN"].base_method.q_class,
            "O-LEAP": methods["O-LEAP"].base_method.q_class,
            "SQS-H": methods["SQS-H"].q,
            "SQS-Hbs": methods["SQS-Hbs"].q,
            "PQ": methods["PQ"].q,
        }
        for name, quantifier in quantifiers.items():
            with self.subTest(method=name):
                self.assertEqual(quantifier.classifier.random_state, qp.environ["_R_SEED"])

    def test_sqs_h_variants(self):
        # FIX-2: SQS-H as in the paper (one estimate of the prevalence) and SQS-Hbs (bootstrap of the prevalence)
        methods = dict(gen_methods())
        self.assertFalse(methods["SQS-H"].bootstrap)
        self.assertTrue(methods["SQS-Hbs"].bootstrap)
        for name in ("SQS-H", "SQS-Hbs"):
            with self.subTest(method=name):
                self.assertEqual(methods[name].num_samples, BOOTSTRAP_ITERS)
                self.assertEqual(methods[name].sample_size, SAMPLE_SIZE)

    def test_sqs_h_and_pq_use_the_experiment_seed(self):
        methods = dict(gen_methods())
        for name in ("SQS-H", "SQS-Hbs", "PQ"):
            with self.subTest(method=name):
                self.assertEqual(methods[name].random_state, qp.environ["_R_SEED"])


class PQRhoFactoryTest(unittest.TestCase):
    """The five methods of E6.2 (card WP-17): SQS-S as PQ with the nested split at rho, outside the main set."""

    NAMES = ("PQ-rho0.1", "PQ-rho0.3", "PQ-rho0.5", "PQ-rho0.7", "PQ-rho0.9")

    def setUp(self):
        self.methods = dict(gen_methods())

    def test_five_levels_with_their_rho(self):
        self.assertEqual(tuple(PQ_RHO_VARIANTS), self.NAMES)
        self.assertEqual(list(PQ_RHO_VARIANTS.values()), [0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertIsNone(self.methods["PQ"].rho)
        for name, rho in PQ_RHO_VARIANTS.items():
            with self.subTest(method=name):
                self.assertIs(METHOD_CLASS[name], PrediQuant)
                self.assertIsInstance(self.methods[name], PrediQuant)
                self.assertEqual(self.methods[name].rho, rho)

    def test_same_configuration_as_pq(self):
        pq = self.methods["PQ"]
        for name in self.NAMES:
            with self.subTest(method=name):
                method = self.methods[name]
                for attribute in ("num_samples", "sample_size", "alpha", "predict_train_prev", "random_state"):
                    self.assertEqual(getattr(method, attribute), getattr(pq, attribute))
                self.assertIs(method.error, pq.error)
                self.assertIs(type(method.q), type(pq.q))
                self.assertEqual(method.q.classifier.random_state, qp.environ["_R_SEED"])
                self.assertIsNot(method.q, pq.q)

    def test_excluded_from_main_tables_ranks_and_default_plots(self):
        for name in self.NAMES:
            with self.subTest(method=name):
                self.assertNotIn(name, MAIN_SET)
                self.assertIn(name, main_tables.EXCLUDED_METHODS)
                self.assertIn(name, rank.EXCLUDED_METHODS)
                self.assertNotIn(name, DEFAULT_METHODS)
                self.assertNotIn(name, AVC_METHODS)
                self.assertNotIn(name, BAYES_POOL_METHODS)


class CalFactoryTest(unittest.TestCase):
    """The method of E7.2 (card WP-26): the stored method with bandwidth="cal", outside the main set."""

    def setUp(self):
        self.methods = dict(gen_methods())

    def test_name_class_and_configuration(self):
        self.assertEqual(
            CAL_VARIANTS,
            {
                "BayCAP-KDEy-mlp-real-cal": "BayCAP-KDEy-mlp-real",
                "BayCAP-KDEy-mlp-real-cal-tcal": "BayCAP-KDEy-mlp-real",
            },
        )
        method, reference = self.methods["BayCAP-KDEy-mlp-real-cal"], self.methods["BayCAP-KDEy-mlp-real"]
        self.assertIs(METHOD_CLASS["BayCAP-KDEy-mlp-real-cal"], BayCAPKDEy)
        self.assertIsInstance(method, BayCAPKDEy)
        self.assertEqual((reference.bandwidth, method.bandwidth), (0.1, "cal"))
        self.assertIsNone(method.bandwidth_grid)
        self.assertEqual((method.cal_bags, method.cal_bag_size, method.cal_nuts_check), (50, 1000, False))
        self.assertEqual(method.cal_solver, "slsqp")
        for attribute in (
            "input_classifier",
            "estimand",
            "c_prior",
            "kernel",
            "shrinkage",
            "temperature",
            "prior",
            "num_samples",
            "num_warmup",
            "random_state",
        ):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(method, attribute), getattr(reference, attribute))
        self.assertTrue(cal_method(None, "BayCAP-KDEy-mlp-real-cal", nuts_check=True).cal_nuts_check)

    def test_tcal_is_the_cal_method_with_the_temperature_selected(self):
        cal, tcal = self.methods["BayCAP-KDEy-mlp-real-cal"], self.methods["BayCAP-KDEy-mlp-real-cal-tcal"]
        self.assertIs(METHOD_CLASS["BayCAP-KDEy-mlp-real-cal-tcal"], BayCAPKDEy)
        self.assertEqual((cal.temperature, tcal.temperature), (1.0, "cal"))
        self.assertFalse(cal.calibrates_temperature)
        self.assertTrue(tcal.calibrates_temperature)
        self.assertIsNone(tcal.temperature_grid)
        self.assertEqual(tcal._tau_grid(), (0.5, 1.0, 2.0, 4.0, 8.0))
        for attribute in vars(cal):
            if attribute not in ("temperature",):
                with self.subTest(attribute=attribute):
                    self.assertEqual(getattr(tcal, attribute), getattr(cal, attribute))

    def test_excluded_from_main_tables_ranks_and_default_plots_and_run_in_the_bayes_pool(self):
        for name in CAL_VARIANTS:
            with self.subTest(name=name):
                self.assertNotIn(name, MAIN_SET)
                self.assertIn(name, main_tables.EXCLUDED_METHODS)
                self.assertIn(name, rank.EXCLUDED_METHODS)
                self.assertNotIn(name, DEFAULT_METHODS)
                self.assertNotIn(name, AVC_METHODS)
                self.assertIn(name, BAYES_POOL_METHODS)
                self.assertIn(name, METHOD_COLORS)
                self.assertIn(name, METHOD_PLOT_LABELS)
                self.assertIn(name, METHOD_LATEX_LABELS)


if __name__ == "__main__":
    unittest.main()
