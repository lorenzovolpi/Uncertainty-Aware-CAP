"""Tests of the per-instance pushforward of BayCAP-KDEy (work-plan card WP-31, E3.4 of
``docs/ext3_posterior_propagation.md``, §T.15)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest.mock import patch

import numpy as np
from cap.error import vanilla_acc
from test_baycap_fixed import DATA, SETTINGS

from uncawe.experiments import methods as experiment_methods
from uncawe.methods.baycap import realised_tables
from uncawe.methods.baycap_kdey import (
    INSTANCE,
    INSTANCE_SUMMARIES,
    BayCAPKDEy,
    instance_moments,
    instance_tables,
    instance_weights,
)
from uncawe.methods.baycap_mapls import BayCAPMAPLS
from uncawe.methods.baycap_quantifier import FIXED_SUMMARIES, accuracy_moments, class_precisions
from uncawe.plots import shift_profile
from uncawe.plots.common import method_plot_label
from uncawe.tables import main as main_tables
from uncawe.tables.common import METHOD_LATEX_LABELS

INST = "BayCAP-KDEy-h-inst"


def _method(**kwargs) -> BayCAPKDEy:
    return BayCAPKDEy(vanilla_acc, **(SETTINGS | {"input_classifier": "h", "estimand": INSTANCE} | kwargs))


def _fitted(**kwargs) -> BayCAPKDEy:
    return _method(**kwargs).fit(DATA.val, DATA.val_posteriors)


class InstanceWeightsTest(unittest.TestCase):
    def test_weights_are_bayes_rule_on_a_small_mixture(self):
        # two Gaussian classes in one dimension, three instances: w_nj = pi_j f_j(z_n) / sum_l pi_l f_l(z_n)
        z = np.array([-1.0, 0.2, 1.5])
        means, sd = np.array([0.0, 1.0]), 0.7
        densities = np.exp(-0.5 * ((z[None, :] - means[:, None]) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
        prevalences = np.array([[0.3, 0.7], [0.9, 0.1]])
        expected = prevalences[:, None, :] * densities.T[None, :, :]
        expected /= expected.sum(axis=2, keepdims=True)
        np.testing.assert_allclose(instance_weights(prevalences, np.log(densities)), expected, rtol=1e-12)

    def test_log_sum_exp_handles_tiny_densities_and_impossible_instances(self):
        log_densities = np.array([[-800.0, -np.inf], [-801.0, -np.inf]])
        weights = instance_weights(np.array([[0.5, 0.5]]), log_densities)
        np.testing.assert_allclose(weights[0, 0], [np.e / (1 + np.e), 1 / (1 + np.e)])
        np.testing.assert_allclose(weights[0, 1], [0.5, 0.5])
        # a class with zero prevalence gets zero weight
        np.testing.assert_allclose(instance_weights(np.array([[1.0, 0.0]]), np.zeros((2, 2)))[0], [[1, 0], [1, 0]])


def _rho_log_densities(rates_paper: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Log-densities with f_j(z_n) = C_{yhat_n j}, so that w_nj = rho_{yhat_n j} for every instance."""
    return np.log(rates_paper[pred, :].T)


class InstanceTablesTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)
        # the paper's C_ij = P(Yhat = i | Y = j), columns summing to one
        self.rates_paper = np.array([[0.8, 0.1, 0.2], [0.15, 0.7, 0.1], [0.05, 0.2, 0.7]])
        self.pred = np.repeat([0, 1, 2], [12, 5, 8])
        self.prevalences = self.rng.dirichlet(np.ones(3), size=7)

    def test_margins_and_normalisation(self):
        log_f = self.rng.normal(size=(3, len(self.pred)))
        tables, confidences = instance_tables(self.prevalences, log_f, self.pred, self.rng)
        shares = np.bincount(self.pred, minlength=3) / len(self.pred)
        self.assertEqual(tables.shape, (7, 3, 3))
        # cap-ml layout: predicted classes on columns
        np.testing.assert_allclose(tables.sum(axis=1), np.broadcast_to(shares, (7, 3)))
        np.testing.assert_allclose(tables.sum(axis=(1, 2)), 1.0)
        weights = instance_weights(self.prevalences, log_f)
        np.testing.assert_allclose(confidences, weights[:, np.arange(len(self.pred)), self.pred])

    def test_with_cell_weights_the_tables_follow_the_cell_pushforward(self):
        # with w_nj = rho_{yhat_n j}, the counts of each predicted class are Mult(n^t_i, rho_i.) (§T.15), as in
        # realised_tables: same mean and variance of every cell, by Monte Carlo
        n_draws = 20000
        prevalences = np.broadcast_to(self.prevalences[0], (n_draws, 3))
        tables, _ = instance_tables(prevalences, _rho_log_densities(self.rates_paper, self.pred), self.pred, self.rng)
        pred_counts = np.bincount(self.pred, minlength=3)
        population = prevalences[:, :, None] * self.rates_paper.T[None, :, :]
        reference = realised_tables(population, pred_counts, np.random.default_rng(6))
        np.testing.assert_allclose(tables.mean(axis=0), reference.mean(axis=0), atol=4e-3)
        np.testing.assert_allclose(tables.var(axis=0), reference.var(axis=0), rtol=0.08, atol=2e-5)

    def test_determinism_and_chunks(self):
        log_f = self.rng.normal(size=(3, len(self.pred)))
        first, _ = instance_tables(self.prevalences, log_f, self.pred, np.random.default_rng(1))
        again, _ = instance_tables(self.prevalences, log_f, self.pred, np.random.default_rng(1))
        chunked, _ = instance_tables(self.prevalences, log_f, self.pred, np.random.default_rng(1), chunk_entries=1)
        np.testing.assert_array_equal(first, again)
        np.testing.assert_array_equal(first, chunked)

    def test_shape_check(self):
        with self.assertRaises(ValueError):
            instance_tables(self.prevalences, np.zeros((3, 4)), self.pred, self.rng)


class InstanceMomentsTest(unittest.TestCase):
    def test_identity_of_the_bag_terms(self):
        # sigma_w^2 = sigma_g_tilde^2 - within-class term on every draw (§T.15)
        rng = np.random.default_rng(2)
        pred = rng.integers(0, 4, size=40)
        pred[:4] = [0, 1, 2, 3]
        confidences = rng.uniform(size=(6, 40))
        moments = instance_moments(confidences, pred, 5)
        np.testing.assert_allclose(moments["sigma_w2"], moments["sigma_g_tilde2"] - moments["within"], atol=1e-15)
        self.assertTrue(np.all(moments["sigma_w2"] <= moments["sigma_g_tilde2"] + 1e-15))
        np.testing.assert_allclose(moments["m_w"], confidences.mean(axis=1))
        # equal confidences within each predicted class: no within-class term
        flat = np.tile(np.array([0.9, 0.6, 0.7, 0.8])[pred], (2, 1))
        np.testing.assert_allclose(instance_moments(flat, pred, 5)["within"], 0.0, atol=1e-18)

    def test_accuracy_moments_unchanged_by_the_precisions_helper(self):
        rng = np.random.default_rng(4)
        prevalences = rng.dirichlet(np.ones(3), size=50)
        rates = np.array([[0.8, 0.15, 0.05], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
        counts = np.array([10, 0, 5])
        precision = class_precisions(prevalences, rates)
        sigma_g = np.sqrt(np.mean((precision * (1 - precision)) @ counts / counts.sum() ** 2))
        self.assertEqual(accuracy_moments(prevalences, rates, counts)[1], sigma_g)


class BayCAPKDEyInstanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = _fitted()
        cls.tables = cls.method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        cls.summaries = cls.method.last_posterior_summaries()

    def test_estimand_checks(self):
        with self.assertRaises(ValueError):
            _method(input_classifier="mlp")
        with self.assertRaises(ValueError):
            _method(c_prior="hierarchical-f")
        with self.assertRaises(ValueError):
            _method(bandwidth="cal", temperature="cal")
        with self.assertRaises(ValueError):
            BayCAPMAPLS(vanilla_acc, input_classifier="h", estimand=INSTANCE)

    def test_tables(self):
        n_u = len(DATA.u_posteriors)
        shares = np.bincount(np.argmax(DATA.u_posteriors, axis=1), minlength=3) / n_u
        self.assertEqual(self.tables.shape, (SETTINGS["num_samples"], 3, 3))
        np.testing.assert_allclose(self.tables.sum(axis=1), np.broadcast_to(shares, (SETTINGS["num_samples"], 3)))
        # counts of instances divided by N^Q
        np.testing.assert_allclose(self.tables * n_u, np.round(self.tables * n_u), atol=1e-9)

    def test_same_prevalence_draws_as_the_stored_estimands(self):
        # P19: the population, cell and per-instance methods of h share their draws of pi for the same seeds
        draws = {}
        for estimand in ("population", "realised", INSTANCE):
            method = _fitted(estimand=estimand)
            # the draws predict_ct_range pushes forward (QuantifierBayCAP.predict_ct_range)
            draws[estimand] = method.sample_prevalences(method.outputs(DATA.X_u, DATA.u_posteriors))
        np.testing.assert_array_equal(draws["population"], draws[INSTANCE])
        np.testing.assert_array_equal(draws["realised"], draws[INSTANCE])

    def test_log_densities_reused_from_the_quantifier(self):
        expected = self.method.quantifier_.kde_.log_densities(DATA.u_posteriors)[self.method.quantifier_.class_order_]
        np.testing.assert_allclose(self.method.quantifier_.last_bag_log_densities(), expected)

    def test_weights_do_not_depend_on_the_temperature(self):
        prevalences = np.random.default_rng(0).dirichlet(np.ones(3), size=SETTINGS["num_samples"])
        out = []
        for temperature in (1.0, 3.0):
            method = _fitted(temperature=temperature)
            with patch.object(type(method.quantifier_), "sample_prevalences", autospec=True) as sample:

                def fixed(quantifier, outputs):
                    quantifier._model(outputs)
                    quantifier._last_nuts = None
                    return prevalences

                sample.side_effect = fixed
                out.append(method.predict_ct_range(DATA.X_u, DATA.u_posteriors))
        np.testing.assert_array_equal(out[0], out[1])

    def test_determinism(self):
        again = _fitted().predict_ct_range(DATA.X_u, DATA.u_posteriors)
        np.testing.assert_array_equal(self.tables, again)

    def test_summaries(self):
        self.assertEqual(set(self.summaries), set(FIXED_SUMMARIES) | set(INSTANCE_SUMMARIES))
        reference = _fitted(estimand="realised")
        reference.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        self.assertEqual(self.summaries["sigma_g"], reference.last_posterior_summaries()["sigma_g"])
        self.assertLessEqual(self.summaries["sigma_w"], self.summaries["sigma_g_tilde"] + 1e-12)
        sigma_w2, tilde2 = self.summaries["sigma_w"] ** 2, self.summaries["sigma_g_tilde"] ** 2
        self.assertAlmostEqual(sigma_w2, tilde2 - self.summaries["within_class"], places=12)
        # the mean realised accuracy of the tables is close to the mean of the centres m_w
        accs = np.trace(self.tables, axis1=1, axis2=2)
        n_u = len(DATA.u_posteriors)
        self.assertLess(
            abs(accs.std() - np.sqrt(sigma_w2 + self.summaries["m_w_sd"] ** 2)), 0.35 * accs.std() + 1 / n_u
        )

    def test_m_sd_is_that_of_the_cell_centres(self):
        prevalences = np.random.default_rng(1).dirichlet(np.ones(3), size=SETTINGS["num_samples"])
        method = _fitted()
        pred_counts = np.bincount(np.argmax(DATA.u_posteriors, axis=1), minlength=3)
        method._last_summaries = {}
        method.quantifier_._model(DATA.u_posteriors)
        method._pushforward(prevalences, DATA.u_posteriors, pred_counts)
        centres = class_precisions(prevalences, method.rate_matrix_) @ (pred_counts / pred_counts.sum())
        self.assertAlmostEqual(method._last_summaries["m_sd"], centres.std(), places=12)


class FactoryTest(unittest.TestCase):
    def test_configuration_of_the_factory_entry(self):
        self.assertEqual(experiment_methods.INSTANCE_VARIANTS, {INST: "BayCAP-KDEy-h-real"})
        self.assertIs(experiment_methods.METHOD_CLASS[INST], BayCAPKDEy)
        self.assertIn(INST, experiment_methods.BAYES_POOL_METHODS)
        method = experiment_methods.instance_method(vanilla_acc, INST)
        reference = experiment_methods.fixed_c_method(vanilla_acc, "BayCAP-KDEy-h-real")
        self.assertEqual(method.estimand, INSTANCE)
        self.assertEqual(reference.estimand, "realised")
        settings = ("input_classifier", "c_prior", "kernel", "bandwidth", "shrinkage", "num_warmup", "num_samples")
        for name in (*settings, "temperature", "prior", "random_state"):
            self.assertEqual(getattr(method, name), getattr(reference, name), name)
        names = [name for name, _ in experiment_methods.gen_methods()]
        self.assertIn(INST, names)
        self.assertEqual(experiment_methods.get_method_names().index(INST), names.index("BayCAP-KDEy-h-real") + 1)

    def test_main_tables_and_shift_profile(self):
        # since 2026-11-26 the BayCAP-KDEy of the main tables and of the shift profile is
        # BayCAP-KDEy-h-inst-cb-tcal-keep (tests/test_tcal.py)
        self.assertNotIn(INST, main_tables.MAIN_METHODS)
        self.assertNotIn("BayCAP-KDEy-mlp-real", main_tables.MAIN_METHODS)
        self.assertNotIn(INST, shift_profile.DEFAULT_METHODS)
        self.assertNotIn("BayCAP-KDEy-mlp-real", shift_profile.DEFAULT_METHODS)
        self.assertEqual(METHOD_LATEX_LABELS[INST], "\\bcapkdey{}-h-inst")
        self.assertEqual(METHOD_LATEX_LABELS["BayCAP-KDEy-mlp-real"], "\\bcap{}-KDEy-mlp-real")
        self.assertEqual(method_plot_label(INST), "BayCAP-KDEy-h-inst")


if __name__ == "__main__":
    unittest.main()
