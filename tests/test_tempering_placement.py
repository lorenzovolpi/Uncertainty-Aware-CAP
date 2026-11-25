"""Tests of the placement of the temperature with the realised step (work-plan card WP-35): ``tempering`` of
``BayCAP``, ``BayesKDEy`` and ``BayCAPKDEy``, and the factory entries."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import itertools
import unittest

import jax.numpy as jnp
import numpy as np
import quapy as qp
from cap.error import vanilla_acc
from cap.models.confidence import P_C_COND_Y, P_TEST_Y
from numpyro.infer.util import log_density
from quapy.data import LabelledCollection
from scipy.stats import multinomial
from test_baycap_fixed import DATA, SETTINGS

from uncawe.experiments import methods as experiment_methods
from uncawe.methods.baycap import TEMPERINGS, BayCAP, tempered_class_tables
from uncawe.methods.baycap_kdey import INSTANCE, BayCAPKDEy, instance_weights
from uncawe.methods.bayes_kdey import BayesKDEy, kdey_model
from uncawe.methods.bayesian_quantifier import PREV

# the experiment seed, which BayCAP reads when random_state is None
qp.environ.setdefault("_R_SEED", 0)
TAU = 2.5
NUTS = {"num_warmup": 30, "num_samples": 40, "random_state": 3}


def _brute_force(log_terms: np.ndarray, log_prev: np.ndarray, tau: float, tempering: str) -> float:
    """log of the sum over every assignment of the latent classes of the product over the instances of the tempered
    complete-data likelihood: ``log_terms[n, j]`` = log of the term of the observation of instance n given class j."""
    n_instances, n_classes = log_terms.shape
    total = []
    for classes in itertools.product(range(n_classes), repeat=n_instances):
        value = 0.0
        for n, j in enumerate(classes):
            if tempering == "complete":
                value += (log_prev[j] + log_terms[n, j]) / tau
            else:
                value += log_prev[j] + log_terms[n, j] / tau
        total.append(value)
    return float(np.logaddexp.reduce(total))


def _one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


class BayCAPModelTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.pi = rng.dirichlet(np.ones(3))
        self.rates = rng.dirichlet(np.ones(3), size=3)  # cap-ml layout: rates[y, c] = C_cy
        self.q = np.array([2, 1, 1])
        self.val_counts = np.array([[8, 1, 1], [2, 7, 1], [1, 1, 8]])

    def _log_density(self, tempering: str, temperature: float) -> float:
        method = BayCAP(vanilla_acc, estimand="realised", temperature=temperature, tempering=tempering)
        kwargs = {
            "pred_posterior_count": self.q,
            "train_class_cond_count": self.val_counts,
            "alpha": np.ones(3),
            "c_alpha": np.ones((3, 3)),
        }
        params = {P_TEST_Y: jnp.asarray(self.pi), P_C_COND_Y: jnp.asarray(self.rates)}
        return float(log_density(method.model, (), kwargs, params)[0])

    def test_the_term_of_the_bag_is_the_sum_over_the_latent_classes(self):
        # the instances of the bag by prediction: one row per instance, with log C_{yhat j} for each class j
        pred = np.repeat(np.arange(3), self.q)
        log_terms = np.log(self.rates.T[pred])
        marginal = multinomial.logpmf(self.q, self.q.sum(), self.rates.T @ self.pi) / TAU
        reference = self._log_density("marginal", TAU)
        for tempering in ("complete", "observation"):
            with self.subTest(tempering=tempering):
                expected = _brute_force(log_terms, np.log(self.pi), TAU, tempering) - marginal
                self.assertAlmostEqual(self._log_density(tempering, TAU) - reference, expected, places=4)

    def test_at_tau_one_every_placement_is_the_untempered_model(self):
        reference = self._log_density("marginal", 1.0)
        for tempering in TEMPERINGS:
            self.assertEqual(self._log_density(tempering, 1.0), reference)

    def test_checks(self):
        with self.assertRaises(ValueError):
            BayCAP(vanilla_acc, estimand="realised", tempering="joint")
        with self.assertRaises(ValueError):
            BayCAP(vanilla_acc, estimand="population", tempering="complete")
        self.assertEqual(BayCAP(vanilla_acc).tempering, "marginal")


class TemperedClassTablesTest(unittest.TestCase):
    def test_closed_forms(self):
        rng = np.random.default_rng(1)
        pi = rng.dirichlet(np.ones(3), size=4)
        rates = rng.dirichlet(np.ones(3), size=(4, 3))
        plain = pi[:, :, None] * rates
        np.testing.assert_allclose(tempered_class_tables(pi, rates, TAU, "marginal"), plain)
        np.testing.assert_allclose(tempered_class_tables(pi, rates, TAU, "complete"), plain ** (1 / TAU))
        np.testing.assert_allclose(
            tempered_class_tables(pi, rates, TAU, "observation"), pi[:, :, None] * rates ** (1 / TAU)
        )
        for tempering in TEMPERINGS:
            np.testing.assert_allclose(tempered_class_tables(pi, rates, 1.0, tempering), plain)


class BayCAPDrawsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        val_y = np.repeat([0, 1], 20)
        val_yhat = np.where(np.arange(40) % 5 == 0, 1 - val_y, val_y)
        cls.validation = LabelledCollection(np.zeros((40, 1)), val_y, classes=np.arange(2))
        cls.val_posteriors = _one_hot(val_yhat, 2)
        cls.X = np.zeros((30, 1))
        cls.test_posteriors = _one_hot((np.arange(30) % 3 == 0).astype(int), 2)

    def _tables(self, **kwargs):
        method = BayCAP(vanilla_acc, estimand="realised", c_prior="hierarchical-f", **NUTS, **kwargs)
        method.fit(self.validation, self.val_posteriors)
        return method.predict_ct_range(self.X, self.test_posteriors)

    def test_tau_one_gives_the_draws_of_marginal(self):
        reference = self._tables()
        for tempering in ("complete", "observation"):
            np.testing.assert_array_equal(self._tables(tempering=tempering), reference)

    def test_tempered_realised_tables_keep_the_predicted_counts(self):
        shares = np.bincount(np.argmax(self.test_posteriors, axis=1), minlength=2) / len(self.test_posteriors)
        marginal = self._tables(temperature=TAU)
        for tempering in ("complete", "observation"):
            with self.subTest(tempering=tempering):
                tables = self._tables(temperature=TAU, tempering=tempering)
                self.assertTrue(np.all(np.isfinite(tables)))
                np.testing.assert_allclose(tables.sum(axis=1), np.broadcast_to(shares, (NUTS["num_samples"], 2)))
                self.assertFalse(np.array_equal(tables, marginal))


class KDEyModelTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.pi = rng.dirichlet(np.ones(3))
        self.log_f = rng.normal(size=(3, 4))  # [j, n]

    def _log_density(self, tempering: str, temperature: float) -> float:
        kwargs = {
            "test_log_densities": self.log_f,
            "alpha": np.ones(3),
            "temperature": temperature,
            "tempering": tempering,
        }
        return float(log_density(kdey_model, (), kwargs, {PREV: jnp.asarray(self.pi)})[0])

    def test_the_likelihood_is_the_sum_over_the_latent_classes(self):
        prior = self._log_density("marginal", 1e12)  # the likelihood vanishes: the log-density of the prior
        for tempering in ("complete", "observation"):
            with self.subTest(tempering=tempering):
                expected = _brute_force(self.log_f.T, np.log(self.pi), TAU, tempering)
                self.assertAlmostEqual(self._log_density(tempering, TAU) - prior, expected, places=4)

    def test_at_tau_one_every_placement_is_the_untempered_model(self):
        reference = self._log_density("marginal", 1.0)
        for tempering in TEMPERINGS:
            self.assertEqual(self._log_density(tempering, 1.0), reference)

    def test_checks(self):
        with self.assertRaises(ValueError):
            BayesKDEy(input_classifier="h", tempering="joint")


class InstanceWeightsTest(unittest.TestCase):
    def test_closed_forms(self):
        rng = np.random.default_rng(3)
        pi = rng.dirichlet(np.ones(3), size=5)
        log_f = rng.normal(size=(3, 4))
        f = np.exp(log_f)

        def normalised(terms):
            return terms / terms.sum(axis=2, keepdims=True)

        plain = normalised(pi[:, None, :] * f.T[None, :, :])
        complete = normalised((pi[:, None, :] * f.T[None, :, :]) ** (1 / TAU))
        observation = normalised(pi[:, None, :] * f.T[None, :, :] ** (1 / TAU))
        np.testing.assert_allclose(instance_weights(pi, log_f, TAU, "marginal"), plain)
        np.testing.assert_allclose(instance_weights(pi, log_f, TAU, "complete"), complete)
        np.testing.assert_allclose(instance_weights(pi, log_f, TAU, "observation"), observation)
        for tempering in TEMPERINGS:
            np.testing.assert_allclose(instance_weights(pi, log_f, 1.0, tempering), plain)
        with self.assertRaises(ValueError):
            instance_weights(pi, log_f, TAU, "joint")


def _kdey(**kwargs) -> BayCAPKDEy:
    settings = SETTINGS | {"input_classifier": "h", "estimand": INSTANCE} | kwargs
    return BayCAPKDEy(vanilla_acc, **settings).fit(DATA.val, DATA.val_posteriors)


class BayCAPKDEyTest(unittest.TestCase):
    def test_checks(self):
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, input_classifier="h", estimand="realised", tempering="complete")
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, input_classifier="h", estimand=INSTANCE, tempering="joint")
        method = BayCAPKDEy(vanilla_acc, input_classifier="h", estimand=INSTANCE, tempering="observation")
        self.assertEqual(method._make_quantifier().tempering, "observation")

    def test_tau_one_gives_the_draws_of_marginal(self):
        reference = _kdey().predict_ct_range(DATA.X_u, DATA.u_posteriors)
        for tempering in ("complete", "observation"):
            tables = _kdey(tempering=tempering).predict_ct_range(DATA.X_u, DATA.u_posteriors)
            np.testing.assert_array_equal(tables, reference)

    def test_tempered_tables_keep_the_predicted_counts(self):
        n_u = len(DATA.u_posteriors)
        shares = np.bincount(np.argmax(DATA.u_posteriors, axis=1), minlength=3) / n_u
        marginal = _kdey(temperature=TAU).predict_ct_range(DATA.X_u, DATA.u_posteriors)
        for tempering in ("complete", "observation"):
            with self.subTest(tempering=tempering):
                method = _kdey(temperature=TAU, tempering=tempering)
                self.assertEqual(method.quantifier_.tempering, tempering)
                tables = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
                np.testing.assert_allclose(tables.sum(axis=1), np.broadcast_to(shares, (SETTINGS["num_samples"], 3)))
                self.assertFalse(np.array_equal(tables, marginal))


class FactoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qp.environ.setdefault("_R_SEED", 0)
        cls.methods = dict(experiment_methods.gen_methods())

    def test_names(self):
        taus = experiment_methods.TEMPERATURE_LEVELS
        expected = {f"BayesCAP-real-t{t:g}" for t in taus} | {
            f"{base}-t{t:g}-{suffix}"
            for base in ("BayesCAP-real", "BayesCAP-Hf-real", "BayCAP-KDEy-h-inst-cb")
            for t in taus
            for suffix in ("complete", "obs")
        }
        self.assertEqual(set(experiment_methods.PLACEMENT_VARIANTS), expected)
        self.assertFalse(set(experiment_methods.PLACEMENT_VARIANTS) & set(experiment_methods.TEMPERED_VARIANTS))
        self.assertTrue(set(experiment_methods.PLACEMENT_VARIANTS) <= experiment_methods.BAYES_POOL_METHODS)

    def test_configuration(self):
        settings = {
            BayCAP: ("estimand", "c_prior", "num_warmup", "num_samples", "random_state", "prior", "shares"),
            BayCAPKDEy: (
                "input_classifier",
                "estimand",
                "c_prior",
                "kernel",
                "bandwidth",
                "bandwidth_grid",
                "num_warmup",
                "num_samples",
                "random_state",
                "prior",
            ),
        }
        for name, (base, tau, tempering) in experiment_methods.PLACEMENT_VARIANTS.items():
            with self.subTest(method=name):
                method, reference = self.methods[name], self.methods[base]
                self.assertIs(type(method), type(reference))
                self.assertIs(experiment_methods.METHOD_CLASS[name], type(reference))
                self.assertEqual((method.temperature, method.tempering), (tau, tempering))
                self.assertEqual(reference.tempering, "marginal")
                for attribute in settings[type(method)]:
                    self.assertEqual(getattr(method, attribute), getattr(reference, attribute), attribute)


if __name__ == "__main__":
    unittest.main()
