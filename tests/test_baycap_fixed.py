import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy.functional as F
from cap.error import vanilla_acc
from jax.scipy.special import logsumexp as jax_logsumexp
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from quapy.data import LabelledCollection
from quapy.method import _bayesian
from scipy.special import logsumexp
from scipy.stats import dirichlet, norm
from sklearn.linear_model import LogisticRegression

from uncawe.methods.baycap_fixed import (
    FIXED_SUMMARIES,
    NUTS_DIAGNOSTICS,
    BayCAPFixed,
    accuracy_moments,
    emq_model,
    ilr,
    ilr_inverse,
    kdey_model,
    top_label_ece,
)

# QuaPy's Bayesian module disables itself when pystan is missing (it only needs pystan for other methods); the test
# restores the modules its two quantifiers use, so that their draws can be compared with those of BayCAPFixed
QUAPY_BAYES_MODULES = {
    "DEPENDENCIES_INSTALLED": True,
    "jax": jax,
    "jnp": jnp,
    "jrandom": jax.random,
    "jax_logsumexp": jax_logsumexp,
    "numpyro": numpyro,
    "dist": dist,
    "MCMC": MCMC,
    "NUTS": NUTS,
}

SETTINGS = {"num_warmup": 30, "num_samples": 40, "random_state": 3}


def _blobs(rng, n_per_class, centres):
    X = np.concatenate([rng.normal(c, 1.0, size=(n, 2)) for c, n in zip(centres, n_per_class)])
    y = np.concatenate([np.full(n, k) for k, n in enumerate(n_per_class)])
    return X, y


class _Data:
    """Three Gaussian classes; h is a logistic regression trained on L, separate from V."""

    def __init__(self):
        rng = np.random.default_rng(0)
        centres = [(0.0, 0.0), (2.5, 0.0), (0.0, 2.5)]
        X_l, y_l = _blobs(rng, (40, 40, 40), centres)
        X_v, y_v = _blobs(rng, (30, 20, 25), centres)
        X_u, _ = _blobs(rng, (50, 10, 20), centres)
        self.h = LogisticRegression().fit(X_l, y_l)
        self.val = LabelledCollection(X_v, y_v, classes=np.array([0, 1, 2]))
        self.val_posteriors = self.h.predict_proba(X_v)
        self.X_u = X_u
        self.u_posteriors = self.h.predict_proba(X_u)


DATA = _Data()


def _fitted(**kwargs):
    method = BayCAPFixed(vanilla_acc, **(SETTINGS | kwargs))
    return method.fit(DATA.val, DATA.val_posteriors)


class HelpersTest(unittest.TestCase):
    def test_top_label_ece_by_hand(self):
        outputs = np.array([[0.95, 0.05], [0.95, 0.05], [0.6, 0.4], [0.3, 0.7]])
        labels = np.array([0, 1, 0, 1])
        # bins of width 1/15: {0.95, 0.95} with accuracy 1/2, {0.6} correct, {0.7} correct
        expected = 0.5 * abs(0.5 - 0.95) + 0.25 * abs(1 - 0.6) + 0.25 * abs(1 - 0.7)
        self.assertAlmostEqual(top_label_ece(outputs, labels, np.array([0, 1])), expected)

    def test_accuracy_moments_by_hand(self):
        rates = np.array([[0.8, 0.2], [0.4, 0.6]])  # cap-ml layout: row j is true class j
        prevalences = np.array([[0.5, 0.5], [0.25, 0.75]])
        pred_counts = np.array([30, 70])
        acc = prevalences @ np.diag(rates)
        rho = []
        for pi in prevalences:
            q = pi @ rates
            rho.append(np.diag(rates) * pi / q)
        sigma_g2 = [np.sum(pred_counts * r * (1 - r)) / 100**2 for r in rho]
        acc_sd, sigma_g = accuracy_moments(prevalences, rates, pred_counts)
        self.assertAlmostEqual(acc_sd, np.std(acc))
        self.assertAlmostEqual(sigma_g, np.sqrt(np.mean(sigma_g2)))

    def test_accuracy_moments_impossible_prediction_uses_uniform_precision(self):
        rates = np.eye(2)
        acc_sd, sigma_g = accuracy_moments(np.array([[1.0, 0.0]]), rates, np.array([5, 5]))
        self.assertEqual(acc_sd, 0.0)
        # class 1 is predicted but impossible under the draw: rho_11 = 1/2
        self.assertAlmostEqual(sigma_g, np.sqrt(5 * 0.25 / 100))

    def test_ilr_matches_quapy(self):
        prev = np.array([0.2, 0.5, 0.3])
        np.testing.assert_allclose(ilr(prev), F.ILRtransformation()(prev), rtol=1e-6)
        np.testing.assert_allclose(ilr_inverse(ilr(prev)), prev, rtol=1e-5)

    def test_kdey_model_log_density(self):
        log_dens = np.log(np.array([[0.5, 2.0, 1.0], [1.5, 0.2, 0.7]]))
        prev = np.array([0.3, 0.7])
        alpha = np.array([1.0, 2.0])
        value, _ = log_density(
            kdey_model,
            (),
            {"test_log_densities": log_dens, "alpha": alpha, "temperature": 2.0},
            {"prev": jnp.asarray(prev)},
        )
        expected = dirichlet.logpdf(prev, alpha) + 0.5 * np.sum(logsumexp(np.log(prev)[:, None] + log_dens, axis=0))
        self.assertAlmostEqual(float(value), expected, places=4)

    def test_emq_model_log_density(self):
        outputs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
        train_prev = np.array([0.5, 0.25, 0.25])
        z = np.array([0.3, -0.4])
        prev = np.asarray(ilr_inverse(z))
        kwargs = {"test_posteriors": outputs, "alpha": np.ones(3), "train_prev": train_prev, "temperature": 1.0}
        value, _ = log_density(emq_model, (), kwargs, {"z": jnp.asarray(z)})
        likelihood = np.sum(logsumexp(np.log(outputs) + np.log(prev) - np.log(train_prev), axis=-1))
        expected = norm.logpdf(z).sum() + dirichlet.logpdf(prev, np.ones(3)) + likelihood
        self.assertAlmostEqual(float(value), expected, places=4)


class BayCAPFixedTest(unittest.TestCase):
    def test_rejects_unknown_options(self):
        for kwargs in ({"quantifier": "pacc"}, {"input_classifier": "svm"}, {"estimand": "both"}, {"prior": "map"}):
            with self.assertRaises(ValueError):
                BayCAPFixed(vanilla_acc, **kwargs)

    def test_rate_matrix_is_measured_on_validation(self):
        method = _fitted(input_classifier="h")
        yhat = DATA.val_posteriors.argmax(axis=1)
        for j in range(3):
            in_class = DATA.val.y == j
            np.testing.assert_allclose(
                method.rate_matrix_[j], np.bincount(yhat[in_class], minlength=3) / in_class.sum()
            )

    @patch.multiple(_bayesian, **QUAPY_BAYES_MODULES)
    def test_draws_equal_quapy_sample_from_posterior(self):
        for quantifier in ("kdey", "emq"):
            with self.subTest(quantifier=quantifier):
                method = _fitted(quantifier=quantifier, input_classifier="h")
                common = {
                    "fit_classifier": False,
                    "val_split": None,
                    "num_warmup": SETTINGS["num_warmup"],
                    "num_samples": SETTINGS["num_samples"],
                    "mcmc_seed": SETTINGS["random_state"],
                }
                classifier = LogisticRegression().fit(DATA.val.X, DATA.val.y)  # only carries the classes
                if quantifier == "kdey":
                    q = _bayesian.BayesianKDEy(classifier, kernel="gaussian", bandwidth=0.1, **common)
                else:
                    q = _bayesian.BayesianMAPLS(classifier, **common)
                q.aggregation_fit(DATA.val_posteriors, DATA.val.y)
                expected = np.asarray(q.sample_from_posterior(DATA.u_posteriors), dtype=float)
                np.testing.assert_allclose(method.sample_prevalences(DATA.u_posteriors), expected, rtol=1e-6, atol=1e-7)

    def test_h_input_reads_the_given_outputs(self):
        for quantifier in ("kdey", "emq"):
            with self.subTest(quantifier=quantifier):
                method = _fitted(quantifier=quantifier, input_classifier="h")
                self.assertIsNone(method.mlp_)
                np.testing.assert_array_equal(method.val_outputs_, DATA.val_posteriors)
                seen = []
                original = method.sample_prevalences

                def spy(outputs, seen=seen, original=original):
                    seen.append(outputs)
                    return original(outputs)

                method.sample_prevalences = spy
                method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
                np.testing.assert_array_equal(seen[0], DATA.u_posteriors)

    def test_emq_source_prevalence_is_that_of_validation(self):
        for input_classifier in ("h", "mlp"):
            method = _fitted(quantifier="emq", input_classifier=input_classifier)
            np.testing.assert_allclose(method.train_prevalence_, DATA.val.prevalence())

    def test_mlp_is_seeded(self):
        first = _fitted(input_classifier="mlp")
        second = _fitted(input_classifier="mlp")
        self.assertEqual(first.mlp_.random_state, SETTINGS["random_state"])
        np.testing.assert_array_equal(first.outputs(DATA.X_u, None), second.outputs(DATA.X_u, None))
        self.assertEqual(first.val_ece_, second.val_ece_)

    def test_estimands_share_prevalence_draws(self):
        for quantifier in ("kdey", "emq"):
            with self.subTest(quantifier=quantifier):
                population = _fitted(quantifier=quantifier, input_classifier="h", estimand="population")
                realised = _fitted(quantifier=quantifier, input_classifier="h", estimand="realised")
                pop_cts = population.predict_ct_range(DATA.X_u, DATA.u_posteriors)
                realised.predict_ct_range(DATA.X_u, DATA.u_posteriors)
                np.testing.assert_allclose(population._last_nuts[0]["prev"], realised._last_nuts[0]["prev"])
                np.testing.assert_allclose(pop_cts.sum(axis=2), population._last_nuts[0]["prev"][0])

    def test_population_accuracy_closed_form(self):
        method = _fitted(input_classifier="h")
        cts = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        accuracies = np.trace(cts, axis1=1, axis2=2)
        prev = method._last_nuts[0]["prev"][0]
        d = np.diag(method.rate_matrix_)
        np.testing.assert_allclose(accuracies.mean(), d @ prev.mean(axis=0), rtol=1e-5)
        centred = d - d.mean()
        np.testing.assert_allclose(
            accuracies.var(), centred @ np.cov(prev, rowvar=False, bias=True) @ centred, rtol=1e-4
        )
        self.assertAlmostEqual(method.last_posterior_summaries()["acc_sd"], accuracies.std(), places=6)

    def test_realised_tables_keep_the_predicted_margins(self):
        method = _fitted(input_classifier="h", estimand="realised")
        cts = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        pred_counts = np.bincount(DATA.u_posteriors.argmax(axis=1), minlength=3)
        np.testing.assert_allclose(cts.sum(axis=1), np.broadcast_to(pred_counts / pred_counts.sum(), (len(cts), 3)))

    def test_realised_tables_with_identity_rates_return_the_observed_counts(self):
        method = _fitted(input_classifier="h", estimand="realised")
        method.rate_matrix_ = np.eye(3)
        cts = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        pred_counts = np.bincount(DATA.u_posteriors.argmax(axis=1), minlength=3)
        np.testing.assert_allclose(cts, np.broadcast_to(np.diag(pred_counts / pred_counts.sum()), cts.shape))

    def test_diagnostics_and_summaries(self):
        method = _fitted(quantifier="emq", input_classifier="h", estimand="realised")
        with self.assertRaises(RuntimeError):
            method.last_nuts_diagnostics()
        method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        diagnostics = method.last_nuts_diagnostics()
        self.assertEqual(set(diagnostics), set(NUTS_DIAGNOSTICS))
        self.assertGreater(diagnostics["nuts_min_ess"], 0)
        summaries = method.last_posterior_summaries()
        self.assertEqual(set(summaries), set(FIXED_SUMMARIES))
        self.assertEqual(summaries["prev_mean"].shape, (3,))
        self.assertAlmostEqual(summaries["prev_mean"].sum(), 1.0)
        self.assertGreater(summaries["sigma_g"], 0)
        self.assertAlmostEqual(
            summaries["val_ece"], top_label_ece(DATA.val_posteriors, DATA.val.y, np.array([0, 1, 2]))
        )

    def test_confidence_api_applies_the_metric_to_each_table(self):
        method = _fitted(input_classifier="h")
        interval = method.predict_with_confidence(DATA.X_u, DATA.u_posteriors)
        self.assertEqual(len(interval.samples), SETTINGS["num_samples"])
        self.assertTrue(np.all((interval.samples >= 0) & (interval.samples <= 1)))


if __name__ == "__main__":
    unittest.main()
