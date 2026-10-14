import unittest
from unittest.mock import patch

import numpy as np
from cap.error import vanilla_acc
from quapy.data import LabelledCollection
from sklearn.neural_network import MLPClassifier

from uncawe.methods import BayCAPKDEy


class _PosteriorQuantifier:
    def __init__(self, samples, classes=(0, 1)):
        self.samples = np.asarray(samples, dtype=float)
        self.classes_ = np.asarray(classes)
        self.fit_args = None
        self.classified = None
        self.sampled_from = None

    def fit(self, X, y):
        self.fit_args = (X, y)
        return self

    def classify(self, X):
        self.classified = X
        return np.full((len(X), 2), 0.5)

    def sample_from_posterior(self, predictions):
        self.sampled_from = predictions
        return self.samples


class BayCAPKDEyTest(unittest.TestCase):
    def setUp(self):
        self.validation = LabelledCollection(
            instances=np.arange(8).reshape(4, 2),
            labels=np.array([0, 0, 1, 1]),
            classes=np.array([0, 1]),
        )
        self.target_posteriors = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.4, 0.6],
            ]
        )
        self.samples = np.array([[0.75, 0.25], [0.2, 0.8]])

    def make_method(self, acc_fn, quantifier, **kwargs):
        with patch("uncawe.methods.baycap_kdey.BayesianKDEy", return_value=quantifier):
            return BayCAPKDEy(acc_fn, **kwargs)

    def test_constructor_configures_internal_bayesian_kdey(self):
        quantifier = _PosteriorQuantifier(self.samples)
        with patch("uncawe.methods.baycap_kdey.BayesianKDEy", return_value=quantifier) as constructor:
            method = BayCAPKDEy(
                vanilla_acc,
                kernel="aitchison",
                bandwidth="scott",
                shrinkage=0.1,
                num_warmup=20,
                num_samples=30,
                mcmc_seed=7,
            )

        classifier = constructor.call_args.args[0]
        self.assertIsInstance(classifier, MLPClassifier)
        self.assertEqual(
            constructor.call_args.kwargs,
            {
                "kernel": "aitchison",
                "bandwidth": "scott",
                "shrinkage": 0.1,
                "num_warmup": 20,
                "num_samples": 30,
                "mcmc_seed": 7,
            },
        )
        self.assertIs(method.q, quantifier)

    def test_fixed_rate_matrix_pushforward_matches_cap_table_orientation(self):
        quantifier = _PosteriorQuantifier(self.samples)
        method = self.make_method(vanilla_acc, quantifier).fit(self.validation, self.target_posteriors)

        expected_rate_matrix = np.array([[0.5, 0.5], [0.0, 1.0]])
        np.testing.assert_allclose(method.rate_matrix_, expected_rate_matrix)

        cts = method.predict_ct_range(np.ones((3, 2)), np.zeros((3, 2)))
        expected_cts = np.array(
            [
                [[0.375, 0.375], [0.0, 0.25]],
                [[0.1, 0.1], [0.0, 0.8]],
            ]
        )
        np.testing.assert_allclose(cts, expected_cts)
        np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0)
        np.testing.assert_allclose(method.predict_ct(np.ones((3, 2)), np.zeros((3, 2))), expected_cts.mean(axis=0))

    def test_cap_confidence_api_propagates_nonlinear_metric_per_table(self):
        def squared_accuracy(ct):
            return np.trace(ct) ** 2

        method = self.make_method(squared_accuracy, _PosteriorQuantifier(self.samples)).fit(
            self.validation,
            self.target_posteriors,
        )

        interval = method.predict_with_confidence(np.ones((3, 2)), np.zeros((3, 2)))

        np.testing.assert_allclose(interval.samples, [0.625**2, 0.9**2])
        self.assertAlmostEqual(interval.point_estimate, (0.625**2 + 0.9**2) / 2)

    def test_posterior_class_order_is_aligned_with_rate_matrix(self):
        quantifier = _PosteriorQuantifier([[0.25, 0.75]], classes=(1, 0))
        method = self.make_method(vanilla_acc, quantifier).fit(self.validation, self.target_posteriors)

        ct = method.predict_ct_range(np.ones((2, 2)), np.zeros((2, 2)))[0]

        np.testing.assert_allclose(ct, [[0.375, 0.375], [0.0, 0.25]])

    def test_rejects_posterior_draws_outside_simplex(self):
        method = self.make_method(vanilla_acc, _PosteriorQuantifier([[0.8, 0.8]])).fit(
            self.validation,
            self.target_posteriors,
        )

        with self.assertRaisesRegex(ValueError, "probability simplex"):
            method.predict_ct_range(np.ones((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
