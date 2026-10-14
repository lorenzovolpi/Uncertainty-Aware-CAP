import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from cap.error import vanilla_acc
from cap.models.confidence import ACC_BayesCAP
from quapy.data import LabelledCollection

from uncawe.methods import BayCAP, realised_tables

NUTS_PARAMS = {"num_warmup": 50, "num_samples": 40}


def _one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


class RealisedTablesTest(unittest.TestCase):
    def setUp(self):
        # cap-ml layout: true classes on rows, predicted classes on columns
        self.cts = np.array(
            [
                [[0.3, 0.0], [0.2, 0.5]],
                [[0.1, 0.3], [0.1, 0.5]],
            ]
        )
        self.pred_counts = np.array([5, 15])

    def test_hand_computed_2x2_layout(self):
        # in draw 0, predicted class 1 has only true class 1 (column [0.0, 0.5]),
        # so its 15 instances must all land in the cell (true 1, predicted 1)
        ct = realised_tables(self.cts[:1], self.pred_counts, np.random.default_rng(0))[0]

        self.assertEqual(ct[0, 1], 0.0)
        self.assertAlmostEqual(ct[1, 1], 15 / 20)
        self.assertAlmostEqual(ct[:, 0].sum(), 5 / 20)
        np.testing.assert_allclose(ct * 20, np.round(ct * 20))

    def test_predicted_class_margins_equal_observed_counts(self):
        cts = realised_tables(self.cts, self.pred_counts, np.random.default_rng(0))

        self.assertEqual(cts.shape, self.cts.shape)
        np.testing.assert_allclose(cts.sum(axis=1), np.broadcast_to(self.pred_counts / 20, (2, 2)))
        np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0)

    def test_perfect_classifier_returns_observed_counts(self):
        pi = np.array([[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]])
        cts = pi[:, :, None] * np.eye(3)[None]
        pred_counts = np.array([3, 0, 7])

        realised = realised_tables(cts, pred_counts, np.random.default_rng(0))

        np.testing.assert_allclose(realised, np.broadcast_to(np.diag(pred_counts / 10), (2, 3, 3)))

    def test_mean_count_table_matches_expected(self):
        draw = self.cts[1]
        n_draws = 20_000
        cts = np.broadcast_to(draw, (n_draws, 2, 2))

        mean_counts = realised_tables(cts, self.pred_counts, np.random.default_rng(0)).mean(axis=0) * 20

        rho = draw / draw.sum(axis=0, keepdims=True)  # P(Y=y | Yhat=c) in column c
        expected = rho * self.pred_counts[None, :]
        np.testing.assert_allclose(mean_counts, expected, atol=0.05)

    def test_fixed_seed_is_reproducible(self):
        first = realised_tables(self.cts, self.pred_counts, np.random.default_rng(3))
        second = realised_tables(self.cts, self.pred_counts, np.random.default_rng(3))

        np.testing.assert_array_equal(first, second)

    def test_rejects_invalid_inputs(self):
        rng = np.random.default_rng(0)
        with self.assertRaisesRegex(ValueError, "pred_counts must have shape"):
            realised_tables(self.cts, np.array([1, 2, 3]), rng)
        with self.assertRaisesRegex(ValueError, "not all zero"):
            realised_tables(self.cts, np.array([0, 0]), rng)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            realised_tables(-self.cts, self.pred_counts, rng)


class BayCAPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n_classes = 3
        val_y = np.repeat(np.arange(n_classes), 30)
        val_yhat = np.where(rng.random(len(val_y)) < 0.8, val_y, rng.integers(0, n_classes, len(val_y)))
        cls.validation = LabelledCollection(
            instances=np.zeros((len(val_y), 1)),
            labels=val_y,
            classes=np.arange(n_classes),
        )
        cls.val_posteriors = _one_hot(val_yhat, n_classes)
        cls.test_posteriors = _one_hot(rng.integers(0, n_classes, 50), n_classes)
        cls.pred_counts = np.bincount(cls.test_posteriors.argmax(axis=1), minlength=n_classes)
        cls.X = np.zeros((50, 1))

    def fit(self, method):
        return method.fit(self.validation, self.val_posteriors)

    def test_population_reproduces_acc_bayescap(self):
        for prior in ("uniform", 2.0, np.array([1.0, 2.0, 3.0])):
            with self.subTest(prior=prior):
                ours = self.fit(BayCAP(vanilla_acc, prior=prior, random_state=5, **NUTS_PARAMS))
                ref = self.fit(ACC_BayesCAP(vanilla_acc, prior=prior, random_state=5, **NUTS_PARAMS))

                np.testing.assert_array_equal(
                    ours.predict_ct_range(self.X, self.test_posteriors),
                    ref.predict_ct_range(self.X, self.test_posteriors),
                )

    def test_estimands_share_posterior_draws(self):
        population = self.fit(BayCAP(vanilla_acc, estimand="population", random_state=5, **NUTS_PARAMS))
        realised = self.fit(BayCAP(vanilla_acc, estimand="realised", random_state=5, **NUTS_PARAMS))

        pop_cts = population.predict_ct_range(self.X, self.test_posteriors)
        real_cts = realised.predict_ct_range(self.X, self.test_posteriors)

        expected = realised_tables(pop_cts, self.pred_counts, np.random.default_rng(5))
        np.testing.assert_array_equal(real_cts, expected)
        np.testing.assert_allclose(real_cts.sum(axis=1), np.broadcast_to(self.pred_counts / 50, (40, 3)))

    def test_realised_is_reproducible_and_feeds_confidence_api(self):
        method = self.fit(BayCAP(vanilla_acc, estimand="realised", random_state=5, **NUTS_PARAMS))

        first = method.predict_ct_range(self.X, self.test_posteriors)
        second = method.predict_ct_range(self.X, self.test_posteriors)
        np.testing.assert_array_equal(first, second)

        interval = method.predict_with_confidence(self.X, self.test_posteriors)
        np.testing.assert_allclose(interval.samples, np.trace(first, axis1=1, axis2=2))

    def test_rejects_unknown_estimand(self):
        with self.assertRaisesRegex(ValueError, "estimand must be one of"):
            BayCAP(vanilla_acc, estimand="realized")


if __name__ == "__main__":
    unittest.main()
