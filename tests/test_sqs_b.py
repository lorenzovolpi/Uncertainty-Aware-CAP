import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from cap.error import vanilla_acc
from cap.models.utils import smooth
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from test_baycap_fixed import DATA, SETTINGS

from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_quantifier import NUTS_DIAGNOSTICS, accuracy_moments
from uncawe.methods.bayes_kdey import BayesKDEy
from uncawe.methods.sqs_b import (
    SMOOTHING,
    SQSB,
    SQSB_SUMMARIES,
    bag_tables,
    forward_bag_variances,
    luck_spreads,
)

SAMPLE_SIZE = 200
N_CLASSES = 3


def _baycap_kdey(**kwargs) -> BayCAPKDEy:
    """The BayCAP method whose prevalence draws SQS-B shares: BayesKDEy with an MLP input and a fixed C."""
    return BayCAPKDEy(vanilla_acc, **({"input_classifier": "mlp", "c_prior": "fixed"} | SETTINGS | kwargs))


def _sqsb(random_state: int = SETTINGS["random_state"], val: LabelledCollection = DATA.val, **kwargs) -> SQSB:
    settings = SETTINGS | {"random_state": random_state}
    method = SQSB(vanilla_acc, input_classifier="mlp", sample_size=SAMPLE_SIZE, **settings, **kwargs)
    return method.fit(val, DATA.val_posteriors)


class HelpersTest(unittest.TestCase):
    def test_luck_spreads_example_of_t14(self):
        # §T.14: recalls (0.90, 0.70), pi = (0.5, 0.5); cap-ml layout, row j is the column C_{:j} of the paper
        rates = np.array([[0.9, 0.1], [0.3, 0.7]])
        s_rec, s_prec = luck_spreads(np.array([0.5, 0.5]), rates)
        self.assertAlmostEqual(s_rec, 0.010)
        self.assertAlmostEqual(s_prec, 0.00375)

    def test_total_variance_identities(self):
        rng = np.random.default_rng(0)
        rates = rng.dirichlet(np.ones(4), size=4) + 2 * np.eye(4)
        rates /= rates.sum(axis=1, keepdims=True)
        recalls = np.diag(rates)
        for prevalence in rng.dirichlet(np.ones(4), size=20):
            accuracy = prevalence @ recalls
            s_rec, s_prec = luck_spreads(prevalence, rates)
            n = 1000
            sigma_f2 = forward_bag_variances(prevalence[None, :], rates, n)[0]
            self.assertAlmostEqual(accuracy * (1 - accuracy), n * sigma_f2 + s_rec)
            predicted = prevalence @ rates
            precision = prevalence * recalls / predicted
            self.assertAlmostEqual(accuracy * (1 - accuracy), predicted @ (precision * (1 - precision)) + s_prec)

    def test_luck_spreads_ignore_a_class_never_predicted(self):
        rates = np.array([[1.0, 0.0], [1.0, 0.0]])  # h always predicts class 0
        s_rec, s_prec = luck_spreads(np.array([0.6, 0.4]), rates)
        self.assertAlmostEqual(s_rec, 0.6 * 0.4**2 + 0.4 * 0.6**2)
        self.assertAlmostEqual(s_prec, (0.6 - 0.6) ** 2)

    def test_bag_tables_equal_cap_ml_contingency_table(self):
        rng = np.random.default_rng(1)
        labels = rng.integers(0, N_CLASSES, size=(5, 50))
        predictions = rng.integers(0, N_CLASSES, size=(5, 50))
        expected = [contingency_table(y, p, N_CLASSES) for y, p in zip(labels, predictions)]
        np.testing.assert_allclose(bag_tables(labels, predictions, N_CLASSES), expected)


class SQSBTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = _sqsb()
        cls.cts = cls.method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        cls.indices = cls.method.last_bag_indices_
        cls.prevalences = cls.method.quantifier_._last_nuts[0]["prev"][0]

    def test_checks_its_hyperparameters_and_needs_a_fit(self):
        with self.assertRaises(ValueError):
            SQSB(vanilla_acc, input_classifier="svm", sample_size=SAMPLE_SIZE)
        with self.assertRaises(ValueError):
            SQSB(vanilla_acc, kernel="aitchison", bandwidth="loo", sample_size=SAMPLE_SIZE)
        with self.assertRaises(RuntimeError):
            unfitted = SQSB(vanilla_acc, sample_size=SAMPLE_SIZE, random_state=0)
            unfitted.predict_ct_range(DATA.X_u, DATA.u_posteriors)

    def test_quantifier_is_bayes_kdey_with_the_hyperparameters_of_the_method(self):
        quantifier = self.method.quantifier_
        self.assertIsInstance(quantifier, BayesKDEy)
        for attribute in ("input_classifier", "kernel", "bandwidth", "shrinkage", "num_samples", "random_state"):
            with self.subTest(attribute=attribute):
                self.assertEqual(getattr(quantifier, attribute), getattr(self.method, attribute))

    def test_draws_equal_those_of_baycap_kdey(self):
        baycap = _baycap_kdey(estimand="realised").fit(DATA.val, DATA.val_posteriors)
        baycap.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        np.testing.assert_array_equal(self.prevalences, baycap._last_nuts[0]["prev"][0])
        np.testing.assert_array_equal(self.method.rate_matrix_, baycap.rate_matrix_)
        self.assertEqual(self.method.num_samples, SETTINGS["num_samples"])

    def test_one_bag_of_sample_size_per_draw_with_the_counts_of_sampling_index(self):
        self.assertEqual(self.indices.shape, (SETTINGS["num_samples"], SAMPLE_SIZE))
        self.assertEqual(self.cts.shape, (SETTINGS["num_samples"], N_CLASSES, N_CLASSES))
        # the bags drawn sequentially from the NumPy state seeded by random_state, as RQBS
        val_post = LabelledCollection(DATA.val_posteriors, DATA.val.y, classes=np.arange(N_CLASSES))
        np.random.seed(self.method.random_state)
        expected = [
            val_post.sampling_index(SAMPLE_SIZE, *smooth(p.astype(np.float64), epsilon=SMOOTHING))
            for p in self.prevalences
        ]
        np.testing.assert_array_equal(self.indices, expected)
        counts = np.stack([np.bincount(DATA.val.y[idx], minlength=N_CLASSES) for idx in self.indices])
        self.assertTrue(np.all(np.abs(counts - SAMPLE_SIZE * self.prevalences) <= N_CLASSES))

    def test_tables_are_those_of_h_on_the_bags(self):
        predictions = DATA.val_posteriors.argmax(axis=1)
        expected = [contingency_table(DATA.val.y[idx], predictions[idx], N_CLASSES) for idx in self.indices]
        np.testing.assert_allclose(self.cts, expected)
        np.testing.assert_allclose(self.cts.sum(axis=(1, 2)), 1.0)
        # cap-ml layout: the row margins are the true class shares of the bags
        counts = np.stack([np.bincount(DATA.val.y[idx], minlength=N_CLASSES) for idx in self.indices])
        np.testing.assert_allclose(self.cts.sum(axis=2), counts / SAMPLE_SIZE)
        np.testing.assert_allclose(self.method.predict_ct(DATA.X_u, DATA.u_posteriors), self.cts.mean(axis=0))

    def test_same_seed_same_tables_and_the_global_state_is_restored(self):
        np.random.seed(123)
        before = np.random.random()
        np.random.seed(123)
        second = _sqsb().predict_ct_range(DATA.X_u, DATA.u_posteriors)
        self.assertEqual(np.random.random(), before)
        np.testing.assert_array_equal(second, self.cts)
        other = _sqsb(random_state=SETTINGS["random_state"] + 1).predict_ct_range(DATA.X_u, DATA.u_posteriors)
        self.assertFalse(np.array_equal(other, self.cts))

    def test_labels_other_than_class_indices(self):
        shifted = LabelledCollection(DATA.val.X, 10 * (DATA.val.y + 1), classes=np.array([10, 20, 30]))
        np.testing.assert_allclose(_sqsb(val=shifted).predict_ct_range(DATA.X_u, DATA.u_posteriors), self.cts)

    def test_centre_is_the_population_accuracy_of_the_draws(self):
        # §T.14: given a draw, the accuracy of the bag is a mean of binomials with mean sum_j pi_j d_j
        accuracies = np.trace(self.cts, axis1=1, axis2=2)
        recalls = np.diag(self.method.rate_matrix_)
        population = self.prevalences @ recalls
        sigma_f = self.method.last_posterior_summaries()["sigma_f"]
        standard_error = sigma_f / np.sqrt(len(accuracies))
        self.assertLess(abs(accuracies.mean() - population.mean()), 4 * standard_error + 1.0 / SAMPLE_SIZE)

    def test_diagnostics_and_summaries(self):
        self.assertEqual(set(self.method.last_nuts_diagnostics()), set(NUTS_DIAGNOSTICS))
        summaries = self.method.last_posterior_summaries()
        self.assertEqual(tuple(summaries), SQSB_SUMMARIES)
        rates = self.method.rate_matrix_
        pred_counts = np.bincount(DATA.u_posteriors.argmax(axis=1), minlength=N_CLASSES)
        acc_sd, sigma_g = accuracy_moments(self.prevalences, rates, pred_counts)
        np.testing.assert_allclose(summaries["prev_mean"], self.prevalences.mean(axis=0))
        self.assertAlmostEqual(summaries["acc_sd"], acc_sd)
        self.assertAlmostEqual(summaries["sigma_g"], sigma_g)
        self.assertEqual(summaries["val_ece"], self.method.quantifier_.val_ece_)
        recalls = np.diag(rates)
        sigma_f2 = [p @ (recalls * (1 - recalls)) / SAMPLE_SIZE for p in self.prevalences]
        self.assertAlmostEqual(summaries["sigma_f"], np.sqrt(np.mean(sigma_f2)))
        self.assertEqual((summaries["s_rec"], summaries["s_prec"]), luck_spreads(summaries["prev_mean"], rates))

    def test_confidence_api_applies_the_metric_to_each_table(self):
        interval = self.method.ci_from_cts(self.cts)
        np.testing.assert_allclose(interval.samples, np.trace(self.cts, axis1=1, axis2=2))


if __name__ == "__main__":
    unittest.main()
