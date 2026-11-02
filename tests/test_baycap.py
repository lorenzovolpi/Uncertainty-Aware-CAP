import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from cap.error import vanilla_acc
from cap.models.confidence import ACC_BayesCAP
from quapy.data import LabelledCollection
from scipy.special import gammaln
from scipy.stats import betabinom

from uncawe.methods import BayCAP, beta_binomial_loglik, eb_concentration, fit_beta_binomial, realised_tables
from uncawe.methods.baycap import (
    EB_NU_BOUNDS,
    H_KAPPA,
    H_M,
    H_NU,
    H_RECALLS,
    H_RHO,
    H_RHO_KAPPA,
    H_SHARES,
    HIERARCHICAL_SUMMARIES,
    KAPPA_SUMMARIES,
    NUTS_DIAGNOSTICS,
    P_C_COND_Y,
    P_TEST_Y,
    failed_summaries,
    hierarchical_rates,
    hierarchical_summaries,
    kappa_summaries,
    nu_from_u,
    nuts_diagnostics,
    strength_from_rho,
)

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


class EmpiricalBayesFitTest(unittest.TestCase):
    def test_loglik_matches_scipy_betabinom(self):
        f = np.array([0, 3, 9, 10, 47])
        n = np.array([4, 5, 10, 10, 50])
        for m, nu in ((0.3, 2.0), (0.9, 40.0), (0.5, 1e3)):
            with self.subTest(m=m, nu=nu):
                expected = betabinom.logpmf(f, n, nu * m, nu * (1 - m)).sum()
                log_binom = np.sum(gammaln(n + 1) - gammaln(f + 1) - gammaln(n - f + 1))
                self.assertAlmostEqual(beta_binomial_loglik(m, nu, f, n) + log_binom, expected, places=8)

    def test_recovers_parameters_with_many_classes(self):
        rng = np.random.default_rng(0)
        m, nu, n_classes = 0.9, 50.0, 400
        n = rng.integers(50, 300, n_classes)
        f = rng.binomial(n, rng.beta(nu * m, nu * (1 - m), n_classes))

        fit = fit_beta_binomial(f, n)

        self.assertFalse(fit.at_bound)
        self.assertAlmostEqual(fit.m, m, delta=0.01)
        self.assertLess(abs(np.log(fit.nu / nu)), np.log(1.3))
        self.assertGreater(fit.sd_m, 0.0)
        self.assertLess(fit.sd_m, 0.01)

    def test_common_recall_reaches_upper_bound_and_pools(self):
        n = np.array([100, 200, 50, 400, 150])
        f = (0.8 * n).astype(int)

        fit = fit_beta_binomial(f, n)

        self.assertTrue(fit.nu_at_bound)
        self.assertFalse(fit.m_at_bound)
        self.assertAlmostEqual(fit.nu, EB_NU_BOUNDS[1], delta=1e-6 * EB_NU_BOUNDS[1])
        # the pooled recall is the maximum only as nu -> infinity; at the bound it is off by O(1/nu)
        self.assertAlmostEqual(fit.m, f.sum() / n.sum(), delta=1e-4)
        self.assertTrue(np.isfinite(fit.sd_m))

    def test_perfect_recalls_reach_the_bound_of_m(self):
        n = np.array([30, 40, 50])

        fit = fit_beta_binomial(n, n)

        self.assertTrue(fit.m_at_bound)
        self.assertTrue(np.isnan(fit.sd_m))

    def test_rejects_invalid_counts(self):
        with self.assertRaisesRegex(ValueError, "0 <= f <= n"):
            fit_beta_binomial(np.array([5, 3]), np.array([4, 3]))
        with self.assertRaisesRegex(ValueError, "same shape"):
            fit_beta_binomial(np.array([1, 2]), np.array([3, 4, 5]))

    def test_concentration_matrix(self):
        m, nu, n_classes = 0.95, 41.0, 4
        alpha = eb_concentration(n_classes, m, nu)

        np.testing.assert_allclose(np.diag(alpha), nu * m)
        # cap-ml layout: row j is the column C_{:j}; its off-diagonal entries sum to nu (1 - m)
        np.testing.assert_allclose(alpha.sum(axis=1) - np.diag(alpha), nu * (1 - m))
        np.testing.assert_allclose(alpha[~np.eye(n_classes, dtype=bool)], nu * (1 - m) / (n_classes - 1))


class NutsDiagnosticsTest(unittest.TestCase):
    def test_independent_draws_converge(self):
        rng = np.random.default_rng(0)
        draws = {"pi": rng.normal(size=(1, 1000, 3)), "C": rng.normal(size=(1, 1000, 3, 3))}
        out = nuts_diagnostics(draws, np.zeros(1000, dtype=bool))

        self.assertEqual(tuple(out), NUTS_DIAGNOSTICS)
        self.assertEqual(out["nuts_divergences"], 0)
        self.assertLess(out["nuts_max_rhat"], 1.02)
        self.assertGreater(out["nuts_min_ess"], 500)

    def test_drifting_chain_is_flagged(self):
        rng = np.random.default_rng(0)
        # one entry drifts: its two halves disagree, and its draws are strongly autocorrelated
        stuck = np.cumsum(rng.normal(size=1000))
        draws = {"pi": np.stack([rng.normal(size=1000), stuck], axis=-1)[None]}
        diverging = np.zeros(1000, dtype=bool)
        diverging[[3, 500]] = True
        out = nuts_diagnostics(draws, diverging)

        self.assertEqual(out["nuts_divergences"], 2)
        self.assertGreater(out["nuts_max_rhat"], 1.1)
        self.assertLess(out["nuts_min_ess"], 50)


class HierarchicalPriorTest(unittest.TestCase):
    def test_prior_on_nu_has_the_stated_distribution(self):
        # u uniform: P(nu <= x) = 1 - (1 + x)^(-1/2), median 3, P(nu < 1) = 1 - 1/sqrt(2)
        u = (np.arange(200_000) + 0.5) / 200_000
        nu = nu_from_u(u)
        self.assertAlmostEqual(np.median(nu), 3.0, places=3)
        self.assertAlmostEqual(np.mean(nu < 1), 1 - 1 / np.sqrt(2), places=4)
        self.assertAlmostEqual(nu_from_u(1.0), 0.0)

    def test_rates_in_the_cap_layout(self):
        recalls = np.array([0.9, 0.6, 0.3])
        shares = np.array([[0.25, 0.75], [0.5, 0.5], [1.0, 0.0]])
        rates = np.asarray(hierarchical_rates(recalls, shares))
        # row j is the column C_{:j}: recall on the diagonal, the mistakes shared among the other classes in order
        expected = np.array([[0.9, 0.025, 0.075], [0.2, 0.6, 0.2], [0.7, 0.0, 0.3]])
        np.testing.assert_allclose(rates, expected, rtol=1e-6)
        np.testing.assert_allclose(rates.sum(axis=1), 1.0, rtol=1e-6)

    def test_rates_with_two_classes(self):
        rates = np.asarray(hierarchical_rates(np.array([0.8, 0.3]), np.ones((2, 1))))
        np.testing.assert_allclose(rates, [[0.8, 0.2], [0.7, 0.3]], rtol=1e-6)

    def test_summaries_of_hand_built_draws(self):
        draws = {
            H_M: np.array([0.2, 0.4, 0.6, 0.8]),
            H_NU: np.array([1.0, 10.0, 100.0, 1000.0]),
            H_RECALLS: np.array([[0.5, 0.7], [0.6, 0.8], [0.7, 0.9], [0.8, 1.0]]),
            P_TEST_Y: np.array([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2]]),
        }
        out = hierarchical_summaries(draws, min_val_count=50)
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES)
        self.assertEqual(out["nuts_failed"], 0)
        self.assertAlmostEqual(out["h_m_mean"], 0.5)
        self.assertAlmostEqual(out["h_ln_nu_median"], np.log(np.sqrt(10.0 * 100.0)))
        self.assertAlmostEqual(out["h_p_nu_above_min_n"], 0.5)
        np.testing.assert_allclose(out["h_recall_mean"], [0.65, 0.85])
        np.testing.assert_allclose(out["h_pi_mean"], [0.5, 0.5])
        var = np.var([0.2, 0.4, 0.6, 0.8], ddof=1)
        np.testing.assert_allclose(out["h_pi_cov"], [var, -var, -var, var])

    def test_failed_summaries(self):
        out = failed_summaries(3)
        self.assertEqual(set(out), set(HIERARCHICAL_SUMMARIES))
        self.assertEqual(out["nuts_failed"], 1)
        self.assertEqual(out["h_pi_cov"].shape, (9,))
        self.assertTrue(np.isnan(out["h_m_mean"]) and np.all(np.isnan(out["h_pi_mean"])))
        out = failed_summaries(3, with_kappa=True)
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES)
        self.assertTrue(all(np.isnan(out[name]) for name in KAPPA_SUMMARIES))


class LightHierarchicalPriorTest(unittest.TestCase):
    def test_prior_on_the_strength_has_the_stated_distribution(self):
        # rho = 1/(1+nu) uniform: P(nu > x) = 1/(1+x), median 1; at x = 141, 1/142 against (142)^(-1/2) for BayCAP-H
        rho = (np.arange(200_000) + 0.5) / 200_000
        nu = strength_from_rho(rho)
        self.assertAlmostEqual(np.median(nu), 1.0, places=3)
        for x in (0.5, 3.0, 141.0):
            self.assertAlmostEqual(np.mean(nu > x), 1 / (1 + x), places=4)
        self.assertAlmostEqual(strength_from_rho(1.0), 0.0)
        self.assertLess(1 / 142, 142**-0.5)

    def test_rho_is_the_share_of_the_largest_variance_of_a_recall(self):
        # Var(r | m, nu) = m(1-m) rho for r ~ Beta(nu m, nu(1-m)) (§T.16)
        m, nu = 0.7, 4.0
        a, b = nu * m, nu * (1 - m)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        self.assertAlmostEqual(var, m * (1 - m) / (1 + nu))

    def test_kappa_summaries(self):
        out = kappa_summaries(np.array([0.1, 0.5, 2.0, 8.0]))
        self.assertEqual(tuple(out), KAPPA_SUMMARIES)
        self.assertAlmostEqual(out["h_ln_kappa_median"], np.log(1.0))
        self.assertAlmostEqual(out["h_p_kappa_below_1"], 0.5)
        self.assertLess(out["h_ln_kappa_q05"], out["h_ln_kappa_q95"])
        self.assertTrue(all(np.isnan(v) for v in kappa_summaries(None).values()))

    def test_summaries_with_kappa(self):
        draws = {
            H_M: np.array([0.4, 0.6]),
            H_NU: np.array([1.0, 3.0]),
            H_RECALLS: np.array([[0.5, 0.7, 0.9], [0.6, 0.8, 1.0]]),
            P_TEST_Y: np.array([[0.2, 0.3, 0.5], [0.4, 0.3, 0.3]]),
            H_KAPPA: np.array([0.5, 2.0]),
        }
        out = hierarchical_summaries(draws, min_val_count=2, with_kappa=True)
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES)
        self.assertAlmostEqual(out["h_p_kappa_below_1"], 0.5)
        self.assertAlmostEqual(out["h_ln_kappa_median"], 0.0)
        # without kappa (BayCAP-H) the summaries are unchanged
        self.assertEqual(tuple(hierarchical_summaries(draws, min_val_count=2)), HIERARCHICAL_SUMMARIES)


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

    def test_rejects_unknown_c_prior(self):
        with self.assertRaisesRegex(ValueError, "c_prior must be one of"):
            BayCAP(vanilla_acc, c_prior="empirical_bayes")

    def test_uniform_c_prior_reproduces_current_draws(self):
        default = self.fit(BayCAP(vanilla_acc, random_state=5, **NUTS_PARAMS))
        uniform = self.fit(BayCAP(vanilla_acc, c_prior="uniform", random_state=5, **NUTS_PARAMS))
        ref = self.fit(ACC_BayesCAP(vanilla_acc, random_state=5, **NUTS_PARAMS))

        self.assertIsNone(uniform.eb_fit)
        np.testing.assert_array_equal(uniform.c_alpha, np.ones((3, 3)))
        expected = ref.predict_ct_range(self.X, self.test_posteriors)
        np.testing.assert_array_equal(default.predict_ct_range(self.X, self.test_posteriors), expected)
        np.testing.assert_array_equal(uniform.predict_ct_range(self.X, self.test_posteriors), expected)

    def test_empirical_bayes_fits_validation_counts(self):
        method = self.fit(BayCAP(vanilla_acc, c_prior="empirical-bayes", random_state=5, **NUTS_PARAMS))

        val_ct = method.val_ct  # cap-ml layout: true classes on rows
        expected = fit_beta_binomial(np.diag(val_ct), val_ct.sum(axis=1))
        self.assertEqual(method.eb_fit, expected)
        np.testing.assert_allclose(method.c_alpha, eb_concentration(3, expected.m, expected.nu))

        for estimand in ("population", "realised"):
            with self.subTest(estimand=estimand):
                method.estimand = estimand
                cts = method.predict_ct_range(self.X, self.test_posteriors)
                self.assertEqual(cts.shape, (NUTS_PARAMS["num_samples"], 3, 3))
                self.assertTrue(np.all(np.isfinite(cts)))
                np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0, rtol=1e-5)

    def test_diagnostics_of_the_last_run_leave_draws_unchanged(self):
        method = BayCAP(vanilla_acc, random_state=5, **NUTS_PARAMS)
        with self.assertRaisesRegex(RuntimeError, "predict_ct_range"):
            method.last_nuts_diagnostics()
        self.fit(method)
        cts = method.predict_ct_range(self.X, self.test_posteriors)
        out = method.last_nuts_diagnostics()

        self.assertEqual(tuple(out), NUTS_DIAGNOSTICS)
        self.assertIsInstance(out["nuts_divergences"], int)
        self.assertGreaterEqual(out["nuts_divergences"], 0)
        self.assertTrue(np.isfinite(out["nuts_max_rhat"]) and out["nuts_max_rhat"] > 0)
        self.assertTrue(0 < out["nuts_min_ess"])
        # the draws of pi and C behind the diagnostics are those of the returned tables
        draws, diverging = method._last_nuts
        self.assertEqual(diverging.shape, (NUTS_PARAMS["num_samples"],))
        pi, c = draws[P_TEST_Y][0], draws[P_C_COND_Y][0]
        np.testing.assert_allclose(np.einsum("sy,syc->syc", pi, c), cts, rtol=1e-6)

    def test_hierarchical_prior_returns_valid_tables_and_summaries(self):
        for estimand in ("population", "realised"):
            with self.subTest(estimand=estimand):
                method = self.fit(
                    BayCAP(vanilla_acc, estimand=estimand, c_prior="hierarchical", random_state=5, **NUTS_PARAMS)
                )
                self.assertIsNone(method.c_alpha)
                cts = method.predict_ct_range(self.X, self.test_posteriors)
                self.assertEqual(cts.shape, (NUTS_PARAMS["num_samples"], 3, 3))
                self.assertTrue(np.all(np.isfinite(cts)) and np.all(cts >= 0))
                np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0, rtol=1e-5)

                self.assertEqual(tuple(method.last_nuts_diagnostics()), NUTS_DIAGNOSTICS)
                out = method.last_posterior_summaries()
                self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES)
                self.assertEqual(out["nuts_failed"], 0)
                self.assertTrue(0 < out["h_m_mean"] < 1 and 0 <= out["h_p_nu_above_min_n"] <= 1)
                self.assertLessEqual(out["h_ln_nu_q05"], out["h_ln_nu_median"])
                self.assertLessEqual(out["h_ln_nu_median"], out["h_ln_nu_q95"])
                cov = out["h_pi_cov"].reshape(3, 3)
                np.testing.assert_allclose(cov, cov.T)
                self.assertTrue(np.all(np.linalg.eigvalsh(cov) > -1e-9))
                # the draws of C have the recalls of the summaries on their diagonal
                draws, _ = method._last_nuts
                recalls = np.diagonal(draws[P_C_COND_Y][0], axis1=1, axis2=2)
                np.testing.assert_allclose(recalls.mean(axis=0), out["h_recall_mean"], rtol=1e-5)

    def test_hierarchical_prior_with_two_classes(self):
        val_y = np.repeat([0, 1], 20)
        val_yhat = np.where(np.arange(40) % 5 == 0, 1 - val_y, val_y)
        validation = LabelledCollection(np.zeros((40, 1)), val_y, classes=np.arange(2))
        method = BayCAP(vanilla_acc, c_prior="hierarchical", random_state=5, **NUTS_PARAMS)
        method.fit(validation, _one_hot(val_yhat, 2))
        cts = method.predict_ct_range(np.zeros((30, 1)), _one_hot(np.arange(30) % 2, 2))
        self.assertTrue(np.all(np.isfinite(cts)))
        draws, _ = method._last_nuts
        np.testing.assert_allclose(draws[P_C_COND_Y][0].sum(axis=2), 1.0, rtol=1e-5)

    def test_hierarchical_prior_is_reproducible(self):
        first = self.fit(BayCAP(vanilla_acc, c_prior="hierarchical", random_state=5, **NUTS_PARAMS))
        second = self.fit(BayCAP(vanilla_acc, c_prior="hierarchical", random_state=5, **NUTS_PARAMS))
        np.testing.assert_array_equal(
            first.predict_ct_range(self.X, self.test_posteriors), second.predict_ct_range(self.X, self.test_posteriors)
        )

    def test_failed_start_keeps_the_bag_as_nan(self):
        method = self.fit(BayCAP(vanilla_acc, c_prior="hierarchical", random_state=5, **NUTS_PARAMS))

        def no_start(*args, **kwargs):
            raise RuntimeError("Cannot find valid initial parameters. Please check your model again.")

        method.sample_posterior = no_start
        cts = method.predict_ct_range(self.X, self.test_posteriors)
        self.assertEqual(cts.shape, (NUTS_PARAMS["num_samples"], 3, 3))
        self.assertTrue(np.all(np.isnan(cts)))
        self.assertTrue(all(np.isnan(v) for v in method.last_nuts_diagnostics().values()))
        self.assertEqual(method.last_posterior_summaries()["nuts_failed"], 1)

    def test_light_hierarchical_prior_returns_valid_tables_and_summaries(self):
        method = self.fit(
            BayCAP(vanilla_acc, estimand="realised", c_prior="hierarchical-f", random_state=5, **NUTS_PARAMS)
        )
        self.assertIsNone(method.c_alpha)
        cts = method.predict_ct_range(self.X, self.test_posteriors)
        self.assertEqual(cts.shape, (NUTS_PARAMS["num_samples"], 3, 3))
        self.assertTrue(np.all(np.isfinite(cts)) and np.all(cts >= 0))
        np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0, rtol=1e-5)

        self.assertEqual(tuple(method.last_nuts_diagnostics()), NUTS_DIAGNOSTICS)
        draws, _ = method._last_nuts
        # the diagnostics cover m, rho and rho_kappa, not u
        self.assertEqual(set(draws), {P_TEST_Y, P_C_COND_Y, H_M, H_RHO, H_RHO_KAPPA})
        out = method.last_posterior_summaries()
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES)
        self.assertEqual(out["nuts_failed"], 0)
        self.assertLessEqual(out["h_ln_kappa_q05"], out["h_ln_kappa_median"])
        self.assertLessEqual(out["h_ln_kappa_median"], out["h_ln_kappa_q95"])
        self.assertTrue(0 <= out["h_p_kappa_below_1"] <= 1)

    def test_light_hierarchical_prior_samples_kappa_and_nu_from_rho(self):
        from numpyro.handlers import seed, trace

        method = self.fit(BayCAP(vanilla_acc, c_prior="hierarchical-f", random_state=5, **NUTS_PARAMS))
        model = seed(method.model, rng_seed=0)
        sites = trace(model).get_trace(
            pred_posterior_count=self.pred_counts,
            train_class_cond_count=method.val_ct,
            alpha=np.ones(3),
            c_alpha=None,
        )
        self.assertNotIn("h_u", sites)
        rho, rho_kappa = sites[H_RHO]["value"], sites[H_RHO_KAPPA]["value"]
        self.assertAlmostEqual(float(sites[H_NU]["value"]), float(1 / rho - 1), places=4)
        kappa = float(sites[H_KAPPA]["value"])
        self.assertAlmostEqual(kappa, float(1 / rho_kappa - 1), places=4)
        concentration = np.asarray(sites[H_SHARES]["fn"].concentration)
        np.testing.assert_allclose(concentration, np.full((3, 2), kappa), rtol=1e-5)

    def test_light_hierarchical_prior_with_two_classes_has_no_kappa(self):
        val_y = np.repeat([0, 1], 20)
        val_yhat = np.where(np.arange(40) % 5 == 0, 1 - val_y, val_y)
        validation = LabelledCollection(np.zeros((40, 1)), val_y, classes=np.arange(2))
        method = BayCAP(vanilla_acc, c_prior="hierarchical-f", random_state=5, **NUTS_PARAMS)
        method.fit(validation, _one_hot(val_yhat, 2))
        cts = method.predict_ct_range(np.zeros((30, 1)), _one_hot(np.arange(30) % 2, 2))
        self.assertTrue(np.all(np.isfinite(cts)))
        draws, _ = method._last_nuts
        self.assertEqual(set(draws), {P_TEST_Y, P_C_COND_Y, H_M, H_RHO})
        out = method.last_posterior_summaries()
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES)
        self.assertTrue(all(np.isnan(out[name]) for name in KAPPA_SUMMARIES))

    def test_light_hierarchical_failed_start_keeps_the_kappa_columns(self):
        method = self.fit(BayCAP(vanilla_acc, c_prior="hierarchical-f", random_state=5, **NUTS_PARAMS))

        def no_start(*args, **kwargs):
            raise RuntimeError("Cannot find valid initial parameters. Please check your model again.")

        method.sample_posterior = no_start
        self.assertTrue(np.all(np.isnan(method.predict_ct_range(self.X, self.test_posteriors))))
        out = method.last_posterior_summaries()
        self.assertEqual(tuple(out), HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES)
        self.assertEqual(out["nuts_failed"], 1)

    def test_other_priors_have_no_posterior_summaries(self):
        method = self.fit(BayCAP(vanilla_acc, random_state=5, **NUTS_PARAMS))
        method.predict_ct_range(self.X, self.test_posteriors)
        self.assertEqual(method.last_posterior_summaries(), {})


if __name__ == "__main__":
    unittest.main()
