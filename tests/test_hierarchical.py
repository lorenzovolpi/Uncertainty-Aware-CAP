import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import betabinom

from uncawe.plots import hierarchical as hierarchical_plots
from uncawe.tables import eb_prior
from uncawe.tables import hierarchical as h
from uncawe.tables.displacement import smoothed_recalls

# cap-ml layout: true classes on rows
VAL_COUNTS = {
    ("wdbc", "LR"): np.array([[90, 10], [30, 70]]),
    ("german", "LR"): np.array([[180, 20], [40, 60]]),
    ("letter", "LR"): np.array([[45, 3, 2], [4, 40, 6], [1, 2, 47]]),
}
PAIRS = [("letter", "LR"), ("wdbc", "LR"), ("german", "LR")]


def results(pairs, estim_shift: float, seed: int = 0, props_shift: float = 0.0) -> pd.DataFrame:
    """Stored results of one method: per bag a true table, a mean table and accuracy draws around its trace.

    The bags (true tables) are the same for every method; ``estim_shift`` moves the recalls of the mean table and
    ``props_shift`` its prevalence of the first class.
    """
    bag_rng, rng = np.random.default_rng(0), np.random.default_rng(seed)
    rows = []
    for dataset, classifier in pairs:
        k = VAL_COUNTS[(dataset, classifier)].shape[0]
        for bag in range(4):
            props = bag_rng.dirichlet(np.ones(k))
            rates = np.full((k, k), 0.1 / (k - 1))
            np.fill_diagonal(rates, 0.9)
            true_ct = props[:, None] * rates
            estim_ct = true_ct.copy()
            estim_ct[np.diag_indices(k)] = np.maximum(np.diag(estim_ct) + estim_shift / k, 1e-3)
            estim_ct /= estim_ct.sum()
            estim_ct[0] *= 1 + props_shift
            estim_ct /= estim_ct.sum()
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": k,
                    "true_cts": true_ct,
                    "estim_cts": estim_ct,
                    "estim_acc_ranges": np.trace(estim_ct) + rng.normal(0, 0.05, 200),
                    "nuts_divergences": bag % 2,
                    "nuts_max_rhat": 1.0,
                    "nuts_min_ess": 400.0,
                }
            )
    return pd.DataFrame(rows)


def method_results(pairs) -> dict[str, pd.DataFrame]:
    shifts = {"BayesCAP": -0.05, "BayesCAP-real": -0.05, "BayesCAP-EB": 0.0, "BayesCAP-EB-real": 0.0}
    return {
        method: results(pairs, shift, seed=i, props_shift=0.0 if shift < 0 else 0.3)
        for i, (method, shift) in enumerate(shifts.items())
    }


def synthetic_table() -> pd.DataFrame:
    def load(method, experiment, domain):
        return method_results(PAIRS)[method]

    with (
        patch.object(h.estimand_gap, "load_results", side_effect=load),
        patch.object(h, "load_validation_counts", side_effect=lambda pairs, domain: VAL_COUNTS),
    ):
        return h.compute_table()


class PosteriorTest(unittest.TestCase):
    def test_class_log_marginals_are_beta_binomial(self):
        f, n = np.array([7.0, 40.0]), np.array([10.0, 50.0])
        ll = h.class_log_marginals(f, n)
        i, j = 50, 120
        m, nu = h.GRID_M[i], np.exp(h.GRID_LN_NU[j])
        for c in range(2):
            expected = betabinom.logpmf(f[c], n[c], nu * m, nu * (1 - m))
            self.assertAlmostEqual(ll[c, i, j], expected, places=8)

    def test_grid_weights_are_normalised_and_prior_matches_density(self):
        ll = h.class_log_marginals(np.array([7.0]), np.array([10.0])).sum(axis=0)
        for prior in h.NU_PRIORS:
            self.assertAlmostEqual(h.grid_weights(ll, prior).sum(), 1.0)
        # log-uniform prior: flat on the grid of ln(nu); flat prior: weights grow as nu
        flat_ln = h.prior_weights("loguniform").sum(axis=0)
        np.testing.assert_allclose(flat_ln, flat_ln[0])
        flat_nu = h.prior_weights("flat").sum(axis=0)
        np.testing.assert_allclose(flat_nu / flat_nu[0], np.exp(h.GRID_LN_NU - h.GRID_LN_NU[0]))
        # the 90% width of a flat density on [a, b] is 0.9 (b - a)
        width = h.ln_nu_width(h.prior_weights("loguniform"))
        step = h.GRID_LN_NU[1] - h.GRID_LN_NU[0]
        self.assertAlmostEqual(width, 0.9 * (h.GRID_LN_NU[-1] - h.GRID_LN_NU[0]), delta=step)

    def test_mixture_moments_match_monte_carlo(self):
        f, n = np.array([45.0, 40.0, 47.0]), np.array([50.0, 50.0, 50.0])
        weights = h.grid_weights(h.class_log_marginals(f, n).sum(axis=0), "loguniform")
        post = h.mixture_posterior(f, n, weights)
        mean, var = h.recall_moments(post)
        rng = np.random.default_rng(0)
        idx = rng.choice(post.weights.size, 200_000, p=post.weights)
        # recover (a, b) of each point from its mean and variance
        mu, v = post.mean[:, idx], post.var[:, idx]
        s = mu * (1 - mu) / v - 1
        draws = rng.beta(mu * s, (1 - mu) * s)
        np.testing.assert_allclose(draws.mean(axis=1), mean, atol=2e-3)
        np.testing.assert_allclose(draws.var(axis=1), var, rtol=0.05)
        props = np.array([[0.2, 0.3, 0.5]])
        acc_mean, acc_var = h.accuracy_moments(post, props)
        acc = props @ draws
        self.assertAlmostEqual(acc_mean[0], acc.mean(), delta=2e-3)
        self.assertAlmostEqual(acc_var[0], acc.var(), delta=0.05 * acc.var())

    def test_single_point_has_no_second_term(self):
        post = h.RecallPosterior(np.array([[0.8], [0.6]]), np.array([[0.01], [0.02]]), np.ones(1))
        props = np.array([[0.25, 0.75]])
        mean, var = h.accuracy_moments(post, props)
        self.assertAlmostEqual(mean[0], 0.25 * 0.8 + 0.75 * 0.6)
        self.assertAlmostEqual(var[0], 0.25**2 * 0.01 + 0.75**2 * 0.02)

    def test_uniform_and_eb_posteriors(self):
        counts = VAL_COUNTS[("letter", "LR")]
        np.testing.assert_allclose(h.uniform_posterior(counts).mean[:, 0], smoothed_recalls(counts, "uniform"))
        np.testing.assert_allclose(h.eb_posterior(counts).mean[:, 0], smoothed_recalls(counts, "empirical-bayes"))
        # Beta(f+1, n-f+K-1) variance
        a, b = 45 + 1, 50 - 45 + 2
        self.assertAlmostEqual(h.uniform_posterior(counts).var[0, 0], a * b / ((a + b) ** 2 * (a + b + 1)))


class ScreensTest(unittest.TestCase):
    def test_predict_draws(self):
        samples = np.random.default_rng(0).normal(0.8, 0.02, 500)
        np.testing.assert_allclose(h.predict_draws(samples, 0.0, 1e-4, 1e-4), samples)
        moved = h.predict_draws(samples, 0.01, 1e-4, 4e-4)
        self.assertAlmostEqual(moved.mean(), samples.mean() + 0.01)
        self.assertAlmostEqual(moved.var(), samples.var() + 3e-4)
        self.assertIsNone(h.predict_draws(samples, 0.0, 1.0, 0.0))

    def test_loo_pvalue_flags_an_outlier_class(self):
        f = np.array([90.0, 91.0, 89.0, 90.0, 30.0])
        n = np.full(5, 100.0)
        ll = h.class_log_marginals(f, n)
        total = ll.sum(axis=0)
        p_outlier = h.loo_pvalue(f[4], n[4], h.grid_weights(total - ll[4], "loguniform"))
        p_typical = h.loo_pvalue(f[0], n[0], h.grid_weights(total - ll[0], "loguniform"))
        self.assertLess(p_outlier, 0.01)
        self.assertGreater(p_typical, 0.1)

    def test_alpha_risk(self):
        weights = np.zeros((h.GRID_M.size, h.GRID_LN_NU.size))
        weights[-1, 0] = 1.0  # m near 1, nu = 0.1: both parameters below 1
        self.assertAlmostEqual(h.alpha_risk(weights), 1.0)
        weights[:] = 0.0
        weights[100, -1] = 1.0  # m near 0.5, nu = 1e4
        self.assertAlmostEqual(h.alpha_risk(weights), 0.0)

    def test_predicted_scores_with_the_uniform_prior_equal_the_stored_ones(self):
        df = results([("letter", "LR")], -0.05)
        uniform = h.uniform_posterior(VAL_COUNTS[("letter", "LR")])
        stored = h.stored_scores(df, "BayesCAP-real")
        predicted = h.predicted_scores(df, "BayesCAP-real", uniform, uniform)
        for score in h.SCORES:
            self.assertAlmostEqual(predicted[score], stored[score])
        self.assertEqual(predicted["undefined"], 0.0)


class CentreSplitTest(unittest.TestCase):
    def setUp(self):
        self.pair = [("letter", "LR")]
        self.uniform_df = results(self.pair, -0.05, seed=1)
        self.eb_df = results(self.pair, 0.0, seed=2, props_shift=0.3)
        counts = VAL_COUNTS[self.pair[0]]
        self.uniform, self.eb = h.uniform_posterior(counts), h.eb_posterior(counts)

    def test_parts_add_up_to_the_change_of_the_centre(self):
        split = h.centre_split(self.uniform_df, self.eb_df, self.uniform, self.eb)
        change = [
            np.trace(e / e.sum()) - np.trace(u / u.sum())
            for u, e in zip(self.uniform_df["estim_cts"], self.eb_df["estim_cts"])
        ]
        np.testing.assert_allclose(split["recall_part"] + split["prevalence_part"], change, atol=1e-12)
        self.assertTrue((np.abs(split["prevalence_part"]) > 0).all())
        self.assertTrue((split["props_l1"] > 0).all())
        props = np.stack([h._bag_props(ct) for ct in self.uniform_df["estim_cts"]])
        expected = h.accuracy_moments(self.eb, props)[0] - h.accuracy_moments(self.uniform, props)[0]
        np.testing.assert_allclose(split["d4_shift"], expected)

    def test_same_tables_have_no_parts(self):
        split = h.centre_split(self.uniform_df, self.uniform_df, self.uniform, self.eb)
        np.testing.assert_allclose(split["recall_part"], 0.0, atol=1e-15)
        np.testing.assert_allclose(split["prevalence_part"], 0.0, atol=1e-15)

    def test_bags_must_match(self):
        other = results(self.pair, 0.0, seed=2)
        other.at[0, "true_cts"] = other.at[0, "true_cts"][::-1]
        with self.assertRaises(ValueError):
            h.centre_split(self.uniform_df, other, self.uniform, self.eb)
        with self.assertRaises(ValueError):
            h.centre_split(self.uniform_df, other.iloc[:2], self.uniform, self.eb)

    def test_extra_shift_moves_the_predicted_draws(self):
        method = "BayesCAP-real"
        base = h.predicted_scores(self.uniform_df, method, self.uniform, self.uniform)
        zero = h.predicted_scores(self.uniform_df, method, self.uniform, self.uniform, extra_shift=np.zeros(4))
        far = h.predicted_scores(self.uniform_df, method, self.uniform, self.uniform, extra_shift=np.full(4, 1.0))
        for score in h.SCORES:
            self.assertAlmostEqual(zero[score], base[score])
        self.assertEqual(far["coverage"], 0.0)


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = synthetic_table()

    def test_rows_order_and_columns(self):
        self.assertNotIn("source", self.table)
        self.assertEqual(list(self.table["dataset"].astype(str)), ["german", "wdbc", "letter"])
        for prior in h.NU_PRIORS:
            for column in ("width_ratio", "alpha_risk", "min_pvalue", "max_abs_z"):
                self.assertIn(f"{prior}_{column}", self.table)
            self.assertIn(f"realised_{prior}_pred_coverage", self.table)
        self.assertTrue((self.table["realised_eb_pred_undefined"] == 0).all())
        np.testing.assert_allclose(self.table["BayesCAP-EB_share_divergent"], 0.5)
        for name in h.PUSHFORWARDS:
            for column in ("d4_shift", "recall_part", "prevalence_part", "abs_prevalence_part", "props_l1"):
                self.assertIn(f"{name}_{column}", self.table)
            for target in ("ebprev", "ebcentre"):
                self.assertIn(f"{name}_{target}_pred_coverage", self.table)
        self.assertTrue((self.table["realised_abs_prevalence_part"] > 0).all())

    def test_analysis_summary(self):
        summary = h.analysis_summary(self.table)
        validation = summary["validation"]
        self.assertEqual(set(validation["pushforward"]), set(h.PUSHFORWARDS))
        first = validation.iloc[0]
        self.assertEqual((first["K"], first["pairs"]), ("all", 3))
        for column in ("mean_abs_cov_diff_prev", "within_0.05_centre", "corr_shift_recall", "mean_abs_prevalence_part"):
            self.assertIn(column, validation)
        stored = self.table["BayesCAP-EB-real_coverage"]
        realised = validation.set_index(["K", "pushforward"]).loc[("all", "realised")]
        self.assertAlmostEqual(
            realised["mean_abs_cov_diff_prev"], (self.table["realised_ebprev_pred_coverage"] - stored).abs().mean()
        )
        prediction = summary["prediction"].set_index("K")
        self.assertAlmostEqual(
            prediction.loc["K=2", "uniform_cov"], self.table.loc[:1, "BayesCAP-real_coverage"].mean()
        )
        self.assertIn("3<=K<=7", prediction.index)

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in h.TABLES.items():
            with self.subTest(table=name):
                latex = eb_prior._latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("\\begin{longtable}", latex)

    def test_figures(self):
        for name, plotter in hierarchical_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(self.table)
                self.assertTrue(fig.axes)
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
