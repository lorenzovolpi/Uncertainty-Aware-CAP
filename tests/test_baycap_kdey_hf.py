"""BayCAP-KDEy with a random $C$ and the Hf prior (E3.2 of ``docs/ext3_posterior_propagation.md``, work-plan card
WP-13): the cells of $V$, the likelihood of §T.10, the empty cells of §T.12, the Hf sites, the realised tables and the
seeds."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from cap.error import vanilla_acc
from cap.models.confidence import P_C_COND_Y, P_TEST_Y
from numpyro import handlers
from quapy.data import LabelledCollection
from scipy.special import logsumexp
from test_baycap_fixed import DATA

from uncawe.methods.baycap import H_KAPPA, NUTS_DIAGNOSTICS, BayCAP
from uncawe.methods.baycap_kdey import (
    CELL_CLASS,
    CELL_OWN,
    CELL_PREDICTED,
    KDEY_HF_SUMMARIES,
    BayCAPKDEy,
    kdey_hf_loglik,
    kdey_hf_model,
)

SETTINGS = {"num_warmup": 40, "num_samples": 30, "random_state": 5}


def _fitted(input_classifier="h", val=DATA.val, val_posteriors=DATA.val_posteriors, **kwargs):
    method = BayCAPKDEy(vanilla_acc, input_classifier=input_classifier, c_prior="hierarchical-f", **(SETTINGS | kwargs))
    return method.fit(val, val_posteriors)


def _latent_sites(model, *args, **kwargs) -> set[str]:
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(*args, **kwargs)
    return {name for name, site in trace.items() if site["type"] == "sample" and not site["is_observed"]}


class _EmptyCells:
    """Three classes with outputs chosen by hand: $h$ never predicts class 2 on $V$, and never predicts class 0 for an
    instance of class 1, so the cells (2, j) take the class densities and the cell (0, 1) the density of the
    instances predicted as 0."""

    def __init__(self):
        rng = np.random.default_rng(1)
        y = np.repeat([0, 1, 2], 12)
        pred = np.concatenate([np.tile([0, 1], 6), np.ones(12, int), np.tile([0, 1], 6)])
        outputs = np.full((len(y), 3), 0.1)
        outputs[np.arange(len(y)), pred] = 0.8
        outputs += rng.uniform(0, 0.02, size=outputs.shape)
        self.val_posteriors = outputs / outputs.sum(axis=1, keepdims=True)
        self.val = LabelledCollection(rng.normal(size=(len(y), 2)), y, classes=np.array([0, 1, 2]))
        self.pred = pred
        # a bag with predictions of every class, also of class 2
        bag = np.full((9, 3), 0.1)
        bag[np.arange(9), np.repeat([0, 1, 2], 3)] = 0.8
        self.u_posteriors = bag / bag.sum(axis=1, keepdims=True)


class CellsTest(unittest.TestCase):
    def test_rejects_unknown_c_prior(self):
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, c_prior="hierarchical")

    def test_cells_partition_v_and_mix_into_the_class_densities(self):
        for source in ("h", "mlp"):
            with self.subTest(input_classifier=source):
                method = _fitted(source)
                counts = method.val_counts_
                np.testing.assert_array_equal(counts.sum(axis=1), np.bincount(DATA.val.y, minlength=3))
                self.assertEqual(counts.sum(), len(DATA.val))
                outputs = method.outputs(DATA.X_u, DATA.u_posteriors)
                cells = np.exp(method.cell_log_densities(outputs))
                # sum_k C-hat_kj phi_kj(z) = f_j(z), with rate_matrix_[j, k] = C-hat_kj; empty cells have weight 0
                mixture = np.einsum("jk,ujk->uj", method.rate_matrix_, cells)
                class_densities = np.exp(method.kde_.log_densities(outputs).T)[:, method._quantifier_to_rate_order]
                np.testing.assert_allclose(mixture, class_densities, rtol=1e-8)

    def test_loglik_at_c_hat_is_that_of_bayes_kdey(self):
        method = _fitted("h")
        outputs = method.outputs(DATA.X_u, DATA.u_posteriors)
        prev = np.array([0.5, 0.2, 0.3])
        got = kdey_hf_loglik(jnp.asarray(prev), jnp.asarray(method.rate_matrix_), method.cell_log_densities(outputs))
        expected = np.sum(logsumexp(np.log(prev)[:, None] + method.kde_.log_densities(outputs), axis=0))
        np.testing.assert_allclose(float(got), expected, rtol=1e-5)

    def test_joint_loglik_summed_over_the_predictions_is_that_of_the_outputs(self):
        method = _fitted("mlp")
        outputs = method.outputs(DATA.X_u, DATA.u_posteriors)[:6]
        prev = jnp.asarray([0.3, 0.3, 0.4])
        rates = jnp.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
        full = method.cell_log_densities(outputs)
        for u in range(len(outputs)):
            joint = [
                float(
                    kdey_hf_loglik(
                        prev, rates, method.cell_log_densities(outputs[u : u + 1], np.array([k])), np.array([k])
                    )
                )
                for k in range(3)
            ]
            alone = float(kdey_hf_loglik(prev, rates, full[u : u + 1]))
            np.testing.assert_allclose(logsumexp(joint), alone, rtol=1e-5)

    def test_selected_cells_match_the_full_array(self):
        method = _fitted("mlp")
        outputs = method.outputs(DATA.X_u, DATA.u_posteriors)
        pred = DATA.u_posteriors.argmax(axis=1)
        full = method.cell_log_densities(outputs)
        np.testing.assert_allclose(method.cell_log_densities(outputs, pred), full[np.arange(len(pred)), :, pred])

    def test_empty_cells_borrow_densities(self):
        data = _EmptyCells()
        method = _fitted("h", val=data.val, val_posteriors=data.val_posteriors)
        # cap-ml layout: cell_source_[j, k] for true class j and prediction k
        expected = np.full((3, 3), CELL_OWN)
        expected[:, 2] = CELL_CLASS
        expected[1, 0] = CELL_PREDICTED
        np.testing.assert_array_equal(method.cell_source_, expected)
        outputs = data.u_posteriors
        cells = method.cell_log_densities(outputs)
        borrowed = method.kde_.density(data.val_posteriors[data.pred == 0])
        np.testing.assert_allclose(cells[:, 1, 0], method.kde_.log_density(borrowed, outputs))
        np.testing.assert_allclose(cells[:, :, 2], method.kde_.log_densities(outputs).T)

        cts = method.predict_ct_range(None, data.u_posteriors)
        self.assertTrue(np.all(np.isfinite(cts)))
        summaries = method.last_posterior_summaries()
        self.assertEqual(summaries["nuts_failed"], 0)
        self.assertEqual((summaries["n_empty_cells"], summaries["n_class_density_cells"]), (4, 3))


class ModelTest(unittest.TestCase):
    def test_hf_sites_are_those_of_baycap(self):
        for n_classes in (2, 3):
            with self.subTest(n_classes=n_classes):
                val_counts = np.full((n_classes, n_classes), 3) + 5 * np.eye(n_classes, dtype=int)
                pred_counts = np.full(n_classes, 4)
                alpha = np.ones(n_classes)
                baycap = BayCAP(vanilla_acc, c_prior="hierarchical-f", random_state=0)
                expected = _latent_sites(baycap.model, pred_counts, val_counts, alpha, None)
                got = _latent_sites(kdey_hf_model, np.zeros((4, n_classes, n_classes)), jnp.asarray(val_counts), alpha)
                self.assertEqual(got, expected)
                self.assertIn(P_TEST_Y, got)
                trace = handlers.trace(handlers.seed(kdey_hf_model, jax.random.PRNGKey(0))).get_trace(
                    np.zeros((4, n_classes, n_classes)), jnp.asarray(val_counts), alpha
                )
                self.assertEqual(H_KAPPA in trace, n_classes > 2)
                self.assertIn(P_C_COND_Y, trace)


class PredictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods = {source: _fitted(source, estimand="realised") for source in ("h", "mlp")}
        cls.cts = {source: m.predict_ct_range(DATA.X_u, DATA.u_posteriors) for source, m in cls.methods.items()}

    def test_realised_tables_keep_the_predicted_margins(self):
        pred_counts = np.bincount(DATA.u_posteriors.argmax(axis=1), minlength=3)
        for source, cts in self.cts.items():
            with self.subTest(input_classifier=source):
                self.assertEqual(cts.shape, (SETTINGS["num_samples"], 3, 3))
                np.testing.assert_allclose(
                    cts.sum(axis=1), np.broadcast_to(pred_counts / pred_counts.sum(), (len(cts), 3))
                )

    def test_same_seeds_give_the_same_draws(self):
        for source, cts in self.cts.items():
            with self.subTest(input_classifier=source):
                again = _fitted(source, estimand="realised").predict_ct_range(DATA.X_u, DATA.u_posteriors)
                np.testing.assert_array_equal(again, cts)

    def test_population_tables_are_the_draws_of_c_diag_pi(self):
        method = _fitted("h", estimand="population")
        cts = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        draws = method._last_nuts[0]
        np.testing.assert_allclose(cts, draws[P_TEST_Y][0][:, :, None] * draws[P_C_COND_Y][0], rtol=1e-6)
        np.testing.assert_allclose(cts.sum(axis=(1, 2)), 1.0, rtol=1e-5)

    def test_diagnostics_and_summaries(self):
        pred_counts = np.bincount(DATA.u_posteriors.argmax(axis=1), minlength=3)
        for source, method in self.methods.items():
            with self.subTest(input_classifier=source):
                self.assertEqual(set(method.last_nuts_diagnostics()), set(NUTS_DIAGNOSTICS))
                self.assertGreater(method.last_nuts_diagnostics()["nuts_min_ess"], 0)
                summaries = method.last_posterior_summaries()
                self.assertEqual(tuple(summaries), KDEY_HF_SUMMARIES)
                self.assertEqual(summaries["nuts_failed"], 0)
                c_mean = summaries["c_mean"].reshape(3, 3)
                np.testing.assert_allclose(c_mean.sum(axis=1), 1.0, rtol=1e-5)
                self.assertTrue(np.all(summaries["c_sd"] > 0))
                shares = pred_counts / pred_counts.sum()
                expected = np.abs(shares - summaries["h_pi_mean"] @ c_mean).sum()
                self.assertAlmostEqual(summaries["mismatch"], expected, places=5)
                self.assertEqual(summaries["n_empty_cells"], int(np.sum(method.val_counts_ == 0)))
                self.assertFalse(np.isnan(summaries["h_ln_kappa_median"]))

    def test_failed_start_gives_nan_tables(self):
        method = _fitted("h", estimand="realised")
        with patch.object(BayCAPKDEy, "_run_nuts", side_effect=RuntimeError("Cannot find valid initial parameters.")):
            cts = method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        self.assertTrue(np.all(np.isnan(cts)))
        self.assertTrue(all(np.isnan(v) for v in method.last_nuts_diagnostics().values()))
        summaries = method.last_posterior_summaries()
        self.assertEqual(tuple(summaries), KDEY_HF_SUMMARIES)
        self.assertEqual(summaries["nuts_failed"], 1)
        self.assertTrue(np.all(np.isnan(summaries["c_mean"])))


if __name__ == "__main__":
    unittest.main()
