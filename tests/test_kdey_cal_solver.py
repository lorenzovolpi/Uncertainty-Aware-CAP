"""The gate of the solver of E7.2a (work-plan card WP-26): EM against the standard KDEy, QuaPy's ``KDEyML`` (SLSQP), in
the selection of the bandwidth on calibration bags."""

import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import quapy as qp
from cap.error import vanilla_acc
from quapy.method.aggregative import KDEyML
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from test_baycap_fixed import DATA
from test_kdey_cal import GRID, _log_likelihood, _method

from uncawe.experiments import kdey_cal_solver as gate_run
from uncawe.methods.baycap_kdey import CAL_SOLVERS, BayCAPKDEy
from uncawe.methods.bayes_kdey import bag_log_likelihoods, em_prevalences, slsqp_prevalences
from uncawe.tables import kdey_cal_solver as kcs

qp.environ.setdefault("_R_SEED", 0)


def _simplex_pool(rng, n_classes: int, n: int):
    """Outputs on the simplex of three classes (or more), each concentrated near its own vertex, with their labels."""
    labels = rng.integers(0, n_classes, size=n)
    alpha = np.ones((n, n_classes))
    alpha[np.arange(n), labels] = 4.0
    return np.stack([rng.dirichlet(a) for a in alpha]), labels


class SolverTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        points, labels = _simplex_pool(rng, 3, 300)
        self.kdes = [KernelDensity(bandwidth=0.1).fit(points[labels == j]) for j in range(3)]
        pool, _ = _simplex_pool(rng, 3, 120)
        self.pool = pool
        self.log_densities = np.stack([kde.score_samples(pool) for kde in self.kdes])
        self.bag_index = rng.integers(0, len(pool), size=(5, 200))
        self.counts = np.stack([np.bincount(index, minlength=len(pool)) for index in self.bag_index]).astype(float)

    def test_slsqp_is_quapy_kdeyml(self):
        quantifier = KDEyML(LogisticRegression(), bandwidth=0.1)
        quantifier.mix_densities = self.kdes
        estimates, success = slsqp_prevalences(self.log_densities, self.bag_index)
        for s, index in enumerate(self.bag_index):
            with self.subTest(bag=s):
                np.testing.assert_array_equal(estimates[s], quantifier.aggregate(self.pool[index]))
        self.assertTrue(success.all())

    def test_slsqp_and_em_reach_the_same_maximum(self):
        slsqp, _ = slsqp_prevalences(self.log_densities, self.bag_index)
        em, converged = em_prevalences(self.log_densities, self.counts)
        self.assertTrue(converged.all())
        np.testing.assert_allclose(slsqp, em, atol=1e-3)
        ll_em = bag_log_likelihoods(self.log_densities, self.counts, em)
        ll_slsqp = bag_log_likelihoods(self.log_densities, self.counts, slsqp)
        # EM, with its tight tolerance, is at least as good a maximiser up to rounding
        self.assertTrue(np.all(ll_em >= ll_slsqp - 1e-6))

    def test_bag_log_likelihoods(self):
        estimates = np.random.default_rng(1).dirichlet(np.ones(3), size=5)
        expected = [_log_likelihood(p, self.log_densities, c) for p, c in zip(estimates, self.counts, strict=True)]
        np.testing.assert_allclose(bag_log_likelihoods(self.log_densities, self.counts, estimates), expected)
        # a slightly negative component, as SLSQP may return, counts as zero
        estimates[0] = [0.6, 0.4 + 1e-17, -1e-17]
        self.assertTrue(np.isfinite(bag_log_likelihoods(self.log_densities, self.counts, estimates)).all())


class MethodTest(unittest.TestCase):
    def test_cal_solver_is_checked(self):
        self.assertEqual(CAL_SOLVERS, ("em", "slsqp"))
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, bandwidth="cal", cal_solver="newton")
        self.assertEqual(BayCAPKDEy(vanilla_acc, bandwidth="cal").cal_solver, "em")

    def test_solver_check_reproduces_the_selection_of_both_solvers(self):
        check = _method(bandwidth_grid=GRID).solver_check(DATA.val, DATA.val_posteriors)
        self.assertFalse(hasattr(_method(bandwidth_grid=GRID), "cal_bandwidth_"))
        for solver in CAL_SOLVERS:
            method = _method(bandwidth_grid=GRID, cal_solver=solver).fit(DATA.val, DATA.val_posteriors)
            with self.subTest(solver=solver):
                np.testing.assert_array_equal(check[f"{solver}_curve"], method.cal_curve_)
                np.testing.assert_array_equal(check[f"{solver}_estimates"], method.cal_estimates_)
                self.assertEqual(check[f"{solver}_bandwidth"], method.cal_bandwidth_)
                self.assertEqual(check[f"{solver}_default"], method.cal_default_)
                self.assertEqual(check[f"{solver}_converged"].shape, (len(GRID), 8))
                self.assertEqual(check[f"{solver}_seconds"].shape, (len(GRID),))
                self.assertTrue(np.all(check[f"{solver}_seconds"] > 0))
        np.testing.assert_allclose(check["em_estimates"], check["slsqp_estimates"], atol=1e-2)
        np.testing.assert_array_equal(check["prevalences"], method.cal_prevalences_)

    def test_solver_check_needs_cal(self):
        with self.assertRaises(RuntimeError):
            BayCAPKDEy(vanilla_acc, bandwidth=0.1).solver_check(DATA.val, DATA.val_posteriors)


class _P:
    """The fields of a ``PretrainInfo`` that the gate reads."""

    def __init__(self, dataset: str, classifier: str, n_classes: int):
        self.domain = "classic"
        self.d_info = type("D", (), {"name": dataset, "n_classes": n_classes})()
        self.h_info = type("H", (), {"name": classifier, "full_name": classifier})()


class TableTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        check = _method(bandwidth_grid=GRID).solver_check(DATA.val, DATA.val_posteriors)
        # a second pair where SLSQP selects another bandwidth, by a large gap, and takes much longer
        other = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in check.items()}
        other["slsqp_bandwidth"] = GRID[0] if check["em_bandwidth"] != GRID[0] else GRID[-1]
        other["slsqp_curve"] = other["slsqp_curve"] + 0.01 * (np.asarray(GRID) != other["slsqp_bandwidth"])
        other["slsqp_curve"][list(GRID).index(other["slsqp_bandwidth"])] -= 0.01
        other["slsqp_seconds"] = other["slsqp_seconds"] + 100.0
        frames = {("toy", "LR", 3): (check, 0.5), ("toy", "kNN", 3): (other, 0.5)}
        cls.frames = {}
        for (dataset, classifier, k), (result, refit) in frames.items():
            frame = gate_run.solver_frame(_P(dataset, classifier, k), result, refit)
            cls.frames[classifier] = frame
            path = os.path.join(
                cls.root,
                gate_run.EXPERIMENT,
                "classic",
                gate_run.ACC_FOLDER,
                dataset,
                classifier,
                f"{gate_run.METHOD}.parquet",
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            frame.save_result(path)
        stored = pd.DataFrame(
            {
                "dataset": ["toy"] * 2,
                "classifier": ["LR", "kNN"],
                "test_seconds": [1000.0, 1000.0],
                "pilot_curve": [check["em_curve"], check["em_curve"]],
            }
        )
        with patch.object(kcs, "stored_pilot", return_value=stored):
            cls.table = kcs.pair_table(cls.root)
        cls.verdict = kcs.gate(cls.table)

    def test_rows(self):
        frame = self.frames["LR"]
        self.assertEqual(len(frame), len(GRID) * 8)
        self.assertTrue(set(gate_run.ROW_COLUMNS) | set(gate_run.PAIR_COLUMNS) <= set(frame.columns))

    def test_pair_table(self):
        table = self.table.set_index("classifier")
        same, other = table.loc["LR"], table.loc["kNN"]
        self.assertTrue(same["same_selection"])
        self.assertEqual(same["tie_gap"], 0.0)
        self.assertEqual(same["pilot_curve_max_diff"], 0.0)
        self.assertFalse(other["same_selection"])
        self.assertGreater(other["tie_gap"], kcs.TIE_TOL)
        self.assertNotEqual(other["steps"], 0)
        for row in (same, other):
            self.assertAlmostEqual(
                row["run_seconds"], row["shared_seconds"] + row["em_seconds"] + row["refit_seconds"] + 1000.0
            )
        self.assertGreater(other["extra_seconds"], 100.0 * len(GRID) - 1)

    def test_gate(self):
        self.assertEqual(self.verdict["pairs"], 2)
        self.assertEqual(self.verdict["pairs_same_selection"], 1)
        self.assertFalse(self.verdict["a_selection"])
        # 300 extra seconds over about 2000 of run time: 15% > 10%
        self.assertFalse(self.verdict["b_time"])
        self.assertFalse(self.verdict["passed"])
        passing = kcs.gate(self.table.loc[self.table["classifier"] == "LR"])
        self.assertTrue(passing["a_selection"])
        self.assertEqual(passing["b_time"], passing["extra_share"] <= kcs.TIME_SHARE)

    def test_latex(self):
        latex = kcs._pairs_styler(self.table).to_latex()
        self.assertIn("SLSQP", latex)
        self.assertIn("E7.2a", kcs.caption(self.verdict))


if __name__ == "__main__":
    unittest.main()
