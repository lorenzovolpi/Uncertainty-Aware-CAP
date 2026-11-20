"""The temperature of BayesKDEy chosen on the calibration bags after the bandwidth (E7.2b of
``docs/ext7_kdey_bandwidth.md``, work-plan card WP-26)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
import quapy as qp
from cap.error import vanilla_acc
from cap.utils.commons import contingency_table
from test_baycap_fixed import DATA, SETTINGS
from test_kdey_cal import GRID, _method

from uncawe.experiments.methods import CONFIDENCE_LEVEL
from uncawe.methods.baycap_kdey import (
    CAL_SUMMARIES,
    CAL_TAU_CONFIDENCE,
    CAL_TAU_SUMMARIES,
    TAU_GRID,
    BayCAPKDEy,
    interval_scores,
    select_temperature_from_scores,
)
from uncawe.tables import main as main_tables
from uncawe.tables.tempering import TAUS, best_tau

qp.environ.setdefault("_R_SEED", 0)

# a small calibration: 4 bags of 200 instances
SMALL = {"cal_bags": 4}
TAUS_SMALL = (0.5, 1.0, 4.0)


def _tcal(**kwargs) -> BayCAPKDEy:
    return _method(**({"temperature": "cal", "bandwidth_grid": GRID} | SMALL | kwargs))


class RulesTest(unittest.TestCase):
    def test_grid_and_level_are_those_of_e5_3_and_main(self):
        self.assertEqual(TAU_GRID, TAUS)
        self.assertEqual(CAL_TAU_CONFIDENCE, CONFIDENCE_LEVEL)

    def test_interval_scores_are_those_of_main(self):
        rng = np.random.default_rng(5)
        accs = rng.beta(20, 5, size=400)
        for true_acc in (0.1, 0.8, 0.99, float(np.median(accs))):
            winkler, coverage = interval_scores(accs, true_acc)
            with self.subTest(true_acc=true_acc):
                self.assertEqual(winkler, main_tables.get_acc_winkler("", true_acc, accs))
                self.assertEqual(coverage, main_tables.get_acc_coverage("", true_acc, accs))

    def test_tie_rule_is_that_of_e5_3(self):
        cases = (
            [0.3, 0.2, 0.2, 0.4, 0.5],
            [0.2, 0.3, 0.3, 0.3, 0.2],
            [0.1, 0.3, 0.3, 0.3, 0.1],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [np.nan, 0.4, 0.4, np.nan, 0.4],
            [0.2, 0.3, 0.3, 0.2, 0.3],
        )
        for scores in cases:
            with self.subTest(scores=scores):
                index = select_temperature_from_scores(TAU_GRID, scores)
                self.assertEqual(TAU_GRID[index], best_tau(dict(zip(TAU_GRID, scores, strict=True))))
        with self.assertRaises(ValueError):
            select_temperature_from_scores(TAU_GRID, [np.nan] * 5)


class MethodTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.method = _tcal(temperature_grid=TAUS_SMALL).fit(DATA.val, DATA.val_posteriors)

    def test_cal_temperature_needs_cal_bandwidth_and_a_valid_grid(self):
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, temperature="cal")
        with self.assertRaises(ValueError):
            BayCAPKDEy(vanilla_acc, bandwidth="cal", temperature="hot")
        for grid in ([], [2.0, 1.0], [0.0, 1.0]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                BayCAPKDEy(vanilla_acc, bandwidth="cal", temperature="cal", temperature_grid=grid)
        method = BayCAPKDEy(vanilla_acc, bandwidth="cal", temperature="cal")
        self.assertTrue(method.calibrates_temperature)
        self.assertEqual(method._tau_grid(), TAU_GRID)
        self.assertFalse(BayCAPKDEy(vanilla_acc, bandwidth="cal").calibrates_temperature)

    def test_the_bandwidth_is_the_same_with_and_without_the_temperature_step(self):
        plain = _method(bandwidth_grid=GRID, **SMALL).fit(DATA.val, DATA.val_posteriors)
        self.assertEqual(self.method.cal_bandwidth_, plain.cal_bandwidth_)
        np.testing.assert_array_equal(self.method.cal_curve_, plain.cal_curve_)
        np.testing.assert_array_equal(self.method.cal_bag_index_, plain.cal_bag_index_)

    def test_selection_and_summaries(self):
        method = self.method
        shape = (len(TAUS_SMALL), SMALL["cal_bags"])
        self.assertIn(method.cal_temperature_, TAUS_SMALL)
        self.assertEqual(method.cal_tau_winkler_bags_.shape, shape)
        np.testing.assert_allclose(method.cal_tau_winkler_, method.cal_tau_winkler_bags_.mean(axis=1))
        self.assertEqual(
            method.cal_temperature_, TAUS_SMALL[select_temperature_from_scores(TAUS_SMALL, method.cal_tau_winkler_)]
        )
        self.assertTrue(np.all(np.isin(method.cal_tau_coverage_bags_, (0.0, 1.0))))
        self.assertEqual(method.quantifier_.temperature, method.cal_temperature_)
        self.assertEqual(method.quantifier_.bandwidth, method.cal_bandwidth_)
        method.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        summaries = method.last_posterior_summaries()
        self.assertTrue(set(CAL_SUMMARIES) | set(CAL_TAU_SUMMARIES) <= set(summaries))
        self.assertEqual(summaries["cal_temperature"], method.cal_temperature_)
        for name in CAL_TAU_SUMMARIES[1:]:
            with self.subTest(name=name):
                self.assertEqual(np.shape(summaries[name]), (len(TAUS_SMALL),))

    def test_the_true_accuracy_of_a_calibration_bag_is_that_of_its_realised_table(self):
        method = self.method
        _, cal_index = method.cal_split_
        labels = np.asarray(DATA.val.y)[cal_index]
        predicted = np.argmax(DATA.val_posteriors, axis=1)[cal_index]
        for s, index in enumerate(method.cal_bag_index_):
            table = contingency_table(labels[index], predicted[index], 3)
            self.assertAlmostEqual(method.cal_true_accs_[s], vanilla_acc(table), places=12)

    def test_wider_temperature_widens_the_calibration_intervals(self):
        # tau multiplies the posterior variance of pi: the coverage of the calibration intervals does not fall
        coverage = self.method.cal_tau_coverage_bags_.mean(axis=1)
        self.assertLessEqual(coverage[0], coverage[-1])

    def test_a_grid_of_1_reproduces_the_method_of_e7_2a(self):
        tcal = _tcal(temperature_grid=(1.0,)).fit(DATA.val, DATA.val_posteriors)
        cal = _method(bandwidth_grid=GRID, **SMALL).fit(DATA.val, DATA.val_posteriors)
        self.assertEqual(tcal.cal_temperature_, 1.0)
        np.testing.assert_array_equal(
            tcal.predict_ct_range(DATA.X_u, DATA.u_posteriors), cal.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        )

    def test_grids_of_one_value_reproduce_the_stored_tempered_method(self):
        for tau in (0.5, 4.0):
            with self.subTest(tau=tau):
                tcal = _tcal(bandwidth_grid=(0.1,), temperature_grid=(tau,)).fit(DATA.val, DATA.val_posteriors)
                stored = BayCAPKDEy(
                    vanilla_acc, estimand="realised", temperature=tau, **(SETTINGS | {"input_classifier": "h"})
                ).fit(DATA.val, DATA.val_posteriors)
                np.testing.assert_array_equal(
                    tcal.predict_ct_range(DATA.X_u, DATA.u_posteriors),
                    stored.predict_ct_range(DATA.X_u, DATA.u_posteriors),
                )

    def test_the_selection_uses_no_target_bag_and_is_reproducible(self):
        method = self.method
        scores, tau = method.cal_tau_winkler_.copy(), method.cal_temperature_
        for size in (80, 30):
            method.predict_ct_range(DATA.X_u[:size], DATA.u_posteriors[:size])
            summaries = method.last_posterior_summaries()
            np.testing.assert_array_equal(summaries["cal_tau_winkler"], scores)
            self.assertEqual(summaries["cal_temperature"], tau)
        again = _tcal(temperature_grid=TAUS_SMALL).fit(DATA.val, DATA.val_posteriors)
        np.testing.assert_array_equal(again.cal_tau_winkler_bags_, method.cal_tau_winkler_bags_)

    def test_mlp_input(self):
        method = _tcal(input_classifier="mlp", bandwidth_grid=(0.1,), temperature_grid=(1.0, 2.0), cal_bags=2)
        method.fit(DATA.val, DATA.val_posteriors)
        self.assertIn(method.cal_temperature_, (1.0, 2.0))
        self.assertTrue(np.all(np.isfinite(method.cal_tau_winkler_bags_)))


if __name__ == "__main__":
    unittest.main()
