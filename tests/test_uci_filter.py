"""FIX-8: the class filter of the UCI multiclass datasets (uncawe.pretrain.uci)."""

import os
import unittest

import numpy as np
import quapy as qp
from cap.environment import env
from quapy.data.datasets import UCI_MULTICLASS_DATASETS

from uncawe.pretrain.uci import (
    MAX_TRAIN_INSTANCES,
    QUAPY_MIN_CLASS_SUPPORT,
    class_support_threshold,
    fetch_uci_multiclass,
    min_class_support,
    n_train_instances,
)

POKER_HAND_N = 1_024_985  # size of poker_hand after QuaPy's default filter (100)


def _local(name: str) -> bool:
    return os.path.exists(os.path.join(env["QUAPY_DATA"], "uci_multiclass", f"{name}.pkl"))


class TestThreshold(unittest.TestCase):
    def test_n_train(self):
        self.assertEqual(n_train_instances(1000), 700)
        self.assertEqual(n_train_instances(POKER_HAND_N), MAX_TRAIN_INSTANCES)

    def test_uncapped_datasets_keep_quapy_threshold(self):
        # with n_train = 0.7 n the rule asks for ceil(20 / 0.7) = 29 < 100 instances
        for n in (1000, 5000, 20000, 35714):
            self.assertEqual(class_support_threshold(n), QUAPY_MIN_CLASS_SUPPORT)

    def test_poker_hand_threshold(self):
        # 10 * 1,024,985 / 12,500 = 819.988 -> 820
        self.assertEqual(class_support_threshold(POKER_HAND_N), 820)

    def test_threshold_gives_k_expected_instances_in_V(self):
        for n in (40_000, 67_557, 300_000, POKER_HAND_N):
            t = class_support_threshold(n, k=10)
            share = n_train_instances(n) * 0.5 / n
            self.assertGreaterEqual(t * share, 10 - 1e-9)
            if t > QUAPY_MIN_CLASS_SUPPORT:
                self.assertLess((t - 1) * share, 10)


@unittest.skipUnless(_local("poker_hand"), "poker_hand not in the local QuaPy data")
class TestPokerHand(unittest.TestCase):
    def setUp(self):
        self._seed = qp.environ.get("_R_SEED")
        qp.environ["_R_SEED"] = 0  # as in experiments/main.py

    def tearDown(self):
        if self._seed is None:
            qp.environ.pop("_R_SEED", None)
        else:
            qp.environ["_R_SEED"] = self._seed

    def test_threshold_and_classes(self):
        self.assertEqual(min_class_support("poker_hand"), 820)
        T, V, U = fetch_uci_multiclass("poker_hand")
        self.assertEqual(V.n_classes, 7)
        for coll in (T, V, U):
            np.testing.assert_array_equal(np.unique(coll.y), np.arange(7))
        self.assertGreaterEqual(V.counts().min(), 10)


class TestOtherDatasetsUnchanged(unittest.TestCase):
    def test_threshold_is_quapy_default(self):
        names = [d for d in UCI_MULTICLASS_DATASETS if d != "poker_hand" and _local(d)]
        if not names:
            self.skipTest("no UCI multiclass dataset in the local QuaPy data")
        for name in names:
            with self.subTest(dataset=name):
                self.assertEqual(min_class_support(name), QUAPY_MIN_CLASS_SUPPORT)


if __name__ == "__main__":
    unittest.main()
