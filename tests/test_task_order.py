"""Longest-first ordering of the tasks of main from stored times (uncawe.experiments.task_order)."""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from uncawe.experiments import main
from uncawe.experiments.task_order import (
    TaskKey,
    default_reference_experiments,
    estimate_costs,
    sort_longest_first,
    stored_cost,
)

ACCS = ["vanilla_accuracy", "macro-F1", "macro-K"]


def _write(root, experiment, key: TaskKey, acc, t_train, t_bag, n_bags=100):
    d = os.path.join(root, experiment, key.domain, acc, key.dataset, key.classifier)
    os.makedirs(d, exist_ok=True)
    pd.DataFrame({"t_train": [t_train] * n_bags, "t_test_ave": [t_bag] * n_bags}).to_parquet(
        os.path.join(d, f"{key.method}.parquet")
    )


def _key(dataset="chess", classifier="LR", method="O-LEAP"):
    return TaskKey("classic", dataset, classifier, method)


class TestStoredCost(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_shared_times_counted_once(self):
        # a contingency-table method writes the same times in the three metric files
        key = _key()
        for acc in ACCS:
            _write(self.root, "main", key, acc, 10.0, 2.0)
        self.assertAlmostEqual(stored_cost(key, ACCS, ["main"], self.root), 10.0 + 100 * 2.0)

    def test_per_metric_times_summed(self):
        # a direct method fits per metric and writes different times
        key = _key(method="QuAcc_NxN")
        for i, acc in enumerate(ACCS):
            _write(self.root, "main", key, acc, 1.0 + i, 1.0)
        self.assertAlmostEqual(stored_cost(key, ACCS, ["main"], self.root), 3 * 100.0 + 1.0 + 2.0 + 3.0)

    def test_first_reference_with_the_task_wins(self):
        key = _key()
        _write(self.root, "main_pre_fix3", key, ACCS[0], 0.0, 5.0)
        _write(self.root, "main_old", key, ACCS[0], 0.0, 9.0)
        self.assertAlmostEqual(stored_cost(key, ACCS, ["main", "main_pre_fix3", "main_old"], self.root), 500.0)
        self.assertIsNone(stored_cost(_key(dataset="letter"), ACCS, ["main", "main_pre_fix3"], self.root))

    def test_default_references(self):
        for d in ("main", "main_pre_fix8", "main_pre_fix3", "binform", "mainly"):
            os.makedirs(os.path.join(self.root, d))
        self.assertEqual(default_reference_experiments(self.root), ["main", "main_pre_fix3", "main_pre_fix8"])


class TestOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        _write(self.root, "main_pre_fix3", _key("chess", "LR"), ACCS[0], 0.0, 8.0)
        _write(self.root, "main_pre_fix3", _key("chess", "kNN"), ACCS[0], 0.0, 6.0)
        _write(self.root, "main_pre_fix3", _key("german", "LR"), ACCS[0], 0.0, 1.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fallback_to_other_classifiers(self):
        keys = [_key("chess", "SVM"), _key("letter", "LR")]
        costs, sources = estimate_costs(
            keys + [_key("chess", "LR"), _key("chess", "kNN")], ACCS, ["main", "main_pre_fix3"], self.root
        )
        self.assertEqual(sources, ["classifiers", "unknown", "stored", "stored"])
        self.assertAlmostEqual(costs[0], 700.0)
        self.assertIsNone(costs[1])

    def test_longest_first_unknown_first_stable(self):
        tasks = [
            ("german", "LR"),  # 100 s
            ("chess", "kNN"),  # 600 s
            ("letter", "LR"),  # unknown
            ("chess", "LR"),  # 800 s
            ("wine", "RF"),  # unknown
        ]
        order = sort_longest_first(
            tasks, lambda t: _key(t[0], t[1]), ACCS, ["main", "main_pre_fix3"], log=Mock(), root_dir=self.root
        )
        self.assertEqual(order, [("letter", "LR"), ("wine", "RF"), ("chess", "LR"), ("chess", "kNN"), ("german", "LR")])
        self.assertEqual(sorted(order), sorted(tasks))


class TestMainOption(unittest.TestCase):
    def _run(self, **kwargs):
        paths = ["a", "b"]
        infos = {
            "a": Mock(d_info=Mock(name="d"), h_info=Mock(), domain="classic"),
            "b": Mock(d_info=Mock(name="d"), h_info=Mock(), domain="classic"),
        }
        with (
            patch.object(main, "load_info_paths", return_value=paths),
            patch.object(main.PretrainInfo, "load", side_effect=lambda p, fast=True: infos[p]),
            patch.object(main, "get_filtered_datasets", return_value=None),
            patch.object(main, "all_results_exist", return_value=False),
            patch.object(main, "gen_methods", return_value=[("M", object())]),
            patch.object(main, "get_method_names", return_value=["M"]),
            patch.object(main, "run_experiment_args") as run,
            patch.object(main, "sort_longest_first", side_effect=lambda tasks, *a, **k: list(reversed(tasks))) as srt,
            patch.object(main, "default_reference_experiments", return_value=["main", "main_x"]),
        ):
            for info in infos.values():
                info.d_info.name = "d"
            with patch.object(main, "get_filtered_datasets", return_value=["d"]):
                main.experiments(Mock(), "classic", **kwargs)
        return run, srt

    def test_default_keeps_order(self):
        run, srt = self._run()
        srt.assert_not_called()
        self.assertEqual([t[0] for t in run.call_args_list[0].args[1]], ["a", "b"])

    def test_option_sorts_each_pool(self):
        run, srt = self._run(longest_first=[])
        self.assertEqual(srt.call_count, 2)
        self.assertEqual(srt.call_args_list[0].args[3], ["main", "main_x"])
        self.assertEqual([t[0] for t in run.call_args_list[0].args[1]], ["b", "a"])

    def test_option_with_folders(self):
        _, srt = self._run(longest_first=["main_pre_fix3"])
        self.assertEqual(srt.call_args_list[0].args[3], ["main_pre_fix3"])


if __name__ == "__main__":
    unittest.main()
