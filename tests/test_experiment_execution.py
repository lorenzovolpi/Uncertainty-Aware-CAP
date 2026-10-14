import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from uncawe.experiments.execution import ExperimentOutcome, OutcomeStatus, run_tasks


class ExperimentOutcomeTest(unittest.TestCase):
    def setUp(self):
        self.pretrain = SimpleNamespace(
            domain="classic",
            d_info=SimpleNamespace(name="dataset"),
            h_info=SimpleNamespace(name="LR", full_name="LR"),
        )

    def test_status_constructors_preserve_legacy_codes(self):
        success = ExperimentOutcome.SUCCESS(self.pretrain, "metric", "method")
        exists = ExperimentOutcome.EXISTS(self.pretrain, "metric", "method")
        error = ExperimentOutcome.ERROR(ValueError("failed"), self.pretrain, "metric", "method")

        self.assertEqual(success.code, OutcomeStatus.SUCCESS)
        self.assertEqual(exists.code, OutcomeStatus.EXISTS)
        self.assertEqual(error.code, OutcomeStatus.ERROR)
        self.assertTrue(success.ok)
        self.assertTrue(exists.old)
        self.assertTrue(error.error())

    @patch("uncawe.experiments.execution.save_experiment_results")
    def test_serial_runner_saves_each_task_immediately(self, save):
        worker = Mock(side_effect=lambda task: [task])
        tasks = ["first", "second"]

        run_tasks(Mock(), tasks, worker=worker, n_jobs=1, experiment="main")

        self.assertEqual(worker.call_args_list[0].args, ("first",))
        self.assertEqual(worker.call_args_list[1].args, ("second",))
        self.assertEqual(save.call_count, 2)
        self.assertEqual(save.call_args_list[0].args[1], [["first"]])

    @patch("uncawe.experiments.execution.save_experiment_results")
    @patch("uncawe.experiments.execution.parallel")
    def test_parallel_runner_preserves_scheduling_arguments(self, parallel, save):
        worker = Mock()
        tasks = ["first", "second"]
        generated = iter([["result"]])
        parallel.return_value = generated

        run_tasks(Mock(), tasks, worker=worker, n_jobs=3, experiment="main")

        parallel.assert_called_once_with(
            func=worker,
            args_list=tasks,
            n_jobs=3,
            return_as="generator_unordered",
            max_nbytes=None,
        )
        save.assert_called_once_with(ANY, generated, experiment="main")


if __name__ == "__main__":
    unittest.main()
