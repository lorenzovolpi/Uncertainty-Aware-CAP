import unittest
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch

from uncawe.experiments.execution import ExperimentOutcome, OutcomeStatus, run_tasks, save_experiment_results


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
    def test_serial_runner_passes_the_task_count(self, save):
        run_tasks(Mock(), ["first", "second"], worker=lambda task: [task], n_jobs=1, experiment="main")

        self.assertEqual([call.kwargs["done"] for call in save.call_args_list], [0, 1])
        self.assertEqual({call.kwargs["total"] for call in save.call_args_list}, {2})

    def test_result_lines_end_with_the_task_count(self):
        log = Mock()
        exists = [ExperimentOutcome.EXISTS(self.pretrain, metric, "method") for metric in ("acc", "f1")]
        error = [ExperimentOutcome.ERROR(ValueError("failed"), self.pretrain, "acc", "method")]

        save_experiment_results(log, [exists, error], experiment="main", total=5, done=2)

        infos = [call.args[0] for call in log.info.call_args_list]
        self.assertEqual(infos, [f"[LR@dataset] method on {m} exists, skipping [3/5]" for m in ("acc", "f1")])
        self.assertTrue(log.warning.call_args.args[0].endswith("skipping [4/5]"))

    def test_result_lines_have_no_count_without_total(self):
        log = Mock()
        save_experiment_results(log, [[ExperimentOutcome.EXISTS(self.pretrain, "acc", "method")]], experiment="main")

        self.assertEqual(log.info.call_args.args[0], "[LR@dataset] method on acc exists, skipping")

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
        save.assert_called_once_with(ANY, ANY, experiment="main", total=2)
        self.assertEqual(list(save.call_args.args[1]), [["result"]])


if __name__ == "__main__":
    unittest.main()
