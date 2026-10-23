import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from uncawe.experiments import main
from uncawe.experiments.records import EvaluationContext


class _Protocol:
    def __init__(self, samples):
        self.samples = samples

    def __call__(self):
        return iter(self.samples)

    def total(self):
        return len(self.samples)


class _Interval:
    def __init__(self, value):
        self.samples = np.array([value])
        self.point_estimate = value


class _ContingencyMethod:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0
        self.switch_calls = 0

    def fit(self, validation, posteriors):
        self.fit_calls += 1
        return self

    def predict_ct_range(self, X, posteriors):
        self.predict_calls += 1
        return np.array([np.eye(2)])

    def switch(self, acc_fn):
        self.switch_calls += 1

    def ci_from_cts(self, cts):
        return _Interval(0.5)


class _DirectMethod:
    def __init__(self):
        self.fit_calls = 0
        self.predict_calls = 0

    def switch_and_fit(self, acc_fn, validation, posteriors):
        self.fit_calls += 1
        return self

    def predict_range(self, X, posteriors):
        self.predict_calls += 1
        return np.array([0.5])

    def ci_from_accs(self, values):
        return _Interval(0.5)


class MainProtocolTest(unittest.TestCase):
    def setUp(self):
        self.pretrain = SimpleNamespace(
            d_info=SimpleNamespace(name="dataset", collection="uci", n_classes=2),
            h_info=SimpleNamespace(name="LR", full_name="LR"),
        )
        samples = [SimpleNamespace(X=np.zeros((1, 1))) for _ in range(2)]
        self.dataset = SimpleNamespace(
            test_prot=_Protocol(samples),
            test_prot_posteriors=[np.ones((1, 2)), np.ones((1, 2))],
        )
        self.context = EvaluationContext(
            pretrain=self.pretrain,
            shifts=[0.1, 0.2],
            true_cts=[np.eye(2), np.eye(2)],
            train_prev=0.5,
            val_prev=0.5,
        )

    @patch("uncawe.experiments.main._result_path", return_value="result.parquet")
    @patch("uncawe.experiments.main.os.path.exists", return_value=False)
    @patch("uncawe.experiments.main.build_result_frame", return_value=Mock())
    def test_contingency_method_fits_and_predicts_once_across_metrics(self, build, _exists, _path):
        method = _ContingencyMethod()
        results = main._evaluate_contingency_method(
            self.pretrain,
            "method",
            method,
            self.dataset,
            Mock(),
            Mock(),
            self.context,
        )

        self.assertEqual(method.fit_calls, 1)
        self.assertEqual(method.predict_calls, 2)
        self.assertEqual(method.switch_calls, 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(build.call_count, 3)

    @patch("uncawe.experiments.main._result_path", return_value="result.parquet")
    @patch("uncawe.experiments.main.os.path.exists", return_value=False)
    @patch("uncawe.experiments.main.build_result_frame", return_value=Mock())
    def test_direct_method_refits_and_predicts_for_each_metric(self, build, _exists, _path):
        method = _DirectMethod()
        results = main._evaluate_direct_method(
            self.pretrain,
            "method",
            method,
            self.dataset,
            Mock(),
            Mock(),
            self.context,
        )

        self.assertEqual(method.fit_calls, 3)
        self.assertEqual(method.predict_calls, 6)
        self.assertEqual(len(results), 3)
        self.assertEqual(build.call_count, 3)


class MainSchedulingTest(unittest.TestCase):
    def test_pools_get_their_workers_and_environments(self):
        info = SimpleNamespace(
            domain="classic", d_info=SimpleNamespace(name="a"), h_info=SimpleNamespace(name="LR", full_name="LR")
        )
        methods = [("DoC", "doc"), ("BayesCAP", "pop")]
        with (
            patch.object(main, "load_info_paths", return_value=["a_LR"]),
            patch.object(main.PretrainInfo, "load", return_value=info),
            patch.object(main, "get_filtered_datasets", return_value=["a"]),
            patch.object(main, "all_results_exist", return_value=False),
            patch.object(main, "gen_methods", return_value=methods),
            patch.object(main, "capenv", {"N_JOBS": 32}),
            patch.object(main, "run_experiment_args") as run,
        ):
            main.experiments(Mock(), "classic")

        regular, bayes = run.call_args_list
        self.assertEqual([args[1] for args in regular.args[1]], ["DoC"])
        self.assertEqual(regular.kwargs, {"n_jobs": 32, "env": main.THREAD_LIMITS})
        self.assertEqual([args[1] for args in bayes.args[1]], ["BayesCAP"])
        self.assertEqual(bayes.kwargs, {"n_jobs": main.BAYES_N_JOBS, "env": main.BAYES_ENV})
        self.assertEqual(main.BAYES_N_JOBS, 16)
        self.assertEqual(main.BAYES_ENV["JAX_PLATFORMS"], "cpu")
        self.assertIn("--xla_cpu_multi_thread_eigen=false", main.BAYES_ENV["XLA_FLAGS"])
        self.assertTrue(all(main.BAYES_ENV[name] == "1" for name in main.THREAD_LIMITS))


class MainRunFilterTest(unittest.TestCase):
    def _run(self, **filters):
        infos = {
            f"{d}_{h}": SimpleNamespace(
                domain="classic", d_info=SimpleNamespace(name=d), h_info=SimpleNamespace(name=h, full_name=h)
            )
            for d in ("a", "b", "c")
            for h in ("LR", "kNN")
        }
        methods = [("DoC", "doc"), ("BayesCAP", "pop"), ("BayesCAP-real", "real")]
        with (
            patch.object(main, "load_info_paths", return_value=list(infos)),
            patch.object(main.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
            patch.object(main, "get_filtered_datasets", return_value=["a", "b"]),
            patch.object(main, "all_results_exist", return_value=False) as exists,
            patch.object(main, "gen_methods", return_value=methods),
            patch.object(main, "get_method_names", return_value=[name for name, _ in methods]),
            patch.object(main, "capenv", {"N_JOBS": 1}),
            patch.object(main, "run_experiment_args") as run,
        ):
            main.experiments(Mock(), "classic", **filters)
        (_, regular), (_, bayes) = (call.args for call in run.call_args_list)
        return regular, bayes, exists

    def test_default_runs_every_filtered_pair_and_method(self):
        regular, bayes, _ = self._run()
        self.assertEqual(len(regular), 4)
        self.assertEqual(len(bayes), 8)
        self.assertEqual({args[1] for args in bayes}, {"BayesCAP", "BayesCAP-real"})

    def test_pilot_filters_pairs_and_methods(self):
        regular, bayes, exists = self._run(datasets=["b", "c"], classifiers=["kNN"], methods=["BayesCAP-real"])
        self.assertEqual(regular, [])
        self.assertEqual(bayes, [("b_kNN", "BayesCAP-real", "real")])
        self.assertEqual(exists.call_args.args[3], ["BayesCAP-real"])

    def test_unknown_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "BayesCAP-rea"):
            main.experiments(Mock(), "classic", methods=["BayesCAP-rea"])


if __name__ == "__main__":
    unittest.main()
