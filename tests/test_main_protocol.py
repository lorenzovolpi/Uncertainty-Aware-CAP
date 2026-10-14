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


if __name__ == "__main__":
    unittest.main()
