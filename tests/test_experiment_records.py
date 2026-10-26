import unittest
from types import SimpleNamespace

import numpy as np

from uncawe.experiments.records import EvaluationContext, build_result_frame


class ExperimentRecordsTest(unittest.TestCase):
    def test_result_frame_preserves_main_schema_and_scalar_metadata(self):
        pretrain = SimpleNamespace(
            d_info=SimpleNamespace(name="dataset", collection="uci_binary", n_classes=2),
            h_info=SimpleNamespace(name="LR"),
        )
        context = EvaluationContext(
            pretrain=pretrain,
            shifts=[0.1, 0.2],
            true_cts=[np.eye(2), np.eye(2)],
            train_prev=0.4,
            val_prev=0.6,
        )
        frame = build_result_frame(
            context,
            method_name="method",
            acc_name="metric",
            acc_fn=lambda ct: 0.5,
            estim_acc_ranges=[np.array([0.4, 0.6]), np.array([0.5, 0.7])],
            estim_accs=[0.5, 0.6],
            estim_cts=[None, None],
            t_train=1.0,
            t_test_ave=0.1,
        )

        self.assertEqual(
            list(frame.columns),
            [
                "shifts",
                "true_cts",
                "estim_accs",
                "estim_acc_ranges",
                "acc_err",
                "estim_cts",
                "classifier",
                "method",
                "dataset",
                "collection",
                "n_classes",
                "acc_name",
                "train_prev",
                "val_prev",
                "t_train",
                "t_test_ave",
            ],
        )
        self.assertEqual(frame["method"].tolist(), ["method", "method"])
        self.assertEqual(frame["train_prev"].tolist(), [0.4, 0.4])
        np.testing.assert_allclose(frame["acc_err"].tolist(), [0.0, 0.1])

    def test_extra_columns_are_appended_per_bag(self):
        pretrain = SimpleNamespace(
            d_info=SimpleNamespace(name="dataset", collection="uci_binary", n_classes=2),
            h_info=SimpleNamespace(name="LR"),
        )
        context = EvaluationContext(
            pretrain=pretrain, shifts=[0.1, 0.2], true_cts=[np.eye(2), np.eye(2)], train_prev=0.4, val_prev=0.6
        )
        frame = build_result_frame(
            context,
            method_name="method",
            acc_name="metric",
            acc_fn=lambda ct: 0.5,
            estim_acc_ranges=[np.array([0.4, 0.6]), np.array([0.5, 0.7])],
            estim_accs=[0.5, 0.6],
            estim_cts=[None, None],
            t_train=1.0,
            t_test_ave=0.1,
            extra_columns={"nuts_divergences": [0, 3], "nuts_min_ess": [400.0, 12.0]},
        )

        self.assertEqual(list(frame.columns[-2:]), ["nuts_divergences", "nuts_min_ess"])
        self.assertEqual(frame["nuts_divergences"].tolist(), [0, 3])
        self.assertEqual(frame["nuts_min_ess"].tolist(), [400.0, 12.0])


if __name__ == "__main__":
    unittest.main()
