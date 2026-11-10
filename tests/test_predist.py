"""Tests of the predist distance diagnostic after WP-16 (E6.1): pairs, quantifier, true-space distance, tables."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import quapy as qp
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC
from sklearn.neural_network import MLPClassifier

from uncawe.experiments import predist, predist_sampling
from uncawe.experiments.methods import gen_methods
from uncawe.experiments.metrics import acc_from_ct
from uncawe.experiments.predist_config import DEFAULT_CLASSIFIERS, PREDIQUANT_ALPHA, SAMPLE_REGIONS
from uncawe.experiments.predist_sampling import TestSample
from uncawe.tables import predist as predist_tables


def _info(dataset, classifier, n_classes=3, collection="uci_multiclass"):
    return SimpleNamespace(
        d_info=SimpleNamespace(name=dataset, n_classes=n_classes, collection=collection),
        h_info=SimpleNamespace(name=classifier, full_name=classifier),
    )


class PretrainSelectionTest(unittest.TestCase):
    def test_one_pretrain_per_dataset_classifier_pair(self):
        infos = [_info(d, h) for d in ("cmc", "mhr", "letter") for h in (*DEFAULT_CLASSIFIERS, "XGB")]
        infos.append(_info("cmc", "LR", n_classes=2))
        infos.append(_info("mhr", "LR", collection="uci_binary"))
        paths = list(range(len(infos)))
        with (
            mock.patch.object(predist_sampling, "load_info_paths", return_value=paths),
            mock.patch.object(predist_sampling.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
        ):
            pretrains = predist_sampling.load_filtered_pretrains("classic", ["RF", "LR", "kNN"], ["mhr", "cmc"])
        pairs = [(p.d_info.name, p.h_info.name) for p in pretrains]
        self.assertEqual(
            pairs, [("mhr", "RF"), ("mhr", "LR"), ("mhr", "kNN"), ("cmc", "RF"), ("cmc", "LR"), ("cmc", "kNN")]
        )

    def test_parse_classifiers(self):
        self.assertEqual(predist_sampling.parse_classifiers(None), list(DEFAULT_CLASSIFIERS))
        self.assertEqual(predist_sampling.parse_classifiers(["LR,kNN", "RF", "LR"]), ["LR", "kNN", "RF"])
        with self.assertRaises(ValueError):
            predist_sampling.parse_classifiers(["XGB"])

    def test_regions_of_the_test_bags(self):
        points = predist_sampling.simplex_points(np.random.RandomState(0), 5)
        self.assertEqual(len(points), len(SAMPLE_REGIONS))
        self.assertEqual(SAMPLE_REGIONS[0], "centre")
        np.testing.assert_allclose(points[0], np.full(3, 1 / 3))
        for sample_id in (1, 2, 3):
            self.assertEqual(SAMPLE_REGIONS[sample_id], "vertex")
            self.assertAlmostEqual(points[sample_id].max(), 0.99)


class PrediQuantSettingsTest(unittest.TestCase):
    def test_quantifier_and_settings_of_pq(self):
        method = predist.build_prediquant(lambda ct: 0.0, 1000, 1000, "hellinger", seed=0)
        pq = dict(gen_methods())["PQ"]
        self.assertIsInstance(method.q, PACC)
        self.assertIsInstance(method.q.classifier, MLPClassifier)
        self.assertEqual(method.q.classifier.random_state, qp.environ["_R_SEED"])
        for name in ("alpha", "num_samples", "sample_size", "predict_train_prev", "rho"):
            with self.subTest(setting=name):
                self.assertEqual(getattr(method, name), getattr(pq, name))
        self.assertEqual(PREDIQUANT_ALPHA, pq.alpha)
        self.assertIs(method.error, predist.PREDIQUANT_DISTANCE_FNS["hellinger"])
        self.assertEqual(method.random_state, 0)

    def test_prediction_uses_the_selected_bags(self):
        rng = np.random.RandomState(0)
        n = 300
        y = np.repeat([0, 1, 2], n // 3)
        X = rng.normal(size=(n, 2)) + 2.0 * np.eye(3)[y][:, :2]
        posteriors = rng.dirichlet(np.ones(3), size=n)
        val = LabelledCollection(X, y, classes=[0, 1, 2])
        acc_fn = lambda ct: acc_from_ct("vanilla_accuracy", ct, type="selection")
        method = predist.build_prediquant(acc_fn, num_samples=10, sample_size=30, distance="l1", seed=0)
        method.q.classifier.set_params(max_iter=50)
        method.fit(val, posteriors)
        test = val.sampling_from_index(np.arange(0, n, 7))
        sample = TestSample(0, np.full(3, 1 / 3), test.prevalence(), test, posteriors[: len(test)])
        accs, test_prev, val_prevs, val_true_prevs = predist._prediquant_predict_with_val_prevs(method, sample)
        idx = method.last_selected_bags
        self.assertEqual(len(idx), 10)
        np.testing.assert_allclose(accs, [acc_fn(method.sigma_ct[i]) for i in idx])
        np.testing.assert_allclose(test_prev, method.q.predict(test.X))
        np.testing.assert_allclose(val_prevs, np.asarray(method.sigma_pred_prevs)[idx])
        np.testing.assert_allclose(val_true_prevs, np.asarray(method.sigma_true_prevs)[idx])
        # the selected bags are the closest candidates in L1 to the estimated prevalence of the test bag
        dists = np.abs(np.asarray(method.sigma_pred_prevs) - test_prev).sum(axis=1)
        self.assertLessEqual(dists[idx].max(), np.delete(dists, idx).min() + 1e-12)


class TrueSpaceDistanceTest(unittest.TestCase):
    def test_mean_l1_to_the_test_prevalence(self):
        actual = np.array([0.5, 0.3, 0.2])
        selected = np.array([[0.5, 0.3, 0.2], [0.6, 0.2, 0.2], [0.2, 0.3, 0.5]])
        # L1 distances 0, 0.2 and 0.6
        self.assertAlmostEqual(predist.true_space_l1(selected, actual), 0.8 / 3)

    def test_record_stores_the_new_columns(self):
        y = np.array([0, 0, 1, 2])
        posteriors = np.eye(3)[[0, 1, 1, 2]]
        collection = LabelledCollection(np.zeros((4, 1)), y, classes=[0, 1, 2])
        sample = TestSample(4, np.full(3, 1 / 3), collection.prevalence(), collection, posteriors)
        p = SimpleNamespace(
            d_info=SimpleNamespace(name="cmc", collection="uci_multiclass"),
            h_info=SimpleNamespace(name="kNN", full_name="kNN"),
        )
        val_true = np.array([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])
        row = predist._record(
            p,
            np.full(3, 1 / 3),
            np.full(3, 1 / 3),
            "vanilla_accuracy",
            "l1",
            sample,
            np.array([0.4, 0.3, 0.3]),
            val_true,
            val_true,
            np.array([0.7, 0.8, 0.75]),
            0.0,
        )
        self.assertEqual(row["quantifier"], "PACC(MLP)")
        self.assertEqual(row["region"], "random")
        self.assertEqual(row["schema_version"], predist.RESULT_SCHEMA_VERSION)
        self.assertAlmostEqual(row["true_dist_l1_mean"], 0.25)
        ct = contingency_table(y, np.argmax(posteriors, axis=-1), 3)
        self.assertAlmostEqual(row["true_acc"], acc_from_ct("vanilla_accuracy", ct, type="evaluation"))


def _synthetic_results(winkler: dict) -> pd.DataFrame:
    """Rows of ``predist`` with the given per-bag Winkler scores {(pair, distance): [scores of the bags]}."""
    rows = []
    for ((dataset, classifier), distance), scores in winkler.items():
        for sample_id, score in enumerate(scores):
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "distance": distance,
                    "sample_id": sample_id,
                    "region": SAMPLE_REGIONS[sample_id],
                    "acc_abs_error": score / 10,
                    "ci_coverage": float(score < 0.5),
                    "ci_amplitude": score / 2,
                    "ci_winkler": score,
                    "true_dist_l1_mean": score / 5,
                }
            )
    return pd.DataFrame(rows)


class PredistTablesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        winkler = {}
        self.pairs = [(f"d{i}", h) for i in range(4) for h in ("LR", "RF")]
        for pair in self.pairs:
            base = rng.uniform(0.3, 1.0, size=5)
            winkler[(pair, "l1")] = base
            winkler[(pair, "hellinger")] = base - 0.1  # better than L1 on every bag
            winkler[(pair, "jensen-shannon")] = base + rng.normal(0, 0.05, size=5)
        self.winkler = winkler
        self.df = _synthetic_results(winkler)

    def test_summary(self):
        pairs, summary, region = predist_tables.compute(self.df)
        self.assertEqual(len(pairs), 3 * len(self.pairs))
        summary = summary.set_index("distance")
        self.assertEqual(list(summary.index), ["l1", "hellinger", "jensen-shannon"])
        self.assertAlmostEqual(summary.loc["l1", "ci_winkler"], np.mean([self.winkler[(p, "l1")] for p in self.pairs]))
        self.assertEqual(summary.loc["hellinger", "n_best"], len(self.pairs))
        self.assertEqual(summary.loc["l1", "n_best"], 0)
        self.assertEqual(summary.loc["hellinger", "n_better_than_l1"], len(self.pairs))
        # 8 pairs all with the same sign: the exact two-sided p-value is 2 / 2^8
        self.assertAlmostEqual(summary.loc["hellinger", "p_wilcoxon"], 2 / 2**8)
        self.assertTrue(np.isnan(summary.loc["l1", "p_wilcoxon"]))
        self.assertEqual(int(summary.loc["l1", "n_pairs"]), len(self.pairs))
        self.assertEqual(list(region["region"].unique()), ["centre", "vertex", "random"])
        self.assertEqual(int(region.loc[(region.region == "vertex") & (region.distance == "l1"), "n_bags"].iloc[0]), 24)

    def test_decision_rule(self):
        _, summary, _ = predist_tables.compute(self.df)
        result = predist_tables.decision(summary)
        self.assertEqual(result["chosen"], "hellinger")
        self.assertEqual(result["qualified"], ["hellinger"])
        # without the better distance, L1 is kept
        _, summary, _ = predist_tables.compute(self.df.loc[self.df.distance != "hellinger"])
        result = predist_tables.decision(summary)
        self.assertEqual(result["chosen"], "l1")

    def test_incomplete_pairs_are_dropped(self):
        df = self.df.loc[~((self.df.dataset == "d0") & (self.df.distance == "hellinger") & (self.df.sample_id == 2))]
        kept = predist_tables.complete_pairs(df)
        # the two pairs of d0 miss one (distance, bag) cell, so they are dropped whole
        self.assertEqual(set(zip(kept.dataset, kept.classifier)), set(self.pairs) - {("d0", "LR"), ("d0", "RF")})

    def test_latex(self):
        _, summary, region = predist_tables.compute(self.df)
        latex = predist_tables.summary_latex(summary)
        self.assertIn("Hellinger", latex)
        self.assertIn("\\textbf{", latex)
        self.assertIn("vertices", predist_tables.region_latex(region))


if __name__ == "__main__":
    unittest.main()
