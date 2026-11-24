"""Tests of E8.2 of docs/ext8_average_confidence.md (card WP-30): experiments/ctt.py, tables/ctt.py, plots/ctt.py."""

import os

# before any import of JAX (experiments/main.py imports it), as the other test modules
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import quapy as qp
from cap.error import vanilla_acc
from cap.models.cont_table import ContTableTransferCAP
from cap.utils.commons import contingency_table
from quapy.data import LabelledCollection

from uncawe.experiments import ctt
from uncawe.experiments.metrics import get_acc_names
from uncawe.experiments.records import EvaluationContext
from uncawe.tables import ctt as tables


def _data(n_classes: int = 3, seed: int = 0, n_val: int = 300, n_bags: int = 4, n_bag: int = 200):
    """A validation set and bags with shifted prevalences; features informative of the class; posteriors of a noisy
    classifier (seeded)."""
    rng = np.random.default_rng(seed)

    def draw(y):
        X = rng.normal(size=(len(y), 2)) + 2.0 * np.eye(n_classes)[y][:, :2] + 0.5 * y[:, None]
        logits = 2.0 * np.eye(n_classes)[y] + rng.normal(scale=1.5, size=(len(y), n_classes))
        post = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        return X, post

    y_val = rng.integers(0, n_classes, n_val)
    X_val, val_post = draw(y_val)
    validation = LabelledCollection(X_val, y_val, classes=np.arange(n_classes))
    bags, bag_posts, true_cts = [], [], []
    for _ in range(n_bags):
        prev = rng.dirichlet(np.ones(n_classes))
        y = rng.choice(n_classes, size=n_bag, p=prev)
        X, post = draw(y)
        bags.append(LabelledCollection(X, y, classes=np.arange(n_classes)))
        bag_posts.append(post)
        true_cts.append(contingency_table(y, post.argmax(axis=1), n_classes))
    dataset = SimpleNamespace(test_prot=lambda: iter(bags), test_prot_posteriors=bag_posts)
    return validation, val_post, dataset, true_cts


def _paper_rates(validation, val_post) -> np.ndarray:
    """$\\hat C_{ij}=N^V_{ij}/n^V_j$ in the paper's orientation (predicted classes on rows), from the counts of $V$."""
    n = len(validation.classes_)
    counts = np.zeros((n, n))
    for y, y_hat in zip(validation.y, val_post.argmax(axis=1)):
        counts[y_hat, y] += 1
    return counts / counts.sum(axis=0, keepdims=True)


class FixedQuantifier:
    """A quantifier that returns a fixed prevalence, with the interface ``ContTableTransferCAP`` uses."""

    def __init__(self, prev):
        self.prev = np.asarray(prev, dtype=float)

    def classifier_fit_predict(self, X, y):
        return None, None

    def aggregation_fit(self, predictions, y):
        return self

    def predict(self, X):
        return self.prev


class TestTables(unittest.TestCase):
    def test_transfer_equals_rates_times_prevalence(self):
        validation, val_post, _, _ = _data()
        rates = _paper_rates(validation, val_post)
        prev = np.array([0.5, 0.3, 0.2])
        method = ContTableTransferCAP(vanilla_acc, FixedQuantifier(prev)).fit(validation, val_post)
        table = method.predict_ct(np.zeros((1, 2)), None)
        # the code stores the transpose of the paper's table C diag(pi)
        np.testing.assert_allclose(table, (rates @ np.diag(prev)).T, atol=1e-12)
        np.testing.assert_allclose(table.sum(axis=1), prev, atol=1e-12)

    def test_transfer_with_validation_prevalence_is_validation_table(self):
        validation, val_post, _, _ = _data()
        method = ContTableTransferCAP(vanilla_acc, FixedQuantifier(validation.prevalence())).fit(validation, val_post)
        table = method.predict_ct(np.zeros((1, 2)), None)
        expected = contingency_table(validation.y, val_post.argmax(axis=1), 3)
        np.testing.assert_allclose(table, expected, atol=1e-12)

    def test_oracle_is_rates_times_bag_prevalence(self):
        validation, val_post, _, true_cts = _data()
        rates = _paper_rates(validation, val_post)
        table, val_prev = ctt.validation_table(validation, val_post)
        for ct in true_cts:
            bag_prev = ct.sum(axis=1)
            np.testing.assert_allclose(ctt.oracle_table(table, val_prev, ct), (rates @ np.diag(bag_prev)).T, atol=1e-12)

    def test_recalls(self):
        validation, val_post, _, _ = _data()
        table, val_prev = ctt.validation_table(validation, val_post)
        np.testing.assert_allclose(ctt.val_recalls(table, val_prev), np.diag(_paper_rates(validation, val_post)))


class TestErrorSplit(unittest.TestCase):
    def test_split_sums_to_error_and_bound_holds(self):
        rng = np.random.default_rng(1)
        for n_classes in (2, 3, 5, 10):
            for _ in range(50):
                recalls = rng.uniform(0.2, 1.0, n_classes)
                counts = rng.integers(0, 50, (n_classes, n_classes)).astype(float)
                counts[rng.integers(n_classes)] = 0  # a class absent from the bag
                true_ct = counts / counts.sum()
                test_prev = rng.dirichlet(np.ones(n_classes))
                e_pi, e_c = ctt.error_split(recalls, test_prev, true_ct)
                estimate = float(np.sum(test_prev * recalls))
                self.assertAlmostEqual(e_pi + e_c, estimate - np.trace(true_ct), places=12)
                bound = (recalls.max() - recalls.min()) / 2 * np.abs(test_prev - true_ct.sum(axis=1)).sum()
                self.assertLessEqual(abs(e_pi), bound + 1e-12)

    def test_oracle_has_no_quantification_part(self):
        validation, val_post, _, true_cts = _data()
        table, val_prev = ctt.validation_table(validation, val_post)
        recalls = ctt.val_recalls(table, val_prev)
        for ct in true_cts:
            oracle = ctt.oracle_table(table, val_prev, ct)
            e_pi, e_c = ctt.error_split(recalls, oracle.sum(axis=1), ct)
            self.assertAlmostEqual(e_pi, 0.0, places=12)
            self.assertAlmostEqual(e_c, np.trace(oracle) - np.trace(ct), places=12)


class TestAbsentClass(unittest.TestCase):
    def test_class_absent_from_validation_raises(self):
        validation, val_post, dataset, true_cts = _data()
        keep = validation.y != 2
        reduced = LabelledCollection(validation.X[keep], validation.y[keep], classes=np.arange(3))
        np.testing.assert_array_equal(ctt.absent_classes(reduced), [2])
        for level in ctt.LEVELS:
            with self.assertRaisesRegex(ValueError, r"classes \[2\] are absent from V"):
                ctt.predict_tables(level, dataset, reduced, val_post[keep], true_cts)


class TestLevels(unittest.TestCase):
    def test_quantifiers_are_deterministic(self):
        validation, val_post, dataset, true_cts = _data(seed=2)
        for level in ctt.METHODS:
            with self.subTest(level=level):
                first = ctt.predict_tables(level, dataset, validation, val_post, true_cts)
                second = ctt.predict_tables(level, dataset, validation, val_post, true_cts)
                self.assertFalse(any(first["failed"]))
                np.testing.assert_array_equal(np.stack(first["tables"]), np.stack(second["tables"]))
                for t in first["tables"]:
                    self.assertAlmostEqual(t.sum(), 1.0, places=9)

    def test_quantifier_settings(self):
        self.assertEqual(ctt.emq().calib, None)
        self.assertTrue(ctt.emq().exact_train_prev)
        # the MLP takes the experiment seed when the factory is called (other test modules change the global seed)
        with patch.dict(qp.environ, {"_R_SEED": 0}):
            for level, factory in ctt.QUANTIFIERS.items():
                self.assertEqual(factory().classifier.random_state, 0, level)
        self.assertEqual(ctt.QUANTIFIERS["CTT-KDEy"]().bandwidth, 0.1)
        self.assertEqual(ctt.QUANTIFIERS["CTT-PACC"]().val_split, 5)
        self.assertEqual(ctt.QUANTIFIERS["CTT-ACC"]().val_split, 5)
        self.assertEqual((ctt.acc().solver, ctt.acc().norm), ("minimize", "clip"))

    def test_failed_bag_is_flagged(self):
        validation, val_post, dataset, true_cts = _data()
        calls = {"n": 0}

        class Flaky(FixedQuantifier):
            def predict(self, X):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("quantifier failed")
                return self.prev

        original = ctt.QUANTIFIERS["CTT-EMQ"]
        ctt.QUANTIFIERS["CTT-EMQ"] = lambda: Flaky([0.2, 0.3, 0.5])
        try:
            prediction = ctt.predict_tables("CTT-EMQ", dataset, validation, val_post, true_cts)
        finally:
            ctt.QUANTIFIERS["CTT-EMQ"] = original
        self.assertEqual(prediction["failed"], [False, True, False, False])
        self.assertTrue(np.all(np.isnan(prediction["tables"][1])))
        split = ctt.split_columns(prediction, true_cts)
        self.assertTrue(np.isnan(split["e_pi"][1]))
        self.assertTrue(np.isfinite(split["e_pi"][0]))


def _pretrain(n_classes=3, dataset="toy", classifier="LR"):
    return SimpleNamespace(
        domain="classic",
        d_info=SimpleNamespace(name=dataset, n_classes=n_classes, collection="toy"),
        h_info=SimpleNamespace(name=classifier, full_name=classifier),
    )


def _run_levels(base_dir: str, n_classes: int, dataset: str, seed: int):
    """Run the four levels on synthetic data and write their results under ``base_dir`` (the results folder)."""
    validation, val_post, data, true_cts = _data(n_classes=n_classes, seed=seed)
    p = _pretrain(n_classes, dataset)
    context = EvaluationContext(
        pretrain=p, shifts=[0.0] * len(true_cts), true_cts=true_cts, train_prev=[0.5] * n_classes, val_prev=[0.5]
    )
    original = ctt.local_path
    ctt.local_path = lambda *a, **k: os.path.join(base_dir, "missing")  # nothing exists
    try:
        outcomes = []
        for level in ctt.LEVELS:
            outcomes += ctt.evaluate_level(p, level, data, validation, val_post, context)
    finally:
        ctt.local_path = original
    for outcome in outcomes:
        assert outcome.ok, outcome.err
        folder = os.path.join(base_dir, "classic", outcome.acc_name, dataset, "LR")
        os.makedirs(folder, exist_ok=True)
        outcome.df.save_result(os.path.join(folder, f"{outcome.method_name}.parquet"))
    return outcomes


def _main_results(results: dict) -> dict:
    """Synthetic results of the methods of main on the same bags: the oracle's frames, renamed, with the AE shifted."""
    out = {}
    for method, offset in zip(tables.MAIN_COMPARED, (0.5, 0.25)):
        for acc in get_acc_names():
            df = results[ctt.ORACLE, acc].copy()
            df["method"] = method
            df["acc_err"] = df["acc_err"] + offset
            out[method, acc] = df
    return out


class TestEvaluationAndTables(unittest.TestCase):
    def test_columns_and_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "ctt")
            outcomes = _run_levels(base, 3, "toy3", 0)
            _run_levels(base, 2, "toy2", 1)
            self.assertEqual(len(outcomes), len(ctt.LEVELS) * len(get_acc_names()))
            vanilla = next(o.df for o in outcomes if o.method_name == "CTT-KDEy" and o.acc_name == "vanilla_accuracy")
            for column in ("test_prev", "bag_prev", "q_l1", "q_failed", "e_pi", "e_c", "recall_range"):
                self.assertIn(column, vanilla.columns)
            macro = next(o.df for o in outcomes if o.method_name == "CTT-KDEy" and o.acc_name == "macro-F1")
            self.assertNotIn("e_pi", macro.columns)
            oracle = next(o.df for o in outcomes if o.method_name == ctt.ORACLE and o.acc_name == "vanilla_accuracy")
            np.testing.assert_allclose(oracle["q_l1"], 0.0, atol=1e-12)

            results = tables.load_results(base)
            main_results = _main_results(results)
            ae, split, check = tables.write_tables(results, os.path.join(tmp, "tables"), main_results)
            # the methods of main are read on the same pairs
            for method, offset in zip(tables.MAIN_COMPARED, (0.5, 0.25)):
                column = f"vanilla_accuracy|{method}"
                self.assertAlmostEqual(ae.iloc[-1][column], ae.iloc[-1]["vanilla_accuracy|CTT-oracle"] + offset)
            self.assertEqual(
                ae["dataset"].tolist(),
                ["toy2", tables.SUMMARY_BINARY, "toy3", tables.SUMMARY_MULTICLASS, tables.SUMMARY_ALL],
            )
            self.assertLess(check["max_split_residual"].max(), 1e-12)
            self.assertLess(check["max_e_c_diff"].max(), 1e-12)
            self.assertLess(check["max_oracle_e_pi"].max(), 1e-12)
            self.assertEqual(check["n_bound_violations"].sum(), 0)
            # the RMS of e_C equals the RMS of the error of the oracle
            for row in split[~split["dataset"].isin(tables.SUMMARIES)].itertuples():
                df = results[ctt.ORACLE, "vanilla_accuracy"]
                df = df[df["dataset"] == row.dataset]
                self.assertAlmostEqual(row.e_c_rms, float(np.sqrt(np.mean(df["acc_err"] ** 2))), places=12)
            # the mean over all pairs is the mean of the per-pair values
            pairs = ae[~ae["dataset"].isin(tables.SUMMARIES)]
            column = "vanilla_accuracy|CTT-PACC"
            self.assertAlmostEqual(ae.iloc[-1][column], pairs[column].mean(), places=12)
            names = ["ctt_ae.csv", "ctt_split.csv", "ctt_check.csv", "ctt_doc.tex"]
            names += [f"ctt_ae_{acc}.csv" for acc in get_acc_names()]
            names += [f"tabular/ctt_ae_{acc}_long.tex" for acc in get_acc_names()]
            for name in names:
                self.assertTrue(os.path.exists(os.path.join(tmp, "tables", name)), name)
            per_metric = pd.read_csv(os.path.join(tmp, "tables", "ctt_ae_macro-K.csv"))
            self.assertEqual(list(per_metric.columns[-len(tables.AE_COLUMNS) :]), list(tables.AE_COLUMNS))
            with open(os.path.join(tmp, "tables", "tabular", "ctt_ae_vanilla_accuracy.tex")) as f:
                latex = f.read()
            self.assertIn("\\cellcolor[HTML]", latex)
            self.assertIn("\\bfseries", latex)
            # since WP-31 the label of BayCAP-KDEy-mlp-real names its input and pushforward
            self.assertIn("\\bcap{}-KDEy-mlp-real", latex)
            with open(os.path.join(tmp, "tables", "ctt_doc.tex")) as f:
                document = f.read()
            self.assertIn("\\usepackage[table]{xcolor}", document)
            self.assertIn("\\newcommand{\\bcapkdey}", document)

            from uncawe.plots import ctt as plots

            fig = plots.plot_ae_vs_qerr(ae, split)
            self.assertEqual(len(fig.axes), len(ctt.METHODS))

    def test_mismatched_bags_of_main_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "ctt")
            _run_levels(base, 3, "toy3", 0)
            results = tables.load_results(base)
            main_results = _main_results(results)
            df = main_results[tables.MAIN_COMPARED[0], "vanilla_accuracy"]
            df["true_cts"] = [ct[::-1] for ct in df["true_cts"]]
            with self.assertRaisesRegex(ValueError, "are not the same"):
                tables.compute(results, main_results)

    def test_ae_styles(self):
        labels = [tables._level_label(level) for level in tables.AE_COLUMNS]
        frame = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.01], [np.nan] * 7], columns=labels)
        styles = tables.ae_styles(frame)
        green = "#" + "".join(f"{c:02x}" for c in tables.BEST_RGB)
        self.assertIn(f"background-color: {green}", styles.iloc[0, 0])
        self.assertIn("font-weight: bold", styles.iloc[0, 0])
        self.assertIn("background-color: #ffffff", styles.iloc[0, 5])
        self.assertNotIn("bold", styles.iloc[0, 5])
        # 0.35 is halfway between 0.1 and 0.6 in AE, so halfway in colour; checked on a frame with that value
        frame.iloc[0, 2] = 0.35
        middle = "#" + "".join(f"{round((255 + c) / 2):02x}" for c in tables.BEST_RGB)
        self.assertIn(f"background-color: {middle}", tables.ae_styles(frame).iloc[0, 2])
        # the reference (smallest AE here) is neither coloured nor compared
        self.assertEqual(styles.iloc[0, 6], "")
        self.assertTrue((styles.iloc[1] == "").all())

    def test_mismatched_bags_raise(self):
        a = pd.DataFrame({"true_cts": [np.eye(2) / 2]})
        b = pd.DataFrame({"true_cts": [np.array([[0.4, 0.1], [0.0, 0.5]])]})
        with self.assertRaises(ValueError):
            tables._check_paired({"a": a, "b": b}, ("d", "c"))


class TestTask(unittest.TestCase):
    def test_unknown_level(self):
        with self.assertRaises(ValueError):
            ctt.experiment_args(Mock(), levels=["CTT-foo"])


if __name__ == "__main__":
    unittest.main()
