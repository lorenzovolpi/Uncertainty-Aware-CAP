"""E5.1 in ``experiments/binform.py`` and its table and plot modules (work-plan card WP-19)."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection

from uncawe.experiments import binform
from uncawe.experiments.methods import BAYES_WARMUPS, BOOTSTRAP_ITERS, gen_methods
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.plots import binform_prior as plots
from uncawe.tables import binform_prior as bp

N_BAGS = 12


class MeansTest(unittest.TestCase):
    def test_means_are_reproducible_on_the_simplex_and_far_apart(self):
        for name, k in (("haberman", 2), ("connect-4", 3), ("dry-bean", 7), ("letter", 26)):
            mu_star, mu_dagger = binform.e51_means(name, k)
            again = binform.e51_means(name, k)
            np.testing.assert_array_equal(mu_star, again[0])
            np.testing.assert_array_equal(mu_dagger, again[1])
            centre = np.full(k, 1 / k)
            for mu in (mu_star, mu_dagger):
                self.assertEqual(mu.shape, (k,))
                self.assertTrue(np.all(mu >= 0))
                self.assertAlmostEqual(mu.sum(), 1.0)
                self.assertGreaterEqual(np.abs(mu - centre).sum(), binform.E51_MIN_DISTANCE)
            self.assertGreaterEqual(np.abs(mu_star - mu_dagger).sum(), binform.E51_MIN_DISTANCE)

    def test_means_depend_on_the_seed_and_the_dataset(self):
        reference = binform.e51_means("wine", 3)[0]
        self.assertFalse(np.allclose(reference, binform.e51_means("wine", 3, seed=1)[0]))
        self.assertFalse(np.allclose(reference, binform.e51_means("yeast", 3)[0]))

    def test_binary_distance_conditions(self):
        # with K=2 the conditions read |mu_1 - 1/2| >= 1/4 and |mu*_1 - mu+_1| >= 1/4
        mu_star, mu_dagger = binform.e51_means("spambase", 2)
        self.assertGreaterEqual(abs(mu_star[0] - 0.5), 0.25 - 1e-12)
        self.assertGreaterEqual(abs(mu_dagger[0] - 0.5), 0.25 - 1e-12)
        self.assertGreaterEqual(abs(mu_star[0] - mu_dagger[0]), 0.25 - 1e-12)

    def test_concentrations_are_3k_mu(self):
        mu_star, mu_dagger = binform.e51_means("wine", 3)
        np.testing.assert_allclose(binform.e51_concentration(mu_star), 9 * mu_star)
        self.assertEqual(binform.e51_prior("uniform", mu_star, mu_dagger), "uniform")
        np.testing.assert_allclose(binform.e51_prior("well", mu_star, mu_dagger), 9 * mu_star)
        np.testing.assert_allclose(binform.e51_prior("wrong", mu_star, mu_dagger), 9 * mu_dagger)
        with self.assertRaises(ValueError):
            binform.e51_prior("other", mu_star, mu_dagger)

    def test_requested_prevalences(self):
        mu_star, _ = binform.e51_means("letter", 26)
        prevs = binform.e51_requested_prevs("letter", mu_star)
        self.assertEqual(prevs.shape, (binform.NUM_TEST, 26))
        self.assertTrue(np.all(prevs >= 0))
        np.testing.assert_allclose(prevs.sum(axis=1), 1.0)
        np.testing.assert_array_equal(prevs, binform.e51_requested_prevs("letter", mu_star))
        # the means of the draws are close to mu*
        mu_star, _ = binform.e51_means("wine", 3)
        prevs = binform.e51_requested_prevs("wine", mu_star)
        np.testing.assert_allclose(prevs.mean(axis=0), mu_star, atol=0.08)


class _Classifier:
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-X[:, 0]))
        return np.stack([1 - p, p], axis=1)


def _collection(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.standard_normal((n, 2)) + y[:, None]
    return LabelledCollection(X, y, classes=[0, 1])


class DefaultBinformTest(unittest.TestCase):
    def test_default_requested_prevalences_are_unchanged(self):
        val_prev = np.array([0.3, 0.3, 0.4])
        expected = np.random.RandomState(0).dirichlet((1 - val_prev) / 2 * binform.TEST_PRIOR_C, size=binform.NUM_TEST)
        np.testing.assert_array_equal(binform._requested_test_prevs(val_prev), expected)

    def test_sized_bags_at_the_default_size_are_the_default_bags(self):
        pool, h = _collection(), _Classifier()
        prevs = np.random.default_rng(1).dirichlet([2.0, 2.0], size=4)
        default = binform._make_test_samples(pool, h, prevs, 0)
        sized = binform._make_sized_test_samples(pool, h, prevs, 0, binform.SAMPLE_SIZE)
        for a, b in zip(default, sized):
            np.testing.assert_array_equal(a.collection.X, b.collection.X)
            np.testing.assert_array_equal(a.true_ct, b.true_ct)
        small = binform._make_sized_test_samples(pool, h, prevs, 0, 100)
        self.assertTrue(all(len(sample.collection) == 100 for sample in small))

    def test_default_methods_and_arguments(self):
        self.assertEqual(binform._method_names(), ["O-LEAP", "SQS-Hbs", "BayesCAP_uniform", "BayesCAP_transformed_val"])
        self.assertFalse(binform.parse_args([]).e51)
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            binform.parse_args(["--datasets", "wine"])
        pargs = binform.parse_args(["--e51", "--sizes", "100", "--datasets", "wine", "yeast"])
        self.assertEqual((pargs.sizes, pargs.datasets), ([100], ["wine", "yeast"]))

    def test_output_folders(self):
        self.assertEqual(binform.e51_experiment(100), os.path.join("binform_e51", "u100"))
        self.assertNotEqual(binform.e51_experiment(1000), binform.EXPERIMENT)


class MethodsTest(unittest.TestCase):
    def test_six_methods(self):
        self.assertEqual(
            list(binform.E51_METHODS),
            [f"{b}_{p}" for b in ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real") for p in ("uniform", "well", "wrong")],
        )

    def test_methods_are_those_of_main_with_the_prior(self):
        main = dict(gen_methods())
        mu_star, mu_dagger = binform.e51_means("wine", 3)
        for name, (base, kind) in binform.E51_METHODS.items():
            method = binform.e51_method(name, mu_star, mu_dagger)
            reference = main[base]
            self.assertIs(type(method), type(reference))
            for attribute in ("estimand", "c_prior", "num_samples", "num_warmup", "random_state"):
                self.assertEqual(getattr(method, attribute), getattr(reference, attribute), (name, attribute))
            if isinstance(method, BayCAP):
                self.assertEqual(method.shares, reference.shares)
            if isinstance(method, BayCAPKDEy):
                for attribute in ("input_classifier", "kernel", "bandwidth", "shrinkage", "temperature"):
                    self.assertEqual(getattr(method, attribute), getattr(reference, attribute), (name, attribute))
            self.assertEqual((method.num_samples, method.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
            expected = binform.e51_prior(kind, mu_star, mu_dagger)
            if kind == "uniform":
                self.assertEqual(method.prior, "uniform")
            else:
                np.testing.assert_allclose(method.prior, expected)

    def test_tasks_order_and_filters(self):
        infos = {
            "a": SimpleNamespace(d_info=SimpleNamespace(name="wine"), h_info=SimpleNamespace(name="LR")),
            "b": SimpleNamespace(d_info=SimpleNamespace(name="wine"), h_info=SimpleNamespace(name="kNN")),
            "c": SimpleNamespace(d_info=SimpleNamespace(name="other"), h_info=SimpleNamespace(name="LR")),
        }
        with (
            patch.object(binform, "load_info_paths", return_value=list(infos)),
            patch.object(binform.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
            patch.object(binform, "get_filtered_datasets", return_value=["wine"]),
            patch.object(binform, "_e51_all_results_exist", side_effect=lambda p, m, s: m.endswith("_well")),
        ):
            tasks = binform.e51_tasks(classifiers=["LR", "kNN"])
            self.assertEqual({path for path, _, _ in tasks}, {"a", "b"})
            self.assertEqual([size for _, _, size in tasks][0], 1000)
            self.assertFalse(any(name.endswith("_well") for _, name, _ in tasks))
            first_kdey = next(i for i, (_, name, _) in enumerate(tasks) if name.startswith("BayCAP-KDEy"))
            self.assertTrue(all(name.startswith("BayesCAP-Hf") for _, name, s in tasks[:first_kdey]))
            self.assertEqual(len(tasks), 2 * 4 * 2)  # sizes x (methods without _well) x pairs
            tasks = binform.e51_tasks(sizes=[100], methods=["BayCAP-KDEy-mlp-real_wrong"], classifiers=["LR"])
            self.assertEqual(tasks, [("a", "BayCAP-KDEy-mlp-real_wrong", 100)])
            with self.assertRaises(ValueError):
                binform.e51_tasks(methods=["BayesCAP-Hf-real"])


class ProtocolCachesTest(unittest.TestCase):
    """``e51_protocol`` clears the caches of JAX after each task, also when the task fails."""

    def test_clears_the_compiled_functions_after_a_task(self):
        square = jax.jit(lambda x: x * x)

        def task(info_path, method_name, size):
            square(jnp.ones(3))
            self.assertGreater(square._cache_size(), 0)
            return ["outcome", info_path, method_name, size]

        with patch.object(binform, "_e51_task", side_effect=task):
            outcome = binform.e51_protocol(("path", "BayesCAP-Hf-real_well", 100))
        self.assertEqual(outcome, ["outcome", "path", "BayesCAP-Hf-real_well", 100])
        self.assertEqual(square._cache_size(), 0)

    def test_clears_the_caches_when_the_task_raises(self):
        with (
            patch.object(binform, "_e51_task", side_effect=RuntimeError("boom")),
            patch.object(binform.jax, "clear_caches") as clear,
            self.assertRaises(RuntimeError),
        ):
            binform.e51_protocol(("path", "BayesCAP-Hf-real_well", 100))
        clear.assert_called_once_with()


# --------------------------------------------------------------------------------------------------- tables


def _true_ct(rng, n_classes):
    """A realised table (cap-ml layout) with recalls about 0.8, summing to 1."""
    props = rng.dirichlet(np.full(n_classes, 3.0))
    ct = np.outer(props, np.full(n_classes, 0.2 / (n_classes - 1)))
    np.fill_diagonal(ct, props * 0.8)
    return ct


# width of the samples per prior: the well-specified prior is narrower, the misspecified one shifted
PRIOR_SHAPES = {"uniform": (0.0, 0.03), "well": (0.0, 0.02), "wrong": (0.08, 0.03)}


def _frame(method, dataset, classifier, n_classes, true_cts, rng, shift, width, failed=()):
    rows = []
    for bag, ct in enumerate(true_cts):
        acc = float(np.trace(ct))
        samples = acc + shift + width * rng.standard_normal(200)
        if bag in failed:
            samples = np.full(200, np.nan)
        rows.append(
            {
                "dataset": dataset,
                "classifier": classifier,
                "n_classes": n_classes,
                "method": method,
                "acc_name": "vanilla_accuracy",
                "true_cts": ct,
                "estim_cts": ct,
                "estim_accs": float(np.mean(samples)),
                "estim_acc_ranges": samples,
                "acc_err": abs(float(np.mean(samples)) - acc),
                "nuts_divergences": 1 if bag == 0 else 0,
            }
        )
    return pd.DataFrame(rows)


PAIRS = [("wine", "LR", 3), ("spam", "kNN", 2), ("letter", "SVM", 26)]


def _frames(base, seed=0):
    rng = np.random.default_rng(seed)
    frames = {prior: [] for prior in binform.E51_PRIORS}
    for dataset, classifier, k in PAIRS:
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        for prior, (shift, width) in PRIOR_SHAPES.items():
            failed = (2,) if prior == "wrong" and dataset == "wine" else ()
            frame = _frame(f"{base}_{prior}", dataset, classifier, k, true_cts, rng, shift, width, failed)
            frames[prior].append(frame)
    return {prior: pd.concat(parts, ignore_index=True) for prior, parts in frames.items()}


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for base in binform.E51_BASE_METHODS:
            for size in (100, 1000):
                rows.extend(bp.pair_rows(_frames(base), "vanilla_accuracy", size, base))
        cls.table = bp.ordered(pd.DataFrame(rows))
        cls.summary = bp.summary_table(cls.table)

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * 2 * len(PAIRS))
        self.assertTrue((self.table["d_w_know"] > 0).all())
        self.assertTrue((self.table["d_w_wrong"] > 0).all())
        self.assertTrue((self.table["well_amplitude_ratio"] < 1).all())
        np.testing.assert_allclose(self.table["d_w_know"], self.table["uniform_winkler"] - self.table["well_winkler"])
        wine = self.table.loc[self.table["dataset"] == "wine"]
        np.testing.assert_allclose(wine["wrong_share_failed"], 1 / N_BAGS)
        np.testing.assert_allclose(self.table["uniform_share_divergent"], 1 / N_BAGS)
        # ordered by size, method, then K
        first = self.table.iloc[: len(PAIRS)]
        self.assertEqual(first["size"].unique().tolist(), [100])
        self.assertEqual(first["n_classes"].tolist(), [2, 3, 26])

    def test_summary(self):
        allk = self.summary.loc[self.summary["K"] == "all"]
        self.assertEqual(len(allk), 2 * 2 * 3)
        well = allk.loc[allk["prior"] == "well"]
        self.assertTrue((well["pairs_lower"] == len(PAIRS)).all())
        self.assertTrue((well["d_winkler"] < 0).all())
        wrong = allk.loc[allk["prior"] == "wrong"]
        self.assertTrue((wrong["d_winkler"] > 0).all())
        self.assertTrue(allk.loc[allk["prior"] == "uniform", "p_value"].isna().all())
        self.assertEqual(set(self.summary["K"]), {"all", "K=2", "3<=K<=7", "K>=8"})

    def test_paired_wilcoxon(self):
        self.assertTrue(np.isnan(bp.paired_wilcoxon(np.zeros(5), np.zeros(5))))
        self.assertLess(bp.paired_wilcoxon(np.arange(10.0), np.arange(10.0) + 1), 0.01)

    def test_different_bags_raise(self):
        frames = _frames("BayesCAP-Hf-real")
        frames["well"] = _frames("BayesCAP-Hf-real", seed=1)["well"]
        with self.assertRaises(ValueError):
            bp.pair_rows(frames, "vanilla_accuracy", 100, "BayesCAP-Hf-real")

    def test_stored_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            for base in binform.E51_BASE_METHODS[:1]:
                for prior, df in _frames(base).items():
                    for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                        folder = os.path.join(
                            root, binform.e51_experiment(100), "classic", "vanilla_accuracy", dataset, classifier
                        )
                        os.makedirs(folder, exist_ok=True)
                        part = part.assign(
                            true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                        )
                        part.to_parquet(os.path.join(folder, f"{base}_{prior}.parquet"))
            table = bp.pair_table(["vanilla_accuracy"], root=root)
            self.assertEqual(len(table), len(PAIRS))
            self.assertEqual(set(table["method"]), {binform.E51_BASE_METHODS[0]})
            self.assertEqual(set(table["size"]), {100})
            with self.assertRaises(RuntimeError):
                bp.pair_table(["macro-F1"], root=root)

    def test_latex(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"]
        latex = bp._summary_styler(part).to_latex()
        self.assertIn("well specified", latex)
        self.assertIn("\\bcap{}-Hf", latex)
        self.assertNotIn("vanilla_accuracy", bp.caption("vanilla_accuracy"))

    def test_plot(self):
        fig = plots.plot_winkler(self.table, "BayesCAP-Hf-real")
        self.assertEqual(len(fig.axes), 2 * 2 + 1)  # two priors x two sizes, and the colour bar
        plt.close(fig)
        self.assertIsNone(plots.plot_winkler(self.table, "BayesCAP-Hf-real", "macro-K"))


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
    unittest.main()
