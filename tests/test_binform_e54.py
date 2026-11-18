"""E5.4 in ``experiments/binform.py`` and its table and plot modules (work-plan card WP-24)."""

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
from test_binform_e51 import _frame, _true_ct

from uncawe.experiments import binform
from uncawe.experiments.methods import BAYES_WARMUPS, BOOTSTRAP_ITERS, gen_methods
from uncawe.experiments.metrics import acc_from_ct
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.plots import binform_centre_edge as plots
from uncawe.tables import binform_centre_edge as bce

N_BAGS = 12


class DistributionsTest(unittest.TestCase):
    def test_alphas_and_priors(self):
        np.testing.assert_allclose(binform.e54_alpha("centre", 4), np.full(4, 3.0))
        np.testing.assert_allclose(binform.e54_alpha("edge", 4), np.full(4, 1 / 3))
        np.testing.assert_allclose(binform.e54_alpha("uniform", 4), np.ones(4))
        self.assertEqual(binform.e54_prior("uniform", 4), "uniform")
        np.testing.assert_allclose(binform.e54_prior("centre", 26), np.full(26, 3.0))
        np.testing.assert_allclose(binform.e54_prior("edge", 2), np.full(2, 1 / 3))
        with self.assertRaises(ValueError):
            binform.e54_alpha("other", 3)
        self.assertEqual(binform.E54_PRIORS, ("uniform", "centre", "edge"))

    def test_requested_prevalences_are_reproducible_and_on_the_simplex(self):
        for name, k in (("haberman", 2), ("connect-4", 3), ("letter", 26)):
            for distribution in binform.E54_DISTRIBUTIONS:
                prevs = binform.e54_requested_prevs(name, k, distribution)
                self.assertEqual(prevs.shape, (binform.NUM_TEST, k))
                self.assertTrue(np.all(np.isfinite(prevs)))
                self.assertTrue(np.all(prevs >= 0))
                np.testing.assert_allclose(prevs.sum(axis=1), 1.0)
                np.testing.assert_array_equal(prevs, binform.e54_requested_prevs(name, k, distribution))
        with self.assertRaises(ValueError):
            binform.e54_requested_prevs("wine", 3, "uniform")

    def test_requested_prevalences_differ_by_distribution_dataset_seed_and_from_e51(self):
        centre = binform.e54_requested_prevs("wine", 3, "centre")
        self.assertFalse(np.allclose(centre, binform.e54_requested_prevs("wine", 3, "edge")))
        self.assertFalse(np.allclose(centre, binform.e54_requested_prevs("yeast", 3, "centre")))
        self.assertFalse(np.allclose(centre, binform.e54_requested_prevs("wine", 3, "centre", seed=1)))
        mu_star, _ = binform.e51_means("wine", 3)
        self.assertFalse(np.allclose(centre, binform.e51_requested_prevs("wine", mu_star)))
        self.assertEqual(len(set(binform._E54_PREVS_STREAMS.values())), 2)
        self.assertFalse(
            {binform._E51_MEANS_STREAM, binform._E51_PREVS_STREAM} & set(binform._E54_PREVS_STREAMS.values())
        )

    def test_centre_is_concentrated_and_edge_sparse(self):
        k = 26
        centre = binform.e54_requested_prevs("letter", k, "centre")
        edge = binform.e54_requested_prevs("letter", k, "edge")
        # both have mean 1/K
        np.testing.assert_allclose(centre.mean(), 1 / k)
        np.testing.assert_allclose(edge.mean(), 1 / k)
        # variance of a component: (1/K)(1-1/K)/(alpha K + 1), within the Monte Carlo error of 100 draws
        for prevs, alpha in ((centre, 3.0), (edge, 1 / 3)):
            expected = (1 / k) * (1 - 1 / k) / (alpha * k + 1)
            self.assertAlmostEqual(prevs.var(axis=0).mean() / expected, 1.0, delta=0.3)
        self.assertGreater(np.mean(edge < 0.01), np.mean(centre < 0.01) + 0.2)


class MethodsTest(unittest.TestCase):
    def test_six_methods(self):
        self.assertEqual(
            list(binform.E54_METHODS),
            [f"{b}_{p}" for b in ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real") for p in ("uniform", "centre", "edge")],
        )

    def test_methods_are_those_of_main_with_the_prior(self):
        main = dict(gen_methods())
        for name, (base, kind) in binform.E54_METHODS.items():
            method = binform.e54_method(name, 3)
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
            if kind == "uniform":
                self.assertEqual(method.prior, "uniform")
            else:
                np.testing.assert_allclose(method.prior, np.full(3, binform.E54_ALPHAS[kind]))

    def test_output_folders(self):
        self.assertEqual(binform.e54_experiment("edge", 100), os.path.join("binform_e54", "edge", "u100"))
        self.assertNotEqual(binform.e54_experiment("centre", 1000), binform.e51_experiment(1000))


def _info(name="wine", classifier="LR", k=3):
    return SimpleNamespace(
        domain="classic",
        d_info=SimpleNamespace(name=name, n_classes=k),
        h_info=SimpleNamespace(name=classifier, full_name=classifier),
    )


class TaskTest(unittest.TestCase):
    """The tasks of E5.1 and E5.4 share ``_prior_task``; what each passes to it."""

    def test_e54_task_passes_its_bags_method_and_columns(self):
        p = _info("wine", "LR", 3)
        with (
            patch.object(binform.PretrainInfo, "load", return_value=p),
            patch.object(binform, "_e54_all_results_exist", return_value=False),
            patch.object(binform, "_prior_task", return_value=["done"]) as core,
        ):
            self.assertEqual(binform._e54_task("path", "BayesCAP-Hf-real_centre", "edge", 100), ["done"])
        args, kwargs = core.call_args
        self.assertEqual(args, (p, "BayesCAP-Hf-real_centre"))
        self.assertEqual(kwargs["size"], 100)
        np.testing.assert_array_equal(kwargs["requested_prevs"], binform.e54_requested_prevs("wine", 3, "edge"))
        self.assertEqual(kwargs["static_columns"], {"distribution": "edge", "prior_alpha": [3.0, 3.0, 3.0]})
        np.testing.assert_allclose(kwargs["build_method"]().prior, np.full(3, 3.0))
        self.assertIn(os.path.join("binform_e54", "edge", "u100"), kwargs["result_path"]("macro-F1"))

    def test_e51_task_passes_what_it_passed_before(self):
        p = _info("wine", "LR", 3)
        mu_star, mu_dagger = binform.e51_means("wine", 3)
        with (
            patch.object(binform.PretrainInfo, "load", return_value=p),
            patch.object(binform, "_e51_all_results_exist", return_value=False),
            patch.object(binform, "_prior_task", return_value=["done"]) as core,
        ):
            binform._e51_task("path", "BayCAP-KDEy-mlp-real_uniform", 1000)
        kwargs = core.call_args.kwargs
        np.testing.assert_array_equal(kwargs["requested_prevs"], binform.e51_requested_prevs("wine", mu_star))
        self.assertEqual(
            kwargs["static_columns"],
            {"mu_star": mu_star.tolist(), "mu_dagger": mu_dagger.tolist(), "prior_alpha": [1.0, 1.0, 1.0]},
        )
        self.assertEqual(list(kwargs["static_columns"]), ["mu_star", "mu_dagger", "prior_alpha"])
        self.assertEqual(kwargs["build_method"]().prior, "uniform")
        self.assertIn(os.path.join("binform_e51", "u1000"), kwargs["result_path"]("vanilla_accuracy"))

    def test_existing_results_are_skipped(self):
        with (
            patch.object(binform.PretrainInfo, "load", return_value=_info()),
            patch.object(binform, "_e54_all_results_exist", return_value=True),
            patch.object(binform, "_prior_task") as core,
        ):
            outcome = binform._e54_task("path", "BayesCAP-Hf-real_edge", "centre", 100)
        core.assert_not_called()
        self.assertEqual(len(outcome), 1)

    def test_tasks_order_and_filters(self):
        infos = {"a": _info("wine", "LR"), "b": _info("wine", "kNN"), "c": _info("other", "LR")}
        with (
            patch.object(binform, "load_info_paths", return_value=list(infos)),
            patch.object(binform.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
            patch.object(binform, "get_filtered_datasets", return_value=["wine"]),
            patch.object(binform, "_e54_all_results_exist", side_effect=lambda p, m, d, s: m.endswith("_edge")),
        ):
            tasks = binform.e54_tasks()
            self.assertEqual({path for path, *_ in tasks}, {"a", "b"})
            self.assertFalse(any(name.endswith("_edge") for _, name, _, _ in tasks))
            self.assertEqual(len(tasks), 2 * 2 * 4 * 2)  # sizes x distributions x (methods without _edge) x pairs
            self.assertEqual([(s, d) for _, _, d, s in tasks[:1]], [(1000, "centre")])
            cells = [(s, d) for _, _, d, s in tasks]
            self.assertEqual(cells, sorted(cells, key=lambda c: (-c[0], binform.E54_DISTRIBUTIONS.index(c[1]))))
            block = [name for _, name, d, s in tasks if (s, d) == (1000, "centre")]
            first_kdey = next(i for i, name in enumerate(block) if name.startswith("BayCAP-KDEy"))
            self.assertTrue(all(name.startswith("BayesCAP-Hf") for name in block[:first_kdey]))
            tasks = binform.e54_tasks(
                classifiers=["LR"], distributions=["edge"], sizes=[100], methods=["BayCAP-KDEy-mlp-real_centre"]
            )
            self.assertEqual(tasks, [("a", "BayCAP-KDEy-mlp-real_centre", "edge", 100)])
            with self.assertRaises(ValueError):
                binform.e54_tasks(methods=["BayesCAP-Hf-real_well"])
            with self.assertRaises(ValueError):
                binform.e54_tasks(distributions=["uniform"])

    def test_arguments(self):
        pargs = binform.parse_args(["--e54", "--datasets", "connect-4", "letter", "--distributions", "edge"])
        self.assertTrue(pargs.e54)
        self.assertFalse(pargs.e51)
        self.assertEqual((pargs.datasets, pargs.distributions), (["connect-4", "letter"], ["edge"]))
        for argv in (["--e51", "--e54"], ["--e51", "--distributions", "edge"], ["--distributions", "edge"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                binform.parse_args(argv)
        self.assertFalse(binform.parse_args([]).e54)


class ProtocolCachesTest(unittest.TestCase):
    def test_clears_the_compiled_functions_after_a_task(self):
        square = jax.jit(lambda x: x * x)

        def task(*args):
            square(jnp.ones(3))
            self.assertGreater(square._cache_size(), 0)
            return list(args)

        with patch.object(binform, "_e54_task", side_effect=task):
            outcome = binform.e54_protocol(("path", "BayesCAP-Hf-real_edge", "centre", 100))
        self.assertEqual(outcome, ["path", "BayesCAP-Hf-real_edge", "centre", 100])
        self.assertEqual(square._cache_size(), 0)

    def test_clears_the_caches_when_the_task_raises(self):
        with (
            patch.object(binform, "_e54_task", side_effect=RuntimeError("boom")),
            patch.object(binform.jax, "clear_caches") as clear,
            self.assertRaises(RuntimeError),
        ):
            binform.e54_protocol(("path", "BayesCAP-Hf-real_edge", "centre", 100))
        clear.assert_called_once_with()


class AbsentClassMetricsTest(unittest.TestCase):
    """How the scored metrics treat a class absent from the bag (frequent under the sparse distribution; Part IV,
    FIX-6): the truth uses the smoothed macro-F1, which scores an absent, never predicted class about 1/K instead of
    1; macro-K scores it 2 specificity - 1."""

    def setUp(self):
        # three classes, class 2 absent from the bag and never predicted, the other two perfectly classified
        self.ct = np.diag([0.6, 0.4, 0.0])

    def test_metrics_are_finite(self):
        for acc_name in ("vanilla_accuracy", "macro-F1", "macro-K"):
            for kind in ("selection", "evaluation"):
                self.assertTrue(np.isfinite(acc_from_ct(acc_name, self.ct, kind)), (acc_name, kind))

    def test_macro_f1_of_an_absent_class(self):
        self.assertAlmostEqual(acc_from_ct("macro-F1", self.ct, "evaluation"), 1.0)
        # smoothed: F1 of the absent class 2e / (2e + 2 (K-1) e) = 1/K, the others about 1
        self.assertAlmostEqual(acc_from_ct("macro-F1", self.ct, "selection"), (1 + 1 + 1 / 3) / 3, places=3)

    def test_macro_k_of_an_absent_class(self):
        self.assertAlmostEqual(acc_from_ct("macro-K", self.ct), 1.0)
        # one instance of class 0 predicted as the absent class 2: specificity of class 2 is 0.9, K_2 = 0.8
        ct = np.array([[0.5, 0.0, 0.1], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
        k0 = 0.5 / 0.6 + 1.0 - 1.0
        self.assertAlmostEqual(acc_from_ct("macro-K", ct), (k0 + 1.0 + 0.8) / 3)


# --------------------------------------------------------------------------------------------------- tables

# shift and width of the samples per role: well informed narrower, ill informed shifted
ROLE_SHAPES = {"uniform": (0.0, 0.03), "well": (0.0, 0.02), "ill": (0.08, 0.03)}
PAIRS = [("wine", "LR", 3), ("wine", "kNN", 3), ("spam", "kNN", 2), ("letter", "SVM", 26)]


def _absent_ct(rng, k):
    """A realised table with its last class absent (zero row, cap-ml layout)."""
    ct = _true_ct(rng, k)
    ct[-1, :] = 0.0
    return ct / ct.sum()


def _frames(base, distribution, seed=0):
    """``role -> rows`` for one distribution; the bags of ``edge`` have one absent class in half of the bags."""
    rng = np.random.default_rng(seed)
    roles = bce.role_priors(distribution)
    frames = {role: [] for role in bce.ROLES}
    for dataset, classifier, k in PAIRS:
        true_cts = [
            _absent_ct(rng, k) if distribution == "edge" and bag % 2 else _true_ct(rng, k) for bag in range(N_BAGS)
        ]
        for role, (shift, width) in ROLE_SHAPES.items():
            failed = (2,) if role == "ill" and dataset == "spam" else ()
            name = f"{base}_{roles[role]}"
            frames[role].append(_frame(name, dataset, classifier, k, true_cts, rng, shift, width, failed))
    return {role: pd.concat(parts, ignore_index=True) for role, parts in frames.items()}


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for distribution in binform.E54_DISTRIBUTIONS:
            for base in binform.E51_BASE_METHODS:
                for size in (100, 1000):
                    rows.extend(
                        bce.pair_rows(_frames(base, distribution), "vanilla_accuracy", distribution, size, base)
                    )
        cls.table = bce.ordered(pd.DataFrame(rows))
        cls.summary = bce.summary_table(cls.table)

    def test_roles(self):
        self.assertEqual(bce.role_priors("centre"), {"uniform": "uniform", "well": "centre", "ill": "edge"})
        self.assertEqual(bce.role_priors("edge"), {"uniform": "uniform", "well": "edge", "ill": "centre"})

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * 2 * 2 * len(PAIRS))
        self.assertTrue((self.table["d_w_know"] > 0).all())
        self.assertTrue((self.table["d_w_wrong"] > 0).all())
        self.assertTrue((self.table["well_amplitude_ratio"] < 1).all())
        np.testing.assert_allclose(self.table["d_w_know"], self.table["uniform_winkler"] - self.table["well_winkler"])
        spam = self.table.loc[self.table["dataset"] == "spam"]
        np.testing.assert_allclose(spam["ill_share_failed"], 1 / N_BAGS)
        np.testing.assert_allclose(self.table["uniform_share_divergent"], 1 / N_BAGS)
        edge = self.table["distribution"] == "edge"
        np.testing.assert_allclose(self.table.loc[edge, "absent_classes"], 0.5)
        np.testing.assert_allclose(self.table.loc[~edge, "absent_classes"], 0.0)
        self.assertEqual(set(self.table.loc[edge, "well_prior"]), {"edge"})
        self.assertEqual(set(self.table.loc[edge, "ill_prior"]), {"centre"})
        first = self.table.iloc[: len(PAIRS)]
        self.assertEqual(
            (first["distribution"].unique().tolist(), first["size"].unique().tolist()), (["centre"], [100])
        )
        self.assertEqual(first["n_classes"].tolist(), [2, 3, 3, 26])

    def test_summary(self):
        allk = self.summary.loc[self.summary["K"] == "all"]
        self.assertEqual(len(allk), 2 * 2 * 2 * 3)
        well = allk.loc[allk["role"] == "well"]
        self.assertTrue((well["pairs_lower"] == len(PAIRS)).all())
        self.assertTrue((well["datasets_lower"] == 3).all())
        self.assertTrue((well["datasets"] == 3).all())
        self.assertTrue((well["d_winkler"] < 0).all())
        self.assertTrue((allk.loc[allk["role"] == "ill", "d_winkler"] > 0).all())
        uniform = allk.loc[allk["role"] == "uniform"]
        self.assertTrue(uniform[["p_value", "p_value_datasets", "pairs_lower"]].isna().all().all())
        self.assertEqual(set(allk.loc[allk["distribution"] == "edge", "prior"]), {"uniform", "edge", "centre"})
        self.assertEqual(set(self.summary["K"]), {"all", "K=2", "3<=K<=7", "K>=8"})

    def test_different_bags_raise(self):
        frames = _frames("BayesCAP-Hf-real", "centre")
        frames["well"] = _frames("BayesCAP-Hf-real", "centre", seed=1)["well"]
        with self.assertRaises(ValueError):
            bce.pair_rows(frames, "vanilla_accuracy", "centre", 100, "BayesCAP-Hf-real")

    def test_stored_round_trip(self):
        base = binform.E51_BASE_METHODS[0]
        with tempfile.TemporaryDirectory() as root:
            for df in _frames(base, "edge").values():
                for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                    folder = os.path.join(
                        root, binform.e54_experiment("edge", 100), "classic", "vanilla_accuracy", dataset, classifier
                    )
                    os.makedirs(folder, exist_ok=True)
                    part = part.assign(
                        true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                    )
                    part.to_parquet(os.path.join(folder, f"{part['method'].iloc[0]}.parquet"))
            table = bce.pair_table(["vanilla_accuracy"], root=root)
            self.assertEqual(len(table), len(PAIRS))
            self.assertEqual((set(table["method"]), set(table["distribution"])), ({base}, {"edge"}))
            np.testing.assert_allclose(table["absent_classes"], 0.5)
            with self.assertRaises(RuntimeError):
                bce.pair_table(["macro-F1"], root=root)

    def test_latex_and_outputs(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"]
        latex = bce._summary_styler(part).to_latex()
        for text in ("well informed", "ill informed", "\\bcaphf", "Dir"):
            self.assertIn(text, latex)
        self.assertNotIn("vanilla_accuracy", bce.caption("vanilla_accuracy"))
        with tempfile.TemporaryDirectory() as out_dir:
            bce.write_tables(self.table, out_dir)
            stem = bce._stem("vanilla_accuracy")
            for path in (
                bce.csv_path(out_dir),
                bce.csv_path(out_dir, "summary"),
                os.path.join(out_dir, "tabular", f"{stem}.tex"),
                os.path.join(out_dir, "tabular", f"{stem}_long.tex"),
            ):
                self.assertTrue(os.path.exists(path), path)
            with open(os.path.join(out_dir, f"{stem}_doc.tex")) as f:
                document = f.read()
            self.assertIn("\\newcommand", document)
            self.assertIn(f"tabular/{stem}_long.tex", document)
            reread = pd.read_csv(bce.csv_path(out_dir))
            self.assertEqual(len(reread), len(self.table))

    def test_plot(self):
        for distribution in binform.E54_DISTRIBUTIONS:
            fig = plots.plot_distribution(self.table, "BayCAP-KDEy-mlp-real", distribution)
            self.assertEqual(len(fig.axes), 2 * 2 + 1)  # two roles x two sizes, and the colour bar
            title = fig.get_suptitle()
            self.assertIn("E5.4", title)
            self.assertIn("vanilla accuracy", title)
            titles = [ax.get_title() for ax in fig.axes[:4]]
            self.assertTrue(any("well informed" in t for t in titles))
            self.assertTrue(any("ill informed" in t for t in titles))
            fig.canvas.draw()
            plt.close(fig)
        self.assertIsNone(plots.plot_distribution(self.table, "BayCAP-KDEy-mlp-real", "edge", "macro-K"))
        self.assertIsNone(plots.plot_distribution(self.table.iloc[:0], "BayCAP-KDEy-mlp-real", "edge"))


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
    unittest.main()
