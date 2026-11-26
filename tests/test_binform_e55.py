"""E5.5 in ``experiments/binform.py`` and its table and plot modules (work-plan card WP-27)."""

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
from cap.models.confidence import RQBS, PrediQuant
from quapy.data import LabelledCollection
from test_binform_e51 import _frame, _true_ct

from uncawe.experiments import binform
from uncawe.experiments.methods import BAYES_N_JOBS, SQS_B, gen_methods
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.sqs_b import SQSB
from uncawe.plots import binform_side_prior as plots
from uncawe.tables import binform_side_prior as bsp

N_BAGS = 12


class DistributionsTest(unittest.TestCase):
    def test_bag_distributions_and_priors(self):
        np.testing.assert_allclose(binform.e55_bag_alpha("centre", 4), np.full(4, 10.0))
        np.testing.assert_allclose(binform.e55_bag_alpha("edge", 4), np.full(4, 0.1))
        with self.assertRaises(ValueError):
            binform.e55_bag_alpha("uniform", 3)
        # the priors are those of E5.4, set apart from the distributions of the bags
        self.assertEqual(binform.E55_PRIORS, ("uniform", "centre", "edge"))
        self.assertEqual(binform.E54_ALPHAS, {"centre": 3.0, "edge": 1 / 3})
        self.assertEqual(binform.E55_DISTRIBUTIONS, binform.E54_DISTRIBUTIONS)
        self.assertEqual(binform.E55_SIZE, 1000)

    def test_requested_prevalences_are_reproducible_and_on_the_simplex(self):
        for name, k in (("haberman", 2), ("connect-4", 3), ("letter", 26)):
            for distribution in binform.E55_DISTRIBUTIONS:
                prevs = binform.e55_requested_prevs(name, k, distribution)
                self.assertEqual(prevs.shape, (binform.NUM_TEST, k))
                self.assertTrue(np.all(np.isfinite(prevs)))
                self.assertTrue(np.all(prevs >= 0))
                np.testing.assert_allclose(prevs.sum(axis=1), 1.0)
                np.testing.assert_array_equal(prevs, binform.e55_requested_prevs(name, k, distribution))

    def test_requested_prevalences_differ_from_e51_and_e54(self):
        centre = binform.e55_requested_prevs("wine", 3, "centre")
        self.assertFalse(np.allclose(centre, binform.e55_requested_prevs("wine", 3, "edge")))
        self.assertFalse(np.allclose(centre, binform.e55_requested_prevs("yeast", 3, "centre")))
        self.assertFalse(np.allclose(centre, binform.e55_requested_prevs("wine", 3, "centre", seed=1)))
        for distribution in binform.E55_DISTRIBUTIONS:
            self.assertFalse(
                np.allclose(
                    binform.e55_requested_prevs("wine", 3, distribution),
                    binform.e54_requested_prevs("wine", 3, distribution),
                )
            )
        mu_star, _ = binform.e51_means("wine", 3)
        self.assertFalse(np.allclose(centre, binform.e51_requested_prevs("wine", mu_star)))
        streams = set(binform._E55_PREVS_STREAMS.values())
        self.assertEqual(len(streams), 2)
        self.assertFalse(streams & {binform._E51_MEANS_STREAM, binform._E51_PREVS_STREAM})
        self.assertFalse(streams & set(binform._E54_PREVS_STREAMS.values()))

    def test_more_extreme_than_e54(self):
        k = 26
        centre = binform.e55_requested_prevs("letter", k, "centre")
        edge = binform.e55_requested_prevs("letter", k, "edge")
        for prevs, alpha in ((centre, 10.0), (edge, 0.1)):
            expected = (1 / k) * (1 - 1 / k) / (alpha * k + 1)
            self.assertAlmostEqual(prevs.var(axis=0).mean() / expected, 1.0, delta=0.3)
        self.assertLess(centre.var(), binform.e54_requested_prevs("letter", k, "centre").var())
        self.assertGreater(edge.var(), binform.e54_requested_prevs("letter", k, "edge").var())


# the methods with a prior on pi^Q (four since card WP-33: BayCAP-Hf, the two per-instance BayCAP-KDEy, SQS-B)
N_PRIOR_BASES = len(binform.E55_PRIOR_BASES)
# the methods without a prior: the PACC references and, since WP-36, those with KDEy
N_REFS = len(binform.E55_REFERENCES)


class MethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = dict(gen_methods())

    def test_fifteen_methods(self):
        # the per-instance BayCAP-KDEy methods in place of BayCAP-KDEy-mlp-real since card WP-33
        expected = [
            f"{b}_{p}"
            for b in ("BayesCAP-Hf-real", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb", "SQS-B")
            for p in ("uniform", "centre", "edge")
        ]
        # the methods of WP-34 with each prior, after those of E5.5
        tcal = [
            f"{b}_{p}"
            for b in ("BayesCAP-Hf-real-tcal", "BayCAP-KDEy-h-inst-cb-tcal-keep", "BayCAP-KDEy-h-inst-cb-tcal-resel")
            for p in ("uniform", "centre", "edge")
        ]
        # the references, the PACC ones and, since WP-36, those with KDEy
        references = ["SQS-H", "SQS-Hbs", "PQ", "SQS-H-KDEy", "SQS-Hbs-KDEy", "PQ-KDEy"]
        self.assertEqual(list(binform.E55_METHODS), [*expected, *tcal, *references])
        self.assertTrue(all(binform.E55_METHODS[name][1] is None for name in references))

    def test_methods_with_a_prior_are_those_of_main_with_the_prior(self):
        for name, (base, kind) in binform.E55_METHODS.items():
            # the methods of WP-34 are checked in tests/test_tcal.py
            if kind is None or base in binform.E55_TCAL_BASES:
                continue
            method = binform.e55_method(name, 3)
            reference = self.main[base]
            self.assertIs(type(method), type(reference))
            attributes = ["num_samples", "num_warmup", "random_state"]
            if isinstance(method, BayCAP):
                attributes += ["estimand", "c_prior", "shares"]
            if isinstance(method, BayCAPKDEy):
                attributes += ["estimand", "input_classifier", "kernel", "bandwidth", "shrinkage", "temperature"]
            if isinstance(method, SQSB):
                attributes += ["input_classifier", "kernel", "bandwidth", "shrinkage", "temperature", "sample_size"]
            for attribute in attributes:
                self.assertEqual(getattr(method, attribute), getattr(reference, attribute), (name, attribute))
            if kind == "uniform":
                self.assertEqual(method.prior, "uniform")
            else:
                np.testing.assert_allclose(method.prior, np.full(3, binform.E54_ALPHAS[kind]))

    def test_sqs_b_receives_the_prior(self):
        method = binform.e55_method("SQS-B_edge", 4)
        self.assertIsInstance(method, SQSB)
        np.testing.assert_allclose(method.prior, np.full(4, 1 / 3))
        np.testing.assert_allclose(method._make_quantifier().prior, np.full(4, 1 / 3))
        # main's SQS-B keeps the uniform prior
        self.assertEqual(self.main[SQS_B].prior, "uniform")

    def test_kdey_bases_receive_the_prior_and_e51_e54_keep_theirs(self):
        self.assertEqual(binform.E55_KDEY_BASES, ("BayCAP-KDEy-h-inst", "BayCAP-KDEy-h-inst-cb"))
        self.assertNotIn("BayCAP-KDEy-mlp-real", binform.E55_PRIOR_BASES)
        self.assertEqual(binform.E51_BASE_METHODS, ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real"))
        inst = binform.e55_method("BayCAP-KDEy-h-inst_centre", 4)
        cb = binform.e55_method("BayCAP-KDEy-h-inst-cb_edge", 4)
        self.assertEqual((inst.estimand, inst.input_classifier, inst.bandwidth), ("realised-instance", "h", 0.1))
        self.assertEqual((cb.estimand, cb.input_classifier, cb.bandwidth), ("realised-instance", "h", "logloss"))
        np.testing.assert_allclose(inst.prior, np.full(4, 3.0))
        np.testing.assert_allclose(cb.prior, np.full(4, 1 / 3))
        np.testing.assert_allclose(cb._make_quantifier().prior, np.full(4, 1 / 3))
        # main's methods keep the uniform prior
        self.assertEqual(self.main["BayCAP-KDEy-h-inst-cb"].prior, "uniform")

    def test_methods_without_a_prior_are_those_of_main(self):
        for name, cls in (
            ("SQS-H", RQBS),
            ("SQS-Hbs", RQBS),
            ("PQ", PrediQuant),
            ("SQS-H-KDEy", RQBS),
            ("SQS-Hbs-KDEy", RQBS),
            ("PQ-KDEy", PrediQuant),
        ):
            method = binform.e55_method(name, 3)
            reference = self.main[name]
            self.assertIs(type(method), cls)
            self.assertEqual(vars(method).keys(), vars(reference).keys())
            for attribute, value in vars(reference).items():
                if isinstance(value, (bool, int, float, str, type(None))):
                    self.assertEqual(getattr(method, attribute), value, (name, attribute))
        self.assertFalse(binform.e55_method("SQS-H", 3).bootstrap)
        self.assertTrue(binform.e55_method("SQS-Hbs", 3).bootstrap)

    def test_output_folders(self):
        self.assertEqual(binform.e55_experiment("edge"), os.path.join("binform_e55", "edge"))
        self.assertNotEqual(binform.e55_experiment("centre"), binform.e54_experiment("centre", 1000))


def _info(name="wine", classifier="LR", k=3):
    return SimpleNamespace(
        domain="classic",
        d_info=SimpleNamespace(name=name, n_classes=k),
        h_info=SimpleNamespace(name=classifier, full_name=classifier),
    )


class TaskTest(unittest.TestCase):
    def _task(self, method_name, distribution):
        p = _info("wine", "LR", 3)
        with (
            patch.object(binform.PretrainInfo, "load", return_value=p),
            patch.object(binform, "_e55_all_results_exist", return_value=False),
            patch.object(binform, "_prior_task", return_value=["done"]) as core,
        ):
            self.assertEqual(binform._e55_task("path", method_name, distribution), ["done"])
        args, kwargs = core.call_args
        self.assertEqual(args, (p, method_name))
        self.assertEqual(kwargs["size"], 1000)
        np.testing.assert_array_equal(kwargs["requested_prevs"], binform.e55_requested_prevs("wine", 3, distribution))
        self.assertIn(os.path.join("binform_e55", distribution), kwargs["result_path"]("macro-F1"))
        return kwargs

    def test_task_with_a_prior(self):
        kwargs = self._task("SQS-B_centre", "edge")
        self.assertEqual(
            kwargs["static_columns"], {"distribution": "edge", "bag_alpha": 0.1, "prior_alpha": [3.0, 3.0, 3.0]}
        )
        np.testing.assert_allclose(kwargs["build_method"]().prior, np.full(3, 3.0))

    def test_task_without_a_prior(self):
        kwargs = self._task("PQ", "centre")
        self.assertEqual(kwargs["static_columns"], {"distribution": "centre", "bag_alpha": 10.0})
        self.assertIsInstance(kwargs["build_method"](), PrediQuant)

    def test_existing_results_are_skipped(self):
        with (
            patch.object(binform.PretrainInfo, "load", return_value=_info()),
            patch.object(binform, "_e55_all_results_exist", return_value=True),
            patch.object(binform, "_prior_task") as core,
        ):
            outcome = binform._e55_task("path", "SQS-H", "centre")
        core.assert_not_called()
        self.assertEqual(len(outcome), 1)

    def _patched_paths(self):
        infos = {"a": _info("wine", "LR"), "b": _info("wine", "kNN"), "c": _info("other", "LR")}
        return (
            patch.object(binform, "load_info_paths", return_value=list(infos)),
            patch.object(binform.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
            patch.object(binform, "get_filtered_datasets", return_value=["wine"]),
        )

    def test_tasks_order_and_filters(self):
        a, b, c = self._patched_paths()
        with a, b, c, patch.object(binform, "_e55_all_results_exist", side_effect=lambda p, m, d: m.endswith("_edge")):
            tasks = binform.e55_tasks()
            self.assertEqual({path for path, *_ in tasks}, {"a", "b"})
            self.assertFalse(any(name.endswith("_edge") for _, name, _ in tasks))
            # distributions x (methods without _edge) x pairs; the methods without a prior once per distribution
            self.assertEqual(
                len(tasks), 2 * ((N_PRIOR_BASES + len(binform.E55_TCAL_BASES)) * 2 + len(binform.E55_REFERENCES)) * 2
            )
            for distribution in binform.E55_DISTRIBUTIONS:
                block = [name for _, name, d in tasks if d == distribution]
                for reference in binform.E55_REFERENCES:
                    self.assertEqual(block.count(reference), 2)
                order = [binform._e55_method_order(name) for name in block]
                self.assertEqual(order, sorted(order))
                self.assertTrue(block[0].startswith("BayesCAP-Hf"))
            self.assertEqual(
                [d for _, _, d in tasks], sorted((d for _, _, d in tasks), key=binform.E55_DISTRIBUTIONS.index)
            )
            tasks = binform.e55_tasks(classifiers=["LR"], distributions=["edge"], methods=["SQS-Hbs"])
            self.assertEqual(tasks, [("a", "SQS-Hbs", "edge")])
            with self.assertRaises(ValueError):
                binform.e55_tasks(methods=["BayesCAP-Hf-real_well"])
            with self.assertRaises(ValueError):
                binform.e55_tasks(distributions=["uniform"])

    def test_pools(self):
        tasks = [
            ("a", "BayesCAP-Hf-real_uniform", "centre"),
            ("a", "SQS-H", "centre"),
            ("a", "SQS-B_edge", "edge"),
            ("a", "PQ", "edge"),
        ]
        with (
            patch.object(binform, "e55_tasks", return_value=tasks),
            patch.object(binform, "run_tasks") as run,
            patch.dict(binform.capenv, {"N_JOBS": 7}),
        ):
            binform.e55_experiments(binform.get_logger(id="test"))
        calls = [(c.args[1], c.kwargs["n_jobs"], c.kwargs["experiment"], c.kwargs["env"]) for c in run.call_args_list]
        self.assertEqual(
            calls,
            [
                ([tasks[1]], 7, os.path.join("binform_e55", "centre"), binform.THREAD_LIMITS),
                ([tasks[0]], BAYES_N_JOBS, os.path.join("binform_e55", "centre"), binform.BAYES_ENV),
                ([tasks[3]], 7, os.path.join("binform_e55", "edge"), binform.THREAD_LIMITS),
                ([tasks[2]], BAYES_N_JOBS, os.path.join("binform_e55", "edge"), binform.BAYES_ENV),
            ],
        )
        self.assertTrue(all(c.kwargs["worker"] is binform.e55_protocol for c in run.call_args_list))

    def test_arguments(self):
        pargs = binform.parse_args(["--e55", "--datasets", "connect-4", "--distributions", "edge", "--methods", "PQ"])
        self.assertTrue(pargs.e55)
        self.assertFalse(pargs.e54 or pargs.e51)
        self.assertEqual((pargs.datasets, pargs.distributions, pargs.methods), (["connect-4"], ["edge"], ["PQ"]))
        for argv in (["--e54", "--e55"], ["--e51", "--e55"], ["--e55", "--sizes", "100"], ["--methods", "PQ"]):
            with self.assertRaises(SystemExit), patch("sys.stderr"):
                binform.parse_args(argv)
        # E5.4 and the default configuration parse as before
        self.assertEqual(binform.parse_args(["--e54", "--distributions", "edge", "--sizes", "100"]).sizes, [100])
        self.assertFalse(binform.parse_args([]).e55)


class _FakeInterval:
    def __init__(self, samples):
        self.samples = samples
        self.point_estimate = float(np.mean(samples))


class _NoNutsMethod:
    """A method without NUTS diagnostics, as SQS-H, SQS-Hbs and SQS-S."""

    def fit(self, validation, posteriors):
        return self

    def predict_ct_range(self, X, posteriors):
        return np.tile(np.eye(2)[None] / 2, (5, 1, 1))

    def switch(self, acc_fn):
        self.acc_fn = acc_fn

    def ci_from_cts(self, cts):
        return _FakeInterval(np.array([self.acc_fn(ct) for ct in cts]))


class PriorTaskWithoutNutsTest(unittest.TestCase):
    """``_prior_task`` runs a method without NUTS diagnostics and stores no diagnostic column."""

    def test_no_diagnostics(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 2))
        y = np.repeat([0, 1], 200)
        collection = LabelledCollection(X, y, classes=[0, 1])
        dataset = SimpleNamespace(V=collection, U=collection, L_prevalence=np.array([0.5, 0.5]))
        classifier = SimpleNamespace(predict_proba=lambda Z: np.tile([0.3, 0.7], (len(Z), 1)))
        p = SimpleNamespace(
            domain="classic",
            d_info=SimpleNamespace(name="toy", n_classes=2),
            h_info=SimpleNamespace(name="LR", full_name="LR"),
            load_dataset_bundle=lambda: dataset,
            load_pretrained_classifier=lambda d: classifier,
        )
        with (
            patch.object(binform, "EvaluationContext", side_effect=lambda **kw: SimpleNamespace(size=3, **kw)),
            patch.object(binform, "build_result_frame", side_effect=lambda context, **kw: kw),
        ):
            outcomes = binform._prior_task(
                p,
                "SQS-H",
                build_method=_NoNutsMethod,
                requested_prevs=np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.1]]),
                size=50,
                result_path=lambda acc_name: "/nonexistent",
                static_columns={"distribution": "centre", "bag_alpha": 10.0},
            )
        self.assertTrue(outcomes)
        extra = outcomes[0].df["extra_columns"]
        self.assertEqual(set(extra), {"bag_size", "requested_prev", "distribution", "bag_alpha"})
        self.assertEqual(extra["bag_size"], [50] * 3)


class ProtocolCachesTest(unittest.TestCase):
    def test_clears_the_compiled_functions_after_a_task(self):
        square = jax.jit(lambda x: x * x)

        def task(*args):
            square(jnp.ones(3))
            self.assertGreater(square._cache_size(), 0)
            return list(args)

        with patch.object(binform, "_e55_task", side_effect=task):
            outcome = binform.e55_protocol(("path", "SQS-B_edge", "centre"))
        self.assertEqual(outcome, ["path", "SQS-B_edge", "centre"])
        self.assertEqual(square._cache_size(), 0)

    def test_clears_the_caches_when_the_task_raises(self):
        with (
            patch.object(binform, "_e55_task", side_effect=RuntimeError("boom")),
            patch.object(binform.jax, "clear_caches") as clear,
            self.assertRaises(RuntimeError),
        ):
            binform.e55_protocol(("path", "SQS-B_edge", "centre"))
        clear.assert_called_once_with()


# --------------------------------------------------------------------------------------------------- tables

# shift and width of the samples per role: right side narrower, wrong side shifted
ROLE_SHAPES = {"uniform": (0.0, 0.03), "right": (0.0, 0.02), "wrong": (0.08, 0.03), "none": (0.02, 0.04)}
PAIRS = [("wine", "LR", 3), ("wine", "kNN", 3), ("spam", "kNN", 2), ("letter", "SVM", 26)]


def _absent_ct(rng, k):
    ct = _true_ct(rng, k)
    ct[-1, :] = 0.0
    return ct / ct.sum()


def _frames(distribution, seed=0):
    """``base -> role -> rows`` for one distribution; the bags of ``edge`` have one absent class in half of the
    bags."""
    rng = np.random.default_rng(seed)
    frames = {base: {} for base in bsp.METHODS}
    parts = {}
    for dataset, classifier, k in PAIRS:
        true_cts = [
            _absent_ct(rng, k) if distribution == "edge" and bag % 2 else _true_ct(rng, k) for bag in range(N_BAGS)
        ]
        for base in bsp.METHODS:
            for role, name in bsp.method_roles(base, distribution).items():
                shift, width = ROLE_SHAPES[role]
                failed = (2,) if role == "wrong" and dataset == "spam" else ()
                frame = _frame(name, dataset, classifier, k, true_cts, rng, shift, width, failed)
                if base in binform.E55_REFERENCES:
                    frame = frame.drop(columns="nuts_divergences")
                parts.setdefault((base, role), []).append(frame)
    for (base, role), rows in parts.items():
        frames[base][role] = pd.concat(rows, ignore_index=True)
    return frames


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pair_rows, bag_rows = [], []
        for distribution in binform.E55_DISTRIBUTIONS:
            pairs, bags = bsp.distribution_rows(_frames(distribution), "vanilla_accuracy", distribution)
            pair_rows.extend(pairs)
            bag_rows.extend(bags)
        cls.table = bsp.ordered(pd.DataFrame(pair_rows))
        cls.bags = bsp.ordered(pd.DataFrame(bag_rows))
        cls.summary = bsp.summary_table(cls.table)

    def test_roles(self):
        self.assertEqual(bsp.role_priors("centre"), {"uniform": "uniform", "right": "centre", "wrong": "edge"})
        self.assertEqual(bsp.role_priors("edge"), {"uniform": "uniform", "right": "edge", "wrong": "centre"})
        self.assertEqual(bsp.method_roles("PQ", "edge"), {"none": "PQ"})
        self.assertEqual(bsp.method_roles("SQS-B", "edge")["right"], "SQS-B_edge")

    def test_pair_table(self):
        self.assertEqual(len(self.table), 2 * len(bsp.METHODS) * len(PAIRS))
        informed = self.table["method"].isin(binform.E55_PRIOR_BASES)
        part = self.table.loc[informed]
        self.assertTrue((part["d_w_know"] > 0).all())
        self.assertTrue((part["d_w_wrong"] > 0).all())
        self.assertTrue((part["right_amplitude_ratio"] < 1).all())
        self.assertTrue(part["none_winkler"].isna().all())
        references = self.table.loc[~informed]
        self.assertTrue(references["uniform_winkler"].isna().all())
        self.assertTrue(references["none_winkler"].notna().all())
        self.assertTrue(references["d_w_know"].isna().all())
        np.testing.assert_allclose(references["none_share_divergent"], 0.0)
        np.testing.assert_allclose(part["uniform_share_divergent"], 1 / N_BAGS)
        spam = part.loc[part["dataset"] == "spam"]
        np.testing.assert_allclose(spam["wrong_share_failed"], 1 / N_BAGS)
        edge = self.table["distribution"] == "edge"
        np.testing.assert_allclose(self.table.loc[edge, "absent_classes"], 0.5)
        self.assertEqual(set(self.table.loc[edge & informed, "right_prior"]), {"edge"})
        self.assertEqual(set(self.table["size"]), {1000})

    def test_bag_table(self):
        self.assertEqual(len(self.bags), 2 * (N_PRIOR_BASES * 3 + N_REFS) * len(PAIRS) * N_BAGS)
        self.assertEqual(set(self.bags["role"]), set(bsp.ALL_ROLES))
        self.assertTrue(self.bags.loc[self.bags["failed"], "winkler"].isna().all())
        # wrong side, spam, one bag per method with a prior, two distributions
        self.assertEqual(int(self.bags["failed"].sum()), 2 * N_PRIOR_BASES)

    def test_summary(self):
        allk = self.summary.loc[self.summary["K"] == "all"]
        self.assertEqual(len(allk), 2 * (N_PRIOR_BASES * 3 + N_REFS))
        right = allk.loc[allk["role"] == "right"]
        self.assertTrue((right["pairs_lower"] == len(PAIRS)).all())
        self.assertTrue((right["datasets_lower"] == 3).all())
        self.assertTrue((right["d_winkler"] < 0).all())
        self.assertTrue((allk.loc[allk["role"] == "wrong", "d_winkler"] > 0).all())
        none = allk.loc[allk["role"] == "none"]
        self.assertEqual(set(none["method"]), set(binform.E55_REFERENCES))
        self.assertTrue(none[["p_value", "pairs_lower", "d_winkler"]].isna().all().all())
        self.assertTrue(none["winkler"].notna().all())
        self.assertEqual(set(self.summary["K"]), {"all", "K=2", "3<=K<=7", "K>=8"})

    def test_different_bags_raise(self):
        frames = _frames("centre")
        frames["PQ"]["none"] = _frames("centre", seed=1)["PQ"]["none"]
        with self.assertRaises(ValueError):
            bsp.distribution_rows(frames, "vanilla_accuracy", "centre")

    def test_restrict(self):
        frames = _frames("centre")
        frames["SQS-B"]["wrong"] = pd.DataFrame()
        frames["PQ"]["none"] = frames["PQ"]["none"].loc[frames["PQ"]["none"]["dataset"] != "letter"]
        kept = bsp.restrict(frames)
        self.assertNotIn("SQS-B", kept)
        self.assertEqual(len(kept), len(bsp.METHODS) - 1)
        for roles in kept.values():
            for df in roles.values():
                self.assertNotIn("letter", set(df["dataset"]))

    def test_stored_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            for roles in _frames("edge").values():
                for df in roles.values():
                    for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                        folder = os.path.join(
                            root, binform.e55_experiment("edge"), "classic", "vanilla_accuracy", dataset, classifier
                        )
                        os.makedirs(folder, exist_ok=True)
                        part = part.assign(
                            true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                        )
                        part.to_parquet(os.path.join(folder, f"{part['method'].iloc[0]}.parquet"))
            table, bags = bsp.tables(["vanilla_accuracy"], root=root)
            self.assertEqual(len(table), len(bsp.METHODS) * len(PAIRS))
            self.assertEqual(set(table["distribution"]), {"edge"})
            self.assertEqual(len(bags), (N_PRIOR_BASES * 3 + N_REFS) * len(PAIRS) * N_BAGS)
            with self.assertRaises(RuntimeError):
                bsp.tables(["macro-F1"], root=root)

    def test_latex_and_outputs(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"]
        latex = bsp._summary_styler(part).to_latex()
        for text in ("right side", "wrong side", "no prior", "\\bcaphf", "\\sqsb", "\\pq", "Dir"):
            self.assertIn(text, latex)
        self.assertNotIn("vanilla_accuracy", bsp.caption("vanilla_accuracy"))
        with tempfile.TemporaryDirectory() as out_dir:
            bsp.write_tables(self.table, self.bags, out_dir)
            stem = bsp._stem("vanilla_accuracy")
            for path in (
                bsp.csv_path(out_dir),
                bsp.csv_path(out_dir, "bags"),
                bsp.csv_path(out_dir, "summary"),
                os.path.join(out_dir, "tabular", f"{stem}.tex"),
                os.path.join(out_dir, "tabular", f"{stem}_long.tex"),
            ):
                self.assertTrue(os.path.exists(path), path)
            with open(os.path.join(out_dir, f"{stem}_doc.tex")) as f:
                document = f.read()
            for macro in ("\\newcommand{\\sqsb}", "\\newcommand{\\pq}", "\\newcommand{\\sqshbs}"):
                self.assertIn(macro, document)
            self.assertIn(f"tabular/{stem}_long.tex", document)
            self.assertEqual(len(pd.read_csv(bsp.csv_path(out_dir))), len(self.table))
            self.assertEqual(len(pd.read_csv(bsp.csv_path(out_dir, "bags"))), len(self.bags))

    # ----------------------------------------------------------------------------------------------- plots

    def test_violin_cells_and_floor(self):
        cells = plots.violin_cells()
        self.assertEqual(len(cells), 3 + 4 * 3)
        methods = list(dict.fromkeys(method for method, _, _ in cells))
        self.assertEqual(
            methods,
            # by default the references with KDEy (WP-36), the PACC ones left out
            [
                "PQ-KDEy",
                "SQS-H-KDEy",
                "SQS-Hbs-KDEy",
                "SQS-B",
                "BayesCAP-Hf-real",
                "BayCAP-KDEy-h-inst",
                "BayCAP-KDEy-h-inst-cb",
            ],
        )
        self.assertEqual(
            [plots.short_label(m) for m in methods][-3:],
            ["BayCAP-Hf-untempered", "BayCAP-KDEy-h-inst", "BayCAP-KDEy-cb"],
        )
        # wrong side on the left, uniform in the middle, right side on the right
        self.assertEqual([role for method, role, _ in cells if method == "SQS-B"], ["wrong", "uniform", "right"])
        self.assertEqual([role for method, role, _ in cells if method == "PQ-KDEy"], ["none"])
        self.assertEqual(len(plots.violin_cells(())), N_REFS + 4 * 3)
        positions = [position for _, _, position in cells]
        self.assertEqual(positions, sorted(positions))
        values = plots.log_values(pd.DataFrame({"ae": [0.0, 1e-7, 0.01, np.nan]}), "ae")
        np.testing.assert_allclose(values, [-5.0, -5.0, -2.0])

    def test_excluded_methods_have_no_violin(self):
        exclude = [*plots.DEFAULT_EXCLUDE, "SQS-B"]
        cells = plots.violin_cells(exclude)
        self.assertEqual(len(cells), 3 + 3 * 3)
        self.assertNotIn("SQS-B", [method for method, _, _ in cells])
        fig = plots.plot_violins(self.bags, "edge", exclude=exclude)
        self.assertEqual(len(fig.axes[0].get_xticklabels()), len(cells))
        plt.close(fig)
        args = plots.parse_args(["--exclude-methods", "SQS-B"])
        self.assertEqual(args.exclude_methods, ["SQS-B"])

    def test_violin_stats_closed_tails(self):
        values = np.array([-2.0, -1.5, -1.2, -1.0, -0.4])
        stats = plots.violin_stats(values, 0.3)
        # the density extends beyond the extreme values and closes towards zero there
        self.assertLess(stats["coords"][0], values.min())
        self.assertGreater(stats["coords"][-1], values.max())
        self.assertLess(stats["vals"][0], 0.2 * stats["vals"].max())
        self.assertLess(stats["vals"][-1], 0.2 * stats["vals"].max())
        # not below the floor
        floored = plots.violin_stats(np.array([-5.0, -5.0, -4.9, -3.0]), 0.3)
        self.assertGreaterEqual(floored["coords"][0], np.log10(plots.FLOOR))
        # one value or equal values
        single = plots.violin_stats(np.array([-2.0]), 0.3)
        self.assertTrue(np.all(np.isfinite(single["vals"])))

    def test_violins(self):
        bags = self.bags.copy()
        bags.loc[bags.index[:5], "ae"] = 0.0  # floored
        for distribution in binform.E55_DISTRIBUTIONS:
            for level in plots.LEVELS:
                fig = plots.plot_violins(bags, distribution, level=level)
                self.assertEqual(len([ax for ax in fig.axes if ax.get_title()]), 2)
                self.assertIn("E5.5", fig.get_suptitle())
                self.assertIn("all pairs", fig.get_suptitle())
                labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
                self.assertEqual(labels[:4], ["no prior", "no prior", "no prior", "wrong"])
                self.assertEqual(len(labels), len(plots.violin_cells()))
                fig.canvas.draw()
                plt.close(fig)
        self.assertIsNone(plots.plot_violins(bags, "edge", "macro-K"))
        with self.assertRaises(ValueError):
            plots.plot_violins(bags, "edge", level="datasets")
        with self.assertRaises(ValueError):
            plots.plot_violins(bags, "edge", group="ternary")

    def test_violin_groups(self):
        binary = plots.pair_group(self.bags, "binary")
        multiclass = plots.pair_group(self.bags, "multiclass")
        self.assertTrue((binary["n_classes"] == 2).all())
        self.assertTrue((multiclass["n_classes"] > 2).all())
        self.assertEqual(len(binary) + len(multiclass), len(self.bags))
        for group in plots.PAIR_GROUPS:
            fig = plots.plot_violins(self.bags, "centre", group=group)
            if fig is None:
                self.assertTrue(plots.pair_group(self.bags, group).empty)
                continue
            self.assertIn(plots.PAIR_GROUPS[group], fig.get_suptitle())
            plt.close(fig)

    def test_violin_means(self):
        bags = self.bags
        fig = plots.plot_violins(bags, "edge", level="bags")
        ax = fig.axes[1]  # Winkler score
        n_cells = len(plots.violin_cells())
        diamonds = [c for c in ax.collections if c.get_offsets().shape[0] == n_cells][-1]
        part = bags.loc[(bags["distribution"] == "edge") & (bags["method"] == "PQ-KDEy") & (bags["role"] == "none")]
        expected = np.log10(max(part["winkler"].mean(), plots.FLOOR))
        self.assertAlmostEqual(diamonds.get_offsets()[0][1], expected)
        self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], ["median", "mean"])
        plt.close(fig)

    def test_pair_violins(self):
        means = plots.pair_means(self.bags)
        self.assertEqual(len(means), len(self.bags.groupby(plots.PAIR_KEYS)))
        # the per-pair means are those of the per-pair table
        for method in binform.E55_PRIOR_BASES:
            for role in bsp.ROLES:
                for distribution in binform.E55_DISTRIBUTIONS:
                    mine = means.loc[
                        (means["method"] == method) & (means["role"] == role) & (means["distribution"] == distribution)
                    ].sort_values(["dataset", "classifier"])
                    table = self.table.loc[
                        (self.table["method"] == method) & (self.table["distribution"] == distribution)
                    ].sort_values(["dataset", "classifier"])
                    np.testing.assert_allclose(mine["winkler"], table[f"{role}_winkler"])
        fig = plots.plot_violins(self.bags, "centre", level="pairs")
        self.assertIn("per-pair means", fig.get_suptitle())
        fig.canvas.draw()
        plt.close(fig)

    def test_scatter(self):
        for distribution in binform.E55_DISTRIBUTIONS:
            fig = plots.plot_scatter(self.table, "SQS-B", distribution)
            self.assertEqual(len(fig.axes), 2 + 1)  # two roles, one size, and the colour bar
            titles = [ax.get_title() for ax in fig.axes[:2]]
            self.assertTrue(any("right side" in t for t in titles))
            self.assertTrue(any("wrong side" in t for t in titles))
            fig.canvas.draw()
            plt.close(fig)
        self.assertIsNone(plots.plot_scatter(self.table, "SQS-H", "edge"))

    def test_generate_plots(self):
        with tempfile.TemporaryDirectory() as out_dir:
            pairs, bags = os.path.join(out_dir, "pairs.csv"), os.path.join(out_dir, "bags.csv")
            self.table.to_csv(pairs, index=False)
            self.bags.to_csv(bags, index=False)
            args = plots.parse_args(["--csv", pairs, "--bags-csv", bags, "--no-svg", "--distributions", "edge"])
            with patch.object(plots, "plot_dir", return_value=out_dir), patch("builtins.print"):
                plots.generate_plots(args)
            names = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
            groups = [g for g in plots.PAIR_GROUPS if not plots.pair_group(self.bags, g).empty]
            # by default the pair violins and the scatters, not the bag violins
            self.assertEqual(len(names), len(groups) + N_PRIOR_BASES)
            self.assertIn("binform_side_prior_pair_violins_edge_all_vanilla_accuracy.png", names)
            self.assertFalse(any("bag_violins" in name for name in names))
            args = plots.parse_args(
                ["--bags-csv", bags, "--no-svg", "--distributions", "edge", "--figures", "bag-violins"]
            )
            with patch.object(plots, "plot_dir", return_value=out_dir), patch("builtins.print"):
                plots.generate_plots(args)
            self.assertIn("binform_side_prior_bag_violins_edge_all_vanilla_accuracy.png", os.listdir(out_dir))


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
    unittest.main()
