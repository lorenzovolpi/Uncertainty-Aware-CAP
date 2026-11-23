"""Tests of E4.4 (card WP-28) in the study target_size: levels, threshold, SQS bag size, hash, scores, tables, plots."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from quapy.data import LabelledCollection

from uncawe.experiments import target_size as ts
from uncawe.experiments.methods import CONFIDENCE_LEVEL, SAMPLE_SIZE
from uncawe.plots import target_size_scores as plots
from uncawe.tables import main as main_tables
from uncawe.tables import target_size as e42_tables
from uncawe.tables import target_size_scores as tables


def _val_labels(counts):
    return np.repeat(np.arange(len(counts)), counts)


class TestLevels(unittest.TestCase):
    def test_all_levels_by_decreasing_fraction_with_their_folders(self):
        fractions = list(ts.ALL_LEVELS.values())
        self.assertEqual(fractions, sorted(fractions, reverse=True))
        self.assertEqual(set(ts.ALL_LEVELS), set(ts.LEVEL_DIRS))
        self.assertEqual(len(set(ts.LEVEL_DIRS.values())), len(ts.LEVEL_DIRS))
        self.assertEqual(list(ts.E44_LEVELS), ["full", "1/sqrt2", "1/2", "1/(2sqrt2)", "1/4"])
        self.assertAlmostEqual(ts.E44_LEVELS["1/(2sqrt2)"] * 2 * np.sqrt(2), 1.0)
        # the levels of E4.2 are unchanged (they enter config_hash)
        self.assertEqual(ts.LEVELS, {"full": 1.0, "1/4": 1 / 4, "1/16": 1 / 16})
        self.assertEqual(
            ts.result_folder(1000, "1/(2sqrt2)", 1), os.path.join("target_size", "u1000", "v1_2sqrt2", "s1")
        )

    def test_new_levels_are_nested_with_those_of_e42(self):
        val_counts = np.array([400, 160, 37, 18, 3, 1])
        val_y = np.random.default_rng(5).permutation(_val_labels(val_counts))
        order = sorted(ts.ALL_LEVELS.items(), key=lambda item: -item[1])
        for seed in range(3):
            previous = None
            for _, fraction in order:
                index = ts.subsample_index(val_y, fraction, "toy", seed)
                counts = np.bincount(val_y[index], minlength=len(val_counts))
                np.testing.assert_array_equal(counts, ts.subsample_counts(val_counts, fraction))
                self.assertTrue(np.all(counts >= 1))
                if previous is not None:
                    self.assertTrue(set(index) <= set(previous))
                previous = index
        # the cells of E4.2 at 1/4 are the same subsamples
        quarter = ts.subsample_index(val_y, ts.E44_LEVELS["1/4"], "toy", 2)
        np.testing.assert_array_equal(quarter, ts.subsample_index(val_y, ts.LEVELS["1/4"], "toy", 2))

    def test_threshold_option(self):
        counts = np.array([18, 400])
        # 18/(2 sqrt 2) = 6.4 -> 6 < 10
        self.assertFalse(ts.level_available(counts, "1/(2sqrt2)"))
        self.assertTrue(ts.level_available(counts, "1/(2sqrt2)", None))
        self.assertTrue(ts.level_available(counts, "1/sqrt2"))
        self.assertTrue(ts.level_available(counts, "1/16", None))

    def test_tasks_without_threshold_run_every_level(self):
        p = SimpleNamespace(
            d_info=SimpleNamespace(name="poker_hand", n_classes=2), h_info=SimpleNamespace(name="LR", full_name="LR_x")
        )
        kwargs = {"methods": ("SQS-H",), "sizes": (1000,), "levels": tuple(ts.E44_LEVELS)}
        with (
            patch.object(ts, "_pretrains", return_value=[("info", p)]),
            patch.object(ts, "_val_counts", return_value=np.array([18, 400])),
            patch.object(ts, "_all_results_exist", return_value=False),
        ):
            with_threshold = ts.tasks(**kwargs)
            without = ts.tasks(min_class_count=None, **kwargs)
        self.assertEqual({t.level for t in with_threshold}, {"full", "1/sqrt2"})
        self.assertEqual({t.level for t in without}, set(ts.E44_LEVELS))
        self.assertEqual(len(without), 1 + 4 * ts.N_V_SEEDS)
        self.assertTrue(all(t.min_class_count is None for t in without))

    def test_unknown_level_or_method(self):
        with self.assertRaises(ValueError):
            ts.tasks(methods=("O-LEAP",))
        with self.assertRaises(ValueError):
            ts.tasks(levels=("1/8",))


class TestMethods(unittest.TestCase):
    def test_sqs_methods_get_the_size_of_the_task(self):
        for size in (100, 1000, 10000):
            for name in ts.SQS_METHODS:
                self.assertEqual(ts.make_method(name, size).sample_size, size, name)
        self.assertFalse(ts.make_method("SQS-H", 300).bootstrap)
        self.assertTrue(ts.make_method("SQS-Hbs", 300).bootstrap)

    def test_at_1000_the_sqs_methods_are_those_of_main(self):
        from uncawe.experiments.methods import gen_methods

        stored = dict(gen_methods())
        for name in ts.SQS_METHODS:
            ours, theirs = ts.make_method(name, SAMPLE_SIZE), stored[name]
            self.assertIs(type(ours), type(theirs))
            for attribute in ("sample_size", "num_samples", "random_state", "bootstrap", "alpha", "rho"):
                if hasattr(theirs, attribute):
                    self.assertEqual(getattr(ours, attribute), getattr(theirs, attribute), (name, attribute))

    def test_nuts_methods(self):
        self.assertTrue(ts._samples_with_nuts(ts.make_method("BayesCAP-Hf-real")))
        self.assertTrue(ts._samples_with_nuts(ts.make_method("SQS-B")))
        self.assertFalse(ts._samples_with_nuts(ts.make_method("SQS-H")))
        self.assertFalse(ts._samples_with_nuts(ts.make_method("PQ")))

    def test_pools(self):
        tasks = [ts.Task("i", name, 1000, "full", 0, 30) for name in ts.E44_METHODS]
        bayes, regular = ts.split_pools(tasks)
        self.assertEqual({t.method_name for t in bayes}, {"BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real", "SQS-B"})
        self.assertEqual({t.method_name for t in regular}, {"SQS-H", "SQS-Hbs", "PQ"})


class TestHashAndArgs(unittest.TestCase):
    def test_hash_of_the_stored_cells_of_e42_is_unchanged(self):
        # the hash stored in every file of E4.2 (R=30), e.g. u1000/v1_16/s2/.../letter/RF/BayesCAP-Hf-real.parquet
        self.assertEqual(ts.config_hash(30), "25fc61de00cb")

    def test_presets(self):
        main15 = [f"d{i}" for i in range(15)]
        with patch.object(ts, "main_datasets", return_value=main15):
            a = ts.parse_args(["--run", "--e44a"])
            b = ts.parse_args(["--run", "--e44b"])
            pilot = ts.parse_args(["--run", "--e44a", "--datasets", "letter", "--methods", "PQ"])
        self.assertEqual((a.sizes, a.levels, a.min_class_count), (list(ts.U_SIZES), ["full"], ts.MIN_CLASS_COUNT))
        self.assertEqual((a.datasets, a.classifiers, a.methods), (main15, ["LR"], list(ts.E44_METHODS)))
        self.assertEqual((b.sizes, b.levels, b.min_class_count), ([1000], list(ts.E44_LEVELS), None))
        self.assertEqual((pilot.datasets, pilot.methods, pilot.classifiers), (["letter"], ["PQ"], ["LR"]))
        default = ts.parse_args(["--run"])
        self.assertEqual((default.datasets, default.methods), (list(ts.DATASETS), list(ts.METHODS)))
        self.assertEqual((default.levels, default.min_class_count, default.preset), (list(ts.LEVELS), 10, None))
        with patch("sys.stderr"):
            for argv in (["--run", "--e44a", "--sizes", "100"], ["--run", "--e44b", "--u-grid"]):
                with self.assertRaises(SystemExit):
                    ts.parse_args(argv)


class TestRunTask(unittest.TestCase):
    """A task of a method without NUTS (SQS-H) on a toy pair, end to end, without writing anything."""

    def test_sqs_h_task(self):
        rng = np.random.default_rng(0)
        n_pool = 400
        val_y = _val_labels([150, 150])
        pool_y = rng.integers(0, 2, n_pool)

        def posteriors(y):
            p1 = np.clip(0.75 * y + 0.25 * (1 - y) + rng.normal(0, 0.15, y.size), 0.01, 0.99)
            return np.column_stack([1 - p1, p1])

        val_post, pool_post = posteriors(val_y), posteriors(pool_y)
        bundle = SimpleNamespace(
            V=LabelledCollection(val_post.copy(), val_y, classes=[0, 1]),
            U=SimpleNamespace(X=pool_post.copy(), y=pool_y),
            L_prevalence=np.array([0.5, 0.5]),
        )
        p = SimpleNamespace(
            domain="classic",
            d_info=SimpleNamespace(name="toy", n_classes=2, collection="uci_binary"),
            h_info=SimpleNamespace(name="LR", full_name="LR_x"),
            load_dataset_bundle=lambda: bundle,
            load_posteriors=lambda: (val_post, pool_post),
        )
        task = ts.Task("info", "SQS-H", 100, "1/2", 1, 2, None)
        with (
            patch.object(ts.PretrainInfo, "load", return_value=p),
            patch.object(ts, "_all_results_exist", return_value=False),
            patch.object(ts, "_result_path", return_value="/nonexistent.parquet"),
        ):
            outcomes = ts.protocol(task)
        self.assertTrue(outcomes)
        frame = outcomes[0].df
        self.assertEqual(len(frame), 2 * len(ts.PREVALENCE_KINDS))
        self.assertNotIn("nuts_divergences", frame.columns)
        self.assertTrue((frame["bag_size"] == 100).all())
        self.assertTrue((frame["v_size"] == 150).all())
        self.assertEqual(frame["v_class_counts"].iloc[0], [75, 75])
        self.assertTrue((frame["config_hash"] == ts.config_hash(2)).all())
        self.assertTrue(np.isfinite(frame["estim_accs"].to_numpy(dtype=float)).all())


# --------------------------------------------------------------------------------------------------- tables, plots


def _synthetic_results(seed=0, n_bags=4):
    """Stored-like rows of E4.4: two datasets (one of them poker_hand, flagged at 1/4), three methods, part a (full V at
    three sizes) and part b (|U|=1000, three levels, two seeds each), three prevalences."""
    rng = np.random.default_rng(seed)
    rows = []
    methods = ("BayesCAP-Hf-real", "SQS-H", "PQ")
    cells = [(size, "full", 0) for size in (100, 1000, 10000)]
    cells += [(1000, level, s) for level in ("1/2", "1/4") for s in (0, 1)]
    for dataset in ("toy", "poker_hand"):
        for method in methods:
            for size, level, v_seed in cells:
                fraction = ts.ALL_LEVELS[level]
                counts = [
                    int(400 * fraction),
                    8 if (dataset == "poker_hand" and level == "1/4") else int(40 * fraction),
                ]
                for kind in ts.PREVALENCE_KINDS:
                    for b in range(n_bags):
                        pi = rng.dirichlet([5, 5])
                        recalls = np.array([0.8, 0.7])
                        true = np.array(
                            [
                                [pi[0] * recalls[0], pi[0] * (1 - recalls[0])],
                                [pi[1] * (1 - recalls[1]), pi[1] * recalls[1]],
                            ]
                        )
                        width = 0.3 / np.sqrt(size)
                        estim = np.trace(true) + rng.normal(0, 0.02)
                        draws = estim + rng.normal(0, width, 200)
                        rows.append(
                            {
                                "method": method,
                                "dataset": dataset,
                                "classifier": "LR",
                                "n_classes": 2,
                                "bag_size": size,
                                "v_level": level,
                                "v_fraction": fraction,
                                "v_seed": v_seed,
                                "v_size": sum(counts),
                                "v_class_counts": counts,
                                "prevalence_kind": kind,
                                "bag_index": b,
                                "config_hash": "h",
                                "true_cts": true,
                                "estim_cts": true + rng.normal(0, 0.005, (2, 2)),
                                "estim_accs": estim,
                                "estim_acc_ranges": draws,
                                "acc_err": abs(estim - np.trace(true)),
                                "t_train": 1.0,
                                "t_test_ave": 0.1,
                            }
                        )
    return pd.DataFrame(rows)


class TestScores(unittest.TestCase):
    def setUp(self):
        self.results = _synthetic_results()
        pool = {(d, "LR"): {"d_pool": np.array([0.8, 0.7])} for d in ("toy", "poker_hand")}
        self.out = tables.compute(self.results, pool)

    def test_winkler_is_the_amplitude_plus_the_penalty_of_a_miss(self):
        alpha = 1 - CONFIDENCE_LEVEL
        draws = np.linspace(0.6, 0.8, 201)
        low, high = main_tables._confidence_interval(draws).interval()
        self.assertGreater(high, low)
        for true in (0.7, low, high):
            self.assertAlmostEqual(main_tables.get_acc_winkler("m", true, draws), high - low)
            self.assertEqual(main_tables.get_acc_coverage("m", true, draws), 1)
        for miss in (0.01, 0.1):
            self.assertAlmostEqual(main_tables.get_acc_winkler("m", low - miss, draws), high - low + 2 / alpha * miss)
            self.assertAlmostEqual(main_tables.get_acc_winkler("m", high + miss, draws), high - low + 2 / alpha * miss)
            self.assertEqual(main_tables.get_acc_coverage("m", high + miss, draws), 0)

    def test_bag_scores(self):
        bags = self.out["bags"]
        self.assertEqual(len(bags), len(self.results))
        np.testing.assert_allclose(bags["ae"], np.abs(bags["estim_acc"] - bags["true_acc"]), atol=1e-12)
        self.assertTrue((bags["winkler"] >= bags["amplitude"] - 1e-12).all())
        covered = bags["coverage"] == 1
        np.testing.assert_allclose(bags.loc[covered, "winkler"], bags.loc[covered, "amplitude"])
        self.assertTrue(np.isfinite(bags[["e1", "e2", "e3"]].to_numpy()).all())
        flagged = bags.loc[bags["below_threshold"]]
        self.assertEqual(set(flagged["dataset"]), {"poker_hand"})
        self.assertEqual(set(flagged["v_level"]), {"1/4"})

    def test_seeds_and_pairs(self):
        seeds, pairs = self.out["seeds"], self.out["pairs"]
        self.assertTrue((seeds["n_bags"] == 3 * 4).all())
        one = seeds.loc[(seeds["method"] == "PQ") & (seeds["dataset"] == "toy") & (seeds["v_level"] == "1/2")]
        row = pairs.loc[(pairs["method"] == "PQ") & (pairs["dataset"] == "toy") & (pairs["v_level"] == "1/2")].iloc[0]
        self.assertEqual(row["n_seeds"], 2)
        self.assertAlmostEqual(row["winkler"], one["winkler"].mean())
        self.assertAlmostEqual(row["winkler_sd"], one["winkler"].std(ddof=1))
        full = pairs.loc[pairs["v_level"] == "full"]
        self.assertTrue(full["winkler_sd"].isna().all())
        self.assertEqual(list(pairs["method"].unique()), ["BayesCAP-Hf-real", "SQS-H", "PQ"])

    def test_summary(self):
        summary = self.out["summary"]
        a = summary.loc[(summary["part"] == "a") & (summary["subset"] == "all")]
        self.assertEqual(sorted(a["x"].unique()), [100.0, 1000.0, 10000.0])
        self.assertTrue((a["n_pairs"] == 2).all())
        b = summary.loc[(summary["part"] == "b") & (summary["subset"] == "all")]
        self.assertEqual(sorted(b["x"].unique()), [0.25, 0.5, 1.0])
        self.assertEqual(set(b.loc[b["n_flagged"] > 0, "x"]), {0.25})
        check = summary.loc[(summary["part"] == "b") & (summary["subset"] == "without poker_hand")]
        self.assertTrue((check["n_pairs"] == 1).all() and (check["n_flagged"] == 0).all())
        pairs = self.out["pairs"]
        sub = pairs.loc[(pairs["method"] == "SQS-H") & (pairs["v_level"] == "full") & (pairs["bag_size"] == 100)]
        row = a.loc[(a["method"] == "SQS-H") & (a["x"] == 100)].iloc[0]
        self.assertAlmostEqual(row["ae_median"], sub["ae"].median())
        self.assertAlmostEqual(row["winkler_q75"], sub["winkler"].quantile(0.75))
        self.assertAlmostEqual(row["ae_mean"], sub["ae"].mean())
        self.assertAlmostEqual(row["winkler_se"], sub["winkler"].std(ddof=1) / np.sqrt(len(sub)))
        self.assertTrue(check["ae_se"].isna().all())  # one pair: no standard error
        for name in tables.SCORES:
            self.assertTrue((summary[f"{name}_q25"] <= summary[f"{name}_median"] + 1e-12).all())
            self.assertTrue((summary[f"{name}_median"] <= summary[f"{name}_q75"] + 1e-12).all())

    def test_slopes(self):
        slopes = self.out["slopes"]
        a = slopes.loc[slopes["part"] == "a"]
        self.assertEqual(len(a), 3 * 2)
        # the synthetic amplitude is proportional to |U|^{-1/2}
        np.testing.assert_allclose(a["amplitude_slope"], -0.5, atol=0.05)

    def test_dropped_bags_are_counted(self):
        results = self.results.copy()
        results.at[0, "estim_accs"] = np.nan
        results.at[1, "estim_acc_ranges"] = np.array([np.nan])
        out = tables.compute(results, split=False)
        self.assertEqual(int(out["seeds"]["n_dropped"].sum()), 2)
        self.assertEqual(len(out["bags"]), len(results) - 2)
        self.assertTrue(out["bags"]["e1"].isna().all())

    def test_latex_frames(self):
        a = tables.part_frame(self.out["pairs"], "a")
        self.assertEqual(a.index.names, ["method", "dataset", "$|U|$"])
        self.assertEqual(a.shape, (3 * 2 * 3, 5))
        b = tables.part_frame(self.out["pairs"], "b")
        self.assertEqual(b.shape, (3 * 2 * 3, 9))
        levels = [index[2] for index in b.index]
        self.assertIn("1/4$^*$", levels)
        self.assertIn("SQS-S", {index[0] for index in a.index})

    def test_tables_and_plots_are_written(self):
        with tempfile.TemporaryDirectory() as folder:
            pargs = SimpleNamespace(table_dir=folder, datasets=None)
            with (
                patch.object(tables, "load_results", return_value=self.results),
                patch.object(
                    tables,
                    "load_pool_info",
                    return_value={(d, "LR"): {"d_pool": np.array([0.8, 0.7])} for d in ("toy", "poker_hand")},
                ),
            ):
                tables.gen_tables(pargs)
            for name in tables.CSV_NAMES:
                self.assertTrue(os.path.exists(os.path.join(folder, f"{tables.TABLE_NAME}_{name}.csv")), name)
            for part in ("a", "b"):
                self.assertTrue(os.path.exists(os.path.join(folder, f"{tables.stem(part)}_doc.tex")))
            args = plots.parse_args(["--no-svg", "--add-stem", "test"])
            with patch.object(plots.env, "root_dir", folder):
                plots.generate_plots(args, summary=self.out["summary"])
                args = plots.parse_args(["--no-svg", "--without", "--figures", "v-scores"])
                plots.generate_plots(args, summary=self.out["summary"])
            for figure, stem in (
                ("u_scores", "target_size_u-scores_test"),
                ("v_scores", "target_size_v-scores_test"),
                ("v_scores", "target_size_v-scores_without_poker_hand"),
            ):
                self.assertTrue(os.path.exists(os.path.join(folder, "plots", "target_size", figure, f"{stem}.png")))

    def test_plot(self):
        for part in ("a", "b"):
            fig = plots.plot_scores(self.out["summary"], part)
            left, right = fig.axes[:2]
            self.assertEqual(right.get_yscale(), "log")
            self.assertEqual(left.get_xscale(), "log")
            # one error-bar line per method, no band and no open marker
            self.assertEqual(len(left.containers), 3)
            self.assertEqual(len(left.collections), 3)  # the vertical bars of the three methods
            self.assertFalse(any(line.get_markerfacecolor() == "white" for line in left.get_lines()))
            # the methods are shifted sideways around each size, symmetrically
            xs = [np.asarray(container.lines[0].get_xdata(), dtype=float) for container in left.containers]
            sizes = np.sort(self.out["summary"].query("part == @part")["x"].unique())
            self.assertTrue(np.allclose(np.sqrt(xs[0] * xs[-1]), sizes))
            self.assertTrue(np.allclose(xs[1], sizes))
            self.assertTrue((xs[0] < xs[1]).all() and (xs[1] < xs[2]).all())
            plt.close(fig)
        with self.assertRaises(RuntimeError):
            plots.plot_scores(self.out["summary"].iloc[:0], "a")


class TestE42TablesKeepTheirGrid(unittest.TestCase):
    def test_load_results_leaves_out_the_cells_of_e44(self):
        rows = []
        for dataset, level, method in (
            ("letter", "1/4", "BayesCAP-Hf-real"),
            ("letter", "1/2", "BayesCAP-Hf-real"),
            ("isolet", "full", "BayesCAP-Hf-real"),
            ("letter", "full", "SQS-H"),
        ):
            rows.append({"dataset": dataset, "v_level": level, "method": method, "schema_version": 1})
        frame = pd.DataFrame(rows)
        with (
            patch.object(e42_tables, "result_files", return_value=["x"]),
            patch.object(e42_tables.ResultsDataFrame, "load_result", return_value=frame),
        ):
            loaded = e42_tables.load_results()
        self.assertEqual(
            loaded[["dataset", "v_level", "method"]].values.tolist(), [["letter", "1/4", "BayesCAP-Hf-real"]]
        )


if __name__ == "__main__":
    unittest.main()
