import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uncawe import error_split as es
from uncawe.experiments import target_size as ts
from uncawe.experiments.execution import ExperimentOutcome, save_experiment_results
from uncawe.plots import target_size as plots
from uncawe.tables import target_size as tables


def _val_labels(counts):
    return np.repeat(np.arange(len(counts)), counts)


class TestBags(unittest.TestCase):
    def test_class_counts_sum_to_the_size_and_follow_the_prevalence(self):
        rng = np.random.default_rng(0)
        for k in (2, 5, 26):
            for size in (100, 1000, 10000):
                prev = rng.dirichlet(np.ones(k))
                counts = ts.class_counts(prev, size)
                self.assertEqual(counts.sum(), size)
                self.assertTrue(np.all(np.abs(counts - prev * size) < 1))
        np.testing.assert_array_equal(ts.class_counts(np.full(3, 1 / 3), 100), [34, 33, 33])

    def test_bags_have_the_requested_counts_and_size(self):
        pool_y = np.repeat(np.arange(3), [50, 30, 20])
        pool_pred = pool_y.copy()
        pool_pred[::7] = (pool_pred[::7] + 1) % 3
        bags = ts.make_bags("toy", pool_y, pool_pred, np.array([40, 30, 15]), size=1000, n_bags=4)
        self.assertEqual(len(bags), 4 * len(ts.PREVALENCE_KINDS))
        for bag in bags:
            counts = np.bincount(pool_y[bag.index], minlength=3)
            np.testing.assert_array_equal(counts, ts.class_counts(bag.requested_prev, 1000))
            self.assertEqual(bag.index.size, 1000)
            np.testing.assert_allclose(bag.true_ct.sum(axis=1), counts / 1000)
            self.assertAlmostEqual(bag.true_ct.sum(), 1.0)

    def test_bags_are_reproducible_and_depend_on_the_bag_index(self):
        pool_y = np.repeat(np.arange(2), [60, 40])
        first = ts.make_bags("toy", pool_y, pool_y, np.array([10, 20]), size=200, n_bags=3)
        again = ts.make_bags("toy", pool_y, pool_y, np.array([10, 20]), size=200, n_bags=3)
        for a, b in zip(first, again):
            np.testing.assert_array_equal(a.index, b.index)
            self.assertEqual(a.seed, b.seed)
        self.assertFalse(np.array_equal(first[0].index, first[1].index))
        self.assertNotEqual(ts.bag_seed("toy", 200, 0, 0), ts.bag_seed("other", 200, 0, 0))
        self.assertNotEqual(ts.bag_seed("toy", 200, 0, 0), ts.bag_seed("toy", 1000, 0, 0))

    def test_bags_larger_than_the_pool_repeat_instances(self):
        pool_y = np.repeat(np.arange(2), [20, 10])
        bags = ts.make_bags("toy", pool_y, pool_y, np.array([5, 5]), size=300, n_bags=2)
        for bag in bags:
            self.assertEqual(bag.index.size, 300)
            self.assertLess(np.unique(bag.index).size, bag.index.size)
            self.assertTrue(np.all((bag.index >= 0) & (bag.index < pool_y.size)))
            np.testing.assert_array_equal(
                np.bincount(pool_y[bag.index], minlength=2), ts.class_counts(bag.requested_prev, 300)
            )

    def test_prevalences(self):
        val_counts = np.array([30, 5, 20, 5])
        prevs = ts.target_prevalences("toy", val_counts)
        self.assertEqual(tuple(prevs), ts.PREVALENCE_KINDS)
        for prev in prevs.values():
            self.assertAlmostEqual(prev.sum(), 1.0)
        np.testing.assert_allclose(prevs["uniform"], 0.25)
        self.assertEqual(int(np.argmax(prevs["vertex"])), 1)
        self.assertAlmostEqual(prevs["vertex"][1], ts.VERTEX_MASS)
        np.testing.assert_allclose(prevs["skewed"], ts.target_prevalences("toy", val_counts)["skewed"])


class TestSubsamples(unittest.TestCase):
    def test_nested_stratified_reproducible(self):
        val_counts = np.array([400, 160, 37])
        val_y = np.random.default_rng(3).permutation(_val_labels(val_counts))
        for seed in range(3):
            quarter = ts.subsample_index(val_y, 1 / 4, "toy", seed)
            sixteenth = ts.subsample_index(val_y, 1 / 16, "toy", seed)
            self.assertTrue(set(sixteenth) <= set(quarter))
            for fraction, index in ((1 / 4, quarter), (1 / 16, sixteenth)):
                np.testing.assert_array_equal(
                    np.bincount(val_y[index], minlength=3), ts.subsample_counts(val_counts, fraction)
                )
                self.assertEqual(len(set(index)), index.size)
            np.testing.assert_array_equal(quarter, ts.subsample_index(val_y, 1 / 4, "toy", seed))
        self.assertFalse(
            np.array_equal(ts.subsample_index(val_y, 1 / 4, "toy", 0), ts.subsample_index(val_y, 1 / 4, "toy", 1))
        )
        np.testing.assert_array_equal(ts.subsample_index(val_y, 1.0, "toy", 5), np.arange(val_y.size))

    def test_levels_need_the_minimum_class_count(self):
        # 1/16 of 150 is 9.375 -> 9 < 10; 1/16 of 160 is 10
        self.assertFalse(ts.level_available(np.array([150, 200]), "1/16"))
        self.assertTrue(ts.level_available(np.array([160, 200]), "1/16"))
        self.assertFalse(ts.level_available(np.array([37, 200]), "1/4"))
        self.assertTrue(ts.level_available(np.array([37, 200]), "full"))

    def test_mean_recall_over_subsamples_tends_to_the_full_recall(self):
        # §T.4: E[d^{V'} | V] = d^V, and the variance of a share drawn without replacement
        val_counts = np.array([200, 120])
        val_y = _val_labels(val_counts)
        correct = np.random.default_rng(1).random(val_y.size) < np.where(val_y == 0, 0.8, 0.6)
        full = np.array([correct[val_y == j].mean() for j in range(2)])
        draws = []
        for seed in range(2000):
            index = ts.subsample_index(val_y, 1 / 4, "toy", seed)
            draws.append([correct[index][val_y[index] == j].mean() for j in range(2)])
        draws = np.array(draws)
        n_sub = ts.subsample_counts(val_counts, 1 / 4)
        variance = full * (1 - full) / n_sub * (val_counts - n_sub) / (val_counts - 1)
        np.testing.assert_array_less(np.abs(draws.mean(axis=0) - full), 4 * np.sqrt(variance / len(draws)))
        np.testing.assert_allclose(draws.var(axis=0), variance, rtol=0.1)


class TestCellParts(unittest.TestCase):
    def test_parts_sum_and_bag_parts_sum_to_zero(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=60)
        cells = np.repeat(["a", "b", "c"], 20)
        fixed, bag = es.cell_parts(values, cells)
        np.testing.assert_allclose(fixed + bag, values)
        for cell in "abc":
            self.assertAlmostEqual(bag[cells == cell].sum(), 0.0, places=12)
            np.testing.assert_allclose(fixed[cells == cell], values[cells == cell].mean())

    def test_se_and_seeds(self):
        bag = np.array([-1.0, 1.0, -1.0, 1.0])
        self.assertAlmostEqual(es.fixed_se(bag), np.std(bag, ddof=1) / 2)
        self.assertTrue(np.isnan(es.fixed_se([0.3])))
        mean, sd = es.over_seeds([0.1, 0.2, 0.3])
        self.assertAlmostEqual(mean, 0.2)
        self.assertAlmostEqual(sd, 0.1)
        self.assertTrue(np.isnan(es.over_seeds([0.1])[1]))

    def test_cell_parts_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            es.cell_parts([0.0, np.nan], [0, 0])


class TestOutcomeFolder(unittest.TestCase):
    def test_an_outcome_is_saved_in_its_own_folder(self):
        p = SimpleNamespace(
            domain="classic", d_info=SimpleNamespace(name="d"), h_info=SimpleNamespace(name="LR", full_name="LR_x")
        )
        df = Mock()
        outcome = ExperimentOutcome.SUCCESS(
            p, "acc", "m", df=df, t_train=0.0, t_test_ave=0.0, experiment="target_size/u1000/v1/s0"
        )
        log = Mock()
        with patch("uncawe.experiments.execution.local_path", return_value="/x.parquet") as local_path:
            save_experiment_results(log, [[outcome]], experiment="target_size")
        self.assertEqual(local_path.call_args.kwargs["experiment"], "target_size/u1000/v1/s0")
        df.save_result.assert_called_once_with("/x.parquet")
        self.assertIn("(target_size/u1000/v1/s0)", log.info.call_args.args[0])


def _synthetic_results(seed=0, n_bags=6):
    """Stored-like rows of the two methods on one binary pair: full V and two seeds of 1/4, three prevalences."""
    rng = np.random.default_rng(seed)
    rows = []
    for p_index, kind in enumerate(ts.PREVALENCE_KINDS):
        pi = (
            np.array([0.5, 0.5])
            if kind == "uniform"
            else np.array([0.3, 0.7])
            if kind == "skewed"
            else np.array([0.9, 0.1])
        )
        true_cts = []
        for b in range(n_bags):
            recalls = np.clip(np.array([0.8, 0.7]) + rng.normal(0, 0.02, 2), 0, 1)
            true_cts.append(_table(pi, recalls))
        for method in ts.METHODS:
            for level, seeds, v_size in (("full", [0], 400), ("1/4", [0, 1], 100)):
                for v_seed in seeds:
                    offset = rng.normal(0, 0.02 if level == "1/4" else 0.005)
                    for b in range(n_bags):
                        est_pi = np.clip(pi + rng.normal(0, 0.03), 0.01, 0.99)
                        est_pi = np.array([est_pi[0], 1 - est_pi[0]])
                        est = _table(est_pi, np.clip(np.array([0.8, 0.7]) + offset, 0, 1))
                        rows.append(
                            {
                                "method": method,
                                "dataset": "toy",
                                "classifier": "LR",
                                "bag_size": 1000,
                                "v_level": level,
                                "v_seed": v_seed,
                                "prevalence_kind": kind,
                                "n_classes": 2,
                                "v_size": v_size,
                                "bag_index": b,
                                "bag_seed": 1000 * p_index + b,
                                "config_hash": "h",
                                "true_cts": true_cts[b],
                                "estim_cts": est,
                                "estim_accs": float(np.trace(est)),
                            }
                        )
    return pd.DataFrame(rows)


def _table(props, recalls):
    props, recalls = np.asarray(props, dtype=float), np.asarray(recalls, dtype=float)
    ct = np.outer(props * (1 - recalls), np.ones(2))
    np.fill_diagonal(ct, props * recalls)
    return ct


class TestTables(unittest.TestCase):
    def setUp(self):
        self.results = _synthetic_results()
        self.pool = {("toy", "LR"): {"d_pool": np.array([0.8, 0.7])}}
        self.out = tables.compute(self.results, self.pool)

    def test_cells_and_levels(self):
        cells, levels = self.out["cells"], self.out["levels"]
        # 2 methods x 3 prevalences x (1 + 2) V cells
        self.assertEqual(len(cells), 18)
        self.assertTrue((cells["n_bags"] == 6).all())
        self.assertTrue((cells["n_dropped"] == 0).all())
        self.assertEqual(len(levels), 4)
        self.assertLess(self.out["bags"]["identity_gap"].max(), 1e-12)
        for _, group in self.out["bags"].groupby(tables.CELL_KEYS):
            for name in tables.PARTS:
                self.assertAlmostEqual(group[f"{name}_bag"].sum(), 0.0, places=12)
        quarter = levels.loc[levels["v_level"] == "1/4"].iloc[0]
        self.assertEqual(quarter["n_seeds"], 2)
        self.assertTrue(np.isfinite(quarter["e2_seed_sd"]))
        full = levels.loc[levels["v_level"] == "full"].iloc[0]
        self.assertTrue(np.isnan(full["e2_seed_sd"]))
        self.assertAlmostEqual(full["e2_seed_mean"], full["e2_full"])
        self.assertEqual(self.out["n_unshared_bags"], 0)

    def test_fixed_rms_is_the_rms_over_the_cells(self):
        cells, levels = self.out["cells"], self.out["levels"]
        method = ts.METHODS[0]
        sub = cells.loc[(cells["method"] == method) & (cells["v_level"] == "1/4")]
        row = levels.loc[(levels["method"] == method) & (levels["v_level"] == "1/4")].iloc[0]
        self.assertAlmostEqual(row["e2_fixed_rms"], es.rms(sub["e2_fixed"]))

    def test_slopes(self):
        slopes = self.out["slopes"]
        self.assertEqual(len(slopes), 2)
        levels = self.out["levels"]
        row = levels.loc[levels["method"] == ts.METHODS[0]].set_index("v_level")
        expected = (np.log(row.loc["1/4", "e2_fixed_rms"]) - np.log(row.loc["full", "e2_fixed_rms"])) / (
            np.log(100) - np.log(400)
        )
        self.assertAlmostEqual(slopes.loc[slopes["method"] == ts.METHODS[0], "e2_slope"].iloc[0], expected)

    def test_dropped_bags_are_counted(self):
        results = self.results.copy()
        results.at[0, "estim_cts"] = np.full((2, 2), np.nan)
        out = tables.compute(results, self.pool)
        self.assertEqual(int(out["cells"]["n_dropped"].sum()), 1)
        self.assertEqual(int(out["cells"]["n_bags"].sum()), len(results) - 1)

    def test_unshared_bags_are_detected(self):
        results = self.results.copy()
        index = results.index[(results["v_level"] == "1/4") & (results["bag_index"] == 0)][0]
        results.at[index, "true_cts"] = _table([0.5, 0.5], [0.1, 0.1])
        self.assertEqual(tables.compute(results, self.pool)["n_unshared_bags"], 1)

    def test_latex_frame(self):
        frame = tables.levels_frame(self.out["levels"], 1000)
        self.assertEqual(frame.shape, (4, 10))
        latex = frame.style.to_latex()
        self.assertIn("fixed part, RMS", latex)
        with self.assertRaises(RuntimeError):
            tables.levels_frame(self.out["levels"], 100)

    def test_plot(self):
        fig = plots.plot_v_size(self.out["cells"], classifier="LR", size=1000)
        self.assertEqual(len(fig.axes), len(tables.PARTS) * len(ts.METHODS))
        plt.close(fig)
        points = plots.seed_points(self.out["cells"], "LR", 1000)
        # per method: one point at the full V and two seeds at 1/4
        self.assertEqual(len(points), 2 * 3)


def _size_results(seed=0, sizes=(100, 1000, 10000), n_bags=8):
    """Stored-like rows of the two methods on one binary pair, full V only, at several bag sizes: the realised recalls
    and the estimated prevalence fluctuate by $1/\\sqrt{|U|}$, the estimated recalls carry an offset shared by the
    bags."""
    rng = np.random.default_rng(seed)
    pis = {"uniform": np.array([0.5, 0.5]), "skewed": np.array([0.3, 0.7]), "vertex": np.array([0.9, 0.1])}
    rows = []
    for size in sizes:
        noise = 1 / np.sqrt(size)
        for p_index, kind in enumerate(ts.PREVALENCE_KINDS):
            pi = pis[kind]
            true_cts = [_table(pi, np.clip([0.8, 0.7] + rng.normal(0, noise, 2), 0, 1)) for _ in range(n_bags)]
            for method in ts.METHODS:
                for b in range(n_bags):
                    first = np.clip(pi[0] + rng.normal(0, noise), 0.01, 0.99)
                    est = _table([first, 1 - first], [0.81, 0.72])
                    rows.append(
                        {
                            "method": method,
                            "dataset": "toy",
                            "classifier": "LR",
                            "bag_size": size,
                            "v_level": "full",
                            "v_seed": 0,
                            "prevalence_kind": kind,
                            "n_classes": 2,
                            "v_size": 400,
                            "bag_index": b,
                            "bag_seed": 10 * size + 1000 * p_index + b,
                            "config_hash": "h",
                            "true_cts": true_cts[b],
                            "estim_cts": est,
                            "estim_accs": float(np.trace(est)),
                            "t_train": 2.0,
                            "t_test_ave": size / 1000,
                        }
                    )
    return pd.DataFrame(rows)


class TestSizes(unittest.TestCase):
    def setUp(self):
        self.results = _size_results()
        self.pool = {("toy", "LR"): {"d_pool": np.array([0.8, 0.7])}}
        self.out = tables.compute(self.results, self.pool)

    def test_levels_per_size(self):
        levels = self.out["levels"]
        self.assertEqual(len(levels), 3 * len(ts.METHODS))
        self.assertEqual(sorted(set(levels["bag_size"])), [100, 1000, 10000])
        self.assertTrue((levels["n_bags"] == 3 * 8).all())
        # one level per size: no slope against |V|
        self.assertTrue(self.out["slopes"].empty)
        self.assertEqual(self.out["n_unshared_bags"], 0)

    def test_size_slopes(self):
        slopes = self.out["size_slopes"]
        self.assertEqual(len(slopes), len(ts.METHODS))
        self.assertTrue((slopes["n_sizes"] == 3).all())
        levels = self.out["levels"]
        rows = levels.loc[levels["method"] == ts.METHODS[0]].sort_values("bag_size")
        row = slopes.loc[slopes["method"] == ts.METHODS[0]].iloc[0]
        for name in tables.PARTS:
            for kind in ("fixed", "bag"):
                expected = np.polyfit(np.log(rows["bag_size"]), np.log(rows[f"{name}_{kind}_rms"]), 1)[0]
                self.assertAlmostEqual(row[f"{name}_{kind}_slope"], expected)
        # the bag parts of e1 and e3 come from fluctuations of order 1/sqrt(|U|)
        self.assertLess(row["e3_bag_slope"], -0.3)
        self.assertLess(row["e1_bag_slope"], -0.3)

    def test_size_slopes_need_two_sizes(self):
        out = tables.compute(_size_results(sizes=(1000,)), self.pool)
        self.assertTrue(out["size_slopes"].empty)
        with self.assertRaises(RuntimeError):
            tables.sizes_frame(out["levels"])

    def test_sizes_frame(self):
        frame = tables.sizes_frame(self.out["levels"])
        self.assertEqual(frame.shape, (3 * len(ts.METHODS), 7))
        self.assertEqual(list(frame.index.get_level_values("$|U|$")[:3]), ["100", "1000", "10000"])
        self.assertIn("bag part, RMS", frame.style.to_latex())

    def test_cost_and_diagnostics_by_size(self):
        cost = self.out["cost"]
        self.assertEqual(len(cost), 3 * len(ts.METHODS))
        self.assertTrue((cost["tasks"] == 1).all())
        np.testing.assert_allclose(cost["t_test_ave_mean"], cost["bag_size"] / 1000)
        diagnostics = self.out["diagnostics"]
        self.assertEqual(sorted(set(diagnostics["bag_size"])), [100, 1000, 10000])
        self.assertTrue((diagnostics["bags"] == 3 * 8).all())

    def test_plot(self):
        fig = plots.plot_u_size(self.out["levels"], classifier="LR")
        self.assertEqual(len(fig.axes), len(tables.PARTS) * len(ts.METHODS))
        # per panel and dataset: the fixed and the bag line
        self.assertEqual(len(fig.axes[0].get_lines()), 2)
        plt.close(fig)
        with self.assertRaises(RuntimeError):
            plots.plot_u_size(self.out["levels"], classifier="RF")

    def test_tables_and_plots_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            pargs = SimpleNamespace(size=1000, table_dir=tmp)
            with patch.object(tables, "compute", return_value=self.out), patch("builtins.print"):
                tables.gen_tables(pargs)
            for name in tables.CSV_NAMES:
                self.assertTrue(os.path.exists(os.path.join(tmp, f"{tables.TABLE_NAME}_{name}.csv")), name)
            self.assertTrue(os.path.exists(os.path.join(tmp, f"{tables.SIZES_STEM}_doc.tex")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "tabular", f"{tables.SIZES_STEM}_long.tex")))
            # the E4.2b table at |U|=1000 is written from the full-V rows too
            self.assertTrue(os.path.exists(os.path.join(tmp, f"{tables._stem(1000)}_doc.tex")))
            args = plots.parse_args(["--figures", "u-size", "v-size", "--no-svg"])
            with (
                patch.object(plots, "compute", return_value=self.out),
                patch.object(plots, "out_dir", return_value=tmp),
                patch("builtins.print"),
            ):
                plots.generate_plots(args)
            self.assertTrue(os.path.exists(os.path.join(tmp, f"{ts.EXPERIMENT}_u-size_LR.png")))
            self.assertTrue(os.path.exists(os.path.join(tmp, f"{ts.EXPERIMENT}_v-size_LR_u1000.png")))


class TestRecallChecks(unittest.TestCase):
    def test_recall_gap(self):
        # V: recalls (0.9, 0.6) on (10, 20) instances; pool: (0.8, 0.7) on (10, 10)
        val_y = _val_labels([10, 20])
        val_pred = val_y.copy()
        val_pred[:1] = 1
        val_pred[10:18] = 0
        pool_y = _val_labels([10, 10])
        pool_pred = pool_y.copy()
        pool_pred[:2] = 1
        pool_pred[10:13] = 0
        prev = np.array([0.25, 0.75])
        out = tables.recall_gap(val_y, val_pred, pool_y, pool_pred, prev, 2)
        self.assertAlmostEqual(out["gap"], 0.25 * 0.1 + 0.75 * (-0.1))
        d_both = np.array([17 / 20, 19 / 30])
        expected_sd = np.sqrt(
            np.sum(prev**2 * d_both * (1 - d_both) * (1 / np.array([10, 20]) + 1 / np.array([10, 10])))
        )
        self.assertAlmostEqual(out["sd_two_sample"], expected_sd)
        self.assertAlmostEqual(out["z"], out["gap"] / expected_sd)

    def test_subsample_sd_matches_the_spread_of_subsamples(self):
        rng = np.random.default_rng(1)
        counts = [80, 160]
        val_y = _val_labels(counts)
        val_pred = np.where(rng.random(val_y.size) < np.where(val_y == 0, 0.7, 0.85), val_y, 1 - val_y)
        prev = np.array([0.6, 0.4])
        self.assertEqual(tables.subsample_sd(val_y, val_pred, prev, 1.0, 2), 0.0)
        draws = []
        for seed in range(4000):
            index = ts.subsample_index(val_y, 0.25, f"toy{seed}", 0)
            draws.append(np.sum(prev * es.pool_recalls(val_y[index], val_pred[index], 2)))
        self.assertAlmostEqual(np.std(draws), tables.subsample_sd(val_y, val_pred, prev, 0.25, 2), delta=0.002)

    def test_drift_and_diagnostics(self):
        results = _synthetic_results()
        out = tables.compute(results, {("toy", "LR"): {"d_pool": np.array([0.8, 0.7])}})
        drift = out["drift"]
        rows = drift.loc[drift["subset"] == "all"]
        self.assertEqual(sorted(rows["v_level"]), ["1/4", "1/4"])
        seeds = out["seeds"]
        sub = seeds.loc[(seeds["v_level"] == "1/4") & (seeds["method"] == ts.METHODS[0])]
        row = rows.loc[rows["method"] == ts.METHODS[0]].iloc[0]
        self.assertAlmostEqual(row["drift_rms"], es.rms(sub["e2_seed_mean"] - sub["e2_full"]))
        self.assertAlmostEqual(row["seed_sd_rms"], es.rms(sub["e2_seed_sd"]))
        # the toy dataset does not reach 1/16
        self.assertTrue(drift.loc[drift["subset"] == "all levels"].empty)
        self.assertTrue(out["recalls"].empty)

        diagnostics = out["diagnostics"]
        self.assertEqual(int(diagnostics["bags"].sum()), len(results))
        self.assertEqual(int(diagnostics[["divergent_bags", "rhat_above", "ess_below", "failed"]].to_numpy().sum()), 0)
        results = results.assign(nuts_divergences=0, nuts_max_rhat=1.0, nuts_min_ess=1000.0)
        results.loc[:2, "nuts_divergences"] = 3
        results.loc[0, "nuts_max_rhat"] = 1.05
        diagnostics = tables.diagnostics_table(results)
        self.assertEqual(int(diagnostics["divergent_bags"].sum()), 3)
        self.assertEqual(int(diagnostics["rhat_above"].sum()), 1)


class TestTasks(unittest.TestCase):
    def test_config_hash_depends_on_r(self):
        self.assertNotEqual(ts.config_hash(30), ts.config_hash(20))
        self.assertEqual(ts.config_hash(30), ts.config_hash(30))

    def test_folders(self):
        self.assertEqual(ts.result_folder(1000, "1/16", 2), os.path.join("target_size", "u1000", "v1_16", "s2"))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            ts.make_method("BayesCAP-real")

    def test_parse_args_grids(self):
        default = ts.parse_args(["--run"])
        self.assertEqual(default.sizes, list(ts.BAG_SIZES))
        self.assertEqual(default.levels, list(ts.LEVELS))
        grid = ts.parse_args(["--run", "--u-grid"])
        self.assertEqual(grid.sizes, [100, 300, 1000, 3000, 10000])
        self.assertEqual(grid.levels, ["full"])
        explicit = ts.parse_args(["--run", "--sizes", "100", "300", "--levels", "full"])
        self.assertEqual((explicit.sizes, explicit.levels), ([100, 300], ["full"]))
        with patch("sys.stderr"):
            for argv in (["--run", "--u-grid", "--sizes", "100"], ["--run", "--u-grid", "--levels", "full"]):
                with self.assertRaises(SystemExit):
                    ts.parse_args(argv)
            with self.assertRaises(SystemExit):
                ts.parse_args(["--run", "--sizes", "0"])

    def test_u_grid_tasks_skip_the_stored_cells(self):
        p = SimpleNamespace(
            d_info=SimpleNamespace(name="letter", n_classes=3), h_info=SimpleNamespace(name="LR", full_name="LR_x")
        )
        with (
            patch.object(ts, "_pretrains", return_value=[("info", p)]),
            patch.object(ts, "_val_counts", return_value=np.array([400, 300, 200])),
            patch.object(ts, "_all_results_exist", side_effect=lambda _, task: task.size == 1000),
        ):
            todo = ts.tasks(sizes=ts.U_SIZES, levels=("full",))
        self.assertEqual(sorted({t.size for t in todo}), [100, 300, 3000, 10000])
        self.assertEqual(len(todo), 4 * len(ts.METHODS))
        self.assertTrue(all(t.level == "full" and t.v_seed == 0 for t in todo))
        self.assertEqual(
            sorted({t.folder for t in todo}),
            sorted(ts.result_folder(size, "full", 0) for size in (100, 300, 3000, 10000)),
        )


if __name__ == "__main__":
    unittest.main()
