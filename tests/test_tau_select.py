"""The study ``tau_select`` (work-plan card WP-23, E5.3): the split of $V$, the calibration bags, the tasks, and the
table and plot modules on synthetic results."""

import os
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quapy as qp
from cap.utils.commons import contingency_table

from uncawe.experiments import tau_select as ts
from uncawe.experiments.methods import TEMPERED_BASES
from uncawe.methods.baycap import BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.plots import tau_select as plots
from uncawe.tables import tau_select as tables
from uncawe.tables import tempering as tt


def _labels(counts):
    rng = np.random.default_rng(1)
    return rng.permutation(np.repeat(np.arange(len(counts)), counts))


# --------------------------------------------------------------------------------------------------- experiment


class SplitTest(unittest.TestCase):
    def test_stratified_disjoint_covering(self):
        counts = [7, 20, 3, 101]
        y = _labels(counts)
        fit, cal = ts.split_index(y, len(counts), "toy")
        self.assertEqual(len(np.intersect1d(fit, cal)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate([fit, cal])), np.arange(y.size))
        np.testing.assert_array_equal(np.bincount(y[cal], minlength=4), np.array(counts) // 2)
        np.testing.assert_array_equal(np.bincount(y[fit], minlength=4), np.array(counts) - np.array(counts) // 2)
        self.assertTrue(np.all(np.diff(fit) > 0) and np.all(np.diff(cal) > 0))

    def test_reproducible_and_seeded_by_the_dataset_only(self):
        y = _labels([30, 40])
        first = ts.split_index(y, 2, "toy")
        np.random.seed(123)  # the global state plays no part
        second = ts.split_index(y, 2, "toy")
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)
        other = ts.split_index(y, 2, "other")
        self.assertFalse(np.array_equal(first[1], other[1]))
        self.assertEqual(ts.split_seed("toy"), ts.split_seed("toy"))
        self.assertNotEqual(ts.split_seed("toy"), ts.split_seed("other"))

    def test_seed_follows_the_experiment_seed(self):
        seed = ts.split_seed("toy")
        old = qp.environ["_R_SEED"]
        try:
            qp.environ["_R_SEED"] = 1
            self.assertNotEqual(ts.split_seed("toy"), seed)
        finally:
            qp.environ["_R_SEED"] = old

    def test_a_class_with_fewer_than_two_instances_raises(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            ts.split_index(_labels([10, 1]), 2, "toy")
        with self.assertRaisesRegex(ValueError, "at least 2"):
            ts.split_index(_labels([10, 10]), 3, "toy")

    def test_recalls(self):
        y = np.array([0, 0, 0, 0, 1, 1, 2])
        pred = np.array([0, 0, 0, 1, 1, 0, 2])
        np.testing.assert_allclose(ts.recalls(y, pred, 4)[:3], [0.75, 0.5, 1.0])
        self.assertTrue(np.isnan(ts.recalls(y, pred, 4)[3]))


class CalibrationBagsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = [40, 90, 15]
        cls.y = _labels(cls.counts)
        rng = np.random.default_rng(2)
        cls.pred = np.where(rng.random(cls.y.size) < 0.7, cls.y, rng.integers(0, 3, cls.y.size))
        cls.fit, cls.cal = ts.split_index(cls.y, 3, "toy")
        cls.bags = ts.calibration_bags("toy", cls.y, cls.pred, cls.cal, 3, n_bags=20)

    def test_only_instances_of_v_cal(self):
        for bag in self.bags:
            self.assertTrue(np.isin(bag.index, self.cal).all())
            self.assertFalse(np.isin(bag.index, self.fit).any())

    def test_size_prevalence_and_realised_table(self):
        self.assertEqual(len(self.bags), 20)
        for b, bag in enumerate(self.bags):
            self.assertEqual(bag.bag_index, b)
            self.assertEqual(bag.index.size, ts.SAMPLE_SIZE)
            self.assertAlmostEqual(bag.requested_prev.sum(), 1.0)
            np.testing.assert_array_equal(
                np.bincount(self.y[bag.index], minlength=3), ts.class_counts(bag.requested_prev, ts.SAMPLE_SIZE)
            )
            np.testing.assert_array_equal(bag.true_ct, contingency_table(self.y[bag.index], self.pred[bag.index], 3))
        # with replacement: V_cal has 72 instances, a bag 1000
        self.assertLess(np.unique(self.bags[0].index).size, ts.SAMPLE_SIZE)

    def test_reproducible_and_independent_of_the_classifier(self):
        other_pred = np.zeros_like(self.pred)
        again = ts.calibration_bags("toy", self.y, other_pred, self.cal, 3, n_bags=20)
        for a, b in zip(self.bags, again):
            np.testing.assert_array_equal(a.index, b.index)
            self.assertEqual(a.seed, b.seed)
        self.assertNotEqual(self.bags[0].seed, self.bags[1].seed)
        self.assertFalse(np.allclose(self.bags[0].requested_prev, self.bags[1].requested_prev))
        self.assertEqual(self.bags[3].seed, ts.calibration_bag_seed("toy", 3))
        self.assertNotEqual(ts.calibration_bag_seed("toy", 3), ts.calibration_bag_seed("other", 3))

    def test_prevalences_cover_the_simplex(self):
        bags = ts.calibration_bags("toy", self.y, self.pred, self.cal, 3, n_bags=400)
        prevs = np.array([bag.requested_prev for bag in bags])
        # Dir(1) on K=3: each component has mean 1/3 and variance (K-1)/(K^2 (K+1)) = 1/18
        np.testing.assert_allclose(prevs.mean(axis=0), 1 / 3, atol=0.04)
        np.testing.assert_allclose(prevs.var(axis=0), 1 / 18, atol=0.015)


class MethodsAndTasksTest(unittest.TestCase):
    def test_method_names(self):
        self.assertEqual(ts.TAUS, (0.5, 1.0, 2.0, 4.0, 8.0))
        self.assertEqual(ts.method_name("BayesCAP-Hf-real", 1.0), "BayesCAP-Hf-real")
        self.assertEqual(ts.method_name("BayCAP-KDEy-mlp-real", 8.0), "BayCAP-KDEy-mlp-real-t8")
        for base in TEMPERED_BASES:
            for tau in ts.TAUS:
                self.assertEqual(ts.method_name(base, tau), tt.method_name(base, tau))
        with self.assertRaises(ValueError):
            ts.method_name("BayesCAP-Hf-real", 3.0)
        with self.assertRaises(ValueError):
            ts.method_name("BayesCAP-real", 1.0)

    def test_methods_are_those_of_main_at_tau(self):
        hf = ts.make_method("BayesCAP-Hf-real", 2.0)
        self.assertIsInstance(hf, BayCAP)
        self.assertEqual((hf.temperature, hf.c_prior, hf.estimand), (2.0, "hierarchical-f", "realised"))
        kdey = ts.make_method("BayCAP-KDEy-mlp-real", 0.5)
        self.assertIsInstance(kdey, BayCAPKDEy)
        self.assertEqual((kdey.temperature, kdey.input_classifier, kdey.estimand), (0.5, "mlp", "realised"))

    def test_config_hash(self):
        self.assertEqual(ts.config_hash(50), ts.config_hash(50))
        self.assertNotEqual(ts.config_hash(50), ts.config_hash(10))

    def test_parse_args(self):
        args = ts.parse_args([])
        self.assertIsNone(args.datasets)
        self.assertEqual(args.taus, list(ts.TAUS))
        self.assertEqual(args.n_bags, 50)
        self.assertEqual(ts.parse_args(["--pilot"]).datasets, list(ts.PILOT_DATASETS))
        self.assertEqual(ts.parse_args(["--taus", "2", "0.5"]).taus, [2.0, 0.5])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            ts.parse_args(["--pilot", "--datasets", "letter"])
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            ts.parse_args(["--taus", "3"])

    def test_tasks(self):
        p = SimpleNamespace(
            d_info=SimpleNamespace(name="toy", n_classes=3), h_info=SimpleNamespace(name="LR", full_name="LR_x")
        )
        with (
            patch.object(ts, "_pretrains", return_value=[("toy_LR", p)]),
            patch.object(ts, "_all_results_exist", side_effect=lambda _, name: name == "BayesCAP-Hf-real-t2"),
        ):
            todo = ts.tasks()
            self.assertEqual(len(todo), len(TEMPERED_BASES) * len(ts.TAUS) - 1)
            self.assertNotIn("BayesCAP-Hf-real-t2", [task.method_name for task in todo])
            self.assertEqual(todo[0].base, "BayesCAP-Hf-real")
            self.assertTrue(all(task.n_bags == ts.N_BAGS for task in todo))
            self.assertEqual(len(ts.tasks(methods=["BayCAP-KDEy-mlp-real"], taus=[1.0])), 1)
            with self.assertRaises(ValueError):
                ts.tasks(taus=[3.0])


# --------------------------------------------------------------------------------------------------- tables

PAIRS = [("wine", "LR", 2), ("wine", "RF", 2), ("spam", "LR", 2), ("spam", "RF", 2), ("letter", "LR", 26)]
N_BAGS = 8
# per dataset: the half-width of the samples at tau=1, the displacement on the calibration bags and on the test bags
SHAPES = {"wine": (0.02, 0.06, 0.05), "spam": (0.03, 0.0, 0.0), "letter": (0.02, 0.03, 0.0)}


def _true_ct(rng, n_classes):
    props = rng.dirichlet(np.full(n_classes, 3.0))
    ct = np.outer(props, np.full(n_classes, 0.2 / (n_classes - 1)))
    np.fill_diagonal(ct, props * 0.8)
    return ct


def _frames(base: str, calibration: bool, seed: int = 0, pairs=PAIRS) -> dict[float, pd.DataFrame]:
    """Synthetic stored rows of ``base`` at each tau, on the calibration bags or on the test bags: samples of width
    ``width * sqrt(tau)`` around the true accuracy plus the displacement of the dataset; the same bags at every tau."""
    rng = np.random.default_rng(seed + (100 if calibration else 0))
    frames = {tau: [] for tau in tt.TAUS}
    for dataset, classifier, k in pairs:
        width, cal_shift, test_shift = SHAPES[dataset]
        shift = cal_shift if calibration else test_shift
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        noise = rng.standard_normal((N_BAGS, 200))
        for tau in tt.TAUS:
            for bag, ct in enumerate(true_cts):
                acc = float(np.trace(ct))
                samples = acc + shift + width * np.sqrt(tau) * noise[bag]
                row = {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": k,
                    "method": tt.method_name(base, tau),
                    "acc_name": "vanilla_accuracy",
                    "true_cts": ct,
                    "estim_cts": ct,
                    "estim_accs": float(np.mean(samples)),
                    "estim_acc_ranges": samples,
                    "acc_err": abs(float(np.mean(samples)) - acc),
                    "nuts_divergences": int(tau == 0.5 and bag == 0),
                    "nuts_max_rhat": 1.0 + 0.01 * tau,
                    "acc_sd": width,
                    "sigma_g": width,
                }
                if calibration:
                    row |= {
                        "v_size": 200,
                        "v_fit_size": 100,
                        "v_cal_size": 100,
                        "v_recalls": np.full(k, 0.8),
                        "v_class_counts": np.full(k, 200 // k),
                    }
                frames[tau].append(row)
    return {tau: pd.DataFrame(rows) for tau, rows in frames.items()}


def _cal_table(seed=0) -> pd.DataFrame:
    parts = [tables.calibration_rows(_frames(base, True, seed), "vanilla_accuracy", base) for base in TEMPERED_BASES]
    return tt.ordered(pd.concat(parts, ignore_index=True))


def _test_table(seed=0, pairs=PAIRS) -> pd.DataFrame:
    rows = []
    for base in TEMPERED_BASES:
        rows.extend(tt.pair_rows(_frames(base, False, seed, pairs), "vanilla_accuracy", base))
    return tt.ordered(pd.DataFrame(rows))


def _computed() -> dict[str, pd.DataFrame]:
    cal, test = _cal_table(), _test_table()
    compared = tables.compare(cal, test)
    return {
        "calibration": cal,
        "pairs": compared["pairs"],
        "datasets": compared["datasets"],
        "tests": compared["tests"],
        "summary": tables.summary_table(compared),
        "crosstab": tables.crosstab(compared),
        "taus": tables.tau_distribution(compared),
        "nuts": tables.nuts_table(cal),
    }


class SigmaTest(unittest.TestCase):
    def test_hand_computed(self):
        # K=2: 2/(K(K+1)) = 1/3; 0.8*0.2/100 + 0.5*0.5/50 = 0.0016 + 0.005 = 0.0066; /3 = 0.0022
        self.assertAlmostEqual(tables.sigma_e2([0.8, 0.5], [100, 50]), np.sqrt(0.0022))
        # K=3: 2/12 = 1/6; 3 * 0.25/10 = 0.075; /6 = 0.0125
        self.assertAlmostEqual(tables.sigma_e2([0.5, 0.5, 0.5], [10, 10, 10]), np.sqrt(0.0125))
        self.assertEqual(tables.sigma_e2([1.0, 0.0], [10, 10]), 0.0)

    def test_invalid(self):
        self.assertTrue(np.isnan(tables.sigma_e2([0.8, 0.5], [100, 0])))
        self.assertTrue(np.isnan(tables.sigma_e2([0.8, np.nan], [100, 10])))
        self.assertTrue(np.isnan(tables.sigma_e2([0.8], [100, 10])))


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = _computed()

    def test_calibration_table(self):
        cal = self.out["calibration"]
        self.assertEqual(len(cal), len(TEMPERED_BASES) * len(PAIRS))
        np.testing.assert_allclose(cal["sigma_e2"].iloc[0], tables.sigma_e2([0.8, 0.8], [100, 100]))
        self.assertTrue((cal["v_fit_size"] == 100).all())

    def test_tau_cal_is_the_best_on_the_calibration_bags(self):
        pairs, cal = self.out["pairs"], self.out["calibration"]
        merged = pairs.merge(cal, on=["acc_name", "method", "dataset", "classifier"], suffixes=("", "_c"))
        for _, row in merged.iterrows():
            means = {tau: row[f"t{tt.tau_label(tau)}_winkler" + ("_c" if tau == 1 else "")] for tau in tt.TAUS}
            self.assertEqual(row["tau_cal"], tt.best_tau(means))
            self.assertEqual(row["calib_winkler_cal"], means[row["tau_cal"]])
        # the displaced calibration bags of wine choose a tau above 1, the centred ones of spam do not
        wine = pairs.loc[pairs["dataset"] == "wine", "tau_cal"]
        self.assertTrue((wine > 1).all())

    def test_tau_cal_does_not_depend_on_the_test_scores(self):
        cal = _cal_table()
        reference = tables.compare(cal, _test_table())["pairs"]
        changed_test = _test_table(seed=5)
        for tau in tt.TAUS:
            changed_test[f"t{tt.tau_label(tau)}_winkler"] = np.random.default_rng(3).random(len(changed_test)) * 50
        changed = tables.compare(cal, changed_test)["pairs"]
        np.testing.assert_array_equal(changed["tau_cal"], reference["tau_cal"])
        self.assertFalse(np.allclose(changed["cal_winkler"], reference["cal_winkler"]))

    def test_test_scores_are_read_at_tau_cal(self):
        pairs = self.out["pairs"]
        test = _test_table().set_index(["acc_name", "method", "dataset", "classifier"])
        for _, row in pairs.iterrows():
            stored = test.loc[(row["acc_name"], row["method"], row["dataset"], row["classifier"])]
            for score in tt.SCORES:
                self.assertEqual(row[f"cal_{score}"], stored[f"t{tt.tau_label(row['tau_cal'])}_{score}"])
                self.assertEqual(row[f"t1_{score}"], stored[f"t1_{score}"])
            self.assertAlmostEqual(row["gain_test"], stored["t1_winkler"] - row["cal_winkler"])
            self.assertAlmostEqual(row["half_amplitude_t1"], stored["t1_amplitude"] / 2)

    def test_lodo_and_oracle_come_from_all_the_test_pairs(self):
        # the test runs have one more pair than the calibration runs: it enters tau_{-d} but not the outputs
        extra = [*PAIRS, ("spam", "kNN", 2)]
        cal, test = _cal_table(), _test_table(pairs=extra)
        compared = tables.compare(cal, test)
        self.assertEqual(len(compared["pairs"]), len(cal))
        lodo = tt.selection(test)["pairs"]
        merged = compared["pairs"].merge(lodo, on=["acc_name", "method", "dataset", "classifier"], suffixes=("", "_t"))
        np.testing.assert_array_equal(merged["tau_lodo"], merged["tau_lodo_t"])
        np.testing.assert_array_equal(merged["tau_oracle"], merged["tau_oracle_t"])

    def test_no_common_pair_raises(self):
        with self.assertRaises(RuntimeError):
            tables.compare(_cal_table(), _test_table(pairs=[("spam", "kNN", 2)]))

    def test_datasets_and_tests(self):
        datasets, tests = self.out["datasets"], self.out["tests"]
        self.assertEqual(len(datasets), len(TEMPERED_BASES) * 3)
        self.assertEqual(len(tests), len(TEMPERED_BASES))
        self.assertTrue((tests["datasets"] == 3).all())
        # three datasets cannot reach p < 0.05 in a one-sided signed-rank test
        self.assertFalse(tests["retained"].any())
        shares = tests[["share_cal_above_oracle", "share_cal_equal_oracle", "share_cal_below_oracle"]].sum(axis=1)
        np.testing.assert_allclose(shares, 1.0)
        wine = datasets.loc[datasets["dataset"] == "wine"]
        pairs = self.out["pairs"]
        for _, row in wine.iterrows():
            sub = pairs.loc[(pairs["method"] == row["method"]) & (pairs["dataset"] == "wine")]
            self.assertAlmostEqual(row["winkler_cal"], sub["cal_winkler"].mean())
            self.assertAlmostEqual(row["winkler_t1"], sub["t1_winkler"].mean())

    def test_spearman(self):
        self.assertAlmostEqual(tables._spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
        self.assertTrue(np.isnan(tables._spearman([1, 1, 1, 1], [1, 2, 3, 4])))
        self.assertTrue(np.isnan(tables._spearman([1, 2], [1, 2])))

    def test_summary_crosstab_taus_nuts(self):
        summary = self.out["summary"]
        allk = summary.loc[summary["K"] == "all"]
        self.assertEqual(allk.loc[allk["method"] == TEMPERED_BASES[0], "column"].tolist(), list(tables.COLUMNS))
        self.assertEqual(allk.loc[allk["reference"], "column"].unique().tolist(), ["oracle"])
        self.assertEqual(set(summary["K"]), {"all", "K=2", "K>=8"})
        labels = [tt.tau_label(tau) for tau in tt.TAUS]
        cross = self.out["crosstab"]
        self.assertEqual(int(cross[labels].to_numpy().sum()), len(self.out["pairs"]))
        taus = self.out["taus"]
        np.testing.assert_array_equal(taus[labels].sum(axis=1), taus["pairs"])
        nuts = self.out["nuts"]
        row = nuts.loc[(nuts["K"] == "all") & (nuts["tau"] == 0.5)].iloc[0]
        self.assertAlmostEqual(row["share_divergent"], 1 / N_BAGS)
        self.assertAlmostEqual(nuts.loc[(nuts["K"] == "all") & (nuts["tau"] == 4.0), "median_max_rhat"].iloc[0], 1.04)

    def test_latex(self):
        tests = self.out["tests"]
        latex = tables._summary_styler(self.out["summary"], tests).to_latex()
        self.assertIn("\\bcap{}-Hf", latex)
        self.assertIn("\\color{gray}", latex)
        self.assertIn("\\tau_{\\mathrm{cal}}", latex)
        latex = tables._crosstab_styler(self.out["crosstab"]).to_latex()
        self.assertIn("oracle-best", latex)
        self.assertNotIn("vanilla_accuracy", tables.caption("vanilla_accuracy", "summary"))

    def test_tables_are_written_and_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables.write_tables(self.out, tmp)
            for name in tables.CSV_NAMES:
                self.assertTrue(os.path.exists(tables.csv_path(tmp, name)), name)
            for name in tables.TABLES:
                stem = tables._stem("vanilla_accuracy", name)
                self.assertTrue(os.path.exists(os.path.join(tmp, f"{stem}_doc.tex")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "tabular", f"{stem}.tex")))
            if shutil.which("pdflatex") is None:
                self.skipTest("pdflatex is not installed")
            pargs = SimpleNamespace(table_dir=tmp, metrics=["vanilla_accuracy"])
            try:
                tables.gen_pdf(pargs)
            except subprocess.CalledProcessError as error:
                self.fail(f"pdflatex failed: {error}")
            for name in tables.TABLES:
                self.assertTrue(os.path.exists(os.path.join(tmp, f"{tables._stem('vanilla_accuracy', name)}_doc.pdf")))

    def test_stored_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            base = TEMPERED_BASES[1]
            for experiment, calibration in ((ts.EXPERIMENT, True), ("main", False)):
                for tau, df in _frames(base, calibration).items():
                    for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                        folder = os.path.join(root, experiment, "classic", "vanilla_accuracy", dataset, classifier)
                        os.makedirs(folder, exist_ok=True)
                        part = part.assign(
                            true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                        )
                        part.to_parquet(os.path.join(folder, f"{tt.method_name(base, tau)}.parquet"))
            out = tables.compute(["vanilla_accuracy"], root=root)
            self.assertEqual(len(out["pairs"]), len(PAIRS))
            self.assertEqual(set(out["pairs"]["method"]), {base})
            np.testing.assert_allclose(out["calibration"]["sigma_e2"].iloc[0], tables.sigma_e2([0.8, 0.8], [100, 100]))
            with self.assertRaises(RuntimeError):
                tables.compute(["macro-F1"], root=root)


class PlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = _computed()

    def test_jitter_keeps_the_points_near_their_tau(self):
        pairs = self.out["pairs"]
        jittered = plots.jittered_tau(pairs)
        ratio = np.log2(jittered / pairs["tau_cal"].to_numpy())
        self.assertLessEqual(np.abs(ratio).max(), plots.JITTER_SPAN / 2 + 1e-12)
        self.assertGreater(np.unique(np.round(ratio, 6)).size, 1)

    def test_figures_render(self):
        for figure in plots.FIGURES:
            with self.subTest(figure=figure):
                fig = plots.PLOTTERS[figure](self.out["pairs"])
                self.assertIsNotNone(fig)
                fig.canvas.draw()  # renders the labels, whose mathtext must parse
                plt.close(fig)
                self.assertIsNone(plots.PLOTTERS[figure](self.out["pairs"], "macro-K"))

    def test_plots_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables.write_tables(self.out, tmp)
            args = plots.parse_args(["--csv", tables.csv_path(tmp), "--no-svg"])
            with patch("builtins.print"):
                plots.generate_plots(args, out_dir=tmp)
            for figure in plots.FIGURES:
                self.assertTrue(os.path.exists(os.path.join(tmp, f"{plots.PLOT_NAME}_{figure}_vanilla_accuracy.png")))


if __name__ == "__main__":
    unittest.main()
