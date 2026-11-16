"""Tempering of BayCAP-Hf and BayCAP-KDEy (work-plan card WP-20, E5.2): the ``temperature`` of ``BayCAP``, the tempered
methods of ``main``, the reproduction check of the untempered runs, and the table and plot modules."""

import io
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import jax
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
from cap.error import vanilla_acc
from numpyro.infer.util import log_density
from quapy.data import LabelledCollection

from uncawe.experiments import tempering_check
from uncawe.experiments.methods import (
    BAYCAP_OPTIONS,
    BAYCAP_VARIANTS,
    BAYES_POOL_METHODS,
    METHOD_CLASS,
    TEMPERATURE_LEVELS,
    TEMPERED_BASES,
    TEMPERED_VARIANTS,
    gen_methods,
)
from uncawe.methods.baycap import C_PRIORS, P_C_COND_Y, BayCAP
from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.plots import tempering as plots
from uncawe.plots.amplitude_vs_coverage import PLOT_METHODS as AVC_METHODS
from uncawe.plots.common import DEFAULT_METHODS
from uncawe.tables import main as main_tables
from uncawe.tables import rank
from uncawe.tables import tempering as tt

NUTS_PARAMS = {"num_warmup": 50, "num_samples": 40}


def _one_hot(labels, n_classes):
    return np.eye(n_classes)[labels]


# --------------------------------------------------------------------------------------------------- BayCAP


class BayCAPTemperatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        n_classes = 3
        val_y = np.repeat(np.arange(n_classes), 30)
        val_yhat = np.where(rng.random(len(val_y)) < 0.8, val_y, rng.integers(0, n_classes, len(val_y)))
        cls.validation = LabelledCollection(
            instances=np.zeros((len(val_y), 1)), labels=val_y, classes=np.arange(n_classes)
        )
        cls.val_posteriors = _one_hot(val_yhat, n_classes)
        cls.test_posteriors = _one_hot(rng.integers(0, n_classes, 50), n_classes)
        cls.X = np.zeros((50, 1))

    def fit(self, method):
        return method.fit(self.validation, self.val_posteriors)

    def test_temperature_one_reproduces_the_untempered_draws(self):
        options = [{"c_prior": c_prior} for c_prior in C_PRIORS]
        options.append({"c_prior": "hierarchical-f", "shares": "non-centred"})
        for option in options:
            for estimand in ("population", "realised"):
                with self.subTest(estimand=estimand, **option):
                    default = self.fit(BayCAP(vanilla_acc, estimand=estimand, random_state=5, **option, **NUTS_PARAMS))
                    tempered = self.fit(
                        BayCAP(vanilla_acc, estimand=estimand, random_state=5, temperature=1, **option, **NUTS_PARAMS)
                    )
                    self.assertEqual(tempered.temperature, 1.0)
                    np.testing.assert_array_equal(
                        tempered.predict_ct_range(self.X, self.test_posteriors),
                        default.predict_ct_range(self.X, self.test_posteriors),
                    )

    def test_rejects_a_temperature_that_is_not_positive(self):
        for temperature in (0, -1.0, float("nan")):
            with (
                self.subTest(temperature=temperature),
                self.assertRaisesRegex(ValueError, "temperature must be positive"),
            ):
                BayCAP(vanilla_acc, temperature=temperature)

    def test_both_likelihood_terms_are_divided_by_tau_and_the_priors_are_not(self):
        """At fixed parameters, the log-density at tau is the log-prior plus the log-likelihood at tau=1 divided by
        tau, for every prior on C."""
        counts = np.array([20, 18, 12])
        for c_prior in C_PRIORS:
            with self.subTest(c_prior=c_prior):
                methods = {
                    tau: self.fit(BayCAP(vanilla_acc, c_prior=c_prior, temperature=tau, **NUTS_PARAMS))
                    for tau in (1.0, 4.0)
                }
                method = methods[1.0]
                kwargs = {
                    "pred_posterior_count": counts,
                    "train_class_cond_count": method.val_ct,
                    "alpha": np.ones(3),
                    "c_alpha": method.c_alpha,
                }
                # one draw of the prior fixes every latent site
                trace = numpyro.handlers.trace(numpyro.handlers.seed(method.model, jax.random.PRNGKey(1))).get_trace(
                    **kwargs
                )
                params = {
                    name: site["value"]
                    for name, site in trace.items()
                    if site["type"] == "sample" and not site["is_observed"]
                }
                observed = {name for name, site in trace.items() if site["type"] == "sample" and site["is_observed"]}
                self.assertEqual(observed, {"F_yc", "N_c"})
                densities = {}
                for tau, m in methods.items():
                    total, sites = log_density(m.model, (), kwargs, params)
                    lik = sum(float(sites[name]["fn"].log_prob(sites[name]["value"]).sum()) for name in observed)
                    densities[tau] = (float(total), lik)
                prior = densities[1.0][0] - densities[1.0][1]
                np.testing.assert_allclose(densities[4.0][0], prior + densities[1.0][1] / 4.0, rtol=1e-5)

    def test_uniform_c_prior_without_bag_matches_the_conjugate_posterior(self):
        """With c_prior="uniform" and an empty bag, column j of C is Dir(1 + f_{.j}/tau) a posteriori, with mean
        (f_{ij}/tau + 1)/(n_j/tau + K)."""
        n_classes, n_per_class, tau = 3, 200, 20.0
        rng = np.random.default_rng(3)
        val_y = np.repeat(np.arange(n_classes), n_per_class)
        val_yhat = np.where(rng.random(len(val_y)) < 0.8, val_y, rng.integers(0, n_classes, len(val_y)))
        validation = LabelledCollection(np.zeros((len(val_y), 1)), val_y, classes=np.arange(n_classes))
        results = {}
        for temperature in (1.0, tau):
            method = BayCAP(vanilla_acc, temperature=temperature, num_warmup=300, num_samples=2000, random_state=2)
            method.fit(validation, _one_hot(val_yhat, n_classes))
            draws = np.asarray(method.sample_posterior(np.zeros(n_classes, dtype=int), np.ones(n_classes))[P_C_COND_Y])
            # cap-ml layout: row y of the draws (and of val_ct) is the column C_{:y} of the paper
            expected = (method.val_ct / temperature + 1) / (method.val_ct.sum(axis=1, keepdims=True) / temperature + 3)
            results[temperature] = (draws.mean(axis=0), expected)
        for temperature, (mean, expected) in results.items():
            with self.subTest(temperature=temperature):
                np.testing.assert_allclose(mean, expected, atol=0.02)
        # the two conjugate means are far apart, so the test tells the two temperatures apart
        self.assertGreater(np.abs(results[1.0][1] - results[tau][1]).max(), 0.08)


# --------------------------------------------------------------------------------------------------- main


class TemperedMethodsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods = dict(gen_methods())

    def test_names_and_levels(self):
        self.assertEqual(TEMPERATURE_LEVELS, (0.5, 2.0, 4.0, 8.0))
        self.assertEqual(TEMPERED_BASES, ("BayesCAP-Hf-real", "BayCAP-KDEy-mlp-real"))
        expected = [f"{base}-t{tau}" for base in TEMPERED_BASES for tau in ("0.5", "2", "4", "8")]
        self.assertEqual(list(TEMPERED_VARIANTS), expected)

    def test_each_is_its_untempered_method_with_the_temperature(self):
        for name, (base, tau) in TEMPERED_VARIANTS.items():
            with self.subTest(method=name):
                method, reference = self.methods[name], self.methods[base]
                self.assertIs(type(method), type(reference))
                self.assertIs(METHOD_CLASS[name], METHOD_CLASS[base])
                self.assertEqual(method.temperature, tau)
                self.assertEqual(reference.temperature, 1.0)
                attributes = ["estimand", "c_prior", "num_samples", "num_warmup", "random_state", "prior"]
                if isinstance(method, BayCAP):
                    attributes.append("shares")
                    self.assertEqual(BAYCAP_VARIANTS[base], (method.estimand, method.c_prior))
                    self.assertNotIn(base, BAYCAP_OPTIONS)
                else:
                    self.assertIsInstance(method, BayCAPKDEy)
                    attributes += ["input_classifier", "kernel", "bandwidth", "shrinkage"]
                for attribute in attributes:
                    self.assertEqual(getattr(method, attribute), getattr(reference, attribute), attribute)

    def test_in_the_bayes_pool_and_outside_the_main_artefacts(self):
        for name in TEMPERED_VARIANTS:
            with self.subTest(method=name):
                self.assertIn(name, BAYES_POOL_METHODS)
                self.assertIn(name, main_tables.EXCLUDED_METHODS)
                self.assertIn(name, rank.EXCLUDED_METHODS)
                self.assertNotIn(name, DEFAULT_METHODS)
                self.assertNotIn(name, AVC_METHODS)


# --------------------------------------------------------------------------------------------------- check


def _stored(seed: int, n_bags: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "true_cts": [rng.dirichlet(np.ones(4)) for _ in range(n_bags)],
            "estim_acc_ranges": [rng.random(10) for _ in range(n_bags)],
            "estim_accs": rng.random(n_bags),
            "estim_cts": [rng.dirichlet(np.ones(4)) for _ in range(n_bags)],
        }
    )


class TemperingCheckTest(unittest.TestCase):
    def test_equal_frames_are_reproduced(self):
        row = tempering_check.compare_frames(_stored(0), _stored(0))
        self.assertTrue(row["reproduced"])
        self.assertEqual(row["estim_acc_ranges_n_diff"], 0)

    def test_nan_in_the_same_places_is_equal(self):
        a, b = _stored(0), _stored(0)
        for df in (a, b):
            df.at[1, "estim_acc_ranges"] = np.full(10, np.nan)
        self.assertTrue(tempering_check.compare_frames(a, b)["reproduced"])

    def test_a_different_bag_is_not_reproduced(self):
        a, b = _stored(0), _stored(0)
        b.at[2, "estim_acc_ranges"] = b.at[2, "estim_acc_ranges"] + 1e-9
        row = tempering_check.compare_frames(a, b)
        self.assertFalse(row["reproduced"])
        self.assertEqual(row["estim_acc_ranges_n_diff"], 1)
        self.assertAlmostEqual(row["estim_acc_ranges_max_abs_diff"], 1e-9, delta=1e-12)
        self.assertFalse(tempering_check.compare_frames(a, a.iloc[:3])["reproduced"])

    def test_comparison_walks_the_checked_cells(self):
        with tempfile.TemporaryDirectory() as root:
            cell = os.path.join("classic", "vanilla_accuracy", "wine", "LR")
            for experiment, seed in (("main", 0), (tempering_check.EXPERIMENT, 0)):
                os.makedirs(os.path.join(root, experiment, cell))
                _stored(seed).to_parquet(os.path.join(root, experiment, cell, "BayesCAP-Hf-real.parquet"))
            _stored(1).to_parquet(os.path.join(root, tempering_check.EXPERIMENT, cell, "BayCAP-KDEy-mlp-real.parquet"))
            table = tempering_check.comparison(root).set_index("method")
            self.assertTrue(table.loc["BayesCAP-Hf-real", "reproduced"])
            self.assertFalse(table.loc["BayCAP-KDEy-mlp-real", "stored_exists"])
            self.assertFalse(table.loc["BayCAP-KDEy-mlp-real", "reproduced"])


# --------------------------------------------------------------------------------------------------- tables

N_BAGS = 10
# (dataset, classifier, K); two classifiers per dataset, so that a dataset holds several pairs
PAIRS = [(d, c, k) for d, k in (("wine", 3), ("spam", 2), ("letter", 26), ("dry", 7)) for c in ("LR", "kNN")]
# the half-width of the samples at tau=1 and the displacement of the point estimate, per dataset: a displaced interval
# gains from tempering up to a point, a centred one does not
SHAPES = {"wine": (0.02, 0.05), "spam": (0.03, 0.0), "letter": (0.02, 0.04), "dry": (0.03, 0.0)}


def _true_ct(rng, n_classes):
    """A realised table (cap-ml layout) with recalls about 0.8, summing to 1."""
    props = rng.dirichlet(np.full(n_classes, 3.0))
    ct = np.outer(props, np.full(n_classes, 0.2 / (n_classes - 1)))
    np.fill_diagonal(ct, props * 0.8)
    return ct


def _frames(base: str, seed: int = 0) -> dict[float, pd.DataFrame]:
    """Synthetic stored rows of ``base`` at each tau: samples of width ``width * sqrt(tau)`` around the true accuracy
    plus the displacement of the dataset; the same bags at every tau."""
    rng = np.random.default_rng(seed)
    frames = {tau: [] for tau in tt.TAUS}
    for dataset, classifier, k in PAIRS:
        width, shift = SHAPES[dataset]
        true_cts = [_true_ct(rng, k) for _ in range(N_BAGS)]
        noise = rng.standard_normal((N_BAGS, 200))
        for tau in tt.TAUS:
            for bag, ct in enumerate(true_cts):
                acc = float(np.trace(ct))
                samples = acc + shift + width * np.sqrt(tau) * noise[bag]
                frames[tau].append(
                    {
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
                )
    return {tau: pd.DataFrame(rows) for tau, rows in frames.items()}


def _pair_table() -> pd.DataFrame:
    rows = []
    for base in TEMPERED_BASES:
        rows.extend(tt.pair_rows(_frames(base), "vanilla_accuracy", base))
    return tt.ordered(pd.DataFrame(rows))


class TablesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = _pair_table()
        cls.selected = tt.selection(cls.table)
        cls.summary = tt.summary_table(cls.table, cls.selected)

    def test_method_names(self):
        self.assertEqual(tt.method_name("BayesCAP-Hf-real", 1.0), "BayesCAP-Hf-real")
        self.assertEqual(tt.method_name("BayCAP-KDEy-mlp-real", 0.5), "BayCAP-KDEy-mlp-real-t0.5")
        self.assertEqual(tt.TAUS, (0.5, 1.0, 2.0, 4.0, 8.0))
        with self.assertRaises(ValueError):
            tt.method_name("BayesCAP-Hf-real", 3.0)

    def test_pair_table(self):
        self.assertEqual(len(self.table), len(TEMPERED_BASES) * len(PAIRS))
        np.testing.assert_allclose(self.table["t0.5_share_divergent"], 1 / N_BAGS)
        np.testing.assert_allclose(self.table["t1_share_divergent"], 0.0)
        np.testing.assert_allclose(self.table["t0.5_share_divergent_draws"], 1 / N_BAGS / tt.NUTS_DRAWS)
        np.testing.assert_allclose(self.table["t1_share_divergent_draws"], 0.0)
        np.testing.assert_allclose(self.table["t4_max_rhat"], 1.04)
        # the samples widen by sqrt(tau)
        for tau in tt.TAUS:
            np.testing.assert_allclose(self.table[f"t{tt.tau_label(tau)}_widening"], np.sqrt(tau), rtol=1e-6)
        # with r = 1 the prediction is sqrt((tau + 1)/2), for BayCAP-KDEy only
        kdey = self.table.loc[self.table["method"] == tt.WIDENING_METHOD]
        hf = self.table.loc[self.table["method"] != tt.WIDENING_METHOD]
        np.testing.assert_allclose(kdey["t4_widening_predicted"], np.sqrt(5 / 2))
        self.assertTrue(hf["t4_widening_predicted"].isna().all())
        self.assertEqual(self.table.iloc[: len(PAIRS)]["n_classes"].tolist(), [2, 2, 3, 3, 7, 7, 26, 26])

    def test_predicted_widening(self):
        np.testing.assert_allclose(tt.predicted_widening(4.0, np.array([1.0, 0.0]), np.array([0.0, 1.0])), [2.0, 1.0])
        self.assertTrue(np.isnan(tt.predicted_widening(2.0, np.zeros(1), np.zeros(1))[0]))

    def test_different_bags_raise(self):
        frames = _frames("BayesCAP-Hf-real")
        frames[2.0] = _frames("BayesCAP-Hf-real", seed=1)[2.0]
        with self.assertRaises(ValueError):
            tt.pair_rows(frames, "vanilla_accuracy", "BayesCAP-Hf-real")

    def test_best_tau_and_its_ties(self):
        self.assertEqual(tt.best_tau({0.5: 3.0, 1.0: 2.0, 2.0: 1.0}), 2.0)
        self.assertEqual(tt.best_tau({0.5: 1.0, 1.0: 1.0, 2.0: 1.0}), 1.0)
        self.assertEqual(tt.best_tau({0.5: 1.0, 1.0: 2.0, 2.0: 1.0}), 0.5)
        self.assertEqual(tt.best_tau({1.0: np.nan, 2.0: 4.0}), 2.0)
        self.assertTrue(np.isnan(tt.best_tau({1.0: np.nan})))

    def test_lodo_tau_never_depends_on_the_held_out_dataset(self):
        part = self.table.loc[self.table["method"] == TEMPERED_BASES[0]]
        reference = tt.lodo_taus(part)
        for dataset in part["dataset"].unique():
            with self.subTest(dataset=dataset):
                changed = part.copy()
                rows = changed["dataset"] == dataset
                rng = np.random.default_rng(7)
                for tau in tt.TAUS:
                    column = f"t{tt.tau_label(tau)}_winkler"
                    changed.loc[rows, column] = rng.random(int(rows.sum())) * 100
                self.assertEqual(tt.lodo_taus(changed)[dataset], reference[dataset])

    def test_lodo_tau_is_the_best_on_the_other_datasets(self):
        part = self.table.loc[self.table["method"] == TEMPERED_BASES[0]]
        for dataset, tau in tt.lodo_taus(part).items():
            others = part.loc[part["dataset"] != dataset]
            means = {t: others[f"t{tt.tau_label(t)}_winkler"].mean() for t in tt.TAUS}
            self.assertEqual(tau, min(means, key=means.get))

    def test_selection_and_decision(self):
        tests = self.selected["tests"]
        self.assertEqual(len(tests), len(TEMPERED_BASES))
        self.assertTrue((tests["datasets"] == 4).all())
        self.assertTrue(tests["decision_metric"].all())
        # four datasets cannot reach p < 0.05 in a one-sided signed-rank test (smallest p is 1/16)
        self.assertFalse(tests["retained"].any())
        datasets = self.selected["datasets"]
        self.assertEqual(len(datasets), len(TEMPERED_BASES) * 4)
        pairs = self.selected["pairs"]
        self.assertEqual(len(pairs), len(self.table))
        # the scores at the oracle tau are never worse than those at any fixed tau
        merged = pairs.merge(self.table, on=["acc_name", "method", "dataset", "classifier"])
        for tau in tt.TAUS:
            self.assertTrue((merged["oracle_winkler"] <= merged[f"t{tt.tau_label(tau)}_winkler"] + 1e-12).all())

    def test_one_sided_wilcoxon(self):
        self.assertTrue(np.isnan(tt.one_sided_wilcoxon(np.ones(5), np.ones(5))))
        self.assertLess(tt.one_sided_wilcoxon(np.arange(10.0), np.arange(10.0) + 1), 0.01)
        self.assertGreater(tt.one_sided_wilcoxon(np.arange(10.0) + 1, np.arange(10.0)), 0.5)

    def test_summary(self):
        allk = self.summary.loc[self.summary["K"] == "all"]
        columns = [tt.tau_label(tau) for tau in tt.TAUS] + ["lodo", "global", "oracle"]
        self.assertEqual(allk.loc[allk["method"] == TEMPERED_BASES[0], "column"].tolist(), columns)
        self.assertEqual(allk.loc[allk["reference"], "column"].unique().tolist(), ["global", "oracle"])
        self.assertTrue((allk["pairs"] == len(PAIRS)).all())
        self.assertEqual(set(self.summary["K"]), {"all", "K=2", "3<=K<=7", "K>=8"})
        fixed = allk.loc[allk["column"] == "1"]
        np.testing.assert_allclose(fixed["median_max_rhat"], 1.01)
        self.assertTrue((allk.loc[allk["column"] == "0.5", "pairs_divergent_draws"] == 0).all())
        oracle = tt.oracle_distribution(self.selected)
        self.assertTrue((oracle[[tt.tau_label(tau) for tau in tt.TAUS]].sum(axis=1) == len(PAIRS)).all())

    def test_stored_round_trip(self):
        with tempfile.TemporaryDirectory() as root:
            base = TEMPERED_BASES[1]
            for tau, df in _frames(base).items():
                for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                    folder = os.path.join(root, "main", "classic", "vanilla_accuracy", dataset, classifier)
                    os.makedirs(folder, exist_ok=True)
                    part = part.assign(
                        true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                    )
                    part.to_parquet(os.path.join(folder, f"{tt.method_name(base, tau)}.parquet"))
            table = tt.pair_table(["vanilla_accuracy"], root=root)
            self.assertEqual(len(table), len(PAIRS))
            self.assertEqual(set(table["method"]), {base})
            with self.assertRaises(RuntimeError):
                tt.pair_table(["macro-F1"], root=root)

    def test_latex(self):
        part = self.summary.loc[self.summary["acc_name"] == "vanilla_accuracy"]
        tests = self.selected["tests"]
        latex = tt._summary_styler(part, tests).to_latex()
        self.assertIn("\\bcap{}-Hf", latex)
        self.assertIn("\\color{gray}", latex)
        self.assertIn("\\bfseries", latex)
        latex = tt._datasets_styler(self.selected["datasets"]).to_latex()
        self.assertIn("\\texttt{wine}", latex)
        datasets = self.selected["datasets"].replace({"dataset": {"wine": "poker_hand"}})
        self.assertIn("\\texttt{poker\\_hand}", tt._datasets_styler(datasets).to_latex())
        self.assertNotIn("vanilla_accuracy", tt.caption("vanilla_accuracy", "summary"))
        self.assertIn("xcolor", tt._latex_document("x.tex"))

    def test_pairs_table(self):
        part = self.table.loc[self.table["acc_name"] == "vanilla_accuracy"]
        styler = tt._pairs_styler(part, self.selected["pairs"])
        self.assertEqual(len(styler.data), len(part))
        self.assertEqual(styler.data.shape[1], 1 + 2 * len(tt.TAUS) + 2)
        # the oracle column is the tau of the lowest Winkler score of the row
        winkler = styler.data[tt.SCORE_LABELS["winkler"]].to_numpy(dtype=float)
        oracle = styler.data[("selected $\\tau$", "oracle")].to_numpy()
        best = [tt.tau_label(tt.TAUS[i]) for i in np.argmin(winkler, axis=1)]
        self.assertEqual(list(oracle), best)
        latex = styler.to_latex()
        self.assertIn("\\bfseries", latex)
        self.assertIn("\\color{gray}", latex)
        self.assertIn("\\tau_{-d}", latex)
        self.assertIn("\\cellcolor[HTML]", latex)
        self.assertIn("[table]{xcolor}", tt._latex_document("x.tex"))
        self.assertNotIn("vanilla_accuracy", tt.caption("vanilla_accuracy", "pairs"))

    def test_tables_are_written_and_build(self):
        with tempfile.TemporaryDirectory() as root:
            base = TEMPERED_BASES[0]
            for tau, df in _frames(base).items():
                for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                    folder = os.path.join(root, "main", "classic", "vanilla_accuracy", dataset, classifier)
                    os.makedirs(folder, exist_ok=True)
                    part = part.assign(
                        true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel)
                    )
                    part.to_parquet(os.path.join(folder, f"{tt.method_name(base, tau)}.parquet"))
            out = os.path.join(root, "tables")
            pargs = SimpleNamespace(table_dir=out, metrics=["vanilla_accuracy"])
            with patch.object(tt.env, "root_dir", root), redirect_stdout(io.StringIO()):
                tt.gen_tables(pargs)
            for name in tt.TABLES:
                self.assertTrue(os.path.exists(os.path.join(out, f"{tt._stem('vanilla_accuracy', name)}_doc.tex")))
            if shutil.which("pdflatex") is None:
                self.skipTest("pdflatex is not installed")
            try:
                tt.gen_pdf(pargs)
            except subprocess.CalledProcessError as error:
                self.fail(f"pdflatex failed: {error}")
            for name in tt.TABLES:
                self.assertTrue(os.path.exists(os.path.join(out, f"{tt._stem('vanilla_accuracy', name)}_doc.pdf")))


class PlotsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.table = _pair_table()

    def test_lines(self):
        fig = plots.plot_lines(self.table, "BayesCAP-Hf-real")
        self.assertEqual(len(fig.axes), len(plots.MEASURES) + 1)  # the measures and the colour bar
        fig.canvas.draw()
        plt.close(fig)
        fig = plots.plot_lines(self.table, "BayesCAP-Hf-real", by_k=True)
        self.assertEqual(len(fig.axes), 4 * len(plots.MEASURES) + 1)  # all pairs and three groups of K
        fig.canvas.draw()  # renders the titles, whose mathtext must parse
        plt.close(fig)
        self.assertIsNone(plots.plot_lines(self.table, "BayesCAP-Hf-real", "macro-K"))

    def test_widening(self):
        points = plots.widening_points(self.table)
        self.assertEqual(len(points), len(PAIRS) * len(TEMPERATURE_LEVELS))
        fig = plots.plot_widening(self.table)
        self.assertIsNotNone(fig)
        fig.canvas.draw()
        plt.close(fig)
        self.assertIsNone(plots.plot_widening(self.table.loc[self.table["method"] != tt.WIDENING_METHOD]))


if __name__ == "__main__":
    unittest.main()
