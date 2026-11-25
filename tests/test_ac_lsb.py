"""Tests of E8.3 of docs/ext8_average_confidence.md (card WP-32): AC-LSB and L-CBPE-LSB in main, the per-bag columns
of label-shift binning, the tables (tables/ac_lsb.py) and the plot (plots/ac_lsb.py).

The calibrator (cap-ml's cap.calibration.LSBinning) and the two methods are tested in cap-ml (tests/test_ls_binning.py,
tests/test_ac.py); this module tests their use in uncawe."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from cap.calibration import LasCalPerBag, LSBinning
from cap.models.confidence import AC, CBPE, CentralPoissonInterval, PoissonConfidenceInterval, poisson_binomial_pmf
from quapy.data import LabelledCollection
from scipy.special import softmax

from uncawe.experiments import main
from uncawe.experiments.methods import (
    AC_LSB_METHOD,
    AC_METHOD,
    CONFIDENCE_LEVEL,
    L_CBPE_LSB_METHOD,
    gen_methods,
    get_method_names,
)
from uncawe.experiments.records import EvaluationContext
from uncawe.plots import ac_lsb as plots
from uncawe.plots.amplitude_vs_coverage import PLOT_METHODS
from uncawe.plots.common import METHOD_COLORS
from uncawe.tables import ac_lsb as tables
from uncawe.tables import main as main_tables
from uncawe.tables import rank

LSB_COLUMNS = set(tables.LSB_COLUMNS)


def _posteriors(rng: np.random.Generator, y: np.ndarray, n_classes: int) -> np.ndarray:
    logits = 2.0 * np.eye(n_classes)[y] + rng.normal(scale=1.5, size=(len(y), n_classes))
    return softmax(logits, axis=1)


def _setup(n_classes: int, seed: int = 0, n_val: int = 400, n_bag: int = 200, n_bags: int = 2):
    """A pretrain stub, a dataset with ``n_bags`` bags (first class at prevalence 0.7), the evaluation context, V and
    the posteriors of V (seeded)."""
    rng = np.random.default_rng(seed)
    y_val = rng.integers(0, n_classes, n_val)
    prev = np.full(n_classes, 0.3 / (n_classes - 1))
    prev[0] = 0.7
    ys = [rng.choice(n_classes, size=n_bag, p=prev) for _ in range(n_bags)]
    bag_posteriors = [_posteriors(rng, y, n_classes) for y in ys]
    pretrain = SimpleNamespace(
        d_info=SimpleNamespace(name="dataset", collection="uci", n_classes=n_classes),
        h_info=SimpleNamespace(name="LR", full_name="LR"),
    )
    samples = [SimpleNamespace(X=np.zeros((n_bag, 1))) for _ in ys]
    test_prot = Mock(side_effect=lambda: iter(samples))
    test_prot.total.return_value = n_bags
    dataset = SimpleNamespace(test_prot=test_prot, test_prot_posteriors=bag_posteriors)
    true_cts = []
    for y, post in zip(ys, bag_posteriors):
        ct = np.zeros((n_classes, n_classes))
        np.add.at(ct, (y, post.argmax(axis=1)), 1)
        true_cts.append(ct / n_bag)
    context = EvaluationContext(
        pretrain=pretrain, shifts=[0.1] * n_bags, true_cts=true_cts, train_prev=0.5, val_prev=0.5
    )
    val = LabelledCollection(np.zeros((n_val, 1)), y_val, classes=np.arange(n_classes))
    return pretrain, dataset, context, val, _posteriors(rng, y_val, n_classes)


def _evaluate(method_name, method, n_classes):
    """The keyword arguments of build_result_frame per metric, for one run of the direct-method path of main."""
    pretrain, dataset, context, val, val_post = _setup(n_classes)
    with (
        patch("uncawe.experiments.main._result_path", return_value="result.parquet"),
        patch("uncawe.experiments.main.os.path.exists", return_value=False),
        patch("uncawe.experiments.main.build_result_frame", return_value=Mock()) as build,
    ):
        outcomes = main._evaluate_direct_method(pretrain, method_name, method, dataset, val, val_post, context)
    return {call.kwargs["acc_name"]: call.kwargs for call in build.call_args_list}, outcomes


class FactoryTest(unittest.TestCase):
    def test_entries(self):
        names = get_method_names()
        methods = dict(gen_methods())
        self.assertIn(AC_LSB_METHOD, names)
        self.assertIn(L_CBPE_LSB_METHOD, names)
        ac_lsb, l_cbpe_lsb = methods[AC_LSB_METHOD], methods[L_CBPE_LSB_METHOD]
        self.assertIsInstance(ac_lsb, AC)
        self.assertIsInstance(ac_lsb.calibrator, LSBinning)
        # LSBinning at its defaults: the number of bins selected on the grid
        self.assertIsNone(ac_lsb.calibrator.nbins)
        self.assertEqual(ac_lsb.confidence_level, CONFIDENCE_LEVEL)
        self.assertIsInstance(l_cbpe_lsb, CBPE)
        self.assertEqual(l_cbpe_lsb.calib_method, "lsb")
        # AC keeps LasCal (E8.1)
        self.assertIsInstance(methods[AC_METHOD].calibrator, LasCalPerBag)

    def test_out_of_the_main_tables_ranks_and_plots(self):
        for method in (AC_LSB_METHOD, L_CBPE_LSB_METHOD):
            self.assertNotIn(method, main_tables.MAIN_METHODS)
            self.assertIn(method, main_tables.EXCLUDED_METHODS)
            self.assertIn(method, rank.EXCLUDED_METHODS)
            self.assertNotIn(method, rank._methods())
            self.assertNotIn(method, PLOT_METHODS)
            self.assertIn(method, METHOD_COLORS)

    def test_intervals_read_from_the_pmf(self):
        self.assertIs(main_tables.PMF_INTERVALS[AC_LSB_METHOD], CentralPoissonInterval)
        self.assertIs(main_tables.PMF_INTERVALS[L_CBPE_LSB_METHOD], PoissonConfidenceInterval)


class MainProtocolTest(unittest.TestCase):
    def test_ac_lsb_stores_the_columns_of_label_shift_binning_for_vanilla_accuracy_only(self):
        by_acc, _ = _evaluate(AC_LSB_METHOD, AC(calibrator=LSBinning()), 3)
        columns = by_acc["vanilla_accuracy"]["extra_columns"]
        self.assertEqual(set(columns), LSB_COLUMNS)
        self.assertTrue(all(len(values) == 2 for values in columns.values()))
        # arrays stored as lists, one value per class
        self.assertIsInstance(columns["lsb_prevalence"][0], list)
        self.assertEqual(len(columns["lsb_prevalence"][0]), 3)
        self.assertEqual(len(columns["lsb_nbins_eff"][0]), 3)
        self.assertTrue(np.isfinite(by_acc["vanilla_accuracy"]["estim_accs"]).all())
        for name in ("macro-F1", "macro-K"):
            self.assertIsNone(by_acc[name]["extra_columns"])
            self.assertTrue(np.isnan(by_acc[name]["estim_accs"]).all())

    def test_l_cbpe_lsb_on_binary_data_equals_ac_lsb(self):
        by_acc, _ = _evaluate(L_CBPE_LSB_METHOD, CBPE("vanilla_accuracy", calib_method="lsb"), 2)
        ac_by_acc, _ = _evaluate(AC_LSB_METHOD, AC(calibrator=LSBinning()), 2)
        self.assertEqual(set(by_acc["vanilla_accuracy"]["extra_columns"]), LSB_COLUMNS)
        self.assertEqual(by_acc["vanilla_accuracy"]["extra_columns"], ac_by_acc["vanilla_accuracy"]["extra_columns"])
        np.testing.assert_allclose(
            by_acc["vanilla_accuracy"]["estim_accs"], ac_by_acc["vanilla_accuracy"]["estim_accs"], atol=1e-12
        )

    def test_l_cbpe_lsb_is_nan_on_multiclass_data(self):
        by_acc, _ = _evaluate(L_CBPE_LSB_METHOD, CBPE("vanilla_accuracy", calib_method="lsb"), 3)
        for kwargs in by_acc.values():
            self.assertTrue(np.isnan(kwargs["estim_accs"]).all())
            self.assertIsNone(kwargs["extra_columns"])

    def test_l_cbpe_stores_no_columns(self):
        # as before WP-32, so that the stored L-CBPE results keep their columns
        method = CBPE("vanilla_accuracy", calib_method="lascal")
        with patch("cap.calibration.lascal.lascal_calibrate", side_effect=lambda v, y, p: (p, 1.0, False)):
            by_acc, _ = _evaluate("L-CBPE", method, 2)
        self.assertIsNone(by_acc["vanilla_accuracy"]["extra_columns"])
        self.assertIsNone(main.calibration_columns(method))


def _rows(method: str, dataset: str, n_classes: int, pmfs: list, true_accs: list, lsb: bool) -> pd.DataFrame:
    """Stored rows of one pair: true tables (cap-ml layout, true classes on rows) with the given accuracies, the bag
    prevalence (0.6, 0.4, 0, ...) and the prevalence of V uniform."""
    rows = []
    for index, (pmf, acc) in enumerate(zip(pmfs, true_accs)):
        ct = np.zeros((n_classes, n_classes))
        ct[0, 0], ct[0, 1] = acc - 0.4, 0.6 - (acc - 0.4)
        ct[1, 1] = 0.4
        no_estimate = main_tables._no_estimate(pmf)
        estimate = np.nan if no_estimate else CentralPoissonInterval(pmf).point_estimate
        row = {
            "dataset": dataset,
            "classifier": "LR",
            "method": method,
            "n_classes": n_classes,
            "true_cts": ct,
            "estim_acc_ranges": pmf,
            "estim_accs": estimate,
            "acc_err": abs(estimate - acc) if np.isfinite(estimate) else np.nan,
            "val_prev": 0.5 if n_classes == 2 else [1 / n_classes] * (n_classes - 1),
        }
        if lsb:
            estimate_prev = np.zeros(n_classes)
            estimate_prev[:2] = (0.5, 0.5)
            row |= {
                "lsb_nbins": 20,
                "lsb_nbins_eff": [20] * (n_classes - 1) + [15],
                "lsb_grid_fallback": False,
                "lsb_prevalence": estimate_prev.tolist(),
                "lsb_n_degenerate": index,
                "lsb_argmax_changed": 0.1 * index,
            }
        rows.append(row)
    return pd.DataFrame(rows)


class TablesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.pmfs = [poisson_binomial_pmf(rng.uniform(0.6, 1.0, size=10)) for _ in range(3)]
        self.other = [poisson_binomial_pmf(rng.uniform(0.5, 1.0, size=10)) for _ in range(3)]
        self.true_accs = [0.8, 0.9, 0.5]

        def pair_frames(method, pmfs, lsb):
            binary = _rows(method, "binary", 2, pmfs, self.true_accs, lsb)
            if method in tables.BINARY_ONLY:
                multi = _rows(method, "multi", 3, [np.nan] * 3, self.true_accs, lsb)
            else:
                multi = _rows(method, "multi", 3, pmfs, self.true_accs, lsb)
            return pd.concat([binary, multi], ignore_index=True)

        self.results = {
            AC_METHOD: pair_frames(AC_METHOD, self.other, False),
            AC_LSB_METHOD: pair_frames(AC_LSB_METHOD, self.pmfs, True),
            tables.L_CBPE: pair_frames(tables.L_CBPE, self.other, False),
            L_CBPE_LSB_METHOD: pair_frames(L_CBPE_LSB_METHOD, self.pmfs, True),
        }

    def test_shift_and_prevalences(self):
        df = self.results[AC_LSB_METHOD]
        binary = df[df["dataset"] == "binary"]
        np.testing.assert_allclose(tables.bag_prevalences(binary), [[0.6, 0.4]] * 3)
        np.testing.assert_allclose(tables.bag_shifts(binary), [0.2] * 3)
        multi = df[df["dataset"] == "multi"]
        np.testing.assert_allclose(tables.bag_shifts(multi), [abs(0.6 - 1 / 3) + abs(0.4 - 1 / 3) + 1 / 3] * 3)
        np.testing.assert_allclose(tables.validation_prevalence(0.3, 2), [0.7, 0.3])
        with self.assertRaises(ValueError):
            tables.validation_prevalence([0.3], 3)

    def test_pair_tables(self):
        comparison, differences, diagnostics = tables.pair_tables(self.results)
        self.assertEqual(comparison["dataset"].tolist(), ["binary", "multi"])
        binary, multi = comparison.iloc[0], comparison.iloc[1]
        expected = np.mean([CentralPoissonInterval(p).winkler(a) for p, a in zip(self.pmfs, self.true_accs)])
        self.assertAlmostEqual(binary[f"{AC_LSB_METHOD}_winkler"], expected)
        expected = np.mean([PoissonConfidenceInterval(p).amplitude() for p in self.pmfs])
        self.assertAlmostEqual(binary[f"{L_CBPE_LSB_METHOD}_amplitude"], expected)
        for method in tables.BINARY_ONLY:
            self.assertTrue(np.isnan(multi[f"{method}_ae"]))
        self.assertFalse(np.isnan(multi[f"{AC_LSB_METHOD}_ae"]))

        for score in tables.SCORES:
            for lsb, lascal in tables.PAIRS:
                self.assertAlmostEqual(
                    differences.iloc[0][f"{lsb}_{score}"], binary[f"{lsb}_{score}"] - binary[f"{lascal}_{score}"]
                )
        self.assertTrue(np.isnan(differences.iloc[1][f"{L_CBPE_LSB_METHOD}_ae"]))

        row = diagnostics.iloc[0]
        self.assertEqual((row["nbins"], row["nbins_eff_min"], row["nbins_eff_max"]), (20, 15, 20))
        self.assertEqual(row["grid_fallback"], 0)
        self.assertAlmostEqual(row["mean_n_degenerate"], 1.0)
        self.assertAlmostEqual(row["mean_argmax_changed"], 0.1)
        # |0.5 - 0.6| + |0.5 - 0.4|
        self.assertAlmostEqual(row["mean_q_l1"], 0.2)
        self.assertEqual(row["max_abs_point_diff"], 0.0)
        self.assertTrue(np.isnan(diagnostics.iloc[1]["max_abs_point_diff"]))

        summary = tables.summary_rows(comparison, tables.comparison_columns())
        self.assertEqual(summary["n_pairs"].tolist(), [1, 2])
        self.assertAlmostEqual(summary.iloc[0][f"{L_CBPE_LSB_METHOD}_ae"], binary[f"{L_CBPE_LSB_METHOD}_ae"])
        self.assertTrue(np.isnan(summary.iloc[1][f"{L_CBPE_LSB_METHOD}_ae"]))
        self.assertAlmostEqual(summary.iloc[1][f"{AC_LSB_METHOD}_ae"], comparison[f"{AC_LSB_METHOD}_ae"].mean())

    def test_pairs_without_lascal_results(self):
        # a pilot of AC-LSB on pairs where AC was not stored: the LasCal columns and differences are NaN
        results = dict(self.results)
        results[AC_METHOD] = pd.DataFrame()
        comparison, differences, _ = tables.pair_tables(results)
        self.assertTrue(np.isnan(comparison[f"{AC_METHOD}_ae"]).all())
        self.assertTrue(np.isnan(differences[f"{AC_LSB_METHOD}_ae"]).all())

    def test_unpaired_bags_raise(self):
        self.results[AC_METHOD].loc[0, "true_cts"][1, 1] = 0.1
        with self.assertRaises(ValueError):
            tables.pair_tables(self.results)

    def test_write_tables(self):
        with tempfile.TemporaryDirectory() as out_dir:
            tables.write_tables(self.results, out_dir)
            for name in (
                "ac_lsb.csv",
                "ac_lsb_diff.csv",
                "ac_lsb_diagnostics.csv",
                "ac_lsb_doc.tex",
                "tabular/ac_lsb.tex",
                "tabular/ac_lsb_diff_long.tex",
                "tabular/ac_lsb_diagnostics_long.tex",
            ):
                with open(f"{out_dir}/{name}") as f:
                    self.assertTrue(f.read())
            csv = pd.read_csv(f"{out_dir}/ac_lsb.csv")
            self.assertEqual(csv["dataset"].tolist()[-2:], list(tables.SUMMARIES))

    def test_plot(self):
        data = plots.points(self.results)
        self.assertEqual(set(data["method"]), set(plots.PANEL_METHODS))
        # 3 bags on each of the 2 pairs
        self.assertEqual(len(data), 2 * 6)
        part = data[(data["method"] == AC_LSB_METHOD) & data["binary"]]
        expected = [CentralPoissonInterval(p).point_estimate - a for p, a in zip(self.pmfs, self.true_accs)]
        np.testing.assert_allclose(part["signed_error"], expected)
        np.testing.assert_allclose(part["shift"], [0.2] * 3)
        fig = plots.plot_signed_error_vs_shift(self.results)
        self.assertEqual(len(fig.axes), len(plots.PANEL_METHODS))


if __name__ == "__main__":
    unittest.main()
