import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cap.models.cont_table import CAPContingencyTable
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS

from uncawe.experiments import eb_binary as experiment
from uncawe.experiments import main
from uncawe.methods.baycap import fit_beta_binomial
from uncawe.plots import eb_binary as eb_binary_plots
from uncawe.pretrain import classic, classic_binary
from uncawe.tables import eb_binary, eb_prior
from uncawe.tables.eb_binary import TABLES, extend, fit_diagnostics, gap_statistic, ordered, pull

MAIN_BINARY = {"german", "mammographic", "semeion", "spambase", "tictactoe"}
# cap-ml layout: true classes on rows
VAL_COUNTS = {
    ("wdbc", "LR"): np.array([[90, 10], [30, 70]]),
    ("wdbc", "kNN"): np.array([[95, 5], [5, 95]]),
    ("german", "LR"): np.array([[180, 20], [40, 60]]),
}


def results(estim_shift: float, pairs=tuple(VAL_COUNTS)) -> pd.DataFrame:
    """Stored results of one method on binary pairs: the mean table moves accuracy by ``estim_shift``."""
    rows = []
    for dataset, classifier in pairs:
        for bag, pos in enumerate((0.3, 0.6)):
            true_ct = np.array([[0.8 * (1 - pos), 0.2 * (1 - pos)], [0.3 * pos, 0.7 * pos]])
            estim_ct = true_ct + estim_shift * np.array([[0.5, -0.5], [-0.5, 0.5]])
            rows.append(
                {
                    "dataset": dataset,
                    "classifier": classifier,
                    "n_classes": 2,
                    "true_cts": true_ct,
                    "estim_cts": estim_ct,
                    "estim_acc_ranges": np.trace(estim_ct) + np.linspace(-0.02, 0.02, 101),
                    "nuts_divergences": bag,
                    "nuts_max_rhat": 1.0,
                    "nuts_min_ess": 400.0,
                }
            )
    return pd.DataFrame(rows)


def method_results(pairs=tuple(VAL_COUNTS)) -> dict[str, pd.DataFrame]:
    shifts = {"BayesCAP": -0.1, "BayesCAP-real": -0.1, "BayesCAP-EB": 0.0, "BayesCAP-EB-real": 0.0}
    return {method: results(shift, pairs) for method, shift in shifts.items()}


def source_tables():
    """The E2.4 per-pair table of the pairs of E2.5 and of main, each extended with the E2.5 columns."""
    out = []
    for source, pairs in (("E2.5", list(VAL_COUNTS)[:2]), ("E2.4", list(VAL_COUNTS)[2:])):
        res = method_results(pairs)
        table = eb_prior.pair_table(res, VAL_COUNTS)
        table = table.assign(dataset=table["dataset"].astype(str), classifier=table["classifier"].astype(str))
        out.append(extend(table, res["BayesCAP"], VAL_COUNTS, source))
    return out


class DatasetsTest(unittest.TestCase):
    def test_datasets_are_the_binary_datasets_of_quapy_outside_main(self):
        self.assertEqual(len(classic_binary.DATASETS), 25)
        self.assertEqual(set(classic_binary.DATASETS) | MAIN_BINARY, set(UCI_BINARY_DATASETS))
        self.assertFalse(set(classic_binary.DATASETS) & MAIN_BINARY)
        self.assertNotEqual(classic_binary.DOMAIN, main.DOMAIN)

    def test_pretraining_stores_under_the_domain(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(40, 2)), np.repeat([0, 1], 20)
        split = LabelledCollection(X, y, classes=[0, 1])
        dumped = []
        with (
            patch.object(classic, "gen_classifiers", return_value=[(Mock(), SimpleNamespace(name="LR"))]),
            patch.object(classic, "PretrainInfo") as info,
            patch.object(classic, "parallel", side_effect=lambda func, args_list, **_: map(func, args_list)),
        ):
            info.return_value.exists = False
            info.return_value.dump.side_effect = lambda posteriors: dumped.append(posteriors)
            classic.pretrain(Mock(), datasets=[("wdbc", "uci_binary", (split, split, split))], domain="classic_binary")
        self.assertTrue(all(call.args[0] == "classic_binary" for call in info.call_args_list))
        self.assertEqual(len(dumped), 1)


class ExperimentTest(unittest.TestCase):
    def _tasks(self, **filters):
        infos = {
            f"{d}_{h}": SimpleNamespace(
                domain="classic_binary", d_info=SimpleNamespace(name=d), h_info=SimpleNamespace(name=h, full_name=h)
            )
            for d in ("wdbc", "sonar", "unknown")
            for h in ("LR", "kNN")
        }
        methods = [(name, name.lower()) for name in experiment.METHODS] + [("DoC", "doc")]
        with (
            patch.object(experiment, "load_info_paths", return_value=list(infos)) as paths,
            patch.object(experiment.PretrainInfo, "load", side_effect=lambda path, fast: infos[path]),
            patch.object(experiment, "all_results_exist", return_value=False) as exists,
            patch.object(experiment, "gen_methods", return_value=methods),
        ):
            tasks = experiment.experiment_args(Mock(), **filters)
        self.assertEqual(paths.call_args.kwargs["domain"], "classic_binary")
        return tasks, exists

    def test_default_runs_the_four_baycap_methods_on_the_e25_pairs(self):
        tasks, exists = self._tasks()
        self.assertEqual(len(tasks), 4 * 4)
        self.assertEqual({task[1] for task in tasks}, set(experiment.METHODS))
        self.assertEqual({task[0].split("_")[0] for task in tasks}, {"wdbc", "sonar"})
        self.assertEqual(exists.call_args.args[5], "eb_binary")

    def test_pilot_filters_pairs_and_methods(self):
        tasks, _ = self._tasks(datasets=["sonar"], classifiers=["LR"], methods=["BayesCAP-EB"])
        self.assertEqual(tasks, [("sonar_LR", "BayesCAP-EB", "bayescap-eb")])

    def test_other_methods_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "DoC"):
            experiment.experiment_args(Mock(), methods=["DoC"])

    def test_tasks_run_in_the_baycap_pool_with_the_e25_results_folder(self):
        with (
            patch.object(experiment, "experiment_args", return_value=["task"]),
            patch.object(experiment, "run_tasks") as run,
        ):
            experiment.experiments(Mock())
        kwargs = run.call_args.kwargs
        self.assertEqual(run.call_args.args[1], ["task"])
        self.assertEqual((kwargs["n_jobs"], kwargs["env"], kwargs["experiment"]), (16, main.BAYES_ENV, "eb_binary"))
        self.assertIs(kwargs["worker"].func, main.exp_protocol)
        self.assertEqual(kwargs["worker"].keywords, {"experiment": "eb_binary"})

    def test_protocol_writes_to_the_results_folder_of_the_experiment(self):
        p = SimpleNamespace(
            domain="classic_binary", d_info=SimpleNamespace(name="wdbc"), h_info=SimpleNamespace(full_name="LR")
        )
        with patch.object(main, "local_path", return_value="x") as path:
            main._result_path(p, "BayesCAP", "vanilla_accuracy", "eb_binary")
        self.assertEqual(path.call_args.kwargs["experiment"], "eb_binary")
        method = Mock(spec=CAPContingencyTable)
        with (
            patch.object(main.PretrainInfo, "load", return_value=p),
            patch.object(main, "_all_method_results_exist", return_value=False) as exists,
            patch.object(main, "_prepare_evaluation", return_value=(1, 2, 3, 4)),
            patch.object(main, "_evaluate_contingency_method", return_value=[]) as evaluate,
        ):
            main.exp_protocol(("path", "BayesCAP", method), experiment="eb_binary")
        self.assertEqual(exists.call_args.args[2], "eb_binary")
        self.assertEqual(evaluate.call_args.args[-1], "eb_binary")


class FitDiagnosticsTest(unittest.TestCase):
    def test_gap_statistic(self):
        r, n = np.array([0.9, 0.7]), np.array([100, 100])
        self.assertAlmostEqual(gap_statistic(r, n), 0.2 / np.sqrt(0.09 / 100 + 0.21 / 100))
        self.assertTrue(np.isnan(gap_statistic(np.array([1.0, 1.0]), n)))
        self.assertEqual(gap_statistic(np.array([1.0, 0.0]), n), np.inf)
        self.assertEqual(gap_statistic(np.array([0.0, 1.0]), n), -np.inf)

    def test_fit_diagnostics(self):
        record = fit_diagnostics(VAL_COUNTS[("wdbc", "LR")])
        self.assertEqual((record["n_val_1"], record["n_val_2"]), (100, 100))
        self.assertAlmostEqual(record["recall_1"], 0.9)
        self.assertAlmostEqual(record["recall_2"], 0.7)
        self.assertAlmostEqual(record["recall_gap"], 0.2)
        with self.assertRaisesRegex(ValueError, "binary"):
            fit_diagnostics(np.eye(3))

    def test_pull_is_the_prior_part_of_empirical_bayes(self):
        counts, props = VAL_COUNTS[("wdbc", "LR")], np.array([0.4, 0.6])
        fit = fit_beta_binomial(np.array([90, 70]), np.array([100, 100]))
        w = 100 / (100 + fit.nu)
        expected = np.sum(props * (1 - w) * (fit.m - np.array([0.9, 0.7])))
        self.assertAlmostEqual(pull(counts, props), expected)
        estimates, _ = eb_prior.recall_estimates(counts)
        self.assertAlmostEqual(pull(counts, props), np.sum(props * (estimates["eb"] - estimates["plain"])))


class TableTest(unittest.TestCase):
    def setUp(self):
        self.table = ordered(pd.concat(source_tables(), ignore_index=True))

    def test_columns_order_and_coverage_change(self):
        self.assertEqual(list(self.table["source"]), ["E2.5", "E2.5", "E2.4"])
        self.assertEqual(list(self.table["dataset"].astype(str)), ["wdbc", "wdbc", "german"])
        for name, (uniform, eb) in eb_prior.PUSHFORWARDS.items():
            np.testing.assert_allclose(
                self.table[f"{name}_coverage_change"], self.table[f"{eb}_coverage"] - self.table[f"{uniform}_coverage"]
            )
        self.assertAlmostEqual(self.table["realised_coverage_change"].iloc[0], 1.0)
        # the pull, averaged over the bags, is the prior part of the empirical-Bayes method
        np.testing.assert_allclose(self.table["pull"], self.table["BayesCAP-EB_prior_part"])

    def test_compute_table_reads_both_sources(self):
        def load(method, experiment, domain):
            pairs = list(VAL_COUNTS)[:2] if experiment == "eb_binary" else list(VAL_COUNTS)[2:]
            return method_results(pairs)[method]

        def pair_table(experiment, domain):
            return eb_prior.pair_table({m: load(m, experiment, domain) for m in eb_prior.METHODS}, VAL_COUNTS)

        with (
            patch.object(eb_binary.eb_prior, "compute_pair_table", side_effect=pair_table) as tables,
            patch.object(eb_binary.eb_prior, "load_results", side_effect=load),
            patch.object(eb_binary, "load_validation_counts", side_effect=lambda pairs, domain: VAL_COUNTS) as counts,
        ):
            table = eb_binary.compute_table()
        self.assertEqual(
            [call.kwargs for call in tables.call_args_list],
            [{"experiment": "eb_binary", "domain": "classic_binary"}, {"experiment": "main", "domain": "classic"}],
        )
        self.assertEqual([call.kwargs["domain"] for call in counts.call_args_list], ["classic_binary", "classic"])
        self.assertEqual(list(table["source"]), ["E2.5", "E2.5", "E2.4"])

    def test_latex_tables(self):
        for name, (styler_fn, caption, label) in TABLES.items():
            with self.subTest(table=name):
                latex = eb_prior._latex_longtable(styler_fn(self.table), caption, label)
                self.assertIn("\\begin{longtable}", latex)
                self.assertIn(eb_binary.E24_MARK, latex)

    def test_figures(self):
        for name, plotter in eb_binary_plots.PLOTTERS.items():
            with self.subTest(figure=name):
                fig = plotter(self.table)
                # drawing parses the mathtext of every label, as savefig does
                fig.canvas.draw()
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
