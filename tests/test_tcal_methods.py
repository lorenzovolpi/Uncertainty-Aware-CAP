"""Tests of the -tcal methods of E5.3 in the main and rank tables (work-plan card WP-33) on synthetic stored runs."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from test_tau_select import N_BAGS, PAIRS, _frames

from uncawe.experiments.methods import TCAL_VARIANTS, TEMPERED_BASES
from uncawe.experiments.tau_select import EXPERIMENT as TAU_SELECT
from uncawe.tables import main as main_tables
from uncawe.tables import rank
from uncawe.tables import tau_select as ts
from uncawe.tables import tcal_methods as tm
from uncawe.tables import tempering as tt
from uncawe.tables.common import METHOD_LATEX_LABELS, TCAL_METHODS

# the only base whose -tcal column is still assembled: the -asm columns of the bases with a stored -tcal method were
# removed on 2026-11-26 (WP-34)
BASE = "BayCAP-KDEy-h-inst"
TCAL = f"{BASE}-tcal"


def _store(root: str, base: str) -> None:
    for experiment, calibration in ((TAU_SELECT, True), ("main", False)):
        for tau, df in _frames(base, calibration).items():
            for (dataset, classifier), part in df.groupby(["dataset", "classifier"]):
                folder = os.path.join(root, experiment, "classic", "vanilla_accuracy", dataset, classifier)
                os.makedirs(folder, exist_ok=True)
                part = part.assign(true_cts=part["true_cts"].map(np.ravel), estim_cts=part["estim_cts"].map(np.ravel))
                part.to_parquet(os.path.join(folder, f"{tt.method_name(base, tau)}.parquet"))


class TcalMethodsTest(unittest.TestCase):
    def test_one_tcal_method_per_base_with_a_label(self):
        stored = set(TCAL_VARIANTS.values())
        self.assertEqual(TCAL_METHODS, {f"{base}-tcal": base for base in TEMPERED_BASES if base not in stored})
        self.assertEqual(TCAL_METHODS, {TCAL: BASE})
        self.assertFalse(any(name.endswith("-asm") for name in METHOD_LATEX_LABELS))
        self.assertFalse(set(TCAL_METHODS) & set(TCAL_VARIANTS))
        for name in TCAL_METHODS:
            self.assertIn(name, METHOD_LATEX_LABELS)

    def test_order_and_validation(self):
        names = main_tables.ordered_method_names()
        self.assertEqual(len(names), len(set(names)))
        for name, base in TCAL_METHODS.items():
            # after the base, and after the stored -tcal method of the base (WP-34)
            offset = 1 + list(TCAL_VARIANTS.values()).count(base)
            self.assertEqual(names.index(name), names.index(base) + offset)
        for name, base in TCAL_VARIANTS.items():
            stored = [n for n, b in TCAL_VARIANTS.items() if b == base]
            self.assertEqual(names.index(name), names.index(base) + 1 + stored.index(name))
        # since 2026-11-26 BayesCAP-Hf-real-tcal is in the main tables, in place of BayesCAP-Hf-real
        shown = main_tables.table_method_names((TCAL, BASE, "BayesCAP-Hf-real"))
        self.assertIn(TCAL, shown)
        self.assertEqual(shown.index("BayesCAP-Hf-real-tcal"), shown.index("BayesCAP-Hf-real") + 1)
        self.assertEqual(shown.index(TCAL), shown.index(BASE) + 1)
        with self.assertRaises(ValueError):
            main_tables._included(SimpleNamespace(include=["BayesCAP-Hf-real-tcal-asm"]))
        self.assertNotIn(TCAL, main_tables.table_method_names())
        self.assertEqual(main_tables._included(SimpleNamespace(include=[TCAL])), (TCAL,))
        self.assertEqual(rank._included(SimpleNamespace(include=[TCAL])), (TCAL,))
        self.assertIn(TCAL, rank._methods((TCAL,)))
        self.assertNotIn(TCAL, rank._methods())
        with self.assertRaises(ValueError):
            main_tables._included(SimpleNamespace(include=["BayCAP-KDEy-mlp-real-tcal"]))

    def test_rows_are_the_test_run_at_the_calibrated_tau(self):
        with tempfile.TemporaryDirectory() as root:
            _store(root, BASE)
            base_dir = os.path.join(root, "main")
            classifiers = sorted({c for _, c, _ in PAIRS})
            taus = tm.tcal_taus("vanilla_accuracy", [TCAL], root)
            rows = tm.tcal_rows(base_dir, "vanilla_accuracy", "classic", classifiers, [TCAL], root)
            # the same tau_cal as the tables of E5.3
            expected = ts.compute(["vanilla_accuracy"], root=root, bases=(BASE,))["pairs"].set_index(tt.KEYS)
            stored = _frames(BASE, False)
        self.assertEqual(len(taus), len(PAIRS))
        self.assertEqual(set(rows["method"]), {TCAL})
        self.assertEqual(len(rows), N_BAGS * len(PAIRS))
        for row in taus.to_dict("records"):
            key = (row["dataset"], row["classifier"])
            self.assertEqual(row["tau_cal"], expected.loc[key, "tau_cal"])
            mine = rows.loc[rows["dataset"].eq(key[0]) & rows["classifier"].eq(key[1])]
            ref = stored[row["tau_cal"]]
            ref = ref.loc[ref["dataset"].eq(key[0]) & ref["classifier"].eq(key[1])]
            np.testing.assert_allclose(np.sort(mine["estim_accs"]), np.sort(ref["estim_accs"]))
        self.assertGreater(len(set(taus["tau_cal"])), 1)

    def test_no_calibration_runs_no_rows(self):
        with tempfile.TemporaryDirectory() as root:
            rows = tm.tcal_rows(os.path.join(root, "main"), "vanilla_accuracy", "classic", ["LR"], [TCAL], root)
        self.assertTrue(rows.empty)


if __name__ == "__main__":
    unittest.main()
