import unittest

from cap.models.confidence import ACC_BayesCAP

from uncawe.experiments.methods import (
    BAYES_WARMUPS,
    BOOTSTRAP_ITERS,
    METHOD_CLASS,
    gen_methods,
    get_method_names,
)
from uncawe.methods.baycap import BayCAP
from uncawe.plots.common import DEFAULT_METHODS
from uncawe.tables import main as main_tables
from uncawe.tables import rank


class BayesCAPRealFactoryTest(unittest.TestCase):
    def setUp(self):
        self.methods = dict(gen_methods())

    def test_factory_builds_realised_baycap_with_main_settings(self):
        self.assertIs(METHOD_CLASS["BayesCAP-real"], BayCAP)
        self.assertIn("BayesCAP-real", get_method_names())
        real, population = self.methods["BayesCAP-real"], self.methods["BayesCAP"]
        self.assertIsInstance(real, BayCAP)
        self.assertIsInstance(population, ACC_BayesCAP)
        self.assertEqual(real.estimand, "realised")
        self.assertEqual((real.num_samples, real.num_warmup), (BOOTSTRAP_ITERS, BAYES_WARMUPS))
        self.assertEqual((real.num_samples, real.num_warmup), (population.num_samples, population.num_warmup))
        # ACC_BayesCAP stores its seed as ``randm_state``; equal seeds give equal NUTS draws
        self.assertEqual(real.random_state, population.randm_state)
        self.assertEqual(real.prior, population.prior)

    def test_factory_names_match_method_class(self):
        self.assertEqual(list(self.methods), get_method_names())

    def test_main_tables_ranks_and_plots_exclude_it(self):
        self.assertNotIn("BayesCAP-real", main_tables.table_method_names())
        self.assertIn("BayesCAP", main_tables.table_method_names())
        self.assertNotIn("BayesCAP-real", rank._methods())
        self.assertNotIn("L-CBPE", rank._methods())
        self.assertNotIn("BayesCAP-real", DEFAULT_METHODS)


if __name__ == "__main__":
    unittest.main()
