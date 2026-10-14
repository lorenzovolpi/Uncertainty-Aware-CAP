import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from uncawe.experiments import binform, predist


class BinformProtocolTest(unittest.TestCase):
    def test_prior_transformation_matches_existing_formula(self):
        prior = np.array([0.2, 0.3, 0.5])
        expected = (1 - prior) / 2 * binform.BAYESCAP_PRIOR_C
        np.testing.assert_allclose(binform._transform_prior(prior, binform.BAYESCAP_PRIOR_C), expected)

    def test_method_order_is_stable(self):
        self.assertEqual(
            binform._method_names(),
            ["O-LEAP", "RQBS", "BayesCAP_uniform", "BayesCAP_transformed_val"],
        )


class PredistProtocolTest(unittest.TestCase):
    def test_simplex_points_are_deterministic_and_keep_original_order(self):
        points = predist._simplex_points(np.random.RandomState(0), 4)
        np.testing.assert_allclose(points[0], np.full(3, 1 / 3))
        np.testing.assert_allclose(points[1], [0.99, 0.005, 0.005])
        np.testing.assert_allclose(points[2], [0.005, 0.99, 0.005])
        np.testing.assert_allclose(points[3], [0.005, 0.005, 0.99])

    def test_distance_parsing_preserves_requested_order(self):
        with patch.object(predist.PrediQuant, "DISTANCES", ("l1", "hellinger", "jensen-shannon")):
            self.assertEqual(
                predist._parse_distances(["hellinger,l1", "jensen-shannon"]),
                ["hellinger", "l1", "jensen-shannon"],
            )

    def test_completed_keys_use_distance_and_sample_id(self):
        df = pd.DataFrame({"distance": ["l1", "hellinger"], "sample_id": [0, 2]})
        self.assertEqual(predist._completed_keys(df), {("l1", 0), ("hellinger", 2)})

    def test_interval_metrics_preserve_winkler_penalty(self):
        coverage, amplitude, winkler = predist._interval_metrics(0.1, 0.2, 0.4)
        self.assertEqual(coverage, 0.0)
        self.assertAlmostEqual(amplitude, 0.2)
        self.assertAlmostEqual(winkler, 4.2)


if __name__ == "__main__":
    unittest.main()
