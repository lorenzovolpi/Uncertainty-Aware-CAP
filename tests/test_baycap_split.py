"""The split of ``BayCAPFixed`` (work-plan card WP-13) changes no draw: ``BayCAPKDEy`` with ``c_prior="fixed"`` and
``BayCAPMAPLS`` give, for the same seeds, the tables and summaries of the frozen ``BayCAPFixed`` of commit ``55247a1``
(``tests/_reference_baycap_fixed.py``) in the eight configurations of E3.1."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from _reference_baycap_fixed import BayCAPFixed
from cap.error import vanilla_acc
from test_baycap_fixed import DATA, SETTINGS

from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS

KDEY_SETTINGS = {"kernel": "gaussian", "bandwidth": 0.1, "shrinkage": 0.0}


class SplitTest(unittest.TestCase):
    def test_split_classes_reproduce_baycap_fixed(self):
        for quantifier in ("kdey", "emq"):
            for source in ("h", "mlp"):
                for estimand in ("population", "realised"):
                    with self.subTest(quantifier=quantifier, input_classifier=source, estimand=estimand):
                        common = SETTINGS | {"input_classifier": source, "estimand": estimand}
                        reference = BayCAPFixed(vanilla_acc, quantifier=quantifier, **KDEY_SETTINGS, **common)
                        if quantifier == "kdey":
                            method = BayCAPKDEy(vanilla_acc, c_prior="fixed", **KDEY_SETTINGS, **common)
                        else:
                            method = BayCAPMAPLS(vanilla_acc, **common)
                        reference.fit(DATA.val, DATA.val_posteriors)
                        method.fit(DATA.val, DATA.val_posteriors)
                        np.testing.assert_array_equal(method.rate_matrix_, reference.rate_matrix_)
                        np.testing.assert_array_equal(method.val_outputs_, reference.val_outputs_)
                        expected = reference.predict_ct_range(DATA.X_u, DATA.u_posteriors)
                        np.testing.assert_array_equal(method.predict_ct_range(DATA.X_u, DATA.u_posteriors), expected)
                        self.assertEqual(method.last_nuts_diagnostics(), reference.last_nuts_diagnostics())
                        got, want = method.last_posterior_summaries(), reference.last_posterior_summaries()
                        self.assertEqual(set(got), set(want))
                        for name in want:
                            np.testing.assert_array_equal(got[name], want[name])


if __name__ == "__main__":
    unittest.main()
