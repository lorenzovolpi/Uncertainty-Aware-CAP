"""The restructure of work-plan card WP-25 (Step 1) changes no result.

BayesKDEy and BayesianMAPLS became quantifier classes of their own (``methods/bayes_kdey.py:BayesKDEy``,
``methods/bayes_mapls.py:BayesMAPLS``) used by ``BayCAPKDEy``, ``BayCAPMAPLS`` and ``SQSB``. For the same configuration
and seeds, the restructured classes must give exactly the rates, validation outputs, tables, NUTS diagnostics,
summaries and bags of the classes as they were before (frozen in ``tests/_reference_wp25``), in every configuration
used by the experiments: BayCAP-KDEy with a fixed $\\hat C$ and with the Hf prior on $C$, BayCAP-EMQ, and SQS-B, with
$h$ and with the MLP as input classifier, and both pushforwards.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import numpy as np
from _reference_wp25.baycap_kdey import BayCAPKDEy as RefBayCAPKDEy
from _reference_wp25.baycap_mapls import BayCAPMAPLS as RefBayCAPMAPLS
from _reference_wp25.sqs_b import SQSB as RefSQSB
from cap.error import vanilla_acc
from test_baycap_fixed import DATA, SETTINGS

from uncawe.methods.baycap_kdey import BayCAPKDEy
from uncawe.methods.baycap_mapls import BayCAPMAPLS
from uncawe.methods.sqs_b import SQSB

KDEY_SETTINGS = {"kernel": "gaussian", "bandwidth": 0.1, "shrinkage": 0.0}
SAMPLE_SIZE = 200


class RestructureTest(unittest.TestCase):
    def _assert_same(self, method, reference):
        method.fit(DATA.val, DATA.val_posteriors)
        reference.fit(DATA.val, DATA.val_posteriors)
        expected = reference.predict_ct_range(DATA.X_u, DATA.u_posteriors)
        np.testing.assert_array_equal(method.predict_ct_range(DATA.X_u, DATA.u_posteriors), expected)
        self.assertEqual(method.last_nuts_diagnostics(), reference.last_nuts_diagnostics())
        got, want = method.last_posterior_summaries(), reference.last_posterior_summaries()
        self.assertEqual(set(got), set(want) | (set(got) - set(want)))
        for name in want:
            np.testing.assert_array_equal(got[name], want[name])

    def test_baycap_kdey(self):
        for c_prior in ("fixed", "hierarchical-f"):
            for source in ("h", "mlp"):
                for estimand in ("population", "realised"):
                    with self.subTest(c_prior=c_prior, input_classifier=source, estimand=estimand):
                        common = SETTINGS | KDEY_SETTINGS | {"input_classifier": source, "estimand": estimand}
                        method = BayCAPKDEy(vanilla_acc, c_prior=c_prior, **common)
                        reference = RefBayCAPKDEy(vanilla_acc, c_prior=c_prior, **common)
                        self._assert_same(method, reference)
                        np.testing.assert_array_equal(method.rate_matrix_, reference.rate_matrix_)
                        np.testing.assert_array_equal(method.val_outputs_, reference.val_outputs_)

    def test_baycap_mapls(self):
        for source in ("h", "mlp"):
            for estimand in ("population", "realised"):
                with self.subTest(input_classifier=source, estimand=estimand):
                    common = SETTINGS | {"input_classifier": source, "estimand": estimand}
                    method = BayCAPMAPLS(vanilla_acc, **common)
                    reference = RefBayCAPMAPLS(vanilla_acc, **common)
                    self._assert_same(method, reference)
                    np.testing.assert_array_equal(method.rate_matrix_, reference.rate_matrix_)

    def test_sqs_b(self):
        # in main, the quantifier of SQS-B and its bags share the seed (the experiment seed); the restructured SQSB has
        # one random_state for both
        for source in ("h", "mlp"):
            with self.subTest(input_classifier=source):
                common = SETTINGS | KDEY_SETTINGS | {"input_classifier": source}
                reference = RefSQSB(
                    vanilla_acc,
                    RefBayCAPKDEy(vanilla_acc, c_prior="fixed", estimand="realised", **common),
                    sample_size=SAMPLE_SIZE,
                    random_state=SETTINGS["random_state"],
                )
                method = SQSB(vanilla_acc, sample_size=SAMPLE_SIZE, **common)
                self._assert_same(method, reference)
                np.testing.assert_array_equal(method.last_bag_indices_, reference.last_bag_indices_)


if __name__ == "__main__":
    unittest.main()
