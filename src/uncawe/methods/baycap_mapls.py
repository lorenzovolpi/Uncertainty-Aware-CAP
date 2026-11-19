r"""BayCAP-EMQ: BayCAP with a fixed rate matrix $\hat C$ and the prevalence posterior of BayesianMAPLS.

The methods ``BayCAP-EMQ-<s>`` and ``BayCAP-EMQ-<s>-real`` of E3.1 of ``docs/ext3_posterior_propagation.md`` (work-plan
card WP-8). Work-plan card WP-13 moved them from ``methods/baycap_fixed.py:BayCAPFixed`` (``quantifier="emq"``) to this
class, with the same draws (``tests/test_baycap_split.py``); work-plan card WP-25 moved the quantifier into
``methods/bayes_mapls.py:BayesMAPLS`` (the NUTS model of BayesianMAPLS, copied from QuaPy), which the class builds from
its own hyperparameters, with the same draws (``tests/test_wp25_restructure.py``).
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal

import numpy as np

from uncawe.methods.baycap_quantifier import QuantifierBayCAP
from uncawe.methods.bayes_mapls import ILR_Z, BayesMAPLS, emq_model, ilr, ilr_inverse


class BayCAPMAPLS(QuantifierBayCAP):
    r"""Push the posterior draws of BayesianMAPLS through the fixed rate matrix of $h$ on $V$ (BayCAP-EMQ, E3.1).

    BayesianMAPLS reads the outputs of the input classifier as calibrated probabilities (:func:`emq_model`, through
    ``quantifier_``, a :class:`uncawe.methods.bayes_mapls.BayesMAPLS`). Its source prevalence $\pi^P$ is that of the
    labels of $V$ (QuaPy's ``exact_train_prev=True``), its chain starts at the EM estimate of QuaPy's ``mapls``
    (``mapls_chain_init=True``), and its effective prior on $\pi$ is logistic-normal even with ``prior="uniform"``
    (§T.4). The input classifier, the estimands, the seeds and the diagnostics are those of
    :class:`uncawe.methods.baycap_quantifier.QuantifierBayCAP`; after each ``predict_ct_range`` call,
    :meth:`last_posterior_summaries` returns ``FIXED_SUMMARIES``. The rate matrix $\hat C$ is fixed (not sampled).

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param estimand: ``"population"`` or ``"realised"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS, of the MLP and of the multinomial draws; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        input_classifier: Literal["h", "mlp"] = "mlp",
        estimand: Literal["population", "realised"] = "population",
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            acc_fn,
            input_classifier=input_classifier,
            estimand=estimand,
            num_warmup=num_warmup,
            num_samples=num_samples,
            temperature=temperature,
            prior=prior,
            random_state=random_state,
        )
        # checks the hyperparameters of the quantifier
        self._make_quantifier()

    def _make_quantifier(self) -> BayesMAPLS:
        return BayesMAPLS(**self._quantifier_settings())

    @property
    def train_prevalence_(self) -> np.ndarray:
        """The source prevalence $\\pi^P$ of the quantifier, that of the labels of $V$."""
        return self.quantifier_.train_prevalence_


__all__ = ["ILR_Z", "BayCAPMAPLS", "emq_model", "ilr", "ilr_inverse"]
