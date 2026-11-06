r"""BayCAP-EMQ: BayCAP with a fixed rate matrix $\hat C$ and the prevalence posterior of BayesianMAPLS.

The methods ``BayCAP-EMQ-<s>`` and ``BayCAP-EMQ-<s>-real`` of E3.1 of ``docs/ext3_posterior_propagation.md`` (work-plan
card WP-8). Work-plan card WP-13 moved them from ``methods/baycap_fixed.py:BayCAPFixed`` (``quantifier="emq"``) to this
class, with the same draws (``tests/test_baycap_split.py``).

The NUTS model of BayesianMAPLS and its preparation are copied from QuaPy's ``quapy.method._bayesian``
(``BayesianMAPLS``, commit ``31f36f6`` of branch ``devel``): QuaPy builds the class only when ``pystan`` is installed,
and ``pystan`` is not in the environment of the project (decision of the user, 2026-11-03). The copy keeps QuaPy's
sites, prior, likelihood, initialisation and seed, so where ``pystan`` is installed it gives the draws of QuaPy's
``sample_from_posterior`` (``tests/test_baycap_fixed.py``). The EM start is QuaPy's ``mapls``, which does not need
``pystan``.
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
import quapy.functional as F
from jax.scipy.special import logsumexp
from quapy.method._bayesian import _resolve_dirichlet_prior, mapls

from uncawe.methods.baycap_quantifier import QuantifierBayCAP

# sampled site of the model (QuaPy's name): the ILR coordinates of pi
ILR_Z = "z"


def ilr_basis(n_classes: int) -> np.ndarray:
    """The orthonormal Helmert basis of QuaPy's ILR transformation (``quapy.functional.ILRtransformation.get_V``)."""
    return F.ILRtransformation().get_V(n_classes)


def ilr(prevalence) -> jnp.ndarray:
    """ILR coordinates of a prevalence vector, as QuaPy's ``_JaxILRTransformation.__call__`` (smoothed by $10^{-12}$)."""
    smoothed = jnp.asarray(qp.error.smooth(np.asarray(prevalence), F.CompositionalTransformation.EPSILON))
    return jnp.log(smoothed) @ jnp.asarray(ilr_basis(smoothed.shape[-1])).T


def ilr_inverse(z) -> jnp.ndarray:
    """Prevalence vectors of ILR coordinates, as QuaPy's ``_JaxILRTransformation.inverse``."""
    z = jnp.asarray(z)
    p = jnp.exp(z @ jnp.asarray(ilr_basis(z.shape[-1] + 1)))
    return p / jnp.sum(p, axis=-1, keepdims=True)


def emq_model(test_posteriors, alpha, train_prev, temperature: float = 1.0) -> None:
    r"""The NUTS model of BayesianMAPLS (QuaPy's ``BayesianMAPLS._numpyro_model``).

    The ILR coordinates $z$ of $\pi$ are $\mathcal N(\mathbf 0,I)$, the density $\operatorname{Dir}(\alpha)$ of $\pi$ is
    added as a factor (with no Jacobian, so the effective prior on $\pi$ is not the Dirichlet, §T.4), and the
    log-likelihood $\sum_n\ln\sum_jp_{nj}\pi_j/\pi^P_j$ of the outputs $p_{nj}$ of the bag, with $\pi^P$ the source
    prevalence, is divided by the temperature $\tau$.
    """
    test_posteriors = jnp.asarray(test_posteriors)
    n_classes = test_posteriors.shape[1]
    z = numpyro.sample(ILR_Z, dist.Normal(jnp.zeros(n_classes - 1), 1.0))
    prev = ilr_inverse(z)
    numpyro.factor("dirichlet_prior", dist.Dirichlet(jnp.asarray(alpha)).log_prob(prev))
    log_w = jnp.log(prev) - jnp.log(jnp.asarray(train_prev))
    log_likelihood = jnp.sum(logsumexp(jnp.log(test_posteriors) + log_w, axis=-1))
    numpyro.factor("likelihood", (1.0 / temperature) * log_likelihood)


class BayCAPMAPLS(QuantifierBayCAP):
    r"""Push the posterior draws of BayesianMAPLS through the fixed rate matrix of $h$ on $V$ (BayCAP-EMQ, E3.1).

    BayesianMAPLS reads the outputs of the input classifier as calibrated probabilities (:func:`emq_model`). Its source
    prevalence $\pi^P$ is that of the labels of $V$ (QuaPy's ``exact_train_prev=True``), its chain starts at the EM
    estimate of QuaPy's ``mapls`` (``mapls_chain_init=True``), and its effective prior on $\pi$ is logistic-normal even
    with ``prior="uniform"`` (§T.4). The input classifier, the estimands, the seeds and the diagnostics are those of
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

    def _fit_quantifier(self, val_outputs: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> None:
        # BayesianMAPLS.aggregation_fit with exact_train_prev=True
        self.train_prevalence_ = F.prevalence_from_labels(labels, classes=self.quantifier_classes_)

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model, its arguments and the initial parameters, as QuaPy's ``sample_from_posterior``."""
        alpha = _resolve_dirichlet_prior(self.prior, outputs.shape[1])
        map_estimate = mapls(self.val_outputs_, test_probs=outputs, pz=self.train_prevalence_, return_lambda=True)[0]
        args = {"test_posteriors": outputs, "alpha": alpha, "train_prev": self.train_prevalence_}
        return emq_model, args | {"temperature": self.temperature}, {ILR_Z: ilr(map_estimate)}

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        return np.asarray(ilr_inverse(grouped[ILR_Z]))


__all__ = ["ILR_Z", "BayCAPMAPLS", "emq_model", "ilr", "ilr_inverse"]
