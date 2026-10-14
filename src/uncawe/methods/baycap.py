"""BayCAP with a population or a realised-table estimand.

This module implements the method of work-plan card WP-1, motivated by the
realised-table posterior predictive of ``docs/ext2_baycap_priors.md``. The
posterior model, the priors and the NUTS sampler are copied from ``cap-ml``
``cap.models.confidence.ACC_BayesCAP`` (commit ``bb20aa5`` of branch
``devel``); only the pushforward of the posterior draws to contingency tables
is extended.
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
from cap.models.confidence import P_C_COND_Y, P_TEST_C, P_TEST_Y, BayesCAP, CTCAPWithConfidence
from cap.models.cont_table import CAPContingencyTable
from quapy.data import LabelledCollection
from sklearn.metrics import confusion_matrix

ESTIMANDS = ("population", "realised")


def realised_tables(cts: np.ndarray, pred_counts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    r"""Sample one realised contingency table of the target sample per table draw.

    For each draw $r$ and each predicted class $\omega_i$, the true classes of
    the $n^t_i$ target instances predicted as $\omega_i$ are drawn as
    $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, with
    $\rho^{(r)}_{ij}=C^{(r)}_{ij}\pi^{(r)}_j/\sum_l C^{(r)}_{il}\pi^{(r)}_l$,
    i.e. the draw's table restricted to predicted class $\omega_i$ and
    normalised to one. The sampled counts are divided by
    $N^Q=\sum_i n^t_i$.

    Tables follow the ``cap-ml`` layout (true classes on rows, predicted
    classes on columns), so each column of a draw is resampled.

    :param cts: table draws of shape ``(n_draws, n_classes, n_classes)``,
        non-negative, in the ``cap-ml`` layout.
    :param pred_counts: number of target instances predicted as each class,
        of shape ``(n_classes,)``.
    :param rng: NumPy generator used for the multinomial draws.
    :return: normalised realised tables with the shape of ``cts``; the column
        sums of every table equal ``pred_counts / pred_counts.sum()``.
    """
    cts = np.asarray(cts, dtype=float)
    pred_counts = np.asarray(pred_counts, dtype=np.int64)
    if cts.ndim != 3 or cts.shape[1] != cts.shape[2]:
        raise ValueError(f"cts must have shape (n_draws, n_classes, n_classes); found {cts.shape}")
    if pred_counts.shape != (cts.shape[2],):
        raise ValueError(f"pred_counts must have shape ({cts.shape[2]},); found {pred_counts.shape}")
    if np.any(pred_counts < 0) or pred_counts.sum() == 0:
        raise ValueError("pred_counts must be non-negative and not all zero")
    if np.any(cts < 0) or not np.all(np.isfinite(cts)):
        raise ValueError("cts must contain finite non-negative values")

    # rho[r, c, y] = P(Y=y | Yhat=c) under draw r; empty columns fall back to uniform
    rho = np.swapaxes(cts, 1, 2)
    col_mass = rho.sum(axis=2, keepdims=True)
    n_classes = cts.shape[1]
    rho = np.divide(rho, col_mass, out=np.full_like(rho, 1.0 / n_classes), where=col_mass > 0)

    n = np.broadcast_to(pred_counts, rho.shape[:2])
    counts = rng.multinomial(n, rho)
    return np.swapaxes(counts, 1, 2) / pred_counts.sum()


class BayCAP(CAPContingencyTable, CTCAPWithConfidence, BayesCAP):
    r"""BayCAP with a selectable estimand.

    The posterior over the target prevalence $\pi^Q$ and the rate matrix $C$
    is the BayesianCC model of ``ACC_BayesCAP``: a Dirichlet prior on
    $\pi^Q$ (uniform by default), a uniform Dirichlet prior on each column of
    $C$, a multinomial likelihood for the validation contingency table and a
    multinomial likelihood for the predicted counts $n^t$ of the target
    sample. It is sampled with NumPyro NUTS from ``PRNGKey(random_state)``.

    The estimand decides how each posterior draw becomes a contingency table:

    - ``"population"`` returns the population table
      $C^{(r)}\operatorname{diag}(\pi^{(r)})$, exactly as ``ACC_BayesCAP``.
      It represents the uncertainty on $\pi^Q$ and on $C$, not the
      finite-sample fluctuation of the target sample's table.
    - ``"realised"`` returns the realised-table posterior predictive (see
      :func:`realised_tables`): for each draw, the true classes behind the
      observed predictions are sampled. It represents the uncertainty on
      $\pi^Q$ and on $C$ and the finite-sample fluctuation of the realised
      table of the target sample, which is what the experiments score
      (``true_cts``).

    Both estimands share the NUTS draws for the same ``random_state``; the
    multinomial draws of ``"realised"`` use a NumPy generator seeded with
    ``random_state``. As in ``cap-ml``, tables have true classes on rows, so
    the paper's $C\operatorname{diag}(\pi)$ is returned transposed, as
    $\operatorname{diag}(\pi)C^\top$.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param estimand: ``"population"`` or ``"realised"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param prior: Dirichlet parameter of the prior on $\pi^Q$: ``"uniform"``,
        a positive scalar repeated for every class, or a vector with one
        value per class.
    :param random_state: seed of the NUTS sampler and of the multinomial
        draws; ``qp.environ["_R_SEED"]`` if ``None``, as in ``cap-ml``.
    """

    def __init__(
        self,
        acc_fn: Callable,
        estimand: Literal["population", "realised"] = "population",
        num_warmup: int = 500,
        num_samples: int = 1000,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ):
        if estimand not in ESTIMANDS:
            raise ValueError(f"estimand must be one of {ESTIMANDS}; found {estimand!r}")
        CAPContingencyTable.__init__(self, acc_fn)
        self.estimand = estimand
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state
        self.prior = prior

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        val_yhat = np.argmax(posteriors, axis=-1)
        self.classes = val.classes
        self.val_ct = confusion_matrix(val.y, val_yhat, labels=val.classes)
        return self

    def model(self, pred_posterior_count: np.ndarray, train_class_cond_count: np.ndarray, alpha: np.ndarray):
        train_class_count = train_class_cond_count.sum(axis=1)

        K = len(pred_posterior_count)
        L = len(train_class_count)

        pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.asarray(alpha, dtype=jnp.float32)))
        p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.ones(K).repeat(L).reshape(L, K)))

        with numpyro.plate("plate", L):
            numpyro.sample("F_yc", dist.Multinomial(train_class_count, p_c_cond_y), obs=train_class_cond_count)

        p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
        numpyro.sample("N_c", dist.Multinomial(jnp.sum(pred_posterior_count), p_c), obs=pred_posterior_count)

    def sample_posterior(self, posterior_count: np.ndarray, alpha: np.ndarray) -> dict:
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self.model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(self.random_state)
        mcmc.run(rng_key, pred_posterior_count=posterior_count, train_class_cond_count=self.val_ct, alpha=alpha)
        return mcmc.get_samples()

    def _alpha(self) -> np.ndarray:
        n_classes = len(self.classes)
        if isinstance(self.prior, str) and self.prior == "uniform":
            return np.ones(n_classes, dtype=float)
        if isinstance(self.prior, Number):
            return np.full(n_classes, float(self.prior), dtype=float)
        alpha = np.asarray(self.prior, dtype=float)
        if alpha.ndim != 1 or len(alpha) != n_classes:
            raise ValueError(f"wrong shape for prior; expected {n_classes} values, found shape {alpha.shape}")
        return alpha

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        """Return one normalised contingency table per posterior draw (true classes on rows)."""
        yhat = np.argmax(posteriors, axis=-1)
        posterior_count = qp.functional.counts_from_labels(yhat, self.classes)

        samples = self.sample_posterior(posterior_count, alpha=self._alpha())

        # compute P(Y,C) for all samples from P(Y) and P(C|Y)
        p_y_and_c_test = jnp.einsum("sy,syc->syc", samples[P_TEST_Y], samples[P_C_COND_Y])
        cts = np.array(jax.device_get(p_y_and_c_test))
        if self.estimand == "realised":
            cts = realised_tables(cts, posterior_count, np.random.default_rng(self.random_state))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)
