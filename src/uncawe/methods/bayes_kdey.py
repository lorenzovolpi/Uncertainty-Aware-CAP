r"""BayesKDEy: the posterior of $\pi^Q$ under kernel density estimates of the outputs within each class.

The NUTS model is copied from QuaPy's ``quapy.method._bayesian`` (``BayesianKDEy``, commit ``31f36f6`` of branch
``devel``), since QuaPy builds the class only when ``pystan`` is installed (decision of the user, 2026-11-03); the copy
gives the draws of QuaPy's ``sample_from_posterior`` (``tests/test_baycap_fixed.py``). The kernel density estimates are
QuaPy's (``quapy.method._kdey.KDEBase``). Work-plan card WP-25 moved the quantifier out of
``methods/baycap_kdey.py:BayCAPKDEy`` into this class, used by ``BayCAPKDEy`` and ``methods/sqs_b.py:SQSB``, with the
same draws (``tests/test_wp25_restructure.py``).

WP-25 (E7.1 of ``docs/ext7_kdey_bandwidth.md``) adds ``bandwidth="loo"``: each class $\omega_j$ gets its own bandwidth
$\hat b_j$, the value of a grid that maximises the leave-one-out log-likelihood of the class density on its validation
outputs (§T.2, §T.3), and the class densities are normalised on the simplex, where the outputs lie, in $K-1$ dimensions
instead of the $K$ of QuaPy's Gaussian kernel (§T.4). With one bandwidth shared by the classes the two normalisations
differ by a factor common to the classes, which cancels in the posterior; with per-class bandwidths it would not.

WP-26 (E7.2) adds what the selection of one shared bandwidth on calibration bags needs
(``methods/baycap_kdey.py:BayCAPKDEy`` with ``bandwidth="cal"``): the class densities at any shared bandwidth
(:meth:`BayesKDEy.class_log_densities`) and the maximum-likelihood estimate of $\pi$ on many bags by EM
(:func:`em_prevalences`, §T.8).
"""

from collections.abc import Callable
from numbers import Number
from typing import Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp as jax_logsumexp
from quapy.method._bayesian import _resolve_dirichlet_prior
from quapy.method._kdey import KDEBase
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity

from uncawe.methods.bayesian_quantifier import PREV, BayesianQuantifier

# the selection of the bandwidth by leave-one-out likelihood (E7.1)
LOO = "loo"
# the grid of E7.1: 21 values log-spaced from 0.01 to 1, step 0.1 in log10(b), containing QuaPy's default 0.1
LOO_GRID = tuple(float(b) for b in np.logspace(-2, 0, 21))
# the bandwidth of a class with fewer than two validation outputs, which has no leave-one-out likelihood (§T.3); also
# the reference value at which the leave-one-out likelihood is reported
LOO_DEFAULT_BANDWIDTH = 0.1
# rows of the matrix of squared distances computed at once, bounded so that a block holds about 10^7 entries
LOO_BLOCK_ENTRIES = 10_000_000
# per-bag summaries added by bandwidth="loo", vectors over the classes in the order of val.classes_: the bandwidth of
# each class, its number of validation outputs n_j, -1/0/+1 when the bandwidth is at the lower edge of the grid, inside
# it, or at its upper edge (0 for a class that keeps the default), and the leave-one-out log-likelihood per instance at
# the selected bandwidth and at LOO_DEFAULT_BANDWIDTH (NaN for n_j < 2)
LOO_SUMMARIES = ("bandwidths", "class_sizes", "bandwidth_edge", "loo_selected", "loo_default")
# the EM estimate of the prevalences (E7.2, §T.8): a bag stops when no component changes by more than EM_TOL, after at
# most EM_MAX_ITER steps
EM_TOL = 1e-9
EM_MAX_ITER = 10_000


def kdey_model(test_log_densities, alpha, temperature: float = 1.0) -> None:
    r"""The NUTS model of BayesKDEy (QuaPy's ``BayesianKDEy._numpyro_model``).

    $\pi\sim\operatorname{Dir}(\alpha)$, and the log-likelihood $\sum_n\ln\sum_j\pi_jp_j(x_n)$ of the outputs $x_n$ of
    the bag, with $p_j$ the kernel density of class $\omega_j$ on $V$ (``test_log_densities[j, n]``$=\ln p_j(x_n)$),
    divided by the temperature $\tau$.
    """
    prev = numpyro.sample(PREV, dist.Dirichlet(jnp.asarray(alpha)))
    log_likelihood = jnp.sum(jax_logsumexp(jnp.log(prev)[:, None] + test_log_densities, axis=0))
    numpyro.factor("loglik", (1.0 / temperature) * log_likelihood)


def em_prevalences(
    log_densities: np.ndarray, counts: np.ndarray, tol: float = EM_TOL, max_iter: int = EM_MAX_ITER
) -> tuple[np.ndarray, np.ndarray]:
    r"""The maximum-likelihood estimates of $\pi$ of the BayesKDEy model on several bags, by EM (E7.2, §T.8).

    The bags are made of instances of one pool, possibly repeated: bag $s$ holds instance $u$ of the pool
    ``counts[s, u]`` times. The log-likelihood of bag $s$ is $\ell_s(\pi)=\sum_u c_{su}\ln\sum_j\pi_j\hat p_j(z_u)$,
    concave in $\pi$. From the uniform $\pi$, each EM step computes, for every instance, the probabilities
    $r_{uj}=\pi_j\hat p_j(z_u)/\sum_k\pi_k\hat p_k(z_u)$ that it comes from each class and sets
    $\pi_j\leftarrow\frac1{N_s}\sum_uc_{su}r_{uj}$, with $N_s=\sum_uc_{su}$; each step increases $\ell_s$. A bag stops
    when no component of $\pi$ changes by more than ``tol``; the others go on, up to ``max_iter`` steps. An instance whose
    log-density is $-\infty$ under every class has a likelihood that does not depend on $\pi$ and is left out.

    :param log_densities: ``[j, u]``$=\ln\hat p_j(z_u)$, shape ``(K, n)``.
    :param counts: the multiplicity of each instance in each bag, shape ``(S, n)``, non-negative, each row with a
        positive sum.
    :param tol: the largest change of a component of $\pi$ at which a bag has converged.
    :param max_iter: the largest number of EM steps.
    :return: the estimates, shape ``(S, K)``, and a boolean array of shape ``(S,)``, true for the bags that converged.
    """
    log_densities = np.asarray(log_densities, dtype=float)
    counts = np.asarray(counts, dtype=float)
    n_classes = log_densities.shape[0]
    if counts.ndim != 2 or counts.shape[1] != log_densities.shape[1]:
        raise ValueError(f"counts must have shape (S, {log_densities.shape[1]}); found {counts.shape}")
    if np.any(counts < 0) or np.any(counts.sum(axis=1) <= 0):
        raise ValueError("counts must be non-negative, with a positive sum in each bag")
    # the instances that appear in no bag do not enter any estimate, nor those whose density is numerically 0 under
    # every class (possible at small bandwidths far from every validation output), whose likelihood does not depend on pi
    used = (counts.sum(axis=0) > 0) & np.isfinite(log_densities.max(axis=0))
    # the densities of an instance are scaled by their largest value, which cancels in r_uj
    log_p = log_densities[:, used]
    densities = np.exp(log_p - log_p.max(axis=0, keepdims=True))
    counts = counts[:, used]
    # a bag left without instances keeps the uniform pi
    sizes = np.maximum(counts.sum(axis=1), np.finfo(float).tiny)
    prev = np.full((len(counts), n_classes), 1.0 / n_classes)
    active = np.ones(len(counts), dtype=bool)
    for _ in range(max_iter):
        current = prev[active]
        # mixture[s, u] = sum_k pi_sk p_k(z_u); new pi_sj = pi_sj / N_s sum_u c_su p_j(z_u) / mixture[s, u]
        mixture = np.maximum(current @ densities, np.finfo(float).tiny)
        updated = current * ((counts[active] / mixture) @ densities.T) / sizes[active, None]
        changed = np.abs(updated - current).max(axis=1) > tol
        prev[active] = updated
        active[np.flatnonzero(active)[~changed]] = False
        if not active.any():
            break
    return prev / prev.sum(axis=1, keepdims=True), ~active


def loo_log_likelihood(points: np.ndarray, bandwidths, block_entries: int = LOO_BLOCK_ENTRIES) -> np.ndarray:
    r"""The leave-one-out log-likelihood $\operatorname{LOO}(b)=\sum_i\ln\hat p_{-i}(z_i)$ of a Gaussian kernel density
    on points of the simplex, for each bandwidth $b$ (§T.2).

    $\hat p_{-i}(z_i)=\frac{1}{n-1}\sum_{l\ne i}\varphi_b(z_i-z_l)$, with $\varphi_b$ the Gaussian density in the
    $d=K-1$ dimensions of the simplex (§T.4). The squared distances are computed once per block of rows and reused for
    every bandwidth.

    :param points: the $n\ge2$ points, of shape ``(n, K)``, rows on the simplex.
    :param bandwidths: the bandwidths, a sequence of positive numbers.
    :param block_entries: the size of a block of the matrix of squared distances.
    :return: an array with one value per bandwidth.
    """
    points = np.asarray(points, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
    n, n_columns = points.shape
    if n < 2:
        raise ValueError("the leave-one-out likelihood needs at least two points")
    dim = n_columns - 1
    norms = np.einsum("ij,ij->i", points, points)
    total = np.zeros(len(bandwidths))
    rows = max(1, block_entries // n)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        sq = norms[start:stop, None] + norms[None, :] - 2.0 * points[start:stop] @ points.T
        np.maximum(sq, 0.0, out=sq)
        # leave each point out of its own density
        sq[np.arange(stop - start), np.arange(start, stop)] = np.inf
        for g, b in enumerate(bandwidths):
            total[g] += logsumexp(-sq / (2.0 * b * b), axis=1).sum()
    return total - n * (np.log(n - 1) + 0.5 * dim * np.log(2.0 * np.pi * bandwidths**2))


def simplex_log_density(kde: KernelDensity, outputs: np.ndarray) -> np.ndarray:
    r"""The log-density of a Gaussian ``KernelDensity`` fitted on points of the simplex, normalised in the $K-1$
    dimensions of the simplex: scikit-learn's log-density, normalised in $K$ dimensions, plus $\frac12\ln(2\pi b^2)$
    (§T.4)."""
    return kde.score_samples(outputs) + 0.5 * np.log(2.0 * np.pi * kde.bandwidth**2)


class _KernelDensities(KDEBase):
    """QuaPy's kernel density estimates of the outputs within each class, as fitted by ``BayesianKDEy``."""

    def __init__(self, kernel: str, bandwidth, shrinkage: float):
        self.kernel = KDEBase._check_kernel(kernel)
        self.bandwidth = KDEBase._check_bandwidth(bandwidth, self.kernel)
        if not 0 <= shrinkage < 1:
            raise ValueError("shrinkage must be in [0,1)")
        if self.kernel == "gaussian" and shrinkage != 0:
            raise ValueError("shrinkage is only supported for Aitchison/ILR kernels")
        self.shrinkage = float(shrinkage)

    def fit(self, outputs: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> Self:
        self.mix_densities = self.get_mixture_components(outputs, labels, classes, self.bandwidth, self.kernel)
        return self

    def log_densities(self, outputs: np.ndarray) -> np.ndarray:
        """``[j, n]``: the log-density of class $j$ at the outputs of instance $n$."""
        return np.asarray([self.pdf(kde, outputs, self.kernel, log_densities=True) for kde in self.mix_densities])

    def density(self, outputs: np.ndarray):
        """A kernel density with the kernel and bandwidth of the class densities, fitted on ``outputs``."""
        return self.get_kde_function(outputs, self.bandwidth, self.kernel)

    def log_density(self, kde, outputs: np.ndarray) -> np.ndarray:
        """The log-density of ``kde`` at ``outputs``."""
        return self.pdf(kde, outputs, self.kernel, log_densities=True)


class _SelectedKernelDensities:
    r"""Gaussian kernel densities of the outputs within each class, each with the bandwidth $\hat b_j$ of the grid that
    maximises its leave-one-out log-likelihood (§T.2, §T.3), normalised on the simplex (§T.4). A class with fewer than
    two outputs keeps ``LOO_DEFAULT_BANDWIDTH``."""

    def __init__(self, grid):
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
            raise ValueError("bandwidth_grid must be a non-empty increasing sequence of positive numbers")
        self.grid = grid

    def fit(self, outputs: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> Self:
        n_classes = len(classes)
        self.bandwidths = np.full(n_classes, LOO_DEFAULT_BANDWIDTH)
        self.class_sizes = np.zeros(n_classes, dtype=int)
        self.edges = np.zeros(n_classes, dtype=int)
        self.loo_selected = np.full(n_classes, np.nan)
        self.loo_default = np.full(n_classes, np.nan)
        self.loo_curves = np.full((n_classes, len(self.grid)), np.nan)
        self.mix_densities = []
        for j, class_ in enumerate(classes):
            points = outputs[labels == class_]
            if len(points) == 0:
                raise ValueError(f"empty class {class_}")
            n = len(points)
            self.class_sizes[j] = n
            if n >= 2:
                values = loo_log_likelihood(points, np.append(self.grid, LOO_DEFAULT_BANDWIDTH))
                curve, default = values[:-1], values[-1]
                best = int(np.argmax(curve))
                self.bandwidths[j] = self.grid[best]
                self.edges[j] = -1 if best == 0 else (1 if best == len(self.grid) - 1 else 0)
                self.loo_curves[j] = curve
                self.loo_selected[j] = curve[best] / n
                self.loo_default[j] = default / n
            self.mix_densities.append(KernelDensity(bandwidth=float(self.bandwidths[j])).fit(points))
        return self

    def log_densities(self, outputs: np.ndarray) -> np.ndarray:
        """``[j, n]``: the log-density of class $j$ at the outputs of instance $n$, normalised on the simplex."""
        return np.asarray([simplex_log_density(kde, outputs) for kde in self.mix_densities])


class BayesKDEy(BayesianQuantifier):
    r"""BayesKDEy: a Dirichlet prior on $\pi$ and the likelihood $\prod_n\sum_j\pi_j\hat p_j(z_n)$ of the outputs $z_n$
    of the input classifier on the bag, with $\hat p_j$ the kernel density of the validation outputs of class
    $\omega_j$ (:func:`kdey_model`).

    The input classifier, the seeds and the NUTS run are those of
    :class:`uncawe.methods.bayesian_quantifier.BayesianQuantifier`.

    - A positive number, ``"scott"`` or ``"silverman"`` as ``bandwidth``: QuaPy's class densities, one bandwidth for
      all classes, with the given ``kernel`` and ``shrinkage``.
    - ``bandwidth="loo"`` (E7.1, Gaussian kernel and no shrinkage only): one bandwidth per class, selected over
      ``bandwidth_grid`` by leave-one-out likelihood, with densities normalised on the simplex
      (:func:`loo_log_likelihood`, :func:`simplex_log_density`).

    After ``fit``, ``bandwidths_`` holds the bandwidth of each class (in the order of ``val.classes_``) and, with
    ``"loo"``, :meth:`selection_summaries` the ``LOO_SUMMARIES``.

    :param kernel: ``"gaussian"``, ``"aitchison"`` or ``"ilr"``.
    :param bandwidth: a positive number, ``"scott"``, ``"silverman"`` or ``"loo"``.
    :param bandwidth_grid: the grid of ``"loo"``; ``LOO_GRID`` if ``None``.
    :param shrinkage: shrinkage in $[0,1)$, $0$ with the Gaussian kernel.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the bag.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS and of the MLP; ``qp.environ["_R_SEED"]`` if ``None``.
    """

    def __init__(
        self,
        input_classifier: Literal["h", "mlp"] = "mlp",
        kernel: Literal["gaussian", "aitchison", "ilr"] = "gaussian",
        bandwidth: float | Literal["scott", "silverman", "loo"] = 0.1,
        bandwidth_grid=None,
        shrinkage: float = 0.0,
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            input_classifier=input_classifier,
            num_warmup=num_warmup,
            num_samples=num_samples,
            temperature=temperature,
            prior=prior,
            random_state=random_state,
        )
        if bandwidth == LOO and (kernel != "gaussian" or shrinkage != 0):
            raise ValueError('bandwidth="loo" needs the Gaussian kernel and no shrinkage')
        if bandwidth != LOO and bandwidth_grid is not None:
            raise ValueError('bandwidth_grid is used only with bandwidth="loo"')
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.shrinkage = shrinkage

    @property
    def selects_bandwidth(self) -> bool:
        """Whether the bandwidths are selected per class by leave-one-out likelihood."""
        return isinstance(self.bandwidth, str) and self.bandwidth == LOO

    def _fit(self, val_outputs: np.ndarray, labels: np.ndarray) -> None:
        if self.selects_bandwidth:
            grid = LOO_GRID if self.bandwidth_grid is None else self.bandwidth_grid
            self.kde_ = _SelectedKernelDensities(grid).fit(val_outputs, labels, self.quantifier_classes_)
            bandwidths = self.kde_.bandwidths
        else:
            self.kde_ = _KernelDensities(self.kernel, self.bandwidth, self.shrinkage)
            self.kde_.fit(val_outputs, labels, self.quantifier_classes_)
            bandwidths = np.asarray([kde.bandwidth_ for kde in self.kde_.mix_densities], dtype=float)
        self.bandwidths_ = bandwidths[self.class_order_]

    def class_log_densities(self, outputs: np.ndarray, bandwidth: float) -> np.ndarray:
        r"""The log-densities at ``outputs`` of QuaPy's class densities fitted on the validation outputs of the
        quantifier with one bandwidth ``bandwidth`` shared by the classes (and the kernel and shrinkage of the
        quantifier), whatever its own bandwidth (E7.2).

        :param outputs: outputs of the input classifier, shape ``(n, K)``, columns in the order of
            ``quantifier_classes_``.
        :return: ``[j, u]``$=\ln\hat p_j(z_u)$, shape ``(K, n)``, rows in the order of ``val.classes_``.
        """
        self._check_is_fitted()
        densities = _KernelDensities(self.kernel, bandwidth, self.shrinkage)
        densities.fit(self.val_outputs_, self.val_labels_, self.quantifier_classes_)
        return densities.log_densities(outputs)[self.class_order_]

    def selection_summaries(self) -> dict:
        """The ``LOO_SUMMARIES`` of the selection, vectors over the classes in the order of ``val.classes_``."""
        if not self.selects_bandwidth:
            raise RuntimeError('selection_summaries exist only with bandwidth="loo"')
        self._check_is_fitted()
        order = self.class_order_
        return {
            "bandwidths": self.kde_.bandwidths[order],
            "class_sizes": self.kde_.class_sizes[order],
            "bandwidth_edge": self.kde_.edges[order],
            "loo_selected": self.kde_.loo_selected[order],
            "loo_default": self.kde_.loo_default[order],
        }

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model of BayesKDEy, its arguments and the initial parameters, as QuaPy's ``sample_from_posterior``."""
        alpha = _resolve_dirichlet_prior(self.prior, outputs.shape[1])
        args = {"test_log_densities": self.kde_.log_densities(outputs), "alpha": alpha}
        return kdey_model, args | {"temperature": self.temperature}, None

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        return np.asarray(jax.device_get(grouped[PREV]))


__all__ = [
    "EM_MAX_ITER",
    "EM_TOL",
    "LOO",
    "LOO_DEFAULT_BANDWIDTH",
    "LOO_GRID",
    "LOO_SUMMARIES",
    "BayesKDEy",
    "em_prevalences",
    "kdey_model",
    "loo_log_likelihood",
    "simplex_log_density",
]
