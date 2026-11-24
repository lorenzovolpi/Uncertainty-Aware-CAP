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
(:func:`em_prevalences`, §T.8). The gate of the solver of E7.2a adds the same estimate by QuaPy's ``KDEyML``, the standard
KDEy, which maximises the same log-likelihood with SciPy's SLSQP (:func:`slsqp_prevalences`), and the log-likelihood of
an estimate on each bag (:func:`bag_log_likelihoods`), to tell which of two estimates is the better maximiser.

WP-33 (E3.5 of ``docs/ext3_posterior_propagation.md``, §T.16) adds ``bandwidth="logloss"``: one bandwidth $b^\star$
shared by the classes, the value of a grid that minimises the class-balanced leave-one-out log-loss of the kernel
(Nadaraya–Watson) class probabilities of the validation instances (:func:`class_logloss_curve`); the class densities are
then fitted at $b^\star$ exactly as with that numeric bandwidth, so the likelihood of the bag uses $b^\star$.
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
from quapy.error import smooth
from quapy.functional import uniform_prevalence
from quapy.method._bayesian import _resolve_dirichlet_prior
from quapy.method._kdey import KDEBase
from scipy import optimize
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity

from uncawe.methods.bayesian_quantifier import PREV, BayesianQuantifier, confidence_ece

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
# the constants of QuaPy's KDEyML.aggregate (quapy.method._kdey): the smoothing of the mixture density in the logarithm,
# and the number of classes from which it works with log-densities (for the Gaussian kernel)
KDEYML_EPSILON = 1e-12
KDEYML_LOG_CLASSES = 30
# the selection of one shared bandwidth by the class-balanced leave-one-out log-loss of the kernel class probabilities on
# V (E3.5, §T.16), over LOO_GRID by default, ties broken towards LOO_DEFAULT_BANDWIDTH (select_from_curve)
LOGLOSS = "logloss"
# the classes that enter the log-loss: those with at least this number of validation instances (§T.16)
LOGLOSS_MIN_CLASS_SIZE = 2
# per-bag summaries added by bandwidth="logloss" (fixed by the fit, so the same for every bag): the selected bandwidth
# b*, the curve L(b) over the grid (a vector), its values at b* and at LOO_DEFAULT_BANDWIDTH, -1/0/+1 when b* is at the
# lower edge of the grid, inside it, or at its upper edge, the number |J| of classes in the log-loss, and the ECE of the
# leave-one-out confidences of the validation instances (the probability given to the class predicted by the input
# classifier, against whether that prediction is right) at LOO_DEFAULT_BANDWIDTH and at b*
LOGLOSS_SUMMARIES = (
    "logloss_bandwidth",
    "logloss_curve",
    "logloss_selected",
    "logloss_default",
    "logloss_edge",
    "logloss_classes",
    "loo_ece_default",
    "loo_ece_selected",
)


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


def slsqp_prevalences(log_densities: np.ndarray, bag_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""The maximum-likelihood estimates of $\pi$ of the BayesKDEy model on several bags by QuaPy's ``KDEyML`` (the gate
    of the solver of E7.2a).

    The same log-likelihood as :func:`em_prevalences`, $\ell_s(\pi)=\sum_{u\in s}\ln\sum_j\pi_j\hat p_j(z_u)$, is
    maximised bag by bag as ``KDEyML.aggregate`` does (``quapy.method._kdey``, Gaussian kernel): SciPy's SLSQP through
    ``quapy.functional.optim_minimize``, from the uniform $\pi$, with bounds $[0,1]$ per component, the constraint
    $\sum_j\pi_j=1$, SciPy's default tolerances and finite-difference gradients, on
    $-\sum_u\ln(\sum_j\pi_j\hat p_j(z_u)+10^{-12})$ with $\hat p_j=\exp(\ln\hat p_j)$ (QuaPy's ``pdf``); from
    ``KDEYML_LOG_CLASSES`` classes on, on $-\sum_u\operatorname{logsumexp}_j(\ln\pi_j+\ln\hat p_j(z_u))$ with $\pi$
    smoothed by ``quapy.error.smooth`` ($\epsilon=10^{-12}$), as QuaPy. The minimiser is called directly, with the
    arguments of ``optim_minimize``, so that its convergence flag can be read; the estimate is SciPy's, not
    normalised, as QuaPy returns it.

    :param log_densities: ``[j, u]``$=\ln\hat p_j(z_u)$ on a pool, shape ``(K, n)``.
    :param bag_index: the indices into the pool of the instances of each bag, shape ``(S, N)``, repeated as drawn.
    :return: the estimates, shape ``(S, K)``, and a boolean array of shape ``(S,)``, true where SLSQP reports success.
    """
    log_densities = np.asarray(log_densities, dtype=float)
    n_classes = log_densities.shape[0]
    use_logs = n_classes >= KDEYML_LOG_CLASSES
    densities = log_densities if use_logs else np.exp(log_densities)
    bounds = tuple((0, 1) for _ in range(n_classes))
    constraints = {"type": "eq", "fun": lambda x: 1 - sum(x)}
    estimates = np.empty((len(bag_index), n_classes))
    success = np.empty(len(bag_index), dtype=bool)
    for s, index in enumerate(np.asarray(bag_index)):
        bag = densities[:, index]
        if use_logs:

            def neg_loglikelihood(prev, bag=bag):
                prev = smooth(prev, eps=KDEYML_EPSILON)
                return -np.sum(logsumexp(np.log(prev)[:, None] + bag, axis=0))
        else:

            def neg_loglikelihood(prev, bag=bag):
                return -np.sum(np.log(prev @ bag + KDEYML_EPSILON))

        result = optimize.minimize(
            neg_loglikelihood,
            x0=uniform_prevalence(n_classes=n_classes),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        estimates[s], success[s] = result.x, bool(result.success)
    return estimates, success


def bag_log_likelihoods(log_densities: np.ndarray, counts: np.ndarray, estimates: np.ndarray) -> np.ndarray:
    r"""The log-likelihood $\ell_s(\hat\pi_s)=\sum_uc_{su}\ln\sum_j\hat\pi_{sj}\hat p_j(z_u)$ of each bag at its
    estimate, by log-sum-exp, over the instances that :func:`em_prevalences` uses (those whose log-density is finite
    under some class); negative components of an estimate (SLSQP may return $-10^{-17}$) count as $0$.

    :param log_densities: ``[j, u]``$=\ln\hat p_j(z_u)$, shape ``(K, n)``.
    :param counts: the multiplicity of each instance in each bag, shape ``(S, n)``.
    :param estimates: $\hat\pi_s$, shape ``(S, K)``.
    :return: shape ``(S,)``.
    """
    log_densities = np.asarray(log_densities, dtype=float)
    counts = np.asarray(counts, dtype=float)
    used = (counts.sum(axis=0) > 0) & np.isfinite(log_densities.max(axis=0))
    with np.errstate(divide="ignore"):
        log_prev = np.log(np.clip(np.asarray(estimates, dtype=float), 0.0, None))
    # [s, u] = logsumexp_j(ln pi_sj + ln p_j(z_u))
    mixture = logsumexp(log_prev[:, :, None] + log_densities[None, :, used], axis=1)
    return np.sum(np.where(counts[:, used] > 0, counts[:, used] * mixture, 0.0), axis=1)


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


def select_from_curve(grid, curve: np.ndarray, reference: float = LOO_DEFAULT_BANDWIDTH) -> int:
    r"""The index of the smallest value of ``curve`` over ``grid``; among values equal up to rounding (relative
    difference $10^{-9}$), the bandwidth closest to ``reference`` on the $\log_{10}$ scale, and the smaller of two at the
    same distance."""
    grid, curve = np.asarray(grid, dtype=float), np.asarray(curve, dtype=float)
    if not np.isfinite(curve).any():
        raise ValueError("the criterion is not finite at any bandwidth")
    best = np.nanmin(curve)
    ties = np.flatnonzero(np.isclose(curve, best, rtol=1e-9, atol=0.0))
    return int(ties[np.argmin(np.abs(np.log10(grid[ties]) - np.log10(reference)))])


def _class_codes(labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """The index into ``classes`` of each label; ``ValueError`` for a label that is not a class."""
    labels, classes = np.asarray(labels), np.asarray(classes)
    matches = labels[:, None] == classes[None, :]
    if not np.all(matches.any(axis=1)):
        raise ValueError("every label must be one of the classes")
    return np.argmax(matches, axis=1)


def _loo_class_log_sums(points: np.ndarray, codes: np.ndarray, n_classes: int, bandwidths, block_entries: int):
    r"""Blocks of the leave-one-out class sums of §T.16: for the rows $u$ of a block, ``[g, u, j]``$=\ln\sum_{v\in
    V_j,v\ne u}\exp(-\lVert z_u-z_v\rVert^2/2b_g^2)$, $-\infty$ when class $j$ has no instance other than $u$.

    Each sum is computed relative to its largest term: with $d_{uj}=\min_{v\in V_j,v\ne u}\lVert z_u-z_v\rVert^2$
    (once per block, for every bandwidth), $\ln\sum_v e^{-s_{uv}/2b^2}=-d_{uj}/2b^2+\ln\sum_v
    e^{-(s_{uv}-d_{uj})/2b^2}$, and the last sum is at least $1$, so no class sum underflows at small $b$. The sums
    over the classes are a product with the one-hot matrix of the labels.

    :return: an iterator of ``(start, stop, sums)``, ``sums`` of shape ``(G, stop - start, K)``.
    """
    n = len(points)
    norms = np.einsum("ij,ij->i", points, points)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), codes] = 1.0
    members = [np.flatnonzero(codes == j) for j in range(n_classes)]
    rows = max(1, block_entries // n)
    for start in range(0, n, rows):
        stop = min(start + rows, n)
        sq = norms[start:stop, None] + norms[None, :] - 2.0 * points[start:stop] @ points.T
        np.maximum(sq, 0.0, out=sq)
        # leave each instance out of its own sums
        sq[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = np.full((stop - start, n_classes), np.inf)
        for j, cols in enumerate(members):
            if len(cols):
                nearest[:, j] = sq[:, cols].min(axis=1)
        present = np.isfinite(nearest)
        shift = np.where(present, nearest, 0.0)
        excess = sq - shift[:, codes]
        sums = np.empty((len(bandwidths), stop - start, n_classes))
        for g, b in enumerate(bandwidths):
            scale = 2.0 * b * b
            with np.errstate(divide="ignore"):
                sums[g] = np.log(np.exp(-excess / scale) @ one_hot) - shift / scale
            sums[g][~present] = -np.inf
        yield start, stop, sums


def loo_class_log_probabilities(
    points: np.ndarray, labels: np.ndarray, classes: np.ndarray, bandwidths, block_entries: int = LOO_BLOCK_ENTRIES
) -> np.ndarray:
    r"""The leave-one-out class probabilities of the Gaussian kernel (Nadaraya–Watson) classifier on $V$ (§T.16), in
    logarithms: for each bandwidth $b$ and each instance $u$,
    $$\ln w^{(-u)}_j(z_u;b)=\ln\frac{\sum_{v\in V_j,v\ne u}\exp(-\lVert z_u-z_v\rVert^2/2b^2)}{\sum_{v\in V,
    v\ne u}\exp(-\lVert z_u-z_v\rVert^2/2b^2)},$$
    Bayes' rule with the prevalence and the kernel densities of $V$ without $u$, one bandwidth for all the classes.
    The squared distances are computed once per block of rows and reused for every bandwidth; the sums are
    log-sum-exps (:func:`_loo_class_log_sums`).

    :param points: the outputs $z_u$, shape ``(n, K')``, $n\ge2$.
    :param labels: the labels, shape ``(n,)``.
    :param classes: the classes, in the order of the last axis of the result.
    :param bandwidths: the bandwidths, a sequence of positive numbers.
    :param block_entries: the size of a block of the matrix of squared distances.
    :return: ``[g, u, j]``$=\ln w^{(-u)}_j(z_u;b_g)$, shape ``(G, n, K)``.
    """
    points, bandwidths = np.asarray(points, dtype=float), np.asarray(bandwidths, dtype=float)
    if len(points) < 2:
        raise ValueError("the leave-one-out class probabilities need at least two points")
    codes = _class_codes(labels, classes)
    out = np.empty((len(bandwidths), len(points), len(classes)))
    for start, stop, sums in _loo_class_log_sums(points, codes, len(classes), bandwidths, block_entries):
        out[:, start:stop] = sums - logsumexp(sums, axis=2, keepdims=True)
    return out


def class_logloss_curve(
    points: np.ndarray, labels: np.ndarray, classes: np.ndarray, bandwidths, block_entries: int = LOO_BLOCK_ENTRIES
) -> tuple[np.ndarray, np.ndarray]:
    r"""The class-balanced leave-one-out log-loss of §T.16 for each bandwidth,
    $$\mathcal L(b)=-\frac1{\lvert J\rvert}\sum_{j\in J}\frac1{n_j}\sum_{u\in V_j}\ln w^{(-u)}_j(z_u;b),$$
    with $w^{(-u)}$ the leave-one-out class probabilities (:func:`loo_class_log_probabilities`) and $J$ the classes with
    at least ``LOGLOSS_MIN_CLASS_SIZE`` instances; the instances of the other classes stay in the denominators.

    :return: the curve, shape ``(G,)``, and the boolean mask of $J$ over ``classes``.
    """
    points, bandwidths = np.asarray(points, dtype=float), np.asarray(bandwidths, dtype=float)
    if len(points) < 2:
        raise ValueError("the leave-one-out log-loss needs at least two points")
    codes = _class_codes(labels, classes)
    n_classes = len(classes)
    sizes = np.bincount(codes, minlength=n_classes)
    used = sizes >= LOGLOSS_MIN_CLASS_SIZE
    if not used.any():
        raise ValueError(f"the log-loss needs a class with at least {LOGLOSS_MIN_CLASS_SIZE} instances")
    # totals[g, j] = sum over the instances u of class j of ln w^(-u)_j(z_u; b_g)
    totals = np.zeros((len(bandwidths), n_classes))
    for start, stop, sums in _loo_class_log_sums(points, codes, n_classes, bandwidths, block_entries):
        block = codes[start:stop]
        own = sums[:, np.arange(stop - start), block] - logsumexp(sums, axis=2)
        keep = used[block]
        for g in range(len(bandwidths)):
            totals[g] += np.bincount(block[keep], weights=own[g, keep], minlength=n_classes)
    return -np.mean(totals[:, used] / sizes[used], axis=1), used


def loo_confidence_ece(
    points: np.ndarray, labels: np.ndarray, classes: np.ndarray, bandwidths, block_entries: int = LOO_BLOCK_ENTRIES
) -> np.ndarray:
    r"""The ECE of the leave-one-out confidences on $V$ (§T.16) for each bandwidth: the confidence of instance $u$ is
    $w^{(-u)}_{\hat y_u}(z_u;b)$, the leave-one-out probability of the class $\hat y_u$ of the largest output $z_u$
    (the prediction of $h$ when the outputs are those of $h$), against whether $\hat y_u$ is the label of $u$; the bins
    of :func:`uncawe.methods.bayesian_quantifier.confidence_ece`.

    :return: shape ``(G,)``.
    """
    points = np.asarray(points, dtype=float)
    log_w = loo_class_log_probabilities(points, labels, classes, bandwidths, block_entries)
    predicted = np.argmax(points, axis=1)
    correct = predicted == _class_codes(labels, classes)
    rows = np.arange(len(points))
    return np.asarray([confidence_ece(np.exp(log_w[g, rows, predicted]), correct) for g in range(len(log_w))])


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
    - ``bandwidth="logloss"`` (E3.5, Gaussian kernel and no shrinkage only): one bandwidth $b^\star$ shared by the
      classes, the value of ``bandwidth_grid`` of smallest class-balanced leave-one-out log-loss of the kernel class
      probabilities of $V$ (:func:`class_logloss_curve`), ties broken towards ``LOO_DEFAULT_BANDWIDTH``
      (:func:`select_from_curve`); the class densities are QuaPy's at $b^\star$, exactly as with that numeric
      bandwidth. The selection draws no random number.

    After ``fit``, ``bandwidths_`` holds the bandwidth of each class (in the order of ``val.classes_``) and, with
    ``"loo"``, :meth:`selection_summaries` the ``LOO_SUMMARIES``; with ``"logloss"``, :meth:`logloss_summaries` the
    ``LOGLOSS_SUMMARIES``.

    :param kernel: ``"gaussian"``, ``"aitchison"`` or ``"ilr"``.
    :param bandwidth: a positive number, ``"scott"``, ``"silverman"``, ``"loo"`` or ``"logloss"``.
    :param bandwidth_grid: the grid of ``"loo"`` and ``"logloss"``; ``LOO_GRID`` if ``None``.
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
        bandwidth: float | Literal["scott", "silverman", "loo", "logloss"] = 0.1,
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
        selected = isinstance(bandwidth, str) and bandwidth in (LOO, LOGLOSS)
        if selected and (kernel != "gaussian" or shrinkage != 0):
            raise ValueError(f'bandwidth="{bandwidth}" needs the Gaussian kernel and no shrinkage')
        if not selected and bandwidth_grid is not None:
            raise ValueError('bandwidth_grid is used only with bandwidth="loo" or "logloss"')
        if isinstance(bandwidth, str) and bandwidth == LOGLOSS:
            grid = np.asarray(LOO_GRID if bandwidth_grid is None else bandwidth_grid, dtype=float)
            if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
                raise ValueError("bandwidth_grid must be a non-empty increasing sequence of positive numbers")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.shrinkage = shrinkage

    @property
    def selects_bandwidth(self) -> bool:
        """Whether the bandwidths are selected per class by leave-one-out likelihood."""
        return isinstance(self.bandwidth, str) and self.bandwidth == LOO

    @property
    def selects_logloss(self) -> bool:
        """Whether one bandwidth is selected by the leave-one-out log-loss of the class probabilities."""
        return isinstance(self.bandwidth, str) and self.bandwidth == LOGLOSS

    def _select_logloss(self, val_outputs: np.ndarray, labels: np.ndarray) -> float:
        r"""Select $b^\star$ over the grid by :func:`class_logloss_curve` on the validation outputs and labels
        (E3.5, §T.16), and the ECE of the leave-one-out confidences at ``LOO_DEFAULT_BANDWIDTH`` and at $b^\star$."""
        grid = np.asarray(LOO_GRID if self.bandwidth_grid is None else self.bandwidth_grid, dtype=float)
        in_grid = np.isclose(grid, LOO_DEFAULT_BANDWIDTH)
        with_default = grid if in_grid.any() else np.append(grid, LOO_DEFAULT_BANDWIDTH)
        default = int(np.flatnonzero(np.isclose(with_default, LOO_DEFAULT_BANDWIDTH))[0])
        classes = self.quantifier_classes_
        values, used = class_logloss_curve(val_outputs, labels, classes, with_default)
        curve = values[: len(grid)]
        best = select_from_curve(grid, curve)
        selected = float(grid[best])
        ece = loo_confidence_ece(val_outputs, labels, classes, [with_default[default], selected])
        self.logloss_grid_ = grid
        self.logloss_curve_ = curve
        self.logloss_bandwidth_ = selected
        self.logloss_selected_ = float(curve[best])
        self.logloss_default_ = float(values[default])
        self.logloss_edge_ = -1 if best == 0 else (1 if best == len(grid) - 1 else 0)
        self.logloss_classes_ = int(used.sum())
        self.loo_ece_default_, self.loo_ece_selected_ = float(ece[0]), float(ece[1])
        return selected

    def _fit(self, val_outputs: np.ndarray, labels: np.ndarray) -> None:
        if self.selects_logloss:
            # the class densities at the selected bandwidth, built exactly as with that numeric bandwidth
            self.kde_ = _KernelDensities(self.kernel, self._select_logloss(val_outputs, labels), self.shrinkage)
            self.kde_.fit(val_outputs, labels, self.quantifier_classes_)
            bandwidths = np.asarray([kde.bandwidth_ for kde in self.kde_.mix_densities], dtype=float)
        elif self.selects_bandwidth:
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

    def logloss_summaries(self) -> dict:
        """The ``LOGLOSS_SUMMARIES`` of the selection of ``bandwidth="logloss"``."""
        if not self.selects_logloss:
            raise RuntimeError('logloss_summaries exist only with bandwidth="logloss"')
        self._check_is_fitted()
        return {
            "logloss_bandwidth": self.logloss_bandwidth_,
            "logloss_curve": self.logloss_curve_.copy(),
            "logloss_selected": self.logloss_selected_,
            "logloss_default": self.logloss_default_,
            "logloss_edge": self.logloss_edge_,
            "logloss_classes": self.logloss_classes_,
            "loo_ece_default": self.loo_ece_default_,
            "loo_ece_selected": self.loo_ece_selected_,
        }

    def _model(self, outputs: np.ndarray) -> tuple[Callable, dict, dict | None]:
        """The NUTS model of BayesKDEy, its arguments and the initial parameters, as QuaPy's ``sample_from_posterior``."""
        alpha = _resolve_dirichlet_prior(self.prior, outputs.shape[1])
        args = {"test_log_densities": self.kde_.log_densities(outputs), "alpha": alpha}
        # kept for the per-instance pushforward of BayCAPKDEy (E3.4), which reuses the log-densities of the bag
        self._last_log_densities = args["test_log_densities"]
        return kdey_model, args | {"temperature": self.temperature}, None

    def last_bag_log_densities(self) -> np.ndarray:
        r"""The log-densities of the classes at the outputs of the last bag sampled (:meth:`sample_prevalences`), the
        array that the likelihood of the bag reads.

        :return: ``[j, n]``$=\ln\hat p_j(z_n)$, shape ``(K, N)``, rows in the order of ``val.classes_``.
        """
        if getattr(self, "_last_log_densities", None) is None:
            raise RuntimeError("No bag sampled yet: call sample_prevalences first")
        return np.asarray(self._last_log_densities, dtype=float)[self.class_order_]

    def _prevalence_draws(self, grouped: dict) -> np.ndarray:
        return np.asarray(jax.device_get(grouped[PREV]))


__all__ = [
    "EM_MAX_ITER",
    "EM_TOL",
    "KDEYML_EPSILON",
    "KDEYML_LOG_CLASSES",
    "LOGLOSS",
    "LOGLOSS_MIN_CLASS_SIZE",
    "LOGLOSS_SUMMARIES",
    "LOO",
    "LOO_DEFAULT_BANDWIDTH",
    "LOO_GRID",
    "LOO_SUMMARIES",
    "BayesKDEy",
    "bag_log_likelihoods",
    "class_logloss_curve",
    "em_prevalences",
    "kdey_model",
    "loo_class_log_probabilities",
    "loo_confidence_ece",
    "loo_log_likelihood",
    "select_from_curve",
    "simplex_log_density",
    "slsqp_prevalences",
]
