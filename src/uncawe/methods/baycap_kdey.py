r"""BayCAP-KDEy: BayCAP with the posterior of BayesKDEy, with a fixed rate matrix $\hat C$ (E3.1) or a random $C$ (E3.2).

``c_prior="fixed"`` gives the methods ``BayCAP-KDEy-<s>`` and ``BayCAP-KDEy-<s>-real`` of E3.1 of
``docs/ext3_posterior_propagation.md`` (work-plan card WP-8): the posterior draws of $\pi$ of BayesKDEy are pushed
through the rate matrix $\hat C$ of $h$ on $V$. Work-plan card WP-13 moved them from
``methods/baycap_fixed.py:BayCAPFixed`` (``quantifier="kdey"``) to this class, with the same draws
(``tests/test_baycap_split.py``). The NUTS model of BayesKDEy is copied from QuaPy's ``quapy.method._bayesian``
(``BayesianKDEy``, commit ``31f36f6`` of branch ``devel``), since QuaPy builds the class only when ``pystan`` is
installed (decision of the user, 2026-11-03); the copy gives the draws of QuaPy's ``sample_from_posterior``
(``tests/test_baycap_fixed.py``). The kernel density estimates are QuaPy's (``quapy.method._kdey.KDEBase``).

Work-plan card WP-25 moved the quantifier into ``methods/bayes_kdey.py:BayesKDEy``, which the class builds from its own
hyperparameters (``quantifier_``), with the same draws (``tests/test_wp25_restructure.py``), and added
``bandwidth="loo"`` (E7.1 of ``docs/ext7_kdey_bandwidth.md``), with ``c_prior="fixed"`` only: one bandwidth per class,
selected by leave-one-out likelihood on $V$.

Work-plan card WP-26 (E7.2 of ``docs/ext7_kdey_bandwidth.md``, §T.7–§T.9) adds ``bandwidth="cal"``, with
``c_prior="fixed"`` only: one bandwidth shared by the classes, selected per pair inside ``fit`` from $V$ alone, on
calibration bags drawn from half of $V$, by the accuracy error that the quantifier causes
(:meth:`BayCAPKDEy.select_bandwidth`); the deployed method is then fitted on the full $V$ at the selected bandwidth
exactly as with a numeric one. The estimate of $\pi$ on each calibration bag is the maximum-likelihood one, by EM
(``cal_solver="em"``, the default and the pilot of E7.2a) or by QuaPy's ``KDEyML`` (``cal_solver="slsqp"``, SLSQP);
:meth:`BayCAPKDEy.solver_check` computes both on the same calibration bags, with their times, for the gate of the
solver of E7.2a (``experiments/kdey_cal_solver.py``).

``c_prior="hierarchical-f"`` gives the methods ``BayCAP-KDEy-Hf-<s>-real`` of E3.2 (work-plan card WP-13, §T.10–§T.13):
each class of $V$ is split into cells by the prediction of $h$, the class density of BayesKDEy becomes the mixture
$f_j=\sum_kC_{kj}\hat\phi_{kj}$ of the kernel densities of its cells, and $(\pi,C)$ are sampled jointly by NUTS, with the
Hf prior on $C$ of ``methods/baycap.py`` (centred shares) and the validation counts observed as multinomials, as in
``BayCAP``. This model is the project's own, not a copy of QuaPy's.
"""

from collections.abc import Callable
from numbers import Number
from time import perf_counter
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
from cap.models.confidence import P_C_COND_Y, P_TEST_Y
from jax.scipy.special import logsumexp
from quapy.data import LabelledCollection
from quapy.method._bayesian import _resolve_dirichlet_prior
from quapy.protocol import UPP

from uncawe.methods.baycap import (
    H_KAPPA,
    H_M,
    H_NU,
    H_RECALLS,
    H_RHO,
    H_RHO_KAPPA,
    HIERARCHICAL_SUMMARIES,
    KAPPA_SUMMARIES,
    failed_summaries,
    hierarchical_rate_sites,
    hierarchical_summaries,
    realised_tables,
)
from uncawe.methods.baycap_quantifier import QuantifierBayCAP, _validate_prevalence_samples
from uncawe.methods.bayes_kdey import (
    LOO,
    LOO_DEFAULT_BANDWIDTH,
    LOO_GRID,
    LOO_SUMMARIES,
    BayesKDEy,
    bag_log_likelihoods,
    em_prevalences,
    kdey_model,
    slsqp_prevalences,
)
from uncawe.methods.bayesian_quantifier import PREV, _validate_outputs

# the prior on the columns of C: "fixed" holds C at the rates of h on V (E3.1), "hierarchical-f" samples C with the Hf
# prior of BayCAP (E3.2)
KDEY_C_PRIORS = ("fixed", "hierarchical-f")
# source of the density of a cell (§T.12): its own validation outputs; those of all the validation instances that h
# predicts as the class of the cell (the cell is empty); those of the true class of the cell (h never predicts the
# class of the cell on V)
CELL_OWN, CELL_PREDICTED, CELL_CLASS = 0, 1, 2
# per-bag summaries of the cells added by "hierarchical-f" (E3.2): the posterior mean and standard deviation of C, in
# the cap-ml layout (true classes on rows), flattened row by row; the label-free mismatch ||n^t/N^Q - mean(C) mean(pi)||_1
# (§T.8); the numbers of empty cells of V and, among them, of cells that take the density of their true class; the
# top-label ECE of the input classifier on its validation outputs
CELL_SUMMARIES = ("c_mean", "c_sd", "mismatch", "n_empty_cells", "n_class_density_cells", "val_ece")
KDEY_HF_SUMMARIES = HIERARCHICAL_SUMMARIES + KAPPA_SUMMARIES + CELL_SUMMARIES

# the selection of one shared bandwidth on calibration bags drawn from half of V (E7.2): the number and the size of the
# calibration bags, and the reference bandwidth (the stored one), towards which ties are broken and at which the
# criterion is also reported
CAL = "cal"
CAL_BAGS = 50
CAL_BAG_SIZE = 1000
CAL_REFERENCE = LOO_DEFAULT_BANDWIDTH
# the streams of the seeds of the calibration, derived from random_state (calibration_seed)
CAL_SPLIT_STREAM, CAL_BAG_STREAM = 0, 1
# the solvers of the estimate of pi on a calibration bag: EM (em_prevalences) or QuaPy's KDEyML (slsqp_prevalences)
CAL_SOLVERS = ("em", "slsqp")
# per-bag summaries added by bandwidth="cal" (the selection is fixed by the fit, so the same for every bag): the selected
# bandwidth b_cal, the criterion over bandwidth_grid (a vector), its values at b_cal and at CAL_REFERENCE, and the
# number of (bandwidth, calibration bag) cells whose estimate did not converge (EM: the step cap reached; SLSQP: no
# success; the name keeps that of the pilot), the sizes of V and of V_cal; with
# cal_nuts_check, also the
# bandwidth selected and the criterion curve computed with the NUTS posterior mean in place of the EM estimate
CAL_SUMMARIES = (
    "cal_bandwidth",
    "cal_curve",
    "cal_selected",
    "cal_default",
    "cal_em_unconverged",
    "cal_n_val",
    "cal_n_cal",
)
CAL_NUTS_SUMMARIES = ("cal_bandwidth_nuts", "cal_curve_nuts")


def calibration_seed(random_state: int, stream: int) -> int:
    """A 32-bit seed of the calibration of ``bandwidth="cal"``, from ``random_state`` and the stream (``CAL_*_STREAM``)
    only, so that the split and the bags do not depend on the classifier $h$ or on any global random state."""
    return int(np.random.SeedSequence([int(random_state), int(stream)]).generate_state(1)[0])


def split_halves(labels: np.ndarray, classes: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Indices into $V$ of $V_{\text{fit}}$ and $V_{\text{cal}}$, each in increasing order (§T.9).

    The instances of each class are permuted by one generator seeded by ``seed`` (class by class, in the order of
    ``classes``); the first $\lfloor n_j/2\rfloor$ go to $V_{\text{cal}}$ and the others to $V_{\text{fit}}$, as the
    split of E5.3 (``experiments/tau_select.py:split_index``). Raises ``ValueError`` if a class has fewer than $2$
    instances.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    fit, cal = [], []
    for class_ in classes:
        members = np.flatnonzero(labels == class_)
        if len(members) < 2:
            raise ValueError(
                f"every class needs at least 2 instances in V to be split; class {class_} has {len(members)}"
            )
        permuted = rng.permutation(members)
        cal.append(permuted[: len(members) // 2])
        fit.append(permuted[len(members) // 2 :])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(cal))


def calibration_bags(
    labels: np.ndarray, classes: np.ndarray, n_bags: int, size: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""The calibration bags, drawn from a labelled pool (here $V_{\text{cal}}$) by QuaPy's UPP, as ``main`` draws its
    target bags: prevalences uniform on the simplex ($\operatorname{Dir}(\mathbf 1)$), instances drawn with
    replacement within each class, ``random_state=seed``.

    :return: the indices into the pool of the instances of each bag, shape ``(n_bags, size)``, and the true prevalence
        of each bag (its class proportions), shape ``(n_bags, K)``, columns in the order of ``classes``.
    """
    labels = np.asarray(labels)
    pool = LabelledCollection(np.arange(len(labels)).reshape(-1, 1), labels, classes=classes)
    protocol = UPP(pool, sample_size=size, repeats=n_bags, random_state=seed, return_type="labelled_collection")
    # QuaPy's temp_seed restores the global NumPy state but writes the seed into qp.environ["_R_SEED"], the experiment
    # seed, which must stay as it is
    had_seed, experiment_seed = "_R_SEED" in qp.environ, qp.environ.get("_R_SEED")
    try:
        samples = list(protocol())
    finally:
        if had_seed:
            qp.environ["_R_SEED"] = experiment_seed
        else:
            qp.environ.pop("_R_SEED", None)
    index = np.stack([np.asarray(sample.X).ravel().astype(int) for sample in samples])
    prevalences = np.stack([np.asarray(sample.prevalence(), dtype=float) for sample in samples])
    return index, prevalences


def class_recalls(labels: np.ndarray, predicted: np.ndarray, classes: np.ndarray) -> np.ndarray:
    r"""The recall $\hat d_j=\hat C_{jj}$ of $h$ on each class, the share of its instances predicted as itself."""
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    return np.asarray([np.mean(predicted[labels == class_] == class_) for class_ in classes], dtype=float)


def accuracy_criterion(estimates: np.ndarray, prevalences: np.ndarray, recalls: np.ndarray) -> np.ndarray:
    r"""The calibration criterion of §T.7, $\frac1S\sum_s\lvert\hat d^\top(\hat\pi_s-\pi_s)\rvert$: the mean over
    the bags of the accuracy error that the quantifier causes if the recalls of $h$ are $\hat d$.

    :param estimates: $\hat\pi_s$, shape ``(..., S, K)`` (e.g. one row of bags per bandwidth).
    :param prevalences: the true prevalences $\pi_s$ of the bags, shape ``(S, K)``.
    :param recalls: $\hat d$, shape ``(K,)``.
    :return: the criterion, shape ``(...)``.
    """
    errors = (np.asarray(estimates, dtype=float) - np.asarray(prevalences, dtype=float)) @ np.asarray(recalls, float)
    return np.abs(errors).mean(axis=-1)


def select_from_curve(grid, curve: np.ndarray, reference: float = CAL_REFERENCE) -> int:
    r"""The index of the smallest value of ``curve`` over ``grid``; among values equal up to rounding (relative
    difference $10^{-9}$), the bandwidth closest to ``reference`` on the $\log_{10}$ scale, and the smaller of two at the
    same distance."""
    grid, curve = np.asarray(grid, dtype=float), np.asarray(curve, dtype=float)
    if not np.isfinite(curve).any():
        raise ValueError("the criterion is not finite at any bandwidth")
    best = np.nanmin(curve)
    ties = np.flatnonzero(np.isclose(curve, best, rtol=1e-9, atol=0.0))
    return int(ties[np.argmin(np.abs(np.log10(grid[ties]) - np.log10(reference)))])


def kdey_hf_loglik(prev, rates, log_cells, pred=None):
    r"""The log-likelihood of the target bag of E3.2 (§T.10), for one $(\pi,C)$.

    With ``pred=None`` (input classifier ``h``), the likelihood of the outputs alone,
    $\sum_u\ln\sum_j\pi_j\sum_kC_{kj}\hat\phi_{kj}(z_u)$, with ``log_cells[u, j, k]``$=\ln\hat\phi_{kj}(z_u)$. With
    ``pred`` the predictions $\hat y_u$ of $h$ on the bag (input classifier ``mlp``), the joint likelihood of the outputs
    and the predictions, $\sum_u\ln\sum_j\pi_jC_{\hat y_uj}\hat\phi_{\hat y_uj}(z_u)$, with
    ``log_cells[u, j]``$=\ln\hat\phi_{\hat y_uj}(z_u)$. Both are computed by log-sum-exp from the logarithms.

    :param prev: $\pi$, shape ``(K,)``.
    :param rates: $C$ in the ``cap-ml`` layout, ``rates[j, k]``$=C_{kj}$.
    :param log_cells: the log-densities of the cells at the target outputs, shape ``(N, K, K)`` or ``(N, K)``.
    :param pred: the indices of the predictions of $h$ on the bag, shape ``(N,)``, or ``None``.
    """
    # log_w[j, k] = ln(pi_j C_kj)
    log_w = jnp.log(prev)[:, None] + jnp.log(rates)
    if pred is None:
        return jnp.sum(logsumexp(log_w[None, :, :] + log_cells, axis=(1, 2)))
    return jnp.sum(logsumexp(log_w[:, pred].T + log_cells, axis=1))


def kdey_hf_model(log_cells, val_counts, alpha, pred=None, temperature: float = 1.0) -> None:
    r"""The NUTS model of E3.2 (§T.10, §T.11): BayesKDEy with a random $C$ and the Hf prior.

    $\pi\sim\operatorname{Dir}(\alpha)$; $C$ from the Hf prior with centred shares
    (:func:`uncawe.methods.baycap.hierarchical_rate_sites`, the sites of ``BayCAP`` with ``c_prior="hierarchical-f"``);
    the validation counts of each true class $\operatorname{Mult}(n^s_j,C_{\cdot j})$, observed, as in ``BayCAP.model``;
    the log-likelihood of the bag (:func:`kdey_hf_loglik`) divided by the temperature $\tau$.

    :param log_cells: see :func:`kdey_hf_loglik`.
    :param val_counts: the validation counts in the ``cap-ml`` layout, ``val_counts[j, k]``$=f_{kj}$.
    :param alpha: the Dirichlet parameter of the prior on $\pi$.
    :param pred: see :func:`kdey_hf_loglik`.
    :param temperature: $\tau$.
    """
    n_classes = val_counts.shape[0]
    prev = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.asarray(alpha)))
    p_c_cond_y = hierarchical_rate_sites(n_classes, light=True, shares="centred")
    with numpyro.plate("plate", n_classes):
        numpyro.sample("F_yc", dist.Multinomial(val_counts.sum(axis=1), p_c_cond_y), obs=val_counts)
    numpyro.factor("loglik", (1.0 / temperature) * kdey_hf_loglik(prev, p_c_cond_y, log_cells, pred))


class BayCAPKDEy(QuantifierBayCAP):
    r"""BayCAP with the posterior of BayesKDEy, with a fixed $\hat C$ (E3.1) or a random $C$ with the Hf prior (E3.2).

    BayesKDEy estimates the densities of the outputs of the input classifier within each class from $V$ by kernel
    density estimation (``kernel``, ``bandwidth``, ``shrinkage``) and puts a Dirichlet prior on $\pi$. The input
    classifier, the estimands, the seeds and the NUTS diagnostics are those of
    :class:`uncawe.methods.baycap_quantifier.QuantifierBayCAP`.

    - ``c_prior="fixed"`` (E3.1): NUTS samples $\pi$ alone (:func:`kdey_model`, through ``quantifier_``, a
      :class:`uncawe.methods.bayes_kdey.BayesKDEy`), and each draw is pushed through the rate matrix $\hat C$ of $h$ on
      $V$. After each ``predict_ct_range`` call, :meth:`last_posterior_summaries` returns ``FIXED_SUMMARIES``, and, with
      ``bandwidth="loo"``, the ``LOO_SUMMARIES`` of the selection too.
    - ``bandwidth="cal"`` (E7.2, ``c_prior="fixed"``, Gaussian kernel and no shrinkage only): ``fit`` first selects
      one bandwidth shared by the classes from $V$ alone (:meth:`select_bandwidth`, §T.7–§T.9), then fits the method on
      the full $V$ at the selected bandwidth ``cal_bandwidth_`` exactly as with that numeric bandwidth: the selection
      draws its randomness from its own seeds (:func:`calibration_seed`) and consumes no random state of the deployed
      fit, so a selected $0.1$ gives the draws of ``bandwidth=0.1``. The per-bag summaries add ``CAL_SUMMARIES`` (and
      ``CAL_NUTS_SUMMARIES`` with ``cal_nuts_check``).
    - ``c_prior="hierarchical-f"`` (E3.2, §T.10–§T.13): ``fit`` splits each class $\omega_j$ of $V$ into the cells
      $V_{kj}$ of the instances that $h$ predicts as $\omega_k$ and fits one kernel density $\hat\phi_{kj}$ per cell on
      the validation outputs of the input classifier (cross-fitted for ``mlp``). An empty cell takes the density of all
      the validation instances that $h$ predicts as $\omega_k$, or, if $h$ never predicts $\omega_k$ on $V$, the class
      density $\hat f_j$ (§T.12). NUTS samples $(\pi,C)$ jointly (:func:`kdey_hf_model`): $C$ has the Hf prior of
      ``BayCAP`` with centred shares and the validation counts as data, and the bag enters through
      :func:`kdey_hf_loglik`, with the outputs alone for ``h`` and the outputs and the predictions of $h$ for ``mlp``.
      Each draw becomes the table $C^{(r)}\operatorname{diag}(\pi^{(r)})$, then the estimand applies. The NUTS
      diagnostics cover $\pi$, $C$, $m$, $\rho$ and $1/(1+\kappa)$; :meth:`last_posterior_summaries` returns
      ``KDEY_HF_SUMMARIES``. If NUTS cannot start, the bag gets tables of NaN, NaN diagnostics and ``nuts_failed``
      equal to $1$, as in ``BayesCAP-Hf-real``.

    :param acc_fn: a ``cap-ml`` contingency-table performance function.
    :param input_classifier: ``"h"`` or ``"mlp"``.
    :param estimand: ``"population"`` or ``"realised"``.
    :param c_prior: ``"fixed"`` or ``"hierarchical-f"``.
    :param kernel: kernel of BayesKDEy (``"gaussian"``, ``"aitchison"`` or ``"ilr"``).
    :param bandwidth: bandwidth of BayesKDEy, a positive number, ``"scott"``, ``"silverman"``, ``"loo"`` (per-class
        bandwidths selected by leave-one-out likelihood) or ``"cal"`` (one bandwidth selected on calibration bags);
        ``"loo"`` and ``"cal"`` need the Gaussian kernel, no shrinkage and ``c_prior="fixed"``.
    :param bandwidth_grid: the grid of ``"loo"`` and ``"cal"``; ``bayes_kdey.LOO_GRID`` if ``None``.
    :param shrinkage: shrinkage of BayesKDEy, in $[0,1)$, $0$ with the Gaussian kernel.
    :param num_warmup: number of NUTS warm-up iterations.
    :param num_samples: number of posterior draws retained after warm-up.
    :param temperature: temperature $\tau$ of the likelihood of the bag.
    :param prior: Dirichlet parameter of the prior on $\pi$ (``"uniform"``, a positive scalar or a vector).
    :param random_state: seed of NUTS, of the MLP and of the multinomial draws, and the root of the seeds of the
        calibration of ``"cal"``; ``qp.environ["_R_SEED"]`` if ``None``.
    :param cal_bags: the number $S$ of calibration bags of ``"cal"``.
    :param cal_bag_size: the size of a calibration bag of ``"cal"``.
    :param cal_nuts_check: with ``"cal"``, also compute the criterion with the NUTS posterior mean in place of the EM
        estimate, for every bandwidth and calibration bag (the pilot check of E7.2, §T.8); it does not change the
        selection.
    :param cal_solver: with ``"cal"``, the maximum-likelihood estimate on a calibration bag: ``"em"``
        (:func:`uncawe.methods.bayes_kdey.em_prevalences`) or ``"slsqp"``, QuaPy's ``KDEyML``
        (:func:`uncawe.methods.bayes_kdey.slsqp_prevalences`).
    """

    def __init__(
        self,
        acc_fn: Callable,
        input_classifier: Literal["h", "mlp"] = "mlp",
        estimand: Literal["population", "realised"] = "population",
        c_prior: Literal["fixed", "hierarchical-f"] = "fixed",
        kernel: Literal["gaussian", "aitchison", "ilr"] = "gaussian",
        bandwidth: float | Literal["scott", "silverman", "loo", "cal"] = 0.1,
        bandwidth_grid=None,
        shrinkage: float = 0.0,
        num_warmup: int = 500,
        num_samples: int = 1000,
        temperature: float = 1.0,
        prior: str | Number | np.ndarray = "uniform",
        random_state: int | None = None,
        cal_bags: int = CAL_BAGS,
        cal_bag_size: int = CAL_BAG_SIZE,
        cal_nuts_check: bool = False,
        cal_solver: Literal["em", "slsqp"] = "em",
    ) -> None:
        if c_prior not in KDEY_C_PRIORS:
            raise ValueError(f"c_prior must be one of {KDEY_C_PRIORS}; found {c_prior!r}")
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
        if c_prior != "fixed" and bandwidth in (LOO, CAL):
            raise ValueError(f'bandwidth="{bandwidth}" is implemented only with c_prior="fixed"')
        self.c_prior = c_prior
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.bandwidth_grid = bandwidth_grid
        self.shrinkage = shrinkage
        self.cal_bags = int(cal_bags)
        self.cal_bag_size = int(cal_bag_size)
        self.cal_nuts_check = bool(cal_nuts_check)
        if cal_solver not in CAL_SOLVERS:
            raise ValueError(f"cal_solver must be one of {CAL_SOLVERS}; found {cal_solver!r}")
        self.cal_solver = cal_solver
        if self.calibrates_bandwidth:
            if kernel != "gaussian" or shrinkage != 0:
                raise ValueError('bandwidth="cal" needs the Gaussian kernel and no shrinkage')
            grid = np.asarray(self._cal_grid(), dtype=float)
            if grid.ndim != 1 or len(grid) == 0 or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
                raise ValueError("bandwidth_grid must be a non-empty increasing sequence of positive numbers")
            if self.cal_bags < 1 or self.cal_bag_size < 1:
                raise ValueError("cal_bags and cal_bag_size must be positive")
        # checks the hyperparameters of the quantifier
        self._make_quantifier()

    @property
    def calibrates_bandwidth(self) -> bool:
        """Whether the bandwidth is selected on calibration bags (``bandwidth="cal"``)."""
        return isinstance(self.bandwidth, str) and self.bandwidth == CAL

    def _cal_grid(self) -> tuple:
        return LOO_GRID if self.bandwidth_grid is None else tuple(self.bandwidth_grid)

    def _make_quantifier(self) -> BayesKDEy:
        if self.calibrates_bandwidth:
            # the deployed quantifier: BayesKDEy at the selected bandwidth (the reference before the selection, to check
            # the hyperparameters), built exactly as with a numeric bandwidth
            bandwidth = getattr(self, "cal_bandwidth_", CAL_REFERENCE)
            return BayesKDEy(
                kernel=self.kernel, bandwidth=bandwidth, shrinkage=self.shrinkage, **self._quantifier_settings()
            )
        return BayesKDEy(
            kernel=self.kernel,
            bandwidth=self.bandwidth,
            bandwidth_grid=self.bandwidth_grid,
            shrinkage=self.shrinkage,
            **self._quantifier_settings(),
        )

    def fit(self, val: LabelledCollection, posteriors: np.ndarray):
        """Measure the rates of $h$ on $V$ and fit the quantifier; with ``bandwidth="cal"``, select the bandwidth first
        (:meth:`select_bandwidth`)."""
        if self.calibrates_bandwidth:
            self.select_bandwidth(val, posteriors)
        return super().fit(val, posteriors)

    def select_bandwidth(self, val: LabelledCollection, posteriors: np.ndarray) -> float:
        r"""Select one bandwidth shared by the classes on calibration bags drawn from half of $V$ (E7.2, §T.7–§T.9).

        1. $V$ is split per class into $V_{\text{fit}}$ and $V_{\text{cal}}$, half each (:func:`split_halves`, seed
           ``calibration_seed(random_state, CAL_SPLIT_STREAM)``).
        2. On $V_{\text{fit}}$: BayesKDEy with the input classifier of the method (for ``mlp``, the MLP with its
           cross-fitted outputs, then trained on all of $V_{\text{fit}}$) and the recalls $\hat d^{\text{fit}}$ of $h$.
        3. ``cal_bags`` bags of ``cal_bag_size`` instances drawn from $V_{\text{cal}}$ by UPP
           (:func:`calibration_bags`, seed ``calibration_seed(random_state, CAL_BAG_STREAM)``), the same for every
           bandwidth, each with its true prevalence $\pi_s$ and the outputs of the input classifier of $V_{\text{fit}}$.
        4. For each $b$ of the grid: the class densities on $V_{\text{fit}}$ at $b$, their log-densities at the outputs
           of $V_{\text{cal}}$ (once per $b$), the maximum-likelihood estimate $\hat\pi_s(b)$ of each bag and
           the criterion $\frac1S\sum_s\lvert\hat d^{\text{fit}\top}(\hat\pi_s(b)-\pi_s)\rvert$
           (:func:`accuracy_criterion`); the criterion is also computed at ``CAL_REFERENCE`` if it is not in the grid.
        5. $b_{\text{cal}}$ is the bandwidth of smallest criterion, ties broken towards ``CAL_REFERENCE``
           (:func:`select_from_curve`).

        The estimate of step 4 is that of ``cal_solver`` (EM or QuaPy's ``KDEyML``). Only the labels of $V$ are used.
        With ``cal_nuts_check``, step 4 is repeated with the posterior mean of a NUTS
        run of the model of BayesKDEy (prior, temperature and NUTS settings of the method) in place of the EM estimate;
        the selection still uses the EM estimate.

        :return: $b_{\text{cal}}$, also stored in ``cal_bandwidth_``.
        """
        setup = self._calibration_setup(val, posteriors)
        half, cal_outputs, bag_index, counts = setup["half"], setup["cal_outputs"], setup["bag_index"], setup["counts"]
        prevalences, recalls, n_classes = setup["prevalences"], setup["recalls"], len(setup["recalls"])

        grid = np.asarray(self._cal_grid(), dtype=float)
        with_reference = self._with_reference(grid)
        estimates = np.empty((len(with_reference), len(bag_index), n_classes))
        nuts_means = np.empty_like(estimates) if self.cal_nuts_check else None
        unconverged = 0
        alpha = _resolve_dirichlet_prior(self.prior, n_classes)
        for g, bandwidth in enumerate(with_reference):
            log_densities = half.class_log_densities(cal_outputs, float(bandwidth))
            estimates[g], converged = self._solve(self.cal_solver, log_densities, counts, bag_index)
            unconverged += int(np.sum(~converged))
            if nuts_means is not None:
                for s, index in enumerate(bag_index):
                    args = {
                        "test_log_densities": log_densities[:, index],
                        "alpha": alpha,
                        "temperature": self.temperature,
                    }
                    mcmc = half.run_nuts(kdey_model, args, None)
                    nuts_means[g, s] = np.asarray(jax.device_get(mcmc.get_samples()[PREV])).mean(axis=0)

        curve = accuracy_criterion(estimates, prevalences, recalls)
        reference = int(np.flatnonzero(np.isclose(with_reference, CAL_REFERENCE))[0])
        best = select_from_curve(grid, curve[: len(grid)])
        self.cal_grid_ = grid
        self.cal_split_ = (setup["fit_index"], setup["cal_index"])
        self.cal_bag_index_ = bag_index
        self.cal_prevalences_ = prevalences
        self.cal_recalls_ = recalls
        self.cal_estimates_ = estimates[: len(grid)]
        self.cal_curve_ = curve[: len(grid)]
        self.cal_selected_ = float(curve[best])
        self.cal_default_ = float(curve[reference])
        self.cal_em_unconverged_ = unconverged
        self.cal_bandwidth_ = float(grid[best])
        if nuts_means is not None:
            curve_nuts = accuracy_criterion(nuts_means, prevalences, recalls)
            self.cal_nuts_means_ = nuts_means[: len(grid)]
            self.cal_curve_nuts_ = curve_nuts[: len(grid)]
            self.cal_bandwidth_nuts_ = float(grid[select_from_curve(grid, self.cal_curve_nuts_)])
        return self.cal_bandwidth_

    def _calibration_setup(self, val: LabelledCollection, posteriors: np.ndarray) -> dict:
        r"""Steps 1–3 of :meth:`select_bandwidth`: the split, BayesKDEy and the recalls on $V_{\text{fit}}$, the
        calibration bags with their outputs and true prevalences, and the multiplicity of each instance of
        $V_{\text{cal}}$ in each bag."""
        posteriors = np.asarray(posteriors, dtype=float)
        classes = np.asarray(val.classes_)
        _validate_outputs(posteriors, len(val), len(classes))
        labels = np.asarray(val.y)
        predicted = classes[np.argmax(posteriors, axis=1)]
        fit_index, cal_index = split_halves(labels, classes, calibration_seed(self.random_state, CAL_SPLIT_STREAM))
        half = BayesKDEy(
            kernel=self.kernel, bandwidth=CAL_REFERENCE, shrinkage=self.shrinkage, **self._quantifier_settings()
        )
        half.fit(LabelledCollection(val.X[fit_index], labels[fit_index], classes=classes), posteriors[fit_index])
        cal_outputs = half.outputs(val.X[cal_index], posteriors[cal_index])
        recalls = class_recalls(labels[fit_index], predicted[fit_index], classes)
        bag_index, prevalences = calibration_bags(
            labels[cal_index],
            classes,
            self.cal_bags,
            self.cal_bag_size,
            calibration_seed(self.random_state, CAL_BAG_STREAM),
        )
        counts = np.stack([np.bincount(index, minlength=len(cal_index)) for index in bag_index])
        return {
            "fit_index": fit_index,
            "cal_index": cal_index,
            "half": half,
            "cal_outputs": cal_outputs,
            "recalls": recalls,
            "bag_index": bag_index,
            "prevalences": prevalences,
            "counts": counts,
        }

    @staticmethod
    def _with_reference(grid: np.ndarray) -> np.ndarray:
        """The grid, with ``CAL_REFERENCE`` appended if it is not in it."""
        return grid if np.any(np.isclose(grid, CAL_REFERENCE)) else np.append(grid, CAL_REFERENCE)

    @staticmethod
    def _solve(solver: str, log_densities: np.ndarray, counts: np.ndarray, bag_index: np.ndarray):
        r"""The estimates of $\pi$ on the calibration bags by ``solver``, and whether each converged."""
        if solver == "em":
            return em_prevalences(log_densities, counts)
        return slsqp_prevalences(log_densities, bag_index)

    def solver_check(self, val: LabelledCollection, posteriors: np.ndarray) -> dict:
        r"""The gate of the solver of E7.2a: the selection of :meth:`select_bandwidth` with both solvers, on the same
        calibration bags and class densities, and their times. It does not fit the method nor set ``cal_bandwidth_``.

        For each bandwidth of the grid (and ``CAL_REFERENCE``), the class log-densities at the outputs of
        $V_{\text{cal}}$ are computed once; then EM (:func:`em_prevalences`) and QuaPy's ``KDEyML``
        (:func:`slsqp_prevalences`) estimate $\pi$ on every bag, each timed by wall clock (``perf_counter``) without the
        densities, and the log-likelihood of each bag at both estimates is computed (:func:`bag_log_likelihoods`).

        :return: a dict with ``grid`` (the grid, shape ``(G,)``); ``prevalences`` (true, ``(S, K)``); ``recalls``
            (``(K,)``); per solver ``<solver>_estimates`` (``(G, S, K)``), ``<solver>_converged`` (``(G, S)``),
            ``<solver>_loglik`` (``(G, S)``), ``<solver>_seconds`` (``(G,)``), ``<solver>_curve`` (``(G,)``),
            ``<solver>_bandwidth`` ($b_{\text{cal}}$ of that solver), ``<solver>_default`` (criterion at
            ``CAL_REFERENCE``); ``densities_seconds`` (``(G,)``, the class densities, shared); ``setup_seconds`` (steps
            1–3 of :meth:`select_bandwidth`, shared); ``n_val``, ``n_cal``.
        """
        if not self.calibrates_bandwidth:
            raise RuntimeError('solver_check needs bandwidth="cal"')
        started = perf_counter()
        setup = self._calibration_setup(val, posteriors)
        setup_seconds = perf_counter() - started
        half, cal_outputs, bag_index, counts = setup["half"], setup["cal_outputs"], setup["bag_index"], setup["counts"]
        prevalences, recalls = setup["prevalences"], setup["recalls"]
        grid = np.asarray(self._cal_grid(), dtype=float)
        with_reference = self._with_reference(grid)
        shape = (len(with_reference), len(bag_index))
        out = {"densities_seconds": np.empty(len(with_reference))}
        for solver in CAL_SOLVERS:
            out |= {
                f"{solver}_estimates": np.empty((*shape, len(recalls))),
                f"{solver}_converged": np.empty(shape, dtype=bool),
                f"{solver}_loglik": np.empty(shape),
                f"{solver}_seconds": np.empty(len(with_reference)),
            }
        for g, bandwidth in enumerate(with_reference):
            started = perf_counter()
            log_densities = half.class_log_densities(cal_outputs, float(bandwidth))
            out["densities_seconds"][g] = perf_counter() - started
            for solver in CAL_SOLVERS:
                started = perf_counter()
                estimates, converged = self._solve(solver, log_densities, counts, bag_index)
                out[f"{solver}_seconds"][g] = perf_counter() - started
                out[f"{solver}_estimates"][g], out[f"{solver}_converged"][g] = estimates, converged
                out[f"{solver}_loglik"][g] = bag_log_likelihoods(log_densities, counts, estimates)

        reference = int(np.flatnonzero(np.isclose(with_reference, CAL_REFERENCE))[0])
        n = len(grid)
        result = {
            "grid": grid,
            "prevalences": prevalences,
            "recalls": recalls,
            "densities_seconds": out["densities_seconds"][:n],
            "setup_seconds": setup_seconds,
            "n_val": int(len(setup["fit_index"]) + len(setup["cal_index"])),
            "n_cal": len(setup["cal_index"]),
        }
        for solver in CAL_SOLVERS:
            curve = accuracy_criterion(out[f"{solver}_estimates"], prevalences, recalls)
            result |= {
                f"{solver}_estimates": out[f"{solver}_estimates"][:n],
                f"{solver}_converged": out[f"{solver}_converged"][:n],
                f"{solver}_loglik": out[f"{solver}_loglik"][:n],
                f"{solver}_seconds": out[f"{solver}_seconds"][:n],
                f"{solver}_curve": curve[:n],
                f"{solver}_bandwidth": float(grid[select_from_curve(grid, curve[:n])]),
                f"{solver}_default": float(curve[reference]),
            }
        return result

    def calibration_summaries(self) -> dict:
        """The ``CAL_SUMMARIES`` of the selection (and ``CAL_NUTS_SUMMARIES`` with ``cal_nuts_check``)."""
        if not hasattr(self, "cal_bandwidth_"):
            raise RuntimeError('calibration_summaries exist only after fitting with bandwidth="cal"')
        summaries = {
            "cal_bandwidth": self.cal_bandwidth_,
            "cal_curve": self.cal_curve_.copy(),
            "cal_selected": self.cal_selected_,
            "cal_default": self.cal_default_,
            "cal_em_unconverged": self.cal_em_unconverged_,
            "cal_n_val": int(len(self.cal_split_[0]) + len(self.cal_split_[1])),
            "cal_n_cal": len(self.cal_split_[1]),
        }
        if self.cal_nuts_check:
            summaries |= {"cal_bandwidth_nuts": self.cal_bandwidth_nuts_, "cal_curve_nuts": self.cal_curve_nuts_.copy()}
        return summaries

    @property
    def kde_(self):
        """The class densities of the quantifier."""
        return self.quantifier_.kde_

    @property
    def bandwidths_(self) -> np.ndarray:
        """The bandwidth of each class, in the order of ``classes_``."""
        return self.quantifier_.bandwidths_

    def _after_fit(self, labels: np.ndarray, predicted: np.ndarray) -> None:
        if self.c_prior == "hierarchical-f":
            self._fit_cells(self.val_outputs_, labels, predicted)

    def _fit_cells(self, val_outputs: np.ndarray, labels: np.ndarray, predicted: np.ndarray) -> None:
        r"""Fit the densities of the cells (§T.10) and choose those of the empty cells (§T.12).

        ``cell_source_[j, k]`` is ``CELL_OWN``, ``CELL_PREDICTED`` or ``CELL_CLASS``; ``_cell_key[j][k]`` names the
        density of cell $(k,j)$ in ``_densities``, where each distinct density is fitted once.
        """
        classes = self.classes_
        n_classes = len(classes)
        self.cell_source_ = np.full((n_classes, n_classes), CELL_OWN, dtype=int)
        self._densities, self._cell_key = {}, [[None] * n_classes for _ in range(n_classes)]
        for k in range(n_classes):
            predicted_k = predicted == classes[k]
            for j in range(n_classes):
                in_cell = predicted_k & (labels == classes[j])
                if in_cell.any():
                    key = ("cell", j, k)
                    if key not in self._densities:
                        self._densities[key] = self.kde_.density(val_outputs[in_cell])
                elif predicted_k.any():
                    key = ("predicted", k)
                    self.cell_source_[j, k] = CELL_PREDICTED
                    if key not in self._densities:
                        self._densities[key] = self.kde_.density(val_outputs[predicted_k])
                else:
                    key = ("class", j)
                    self.cell_source_[j, k] = CELL_CLASS
                    if key not in self._densities:
                        self._densities[key] = self.kde_.mix_densities[self._quantifier_to_rate_order[j]]
                self._cell_key[j][k] = key

    def cell_log_densities(self, outputs: np.ndarray, pred: np.ndarray | None = None) -> np.ndarray:
        r"""The log-densities of the cells at the outputs of a bag (§T.10), computed before sampling.

        :param outputs: the outputs of the input classifier on the bag, shape ``(N, K)``.
        :param pred: ``None`` for all cells, ``[u, j, k]``$=\ln\hat\phi_{kj}(z_u)$, shape ``(N, K, K)``; or the indices
            of the predictions of $h$ on the bag, for ``[u, j]``$=\ln\hat\phi_{\hat y_uj}(z_u)$, shape ``(N, K)``.
        """
        n_classes = len(self.classes_)
        if pred is None:
            cache = {}
            out = np.empty((len(outputs), n_classes, n_classes))
            for j in range(n_classes):
                for k in range(n_classes):
                    key = self._cell_key[j][k]
                    if key not in cache:
                        cache[key] = self.kde_.log_density(self._densities[key], outputs)
                    out[:, j, k] = cache[key]
            return out
        out = np.empty((len(outputs), n_classes))
        for k in np.unique(pred):
            rows = pred == k
            cache = {}
            for j in range(n_classes):
                key = self._cell_key[j][k]
                if key not in cache:
                    cache[key] = self.kde_.log_density(self._densities[key], outputs[rows])
                out[rows, j] = cache[key]
        return out

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        r"""Return one normalised contingency table per posterior draw (true classes on rows).

        ``posteriors`` are the outputs of $h$ on the bag: they give $n^t$, the predictions $\hat y_u$ and, with
        ``input_classifier="h"``, the outputs the quantifier reads; with ``"mlp"`` it reads the outputs of its MLP on
        ``X``.
        """
        if self.c_prior == "fixed":
            cts = super().predict_ct_range(X, posteriors)
            if self.quantifier_.selects_bandwidth:
                self._last_summaries |= self.quantifier_.selection_summaries()
            if self.calibrates_bandwidth:
                self._last_summaries |= self.calibration_summaries()
            return cts
        self._check_is_fitted()
        posteriors = np.asarray(posteriors, dtype=float)
        n_classes = len(self.classes_)
        pred = np.argmax(posteriors, axis=1)
        pred_counts = np.bincount(pred, minlength=n_classes)
        cell_pred = None if self.input_classifier == "h" else pred
        model_args = {
            "log_cells": self.cell_log_densities(self.outputs(X, posteriors), cell_pred),
            "val_counts": self.val_counts_,
            "alpha": _resolve_dirichlet_prior(self.prior, n_classes),
            "pred": cell_pred,
            "temperature": self.temperature,
        }
        try:
            mcmc = self.quantifier_.run_nuts(kdey_hf_model, model_args, None)
        except RuntimeError as error:
            # numpyro raises a RuntimeError when it finds no valid initial parameters
            if "initial parameters" not in str(error):
                raise
            self._last_nuts = "failed"
            self._last_summaries = failed_summaries(n_classes, with_kappa=True) | self._cell_summaries(None, None, None)
            return np.full((self.num_samples, n_classes, n_classes), np.nan)

        grouped = mcmc.get_samples(group_by_chain=True)
        sites = (P_TEST_Y, P_C_COND_Y, H_M, H_RHO) + ((H_RHO_KAPPA,) if n_classes > 2 else ())
        self._last_nuts = (
            {name: np.asarray(jax.device_get(grouped[name])) for name in sites},
            np.asarray(jax.device_get(mcmc.get_extra_fields()["diverging"])),
        )
        samples = {name: np.asarray(jax.device_get(value)) for name, value in mcmc.get_samples().items()}
        prevalences, rates = samples[P_TEST_Y].astype(float), samples[P_C_COND_Y].astype(float)
        _validate_prevalence_samples(prevalences, n_classes)
        summary_sites = (H_M, H_NU, H_RECALLS, P_TEST_Y) + ((H_KAPPA,) if H_KAPPA in samples else ())
        self._last_summaries = hierarchical_summaries(
            {name: samples[name] for name in summary_sites},
            float(self.val_counts_.sum(axis=1).min()),
            with_kappa=True,
        ) | self._cell_summaries(prevalences, rates, pred_counts)

        cts = prevalences[:, :, None] * rates
        if self.estimand == "realised":
            cts = realised_tables(cts, pred_counts, np.random.default_rng(self.random_state))
        return cts

    def _cell_summaries(self, prevalences, rates, pred_counts) -> dict:
        """The summaries ``CELL_SUMMARIES`` of a bag; NaN for the posterior ones when NUTS failed (``rates=None``)."""
        n_classes = len(self.classes_)
        counts = {
            "n_empty_cells": int(np.sum(self.val_counts_ == 0)),
            "n_class_density_cells": int(np.sum(self.cell_source_ == CELL_CLASS)),
            "val_ece": self.val_ece_,
        }
        if rates is None:
            nan = np.full(n_classes * n_classes, np.nan)
            return {"c_mean": nan, "c_sd": nan.copy(), "mismatch": np.nan} | counts
        c_mean = rates.mean(axis=0)
        shares = np.asarray(pred_counts, dtype=float) / np.sum(pred_counts)
        return {
            "c_mean": c_mean.ravel(),
            "c_sd": rates.std(axis=0).ravel(),
            # the predicted share of class k under the posterior means: sum_j mean(pi)_j mean(C)_kj
            "mismatch": float(np.abs(shares - prevalences.mean(axis=0) @ c_mean).sum()),
        } | counts


__all__ = [
    "CAL",
    "CAL_BAGS",
    "CAL_BAG_SIZE",
    "CAL_NUTS_SUMMARIES",
    "CAL_REFERENCE",
    "CAL_SUMMARIES",
    "CELL_CLASS",
    "CELL_OWN",
    "CELL_PREDICTED",
    "CELL_SUMMARIES",
    "KDEY_C_PRIORS",
    "KDEY_HF_SUMMARIES",
    "LOO_SUMMARIES",
    "BayCAPKDEy",
    "accuracy_criterion",
    "calibration_bags",
    "calibration_seed",
    "class_recalls",
    "kdey_hf_loglik",
    "kdey_hf_model",
    "kdey_model",
    "select_from_curve",
    "split_halves",
]
