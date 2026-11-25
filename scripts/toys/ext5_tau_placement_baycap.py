r"""Toy simulation for Extension 5: where the temperature $\tau$ of BayesCAP-real and BayesCAP-Hf-real acts, and what
that does to the selection of $\tau$ of E5.3 (the companion of ``ext5_tau_placement.py`` for BayCAP-KDEy-h-inst).

Binary problem, $K=2$, $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ with recalls $d_0=C_{00}$, $d_1=C_{11}$. The
classifier has true recalls $d^\star_0=a+0.04$, $d^\star_1=a-0.04$ ($a$ in ``ACCURACIES``). $V$ has $n_j$ instances per
class, each predicted correctly with probability $d^\star_j$; its counts $F_{jj}$ give
$\mathcal L_V=\prod_j\operatorname{Bin}(F_{jj};n_j,d_j)$. A bag $U$ of $N$ instances is seen through its predicted
counts $q_0,q_1$. Priors: $\pi\sim\operatorname{Dir}(1,1)$; on the recalls either ``uniform``
($d_j\sim\operatorname{Beta}(1,1)$, BayesCAP-real) or ``hf`` (the Hf prior of ``methods/baycap.py:
hierarchical_rate_sites`` for $K=2$: $m\sim U(0,1)$, $\rho=1/(1+\nu)\sim U(0,1)$, $d_j\mid m,\nu\sim
\operatorname{Beta}(\nu m,\nu(1-m))$, integrated on a grid of $(m,\rho)$; BayesCAP-Hf-real). The priors and
$\mathcal L_V^{1/\tau}$ are the same in every variant; the three ways of putting $\tau$ in the term of $U$, with the
per-instance complete-data likelihood $\prod_n C_{\hat y_nY_n}\pi_{Y_n}$:

- ``c`` (the implemented method): $\mathcal L_U^{1/\tau}=\prod_i(\sum_jC_{ij}\pi_j)^{q_i/\tau}$, untempered realised
  step $n_{i\cdot}\sim\operatorname{Mult}(q_i,\rho_{i\cdot})$, $\rho_{ij}\propto C_{ij}\pi_j$
  (``methods/baycap.py:realised_tables``);
- ``a`` (whole complete-data likelihood tempered): $\prod_i[\sum_j(C_{ij}\pi_j)^{1/\tau}]^{q_i}$,
  $\rho_{ij}\propto(C_{ij}\pi_j)^{1/\tau}$;
- ``b`` (the term of the prediction tempered only): $\prod_i[\sum_jC_{ij}^{1/\tau}\pi_j]^{q_i}$,
  $\rho_{ij}\propto C_{ij}^{1/\tau}\pi_j$.

At $\tau=1$ the three coincide. The posterior of $(\pi_1,d_0,d_1)$ is computed exactly on a grid (no MCMC; the grid of
each recall follows its posterior under the uniform prior and $\mathcal L_V^{1/\tau}$ at each $\tau$); $R$ draws are taken from it, a realised
table per draw, and the interval of the accuracy of the realised table is the central 95% interval of the draws
(``methods/baycap_kdey.py:interval_scores``), scored against the realised accuracy of the bag. Every variant targets
the realised table.

The selection follows E5.3, as in ``ext5_tau_placement.py``: $V$ split per class into $V_{\text{fit}}$ and
$V_{\text{cal}}$, $50$ calibration bags of $1000$ from $V_{\text{cal}}$ by UPP, the model on the counts of
$V_{\text{fit}}$, $\tau_{\text{cal}}$ the lowest mean Winkler score over ``TAU_GRID`` with ties towards $1$; the test
bags use the counts of the full $V$. Test bags: $N=1000$, $\pi_1\sim U(0,1)$, $n_1=\operatorname{round}(N\pi_1)$, each
instance predicted correctly with probability $d^\star_j+\delta$ ($\delta$ in ``SHIFTS``: the recalls on the target
differ from those of $V$ in the misspecified scenarios, unseen by $V$ and by the calibration bags). The oracle $\tau$
is the one with the lowest mean Winkler score on the test bags of the replicate (a reference, it uses the test labels).
Every replicate has its own generator, seeded by ``(--seed, scenario, replicate)``. Toy results, not experiment
results.

Output: per (scenario, replicate, prior, variant, $\tau$) the mean test coverage, width, signed error of the centre and
Winkler score, the calibration Winkler score, $\tau_{\text{cal}}$ and $\tau_{\text{oracle}}$, in
``<OUT_DIR>/uncawe/toys/ext5_tau_placement_baycap.csv`` (``--csv`` to change it); the summaries are printed.

Usage: ``uv run python scripts/toys/ext5_tau_placement_baycap.py [--reps 20] [--test-bags 50] [--draws 400]
[--jobs 16] [--seed 0]``
"""

import argparse
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

# one thread per worker process
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
from scipy.special import betaln, expit
from scipy.stats import beta

from uncawe import env
from uncawe.methods.baycap_kdey import (
    CAL_BAG_SIZE,
    CAL_BAGS,
    TAU_GRID,
    calibration_bags,
    interval_scores,
    select_temperature_from_scores,
    split_halves,
)

ACCURACIES = (0.70, 0.85, 0.95)
RECALL_GAP = 0.04
VAL_SIZES = (50, 200, 1000)
SHIFTS = (0.0, -0.04)
PRIORS = ("uniform", "hf")
BAG_SIZE = 1000
PI_GRID_SIZE = 300
RECALL_GRID_SIZE = 30
# the grid of each recall: the central 1 - 2e-6 of its posterior under the uniform prior and L_V^(1/tau), widened by
# RECALL_MARGIN of its length on each side
RECALL_TAIL = 1e-6
RECALL_MARGIN = 0.25
HYPER_GRID_SIZE = 100
VARIANTS = ("c", "a", "b")
CLASSES = np.array([0, 1])
PI = (np.arange(PI_GRID_SIZE) + 0.5) / PI_GRID_SIZE
LOG_PI1, LOG_PI0 = np.log(PI)[:, None, None], np.log1p(-PI)[:, None, None]


def scenarios() -> list[tuple[float, int, float]]:
    return list(itertools.product(ACCURACIES, VAL_SIZES, SHIFTS))


def true_recalls(accuracy: float) -> np.ndarray:
    return np.array([accuracy + RECALL_GAP, accuracy - RECALL_GAP])


class Fit:
    """The grid of the recalls and the parts of the log posterior that depend only on the validation counts."""

    def __init__(self, correct: np.ndarray, sizes: np.ndarray, tau: float):
        self.axes, self.steps = [], []
        for f, n in zip(correct, sizes):
            low, high = beta.ppf([RECALL_TAIL, 1 - RECALL_TAIL], f / tau + 1, (n - f) / tau + 1)
            margin = RECALL_MARGIN * (high - low)
            low, high = max(low - margin, 1e-9), min(high + margin, 1 - 1e-9)
            edges = np.linspace(low, high, RECALL_GRID_SIZE + 1)
            self.axes.append(0.5 * (edges[1:] + edges[:-1]))
            self.steps.append(edges[1] - edges[0])
        d0, d1 = self.axes
        self.log_d = (np.log(d0)[None, :, None], np.log(d1)[None, None, :])
        self.log_e = (np.log1p(-d0)[None, :, None], np.log1p(-d1)[None, None, :])  # log(1 - d_j)
        # log L_V on the grid of (d0, d1)
        self.loglik_v = (correct[0] * self.log_d[0] + (sizes[0] - correct[0]) * self.log_e[0]) + (
            correct[1] * self.log_d[1] + (sizes[1] - correct[1]) * self.log_e[1]
        )
        self.log_prior = {"uniform": np.zeros((1, RECALL_GRID_SIZE, RECALL_GRID_SIZE)), "hf": self._hf_prior()}

    def _hf_prior(self) -> np.ndarray:
        """log of the prior density of (d0, d1) under the Hf prior, by the midpoint rule on (m, rho)."""
        grid = (np.arange(HYPER_GRID_SIZE) + 0.5) / HYPER_GRID_SIZE
        m, rho = np.meshgrid(grid, grid, indexing="ij")
        nu = 1.0 / rho - 1.0
        a, b = (nu * m).ravel()[:, None], (nu * (1 - m)).ravel()[:, None]
        parts = [
            (a - 1) * np.log(d)[None, :] + (b - 1) * np.log1p(-d)[None, :] - betaln(a, b) for d in self.axes
        ]  # (H, G) each
        # log sum_h exp(parts0[h, i] + parts1[h, k]) as a matrix product, each row scaled by its maximum
        m0, m1 = parts[0].max(axis=1, keepdims=True), parts[1].max(axis=1, keepdims=True)
        shift = m0 + m1
        top = shift.max()
        weights = np.exp(shift - top)
        mixed = (np.exp(parts[0] - m0) * weights).T @ np.exp(parts[1] - m1)
        with np.errstate(divide="ignore"):
            return (top + np.log(mixed) - np.log(len(a)))[None, :, :]


def loglik_u(fit: Fit, q: np.ndarray, variant: str, tau: float) -> np.ndarray:
    """log of the term of $U$ of the variant on the grid $(\\pi_1,d_0,d_1)$."""
    (ld0, ld1), (le0, le1) = fit.log_d, fit.log_e
    # log C_ij pi_j: predicted 1 from class 1 (d1 pi1) and from class 0 ((1-d0) pi0); predicted 0 likewise
    p11, p10 = ld1 + LOG_PI1, le0 + LOG_PI0
    p00, p01 = ld0 + LOG_PI0, le1 + LOG_PI1
    if variant == "c":
        return (q[1] * np.logaddexp(p11, p10) + q[0] * np.logaddexp(p00, p01)) / tau
    if variant == "a":
        return q[1] * np.logaddexp(p11 / tau, p10 / tau) + q[0] * np.logaddexp(p00 / tau, p01 / tau)
    return q[1] * np.logaddexp(ld1 / tau + LOG_PI1, le0 / tau + LOG_PI0) + q[0] * np.logaddexp(
        ld0 / tau + LOG_PI0, le1 / tau + LOG_PI1
    )


def accuracy_draws(fit: Fit, logpost: np.ndarray, q, variant, tau, rng, n_draws) -> np.ndarray:
    post = np.exp(logpost - logpost.max()).ravel()
    post /= post.sum()
    cells = np.unravel_index(rng.choice(post.size, size=n_draws, p=post), logpost.shape)
    pi1 = (cells[0] + rng.random(n_draws)) / PI_GRID_SIZE
    d = [fit.axes[j][cells[j + 1]] + (rng.random(n_draws) - 0.5) * fit.steps[j] for j in (0, 1)]
    d = [np.clip(x, 1e-9, 1 - 1e-9) for x in d]
    lp1, lp0 = np.log(pi1), np.log1p(-pi1)
    ld0, ld1, le0, le1 = np.log(d[0]), np.log(d[1]), np.log1p(-d[0]), np.log1p(-d[1])
    # log-odds of the right class among the instances predicted 1 (class 1 against 0) and predicted 0
    if variant == "c":
        odds1, odds0 = (ld1 + lp1) - (le0 + lp0), (ld0 + lp0) - (le1 + lp1)
    elif variant == "a":
        odds1, odds0 = ((ld1 + lp1) - (le0 + lp0)) / tau, ((ld0 + lp0) - (le1 + lp1)) / tau
    else:
        odds1, odds0 = (ld1 - le0) / tau + (lp1 - lp0), (ld0 - le1) / tau + (lp0 - lp1)
    right = rng.binomial(q[1], expit(odds1)) + rng.binomial(q[0], expit(odds0))
    return right / (q[0] + q[1])


def score_bags(bags, fits: dict, n_draws, rng) -> dict:
    """Per (prior, variant, tau): mean Winkler, coverage, width and signed error of the centre over the bags."""
    sums = {}
    for q, true_acc in bags:
        for variant, tau in itertools.product(VARIANTS, TAU_GRID):
            fit = fits[tau]
            base = loglik_u(fit, q, variant, tau) + fit.loglik_v / tau
            for prior in PRIORS:
                accs = accuracy_draws(fit, base + fit.log_prior[prior], q, variant, tau, rng, n_draws)
                winkler, coverage = interval_scores(accs, true_acc)
                low, high = np.quantile(accs, [0.025, 0.975])
                row = np.array([winkler, coverage, high - low, accs.mean() - true_acc])
                key = (prior, variant, tau)
                sums[key] = sums.get(key, 0) + row
    return {key: value / len(bags) for key, value in sums.items()}


def counts(labels, predicted) -> tuple[np.ndarray, np.ndarray]:
    """The correct predictions F_jj and the sizes n_j of each class."""
    right = labels == predicted
    return np.array([right[labels == j].sum() for j in CLASSES]), np.array([(labels == j).sum() for j in CLASSES])


def replicate(args) -> list[dict]:
    seed, s_index, rep, n_test, n_draws = args
    accuracy, n_val, shift = scenarios()[s_index]
    recalls = true_recalls(accuracy)
    rng = np.random.default_rng(np.random.SeedSequence([seed, s_index, rep]))
    labels = np.repeat(CLASSES, n_val)
    predicted = np.where(rng.random(len(labels)) < recalls[labels], labels, 1 - labels)
    fit_idx, cal_idx = split_halves(labels, CLASSES, int(rng.integers(2**31)))
    fit_cal = {tau: Fit(*counts(labels[fit_idx], predicted[fit_idx]), tau) for tau in TAU_GRID}
    fit_full = {tau: Fit(*counts(labels, predicted), tau) for tau in TAU_GRID}
    index, _ = calibration_bags(labels[cal_idx], CLASSES, CAL_BAGS, CAL_BAG_SIZE, int(rng.integers(2**31)))
    cal_bags = []
    for bag in index:
        y, p = labels[cal_idx][bag], predicted[cal_idx][bag]
        cal_bags.append((np.bincount(p, minlength=2), float(np.mean(y == p))))
    test_bags = []
    for _ in range(n_test):
        n1 = round(BAG_SIZE * rng.random())
        sizes = np.array([BAG_SIZE - n1, n1])
        right = rng.binomial(sizes, np.clip(recalls + shift, 0, 1))
        q = np.array([right[0] + sizes[1] - right[1], right[1] + sizes[0] - right[0]])
        test_bags.append((q, float(right.sum() / BAG_SIZE)))
    cal_scores = score_bags(cal_bags, fit_cal, n_draws, rng)
    test_scores = score_bags(test_bags, fit_full, n_draws, rng)
    rows = []
    for prior, variant in itertools.product(PRIORS, VARIANTS):
        cal_w = np.array([cal_scores[(prior, variant, tau)][0] for tau in TAU_GRID])
        test_w = np.array([test_scores[(prior, variant, tau)][0] for tau in TAU_GRID])
        tau_cal = TAU_GRID[select_temperature_from_scores(TAU_GRID, cal_w)]
        tau_oracle = TAU_GRID[select_temperature_from_scores(TAU_GRID, test_w)]
        for tau in TAU_GRID:
            winkler, coverage, width, error = test_scores[(prior, variant, tau)]
            rows.append(
                {
                    "accuracy": accuracy,
                    "n_val": n_val,
                    "shift": shift,
                    "rep": rep,
                    "prior": prior,
                    "variant": variant,
                    "tau": tau,
                    "winkler": winkler,
                    "coverage": coverage,
                    "width": width,
                    "error": error,
                    "cal_winkler": cal_scores[(prior, variant, tau)][0],
                    "cal_coverage": cal_scores[(prior, variant, tau)][1],
                    "tau_cal": tau_cal,
                    "tau_oracle": tau_oracle,
                }
            )
    return rows


def summaries(df: pd.DataFrame) -> None:
    keys = ["prior", "accuracy", "n_val", "shift", "variant"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 1000)
    print("\n== per tau: mean over replicates of the test scores ==")
    by_tau = df.groupby([*keys, "tau"])[["coverage", "width", "error", "winkler", "cal_winkler"]].mean()
    print(by_tau.round(4).to_string())
    print(
        "\n== selection: tau_cal and tau_oracle over the replicates, test scores at tau_cal, at 1 and at the oracle =="
    )
    rows = []
    for key, part in df.groupby(keys):
        at = {col: part.loc[part["tau"] == part[col]].sort_values("rep") for col in ("tau_cal", "tau_oracle")}
        one = part.loc[part["tau"] == 1.0]
        cal, oracle = at["tau_cal"], at["tau_oracle"]
        rows.append(
            {
                **dict(zip(keys, key)),
                "tau_cal": " ".join(f"{t:g}:{n}" for t, n in cal["tau_cal"].value_counts().sort_index().items()),
                "tau_oracle": " ".join(
                    f"{t:g}:{n}" for t, n in oracle["tau_oracle"].value_counts().sort_index().items()
                ),
                "agree": float((cal["tau_cal"].to_numpy() == cal["tau_oracle"].to_numpy()).mean()),
                "cov_cal": cal["coverage"].mean(),
                "width_cal": cal["width"].mean(),
                "err_cal": cal["error"].mean(),
                "wink_cal": cal["winkler"].mean(),
                "wink_1": one["winkler"].mean(),
                "wink_oracle": oracle["winkler"].mean(),
                "cov_1": one["coverage"].mean(),
            }
        )
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--test-bags", type=int, default=50)
    parser.add_argument("--draws", type=int, default=400)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--summary-only", action="store_true", help="print the summaries of an existing --csv")
    args = parser.parse_args()
    path = args.csv or os.path.join(env.root_dir, "toys", "ext5_tau_placement_baycap.csv")
    if args.summary_only:
        summaries(pd.read_csv(path))
        return
    tasks = [
        (args.seed, s, rep, args.test_bags, args.draws) for s in range(len(scenarios())) for rep in range(args.reps)
    ]
    print(
        f"scenarios={len(scenarios())} reps={args.reps} test_bags={args.test_bags} draws={args.draws} "
        f"cal_bags={CAL_BAGS}x{CAL_BAG_SIZE} grid={PI_GRID_SIZE}x{RECALL_GRID_SIZE}^2 seed={args.seed}",
        flush=True,
    )
    start, rows = time.time(), []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done, result in enumerate(pool.map(replicate, tasks), start=1):
            rows.extend(result)
            if done % max(1, len(tasks) // 20) == 0 or done == len(tasks):
                print(f"[{done}/{len(tasks)}] {time.time() - start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"written: {path}")
    summaries(df)


if __name__ == "__main__":
    main()
