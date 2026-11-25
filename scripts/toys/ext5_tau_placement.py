r"""Toy simulation for Extensions 3 and 5: where the temperature $\tau$ of BayCAP-KDEy-h-inst acts, and what that does
to the selection of $\tau$ of E5.3.

Binary problem, $K=2$. The classifier outputs a score $z$ with $z\mid Y=j\sim\mathcal N(\mu_j,1)$,
$\mu_0=-d/2$, $\mu_1=d/2$, and predicts $\hat y=\mathbb 1[z>0]$ (accuracy $\Phi(d/2)$ on every bag). The class
densities $\hat f_j$ are Gaussian kernel densities of the scores of $V_j$ at a fixed bandwidth (``BANDWIDTH``), so they
differ from the true densities when $V$ is small. Prior $\pi\sim\operatorname{Dir}(1,1)$. For a bag $z_{1:N}$ and
$m_n(\pi)=\sum_j\pi_j\hat f_j(z_n)$, three ways of putting $\tau$ in the model of the realised table:

- ``c`` (the implemented method): marginal $\propto\operatorname{Dir}(\pi)\prod_n m_n(\pi)^{1/\tau}$, untempered
  per-instance weights $w_{nj}=\pi_j\hat f_j(z_n)/m_n(\pi)$;
- ``a`` (whole complete-data likelihood tempered): marginal $\propto\operatorname{Dir}(\pi)\prod_n\sum_j
  (\pi_j\hat f_j(z_n))^{1/\tau}$, weights $w_{nj}\propto(\pi_j\hat f_j(z_n))^{1/\tau}$;
- ``b`` (output term tempered only): marginal $\propto\operatorname{Dir}(\pi)\prod_n\sum_j\pi_j\hat f_j(z_n)^{1/\tau}$,
  weights $w_{nj}\propto\pi_j\hat f_j(z_n)^{1/\tau}$.

At $\tau=1$ the three coincide. The marginal of $\pi_1$ is computed exactly on a grid (no MCMC); $R$ values of $\pi$
are drawn from it, the true class of each instance from its weights under each draw, and the interval of the accuracy
of the realised table is the central 95% interval of the draws (``methods/baycap_kdey.py:interval_scores``), scored
against the realised accuracy of the bag. Every variant targets the realised table.

The selection follows E5.3 (``methods/baycap_kdey.py``): $V$ split per class into $V_{\text{fit}}$ and $V_{\text{cal}}$
(``split_halves``), $50$ calibration bags of $1000$ from $V_{\text{cal}}$ by UPP (``calibration_bags``), $\hat f_j$ on
$V_{\text{fit}}$, $\tau_{\text{cal}}$ the lowest mean Winkler score over ``TAU_GRID`` with ties towards $1$
(``select_temperature_from_scores``); the test bags use $\hat f_j$ fitted on the full $V$. Test bags: $N=1000$,
$\pi_1\sim U(0,1)$, $n_1=\operatorname{round}(N\pi_1)$, scores drawn from the true laws, shifted by $\delta$ in the
misspecified scenarios (the same shift for both classes, not seen by $V$ or by the calibration bags). The oracle
$\tau$ is the one with the lowest mean Winkler score on the test bags of the replicate (a reference, it uses the test
labels).

Scenarios: $d$ in ``SEPARATIONS`` (accuracy about $0.70$, $0.85$, $0.95$) $\times$ $n_j$ in ``VAL_SIZES`` $\times$
$\delta$ in ``SHIFTS``; ``--reps`` replicates of $V$ per scenario, each with its calibration and test bags. Every
replicate has its own generator, seeded by ``(--seed, scenario, replicate)``, so the results do not depend on
``--jobs``. Toy results, not experiment results.

Output: per (scenario, replicate, variant, $\tau$) the mean test coverage, width, signed error of the centre and
Winkler score, the calibration Winkler score, $\tau_{\text{cal}}$ and $\tau_{\text{oracle}}$, in
``<OUT_DIR>/uncawe/toys/ext5_tau_placement.csv`` (``--csv`` to change it); the summaries are printed.

Usage: ``uv run python scripts/toys/ext5_tau_placement.py [--reps 20] [--test-bags 50] [--draws 400] [--jobs 16]
[--seed 0]``
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
from scipy.special import expit, logsumexp
from scipy.stats import norm

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

# accuracy Phi(d/2) of the classifier: about 0.70, 0.85, 0.95
SEPARATIONS = (1.05, 2.07, 3.29)
VAL_SIZES = (50, 200, 1000)
SHIFTS = (0.0, 0.3)
BANDWIDTH = 0.3
BAG_SIZE = 1000
GRID_SIZE = 1000
VARIANTS = ("c", "a", "b")
CLASSES = np.array([0, 1])


def scenarios() -> list[tuple[float, int, float]]:
    return list(itertools.product(SEPARATIONS, VAL_SIZES, SHIFTS))


def kde_log_densities(train: np.ndarray, labels: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``[j, n]`` = log of the Gaussian kernel density of the scores of class ``j`` of ``train`` at ``z[n]``."""
    out = np.empty((2, len(z)))
    for j in CLASSES:
        points = train[labels == j]
        sq = -0.5 * ((z[:, None] - points[None, :]) / BANDWIDTH) ** 2
        out[j] = logsumexp(sq, axis=1) - np.log(len(points) * BANDWIDTH * np.sqrt(2 * np.pi))
    return out


GRID = (np.arange(GRID_SIZE) + 0.5) / GRID_SIZE
LOG_P1, LOG_P0 = np.log(GRID), np.log1p(-GRID)


def marginal_loglik(lf: np.ndarray, variant: str, tau: float, base_c: np.ndarray | None = None) -> np.ndarray:
    """The log of the (tempered) marginal likelihood of the bag on ``GRID`` (``lf[j, n]`` = $\\ln\\hat f_j(z_n)$)."""
    if variant == "c":
        return base_c / tau
    if variant == "a":
        terms = np.logaddexp((LOG_P1[:, None] + lf[1][None, :]) / tau, (LOG_P0[:, None] + lf[0][None, :]) / tau)
    else:
        terms = np.logaddexp(LOG_P1[:, None] + lf[1][None, :] / tau, LOG_P0[:, None] + lf[0][None, :] / tau)
    return terms.sum(axis=1)


def accuracy_draws(lf, pred, variant, tau, rng, n_draws, base_c) -> np.ndarray:
    """``n_draws`` accuracies of the realised table of the bag under the variant and temperature."""
    loglik = marginal_loglik(lf, variant, tau, base_c)  # uniform prior: the posterior is the likelihood
    post = np.exp(loglik - loglik.max())
    post /= post.sum()
    cells = rng.choice(GRID_SIZE, size=n_draws, p=post)
    pi1 = (cells + rng.random(n_draws)) / GRID_SIZE
    logit_pi = np.log(pi1) - np.log1p(-pi1)
    diff = lf[1] - lf[0]
    if variant == "c":
        logit_w = logit_pi[:, None] + diff[None, :]
    elif variant == "a":
        logit_w = (logit_pi[:, None] + diff[None, :]) / tau
    else:
        logit_w = logit_pi[:, None] + diff[None, :] / tau
    w1 = expit(logit_w)
    drawn = rng.random(w1.shape) < w1
    return (drawn == pred[None, :].astype(bool)).mean(axis=1)


def score_bags(bags, lfs, n_draws, rng) -> dict:
    """Per (variant, tau): mean Winkler, coverage, width and signed error of the centre over the bags."""
    sums = {}
    for (z, true_acc), lf in zip(bags, lfs):
        pred = (z > 0).astype(int)
        base_c = np.logaddexp(LOG_P1[:, None] + lf[1][None, :], LOG_P0[:, None] + lf[0][None, :]).sum(axis=1)
        for variant, tau in itertools.product(VARIANTS, TAU_GRID):
            accs = accuracy_draws(lf, pred, variant, tau, rng, n_draws, base_c)
            winkler, coverage = interval_scores(accs, true_acc)
            low, high = np.quantile(accs, [0.025, 0.975])
            row = np.array([winkler, coverage, high - low, accs.mean() - true_acc])
            sums[(variant, tau)] = sums.get((variant, tau), 0) + row
    return {key: value / len(bags) for key, value in sums.items()}


def replicate(args) -> list[dict]:
    seed, s_index, rep, n_test, n_draws = args
    d, n_val, shift = scenarios()[s_index]
    rng = np.random.default_rng(np.random.SeedSequence([seed, s_index, rep]))
    mu = np.array([-d / 2, d / 2])
    labels = np.repeat(CLASSES, n_val)
    scores = rng.normal(mu[labels], 1.0)
    fit, cal = split_halves(labels, CLASSES, int(rng.integers(2**31)))
    # calibration bags from V_cal, fitted on V_fit
    index, _ = calibration_bags(labels[cal], CLASSES, CAL_BAGS, CAL_BAG_SIZE, int(rng.integers(2**31)))
    cal_bags, cal_lfs = [], []
    for bag in index:
        z, y = scores[cal][bag], labels[cal][bag]
        cal_bags.append((z, float(np.mean((z > 0).astype(int) == y))))
        cal_lfs.append(kde_log_densities(scores[fit], labels[fit], z))
    # test bags from the true laws (shifted by delta), fitted on the full V
    test_bags, test_lfs = [], []
    for _ in range(n_test):
        n1 = round(BAG_SIZE * rng.random())
        y = np.repeat(CLASSES, [BAG_SIZE - n1, n1])
        z = rng.normal(mu[y] + shift, 1.0)
        test_bags.append((z, float(np.mean((z > 0).astype(int) == y))))
        test_lfs.append(kde_log_densities(scores, labels, z))
    cal_scores = score_bags(cal_bags, cal_lfs, n_draws, rng)
    test_scores = score_bags(test_bags, test_lfs, n_draws, rng)
    rows = []
    for variant in VARIANTS:
        cal_w = np.array([cal_scores[(variant, tau)][0] for tau in TAU_GRID])
        test_w = np.array([test_scores[(variant, tau)][0] for tau in TAU_GRID])
        tau_cal = TAU_GRID[select_temperature_from_scores(TAU_GRID, cal_w)]
        tau_oracle = TAU_GRID[select_temperature_from_scores(TAU_GRID, test_w)]
        for tau in TAU_GRID:
            winkler, coverage, width, error = test_scores[(variant, tau)]
            rows.append(
                {
                    "separation": d,
                    "accuracy": float(norm.cdf(d / 2)),
                    "n_val": n_val,
                    "shift": shift,
                    "rep": rep,
                    "variant": variant,
                    "tau": tau,
                    "winkler": winkler,
                    "coverage": coverage,
                    "width": width,
                    "error": error,
                    "cal_winkler": cal_scores[(variant, tau)][0],
                    "cal_coverage": cal_scores[(variant, tau)][1],
                    "tau_cal": tau_cal,
                    "tau_oracle": tau_oracle,
                }
            )
    return rows


def summaries(df: pd.DataFrame) -> None:
    keys = ["separation", "n_val", "shift", "variant"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print("\n== per tau: mean over replicates of the test scores ==")
    by_tau = df.groupby([*keys, "tau"])[["coverage", "width", "error", "winkler", "cal_winkler"]].mean()
    print(by_tau.round(4).to_string())
    print(
        "\n== selection: tau_cal and tau_oracle over the replicates, test scores at tau_cal, at 1 and at the oracle =="
    )
    rows = []
    for key, part in df.groupby(keys):
        reps = part.groupby("rep")
        at = {
            name: reps.apply(lambda g, col=col: g.loc[g["tau"] == g[col].iloc[0]].iloc[0], include_groups=False)
            for name, col in (("cal", "tau_cal"), ("oracle", "tau_oracle"))
        }
        one = part.loc[part["tau"] == 1.0]
        rows.append(
            {
                **dict(zip(keys, key)),
                "tau_cal": " ".join(f"{t:g}:{n}" for t, n in at["cal"]["tau_cal"].value_counts().sort_index().items()),
                "tau_oracle": " ".join(
                    f"{t:g}:{n}" for t, n in at["oracle"]["tau_oracle"].value_counts().sort_index().items()
                ),
                "agree": float((at["cal"]["tau_cal"] == at["cal"]["tau_oracle"]).mean()),
                "cov_cal": at["cal"]["coverage"].mean(),
                "width_cal": at["cal"]["width"].mean(),
                "err_cal": at["cal"]["error"].mean(),
                "wink_cal": at["cal"]["winkler"].mean(),
                "wink_1": one["winkler"].mean(),
                "wink_oracle": at["oracle"]["winkler"].mean(),
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
    path = args.csv or os.path.join(env.root_dir, "toys", "ext5_tau_placement.csv")
    if args.summary_only:
        summaries(pd.read_csv(path))
        return
    tasks = [
        (args.seed, s, rep, args.test_bags, args.draws) for s in range(len(scenarios())) for rep in range(args.reps)
    ]
    print(
        f"scenarios={len(scenarios())} reps={args.reps} test_bags={args.test_bags} draws={args.draws} "
        f"cal_bags={CAL_BAGS}x{CAL_BAG_SIZE} bandwidth={BANDWIDTH} grid={GRID_SIZE} seed={args.seed}",
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
