"""Toy simulation for Extension 2: sources of BayCAP miscalibration.

Binary problem in the paper's notation, $C_{ij}=P(\\hat Y=\\omega_i\\mid Y=\\omega_j)$, with
$C_{10}=0.1$ and $C_{11}=0.7$. For each simulated configuration:

1. a validation set with ``n_s`` instances per class gives the counts that inform $C$;
2. a target bag of size $N^Q$ is drawn with prevalence $\\pi_1$, either with fixed class counts
   (APP, as in the paper) or with multinomial class counts (i.i.d.);
3. the BayCAP posterior over $(\\pi^Q, C)$ is computed (uniform priors) by sampling $C$ from its
   validation posterior and weighting a grid over $\\pi_1$ by the target likelihood
   $\\operatorname{Mult}(n^t\\mid N^Q, C\\pi^Q)$;
4. two 95% intervals for accuracy are built from the same posterior draws:
   - population interval: percentiles of $\\sum_j C_{jj}\\pi_j$ (BayCAP as in the paper);
   - realised-table predictive: for each draw, the true classes behind the observed predictions
     are sampled with probability $\\rho_{ij}\\propto C_{ij}\\pi_j$ and accuracy is computed on that
     table.

Coverage is reported against the population accuracy and against the realised accuracy of the bag
(what the evaluation in ``main`` scores).

Reproduces the table in ``docs/ext2_baycap.md`` (section "Why BayCAP can be miscalibrated").

Usage: ``uv run python scripts/toys/ext2_estimand_gap.py [--bags 3000] [--seed 0]``
"""

import argparse

import numpy as np
from scipy.stats import binom

N_Q = 1000
C10, C11 = 0.1, 0.7  # P(yhat=1 | y=0), P(yhat=1 | y=1)
PI1 = 0.5
N_S = (100, 500, 5000)
ALPHA = 0.05
GRID = np.linspace(0, 1, 401)
M_C = 200  # draws of C from its validation posterior
M_OUT = 2000  # resampled posterior draws used to build the intervals


def posterior_draws(rng, n1_pred, f10, f11, n_s):
    """Return joint posterior draws (C10, C11, pi1) given validation and target counts."""
    c10 = rng.beta(1 + f10, 1 + n_s - f10, M_C)
    c11 = rng.beta(1 + f11, 1 + n_s - f11, M_C)
    p_pred1 = c10[:, None] * (1 - GRID) + c11[:, None] * GRID
    logw = binom.logpmf(n1_pred, N_Q, p_pred1)
    w = np.exp(logw - logw.max()).ravel()
    idx = rng.choice(w.size, M_OUT, p=w / w.sum())
    i, j = np.unravel_index(idx, (M_C, GRID.size))
    return c10[i], c11[i], GRID[j]


def intervals(rng, n1_pred, c10, c11, pi1):
    q = [ALPHA / 2, 1 - ALPHA / 2]
    pop = (1 - c10) * (1 - pi1) + c11 * pi1
    n0_pred = N_Q - n1_pred
    # P(Y=1 | yhat=1) and P(Y=0 | yhat=0) for each draw
    rho11 = c11 * pi1 / (c10 * (1 - pi1) + c11 * pi1)
    rho00 = (1 - c10) * (1 - pi1) / ((1 - c10) * (1 - pi1) + (1 - c11) * pi1)
    realised = (rng.binomial(n1_pred, rho11) + rng.binomial(n0_pred, rho00)) / N_Q
    return np.quantile(pop, q), np.quantile(realised, q)


def run(rng, n_s, mode, n_bags):
    hits = {"pop_vs_pop": 0, "pop_vs_real": 0, "pred_vs_real": 0}
    widths = {"pop": [], "pred": []}
    acc_pop = (1 - C10) * (1 - PI1) + C11 * PI1
    for _ in range(n_bags):
        f10, f11 = rng.binomial(n_s, C10), rng.binomial(n_s, C11)
        n1 = round(N_Q * PI1) if mode == "app" else rng.binomial(N_Q, PI1)
        n0 = N_Q - n1
        fp, tp = rng.binomial(n0, C10), rng.binomial(n1, C11)
        acc_real = ((n0 - fp) + tp) / N_Q
        (lo, hi), (plo, phi) = intervals(rng, fp + tp, *posterior_draws(rng, fp + tp, f10, f11, n_s))
        hits["pop_vs_pop"] += lo <= acc_pop <= hi
        hits["pop_vs_real"] += lo <= acc_real <= hi
        hits["pred_vs_real"] += plo <= acc_real <= phi
        widths["pop"].append(hi - lo)
        widths["pred"].append(phi - plo)
    return {k: v / n_bags for k, v in hits.items()} | {f"w_{k}": np.mean(v) for k, v in widths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bags", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"N_Q={N_Q}, C10={C10}, C11={C11}, pi1={PI1}, bags={args.bags}, seed={args.seed}")
    header = f"{'n_s':>5} {'bags':>4} {'pop/pop':>8} {'pop/real':>9} {'pred/real':>10} {'w_pop':>7} {'w_pred':>7}"
    print(header)
    for n_s in N_S:
        for mode in ("app", "iid"):
            r = run(rng, n_s, mode, args.bags)
            print(
                f"{n_s:5d} {mode:>4} {r['pop_vs_pop']:8.3f} {r['pop_vs_real']:9.3f} "
                f"{r['pred_vs_real']:10.3f} {r['w_pop']:7.3f} {r['w_pred']:7.3f}"
            )


if __name__ == "__main__":
    main()
