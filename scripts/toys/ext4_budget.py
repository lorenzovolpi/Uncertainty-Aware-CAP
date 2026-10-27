"""Toy simulation for Extension 4: the uncertainty budget of CAP intervals as a function of |U| and |V|.

Binary problem in the paper's notation, $C_{ij}=P(\\hat Y=\\omega_i\\mid Y=\\omega_j)$. The target
classifier $h$ has rates $d=\\operatorname{diag}(C)$; the surrogate classifier $s$ used by the
quantifier has its own rates and, as a simplification, is conditionally independent of $h$ given $Y$.
For every configuration $(h, n, n_s)$, with $n=|U|$ and $n_s$ validation instances per class:

1. an outer loop draws ``--vseeds`` validation sets $V$ (counts of correct predictions of $h$ and of
   predictions $\\hat Y_s=\\omega_1$ of $s$ in each class);
2. for each $V$, ``--bags`` target bags are drawn with fixed class counts $n_j=\\operatorname{round}(n\\pi^Q_j)$
   (APP, as in the paper) and true prevalence $\\pi^Q_1$ fixed;
3. the quantifier is ACC on the predictions of $s$, with the rates of $s$ estimated on $V$;
4. 95% intervals for accuracy are built by
   - SQS-H: fixed-count bags of size $n$ at $\\hat\\pi$ drawn from $V$ (carries term (3));
   - SQS-H + U bootstrap (RQBS-like): bootstrap of $U$ for $\\hat\\pi$, one bag per draw (terms (1), (3));
   - SQS-H + U and V bootstrap: as above, also resampling $V$ per draw (terms (1), (2), (3));
   - BayCAP, population pushforward (terms (1), (2)), exact posterior on a grid as in
     ``ext2_estimand_gap.py``;
   - BayCAP, realised-table predictive of Extension 2 (terms (1), (2), (3));
   - oracle-$\\pi$ SQS (reference, uses the true counts): fixed-count bags from $V$ at the true counts;
5. coverage is computed against the realised accuracy of each bag (what ``main`` scores).

It also splits the error of the SQS-H point estimate, $\\hat A-A(h,U)=e_1+e_2+e_3$, into prevalence
($e_1$), rate-matrix ($e_2$) and finite-$U$ ($e_3$) parts and reports the root mean square of each.
Next to the oracle-$\\pi$ coverage it prints the closed-form prediction $2\\Phi(1.96/\\sqrt{1+r})-1$,
with $r=\\operatorname{Var}(e_2)/\\operatorname{Var}(e_3)$ computed from $d$, $\\pi^Q$, $n$ and $n_s$.

Reproduces the tables in ``docs/ext4_uncertainty_budget.md`` (Appendix A).

Usage: ``uv run python scripts/toys/ext4_budget.py [--vseeds 40] [--bags 50] [--seed 0]``
"""

import argparse

import numpy as np
from scipy.stats import norm

PI1 = 0.5
S10, S11 = 0.25, 0.75  # surrogate rates P(yhat_s=1 | y=0), P(yhat_s=1 | y=1)
H = {"flat": (0.90, 0.70), "steep": (0.95, 0.55)}  # d = (C_00, C_11) of the target classifier
N_U = (100, 300, 1000, 3000)
N_S = (100, 1000)
ALPHA = 0.05
Q = [ALPHA / 2, 1 - ALPHA / 2]
M = 1000  # performance samples per interval
GRID = np.linspace(0.0005, 0.9995, 1000)
M_C = 200  # draws of C from its validation posterior (BayCAP)
METHODS = ("sqsh", "sqsh_u", "sqsh_uv", "bay_pop", "bay_pred", "oracle_pi")


def acc_quant(n1_pred, n, s10, s11):
    """ACC estimate of pi_1 from the surrogate's predicted positives, clipped to [0, 1]."""
    den = np.where(np.abs(s11 - s10) < 1e-9, 1e-9, s11 - s10)
    return np.clip((n1_pred / n - s10) / den, 0.0, 1.0)


def bag_acc(rng, counts1, n, d0, d1):
    """Accuracy of fixed-count bags drawn with replacement from V (h correct w.p. d_j)."""
    counts1 = np.asarray(counts1)
    return (rng.binomial(n - counts1, d0) + rng.binomial(counts1, d1)) / n


def bay_draws(rng, n1_pred, n, f10, f11, n_s):
    """Joint BayCAP posterior draws (C10, C11, pi1), uniform priors, grid over pi1."""
    c10 = rng.beta(1 + f10, 1 + n_s - f10, M_C)
    c11 = rng.beta(1 + f11, 1 + n_s - f11, M_C)
    p = c10[:, None] * (1 - GRID) + c11[:, None] * GRID
    logw = n1_pred * np.log(p) + (n - n1_pred) * np.log1p(-p)
    w = np.exp(logw - logw.max()).ravel()
    idx = rng.choice(w.size, M, p=w / w.sum())
    i, j = np.unravel_index(idx, (M_C, GRID.size))
    return c10[i], c11[i], GRID[j]


def run(rng, d, n, n_s, n_v, n_bags):
    d0, d1 = d
    n1 = round(n * PI1)
    n0 = n - n1
    hits = {k: [] for k in METHODS}
    width = {k: [] for k in METHODS}
    errs = []
    for _ in range(n_v):
        # validation set: h correct counts per class, s predicted-positive counts per class
        k0, k1 = rng.binomial(n_s, d0), rng.binomial(n_s, d1)
        s0, s1 = rng.binomial(n_s, S10), rng.binomial(n_s, S11)
        dh0, dh1, sh10, sh11 = k0 / n_s, k1 / n_s, s0 / n_s, s1 / n_s
        vhits = {k: 0 for k in METHODS}
        for _ in range(n_bags):
            # target bag: h correct per class, s predicted positives, h predicted positives
            c0, c1 = rng.binomial(n0, d0), rng.binomial(n1, d1)
            a_real = (c0 + c1) / n
            n1_s = rng.binomial(n0, S10) + rng.binomial(n1, S11)
            n1_h = (n0 - c0) + c1
            pi_hat = acc_quant(n1_s, n, sh10, sh11)
            cnt_hat = round(n * pi_hat)
            samples = {"sqsh": bag_acc(rng, np.full(M, cnt_hat), n, dh0, dh1)}
            # U bootstrap of the quantifier input
            n1_s_b = rng.binomial(n, n1_s / n, M)
            cnt_u = np.rint(n * acc_quant(n1_s_b, n, sh10, sh11)).astype(int)
            samples["sqsh_u"] = bag_acc(rng, cnt_u, n, dh0, dh1)
            # U and V bootstrap: resample the rates of s and h as well
            sb10, sb11 = rng.binomial(n_s, sh10, M) / n_s, rng.binomial(n_s, sh11, M) / n_s
            db0, db1 = rng.binomial(n_s, dh0, M) / n_s, rng.binomial(n_s, dh1, M) / n_s
            cnt_uv = np.rint(n * acc_quant(n1_s_b, n, sb10, sb11)).astype(int)
            samples["sqsh_uv"] = bag_acc(rng, cnt_uv, n, db0, db1)
            # BayCAP on the predictions of h (f10 = false positives, f11 = true positives in V)
            c10, c11, p1 = bay_draws(rng, n1_h, n, n_s - k0, k1, n_s)
            samples["bay_pop"] = (1 - c10) * (1 - p1) + c11 * p1
            rho11 = c11 * p1 / (c10 * (1 - p1) + c11 * p1)
            rho00 = (1 - c10) * (1 - p1) / ((1 - c10) * (1 - p1) + (1 - c11) * p1)
            samples["bay_pred"] = (rng.binomial(n1_h, rho11) + rng.binomial(n - n1_h, rho00)) / n
            samples["oracle_pi"] = bag_acc(rng, np.full(M, n1), n, dh0, dh1)
            for k, s in samples.items():
                lo, hi = np.quantile(s, Q)
                vhits[k] += lo <= a_real <= hi
                width[k].append(hi - lo)
            # error split of the SQS-H point estimate
            e1 = ((cnt_hat - n1) * (dh1 - dh0)) / n
            e2 = (n0 * (dh0 - d0) + n1 * (dh1 - d1)) / n
            e3 = (n0 * d0 + n1 * d1) / n - a_real
            errs.append((e1, e2, e3))
        for k in METHODS:
            hits[k].append(vhits[k] / n_bags)
    errs = np.array(errs)
    out = {k: np.mean(v) for k, v in hits.items()}
    out |= {f"w_{k}": np.mean(v) for k, v in width.items()}
    out["sqsh_p10"], out["sqsh_p90"] = np.quantile(hits["sqsh"], [0.1, 0.9])
    out["rmse"] = np.sqrt(np.mean(errs**2, axis=0))
    out["bias"] = errs.mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vseeds", type=int, default=40)
    parser.add_argument("--bags", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"pi1={PI1}, S=({S10},{S11}), vseeds={args.vseeds}, bags/V={args.bags}, m={M}, seed={args.seed}")
    for name, d in H.items():
        print(f"\n# h={name}, d=(C00, C11)={d}")
        print(
            f"{'n_s':>5} {'n':>5} {'ratio':>6} | "
            + " ".join(f"{k:>9}" for k in METHODS)
            + f" {'theory':>7} | {'sqsh p10-p90':>12} | {'rmse e1':>8} {'e2':>7} {'e3':>7}"
        )
        for n_s in N_S:
            for n in N_U:
                r = run(rng, d, n, n_s, args.vseeds, args.bags)
                # r = Var(e_2) / Var(e_3); closed-form coverage of the oracle-pi interval
                pi, v = np.array([1 - PI1, PI1]), np.array(d) * (1 - np.array(d))
                ratio = (np.sum(pi**2 * v) / n_s) / (np.sum(pi * v) / n)
                theory = 2 * norm.cdf(1.96 / np.sqrt(1 + ratio)) - 1
                cov = " ".join(f"{r[k]:9.3f}" for k in METHODS)
                print(
                    f"{n_s:5d} {n:5d} {ratio:6.2f} | {cov} {theory:7.3f} | "
                    f"{r['sqsh_p10']:5.2f}-{r['sqsh_p90']:<6.2f} | " + " ".join(f"{x:7.4f}" for x in r["rmse"])
                )
                wid = " ".join(f"{r['w_' + k]:9.3f}" for k in METHODS)
                print(f"{'':>5} {'':>5} {'width':>6} | {wid}")


if __name__ == "__main__":
    main()
