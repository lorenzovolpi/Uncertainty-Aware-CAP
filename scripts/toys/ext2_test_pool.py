"""Toy simulation for Extension 2: cause 4, the bags of ``main`` are re-drawn from one finite test pool.

Binary problem in the paper's notation, sized like ``mammographic`` in ``main``: $n^s_j=141$
validation instances per class and target bags of $N^Q=1000$ instances, drawn with UPP
($\\pi_1\\sim U(0,1)$, $\\operatorname{round}(N^Q\\pi_j)$ instances of class $j$) with replacement from a
test pool $U$ with $P_j$ instances per class, as ``quapy``'s ``sampling_index`` does. Each replication
draws one validation set and one pool, shared by ``CLASSIFIERS`` classifiers and by ``BAGS`` bags,
as a dataset of ``main`` is shared by its classifiers and bags.

Instances have a latent difficulty shared by the classifiers: instance $i$ is classified correctly by
classifier $k$ when $\\sqrt\\rho\\,u_i+\\sqrt{1-\\rho}\\,e_{ik}<\\Phi^{-1}(r_{jk})$, with $u_i,e_{ik}$
standard normal, so that the recall of classifier $k$ on class $j$ is $r_{jk}$ and the classifiers err
on the same hard instances when $\\rho>0$. The recall of classifier $k$ on the pool is $g_{jk}$; every bag
of the replication has $\\operatorname{Bin}(n^Q_j,g_{jk})$ correct instances of class $j$.

For each bag the BayCAP posterior over $(C,\\pi^Q)$ is computed with the $\\operatorname{Dir}(\\mathbf 1)$
prior, as in ``ext2_eb_binary.py``, and its realised 95% interval is scored against the realised
accuracy of the bag. The ``pool-aware`` interval is a reference, since it uses $P_j$, which BayCAP
does not know: each draw of $C$ is replaced by a draw of the pool's own recalls,
$g_j\\sim\\operatorname{Bin}(P_j,r_j)/P_j$, before the target likelihood and the pushforward.

The first table reports, for each pool size: realised coverage and Winkler score of BayCAP and of
the pool-aware reference; the share of (replication, classifier) pairs whose coverage over the bags is
below $0.5$; the standard deviation over pairs of the shift (mean over bags of the posterior-mean
accuracy minus the realised accuracy) and its prediction
$\\sqrt{\\sum_j\\tfrac14\\bar r_j(1-\\bar r_j)(1/n^s_j+1/P_j)}$, with $\\bar r_j$ the mean recall of the classifiers and the mean prevalence of the bags about $(1/2,1/2)$, for the $V$-vs-$U$ gap; the
correlation and slope of the shift against its cause-4 prediction
$\\overline{\\sum_j\\pi_j(\\tilde r_j-g_j)}$, with $\\tilde r_j=(f_j+1)/(n^s_j+2)$ the posterior-mean
recall from $V$; the mean correlation of the shifts of two classifiers across replications; and the
share of replications in which all classifiers have shifts of the same sign. Toy results, not results
of ``main``.

Usage: ``uv run python scripts/toys/ext2_test_pool.py [--reps 200] [--bags 20] [--rho 0.5] [--seed 0]``
"""

import argparse

import numpy as np
from scipy.stats import norm

N_Q = 1000
N_S = 141
POOL_SIZES = (25, 62, 125, 250, 500, 2000, None)  # instances per class; None is an infinite pool
CLASSIFIERS = ((0.80, 0.84), (0.78, 0.86), (0.82, 0.82), (0.76, 0.88), (0.81, 0.85))  # (r_0k, r_1k)
ALPHA = 0.05
GRID = np.linspace(0, 1, 201)
M_C = 300  # draws of C from its validation posterior, shared by the bags of a replication
M_OUT = 2000  # resampled posterior draws used to build the intervals


def correct(rng, n, recall, u, rho):
    """Whether each of the n instances with latent difficulties u is classified correctly."""
    e = rng.standard_normal(n)
    return np.sqrt(rho) * u + np.sqrt(1 - rho) * e < norm.ppf(recall)


def winkler(y, lo, hi):
    return hi - lo + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def interval(rng, n1_pred, c00, c11, log_p, log_q):
    """Realised 95% interval and posterior-mean accuracy for one bag, given draws of the bag's recalls."""
    logw = n1_pred * log_p + (N_Q - n1_pred) * log_q
    w = np.exp(logw - logw.max()).ravel()
    idx = rng.choice(w.size, M_OUT, p=w / w.sum())
    i, j = np.unravel_index(idx, log_p.shape)
    c00, c11, pi1 = c00[i], c11[i], GRID[j]
    rho11 = c11 * pi1 / ((1 - c00) * (1 - pi1) + c11 * pi1)
    rho00 = c00 * (1 - pi1) / (c00 * (1 - pi1) + (1 - c11) * pi1)
    realised = (rng.binomial(n1_pred, rho11) + rng.binomial(N_Q - n1_pred, rho00)) / N_Q
    lo, hi = np.quantile(realised, [ALPHA / 2, 1 - ALPHA / 2])
    return lo, hi, float(np.mean(c00 * (1 - pi1) + c11 * pi1))


def log_lik_grid(c00, c11):
    p = np.clip((1 - c00)[:, None] * (1 - GRID) + c11[:, None] * GRID, 1e-300, 1 - 1e-16)
    return np.log(p), np.log1p(-p)


def run(rng, pool, n_reps, n_bags, rho):
    """Simulate one pool size; return per-(replication, classifier) scores."""
    n_k = len(CLASSIFIERS)
    recalls = np.asarray(CLASSIFIERS).T  # shape (2 classes, n_k)
    out = {name: np.zeros((n_reps, n_k)) for name in ("cov", "W", "cov_aware", "W_aware", "shift", "pred")}
    for rep in range(n_reps):
        f, g = np.zeros((2, n_k)), np.zeros((2, n_k))
        for j in range(2):
            u_v = rng.standard_normal(N_S)
            f[j] = [correct(rng, N_S, recalls[j, k], u_v, rho).sum() for k in range(n_k)]
            if pool is None:
                g[j] = recalls[j]
            else:
                u_p = rng.standard_normal(pool)
                g[j] = [correct(rng, pool, recalls[j, k], u_p, rho).mean() for k in range(n_k)]
        pi1s = rng.uniform(size=n_bags)
        n1s = np.round(N_Q * pi1s).astype(int)
        r_tilde = (f + 1) / (N_S + 2)
        for k in range(n_k):
            r0 = rng.beta(1 + f[0, k], 1 + N_S - f[0, k], M_C)
            r1 = rng.beta(1 + f[1, k], 1 + N_S - f[1, k], M_C)
            if pool is None:
                g0, g1 = r0, r1
            else:
                g0, g1 = rng.binomial(pool, r0) / pool, rng.binomial(pool, r1) / pool
            grids = {"plain": (r0, r1, *log_lik_grid(r0, r1)), "aware": (g0, g1, *log_lik_grid(g0, g1))}
            cov, w, cov_a, w_a, shift, pred = [], [], [], [], [], []
            for pi1, n1 in zip(pi1s, n1s):
                n0 = N_Q - n1
                tn, tp = rng.binomial(n0, g[0, k]), rng.binomial(n1, g[1, k])
                acc = (tn + tp) / N_Q
                n1_pred = tp + (n0 - tn)
                lo, hi, mean_acc = interval(rng, n1_pred, *grids["plain"])
                cov.append(lo <= acc <= hi)
                w.append(winkler(acc, lo, hi))
                shift.append(mean_acc - acc)
                pred.append((1 - pi1) * (r_tilde[0, k] - g[0, k]) + pi1 * (r_tilde[1, k] - g[1, k]))
                lo, hi, _ = interval(rng, n1_pred, *grids["aware"])
                cov_a.append(lo <= acc <= hi)
                w_a.append(winkler(acc, lo, hi))
            for name, v in zip(out, (cov, w, cov_a, w_a, shift, pred)):
                out[name][rep, k] = np.mean(v)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--bags", type=int, default=20)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"N_Q={N_Q}, n_s={N_S}, classifiers={CLASSIFIERS}, rho={args.rho}")
    print(f"reps={args.reps}, bags={args.bags}, seed={args.seed}")
    print(
        f"{'P_j':>5} {'cov':>6} {'W':>6} {'cov_aw':>6} {'W_aw':>6} {'cov<.5':>6} {'sd_sh':>6} {'sd_pr':>6} "
        f"{'corr':>5} {'slope':>5} {'corr_k':>6} {'same':>5}"
    )
    rbar = np.mean(CLASSIFIERS, axis=0)
    mean_pi2 = 1 / 4  # the shift averages the bags, whose mean prevalence is about (1/2, 1/2)
    for pool in POOL_SIZES:
        o = run(rng, pool, args.reps, args.bags, args.rho)
        inv_p = 0 if pool is None else 1 / pool
        sd_pred = np.sqrt(mean_pi2 * np.sum(rbar * (1 - rbar) * (1 / N_S + inv_p)))
        sh, pr = o["shift"].ravel(), o["pred"].ravel()
        corr = np.corrcoef(sh, pr)[0, 1]
        slope = np.polyfit(pr, sh, 1)[0]
        ck = np.corrcoef(o["shift"].T)
        corr_k = ck[np.triu_indices_from(ck, 1)].mean()
        same = np.mean(np.all(o["shift"] > 0, axis=1) | np.all(o["shift"] < 0, axis=1))
        print(
            f"{'inf' if pool is None else pool:>5} {o['cov'].mean():6.3f} {o['W'].mean():6.3f} "
            f"{o['cov_aware'].mean():6.3f} {o['W_aware'].mean():6.3f} {np.mean(o['cov'] < 0.5):6.3f} "
            f"{sh.std():6.3f} {sd_pred:6.3f} {corr:5.2f} {slope:5.2f} {corr_k:6.2f} {same:5.2f}"
        )


if __name__ == "__main__":
    main()
