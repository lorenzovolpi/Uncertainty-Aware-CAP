"""Toy simulation for Extension 2: the empirical-Bayes prior of BayCAP on binary problems.

Binary problem in the paper's notation, with recalls $r_0=C_{00}$ and $r_1=C_{11}$, sized like
``mammographic`` in ``main``: $n^s_j=141$ validation instances per class and target bags of
$N^Q=1000$ instances. Each replication plays the role of one dataset/classifier pair:

1. the validation counts $f_j\\sim\\operatorname{Bin}(n^s_j,r_j)$ inform $C$;
2. the prior on each recall is one of
   - ``uniform``: $\\operatorname{Beta}(1,1)$, the $\\operatorname{Dir}(\\mathbf 1)$ prior of BayCAP;
   - ``EB``: $\\operatorname{Beta}(\\hat\\nu\\hat m,\\hat\\nu(1-\\hat m))$, fitted on the counts with
     ``fit_beta_binomial`` of BayCAP (``src/uncawe/methods/baycap.py``);
   - ``EB-nu*``: the EB prior with $\\nu^*$ from
     $1/(\\nu^*+1)=1/(\\hat\\nu+1)+\\operatorname{sd}(\\hat m)^2/(\\hat m(1-\\hat m))$, which adds the
     uncertainty on $\\hat m$ to the spread of the prior;
3. ``BAGS`` target bags are drawn with UPP, $\\pi_1\\sim U(0,1)$ and $\\operatorname{round}(N^Q\\pi_j)$
   instances of class $j$, as ``quapy``'s ``sampling_index`` does, either from the population
   (``pool=inf``) or with replacement from a finite test pool drawn once per replication
   (``pool=249``, the size of $U$ on ``mammographic``: $125$ and $124$ instances per class), so that
   every bag of the replication sees the recalls of $h$ on the pool instead of $r_j$ (cause 4);
4. the posterior over $(C,\\pi^Q)$ is computed as in BayCAP (uniform prior on $\\pi^Q$): draws of $C$
   from its validation posterior are paired with a grid over $\\pi_1$ and weighted by the target
   likelihood $\\operatorname{Bin}(n^t_1\\mid N^Q,(1-r_0)(1-\\pi_1)+r_1\\pi_1)$;
5. two 95% intervals for accuracy are built from the resampled draws: the population pushforward
   $r_0\\pi_0+r_1\\pi_1$ and the realised pushforward (true classes behind the predictions drawn with
   $\\rho_{ij}\\propto C_{ij}\\pi_j$), and both are scored against the realised accuracy of the bag.

The table reports, for each scenario of true recalls, pool and prior: the share of replications with
$\\hat\\nu$ at its upper bound $10^4$ and the median $\\hat\\nu$; coverage of the population (``cov_pop``)
and realised (``cov_real``) intervals; their mean amplitude; and the mean Winkler score of the realised
interval at $\\alpha=0.05$. Toy results, not results of ``main``.

Usage: ``uv run python scripts/toys/ext2_eb_binary.py [--reps 200] [--bags 20] [--seed 0]``
"""

import argparse

import numpy as np

from uncawe.methods.baycap import EB_NU_BOUNDS, fit_beta_binomial

N_Q = 1000
N_S = (141, 141)
POOL = (125, 124)
SCENARIOS = {"equal": (0.83, 0.83), "close": (0.80, 0.86), "far": (0.70, 0.90)}
PRIORS = ("uniform", "EB", "EB-nu*")
ALPHA = 0.05
GRID = np.linspace(0, 1, 201)
M_C = 300  # draws of C from its validation posterior, shared by the bags of a replication
M_OUT = 2000  # resampled posterior draws used to build the intervals


def prior_params(prior, f, n):
    """Return the Beta parameters (a, b) of the prior on each recall, and the fitted nu (NaN if none)."""
    if prior == "uniform":
        return 1.0, 1.0, np.nan
    fit = fit_beta_binomial(f, n)
    nu = fit.nu
    if prior == "EB-nu*" and np.isfinite(fit.sd_m):
        nu = 1 / (1 / (fit.nu + 1) + fit.sd_m**2 / (fit.m * (1 - fit.m))) - 1
    return nu * fit.m, nu * (1 - fit.m), fit.nu


def winkler(y, lo, hi):
    return hi - lo + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def bag_intervals(rng, n1_pred, r0, r1, log_p, log_q):
    """Population and realised 95% intervals for one bag, given the draws of C of the replication."""
    logw = n1_pred * log_p + (N_Q - n1_pred) * log_q
    w = np.exp(logw - logw.max()).ravel()
    idx = rng.choice(w.size, M_OUT, p=w / w.sum())
    i, j = np.unravel_index(idx, log_p.shape)
    c00, c11, pi1 = r0[i], r1[i], GRID[j]
    pop = c00 * (1 - pi1) + c11 * pi1
    rho11 = c11 * pi1 / ((1 - c00) * (1 - pi1) + c11 * pi1)
    rho00 = c00 * (1 - pi1) / (c00 * (1 - pi1) + (1 - c11) * pi1)
    realised = (rng.binomial(n1_pred, rho11) + rng.binomial(N_Q - n1_pred, rho00)) / N_Q
    q = [ALPHA / 2, 1 - ALPHA / 2]
    return np.quantile(pop, q), np.quantile(realised, q)


def run(rng, recalls, finite_pool, n_reps, n_bags):
    """Simulate one scenario and pool setting; return the scores of each prior and the fitted nu."""
    scores = {p: {"cov_pop": [], "cov_real": [], "amp_pop": [], "amp_real": [], "W_real": []} for p in PRIORS}
    nus = []
    r = np.asarray(recalls)
    n = np.asarray(N_S)
    for _ in range(n_reps):
        f = rng.binomial(n, r)
        # recalls of h on the test pool: the bags of this replication are drawn from it
        r_pool = rng.binomial(np.asarray(POOL), r) / np.asarray(POOL) if finite_pool else r
        pi1s = rng.uniform(size=n_bags)
        bags = []
        for pi1 in pi1s:
            n1 = round(N_Q * pi1)
            n0 = N_Q - n1
            tn, tp = rng.binomial(n0, r_pool[0]), rng.binomial(n1, r_pool[1])
            bags.append((tp + (n0 - tn), (tn + tp) / N_Q))
        for prior in PRIORS:
            a, b, nu_hat = prior_params(prior, f, n)
            if prior == "EB":
                nus.append(nu_hat)
            r0 = rng.beta(a + f[0], b + n[0] - f[0], M_C)
            r1 = rng.beta(a + f[1], b + n[1] - f[1], M_C)
            p_pred1 = np.clip((1 - r0)[:, None] * (1 - GRID) + r1[:, None] * GRID, 1e-300, 1 - 1e-16)
            log_p, log_q = np.log(p_pred1), np.log1p(-p_pred1)
            s = scores[prior]
            for n1_pred, acc_real in bags:
                (lo, hi), (plo, phi) = bag_intervals(rng, n1_pred, r0, r1, log_p, log_q)
                s["cov_pop"].append(lo <= acc_real <= hi)
                s["cov_real"].append(plo <= acc_real <= phi)
                s["amp_pop"].append(hi - lo)
                s["amp_real"].append(phi - plo)
                s["W_real"].append(winkler(acc_real, plo, phi))
    return {p: {k: float(np.mean(v)) for k, v in s.items()} for p, s in scores.items()}, np.asarray(nus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--bags", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"N_Q={N_Q}, n_s={N_S}, pool={POOL}, reps={args.reps}, bags={args.bags}, seed={args.seed}")
    header = (
        f"{'scenario':>8} {'r0,r1':>10} {'pool':>4} {'nu@bound':>8} {'med_nu':>8} {'prior':>8} "
        f"{'cov_pop':>7} {'cov_real':>8} {'amp_pop':>7} {'amp_real':>8} {'W_real':>7}"
    )
    print(header)
    for name, recalls in SCENARIOS.items():
        for finite_pool in (False, True):
            res, nus = run(rng, recalls, finite_pool, args.reps, args.bags)
            at_bound = np.mean(np.isclose(nus, EB_NU_BOUNDS[1], rtol=1e-4))
            for prior in PRIORS:
                s = res[prior]
                print(
                    f"{name:>8} {recalls[0]:.2f},{recalls[1]:.2f} {'249' if finite_pool else 'inf':>4} "
                    f"{at_bound:8.2f} {np.median(nus):8.1f} {prior:>8} {s['cov_pop']:7.3f} {s['cov_real']:8.3f} "
                    f"{s['amp_pop']:7.3f} {s['amp_real']:8.3f} {s['W_real']:7.3f}"
                )


if __name__ == "__main__":
    main()
