"""Toy simulation for Extension 2: the corrected strength $\\nu^*$ against a hierarchical prior on the recalls.

$K$ classes, $n^s_j$ validation instances per class with fixed true recalls $r_j$, and target bags of
$N^Q=1000$ instances with UPP prevalences $\\pi\\sim\\operatorname{Dir}(\\mathbf 1_K)$ and multinomial class
counts. To compare the priors on the recalls without NUTS, the class counts of each bag are taken
as known (as in the check of E2.4): the interval reflects only the uncertainty on the recalls and the
bag-level spread, not the estimation of $\\pi^Q$. Under a Dirichlet prior on each column of $C$ the
recall $C_{jj}$ has a Beta prior and, given $V$, the Beta posterior
$\\operatorname{Beta}(f_j+a,\\,n_j-f_j+b)$, so the off-diagonal entries are not needed. Priors:

- ``uniform``: $\\operatorname{Dir}(\\mathbf 1)$, i.e. $\\operatorname{Beta}(1,K-1)$ on each recall;
- ``EB``: $\\operatorname{Beta}(\\hat\\nu\\hat m,\\hat\\nu(1-\\hat m))$, fitted with ``fit_beta_binomial`` of BayCAP;
- ``EB-nu*``: the EB prior with the corrected strength $\\nu^*$ of §T.10;
- ``H-*``: hierarchical, $(m,\\nu)$ with a prior of their own and the posterior
  $p(m,\\nu\\mid V)\\propto p(m,\\nu)\\prod_j\\operatorname{BetaBinomial}(f_j\\mid n_j,\\nu m,\\nu(1-m))$ computed on a
  grid ($m$ uniform; $\\nu\\in[10^{-1},10^4]$); each draw takes $(m,\\nu)$ from the grid and then the
  recalls from their conjugate posterior given $(m,\\nu)$. Three priors on $\\nu$:
  ``H-nu^-3/2`` ($p(\\nu)\\propto\\nu^{-3/2}$, i.e. uniform on $1/\\sqrt\\nu$), ``H-logunif`` ($\\propto1/\\nu$)
  and ``H-flat`` ($\\propto1$, which favours large $\\nu$).

For each draw of the recalls, the realised pushforward draws $\\operatorname{Bin}(n^Q_j,r_j)$ correct
instances of each class; the 95% interval of the resulting accuracy is scored against the realised
accuracy of the bag. The table reports, per $K$, scenario of true recalls and prior: the share of
replications with $\\hat\\nu$ at its upper bound, coverage, mean amplitude and mean Winkler score.
Toy results, not results of ``main``.

Usage: ``uv run python scripts/toys/ext2_hierarchical.py [--reps 200] [--bags 20] [--seed 0]``
"""

import argparse

import numpy as np
from scipy.special import betaln, logsumexp

from uncawe.methods.baycap import EB_NU_BOUNDS, fit_beta_binomial

N_Q = 1000
N_S = 141  # validation instances per class, as on mammographic
KS = (2, 3, 7, 26)
SCENARIOS = {"equal": (0.83, 0.83), "close": (0.80, 0.86), "far": (0.70, 0.90)}  # recalls evenly spaced in [lo, hi]
ALPHA = 0.05
M_DRAWS = 1000
GRID_M = np.linspace(0.0025, 0.9975, 200)
GRID_LN_NU = np.linspace(np.log(1e-1), np.log(EB_NU_BOUNDS[1]), 200)
# log-density of each prior on nu with respect to ln(nu), the grid measure
NU_PRIORS = {"H-nu^-3/2": -0.5 * GRID_LN_NU, "H-logunif": 0 * GRID_LN_NU, "H-flat": GRID_LN_NU}
PRIORS = ("uniform", "EB", "EB-nu*", *NU_PRIORS)


def winkler(y, lo, hi):
    return hi - lo + (2 / ALPHA) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def grid_loglik(f, n):
    """Beta-binomial log-likelihood of every (m, nu) of the grid, without the binomial coefficients."""
    m, nu = GRID_M[:, None, None], np.exp(GRID_LN_NU)[None, :, None]
    a, b = nu * m, nu * (1 - m)
    return np.sum(betaln(f + a, n - f + b) - betaln(a, b), axis=-1)


def recall_draws(rng, prior, f, n, k, loglik):
    """Draws of the K recalls from their posterior given V, shape (M_DRAWS, K); and the fitted nu."""
    if prior == "uniform":
        return rng.beta(1 + f, k - 1 + n - f, (M_DRAWS, k)), np.nan
    if prior in ("EB", "EB-nu*"):
        fit = fit_beta_binomial(f, n)
        nu = fit.nu
        if prior == "EB-nu*" and np.isfinite(fit.sd_m):
            nu = 1 / (1 / (fit.nu + 1) + fit.sd_m**2 / (fit.m * (1 - fit.m))) - 1
        a, b = nu * fit.m, nu * (1 - fit.m)
        return rng.beta(a + f, b + n - f, (M_DRAWS, k)), fit.nu
    logpost = (loglik + NU_PRIORS[prior][None, :]).ravel()
    w = np.exp(logpost - logsumexp(logpost))
    i, j = np.unravel_index(rng.choice(w.size, M_DRAWS, p=w), loglik.shape)
    m, nu = GRID_M[i][:, None], np.exp(GRID_LN_NU[j])[:, None]
    return rng.beta(nu * m + f, nu * (1 - m) + n - f), np.nan


def run(rng, k, lo, hi, n_reps, n_bags):
    r = np.linspace(lo, hi, k)
    n = np.full(k, N_S)
    scores = {p: {"cov": [], "amp": [], "W": []} for p in PRIORS}
    nus = []
    for _ in range(n_reps):
        f = rng.binomial(n, r)
        loglik = grid_loglik(f, n)
        bags = []
        for _ in range(n_bags):
            counts = rng.multinomial(N_Q, rng.dirichlet(np.ones(k)))
            bags.append((counts, rng.binomial(counts, r).sum() / N_Q))
        for prior in PRIORS:
            draws, nu_hat = recall_draws(rng, prior, f, n, k, loglik)
            if prior == "EB":
                nus.append(nu_hat)
            s = scores[prior]
            for counts, acc in bags:
                realised = rng.binomial(counts, draws).sum(axis=1) / N_Q
                lo_i, hi_i = np.quantile(realised, [ALPHA / 2, 1 - ALPHA / 2])
                s["cov"].append(lo_i <= acc <= hi_i)
                s["amp"].append(hi_i - lo_i)
                s["W"].append(winkler(acc, lo_i, hi_i))
    return {p: {key: float(np.mean(v)) for key, v in s.items()} for p, s in scores.items()}, np.asarray(nus)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--bags", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"N_Q={N_Q}, n_s={N_S}, reps={args.reps}, bags={args.bags}, seed={args.seed}")
    print(f"{'K':>3} {'scenario':>8} {'nu@bound':>8} {'prior':>10} {'cov':>6} {'amp':>6} {'W':>6}")
    for k in KS:
        for name, (lo, hi) in SCENARIOS.items():
            res, nus = run(rng, k, lo, hi, args.reps, args.bags)
            at_bound = np.mean(np.isclose(nus, EB_NU_BOUNDS[1], rtol=1e-4))
            for prior in PRIORS:
                s = res[prior]
                print(f"{k:3d} {name:>8} {at_bound:8.2f} {prior:>10} {s['cov']:6.3f} {s['amp']:6.3f} {s['W']:6.3f}")


if __name__ == "__main__":
    main()
