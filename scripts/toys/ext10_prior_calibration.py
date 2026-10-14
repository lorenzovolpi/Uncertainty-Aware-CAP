"""Toy simulation for Extension 10: calibration on average of well- and misspecified priors.

Binary problem with a perfect, known classifier, so the observed data are the class counts
$n\\sim\\operatorname{Bin}(N,\\pi_1)$ and the posterior of $\\pi_1$ under a Beta prior is conjugate.
For each bag-generating distribution $G$ and prior, the script draws $\\pi_1\\sim G$, simulates the
counts, computes the equal-tailed 95% credible interval, and reports coverage and mean width
overall and for $\\pi_1\\in[0.4,0.6)$.

Reproduces the table in ``docs/ext10_baycap_priors_tempering.md`` (section "Well-specified prior and calibration on average").

Usage: ``uv run python scripts/toys/ext10_prior_calibration.py [--bags 200000] [--seed 0]``
"""

import argparse

import numpy as np
from scipy.stats import beta

N = 20
ALPHA = 0.05
CONFIGS = [
    # (G, prior), both as Beta parameters
    ((1, 1), (1, 1)),
    ((1, 1), (2, 8)),
    ((2, 8), (2, 8)),
    ((2, 8), (1, 1)),
    ((2, 8), (8, 2)),
]


def simulate(rng: np.random.Generator, g: tuple, prior: tuple, n_bags: int) -> dict:
    pi1 = rng.beta(*g, n_bags)
    n = rng.binomial(N, pi1)
    a, b = prior[0] + n, prior[1] + N - n
    lo, hi = beta.ppf(ALPHA / 2, a, b), beta.ppf(1 - ALPHA / 2, a, b)
    covered = (lo <= pi1) & (pi1 <= hi)
    mid = (pi1 >= 0.4) & (pi1 < 0.6)
    return {
        "coverage": covered.mean(),
        "width": (hi - lo).mean(),
        "coverage_mid": covered[mid].mean() if mid.any() else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bags", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"N={N}, bags={args.bags}, seed={args.seed}, nominal={1 - ALPHA:.2f}")
    print(f"{'G':>10} {'prior':>10} {'coverage':>9} {'width':>7} {'cov[0.4,0.6)':>13}")
    for g, prior in CONFIGS:
        r = simulate(rng, g, prior, args.bags)
        print(
            f"{'Beta' + str(g):>10} {'Beta' + str(prior):>10} "
            f"{r['coverage']:9.3f} {r['width']:7.3f} {r['coverage_mid']:13.3f}"
        )


if __name__ == "__main__":
    main()
