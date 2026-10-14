"""Time ACC_BayesCAP.sample_posterior on the JAX platform set by JAX_PLATFORMS (synthetic counts, seed 0).

Usage: JAX_PLATFORMS=cpu uv run python scripts/bench/baycap_platform.py
The first bag includes JIT compilation; next_mean averages the remaining three bags.
"""
import time, jax, numpy as np
from cap.models.confidence import ACC_BayesCAP
from sklearn.metrics import accuracy_score
rng = np.random.default_rng(0)
for K in (2, 10, 26):
    m = ACC_BayesCAP(accuracy_score, num_samples=1000, num_warmup=500, random_state=0)
    m.val_ct = rng.integers(5, 200, size=(K, K)) + np.eye(K, dtype=int) * 500
    ts = []
    for b in range(4):
        cnt = rng.multinomial(1000, rng.dirichlet(np.ones(K)))
        t = time.time(); s = m.sample_posterior(cnt, np.ones(K)); jax.block_until_ready(s); ts.append(time.time() - t)
    print(jax.devices()[0].platform, f"K={K} first={ts[0]:.1f}s next_mean={np.mean(ts[1:]):.1f}s", flush=True)
