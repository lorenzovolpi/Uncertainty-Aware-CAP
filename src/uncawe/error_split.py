"""The error split of a quantification-based CAP estimate (Extension 4, ``docs/ext4_uncertainty_budget.md`` §T.2 and
§T.5; work-plan card WP-14, used by E4.1 and E4.3 and meant to be reused by E4.2).

Pure numpy: every function takes the tables of a set of bags and returns per-bag arrays. Tables are in the ``cap-ml``
layout used by the stored results (true classes on rows, predicted classes on columns), i.e. the transpose of the
paper's $T=C\\operatorname{diag}(\\pi)$; the column sums and recalls of the paper's orientation are therefore the row
sums and the diagonal over the row sums of the stored tables.

For a bag with realised table $T^Q$ (``true_cts``) and a method's mean table $\\hat T$ (``estim_cts``), both normalised
to sum to $1$:

- $\\tilde\\pi_j$ and $\\tilde d_j=\\hat T_{jj}/\\tilde\\pi_j$, the prevalence and the recalls read from $\\hat T$;
- $\\pi^Q_j$ and $c_j/n_j=T^Q_{jj}/\\pi^Q_j$, the true prevalence and the realised recalls of the bag;
- $d^{\\text{pool}}_j$, the recalls of $h$ on the test pool, which stand for the population recalls (§T.1).

The error of the estimate $\\hat A=\\operatorname{tr}\\hat T$ against the realised accuracy $A=\\operatorname{tr}T^Q$ splits
exactly into

$$\\hat A-A=\\underbrace{\\sum_j(\\tilde\\pi_j-\\pi^Q_j)\\tilde d_j}_{e_1}
+\\underbrace{\\sum_j\\pi^Q_j(\\tilde d_j-d^{\\text{pool}}_j)}_{e_2}
+\\underbrace{\\sum_j\\pi^Q_j(d^{\\text{pool}}_j-c_j/n_j)}_{e_3},$$

and the cross term is $x=\\sum_j(\\tilde\\pi_j-\\pi^Q_j)(\\tilde d_j-d^{\\text{pool}}_j)$. The four oracle estimates of
§T.5 are $\\hat A_{\\pi,d}=\\sum_j\\pi_jd_j$ with $\\pi\\in\\{\\tilde\\pi,\\pi^Q\\}$ and $d\\in\\{\\tilde d,d^{\\text{pool}}\\}$;
their errors against $A$ are $e_1+e_2+e_3$, $e_2+e_3$, $e_1-x+e_3$ and $e_3$.

Conventions (user's decision of 2026-11-09): a class absent from the bag ($\\pi^Q_j=0$, i.e. $n_j=0$) contributes $0$ to
$e_3$ (its realised recall is set to $d^{\\text{pool}}_j$, and its weight is $0$ anyway); a class with an empty estimated
column ($\\tilde\\pi_j=0$) gets $\\tilde d_j:=d^{\\text{pool}}_j$, so its whole contribution goes to $e_1$; the number of
such classes is returned per bag (``n_empty``). Both conventions keep the identity exact.

The split uses the true prevalence of the bag and the labels of the pool, so it is an analysis, not a method; the three
oracle cells are references.
"""

import numpy as np

PARTS = ("e1", "e2", "e3")
# the four cells of the oracle 2x2 (§T.5): (prevalence, recalls), "est" for the method's, "oracle" for the true ones
CELLS = ("est_est", "oracle_est", "est_oracle", "oracle_oracle")
# the per-bag outputs of ``split``, in order
SPLIT_KEYS = ("e1", "e2", "e3", "x", "estimate", "true_acc", "pool_acc", "total", "identity_gap")
SPLIT_KEYS += tuple(f"acc_{cell}" for cell in CELLS) + ("n_empty",)


def _as_tables(cts) -> np.ndarray:
    """Return ``cts`` as a float array of shape ``(B, K, K)``, each table normalised to sum to $1$."""
    cts = np.asarray(cts, dtype=float)
    if cts.ndim == 2:
        cts = cts[None]
    if cts.ndim != 3 or cts.shape[1] != cts.shape[2]:
        raise ValueError(f"Expected tables of shape (B, K, K), got {cts.shape}")
    totals = cts.sum(axis=(1, 2), keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("Every table must have a positive total")
    return cts / totals


def props_and_recalls(cts, fill=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the prevalences, the recalls and the mask of empty classes of tables in the ``cap-ml`` layout.

    ``cts`` has shape ``(K, K)`` or ``(B, K, K)``, true classes on rows; each table is normalised to sum to $1$. The
    prevalence of class $\\omega_j$ is the sum of row $j$ (column $j$ of the paper's table), the recall is the diagonal
    entry over that sum. Where the prevalence is $0$ the recall is ``fill[j]`` (a vector of length $K$), or NaN if
    ``fill`` is None. Returns arrays of shape ``(B, K)``.
    """
    tables = _as_tables(cts)
    props = tables.sum(axis=2)
    diag = np.diagonal(tables, axis1=1, axis2=2)
    empty = props <= 0
    if fill is None:
        fill_values = np.full_like(props, np.nan)
    else:
        fill_values = np.broadcast_to(np.asarray(fill, dtype=float), props.shape)
    recalls = np.divide(diag, props, out=np.array(fill_values, dtype=float), where=~empty)
    return props, recalls, empty


def pool_recalls(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Return $d^{\\text{pool}}$, the recalls of $h$ on the test pool, from its labels and predictions.

    Raises if a class has no instance in the pool, since its recall is then undefined (and no bag can contain it).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    counts = np.bincount(y_true, minlength=n_classes).astype(float)
    if counts.shape[0] != n_classes:
        raise ValueError(f"Labels outside 0..{n_classes - 1}")
    if np.any(counts == 0):
        raise ValueError(f"Classes {np.flatnonzero(counts == 0).tolist()} have no instance in the pool")
    correct = np.bincount(y_true[y_true == y_pred], minlength=n_classes).astype(float)
    return correct / counts


def split(estim_cts, true_cts, d_pool) -> dict[str, np.ndarray]:
    """Return the error split of each bag (§T.2), its cross term and its four oracle estimates (§T.5).

    ``estim_cts`` and ``true_cts`` have shape ``(B, K, K)`` (or ``(K, K)`` for one bag), ``cap-ml`` layout; ``d_pool``
    has length $K$. Every returned array has length $B$:

    - ``e1``, ``e2``, ``e3``: the parts; ``x``: the cross term;
    - ``estimate`` ($\\operatorname{tr}\\hat T$), ``true_acc`` ($A$), ``pool_acc`` ($\\sum_j\\pi^Q_jd^{\\text{pool}}_j$);
    - ``total``: $\\hat A-A$; ``identity_gap``: $|e_1+e_2+e_3-(\\hat A-A)|$, zero up to floating point;
    - ``acc_<cell>`` for the cells of ``CELLS``: the estimate $\\sum_j\\pi_jd_j$ of that cell;
    - ``n_empty``: the classes with an empty estimated column ($\\tilde\\pi_j=0$, recall set to $d^{\\text{pool}}_j$).
    """
    d_pool = np.asarray(d_pool, dtype=float)
    est = _as_tables(estim_cts)
    true = _as_tables(true_cts)
    if est.shape != true.shape or est.shape[1] != d_pool.shape[0]:
        raise ValueError(f"Shapes differ: estim {est.shape}, true {true.shape}, d_pool {d_pool.shape}")
    if not np.all(np.isfinite(d_pool)):
        raise ValueError("d_pool must be finite")

    pi_t, d_t, empty = props_and_recalls(est, fill=d_pool)
    pi_q, r_q, _ = props_and_recalls(true, fill=d_pool)

    e1 = np.sum((pi_t - pi_q) * d_t, axis=1)
    e2 = np.sum(pi_q * (d_t - d_pool), axis=1)
    e3 = np.sum(pi_q * (d_pool - r_q), axis=1)
    x = np.sum((pi_t - pi_q) * (d_t - d_pool), axis=1)

    estimate = np.trace(est, axis1=1, axis2=2)
    true_acc = np.trace(true, axis1=1, axis2=2)
    total = estimate - true_acc
    pool_acc = np.sum(pi_q * d_pool, axis=1)
    return {
        "e1": e1,
        "e2": e2,
        "e3": e3,
        "x": x,
        "estimate": estimate,
        "true_acc": true_acc,
        "pool_acc": pool_acc,
        "total": total,
        "identity_gap": np.abs(e1 + e2 + e3 - total),
        "acc_est_est": np.sum(pi_t * d_t, axis=1),
        "acc_oracle_est": np.sum(pi_q * d_t, axis=1),
        "acc_est_oracle": np.sum(pi_t * d_pool, axis=1),
        "acc_oracle_oracle": pool_acc,
        "n_empty": empty.sum(axis=1),
    }


def rms(values) -> float:
    """Root mean square of ``values``: $\\sqrt{\\frac1B\\sum_b v_b^2}$ (NaN if empty)."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values**2))) if values.size else np.nan


def largest_part(rms_parts) -> int:
    """Return $1$, $2$ or $3$, the part with the largest RMS among ``(rms_e1, rms_e2, rms_e3)`` (0 if all are NaN)."""
    values = np.asarray(rms_parts, dtype=float)
    if np.all(np.isnan(values)):
        return 0
    return int(np.nanargmax(values)) + 1
