# Extension 1: metrics beyond vanilla accuracy

## Status and role

Research proposal; analysis only. The `main` experiment already computes vanilla accuracy, macro-F1, and macro-K for every method, so this extension requires no new runs.

Extension 1 is a **cross-cutting evaluation axis**. It fixes the metric protocol (definitions, table convention, edge cases, point estimator, and evaluation measures) that every other extension reuses. It also contributes one analysis of its own: whether interval quality depends on how strongly each metric depends on target prevalence (Section "Prevalence dependence").

## Notation

We follow the convention of the reference paper. Let

$$
C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j),
\qquad
T=C\operatorname{diag}(\pi),
\qquad
T_{ij}=Q(\hat Y=\omega_i,\,Y=\omega_j),
$$

so rows index predicted classes and columns index true classes. For class $k$:

$$
\mathrm{TP}_k=T_{kk}=C_{kk}\pi_k,
\qquad
\mathrm{FP}_k=\sum_{j\neq k}T_{kj}=\sum_{j\neq k}C_{kj}\pi_j,
\qquad
\mathrm{FN}_k=\pi_k-C_{kk}\pi_k.
$$

*Implementation note:* `cap.error` stores the transpose (true classes on rows, predicted classes on columns); the methods in `cap` and UNCAWE already return tables in that layout. Swapping the two conventions would leave F1 unchanged but turn macro-K into macro-markedness, so the orientation must be checked whenever a new method produces tables.

## Metrics

| Code key | Name in documents | Definition |
|---|---|---|
| `vanilla_accuracy` | Acc | $\sum_k T_{kk}$ |
| `macro-F1` | macro-F1 (multiclass); F1 (binary) | mean over classes of $\mathrm{F1}_k$; binary: F1 of the positive class only |
| `macro-K` | macro-K | mean over classes of $\mathrm{TPR}_k+\mathrm{TNR}_k-1$ |

Macro-K is the macro-averaged informedness (Youden's $J$); the name macro-K is kept throughout.

For binary data, the code computes F1 of class $1$ (positive class) rather than macro-F1. Binary F1 and multiclass macro-F1 are therefore different quantities. They must be reported separately and never pooled in one table or ranking, and the positive class of every binary dataset must be stated. Macro-K has no such issue: for $K=2$ both classes have the same informedness.

## Point estimate and intervals

As in the paper, every sample-based method reports the empirical mean of the metric samples,

$$
\hat A=\frac1m\sum_{r=1}^m A\!\left(T^{(r)}\right),
$$

and the $100(1-\alpha)\%$ percentile interval of the same samples. Each metric is applied to every sampled table; for nonlinear metrics $\hat A\neq A(\bar T)$ in general, where $\bar T=\frac1m\sum_r T^{(r)}$. Methods that do not produce tables are refitted separately for each metric.

Evaluation follows the paper: AE for point estimates; coverage, amplitude, and Winkler score at $\alpha=0.05$ for intervals.

## Edge-case conventions

These conventions shape the tails of the metric distributions on rare or absent classes and must be reported:

- $\mathrm{F1}_k=1$ when $2\mathrm{TP}_k+\mathrm{FP}_k+\mathrm{FN}_k=0$ (class $k$ neither present nor predicted);
- $\mathrm{F1}_k=0$ when class $k$ is predicted but absent from the true labels;
- informedness becomes $2\,\mathrm{TNR}_k-1$ when class $k$ has no positives and $2\,\mathrm{TPR}_k-1$ when it has no negatives;
- methods use a smoothed table ($T+10^{-5}$, renormalized) for macro-F1 during internal selection; evaluation always uses the raw table.

## Prevalence dependence

Under PPS, $C$ is shared between source and target, so the three metrics depend on the target prevalence $\pi$ to different degrees:

$$
\operatorname{Acc}(C,\pi)=\operatorname{diag}(C)^\top\pi
\qquad\text{(linear in }\pi\text{)},
$$

$$
\mathrm{F1}_k(C,\pi)=\frac{2\,C_{kk}\pi_k}{(C\pi)_k+\pi_k}
\qquad\text{(ratio of linear functions of }\pi\text{)},
$$

$$
\mathrm{TPR}_k=C_{kk},
\qquad
\mathrm{TNR}_k=1-\sum_{j\neq k}C_{kj}\,\frac{\pi_j}{1-\pi_k}.
$$

For $K=2$, $\mathrm{TNR}_1=C_{00}$ and macro-K reduces to $J=C_{00}+C_{11}-1$, which **does not depend on $\pi$**. For $K>2$, macro-K depends on $\pi$ only through the relative proportions of the classes other than $k$, so weakly.

**Hypothesis (new, untested).** If the dominant error source is the estimate of $\pi$, interval quality should degrade most for Acc and macro-F1 and least for macro-K (not at all for binary macro-K). If the dominant source is the estimate of $C$, the three metrics should degrade comparably. Comparing the metrics on the same bags therefore separates prevalence error from conditional-rate error without new runs.

## Analyses

All analyses use the existing `main` results, which store the true tables (`true_cts`), the mean estimated tables (`estim_cts`), and the metric samples per method.

1. **Per-metric quality.** AE, coverage, amplitude, and Winkler score for every method and metric; binary F1 and multiclass macro-F1 kept separate.
2. **Nonlinearity gap.** $\lvert A(\bar T)-\hat A\rvert$ per bag (table-based methods only), compared with the AE of the method, to check whether the nonlinearity is negligible relative to estimation error.
3. **Asymmetry.** Skewness of the metric samples and the position of $\hat A$ inside the interval.
4. **Rare and absent classes.** Flag bags where a class is absent or below a small prevalence threshold; report the metrics above separately on these bags.
5. **Prevalence dependence.** For each bag, compute the prevalence error $\lVert\hat\pi-\pi\rVert_1$ from the column sums of `estim_cts` and `true_cts`; relate it to the per-metric AE and coverage, and test the ordering Acc / macro-F1 vs macro-K stated above, separately for binary and multiclass data. This applies only to methods that produce tables, since the others store no `estim_cts`.

## Out of scope (possible future work)

- recomputing a true binary macro-F1 (requires reruns, since individual table samples are not stored and non-table methods are refitted per metric);
- adding further metrics such as balanced accuracy or Cohen's $\kappa$.
