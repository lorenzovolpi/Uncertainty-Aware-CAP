# Extensions under evaluation

This note collects candidate extensions of the paper
[Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
that were introduced after the proposals in [[extensions]].

**Status.** Everything below is a *hypothesis that still needs evaluation*. None
of these ideas is implemented, and none is ready to be implemented. Each needs a
feasibility check, a literature check for novelty, and, where noted, a
preliminary experiment before it can be promoted to [[extensions]].

Labels used throughout:

- **[paper]**: implemented and evaluated in the reference paper;
- **[proposal]**: documented in [[extensions]] or its linked notes, not implemented;
- **[proposal, implemented]**: a proposal implemented in UNCAWE but not yet evaluated;
- **[new]**: hypothesis introduced in this note.

## A. Estimand clarification and variance decomposition [promoted]

Promoted into [[ext4_uncertainty_budget|Extension 4]], which now contains the estimand clarification, the exact error split, the variance budget (terms (1)–(3) of the former proposal, plus the parts of the prevalence error that are fixed given $V$), the table of terms represented by each method, and the caveats. The illustrative numbers of the former proposal were replaced by the seeded toy `scripts/toys/ext4_budget.py`.

## B. Transductive BayCAP and the link to CBPE [new]

### Idea

The labels of $U$ are unknown but fixed, and the predictions $h(U)$ are observed.
Instead of evaluating $A$ on the population table $C\operatorname{diag}(\pi)$,
one could impute the latent labels instance by instance for each posterior draw
$\pi^{(r)}$:

$$
y_x^{(r)}\sim p_Q(y\mid x)\propto p_P(y\mid x)\,\frac{\pi^{(r)}_y}{\pi^P_y},
$$

then compute $A$ on the imputed labels. This leads to two observations:

- **Fixed $\pi$.** With $\pi$ fixed and calibrated posteriors, the number of
  correct predictions is Poisson-binomial. This is the CBPE construction
  ([17] in the paper, evaluated there as L-CBPE).
- **Integrated $\pi$.** Integrating over the prevalence posterior gives a
  mixture of Poisson-binomials. This joins BayCAP and CBPE in one model.

A crisp-prediction variant uses only $\hat y_x$. In that case each instance is
correct with probability equal to the *precision*,
$P(Y=\hat y\mid\hat Y=\hat y)=C_{\hat y\hat y}\pi_{\hat y}/(C\pi)_{\hat y}$,
rather than the recall $d_j$. This changes the finite-$U$ term of the budget of
[[ext4_uncertainty_budget|Extension 4]], because it conditions on observed
predictions instead of true class counts.

*Note:* the crisp-prediction variant (realised-table posterior predictive) is now part of [[ext2_baycap|Extension 2]], where it is studied as a correction of the estimand; its comparison with tempering is in [[ext5_baycap_priors_tempering|Extension 5]]. This proposal retains the soft-posterior variant and its link to CBPE.

### What needs evaluation

- whether the reweighted posteriors $p_Q(y\mid x)$ are calibrated enough for
  this to be useful, given that L-CBPE performed poorly in the paper;
- whether the soft-posterior and crisp-prediction variants differ in practice;
- behaviour for small $\lvert U\rvert$, where the transductive estimand matters
  most;
- the literature on finite-population inference and on CBPE extensions, to
  establish novelty.

## C. SQS-S as approximate Bayesian computation [new]

### Idea

SQS-S [paper] is structurally a rejection ABC sampler, with these pieces:

- **prior:** uniform over the simplex;
- **simulator:** a bag drawn from $V_2$;
- **summary statistic:** $q(\sigma)$;
- **tolerance:** set by $\gamma$.

This interpretation, already mentioned in Extension 6 of [[extensions]],
suggests concrete methodological changes rather than a comparison of distances:

- **Regression adjustment** (in the style of Beaumont et al., 2002). Fit
  $a_i\approx\alpha+\beta^\top\big(q(\sigma_i)-q(U)\big)$ on the accepted bags
  and use the adjusted values
  $$\tilde a_i=a_i-\hat\beta^\top\big(q(\sigma_i)-q(U)\big).$$
  This should reduce the bias from the finite neighbourhood, so a larger
  $\gamma$ can be used. That is a possible remedy for the wide SQS-S intervals
  reported in the paper.
- **Extra summary statistics.** Add the histogram of $h(\sigma)$, compared with
  that of $h(U)$, which is observed but currently unused by SQS. The paper's
  circularity argument applies to using $h$ as *the* quantifier, not to using
  it as an extra summary statistic. Whether this helps is an empirical question.
- **Kernel weights and adaptive tolerance.** These follow directly from the ABC
  literature, which provides known asymptotic behaviour.

### What needs evaluation

- whether regression adjustment reduces width without losing coverage;
- whether adding the $h$-prediction summary improves or biases the estimates;
- the dependence on the ABC prior. The candidate prevalences are drawn from the
  same uniform-simplex distribution used by APP to generate test sets, so the
  current evaluation may favour SQS-S. Non-uniform test shifts, for example
  concentrated near $\pi^P$ or skewed, are needed.

## D. Conformal calibration under simulated label shift [new]

### Idea

The idea is a post-hoc wrapper that can calibrate the intervals of *any* CAP
method, including baselines that undercover:

1. Generate APP bags $\{\sigma_b\}$ from labelled data, using cross-fitting so
   that the CAP method never sees the bag it is scored on.
2. For each bag, compute a nonconformity score
   $s_b=\lvert a_b-\hat a_b\rvert/\hat w_b$, where $\hat w_b$ is the method's
   own interval half-width or another scale estimate.
3. Widen the target interval to $\hat a\pm \hat q_{1-\alpha}\,\hat w$, where
   $\hat q_{1-\alpha}$ is the conformal quantile of the scores.

This would give a finite-sample marginal coverage guarantee, but *only* with
respect to the distribution used to simulate the shifts. The exchangeability
between simulated bags and the target is an assumption, not a fact. A localized
variant that weights bags by proximity in $q$-space would connect to proposal C.

### What needs evaluation

- **Novelty.** A literature check is needed, starting from Podkopaev and Ramdas
  (2021) on conformal prediction under label shift. That work operates at the
  instance level, not on accuracy estimation.
- **Mismatch.** How coverage degrades when the target shift distribution differs
  from the simulation distribution.
- **Data cost.** The cost in labelled data, and the interaction with the
  validation split of Extension 6 in [[extensions]].

## Possible combination

If these hypotheses survive evaluation, they would attach to the uncertainty
budget of [[ext4_uncertainty_budget|Extension 4]] (the former proposal A),
which is now the conceptual core:

- evaluate on Extension 1 (metrics) and on the $(\lvert U\rvert,\lvert V\rvert)$
  grid of Extension 4 of [[extensions]];
- include one calibration method, C or D.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- Beaumont, Zhang, Balding. Approximate Bayesian computation in population genetics. *Genetics*, 2002.
- Podkopaev, Ramdas. Distribution-free uncertainty quantification for classification under label shift. *UAI*, 2021.
- [[extensions]] · [[ext3_posterior_propagation]] · [[aitchison_sqs]]
