# Possible extensions of the UNCAWE paper

## Starting point

The paper *Uncertainty-Aware Classifier Accuracy Prediction under Prior
Probability Shift* proposes SQS-H, SQS-S, and BayCAP for estimating both the
performance of a classifier on unlabelled target data and the uncertainty of
that estimate. The current evaluation focuses mainly on vanilla accuracy,
tabular datasets, and artificially generated prior probability shift (PPS).

An extended version should go beyond increasing the number of datasets and
address at least one new scientific question. The directions below may be
combined, although they differ in ambition and cost.

## 1. Systematic interval calibration

The current evaluation mainly uses nominal 95% intervals. To determine whether
the distributions produced by the methods represent uncertainty accurately,
one could evaluate several nominal levels, such as 50%, 80%, 90%, 95%, and 99%.

The analysis would include:

- reliability plots comparing nominal and empirical coverage;
- coverage error, interval amplitude, Winkler score, and CRPS;
- coverage conditioned on shift intensity, $|U|$, number of classes, and
  quantifier quality;
- post-hoc interval calibration, using temperature for BayCAP or corrections
  learned from shifts simulated with the validation set.

This contribution would provide a more rigorous characterization of uncertainty
quality. Without a new calibration method, however, it would remain primarily
an experimental extension.

## 2. Metrics beyond vanilla accuracy

Because several methods estimate a distribution over contingency tables, they
should be able to predict any measure derived from such tables. This property
could be evaluated on macro-F1 and macro-K, which are already supported by the
code.

Nonlinear metrics introduce problems that are not visible with vanilla
accuracy:

- in general, $E[A(T)]\neq A(E[T])$;
- metric distributions may be asymmetric;
- macro-F1 is unstable when classes are rare or absent;
- a well-estimated contingency table does not guarantee calibrated intervals
  for every functional.

This extension would strengthen the metric-agnosticism claim. By itself,
however, it would probably be incremental and should be paired with a
methodological extension.

## 3. BayCAP with informative priors and tempering

BayCAP permits the incorporation of prior knowledge and adjustment of posterior
dispersion through temperature. A complete evaluation should compare:

- a uniform prior;
- a partially informative prior;
- a correctly specified prior;
- a deliberately misspecified prior;
- different concentration levels;
- different posterior temperatures.

The goal should not be merely to show the advantage of an oracle prior, but to
measure the trade-off between accuracy, sharpness, coverage, and robustness to
misspecification. Temperature should be selected without target labels, for
example through synthetic shifts built from the validation set.

This direction develops the future work stated in the paper and could produce a
*Calibrated BayCAP* variant.

## 4. Decomposition of uncertainty sources

Current intervals may aggregate distinct phenomena:

- uncertainty about target prevalence $\pi^Q$;
- uncertainty about the conditional matrix $C$;
- variability caused by finite $|U|$;
- validation-set variability;
- error caused by misspecification or PPS violations.

Controlled experiments and ablations could isolate these components. The
results could help explain why SQS-H may outperform SQS-S even though the latter
attempts to represent quantifier error more explicitly. This decomposition
integrates naturally with the Aitchison-SQS and SQS-B proposals described below.

## 5. Robustness beyond exact PPS

The current protocol generates pure PPS by resampling examples from the same
pool. Reliability under more realistic conditions could be studied by
introducing a controlled violation:

$$
Q_\lambda(X\mid Y)
=(1-\lambda)P(X\mid Y)+\lambda R(X\mid Y),
$$

with increasing $\lambda$. The experiment would measure the degradation of
point estimates and coverage as conditional shift increases.

A methodological extension could add:

- a diagnostic score for PPS compatibility;
- more conservative intervals when the diagnostic score deteriorates;
- abstention when the assumption appears incompatible with the data.

This is one of the most ambitious and potentially consequential directions,
because it changes the problem from estimation under an ideal assumption to
monitoring the reliability of that assumption.

## 6. Geometric analysis and kernel variants of SQS-S

SQS-S selects validation bags by comparing estimated prevalences through a
distance on the simplex. One could systematically study how the distance and
neighbourhood definition affect bias and variance:

- compare $L_1$, Hellinger, Jensen--Shannon, and Aitchison distances;
- measure the correspondence between proximity in predicted and true
  prevalence space;
- study sensitivity to $\gamma$;
- replace hard neighbour selection with kernel weights;
- select bandwidth adaptively according to local density.

This interpretation connects SQS-S to local regression or approximate Bayesian
computation on the simplex. The contribution becomes substantial if it yields
an adaptive rule, rather than only a comparison of distance functions.

## 7. Multiclass scalability and new domains

The evaluation could be extended to text and image data, pretrained models, and
problems with many classes. In addition to predictive quality, the analysis
should measure:

- time and memory as functions of $K$, $|V|$, $|U|$, and the number of samples;
- the cost of BayCAP and NUTS sampling;
- behaviour with rare or absent classes;
- stability when the confusion matrix is ill-conditioned.

New domains would make the empirical evidence more convincing, but should not
constitute the central contribution on their own.

## 8. Aitchison-SQS: propagating quantifier uncertainty

Aitchison-SQS is a research proposal, not an implemented method. It models
prevalences as compositional data. The isometric log-ratio transformation maps a
prevalence $\pi\in\Delta^{K-1}$ to Euclidean space:

$$
z=\operatorname{ilr}(\pi)\in\mathbb R^{K-1}.
$$

A Gaussian distribution in `ilr` coordinates induces a logistic-normal
distribution on the simplex. Given the point estimate $\hat\pi^Q=q(U)$, one may
define

$$
Z\sim\mathcal N\left(
\operatorname{ilr}(\hat\pi^Q)+\mu_e,
\Sigma
\right),
\qquad
\Pi=\operatorname{ilr}^{-1}(Z),
$$

sample prevalence vectors $\pi^{(r)}$, and draw a validation bag for each vector
on which performance is evaluated. Quantifier uncertainty is thereby propagated
to the predictive distribution of the metric.

### Estimation with validation bags

On bags with known prevalence, one can observe residuals

$$
e_b=\operatorname{ilr}(\pi_b)-\operatorname{ilr}(q(\sigma_b)),
$$

and estimate bias $\mu_e$ and covariance $\Sigma_e$. This measures empirical
quantifier error, but requires out-of-sample evaluation through splitting or
cross-fitting and may not be local to a particular target set.

### Estimation with target bootstrap

Alternatively, bootstrap samples $U^{*(b)}$ can be generated from the target,
the quantifier can be applied to each sample, and the covariance of the resulting
`ilr` coordinates can be estimated:

$$
\widehat\Sigma_U=
\operatorname{Cov}\left(
\operatorname{ilr}(q(U^{*(1)})),\ldots,
\operatorname{ilr}(q(U^{*(B)}))
\right).
$$

This estimate is local, adaptive, and label-free, but captures only sampling
uncertainty conditional on the fitted quantifier. It does not identify systematic
bias, training uncertainty, or misspecification and may therefore produce
intervals that are too narrow for stable but incorrect quantifiers.

### Hybrid variant

A complete formulation may approximately separate

$$
\Sigma_{\mathrm{total}}
\approx
\Sigma_{\mathrm{target}}+
\Sigma_{\mathrm{quantifier}}.
$$

The first component comes from bootstrapping $U$ and the second from
out-of-sample errors or quantifier-training bootstrap. Double counting must be
avoided because residuals on validation bags already contain sampling
variability.

Required baselines include direct prevalence bootstrap, isotropic Gaussian
Aitchison, shrinkage covariance, nonparametric `ilr` residuals, and a Dirichlet
distribution; the binary setting should also include a Beta distribution. The
logistic-normal is more flexible than the Dirichlet, but requires explicit zero
handling, regularization for large $K$, and empirical checks of normality.

The full formulation, caveats, and baselines are described in
[aitchison_sqs.md](aitchison_sqs.md).

## 9. SQS-B: Bayesian prevalence propagation through validation sampling

SQS-B is a research proposal that replaces the point or bootstrap prevalence
representation used by SQS-style methods with joint draws from a Bayesian
quantifier:

$$
\pi^{(r)}\sim p_q(\pi^Q\mid U,V).
$$

For every posterior draw, the method samples one target-sized validation bag
with replacement and with true prevalence approximately equal to
$\pi^{(r)}$. Applying the classifier and the performance functional to every bag
produces an empirical distribution of target-performance values.

The quantifier interface is generic: BayesianCC is a natural instantiation, but
SQS-B only requires joint posterior draws on the simplex. Marginal intervals for
individual classes are insufficient because independent sampling would discard
posterior dependence and could generate invalid prevalence vectors.

SQS-B differs from BayCAP because it propagates only the posterior over target
prevalence and uses empirical validation sampling for conditional classifier
behaviour. It also differs from Aitchison-SQS, which constructs or estimates a
distribution around a non-Bayesian point estimate. The resulting SQS-B interval
is hybrid Bayesian and empirical and should not automatically be described as a
fully Bayesian credible interval.

The current RQBS implementation already maps bootstrap prevalence estimates to
validation bags. Consequently, merely replacing bootstrap draws with posterior
draws is likely an incremental contribution. A convincing evaluation must
compare posterior-mean SQS, bootstrap RQBS, full SQS-B, BayCAP, and an oracle
using the true target prevalence, while measuring both prevalence-posterior
calibration and downstream interval quality.

The complete formulation, interpretation, limitations, and decisive ablations
are described in [sqs_b.md](sqs_b.md). SQS-B is not currently implemented.

## Recommended combinations

### Incremental and manageable extension

Evaluate multiple metrics and nominal levels, adding reliability plots,
sample-size analysis, and a systematic study of BayCAP priors and temperature.
This is the lowest-risk option, but it needs at least one methodological element
to become more than an expanded empirical evaluation.

### Methodological extension centred on Aitchison-SQS

Propose Aitchison-SQS with covariance estimated through target bootstrap and/or
validation residuals, compare parametric and nonparametric variants, and evaluate
calibration at several nominal levels. This combination is consistent with the
original contribution and directly addresses how SQS-H handles quantifier
error.

### Posterior-propagation extension

Compare SQS-B directly with bootstrap-based RQBS, BayCAP, and Aitchison-SQS at
multiple nominal coverage levels. This design isolates the effect of replacing a
bootstrap or constructed prevalence distribution with an actual Bayesian
posterior, while also comparing empirical validation sampling against BayCAP's
joint parametric propagation. This is the most natural setting in which SQS-B
could support a substantive scientific claim rather than a component
substitution.

### More ambitious extension

Combine Aitchison-SQS, SQS-B, or Calibrated BayCAP with stress tests beyond PPS,
a diagnostic score, and abstention. The scientific message would become: when
are predictive performance distributions reliable, and how can we recognize
cases in which the label-shift assumption is no longer adequate?
