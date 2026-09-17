# SQS-B: Bayesian prevalence propagation through validation sampling

## Status and scope

SQS-B is a research proposal. It is not currently implemented in UNCAWE or
established by the reference paper.

The proposal extends Surrogate Quantifier-based Sampling (SQS) by replacing a
single target-prevalence estimate, or a bootstrap distribution of such
estimates, with joint prevalence vectors drawn from the posterior of a Bayesian
quantifier. Each posterior draw is propagated through the empirical validation
sampling mechanism of SQS-H.

The intended contribution is not a new Bayesian quantifier. SQS-B defines a
generic way of converting a posterior over target prevalence into a predictive
distribution over classifier-performance measures.

## Motivation

Let $V$ be a labelled validation set, $U$ an unlabelled target set, and $h$ the
classifier whose target performance is to be estimated. Under prior probability
shift (PPS),

$$
P(X\mid Y)=Q(X\mid Y),
$$

while the class prevalences may change. SQS-H first estimates the target
prevalence $\hat\pi^Q=q(U)$ and then draws validation bags whose true class
prevalence approximately matches $\hat\pi^Q$. Their performance values form an
empirical predictive distribution.

This procedure conditions all validation bags on one estimated prevalence
vector. It therefore does not, in its original formulation, propagate
uncertainty about $\pi^Q$. The current `RQBS` implementation used by UNCAWE
already addresses part of this limitation by obtaining a bootstrap distribution
of quantifier estimates and drawing a validation bag for every bootstrap
prevalence vector.

SQS-B replaces that bootstrap distribution with a posterior supplied directly
by a Bayesian quantifier:

$$
\pi^{(r)}\sim p_q(\pi^Q\mid U,V),
\qquad r=1,\ldots,m.
$$

This preserves posterior asymmetry, dependence between classes, multimodality,
and boundary behaviour whenever they are represented by the quantifier.

## Required quantifier interface

SQS-B is defined for any quantifier that can return joint posterior draws

$$
\Pi=
\left\{\pi^{(1)},\ldots,\pi^{(m)}\right\},
\qquad
\pi^{(r)}\in\Delta^{K-1}.
$$

The method does not require a specific posterior family or inference engine.
BayesianCC is a natural first instantiation because it returns a posterior over
target prevalence under a black-box label-shift model, but it is not part of the
definition of SQS-B.

Joint posterior samples are essential. Sampling each class independently from
a marginal credible interval would discard posterior dependence and could
produce vectors outside the simplex. A collection of marginal lower and upper
bounds is therefore not a sufficient quantifier interface.

## Algorithm

Given a performance functional $A$, a posterior quantifier $q$, a labelled
validation set $V$, and an unlabelled target set $U$:

1. Fit or condition $q$ using the labelled data required by the quantifier.
2. Obtain $m$ joint draws
   $\{\pi^{(r)}\}_{r=1}^m$ from $p_q(\pi^Q\mid U,V)$.
3. For every $\pi^{(r)}$, draw one bag $\sigma^{(r)}$ with replacement from
   $V$, with $|\sigma^{(r)}|=|U|$ and true class prevalence approximately equal
   to $\pi^{(r)}$. Approximation is unavoidable when $|U|\pi^{(r)}$ is not
   integer-valued.
4. Apply $h$ to $\sigma^{(r)}$ and construct its contingency table $T^{(r)}$.
5. Compute one performance draw

   $$
   a^{(r)}=A(T^{(r)}).
   $$

6. Return the Monte Carlo mean

   $$
   \widehat A_{\mathrm{SQS-B}}
   =\frac{1}{m}\sum_{r=1}^m a^{(r)}
   $$

   as the point estimate and the empirical
   $[\alpha/2,1-\alpha/2]$ quantiles of the $a^{(r)}$ values as the interval
   estimate.

The metric must be evaluated separately on every sampled contingency table.
For nonlinear metrics, in general,

$$
E[A(T)]\ne A(E[T]).
$$

One validation bag per posterior draw is the default SQS-B definition. Nested
sampling with multiple bags per draw may be useful for decomposing sources of
variance, but it is an experimental diagnostic rather than part of the base
method.

## Statistical interpretation

SQS-B combines two different uncertainty mechanisms:

- posterior uncertainty about the target prevalence, represented by
  $p_q(\pi^Q\mid U,V)$;
- empirical variability of classifier performance conditional on prevalence,
  represented by target-sized validation bags.

The resulting distribution is therefore hybrid Bayesian and empirical. Its
quantiles should not automatically be called Bayesian credible intervals. A
fully Bayesian interpretation would require a joint probabilistic model for
the prevalence and the class-conditional behaviour of $h$, including the
dependence introduced when the same validation observations inform both parts.
Without such a model, `posterior-propagated empirical interval` is a more precise
description.

This distinction is especially important when the Bayesian quantifier uses $V$
to infer a conditional prediction matrix and SQS-B subsequently samples bags
from the same $V$. Reusing the data is compatible with an empirical SQS
procedure, but the resulting dependence is not represented by merely combining
posterior prevalence draws with independent validation resampling. Data
splitting and cross-fitting are useful sensitivity analyses, although they also
reduce the effective validation sample size.

## Relationship to existing and proposed methods

| Method | Prevalence representation | Conditional classifier behaviour | Status in UNCAWE |
|---|---|---|---|
| SQS-H | Single point estimate | Target-sized validation bags | Proposed and evaluated in the reference paper |
| RQBS implementation of SQS-H | Bootstrap quantifier estimates | One validation bag per bootstrap estimate | Used by the current experiments through `cap-ml` |
| SQS-S | Neighbourhood of bags matched in estimated-prevalence space | Selected validation bags | Proposed and evaluated in the reference paper |
| SQS-B | Joint Bayesian posterior draws | One validation bag per posterior draw | Research proposal; not implemented |
| BayCAP | Joint posterior draws of prevalence and conditional prediction matrix | Parametric posterior over the conditional matrix | Implemented and evaluated |
| Aitchison-SQS | Estimated or constructed distribution in `ilr` coordinates | Validation bags | Research proposal; not implemented |

### SQS-B versus RQBS

Both methods propagate multiple prevalence vectors through the same validation
sampling mechanism. Their main difference is the source and interpretation of
those vectors:

$$
\text{RQBS: }q(U^{*(r)})
\qquad\text{versus}\qquad
\text{SQS-B: }\pi^{(r)}\sim p_q(\pi^Q\mid U,V).
$$

The bootstrap distribution reflects the sampling behaviour of the fitted
quantifier under its resampling scheme. The Bayesian posterior also depends on
the likelihood, prior, and model assumptions of the Bayesian quantifier. Neither
is guaranteed to be calibrated under misspecification.

### SQS-B versus BayCAP

BayCAP draws prevalence and the conditional prediction matrix jointly and maps
each pair directly to a normalized target contingency table:

$$
(\pi^{(r)},C^{(r)})\sim p(\pi^Q,C\mid U,V),
\qquad
T^{(r)}=C^{(r)}\operatorname{diag}(\pi^{(r)}).
$$

SQS-B retains only posterior prevalence draws and represents conditional
performance by resampling observed validation instances. It is thus less
parametric in this second stage, but it gives up the joint Bayesian propagation
of $C$ performed by BayCAP. Comparing the two isolates whether empirical
validation sampling is preferable to the parametric conditional model.

### SQS-B versus Aitchison-SQS

Aitchison-SQS starts from a point estimator and learns or constructs an error
distribution in `ilr` coordinates. SQS-B requires an actual posterior from the
quantifier and does not need a Gaussian or logistic-normal approximation. When
such a posterior is available, SQS-B is conceptually simpler. Aitchison-SQS
remains applicable to non-Bayesian quantifiers and can explicitly incorporate
empirical error or bias estimates unavailable from target-only posterior
sampling.

## Limitations and failure modes

- **Posterior calibration.** Bayesian output is not automatically calibrated.
  Priors, likelihood misspecification, approximate inference, and tempering can
  all make the prevalence posterior too narrow or too wide.
- **PPS dependence.** Validation resampling transfers conditional performance
  only when the relevant class-conditional behaviour remains stable. SQS-B does
  not detect conditional shift.
- **Uninformative quantifier.** If the auxiliary representation cannot identify
  some prevalence directions, the posterior should remain broad. Artificial
  posterior concentration would translate directly into overconfident
  performance intervals.
- **Rare or missing validation classes.** A posterior draw may assign positive
  mass to a class with too few validation observations. Sampling with replacement
  remains possible when at least one example is present, but the empirical
  conditional distribution may be unreliable; a missing class makes the bag
  impossible to construct.
- **Finite-sample interpretation.** Validation bags of size $|U|$ represent
  variability in a target-sized labelled sample. This is not the same uncertainty
  as posterior uncertainty about a population-level conditional matrix.
- **Monte Carlo error.** One bag per posterior draw couples prevalence and bag
  randomness. Sufficient draws and repeated random seeds are needed to show that
  reported differences are not simulation artefacts.
- **Computational cost.** MCMC-based quantifiers may be substantially more
  expensive than target bootstrap, especially across many datasets, classifiers,
  metrics, and test bags.

## Decisive experimental comparison

The minimum comparison should keep the validation data, target bags, number of
performance samples, and random-seed policy aligned across methods:

1. original point-prevalence SQS-H;
2. current bootstrap-based RQBS;
3. SQS-B using only the posterior mean, repeated through validation sampling;
4. SQS-B using full posterior draws;
5. BayCAP, preferably using the same underlying Bayesian quantification model;
6. Oracle-SQS using the true target prevalence.

The posterior-mean ablation separates changes due to the Bayesian point
estimator from changes due to uncertainty propagation. Oracle-SQS estimates the
maximum performance attainable by prevalence matching when quantification error
is removed.

Evaluation should cover nominal levels such as 50%, 80%, 90%, 95%, and 99%,
and report:

- absolute error of the point estimate;
- empirical coverage and interval amplitude;
- coverage error and reliability plots;
- Winkler score and, when full sample distributions are retained, CRPS;
- calibration and sharpness of the prevalence posterior itself;
- results conditioned on shift magnitude, target size, number of classes, and
  quantifier informativeness;
- training and inference time.

Useful sensitivity analyses include informative and misspecified priors,
posterior tempering, split versus reused validation data, multiple bags per
posterior draw, and controlled violations of PPS.

## Assessment of the contribution

SQS-B is a principled and simple extension when a posterior quantifier is
already available. It avoids imposing an auxiliary distribution around a point
estimate and gives a direct route for propagating quantification uncertainty to
arbitrary contingency-table-based metrics.

On its own, however, replacing bootstrap prevalence draws with posterior draws
is likely an incremental methodological contribution. The stronger scientific
question is:

> When does a Bayesian posterior over prevalence, propagated through empirical
> validation sampling, produce better calibrated and sharper performance
> intervals than bootstrap propagation and the fully parametric joint posterior
> used by BayCAP?

Answering this question requires the controlled ablations above and an explicit
assessment of both prevalence-posterior calibration and downstream performance
uncertainty.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- [Bayesian Quantification with Black-Box Estimators](https://arxiv.org/abs/2302.09159)
- [Aitchison-SQS proposal](aitchison_sqs.md)

