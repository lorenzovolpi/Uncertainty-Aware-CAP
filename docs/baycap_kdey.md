# BayCAP-KDEy: two-stage Bayesian pushforward with a fixed rate matrix

## Status and scope

BayCAP-KDEy is a research proposal that is not established by the reference
paper. It is implemented in UNCAWE as `uncawe.methods.BayCAPKDEy`
(`src/uncawe/methods/baycap_kdey.py`), but it has not yet been experimentally
evaluated.

The proposal extends the geometry-aware Bayesian quantifier of
[KDE-Bayes](papers/kde_bayes.pdf) — the shrinkage-regularised, Aitchison-geometry
KDE quantifier, hereafter **BayesKDEy** — into a full classifier-accuracy
prediction (CAP) method by attaching a *parametric* bridge that is **fixed** at
the target classifier's rate matrix.

Its intended contribution is not a new quantifier. Given a Bayesian posterior
over target prevalence, BayCAP-KDEy defines a *parametric* way of propagating
that posterior through the (fixed) conditional prediction matrix of the
classifier into a predictive distribution over any contingency-table-based
performance measure.

This distinguishes it from the two other proposals in this repository:

- it is not [SQS-B](sqs_b.md), which propagates the posterior through
  *empirical validation-bag sampling*; here the conditional classifier
  behaviour is represented *parametrically* by a fixed matrix;
- it is not [Aitchison-SQS](optional-directions/aitchison_sqs.md), which *constructs* a prevalence
  distribution around a non-Bayesian point estimate; here the posterior is
  taken directly from the quantifier.

It is also distinct from the **BayCAP** method of the reference paper, which:

- uses the **BayesianCC** quantifier rather than BayesKDEy; and
- treats the conditional prediction matrix $C$ as a *random* parameter drawn
  jointly with the prevalence.

BayCAP-KDEy therefore occupies a distinct point in the design space:

> a strong Bayesian KDE prevalence posterior, propagated to performance through
> a **fixed** (measured) rate matrix.

## Motivation

Under prior probability shift (PPS),

$$
P(X\mid Y)=Q(X\mid Y),
$$

while the class prevalences change, $P(Y)\neq Q(Y)$. The class-conditional
prediction rates of a fixed classifier $h$ are therefore stationary:

$$
P_h(\hat Y\mid Y)=Q_h(\hat Y\mid Y).
$$

Let $C^h_{ij}=P_h(\hat Y=\omega_i\mid Y=\omega_j)$, estimated on the labelled
validation set $V$. For any contingency-table-based measure $A$, target
performance is a deterministic functional of the target prevalence:

$$
A(h,U)=A\!\left(C^h\!\operatorname{diag}(\pi^Q)\right).
$$

Consequently, once $C^h$ is held fixed, *all* of the uncertainty about target
performance is a deterministic image of the uncertainty about the prevalence
vector $\pi^Q$. The problem of propagating uncertainty to accuracy then reduces
exactly to a *pushforward* problem through a known function.

This is precisely the kind of problem the posterior of BayesKDEy is equipped
to answer. The KDE-Bayes paper derives a Bayesian posterior over prevalence
(temperature-calibrated, NUTS-sampled) and stops there. BayCAP-KDEy closes the
loop: it takes that posterior and pushes it forward, through the fixed rate
matrix and then through the metric, to obtain a predictive distribution over
classifier accuracy.

Three properties make this formulation attractive.

1. **Strong prevalence posterior.** BayesKDEy is one of the strongest Bayesian
   quantifiers reported in the KDE-Bayes paper (best aggregate AE/W rank in the
   Bayesian regime), and its Aitchison geometry with shrinkage is specifically
   designed to behave well near the simplex boundary, where label shift is most
   severe.

2. **Parametric propagation.** The conditional behaviour of the classifier is
   represented by the *measured* matrix $C^h$, rather than by resampling an
   empirical validation bag. This keeps the whole pipeline closed-form and
   avoids the Monte-Carlo noise of bag sampling that SQS-H, SQS-S, and SQS-B
   inherit.

3. **Metric agnosticism.** Because the output is a full predictive
   distribution over contingency tables, any functional $A$ (vanilla accuracy,
   macro-F1, macro-K, ...) is obtained by applying the metric to each sampled
   table. This is exactly the metric-agnosticism claim of the reference paper
   and of BayCAP, but driven by the KDE prevalence posterior.

## Setup

Let $X$ be the input space and $Y=\{\omega_1,\dots,\omega_K\}$ the class set.
Assume PPS between the labelled distribution $P$ (training/validation) and the
unlabelled target $Q$. We assume access to:

- a trained target classifier $h:X\to\Delta^{K-1}$ of interest;
- a labelled validation set $V$ drawn from $P$;
- an unlabelled target set $U$ drawn from $Q$.

Following the reference paper, we keep **distinct** the target classifier $h$
whose accuracy is estimated and the surrogate classifier $s$ used inside the
quantifier. Keeping them distinct avoids the circularity in which the measured
quantity depends on the very classifier being measured.

Let

$$
C^h_{ij}=P_h(\hat Y=\omega_i\mid Y=\omega_j),
$$

using out-of-sample predictions of $h$ on $V$ (e.g. cross-fitting, as in
SQS-S), so that $C^h$ is an unbiased, PPS-stationary estimate shared by $P$ and
$Q$. The key modelling choice of BayCAP-KDEy is to treat $C^h$ as a *fixed
constant*, not as a random variable. This is a deliberate departure from
BayCAP, which places a Dirichlet prior on the columns of $C$ and samples it
jointly with $\pi^Q$.

Let $q$ denote the BayesKDEy quantifier with surrogate $s$, and let

$$
\pi^Q\sim p_T\!\left(\pi^Q\mid U,V; s\right)
$$

be its temperature-$T$ posterior (tempering as in the KDE-Bayes paper,
selected on a held-out validation split). The required interface is the set of
**joint** posterior samples

$$
\left\{\pi^{(1)},\dots,\pi^{(m)}\right\},\qquad
\pi^{(r)}\in\Delta^{K-1}.
$$

As in SQS-B, joint draws are essential: sampling each class independently
from a marginal credible interval would discard posterior dependence and could
produce vectors outside the simplex.

## Algorithm

Given a performance functional $A$, the BayesKDEy quantifier $q$ with
surrogate $s$, the labelled validation set $V$, the target classifier $h$ of
interest, and the unlabelled target set $U$:

1. Fit $s$ and the quantifier $q$ using $V$; obtain the temperature
   $T$ (and, if using a posterior-mean plug-in, the posterior mean) from the
   quantifier's own validation procedure.

2. Measure the rate matrix of the target classifier,

   $$
   C^h_{ij}\approx \frac{1}{\lvert V\rvert}\sum_{x\in V}\mathbf{1}\!\left[\hat y_h(x)=\omega_i\right]\mathbf{1}\!\left[y(x)=\omega_j\right],
   $$

   where the predictions $\hat y_h(x)$ are obtained *out of sample* on $V$
   (cross-fitting within $V$, as in SQS-S), so that $C^h$ is an unbiased,
   PPS-stationary estimate shared by $P$ and $Q$.

3. Draw $m$ joint posterior prevalence vectors
   $\left\{\pi^{(r)}\right\}_{r=1}^m$ from $p_T\left(\pi^Q\mid U,V;s\right)$.

4. **First pushforward (prevalence $\to$ contingency table).** For every draw,
   form the normalised target contingency table

   $$
   T^{(r)} = C^h\,\operatorname{diag}\!\left(\pi^{(r)}\right).
   $$

5. **Second pushforward (contingency table $\to$ metric).** For every table,
   compute one performance draw

   $$
   a^{(r)} = A\!\left(T^{(r)}\right).
   $$

6. Return the Monte-Carlo mean

   $$
   \widehat A_{\mathrm{BayCAP-KDEy}}
   =\frac{1}{m}\sum_{r=1}^m a^{(r)}
   $$

   as the point estimate, and the empirical $[\alpha/2,1-\alpha/2]$ quantiles
   of the $\{a^{(r)}\}$ as the interval estimate.

Equivalently, for vanilla accuracy the two pushforwards collapse into a single
linear map (see below), which permits exact moment computation.

## Theory: the two-stage structure

### Vanilla accuracy is linear

For vanilla accuracy,

$$
\operatorname{Acc}\!\left(C^h\operatorname{diag}(\pi)\right)
= \sum_{j}\!C^h_{jj}\,\pi_j
= \bigl(\operatorname{diag} C^h\bigr)^{\!\top}\pi .
$$

Writing $d=\operatorname{diag}(C^h)$ and treating $\operatorname{Acc}$ as the
random variable $\operatorname{Acc}=d^\top\pi^Q$, the map
$\pi\mapsto\operatorname{Acc}$ is a single linear functional of the posterior
variable. The predictive moments follow immediately from the posterior moments
of $\pi^Q$:

$$
\mathbb{E}[\operatorname{Acc}]
= d^\top\,\mathbb{E}[\pi^Q],
\qquad
\operatorname{Var}(\operatorname{Acc})
= d^\top\,\operatorname{Cov}(\pi^Q)\,d .
$$

This is a direct consequence of holding $C^h$ fixed: the variance of the
accuracy under PPS is exactly the variance induced by the prevalence posterior
projected through the diagonal of the rate matrix, with no sampling noise and
no uncertainty on $C$.

BayCAP-KDEy therefore recovers the same quantity computed by the
cross-validated estimator in the reference paper (paper, Eq. 2),
$\operatorname{Acc}=d^\top\pi^Q$, but replaces the point plug-in by a
posterior: rather than estimating the bias $d^\top(\pi^P-\pi^Q)$ of a
cross-validated estimate, it estimates $\mathbb{E}[d^\top\pi^Q]$ and
$\operatorname{Var}(d^\top\pi^Q)$ directly from the prevalence posterior.

### Nonlinear metrics require the full two-stage map

For nonlinear functionals such as macro-F1 and macro-K,

$$
A(T)=\frac{1}{K}\sum_{k}\!A_k(T),
\qquad A_k(T)\ \text{nonlinear in } T,
$$

the identity $\mathbb E[A(T)]\neq A(\mathbb E[T])$ holds in general, and the
metric distributions may be asymmetric. In this case the second pushforward
cannot be dropped: each posterior draw must be mapped through the *full*
functional $A\bigl(C^h\operatorname{diag}(\pi^{(r)})\bigr)$. This is
identical in structure to the corresponding observation for BayCAP in the
reference paper, where the metric is applied to every posterior table.

BayCAP-KDEy therefore inherits from BayCAP the property that a well-estimated
prevalence distribution does **not** automatically yield calibrated intervals
for every functional; each metric must be propagated and calibrated separately.
This is exactly the point raised by the "metrics beyond vanilla accuracy" and
"systematic interval calibration" extensions.

### Interpretation

Because the conditional classifier behaviour is represented by a single
measured matrix $C^h$ rather than by empirical resampling, the resulting
predictive distribution is *parametric-Bayesian* rather than hybrid
Bayesian-empirical as in SQS-B. Its quantiles are Bayesian credible intervals
for $A(h,U)$ under the conditioning "$C \equiv C^h$": they include prevalence
posterior uncertainty, but not the estimation uncertainty of $C^h$ itself.

This is a stronger (but also more restrictive) claim than SQS-B's. It relies
on $C^h$ measured on $V$ being a faithful proxy for the $P$-population rate
matrix, which holds under PPS. It does **not** account for the estimation
uncertainty in $C^h$: on a small or noisy $V$, the true rate matrix can differ
from $C^h$, and the resulting intervals will be too narrow. BayCAP, which
models $C$ as a random object, does account for that source of uncertainty,
at the cost of a weaker (and less interpretable) prevalence posterior than
BayesKDEy.

The two approaches thus trade off different uncertainty sources:

| Source of uncertainty | BayCAP-KDEy | SQS-B | BayCAP |
|---|:---:|:---:|:---:|
| Prevalence | full posterior | full posterior | full posterior |
| Rate matrix $C$ | **fixed** (omitted) | **empirical bag** (implicit) | **random** (Dirichlet) |
| Bag / sampling noise | none | present | none |
| Quantifier strength (multiclass) | BayesKDEy | BayesKDEy | BayesianCC |

## Relationship to existing and proposed methods

| Method | Prevalence representation | Conditional classifier behaviour | Status in UNCAWE |
|---|---|---|---|
| SQS-H | Single point estimate | Target-sized validation bags | Proposed and evaluated in the reference paper |
| RQBS | Bootstrap quantifier estimates | One validation bag per bootstrap estimate | Implemented (current `cap-ml`) |
| SQS-S | Neighbourhood of bags matched in estimated-prevalence space | Selected validation bags | Proposed and evaluated in the reference paper |
| SQS-B | Joint Bayesian posterior draws | One validation bag per posterior draw | Research proposal; not implemented |
| Aitchison-SQS | Estimated or constructed distribution in `ilr` coordinates | Validation bags | Research proposal; not implemented |
| BayCAP | Joint posterior draws (BayesianCC) | Parametric posterior over $C$ | Implemented and evaluated in the reference paper |
| **BayCAP-KDEy** | **Joint posterior draws (BayesKDEy)** | **Fixed matrix $C^h$ (measured on $V$)** | **Research proposal; implemented in UNCAWE, not yet evaluated** |

### BayCAP-KDEy versus BayCAP

Both are Bayesian methods that propagate a prevalence posterior through a
metric functional, but they differ in both the quantifier and in how the
conditional classifier behaviour is represented. Their differences are:

1. **Rate matrix.** BayCAP draws $(\pi^Q,C)$ jointly and propagates
   $T^{(r)}=C^{(r)}\operatorname{diag}(\pi^{(r)})$. BayCAP-KDEy fixes
   $C=C^h$ and only propagates $\pi^Q$:
   $T^{(r)}=C^h\operatorname{diag}(\pi^{(r)})$.
2. **Quantifier.** BayCAP uses BayesianCC (Ziegler–Czyz, TMLR 2024);
   BayCAP-KDEy uses BayesKDEy (Moreo–González–del Coz, [KDE-Bayes](papers/kde_bayes.pdf)).
3. **Sources of uncertainty.** BayCAP includes uncertainty on $C$;
   BayCAP-KDEy does not.

These differences are *scientifically orthogonal* to the prevalence
posterior. A controlled comparison holding the quantifier fixed and swapping
only the propagation mechanism (C-fixed vs C-random vs empirical-bag) directly
tests which source of uncertainty dominates.

### BayCAP-KDEy versus SQS-B

SQS-B propagates a prevalence posterior through *empirical* validation-bag
sampling. BayCAP-KDEy propagates the *same kind* of posterior through a
*parametric* bridge. The differences are:

- the conditional behaviour is represented analytically, not by resampling;
- there is no Monte-Carlo noise from the bag-drawing step;
- the uncertainty on $C^h$ is not propagated.

A controlled comparison of the two with the same quantifier and the same
number of posterior draws isolates the cost (or benefit) of replacing bag
sampling by a parametric bridge.

### BayCAP-KDEy versus Aitchison-SQS

Aitchison-SQS starts from a point estimate of target prevalence and constructs
a distribution around it (logistic-normal in `ilr` coordinates, with
shrinkage-regularised covariance). BayCAP-KDEy does not need to construct the
distribution: it takes an *already* posterior from the quantifier.
When a posterior is available, BayCAP-KDEy is conceptually simpler.
Aitchison-SQS remains applicable to non-Bayesian quantifiers and can
explicitly use empirical error or bias estimates that posterior sampling does
not expose.

## Limitations and failure modes

- **Fixed rate matrix.** Omitting uncertainty on $C^h$ produces intervals
  that are *conditioned on* a single estimate. On small or noisy $V$ this
  overstates confidence. This is the central trade-off with BayCAP (which
  models $C$ as a random object) and the central reason BayCAP-KDEy can be
  outperformed on hard or noisy datasets even when the prevalence posterior is
  itself calibrated.
- **PPS dependence.** The stationarity $P_h(\hat Y\mid Y)=Q_h(\hat Y\mid Y)$
  is an assumption, not an estimate. Conditional shift, covariate shift, or
  classifier miscalibration break the bridge.
- **Posterior calibration.** As in SQS-B, Bayesian output is not
  automatically calibrated. Temperature $T$ must be selected on a held-out
  validation split and should be selected *separately* from the prevalence
  posterior.
- **Quantifier choice.** The method inherits the quantifier's quality. A weak
  or systematically biased BayesKDEy posterior (e.g. under extreme shift,
  rare classes, or when the surrogate is poorly calibrated) translates
  directly into a biased or overconfident performance distribution.
- **Metric propagation.** For nonlinear metrics, the functional
  $T\mapsto A(T)$ can be sharply non-linear near the boundary of the simplex,
  and the induced predictive distribution can be asymmetric or multimodal.
  Calibration must be assessed per metric.
- **Missing classes.** If a class has zero (or near-zero) prevalence in the
  target, the posterior must be sharp enough to reflect that; otherwise the
  intervals for class-level or macro-level metrics can be misleading.
- **Cost.** BayesKDEy uses NUTS. The cost per dataset-classifier-test-bag is
  the cost of NUTS plus a single matrix-vector product per draw; this is
  typically cheaper than bag-sampling methods on large validation sets but can
  be comparable or more expensive than point estimators.

## Decisive experimental comparison

The minimum comparison must keep the validation data, target bags, metric, and
random-seed policy aligned across methods:

1. SQS-H (point prevalence; current SQS-H implementation);
2. RQBS (bootstrap prevalence; current `cap-ml`);
3. SQS-S (neighbourhood of bags; current `cap-ml`);
4. **BayCAP-KDEy** (this proposal): BayesKDEy posterior, C-fixed;
5. **BayCAP-KDEy-empirical**: *same* BayesKDEy posterior, propagated through
   one empirical validation bag per draw (i.e. SQS-B on BayesKDEy);
6. **BayCAP-CC**: *the same propagation mechanism of BayCAP-KDEy* (C-fixed)
   but with the BayesianCC posterior instead of the BayesKDEy posterior;
7. **BayCAP**: the reference-paper method (BayesianCC posterior, C-random);
8. **Oracle-C**: BayesKDEy posterior, C-fixed, but with true target
   prevalence drawn from the oracle distribution — isolates the propagation
   mechanism from the prevalence posterior.

Rows 4 (C-fixed, KDE), 5 (empirical, KDE), 6 (C-fixed, CC), and 7 (C-random,
CC) isolate, one by one, the three design axes that matter: *which rate-matrix
propagation* (fixed vs empirical vs random), *which prevalence posterior*
(KDE vs BayesianCC), and *how they combine*. This is the cleanest possible
design for a scientific claim of the form:

> "Given a fixed prevalence posterior, parametric propagation with a fixed
> rate matrix matches or beats empirical-bag propagation and random-rate
> propagation on coverage, sharpness, and nonlinear-metric calibration."

Evaluation should cover nominal levels $\{0.50,0.80,0.90,0.95,0.99\}$ and
report, for vanilla accuracy, macro-F1, and macro-K:

- absolute error of the point estimate;
- empirical coverage, coverage error, and interval amplitude;
- reliability plots and Winkler score;
- CRPS and full predictive-distribution quality when all samples are retained;
- coverage conditioned on shift magnitude, target size, number of classes, and
  quantifier informativeness;
- the gap between the BayCAP-KDEy intervals and the Oracle-C intervals under
  the same posterior (isolates how much of the posterior error is due to
  propagation rather than to the posterior itself);
- training and inference time.

Sensitivity analyses that should be run:

- temperature $T\in\{0.5,1,2,5\}$.
- shrinkage $\lambda$ from the KDE-Bayes model-selection grid.
- split versus reused validation data (BayesKDEy posterior uses $V$; bags
  would be drawn from the same $V$ in the empirical variant — the dependence
  is different and must be quantified).
- controlled violation of PPS (a small fraction of covariate shift).
- misspecified informative prior on $\pi^Q$.
- rare or absent classes in the target.

## Assessment of the contribution

BayCAP-KDEy is a clean, minimal, and defensible extension when a good Bayesian
quantifier is already available. It adds no new modelling machinery: it
reuses the temperature-calibrated posterior of BayesKDEy and a single linear
bridge $C^h$. Its novelty is in *where* it propagates uncertainty:

- it is fully parametric on the conditional side (no bag-sampling noise);
- it is fully Bayesian on the prevalence side;
- it therefore yields, per metric, a *parametric-Bayesian* predictive
  distribution with moments amenable to closed-form analysis.

On its own, replacing bag sampling by a fixed matrix is an incremental
methodological step. The scientific question BayCAP-KDEy is designed to
answer, and that distinguishes it from the other extensions in this
repository, is:

> Holding the prevalence posterior fixed, does parametric propagation with a
> fixed rate matrix give calibrated predictive distributions across metrics
> and nominal levels, and does it outperform (a) empirical-bag propagation,
> (b) random-rate propagation (BayCAP), and (c) Bayesian quantifiers with
> weaker multiclass calibration (BayesianCC)?

If the decisive ablation (rows 4-7 of the experimental comparison above)
answers that yes, with a clear and explainable gap to BayesianCC and to bag
sampling, BayCAP-KDEy supports a substantive scientific claim about which
uncertainty source matters most in uncertainty-aware CAP. If the answer is
mixed or the advantage is small, the value of the method lies in its
interpretability and closed-form moments, but the contribution should be
presented as a *baseline* rather than as a state of the art.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- [Geometry-Aware Bayesian Quantification via Compositional Data Analysis (KDE-Bayes)](papers/kde_bayes.pdf)
- [Ziegler, Czyz. Bayesian quantification with black-box estimators. TMLR 2024.](https://arxiv.org/abs/2302.09159)
- [Aitchison-SQS proposal](optional-directions/aitchison_sqs.md)
- [SQS-B proposal](sqs_b.md)
- [QuaPy](https://github.com/HLT-ISTI/QuaPy) · [cap-ml](https://github.com/lorenzovolpi/cap-ml)
