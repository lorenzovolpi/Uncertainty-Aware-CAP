# Work in progress: characterizing CAP difficulty through the conditional rate matrix

## Status and scope

This document develops an early-stage research direction for characterizing the
difficulty of classifier accuracy prediction (CAP) from a class-conditional
prediction-rate matrix. It is not part of the reference paper, is not an
implemented method in UNCAWE, and should not yet be presented as an established
result.

The central simplifying assumption is that the relevant conditional rate matrix
is known accurately enough to be treated as fixed. Estimation error in that
matrix is deliberately excluded. The resulting quantities characterize
structural identifiability, sensitivity, and target-sampling variance. They do
not, by themselves, determine the bias of a particular CAP method.

The initial analysis is restricted to vanilla accuracy under prior probability
shift (PPS). Extensions to nonlinear contingency-table measures require
metric-specific derivatives or direct optimization over the induced metric.

## Setup

Let $h$ be the fixed classifier whose target accuracy must be estimated. Under
PPS,

$$
P(X\mid Y)=Q(X\mid Y),
\qquad
P(Y)\ne Q(Y).
$$

Let

$$
C_{rk}=P(h(X)=r\mid Y=k)
$$

be the column-stochastic conditional prediction matrix of $h$, and let

$$
d=\operatorname{diag}(C)
$$

be its vector of class-specific correct-classification rates. For a target
prevalence vector $\pi\in\Delta^{K-1}$, vanilla accuracy is

$$
a(\pi)=d^\top\pi.
$$

The marginal distribution of the crisp predictions is

$$
\mathbf{c}=C\pi.
$$

This creates two conceptually different maps:

$$
\pi\longmapsto a=d^\top\pi,
\qquad
\pi\longmapsto \mathbf{c}=C\pi.
$$

The first determines how target prevalence affects accuracy. The second
determines how much information about target prevalence is observable through
the classifier predictions. CAP difficulty depends on their interaction.

### $\mathbf{c}$ as the marginal distribution of predicted labels
Given

$$
C_{rk}=P(\widehat Y=r\mid Y=k)
$$

and target prevalence $\pi$, the normalized contingency table is

$$
T=C\operatorname{diag}(\pi),
$$

because

$$
T_{rk}=P(\widehat Y=r,Y=k)=C_{rk}\pi_k.
$$

Instead,

$$
\mathbf{c}=C\pi
$$

is the marginal distribution of predicted labels:

$$
\mathbf{c}_r=P(\widehat Y=r)=\sum_k C_{rk}\pi_k.
$$

Equivalently, $\mathbf{c}$ is obtained by summing the contingency table over
its true-label columns:

$$
\mathbf{c}=T\mathbf 1.
$$

Thus:

- $C$: class-conditional prediction-rate matrix;
- $\pi$: true-label prevalence vector;
- $T=C\operatorname{diag}(\pi)$: normalized contingency table;
- $\mathbf{c}=C\pi$: predicted-label prevalence vector.

If using counts rather than normalized probabilities, the contingency table is approximately

$$
N\,C\operatorname{diag}(\pi),
$$

where $N$ is the target-bag size.

## From invertibility to conditioning to CAP-specific conditioning

The matrix $C$ plays two related but conceptually distinct roles. First, it maps
target prevalence to the observable marginal distribution of predicted labels,

$$
\mathbf{c}=C\pi.
$$

Second, through its diagonal $d=\operatorname{diag}(C)$, it determines how target
prevalence affects vanilla accuracy,

$$
a(\pi)=d^\top\pi.
$$

A useful way to organize the analysis is therefore to proceed in three steps:

1. **invertibility:** can the full prevalence vector $\pi$ be recovered from
   $\mathbf{c}$?
2. **conditioning:** if it can be recovered, how sensitive is that recovery to
   small errors in $\mathbf{c}$?
3. **CAP-specific conditioning:** are the poorly observed prevalence directions
   also directions along which the target accuracy changes?

The third question is the one that is specific to CAP.

### What invertibility of $C$ means

If $C$ is known exactly and is invertible, then the population prevalence is
identified from the predicted-label distribution through

$$
\pi=C^{-1}\mathbf{c}.
$$

Thus, in the idealized noiseless setting, invertibility guarantees that the
complete prevalence vector can be recovered from $\mathbf{c}$.

If $C$ is singular, there exists a nonzero direction $v$ such that

$$
Cv=0.
$$

Therefore,

$$
C(\pi+v)=C\pi,
$$

whenever both $\pi$ and $\pi+v$ are valid prevalence vectors. Two different
prevalence vectors can then induce exactly the same distribution of predicted
labels, so the complete prevalence vector is not identifiable through $C$.

For CAP, however, this is not yet enough to conclude that target accuracy is
unidentifiable. Accuracy changes along $v$ by

$$
a(\pi+v)-a(\pi)=d^\top v.
$$

Hence there are two qualitatively different singular cases:

- if $Cv=0$ and $d^\top v\ne0$, the two prevalences are observationally
  indistinguishable but imply different accuracies; vanilla accuracy is not
  identifiable from $\mathbf{c}$;
- if $Cv=0$ implies $d^\top v=0$ for every admissible direction $v$, the full
  prevalence vector is not identifiable, but vanilla accuracy can still be
  identifiable.

This distinction is central to the present proposal: **CAP does not require
recovering all of $\pi$ when the target metric depends only on a lower-dimensional
functional of $\pi$.**

### Conditioning: invertible does not mean stable

Even when $C$ is invertible, recovering $\pi$ can be numerically and
statistically unstable. Suppose that the observed target prediction frequencies
contain a small error,

$$
\widehat{\mathbf{c}}=\mathbf{c}+\varepsilon.
$$

The inversion estimator becomes

$$
\widehat\pi
=
C^{-1}\widehat{\mathbf{c}}
=
\pi+C^{-1}\varepsilon.
$$

Thus, a small perturbation in prediction space may be strongly amplified in
prevalence space. This is the standard meaning of poor conditioning.

Let

$$
C=U\Sigma V^\top
$$

be the singular value decomposition, with singular values

$$
\sigma_1\ge\cdots\ge\sigma_K\ge0.
$$

The smallest singular value has the geometric interpretation

$$
\sigma_{\min}(C)
=
\min_{\|v\|_2=1}\|Cv\|_2.
$$

Hence, if $\sigma_{\min}(C)$ is small, there exists a non-negligible direction
$v$ that is almost annihilated by $C$: prevalence can change substantially in
that direction while the observable prediction distribution changes very little.
If $\sigma_{\min}(C)=0$, the matrix is singular.

The usual Euclidean condition number is

$$
\kappa_2(C)
=
\frac{\sigma_{\max}(C)}{\sigma_{\min}(C)}.
$$

A value close to one indicates a well-conditioned map. Large values indicate
increasing sensitivity, and $\kappa_2(C)=+\infty$ when $C$ is singular. There is
no universal numerical threshold at which a matrix becomes "badly
conditioned"; the practically relevant scale depends on the noise level, sample
size, estimator, and target functional. For empirical analysis it is therefore
preferable to report the singular values themselves, or at least both
$\sigma_{\min}$ and $\kappa_2$, rather than relying on an arbitrary cutoff.

### Conditioning should be evaluated on the simplex tangent space

Prevalence vectors lie on the simplex, so valid infinitesimal perturbations do
not span all of $\mathbb R^K$. They satisfy

$$
T
=
\{v\in\mathbb R^K:\mathbf 1^\top v=0\}.
$$

Let $B\in\mathbb R^{K\times(K-1)}$ have orthonormal columns spanning $T$.
Every admissible perturbation can be written as $v=Bz$, and its observable
change is

$$
Cv=CBz.
$$

This suggests diagnosing prevalence identifiability through the restricted map

$$
A=CB
$$

rather than through $C$ on the whole ambient space. Useful diagnostics are then

$$
\sigma_{\min}(CB)
$$

and, when $CB$ has full column rank,

$$
\kappa_2(CB)
=
\frac{\sigma_{\max}(CB)}{\sigma_{\min}(CB)}.
$$

A small $\sigma_{\min}(CB)$ means that some *valid prevalence direction* is
weakly observable in predicted-label space. This is more directly relevant than
an ill-conditioned direction that does not correspond to a perturbation within
the simplex.

### Why the standard condition number is still insufficient for CAP

Poor conditioning of $C$ or $CB$ means that some prevalence directions are
hard to identify. Vanilla-accuracy CAP, however, does not require the full
prevalence vector. It only requires

$$
a(\pi)=d^\top\pi.
$$

Consider an admissible direction $v\in T$ for which $\|Cv\|_2$ is very small.
There are two possible situations.

First,

$$
\|Cv\|_2\approx0,
\qquad
|d^\top v|\approx0.
$$

The prevalence direction is poorly observed, but moving along it barely changes
accuracy. This direction makes prevalence estimation difficult but is largely
irrelevant for CAP.

Second,

$$
\|Cv\|_2\approx0,
\qquad
|d^\top v|\ \text{is large}.
$$

Now a weakly observable change in prevalence can correspond to a substantial
change in target accuracy. This is the genuinely difficult case for CAP.

The ordinary condition number does not distinguish these cases. It may therefore
overstate CAP difficulty when the ill-conditioned directions are
accuracy-irrelevant, or understate the practical difficulty when the weakest
observable direction is strongly aligned with the accuracy functional.

The relevant structural object should therefore combine **prevalence
identifiability** with **metric sensitivity**.

## Accuracy susceptibility to prior shift

Before combining the two effects, it is useful to isolate how strongly accuracy
can vary with prevalence. The simplest matrix-derived quantity is

$$
S_{\mathrm{acc}}(C)
=
\max_k d_k-\min_k d_k.
$$

Since a linear functional reaches its extrema at vertices of the simplex,

$$
\max_{\pi\in\Delta^{K-1}}d^\top\pi
-
\min_{\pi\in\Delta^{K-1}}d^\top\pi
=
S_{\mathrm{acc}}(C).
$$

If all class-specific rates are equal, $d=\beta\mathbf 1$, then

$$
a(\pi)=\beta
$$

for every target prevalence. Vanilla-accuracy CAP is then insensitive to PPS,
even if target prevalence itself is difficult or impossible to identify.

More generally, only the component of $d$ tangent to the simplex affects
accuracy changes. Define

$$
d_T
=
\left(I-\frac{1}{K}\mathbf 1\mathbf 1^\top\right)d.
$$

If $\widehat\pi-\pi\in T$, then

$$
\widehat a-a
=
d_T^\top(\widehat\pi-\pi).
$$

The norm $\|d_T\|$ is therefore a second measure of accuracy susceptibility. It
describes how strongly accuracy reacts to valid prevalence changes, but not how
well those changes can be observed through $C$.

## Directional condition number for accuracy

Now combine the two ingredients. For a valid infinitesimal prevalence
perturbation $v\in T$,

$$
\delta\mathbf{c}=Cv,
\qquad
\delta a=d^\top v.
$$

This motivates the directional CAP condition number

$$
\boxed{
\chi_{\mathrm{acc}}(C)
=
\sup_{\substack{v\in T\\v\ne0}}
\frac{|d^\top v|}{\|Cv\|_2}
}.
$$

The numerator is the change in the quantity that CAP wants to estimate. The
denominator is the observable change in predicted-label frequencies. Thus,
$\chi_{\mathrm{acc}}$ can be read as a worst-case ratio

$$
\frac{\text{change in target accuracy}}
     {\text{observable change in prediction frequencies}}.
$$

Interpretation:

- a small value means that accuracy-relevant prevalence changes are clearly
  reflected in predicted-label frequencies, or that accuracy barely changes;
- a large value means that a small observable change can correspond to a large
  change in target accuracy;
- an infinite value means that two locally indistinguishable prevalences can
  have different accuracies.

Specifically, set $\chi_{\mathrm{acc}}(C)=+\infty$ when there exists $v\in T$
such that

$$
Cv=0,
\qquad
d^\top v\ne0.
$$

If every tangent null direction of $C$ also satisfies $d^\top v=0$, complete
prevalence is not identifiable but vanilla accuracy remains identifiable.

### Computation on the simplex tangent space

Let

$$
A=CB,
\qquad
g=B^\top d.
$$

Since every tangent perturbation can be written as $v=Bz$,

$$
\chi_{\mathrm{acc}}(C)
=
\sup_{z\ne0}
\frac{|g^\top z|}{\|Az\|_2}.
$$

If $A$ has full column rank,

$$
\chi_{\mathrm{acc}}^2(C)
=
g^\top(A^\top A)^{-1}g.
$$

If $A$ is rank-deficient, the quantity is finite only when $g$ is orthogonal
to $\operatorname{null}(A)$. In that case,

$$
\chi_{\mathrm{acc}}^2(C)
=
g^\top(A^\top A)^\dagger g,
$$

where $\dagger$ denotes the Moore--Penrose pseudoinverse. Otherwise it is
infinite.

This expression is invariant to the particular orthonormal basis $B$. It still
depends on the norm used in prediction space; the Euclidean norm is a
convenient first formulation, not the only possible one.

### Relationship to singular values

The singular values of $CB$ diagnose how well prevalence directions are
observed, while $\chi_{\mathrm{acc}}$ additionally weights those directions by
their relevance to accuracy.

In particular,

$$
\sigma_{\min}(CB)\ \text{small}
$$

means that at least one valid prevalence direction is nearly invisible, whereas

$$
\chi_{\mathrm{acc}}(C)\ \text{large}
$$

means that an invisible or weakly visible direction is also important for
accuracy. The second condition is the one that should matter most when
characterizing structural CAP difficulty.

## Binary case

The binary case makes the distinction especially transparent. Write

$$
C=
\begin{pmatrix}
\alpha & 1-\beta\\
1-\alpha & \beta
\end{pmatrix},
$$

where $\alpha=C_{11}$ and $\beta=C_{22}$ are the two class-specific correct
classification rates. If $\pi$ is the prevalence of the first class, then

$$
\mathbf{c}_1
=
(\alpha+\beta-1)\pi+(1-\beta)
$$

and

$$
a
=
\beta+(\alpha-\beta)\pi.
$$

The factor

$$
\alpha+\beta-1
$$

controls prevalence identifiability. When it is close to zero, large changes in
$\pi$ produce only small changes in the observed predicted-label prevalence.
When it is exactly zero, the predicted-label distribution does not identify
$\pi$.

The factor

$$
\alpha-\beta
$$

controls accuracy susceptibility: when it is zero, vanilla accuracy is constant
with respect to prevalence.

Whenever $\alpha+\beta-1\ne0$,

$$
\frac{\partial a}{\partial \mathbf{c}_1}
=
\frac{\alpha-\beta}{\alpha+\beta-1}.
$$

Apart from constants induced by the norm convention, the binary directional
complexity is therefore

$$
\boxed{
\chi_{\mathrm{acc}}(C)
=
\left|
\frac{\alpha-\beta}{\alpha+\beta-1}
\right|
}.
$$

The numerator measures how strongly accuracy changes with prevalence; the
denominator measures how visible prevalence changes are through the classifier's
predicted-label frequencies.

Important cases are:

- if $\alpha=\beta$, accuracy is invariant to prevalence and the complexity is
  zero, even if prevalence itself is difficult to estimate;
- if $\alpha+\beta=1$ but $\alpha\ne\beta$, predicted-label frequencies do not
  identify prevalence while accuracy depends on it, so the complexity is
  infinite;
- if both numerator and denominator vanish because $d$ is constant, the general
  tangent-space definition assigns complexity zero: prevalence is not
  identifiable, but it is unnecessary for estimating vanilla accuracy.

This binary expression provides the simplest intuition for the multiclass
quantity: **CAP becomes difficult when an accuracy-relevant prevalence direction
is weakly observable.**

## Target-specific sampling difficulty

The directional condition number is structural and depends only on $C$. The
difficulty of a particular target bag also depends on its prevalence and size.

Assume temporarily that $C$ is invertible and that prevalence is estimated by

$$
\widehat\pi=C^{-1}\widehat{\mathbf{c}},
$$

where $\widehat{\mathbf{c}}$ is the empirical predicted-label distribution in
a target bag of size $n$. Then

$$
\widehat a
=
d^\top C^{-1}\widehat{\mathbf{c}}
=
w^\top\widehat{\mathbf{c}},
\qquad
w=C^{-\top}d.
$$

For multinomial prediction counts,

$$
\operatorname{Cov}(\widehat{\mathbf{c}}\mid\mathbf{c})
=
\frac{1}{n}
\left[\operatorname{diag}(\mathbf{c})-\mathbf{c}\mathbf{c}^\top\right],
\qquad
\mathbf{c}=C\pi.
$$

The induced sampling variance is

$$
\boxed{
V_{\mathrm{acc}}(C,\pi,n)
=
\frac{1}{n}
w^\top
\left[\operatorname{diag}(\mathbf{c})-\mathbf{c}\mathbf{c}^\top\right]
w
}.
$$

This is a target-specific difficulty measure. It accounts for:

- the accuracy direction $d$;
- amplification through $C^{-1}$;
- the current target composition through $\mathbf{c}=C\pi$;
- the target sample size $n$.

In a label-free application, $\mathbf{c}$ can be replaced by the directly
observed $\widehat{\mathbf{c}}$. Near the boundary or under singular $C$,
constrained or pseudoinverse-based formulations are needed. The expression
describes the specific linear inversion estimator and should not automatically
be attributed to every CAP method.

## Fisher-information difficulty

A local information-theoretic formulation avoids using a global condition
number. Parameterize the simplex locally as

$$
\pi=\pi_0+B\theta,
$$

where the columns of $B$ span $T$. Under multinomial predicted-label counts,
the Fisher information for $\theta$ is

$$
I_C(\pi)
=
nB^\top C^\top
\operatorname{diag}\!\left(\frac{1}{C\pi}\right)
CB,
$$

provided the relevant components of $C\pi$ are positive. The accuracy gradient
in these coordinates is

$$
g=B^\top d.
$$

The corresponding Cramér--Rao expression is

$$
\boxed{
D_{\mathrm{Fisher}}(C,\pi,n)
=
g^\top I_C(\pi)^{-1}g
}.
$$

With appropriate regularity conditions, this is a lower bound on the local
variance of a regular unbiased estimator of accuracy based on predicted-label
counts. A pseudoinverse and an explicit identifiability check are required when
the information matrix is singular. Boundary prevalences also require separate
treatment.

This quantity should be described as local information or a variance lower
bound, not as the expected bias of a CAP method.

## Accuracy identification width

A more directly interpretable alternative is to construct the set of target
prevalences compatible with the observed prediction distribution:

$$
\Pi_\varepsilon(C,\widehat{\mathbf{c}})
=
\left\{
\pi\in\Delta^{K-1}:
D(C\pi,\widehat{\mathbf{c}})\le\varepsilon
\right\}.
$$

Here $D$ is a specified discrepancy and $\varepsilon$ represents target-sample
uncertainty. The plausible accuracy range is

$$
a^-_\varepsilon
=
\min_{\pi\in\Pi_\varepsilon}d^\top\pi,
\qquad
a^+_\varepsilon
=
\max_{\pi\in\Pi_\varepsilon}d^\top\pi.
$$

Define the accuracy identification width as

$$
\boxed{
W_{\mathrm{acc}}(C,\widehat{\mathbf{c}},\varepsilon)
=
a^+_\varepsilon-a^-_\varepsilon
}.
$$

It answers the operational question:

> How much can target accuracy vary among all prevalence vectors compatible
> with the observed predictions, the assumed matrix $C$, and the selected
> finite-sample tolerance?

Advantages include:

- direct interpretation on the accuracy scale;
- compatibility with singular or rectangular observation matrices;
- explicit enforcement of simplex constraints;
- separation between structural non-identifiability and finite-sample
  uncertainty;
- computation through convex optimization for suitable choices of $D$.

With $\varepsilon=0$, positive width indicates structural non-identifiability
of accuracy at the observed $\mathbf{c}$. With $\varepsilon>0$, the width also
reflects finite target size. The tolerance must be selected through a justified
multinomial confidence region or concentration result, not tuned using target
labels.

## Bias, variance, and complexity are different quantities

Treating $C$ as exact removes one important source of uncertainty but does not
determine the bias of a CAP method. If a method supplies a prevalence estimate
$\widehat\pi$ with bias

$$
b_\pi
=
\mathbb E[\widehat\pi-\pi],
$$

then the induced accuracy bias is exactly

$$
\operatorname{Bias}(\widehat a)
=
d^\top b_\pi
$$

for the linear plug-in estimator $\widehat a=d^\top\widehat\pi$.

The proposed matrix-derived quantities have different roles:

| Quantity | Main interpretation | Target-specific? |
|---|---|:---:|
| $S_{\mathrm{acc}}(C)$ or $\|d_T\|$ | Susceptibility of accuracy to PPS | No |
| $\chi_{\mathrm{acc}}(C)$ | Worst-case directional amplification and identifiability | No |
| $V_{\mathrm{acc}}(C,\pi,n)$ | Sampling variance for linear inversion | Yes |
| $D_{\mathrm{Fisher}}(C,\pi,n)$ | Local variance lower bound | Yes |
| $W_{\mathrm{acc}}(C,\widehat{\mathbf{c}},\varepsilon)$ | Plausible accuracy ambiguity | Yes |
| $d^\top b_\pi$ | Bias of a particular prevalence-based estimator | Method-specific |

Thus, $C$ can characterize how difficult it is to identify accuracy or how
strongly errors are amplified. Estimating the systematic bias of a particular
CAP method still requires a model or empirical estimate of the direction of its
prevalence error.

## Separating the target classifier from the observation channel

The same matrix need not play both roles in a CAP method. In the reference
paper, the target classifier $h$ determines the performance functional, while
SQS-style methods use a distinct surrogate classifier $s$ for quantification.

Let

$$
d^h=\operatorname{diag}(C^h)
$$

describe how the accuracy of $h$ changes with prevalence, and let

$$
M^s_{rk}=P(s(X)=r\mid Y=k)
$$

describe the observation channel used to identify prevalence. A generalized
directional CAP complexity is

$$
\boxed{
\chi_{\mathrm{CAP}}(d^h,M^s)
=
\sup_{\substack{v\in T\\v\ne0}}
\frac{|(d^h)^\top v|}{\|M^s v\|_2}
}.
$$

This formulation distinguishes:

- **performance sensitivity**, determined by the target classifier $h$;
- **prevalence identifiability**, determined by the surrogate or observation
  model used by the CAP method.

When the predictions of $h$ itself are used to identify prevalence, set
$M^s=C^h$. For soft-output quantifiers, $M^s$ may need to be replaced by the
appropriate moment map or likelihood information rather than a crisp confusion
matrix.

This two-object formulation is likely more appropriate for comparing CAP
methods than a score derived only from $C^h$.

## Candidate empirical study

This is not yet a finalized protocol. A first study could compute, for every
dataset and fitted target classifier:

1. $S_{\mathrm{acc}}(C^h)$ and $\|d_T^h\|$;
2. $\chi_{\mathrm{acc}}(C^h)$;
3. for SQS-style methods, $\chi_{\mathrm{CAP}}(d^h,M^s)$;
4. target-specific $V_{\mathrm{acc}}$ or $D_{\mathrm{Fisher}}$ for every APP
   bag;
5. $W_{\mathrm{acc}}$ under one preregistered confidence-region construction.

Their relationships with signed CAP error, absolute error, interval coverage,
and Winkler score could then be evaluated. The central comparisons should be:

- the ordinary condition number and smallest singular value of the observation
  matrix;
- the restricted diagnostics $\sigma_{\min}(M^sB)$ and, when defined,
  $\kappa_2(M^sB)$ on the simplex tangent space;
- accuracy susceptibility alone;
- the proposed directional condition number;
- the target-specific variance and width measures;
- oracle quantities using true $\pi$ versus deployable plug-in quantities using
  $\widehat{\mathbf{c}}$ or $\widehat\pi$.

Analyses should be performed within method and classifier before aggregating.
APP bags drawn from the same held-out pool are dependent, so pooled bag-level
correlations are not sufficient evidence.

## Limitations and open questions

- **Exact-$C$ assumption.** The present analysis deliberately omits estimation
  error in $C$. Results obtained under this assumption are a best-case or
  conditional analysis, not complete CAP uncertainty.
- **Observation restriction.** Measures based on $\mathbf{c}=C\pi$ assume that
  target information is reduced to crisp predicted-label counts. A method using
  full soft outputs or covariates may contain more information.
- **Method dependence.** A structural condition number does not encode the
  prior, regularization, constraints, or bias of a particular quantifier.
- **Boundary behaviour.** Fisher information, Gaussian approximations, and
  inverse-based formulas require care near zero prevalence or zero predicted
  probability.
- **Choice of norm.** $\chi_{\mathrm{acc}}$ depends on the norm used to measure
  observable perturbations. Euclidean, total-variation, and information-based
  geometries should be compared.
- **Nonlinear metrics.** Macro-F1 and macro-K are nonlinear in the contingency
  table. Their gradients depend on $\pi$, and local sensitivity may diverge or
  become unstable near simplex boundaries.
- **Finite versus population target.** The matrix $C$ describes population
  conditional rates, while the paper evaluates performance on finite target
  bags. The chosen difficulty measure must state which estimand it addresses.
- **Calibration of $\varepsilon$.** Identification width is meaningful only if
  its tolerance corresponds to a valid target-sampling uncertainty statement.
- **Comparability.** Structural and target-specific quantities answer different
  questions and should not be collapsed into one score without justification.

## Current recommendation

The most promising structural quantity is

$$
\chi_{\mathrm{acc}}(C)
=
\sup_{v\in T\setminus\{0\}}
\frac{|d^\top v|}{\|Cv\|_2},
$$

because it combines accuracy sensitivity and prevalence identifiability while
ignoring irrelevant ill-conditioned directions.

The most interpretable operational quantity is

$$
W_{\mathrm{acc}}(C,\widehat{\mathbf{c}},\varepsilon),
$$

because it expresses difficulty directly as a range of plausible target
accuracies.

For SQS-style CAP, the two-matrix generalization

$$
\chi_{\mathrm{CAP}}(d^h,M^s)
$$

should be preferred when the target classifier and prevalence-estimation
surrogate are distinct.

These quantities should initially be presented as candidate characterizations
of identifiability and error amplification, not as estimators of CAP bias. The
method-specific bias remains $d^\top b_\pi$ and requires information about the
prevalence estimator's systematic error.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- [Possible extensions of the UNCAWE paper](extensions.md)
- [Lipton, Wang, and Smola. *Detecting and Correcting for Label Shift with Black Box Predictors*. ICML, 2018.](https://proceedings.mlr.press/v80/lipton18a.html)
- [Garg et al. *A Unified View of Label Shift Estimation*. NeurIPS, 2020.](https://papers.nips.cc/paper_files/paper/2020/hash/219e052492f4008818b8adb6366c7ed6-Abstract.html)
- [Rademacher-complexity intuition](rademacher.md)
