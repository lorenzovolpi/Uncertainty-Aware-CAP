# Work in progress: Rademacher complexity and target-side CAP difficulty

## Status of this document

This document is a research notebook for an early-stage intuition. It is not a
finished proposal, does not define an established method, and does not report an
implemented component of UNCAWE. Its theoretical object, operational estimator,
and experimental claim all require further study.

The current direction is deliberately recorded together with its weaknesses.
Some alternatives are retained because they may become necessary if the main
idea proves ill-posed or uninformative. Nothing described here should yet be
presented as a contribution of the reference paper.

## Starting intuition

The reference paper studies classifier accuracy prediction (CAP) under prior
probability shift (PPS). Let $h$ be the already-trained target classifier whose
performance must be estimated, and let

$$
U_i=\{x_{i1},\ldots,x_{in}\}
$$

be one unlabelled target bag. The current intuition is to characterize the
classification capacity of the *family to which $h$ belongs* on the observed
geometry of $U_i$, and then investigate whether this quantity is related to the
error made by a CAP method on that bag.

More precisely, let $\mathcal H_h$ denote a global hypothesis class associated
with the classifier family and structural hyperparameters of $h$. The tentative
object is

$$
\widehat{\mathfrak R}_{U_i}(\mathcal H_h),
$$

the empirical Rademacher complexity of $\mathcal H_h$ on $U_i$.

The intended construction is:

- **global:** it concerns the full class $\mathcal H_h$, not a neighbourhood
  localized around the fitted instance $h$;
- **marginal:** it is computed from the covariates in $U_i$, without attempting
  to split the bag by its unknown true classes;
- **label-free on the target:** the true labels of $U_i$ are not used to compute
  the score;
- **instance-independent:** the fitted parameters, margins, errors, and bias of
  the particular instance $h$ are intentionally excluded;
- **classifier-family-specific:** the class is meant to reflect the family of
  $h$, rather than the surrogate classifier used by a quantifier.

The motivation for excluding the fitted instance is to isolate a structural
notion of classification capacity from the quality of the particular model
trained on $L$. This is also an important limitation: the proposed score may
discard precisely the information that best predicts the actual behaviour of
$h$.

## Terminology: class complexity is not intrinsic problem difficulty

Rademacher complexity is not an intrinsic property of a dataset and does not,
by itself, measure the difficulty of the true labeling problem. It is a property
of a sample--function-class pair. For a real-valued class $\mathcal F$,

$$
\widehat{\mathfrak R}_{U_i}(\mathcal F)
=
\mathbb E_{\varepsilon}
\left[
\sup_{f\in\mathcal F}
\frac{1}{n}\sum_{j=1}^{n}\varepsilon_j f(x_{ij})
\right],
$$

where the $\varepsilon_j$ are independent signs taking values in
$\{-1,+1\}$ with equal probability.

Two classification problems can have the same marginal distribution $Q(X)$
and the same class $\mathcal H$, but radically different conditional label
distributions $Q(Y\mid X)$. They then have the same marginal Rademacher
complexity even if one problem is perfectly separable and the other is
intrinsically noisy.

The most accurate provisional terminology is therefore:

> target-conditioned hypothesis-class complexity

rather than "intrinsic classification-problem complexity." Whether this
class-dependent quantity is a useful proxy for CAP difficulty is the empirical
question to be investigated.

## Why the intuition may be relevant to CAP

Under PPS,

$$
P(X\mid Y)=Q(X\mid Y),
\qquad
P(Y)\ne Q(Y).
$$

For vanilla accuracy on a finite target bag $U_i$, write

$$
a_i=A(h,U_i)=d_i^\top\pi_i,
\qquad
d_i=\operatorname{diag}(C_i^h),
$$

where $\pi_i$ is the empirical class-prevalence vector and $C_i^h$ is the
empirical class-conditional classification-rate matrix of the fitted classifier
$h$ on $U_i$.

For a plug-in representation $\widehat a_i=\widehat d_i^\top\widehat\pi_i$,
the error has the exact decomposition

$$
\widehat a_i-a_i
=
\widehat d_i^\top(\widehat\pi_i-\pi_i)
+
(\widehat d_i-d_i)^\top\pi_i.
$$

The first term is associated with target-prevalence estimation. The second is
associated with the discrepancy between the conditional rates represented by
the CAP procedure and those realized on the finite target bag. Alternative
algebraic decompositions can separate a cross-term, but they describe the same
two basic sources.

The intuition considered here is that a target-conditioned capacity measure of
$\mathcal H_h$ might contain information about finite-sample variability in the
classification problem and, indirectly, about the stability of the realized
$C_i^h$. A tentative pathway is

$$
\widehat{\mathfrak R}_{U_i}(\mathcal H_h)
\longrightarrow
\text{finite-sample instability of classification behaviour}
\longrightarrow
\|C_i^h-C_V^h\|
\longrightarrow
|\widehat a_i-a_i|.
$$

This is only a hypothesis. Global complexity of $\mathcal H_h$ does not
determine the rate matrix of the particular fitted classifier $h$. At most,
standard Rademacher arguments may provide uniform control over empirical losses
for all classifiers in a suitably constrained class. Establishing a bound that
is relevant to class-conditional rates, finite target bags, and the CAP
estimand would require additional work.

The displayed plug-in decomposition also does not describe every CAP method's
internal computation. SQS-H, SQS-S, BayCAP, and the baselines represent
prevalence and conditional behaviour differently. It is useful as an analysis
of error sources, not yet as a common derivation of all their estimators.

## Fundamental issue: CAP evaluates a fixed classifier

The main theoretical objection to the current intuition is that Rademacher
complexity is especially useful when a learning procedure selects a function
from a class,

$$
\widehat h\in\mathcal H,
$$

and one needs to control the empirical-to-population deviation uniformly over
all candidates:

$$
\sup_{g\in\mathcal H}
\left|\widehat R(g)-R(g)\right|.
$$

The CAP setting in the reference paper is different. The target classifier
$h$ has already been trained on $L$ before the target bag is evaluated. CAP
does not select a new $g\in\mathcal H_h$ using $U_i$; it estimates the
performance of the one fixed function $h$.

For a fixed $h$, finite-sample variation in its class-conditional rates can be
studied directly without a supremum over $\mathcal H_h$. For example, let

$$
d_k=Q(h(X)=k\mid Y=k)
$$

and suppose that $U_i$ contains $n_{ik}$ observations of class $k$. Its
empirical true-positive rate is

$$
d_{ik}
=
\frac{1}{n_{ik}}
\sum_{j:Y_{ij}=k}
\mathbf 1[h(x_{ij})=k].
$$

Under an independent Bernoulli approximation conditional on $n_{ik}$,

$$
\operatorname{Var}(d_{ik}\mid n_{ik})
=
\frac{d_k(1-d_k)}{n_{ik}}.
$$

This variability depends on the rate of the fitted classifier and the number
of target examples in the class. It does not directly depend on the global
complexity of the family $\mathcal H_h$. Standard concentration inequalities
for one fixed function may therefore be sufficient; a uniform-complexity term
can be unnecessary or overly conservative.

### Where Rademacher complexity would enter more naturally

A Rademacher term would have a direct theoretical role if the objective were a
guarantee holding simultaneously for every classifier in the family. For each
class $k$, the relevant object would resemble

$$
\sup_{g\in\mathcal H_h}
\left|
\frac{1}{n_{ik}}
\sum_{j:Y_{ij}=k}\ell(g(x_{ij}),k)
-
\mathbb E_{Q(X\mid Y=k)}[\ell(g(X),k)]
\right|,
$$

which can be related to a class-conditional loss-class complexity such as

$$
\widehat{\mathfrak R}_{U_{ik}}
\left(\ell_k\circ\mathcal H_h\right).
$$

This is not the same object as the current proposal. It is class-conditional
rather than marginal, acts on a loss class rather than only on hard classifier
outputs, and controls every $g\in\mathcal H_h$ even though CAP evaluates a
single fixed $h$. Moreover, the subsets $U_{ik}$ are unavailable without target
labels.

### Consequence for the proposed interpretation

The global marginal score could still correlate with CAP error because
random-label fitting may indirectly capture aspects of the target geometry,
such as sparsity, isolated observations, duplicate structure, arbitrary
separability, or changes in mixture composition. These properties might in
turn correlate with quantification error or finite-sample deviations in
$C_i^h$. No such relationship is guaranteed.

For example, two fitted classifiers $h_1,h_2\in\mathcal H_h$ evaluated on the
same $U_i$ have the same value of
$\widehat{\mathfrak R}_{U_i}(\mathcal H_h)$, while their margins, errors, and
matrices $C_i^{h_1}$ and $C_i^{h_2}$ may be radically different. A family-level
score cannot distinguish their CAP difficulty.

The current motivation must therefore remain modest:

> The ability of the family of $h$ to fit random labels on the geometry of
> $U_i$ may be an exploratory, label-free proxy for aspects of target-bag
> difficulty not captured by prevalence, sample size, or shift magnitude.

There is currently no strong theoretical reason to expect the global marginal
Rademacher complexity of $\mathcal H_h$ to predict the CAP error of the fitted
instance $h$. The absence of an empirical relationship is plausible and would
be an informative negative result. If a direct theoretical connection to CAP
is required, class-conditional loss complexity, a class localized around $h$,
or instance-specific diagnostics such as margins, predictive entropy, model
disagreement, and bootstrap stability are more natural candidates.

## Candidate operational estimator: fitting random labels

### Binary identity

For a binary class $\mathcal H\subseteq\{-1,+1\}^{\mathcal X}$, empirical
Rademacher complexity can be written as

$$
\widehat{\mathfrak R}_{U_i}(\mathcal H)
=
2\,
\mathbb E_{\varepsilon}
\left[
\sup_{g\in\mathcal H}
\operatorname{Acc}(g;U_i,\varepsilon)
\right]-1,
$$

where $\varepsilon=(\varepsilon_1,\ldots,\varepsilon_n)$ is a random binary
labeling of the observations in $U_i$.

This identity suggests the following approximation:

1. clone the classifier family and structural hyperparameters associated with
   $h$, without copying its fitted parameters;
2. generate an independent random binary labeling of the observations in
   $U_i$;
3. train the cloned classifier on $(U_i,\varepsilon)$;
4. measure its in-sample agreement with the random labels;
5. repeat the procedure for several random labelings and, where appropriate,
   several optimizer initializations;
6. for each labeling, retain the highest agreement across training runs,
   transform it to the $[-1,1]$ correlation scale, and then average across
   labelings.

For one random labeling $r$, let $g^{(r)}$ be the classifier fitted on
$(U_i,\varepsilon^{(r)})$. Its *in-sample agreement* with the random labels is
its training accuracy on that same randomly labelled bag:

$$
\operatorname{Agr}^{(r)}
=
\frac{1}{n}\sum_{j=1}^{n}
\mathbf 1\!\left[
g^{(r)}(x_{ij})=\varepsilon_j^{(r)}
\right].
$$

Thus, "in-sample" means that no test split is created: the classifier is both
trained and evaluated on the same observations in $U_i$. This is intentional,
not data leakage. Empirical Rademacher complexity asks how strongly a function
class can correlate with arbitrary signs on the observed sample; it is not a
conventional predictive-generalization experiment.

For predictions and random labels in $\{-1,+1\}$,

$$
\varepsilon_j g(x_{ij})
=
\begin{cases}
+1, & g(x_{ij})=\varepsilon_j,\\
-1, & g(x_{ij})\ne\varepsilon_j.
\end{cases}
$$

It follows that

$$
\frac{1}{n}\sum_{j=1}^{n}\varepsilon_j g(x_{ij})
=
2\operatorname{Agr}(g,\varepsilon)-1.
$$

An agreement of $0.5$ therefore gives a score of $0$, an agreement of $0.75$
gives $0.5$, and perfect agreement gives $1$. With $M$ independent random
labelings, a single-fit operational estimate would be

$$
\widehat{\mathfrak R}^{\,A}_{U_i}
=
\frac{1}{M}\sum_{r=1}^{M}
\left(2\operatorname{Agr}^{(r)}-1\right).
$$

The theoretical definition contains a supremum over $g\in\mathcal H$. One
ordinary call to a training algorithm only returns one candidate and need not
attain that supremum. If $S$ optimizer initializations or randomized training
runs are used for every random labeling, the closer operational approximation
retains the best in-sample agreement for that labeling:

$$
\widehat{\mathfrak R}^{\,A}_{U_i}
=
\frac{1}{M}\sum_{r=1}^{M}
\left(
2\max_{s\in\{1,\ldots,S\}}
\operatorname{Agr}\!\left(g^{(r,s)},\varepsilon^{(r)}\right)
-1
\right).
$$

Multiple runs should therefore be maximized within each random labeling when
the purpose is to approximate the supremum. Averaging across random labelings
approximates the outer expectation over the Rademacher signs.

### What the training procedure actually estimates

A normal training algorithm does not generally solve

$$
\sup_{g\in\mathcal H}
\operatorname{Acc}(g;U_i,\varepsilon).
$$

It optimizes a surrogate objective with particular regularization, stopping
rules, initialization, and numerical approximations. If $A$ denotes one run of
this training procedure, the corresponding single-run expected score is better
written as

$$
\widehat{\mathfrak R}^{\,A}_{U_i}
=
2\,
\mathbb E_{\varepsilon,A}
\left[
\operatorname{Acc}\!\left(A(U_i,\varepsilon);U_i,\varepsilon\right)
\right]-1.
$$

Subject to compatible definitions, this is an algorithm-dependent lower
approximation to the supremum, not automatically the empirical Rademacher
complexity of the entire mathematical class. It mixes:

- representational capacity of the family;
- explicit and implicit regularization;
- optimization effectiveness;
- initialization and training randomness;
- implementation-specific behaviour.

Possible names currently under consideration are:

- *algorithmically approximated empirical Rademacher complexity*;
- *algorithm-dependent Rademacher lower bound*;
- *random-label fitting capacity*.

The last term is the most conservative if no evidence is obtained that the
optimizer approximates the supremum sufficiently well. The terminology remains
an open decision.

## Open definition of the hypothesis classes

Saying that $\mathcal H_h$ is "the class to which $h$ belongs" is not a complete
mathematical definition. The reference experiments use scikit-learn defaults
for Logistic Regression, k-Nearest Neighbours, MultiLayer Perceptron, and
Random Forest, and an RBF-kernel SVM with probability estimates. Each family
requires a different definition.

### Logistic Regression

The natural object is a class of linear scoring functions. A finite and
interpretable bound normally requires a constraint on the weight norm together
with a bound or normalization of the input norm. It is unresolved whether the
regularization parameter used by the training algorithm should be translated
into a hard norm constraint or retained only as part of the operational fitting
procedure.

### RBF Support Vector Machine

A standard formulation uses a ball in the RKHS induced by the RBF kernel. The
kernel bandwidth and RKHS norm radius must be fixed explicitly. The current
training parameter $C$ is a regularized-optimization parameter, not by itself a
complete definition of an RKHS ball. Without a norm constraint, the class can
be too rich for the proposed diagnostic.

### k-Nearest Neighbours

The classifier is defined through a reference set and its labels rather than a
fixed finite-dimensional parameter vector. It is unclear whether the intended
class should range over all possible labelled reference sets, all labelings of
a fixed reference set, or only predictors reachable by the exact training
procedure. These alternatives have different meanings and capacity.

### MultiLayer Perceptron

Architecture alone does not give the desired data-dependent complexity. Bounds
typically require constraints involving layer widths and weight norms. The
default optimizer and weight decay provide an operational training procedure,
but not an obvious global class with directly comparable capacity.

### Random Forest

The class depends on the number of trees, tree depth, splitting rules, feature
subsampling, and bootstrap sampling. Trees with unrestricted depth can fit or
nearly fit arbitrary labels on many finite samples. The random-label score may
therefore saturate, while its exact value may also depend strongly on bootstrap
and optimizer randomness.

These differences make a single cross-family scale difficult to justify. An
initial analysis may have to be performed within classifier family, treating
cross-family comparisons as secondary or unavailable.

## Multiclass formulation remains unresolved

The binary random-label identity does not transfer unchanged to $K>2$. Drawing
uniform random labels from $\{1,\ldots,K\}$ and normalizing accuracy above its
chance value,

$$
\frac{
\operatorname{Acc}_{\mathrm{random}}-1/K
}{
1-1/K
},
$$

would define an intuitive random-label fitting score, but it is not
automatically a standard multiclass Rademacher complexity.

A more direct multiclass object would operate on a vector-valued scoring class,
for example

$$
\mathbb E_{\sigma}
\left[
\sup_{f\in\mathcal F}
\frac{1}{n}
\sum_{j=1}^{n}\sum_{k=1}^{K}
\sigma_{jk}f_k(x_{ij})
\right],
$$

with an appropriate convention and normalization. Standard classifier
training on one random class label per observation does not directly optimize
this objective.

The current possibilities are:

1. begin with binary datasets, where the operational identity is clearest;
2. use normalized multiclass random-label accuracy and explicitly call it a
   proxy rather than Rademacher complexity;
3. define and implement a vector-valued Rademacher objective for every
   classifier family.

No option has yet been selected.

## Main conceptual and experimental problems

### 1. Richness is not labeling difficulty

The score observes $U_i$ and $\mathcal H_h$, but not $Q(Y\mid X)$. It cannot
identify Bayes error, class overlap, label noise, or true separability from the
target covariates alone. Any interpretation as intrinsic problem difficulty
would therefore be too strong.

### 2. The link to the fitted classifier is indirect

The score intentionally excludes the parameters of $h$. Two instances of the
same class can have different margins, biases, errors, and class-conditional
rate matrices on the same target bag. Global class complexity may consequently
explain little of the CAP error for the particular instance under evaluation.

### 3. Marginal complexity and class-conditional rates are different objects

The marginal target distribution is

$$
Q_i(X)=\sum_{k=1}^{K}\pi_{ik}Q(X\mid Y=k),
$$

whereas $C_i^h$ is defined through behaviour within each class. Under exact
PPS, the population distributions $Q(X\mid Y=k)$ do not change between source
and target; the APP bags change primarily through $\pi_i$. A marginal
complexity score may therefore react to mixture composition without measuring
within-class difficulty or rate instability.

### 4. Prevalence is a major confounder

Both the marginal complexity and CAP error may vary with target prevalence.
The apparent relationship

$$
\widehat{\mathfrak R}_{U_i}(\mathcal H_h)
\longleftrightarrow
|\widehat a_i-a_i|
$$

may be induced by the common cause $\pi_i$. Any future analysis must control
for at least shift magnitude, prevalence entropy, rare-class prevalence,
dataset, target-classifier family, and CAP method. It should also determine
whether the complexity score adds information beyond those simpler variables.

### 5. Flexible classes may saturate

RBF-SVM, MLP, and Random Forest classes may fit random labels almost perfectly,
producing scores close to their maximum for nearly all bags. In that case the
score contains too little between-bag variation to explain CAP difficulty.
Conversely, a low score can reflect optimizer failure rather than low class
capacity.

### 6. Cross-family values may not be comparable

The five classifier families use different output spaces, constraints,
regularizers, and optimizers. Raw values may only be interpretable within a
family. Standardizing them statistically does not by itself create a common
mathematical meaning.

### 7. APP sampling with replacement creates duplicates

Target bags are sampled with replacement. If the same covariate vector appears
multiple times and receives conflicting random labels, no deterministic
classifier can satisfy all copies. The random-label score can therefore depend
on the number and multiplicities of unique observations. This is compatible
with empirical Rademacher complexity on a bag, but it may turn the score into a
diagnostic of APP duplication rather than target geometry. Duplicate rate must
be measured as a baseline or control.

### 8. Bag size changes the scale

Rademacher complexity generally decreases with sample size under controlled
classes. Results at the reference size $n=1000$ cannot be pooled naively with
results at other sizes. Sample-size sensitivity, if studied, must be a separate
experimental dimension with explicit normalization and controls.

### 9. CAP error has several other sources

CAP error can arise from prevalence error, validation-rate uncertainty,
quantifier misspecification, finite-bag variability, Monte Carlo error, and PPS
violations. A target-side class-complexity score is not expected to identify all
of them. Its relationship may also differ across SQS-H, SQS-S, BayCAP, and the
baselines because they propagate uncertainty differently.

### 10. Computational cost may be substantial

The reference protocol crosses datasets, five target-classifier families, 100
APP bags, and several CAP methods. Repeating model fitting for many random
labelings and optimizer seeds could require hundreds of thousands of additional
training runs. The number of repetitions cannot be fixed until convergence and
variance are studied on a smaller pilot.

### 11. The APP observations are statistically dependent

APP bags are drawn from the same held-out pool and may share observations.
Pooled correlations or random bag-level train/test splits would overstate the
amount of independent evidence. Any future statistical analysis must respect
the dataset--classifier hierarchy and shared sampling pool.

### 12. Association would not establish causality

Even after adjustment for observed confounders, a relationship would support
the score as a diagnostic, not the claim that hypothesis-class complexity
causes CAP error. The first defensible result would be predictive or
associational.

## Tentative questions for an exploratory study

These are candidate research questions, not a finalized experimental plan.

1. Does the proposed global marginal score vary meaningfully between target
   bags within the same dataset and classifier family, or does it saturate?
2. Is it associated with the finite-sample deviation
   $\|C_i^h-C_V^h\|$, measured retrospectively using target labels?
3. Is it associated with signed CAP error or absolute error?
4. Does an association remain after controlling for target prevalence, shift
   magnitude, entropy, duplicate rate, and quantification error?
5. Does the score add information beyond simpler diagnostics such as input
   covariance, margins, prediction entropy, or target-bootstrap variability?
6. Is the relationship method-specific, as expected from the different
   uncertainty mechanisms of SQS-H, SQS-S, BayCAP, and the baselines?
7. Does a relationship learned on some datasets generalize to a held-out
   dataset, or is it dataset-specific?

Target labels may be used to compute $C_i^h$, true accuracy, and error outcomes
for retrospective scientific evaluation. They must never enter the proposed
complexity score if it is to remain a deployable label-free diagnostic.

## Alternative directions retained for study

The following ideas are not part of the current core intuition, but remain
possible responses to its limitations.

### Local complexity around the fitted classifier

A class localized around $h$ through norm, margin, loss, or parameter distance
would have a more direct connection to the evaluated classifier. It would,
however, intentionally reintroduce instance-specific information that the
current global proposal aims to exclude.

### Class-conditional complexity

One could estimate

$$
\widehat{\mathfrak R}_{V_k}(\ell_k\circ\mathcal H_h)
$$

on labelled validation examples of class $k$, then combine the class-specific
values using estimated target prevalence. This is more closely aligned with
$C^h$, but it is no longer a complexity computed directly and marginally on
$U_i$ and may reduce to a prevalence-weighted class-difficulty score. An oracle
version on the true subsets $U_{ik}$ could be used only for retrospective
analysis.

### Quantifier-aligned representation complexity

The previous version of this document considered a norm-bounded linear or
kernel class on the surrogate classifier's posterior representation. That
quantity may be more directly connected to target-prevalence estimation, but it
answers a different question: quantifier difficulty rather than the complexity
of the classification family of $h$.

### Direct random-label capacity without a Rademacher claim

If the optimizers, multiclass construction, or family definitions prevent a
defensible approximation of empirical Rademacher complexity, random-label
fitting capacity can still be studied as an operational statistic in its own
right. The scientific claim and terminology would then need to change
accordingly.

## Relationship to current UNCAWE work

The statuses of the relevant items must remain distinct:

| Item | Status |
|---|---|
| SQS-H, SQS-S, and BayCAP | Proposed and evaluated in the reference paper |
| RQBS implementation of SQS-H | Used by current UNCAWE experiments through `cap-ml` |
| Aitchison-SQS | Research proposal; not implemented |
| SQS-B | Research proposal; not implemented |
| BayCAP-KDEy | Research proposal; implemented in UNCAWE (`uncawe.methods.BayCAPKDEy`), not yet evaluated |
| Target-conditioned Rademacher intuition | Work in progress; definition unresolved; not implemented |

This draft is related to the uncertainty-decomposition direction in
[extensions.md](extensions.md), but it does not yet define a new CAP method or
an uncertainty-calibration procedure.

## Open decisions and next steps

Before this draft can become a research proposal, the following decisions and
checks are required:

1. define $\mathcal H_h$ mathematically for each classifier family, or restrict
   the initial scope to families for which a controlled definition is possible;
2. decide whether the training-based score is an approximation to Rademacher
   complexity or should be called random-label fitting capacity;
3. select a rigorous or explicitly proxy-based multiclass formulation;
4. determine whether results are interpretable only within family or admit a
   justified cross-family normalization;
5. run a small pilot to measure saturation, optimizer sensitivity, Monte Carlo
   variance, duplicate effects, and computational cost;
6. choose whether signed CAP error, absolute error, deviation in $C_i^h$, or
   interval quality is the primary outcome;
7. specify the confounder adjustment and hierarchical evaluation protocol;
8. compare against simpler label-free statistics before attributing value to
   Rademacher complexity specifically;
9. investigate whether a theoretical uniform-convergence statement can be
   connected to finite-sample class-conditional rates under PPS;
10. only after these steps, decide whether an adaptive CAP calibration method
    is justified.

The current working question is therefore intentionally modest:

> Can the ability of the target classifier family to fit random labels on the
> geometry of an unlabelled target bag serve as a useful, label-free diagnostic
> of finite-sample rate instability and CAP error under prior probability
> shift?

This question may eventually lead to a Rademacher-based diagnostic, a more
conservative random-label capacity measure, or a negative result showing that
global hypothesis-class complexity is not the appropriate notion of difficulty
for CAP.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- [Possible extensions of the UNCAWE paper](extensions.md)
- [Aitchison-SQS proposal](aitchison_sqs.md)
- [SQS-B proposal](sqs_b.md)
- [BayCAP-KDEy proposal](baycap_kdey.md)
- [Bartlett and Mendelson. *Rademacher and Gaussian Complexities: Risk Bounds and Structural Results*. JMLR, 2002.](https://www.jmlr.org/papers/v3/bartlett02a.html)
- [Bartlett, Bousquet, and Mendelson. *Local Rademacher Complexities*. Annals of Statistics, 2005.](https://arxiv.org/abs/math/0508275)
- [El-Yaniv and Pechyony. *Transductive Rademacher Complexity and Its Applications*.](https://arxiv.org/abs/1401.3441)
