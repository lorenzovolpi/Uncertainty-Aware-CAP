# Possible extensions of the UNCAWE paper

## Starting point

The paper [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf) proposes SQS-H, SQS-S, and BayCAP for estimating both the performance of a classifier on unlabelled target data and the uncertainty of that estimate. The current evaluation focuses mainly on vanilla accuracy, tabular datasets, and artificially generated prior probability shift (PPS).

An extended version should go beyond increasing the number of datasets and address at least one new scientific question. The directions below may be combined, although they differ in ambition and cost.

## 1. [[ext1_metrics|Metrics beyond vanilla accuracy]]

A cross-cutting evaluation axis rather than a standalone contribution. It fixes the metric protocol shared by all other extensions: vanilla accuracy (Acc), macro-F1 (F1 of the positive class on binary data, reported separately), and macro-K (macro-averaged informedness), with the paper's table convention $T=C\operatorname{diag}(\pi)$, the sample mean $\hat A=\frac1m\sum_r A(T^{(r)})$ as point estimate, and explicit edge-case conventions for rare or absent classes.

Its own analysis is analysis-only on existing `main` results: per-metric point and interval quality, the gap $\lvert A(\bar T)-\hat A\rvert$, asymmetry of the metric distributions, and behaviour on bags with rare or absent classes. It also tests a new hypothesis: under PPS, Acc is linear in $\pi$, macro-F1 is nonlinear in $\pi$, and macro-K depends on $\pi$ only weakly (not at all for $K=2$), so comparing interval quality across metrics separates prevalence-estimation error from error in $C$.

The full protocol and analyses are described in [ext1_metrics.md](ext1_metrics.md). No new runs are required.

## 2. [[ext2_baycap|A deeper study of BayCAP: estimand, sources of miscalibration, realised-table predictive]]

This extension asks what BayCAP's interval actually estimates and why it is miscalibrated with respect to the quantity CAP is scored on. BayCAP's posterior describes the population table $C\operatorname{diag}(\pi^Q)$, while the evaluation scores the realised table of the target sample $U$. The candidate *Calibrated BayCAP* is a realised-table posterior predictive: for each posterior draw, it samples the true classes behind the observed target predictions and applies the metric to the resulting table of $U$.

A new hypothesis drives the design. BayCAP's miscalibration has four sources: the APP protocol fixes the class counts that the model treats as random (intervals too wide); the interval targets the population table while the evaluation scores the realised table of $U$ (too narrow); finite $V$ leaves uncertainty on $C$ (modelled); and the uniform $\operatorname{Dir}(\mathbf 1)$ prior on the columns of $C$ pulls recalls towards $1/K$, displacing the interval on datasets with many classes and few validation instances per class (biased). In a binary toy simulation, the estimand gap dominates once $V$ has many instances per class, and the realised-table predictive restores near-nominal coverage without tuning. On stored `main` results, BayCAP's coverage is $0$ on `isolet` and `letter` for every classifier, and widening the interval does not recover it, as the fourth source predicts.

The work is experiment-driven. E2.1 approximated the effect of the predictive on accuracy intervals from stored `main` results, at no run cost (`tables/estimand_gap.py`, in review); its draft analysis finds that the predictive improves the Winkler score on every under-covering pair, but that most under-covering pairs also carry a bias that no widening removes. E2.2, scheduled next, checks the predictive exactly on a few datasets with `BayCAP`, a copy of `ACC_BayesCAP` that supports both estimands through a hyperparameter. Candidate experiments, considered after E2.2: a diagnosis of the direction of miscalibration (Q1), a factorial isolating the estimand gap and the fixed counts (Q2), and a decomposition of BayCAP's bias into a recall and a prevalence term (Q5). Tempering and the choice of priors are in Extension 10.

The experiments, their analyses, and the background are described in [ext2_baycap.md](ext2_baycap.md). `BayCAP` is implemented but not evaluated.

## 3. [[ext3_uncertainty_budget|The uncertainty budget of CAP under PPS]]

The theoretical frame of the extended paper. The error $\hat A-A(h,U)$ of every quantification-based CAP estimate splits exactly into a prevalence part, a rate-matrix part, and the finite-sample fluctuation of the realised table of $U$. The finite-$U$ part and the target-sampling part of the prevalence error shrink as $\lvert U\rvert$ grows; the error of $C$ estimated on $V$ and the part of the prevalence error caused by fitting the quantifier on $V$ do not. Each CAP method represents only some of these terms: SQS-H and its bootstrap variant only the shrinking ones, BayCAP only the fixed ones.

A new hypothesis follows: a method's interval is wide only along the parts it simulates, so its coverage is governed by the ratio $\eta$ of missing to represented variance, approximately $2\Phi(1.96/\sqrt{1+\eta})-1$. Because the parts scale differently with $\lvert U\rvert$ and $\lvert V\rvert$, each method is calibrated only in one regime of the design ratio $r=\lvert U\rvert\sum_j(\pi^Q_j)^2/\lvert V_j\rvert$, and the paper's ranking at $\lvert U\rvert=1000$ is one point of a regime map. In a binary toy simulation, SQS-H loses coverage as $r$ grows, BayCAP as $r$ falls, and methods that represent all terms (the realised-table BayCAP of Extension 2, and SQS-H with resampling of both $U$ and $V$, a new candidate) stay near nominal. Extensions 2, 8 and 9 are presented as completing missing terms of the budget.

The study varies $\lvert U\rvert$ and, at fixed allocation, a stratified subsample of $\lvert V\rvert$, with outer seeds for $V$, an oracle ladder (true $\pi^Q$, rates from the labelled pool, population scoring) as reference, and both paper and proposed methods. The validation allocation $\rho$ stays in Extension 6; PPS violations are set aside in [[optional_extensions]] §2. It absorbs proposal A of [[extensions_to_evaluate]].

The full formulation, hypotheses, and design are described in [ext3_uncertainty_budget.md](ext3_uncertainty_budget.md). It is not implemented; the toy simulation is `scripts/toys/ext3_budget.py`.

## 4. Geometric analysis and kernel variants of SQS-S

SQS-S selects validation bags by comparing estimated prevalences through a distance on the simplex. One could systematically study how the distance and neighbourhood definition affect bias and variance:

- compare $L_1$, Hellinger, Jensen--Shannon, and Aitchison distances;
- measure the correspondence between proximity in predicted and true prevalence space;
- study sensitivity to $\gamma$;
- replace hard neighbour selection with kernel weights;
- select bandwidth adaptively according to local density.

This interpretation connects SQS-S to local regression or approximate Bayesian computation on the simplex. The contribution becomes substantial if it yields an adaptive rule, rather than only a comparison of distance functions.

## 5. Multiclass scalability and new domains

The evaluation could be extended to text and image data, pretrained models, and problems with many classes. In addition to predictive quality, the analysis should measure:

- time and memory as functions of $K$, $|V|$, $|U|$, and the number of samples;
- the cost of BayCAP and NUTS sampling;
- behaviour with rare or absent classes;
- stability when the confusion matrix is ill-conditioned.

New domains would make the empirical evidence more convincing, but should not constitute the central contribution on their own.

## 6. Validation-budget allocation in SQS-S

SQS-S divides the labelled validation set $V$ into two parts: one part is used to train the surrogate quantifier, while the other supplies the labelled bags used by the SQS protocol. For a fixed validation budget, let

$$
|V_q|=\rho|V|,
\qquad
|V_b|=(1-\rho)|V|,
$$

where $V_q$ is the quantifier-training subset and $V_b$ is the bag-sampling subset. Increasing $\rho$ can improve the fitted quantifier and therefore the matching of the target sample to validation bags, but it reduces the empirical support available for constructing those bags. Decreasing $\rho$ produces the opposite trade-off. Sampling bags with replacement does not remove the loss of class-conditional diversity caused by a small $V_b$.

The reference paper defines SQS-S with a stratified split $|V_1|\approx|V_2|$, where the quantifier is trained on $V_1=V_q$ and the candidate bags are drawn with replacement from $V_2=V_b$; the paper's baseline is therefore $\rho=0.5$. The `cap-ml` `PrediQuant` implementation used by the current experiments performs the same $50\%/50\%$ stratified split, but fits the quantifier on all of $V$ rather than on $V_1$, so $V_q\cap V_b\neq\emptyset$; this must be corrected before the allocation study, and results of `main` obtained before the correction do not correspond exactly to the paper's SQS-S. The proposed study is not implemented: it should vary $\rho$ while holding $|V|$, $|U|$, the target bags, and the total number of SQS samples fixed. Repeated stratified splits are needed to separate a systematic allocation effect from the particular observations assigned to either subset.
The evaluation should track:

- quantification error on $U$;
- proximity between selected bags and $U$ in estimated- and true-prevalence space;
- point-estimation error, interval coverage, and interval width;
- the effective diversity and class support of the selected validation bags;
- training and inference cost.

This is a validation-budget allocation study, not part of the uncertainty budget of Extension 3, which subsamples $|V|$ at fixed $\rho$. A secondary factorial analysis may later test interactions between $\rho$, $|V|$ and $|U|$, but the primary experiment should vary only $\rho$ so that the effects remain identifiable.

## 7. Grid-search optimization of classifiers and quantifiers

The current experiments mostly use default classifier hyperparameters and fixed configurations for the classifiers embedded in the quantifiers. A proposed study, not currently implemented, should determine whether the conclusions about CAP methods remain stable when these predictive components are optimized by grid search.

The optimization must distinguish three objects with different roles:

- the target classifier $h$, whose performance on $U$ is being estimated;
- the surrogate classifier $s$ used by an aggregative quantifier to produce posterior representations;
- the quantifier $q$ itself, including method-specific hyperparameters such as regularization, bandwidth, kernel, or other aggregation parameters.

These objects should not necessarily use the same selection objective. The target classifier $h$ can be selected by an out-of-sample classification score, possibly including a calibration criterion when its posterior probabilities are consumed downstream. By contrast, $s$ and $q$ should be selected primarily by quantification error on source-generated validation bags spanning multiple prevalences, rather than only by instance-level classification accuracy. Bayesian quantifiers may additionally require a proper score or coverage-based criterion for selecting posterior-shape parameters.

The experimental protocol must remain nested and label-safe. Hyperparameters for $h$ should be selected only within its labelled training data. Parameters for $s$ and $q$ should be selected through inner splits or cross-fitting within the labelled data allocated to quantifier training. Target labels and the bags used for final CAP evaluation must never influence model selection. For SQS-S, the search must also respect the separation between $V_q$ and $V_b$ introduced in Extension 6, so that tuning does not consume the bag-sampling subset.

A factorial ablation should compare:

1. default $h$ and default $(s,q)$;
2. optimized $h$ and default $(s,q)$;
3. default $h$ and optimized $(s,q)$;
4. optimized $h$ and optimized $(s,q)$.

The parameter grids, selection metrics, folds, random seeds, and computational budgets should be fixed in advance and shared across CAP methods whenever their interfaces permit it. Results should report both predictive quality and search cost. Tuning $h$ changes the classifier and therefore the CAP problem itself, whereas tuning $(s,q)$ changes the estimator of target prevalence; these effects must be reported separately rather than summarized as a single gain from "optimization."

## 8. [[sqs_b|SQS-B: Bayesian prevalence propagation through validation sampling]]

SQS-B is a research proposal that replaces the point or bootstrap prevalence representation used by SQS-style methods with joint draws from a Bayesian quantifier:

$$
\pi^{(r)}\sim p_q(\pi^Q\mid U,V).
$$

For every posterior draw, the method samples one target-sized validation bag with replacement and with true prevalence approximately equal to $\pi^{(r)}$. Applying the classifier and the performance functional to every bag produces an empirical distribution of target-performance values.

The quantifier interface is generic: BayesianCC is a natural instantiation, but SQS-B only requires joint posterior draws on the simplex. Marginal intervals for individual classes are insufficient because independent sampling would discard posterior dependence and could generate invalid prevalence vectors.

SQS-B differs from BayCAP because it propagates only the posterior over target prevalence and uses empirical validation sampling for conditional classifier behaviour. It also differs from Aitchison-SQS, which constructs or estimates a distribution around a non-Bayesian point estimate. The resulting SQS-B interval is hybrid Bayesian and empirical and should not automatically be described as a fully Bayesian credible interval.

The current RQBS implementation already maps bootstrap prevalence estimates to validation bags. Consequently, merely replacing bootstrap draws with posterior draws is likely an incremental contribution. A convincing evaluation must compare posterior-mean SQS, bootstrap RQBS, full SQS-B, BayCAP, and an oracle using the true target prevalence, while measuring both prevalence-posterior calibration and downstream interval quality.

The complete formulation, interpretation, limitations, and decisive ablations are described in [sqs_b.md](sqs_b.md). SQS-B is not currently implemented.

## 9. [[baycap_kdey|BayCAP-KDEy: two-stage Bayesian pushforward with a fixed rate matrix]]

A final direction combines the strongest Bayesian quantifier in the KDE-Bayes line [kde_bayes.pdf](papers/kde_bayes.pdf) — the shrinkage-regularised, Aitchison-geometry KDE quantifier (BayesKDEy) — with a *parametric* classifier bridge. Given the posterior over target prevalence $\pi^{(r)}\sim p_T(\pi^Q\mid U,V)$, the method holds fixed the target classifier's rate matrix $C^h$ (measured on the validation set) and applies two successive pushforwards:

$$
T^{(r)}=C^h\,\operatorname{diag}\!\left(\pi^{(r)}\right),
\qquad
a^{(r)}=A\!\left(T^{(r)}\right).
$$

For vanilla accuracy this collapses into a single linear map $\operatorname{Acc}=d^\top\pi^Q$ ($d=\operatorname{diag}C^h$), yielding closed form for the predictive mean and variance without any sampling noise; for nonlinear metrics (macro-F1, macro-K) the full two-stage functional is required and the resulting distributions may be asymmetric. The propagation is therefore *parametric-Bayesian*: it includes all prevalence uncertainty but none of the uncertainty on $C^h$ itself, which is the key trade-off with BayCAP (posterior on $C$) and with SQS-B (empirical-bag propagation).

[[baycap_kdey|BayCAP-KDEy]] occupies a distinct position in this family: where [[aitchison_sqs|Aitchison-SQS]] *constructs* a prevalence distribution and [[sqs_b|SQS-B]] *samples* validation bags, BayCAP-KDEy *propagates analytically through a fixed rate matrix*. The decisive ablation holds the prevalence posterior fixed and swaps only the propagation mechanism (C-fixed vs C-random vs empirical-bag) to isolate which uncertainty source dominates under PPS.

The full formulation, theory, limitations, and decisive ablations are described in [baycap_kdey.md](baycap_kdey.md). BayCAP-KDEy is implemented in UNCAWE as `uncawe.methods.BayCAPKDEy`, but it has not yet been experimentally evaluated.

## 10. [[ext10_baycap_priors_tempering|Informative priors and posterior tempering for BayCAP]]

The paper names two unexplored features of BayCAP: informative priors and posterior tempering. This extension studies both, using the estimands and the realised-table predictive of Extension 2.

Tempering is studied first as a diagnostic: a sweep over the temperature $\tau$, evaluated with target labels, gives the oracle-best $\tau$, an upper bound on what tempering can achieve, which is compared with the realised-table predictive. Selecting $\tau$ without target labels is run only if tempering clearly beats the predictive.

Priors are studied on both parameters. On $\pi^Q$, the paper's bags are drawn uniformly from the simplex, so its uniform prior is already the correctly specified one, and the prior's weight is about $c/(c+|U|/\tau)$, negligible at $|U|=1000$; a minimal check (one non-uniform bag-generating distribution, uniform, well-specified and confidently wrong priors, $|U|\in\{100,1000\}$) precedes the full study. On $C$, a new hypothesis: the uniform prior on each column is informative, since its weight of $K$ imaginary validation instances per class grows with the number of classes; a prior whose total weight does not grow with $K$ (Perks' $\operatorname{Dir}(1/K)$) or a diagonal-aware prior should remove BayCAP's bias on large-$K$ datasets. Tempering with $\tau>1$ increases the weight of the prior, and so this bias.

The full formulation, hypotheses, and design are described in [ext10_baycap_priors_tempering.md](ext10_baycap_priors_tempering.md). `ACC_BayesCAP` accepts a prior on $\pi^Q$ and the `binform` experiment covers one prior configuration; tempered BayCAP and a configurable prior on $C$ are not implemented. It revives two studies previously set aside in [[optional_extensions]] (full informative-prior study, label-free temperature selection).

## Extensions under evaluation

Further candidate extensions are collected in
[[extensions_to_evaluate]] (the variance decomposition of proposal A has been promoted into Extension 3):

- a transductive BayCAP linked to CBPE;
- an ABC reinterpretation of SQS-S with regression adjustment;
- conformal calibration under simulated label shift.

These are hypotheses that still need evaluation (feasibility, novelty, and preliminary evidence). They are not ready to be implemented and should not be treated as part of the proposals above until they are promoted.

## Optional extensions

Additional proposals for extending the paper [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf) can be found in [[optional_extensions]]. These extensions were considered not worthy of being included in the final version of the extended paper and set aside for the moment.

## Recommended combinations

### Incremental and manageable extension

Evaluate multiple metrics, use the uncertainty budget of Extension 3 as the frame and its $(|U|,|V|)$ regime map as the main experiment, add the study of BayCAP (realised-table predictive, Extension 2) with the tempering diagnostic of Extension 10, and verify the main findings with optimized classifiers and quantifiers. This combines Extensions 1--3, 7 and 10 and is the lowest-risk option, although it remains mainly an empirical contribution unless the results support a new calibration or selection rule.

### SQS-S design extension

Combine the geometric and kernel variants of SQS-S with the validation-budget allocation study in Extensions 4 and 6. Jointly selecting the simplex geometry, neighbourhood rule, and allocation $\rho$ would turn two sensitivity analyses into a coherent study of how SQS-S uses limited validation data. Extension 3 can then test whether the selected design remains stable as $|U|$ changes, while Extension 7 determines whether it depends on untuned surrogate models.

### Posterior-propagation extension

Compare [[sqs_b|SQS-B]] directly with bootstrap-based RQBS and BayCAP across the target sizes of Extension 3 and the metrics of Extension 1. This design isolates the effect of replacing a bootstrap prevalence distribution with a Bayesian posterior, while also comparing empirical validation sampling against BayCAP's joint parametric propagation. This is the most natural setting in which SQS-B could support a substantive scientific claim rather than a component substitution. Extension 7 should be applied to all competing quantifiers under the same search budget.

### Parametric-bridge posterior extension (BayCAP-KDEy)

Compare [[baycap_kdey|BayCAP-KDEy]], [[sqs_b|SQS-B]] using the same BayesKDEy posterior, a fixed-matrix propagation of the BayesianCC posterior, and BayCAP across the target sizes of Extension 3 and on at least three contingency-table metrics (vanilla accuracy, macro-F1, macro-K). This design isolates, one axis at a time, the prevalence posterior (KDE vs BayesianCC) from the propagation mechanism (fixed matrix vs empirical bag vs random matrix), which is the cleanest setting in which BayCAP-KDEy could support a substantive scientific claim about which uncertainty source dominates under PPS. The grid-search protocol of Extension 7 should tune each posterior model without using target labels and should retain untuned variants as ablations.

### More ambitious extension

Combine the SQS-S geometry and allocation studies with optimized surrogate models and either SQS-B or BayCAP-KDEy, then evaluate the resulting methods across multiple metrics, target sizes, class counts, and domains. The scientific message would become: which representation and propagation of uncertainty remains reliable as the target sample and problem scale change under strict PPS, and are the conclusions robust to model selection?
