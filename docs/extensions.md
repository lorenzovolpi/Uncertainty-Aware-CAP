# Possible extensions of the UNCAWE paper

## Starting point

The paper [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf) proposes SQS-H, SQS-S, and BayCAP for estimating both the performance of a classifier on unlabelled target data and the uncertainty of that estimate. The current evaluation focuses mainly on vanilla accuracy, tabular datasets, and artificially generated prior probability shift (PPS).

An extended version should go beyond increasing the number of datasets and address at least one new scientific question. The directions below may be combined, although they differ in ambition and cost.

## 1. [[ext1_metrics|Metrics beyond vanilla accuracy]]

A cross-cutting evaluation axis rather than a standalone contribution. It fixes the metric protocol shared by all other extensions: vanilla accuracy (Acc), macro-F1 (F1 of the positive class on binary data, reported separately), and macro-K (macro-averaged informedness), with the paper's table convention $T=C\operatorname{diag}(\pi)$, the sample mean $\hat A=\frac1m\sum_r A(T^{(r)})$ as point estimate, and explicit edge-case conventions for rare or absent classes.

Its own analysis is analysis-only on existing `main` results: per-metric point and interval quality, the gap $\lvert A(\bar T)-\hat A\rvert$, asymmetry of the metric distributions, and behaviour on bags with rare or absent classes. It also tests a new hypothesis: under PPS, Acc is linear in $\pi$, macro-F1 is nonlinear in $\pi$, and macro-K depends on $\pi$ only weakly (not at all for $K=2$), so comparing interval quality across metrics separates prevalence-estimation error from error in $C$.

The full protocol and analyses are described in [ext1_metrics.md](ext1_metrics.md). No new runs are required.

## 2. [[ext2_baycap|Does BayCAP describe the right accuracy?]]

BayCAP's interval describes the population table $C\operatorname{diag}(\pi^Q)$, i.e. the accuracy the classifier has on average over bags with a given prevalence, while the paper scores it against the realised table of the target bag $U$. On a finite bag the two differ, because the classifier is right on a random number of instances. The extension asks whether this mismatch makes BayCAP's intervals miscalibrated on the paper's datasets and, if so, fixes it with the realised-table posterior predictive: for each posterior draw, it samples the true classes behind the observed target predictions and applies the metric to the resulting table of $U$, with no target labels and no tuning. In a binary toy simulation, the population interval covers the realised accuracy only $63\%$ of the time when $V$ has $5000$ instances per class, and the predictive restores near-nominal coverage.

The document is experiment-driven. E2.1 approximates the predictive for accuracy on stored `main` results (`tables/estimand_gap.py`): BayCAP under-covers on $53$ of $75$ (dataset, classifier) pairs, but the predictive repairs only $5$, because on many pairs, and on every pair of `isolet` and `letter`, the interval is displaced rather than too narrow. E2.2, at no run cost, studies three candidate causes of the displacement on stored results: the uniform prior on the columns of $C$, which pulls the recalls towards $1/K$; an ill-conditioned $C$, which amplifies errors into the estimated prevalences; and the difficulty of the problem on i.i.d. data. It splits BayCAP's error into a recall term and a prevalence term, and reports per pair the conditioning of $C$ (condition number $\kappa_0$ and absolute amplification $1/\sigma_{\min}$ on vectors summing to zero; $\kappa_0$ is always $1$ when $K=2$) and the accuracy of the classifier on $V$. Its run (2026-10-20, generated again on the CPU results of `BayesCAP` on 2026-10-22) finds that coverage is almost entirely a matter of displacement: the prior on $C$ accounts for every pair with coverage near $0$ (large $K$, few validation instances per class, recall term equal to its prior part); a badly conditioned $C$ displaces the interval only when the prevalence error has the same sign in every bag (`german`, `wine-quality`), and otherwise just widens it; a low accuracy on $V$ does not act on its own; on binary datasets a smaller recall error remains that the prior does not explain, possibly a difference between $V$ and the test pool shared by all bags (cause 4, a new hypothesis, recorded as a guess and not tested). E2.3 runs the exact predictive in `main` as the method `BayesCAP-real` (`BayCAP` with `estimand="realised"`, excluded by default from the paper's tables) and compares it with the population interval: the realised interval is wider by exactly the bag-level spread, $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$ times, and its centre does not move (within $0.009$), so the causes found by E2.2 are unchanged; coverage never drops, $16$ pairs reach $0.95$ (from $12$ to $28$ pairs at or above $0.95$), and a paired bootstrap finds the Winkler score significantly lower on $31$ pairs, all displaced ones, and slightly higher on $9$ pairs that already covered; the approximation of E2.1 is accurate within about $0.01$ of coverage. E2.4 replaces the uniform prior on the columns of $C$ with an empirical-Bayes prior: the recalls of a classifier get a common Beta prior whose centre (the typical recall) and strength are fitted to the validation counts by maximum likelihood of the beta-binomial, so each recall is pulled towards the typical recall of the classifier instead of towards $1/K$; only the concentration parameters of the Dirichlet prior change, so BayesianCC is otherwise unchanged. It runs the new prior with both pushforwards (`BayesCAP-EB`, `BayesCAP-EB-real`) against `BayesCAP` and `BayesCAP-real`, predicts its effect on the recall term in closed form from $V$ alone (a zero-cost check), and stores the NUTS diagnostics of all four methods. Its run (2026-10-26) finds that the check predicts the change of the recall term almost exactly (correlation $0.9995$); the large-$K$ displacement disappears (`isolet` and `letter` go from coverage $0$ to $0.84$–$0.96$ with the realised pushforward), and with the realised pushforward the pairs with coverage at least $0.90$ go from $34$ to $52$ and the mean Winkler score from $0.84$ to $0.32$. Two problems remain: on binary datasets the fit has two counts to estimate a spread, its strength can reach the upper bound and make the interval too narrow (`mammographic`), and no binary pair gains coverage; and with many classes and accurate classifiers the off-diagonal concentrations fall below $1$, NUTS diverges in most bags on $18$ pairs, and the gains there (all of `isolet`, most of `poker_hand`) cannot yet be trusted. A diagnosis against the size of $V$ (E2.c1) and a factorial separating the estimand gap from the fixed class counts (E2.c2) are further candidates. Tempering and other priors are in Extension 5.

The full description is in [ext2_baycap.md](ext2_baycap.md).

## 3. [[ext3_posterior_propagation|Propagating a Bayesian prevalence posterior to classifier performance]]

Merges the former Extensions 8 (SQS-B) and 9 (BayCAP-KDEy) (2026-10-27). The extension adds two CAP methods that turn the posterior of a Bayesian quantifier over $\pi^Q$ into an interval on performance, and compares them with the methods of the paper. BayCAP-KDEy extends the Bayesian KDEy quantifier (BayesKDEy, [kde_bayes.pdf](papers/kde_bayes.pdf)): each joint posterior draw $\pi^{(r)}$ is pushed through the rate matrix of $h$ measured on $V$, $\hat C\operatorname{diag}(\pi^{(r)})$, either as a population table or, with the realised-table predictive of Extension 2, as a plausible realised table of the target bag. SQS-B does what the other SQS methods do, but draws the prevalences of its validation bags (one bag per joint draw) from the posterior of a Bayesian quantifier, BayesKDEy, instead of from a point estimate or a bootstrap. Each method differs from an existing one in a single component, so the comparison says which component matters; the uncertainty budget of Extension 4 predicts which error terms each method represents, and Extension 4 reuses the same implementations in its regime map.

The theory derives that, with a fixed $\hat C$, accuracy is $\hat d^\top\pi$, so the population interval has variance $(\hat d-\bar d\mathbf 1)^\top\Sigma_\pi(\hat d-\bar d\mathbf 1)$ and is narrow when the recalls are similar; that it leaves out the luck of the bag; that the realised pushforward should widen it by $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$ and, unlike in BayCAP, may move its centre, because BayesKDEy does not fit $\pi^Q$ to the predictions of $h$; and that $\hat C$ has no prior, so the pull towards $1/K$ that displaces BayCAP on large-$K$ datasets is absent (new hypotheses). E3.1 (not scheduled, not run) evaluates `BayCAP-KDEy` and `BayCAP-KDEy-real` on the protocol of `main`, with BayesKDEy at temperature $\tau=1$; its kernel, bandwidth and shrinkage are still to be decided. Candidates: SQS-B on BayesKDEy (E3.c1), BayCAP with a fixed $\hat C$ (E3.c2), and a random rate matrix for BayesKDEy in three steps of increasing cost (E3.c3–E3.c5, a new hypothesis based on the Bayesian bootstrap of $V$). Time and memory measurements, the generic sensitivity analyses and the other ablations of the former documents are set aside in [[optional_extensions]] §8.

The full description is in [ext3_posterior_propagation.md](ext3_posterior_propagation.md). `BayCAPKDEy` is implemented (population pushforward) and not evaluated; the realised pushforward and SQS-B are not implemented.

## 4. [[ext4_uncertainty_budget|The uncertainty budget of CAP under PPS]]

The theoretical frame of the extended paper. The error $\hat A-A(h,U)$ of every quantification-based CAP estimate splits exactly into a prevalence part, a rate-matrix part, and the finite-sample fluctuation of the realised table of $U$. The finite-$U$ part and the target-sampling part of the prevalence error shrink as $\lvert U\rvert$ grows; the error of $C$ estimated on $V$ and the part of the prevalence error caused by fitting the quantifier on $V$ do not. Each CAP method represents only some of these terms: SQS-H and its bootstrap variant only the shrinking ones, BayCAP only the fixed ones.

A new hypothesis follows: a method's interval is wide only along the parts it simulates, so its coverage is governed by the ratio $\eta$ of missing to represented variance, approximately $2\Phi(1.96/\sqrt{1+\eta})-1$. Because the parts scale differently with $\lvert U\rvert$ and $\lvert V\rvert$, each method is calibrated only in one regime of the design ratio $r=\lvert U\rvert\sum_j(\pi^Q_j)^2/\lvert V_j\rvert$, and the paper's ranking at $\lvert U\rvert=1000$ is one point of a regime map. In a binary toy simulation, SQS-H loses coverage as $r$ grows, BayCAP as $r$ falls, and methods that represent all terms (the realised-table BayCAP of Extension 2, and SQS-H with resampling of both $U$ and $V$, a new candidate) stay near nominal. Extensions 2 and 3 are presented as completing missing terms of the budget.

The study varies $\lvert U\rvert$ and, at fixed allocation, a stratified subsample of $\lvert V\rvert$, with outer seeds for $V$, an oracle ladder (true $\pi^Q$, rates from the labelled pool, population scoring) as reference, and both paper and proposed methods. The validation allocation $\rho$ stays in Extension 6; PPS violations are set aside in [[optional_extensions]] §2. It absorbs proposal A of [[extensions_to_evaluate]].

The full formulation, hypotheses, and design are described in [ext4_uncertainty_budget.md](ext4_uncertainty_budget.md). It is not implemented; the toy simulation is `scripts/toys/ext4_budget.py`.

## 5. [[ext5_baycap_priors_tempering|Informative priors and posterior tempering for BayCAP]]

The paper names two unexplored features of BayCAP: informative priors and posterior tempering. This extension studies both, using the estimands and the realised-table predictive of Extension 2. The temperature of BayesKDEy (Extension 3) also belongs here, with no experiment designed yet.

Tempering is studied first as a diagnostic: a sweep over the temperature $\tau$, evaluated with target labels, gives the oracle-best $\tau$, an upper bound on what tempering can achieve, which is compared with the realised-table predictive. Selecting $\tau$ without target labels is run only if tempering clearly beats the predictive.

Priors are studied on both parameters. On $\pi^Q$, the paper's bags are drawn uniformly from the simplex, so its uniform prior is already the correctly specified one, and the prior's weight is about $c/(c+|U|/\tau)$, negligible at $|U|=1000$; a minimal check (one non-uniform bag-generating distribution, uniform, well-specified and confidently wrong priors, $|U|\in\{100,1000\}$) precedes the full study. On $C$, a new hypothesis: the uniform prior on each column is informative, since its weight of $K$ imaginary validation instances per class grows with the number of classes; a prior whose total weight does not grow with $K$ (Perks' $\operatorname{Dir}(1/K)$) or a diagonal-aware prior should remove BayCAP's bias on large-$K$ datasets. Tempering with $\tau>1$ increases the weight of the prior, and so this bias.

The full formulation, hypotheses, and design are described in [ext5_baycap_priors_tempering.md](ext5_baycap_priors_tempering.md). `ACC_BayesCAP` accepts a prior on $\pi^Q$ and the `binform` experiment covers one prior configuration; tempered BayCAP and a configurable prior on $C$ are not implemented. It revives two studies previously set aside in [[optional_extensions]] (full informative-prior study, label-free temperature selection).

## 6. SQS-S design: geometry, kernels and validation-budget allocation

Merges the former Extensions 4 and 6 (2026-10-27), which both study how SQS-S uses the validation set: the first through the distance and neighbourhood that select validation bags, the second through the split of $V$ between the quantifier and the bags. Each part keeps its own primary experiment, so that the effects remain identifiable; a joint selection of geometry, neighbourhood rule and allocation $\rho$ can follow.

### 6.1 Geometric analysis and kernel variants

SQS-S selects validation bags by comparing estimated prevalences through a distance on the simplex. One could systematically study how the distance and neighbourhood definition affect bias and variance:

- compare $L_1$, Hellinger, Jensen--Shannon, and Aitchison distances;
- measure the correspondence between proximity in predicted and true prevalence space;
- study sensitivity to $\gamma$;
- replace hard neighbour selection with kernel weights;
- select bandwidth adaptively according to local density.

This interpretation connects SQS-S to local regression or approximate Bayesian computation on the simplex. The contribution becomes substantial if it yields an adaptive rule, rather than only a comparison of distance functions.

### 6.2 Validation-budget allocation

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

This is a validation-budget allocation study, not part of the uncertainty budget of Extension 4, which subsamples $|V|$ at fixed $\rho$. A secondary factorial analysis may later test interactions between $\rho$, $|V|$ and $|U|$, but the primary experiment should vary only $\rho$ so that the effects remain identifiable.

## 7. Grid-search optimization of classifiers and quantifiers

The current experiments mostly use default classifier hyperparameters and fixed configurations for the classifiers embedded in the quantifiers. A proposed study, not currently implemented, should determine whether the conclusions about CAP methods remain stable when these predictive components are optimized by grid search.

The optimization must distinguish three objects with different roles:

- the target classifier $h$, whose performance on $U$ is being estimated;
- the surrogate classifier $s$ used by an aggregative quantifier to produce posterior representations;
- the quantifier $q$ itself, including method-specific hyperparameters such as regularization, bandwidth, kernel, or other aggregation parameters.

These objects should not necessarily use the same selection objective. The target classifier $h$ can be selected by an out-of-sample classification score, possibly including a calibration criterion when its posterior probabilities are consumed downstream. By contrast, $s$ and $q$ should be selected primarily by quantification error on source-generated validation bags spanning multiple prevalences, rather than only by instance-level classification accuracy. Bayesian quantifiers may additionally require a proper score or coverage-based criterion for selecting posterior-shape parameters.

The experimental protocol must remain nested and label-safe. Hyperparameters for $h$ should be selected only within its labelled training data. Parameters for $s$ and $q$ should be selected through inner splits or cross-fitting within the labelled data allocated to quantifier training. Target labels and the bags used for final CAP evaluation must never influence model selection. For SQS-S, the search must also respect the separation between $V_q$ and $V_b$ introduced in Extension 6 (§6.2), so that tuning does not consume the bag-sampling subset.

A factorial ablation should compare:

1. default $h$ and default $(s,q)$;
2. optimized $h$ and default $(s,q)$;
3. default $h$ and optimized $(s,q)$;
4. optimized $h$ and optimized $(s,q)$.

The parameter grids, selection metrics, folds, random seeds, and computational budgets should be fixed in advance and shared across CAP methods whenever their interfaces permit it. Results should report both predictive quality and search cost. Tuning $h$ changes the classifier and therefore the CAP problem itself, whereas tuning $(s,q)$ changes the estimator of target prevalence; these effects must be reported separately rather than summarized as a single gain from "optimization."

## Extensions under evaluation

Further candidate extensions are collected in
[[extensions_to_evaluate]] (the variance decomposition of proposal A has been promoted into Extension 4):

- a transductive BayCAP linked to CBPE;
- an ABC reinterpretation of SQS-S with regression adjustment;
- conformal calibration under simulated label shift.

These are hypotheses that still need evaluation (feasibility, novelty, and preliminary evidence). They are not ready to be implemented and should not be treated as part of the proposals above until they are promoted.

## Optional extensions

Additional proposals for extending the paper [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf) can be found in [[optional_extensions]]. These extensions were considered not worthy of being included in the final version of the extended paper and set aside for the moment; the former Extension 5 (multiclass scalability and new domains) is there as §7 since 2026-10-27.

## Recommended combinations

### Incremental and manageable extension

Evaluate multiple metrics, use the uncertainty budget of Extension 4 as the frame and its $(|U|,|V|)$ regime map as the main experiment, add the study of BayCAP (realised-table predictive, Extension 2) with the tempering diagnostic of Extension 5, and verify the main findings with optimized classifiers and quantifiers. This combines Extensions 1, 2, 4, 5 and 7 and is the lowest-risk option, although it remains mainly an empirical contribution unless the results support a new calibration or selection rule.

### SQS-S design extension

Carry out both parts of Extension 6: jointly selecting the simplex geometry, neighbourhood rule, and allocation $\rho$ would turn two sensitivity analyses into a coherent study of how SQS-S uses limited validation data. Extension 4 can then test whether the selected design remains stable as $|U|$ changes, while Extension 7 determines whether it depends on untuned surrogate models.

### Posterior-propagation extension

Add the methods of Extension 3 (BayCAP-KDEy in both pushforwards, SQS-B) and compare them with RQBS and BayCAP on the protocol of `main` and on at least three contingency-table metrics (Extension 1), then place them in the regime map of Extension 4. Because each method differs from an existing one in one component (the prevalence posterior, the handling of $C$, the source of the prevalences of the validation bags), the comparison isolates which source of uncertainty matters under PPS, which is the setting in which these methods could support a substantive claim rather than a component substitution. Extension 7 should be applied to all competing quantifiers under the same search budget, retaining untuned variants as ablations.

### More ambitious extension

Combine the SQS-S design study (Extension 6) with optimized surrogate models and the posterior-propagation methods of Extension 3, then evaluate the resulting methods across multiple metrics, target sizes, class counts, and domains. The scientific message would become: which representation and propagation of uncertainty remains reliable as the target sample and problem scale change under strict PPS, and are the conclusions robust to model selection?
