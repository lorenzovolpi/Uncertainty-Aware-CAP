This note includes all possible extensions to the paper [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf) that were discarded or temporarely put aside from [[extensions]].


## 1. Systematic interval calibration

The current evaluation mainly uses nominal 95% intervals. To determine whether
the distributions produced by the methods represent uncertainty accurately,
one could evaluate several nominal levels, such as 50%, 80%, 90%, 95%, and 99%.

The analysis would include:

- reliability plots comparing nominal and empirical coverage;
- coverage error, interval amplitude, Winkler score, and CRPS;
- coverage conditioned on shift magnitude, $|U|$, number of classes, and
  quantifier quality;
- post-hoc interval calibration, using temperature for BayCAP or corrections
  learned from shifts simulated with the validation set.

This contribution would provide a more rigorous characterization of uncertainty
quality. Without a new calibration method, however, it would remain primarily
an experimental extension.



## 2. Robustness beyond exact PPS

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

The marginal stress test previously listed among the secondary analyses of the uncertainty budget (now Extension 4; error caused by model misspecification or violations of PPS) is set aside here; the [[ext4_uncertainty_budget|uncertainty budget]] assumes strict PPS.

## 3. [[aitchison_sqs|Aitchison-SQS: propagating quantifier uncertainty]]

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
[aitchison_sqs.md](optional-directions/aitchison_sqs.md).

## 4. Full informative-prior study for BayCAP

Revived in [[ext5_baycap_priors_tempering|Extension 5]] (then numbered 10, 2026-10-15), as the full grid of experiment E5.4; set aside again on 2026-11-10, when Extension 5 was restructured (§12): only the minimal check, now E5.1, stays in the extension.

## 5. Label-free temperature selection for BayCAP

Revived in [[ext5_baycap_priors_tempering|Extension 5]] (then numbered 10, 2026-10-15), as experiment E5.5, then run only if oracle-best tempering clearly beat the realised-table predictive (hypothesis H5 refuted); set aside again on 2026-11-10, when Extension 5 was restructured (§12): label-free selection of $\tau$ is now the candidate E5.c2, which first tries a leave-one-dataset-out choice of $\tau$ on the results of E5.2.

## 6. A pass/fail gate for the realised-table predictive of BayCAP

Set aside from [[ext2_baycap|Extension 2]] on 2026-10-21: E2.1 and E2.3 are treated as analyses, not as a gate. The provisional decision rule was:

- *Pass* if, in E2.1 and confirmed by the exact run (now E2.3), at least one pair in five under-covers (coverage below $0.90$) and the predictive brings it within $[0.93,0.97]$ with a lower Winkler score.
- *Fail* if $\hat\sigma_g/\hat\sigma_p<0.3$ on almost every pair (the coverage then changes by about one percentage point or less), or if BayCAP over-covers and the predictive worsens the Winkler score everywhere.

All pairs count, including those whose interval is displaced (coverage near $0$ for both pushforwards), which no pushforward can repair (decision of 2026-10-20).

## 7. Multiclass scalability and new domains

Set aside from [[extensions]] on 2026-10-27, where it was Extension 5. The text below is the former summary.

The evaluation could be extended to text and image data, pretrained models, and problems with many classes. In addition to predictive quality, the analysis should measure:

- time and memory as functions of $K$, $|V|$, $|U|$, and the number of samples;
- the cost of BayCAP and NUTS sampling;
- behaviour with rare or absent classes;
- stability when the confusion matrix is ill-conditioned.

New domains would make the empirical evidence more convincing, but should not constitute the central contribution on their own.

## 8. Material set aside from SQS-B and BayCAP-KDEy

On 2026-10-27 the former Extensions 8 (SQS-B, `docs/sqs_b.md`) and 9 (BayCAP-KDEy, `docs/baycap_kdey.md`) were merged into [[ext3_posterior_propagation|Extension 3]], which keeps the two methods, their comparison on the protocol of `main`, and the candidate experiments decided in the discussion. The full former documents are in commit `391220c`. The rest is set aside here:

- **Time and memory.** Training and inference time of the posterior-propagation methods, in particular of the NUTS runs of BayesKDEy; optional, to be added back if time allows (user's decision of 2026-10-27).
- **Interpretation of SQS-B.** Its interval combines a Bayesian posterior over $\pi^Q$ with empirical validation sampling, so it is a "posterior-propagated empirical interval" rather than a Bayesian credible interval; when the quantifier and the bags both use $V$, the dependence this creates is not modelled, and data splitting or cross-fitting were proposed as sensitivity analyses.
- **Ablations of the former SQS-B document.** SQS-B with only the posterior mean, repeated through validation sampling (separates a better point estimate from the propagation of uncertainty); several bags per posterior draw (separates prevalence and bag randomness); an Oracle-SQS with the true $\pi^Q$, as a reference.
- **Ablations of the former BayCAP-KDEy document.** An "Oracle-C" reference with the true prevalence and a fixed $\hat C$, isolating the propagation from the posterior; a full factorial of posterior (BayesKDEy, BayesianCC) by bridge (fixed $C$, validation bags, random $C$), of which Extension 3 keeps the cells it needs as E3.1 and candidates.
- **Sensitivity analyses of both documents.** Informative and misspecified priors on $\pi^Q$; a temperature grid $\tau\in\{0.5,1,2,5\}$ (tempering now belongs to [[ext5_baycap_priors_tempering|Extension 5]]); the shrinkage grid of KDE-Bayes; split versus reused validation data; controlled violations of PPS (§2); rare or absent target classes.
- **Further measures.** Nominal levels $50\%$, $80\%$, $90\%$, $95\%$, $99\%$, reliability plots and CRPS (§1); calibration and sharpness of the prevalence posterior itself; results conditioned on shift magnitude, target size, number of classes and quantifier informativeness.
- **Relation to Aitchison-SQS.** Aitchison-SQS (§3) constructs a distribution around a non-Bayesian point estimate, while SQS-B and BayCAP-KDEy take the posterior from the quantifier; Aitchison-SQS remains applicable to non-Bayesian quantifiers.
- **Fallback framing.** If BayCAP-KDEy does not beat the other methods, it could still be presented as an interpretable baseline with closed-form moments for accuracy.

## 9. Material set aside from Extension 4 (paper plan)

On 2026-10-29 [[ext4_uncertainty_budget|Extension 4]] was restructured into the experiment-driven form of Extensions 2 and 3, which keeps no paper material for now. Its contributions, paper outline and figure plan are kept here as they were (section and hypothesis numbers refer to the earlier version, commit `dbb04dc`); the hypotheses H1–H4 were then in the experiments E4.1a and E4.2a of Extension 4, set aside on 2026-11-09 with the rest of the variance budget (§10 below), and the sections cited as §1–§3 are its §T.1–§T.8 of that version (§10).

### Contributions

1. **An exact error split and a budget** for CAP under PPS, which makes explicit that the realised table is the estimand and identifies which uncertainty terms shrink with $\lvert U\rvert$ and which do not.
2. **A unifying classification** of existing and proposed uncertainty-aware CAP methods by the parts they represent (§3.4), and a coverage rule in terms of the ratio $\eta$ of missing to represented variance (§3.5), which explains why the paper's methods are calibrated in different regimes.
3. **A regime map**, empirical, of coverage as a function of $(\lvert U\rvert,\lvert V\rvert)$, showing whether the paper's ranking at $\lvert U\rvert=1000$ depends on the design ratio $r$.
4. **Completed methods**: evidence that methods representing all terms (realised-table BayCAP from Extension 2, and SQS-H with $U$ and $V$ resampling) are calibrated across the map.

### Hypotheses

Primary:

- **H1 (SQS-type intervals degrade with $n$).** At fixed $V$, the coverage of SQS-H and RQBS decreases as $\lvert U\rvert$ grows, faster when $\lvert V_j\rvert$ is small. *Confirmed if* in E4.2a the coverage of RQBS decreases monotonically with $\lvert U\rvert$ on most (dataset, classifier) pairs and the decrease is larger at the smaller $\lvert V\rvert$ level. *Refuted if* coverage is flat in $\lvert U\rvert$; then $e_2$ and $e_1^{V}$ are negligible on real data and the budget adds little to Extension 2.
- **H2 (BayCAP under-covers at small $n$ and large $V$).** The coverage of BayCAP with the population pushforward increases with $\lvert U\rvert$ at fixed $V$ and is below nominal where its $\eta=\sigma_3^2/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ is large (§3.6). *Confirmed if* in E4.2a coverage increases with $\lvert U\rvert$ at fixed $V$ and decreases with $\lvert V\rvert$ at fixed $\lvert U\rvert$. *Refuted if* BayCAP over-covers at every $r$; then the fixed-count likelihood (source (i) of [[ext2_baycap]] §3) or prior effects dominate, and the budget does not explain BayCAP.
- **H3 (the budget accounts for the error).** In E4.1a, $\operatorname{RMSE}(e_3)$ scales as $n^{-1/2}$, $\operatorname{RMSE}(e_2)$ is constant in $n$ and scales as $\lvert V\rvert^{-1/2}$, and $e_1$ is the sum of a part shrinking with $n$ and a part fixed given $V$. *Confirmed if* the fitted slopes on log–log axes are within $\pm0.15$ of the predicted ones and, for every method, the coverage plotted against its $\eta$ (computed from the variances of the parts estimated in E4.1a) lies within $\pm0.05$ of $2\Phi(1.96/\sqrt{1+\eta})-1$ on most (dataset, classifier) pairs. *Refuted if* coverage does not follow $\eta$; then non-normality, the within-group correlations of §3.2 or quantifier bias $b_q$ matter, and the budget must be refined.

Secondary:

- **H4 (completing the budget restores coverage).** Methods that represent all terms are within $\pm0.03$ of nominal coverage across the grid, while each partial method fails where its missing term dominates. *Confirmed if* in E4.2a the realised-table BayCAP and SQS-H with $U$ and $V$ resampling meet this tolerance at every $(\lvert U\rvert,\lvert V\rvert)$ level. *Refuted if* they fail at large $n$ as well; then quantifier bias $b_q$ is the dominant fixed term and no variance-based correction suffices.

### Paper outline

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §3 (problem), new subsection "Target estimand and uncertainty budget" | §1–§3.3 of this note | The CAP error splits into prevalence, rate-matrix and finite-$U$ parts with different scaling. | derivation; F0 (toy); E4.1a |
| §4–§5 (methods), one table and the coverage rule | §3.4–§3.5 | Each method represents only some parts; its coverage follows from the ratio $\eta$ of missing to represented variance. | tables of §3.4–§3.5; E4.1a (H3) |
| §5, new subsections from Extensions 2 and 3 | completed methods | Adding the missing term restores calibration. | E4.2a (H4); [[ext5_baycap_priors_tempering]] E5.2 |
| §6 (experiments), new subsection "Target and validation size" | E4.1a, E4.2a | The paper's ranking at $\lvert U\rvert=1000$ is regime-dependent. | F1, F2, T1 |
| §7 (conclusions) | one paragraph | Practical guidance: which method to use given $\lvert U\rvert$ and $\lvert V\rvert$. | F2 |

Wording if results are mixed: if H1 is refuted, the budget becomes a short section explaining the estimand gap, and Extension 2 carries the main claim; if H3 is refuted but H1 and H2 hold, the regime map is presented empirically without the closed-form ratio.

#### Figures and tables

| Item | Question | Content | What it must show |
|---|---|---|---|
| F0 (optional) | toy | coverage vs $r$ for each method of Appendix A | the predicted shape of F2 |
| F1 | E4.1a | RMSE of $e_1$, $e_2$, $e_3$ vs $\lvert U\rvert$ (log–log), one panel per $\lvert V\rvert$ level, one line per dataset (LR) | the scaling of H3 |
| F2 | E4.2a | (a) coverage vs $r$ (log), one panel per method, one point per (dataset, classifier, $\lvert U\rvert$, $\lvert V\rvert$), dashed line at $0.95$; (b) coverage vs each method's $\eta$ (log), all methods in one panel, with the curve $2\Phi(1.96/\sqrt{1+\eta})-1$ | (a) the regime map, H1, H2, H4; (b) the coverage rule, H3 |
| T1 | E4.2a | mean Winkler rank per method, one column block per $\lvert U\rvert$ at full $V$; oracle references as unranked columns (`tables/rank.py`) | whether the ranking changes with $\lvert U\rvert$ |
| T2 (appendix) | E4.1a | per-dataset bias and standard deviation of $\hat\pi$ (L1) and $\hat A$ vs $\lvert U\rvert$ | quantifier bias $b_q$, needed to interpret H4 |

Macro-F1 and macro-K versions of F2 go to the appendix.

## 10. Material set aside from Extension 4 (variance budget, coverage, protocol)

On 2026-11-09 the user narrowed [[ext4_uncertainty_budget|Extension 4]] to the error split itself, its behaviour across $\lvert U\rvert$ and $\lvert V\rvert$, and oracle substitutions of its components; the population recalls are taken equal to the recalls on the test pool, so the pool term is no longer a separate part. Everything else is kept here as it was, verbatim: the first-order variance model of the parts and everything built on it (the coverage rule, the ratios $\eta$ and $r$, the regime map, the width floors, the label-free plug-ins, self-diagnosis, the break-even bag size, budget-completed and guaranteed intervals), the parts of BayCAP with joint estimation and the bound on $e_1$, the pool term and the protocol without replacement (cause 4 of [[ext2_baycap|Extension 2]]: candidates E4.c1 and E4.c2), the earlier implementation plan and the toy simulation of the budget (`scripts/toys/ext4_budget.py`, still in the repository). Section, experiment and hypothesis numbers (§T.n, E4.1a–E4.3b, E4.c1, E4.c2, H1–H4) refer to the version of the extension at commit `f19553a`; §T.1 and §T.2 of that version stay in the extension, and the old E4.1a is replaced there by E4.2 and E4.3 (mapping in the extension). The bound on $e_1$ (§T.11 below) was brought back into the extension later on 2026-11-09, as its §T.6, and the old E4.1c as its candidate E4.c12.

#### Claim

A CAP interval is scored against the accuracy of one target bag, and the gap between the estimate and that accuracy has several sources: the error of the estimated prevalence, the error of the rates of $h$ measured on the validation set $V$, and the luck of the bag. Under the paper's protocol, where every bag is re-drawn with replacement from one finite test pool, there is one more: the luck of the pool, which all the bags of a dataset share. The sources scale differently with the size $\lvert U\rvert$ of the bag, the size of $V$ and the size of the pool, and each CAP method's interval is wide only along the sources it simulates. This extension writes the error as an exact sum of these parts (the *uncertainty budget*), uses it to predict in which regime of $(\lvert U\rvert,\lvert V\rvert)$ each method is calibrated, and measures it on real data; the paper's comparison at $\lvert U\rvert=1000$ is one point of this regime map. The budget and the predictions are a new hypothesis, supported so far only by a toy simulation. A further question, opened by cause 4 of [[ext2_baycap|Extension 2]], is what would change if the bags were drawn *without* replacement: each bag would then be a fresh sample of the target distribution, the pool term would merge into the luck of the bag, and the methods and baselines that miss the pool term under the paper's protocol could become calibrated (§T.10, candidate experiments).

This extension is the common frame for the method extensions: the realised-table predictive of Extension 2 adds the luck of the bag to BayCAP, and the methods of [[ext3_posterior_propagation|Extension 3]] propagate a Bayesian prevalence posterior. It absorbs proposal A of [[extensions_to_evaluate]] (variance decomposition).

#### Status

| Item | Status |
|---|---|
| SQS-H, SQS-S, BayCAP at $\lvert U\rvert=1000$, scored against the realised accuracy of each bag | [paper] |
| `RQBS` (SQS-H with a bootstrap of $\hat\pi$, the variant run by `main`), `PrediQuant` (SQS-S), `BayesCAP` (BayCAP, built from `methods/baycap.py:BayCAP`) | [implemented] |
| `BayesCAP-real` (BayCAP with the realised-table predictive, [[ext2_baycap]] E2.3) | [implemented], evaluated in `main` |
| `BayCAPKDEy` with a fixed $\hat C$ and `BayCAPMAPLS` (the methods of E3.1 of [[ext3_posterior_propagation]]; one class, `BayCAPFixed`, until WP-13) | [implemented], evaluated by E3.1 |
| SQS-B and the realised pushforward of BayCAP-KDEy ([[ext3_posterior_propagation]]) | [proposal] |
| Error split, budget with the pool term, coverage rule and ratios $\eta$, $r$, oracle ladder, SQS-H with $U$ and $V$ resampling | [new hypothesis] |
| The protocol without replacement and its budget (§T.10) | [new hypothesis] |
| Bound on $e_1$, width floors, variances and prior bias of BayCAP with joint estimation of $(\pi^Q,C)$, label-free plug-ins (self-diagnosis, break-even $n^*$, completed intervals), finite-sample guaranteed intervals (§T.11–§T.15) | [new hypothesis], no experiment yet |
| Toy simulation of the budget (`scripts/toys/ext4_budget.py`, Appendix A) | [implemented], run |
| E4.1 (is the budget right?), E4.2 (does it explain coverage?), E4.3 (can it fix the intervals?), with their parts; merged on 2026-11-04 from the former E4.1–E4.3 and candidates E4.c3–E4.c10 (§T.0) | [proposal], not implemented |
| Candidate experiments E4.c1, E4.c2 (cause 4, question D of §T.0) | [proposal], not scheduled |

The paper material of earlier versions (contributions, paper outline, figure plan) is in [[optional_extensions]] §9.

### Experiments

On 2026-11-04 the experiments and the candidate experiments of this extension were merged into experiments that answer the questions of §T.0 (questions A–C here; question D, the protocol, was first experiment E4.4 and is now studied by the candidates E4.c1 and E4.c2); each experiment is split into parts, named with a letter. Old numbers map to new ones as follows (E4.1–E4.3 were reused with a new meaning):

| Old | New | Part |
|---|---|---|
| E4.1 budget validation | E4.1a | the parts across $\lvert U\rvert$ and $\lvert V\rvert$ |
| E4.c9 | E4.1b | the parts of BayCAP with joint estimation of $(\pi^Q,C)$ |
| E4.c3 | E4.1c | the bound on the prevalence part |
| E4.2 regime map | E4.2a | regime map |
| E4.3 robustness | E4.2b | robustness of the map |
| E4.c4 | E4.2c | class-level width floor |
| E4.c5 | E4.2d | instance-level floor and the baselines |
| E4.c6 | E4.2e | self-diagnosis of coverage |
| E4.c7 | E4.2f | break-even bag size |
| E4.c8 | E4.3a | budget-completed intervals |
| E4.c10 | E4.3b | intervals with a finite-sample guarantee |
| E4.c1, E4.c2 | E4.c1, E4.c2 (E4.4a, E4.4b until demoted) | the pool gap; with or without replacement |

**Settings of the grid study** (E4.1a, E4.2a, E4.2b; new study `target_size`). APP bags drawn with replacement from the test pool, as in `main`; all intervals scored against the realised table of each bag; measures as in the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$) plus the error split of §T.2; metrics of [[ext1_metrics]]. Coverage is reported with a standard error clustered by $V$ seed, since bags sharing a $V$ are not independent; bags also share the pool, which is fixed per dataset, so the pool term $e_P$ of §T.3 is a constant offset within a dataset (§T.7). Dataset subset shared with [[ext2_baycap]]: binary and multiclass datasets covering small and large $\lvert V_j\rvert$, well- and ill-conditioned $C$, and one large-$K$ dataset; the pool sizes $P_j$ are recorded for every dataset. Classifiers: LR and RF (all five in the pilot). Grids:

- $\lvert U\rvert\in\{100,300,1000,3000,10000\}$, log-spaced, including the paper's $1000$; bags larger than the pool repeat instances, which is consistent with sampling with replacement; at these sizes the pool term, which does not shrink with $\lvert U\rvert$, is part of the fixed error (§T.8);
- $\lvert V\rvert$ levels: the full $V$ and a stratified subsample of $1/4$ (and $1/16$ on the largest datasets), nested ($V_{1/16}\subset V_{1/4}\subset V$) and seeded; the split $L/V$, the classifier $h$, and the allocation $\rho$ used inside SQS-S are unchanged, so Extension 6 stays separate;
- $S_V=5$ outer seeds for each subsampled $V$ level (for the full $V$, the seed changes the $L/V$ split; $h$ is retrained per seed).

**Settings of the analyses on stored results** (E4.1b on real data, E4.1c, E4.2c, E4.2e, E4.2f, E4.3a). They work on the results of `main` ($\lvert U\rvert=1000$, vanilla accuracy, bags drawn with replacement), score every interval against the realised table of each bag (`true_cts`), and read the predictions and labels of $V$ and of the pool from the pretrain files; $\hat\pi$ and the method's recalls are read from the stored estimated tables (`estim_cts`) where a method stores them. Quantities that use the true prevalence or the labels of the pool are references or analyses, not methods. They share one set of label-free plug-ins (§T.14), computed once per (dataset, classifier) pair and bag, and the pool gap $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$ of each bag (§T.3), computed from the recalls of $h$ on $V$ and on the pool as a reference, which the analyses use to read the gaps that the plug-ins cannot explain.

**Order.** First the analyses on stored results: E4.1c, E4.2f, E4.2e, E4.3a, E4.2c; then the toy of E4.1b; then the re-run that stores the per-bag quantities needed by E4.2d and E4.3b; last the grid study (pilot, then E4.1a, E4.2a, E4.2b). If the candidates E4.c1 and E4.c2 are promoted, E4.c1 comes first and the runs of E4.c2 can also serve E4.2d and E4.3b.

#### E4.1: Is the budget right?

**Why.** The split of §T.2 is exact, but the sizes, scalings and correlations of its parts come from first-order formulas under assumptions (§T.4, §T.5, §T.13). Before the budget can explain coverage (E4.2) or correct intervals (E4.3), its parts must be measured on real data and compared with the predictions: for the SQS family across sizes (E4.1a), for BayCAP with joint estimation of $(\pi^Q,C)$, whose parts follow other formulas (E4.1b), and through the exact bound that links the prevalence part to the quantifier's error (E4.1c).

##### E4.1a: the parts across $\lvert U\rvert$ and $\lvert V\rvert$

**Why.** The budget (§T.2–§T.5) predicts how each part of the error scales: the luck of the bag and the quantifier's noise on the bag shrink as $\lvert U\rvert^{-1/2}$, the errors fixed by $V$ do not shrink with $\lvert U\rvert$ but shrink as $\lvert V\rvert^{-1/2}$, and the pool term is fixed by the pool. Before the budget can explain the coverage of the methods (E4.2a), its parts must be measured on real data and their scaling checked; their variances also give each method's ratio $\eta$ (§T.7), which E4.2a needs.

- **Design.**
  - **Inputs:** the subset datasets and their test pools; quantifier PACC as in the paper; methods RQBS and SQS-H with a point $\hat\pi$, which are cheap.
  - **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed; three fixed prevalences per dataset: uniform $\mathbf 1/K$, one skewed vector (a seeded draw from $\operatorname{Dir}(\mathbf 1)$), one near-vertex vector ($0.9$ on the rarest validation class, the rest uniform); $R=100$ bags per (prevalence, $\lvert U\rvert$, $\lvert V\rvert$, seed), i.e. $500$ per cell. Held fixed: the pool of each dataset, the quantifier and the methods.
  - **Measures:** per bag, $\hat\pi$, $\hat A$, $e_1^U$, $e_1^V$, $e_2+e_P$ and $e_3^b$ (§T.3; on real data the population rates $d$ are unknown, so $e_2$ and $e_P$ are observed only through their sum $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$), interval bounds; per cell, the RMSE and mean of each part, the bias and standard deviation of $\hat\pi$ and $\hat A$, and their log–log slopes against $\lvert U\rvert$ and $\lvert V\rvert$. $\pi^\infty(V)$ is approximated, once per ($V$ seed, prevalence), by the estimate of the quantifier fitted on that $V$ on one very large bag (e.g. $10^5$ instances drawn with replacement from the pool at $\pi^Q$); then $e_1^V$ follows from its definition in §T.3 and $e_1^U=e_1-e_1^V$. Oracle ladder (references, never methods): (a) true $\pi^Q$: SQS bags at the true counts, removing $e_1$; (b) $d^{\text{pool}}$ from the labelled pool, removing $e_2$ and $e_P$ together; (c) scoring against $d^{\text{pool}\top}\pi^Q$, isolating $e_3^b$.
  - **Estimand:** the realised table of each bag for every interval; the population accuracy $d^{\text{pool}\top}\pi^Q$ only in oracle (c). The parts of the split use the true prevalence of the bag and the labels of the pool, so they are an analysis, not a method.
  - **Cost:** no NUTS; dominated by bag sampling, of order $m\cdot n$ per bag for $m$ draws of size $n$.
- **Representation.**
  - **RMSE of the parts against $\lvert U\rvert$** (log–log), one panel per $\lvert V\rvert$ level, one line per dataset (LR), for $e_1$, $e_2+e_P$ and $e_3^b$.
  - **Table, quantifier bias:** per dataset, the bias and standard deviation of $\hat\pi$ ($L_1$) and of $\hat A$ against $\lvert U\rvert$.
- **Decision rule (H3, the budget accounts for the error).** $\operatorname{RMSE}(e_3^b)$ should scale as $n^{-1/2}$; $\operatorname{RMSE}(e_2+e_P)$ should be constant in $n$, and its square should be close to $\operatorname{Var}(e_2)+\operatorname{Var}(e_P)$ of §T.4 (it shrinks with $\lvert V\rvert$ only down to the pool part); $e_1$ should be the sum of a part shrinking with $n$ and a part fixed given $V$. The prediction holds if the fitted slopes are within $\pm0.15$ of the predicted ones and, for every method, the coverage measured in E4.2a against its $\eta$ lies within $\pm0.05$ of $2\Phi(1.96/\sqrt{1+\eta})-1$ on most (dataset, classifier) pairs. If coverage does not follow $\eta$, non-normality, the correlations of §T.4 or the quantifier bias $b_q$ matter, and the budget must be refined. The mean of $e_1$ over $V$ seeds estimates $b_q$.
- **Analysis.** Not run.

##### E4.1b: the parts of BayCAP with joint estimation of $(\pi^Q,C)$

**Why.** §T.13 predicts, for the variants of the `BayCAP` class (joint posterior of $(\pi^Q,C)$, with $\pi^Q$ estimated from the predictions of $h$ through $C$), that in the binary case $\operatorname{Var}(e_1^U)\approx\kappa\sigma_{3b}^2$ and $\operatorname{Var}(e_1^V)\approx\kappa\operatorname{Var}(e_2)$, with $\kappa=\big((d_1-d_0)/(d_1+d_0-1)\big)^2$ and structural correlations of $e_1^U$ with $e_3^b$ and of $e_1^V$ with $e_2$, whatever the prior; and that the prior on $C$ biases the recalls, by $(1-Kd_j)/(\lvert V_j\rvert+K)$ with the uniform prior and by the general form of §T.13 with the empirical-Bayes prior. These drive the coverage of these variants in the budget, and the correlations decide whether the budget can treat their parts as independent. The variants of the `BayCAPFixed` class are not tested here: their $e_2$ behaves as in the SQS family and their $e_1$ depends on their quantifier, with no derived formula (§T.13).

- **Design.**
  - **Inputs:** a toy simulation, binary, with BayCAP computed exactly on a grid as in `scripts/toys/ext4_budget.py` (uniform prior); the stored results of `main` of the variants of the `BayCAP` class with their estimated tables: `BayesCAP` and `BayesCAP-real` (uniform prior), `BayesCAP-EB` and `BayesCAP-EB-real` (empirical-Bayes prior, whose fitted concentrations are needed for the general form of the bias), `BayesCAP-H-real`, `BayesCAP-Hf-real`, `BayesCAP-Hfn-real` (hierarchical priors); the validation counts $k_j$, $\lvert V_j\rvert$ and the pool labels from the pretrain files.
  - **Varied:** in the toy, $d$ (including $d_1+d_0$ close to $1$), $n$, $\lvert V_j\rvert$, with many draws of $V$ and of the bag per configuration; on real data, dataset, classifier and variant (prior on $C$; population or realised pushforward, which changes the represented parts but not the error parts of the point estimate).
  - **Measures:** in the toy, the simulated variances and covariances of $e_1^U$, $e_1^V$, $e_2$, $e_3^b$ against the formulas of §T.13, and the bias of the posterior mean recalls against $(1-Kd_j)/(\lvert V_j\rvert+K)$; on binary datasets and for every variant, the error parts per bag (an analysis with the true prevalence and the labels of the pool), their spread over the bags against $\hat\kappa\hat\sigma_{3b}^2$, and the sign of their correlation against the predicted one; for the uniform and empirical-Bayes priors, the posterior mean recalls (from the posterior draws, or approximately from the stored mean table as $\bar T_{jj}/\sum_i\bar T_{ij}$) against their closed form, $(k_j+1)/(\lvert V_j\rvert+K)$ or $(k_j+\alpha_{jj})/(\lvert V_j\rvert+\alpha_{\cdot j})$, whose difference measures the information of the target bag on $C$, and the gap $\bar d_j-d^{\text{pool}}_j$ against the predicted bias, grouped by $K$ (a reference, since it uses the labels of the pool); for the hierarchical priors, which have no closed form, the same gap, measured only.
  - **Estimand:** the realised table of each bag; the parts are an analysis that uses the true prevalence and the labels of the pool.
  - **Cost:** the toy runs in minutes; no runs on real data.
- **Representation.**
  - **Scatter, toy:** simulated against predicted variance and covariance of each part, one point per configuration, with the identity line.
  - **Table, real data:** per binary pair and variant, $\hat\kappa$, the predicted and measured spread of $e_1$ over the bags, and the correlation of $e_1$ with $e_3^b$; per $K$ and prior, the mean gap $\bar d_j-d^{\text{pool}}_j$ against the predicted bias.
- **Analysis.** Not run.

##### E4.1c: the bound on the prevalence part

**Why.** §T.11 bounds $\lvert e_1\rvert$ by $\frac12\lVert\tilde\pi-\pi^Q\rVert_1(\max_j\tilde d_j-\min_j\tilde d_j)$ for both families. The bound is exact, so it cannot fail; its tightness says whether the quantifier's AE, the measure reported by the quantification literature, times the recall spread of $h$ is a good proxy of the accuracy error that the prevalence error causes.

- **Design.**
  - **Inputs:** the stored results of `main` for the methods whose estimated tables give $\tilde\pi$ and $\tilde d$ (`RQBS`, `BayesCAP`, `BayesCAP-real`); the predictions and labels of $V$.
  - **Varied:** dataset, classifier, method; nothing else.
  - **Measures:** per pair, the recall spread $\max\tilde d-\min\tilde d$ from $V$; per bag and method, $\lvert e_1\rvert$ (§T.2, with the true prevalence: an analysis), the bound, their ratio (tightness), and the share of $\lvert\hat A-A\rvert$ that the bound alone accounts for; across pairs, the quantifier's AE against the accuracy error, with and without the factor of the recall spread (rank correlation).
  - **Estimand:** the realised table of each bag.
  - **Cost:** none.
- **Representation.**
  - **Scatter:** $\lvert e_1\rvert$ against the bound, per bag (log–log), one panel per method, with the identity line.
  - **Table:** per pair, the recall spread, the median tightness and the median share of $\lvert\hat A-A\rvert$ covered by the bound.
  - **Scatter:** accuracy error against AE and against $\frac K2\,\mathrm{AE}\,(\max\tilde d-\min\tilde d)$, one point per pair and method.
- **Analysis.** Not run.

#### E4.2: Does the budget explain coverage?

**Why.** If the budget is right, each interval is wide only along the parts it simulates, and its coverage follows the ratio $\eta$ of missing to represented variance (§T.7). This predicts the regime of $(\lvert U\rvert,\lvert V\rvert)$ in which each method is calibrated (E4.2a, E4.2b), floors below which no interval can be valid (E4.2c, E4.2d), and label-free predictions of coverage that a method can compute from its own outputs (E4.2e, E4.2f).

##### E4.2a: regime map

**Why.** The budget predicts that each method is calibrated only in one regime of $(\lvert U\rvert,\lvert V\rvert)$, because its interval represents some parts of the error and misses others whose share changes with the two sizes (§T.8). The paper compares the methods at $\lvert U\rvert=1000$ only; E4.2a measures the coverage of every method across the grid, to see whether the paper's ranking holds elsewhere and whether the methods that represent all parts but the pool term are calibrated across the map.

- **Design.**
  - **Inputs:** the subset datasets; APP prevalences ($100$ uniformly drawn prevalence vectors per cell, as in the paper, with fixed seeds shared by all methods and cells); the variances of the parts estimated by E4.1a for the same cells.
  - **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed; the method: SQS-H (point $\hat\pi$), RQBS, SQS-S (`PrediQuant`, after the fix of [[work_plan]] FIX-1), BayCAP (population pushforward), BayCAP with the realised-table predictive (same draws, no extra NUTS run), SQS-B, BayCAP-KDEy (both pushforwards, with the implementations of [[ext3_posterior_propagation]] E3.1), SQS-H with $U$ and $V$ resampling (§T.8); the oracle references of E4.1a.
  - **Measures:** per bag and method, the metric samples; per cell, coverage (clustered standard error), amplitude, Winkler score, AE; $r$ per cell (§T.7) and $\eta$ per method and cell, from the variances of E4.1a.
  - **Estimand:** the population table for BayCAP and BayCAP-KDEy with the population pushforward, the realised table for the realised pushforwards and for the SQS family; every interval scored against the realised table of each bag.
  - **Cost:** NUTS dominates: one BayCAP run per bag, shared by both pushforwards; BayesKDEy and SQS-B add their own posterior runs. The pilot (two datasets, LR, $20$ bags) estimates the cost; if needed, the BayCAP family runs on $S_V=3$ seeds.
- **Representation.**
  - **Scatter, regime map:** coverage against $r$ (log scale), one panel per method, one point per (dataset, classifier, $\lvert U\rvert$, $\lvert V\rvert$), a dashed line at $0.95$.
  - **Scatter, coverage rule:** coverage against each method's $\eta$ (log scale), all methods in one panel, with the curve $2\Phi(1.96/\sqrt{1+\eta})-1$.
  - **Table, ranking:** mean Winkler rank per method, one column block per $\lvert U\rvert$ at the full $V$, oracle references as unranked columns (`tables/rank.py`).
- **Decision rule (H1, H2, H4 of §T.8).** Three predictions of §T.8. **H1 (SQS-type intervals degrade with $n$):** at fixed $V$ the coverage of RQBS decreases monotonically with $\lvert U\rvert$ on most pairs, faster at the smaller $\lvert V\rvert$ level; if coverage is flat in $\lvert U\rvert$, $e_2$ and $e_1^V$ are negligible on real data. **H2 (BayCAP under-covers at small $n$ and large $V$):** the coverage of BayCAP with the population pushforward increases with $\lvert U\rvert$ at fixed $V$ and decreases with $\lvert V\rvert$ at fixed $\lvert U\rvert$; if it over-covers at every $r$, the fixed class counts of APP bags ([[ext2_baycap]] §T.6) or prior effects dominate. **H4 (completing the budget restores coverage):** the realised-table BayCAP and SQS-H with $U$ and $V$ resampling are within $\pm0.03$ of nominal at every $(\lvert U\rvert,\lvert V\rvert)$ level, while each partial method fails where its missing part dominates; if they fail at large $n$ as well, the quantifier bias $b_q$ or the pool term $e_P$, which no method represents, is the dominant fixed part, and the two are told apart by the pool gap of each bag (settings above; its full study is candidate E4.c1). The cell $\lvert U\rvert=1000$, full $V$, seed of `main` reproduces the paper's setting (comparability check with `main`).
- **Analysis.** Not run.

##### E4.2b: robustness of the map

**Why.** The regime map of E4.2a is measured on a few datasets, on uniformly drawn prevalences and on vanilla accuracy; the budget predicts how it should move with the prevalence, the number of classes and the metric, so these strata test it further.

- **Design.**
  - **Inputs:** the outputs of E4.1a and E4.2a; no new runs, except macro-F1 and macro-K computed from the stored tables.
  - **Varied:** prevalence type (uniform, skewed, near-vertex); binary against multiclass; metric.
  - **Measures:** coverage against $r$ per stratum.
  - **Estimand:** as in E4.2a.
  - **Cost:** analysis only.
- **Representation.** The two scatters of E4.2a, one version per stratum.
- **Decision rule.** Macro-K depends on $\pi^Q$ only weakly ([[ext1_metrics]]), so its $e_1$ should be small and its coverage should follow $e_2$, $e_P$ and $e_3$; near-vertex prevalences raise $\sum_j(\pi^Q_j)^2$ and therefore $r$ (and $\operatorname{Var}(e_P)$) at the same $\lvert U\rvert$.
- **Analysis.** Not run.

##### E4.2c: class-level width floor

**Why.** §T.12 shows that an interval narrower than about $2\cdot1.96\sqrt\rho\,\sigma_{3b}$ cannot reach $95\%$ conditional coverage of the realised accuracy for a method that sees the bag only through class-level quantities. How often each method's intervals fall below the plug-in floor tests whether the floor explains part of the under-coverage, notably of the population-pushforward BayCAP.

- **Design.**
  - **Inputs:** the stored intervals of `main`; $\hat\pi$ from the estimated tables or from the method's quantifier; the predictions and labels of $V$.
  - **Varied:** dataset, classifier, method; binary datasets only, since $\rho$ is derived only for $K=2$ (for $K>2$, the floor with $\rho=1$ is reported as an upper value).
  - **Measures:** per method and pair, the share of bags whose interval is narrower than the plug-in floor $2\cdot1.96\sqrt{\hat\rho}\,\hat\sigma_{3b}$ (§T.14), and the coverage of the bags below and above the floor; the same with the true $\pi^Q$ and $d^{\text{pool}}$ in place of the plug-ins, as a reference.
  - **Estimand:** the realised table of each bag; the floor refers to it.
  - **Cost:** none.
- **Representation.**
  - **Table:** per method and pair, the share of bags below the floor and the coverage below and above it.
  - **Scatter:** interval width against the plug-in floor, per bag, one panel per method, coloured by covered or not.
- **Analysis.** Not run.

##### E4.2d: instance-level floor and the baselines

**Why.** §T.12 predicts that methods using instance-level confidences can have valid intervals narrower than the class-level floor, since $\frac1{n^2}\sum_iq_i(1-q_i)\le\sigma_{3b}^2$. This would explain where confidence-based baselines can beat the paper's methods, above all at small $n$.

- **Design.**
  - **Inputs:** a re-run of `main` on the pairs of the subset, with the same seeds and bags, that stores per bag $\sum_i\hat q_i(1-\hat q_i)$ and $\sum_i\hat q_i$ computed with the calibrated posteriors of $h$, and the calibration map used; the stored results of the methods.
  - **Varied:** dataset, classifier, method (the baselines DoC, O-LEAP, L-CBPE, QuAcc and the paper's methods)
  - **Measures:** per bag, the class-level floor, the instance-level floor, and each method's width and coverage against both; per pair, the ratio of the two floors against the calibration of $h$ (ECE) and the spread of the confidences within classes.
  - **Estimand:** the realised table of each bag; the instance-level floor uses the calibrated confidences, which are available without labels, but their calibration is checked on $V$.
  - **Cost:** the re-run of the methods of `main` on the subset (shared with E4.3b, and with the runs of candidate E4.c2 if it is promoted).
- **Representation.**
  - **Scatter:** width against the two floors, per bag, one panel per method.
  - **Scatter:** ratio of the floors against ECE, one point per pair.
- **Analysis.** Not run.

##### E4.2e: self-diagnosis of coverage

**Why.** §T.14 gives each method a predicted coverage $2\Phi(1.96/\sqrt{1+\hat\eta})-1$, computed without target labels from its own width and the plug-ins of the parts it misses; it is an upper bound to first order if the within-source correlations are negligible. If it tracks the observed coverage, a practitioner can tell from the outputs alone when an interval is too narrow.

- **Design.**
  - **Inputs:** the stored intervals of `main`; the plug-ins of $\sigma_{3b}^2$ and $\sigma_2^2$, and of $\sigma_{1U}^2$, $\sigma_{1V}^2$ (delta method for ACC-type quantifiers, or a bootstrap of $V$ for PACC); the predictions and labels of $V$.
  - **Varied:** dataset, classifier, method; for the variants of the `BayCAP` class on binary datasets, with and without the correlations of §T.13.
  - **Measures:** per bag and method, $\sigma_{\text{in}}$ (half-width$/1.96$), $\hat\eta$ and the predicted coverage; per method and pair, the predicted coverage averaged over the bags against the observed coverage, the share of pairs where the prediction is above the observed coverage (as the upper bound requires), and the residual gap against the pool gap of each bag (settings above), which the diagnosis cannot see.
  - **Estimand:** the realised table of each bag.
  - **Cost:** none, except a bootstrap of $V$ if the delta-method plug-ins of $e_1$ are not available for PACC.
- **Representation.**
  - **Scatter, calibration of the diagnosis:** observed against predicted coverage, one point per pair, one panel per method, with the identity line.
  - **Scatter:** residual gap (predicted minus observed) against the pool gap of each bag.
- **Analysis.** Not run.

##### E4.2f: break-even bag size

**Why.** §T.14 defines $n^*$, the bag size at which the missing $e_2$ of SQS-type intervals equals the luck of the bag they represent, computable without labels. The rule "do not trust SQS-type intervals for $n\gg n^*$" is useful only if the coverage of the SQS family falls with $n/n^*$.

- **Design.**
  - **Inputs:** the stored results of `main`; $\tilde d$ and $\lvert V_j\rvert$ from the pretrain files; for the within-pair version, the grid of E4.2a.
  - **Varied:** dataset, classifier; method (`RQBS`, `PrediQuant`); $\lvert U\rvert$ only within E4.2a.
  - **Measures:** per pair, $n^*$ at the prevalences of the bags; coverage against $1000/n^*$ across pairs at $\lvert U\rvert=1000$, and against $n/n^*$ within pairs on the grid.
  - **Estimand:** the realised table of each bag.
  - **Cost:** none.
- **Representation.**
  - **Scatter:** coverage against $n/n^*$ (log scale), one point per pair (and per $\lvert U\rvert$ on the grid), with the curve $2\Phi(1.96/\sqrt{1+n/n^*})-1$ of the oracle-$\pi$ reference.
- **Analysis.** Not run.

#### E4.3: Can the budget fix the intervals?

**Why.** If the missing parts explain the under-coverage, adding them back should restore it. E4.3a adds the plug-ins of the missing parts to each interval; E4.3b builds intervals with a finite-sample guarantee, the conservative end of the scale. What remains after the corrections measures the parts that have no label-free plug-in, $e_P$ and $b_q$.

##### E4.3a: budget-completed intervals

**Why.** §T.14 proposes widening each interval by the plug-ins of the parts it misses, and moving the centre of BayCAP by its estimated prior bias $-\sum_j\hat\pi_j(\bar d_j-\tilde d_j)$. These cheap post-processings approximate the methods that represent all the parts (SQS-H with $U$ and $V$ resampling, the realised-table BayCAP); if they restore coverage, the plug-ins are adequate, and where they do not, the remaining gap measures $e_P$ and $b_q$.

- **Design.**
  - **Inputs:** the stored results of `main`; the plug-ins of E4.2e; the pool gap of each bag (settings above).
  - **Varied:** dataset, classifier; method and correction: `RQBS` and `RQBS` $+\hat\sigma_{1V}^2+\hat\sigma_2^2$; `BayesCAP` and `BayesCAP` $+\hat\sigma_{3b}^2$, with and without the bias shift; `BayesCAP-real` as the method that represents the parts by construction.
  - **Measures:** per method and pair, coverage, amplitude, Winkler score and AE of the original and of the completed intervals; the paired comparison with `BayesCAP-real` (paired bootstrap of the Winkler score, as in [[ext2_baycap]] E2.3); the residual under-coverage against the pool gap of each bag and against the pool size $P_j$.
  - **Estimand:** `BayesCAP` targets the population table, its completed version and the other methods the realised table; every interval is scored against the realised table of each bag.
  - **Cost:** none.
- **Representation.**
  - **Table:** per method and correction, coverage, amplitude and Winkler score, overall and per dataset, with the paired test against `BayesCAP-real`.
  - **Scatter:** coverage of the completed intervals against the pool gap, one point per pair.
- **Analysis.** Not run.

##### E4.3b: intervals with a finite-sample guarantee

**Why.** §T.15 builds, for binary ACC, an interval that covers the realised accuracy with probability at least $1-\alpha$ for every sample size, without normality and including the quantifier bias. Its width is the price of a guarantee, and comparing it with the methods' widths shows how much their coverage relies on the normal approximation.

- **Design.**
  - **Inputs:** per bag, the ACC quantities $\hat p$ (share of the bag predicted as $\omega_1$ by the surrogate) and $\hat S$ on $V$, which `main` does not store: a re-run on the binary pairs that stores them, or a recomputation from the seeded bags and the surrogate's predictions; the stored results of the methods; the runs of candidate E4.c2 for the protocol without replacement, only if it is promoted.
  - **Varied:** binary dataset, classifier; protocol: with replacement, as in `main`, with the pool terms added as a reference; without replacement, where the guarantee holds as derived, only if candidate E4.c2 is promoted; the split of $\alpha$ among the parts.
  - **Measures:** per pair and protocol, the coverage (expected at least $0.95$) and the width of the guaranteed interval; the ratio of its width to the widths of the methods, notably of SQS-H with $U$ and $V$ resampling.
  - **Estimand:** the realised table of each bag.
  - **Cost:** the re-run or recomputation of the ACC quantities; no NUTS.
- **Representation.**
  - **Table:** per pair and protocol, coverage and width of the guaranteed interval and of the methods.
  - **Scatter:** width ratio (guaranteed over method) against coverage of the method, one point per pair and method.
- **Analysis.** Not run.

### Candidate experiments

Ideas that came up in the discussions; they are not scheduled and are numbered E4.c<n>. One becomes an experiment of the main section only when the user decides so, and then it gets the next free number E4.<n> and a full specification. The two candidates below study cause 4 of [[ext2_baycap|Extension 2]] (the test pool shared by all bags) for every method and classifier. They formed experiment E4.4 ("How much is due to the protocol?", question D of §T.0) from 2026-11-04 until they were demoted again on the same day; they keep their original numbers, and the number E4.4 is not reused. New candidates are numbered from E4.c11.

- **E4.c1, the pool gap on stored results.** *Why:* Extension 2 found on binary datasets a recall error that the prior of BayCAP does not explain, and its toy traces it to the pool (§T.3; [[ext2_baycap]] §T.13, Appendix A.3); the explanation is a hypothesis, and the pool gap $\sum_j\pi^{(b)}_j(\hat d_j-d^{\text{pool}}_j)$ belongs to the (dataset, classifier) pair, not to a method, so it can be measured once and compared with the shift of every method. *What it would show:* per pair, the pool sizes $P_j$, the plain recalls of $h$ on $V$ and on the pool, the pool gap averaged over the bags and its predicted standard deviation $\sqrt{\sum_j\pi_j^2d_j(1-d_j)(1/\lvert V_j\rvert+1/P_j)}$ (§T.4); per method, the shift (mean over the bags of $\hat A-A$), its correlation and slope against the pool gap across pairs, and a reference coverage of the interval moved by the pool gap of each bag (a reference, since it uses the labels of the pool); for the BayCAP methods, $\overline{R-R_0}$ of [[ext2_baycap]] E2.2 against the pool gap. *Needs:* the stored results of `main` (all its methods, vanilla accuracy, $\lvert U\rvert=1000$) and of E2.5; the predictions and labels of $V$ and of the pool, from the pretrain files; no runs.
- **E4.c2, with or without replacement.** *Why:* under the paper's protocol the pool term $e_P$ is an error that no method represents; if the bags are drawn without replacement, each bag is an i.i.d. sample of the target distribution and $e_P$ becomes part of a luck of the bag of the right size (§T.10). Comparing the two protocols on the same pools shows how much of each method's miscalibration the protocol creates, and how the error split changes. *What it would show:* per method and pair, coverage, amplitude, Winkler score and AE under both protocols, their differences (paired by bag index, since the bags share their prevalences and seeds), and the error split of §T.2 under both protocols; coverage both per pair and averaged over pairs, since without replacement a share $n_j/P_j$ of the luck of the bag is common to the bags of a pair (§T.10). Predicted (§T.10): the methods that represent $e_2$ and $e_3$ (BayCAP with the realised-table predictive) gain the most, more on small pools; those that miss $e_2$ (SQS-H, RQBS) gain too but stay short; BayCAP with the population pushforward gains where the pool term is large but stays short, since it still misses the luck of the bag; for the baselines (DoC, QuAcc, O-LEAP, L-CBPE) the change is measured, not predicted. *Needs:* a check of the pool sizes first; a bag size $N_d$ per dataset, the largest of $\{25,50,100,250,500\}$ not above $\min_jP_j/4$, so that every class fits without replacement at every UPP prevalence and the bags of a pair share at most a quarter of the variance of $e_3$, with the datasets whose $\min_jP_j$ is below $100$ excluded (`mammographic`, with about $124$ instances per class in its pool, stays in with $N_d=25$); UPP bags without replacement, a new option of the sampling of `main`; two runs of the methods of `main` at $N_d$, with and without replacement, with the same pools, $V$, $h$, prevalences and seeds, stored apart from `main`; the stored results of `main` at $N=1000$, which link the run with replacement at $N_d$ to the paper; the pool-gap measures of E4.c1 for the run with replacement.

#### T.0 The four questions

*Intuition.* The budget says where the error of a CAP estimate comes from and which of its parts each method's interval accounts for. The experiments of this extension ask four questions, in order: is the budget right; does it explain the coverage of the methods; can it be used to fix their intervals; and how much of what we observe comes from the evaluation protocol rather than from the methods. Every interval is scored against the realised table of each bag, as in `main`; all the claims below are new hypotheses.

**A. Is the budget right? (E4.1; theory §T.2–§T.5, §T.11, §T.13).** The split of the error into $e_1^U$, $e_1^V$, $e_2$, $e_P$ and $e_3^b$ is an identity, but the size of each part comes from first-order formulas that assume independent binomial counts, uncorrelated parts and approximately normal errors. E4.1a measures the parts of the SQS family on real data over a grid of $\lvert U\rvert$ and $\lvert V\rvert$, and tests the predicted scalings: the parts driven by the bag ($e_3^b$, $e_1^U$) shrink as $\lvert U\rvert^{-1/2}$, those driven by $V$ ($e_2$, $e_1^V$) do not depend on $\lvert U\rvert$ and shrink as $\lvert V_j\rvert^{-1/2}$, and the pool term $e_P$ depends on neither. E4.1b tests the formulas of BayCAP with joint estimation of $(\pi^Q,C)$ (the variants of the `BayCAP` class), whose prevalence part is the luck of the bag and the error of $V$ scaled by $\kappa=\big((d_1-d_0)/(d_1+d_0-1)\big)^2$, with structural correlations and a bias $(1-Kd_j)/(\lvert V_j\rvert+K)$ due to the prior. E4.1c measures how tight the exact bound $\lvert e_1\rvert\le\frac12\lVert\tilde\pi-\pi^Q\rVert_1(\max\tilde d-\min\tilde d)$ is, i.e. how well the quantifier's error times the recall spread of $h$ predicts the accuracy error it causes. If the budget fails here, the coverage predictions of B have no basis and the budget must be refined first (correlations, non-normality, quantifier bias).

**B. Does the budget explain coverage? (E4.2; theory §T.6–§T.8, §T.12, §T.14).** An interval is wide only along the parts it simulates, so its coverage is predicted by the ratio $\eta$ of missing to represented variance, $2\Phi(1.96/\sqrt{1+\eta})-1$. E4.2a draws the regime map: the coverage of every method across $\lvert U\rvert$ and $\lvert V\rvert$, against the design ratio $r$ and against $\eta$, testing that SQS-type intervals degrade as $\lvert U\rvert$ grows, that the population-pushforward BayCAP under-covers at small $\lvert U\rvert$ and large $V$, and that the methods representing every part but $e_P$ stay near nominal. E4.2b checks that the map moves as predicted with the prevalence, the number of classes and the metric. E4.2c and E4.2d test the width floors: an interval narrower than the luck of the bag cannot cover, and the floor is lower for methods that see instance-level confidences, which predicts where the baselines can be narrower and still valid. E4.2e tests self-diagnosis: whether the coverage predicted from a method's own width and the label-free plug-ins of the parts it misses tracks the observed coverage. E4.2f tests the break-even bag size $n^*$ as a label-free rule for SQS-type intervals. If B holds, the paper's comparison at $\lvert U\rvert=1000$ is one point of a predictable map.

**C. Can the budget fix the intervals? (E4.3; theory §T.14, §T.15).** If the missing parts explain the under-coverage, adding them back should restore it. E4.3a widens each interval by the plug-ins of the parts it misses, and moves BayCAP's centre by its estimated prior bias, and compares the results with `BayesCAP-real`, which represents those parts by construction. E4.3b builds intervals with a finite-sample guarantee (binary ACC), the conservative end of the scale: they should cover at least $1-\alpha$ at every size, and their width measures the price of not relying on the normal approximation. What remains after the corrections measures the parts that have no label-free plug-in: the pool term $e_P$ and the quantifier bias $b_q$.

**D. How much is due to the protocol? (candidates E4.c1, E4.c2; theory §T.1, §T.3, §T.10).** This question is not scheduled: it was experiment E4.4 and was demoted to candidates on 2026-11-04. In the paper's protocol all the bags of a dataset are re-drawn with replacement from one finite test pool, which adds the pool term $e_P$: shared by all bags and classifiers of a dataset, not shrinking with $\lvert U\rvert$, and represented by no method. Candidate E4.c1 would measure the observable pool gap $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$ on stored results and compare it with the shift of every method (cause 4 of [[ext2_baycap|Extension 2]]); candidate E4.c2 would compare the protocols with and without replacement on the same pools: without replacement, the pool term becomes part of the luck of a fresh bag, so the methods that represent all the other parts should become calibrated. Without D, A–C still read the gaps left by the plug-ins through the pool gap of each bag, computed as a reference in the analyses on stored results; D would study it in full.

**How they depend on each other.** E4.1c, E4.2c, E4.2e, E4.2f and E4.3a run first, on the stored results of `main`, with one shared set of plug-ins and the pool gap of each bag as a reference. E4.1b needs a toy simulation. E4.2d (confidence summaries) and E4.3b (ACC quantities) need a re-run that stores per-bag quantities; if candidate E4.c2 is promoted, its runs can store them. The grid study of E4.1a, E4.2a and E4.2b is the most expensive and comes last, informed by the earlier results.

#### T.3 Sources of each part, and the pool term

*Intuition.* Each part is driven by one of three independent random draws: the target bag $U$, the validation set $V$, or the test pool. The parts driven by $U$ shrink as $U$ grows; those driven by $V$ or by the pool do not.

**The prevalence error.** Let $\pi^\infty(V)$ be the prevalence that the quantifier fitted on this $V$ would return on an infinitely large target sample with prevalence $\pi^Q$. Then

$$
\tilde\pi-\pi^Q=\big(\tilde\pi-\pi^\infty(V)\big)+\big(\pi^\infty(V)-\pi^Q\big),
\qquad
e_1=e_1^U+e_1^V,
$$

with $e_1^U=\sum_j(\tilde\pi_j-\pi^\infty_j(V))\tilde d_j$ and $e_1^V=\sum_j(\pi^\infty_j(V)-\pi^Q_j)\tilde d_j$.

**The finite-target error.** Under the protocol of §T.1 the bags vary around the accuracy of the pool, $\sum_j\pi^Q_jd^{\text{pool}}_j$. Adding and subtracting it splits $e_3$ into two parts:

$$
e_3=d^\top\pi^Q-A(h,U)=\underbrace{\sum_j\pi^Q_j\big(d_j-d^{\text{pool}}_j\big)}_{e_P\ \text{pool}}+\underbrace{\Big(\sum_j\pi^Q_jd^{\text{pool}}_j-A(h,U)\Big)}_{e_3^b\ \text{bag}}.
$$

$e_P$ is the luck of the pool: how much better or worse $h$ does on the instances that happen to be in the pool than on the population. $e_3^b$ is the luck of the bag given the pool: which of the pool's instances the bag happens to contain. Every bag of a dataset is drawn from the same pool, so every bag, and every classifier evaluated on that dataset, has the same $d^{\text{pool}}$; $e_P$ is the same for all bags with the same prevalence.

The five parts are:

| Part | Meaning | Source | Behaviour |
|---|---|---|---|
| $e_3^b$ | luck of the bag: which pool instances it contains, and whether $h$ classifies them correctly | $U$ | mean $0$, variance $\propto1/n$ |
| $e_P$ | luck of the pool: the recalls of $h$ on the pool against the population | pool | fixed given the pool; variance $\propto1/P_j$ over draws of the pool |
| $e_1^U$ | quantifier noise from seeing only $n$ target instances | $U$ | mean about $0$, variance $\propto1/n$ |
| $e_1^V$ | quantifier error from fitting it on a finite $V$, plus its systematic bias $b_q$ | $V$ | fixed given $V$; variance $\propto1/\lvert V\rvert$ over draws of $V$ |
| $e_2$ | error of the rates of $h$ measured on a finite $V$ | $V$ | fixed given $V$; variance $\propto1/\lvert V\rvert$ over draws of $V$ |

$e_3=e_P+e_3^b$ is the sampling variability of the **estimand**: it exists even with $\pi^Q$ and $C$ known, and it is the estimand gap of [[ext2_baycap]] §T.2. $e_1^U$, $e_1^V$ and $e_2$ are sampling errors of the **estimator**. $e_1^V$ is the quantifier's analogue of $e_2$: the first is the error of the surrogate classifier's rates, the second that of the rates of $h$. $e_P$ has the form of $e_2$, with the pool in place of $V$: both are the error of a sample frequency of correct predictions, measured once and shared by all bags. If $d$ is replaced by $d^{\text{pool}}$, as an experiment must do, $e_2$ becomes $e_2+e_P=\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$, the gap between the recalls on $V$ and on the pool, and $e_3$ becomes $e_3^b$.

#### T.4 Variance of each part

All variances below are first order in the estimation errors (Property C) and assume the paper's protocol: fixed class counts $n_j=n\pi^Q_j$, instances drawn with replacement from the pool. Three properties are used throughout; we derive them first.

**Property A: the variance of a frequency.** A Bernoulli variable $B$ equals $1$ with probability $p$ and $0$ otherwise. Its mean is $\mathbb E[B]=p$, and since $B^2=B$, also $\mathbb E[B^2]=p$; so $\operatorname{Var}(B)=\mathbb E[B^2]-\mathbb E[B]^2=p-p^2=p(1-p)$. A count of successes in $m$ independent trials is $k=B_1+\dots+B_m\sim\operatorname{Bin}(m,p)$; the variance of a sum of independent variables is the sum of their variances, so $\operatorname{Var}(k)=m\,p(1-p)$. Multiplying a variable by a constant $a$ multiplies its variance by $a^2$ (from the definition $\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]$), so the frequency $k/m$ has

$$
\operatorname{Var}\Big(\frac km\Big)=\frac{m\,p(1-p)}{m^2}=\frac{p(1-p)}m.
$$

**Property B: the variance of a weighted sum of independent variables.** In general $\operatorname{Var}(\sum_ja_jX_j)=\sum_j\sum_ka_ja_k\operatorname{Cov}(X_j,X_k)$. If the $X_j$ are independent, $\operatorname{Cov}(X_j,X_k)=0$ for $j\ne k$, and $\operatorname{Cov}(X_j,X_j)=\operatorname{Var}(X_j)$, so

$$
\operatorname{Var}\Big(\sum_ja_jX_j\Big)=\sum_ja_j^2\operatorname{Var}(X_j).
$$

**Property C: first order and the delta method.** "First order" means that every estimation error is treated as a small number $\varepsilon$, and every term that is a product of two small errors (or a square, $\varepsilon^2$) is dropped. It comes from the Taylor expansion of a smooth function $f$ around a point $\mu$:

$$
f(\mu+\varepsilon)=f(\mu)+f'(\mu)\,\varepsilon+\tfrac12f''(\mu)\,\varepsilon^2+\dots
$$

The first-order approximation keeps $f(\mu)+f'(\mu)\varepsilon$: near $\mu$, the curve is replaced by its tangent line. The dropped terms are small because the errors here are sampling errors: an error from a sample of size $m$ (here $n$, $\lvert V_j\rvert$ or $P_j$) has a standard deviation proportional to $1/\sqrt m$ (Property A), so $\varepsilon^2$ and products of two errors are of order $1/m$, smaller than $\varepsilon$ by a factor of order $1/\sqrt m$; the larger the samples, the smaller their share. The *delta method* uses this to compute variances. Let $X$ be random with mean $\mu$ and small spread. To first order, $f(X)\approx f(\mu)+f'(\mu)(X-\mu)$; $f(\mu)$ and $f'(\mu)$ are constants, a constant added to a variable does not change its variance, and a constant factor multiplies it by its square, so

$$
\operatorname{Var}\big(f(X)\big)\approx f'(\mu)^2\operatorname{Var}(X).
$$

With several independent variables $X_1,\dots,X_k$, the tangent line becomes $f(\mu)+\sum_i\frac{\partial f}{\partial x_i}(X_i-\mu_i)$, and Property B gives $\operatorname{Var}(f)\approx\sum_i\big(\frac{\partial f}{\partial x_i}\big)^2\operatorname{Var}(X_i)$, with the partial derivatives taken at the means. First order is used in three places below: the cross term inside $e_1$, the nonlinear formula of the quantifier, and the replacement of $d^{\text{pool}}$ by $d$ inside a variance. The independence of the parts within a source, assumed in §T.5, is a separate approximation (see Correlations below): unlike the first-order terms, a correlation does not become negligible as the samples grow.

**$e_3^b$.** Given the pool, $c_j\sim\operatorname{Bin}(n_j,d^{\text{pool}}_j)$, independently across classes (§T.1). The realised accuracy is a weighted sum of the class frequencies: $A(h,U)=\frac1n\sum_jc_j=\sum_j\frac{n_j}n\cdot\frac{c_j}{n_j}=\sum_j\pi^Q_j\frac{c_j}{n_j}$. By linearity of the expectation, $\mathbb E[c_j/n_j\mid\text{pool}]=d^{\text{pool}}_j$, and by Property A, $\operatorname{Var}(c_j/n_j\mid\text{pool})=d^{\text{pool}}_j(1-d^{\text{pool}}_j)/n_j$. Hence, by Property B and $n_j=n\pi^Q_j$,

$$
\mathbb E[A(h,U)\mid\text{pool}]=\sum_j\pi^Q_jd^{\text{pool}}_j,
\qquad
\operatorname{Var}(e_3^b\mid\text{pool})=\sum_j(\pi^Q_j)^2\frac{d^{\text{pool}}_j(1-d^{\text{pool}}_j)}{n\pi^Q_j}=\frac1n\sum_j\pi^Q_jd^{\text{pool}}_j(1-d^{\text{pool}}_j)\approx\frac1n\sum_j\pi^Q_jd_j(1-d_j),
$$

where the last step replaces $d^{\text{pool}}_j$ by $d_j$, a second-order change: it changes $d^{\text{pool}}_j(1-d^{\text{pool}}_j)$ by an amount proportional to $d^{\text{pool}}_j-d_j$, of order $1/\sqrt{P_j}$, and this change is then multiplied by $1/n$, so it is smaller than the term itself by a factor of order $1/\sqrt{P_j}$. One factor $\pi^Q_j$ of the squared weight cancels against the sample size $n_j=n\pi^Q_j$ of the class.

**$e_P$.** The pool's class $\omega_j$ is a random sample of $P_j$ instances of the population's class $\omega_j$, so the number of them that $h$ classifies correctly is $\operatorname{Bin}(P_j,d_j)$ and $d^{\text{pool}}_j$ is its frequency, independently across classes. With the constant weights $\pi^Q_j$,

$$
\operatorname{Var}(e_P)=\sum_j(\pi^Q_j)^2\frac{d_j(1-d_j)}{P_j}.
$$

**$e_3$, by the law of total variance.** The law of total variance says that for two random quantities $X$ and $G$, $\operatorname{Var}(X)=\mathbb E[\operatorname{Var}(X\mid G)]+\operatorname{Var}(\mathbb E[X\mid G])$: the total spread is the average spread within each value of $G$ plus the spread of the averages. With $X=e_3$ and $G$ the pool, $\mathbb E[e_3\mid\text{pool}]=e_P$ (because $e_3^b$ has mean $0$ given the pool) and $\operatorname{Var}(e_3\mid\text{pool})=\operatorname{Var}(e_3^b\mid\text{pool})$, so

$$
\operatorname{Var}(e_3)\approx\frac1n\sum_j\pi^Q_jd_j(1-d_j)+\sum_j(\pi^Q_j)^2\frac{d_j(1-d_j)}{P_j}.
$$

The first term shrinks with $n$; the second does not. Their ratio for class $\omega_j$ is $n\pi^Q_j/P_j=n_j/P_j$, the number of times each pool instance of that class is used by one bag on average: with $n=1000$, $\pi^Q_j=1/2$ and $P_j=125$ the pool term is four times the bag term (illustrative). Under the paper's protocol, the luck of the pool is the larger part of the luck of the bag on every dataset whose pool is smaller than the bags.

**$e_2$.** Model: the class-$\omega_j$ instances of $V$ are $\lvert V_j\rvert$ independent draws from the population of that class, each classified correctly by $h$ with probability $d_j$. Let $k_j$ be the number of them that $h$ classifies correctly; it counts the successes of $\lvert V_j\rvert$ independent trials, so $k_j\sim\operatorname{Bin}(\lvert V_j\rvert,d_j)$, independently across classes. The estimated recall is the frequency $\tilde d_j=k_j/\lvert V_j\rvert$: the count divided by the number of class-$\omega_j$ instances, not $d_j$ divided by a size. The sample size is $\lvert V_j\rvert$, not $\lvert V\rvert$, because a recall is measured only on the instances of its own class. By Property A,

$$
\mathbb E[\tilde d_j]=d_j,\qquad\operatorname{Var}(\tilde d_j)=\frac{d_j(1-d_j)}{\lvert V_j\rvert},
$$

and, since $e_2=\sum_j\pi^Q_j(\tilde d_j-d_j)$ with constant weights $\pi^Q_j$ and constant $d_j$, by Property B

$$
\operatorname{Var}(e_2)=\sum_j(\pi^Q_j)^2\operatorname{Var}(\tilde d_j)=\sum_j\frac{(\pi^Q_j)^2d_j(1-d_j)}{\lvert V_j\rvert}.
$$

This is a *sampling* variance: it describes how $\tilde d_j$ would scatter around the fixed $d_j$ if $V$ were drawn again and again; for a given $V$, $e_2$ is a single number. It applies to the methods that use the frequencies $\tilde d_j$ of $V$ directly (the SQS family); a method that obtains its recalls otherwise, as BayCAP does through a posterior, has an $e_2$ with a different mean and variance. The weights are squared here and in $e_P$, and not in $e_3^b$, because in $e_3^b$ the prevalence also enters through the sample sizes $n_j=n\pi^Q_j$, which cancels one factor, while $\lvert V_j\rvert$ and $P_j$ do not depend on $\pi^Q$. $V$ and the pool are independent, so the observable gap of §T.3 has $\operatorname{Var}(e_2+e_P)=\sum_j(\pi^Q_j)^2d_j(1-d_j)\big(1/\lvert V_j\rvert+1/P_j\big)$.

**$e_1$, the cross term.** Write the error of the recalls as $\delta_j=\tilde d_j-d_j$, so that $\tilde d_j=d_j+\delta_j$. Then

$$
e_1=\sum_j(\tilde\pi_j-\pi^Q_j)\tilde d_j=\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,d_j}_{\text{first order}}+\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,\delta_j}_{\text{product of two errors}},
$$

and, by Property C, the variance of $e_1$ is computed from the first part only, with the true $d$ in place of $\tilde d$.

**$e_1$, binary case with ACC.** Prevalence errors sum to zero, $\tilde\pi_0-\pi^Q_0=-(\tilde\pi_1-\pi^Q_1)$, so the first part is $(\tilde\pi_1-\pi^Q_1)(d_1-d_0)$; $d_1-d_0$ is a constant, so $\operatorname{Var}(e_1)\approx(d_1-d_0)^2\operatorname{Var}(\tilde\pi_1)$. The whole term vanishes when $d=\beta\mathbf 1$, the paper's condition for unbiased cross-validation (Eq. 2). Let the surrogate classifier have rates $S_{10}=P(\hat Y_s=\omega_1\mid Y=\omega_0)$ and $S_{11}$, estimated on $V$ as $\hat S$, and let $\hat p$ be the fraction of $U$ that it predicts as $\omega_1$. ACC returns

$$
\hat\pi_1=\frac{\hat p-\hat S_{10}}{\hat S_{11}-\hat S_{10}}.
$$

$\hat\pi_1$ is a nonlinear function of $\hat p$, $\hat S_{10}$ and $\hat S_{11}$ (the estimated rates are in the denominator), so its variance is computed with the delta method (Property C). Its partial derivatives, at the true values $p$, $S_{10}$, $S_{11}$, follow from the quotient rule and from $(p-S_{10})/(S_{11}-S_{10})=\pi^Q_1$:

$$
\frac{\partial\hat\pi_1}{\partial\hat p}=\frac1{S_{11}-S_{10}},\qquad
\frac{\partial\hat\pi_1}{\partial\hat S_{10}}=\frac{-(S_{11}-S_{10})+(p-S_{10})}{(S_{11}-S_{10})^2}=\frac{\pi^Q_1-1}{S_{11}-S_{10}}=-\frac{\pi^Q_0}{S_{11}-S_{10}},\qquad
\frac{\partial\hat\pi_1}{\partial\hat S_{11}}=-\frac{p-S_{10}}{(S_{11}-S_{10})^2}=-\frac{\pi^Q_1}{S_{11}-S_{10}}.
$$

Each squared derivative multiplies the variance of the corresponding input. Two independent sources of noise enter it:

- **from $U$** ($e_1^U$): with fixed class counts, $\operatorname{Var}(\hat p)=\frac1n\big[\pi^Q_0S_{10}(1-S_{10})+\pi^Q_1S_{11}(1-S_{11})\big]$, so

$$
\operatorname{Var}(e_1^U)\approx\frac{(d_1-d_0)^2}{(S_{11}-S_{10})^2}\cdot\frac{\pi^Q_0S_{10}(1-S_{10})+\pi^Q_1S_{11}(1-S_{11})}{n};
$$

- **from $V$** ($e_1^V$): the derivatives with respect to $\hat S_{10}$ and $\hat S_{11}$ above, with $\operatorname{Var}(\hat S_{1j})=S_{1j}(1-S_{1j})/\lvert V_j\rvert$, so

$$
\operatorname{Var}(e_1^V)\approx\frac{(d_1-d_0)^2}{(S_{11}-S_{10})^2}\left[\frac{(\pi^Q_0)^2S_{10}(1-S_{10})}{\lvert V_0\rvert}+\frac{(\pi^Q_1)^2S_{11}(1-S_{11})}{\lvert V_1\rvert}\right].
$$

$e_1^U$ has the same structure as $e_3^b$ and $e_1^V$ the same structure as $e_2$, with the rates of the surrogate classifier in place of those of $h$, amplified by $1/(S_{11}-S_{10})^2$ (an ill-conditioned surrogate inflates both) and scaled by $(d_1-d_0)^2$. The multiclass case replaces these factors with the inverse of the surrogate's rate matrix and the vector $d$. With replacement, the surrogate's rates on the pool also differ from the population ones, which adds to $e_1^U$ a pool part of the same form as $e_P$; it is left out here, as a second-order effect for the quantifiers that read the surrogate only through $\hat p$.

**Correlations.** Parts driven by different draws ($U$, $V$, the pool) are uncorrelated, because the three draws are independent. Within a group they can be correlated: $e_3^b$ and $e_1^U$ if $h$ and the surrogate tend to misclassify the same target instances, $e_2$ and $e_1^V$ if they tend to misclassify the same validation instances. The toy of Appendix A assumes $h$ and the surrogate conditionally independent given $Y$, which removes both correlations; on real data the oracle ladder measures them. This assumption is doubtful on real data [new hypothesis]: `main` uses `PACC(MLP())` (`src/uncawe/experiments/methods.py`), whose surrogate is a separate MLP, so when $h$ is itself an MLP the two are likely to misclassify the same instances; and when the surrogate's rates are estimated on the same $V$ as $\tilde d$, $e_2$ and $e_1^V$ are computed from the same validation instances. A covariance of either sign then changes the budget of §T.5 by $2\operatorname{Cov}(e_2,e_1^V)$ (and $2\operatorname{Cov}(e_3^b,e_1^U)$), which can be of the same order as the variances.

#### T.5 The budget

Two properties give the budget. First, the mean squared error of any error $X$ is its variance plus its squared mean: $\mathbb E[X^2]=\operatorname{Var}(X)+(\mathbb E X)^2$, which follows from the definition $\operatorname{Var}(X)=\mathbb E[X^2]-(\mathbb E X)^2$. Second, the variance of a sum of uncorrelated parts is the sum of their variances (Property B of §T.4 with all weights equal to $1$). Applied to $\hat A-A(h,U)=e_1^U+e_1^V+e_2+e_P+e_3^b$ and grouping the parts by source gives

$$
\operatorname{MSE}(\hat A)\approx\underbrace{\operatorname{Var}(e_3^b)+\operatorname{Var}(e_1^U)}_{\text{from }U,\ \propto1/n}+\underbrace{\operatorname{Var}(e_2)+\operatorname{Var}(e_1^V)+b_q^2}_{\text{from }V,\ \text{independent of }n}+\underbrace{\operatorname{Var}(e_P)}_{\text{from the pool},\ \text{independent of }n},
$$

where $b_q$ is the quantifier's systematic bias propagated to accuracy (contained in $e_1^V$, written separately because it does not average out over draws of $V$). The approximation is first order and drops the within-group correlations of §T.4. The point of the budget is the scaling: as $n$ grows, only the first group shrinks.

The form of the budget is general, since it follows from the identity of §T.2; the size and the mean of each part are not. They depend on how the method obtains $\tilde\pi$ and $\tilde d$: the variances above are those of a method that uses the frequencies of $V$ and a quantifier such as ACC; for a method with a prior on $C$ (BayCAP) or a different quantifier, the same parts have other variances and may have a nonzero mean.

#### T.6 Which parts each method represents

**What is general and what depends on the method.** The budget is a common frame, but only its structure is shared by all methods:

- **General.** For every method, adding and subtracting $d^\top\pi^Q$ gives $\hat A-A(h,U)=(\hat A-d^\top\pi^Q)+e_3$, and $e_3=e_P+e_3^b$ depends only on the estimand and the protocol, not on the method. For the methods of the form $\hat A=\sum_j\tilde\pi_j\tilde d_j$ (the SQS family and the BayCAP family), the estimator error splits further into $e_1^U+e_1^V+e_2$ (§T.2, §T.3). The baselines of the paper (DoC, QuAcc, O-LEAP, L-CBPE) produce no $\tilde\pi$ and $\tilde d$, so for them only the two-part split exists. The split is exact for vanilla accuracy, which is linear in the table, and first order for nonlinear metrics (§T.9).
- **Depends on how the accuracy is estimated.** The variances of the parts: those of §T.4 are those of a method that uses the frequencies of $V$ and a quantifier such as ACC; BayCAP obtains its recalls from a posterior, and each quantifier (ACC on a surrogate, BayesianCC on $h$, KDEy) has its own $\operatorname{Var}(e_1^U)$ and $\operatorname{Var}(e_1^V)$. The means of the parts: the quantifier bias $b_q$, and a prior on $C$, which can give $e_2$ a nonzero mean that does not average out over draws of $V$ (§T.13). The within-source correlations of §T.4, which depend on whether the quantifier and $h$ share predictions and validation instances.
- **Depends on the interval.** Which parts the method's samples simulate (the table below). Two methods with the same point estimate, hence the same split, can have different coverage, as BayCAP and the realised-table BayCAP do.
- **Depends on the protocol, for all methods alike.** With replacement, $e_P$ adds to $e_3^b$; without replacement, it is part of the luck of a fresh bag (§T.10).

*Intuition.* A method's samples vary only along the sources it simulates; the other sources move the centre of its interval without widening it.

| Method | Status | Spread comes from | $e_3^b$ | $e_P$ | $e_1^U$ | $e_1^V$ | $e_2$ |
|---|---|---|---|---|---|---|---|
| SQS-H as in the paper (§4.1) | [paper] | bags of size $n$ at the counts $n\hat\pi_j$, drawn from $V$ | yes | no | no | no | no |
| `RQBS` (SQS-H as run by `main`) | [implemented] | bootstrap of $\hat\pi$ on $U$, one bag per draw | yes | no | yes | no | no |
| SQS-S (§4.2) | [paper] | bags whose true prevalences lie near $\hat\pi$ in estimated-prevalence space | yes | no | partly (hypothesis) | no | no |
| SQS-B | [proposal] ([[ext3_posterior_propagation]]) | posterior of $\pi^Q$, one bag per draw | yes | no | yes | if the posterior models the quantifier's rates (BayesianCC does); not with BayesKDEy, whose kernel densities are fixed | no |
| BayCAP (§5) | [paper] | joint posterior of $(\pi^Q,C)$, population pushforward | no | no | yes | yes | yes |
| BayCAP with the realised-table predictive (`BayesCAP-real`) | [implemented] ([[ext2_baycap]] E2.3) | as BayCAP, plus a sampled realised table | yes | no | yes | yes | yes |
| BayCAP-KDEy | [implemented] ([[ext3_posterior_propagation]]) | posterior of $\pi^Q$ (BayesKDEy), fixed $\hat C$ | no | no | yes | no (kernel densities fixed) | no |
| BayCAP-KDEy, realised pushforward | [proposal] ([[ext3_posterior_propagation]] E3.1) | as BayCAP-KDEy, plus a sampled realised table | yes | no | yes | no | no |
| SQS-H with $U$ and $V$ resampling | [new hypothesis] | bootstrap of $U$ and of $V$, one bag per draw | yes | no | yes | yes | yes |

Why SQS-H represents $e_3^b$: its bags have the same class counts as the target bag and draw class-$\omega_j$ instances from $V_j$, so the number of correct predictions in class $\omega_j$ is $\operatorname{Bin}(n_j,\tilde d_j)$, the same mechanism that generates the target bag with $d^{\text{pool}}_j$ replaced by $\tilde d_j$. SQS-H is a simulator of the bag's luck, with $V$ as a stand-in for the class-conditional distributions. Why it does not represent $e_2$: every bag uses the same $\tilde d$, so $e_2$ is an offset shared by all draws. BayCAP's posterior describes the uncertainty on $(\pi^Q,C)$, but each draw is a population accuracy $d^{(r)\top}\pi^{(r)}$ with no binomial luck of the bag, so it has no $e_3^b$ ([[ext2_baycap]] §T.2). BayCAP estimates $\pi^Q$ from the predictions of $h$, not of a surrogate, so its $e_1$ differs in size from that of the SQS family. **No method represents $e_P$**: none of them knows that the bags are re-drawn from a pool, nor its size; a method that resamples $V$ represents the uncertainty of $\tilde d$ around $d$, not that of $d^{\text{pool}}$ around $d$.

#### T.7 From the budget to coverage

*Intuition.* An interval is as wide as the noise it simulates; if the actual error also contains noise it does not simulate, it misses.

Write $\sigma^2_{\text{in}}$ for the sum of the variances of the parts a method represents and $\sigma^2_{\text{out}}$ for the sum of those it misses. Assume that the represented parts are simulated at the right size, that all parts are approximately normal (sums of many binomials, by the central limit theorem), and that represented and missing parts are independent. Then:

- the method's samples are approximately normal with standard deviation $\sigma_{\text{in}}$, and the 2.5% and 97.5% percentiles of $N(\mu,\sigma^2)$ are $\mu\pm1.96\,\sigma$, so the interval is approximately $\hat A\pm1.96\,\sigma_{\text{in}}$;
- the interval covers $A(h,U)$ when $\lvert\hat A-A(h,U)\rvert\le1.96\,\sigma_{\text{in}}$, and $\hat A-A(h,U)$, the sum of all parts, has variance $\sigma^2_{\text{in}}+\sigma^2_{\text{out}}$;
- dividing by its standard deviation,

$$
\text{coverage}\approx P\Big(\lvert Z\rvert\le\frac{1.96\,\sigma_{\text{in}}}{\sqrt{\sigma^2_{\text{in}}+\sigma^2_{\text{out}}}}\Big)=2\Phi\Big(\frac{1.96}{\sqrt{1+\eta}}\Big)-1,
\qquad
\eta=\frac{\sigma^2_{\text{out}}}{\sigma^2_{\text{in}}},
$$

with $Z$ standard normal. Checks: $\eta=0$ gives $0.95$; $\eta=1$, a missing variance as large as the represented one, gives $2\Phi(1.386)-1\approx0.83$; $\eta\to\infty$ gives $0$.

**Worked example: SQS-H with the true prevalence** (the oracle-$\pi$ reference). Its bags use the true counts, so $e_1=0$; it represents $e_3^b$ and misses $e_2$ and $e_P$. Its interval is approximately $\tilde d^\top\pi^Q\pm1.96\,\sigma_{3b}$: the width is right, the centre is off by $e_2+e_P$, because $\tilde d^\top\pi^Q=d^\top\pi^Q+e_2$ and $A(h,U)=d^\top\pi^Q-e_P-e_3^b$. Without the pool term, as in the toy of Appendix A, $\eta=\operatorname{Var}(e_2)/\operatorname{Var}(e_3^b)=r$, the design ratio

$$
r=\frac{\sum_j(\pi^Q_j)^2d_j(1-d_j)/\lvert V_j\rvert}{\frac1n\sum_j\pi^Q_jd_j(1-d_j)},
$$

which reduces to $n\sum_j(\pi^Q_j)^2/\lvert V_j\rvert$ when $d_j(1-d_j)$ and $\lvert V_j\rvert$ are equal across classes. In the toy, the closed form $2\Phi(1.96/\sqrt{1+r})-1$ is close to the simulated coverage of the oracle-$\pi$ reference: there $r=0.5\,n/n_s$, with $n_s$ validation instances per class, so $n=1000$ and $n_s=100$ give $r=5$ and a predicted coverage of $0.58$, against a simulated $0.50$ (flat $d$, Appendix A, toy). With the pool term, $\eta$ gains $\operatorname{Var}(e_P)/\operatorname{Var}(e_3^b)$, which by §T.4 is the analogue of $r$ with $P_j$ in place of $\lvert V_j\rvert$; with $P_j\approx\lvert V_j\rvert$, as in the splits of `main`, the two are about equal and $\eta\approx2r$.

**Conditional on one $V$ and one pool.** The formula averages over draws of $V$ and of the pool. For a given $V$ and pool, $e_2+e_P$ is a constant and the coverage of the same interval is $\Phi(1.96-(e_2+e_P)/\sigma_{3b})-\Phi(-1.96-(e_2+e_P)/\sigma_{3b})$: close to $0.95$ if they happen to give a small offset, close to $0$ if $\lvert e_2+e_P\rvert\gg\sigma_{3b}$. Coverage measured on many bags from a single $V$ and a single pool therefore measures that $V$ and that pool, not the method; experiments must repeat over $V$ seeds, and the pool is shared by all the classifiers of a dataset, so their coverages are correlated.

**Ratio for each method.** With $\sigma_k^2$ the variance of part $e_k$:

| Method | Represented | Missing | $\eta$ |
|---|---|---|---|
| SQS-H with true $\pi^Q$ (reference) | $e_3^b$ | $e_2$, $e_P$ | $(\sigma_2^2+\sigma_P^2)/\sigma_{3b}^2$ |
| SQS-H | $e_3^b$ | $e_1^U$, $e_1^V$, $e_2$, $e_P$ | $(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2+\sigma_P^2)/\sigma_{3b}^2$ |
| RQBS | $e_3^b$, $e_1^U$ | $e_1^V$, $e_2$, $e_P$ | $(\sigma_{1V}^2+\sigma_2^2+\sigma_P^2)/(\sigma_{3b}^2+\sigma_{1U}^2)$ |
| BayCAP (population) | $e_1^U$, $e_1^V$, $e_2$ | $e_3^b$, $e_P$ | $(\sigma_{3b}^2+\sigma_P^2)/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ |
| Realised-table BayCAP; SQS-H with $U$ and $V$ resampling | all but $e_P$ | $e_P$ | $\sigma_P^2/(\sigma_{3b}^2+\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ |

All $\sigma_k$ but $\sigma_2$ and $\sigma_P$ separately can be estimated in an experiment with labels (E4.1a); $\sigma_2^2+\sigma_P^2$ is estimated from the observable gap, and its split follows from §T.4. So each method's $\eta$ is measurable and the coverage rule is testable (E4.1a).

#### T.8 The regime hypothesis

*Intuition.* As $n$ grows, the parts driven by $U$ shrink and those driven by $V$ or by the pool stay, so methods that simulate only the first kind become overconfident, and methods that miss the bag's luck become well calibrated.

From §T.4 and the table of §T.7 (new hypothesis):

- **SQS-H and RQBS** miss parts driven by $V$ and by the pool and represent parts driven by $U$: their $\eta$ grows linearly in $n$ and their coverage falls towards $0$ for $n\gg\lvert V_j\rvert$ or $n\gg P_j$. SQS-H also misses $e_1^U$, so it under-covers already at small $n$ when the prevalence term is large (large $\lvert d_1-d_0\rvert$ or ill-conditioned surrogate).
- **BayCAP** misses $e_3^b$ and $e_P$. As $n$ grows, $\sigma_{3b}^2$ and $\sigma_{1U}^2$ shrink while $\sigma_{1V}^2+\sigma_2^2$ stays, so the part of $\eta$ due to $e_3^b$ goes to $0$, and $\eta$ approaches $\sigma_P^2/(\sigma_{1V}^2+\sigma_2^2)$, which is at most about $1$ when $P_j\approx\lvert V_j\rvert$ (equal to $1$ if $e_1^V$ is negligible), so coverage approaches a level between $2\Phi(1.96/\sqrt2)-1\approx0.83$ and $0.95$ rather than $0.95$. At small $n$, or when $V$ is large, $\eta$ is dominated by $\sigma_{3b}^2/\sigma_{1U}^2$: BayCAP's coverage then has a floor set by how the quantifier's own noise compares with the bag's luck.
- **Methods that represent all parts but $e_P$** (realised-table BayCAP, SQS-H with $U$ and $V$ resampling) are near nominal when the pool is much larger than the bags and $V$, apart from the quantifier bias $b_q$; with the pools of `main` they miss $e_P$, and their coverage should fall as $n$ grows past $P_j$.
- **$r$ as the design axis.** $r$ depends only on $n$, $\lvert V_j\rvert$, $\pi^Q$ and $d$, so it can be computed for every cell of an experiment before running any method; the $\eta$ of each method is then measured. The pool adds a second, fixed axis per dataset, $n/P_j$. The paper's comparison at $\lvert U\rvert=1000$ fixes one value of $r$ and of $n/P_j$ per dataset.

**SQS-H with $U$ and $V$ resampling** follows directly: for every draw, resample $U$ (as `RQBS` does) and resample $V$ with stratification, refit the quantifier's rate estimates and the rates of $h$ on the resampled $V$, and draw one bag of size $n$ at the resulting prevalence. It is the non-Bayesian counterpart of the realised-table BayCAP. Its cost is one quantifier refit per draw; for aggregative quantifiers such as PACC only the misclassification rates need refitting, not the surrogate classifier, which is an approximation to be checked.

The toy simulation (Appendix A), which draws its bags from the population and therefore has no pool term, supports the first three points in a binary setting.

#### T.9 Caveats

- **$\pi^Q$ and $C$ are dependent in BayCAP's posterior**, through the observed target predictions $n^t\approx n\,C\pi^Q$. The budget ignores this covariance; the oracle ladder measures it implicitly.
- **Nonlinear metrics.** For macro-F1 the split of §T.2 does not hold exactly and the budget holds only to first order (delta method). For macro-K, which depends on $\pi^Q$ only weakly ([[ext1_metrics]]), $e_1$ should be small and $e_2$, $e_P$, $e_3^b$ should dominate; this is a sharp test of the budget.
- **Natural sampling.** Under natural (i.i.d.) sampling instead of APP, the class counts are multinomial, which adds a term $\frac1n\operatorname{Var}_{\pi^Q}(d_Y)$ to $e_3^b$. All experiments here keep APP, as in the paper.
- **Quantifier bias** is not a variance and is represented by no method. If the completed methods still under-cover at large $n$, bias and the pool term are the likely causes; the oracle ladder separates bias from $e_1^V$ by repeating over $V$ seeds, and E4.c1 measures the pool term.
- **Small $n$.** Rounding $n\pi^Q_j$ can leave classes empty; the conventions of [[ext1_metrics]] for rare and absent classes apply.
- **When first order fails.** The first-order approximation of §T.4 (Property C) needs small errors. It can fail with small $\lvert V_j\rvert$, $n$ or $P_j$ (rare classes), where the dropped products are no longer negligible; with an ill-conditioned surrogate, where $S_{11}-S_{10}$ is close to $0$, ACC is strongly curved and $\hat\pi$ can have heavy tails that the delta method underestimates; and near the vertices of the simplex, where quantifiers clip $\hat\pi$, which is not a smooth operation. In the toy of Appendix A the coverage rule is least accurate at large $r$, where the normal approximation for $e_2$ is coarsest.
- **"Represents a part" is an assumption** [new hypothesis]. The coverage rule of §T.7 assumes that each represented part is simulated at its true size. For BayCAP the spread along $e_1$ and $e_2$ is a posterior width, not a sampling variance, and it can be displaced by the prior ([[ext2_baycap]]). A failure of the rule can therefore come either from a wrong $\sigma_{\text{in}}$ or from a missing part; telling them apart needs a comparison of each method's own spread with $\sqrt{\sum_{\text{represented}}\sigma_k^2}$ before the comparison of its coverage with $\eta$.
- **Measuring $e_1^V$ through a large bag from the pool** [new hypothesis]. The approximation of $\pi^\infty(V)$ in E4.1a uses a bag of $10^5$ instances drawn from the pool, which inherits the surrogate's rates on the pool; the measured $e_1^V$ then contains the surrogate's pool part set aside in §T.4, which may not be second order on small pools.
- **$e_2$ and $e_P$ are not separately identifiable, and $e_P$ is one draw per pool** [new hypothesis]. Only their sum $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$ is observable; their split relies on the variances of §T.4. Within a dataset $e_P$ is a single realisation, so its variance is seen only across datasets, or across outer seeds if these re-draw the pool (to be checked in the splits of `main`).

#### T.10 A protocol without replacement (background for E4.c2)

*Intuition:* a bag re-drawn with replacement from a small pool repeats the pool's peculiarities in every bag. If each bag instead takes distinct instances of the pool, chosen at random, then the bag is just a random sample of the population, as a deployment sample would be: the pool's luck does not disappear, but it becomes part of the bag's own luck, at the size every method that simulates a bag already assumes.

**The protocol.** For each class $\omega_j$, the bag takes $n_j=n\pi^Q_j$ (rounded) *distinct* instances of the pool's class $\omega_j$, chosen uniformly at random, without replacement. This needs $n_j\le P_j$ for every class and every prevalence; since UPP draws prevalences over the whole simplex, $n_j$ can be as large as $n$, so the bag size must satisfy $n\le\min_jP_j$.

**A bag drawn without replacement from a random pool is a random sample of the population.** Take one class and write its pool as $X_1,\dots,X_{P_j}$, independent draws from the class-conditional distribution (the pool is a random sample of the dataset). The bag takes the instances at a set $S$ of $n_j$ distinct indices, chosen at random and independently of the $X_i$. For any fixed set $S$, the instances $\{X_i:i\in S\}$ are $n_j$ of the independent draws, so they are independent draws from the class-conditional distribution. Since $S$ is chosen independently of the $X_i$, this holds for whichever $S$ is chosen, and therefore also without conditioning on $S$. So the bag's instances of class $\omega_j$ are $n_j$ independent draws from the population, each correctly classified with probability $d_j$, and

$$
c_j\sim\operatorname{Bin}(n_j,d_j),\qquad\operatorname{Var}(e_3)=\frac1n\sum_j\pi^Q_jd_j(1-d_j),
$$

exactly the variance of the luck of a fresh bag. With replacement the argument fails, because a bag of $n_j>P_j$ instances must repeat some of them, and repeated instances are not independent draws.

**How the split changes.** The decomposition of §T.3, $e_3=e_P+e_3^b$, still holds, since it is an identity. What changes is the variance of $e_3^b$. Given the pool, the number of correct instances in the bag is hypergeometric: it counts the correct ones among $n_j$ instances drawn without replacement from $P_j$ instances, of which $P_jd^{\text{pool}}_j$ are correct. Its variance is the binomial one times the finite-population correction $(P_j-n_j)/(P_j-1)$, which accounts for the fact that each draw without replacement leaves fewer instances to choose from:

$$
\operatorname{Var}\Big(\frac{c_j}{n_j}\,\Big|\,\text{pool}\Big)=\frac{d^{\text{pool}}_j(1-d^{\text{pool}}_j)}{n_j}\cdot\frac{P_j-n_j}{P_j-1}\approx d_j(1-d_j)\Big(\frac1{n_j}-\frac1{P_j}\Big).
$$

Adding the variance $d_j(1-d_j)/P_j$ of the pool's recall (§T.4), by the law of total variance, gives $d_j(1-d_j)/n_j$: the two terms $\pm1/P_j$ cancel, and the total is the variance of a fresh bag, as the argument above showed. So without replacement the pool term is still there, but it no longer adds to the luck of the bag: it is a share $n_j/P_j$ of it. With replacement, by contrast, the bag part is $d_j(1-d_j)/n_j$ and the pool part adds $d_j(1-d_j)/P_j$ on top (§T.4).

**What this predicts for the methods** (new hypotheses, for E4.c2). Every method that represents the luck of the bag at the binomial size (§T.6) now represents $e_P$ as well, because it is part of that luck:

- the methods that represent $e_3^b$, $e_1$ and $e_2$ (realised-table BayCAP, SQS-H with $U$ and $V$ resampling) have $\eta=0$ and should be near nominal; with replacement they miss $e_P$, so the gain should be largest where $n_j/P_j$ is large, on small pools;
- the methods that represent $e_3^b$ but not $e_2$ (SQS-H, RQBS) lose the $\sigma_P^2$ term of their $\eta$ and gain coverage, but still miss $e_2$ and $e_1^V$;
- BayCAP with the population pushforward misses the luck of the bag in both protocols: with replacement it misses $\sigma_{3b}^2+\sigma_P^2$, without replacement only the variance of a fresh bag, $\frac1n\sum_j\pi^Q_jd_j(1-d_j)$, smaller by about $\sigma_P^2$; it should gain coverage where the pool term is large, but stay short of nominal;
- the baselines of the paper (DoC, QuAcc, O-LEAP, L-CBPE) do not build their intervals from these parts, so the change is measured, not predicted.

**Coverage per pair and over pairs.** Without replacement, the bags of a pair are calibrated on average over the draws of the pool, but given the pool they share the part $e_P$ of their luck, a share $n_j/P_j$ of its variance. Coverage measured on the $100$ bags of one pair is therefore still partly one test of the pool, as it is one test of $V$ (§T.7); it is an unbiased estimate of nominal coverage only when averaged over pairs with different pools, and per pair it is closer to nominal the smaller $n_j/P_j$ is. Choosing $n\le\min_jP_j/4$ keeps this share below a quarter for every class.

**Two readings of cause 4.** If target bags are fresh samples, as in a deployment, the pool term belongs to the evaluation protocol, which re-draws many large bags from a small pool; the protocol without replacement removes it as a separate error. If the deployment really is many bags from one finite pool, the estimand is the pool's own table, whose uncertainty given $C$ a method cannot know without $P_j$: a reference that adds the pool's sampling error, $d^{\text{pool}}_j\sim\operatorname{Bin}(P_j,r_j)/P_j$ for each draw $r_j$ of a recall, restores coverage in the toy of [[ext2_baycap]] Appendix A.3, but needs $P_j$ and the knowledge that the bags come from a pool, so it is a reference, not a method. A further effect, a new hypothesis not checked: with replacement, the target likelihood of BayCAP treats the $n$ instances of a bag as independent while they are repetitions of $P_j$ instances, so it overstates the information of the bag; [[ext2_baycap]] E2.4 found that the target bag barely moves $C$, so the effect would be mostly on $\pi^Q$.

#### T.11 A bound on the prevalence part

§T.11–§T.15 come from a discussion of what the budget can be used for (2026-11-04). All their results are new hypotheses; they are tested by parts of E4.1, E4.2 and E4.3 (§T.0).

*Intuition.* A wrong prevalence changes the estimated accuracy only because the classes have different recalls: giving too much weight to a class matters only if that class is easier or harder for $h$ than the others. So the damage is at most "how much weight was moved" times "how different the recalls are".

**Claim.** Let $\tilde\pi$ and $\pi^Q$ be two prevalence vectors (non-negative entries summing to $1$) and $\tilde d$ a vector of recalls. Write $\lVert u\rVert_1=\sum_j\lvert u_j\rvert$ for the $L_1$ norm of a vector $u$. Then

$$
\lvert e_1\rvert=\Big\lvert\sum_j(\tilde\pi_j-\pi^Q_j)\tilde d_j\Big\rvert\le\tfrac12\lVert\tilde\pi-\pi^Q\rVert_1\cdot\Big(\max_j\tilde d_j-\min_j\tilde d_j\Big).
$$

**Derivation.**

1. Write $u_j=\tilde\pi_j-\pi^Q_j$. Both vectors sum to $1$, so $\sum_ju_j=1-1=0$.
2. Split the classes into those with $u_j>0$ and those with $u_j<0$. Since the $u_j$ sum to $0$, the positive ones and the negative ones have the same total size: $\sum_{u_j>0}u_j=\sum_{u_j<0}\lvert u_j\rvert$. The two totals add up to $\lVert u\rVert_1$, so each is $\frac12\lVert u\rVert_1$.
3. Then $e_1=\sum_ju_j\tilde d_j=\sum_{u_j>0}u_j\tilde d_j-\sum_{u_j<0}\lvert u_j\rvert\tilde d_j$.
4. In the first sum every $\tilde d_j$ is at most $\max\tilde d$; in the second every $\tilde d_j$ is at least $\min\tilde d$. Replacing them can only increase the first sum and decrease what is subtracted, so $e_1\le\frac12\lVert u\rVert_1\max\tilde d-\frac12\lVert u\rVert_1\min\tilde d$.
5. Exchanging the roles of $\max$ and $\min$ gives $e_1\ge-\frac12\lVert u\rVert_1(\max\tilde d-\min\tilde d)$. Together, the claim.

The bound is tight: equality holds when all the prevalence error moves weight from the class with the lowest recall to the class with the highest. In the binary case $\frac12\lVert u\rVert_1=\lvert\tilde\pi_1-\pi^Q_1\rvert$ and the bound is the identity $e_1=(\tilde\pi_1-\pi^Q_1)(\tilde d_1-\tilde d_0)$ of §T.4 in absolute value, so it is an equality. The same derivation bounds the difference between the accuracies of any two prevalence vectors $\pi,\pi'$ with the same recalls: $\lvert\sum_j(\pi_j-\pi'_j)\tilde d_j\rvert\le\frac12\lVert\pi-\pi'\rVert_1(\max\tilde d-\min\tilde d)$.

**Which methods it applies to.** The derivation uses only two facts: the method's estimate has the form $\sum_j\tilde\pi_j\tilde d_j$, and $\tilde\pi$ is a prevalence vector. So it holds for both families of the paper:

- **SQS-H** with a point estimate $\hat\pi$: its bags have the counts $n_j$ obtained by rounding $n\hat\pi_j$, so $\tilde\pi_j=n_j/n$, a prevalence vector. Its point estimate is the mean accuracy of $m$ simulated bags, whose expectation given $V$ is $\sum_j\tilde\pi_j\tilde d_j$; the bound applies to that expectation. The Monte Carlo deviation of the mean of $m$ bags, of order $\sigma_{3b}/\sqrt m$, is not covered.
- **Methods that average over draws** (RQBS, SQS-B, BayCAP and its variants): each draw $r$ has a prevalence vector $\pi^{(r)}$ and recalls $d^{(r)}$, and $e_1$ is the average over the $m$ draws of $e_1^{(r)}=\sum_j(\pi^{(r)}_j-\pi^Q_j)d^{(r)}_j$. The bound holds for each draw, and the absolute value of an average is at most the average of the absolute values (triangle inequality), so

$$
\lvert e_1\rvert\le\frac1m\sum_{r=1}^m\tfrac12\lVert\pi^{(r)}-\pi^Q\rVert_1\Big(\max_jd^{(r)}_j-\min_jd^{(r)}_j\Big).
$$

  For BayCAP the $d^{(r)}$ are posterior draws of the recalls.
- **The baselines** (DoC, QuAcc, O-LEAP, L-CBPE) produce no $\tilde\pi$ and $\tilde d$, so the bound does not apply.

**What it is for.** It translates the error of the quantifier into an error of accuracy. The quantification literature reports the absolute error $\mathrm{AE}=\lVert\tilde\pi-\pi^Q\rVert_1/K$, so $\lvert e_1\rvert\le\frac K2\,\mathrm{AE}\,(\max\tilde d-\min\tilde d)$; the second factor is computable from $V$ for every (dataset, classifier) pair. It recovers the paper's condition of Eq. 2: if all recalls are equal, prevalence errors cost nothing. Per bag it is not computable without labels, since $\pi^Q$ is unknown, but combined with a bound on $\lVert\tilde\pi-\pi^Q\rVert_1$ it becomes one (§T.15), and applied to two prevalence vectors of a method's set of plausible prevalences it bounds the spread of the accuracy interval along $e_1$.

#### T.12 The shortest interval, and floors on the width

*Intuition.* Even someone who knows $\pi^Q$ and $C$ exactly cannot predict the realised accuracy of a bag exactly, because which instances $h$ gets right is random. An interval shorter than this unavoidable spread must miss more often than it should.

**Step 1: the shortest interval of a normal variable.** Let $X\sim N(\mu,\sigma^2)$, with density $f$. Two properties of $f$ are used: it is symmetric about $\mu$ ($f(\mu+x)=f(\mu-x)$), and it decreases as one moves away from $\mu$ on either side. Fix a length $L$ and consider the intervals $[a,a+L]$. Their probability is $G(a)=\int_a^{a+L}f(x)\,dx$, and by the fundamental theorem of calculus its derivative is $G'(a)=f(a+L)-f(a)$.

- If the interval lies more to the left than the centred one ($a<\mu-L/2$), its right end is closer to $\mu$ than its left end, so $f(a+L)>f(a)$ and $G'(a)>0$: moving it right increases its probability. If it lies more to the right, $G'(a)<0$ and moving it left increases its probability.
- So among intervals of length $L$, the centred one $[\mu-L/2,\mu+L/2]$ has the largest probability. Standardising with $Z=(X-\mu)/\sigma$, a standard normal with distribution function $\Phi$, that probability is $\Phi\big(\tfrac L{2\sigma}\big)-\Phi\big(-\tfrac L{2\sigma}\big)=2\Phi\big(\tfrac L{2\sigma}\big)-1$, using the symmetry $\Phi(-z)=1-\Phi(z)$.
- Now take any interval of length $L$ with probability at least $0.95$. The centred interval of the same length has at least that probability, so $2\Phi(L/(2\sigma))-1\ge0.95$, i.e. $\Phi(L/(2\sigma))\ge0.975$. Since $\Phi$ increases and $\Phi(1.96)=0.975$, this means $L/(2\sigma)\ge1.96$, i.e.

$$
L\ge2\cdot1.96\,\sigma,
$$

  with equality for the centred interval $\mu\pm1.96\,\sigma$. This is the shortest interval with probability $0.95$.

**Step 2: what an oracle knows.** Write $\theta$ for the unknowns $(\pi^Q,C)$. Given $\theta$ and the pool, the realised accuracy is $A(h,U)=\frac1n\sum_jc_j$ with independent $c_j\sim\operatorname{Bin}(n_j,d^{\text{pool}}_j)$ (§T.1): a sum of many independent terms, hence approximately normal by the central limit theorem, with variance $\sigma_{3b}^2=\frac1n\sum_j\pi^Q_jd_j(1-d_j)$ (§T.4). By Step 1, an oracle that knows $\theta$ and $d^{\text{pool}}$, and nothing else about the bag, has a shortest $95\%$ interval of length $2\cdot1.96\,\sigma_{3b}$. An oracle that knows $\theta$ but not the pool faces the variance $\sigma_{3b}^2+\sigma_P^2$ (law of total variance, §T.4), so its shortest interval has length $2\cdot1.96\sqrt{\sigma_{3b}^2+\sigma_P^2}$.

**Step 3: a method knows less about $\theta$, which can only widen.** Let $D$ be the data a method sees ($V$ and the unlabelled bag). The law of total variance (§T.4), applied given $D$ with $\theta$ as the second variable, says

$$
\operatorname{Var}(A\mid D)=\mathbb E\big[\operatorname{Var}(A\mid D,\theta)\mid D\big]+\operatorname{Var}\big(\mathbb E[A\mid D,\theta]\mid D\big)\ge\mathbb E\big[\operatorname{Var}(A\mid D,\theta)\mid D\big],
$$

because a variance is never negative. If $D$ tells nothing about which instances of the bag $h$ gets right beyond what $\theta$ tells, then $\operatorname{Var}(A\mid D,\theta)=\sigma_{3b}^2$ (with the pool known), so $\operatorname{Var}(A\mid D)\ge\sigma_{3b}^2$ and, by Step 1, no such method has a $95\%$ interval shorter than about $2\cdot1.96\,\sigma_{3b}$.

**Step 4: what a method can learn about the bag itself.** Two kinds of data do tell something about which instances $h$ gets right, and lower the floor.

- *The predicted counts of the bag* (used by every quantifier, and by BayCAP's target likelihood). In the binary case, let $c_0$ and $c_1$ be the correct predictions in classes $\omega_0$ and $\omega_1$, with variances $v_j=n_jd_j(1-d_j)$. The number of instances predicted as $\omega_1$ is $c_1+(n_0-c_0)$; given $\theta$, $n_0$ is known, so the predicted counts reveal $S=c_1-c_0$, while $A=(c_0+c_1)/n$. To find the variance of $A$ once $S$ is known, write $A=\beta S+R$ with $\beta=\operatorname{Cov}(A,S)/\operatorname{Var}(S)$. This choice makes $R$ uncorrelated with $S$, because $\operatorname{Cov}(R,S)=\operatorname{Cov}(A,S)-\beta\operatorname{Var}(S)=0$; for jointly normal variables (the normal approximation of the binomials) uncorrelated means independent, so knowing $S$ leaves only the variance of $R$. Since $A=\beta S+R$ with uncorrelated parts, $\operatorname{Var}(A)=\beta^2\operatorname{Var}(S)+\operatorname{Var}(R)$, hence $\operatorname{Var}(A\mid S)=\operatorname{Var}(R)=\operatorname{Var}(A)-\operatorname{Cov}(A,S)^2/\operatorname{Var}(S)$. With $c_0$ and $c_1$ independent: $\operatorname{Var}(A)=(v_0+v_1)/n^2$, $\operatorname{Var}(S)=v_0+v_1$, and $\operatorname{Cov}(A,S)=\operatorname{Cov}(c_0+c_1,c_1-c_0)/n=(v_1-v_0)/n$. So

$$
\operatorname{Var}(A\mid S,\theta)=\frac{(v_0+v_1)^2-(v_1-v_0)^2}{n^2(v_0+v_1)}=\frac{4v_0v_1}{n^2(v_0+v_1)}=\rho\,\sigma_{3b}^2,\qquad\rho=\frac{4v_0v_1}{(v_0+v_1)^2}.
$$

  Since $(v_0+v_1)^2-4v_0v_1=(v_0-v_1)^2\ge0$, $\rho\le1$, with $\rho=1$ when $v_0=v_1$: then $A$ and $S$ are uncorrelated and the predicted counts tell nothing about the accuracy. The floor becomes $2\cdot1.96\sqrt\rho\,\sigma_{3b}$.
- *Instance-level confidences* (used by the baselines DoC, O-LEAP and L-CBPE, through the posteriors of $h$). Let $q_i=P(h\text{ correct}\mid x_i)$ be the probability that $h$ classifies instance $i$ correctly given its features. Given the features of the bag, the $n$ outcomes are independent Bernoulli variables with probabilities $q_i$, so $\operatorname{Var}(A\mid X_U)=\frac1{n^2}\sum_iq_i(1-q_i)$. Within class $\omega_j$ the average of $q$ is $d_j$, and since the variance of $q$ is non-negative, $\mathbb E[q^2]\ge d_j^2$, so $\mathbb E[q(1-q)]=d_j-\mathbb E[q^2]\le d_j-d_j^2=d_j(1-d_j)$. The instance-level floor is therefore at most the class-level one, with equality only if $h$ is equally likely to be right on every instance of a class.

**Step 5: conditional and average coverage.** The floors hold for coverage *conditional on the data*: $95\%$ for every value of $D$. Coverage averaged over many bags can be reached by intervals that are shorter on some bags and longer on others, so the floors constrain the typical width, not the width on every bag.

**What it is for** [new hypothesis]. A label-free check: a population-pushforward BayCAP interval shorter than the plug-in floor $2\cdot1.96\sqrt{\hat\rho}\,\hat\sigma_{3b}$ (§T.14) cannot reach $95\%$ conditional coverage of the realised accuracy. And it places the baselines in the frame: the paper's methods see the bag only through class-level quantities and cannot go below the class-level floor, while confidence-based baselines with well-calibrated confidences can, in principle, have valid intervals that are narrower, above all at small $n$. Testing this needs per-bag summaries of the confidences ($\sum_iq_i(1-q_i)$ with the calibrated posteriors of $h$), which `main` does not store and would have to be added to a re-run.

#### T.13 The variances for the SQS family and for BayCAP, and the bias of BayCAP's prior

*Intuition.* The split of §T.2 is the same for every method, but the size and the mean of its parts depend on how $\tilde\pi$ and $\tilde d$ are obtained: frequencies of $V$ and a surrogate quantifier (SQS family), or a posterior with a prior and $h$ itself as quantifier (BayCAP with joint estimation), or a fixed rate matrix and a separate quantifier (BayCAP with a fixed $\hat C$).

**Which BayCAP variants.** The BayCAP methods of `main` (`src/uncawe/experiments/methods.py`) come from two classes, which obtain $C$ and $\pi^Q$ differently, so the results of this section do not hold for all of them alike:

- the `BayCAP` class (`methods/baycap.py`): a joint posterior of $(\pi^Q,C)$, with $\pi^Q$ estimated from the predictions of $h$ through $C$ (BayesianCC); its variants differ in the prior on $C$, uniform (`BayesCAP`, `BayesCAP-real`), empirical-Bayes (`BayesCAP-EB`, `BayesCAP-EB-real`) or hierarchical (`BayesCAP-H-real`, `BayesCAP-Hf-real`, `BayesCAP-Hfn-real`), and in the pushforward;
- the classes `BayCAPKDEy` with `c_prior="fixed"` and `BayCAPMAPLS` (`methods/baycap_kdey.py`, `methods/baycap_mapls.py`; one class, `BayCAPFixed`, until WP-13; [[ext3_posterior_propagation]] E3.1): a fixed $\hat C$, the frequency matrix of $V$, and the prevalence posterior of a separate Bayesian quantifier (BayesKDEy or BayesianMAPLS, on the outputs of $h$ or of an MLP): `BayCAP-{KDEy,EMQ}-{h,mlp}` and their `-real` versions.

The pushforward (population or realised) changes which parts the interval represents (§T.6), not the error parts of the point estimate, whose expectation is the same for both.

**What is the same for every method.** $e_3^b$ and $e_P$ depend only on the estimand and the protocol (§T.3), so their variances of §T.4 hold for every method.

**SQS family.** $e_2$ uses the frequencies $\tilde d_j=k_j/\lvert V_j\rvert$, with mean $d_j$ and variance $d_j(1-d_j)/\lvert V_j\rvert$ (§T.4). $e_1$ comes from the surrogate's rates $S$ through the quantifier (binary ACC: §T.4).

**BayCAP with joint estimation, the recalls.** With the `"uniform"` prior, each column of $C$ has the prior $\operatorname{Dir}(\mathbf 1_K)$. Two properties are used.

- *Conjugacy.* The class-$\omega_j$ instances of $V$ have predictions with multinomial counts of probabilities $C_{\cdot j}$; a Dirichlet prior times a multinomial likelihood gives a Dirichlet posterior whose parameters are the prior parameters plus the counts. Ignoring the information that the target bag adds (small, by [[ext2_baycap]] E2.4), $C_{\cdot j}\mid V\sim\operatorname{Dir}(\mathbf 1+\text{counts of column }j)$.
- *Aggregation.* If $(X_1,\dots,X_K)\sim\operatorname{Dir}(\alpha_1,\dots,\alpha_K)$, merging components adds their parameters; a two-component Dirichlet is a Beta distribution. Merging all the components but $C_{jj}$:

$$
d_j\mid V\sim\operatorname{Beta}(a_j,b_j),\qquad a_j=k_j+1,\qquad b_j=(\lvert V_j\rvert-k_j)+(K-1):
$$

the $k_j$ correct predictions plus one prior pseudo-count, and the $\lvert V_j\rvert-k_j$ wrong ones plus $K-1$ pseudo-counts.

A $\operatorname{Beta}(a,b)$ has mean $a/(a+b)$ and variance $\mu(1-\mu)/(a+b+1)$, with $\mu$ its mean. Here $a_j+b_j=\lvert V_j\rvert+K$, so

$$
\bar d_j=\mathbb E[d_j\mid V]=\frac{k_j+1}{\lvert V_j\rvert+K},\qquad\operatorname{Var}(d_j\mid V)=\frac{\bar d_j(1-\bar d_j)}{\lvert V_j\rvert+K+1}.
$$

- *Spread.* When $\lvert V_j\rvert\gg K$, $\bar d_j\approx\tilde d_j$ and the posterior variance is about $\tilde d_j(1-\tilde d_j)/\lvert V_j\rvert$, the sampling variance of the SQS family with $\tilde d_j$ in place of $d_j$. The meaning differs: the posterior variance is the uncertainty on $d_j$ given this $V$, the sampling variance the scatter of $\tilde d_j$ over draws of $V$. For large samples the two agree (Bernstein–von Mises theorem), so a posterior interval for $d_j$ covers it in about $95\%$ of the draws of $V$: this is the sense in which BayCAP represents $e_2$ (§T.6). For the empirical-Bayes prior the same steps hold with its pseudo-counts (*Other priors* below); for the hierarchical priors, which are not a fixed Dirichlet, the posterior still concentrates around the frequencies with this spread when $\lvert V_j\rvert$ is large, so the statement holds approximately.
- *Mean: the bias of the prior.* For BayCAP, $e_2$ averaged over the draws is $\sum_j\pi^Q_j(\bar d_j-d_j)$. By linearity of the expectation and $\mathbb E[k_j]=\lvert V_j\rvert d_j$,

$$
\mathbb E[\bar d_j]-d_j=\frac{\lvert V_j\rvert d_j+1}{\lvert V_j\rvert+K}-d_j=\frac{\lvert V_j\rvert d_j+1-\lvert V_j\rvert d_j-Kd_j}{\lvert V_j\rvert+K}=\frac{1-Kd_j}{\lvert V_j\rvert+K}.
$$

  This bias does not average out over draws of $V$, and the posterior does not widen for it. It is small for $K=2$ and grows with $K$ and with small $\lvert V_j\rvert$, as the large-$K$ displacement found in [[ext2_baycap]]. Over draws of $V$, $\bar d_j$ has variance $\operatorname{Var}(k_j)/(\lvert V_j\rvert+K)^2=\lvert V_j\rvert d_j(1-d_j)/(\lvert V_j\rvert+K)^2$, slightly smaller than that of $\tilde d_j$: the prior trades variance for bias. So, for BayCAP, $\mathbb E[e_2]=\sum_j\pi^Q_j(1-Kd_j)/(\lvert V_j\rvert+K)$ and $\operatorname{Var}(e_2)=\sum_j(\pi^Q_j)^2\lvert V_j\rvert d_j(1-d_j)/(\lvert V_j\rvert+K)^2$.
- *Other priors.* With a prior $\operatorname{Dir}(\alpha_{\cdot j})$ on column $j$ (the `"empirical-bayes"` prior of `methods/baycap.py`), the same steps give $\operatorname{Beta}(k_j+\alpha_{jj},\lvert V_j\rvert-k_j+\alpha_{\text{off},j})$ with $\alpha_{\text{off},j}=\sum_{i\ne j}\alpha_{ij}$, and, writing $\alpha_{\cdot j}=\alpha_{jj}+\alpha_{\text{off},j}$, the bias $(\alpha_{jj}-d_j\alpha_{\cdot j})/(\lvert V_j\rvert+\alpha_{\cdot j})$. It vanishes when the prior mean $\alpha_{jj}/\alpha_{\cdot j}$ equals $d_j$, which is what a prior centred on the typical recall aims at. The hierarchical priors learn their parameters and are not covered by this closed form.

**BayCAP with joint estimation, the prevalence part (binary).** For every prior on $C$, BayCAP estimates $\pi^Q$ from the predictions of $h$ itself, so its $e_1$ is that of ACC with $h$ as the surrogate: $S_{10}=C_{10}=1-d_0$ and $S_{11}=C_{11}=d_1$, hence $S_{11}-S_{10}=d_1+d_0-1$. Write $g=(d_1-d_0)/(d_1+d_0-1)$ and $\kappa=g^2$. To first order (§T.4, Property C), and ignoring the priors:

- *From $U$.* The share of the bag predicted as $\omega_1$ is $\hat p=(c_1+n_0-c_0)/n$ and its expectation is $p=(n_1d_1+n_0(1-d_0))/n$, so $\hat p-p=\big[(c_1-n_1d_1)-(c_0-n_0d_0)\big]/n$, with variance $(v_0+v_1)/n^2=\sigma_{3b}^2$. Since $e_1^U=g(\hat p-p)$ (the derivative of ACC with respect to $\hat p$, times $d_1-d_0$),

$$
\operatorname{Var}(e_1^U)\approx\kappa\,\sigma_{3b}^2.
$$

  The same correct and wrong predictions drive $e_3^b=-\big[(c_0-n_0d_0)+(c_1-n_1d_1)\big]/n$, so the two are correlated: $\operatorname{Cov}(e_1^U,e_3^b)=-g\,(v_1-v_0)/n^2$, by the covariance of sums of independent terms. They are uncorrelated only when $v_0=v_1$.
- *From $V$.* Write $\delta_j=\tilde d_j-d_j$; then the error of $\hat C_{10}=1-\tilde d_0$ is $-\delta_0$ and that of $\hat C_{11}$ is $\delta_1$. With the derivatives of ACC of §T.4, $\partial\hat\pi_1/\partial S_{10}=-\pi^Q_0/(d_1+d_0-1)$ and $\partial\hat\pi_1/\partial S_{11}=-\pi^Q_1/(d_1+d_0-1)$, the $V$-part of the prevalence error is $(\pi^Q_0\delta_0-\pi^Q_1\delta_1)/(d_1+d_0-1)$, so $e_1^V=g(\pi^Q_0\delta_0-\pi^Q_1\delta_1)$ and, by Property B,

$$
\operatorname{Var}(e_1^V)\approx\kappa\big[(\pi^Q_0)^2\operatorname{Var}(\delta_0)+(\pi^Q_1)^2\operatorname{Var}(\delta_1)\big]=\kappa\operatorname{Var}(e_2),
$$

  with $\operatorname{Cov}(e_1^V,e_2)=g\big[(\pi^Q_0)^2\operatorname{Var}(\delta_0)-(\pi^Q_1)^2\operatorname{Var}(\delta_1)\big]$, because $e_2=\pi^Q_0\delta_0+\pi^Q_1\delta_1$ uses the same $\delta_j$.

So for BayCAP the prevalence part is the luck of the bag and the error of $V$, scaled by $\kappa$: it vanishes when $d_1=d_0$ and grows without limit as $d_1+d_0\to1$, i.e. as $h$ approaches a random classifier. The within-source correlations of §T.4, which the budget drops, are here structural, because the quantifier and the classifier are the same. These are frequentist variances of BayCAP's estimate to first order; the priors (on $C$ and on $\pi^Q$, whose bias is not derived) and the information of the target bag on $C$ change them, but not the mechanism, so the factor $\kappa$ applies to every variant of the `BayCAP` class.

**BayCAP with a fixed rate matrix (`BayCAPKDEy` with `c_prior="fixed"`, `BayCAPMAPLS`).** Its recalls are the frequencies $\tilde d_j=k_j/\lvert V_j\rvert$ of $V$, used in every draw. So its $e_2$ is that of the SQS family: mean $0$ (no prior, hence no prior bias) and variance $\sum_j(\pi^Q_j)^2d_j(1-d_j)/\lvert V_j\rvert$ (§T.4), and it is not represented, since all draws share $\tilde d$ (§T.6). Its prevalence part comes from the quantifier, not from inverting $C$: its variance is set by the kernel densities of BayesKDEy or by the fixed point of the EM algorithm of BayesianMAPLS, and is not derived here; BayesKDEy keeps its densities fixed, so it does not propagate the error of $V$ into $e_1^V$ (§T.6). With the outputs of an MLP as input, the quantifier's errors are not those of $h$, and the correlations with $e_3^b$ and $e_2$ are the ordinary within-source correlations of §T.4 rather than structural ones.

#### T.14 Label-free plug-ins, self-diagnosis and completed intervals

*Intuition.* The variances of §T.4 and §T.13 contain unknown quantities ($d$, $\pi^Q$). Replacing them by their estimates gives numbers that can be computed without the labels of the target bag, and hence by the methods themselves.

**The plug-in principle.** Replace every unknown in a formula by its estimate. If the estimate approaches the true value as the samples grow and the formula is a continuous function of it, the plug-in approaches the true quantity; by Property C its error is of first order in the estimation errors, a relative error of order $1/\sqrt m$ for samples of size $m$.

**A small bias, and its correction.** The plug-in of $d_j(1-d_j)$ is $\tilde d_j(1-\tilde d_j)$. Its expectation follows from $\mathbb E[\tilde d_j]=d_j$ and $\mathbb E[\tilde d_j^2]=\operatorname{Var}(\tilde d_j)+d_j^2=d_j(1-d_j)/\lvert V_j\rvert+d_j^2$:

$$
\mathbb E\big[\tilde d_j(1-\tilde d_j)\big]=d_j-\frac{d_j(1-d_j)}{\lvert V_j\rvert}-d_j^2=d_j(1-d_j)\Big(1-\frac1{\lvert V_j\rvert}\Big),
$$

so $\frac{\lvert V_j\rvert}{\lvert V_j\rvert-1}\tilde d_j(1-\tilde d_j)$ is unbiased. The correction matters only for small classes.

**Plug-ins for the SQS family.** With $\hat\pi$ the quantifier's estimate in place of $\pi^Q$:

$$
\hat\sigma_{3b}^2=\frac1n\sum_j\hat\pi_j\tilde d_j(1-\tilde d_j),\qquad
\hat\sigma_2^2=\sum_j\hat\pi_j^2\frac{\tilde d_j(1-\tilde d_j)}{\lvert V_j\rvert-1},
$$

and $\hat\sigma_{1U}^2$, $\hat\sigma_{1V}^2$ from the formulas of §T.4 with $\hat S$, $\tilde d$, $\hat\pi$ plugged in (binary ACC), or from a bootstrap of $U$ and of $V$. $\hat\pi$ is itself noisy, which adds a first-order error to every plug-in; the error is small where the prevalence term is small (§T.11).

**Plug-ins for BayCAP with joint estimation.** The plug-ins become averages over the posterior draws, which also account for the uncertainty on $(\pi^Q,C)$: $\hat\sigma_{3b}^2=\frac1m\sum_r\frac1n\sum_j\pi^{(r)}_jd^{(r)}_j(1-d^{(r)}_j)$, which is the variance that the realised-table predictive adds by sampling a table for each draw. The spread along $e_2$ is already in the posterior (§T.13); its prior bias is not, but its plug-in is simple: $\frac{1-K\tilde d_j}{\lvert V_j\rvert+K}=\frac{k_j+1}{\lvert V_j\rvert+K}-\frac{k_j}{\lvert V_j\rvert}=\bar d_j-\tilde d_j$ (to check, bring both fractions to the denominator $\lvert V_j\rvert(\lvert V_j\rvert+K)$: the numerator is $\lvert V_j\rvert(k_j+1)-k_j(\lvert V_j\rvert+K)=\lvert V_j\rvert-Kk_j$). The estimated bias is the gap between the posterior mean and the frequency of $V$. The same identity holds for any Dirichlet prior on the column, hence for the empirical-Bayes prior: $\frac{k_j+\alpha_{jj}}{\lvert V_j\rvert+\alpha_{\cdot j}}-\frac{k_j}{\lvert V_j\rvert}=\frac{\alpha_{jj}-\tilde d_j\alpha_{\cdot j}}{\lvert V_j\rvert+\alpha_{\cdot j}}$, the plug-in of its bias; for the hierarchical priors, $\bar d_j-\tilde d_j$ is used as an estimate without a closed form behind it. For $e_1$, the binary formulas of §T.13 with $\bar d$ plugged in.

**Plug-ins for BayCAP with a fixed $\hat C$.** $\hat\sigma_{3b}^2$ as an average over the prevalence draws with the fixed $\tilde d$; $\hat\sigma_2^2$ as for the SQS family, since $e_2$ is not represented and has no prior bias; no plug-in for $e_1^V$, whose formula depends on the quantifier and is not derived.

**Self-diagnosis: each method's coverage, without labels.** A method's interval gives its own $\sigma_{\text{in}}\approx$ half-width$/1.96$ (§T.7). The missing parts that have a plug-in give an estimate $\hat\eta=\hat\sigma^2_{\text{missing}}/\sigma_{\text{in}}^2$, and the coverage rule of §T.7 a predicted coverage $2\Phi(1.96/\sqrt{1+\hat\eta})-1$. Per method: SQS-H misses $e_1^U$, $e_1^V$, $e_2$, all with plug-ins; RQBS misses $e_1^V$ and $e_2$; the population-pushforward BayCAP with joint estimation misses $e_3^b$, and its realised-table version none of the estimable parts; the variants with a fixed $\hat C$ also miss $e_2$ (and $e_1^V$ with BayesKDEy), so their realised versions miss $e_2$ and their population versions $e_2$ and $e_3^b$. The parts without a plug-in, $e_P$ (it needs $P_j$ and the knowledge of the pool) and $b_q^2$ (a bias that variances cannot reveal), are non-negative additions to $\sigma^2_{\text{out}}$; since the coverage rule decreases as $\eta$ grows, the predicted coverage is then an *upper bound* on the coverage, to first order and as long as the within-source correlations, which can have either sign, are negligible. For BayCAP with joint estimation of $(\pi^Q,C)$ they are not (§T.13), and the prediction must include them.

**Break-even bag size.** SQS-H represents $e_3^b$ and misses $e_2$, so it is reliable only while $\hat\sigma_2^2$ is small compared with $\hat\sigma_{3b}^2$. Setting the two plug-ins equal and solving for $n$:

$$
\frac1n\sum_j\hat\pi_j\tilde d_j(1-\tilde d_j)=\sum_j\hat\pi_j^2\frac{\tilde d_j(1-\tilde d_j)}{\lvert V_j\rvert}
\quad\Longrightarrow\quad
n^*=\frac{\sum_j\hat\pi_j\tilde d_j(1-\tilde d_j)}{\sum_j\hat\pi_j^2\tilde d_j(1-\tilde d_j)/\lvert V_j\rvert}.
$$

This is the bag size at which the design ratio $r$ of §T.7 equals $1$, estimated without labels; at $n=n^*$ the oracle-$\pi$ SQS-H has coverage about $2\Phi(1.96/\sqrt2)-1\approx0.83$. It gives a practical rule: SQS-type intervals are not to be trusted for bags much larger than $n^*$.

**Budget-completed intervals.** The same plug-ins correct an interval for the parts it misses: $\hat A\pm1.96\sqrt{\sigma_{\text{in}}^2+\hat\sigma^2_{\text{missing}}}$, for example RQBS widened by $\hat\sigma_{1V}^2+\hat\sigma_2^2$, a closed-form approximation of SQS-H with $U$ and $V$ resampling (§T.8), and the population-pushforward BayCAP widened by $\hat\sigma_{3b}^2$, a closed-form approximation of the realised-table BayCAP. For BayCAP the centre can also be moved by the estimated prior bias, $-\sum_j\hat\pi_j(\bar d_j-\tilde d_j)$. These corrections are post-processing of stored outputs. They do not represent $e_P$ and $b_q$, so under the paper's protocol they should still under-cover on small pools, which is itself a test of the pool term.

#### T.15 Intervals with a finite-sample guarantee

*Intuition.* The normal approximations of §T.7 and §T.12 hold for large samples. Probability inequalities give intervals whose coverage is guaranteed for every sample size and without assuming normality, at the price of a larger width.

**Tools.**

- *Markov's inequality.* For $X\ge0$ and $t>0$, $P(X\ge t)\le\mathbb E[X]/t$. Derivation: let $\mathbb 1\{X\ge t\}$ be $1$ when $X\ge t$ and $0$ otherwise; then $X\ge t\,\mathbb 1\{X\ge t\}$ always (if $X\ge t$ the right side is $t\le X$, otherwise it is $0\le X$), and taking expectations gives $\mathbb E[X]\ge t\,P(X\ge t)$.
- *Chebyshev's inequality.* Markov applied to $(X-\mu)^2$ and $t^2$: $P(\lvert X-\mu\rvert\ge t)=P((X-\mu)^2\ge t^2)\le\operatorname{Var}(X)/t^2$.
- *Hoeffding's inequality* (stated, not derived). If $\bar X$ is the average of $m$ independent variables with values in $[0,1]$, $P(\lvert\bar X-\mathbb E\bar X\rvert\ge t)\le2e^{-2mt^2}$. It is much tighter than Chebyshev for small probabilities, because it uses that the variables are bounded, not only their variance; it allows the variables to have different means. Setting the right side equal to a level $\alpha'$ and solving for $t$: $\lvert\bar X-\mathbb E\bar X\rvert\le t(m,\alpha')=\sqrt{\ln(2/\alpha')/(2m)}$ with probability at least $1-\alpha'$.
- *Union bound.* The probability that at least one of several events happens is at most the sum of their probabilities. So if each of several bounds fails with probability at most $\alpha_k$, all of them hold together with probability at least $1-\sum_k\alpha_k$, whatever the dependence between them.

**Each part, binary case, SQS-H with ACC.** Take the closed-form estimate $\hat A=\sum_j\tilde\pi_j\tilde d_j$ (no Monte Carlo bags) and split the level $\alpha$ into $\alpha_3+\alpha_2+\alpha_1$.

- *$e_3^b$.* Given $\theta$ and the pool, $A(h,U)$ is the average of $n$ independent Bernoulli variables (§T.1), so $\lvert e_3^b\rvert\le t(n,\alpha_3)$.
- *$e_2$.* For each class, $\tilde d_j$ is the average of $\lvert V_j\rvert$ independent Bernoulli variables, so $\lvert\delta_j\rvert\le t(\lvert V_j\rvert,\alpha_2/2)$. A weighted average of numbers is at most the largest of them, so $\lvert e_2\rvert\le\sum_j\pi^Q_j\lvert\delta_j\rvert\le\max_jt(\lvert V_j\rvert,\alpha_2/2)$, without knowing $\pi^Q$.
- *$e_1$.* ACC has an exact error formula. By the law of total probability the expected share predicted as $\omega_1$ is $p=\pi^Q_0S_{10}+\pi^Q_1S_{11}$. Write $\varepsilon_{1j}=\hat S_{1j}-S_{1j}$. Then the numerator of ACC minus $\pi^Q_1$ times its denominator is $\hat p-\hat S_{10}-\pi^Q_1(\hat S_{11}-\hat S_{10})=(\hat p-p)+\big(p-\pi^Q_0\hat S_{10}-\pi^Q_1\hat S_{11}\big)=(\hat p-p)-\pi^Q_0\varepsilon_{10}-\pi^Q_1\varepsilon_{11}$, using $\pi^Q_0+\pi^Q_1=1$. Dividing by the denominator,

$$
\hat\pi_1-\pi^Q_1=\frac{(\hat p-p)-\pi^Q_0\varepsilon_{10}-\pi^Q_1\varepsilon_{11}}{\hat S_{11}-\hat S_{10}},
$$

  exactly, with an observed denominator. $\hat p$ is an average of $n$ independent Bernoulli variables, and $\hat S_{1j}$ of $\lvert V_j\rvert$, so with Hoeffding at levels $\alpha_1/3$ each, and the weighted-average argument above, $\lvert\hat\pi_1-\pi^Q_1\rvert\le B_\pi=\big[t(n,\alpha_1/3)+\max_jt(\lvert V_j\rvert,\alpha_1/3)\big]/\lvert\hat S_{11}-\hat S_{10}\rvert$. Quantifiers clip $\hat\pi_1$ to $[0,1]$; since $\pi^Q_1\in[0,1]$, clipping can only move $\hat\pi_1$ closer to it, so the bound still holds. Then $\lvert e_1\rvert\le B_\pi\lvert\tilde d_1-\tilde d_0\rvert$ by the binary case of §T.11, which is exact and includes the cross term.
- *Together.* By the triangle inequality, $\lvert\hat A-A\rvert\le\lvert e_1\rvert+\lvert e_2\rvert+\lvert e_3^b\rvert$, and by the union bound all the bounds hold together with probability at least $1-\alpha$. So the interval $\hat A\pm\big(B_\pi\lvert\tilde d_1-\tilde d_0\rvert+\max_jt(\lvert V_j\rvert,\alpha_2/2)+t(n,\alpha_3)\big)$ covers the realised accuracy with probability at least $1-\alpha$, for every $n$ and $\lvert V_j\rvert$, without normality, and including the quantifier's bias, since the error formula of ACC is exact under PPS.

**Where it holds.** The derivation assumes that the bag's instances are independent draws with the population rates. That is true without replacement (§T.10, where the bag is an i.i.d. sample of the population). Under the paper's protocol the bag is drawn from the pool, so $A$ and $\hat p$ fluctuate around the pool's rates: the guarantee then needs two more terms of the same form with $P_j$ (one for $e_P$, one for the surrogate's rates on the pool), which require knowing the pool, as in §T.10.

**The price of the guarantee.** For one part, the ratio of the Hoeffding half-width to the normal one is $t(m,\alpha')/(1.96\sqrt{p(1-p)/m})=\sqrt{\ln(2/\alpha')/2}\,/\,\big(1.96\sqrt{p(1-p)}\big)$: it does not depend on $m$, it is larger than $1$ because Hoeffding assumes the worst variance $1/4$ and $\alpha'<\alpha$, and it grows as $p$ approaches $0$ or $1$. Adding the bounds of the parts, rather than adding their variances, widens further. The guaranteed interval is therefore a conservative reference: any method whose interval is wider is wider than needed for $1-\alpha$ coverage, and the gap between a method's width and the guaranteed one measures how much its coverage relies on the normal approximation. In the multiclass case the error formula involves the inverse of $\hat S$, and the bound on $\lVert\hat\pi-\pi^Q\rVert_1$ grows with its norm, so the guarantee becomes loose for ill-conditioned surrogates.

### Implementation

Components (for E4.1a, E4.2a and E4.2b):

- `experiments/sampling.py`: `fixed_prev_grid(pis, n_U_grid, n_repeats)` and APP at any $n$; per-bag seeds from `(seed, dataset, bag_id, n_U)`;
- bag size taken from the target: `RQBS` and `PrediQuant` are built with `sample_size=SAMPLE_SIZE=1000` in `src/uncawe/experiments/methods.py`; the study passes $n$ explicitly or uses `n_U = len(X)`;
- nested, seeded stratified subsampling of $V$; the quantifier is refitted on each subsample, $h$ is not;
- error split per bag (`evaluation/budget.py`): $e_1^U$, $e_1^V$, $e_2+e_P$, $e_3^b$ from the method's $\tilde\pi$ and $\tilde d$, the bag's true prevalence, $d^{\text{pool}}$ and the large-bag estimate of $\pi^\infty(V)$; the pool sizes $P_j$ of each dataset;
- `SQSHResampled` [new]: SQS-H with $U$ and $V$ resampling (§T.8), following the method conventions of [[work_plan]] §V.2;
- oracle classes `OracleSQS` (true $\pi^Q$, validation bags) and `OracleC` (true $\pi^Q$, fixed $C^h$), reported as references, plus scoring against the population table.

Tests:

- the error split satisfies $e_1+e_2+e_P+e_3^b=\hat A-A(h,U)$ exactly on random tables, with $d$ given;
- the closed-form variances of §T.4 match simulated variances of $e_3^b$, $e_P$, $e_2$, $e_1^U$, $e_1^V$ for a binary ACC configuration with a finite pool;
- with $\tilde\pi=\pi^Q$ and $\tilde d=d^{\text{pool}}$, $e_1=e_2+e_P=0$;
- `fixed_prev_grid` returns bags with the requested class counts and size;
- nested subsamples satisfy $V_{1/16}\subset V_{1/4}\subset V$ and are stratified;
- `SQSHResampled` with resampling of $V$ switched off reproduces RQBS for the same seed.

Reproducibility and comparability:

- new study `target_size`, writing to `cap.env["OUT_DIR"]/uncawe/target_size`; `main`, `binform`, `predist` and their outputs are unchanged;
- every row records $n$, $\lvert V\rvert$ level, $V$ seed, prevalence type, bag seed, `config_hash`, `schema_version`;
- the cell ($\lvert U\rvert=1000$, full $V$, seed of `main`) must reproduce `main` within Monte Carlo error for the methods present in both;
- outer seeds that change the $L/V$ split retrain $h$; these rows are not comparable with `main` and are flagged.

#### A. Toy simulation

`uv run python scripts/toys/ext4_budget.py` (defaults: $40$ validation sets, $50$ bags per validation set, $m=1000$ samples per interval, seed $0$; about $3$ minutes). Binary, toy only: $\pi^Q_1=0.5$ with fixed counts, $n_s$ validation instances per class, quantifier ACC on a surrogate classifier with rates $P(\hat Y_s=\omega_1\mid Y=\omega_0)=0.25$, $P(\hat Y_s=\omega_1\mid Y=\omega_1)=0.75$, conditionally independent of $h$ given $Y$ (a simplification). The bags are drawn from the population, not from a finite pool, so the toy has no pool term $e_P$ and $e_3=e_3^b$; the pool term is simulated by the toy of [[ext2_baycap]] Appendix A.3. Two target classifiers: "flat", $d=(0.90,0.70)$, as in [[ext2_baycap]], and "steep", $d=(0.95,0.55)$, for which the prevalence term is larger. BayCAP is computed exactly on a grid, as in `ext2_estimand_gap.py`. "Oracle $\pi$" uses the true counts and is a reference. Coverage of the $95\%$ interval against the realised accuracy, $2000$ bags per row; "closed form" is $2\Phi(1.96/\sqrt{1+r})-1$ (§T.7), the prediction for the oracle-$\pi$ column, with $r=\operatorname{Var}(e_2)/\operatorname{Var}(e_3)$ computed from $d$, $\pi^Q$, $n$, $n_s$ (here $r=0.5\,n/n_s$ for both classifiers); the Monte Carlo error is about $\pm0.01$ for methods whose coverage does not depend on the validation set, and larger for the SQS family in the rows with large $r$, where its coverage varies strongly between validation sets (the script prints the 10th–90th percentile).

Flat $d=(0.90,0.70)$:

| $n_s$ | $n$ | $r$ | SQS-H | RQBS-like | SQS-H, $U$+$V$ | BayCAP | realised-table BayCAP | oracle $\pi$ | closed form |
|---|---|---|---|---|---|---|---|---|---|
| $100$ | $100$ | $0.5$ | $0.85$ | $0.89$ | $0.95$ | $0.78$ | $0.95$ | $0.90$ | $0.89$ |
| $100$ | $300$ | $1.5$ | $0.75$ | $0.81$ | $0.96$ | $0.86$ | $0.94$ | $0.77$ | $0.78$ |
| $100$ | $1000$ | $5$ | $0.43$ | $0.49$ | $0.94$ | $0.90$ | $0.92$ | $0.50$ | $0.58$ |
| $100$ | $3000$ | $15$ | $0.33$ | $0.38$ | $0.93$ | $0.87$ | $0.88$ | $0.34$ | $0.38$ |
| $1000$ | $100$ | $0.05$ | $0.94$ | $0.97$ | $0.97$ | $0.59$ | $0.97$ | $0.96$ | $0.94$ |
| $1000$ | $300$ | $0.15$ | $0.90$ | $0.94$ | $0.96$ | $0.65$ | $0.95$ | $0.94$ | $0.93$ |
| $1000$ | $1000$ | $0.5$ | $0.84$ | $0.88$ | $0.94$ | $0.78$ | $0.94$ | $0.88$ | $0.89$ |
| $1000$ | $3000$ | $1.5$ | $0.72$ | $0.79$ | $0.96$ | $0.90$ | $0.96$ | $0.80$ | $0.78$ |

Steep $d=(0.95,0.55)$:

| $n_s$ | $n$ | $r$ | SQS-H | RQBS-like | SQS-H, $U$+$V$ | BayCAP | realised-table BayCAP | oracle $\pi$ | closed form |
|---|---|---|---|---|---|---|---|---|---|
| $100$ | $100$ | $0.5$ | $0.79$ | $0.93$ | $0.97$ | $0.84$ | $0.96$ | $0.90$ | $0.89$ |
| $100$ | $300$ | $1.5$ | $0.76$ | $0.89$ | $0.98$ | $0.91$ | $0.96$ | $0.83$ | $0.78$ |
| $100$ | $1000$ | $5$ | $0.37$ | $0.54$ | $0.95$ | $0.90$ | $0.92$ | $0.54$ | $0.58$ |
| $100$ | $3000$ | $15$ | $0.21$ | $0.32$ | $0.95$ | $0.94$ | $0.95$ | $0.45$ | $0.38$ |
| $1000$ | $100$ | $0.05$ | $0.87$ | $0.97$ | $0.97$ | $0.75$ | $0.96$ | $0.96$ | $0.94$ |
| $1000$ | $300$ | $0.15$ | $0.85$ | $0.96$ | $0.97$ | $0.81$ | $0.96$ | $0.94$ | $0.93$ |
| $1000$ | $1000$ | $0.5$ | $0.78$ | $0.93$ | $0.97$ | $0.86$ | $0.96$ | $0.91$ | $0.89$ |
| $1000$ | $3000$ | $1.5$ | $0.68$ | $0.85$ | $0.96$ | $0.88$ | $0.93$ | $0.78$ | $0.78$ |

Root mean square of the error parts of the SQS-H point estimate (flat $d$, toy):

| $n_s$ | $n$ | $e_1$ | $e_2$ | $e_3$ |
|---|---|---|---|---|
| $100$ | $100$ | $0.023$ | $0.032$ | $0.037$ |
| $100$ | $3000$ | $0.017$ | $0.031$ | $0.007$ |
| $1000$ | $100$ | $0.019$ | $0.008$ | $0.038$ |
| $1000$ | $3000$ | $0.005$ | $0.008$ | $0.007$ |

What the toy shows:

- the coverage rule of §T.7 predicts the oracle-$\pi$ coverage within $0.04$ for $r\le1.5$ and within about $0.08$ at $r\in\{5,15\}$, where coverage depends strongly on which of the $40$ validation sets is drawn (the normal approximation for $e_2$ is also coarsest there);
- $e_3$ shrinks as $n^{-1/2}$ and $e_2$ is constant in $n$ and shrinks with $n_s$; at $n_s=100$, $e_1$ also keeps a floor (about $0.015$) from fitting the quantifier on $V$ (H3 of E4.1a);
- SQS-H, the RQBS-like variant and even the oracle-$\pi$ reference lose coverage as $r$ grows (down to $0.21$–$0.45$ at $r=15$): the missing term is $e_2$, which no SQS variant without $V$ resampling represents (H1 of E4.2a). Between validation sets, the coverage of SQS-H at $n_s=100$, $n=3000$ ranges from $0.00$ to $0.86$ (10th–90th percentile), i.e. it measures the validation set;
- with steep $d$, SQS-H under-covers already at small $r$ ($0.78$–$0.87$ for $r\le0.5$), while the RQBS-like variant does not: the bootstrap of $\hat\pi$ supplies the prevalence term;
- BayCAP with the population pushforward is lowest at small $r$ ($0.59$ at $r=0.05$) and improves as $r$ grows (H2 of E4.2a);
- the realised-table BayCAP and SQS-H with $U$ and $V$ resampling stay between $0.88$ and $0.98$ in every row (H4 of E4.2a); the realised-table BayCAP dips to $0.88$–$0.92$ at $n_s=100$ and large $n$ with flat $d$, a residual not investigated here;
- the paper's SQS-H runs at $n=1000$; in the toy, the regime of $r$ where SQS-H covers well ($r\lesssim0.2$) needs about five validation instances per class for every target instance of that class.

What is not known: whether real quantifiers (PACC with a cross-validated surrogate), multiclass problems and real validation sizes reproduce these trends, how large the quantifier bias $b_q$ is, and how much the pool term, absent from the toy, adds on the datasets of `main`. The paper's datasets have $\lvert V\rvert$ equal to $35\%$ of the dataset and pools of $30\%$, so $r$ and $n/P_j$ at $\lvert U\rvert=1000$ vary widely across them; E4.2a measures where each dataset falls, and E4.c1 the pool term.

## 11. Material set aside from Extension 6 (earlier version)

On 2026-11-09 the user restructured [[ext6_sqs_s_design|Extension 6]] into its own document, kept thin: one experiment on the distances (E6.1) and one on the allocation $\rho$ (E6.2), with cross-fitting, the sensitivity to $\gamma$ and kernel weights as candidates. Set aside: the adaptive bandwidth by local density, the Aitchison distance (see also §3), the secondary factorial study of $\rho$, $\lvert V\rvert$ and $\lvert U\rvert$, the allocation of $V$ across classes (Neyman allocation), the label-free selection of $\rho$, and, among the measures of the allocation study, the proximity in estimated-prevalence space and the training and inference cost. The earlier text of the extension (its §6 in [[extensions]], at commit `3f080b2`) is kept below verbatim; it predates FIX-1, so its remark that `PrediQuant` fits the quantifier on all of $V$ no longer holds.

Merges the former Extensions 4 and 6 (2026-10-27), which both study how SQS-S uses the validation set: the first through the distance and neighbourhood that select validation bags, the second through the split of $V$ between the quantifier and the bags. Each part keeps its own primary experiment, so that the effects remain identifiable; a joint selection of geometry, neighbourhood rule and allocation $\rho$ can follow.

#### 6.1 Geometric analysis and kernel variants

SQS-S selects validation bags by comparing estimated prevalences through a distance on the simplex. One could systematically study how the distance and neighbourhood definition affect bias and variance:

- compare $L_1$, Hellinger, Jensen--Shannon, and Aitchison distances;
- measure the correspondence between proximity in predicted and true prevalence space;
- study sensitivity to $\gamma$;
- replace hard neighbour selection with kernel weights;
- select bandwidth adaptively according to local density.

This interpretation connects SQS-S to local regression or approximate Bayesian computation on the simplex. The contribution becomes substantial if it yields an adaptive rule, rather than only a comparison of distance functions.

#### 6.2 Validation-budget allocation

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

This is a validation-budget allocation study, not part of Extension 4, whose E4.2 subsamples $|V|$ at fixed $\rho$. A secondary factorial analysis may later test interactions between $\rho$, $|V|$ and $|U|$, but the primary experiment should vary only $\rho$ so that the effects remain identifiable.

**To explore: allocation of $V$ across classes** (reminder from the variance model of [[ext4_uncertainty_budget|Extension 4]], now set aside in [[optional_extensions]] §10; new hypothesis). Besides the split $\rho$, the class composition of $V$ affects the error of the rates: minimising $\operatorname{Var}(e_2)=\sum_j\pi_j^2d_j(1-d_j)/\lvert V_j\rvert$ subject to $\sum_j\lvert V_j\rvert=N$ gives the Neyman allocation $\lvert V_j\rvert\propto\pi_j\sqrt{d_j(1-d_j)}$, and, averaged over APP prevalences drawn uniformly on the simplex (for which $\mathbb E[\pi_j^2]$ is the same for all classes), $\lvert V_j\rvert\propto\sqrt{d_j(1-d_j)}$. Whether a stratified $V$ with this composition improves the SQS and BayCAP intervals, and how it interacts with $\rho$, is to be studied here.

## 12. Material set aside from Extension 5 (earlier version)

On 2026-11-10 the user restructured [[ext5_baycap_priors_tempering|Extension 5]] into the experiment-driven template, kept thin: it now studies only the two methods kept by Extensions 2 and 3 (BayCAP-Hf and BayCAP-KDEy with an MLP input and a fixed $\hat C$), with one experiment on the knowledge of the prior on $\pi^Q$ (E5.1, the former minimal check of E5.4) and one on tempering (E5.2, the former E5.1 and E5.2, BayCAP-Hf tempered jointly). Set aside: the prior on $C$ (former §2.4, H7, E5.3: $\operatorname{Dir}(1/K)$, the diagonal-aware prior, the error-minimising weight; the priors on $C$ are studied in [[ext2_baycap|Extension 2]] E2.4–E2.9), the comparison of tempering with the realised-table predictive (former H5; the predictive is now part of both methods), the amplification of the bias of the prior on $C$ by tempering (former H8), the full grid of priors on $\pi^Q$ (former E5.4, full study, and H1), the prior $\times$ temperature hypothesis (former H4, now the candidate E5.c3), the label-free temperature selection on held-out validation bags (former E5.5, now part of the candidate E5.c2), the contributions, the paper outline and the figure plan. The earlier text (at commit `3d56aa9`) is kept below verbatim, with its headings demoted; its experiment numbers and hypothesis IDs are those of the earlier version (mapping in the Status of [[ext5_baycap_priors_tempering|Extension 5]]).

#### 0. Summary

**Question.** How should BayCAP's priors (on $\pi^Q$ and on the columns of $C$) and its temperature $\tau$ be chosen, and can a choice of prior or temperature calibrate BayCAP as well as the realised-table predictive of [[ext2_baycap|Extension 2]]? The paper names informative priors and posterior tempering as unexplored features of BayCAP; this extension studies both.

**Main claims (new hypotheses).**

1. *Prior on $\pi^Q$.* Under the paper's protocol the uniform prior on $\pi^Q$ is already well specified; informative priors help only under a non-uniform bag-generating distribution, and only when the prior weight is not negligible (small $\lvert U\rvert$, ill-conditioned $C$). Supported only by a toy simulation.
2. *Prior on $C$.* The uniform prior $\operatorname{Dir}(\mathbf 1)$ on each column of $C$ is not uninformative: its weight, $K$ imaginary validation instances per class, grows with the number of classes and pulls every recall towards $1/K$, which biases BayCAP on large-$K$ datasets (source (iv) of [[ext2_baycap|Extension 2]] §3). A symmetric Dirichlet prior whose total weight does not grow with $K$, such as Perks' $\operatorname{Dir}(1/K)$, or a prior that treats the diagonal cell separately, removes most of the bias. Supported by a derivation; no simulation yet.
3. *Tempering.* Tempering can compensate the estimand gap only indirectly, by widening the posterior; it cannot move a displaced interval, and with $\tau>1$ it increases the weight of the prior on $C$ and therefore its bias.

**Candidate methods.** Tempered BayCAP (both likelihood terms divided by $\tau$), evaluated first as a diagnostic: a sweep over $\tau$ with target labels gives the oracle-best $\tau$, an upper bound on what tempering can achieve, compared with the realised-table predictive. BayCAP with a prior on $C$ of total weight independent of $K$. Informative priors on $\pi^Q$. Label-free selection of $\tau$, run only if oracle-best tempering clearly beats the predictive.

**Status.**

| Item | Status |
|---|---|
| `cap-ml` `ACC_BayesCAP(prior=...)`: Dirichlet prior on $\pi^Q$, prior on $C$ fixed to $\operatorname{Dir}(\mathbf 1)$, no temperature | [implemented] |
| `methods/baycap.py:BayCAP`: same priors, `estimand` $\in\{$population, realised$\}$ ([[ext2_baycap|Extension 2]] §8) | [implemented, not evaluated] |
| `quapy` `model_bayesianCC(..., temperature, alpha)`: BayesianCC with both log-likelihood terms divided by $\tau$ | [implemented, external] |
| `quapy` `calibrate_temperature`: selects $\tau$ on **prevalence** regions, not on performance intervals | [implemented, external] |
| `uncawe` `binform` experiment: uniform vs one informative prior on $\pi^Q$, `connect-4` only, $\tau=1$ | [implemented] |
| Toy simulation of §2.3 (`scripts/toys/ext5_prior_calibration.py`) | [implemented] |
| Toy simulation of the prior on $C$ (§2.4) | [proposal] |
| `temperature` and `c_prior` arguments of `BayCAP`, prior registry | [proposal] |
| E5.1–E5.5 | [proposal] |
| All hypotheses (H1, H4, H5, H7, H8) | [new hypothesis] |

Hypothesis IDs continue those of the former joint note (H0, H2, H6 are in [[ext2_baycap|Extension 2]]); IDs are not reused.

**Reader's guide.** Part I explains the priors and the temperature of BayCAP, the prior weight, when a prior is "correct", and why the uniform prior on $C$ is informative. Part II states the contributions, hypotheses, and paper outline. Part III specifies the experiments and the components. The estimands, the sources of miscalibration and the realised-table predictive are in [[ext2_baycap|Extension 2]].

### Part I: Understanding

#### 1. BayCAP's priors and temperature in brief

BayCAP and its notation are introduced in [[ext2_baycap|Extension 2]] §1: the posterior over $(\pi^Q,C)$ combines the validation likelihood $\mathcal L_V(C)$, the target likelihood $\mathcal L_U(\pi^Q,C)$, and the priors $\pi^Q\sim\operatorname{Dir}(\alpha)$ and $C_{:j}\sim\operatorname{Dir}(\mathbf 1)$ for every true class $\omega_j$.

We write the prior on $\pi^Q$ as $\alpha=c\,\mu$, with $\mu\in\Delta^{K-1}$ the prior mean and $c>0$ the concentration (total pseudo-counts). The uniform prior is $\mu=\mathbf 1/K$, $c=K$. The prior on each column of $C$ is written $\operatorname{Dir}(\beta_{:j})$, with total weight $w_j=\sum_i\beta_{ij}$; the implementations fix $\beta_{ij}=1$, so $w_j=K$.

Every interval in this note is computed from BayCAP's posterior, with either the population pushforward (target: the population table $C\operatorname{diag}(\pi^Q)$) or the realised-table predictive (target: the realised table of $U$), and is always scored against the realised table (`true_cts`), as in `main`.

*Implementation note:* `prior` in `ACC_BayesCAP` and `BayCAP` sets $\alpha$ only; the prior on $C$ is hard-coded (`jnp.ones(K)` in `BayCAP.model`, `src/uncawe/methods/baycap.py`), so studying it needs a new argument (§8).

#### 2. Key concepts

##### 2.1 Tempering

The tempered posterior is

$$
p_\tau(\pi^Q,C\mid V,U)\;\propto\;\left[\mathcal L_V(C)\,\mathcal L_U(\pi^Q,C)\right]^{1/\tau}\,p(\pi^Q)\,p(C),
$$

the scheme `quapy` uses for BayesianCC. We write the temperature as $\tau$ because $T$ denotes contingency tables in the reference paper; in the code it is the `temperature` parameter. A value $\tau>1$ flattens the posterior and $\tau<1$ sharpens it.

*Intuition.* Apart from the multinomial coefficients, raising a multinomial likelihood to the power $1/\tau$ is the same as dividing its counts by $\tau$. Tempering therefore acts like pretending that $V$ and $U$ are $\tau$ times smaller. The prior is not tempered, so raising $\tau$ also gives the prior more relative influence, on $\pi^Q$ and on $C$ alike.

How tempering relates to the sources of miscalibration of [[ext2_baycap|Extension 2]] §3: the estimand gap (ii) calls for $\tau>1$, the fixed-count likelihood (i) for $\tau<1$; neither direction moves the centre of a displaced interval (iv).

##### 2.2 Prior weight

*Intuition.* The posterior combines pseudo-counts from the prior with counts from the data; the prior matters only when its pseudo-counts are not negligible.

In the simplest case, a perfect and known classifier ($C=I$), the tempered target likelihood is conjugate to the Dirichlet prior:

$$
\pi^Q\mid U\;\sim\;\operatorname{Dir}\!\left(c\,\mu+\frac{n^t}{\tau}\right),
\qquad
\mathbb E[\pi^Q\mid U]=w\,\mu+(1-w)\,\frac{n^t}{N^Q},
\qquad
w=\frac{c}{c+N^Q/\tau}.
$$

The **prior weight** $w$ is the share of the posterior mean determined by the prior. With $c=10$:

| $N^Q$ | $\tau=1$ | $\tau=4$ |
|---|---|---|
| $1000$ | $w\approx0.010$ | $w\approx0.038$ |
| $100$ | $w\approx0.091$ | $w\approx0.29$ |

For a real classifier, $C$ mixes the classes, so the predicted counts carry less information about $\pi^Q$ than $N^Q$ true labels would; the actual prior weight is larger than $w$, especially along directions that an ill-conditioned $C$ separates poorly. $w$ is a lower bound and a rule of thumb: the prior matters only when $N^Q/\tau$ is comparable to $c$.

##### 2.3 Well-specified prior and calibration on average

*Intuition.* A prior is "correct" when it describes how often each target situation actually occurs; then, and only then, Bayesian intervals are calibrated on average over those situations.

In every experiment, bag prevalences are produced by a known mechanism: $\pi^Q_i\sim G$, then a bag $U_i$ with that prevalence is sampled. A prior is **well specified** when $p(\pi^Q)=G$. Under the paper's protocol (APP), $G=\operatorname{Dir}(\mathbf 1)$, which is exactly BayCAP's uniform prior: **the paper already evaluates BayCAP with a well-specified prior on $\pi^Q$**, and under that protocol any informative prior on $\pi^Q$ is misspecified by construction.

If the prior equals $G$ and the likelihood is the true sampling model, a $95\%$ credible interval contains the true value for exactly $95\%$ of the bags. The reason is Bayes' theorem: the posterior is then the actual conditional distribution of the true $\pi^Q$ among all bags that produce the same observed data. The guarantee is an **average over bags drawn from $G$**, not a per-bag property. A vague wrong prior usually keeps coverage and loses sharpness; a confident wrong prior can lose coverage. The standard check is *simulation-based calibration*: the rank of the true value among the posterior draws should be uniform across bags.

Toy example (`scripts/toys/ext5_prior_calibration.py`): binary, perfect known classifier, $n\sim\operatorname{Bin}(20,\pi_1)$, Beta priors, $200{,}000$ bags, $95\%$ equal-tailed intervals.

| $G$ | prior | coverage | mean width | coverage for $\pi_1\in[0.4,0.6)$ |
|---|---|---|---|---|
| uniform | uniform (well specified) | $0.950$ | $0.33$ | $0.95$ |
| uniform | $\operatorname{Beta}(2,8)$ | $0.65$ | $0.31$ | $0.83$ |
| $\operatorname{Beta}(2,8)$ | $\operatorname{Beta}(2,8)$ (well specified) | $0.950$ | $0.26$ | $0.86$ |
| $\operatorname{Beta}(2,8)$ | uniform | $0.955$ | $0.31$ | $0.95$ |
| $\operatorname{Beta}(2,8)$ | $\operatorname{Beta}(8,2)$ | $0.26$ | $0.34$ | $0.81$ |

With a non-uniform $G$, the uniform prior keeps coverage but its intervals are about $20\%$ wider than those of the well-specified prior: that difference is the value of prior information.

The guarantee has three conditions, and each can fail in BayCAP: the prior must equal $G$; the likelihood must be the true sampling model (fails under APP, source (i) of [[ext2_baycap|Extension 2]] §3); and the interval must target the quantity that is scored (fails because of the estimand gap, source (ii)). If BayCAP with a well-specified prior on $\pi^Q$ misses nominal coverage, that prior is therefore ruled out as the cause. The argument says nothing about the prior on $C$: there is no "bag-generating distribution" of $C$ in the protocol, since $C$ is fixed by the classifier, so a prior on $C$ cannot be well specified in this sense and is judged instead by the bias it induces (§2.4).

##### 2.4 The prior on $C$ is informative

*Intuition.* "Uniform on the simplex" sounds neutral, but in $K$ dimensions almost all of the simplex's volume lies near its centre, so a uniform prior on a column of $C$ says, with near certainty, that the classifier is close to random.

**The marginal of a recall.** Under $C_{:j}\sim\operatorname{Dir}(\beta\mathbf 1_K)$, the aggregation property of the Dirichlet gives

$$
C_{jj}\sim\operatorname{Beta}\!\left(\beta,(K-1)\beta\right),
\qquad
\mathbb E[C_{jj}]=\frac1K .
$$

For $\beta=1$ this is $\operatorname{Beta}(1,K-1)$, with standard deviation of order $1/K$ and $P(C_{jj}>\tfrac12)=2^{-(K-1)}$: the prior statement "the recall is about $1/K$" becomes stronger as $K$ grows.

**A Dirichlet prior is a number of imaginary instances.** With $f_{jj}$ correct validation instances out of $n^s_j$, the posterior mean of the recall is

$$
\tilde r_j=\frac{f_{jj}+\beta}{n^s_j+K\beta},
\qquad
\mathbb E[\tilde r_j]-r_j=\frac{w\,(1/K-r_j)}{n^s_j+w},
\qquad
w=K\beta .
$$

Every symmetric Dirichlet shrinks towards $1/K$; they differ only in the total weight $w$, the number of imaginary validation instances added to class $\omega_j$. $\operatorname{Dir}(\mathbf 1)$ has $w=K$, Jeffreys' $\operatorname{Dir}(\tfrac12)$ has $w=K/2$, Perks' $\operatorname{Dir}(1/K)$ has $w=1$ for every $K$. The requirement behind Perks' prior: refining the label set should not add prior information about whether a prediction is right. Accuracy depends only on "right vs wrong", so a prior whose information grows with the number of ways of being wrong is the wrong default for this estimand. Literature, to verify before citing: Perks (1947) proposed this rule; Berger, Bernardo and Sun, "Overall objective priors" (Bayesian Analysis, 2015), discuss $\operatorname{Dir}(1/K)$ as an overall objective prior for the multinomial when $K$ is large relative to $n$; add-one (Laplace) smoothing is known to over-smooth sparse multinomials.

**Error-minimising weight.** For a fixed true recall $r$, the mean squared error of $\tilde r_j$ is

$$
\operatorname{MSE}(\beta)=\frac{n\,r(1-r)+\beta^2(1-Kr)^2}{(n+K\beta)^2},
\qquad
\beta^*=\frac{K\,r(1-r)}{(Kr-1)^2},
\qquad
w^*=K\beta^*\xrightarrow{K\to\infty}\frac{1-r}{r},
$$

with $n=n^s_j$. The optimal total weight does not depend on $n$ and tends to a constant in $K$, so $\beta^*\propto1/K$; it equals Perks' $w=1$ at $r=\tfrac12$ and is smaller for better classifiers. For the accuracy, the $K$ shifts have the same sign and add up, while the $K$ noise terms partly cancel; with $K$ equal classes the optimum becomes $w^*=K\,r(1-r)/(Kr-1)^2\to(1-r)/(Kr)$, which vanishes as $K$ grows. The same comparison gives the ratio of the bias of the accuracy to its posterior standard deviation, ignoring the uncertainty on $\pi^Q$:

$$
\frac{\text{bias}}{\text{sd}}\approx w\sqrt{\frac{K\,r}{n\,(1-r)}},
$$

so a prior is harmless for coverage only if $w\ll\sqrt{n(1-r)/(Kr)}$. Perks' prior reduces the bias by a factor of about $K$ with respect to $\operatorname{Dir}(\mathbf 1)$, but on datasets with large $K$ and very accurate classifiers it may still leave a visible bias; this is why E5.3 varies $w$ rather than testing one prior.

**Limits.**

- No symmetric Dirichlet is neutral about the recall: even $\operatorname{Dir}(1/K)$ has prior mean $1/K$, with a U-shaped marginal $\operatorname{Beta}(1/K,1-1/K)$. Its advantage is weakness, not correct centring.
- $w\to0$ (Haldane's prior) is improper, and confusion columns are sparse: off-diagonal cells with zero counts leave the posterior degenerate on the boundary of the simplex. Small $w>0$ suits this sparsity, since $\beta<1$ puts mass near the faces of the simplex, where sparse columns lie.
- **Diagonal-aware prior** (new hypothesis): a non-symmetric Dirichlet with $\beta_{jj}=1$ on the diagonal cell and $\beta_{ij}=1/(K-1)$ on each off-diagonal cell. Its recall marginal is $\operatorname{Beta}(1,1)$, uniform and independent of $K$; its total weight is $w=2$ and its shift $(1-2r_j)/(n^s_j+2)$. It replaces the symmetric assumption, which treats the diagonal like any other cell, with the structure of a confusion matrix.

**Interaction with tempering.** Joint tempering divides the validation counts by $\tau$ but leaves the prior unchanged, so the shift becomes $w\,(1/K-r_j)/(n^s_j/\tau+w)$: with $\tau>1$ the bias of the prior on $C$ grows (H8).

All formulas of this section are derivations; the numbers that illustrate them in the paper will come from the toy simulation proposed in Appendix A.

### Part II: For the paper

#### 3. Contributions

1. **Tempering as a diagnostic.** What tempering can achieve with the best temperature (oracle-best $\tau$, a reference that uses target labels), compared with the realised-table predictive of [[ext2_baycap|Extension 2]], which needs neither labels nor tuning.
2. **Priors on $\pi^Q$ (the paper's stated future work).** Under the paper's protocol the uniform prior is already well specified; informative priors help only under a non-uniform bag-generating distribution, and only when the prior weight is not negligible (small $\lvert U\rvert$, ill-conditioned $C$).
3. **Prior on $C$.** The uniform prior on the columns of $C$ is informative and biases BayCAP as $K$ grows; a prior with total weight independent of $K$, or a diagonal-aware prior, removes the bias.
4. **Label-free tempering** (only if H5 is refuted): a calibration procedure for $\tau$ on the performance interval.

#### 4. Hypotheses

Primary:

- **H5 (the predictive needs no tuning).** The realised-table predictive reaches near-nominal coverage without tempering, and at matched coverage its intervals are no wider than those of tempering at the oracle-best $\tau$. *Confirmed if* in E5.2 its coverage is within tolerance of $0.95$ across datasets and its Winkler score is close to or better than that of oracle-best tempering. *Refuted if* oracle-best tempering is clearly better; then label-free selection of $\tau$ (E5.5) becomes worth its cost. Tested only if H0 of [[ext2_baycap|Extension 2]] is confirmed.
- **H7 (a $K$-independent prior on $C$ removes the bias).** On datasets with large $K$ and few validation instances per class, replacing $\operatorname{Dir}(\mathbf 1)$ with a prior of total weight $w\le2$ ($\operatorname{Dir}(1/K)$, diagonal-aware) reduces the bias of BayCAP by about the factor $K/w$ predicted in §2.4 and restores most of the coverage, without changing the results on binary datasets. *Confirmed if* in E5.3 the bias follows $w\,(1/K-r_j)/(n^s_j+w)$ across $w$ and coverage rises towards nominal on `isolet` and `letter`. *Refuted if* the bias does not change with $w$; then source (iv) of [[ext2_baycap|Extension 2]] is not the cause.

Secondary:

- **H1 (prior on $\pi^Q$ matters only when data are weak).** The prior's effect grows with $w=c/(c+N^Q/\tau)$ and with the ill-conditioning of $C$. *Confirmed if* in E5.4 the well-specified prior narrows intervals at $\lvert U\rvert=100$ and not at $\lvert U\rvert=1000$.
- **H8 (tempering amplifies the bias of the prior on $C$).** With $\operatorname{Dir}(\mathbf 1)$ on $C$, the bias on large-$K$ datasets grows with $\tau$ as $w\,(1/K-r_j)/(n^s_j/\tau+w)$. *Confirmed if* in E5.1 the bias at $\tau\in\{2,4\}$ is larger than at $\tau=1$ on these datasets, by about the predicted amount. *Refuted if* the bias does not depend on $\tau$.
- **H4 (prior $\times$ temperature).** If $\tau>1$ is selected, a misspecified prior on $\pi^Q$ gains weight and robustness degrades. *Confirmed if* in E5.5 a confidently wrong prior loses more coverage at the selected $\tau$ than at $\tau=1$. Tested only if E5.5 runs.

#### 5. Paper outline

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §5 (BayCAP), subsection "Calibrated BayCAP" of [[ext2_baycap|Extension 2]], second half | tempering as a diagnostic | The realised-table predictive calibrates BayCAP at least as well as the best temperature. | E5.1; E5.2 |
| §5, paragraph "Prior on $C$" of [[ext2_baycap|Extension 2]], second half | prior weight $w=K\beta$; Perks and diagonal-aware priors | A prior on $C$ whose weight does not grow with $K$ removes BayCAP's bias on large-$K$ datasets. | §2.4; E5.3 |
| §5, short paragraph "Prior on $\pi^Q$" | well-specified prior under APP; prior weight | Under the paper's protocol the uniform prior is the correct one, and informative priors matter only when data are weak. | §2.2–2.3; E5.4 |
| §6 (experiments), subsection "Calibration of BayCAP", shared with [[ext2_baycap|Extension 2]] | F2, T2, T3, T5 (see §5.1) | as above | Part III |
| §7 (conclusions) | one paragraph | Calibrated BayCAP closes (or narrows) the gap with SQS-H in interval quality. | E5.2 (T2), against SQS-H |

Wording if results are mixed: if H5 is refuted, the predictive is presented as a correction of the estimand, and label-free tempering (E5.5) as a calibration step on top of it; if H7 is refuted, the "Prior on $C$" paragraph reports the bias without a remedy; if H1 is refuted at $\lvert U\rvert=100$, priors on $\pi^Q$ get one sentence in the conclusions.

##### 5.1 Figures and tables

All use Acc; macro-F1 and macro-K versions go to the appendix. Conventions follow the existing paper figures.

| Item | Experiment | Content | What it must show |
|---|---|---|---|
| F2 | E5.1, E5.2 | coverage–amplitude trajectories over $\tau$, one path per dataset through its $\tau$ values; solid joint tempering, dashed $\mathcal L_U$-only; separate marker for the predictive at $\tau=1$ | where tempering reaches $0.95$ and at what width, compared with the predictive (H5) |
| T2 | E5.2 | mean ranks per measure with Wilcoxon markers (`tables/rank.py`) for BayCAP at $\tau=1$, predictive at $\tau=1$, and SQS-H from `main`; oracle-best variants as greyed reference columns, not ranked; one block per $\lvert U\rvert$ (SQS-H only at $1000$) | whether Calibrated BayCAP closes the gap with SQS-H |
| T3 | E5.4 | rows prior on $\pi^Q$ (uniform, well-specified, confidently wrong) $\times$ $\lvert U\rvert$; columns coverage, amplitude, width relative to the uniform prior, for both pushforwards; mean $\pm$ std with min/max colouring (`tables/binform.py` style) | that informative priors matter only at small $\lvert U\rvert$ (H1) |
| T5 | E5.3 | rows dataset $\times$ prior on $C$ (ordered by $w$); columns bias, coverage, amplitude, Winkler, for both pushforwards; $K$ and median $n^s_j$ as index columns | that the bias falls with $w$ as predicted and coverage recovers (H7) |

Appendix and internal material: coverage vs $\tau$ per dataset (diagnostic used to read off the oracle-best $\tau$); bias vs $\tau$ on large-$K$ datasets (H8); `plots/amplitude_vs_coverage.py` with the E5.2 variants; `plots/binform_simplex.py` to illustrate $G$ and the E5.4 priors on the simplex; robustness curves of the full prior study.

### Part III: For the implementation

#### 6. Experiments

Common to all: same bags and seeds across conditions, $100$ bags per configuration as in the paper, measures as in the paper (AE, coverage, amplitude, Winkler at $\alpha=0.05$) on the metrics of [[ext1_metrics]], always scored against the realised table of each bag; both pushforwards of [[ext2_baycap|Extension 2]] (population and realised-table predictive) unless stated. Dataset subset: the one of [[ext2_baycap|Extension 2]] (binary and multiclass, small and large $\lvert V\rvert$ per class, well- and ill-conditioned $C$, at least one large-$K$ dataset); `connect-4` for E5.4. All five classifiers unless stated. Order and gates: E5.1 → E5.2 (needs H0 of [[ext2_baycap|Extension 2]] for the predictive; otherwise it compares tempering with BayCAP at $\tau=1$ only); E5.3 is independent; E5.4 runs its minimal check first and the full grid only if H1 is confirmed; E5.5 runs only if H5 is refuted.

##### E5.1: temperature sweep

- **Goal / hypothesis:** what tempering could achieve, with target labels used only to evaluate it; input to H5; H8.
- **Inputs:** BayCAP rerun on the APP bags of the subset (the same bags as [[ext2_baycap|Extension 2]] Q2, whose $\tau=1$ draws are reused).
- **Varied:** $\tau\in\{0.25,0.5,1,2,4\}$; tempering variant: joint ($\mathcal L_V$ and $\mathcal L_U$), $\mathcal L_U$ only, and, as an ablation, $\mathcal L_V$ only (uncertainty on $C$).
- **Outputs:** table draws (sidecar `.npz`) for every $\tau$ and variant, so that E5.2 can apply the predictive to them; metric samples; the oracle-best $\tau$ per (dataset, classifier) and pushforward; the bias split of [[ext2_baycap|Extension 2]] Q5 at every $\tau$.
- **Measures:** the paper's measures; bias and its recall term on large-$K$ datasets.
- **Interpretation:** the direction in which $\tau$ must move to reach nominal coverage ($\tau>1$ on large-$V$ datasets if the estimand gap dominates); how wide the intervals become there, compared with the predictive at the same coverage; whether loosening $\mathcal L_U$ alone gives the extra width. The $\tau$ closest to nominal coverage is the *oracle-best $\tau$*, a reference, not a method. A recall term growing with $\tau$ on large-$K$ datasets supports H8.
- **Visualization:** F2; coverage vs $\tau$ per dataset; bias vs $\tau$ (appendix).
- **Cost:** five runs per bag and per tempering variant.

##### E5.2: Calibrated BayCAP vs tempering

- **Goal / hypothesis:** is the realised-table predictive as good as the best tempering? H5.
- **Inputs:** subset datasets, $\lvert U\rvert\in\{100,1000\}$.
- **Varied:** four variants: BayCAP at $\tau=1$ (population pushforward, as in the paper); realised-table predictive at $\tau=1$; tempering at the oracle-best $\tau$ (population pushforward); realised-table predictive at the oracle-best $\tau$.
- **Outputs:** metric samples per variant, oracle-best $\tau$ for each pushforward.
- **Measures:** the paper's measures; width at matched coverage; comparison with SQS-H from `main` at $\lvert U\rvert=1000$.
- **Interpretation:** H5 holds if the predictive at $\tau=1$ is within tolerance of nominal coverage and its Winkler score is close to that of oracle-best tempering; the predictive at the oracle-best $\tau$ shows whether any residual tempering helps.
- **Comparability:** all variants are fitted on all of $V$ and are directly comparable with `main`. The oracle-best variants use target labels to choose $\tau$ and are references, not methods.
- **Visualization:** T2; F2 (shared with E5.1); `amplitude_vs_coverage` with the four variants in the appendix.
- **Cost:** at $\lvert U\rvert=1000$ none beyond E5.1, since all four variants reuse its draws; $\lvert U\rvert=100$ needs one $\tau$ sweep.

##### E5.3: prior on $C$

- **Goal / hypothesis:** does a prior on $C$ with total weight independent of $K$ remove BayCAP's bias on large-$K$ datasets? H7.
- **Inputs:** `isolet`, `letter` ($K=26$), `hand_digits` ($K=10$), `chess` ($K=15$), and one binary control (`spambase`); the $100$ `main` bags with the same seeds; LR in the pilot, all five classifiers in the full run.
- **Varied:** prior on $C$: $\operatorname{Dir}(\mathbf 1)$ ($w=K$, as in the paper), $\operatorname{Dir}(1/K)$ ($w=1$), $\operatorname{Dir}(0.1/K)$ ($w=0.1$), diagonal-aware ($w=2$); pushforward: population and realised-table predictive; $\tau=1$; prior on $\pi^Q$ uniform.
- **Outputs:** metric samples; the bias split of [[ext2_baycap|Extension 2]] Q5 for every prior; NUTS diagnostics (divergences, effective sample size), since small $\beta$ puts mass near the boundary of the simplex.
- **Measures:** the paper's measures; bias and recall term against the prediction $w\,(1/K-r_j)/(n^s_j+w)$ of §2.4.
- **Interpretation:** see H7. A bias that falls with $w$ but coverage that stays low points to the remaining sources of [[ext2_baycap|Extension 2]] §3; divergences at small $w$ limit how small $w$ can be in practice.
- **Visualization:** T5.
- **Cost:** four priors × five datasets × $100$ bags NUTS runs per classifier; the population and realised pushforwards share each run.

##### E5.4: priors on $\pi^Q$

- **Goal / hypothesis:** answer the paper's future work on priors; H1. A minimal check first; the full study only if H1 is confirmed.
- **Inputs:** non-uniform $G=\operatorname{Dir}(c^\star\mu^\star)$ with the `binform` construction $\mu^\star_k=(1-\pi^V_k)/(K-1)$, $c^\star=10$ (frequent validation classes become rare targets; e.g. $\pi^V\approx(0.66,0.25,0.10)$ gives $\mu^\star\approx(0.17,0.38,0.45)$); `connect-4` plus one binary and one ill-conditioned multiclass dataset.
- **Varied (minimal check):** prior: uniform; well specified, $\operatorname{Dir}(c^\star\mu^\star)$; confidently wrong, $\operatorname{Dir}(c\,\pi^V)$ with large $c$ (the "no-shift" prior); pushforward: population and realised-table predictive; $\tau=1$; $\lvert U\rvert\in\{100,1000\}$. Optionally one per-bag oracle row (prior centred on the bag's true prevalence), as an upper bound only.
- **Varied (full study):** with $\alpha=c\,\mu$, the conditions of the table below, the concentration grid $c/c^\star\in\{0.1,0.3,1,3,10\}$ (also stated relative to $N^Q/\tau$), crossed with $\lvert U\rvert$.
- **Outputs and measures:** the paper's measures; width ratio of well-specified vs uniform prior; in the full study, robustness curves (Winkler score and coverage as functions of the misspecification distance and of $c/c^\star$).
- **Interpretation:** see H1. If the minimal check shows no effect at $\lvert U\rvert=100$, the full study is not run.
- **Visualization:** T3; optionally `binform_simplex` for $G$ and the priors; robustness curves in the appendix.
- **Cost:** minimal check: three priors × two $\lvert U\rvert$ × three datasets × $100$ bags; the full study multiplies this by about ten.
- **Note:** `binform` already covers one configuration of this grid (uniform vs $\operatorname{Dir}(3\mu^\star)$, `connect-4`, $\lvert U\rvert=1000$, $\tau=1$); its results were not inspected for this note.

Conditions of the full study:

| Condition | $\mu$ | $c$ | Role |
|---|---|---|---|
| uniform | $\mathbf 1/K$ | $K$ | paper baseline; well specified only when $G$ is uniform |
| no-shift | $\pi^V$ | grid | practitioner default; misspecified under PPS by construction |
| well-specified | $\mu^\star$ | $c^\star$ | matches $G$ |
| partially informative | $\mu^\star$ | $c<c^\star$ | correct mean, under-confident |
| over-concentrated | $\mu^\star$ | $c>c^\star$ | correct mean, over-confident |
| mean-misspecified | $\mu_\delta=(1-\delta)\mu^\star+\delta\mu^\dagger$ | $c^\star$ | robustness curve in $\delta\in\{0,0.25,0.5,0.75,1\}$ |
| per-bag oracle | $\pi^Q_i$ | large | upper bound only (reference) |

$\mu^\dagger$ is a fixed wrong mean, e.g. $\pi^V$ or the point of the simplex opposite $\mu^\star$, and misspecification is reported as the $L_1$ or Aitchison distance between $\mu_\delta$ and $\mu^\star$.

##### E5.5: label-free temperature selection (gated)

- **Goal / hypothesis:** select $\tau$ without target labels, if oracle-best tempering clearly beats the predictive (H5 refuted); H4.
- **Inputs:** subset datasets; $V$ split into $V_{\mathrm{fit}}$ and $V_{\mathrm{cal}}$ with stratification.
- **Varied:** $\tau\in\{0.25,0.5,1,2,4\}$ on calibration bags; prior on $\pi^Q$ (uniform, confidently wrong) for H4.
- **Outputs:** the selected $\tau$ per (dataset, classifier) and $\lvert U\rvert$; metric samples at deployment.
- **Measures:** the paper's measures; regret in Winkler score between the selected and the oracle-best $\tau$.
- **Interpretation:** comparison of BayCAP at $\tau=1$, label-free tempering, the realised-table predictive, both combined, and the oracle-best $\tau$ (reference).
- **Visualization:** a column of T2; regret per dataset in the appendix.
- **Cost:** one NUTS run per calibration bag per value of $\tau$, i.e. $250$ runs per (dataset, classifier) and $\lvert U\rvert$ with $S=50$ calibration bags and five values of $\tau$, compared with $100$ runs for evaluation; small values of $\tau$ may also increase NUTS divergences.

$\tau$ is calibrated on the **performance interval** of $A$, not on the prevalence region, so `quapy`'s `calibrate_temperature` cannot be used as is; its structure (grid over $\tau$, coverage or Winkler criterion, amplitude filter) carries over. The procedure:

1. Split $V$ into $V_{\mathrm{fit}}$ and $V_{\mathrm{cal}}$ with stratification. $\mathcal L_V$ uses only $V_{\mathrm{fit}}$, at calibration and at deployment, so that its strength is the same in both.
2. Draw $S$ calibration prevalences $\pi^{(s)}\sim\operatorname{Dir}(\mathbf 1)$; for each, draw a bag of size $\lvert U\rvert$ from $V_{\mathrm{cal}}$ with replacement. Its labels are known, so its realised table is known.
3. For each $\tau$, run tempered BayCAP on every calibration bag and compute coverage, amplitude, and Winkler score of the interval of Acc, scored against the realised table.
4. Select $\tau$ by minimum Winkler score among the candidates whose coverage is within a tolerance of $1-\alpha$.

Notes:

- the grid covers both sides of $1$: the estimand gap calls for $\tau>1$, the fixed-count likelihood for $\tau<1$;
- one $\tau$ serves all metrics, so one posterior serves all metrics;
- calibration prevalences stay uniform also with informative priors: drawing them from the prior would reuse the prior's assumption and could not reveal its misspecification;
- bags from a small $V_{\mathrm{cal}}$ repeat instances; $\lvert V_{\mathrm{cal}}\rvert$ should be reported next to $\lvert U\rvert$;
- configurations fitted on $V_{\mathrm{fit}}$ are not directly comparable with `main`; BayCAP at $\tau=1$ on $V_{\mathrm{fit}}$ should be reported as a reference.

Further ablations: a separate temperature per metric instead of one $\tau$ selected on Acc; refitting on all of $V$ after selecting $\tau$ on $V_{\mathrm{fit}}$; prior-predictive calibration bags ($\pi^{(s)}$ drawn from the prior) versus uniform calibration bags, which shows whether calibration can hide a misspecified prior.

#### 7. Components, tests, and reproducibility

Components (all [proposal]):

- **`temperature` argument of `methods/baycap.py:BayCAP`**: both likelihood terms divided by $\tau$, with a flag for $\mathcal L_U$-only and $\mathcal L_V$-only tempering; default $\tau=1$. The alternative, a wrapper of `quapy`'s `model_bayesianCC` (private API; record the commit), is kept only if extending `BayCAP` proves harder.
- **`c_prior` argument of `BayCAP`**: the prior on the columns of $C$: `"uniform"` (default, $\beta_{ij}=1$, as now), a positive scalar $\beta$ repeated in every cell, `"perks"` ($\beta=1/K$), `"diagonal"` (the diagonal-aware prior of §2.4), or a $K\times K$ array in the paper layout (stored transposed in the model, as `p_c_cond_y`).
- **Prior registry**: the E5.4 priors on $\pi^Q$, computed from $\pi^V$ and $G$; `binform` stays unchanged.
- **Label-free $\tau$ selection** (E5.5 only): calibration bags from $V_{\mathrm{cal}}$ and the selection rule of E5.5.

Tests:

- `BayCAP` with $\tau=1$ and `c_prior="uniform"` reproduces `ACC_BayesCAP` exactly for the same `random_state`;
- with a large validation set and $\tau$ large enough that the target term is weak, the posterior mean of each column of $C$ matches the conjugate value $(f_{\cdot j}/\tau+\beta_{\cdot j})/(n^s_j/\tau+w_j)$ within Monte Carlo error;
- `c_prior` arrays are checked against a hand-computed $2\times2$ example in the paper layout (guards against transposed priors);
- the diagonal-aware prior has row sums equal to $2$ and diagonal equal to $1$ in the paper layout.

Reproducibility and comparability:

- all runs go to the `calibration` experiment of [[ext2_baycap|Extension 2]], under `cap.env["OUT_DIR"]/uncawe/calibration`; `main` and `binform` and their output paths are unchanged;
- every row records $G$, prior on $\pi^Q$, prior on $C$, $\tau$ and tempering variant, seeds, pushforward, and bag protocol;
- every configuration of E5.1–E5.4 is fitted on all of $V$, so all rows are directly comparable with `main`; E5.5 is not (see its notes);
- a pilot on two datasets estimates the NUTS cost before the full grid; E5.3 reports NUTS diagnostics for every prior.

### Appendix

#### A. Toy simulations

- `uv run python scripts/toys/ext5_prior_calibration.py` (§2.3; defaults: $200{,}000$ bags, seed $0$) is seeded and prints the table of §2.3.
- [proposal] a toy for §2.4: simulate validation confusion columns for chosen $K$, $n^s_j$ and $r$, compute the conjugate posterior of $C$ for each prior on $C$ (no NUTS), and print the predicted and simulated bias, the ratio bias/sd, and the coverage of the accuracy interval. It checks the formulas of §2.4 before E5.3 and provides the worked numbers for the paper.

#### B. Relation to other extensions

- **[[ext2_baycap|Extension 2]]**: supplies BayCAP, the two pushforwards, the sources of miscalibration (source (iv) motivates E5.3), the dataset subset, the `calibration` experiment and the $\tau=1$ draws; H0 there gates the comparison of E5.2.
- **[[ext4_uncertainty_budget|Extension 4]]**: supplies the $\lvert U\rvert$ axis used by E5.2 and E5.4.
- **[[ext3_posterior_propagation|Extension 3]]** (BayCAP-KDEy, SQS-B): `BayesianKDEy` also has a temperature; Extension 3 keeps $\tau=1$, and tempering BayesKDEy belongs to this extension (user's decision of 2026-10-27), with no experiment designed yet.
- **[[optional_extensions]]**: §1 covers post-hoc interval rescaling, a different estimator from in-model tempering. Its former §4 (full informative-prior study) and §5 (label-free temperature selection) were revived here on 2026-10-15, as E5.4 and E5.5.

#### C. Out of scope

- hierarchical or empirical-Bayes priors learned from a stream of target sets;
- informative priors on $C$ built from external knowledge (e.g. from other classifiers on the same dataset); E5.3 studies only weak default priors;
- post-hoc interval recalibration;
- violations of PPS.
