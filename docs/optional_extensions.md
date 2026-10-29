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

Revived in [[ext5_baycap_priors_tempering|Extension 5]] (then numbered 10, 2026-10-15), as the full grid of experiment E5.4.

## 5. Label-free temperature selection for BayCAP

Revived in [[ext5_baycap_priors_tempering|Extension 5]] (then numbered 10, 2026-10-15), as experiment E5.5, still run only if oracle-best tempering clearly beats the realised-table predictive (hypothesis H5 refuted).

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

On 2026-10-29 [[ext4_uncertainty_budget|Extension 4]] was restructured into the experiment-driven form of Extensions 2 and 3, which keeps no paper material for now. Its contributions, paper outline and figure plan are kept here as they were (section and hypothesis numbers refer to the earlier version, commit `dbb04dc`); the hypotheses H1–H4 are now in the experiments E4.1 and E4.2 of Extension 4, and the sections cited as §1–§3 are its §T.1–§T.8.

### Contributions

1. **An exact error split and a budget** for CAP under PPS, which makes explicit that the realised table is the estimand and identifies which uncertainty terms shrink with $\lvert U\rvert$ and which do not.
2. **A unifying classification** of existing and proposed uncertainty-aware CAP methods by the parts they represent (§3.4), and a coverage rule in terms of the ratio $\eta$ of missing to represented variance (§3.5), which explains why the paper's methods are calibrated in different regimes.
3. **A regime map**, empirical, of coverage as a function of $(\lvert U\rvert,\lvert V\rvert)$, showing whether the paper's ranking at $\lvert U\rvert=1000$ depends on the design ratio $r$.
4. **Completed methods**: evidence that methods representing all terms (realised-table BayCAP from Extension 2, and SQS-H with $U$ and $V$ resampling) are calibrated across the map.

### Hypotheses

Primary:

- **H1 (SQS-type intervals degrade with $n$).** At fixed $V$, the coverage of SQS-H and RQBS decreases as $\lvert U\rvert$ grows, faster when $\lvert V_j\rvert$ is small. *Confirmed if* in E4.2 the coverage of RQBS decreases monotonically with $\lvert U\rvert$ on most (dataset, classifier) pairs and the decrease is larger at the smaller $\lvert V\rvert$ level. *Refuted if* coverage is flat in $\lvert U\rvert$; then $e_2$ and $e_1^{V}$ are negligible on real data and the budget adds little to Extension 2.
- **H2 (BayCAP under-covers at small $n$ and large $V$).** The coverage of BayCAP with the population pushforward increases with $\lvert U\rvert$ at fixed $V$ and is below nominal where its $\eta=\sigma_3^2/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ is large (§3.6). *Confirmed if* in E4.2 coverage increases with $\lvert U\rvert$ at fixed $V$ and decreases with $\lvert V\rvert$ at fixed $\lvert U\rvert$. *Refuted if* BayCAP over-covers at every $r$; then the fixed-count likelihood (source (i) of [[ext2_baycap]] §3) or prior effects dominate, and the budget does not explain BayCAP.
- **H3 (the budget accounts for the error).** In E4.1, $\operatorname{RMSE}(e_3)$ scales as $n^{-1/2}$, $\operatorname{RMSE}(e_2)$ is constant in $n$ and scales as $\lvert V\rvert^{-1/2}$, and $e_1$ is the sum of a part shrinking with $n$ and a part fixed given $V$. *Confirmed if* the fitted slopes on log–log axes are within $\pm0.15$ of the predicted ones and, for every method, the coverage plotted against its $\eta$ (computed from the variances of the parts estimated in E4.1) lies within $\pm0.05$ of $2\Phi(1.96/\sqrt{1+\eta})-1$ on most (dataset, classifier) pairs. *Refuted if* coverage does not follow $\eta$; then non-normality, the within-group correlations of §3.2 or quantifier bias $b_q$ matter, and the budget must be refined.

Secondary:

- **H4 (completing the budget restores coverage).** Methods that represent all terms are within $\pm0.03$ of nominal coverage across the grid, while each partial method fails where its missing term dominates. *Confirmed if* in E4.2 the realised-table BayCAP and SQS-H with $U$ and $V$ resampling meet this tolerance at every $(\lvert U\rvert,\lvert V\rvert)$ level. *Refuted if* they fail at large $n$ as well; then quantifier bias $b_q$ is the dominant fixed term and no variance-based correction suffices.

### Paper outline

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §3 (problem), new subsection "Target estimand and uncertainty budget" | §1–§3.3 of this note | The CAP error splits into prevalence, rate-matrix and finite-$U$ parts with different scaling. | derivation; F0 (toy); E4.1 |
| §4–§5 (methods), one table and the coverage rule | §3.4–§3.5 | Each method represents only some parts; its coverage follows from the ratio $\eta$ of missing to represented variance. | tables of §3.4–§3.5; E4.1 (H3) |
| §5, new subsections from Extensions 2 and 3 | completed methods | Adding the missing term restores calibration. | E4.2 (H4); [[ext5_baycap_priors_tempering]] E5.2 |
| §6 (experiments), new subsection "Target and validation size" | E4.1, E4.2 | The paper's ranking at $\lvert U\rvert=1000$ is regime-dependent. | F1, F2, T1 |
| §7 (conclusions) | one paragraph | Practical guidance: which method to use given $\lvert U\rvert$ and $\lvert V\rvert$. | F2 |

Wording if results are mixed: if H1 is refuted, the budget becomes a short section explaining the estimand gap, and Extension 2 carries the main claim; if H3 is refuted but H1 and H2 hold, the regime map is presented empirically without the closed-form ratio.

#### Figures and tables

| Item | Question | Content | What it must show |
|---|---|---|---|
| F0 (optional) | toy | coverage vs $r$ for each method of Appendix A | the predicted shape of F2 |
| F1 | E4.1 | RMSE of $e_1$, $e_2$, $e_3$ vs $\lvert U\rvert$ (log–log), one panel per $\lvert V\rvert$ level, one line per dataset (LR) | the scaling of H3 |
| F2 | E4.2 | (a) coverage vs $r$ (log), one panel per method, one point per (dataset, classifier, $\lvert U\rvert$, $\lvert V\rvert$), dashed line at $0.95$; (b) coverage vs each method's $\eta$ (log), all methods in one panel, with the curve $2\Phi(1.96/\sqrt{1+\eta})-1$ | (a) the regime map, H1, H2, H4; (b) the coverage rule, H3 |
| T1 | E4.2 | mean Winkler rank per method, one column block per $\lvert U\rvert$ at full $V$; oracle references as unranked columns (`tables/rank.py`) | whether the ranking changes with $\lvert U\rvert$ |
| T2 (appendix) | E4.1 | per-dataset bias and standard deviation of $\hat\pi$ (L1) and $\hat A$ vs $\lvert U\rvert$ | quantifier bias $b_q$, needed to interpret H4 |

Macro-F1 and macro-K versions of F2 go to the appendix.
