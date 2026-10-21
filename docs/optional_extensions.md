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

The marginal stress test previously listed among the secondary analyses of Extension 3 (error caused by model misspecification or violations of PPS) is set aside here; the [[ext3_uncertainty_budget|uncertainty budget]] assumes strict PPS.

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

Revived in [[ext10_baycap_priors_tempering|Extension 10]] (2026-10-15), as the full grid of experiment E10.4.

## 5. Label-free temperature selection for BayCAP

Revived in [[ext10_baycap_priors_tempering|Extension 10]] (2026-10-15), as experiment E10.5, still run only if oracle-best tempering clearly beats the realised-table predictive (hypothesis H5 refuted).

## 6. A pass/fail gate for the realised-table predictive of BayCAP

Set aside from [[ext2_baycap|Extension 2]] on 2026-10-21: E2.1 and E2.3 are treated as analyses, not as a gate. The provisional decision rule was:

- *Pass* if, in E2.1 and confirmed by the exact run (now E2.3), at least one pair in five under-covers (coverage below $0.90$) and the predictive brings it within $[0.93,0.97]$ with a lower Winkler score.
- *Fail* if $\hat\sigma_g/\hat\sigma_p<0.3$ on almost every pair (the coverage then changes by about one percentage point or less), or if BayCAP over-covers and the predictive worsens the Winkler score everywhere.

All pairs count, including those whose interval is displaced (coverage near $0$ for both pushforwards), which no pushforward can repair (decision of 2026-10-20).
