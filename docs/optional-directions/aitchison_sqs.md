# SQS and Aitchison geometry

Yes, this is an interesting approach, with one important conceptual correction:

> Aitchison geometry does not automatically make a distribution on the simplex Gaussian. It makes it possible to represent the simplex in Euclidean `ilr` coordinates; assuming a Gaussian distribution in those coordinates induces a logistic-normal distribution on the simplex.

A composition $\pi\in\Delta^{K-1}$, with strictly positive components, is transformed into

$$
z=\operatorname{ilr}(\pi)\in\mathbb{R}^{K-1}.
$$

If one assumes

$$
Z\sim\mathcal N(\mu,\Sigma),
$$

then

$$
\Pi=\operatorname{ilr}^{-1}(Z)
$$

follows a logistic-normal distribution on the simplex, often described as a “normal distribution on the simplex.” This is precisely the standard interpretation of normality in Aitchison geometry. [This source on compositional data analysis](https://compositionaldata.com/codawork2015/images/ProceedingsBook.pdf) defines it in terms of `ilr` coordinates.

## Where the idea becomes scientifically compelling

The weak version would be:

1. the quantifier produces $\hat\pi^Q$;
2. an isotropic Gaussian is placed arbitrarily around $\operatorname{ilr}(\hat\pi^Q)$;
3. prevalence vectors and validation sets are sampled.

This procedure is geometrically elegant, but leaves the main question unanswered: where does the covariance $\Sigma$ come from? The quantifier's point estimate alone does not contain enough information to determine it. With a manually chosen standard deviation, the method would essentially be a smoothing procedure with a new hyperparameter.

A first compelling version would instead learn the distribution of the quantification error from validation bags with known prevalence.

For example:

$$
\hat\pi_b=q(\sigma_b),\qquad
e_b=\operatorname{ilr}(\pi_b)-\operatorname{ilr}(\hat\pi_b),
$$

where $\pi_b$ is the true prevalence of validation bag $\sigma_b$. From these pairs, one estimates

$$
e_b\sim\mathcal N(\mu_e,\Sigma_e).
$$

Given the target:

$$
\hat z^Q=\operatorname{ilr}(q(U)),
$$

one generates

$$
z^{(r)}=\hat z^Q+\mu_e+\varepsilon^{(r)},
\qquad
\varepsilon^{(r)}\sim\mathcal N(0,\Sigma_e),
$$

and finally

$$
\pi^{(r)}=\operatorname{ilr}^{-1}(z^{(r)}).
$$

For each $\pi^{(r)}$, a bag is sampled from the validation set and its accuracy is computed. The resulting accuracy distribution thus includes:

- quantifier uncertainty;
- any quantifier bias, through $\mu_e$;
- variability in performance conditional on prevalence;
- variability due to the finite bag size.

This would be a genuine extension of SQS-H: rather than fixing prevalence at $\hat\pi^Q$, it propagates uncertainty about target prevalence through to accuracy.

## Estimating covariance by bootstrapping $U$

A second option is to estimate the covariance directly from the unlabeled target data. With the already trained quantifier held fixed, $B$ bootstrap samples of $U$ are generated and prevalence is recomputed for each of them:

$$
U^{*(b)}\sim\operatorname{Bootstrap}(U),\qquad
\hat\pi^{*(b)}=q(U^{*(b)}),
$$

for $b=1,\ldots,B$. The estimates are then transformed into `ilr` coordinates,

$$
z^{*(b)}=\operatorname{ilr}(\hat\pi^{*(b)}),
$$

and one estimates

$$
\widehat\Sigma_U=
\operatorname{Cov}\left(z^{*(1)},\ldots,z^{*(B)}\right).
$$

The logistic-normal distribution used to generate prevalence vectors then becomes

$$
Z\sim\mathcal N\left(
\operatorname{ilr}(\hat\pi^Q),
\widehat\Sigma_U
\right),
\qquad
\Pi=\operatorname{ilr}^{-1}(Z).
$$

This procedure is entirely label-free on the target data and yields a local covariance adapted to the specific set $U$. In particular, it captures:

- variability due to the finite size of $U$;
- the quantifier's sensitivity to the observed target examples;
- any directions of greater instability among classes;
- the dependence of variability on the current target prevalence.

If an isotropic distribution is desired, the matrix can be reduced to a single parameter:

$$
\hat\sigma^2=
\frac{\operatorname{tr}(\widehat\Sigma_U)}{K-1}.
$$

The full covariance, preferably regularized through shrinkage, remains more informative. Reducing it to a single $\sigma$ removes the directions of variability specific to different classes.

### Limitations of the target bootstrap

Bootstrapping $U$ estimates the sampling variability of $q(U)$ conditional on the already trained quantifier. It does not necessarily estimate the full quantification error and, in particular, does not adequately capture:

- the systematic bias of $q$;
- uncertainty due to quantifier training;
- error in the confusion or adjustment matrix estimated from the validation set;
- quantifier misspecification;
- possible violations of the PPS assumption.

A quantifier may be very stable under bootstrapping yet systematically wrong. In this case, one may have

$$
\widehat\Sigma_U\approx 0
$$

even when

$$
\lVert\hat\pi^Q-\pi^Q\rVert
$$

is large. The resulting intervals would therefore be too narrow. The risk increases with $|U|$: the sampling-uncertainty component tends to decrease, while bias or auxiliary-model error may persist.

The bootstrap may be used only to estimate $\widehat\Sigma_U$, as in the Gaussian Aitchison approach above, or the bootstrap prevalence estimates $\hat\pi^{*(b)}$ may be used directly. The latter option is an important nonparametric baseline for establishing whether the Gaussian approximation offers any real benefit.

## Comparing and combining sources of uncertainty

The different procedures are not equivalent, because they estimate different components:

| Approach | Main quantity estimated | Main limitation |
|---|---|---|
| Bootstrap of $U$ | Local variability of $q(U)$ | Does not observe the true error or bias |
| Validation bags with known prevalence | Empirical error $\pi-\hat\pi$ | May not be sufficiently local and requires separate data |
| Bootstrap of $q$ training | Uncertainty due to quantifier estimation | Computationally expensive and does not measure misspecification on its own |

A more complete formulation may separate target sampling uncertainty from quantifier uncertainty:

$$
\Sigma_{\mathrm{total}}
\approx
\Sigma_{\mathrm{target\ sampling}}
+
\Sigma_{\mathrm{quantifier}}.
$$

The first component can be estimated by bootstrapping $U$; the second through out-of-sample errors on validation bags or by bootstrapping the training of $q$. Mean bias $\mu_e$, by contrast, can only be estimated using known prevalence values, such as those of the validation bags. The resulting distribution would be

$$
Z\sim\mathcal N\left(
\operatorname{ilr}(\hat\pi^Q)+\hat\mu_e,
\widehat\Sigma_U+\widehat\Sigma_q
\right).
$$

Double counting must be avoided: the covariance of the residuals on validation bags already includes the bags' sampling variability. One possible decomposition is

$$
\widehat\Sigma_q=
\operatorname{PSD}\left(
\widehat\Sigma_{\mathrm{residuals}}
-
\overline{\widehat\Sigma_{\mathrm{bootstrap,val}}}
\right),
$$

followed by

$$
\widehat\Sigma_{\mathrm{total}}
=
\widehat\Sigma_q
+
\widehat\Sigma_{\mathrm{bootstrap},U}.
$$

Here, $\operatorname{PSD}$ denotes projection onto the cone of positive semidefinite matrices, which is necessary because estimation error may produce negative eigenvalues after subtraction. This decomposition is an approximation to be validated empirically, not a guaranteed identity.

## Beta, Dirichlet, or logistic-normal?

The Beta distribution is an alternative only in the binary case. For $K>2$, its natural counterpart is the Dirichlet distribution.

| Distribution | Applicability | Advantage | Main limitation |
|---|---:|---|---|
| Beta | $K=2$ only | Simple and interpretable | Does not generalize directly to multiclass settings |
| Dirichlet | $K\ge2$ | Simple and easy to sample from | Highly restrictive covariance structure |
| Logistic-normal | $K\ge2$ | Flexible covariance consistent with Aitchison geometry | More parameters and the zero problem |

If the mean of the Dirichlet distribution is fixed at $\hat\pi$,

$$
\Pi\sim\operatorname{Dirichlet}(\kappa\hat\pi),
$$

only one free parameter essentially remains: the concentration $\kappa$. All variances and covariances are determined by $\hat\pi$ and $\kappa$. It is therefore impossible to represent, for example, a quantifier that primarily confuses classes 1 and 2 but estimates class 3 very accurately.

The logistic-normal distribution instead permits a full $\Sigma$ matrix in $\mathbb R^{K-1}$. This greater flexibility is one of the historical reasons for preferring it to the Dirichlet distribution in compositional data analysis; the Dirichlet imposes strong independence and dependence properties. [Aitchison discusses these very limitations](https://academic.oup.com/jrsssc/article/34/2/129/6985271).

In the binary case:

- the logistic-normal becomes a logit-normal;
- the Beta distribution remains a valid baseline;
- neither distribution contains the other exactly.

I would therefore not say that the Beta distribution makes the Aitchison approach unnecessary. Rather, Beta and Dirichlet distributions are essential baselines for determining whether the flexibility of the logistic-normal is actually needed.

## An important detail about the “center”

If one sets

$$
\mu=\operatorname{ilr}(\hat\pi),
$$

then $\hat\pi$ is the Aitchison center, or equivalently the composition corresponding to the mean in `ilr` coordinates. In general, however,

$$
E[\Pi]\ne\hat\pi
$$

in the sense of the usual arithmetic mean on the simplex. This is not an error, but the paper will need to refer explicitly to the “Aitchison center,” rather than simply to the “mean prevalence.”

## The most important caveats

- **Zero prevalence.** `ilr` is not defined when a component is zero. An explicit strategy is required: a pseudocount, zero replacement, or a separate model for the faces of the simplex. This becomes especially important under severe shift.

- **Target-set size.** Quantifier variance depends on $|U|$. The validation bags used to learn $\Sigma_e$ should have the same size as the target set, or the covariance should be modeled as a function of $n$.

- **Bootstrap interpretation.** The bootstrap distribution of $q(U)$ approximates the estimator's sampling distribution; it is not automatically a posterior distribution of the true prevalence. Using its covariance around $\operatorname{ilr}(\hat\pi^Q)$ therefore requires a frequentist interpretation or an explicitly stated approximation.

- **Data leakage.** The quantifier must be evaluated out of sample on the validation bags. I would use data splitting or cross-fitting, as in the motivation for SQS-S.

- **Number of classes.** A full covariance matrix has $O(K^2)$ parameters. With many classes, shrinkage, a low-rank approximation, or an isotropic structure is required.

- **Choice of coordinates.** A full covariance matrix, transformed consistently, does not depend on the particular `ilr` basis; a diagonal covariance matrix, by contrast, may depend on it. Isotropy, $\Sigma=\sigma^2I$, is basis-invariant but much less expressive.

- **Normality is not guaranteed.** Quantifier errors may be skewed, multimodal, or heteroscedastic, especially near the boundaries. Normality must be verified, not assumed as a consequence of the transformation.

## The decisive baselines

I would compare at least:

1. the original SQS-H;
2. direct bootstrapping of prevalence estimates $q(U^{*(b)})$;
3. isotropic Gaussian Aitchison with $\sigma$ estimated by bootstrapping $U$;
4. Gaussian Aitchison with shrinkage covariance estimated by bootstrapping $U$;
5. Gaussian Aitchison with bias and covariance estimated from validation residuals;
6. hybrid Gaussian Aitchison, with separate target and quantifier components;
7. a Dirichlet distribution with calibrated concentration;
8. a Beta distribution in the binary case;
9. nonparametric bootstrapping of the `ilr` residuals.

The last baseline is particularly important: instead of assuming a Gaussian distribution, one samples directly from the observed errors

$$
e_b=\operatorname{ilr}(\pi_b)-\operatorname{ilr}(\hat\pi_b).
$$

If the logistic-normal matches or outperforms this bootstrap with greater regularity and lower cost, the parametric assumption is justified. If it does not, Aitchison geometry may still be useful, but without any need for Gaussianity.

In summary: **this is not needlessly complex, and the Beta distribution does not replace it**, especially in multiclass settings. Bootstrapping $U$ is a useful, local, label-free way to estimate sampling uncertainty, but it does not replace calibration on validation data when the goal is also to represent bias and overall quantifier uncertainty. The compelling novelty, therefore, would not simply be “placing a Gaussian distribution on the simplex”; it would be:

> Estimating and combining the different components of quantification uncertainty in Aitchison geometry and propagating them to the predictive distribution of performance.

This formulation links target bootstrapping to out-of-sample calibration, has a stronger motivation, and directly addresses the stated limitation of SQS-H.

## References

- [Uncertainty-Aware Classifier Accuracy Prediction under Prior Probability Shift](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf)
- [Geometry-Aware Bayesian Quantification via Compositional Data Analysis (KDE-Bayes)](papers/kde_bayes.pdf)
- [Aitchison. *The statistical analysis of compositional data*. JRSS-B 1982.](https://academic.oup.com/jrsssc/article/34/2/129/6985271)
- [SQS-B proposal](sqs_b.md)
- [BayCAP-KDEy proposal](baycap_kdey.md)
- [Possible extensions overview](extensions.md)
