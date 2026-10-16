# Extension 3: The uncertainty budget of CAP under PPS

## 0. Summary

**Question.** Which sources of uncertainty determine the error of a CAP estimate on a finite target sample, how does their balance change with the target size $\lvert U\rvert$ and the validation size $\lvert V\rvert$, and does the ranking of CAP methods reported in the paper at $\lvert U\rvert=1000$ hold across this range?

**Main claim (new hypothesis, supported only by a toy simulation).** The error $\hat A-A(h,U)$ of every quantification-based CAP method splits exactly into three parts: prevalence error, rate-matrix error, and the finite-sample fluctuation of the realised table. Some of these shrink as $\lvert U\rvert$ grows; others are fixed by the validation set $V$ and do not. A method's interval is wide only along the parts it simulates (*represented* parts); the parts it does not simulate (*missing* parts) shift its centre without widening it. Its coverage is therefore governed by the ratio $\eta=\sigma^2_{\text{out}}/\sigma^2_{\text{in}}$ of missing to represented variance, approximately $2\Phi\big(1.96/\sqrt{1+\eta}\big)-1$ for a $95\%$ interval. SQS-H and its bootstrap variant miss the parts fixed by $V$, so their $\eta$ grows with $\lvert U\rvert$ and their coverage falls; BayCAP misses the finite-sample fluctuation, so its $\eta$ is largest at small $\lvert U\rvert$ and large $\lvert V\rvert$. For SQS-H given the true prevalence, $\eta$ reduces to the design ratio

$$
r=\frac{\operatorname{Var}(e_2)}{\operatorname{Var}(e_3)}=\frac{\lvert U\rvert\sum_j(\pi^Q_j)^2}{\lvert V_j\rvert}
$$

(the second form when all classes have equal $d_j(1-d_j)$ and $\lvert V_j\rvert$), which depends only on $\lvert U\rvert$, $\lvert V\rvert$ and $\pi^Q$ and is used as the axis of the experiments. The paper's comparison at $\lvert U\rvert=1000$ is one point on this map.

**Role in the extended paper.** This extension is the common frame for the method extensions: the realised-table predictive of [[ext2_baycap|Extension 2]] adds the finite-$U$ term to BayCAP, [[sqs_b|SQS-B]] (Extension 8) replaces the prevalence term of SQS-H with a posterior, and [[baycap_kdey|BayCAP-KDEy]] (Extension 9) keeps the prevalence term only. It absorbs proposal A of [[extensions_to_evaluate]] (variance decomposition), which is promoted here.

**Candidate methods.** None is required for the main claim; the budget classifies existing and proposed methods. One candidate arises from it: **SQS-H with $U$ and $V$ resampling**, a non-Bayesian completion of SQS-H that resamples both the target sample (for $\hat\pi$) and the validation set (for the quantifier and for $C$) in every draw (§3.6).

**Status.**

| Item | Status |
|---|---|
| SQS-H, SQS-S, BayCAP at $\lvert U\rvert=1000$; coverage scored against the realised accuracy $A(h,U_i)$ | [paper] |
| `RQBS` (SQS-H with a bootstrap of $\hat\pi$, the variant run by `main`), `PrediQuant` (SQS-S), `ACC_BayesCAP` (BayCAP) | [implemented] (`cap-ml`) |
| `BayCAPKDEy` | [proposal, implemented], not evaluated |
| Realised-table BayCAP ([[ext2_baycap]]), SQS-B ([[sqs_b]]) | [proposal] |
| Error split, budget terms, coverage rule and ratios $\eta$, $r$, oracle ladder, SQS-H with $U$ and $V$ resampling | [new] |
| Toy simulation behind the numbers in this note (`scripts/toys/ext3_budget.py`) | [implemented] |
| `target_size` study | [proposal] |
| All hypotheses (H1–H4) | [new hypothesis] |

**Reader's guide.** Part I derives the error split (§2), the variance of each part (§3.1–§3.3), which parts each method represents (§3.4), and how this determines coverage (§3.5–§3.6). Part II states the contributions, the hypotheses, and where they enter the paper. Part III specifies the experiments and components. The toy results are in Appendix A.

# Part I: Understanding

## 1. Setting and estimand

We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$, $d=\operatorname{diag}(C)$, population table $T=C\operatorname{diag}(\pi^Q)$. Under PPS, $C$ is the same on $V$ and on the target.

**Protocol.** The paper's APP draws a bag $U$ of size $n=\lvert U\rvert$ with replacement from a labelled test pool, with class counts $n_j=n\pi^Q_j$ (rounded). Because sampling is with replacement, the number of correct predictions in class $\omega_j$ is exactly $c_j\sim\operatorname{Bin}(n_j,d^{\text{pool}}_j)$, where $d^{\text{pool}}$ is the diagonal of the rate matrix of $h$ on the pool. In experiments the pool is labelled, so $d^{\text{pool}}$ is known; it is the relevant "true" $d$ for the scored quantity.

**Estimand.** Every interval in this note is scored against the **realised accuracy** $A(h,U)=\frac1n\sum_j c_j$, computed from the realised table of the bag (`true_cts` in `main`), as in the paper. The **population accuracy** $d^\top\pi^Q=A(C\operatorname{diag}\pi^Q)$ appears only as an oracle reference. The difference between the two is the estimand gap of [[ext2_baycap]] §2.4.

*Intuition.* Even with $\pi^Q$ and $C$ known, the realised accuracy on $n$ specific instances is random; any interval that ignores this is too narrow for the realised accuracy.

## 2. The error split

*Intuition.* Every quantification-based estimate has the form "estimated prevalence times estimated per-class accuracy"; its error is wrong prevalence, plus wrong per-class accuracy, plus the luck of this particular bag.

**Form of the estimate.** The accuracy of a table $T=C\operatorname{diag}(\pi)$ is its trace, $\sum_jC_{jj}\pi_j=d^\top\pi$. Every method considered here estimates accuracy as $\hat A=\sum_j\tilde\pi_j\tilde d_j$, with $\tilde\pi$ an estimate of $\pi^Q$ and $\tilde d$ an estimate of $d$ from $V$:

- SQS-H: the mean accuracy of its bags is $\sum_j(n\hat\pi_j/n)\,\tilde d_j$, where $\tilde d_j$ is the fraction of the class-$\omega_j$ instances of $V$ that $h$ classifies correctly;
- RQBS and SQS-B: the same for each draw $\hat\pi^{(r)}$, averaged over draws;
- BayCAP and BayCAP-KDEy: $\sum_jC^{(r)}_{jj}\pi^{(r)}_j$ for each posterior draw, averaged over draws.

**Derivation.** Add and subtract $\sum_j\pi^Q_j\tilde d_j$ and $\sum_j\pi^Q_jd_j$:

$$
\hat A-A(h,U)=\Big(\sum_j\tilde\pi_j\tilde d_j-\sum_j\pi^Q_j\tilde d_j\Big)+\Big(\sum_j\pi^Q_j\tilde d_j-\sum_j\pi^Q_jd_j\Big)+\Big(\sum_j\pi^Q_jd_j-A(h,U)\Big).
$$

The inner terms cancel, so the identity is exact for every bag:

$$
\hat A-A(h,U)=\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,\tilde d_j}_{e_1\ \text{prevalence}}+\underbrace{\sum_j\pi^Q_j(\tilde d_j-d_j)}_{e_2\ \text{rate matrix}}+\underbrace{d^\top\pi^Q-A(h,U)}_{e_3\ \text{finite }U}.
$$

For methods that average over draws, the identity holds for each draw and therefore for the average. The order of the add-and-subtract steps is a choice: the second-order cross term $\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d_j)$ is included in $e_1$ (because $e_1$ uses $\tilde d$, not $d$); the other order would move it into $e_2$. In an experiment all three parts are computable with labels: $\pi^Q$ is the bag's true prevalence and $d=d^{\text{pool}}$. This is the basis of the oracle ladder of §7.

## 3. The budget

### 3.1 Sources of each part

*Intuition.* Each part is driven by one of two independent random draws, the target sample $U$ or the validation set $V$; the parts driven by $U$ shrink as $U$ grows, those driven by $V$ do not.

The prevalence error is split further by its source. Let $\pi^\infty(V)$ be the prevalence that the quantifier fitted on this $V$ would return on an infinitely large target sample with prevalence $\pi^Q$. Then

$$
\tilde\pi-\pi^Q=\big(\tilde\pi-\pi^\infty(V)\big)+\big(\pi^\infty(V)-\pi^Q\big),
\qquad
e_1=e_1^U+e_1^V,
$$

with $e_1^U=\sum_j(\tilde\pi_j-\pi^\infty_j(V))\tilde d_j$ and $e_1^V=\sum_j(\pi^\infty_j(V)-\pi^Q_j)\tilde d_j$. Both are parts of $e_1$; neither overlaps with $e_2$ or $e_3$. The four parts are:

| Part | Meaning | Source | Behaviour |
|---|---|---|---|
| $e_3$ | luck of the bag: which of its instances $h$ classifies correctly | $U$ | mean $0$, variance $\propto1/n$ |
| $e_1^U$ | quantifier noise from seeing only $n$ target instances | $U$ | mean about $0$, variance $\propto1/n$ |
| $e_1^V$ | quantifier error from fitting it on a finite $V$, plus its systematic bias $b_q$ | $V$ | fixed given $V$; variance $\propto1/\lvert V\rvert$ over draws of $V$ |
| $e_2$ | error of the rates of $h$ measured on a finite $V$ | $V$ | fixed given $V$; variance $\propto1/\lvert V\rvert$ over draws of $V$ |

$e_3$ is the sampling variability of the **estimand**: it exists even with $\pi^Q$ and $C$ known, and it is the estimand gap of [[ext2_baycap]] §2.4. $e_1^U$, $e_1^V$ and $e_2$ are sampling errors of the **estimator**: $e_1^U$ from sampling $U$, $e_1^V$ and $e_2$ from sampling $V$. $e_1^V$ is the quantifier's analogue of $e_2$: the first is the error of the surrogate classifier's rates, the second that of the rates of $h$.

### 3.2 Variance of each part

All variances below are first order in the estimation errors (delta method) and assume the paper's protocol: fixed class counts $n_j=n\pi^Q_j$, instances drawn with replacement.

**$e_3$.** Each of the $n_j$ instances of class $\omega_j$ in the bag is classified correctly independently with probability $d_j$, so the number of correct predictions is $c_j\sim\operatorname{Bin}(n_j,d_j)$, independently across classes. Hence

$$
\mathbb E[A(h,U)]=\frac1n\sum_jn_jd_j=d^\top\pi^Q,
\qquad
\operatorname{Var}(e_3)=\operatorname{Var}(A(h,U))=\frac1{n^2}\sum_jn_jd_j(1-d_j)=\frac1n\sum_j\pi^Q_jd_j(1-d_j).
$$

**$e_2$.** $\tilde d_j=k_j/\lvert V_j\rvert$, where $k_j\sim\operatorname{Bin}(\lvert V_j\rvert,d_j)$ is the number of class-$\omega_j$ validation instances that $h$ classifies correctly, independently across classes. With the constant weights $\pi^Q_j$,

$$
\operatorname{Var}(e_2)=\sum_j(\pi^Q_j)^2\operatorname{Var}(\tilde d_j)=\sum_j\frac{(\pi^Q_j)^2d_j(1-d_j)}{\lvert V_j\rvert}.
$$

This variance is over draws of $V$; for a given $V$, $e_2$ is a single number. The weights are squared here and not in $e_3$ because in $e_3$ the prevalence enters through the counts $n_j$.

**$e_1$, binary case with ACC.** Prevalence errors sum to zero, so $e_1=(\tilde\pi_1-\pi^Q_1)(\tilde d_1-\tilde d_0)$ and $\operatorname{Var}(e_1)\approx(d_1-d_0)^2\operatorname{Var}(\tilde\pi_1)$; the whole term vanishes when $d=\beta\mathbf 1$, the paper's condition for unbiased cross-validation (Eq. 2). Let the surrogate classifier have rates $S_{10}=P(\hat Y_s=\omega_1\mid Y=\omega_0)$ and $S_{11}$, estimated on $V$ as $\hat S$, and let $\hat p$ be the fraction of $U$ that it predicts as $\omega_1$. ACC returns

$$
\hat\pi_1=\frac{\hat p-\hat S_{10}}{\hat S_{11}-\hat S_{10}}.
$$

Two independent sources of noise enter it:

- **from $U$** ($e_1^U$): with fixed class counts, $\operatorname{Var}(\hat p)=\frac1n\big[\pi^Q_0S_{10}(1-S_{10})+\pi^Q_1S_{11}(1-S_{11})\big]$, so

$$
\operatorname{Var}(e_1^U)\approx\frac{(d_1-d_0)^2}{(S_{11}-S_{10})^2}\cdot\frac{\pi^Q_0S_{10}(1-S_{10})+\pi^Q_1S_{11}(1-S_{11})}{n};
$$

- **from $V$** ($e_1^V$): $\partial\hat\pi_1/\partial S_{10}=-\pi^Q_0/(S_{11}-S_{10})$ and $\partial\hat\pi_1/\partial S_{11}=-\pi^Q_1/(S_{11}-S_{10})$, with $\operatorname{Var}(\hat S_{1j})=S_{1j}(1-S_{1j})/\lvert V_j\rvert$, so

$$
\operatorname{Var}(e_1^V)\approx\frac{(d_1-d_0)^2}{(S_{11}-S_{10})^2}\left[\frac{(\pi^Q_0)^2S_{10}(1-S_{10})}{\lvert V_0\rvert}+\frac{(\pi^Q_1)^2S_{11}(1-S_{11})}{\lvert V_1\rvert}\right].
$$

$e_1^U$ has the same structure as $e_3$ and $e_1^V$ the same structure as $e_2$, with the rates of the surrogate classifier in place of those of $h$, amplified by $1/(S_{11}-S_{10})^2$ (an ill-conditioned surrogate inflates both) and scaled by $(d_1-d_0)^2$. The multiclass case replaces these factors with the inverse of the surrogate's rate matrix and the vector $d$.

**Correlations.** Parts driven by $U$ and parts driven by $V$ are uncorrelated, because $U$ and $V$ are drawn independently. Within a group they can be correlated: $e_3$ and $e_1^U$ if $h$ and the surrogate tend to misclassify the same target instances, $e_2$ and $e_1^V$ if they tend to misclassify the same validation instances. The toy of Appendix A assumes $h$ and the surrogate conditionally independent given $Y$, which removes both correlations; on real data the oracle ladder measures them.

### 3.3 The budget

Grouping the parts by source gives

$$
\operatorname{MSE}(\hat A)\approx\underbrace{\operatorname{Var}(e_3)+\operatorname{Var}(e_1^U)}_{\text{from }U,\ \propto1/n}+\underbrace{\operatorname{Var}(e_2)+\operatorname{Var}(e_1^V)+b_q^2}_{\text{from }V,\ \text{independent of }n},
$$

where $b_q$ is the quantifier's systematic bias propagated to accuracy (contained in $e_1^V$, written separately because it does not average out over draws of $V$). The approximation is first order and drops the within-group correlations of §3.2.

### 3.4 Which parts each method represents

*Intuition.* A method's samples vary only along the sources it simulates; the other sources move the centre of its interval without widening it.

| Method | Status | Spread comes from | $e_3$ | $e_1^U$ | $e_1^V$ | $e_2$ |
|---|---|---|---|---|---|---|
| SQS-H as in the paper (§4.1) | [paper] | bags of size $n$ at the counts $n\hat\pi_j$, drawn from $V$ | yes | no | no | no |
| `RQBS` (SQS-H as run by `main`) | [implemented] | bootstrap of $\hat\pi$ on $U$, one bag per draw | yes | yes | no | no |
| SQS-S (§4.2) | [paper] | bags whose true prevalences lie near $\hat\pi$ in estimated-prevalence space | yes | partly (hypothesis) | no | no |
| SQS-B | [proposal] | posterior of $\pi^Q$, one bag per draw | yes | yes | if the posterior models the quantifier's rates (BayesianCC does) | no |
| BayCAP (§5) | [paper] | joint posterior of $(\pi^Q,C)$, population pushforward | no | yes | yes | yes |
| Realised-table BayCAP | [proposal] ([[ext2_baycap]]) | as BayCAP, plus a sampled realised table | yes | yes | yes | yes |
| BayCAP-KDEy | [proposal, implemented] | posterior of $\pi^Q$, fixed $C^h$ | no | yes | through the posterior | no |
| SQS-H with $U$ and $V$ resampling | [new] | bootstrap of $U$ and of $V$, one bag per draw | yes | yes | yes | yes |

Why SQS-H represents $e_3$: its bags have the same class counts as the target bag and draw class-$\omega_j$ instances from $V_j$, so the number of correct predictions in class $\omega_j$ is $\operatorname{Bin}(n_j,\tilde d_j)$, the same mechanism that generates the target bag with $d_j$ replaced by $\tilde d_j$. SQS-H is a simulator of the bag's luck, with $V$ as a stand-in for the class-conditional distributions. Why it does not represent $e_2$: every bag uses the same $\tilde d$, so $e_2$ is an offset shared by all draws. BayCAP's posterior describes the uncertainty on $(\pi^Q,C)$, but each draw is a population accuracy $d^{(r)\top}\pi^{(r)}$ with no binomial luck of the bag, so it has no $e_3$ ([[ext2_baycap]] §3). BayCAP estimates $\pi^Q$ from the predictions of $h$, not of a surrogate, so its $e_1$ differs in size from that of the SQS family.

### 3.5 From the budget to coverage

*Intuition.* An interval is as wide as the noise it simulates; if the actual error also contains noise it does not simulate, it misses.

Write $\sigma^2_{\text{in}}$ for the sum of the variances of the parts a method represents and $\sigma^2_{\text{out}}$ for the sum of those it misses. Assume that the represented parts are simulated at the right size, that all parts are approximately normal (sums of many binomials, by the central limit theorem), and that represented and missing parts are independent. Then:

- the method's samples are approximately normal with standard deviation $\sigma_{\text{in}}$, and the 2.5% and 97.5% percentiles of $N(\mu,\sigma^2)$ are $\mu\pm1.96\,\sigma$, so the interval is approximately $\hat A\pm1.96\,\sigma_{\text{in}}$;
- the interval covers $A(h,U)$ when $\lvert\hat A-A(h,U)\rvert\le1.96\,\sigma_{\text{in}}$, and $\hat A-A(h,U)=e_1+e_2+e_3$ has variance $\sigma^2_{\text{in}}+\sigma^2_{\text{out}}$;
- dividing by its standard deviation,

$$
\text{coverage}\approx P\Big(\lvert Z\rvert\le\frac{1.96\,\sigma_{\text{in}}}{\sqrt{\sigma^2_{\text{in}}+\sigma^2_{\text{out}}}}\Big)=2\Phi\Big(\frac{1.96}{\sqrt{1+\eta}}\Big)-1,
\qquad
\eta=\frac{\sigma^2_{\text{out}}}{\sigma^2_{\text{in}}},
$$

with $Z$ standard normal. Checks: $\eta=0$ gives $0.95$; $\eta\to\infty$ gives $0$.

**Worked example: SQS-H with the true prevalence** (the oracle-$\pi$ reference). Its bags use the true counts, so $e_1=0$; it represents $e_3$ and misses $e_2$. Its interval is approximately $\tilde d^\top\pi^Q\pm1.96\,\sigma_3$: the width is right, the centre is off by $e_2$, because $\tilde d^\top\pi^Q=d^\top\pi^Q+e_2$ and $A(h,U)=d^\top\pi^Q-e_3$, so $A(h,U)-\text{centre}=-(e_2+e_3)$. Hence $\eta=\operatorname{Var}(e_2)/\operatorname{Var}(e_3)=r$, and with §3.2

$$
r=\frac{\sum_j(\pi^Q_j)^2d_j(1-d_j)/\lvert V_j\rvert}{\frac1n\sum_j\pi^Q_jd_j(1-d_j)},
$$

which reduces to $n\sum_j(\pi^Q_j)^2/\lvert V_j\rvert$ when $d_j(1-d_j)$ and $\lvert V_j\rvert$ are equal across classes. In the toy (Appendix A), the closed form $2\Phi(1.96/\sqrt{1+r})-1$ is close to the simulated coverage of the oracle-$\pi$ reference.

**Conditional on one $V$.** The formula averages over draws of $V$. For a given $V$, $e_2$ is a constant and the coverage of the same interval is $\Phi(1.96-e_2/\sigma_3)-\Phi(-1.96-e_2/\sigma_3)$: close to $0.95$ if that $V$ happens to give a small $e_2$, close to $0$ if $\lvert e_2\rvert\gg\sigma_3$. Coverage measured on many bags from a single $V$ therefore measures that $V$, not the method; experiments must repeat over $V$ seeds.

**Ratio for each method.** With $\sigma_k^2$ the variance of part $e_k$:

| Method | Represented | Missing | $\eta$ |
|---|---|---|---|
| SQS-H with true $\pi^Q$ (reference) | $e_3$ | $e_2$ | $\sigma_2^2/\sigma_3^2=r$ |
| SQS-H | $e_3$ | $e_1^U$, $e_1^V$, $e_2$ | $(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)/\sigma_3^2$ |
| RQBS | $e_3$, $e_1^U$ | $e_1^V$, $e_2$ | $(\sigma_{1V}^2+\sigma_2^2)/(\sigma_3^2+\sigma_{1U}^2)$ |
| BayCAP (population) | $e_1^U$, $e_1^V$, $e_2$ | $e_3$ | $\sigma_3^2/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ |
| Realised-table BayCAP; SQS-H with $U$ and $V$ resampling | all | none | $0$ |

$\eta$ equals $r$ only for the reference; for the other methods, the quantifier's terms enter as well. All $\sigma_k$ can be estimated in an experiment with labels (E3.1), so each method's $\eta$ is measurable and the coverage rule is testable (H3).

### 3.6 The regime hypothesis

*Intuition.* As $n$ grows, the parts driven by $U$ shrink and those driven by $V$ stay, so methods that simulate only the first kind become overconfident, and methods that miss the bag's luck become well calibrated.

From §3.2 and the table of §3.5 (new hypothesis):

- **SQS-H and RQBS** miss parts driven by $V$ and represent parts driven by $U$: their $\eta$ grows linearly in $n$ and their coverage falls towards $0$ for $n\gg\lvert V_j\rvert$. SQS-H also misses $e_1^U$, so it under-covers already at small $n$ when the prevalence term is large (large $\lvert d_1-d_0\rvert$ or ill-conditioned surrogate).
- **BayCAP** misses $e_3$. As $n$ grows, $\sigma_3^2$ and $\sigma_{1U}^2$ shrink while $\sigma_{1V}^2+\sigma_2^2$ stays, so $\eta\to0$ and coverage approaches nominal. At small $n$, or when $V$ is large, $\eta\to\sigma_3^2/\sigma_{1U}^2$: BayCAP's coverage then has a floor set by how the quantifier's own noise compares with the bag's luck, independent of $n$. This is the within-dataset form of hypothesis H2 of [[ext2_baycap]].
- **Methods that represent all parts** (realised-table BayCAP, SQS-H with $U$ and $V$ resampling) have $\eta=0$ and are near nominal across the range, apart from quantifier bias $b_q$.
- **$r$ as the design axis.** $r$ depends only on $n$, $\lvert V_j\rvert$, $\pi^Q$ and $d$, so it can be computed for every cell of an experiment before running any method; the $\eta$ of each method is then measured. The paper's comparison at $\lvert U\rvert=1000$ fixes one value of $r$ per dataset.

**SQS-H with $U$ and $V$ resampling** follows directly: for every draw, resample $U$ (as `RQBS` does) and resample $V$ with stratification, refit the quantifier's rate estimates and the rates of $h$ on the resampled $V$, and draw one bag of size $n$ at the resulting prevalence. It is the non-Bayesian counterpart of the realised-table BayCAP. Its cost is one quantifier refit per draw; for aggregative quantifiers such as PACC only the misclassification rates need refitting, not the surrogate classifier, which is an approximation to be checked.

The toy simulation (Appendix A) supports all four points in a binary setting.

### 3.7 Caveats

- **$\pi^Q$ and $C$ are dependent in BayCAP's posterior**, through the observed target predictions $n^t\approx n\,C\pi^Q$. The budget ignores this covariance; the oracle ladder measures it implicitly.
- **Nonlinear metrics.** For macro-F1 the split of §2 does not hold exactly and the budget holds only to first order (delta method). For macro-K, which depends on $\pi^Q$ only weakly ([[ext1_metrics]]), $e_1$ should be small and $e_2$, $e_3$ should dominate; this is a sharp test of the budget.
- **Natural sampling.** Under natural (i.i.d.) sampling instead of APP, the class counts are multinomial, which adds a term $\frac1n\operatorname{Var}_{\pi^Q}(d_Y)$ to $e_3$. All experiments here keep APP, as in the paper.
- **Quantifier bias** is not a variance and is represented by no method. If the completed methods still under-cover at large $n$, bias is the likely cause; the oracle ladder separates it from $e_1^{V}$ by repeating over $V$ seeds.
- **Small $n$.** Rounding $n\pi^Q_j$ can leave classes empty; the conventions of [[ext1_metrics]] for rare and absent classes apply.

# Part II: For the paper

## 4. Contributions

1. **An exact error split and a budget** for CAP under PPS, which makes explicit that the realised table is the estimand and identifies which uncertainty terms shrink with $\lvert U\rvert$ and which do not.
2. **A unifying classification** of existing and proposed uncertainty-aware CAP methods by the parts they represent (§3.4), and a coverage rule in terms of the ratio $\eta$ of missing to represented variance (§3.5), which explains why the paper's methods are calibrated in different regimes.
3. **A regime map**, empirical, of coverage as a function of $(\lvert U\rvert,\lvert V\rvert)$, showing whether the paper's ranking at $\lvert U\rvert=1000$ depends on the design ratio $r$.
4. **Completed methods**: evidence that methods representing all terms (realised-table BayCAP from Extension 2, and SQS-H with $U$ and $V$ resampling) are calibrated across the map.

## 5. Hypotheses

Primary:

- **H1 (SQS-type intervals degrade with $n$).** At fixed $V$, the coverage of SQS-H and RQBS decreases as $\lvert U\rvert$ grows, faster when $\lvert V_j\rvert$ is small. *Confirmed if* in E3.2 the coverage of RQBS decreases monotonically with $\lvert U\rvert$ on most (dataset, classifier) pairs and the decrease is larger at the smaller $\lvert V\rvert$ level. *Refuted if* coverage is flat in $\lvert U\rvert$; then $e_2$ and $e_1^{V}$ are negligible on real data and the budget adds little to Extension 2.
- **H2 (BayCAP under-covers at small $n$ and large $V$).** The coverage of BayCAP with the population pushforward increases with $\lvert U\rvert$ at fixed $V$ and is below nominal where its $\eta=\sigma_3^2/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ is large (§3.6). *Confirmed if* in E3.2 coverage increases with $\lvert U\rvert$ at fixed $V$ and decreases with $\lvert V\rvert$ at fixed $\lvert U\rvert$. *Refuted if* BayCAP over-covers at every $r$; then the fixed-count likelihood (source (i) of [[ext2_baycap]] §3) or prior effects dominate, and the budget does not explain BayCAP.
- **H3 (the budget accounts for the error).** In E3.1, $\operatorname{RMSE}(e_3)$ scales as $n^{-1/2}$, $\operatorname{RMSE}(e_2)$ is constant in $n$ and scales as $\lvert V\rvert^{-1/2}$, and $e_1$ is the sum of a part shrinking with $n$ and a part fixed given $V$. *Confirmed if* the fitted slopes on log–log axes are within $\pm0.15$ of the predicted ones and, for every method, the coverage plotted against its $\eta$ (computed from the variances of the parts estimated in E3.1) lies within $\pm0.05$ of $2\Phi(1.96/\sqrt{1+\eta})-1$ on most (dataset, classifier) pairs. *Refuted if* coverage does not follow $\eta$; then non-normality, the within-group correlations of §3.2 or quantifier bias $b_q$ matter, and the budget must be refined.

Secondary:

- **H4 (completing the budget restores coverage).** Methods that represent all terms are within $\pm0.03$ of nominal coverage across the grid, while each partial method fails where its missing term dominates. *Confirmed if* in E3.2 the realised-table BayCAP and SQS-H with $U$ and $V$ resampling meet this tolerance at every $(\lvert U\rvert,\lvert V\rvert)$ level. *Refuted if* they fail at large $n$ as well; then quantifier bias $b_q$ is the dominant fixed term and no variance-based correction suffices.

## 6. Paper outline

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §3 (problem), new subsection "Target estimand and uncertainty budget" | §1–§3.3 of this note | The CAP error splits into prevalence, rate-matrix and finite-$U$ parts with different scaling. | derivation; F0 (toy); E3.1 |
| §4–§5 (methods), one table and the coverage rule | §3.4–§3.5 | Each method represents only some parts; its coverage follows from the ratio $\eta$ of missing to represented variance. | tables of §3.4–§3.5; E3.1 (H3) |
| §5, new subsections from Extensions 2 and 8 | completed methods | Adding the missing term restores calibration. | E3.2 (H4); [[ext10_baycap_priors_tempering]] E10.2 |
| §6 (experiments), new subsection "Target and validation size" | E3.1, E3.2 | The paper's ranking at $\lvert U\rvert=1000$ is regime-dependent. | F1, F2, T1 |
| §7 (conclusions) | one paragraph | Practical guidance: which method to use given $\lvert U\rvert$ and $\lvert V\rvert$. | F2 |

Wording if results are mixed: if H1 is refuted, the budget becomes a short section explaining the estimand gap, and Extension 2 carries the main claim; if H3 is refuted but H1 and H2 hold, the regime map is presented empirically without the closed-form ratio.

### 6.1 Figures and tables

| Item | Question | Content | What it must show |
|---|---|---|---|
| F0 (optional) | toy | coverage vs $r$ for each method of Appendix A | the predicted shape of F2 |
| F1 | E3.1 | RMSE of $e_1$, $e_2$, $e_3$ vs $\lvert U\rvert$ (log–log), one panel per $\lvert V\rvert$ level, one line per dataset (LR) | the scaling of H3 |
| F2 | E3.2 | (a) coverage vs $r$ (log), one panel per method, one point per (dataset, classifier, $\lvert U\rvert$, $\lvert V\rvert$), dashed line at $0.95$; (b) coverage vs each method's $\eta$ (log), all methods in one panel, with the curve $2\Phi(1.96/\sqrt{1+\eta})-1$ | (a) the regime map, H1, H2, H4; (b) the coverage rule, H3 |
| T1 | E3.2 | mean Winkler rank per method, one column block per $\lvert U\rvert$ at full $V$; oracle references as unranked columns (`tables/rank.py`) | whether the ranking changes with $\lvert U\rvert$ |
| T2 (appendix) | E3.1 | per-dataset bias and standard deviation of $\hat\pi$ (L1) and $\hat A$ vs $\lvert U\rvert$ | quantifier bias $b_q$, needed to interpret H4 |

Macro-F1 and macro-K versions of F2 go to the appendix.

# Part III: For the implementation

## 7. Experiments

Common to all: APP bags drawn with replacement from the test pool, as in `main`; all intervals scored against the realised table of each bag; measures as in the paper (AE, coverage, amplitude, Winkler at $\alpha=0.05$) plus the error split of §2; metrics of [[ext1_metrics]]. Coverage is reported with a standard error clustered by $V$ seed, since bags sharing a $V$ are not independent. Dataset subset shared with [[ext2_baycap]]: binary and multiclass datasets covering small and large $\lvert V_j\rvert$, well- and ill-conditioned $C$, and one large-$K$ dataset. Classifiers: LR and RF (all five in the pilot). Grids:

- $\lvert U\rvert\in\{100,300,1000,3000,10000\}$, log-spaced, including the paper's $1000$; bags larger than the pool repeat instances, which is consistent with sampling with replacement;
- $\lvert V\rvert$ levels: the full $V$ and a stratified subsample of $1/4$ (and $1/16$ on the largest datasets), nested ($V_{1/16}\subset V_{1/4}\subset V$) and seeded; the split $L/V$, the classifier $h$, and the allocation $\rho$ used inside SQS-S are unchanged, so Extension 6 stays separate;
- $S_V=5$ outer seeds for each subsampled $V$ level (for the full $V$, the seed changes the $L/V$ split; $h$ is retrained per seed).

Order and gates: pilot → E3.1 → E3.2 → E3.3.

### E3.1: budget validation

- **Goal / hypothesis:** measure $e_1$, $e_2$, $e_3$ on real data and their scaling; H3.
- **Inputs:** subset datasets; quantifier PACC as in the paper; methods RQBS and SQS-H (point $\hat\pi$), which are cheap.
- **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed; three fixed prevalences per dataset: uniform $\mathbf 1/K$, one skewed vector (a seeded draw from $\operatorname{Dir}(\mathbf 1)$), one near-vertex vector ($0.9$ on the rarest validation class, the rest uniform); $R=100$ bags per (prevalence, $\lvert U\rvert$, $\lvert V\rvert$, seed), i.e. $500$ per cell.
- **Oracle ladder (references, never methods):** (a) true $\pi^Q$: SQS bags at the true counts, removing $e_1$; (b) $d^{\text{pool}}$ from the labelled pool, removing $e_2$; (c) scoring against $d^{\text{pool}\top}\pi^Q$, isolating $e_3$.
- **Outputs:** per bag: $\hat\pi$, $\hat A$, $e_1^U$, $e_1^V$, $e_2$, $e_3$, interval bounds; prevalence sidecars. $\pi^\infty(V)$ is approximated, once per ($V$ seed, prevalence), by the estimate of the quantifier fitted on that $V$ on one very large bag (e.g. $10^5$ instances drawn with replacement from the pool at $\pi^Q$); then $e_1^V$ follows from its definition in §3.1 and $e_1^U=e_1-e_1^V$.
- **Measures:** RMSE and mean of each $e_k$; bias and standard deviation of $\hat\pi$ and $\hat A$; log–log slopes vs $\lvert U\rvert$ and $\lvert V\rvert$.
- **Interpretation:** H3; the estimated variances of $e_1^U$, $e_1^V$, $e_2$, $e_3$ give each method's $\eta$ for E3.2, and the mean of $e_1$ over $V$ seeds estimates the quantifier bias $b_q$.
- **Visualization:** F1, T2.
- **Cost:** no NUTS; dominated by bag sampling, of order $m\cdot n$ per bag.

### E3.2: regime map

- **Goal / hypothesis:** coverage of every method across $(\lvert U\rvert,\lvert V\rvert)$; H1, H2, H4.
- **Inputs:** subset datasets; APP prevalences ($100$ uniformly drawn prevalence vectors per cell, as in the paper, with fixed seeds shared by all methods and cells).
- **Methods:** SQS-H (point $\hat\pi$), RQBS, SQS-S (`PrediQuant`, after the fix of [[work_plan]] FIX-1), BayCAP (population pushforward), realised-table BayCAP (same draws, no extra NUTS run), SQS-B, BayCAP-KDEy, SQS-H with $U$ and $V$ resampling; oracle references from E3.1.
- **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed.
- **Outputs:** metric samples per bag and method; $r$ per cell; $\eta$ per method and cell, from the variances of the parts estimated in E3.1 for the same cell.
- **Measures:** coverage (clustered standard error), amplitude, Winkler, AE.
- **Interpretation:** H1 from the SQS family, H2 from BayCAP, H4 from the completed methods; the cell $\lvert U\rvert=1000$, full $V$ reproduces the paper's setting (comparability check with `main`).
- **Visualization:** F2, T1.
- **Cost:** NUTS dominates: one BayCAP run per bag, shared by both pushforwards; BayesianKDEy and SQS-B add their own posterior runs. The pilot (two datasets, LR, $20$ bags) estimates the cost; if needed, the BayCAP family runs on $S_V=3$ seeds.

### E3.3: robustness

- **Goal / hypothesis:** whether the regime map depends on the prevalence, on $K$, or on the metric.
- **Inputs:** E3.1 and E3.2 outputs; no new runs except macro-F1 and macro-K computed from stored tables.
- **Varied:** prevalence type (uniform, skewed, near-vertex); binary vs multiclass; metric.
- **Measures:** coverage vs $r$ per stratum.
- **Interpretation:** macro-K should show small $e_1$ ([[ext1_metrics]]); near-vertex prevalences raise $\sum_j(\pi^Q_j)^2$ and therefore $r$ at the same $\lvert U\rvert$.
- **Visualization:** appendix versions of F2.
- **Cost:** analysis only.

## 8. Components, tests, and reproducibility

Components:

- `experiments/sampling.py`: `fixed_prev_grid(pis, n_U_grid, n_repeats)` and APP at any $n$; per-bag seeds from `(seed, dataset, bag_id, n_U)`;
- bag size taken from the target: `RQBS` and `PrediQuant` are built with `sample_size=SAMPLE_SIZE=1000` in `src/uncawe/experiments/methods.py`; the study passes $n$ explicitly or uses `n_U = len(X)`;
- nested, seeded stratified subsampling of $V$; the quantifier is refitted on each subsample, $h$ is not;
- error split per bag (`evaluation/budget.py`): $e_1^U$, $e_1^V$, $e_2$, $e_3$ from the method's $\tilde\pi$ and $\tilde d$, the bag's true prevalence, $d^{\text{pool}}$ and the large-bag estimate of $\pi^\infty(V)$;
- `SQSHResampled` [new]: SQS-H with $U$ and $V$ resampling (§3.6), following the method conventions of [[work_plan]] §V.2;
- oracle classes `OracleSQS` (true $\pi^Q$, validation bags) and `OracleC` (true $\pi^Q$, fixed $C^h$), reported as references, plus scoring against the population table.

Tests:

- the error split satisfies $e_1+e_2+e_3=\hat A-A(h,U)$ exactly on random tables;
- the closed-form variances of §3.2 match simulated variances of $e_3$, $e_2$, $e_1^U$, $e_1^V$ for a binary ACC configuration (as in the toy);
- with $\tilde\pi=\pi^Q$ and $\tilde d=d^{\text{pool}}$, $e_1=e_2=0$;
- `fixed_prev_grid` returns bags with the requested class counts and size;
- nested subsamples satisfy $V_{1/16}\subset V_{1/4}\subset V$ and are stratified;
- `SQSHResampled` with resampling of $V$ switched off reproduces RQBS for the same seed.

Reproducibility and comparability:

- new study `target_size`, writing to `cap.env["OUT_DIR"]/uncawe/target_size`; `main`, `binform`, `predist` and their outputs are unchanged;
- every row records $n$, $\lvert V\rvert$ level, $V$ seed, prevalence type, bag seed, `config_hash`, `schema_version`;
- the cell ($\lvert U\rvert=1000$, full $V$, seed of `main`) must reproduce `main` within Monte Carlo error for the methods present in both;
- outer seeds that change the $L/V$ split retrain $h$; these rows are not comparable with `main` and are flagged.

# Appendix

## A. Toy simulation

`uv run python scripts/toys/ext3_budget.py` (defaults: $40$ validation sets, $50$ bags per validation set, $m=1000$ samples per interval, seed $0$; about $3$ minutes). Binary, toy only: $\pi^Q_1=0.5$ with fixed counts, $n_s$ validation instances per class, quantifier ACC on a surrogate classifier with rates $P(\hat Y_s=\omega_1\mid Y=\omega_0)=0.25$, $P(\hat Y_s=\omega_1\mid Y=\omega_1)=0.75$, conditionally independent of $h$ given $Y$ (a simplification). Two target classifiers: "flat", $d=(0.90,0.70)$, as in [[ext2_baycap]], and "steep", $d=(0.95,0.55)$, for which the prevalence term is larger. BayCAP is computed exactly on a grid, as in `ext2_estimand_gap.py`. "Oracle $\pi$" uses the true counts and is a reference. Coverage of the $95\%$ interval against the realised accuracy, $2000$ bags per row; "closed form" is $2\Phi(1.96/\sqrt{1+r})-1$ (§3.5), the prediction for the oracle-$\pi$ column, with $r=\operatorname{Var}(e_2)/\operatorname{Var}(e_3)$ computed from $d$, $\pi^Q$, $n$, $n_s$ (here $r=0.5\,n/n_s$ for both classifiers); the Monte Carlo error is about $\pm0.01$ for methods whose coverage does not depend on the validation set, and larger for the SQS family in the rows with large $r$, where its coverage varies strongly between validation sets (the script prints the 10th–90th percentile).

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

- the coverage rule of §3.5 predicts the oracle-$\pi$ coverage within $0.04$ for $r\le1.5$ and within about $0.08$ at $r\in\{5,15\}$, where coverage depends strongly on which of the $40$ validation sets is drawn (the normal approximation for $e_2$ is also coarsest there);
- $e_3$ shrinks as $n^{-1/2}$ and $e_2$ is constant in $n$ and shrinks with $n_s$; at $n_s=100$, $e_1$ also keeps a floor (about $0.015$) from fitting the quantifier on $V$ (H3);
- SQS-H, the RQBS-like variant and even the oracle-$\pi$ reference lose coverage as $r$ grows (down to $0.21$–$0.45$ at $r=15$): the missing term is $e_2$, which no SQS variant without $V$ resampling represents (H1). Between validation sets, the coverage of SQS-H at $n_s=100$, $n=3000$ ranges from $0.00$ to $0.86$ (10th–90th percentile), i.e. it measures the validation set;
- with steep $d$, SQS-H under-covers already at small $r$ ($0.78$–$0.87$ for $r\le0.5$), while the RQBS-like variant does not: the bootstrap of $\hat\pi$ supplies the prevalence term;
- BayCAP with the population pushforward is lowest at small $r$ ($0.59$ at $r=0.05$) and improves as $r$ grows (H2);
- the realised-table BayCAP and SQS-H with $U$ and $V$ resampling stay between $0.88$ and $0.98$ in every row (H4); the realised-table BayCAP dips to $0.88$–$0.92$ at $n_s=100$ and large $n$ with flat $d$, a residual not investigated here;
- the paper's SQS-H runs at $n=1000$; in the toy, the regime of $r$ where SQS-H covers well ($r\lesssim0.2$) needs about five validation instances per class for every target instance of that class.

What is not known: whether real quantifiers (PACC with a cross-validated surrogate), multiclass problems and real validation sizes reproduce these trends, and how large the quantifier bias $b_q$ is. The paper's datasets have $\lvert V\rvert$ equal to $35\%$ of the dataset, so $r$ at $\lvert U\rvert=1000$ varies widely across them; E3.2 measures where each dataset falls.

## B. Relation to other extensions

- **[[ext1_metrics|Extension 1]]**: supplies the metrics; macro-K is a sharp test of the budget (small $e_1$).
- **[[ext2_baycap|Extension 2]]**: its estimand gap is the term $e_3$; its hypothesis H2 is tested within datasets by E3.2; its realised-table predictive is one of the completed methods. Both studies share the dataset subset and the BayCAP draws.
- **Extension 6**: varies the allocation $\rho$ at fixed $\lvert V\rvert$; here $\lvert V\rvert$ is subsampled at fixed allocation. A factorial $\rho\times\lvert V\rvert$ is out of scope.
- **[[sqs_b|Extension 8]]** and **[[baycap_kdey|Extension 9]]**: their methods are placed in the table of §3.4 and enter E3.2; a posterior-propagation study comparing them can reuse the grid of E3.2.
- **[[extensions_to_evaluate]] A**: promoted into this note (§2–§3).

## C. Out of scope

- violations of PPS (moved to [[optional_extensions]] §2);
- calibration of the method-specific representation of $\pi^Q$ as such (Extension 8);
- the validation allocation $\rho$ (Extension 6);
- natural sampling of the target bags beyond the remark of §3.7 ([[ext2_baycap]] Q2(b) covers it for BayCAP).
