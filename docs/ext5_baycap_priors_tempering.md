# Extension 5: Informative priors and posterior tempering for BayCAP

## 0. Summary

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

# Part I: Understanding

## 1. BayCAP's priors and temperature in brief

BayCAP and its notation are introduced in [[ext2_baycap|Extension 2]] §1: the posterior over $(\pi^Q,C)$ combines the validation likelihood $\mathcal L_V(C)$, the target likelihood $\mathcal L_U(\pi^Q,C)$, and the priors $\pi^Q\sim\operatorname{Dir}(\alpha)$ and $C_{:j}\sim\operatorname{Dir}(\mathbf 1)$ for every true class $\omega_j$.

We write the prior on $\pi^Q$ as $\alpha=c\,\mu$, with $\mu\in\Delta^{K-1}$ the prior mean and $c>0$ the concentration (total pseudo-counts). The uniform prior is $\mu=\mathbf 1/K$, $c=K$. The prior on each column of $C$ is written $\operatorname{Dir}(\beta_{:j})$, with total weight $w_j=\sum_i\beta_{ij}$; the implementations fix $\beta_{ij}=1$, so $w_j=K$.

Every interval in this note is computed from BayCAP's posterior, with either the population pushforward (target: the population table $C\operatorname{diag}(\pi^Q)$) or the realised-table predictive (target: the realised table of $U$), and is always scored against the realised table (`true_cts`), as in `main`.

*Implementation note:* `prior` in `ACC_BayesCAP` and `BayCAP` sets $\alpha$ only; the prior on $C$ is hard-coded (`jnp.ones(K)` in `BayCAP.model`, `src/uncawe/methods/baycap.py`), so studying it needs a new argument (§8).

## 2. Key concepts

### 2.1 Tempering

The tempered posterior is

$$
p_\tau(\pi^Q,C\mid V,U)\;\propto\;\left[\mathcal L_V(C)\,\mathcal L_U(\pi^Q,C)\right]^{1/\tau}\,p(\pi^Q)\,p(C),
$$

the scheme `quapy` uses for BayesianCC. We write the temperature as $\tau$ because $T$ denotes contingency tables in the reference paper; in the code it is the `temperature` parameter. A value $\tau>1$ flattens the posterior and $\tau<1$ sharpens it.

*Intuition.* Apart from the multinomial coefficients, raising a multinomial likelihood to the power $1/\tau$ is the same as dividing its counts by $\tau$. Tempering therefore acts like pretending that $V$ and $U$ are $\tau$ times smaller. The prior is not tempered, so raising $\tau$ also gives the prior more relative influence, on $\pi^Q$ and on $C$ alike.

How tempering relates to the sources of miscalibration of [[ext2_baycap|Extension 2]] §3: the estimand gap (ii) calls for $\tau>1$, the fixed-count likelihood (i) for $\tau<1$; neither direction moves the centre of a displaced interval (iv).

### 2.2 Prior weight

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

### 2.3 Well-specified prior and calibration on average

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

### 2.4 The prior on $C$ is informative

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

# Part II: For the paper

## 3. Contributions

1. **Tempering as a diagnostic.** What tempering can achieve with the best temperature (oracle-best $\tau$, a reference that uses target labels), compared with the realised-table predictive of [[ext2_baycap|Extension 2]], which needs neither labels nor tuning.
2. **Priors on $\pi^Q$ (the paper's stated future work).** Under the paper's protocol the uniform prior is already well specified; informative priors help only under a non-uniform bag-generating distribution, and only when the prior weight is not negligible (small $\lvert U\rvert$, ill-conditioned $C$).
3. **Prior on $C$.** The uniform prior on the columns of $C$ is informative and biases BayCAP as $K$ grows; a prior with total weight independent of $K$, or a diagonal-aware prior, removes the bias.
4. **Label-free tempering** (only if H5 is refuted): a calibration procedure for $\tau$ on the performance interval.

## 4. Hypotheses

Primary:

- **H5 (the predictive needs no tuning).** The realised-table predictive reaches near-nominal coverage without tempering, and at matched coverage its intervals are no wider than those of tempering at the oracle-best $\tau$. *Confirmed if* in E5.2 its coverage is within tolerance of $0.95$ across datasets and its Winkler score is close to or better than that of oracle-best tempering. *Refuted if* oracle-best tempering is clearly better; then label-free selection of $\tau$ (E5.5) becomes worth its cost. Tested only if H0 of [[ext2_baycap|Extension 2]] is confirmed.
- **H7 (a $K$-independent prior on $C$ removes the bias).** On datasets with large $K$ and few validation instances per class, replacing $\operatorname{Dir}(\mathbf 1)$ with a prior of total weight $w\le2$ ($\operatorname{Dir}(1/K)$, diagonal-aware) reduces the bias of BayCAP by about the factor $K/w$ predicted in §2.4 and restores most of the coverage, without changing the results on binary datasets. *Confirmed if* in E5.3 the bias follows $w\,(1/K-r_j)/(n^s_j+w)$ across $w$ and coverage rises towards nominal on `isolet` and `letter`. *Refuted if* the bias does not change with $w$; then source (iv) of [[ext2_baycap|Extension 2]] is not the cause.

Secondary:

- **H1 (prior on $\pi^Q$ matters only when data are weak).** The prior's effect grows with $w=c/(c+N^Q/\tau)$ and with the ill-conditioning of $C$. *Confirmed if* in E5.4 the well-specified prior narrows intervals at $\lvert U\rvert=100$ and not at $\lvert U\rvert=1000$.
- **H8 (tempering amplifies the bias of the prior on $C$).** With $\operatorname{Dir}(\mathbf 1)$ on $C$, the bias on large-$K$ datasets grows with $\tau$ as $w\,(1/K-r_j)/(n^s_j/\tau+w)$. *Confirmed if* in E5.1 the bias at $\tau\in\{2,4\}$ is larger than at $\tau=1$ on these datasets, by about the predicted amount. *Refuted if* the bias does not depend on $\tau$.
- **H4 (prior $\times$ temperature).** If $\tau>1$ is selected, a misspecified prior on $\pi^Q$ gains weight and robustness degrades. *Confirmed if* in E5.5 a confidently wrong prior loses more coverage at the selected $\tau$ than at $\tau=1$. Tested only if E5.5 runs.

## 5. Paper outline

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §5 (BayCAP), subsection "Calibrated BayCAP" of [[ext2_baycap|Extension 2]], second half | tempering as a diagnostic | The realised-table predictive calibrates BayCAP at least as well as the best temperature. | E5.1; E5.2 |
| §5, paragraph "Prior on $C$" of [[ext2_baycap|Extension 2]], second half | prior weight $w=K\beta$; Perks and diagonal-aware priors | A prior on $C$ whose weight does not grow with $K$ removes BayCAP's bias on large-$K$ datasets. | §2.4; E5.3 |
| §5, short paragraph "Prior on $\pi^Q$" | well-specified prior under APP; prior weight | Under the paper's protocol the uniform prior is the correct one, and informative priors matter only when data are weak. | §2.2–2.3; E5.4 |
| §6 (experiments), subsection "Calibration of BayCAP", shared with [[ext2_baycap|Extension 2]] | F2, T2, T3, T5 (see §5.1) | as above | Part III |
| §7 (conclusions) | one paragraph | Calibrated BayCAP closes (or narrows) the gap with SQS-H in interval quality. | E5.2 (T2), against SQS-H |

Wording if results are mixed: if H5 is refuted, the predictive is presented as a correction of the estimand, and label-free tempering (E5.5) as a calibration step on top of it; if H7 is refuted, the "Prior on $C$" paragraph reports the bias without a remedy; if H1 is refuted at $\lvert U\rvert=100$, priors on $\pi^Q$ get one sentence in the conclusions.

### 5.1 Figures and tables

All use Acc; macro-F1 and macro-K versions go to the appendix. Conventions follow the existing paper figures.

| Item | Experiment | Content | What it must show |
|---|---|---|---|
| F2 | E5.1, E5.2 | coverage–amplitude trajectories over $\tau$, one path per dataset through its $\tau$ values; solid joint tempering, dashed $\mathcal L_U$-only; separate marker for the predictive at $\tau=1$ | where tempering reaches $0.95$ and at what width, compared with the predictive (H5) |
| T2 | E5.2 | mean ranks per measure with Wilcoxon markers (`tables/rank.py`) for BayCAP at $\tau=1$, predictive at $\tau=1$, and SQS-H from `main`; oracle-best variants as greyed reference columns, not ranked; one block per $\lvert U\rvert$ (SQS-H only at $1000$) | whether Calibrated BayCAP closes the gap with SQS-H |
| T3 | E5.4 | rows prior on $\pi^Q$ (uniform, well-specified, confidently wrong) $\times$ $\lvert U\rvert$; columns coverage, amplitude, width relative to the uniform prior, for both pushforwards; mean $\pm$ std with min/max colouring (`tables/binform.py` style) | that informative priors matter only at small $\lvert U\rvert$ (H1) |
| T5 | E5.3 | rows dataset $\times$ prior on $C$ (ordered by $w$); columns bias, coverage, amplitude, Winkler, for both pushforwards; $K$ and median $n^s_j$ as index columns | that the bias falls with $w$ as predicted and coverage recovers (H7) |

Appendix and internal material: coverage vs $\tau$ per dataset (diagnostic used to read off the oracle-best $\tau$); bias vs $\tau$ on large-$K$ datasets (H8); `plots/amplitude_vs_coverage.py` with the E5.2 variants; `plots/binform_simplex.py` to illustrate $G$ and the E5.4 priors on the simplex; robustness curves of the full prior study.

# Part III: For the implementation

## 6. Experiments

Common to all: same bags and seeds across conditions, $100$ bags per configuration as in the paper, measures as in the paper (AE, coverage, amplitude, Winkler at $\alpha=0.05$) on the metrics of [[ext1_metrics]], always scored against the realised table of each bag; both pushforwards of [[ext2_baycap|Extension 2]] (population and realised-table predictive) unless stated. Dataset subset: the one of [[ext2_baycap|Extension 2]] (binary and multiclass, small and large $\lvert V\rvert$ per class, well- and ill-conditioned $C$, at least one large-$K$ dataset); `connect-4` for E5.4. All five classifiers unless stated. Order and gates: E5.1 → E5.2 (needs H0 of [[ext2_baycap|Extension 2]] for the predictive; otherwise it compares tempering with BayCAP at $\tau=1$ only); E5.3 is independent; E5.4 runs its minimal check first and the full grid only if H1 is confirmed; E5.5 runs only if H5 is refuted.

### E5.1: temperature sweep

- **Goal / hypothesis:** what tempering could achieve, with target labels used only to evaluate it; input to H5; H8.
- **Inputs:** BayCAP rerun on the APP bags of the subset (the same bags as [[ext2_baycap|Extension 2]] Q2, whose $\tau=1$ draws are reused).
- **Varied:** $\tau\in\{0.25,0.5,1,2,4\}$; tempering variant: joint ($\mathcal L_V$ and $\mathcal L_U$), $\mathcal L_U$ only, and, as an ablation, $\mathcal L_V$ only (uncertainty on $C$).
- **Outputs:** table draws (sidecar `.npz`) for every $\tau$ and variant, so that E5.2 can apply the predictive to them; metric samples; the oracle-best $\tau$ per (dataset, classifier) and pushforward; the bias split of [[ext2_baycap|Extension 2]] Q5 at every $\tau$.
- **Measures:** the paper's measures; bias and its recall term on large-$K$ datasets.
- **Interpretation:** the direction in which $\tau$ must move to reach nominal coverage ($\tau>1$ on large-$V$ datasets if the estimand gap dominates); how wide the intervals become there, compared with the predictive at the same coverage; whether loosening $\mathcal L_U$ alone gives the extra width. The $\tau$ closest to nominal coverage is the *oracle-best $\tau$*, a reference, not a method. A recall term growing with $\tau$ on large-$K$ datasets supports H8.
- **Visualization:** F2; coverage vs $\tau$ per dataset; bias vs $\tau$ (appendix).
- **Cost:** five runs per bag and per tempering variant.

### E5.2: Calibrated BayCAP vs tempering

- **Goal / hypothesis:** is the realised-table predictive as good as the best tempering? H5.
- **Inputs:** subset datasets, $\lvert U\rvert\in\{100,1000\}$.
- **Varied:** four variants: BayCAP at $\tau=1$ (population pushforward, as in the paper); realised-table predictive at $\tau=1$; tempering at the oracle-best $\tau$ (population pushforward); realised-table predictive at the oracle-best $\tau$.
- **Outputs:** metric samples per variant, oracle-best $\tau$ for each pushforward.
- **Measures:** the paper's measures; width at matched coverage; comparison with SQS-H from `main` at $\lvert U\rvert=1000$.
- **Interpretation:** H5 holds if the predictive at $\tau=1$ is within tolerance of nominal coverage and its Winkler score is close to that of oracle-best tempering; the predictive at the oracle-best $\tau$ shows whether any residual tempering helps.
- **Comparability:** all variants are fitted on all of $V$ and are directly comparable with `main`. The oracle-best variants use target labels to choose $\tau$ and are references, not methods.
- **Visualization:** T2; F2 (shared with E5.1); `amplitude_vs_coverage` with the four variants in the appendix.
- **Cost:** at $\lvert U\rvert=1000$ none beyond E5.1, since all four variants reuse its draws; $\lvert U\rvert=100$ needs one $\tau$ sweep.

### E5.3: prior on $C$

- **Goal / hypothesis:** does a prior on $C$ with total weight independent of $K$ remove BayCAP's bias on large-$K$ datasets? H7.
- **Inputs:** `isolet`, `letter` ($K=26$), `hand_digits` ($K=10$), `chess` ($K=15$), and one binary control (`spambase`); the $100$ `main` bags with the same seeds; LR in the pilot, all five classifiers in the full run.
- **Varied:** prior on $C$: $\operatorname{Dir}(\mathbf 1)$ ($w=K$, as in the paper), $\operatorname{Dir}(1/K)$ ($w=1$), $\operatorname{Dir}(0.1/K)$ ($w=0.1$), diagonal-aware ($w=2$); pushforward: population and realised-table predictive; $\tau=1$; prior on $\pi^Q$ uniform.
- **Outputs:** metric samples; the bias split of [[ext2_baycap|Extension 2]] Q5 for every prior; NUTS diagnostics (divergences, effective sample size), since small $\beta$ puts mass near the boundary of the simplex.
- **Measures:** the paper's measures; bias and recall term against the prediction $w\,(1/K-r_j)/(n^s_j+w)$ of §2.4.
- **Interpretation:** see H7. A bias that falls with $w$ but coverage that stays low points to the remaining sources of [[ext2_baycap|Extension 2]] §3; divergences at small $w$ limit how small $w$ can be in practice.
- **Visualization:** T5.
- **Cost:** four priors × five datasets × $100$ bags NUTS runs per classifier; the population and realised pushforwards share each run.

### E5.4: priors on $\pi^Q$

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

### E5.5: label-free temperature selection (gated)

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

## 7. Components, tests, and reproducibility

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

# Appendix

## A. Toy simulations

- `uv run python scripts/toys/ext5_prior_calibration.py` (§2.3; defaults: $200{,}000$ bags, seed $0$) is seeded and prints the table of §2.3.
- [proposal] a toy for §2.4: simulate validation confusion columns for chosen $K$, $n^s_j$ and $r$, compute the conjugate posterior of $C$ for each prior on $C$ (no NUTS), and print the predicted and simulated bias, the ratio bias/sd, and the coverage of the accuracy interval. It checks the formulas of §2.4 before E5.3 and provides the worked numbers for the paper.

## B. Relation to other extensions

- **[[ext2_baycap|Extension 2]]**: supplies BayCAP, the two pushforwards, the sources of miscalibration (source (iv) motivates E5.3), the dataset subset, the `calibration` experiment and the $\tau=1$ draws; H0 there gates the comparison of E5.2.
- **[[ext4_uncertainty_budget|Extension 4]]**: supplies the $\lvert U\rvert$ axis used by E5.2 and E5.4.
- **[[ext3_posterior_propagation|Extension 3]]** (BayCAP-KDEy, SQS-B): `BayesianKDEy` also has a temperature; Extension 3 keeps $\tau=1$, and tempering BayesKDEy belongs to this extension (user's decision of 2026-10-27), with no experiment designed yet.
- **[[optional_extensions]]**: §1 covers post-hoc interval rescaling, a different estimator from in-model tempering. Its former §4 (full informative-prior study) and §5 (label-free temperature selection) were revived here on 2026-10-15, as E5.4 and E5.5.

## C. Out of scope

- hierarchical or empirical-Bayes priors learned from a stream of target sets;
- informative priors on $C$ built from external knowledge (e.g. from other classifiers on the same dataset); E5.3 studies only weak default priors;
- post-hoc interval recalibration;
- violations of PPS.
