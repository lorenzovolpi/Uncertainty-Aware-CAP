# Extension 4: The uncertainty budget of CAP under PPS

## Claim

A CAP interval is scored against the accuracy of one target bag, and the gap between the estimate and that accuracy has several sources: the error of the estimated prevalence, the error of the rates of $h$ measured on the validation set $V$, and the luck of the bag. Under the paper's protocol, where every bag is re-drawn with replacement from one finite test pool, there is one more: the luck of the pool, which all the bags of a dataset share. The sources scale differently with the size $\lvert U\rvert$ of the bag, the size of $V$ and the size of the pool, and each CAP method's interval is wide only along the sources it simulates. This extension writes the error as an exact sum of these parts (the *uncertainty budget*), uses it to predict in which regime of $(\lvert U\rvert,\lvert V\rvert)$ each method is calibrated, and measures it on real data; the paper's comparison at $\lvert U\rvert=1000$ is one point of this regime map. The budget and the predictions are a new hypothesis, supported so far only by a toy simulation. A further question, opened by cause 4 of [[ext2_baycap|Extension 2]], is what would change if the bags were drawn *without* replacement: each bag would then be a fresh sample of the target distribution, the pool term would merge into the luck of the bag, and the methods and baselines that miss the pool term under the paper's protocol could become calibrated (§T.10, candidate experiments).

This extension is the common frame for the method extensions: the realised-table predictive of Extension 2 adds the luck of the bag to BayCAP, and the methods of [[ext3_posterior_propagation|Extension 3]] propagate a Bayesian prevalence posterior. It absorbs proposal A of [[extensions_to_evaluate]] (variance decomposition).

## Status

| Item | Status |
|---|---|
| SQS-H, SQS-S, BayCAP at $\lvert U\rvert=1000$, scored against the realised accuracy of each bag | [paper] |
| `RQBS` (SQS-H with a bootstrap of $\hat\pi$, the variant run by `main`), `PrediQuant` (SQS-S), `BayesCAP` (BayCAP, built from `methods/baycap.py:BayCAP`) | [implemented] |
| `BayesCAP-real` (BayCAP with the realised-table predictive, [[ext2_baycap]] E2.3) | [implemented], evaluated in `main` |
| `BayCAPKDEy` | [implemented], not evaluated |
| SQS-B and the realised pushforward of BayCAP-KDEy ([[ext3_posterior_propagation]]) | [proposal] |
| Error split, budget with the pool term, coverage rule and ratios $\eta$, $r$, oracle ladder, SQS-H with $U$ and $V$ resampling | [new hypothesis] |
| The protocol without replacement and its budget (§T.10) | [new hypothesis] |
| Toy simulation of the budget (`scripts/toys/ext4_budget.py`, Appendix A) | [implemented], run |
| E4.1 budget validation, E4.2 regime map, E4.3 robustness (study `target_size`) | [proposal], not implemented |
| Candidate experiments E4.c1, E4.c2 (cause 4) | [proposal], not scheduled |

The paper material of earlier versions (contributions, paper outline, figure plan) is in [[optional_extensions]] §9.

# Experiments

Common to all: APP bags drawn with replacement from the test pool, as in `main`; all intervals scored against the realised table of each bag; measures as in the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$) plus the error split of §T.2; metrics of [[ext1_metrics]]. Coverage is reported with a standard error clustered by $V$ seed, since bags sharing a $V$ are not independent; bags also share the pool, which is fixed per dataset, so the pool term $e_P$ of §T.3 is a constant offset within a dataset (§T.7). Dataset subset shared with [[ext2_baycap]]: binary and multiclass datasets covering small and large $\lvert V_j\rvert$, well- and ill-conditioned $C$, and one large-$K$ dataset; the pool sizes $P_j$ are recorded for every dataset. Classifiers: LR and RF (all five in the pilot). Grids:

- $\lvert U\rvert\in\{100,300,1000,3000,10000\}$, log-spaced, including the paper's $1000$; bags larger than the pool repeat instances, which is consistent with sampling with replacement; at these sizes the pool term, which does not shrink with $\lvert U\rvert$, is part of the fixed error (§T.8);
- $\lvert V\rvert$ levels: the full $V$ and a stratified subsample of $1/4$ (and $1/16$ on the largest datasets), nested ($V_{1/16}\subset V_{1/4}\subset V$) and seeded; the split $L/V$, the classifier $h$, and the allocation $\rho$ used inside SQS-S are unchanged, so Extension 6 stays separate;
- $S_V=5$ outer seeds for each subsampled $V$ level (for the full $V$, the seed changes the $L/V$ split; $h$ is retrained per seed).

Order: pilot, then E4.1, E4.2, E4.3.

## E4.1: budget validation

**Why.** The budget (§T.2–§T.5) predicts how each part of the error scales: the luck of the bag and the quantifier's noise on the bag shrink as $\lvert U\rvert^{-1/2}$, the errors fixed by $V$ do not shrink with $\lvert U\rvert$ but shrink as $\lvert V\rvert^{-1/2}$, and the pool term is fixed by the pool. Before the budget can explain the coverage of the methods (E4.2), its parts must be measured on real data and their scaling checked; their variances also give each method's ratio $\eta$ (§T.7), which E4.2 needs.

- **Design.**
  - **Inputs:** the subset datasets and their test pools; quantifier PACC as in the paper; methods RQBS and SQS-H with a point $\hat\pi$, which are cheap.
  - **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed; three fixed prevalences per dataset: uniform $\mathbf 1/K$, one skewed vector (a seeded draw from $\operatorname{Dir}(\mathbf 1)$), one near-vertex vector ($0.9$ on the rarest validation class, the rest uniform); $R=100$ bags per (prevalence, $\lvert U\rvert$, $\lvert V\rvert$, seed), i.e. $500$ per cell. Held fixed: the pool of each dataset, the quantifier and the methods.
  - **Measures:** per bag, $\hat\pi$, $\hat A$, $e_1^U$, $e_1^V$, $e_2+e_P$ and $e_3^b$ (§T.3; on real data the population rates $d$ are unknown, so $e_2$ and $e_P$ are observed only through their sum $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$), interval bounds; per cell, the RMSE and mean of each part, the bias and standard deviation of $\hat\pi$ and $\hat A$, and their log–log slopes against $\lvert U\rvert$ and $\lvert V\rvert$. $\pi^\infty(V)$ is approximated, once per ($V$ seed, prevalence), by the estimate of the quantifier fitted on that $V$ on one very large bag (e.g. $10^5$ instances drawn with replacement from the pool at $\pi^Q$); then $e_1^V$ follows from its definition in §T.3 and $e_1^U=e_1-e_1^V$. Oracle ladder (references, never methods): (a) true $\pi^Q$: SQS bags at the true counts, removing $e_1$; (b) $d^{\text{pool}}$ from the labelled pool, removing $e_2$ and $e_P$ together; (c) scoring against $d^{\text{pool}\top}\pi^Q$, isolating $e_3^b$.
  - **Estimand:** the realised table of each bag for every interval; the population accuracy $d^{\text{pool}\top}\pi^Q$ only in oracle (c). The parts of the split use the true prevalence of the bag and the labels of the pool, so they are an analysis, not a method.
  - **Cost:** no NUTS; dominated by bag sampling, of order $m\cdot n$ per bag for $m$ draws of size $n$.
- **Representation.**
  - **RMSE of the parts against $\lvert U\rvert$** (log–log), one panel per $\lvert V\rvert$ level, one line per dataset (LR), for $e_1$, $e_2+e_P$ and $e_3^b$.
  - **Table, quantifier bias:** per dataset, the bias and standard deviation of $\hat\pi$ ($L_1$) and of $\hat A$ against $\lvert U\rvert$.
- **How the results are read (H3, the budget accounts for the error).** $\operatorname{RMSE}(e_3^b)$ should scale as $n^{-1/2}$; $\operatorname{RMSE}(e_2+e_P)$ should be constant in $n$, and its square should be close to $\operatorname{Var}(e_2)+\operatorname{Var}(e_P)$ of §T.4 (it shrinks with $\lvert V\rvert$ only down to the pool part); $e_1$ should be the sum of a part shrinking with $n$ and a part fixed given $V$. The prediction holds if the fitted slopes are within $\pm0.15$ of the predicted ones and, for every method, the coverage measured in E4.2 against its $\eta$ lies within $\pm0.05$ of $2\Phi(1.96/\sqrt{1+\eta})-1$ on most (dataset, classifier) pairs. If coverage does not follow $\eta$, non-normality, the correlations of §T.4 or the quantifier bias $b_q$ matter, and the budget must be refined. The mean of $e_1$ over $V$ seeds estimates $b_q$.
- **Analysis.** Not run.

## E4.2: regime map

**Why.** The budget predicts that each method is calibrated only in one regime of $(\lvert U\rvert,\lvert V\rvert)$, because its interval represents some parts of the error and misses others whose share changes with the two sizes (§T.8). The paper compares the methods at $\lvert U\rvert=1000$ only; E4.2 measures the coverage of every method across the grid, to see whether the paper's ranking holds elsewhere and whether the methods that represent all parts but the pool term are calibrated across the map.

- **Design.**
  - **Inputs:** the subset datasets; APP prevalences ($100$ uniformly drawn prevalence vectors per cell, as in the paper, with fixed seeds shared by all methods and cells); the variances of the parts estimated by E4.1 for the same cells.
  - **Varied:** $\lvert U\rvert$; $\lvert V\rvert$ level; $V$ seed; the method: SQS-H (point $\hat\pi$), RQBS, SQS-S (`PrediQuant`, after the fix of [[work_plan]] FIX-1), BayCAP (population pushforward), BayCAP with the realised-table predictive (same draws, no extra NUTS run), SQS-B, BayCAP-KDEy (both pushforwards, with the implementations of [[ext3_posterior_propagation]] E3.1), SQS-H with $U$ and $V$ resampling (§T.8); the oracle references of E4.1.
  - **Measures:** per bag and method, the metric samples; per cell, coverage (clustered standard error), amplitude, Winkler score, AE; $r$ per cell (§T.7) and $\eta$ per method and cell, from the variances of E4.1.
  - **Estimand:** the population table for BayCAP and BayCAP-KDEy with the population pushforward, the realised table for the realised pushforwards and for the SQS family; every interval scored against the realised table of each bag.
  - **Cost:** NUTS dominates: one BayCAP run per bag, shared by both pushforwards; BayesKDEy and SQS-B add their own posterior runs. The pilot (two datasets, LR, $20$ bags) estimates the cost; if needed, the BayCAP family runs on $S_V=3$ seeds.
- **Representation.**
  - **Scatter, regime map:** coverage against $r$ (log scale), one panel per method, one point per (dataset, classifier, $\lvert U\rvert$, $\lvert V\rvert$), a dashed line at $0.95$.
  - **Scatter, coverage rule:** coverage against each method's $\eta$ (log scale), all methods in one panel, with the curve $2\Phi(1.96/\sqrt{1+\eta})-1$.
  - **Table, ranking:** mean Winkler rank per method, one column block per $\lvert U\rvert$ at the full $V$, oracle references as unranked columns (`tables/rank.py`).
- **How the results are read.** Three predictions of §T.8. **H1 (SQS-type intervals degrade with $n$):** at fixed $V$ the coverage of RQBS decreases monotonically with $\lvert U\rvert$ on most pairs, faster at the smaller $\lvert V\rvert$ level; if coverage is flat in $\lvert U\rvert$, $e_2$ and $e_1^V$ are negligible on real data. **H2 (BayCAP under-covers at small $n$ and large $V$):** the coverage of BayCAP with the population pushforward increases with $\lvert U\rvert$ at fixed $V$ and decreases with $\lvert V\rvert$ at fixed $\lvert U\rvert$; if it over-covers at every $r$, the fixed class counts of APP bags ([[ext2_baycap]] §T.6) or prior effects dominate. **H4 (completing the budget restores coverage):** the realised-table BayCAP and SQS-H with $U$ and $V$ resampling are within $\pm0.03$ of nominal at every $(\lvert U\rvert,\lvert V\rvert)$ level, while each partial method fails where its missing part dominates; if they fail at large $n$ as well, the quantifier bias $b_q$ or the pool term $e_P$, which no method represents, is the dominant fixed part, and the two are told apart by E4.c1. The cell $\lvert U\rvert=1000$, full $V$, seed of `main` reproduces the paper's setting (comparability check with `main`).
- **Analysis.** Not run.

## E4.3: robustness

**Why.** The regime map of E4.2 is measured on a few datasets, on uniformly drawn prevalences and on vanilla accuracy; the budget predicts how it should move with the prevalence, the number of classes and the metric, so these strata test it further.

- **Design.**
  - **Inputs:** the outputs of E4.1 and E4.2; no new runs, except macro-F1 and macro-K computed from the stored tables.
  - **Varied:** prevalence type (uniform, skewed, near-vertex); binary against multiclass; metric.
  - **Measures:** coverage against $r$ per stratum.
  - **Estimand:** as in E4.2.
  - **Cost:** analysis only.
- **Representation.** The two scatters of E4.2, one version per stratum.
- **How the results are read.** Macro-K depends on $\pi^Q$ only weakly ([[ext1_metrics]]), so its $e_1$ should be small and its coverage should follow $e_2$, $e_P$ and $e_3$; near-vertex prevalences raise $\sum_j(\pi^Q_j)^2$ and therefore $r$ (and $\operatorname{Var}(e_P)$) at the same $\lvert U\rvert$.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussions; they are not scheduled and are numbered E4.c<n>. One becomes an experiment of the main section only when the user decides so, and then it gets the next number E4.<n> and a full specification. The two candidates below study cause 4 of [[ext2_baycap|Extension 2]] (the test pool shared by all bags) for every method and classifier, moved here on 2026-10-29 because the pool term affects every method that learns from $V$.

- **E4.c1, the pool gap on stored results.** *Why:* Extension 2 found on binary datasets a recall error that the prior of BayCAP does not explain, and its toy traces it to the pool (§T.3; [[ext2_baycap]] §T.13, Appendix A.3); the explanation is a hypothesis, and the pool gap $\sum_j\pi^{(b)}_j(\hat d_j-d^{\text{pool}}_j)$ belongs to the (dataset, classifier) pair, not to a method, so it can be measured once and compared with the shift of every method. *What it would show:* per pair, the pool sizes $P_j$, the plain recalls of $h$ on $V$ and on the pool, the pool gap averaged over the bags and its predicted standard deviation $\sqrt{\sum_j\pi_j^2d_j(1-d_j)(1/\lvert V_j\rvert+1/P_j)}$ (§T.4); per method, the shift (mean over the bags of $\hat A-A$), its correlation and slope against the pool gap across pairs, and a reference coverage of the interval moved by the pool gap of each bag (a reference, since it uses the labels of the pool); for the BayCAP methods, $\overline{R-R_0}$ of [[ext2_baycap]] E2.2 against the pool gap. *Needs:* the stored results of `main` (all its methods, vanilla accuracy, $\lvert U\rvert=1000$) and of E2.5; the predictions and labels of $V$ and of the pool, from the pretrain files; no runs.
- **E4.c2, with or without replacement.** *Why:* under the paper's protocol the pool term $e_P$ is an error that no method represents; if the bags are drawn without replacement, each bag is an i.i.d. sample of the target distribution and $e_P$ becomes part of a luck of the bag of the right size (§T.10). Comparing the two protocols on the same pools shows how much of each method's miscalibration the protocol creates, and how the error split changes. *What it would show:* per method and pair, coverage, amplitude, Winkler score and AE under both protocols, their differences (paired by bag index, since the bags share their prevalences and seeds), and the error split of §T.2 under both protocols; coverage both per pair and averaged over pairs, since without replacement a share $n_j/P_j$ of the luck of the bag is common to the bags of a pair (§T.10). Predicted (§T.10): the methods that represent $e_2$ and $e_3$ (BayCAP with the realised-table predictive) gain the most, more on small pools; those that miss $e_2$ (SQS-H, RQBS) gain too but stay short; BayCAP with the population pushforward gains where the pool term is large but stays short, since it still misses the luck of the bag; for the baselines (DoC, QuAcc, O-LEAP, L-CBPE) the change is measured, not predicted. *Needs:* a check of the pool sizes first; a bag size $N_d$ per dataset, the largest of $\{50,100,250,500\}$ not above $\min_jP_j/4$, so that every class fits without replacement at every UPP prevalence and the bags of a pair share at most a quarter of the variance of $e_3$, with the datasets whose $\min_jP_j$ is below $200$ excluded (`mammographic`, with about $124$ instances per class in its pool, among them); UPP bags without replacement, a new option of the sampling of `main`; two runs of the methods of `main` at $N_d$, with and without replacement, with the same pools, $V$, $h$, prevalences and seeds, stored apart from `main`; the stored results of `main` at $N=1000$, which link the run with replacement at $N_d$ to the paper; the pool-gap measures of E4.c1 for the run with replacement.

# Theory

This part explains what the experiments need, step by step. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$, $d=\operatorname{diag}(C)$ the vector of its recalls (the accuracy within each class), and the population table $T=C\operatorname{diag}(\pi^Q)$. Under PPS, $C$ is the same on $V$ and on the target. Numerical examples are illustrative, computed by hand, unless they cite the toy of Appendix A.

## T.1 Setting, protocol and estimand

**Three versions of the recalls.** $d_j$ is the population recall of $h$ on class $\omega_j$: the probability that $h$ classifies correctly a new instance of that class. Two finite samples estimate it. The validation set $V$, with $\lvert V_j\rvert$ instances of class $\omega_j$, gives $\tilde d_j$, the share of them that $h$ classifies correctly. The labelled test pool from which the target bags are drawn, with $P_j$ instances of class $\omega_j$, has its own share $d^{\text{pool}}_j$. $V$ and the pool are disjoint random samples of the same distribution, so $\tilde d_j$ and $d^{\text{pool}}_j$ are two independent noisy versions of $d_j$.

**Protocol.** The paper's APP draws a bag $U$ of size $n=\lvert U\rvert$ from the pool, with class counts $n_j=n\pi^Q_j$ (rounded), and draws the instances of each class *with replacement*. Each instance of class $\omega_j$ in the bag is then a uniformly random instance of the pool's class $\omega_j$, correctly classified with probability $d^{\text{pool}}_j$, independently of the others, so the number of correct predictions in class $\omega_j$ is

$$
c_j\sim\operatorname{Bin}(n_j,d^{\text{pool}}_j),
$$

with $d^{\text{pool}}_j$, not $d_j$. This is the main protocol of this extension, because it is the protocol of the paper and of `main`. A protocol that draws without replacement is studied in §T.10.

**Estimand.** Every interval is scored against the **realised accuracy** $A(h,U)=\frac1n\sum_jc_j$, computed from the realised table of the bag (`true_cts` in `main`), as in the paper. The **population accuracy** $d^\top\pi^Q$ appears only as a reference. The difference between the two is the estimand gap of [[ext2_baycap]] §T.2.

*Intuition.* Even with $\pi^Q$ and $C$ known, the realised accuracy on $n$ specific instances is random; any interval that ignores this is too narrow for the realised accuracy. And when the bags are re-drawn from one pool, they vary around the accuracy of that pool, not around the population accuracy.

## T.2 The error split

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
\hat A-A(h,U)=\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,\tilde d_j}_{e_1\ \text{prevalence}}+\underbrace{\sum_j\pi^Q_j(\tilde d_j-d_j)}_{e_2\ \text{rate matrix}}+\underbrace{d^\top\pi^Q-A(h,U)}_{e_3\ \text{finite target}}.
$$

For methods that average over draws, the identity holds for each draw and therefore for the average. The order of the add-and-subtract steps is a choice: the second-order cross term $\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d_j)$ is included in $e_1$ (because $e_1$ uses $\tilde d$, not $d$); the other order would move it into $e_2$.

**What is observable.** In an experiment the true prevalence $\pi^Q$ of the bag and the labels of the pool are known, but the population recalls $d$ are not. So $e_1$ is computable, while $e_2$ and $e_3$ are computable only after $d$ is replaced by $d^{\text{pool}}$, which moves part of $e_3$ into $e_2$ (§T.3). This is the basis of the oracle ladder of E4.1.

## T.3 Sources of each part, and the pool term

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

## T.4 Variance of each part

All variances below are first order in the estimation errors (delta method) and assume the paper's protocol: fixed class counts $n_j=n\pi^Q_j$, instances drawn with replacement from the pool. Two properties are used throughout: the variance of a binomial frequency, $\operatorname{Var}(k/m)=p(1-p)/m$ for $k\sim\operatorname{Bin}(m,p)$; and the variance of a weighted sum of independent variables, $\operatorname{Var}(\sum_ja_jX_j)=\sum_ja_j^2\operatorname{Var}(X_j)$.

**$e_3^b$.** Given the pool, $c_j\sim\operatorname{Bin}(n_j,d^{\text{pool}}_j)$, independently across classes (§T.1). Hence

$$
\mathbb E[A(h,U)\mid\text{pool}]=\frac1n\sum_jn_jd^{\text{pool}}_j=\sum_j\pi^Q_jd^{\text{pool}}_j,
\qquad
\operatorname{Var}(e_3^b\mid\text{pool})=\frac1{n^2}\sum_jn_jd^{\text{pool}}_j(1-d^{\text{pool}}_j)\approx\frac1n\sum_j\pi^Q_jd_j(1-d_j),
$$

where the last step replaces $d^{\text{pool}}_j$ by $d_j$, a second-order change.

**$e_P$.** The pool's class $\omega_j$ is a random sample of $P_j$ instances of the population's class $\omega_j$, so the number of them that $h$ classifies correctly is $\operatorname{Bin}(P_j,d_j)$ and $d^{\text{pool}}_j$ is its frequency, independently across classes. With the constant weights $\pi^Q_j$,

$$
\operatorname{Var}(e_P)=\sum_j(\pi^Q_j)^2\frac{d_j(1-d_j)}{P_j}.
$$

**$e_3$, by the law of total variance.** The law of total variance says that for two random quantities $X$ and $G$, $\operatorname{Var}(X)=\mathbb E[\operatorname{Var}(X\mid G)]+\operatorname{Var}(\mathbb E[X\mid G])$: the total spread is the average spread within each value of $G$ plus the spread of the averages. With $X=e_3$ and $G$ the pool, $\mathbb E[e_3\mid\text{pool}]=e_P$ (because $e_3^b$ has mean $0$ given the pool) and $\operatorname{Var}(e_3\mid\text{pool})=\operatorname{Var}(e_3^b\mid\text{pool})$, so

$$
\operatorname{Var}(e_3)\approx\frac1n\sum_j\pi^Q_jd_j(1-d_j)+\sum_j(\pi^Q_j)^2\frac{d_j(1-d_j)}{P_j}.
$$

The first term shrinks with $n$; the second does not. Their ratio for class $\omega_j$ is $n\pi^Q_j/P_j=n_j/P_j$, the number of times each pool instance of that class is used by one bag on average: with $n=1000$, $\pi^Q_j=1/2$ and $P_j=125$ the pool term is four times the bag term (illustrative). Under the paper's protocol, the luck of the pool is the larger part of the luck of the bag on every dataset whose pool is smaller than the bags.

**$e_2$.** $\tilde d_j=k_j/\lvert V_j\rvert$, where $k_j\sim\operatorname{Bin}(\lvert V_j\rvert,d_j)$ is the number of class-$\omega_j$ validation instances that $h$ classifies correctly, independently across classes, so

$$
\operatorname{Var}(e_2)=\sum_j(\pi^Q_j)^2\operatorname{Var}(\tilde d_j)=\sum_j\frac{(\pi^Q_j)^2d_j(1-d_j)}{\lvert V_j\rvert}.
$$

This variance is over draws of $V$; for a given $V$, $e_2$ is a single number. The weights are squared here and in $e_P$, and not in $e_3^b$, because in $e_3^b$ the prevalence enters through the counts $n_j$. $V$ and the pool are independent, so the observable gap of §T.3 has $\operatorname{Var}(e_2+e_P)=\sum_j(\pi^Q_j)^2d_j(1-d_j)\big(1/\lvert V_j\rvert+1/P_j\big)$.

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

$e_1^U$ has the same structure as $e_3^b$ and $e_1^V$ the same structure as $e_2$, with the rates of the surrogate classifier in place of those of $h$, amplified by $1/(S_{11}-S_{10})^2$ (an ill-conditioned surrogate inflates both) and scaled by $(d_1-d_0)^2$. The multiclass case replaces these factors with the inverse of the surrogate's rate matrix and the vector $d$. With replacement, the surrogate's rates on the pool also differ from the population ones, which adds to $e_1^U$ a pool part of the same form as $e_P$; it is left out here, as a second-order effect for the quantifiers that read the surrogate only through $\hat p$.

**Correlations.** Parts driven by different draws ($U$, $V$, the pool) are uncorrelated, because the three draws are independent. Within a group they can be correlated: $e_3^b$ and $e_1^U$ if $h$ and the surrogate tend to misclassify the same target instances, $e_2$ and $e_1^V$ if they tend to misclassify the same validation instances. The toy of Appendix A assumes $h$ and the surrogate conditionally independent given $Y$, which removes both correlations; on real data the oracle ladder measures them.

## T.5 The budget

Grouping the parts by source gives

$$
\operatorname{MSE}(\hat A)\approx\underbrace{\operatorname{Var}(e_3^b)+\operatorname{Var}(e_1^U)}_{\text{from }U,\ \propto1/n}+\underbrace{\operatorname{Var}(e_2)+\operatorname{Var}(e_1^V)+b_q^2}_{\text{from }V,\ \text{independent of }n}+\underbrace{\operatorname{Var}(e_P)}_{\text{from the pool},\ \text{independent of }n},
$$

where $b_q$ is the quantifier's systematic bias propagated to accuracy (contained in $e_1^V$, written separately because it does not average out over draws of $V$). The approximation is first order and drops the within-group correlations of §T.4.

## T.6 Which parts each method represents

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

## T.7 From the budget to coverage

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

with $Z$ standard normal. Checks: $\eta=0$ gives $0.95$; $\eta\to\infty$ gives $0$.

**Worked example: SQS-H with the true prevalence** (the oracle-$\pi$ reference). Its bags use the true counts, so $e_1=0$; it represents $e_3^b$ and misses $e_2$ and $e_P$. Its interval is approximately $\tilde d^\top\pi^Q\pm1.96\,\sigma_{3b}$: the width is right, the centre is off by $e_2+e_P$, because $\tilde d^\top\pi^Q=d^\top\pi^Q+e_2$ and $A(h,U)=d^\top\pi^Q-e_P-e_3^b$. Without the pool term, as in the toy of Appendix A, $\eta=\operatorname{Var}(e_2)/\operatorname{Var}(e_3^b)=r$, the design ratio

$$
r=\frac{\sum_j(\pi^Q_j)^2d_j(1-d_j)/\lvert V_j\rvert}{\frac1n\sum_j\pi^Q_jd_j(1-d_j)},
$$

which reduces to $n\sum_j(\pi^Q_j)^2/\lvert V_j\rvert$ when $d_j(1-d_j)$ and $\lvert V_j\rvert$ are equal across classes. In the toy, the closed form $2\Phi(1.96/\sqrt{1+r})-1$ is close to the simulated coverage of the oracle-$\pi$ reference. With the pool term, $\eta$ gains $\operatorname{Var}(e_P)/\operatorname{Var}(e_3^b)$, which by §T.4 is the analogue of $r$ with $P_j$ in place of $\lvert V_j\rvert$; with $P_j\approx\lvert V_j\rvert$, as in the splits of `main`, the two are about equal and $\eta\approx2r$.

**Conditional on one $V$ and one pool.** The formula averages over draws of $V$ and of the pool. For a given $V$ and pool, $e_2+e_P$ is a constant and the coverage of the same interval is $\Phi(1.96-(e_2+e_P)/\sigma_{3b})-\Phi(-1.96-(e_2+e_P)/\sigma_{3b})$: close to $0.95$ if they happen to give a small offset, close to $0$ if $\lvert e_2+e_P\rvert\gg\sigma_{3b}$. Coverage measured on many bags from a single $V$ and a single pool therefore measures that $V$ and that pool, not the method; experiments must repeat over $V$ seeds, and the pool is shared by all the classifiers of a dataset, so their coverages are correlated.

**Ratio for each method.** With $\sigma_k^2$ the variance of part $e_k$:

| Method | Represented | Missing | $\eta$ |
|---|---|---|---|
| SQS-H with true $\pi^Q$ (reference) | $e_3^b$ | $e_2$, $e_P$ | $(\sigma_2^2+\sigma_P^2)/\sigma_{3b}^2$ |
| SQS-H | $e_3^b$ | $e_1^U$, $e_1^V$, $e_2$, $e_P$ | $(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2+\sigma_P^2)/\sigma_{3b}^2$ |
| RQBS | $e_3^b$, $e_1^U$ | $e_1^V$, $e_2$, $e_P$ | $(\sigma_{1V}^2+\sigma_2^2+\sigma_P^2)/(\sigma_{3b}^2+\sigma_{1U}^2)$ |
| BayCAP (population) | $e_1^U$, $e_1^V$, $e_2$ | $e_3^b$, $e_P$ | $(\sigma_{3b}^2+\sigma_P^2)/(\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ |
| Realised-table BayCAP; SQS-H with $U$ and $V$ resampling | all but $e_P$ | $e_P$ | $\sigma_P^2/(\sigma_{3b}^2+\sigma_{1U}^2+\sigma_{1V}^2+\sigma_2^2)$ |

All $\sigma_k$ but $\sigma_2$ and $\sigma_P$ separately can be estimated in an experiment with labels (E4.1); $\sigma_2^2+\sigma_P^2$ is estimated from the observable gap, and its split follows from §T.4. So each method's $\eta$ is measurable and the coverage rule is testable (E4.1).

## T.8 The regime hypothesis

*Intuition.* As $n$ grows, the parts driven by $U$ shrink and those driven by $V$ or by the pool stay, so methods that simulate only the first kind become overconfident, and methods that miss the bag's luck become well calibrated.

From §T.4 and the table of §T.7 (new hypothesis):

- **SQS-H and RQBS** miss parts driven by $V$ and by the pool and represent parts driven by $U$: their $\eta$ grows linearly in $n$ and their coverage falls towards $0$ for $n\gg\lvert V_j\rvert$ or $n\gg P_j$. SQS-H also misses $e_1^U$, so it under-covers already at small $n$ when the prevalence term is large (large $\lvert d_1-d_0\rvert$ or ill-conditioned surrogate).
- **BayCAP** misses $e_3^b$ and $e_P$. As $n$ grows, $\sigma_{3b}^2$ and $\sigma_{1U}^2$ shrink while $\sigma_{1V}^2+\sigma_2^2$ stays, so the part of $\eta$ due to $e_3^b$ goes to $0$, and $\eta$ approaches $\sigma_P^2/(\sigma_{1V}^2+\sigma_2^2)$, which is at most about $1$ when $P_j\approx\lvert V_j\rvert$ (equal to $1$ if $e_1^V$ is negligible), so coverage approaches a level between $2\Phi(1.96/\sqrt2)-1\approx0.83$ and $0.95$ rather than $0.95$. At small $n$, or when $V$ is large, $\eta$ is dominated by $\sigma_{3b}^2/\sigma_{1U}^2$: BayCAP's coverage then has a floor set by how the quantifier's own noise compares with the bag's luck.
- **Methods that represent all parts but $e_P$** (realised-table BayCAP, SQS-H with $U$ and $V$ resampling) are near nominal when the pool is much larger than the bags and $V$, apart from the quantifier bias $b_q$; with the pools of `main` they miss $e_P$, and their coverage should fall as $n$ grows past $P_j$.
- **$r$ as the design axis.** $r$ depends only on $n$, $\lvert V_j\rvert$, $\pi^Q$ and $d$, so it can be computed for every cell of an experiment before running any method; the $\eta$ of each method is then measured. The pool adds a second, fixed axis per dataset, $n/P_j$. The paper's comparison at $\lvert U\rvert=1000$ fixes one value of $r$ and of $n/P_j$ per dataset.

**SQS-H with $U$ and $V$ resampling** follows directly: for every draw, resample $U$ (as `RQBS` does) and resample $V$ with stratification, refit the quantifier's rate estimates and the rates of $h$ on the resampled $V$, and draw one bag of size $n$ at the resulting prevalence. It is the non-Bayesian counterpart of the realised-table BayCAP. Its cost is one quantifier refit per draw; for aggregative quantifiers such as PACC only the misclassification rates need refitting, not the surrogate classifier, which is an approximation to be checked.

The toy simulation (Appendix A), which draws its bags from the population and therefore has no pool term, supports the first three points in a binary setting.

## T.9 Caveats

- **$\pi^Q$ and $C$ are dependent in BayCAP's posterior**, through the observed target predictions $n^t\approx n\,C\pi^Q$. The budget ignores this covariance; the oracle ladder measures it implicitly.
- **Nonlinear metrics.** For macro-F1 the split of §T.2 does not hold exactly and the budget holds only to first order (delta method). For macro-K, which depends on $\pi^Q$ only weakly ([[ext1_metrics]]), $e_1$ should be small and $e_2$, $e_P$, $e_3^b$ should dominate; this is a sharp test of the budget.
- **Natural sampling.** Under natural (i.i.d.) sampling instead of APP, the class counts are multinomial, which adds a term $\frac1n\operatorname{Var}_{\pi^Q}(d_Y)$ to $e_3^b$. All experiments here keep APP, as in the paper.
- **Quantifier bias** is not a variance and is represented by no method. If the completed methods still under-cover at large $n$, bias and the pool term are the likely causes; the oracle ladder separates bias from $e_1^V$ by repeating over $V$ seeds, and E4.c1 measures the pool term.
- **Small $n$.** Rounding $n\pi^Q_j$ can leave classes empty; the conventions of [[ext1_metrics]] for rare and absent classes apply.

## T.10 A protocol without replacement (background for E4.c2)

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

# Implementation

Components (for E4.1–E4.3):

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

# Appendix

## A. Toy simulation

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
- $e_3$ shrinks as $n^{-1/2}$ and $e_2$ is constant in $n$ and shrinks with $n_s$; at $n_s=100$, $e_1$ also keeps a floor (about $0.015$) from fitting the quantifier on $V$ (H3 of E4.1);
- SQS-H, the RQBS-like variant and even the oracle-$\pi$ reference lose coverage as $r$ grows (down to $0.21$–$0.45$ at $r=15$): the missing term is $e_2$, which no SQS variant without $V$ resampling represents (H1 of E4.2). Between validation sets, the coverage of SQS-H at $n_s=100$, $n=3000$ ranges from $0.00$ to $0.86$ (10th–90th percentile), i.e. it measures the validation set;
- with steep $d$, SQS-H under-covers already at small $r$ ($0.78$–$0.87$ for $r\le0.5$), while the RQBS-like variant does not: the bootstrap of $\hat\pi$ supplies the prevalence term;
- BayCAP with the population pushforward is lowest at small $r$ ($0.59$ at $r=0.05$) and improves as $r$ grows (H2 of E4.2);
- the realised-table BayCAP and SQS-H with $U$ and $V$ resampling stay between $0.88$ and $0.98$ in every row (H4 of E4.2); the realised-table BayCAP dips to $0.88$–$0.92$ at $n_s=100$ and large $n$ with flat $d$, a residual not investigated here;
- the paper's SQS-H runs at $n=1000$; in the toy, the regime of $r$ where SQS-H covers well ($r\lesssim0.2$) needs about five validation instances per class for every target instance of that class.

What is not known: whether real quantifiers (PACC with a cross-validated surrogate), multiclass problems and real validation sizes reproduce these trends, how large the quantifier bias $b_q$ is, and how much the pool term, absent from the toy, adds on the datasets of `main`. The paper's datasets have $\lvert V\rvert$ equal to $35\%$ of the dataset and pools of $30\%$, so $r$ and $n/P_j$ at $\lvert U\rvert=1000$ vary widely across them; E4.2 measures where each dataset falls, and E4.c1 the pool term.

## B. Relation to other extensions

- **[[ext1_metrics|Extension 1]]**: supplies the metrics; macro-K is a sharp test of the budget (small $e_1$).
- **[[ext2_baycap|Extension 2]]**: its estimand gap is the term $e_3$; its realised-table predictive is one of the completed methods (`BayesCAP-real`). Its cause 4 is the pool term $e_P$ of §T.3: Extension 2 explains it for BayCAP (§T.13 and Appendix A.3 there), and its study across methods (E4.c1, E4.c2, §T.10) is here. Both studies share the dataset subset and the BayCAP draws.
- **Extension 6** ([[extensions]] §6): varies the allocation $\rho$ at fixed $\lvert V\rvert$; here $\lvert V\rvert$ is subsampled at fixed allocation. A factorial $\rho\times\lvert V\rvert$ is out of scope.
- **[[ext3_posterior_propagation|Extension 3]]**: its methods (SQS-B, BayCAP-KDEy in both pushforwards) are placed in the table of §T.6 and enter E4.2 with the same implementations; Extension 3 itself evaluates them only on the protocol of `main`, so their coverage there includes the pool term.
- **[[extensions_to_evaluate]] A**: promoted into this extension (§T.2–§T.5).

## C. Out of scope

- violations of PPS (moved to [[optional_extensions]] §2);
- calibration of the method-specific representation of $\pi^Q$ as such (Extension 3);
- the validation allocation $\rho$ (Extension 6);
- natural sampling of the target bags beyond the remark of §T.9 ([[ext2_baycap]] E2.c2 covers it for BayCAP).
