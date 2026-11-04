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
| Bound on $e_1$, width floors, variances and prior bias of the BayCAP family, label-free plug-ins (self-diagnosis, break-even $n^*$, completed intervals), finite-sample guaranteed intervals (§T.11–§T.15) | [new hypothesis], no experiment yet |
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
- **E4.c2, with or without replacement.** *Why:* under the paper's protocol the pool term $e_P$ is an error that no method represents; if the bags are drawn without replacement, each bag is an i.i.d. sample of the target distribution and $e_P$ becomes part of a luck of the bag of the right size (§T.10). Comparing the two protocols on the same pools shows how much of each method's miscalibration the protocol creates, and how the error split changes. *What it would show:* per method and pair, coverage, amplitude, Winkler score and AE under both protocols, their differences (paired by bag index, since the bags share their prevalences and seeds), and the error split of §T.2 under both protocols; coverage both per pair and averaged over pairs, since without replacement a share $n_j/P_j$ of the luck of the bag is common to the bags of a pair (§T.10). Predicted (§T.10): the methods that represent $e_2$ and $e_3$ (BayCAP with the realised-table predictive) gain the most, more on small pools; those that miss $e_2$ (SQS-H, RQBS) gain too but stay short; BayCAP with the population pushforward gains where the pool term is large but stays short, since it still misses the luck of the bag; for the baselines (DoC, QuAcc, O-LEAP, L-CBPE) the change is measured, not predicted. *Needs:* a check of the pool sizes first; a bag size $N_d$ per dataset, the largest of $\{25,50,100,250,500\}$ not above $\min_jP_j/4$, so that every class fits without replacement at every UPP prevalence and the bags of a pair share at most a quarter of the variance of $e_3$, with the datasets whose $\min_jP_j$ is below $100$ excluded (`mammographic`, with about $124$ instances per class in its pool, stays in with $N_d=25$); UPP bags without replacement, a new option of the sampling of `main`; two runs of the methods of `main` at $N_d$, with and without replacement, with the same pools, $V$, $h$, prevalences and seeds, stored apart from `main`; the stored results of `main` at $N=1000$, which link the run with replacement at $N_d$ to the paper; the pool-gap measures of E4.c1 for the run with replacement.

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

**Accuracy as a sum over classes.** The table $T=C\operatorname{diag}(\pi)$ has entries $T_{ij}=C_{ij}\pi_j$: the probability that an instance is of class $\omega_j$ and is predicted as $\omega_i$. Accuracy is the probability that the prediction is right, i.e. the sum of the diagonal entries (the trace): $\sum_jT_{jj}=\sum_jC_{jj}\pi_j=d^\top\pi$. In words: the share of class $\omega_j$ times the accuracy within class $\omega_j$, summed over the classes.

**Form of the estimate.** Every method considered here estimates accuracy as $\hat A=\sum_j\tilde\pi_j\tilde d_j$, with $\tilde\pi$ an estimate of $\pi^Q$ and $\tilde d$ an estimate of $d$ from $V$:

- SQS-H: the mean accuracy of its bags is $\sum_j(n\hat\pi_j/n)\,\tilde d_j$, where $\tilde d_j$ is the fraction of the class-$\omega_j$ instances of $V$ that $h$ classifies correctly;
- RQBS and SQS-B: the same for each draw $\hat\pi^{(r)}$, averaged over draws;
- BayCAP and BayCAP-KDEy: $\sum_jC^{(r)}_{jj}\pi^{(r)}_j$ for each posterior draw, averaged over draws.

**Derivation.** Adding and subtracting the same quantity does not change a value. We add and subtract two intermediate quantities: $\sum_j\pi^Q_j\tilde d_j$, the right prevalence with the estimated recalls, and $\sum_j\pi^Q_jd_j$, the right prevalence with the true recalls:

$$
\hat A-A(h,U)=\Big(\sum_j\tilde\pi_j\tilde d_j-\sum_j\pi^Q_j\tilde d_j\Big)+\Big(\sum_j\pi^Q_j\tilde d_j-\sum_j\pi^Q_jd_j\Big)+\Big(\sum_j\pi^Q_jd_j-A(h,U)\Big).
$$

Each intermediate quantity appears once with $+$ and once with $-$, so the inner terms cancel and the right-hand side is again $\hat A-A(h,U)$. Collecting terms, the identity is exact for every bag:

$$
\hat A-A(h,U)=\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,\tilde d_j}_{e_1\ \text{prevalence}}+\underbrace{\sum_j\pi^Q_j(\tilde d_j-d_j)}_{e_2\ \text{rate matrix}}+\underbrace{d^\top\pi^Q-A(h,U)}_{e_3\ \text{finite target}}.
$$

For methods that average over draws, the identity holds for each draw and therefore for the average. The order of the add-and-subtract steps is a choice: the second-order cross term $\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d_j)$ is included in $e_1$ (because $e_1$ uses $\tilde d$, not $d$); the other order would move it into $e_2$.

In words: $e_1$ is the error from the wrong prevalence, weighted by the estimated recalls; $e_2$ is the error from the wrong recalls, weighted by the right prevalence; $e_3$ is the error that remains even with $\pi^Q$ and $d$ known exactly, because the $n$ specific instances of the bag do not have exactly the population accuracy.

**Worked example** (illustrative, computed by hand). Take $\pi^Q=(0.5,0.5)$, $\tilde\pi=(0.45,0.55)$, $d=(0.90,0.70)$, $\tilde d=(0.88,0.73)$ and a realised accuracy $A(h,U)=0.79$. Then $\hat A=0.45\cdot0.88+0.55\cdot0.73=0.7975$ and $\hat A-A(h,U)=0.0075$. The parts are $e_1=(-0.05)(0.88)+(0.05)(0.73)=-0.0075$, $e_2=0.5(-0.02)+0.5(0.03)=0.005$ and $e_3=d^\top\pi^Q-A(h,U)=0.80-0.79=0.01$, and indeed $-0.0075+0.005+0.01=0.0075$.

**What the split does and does not say.** Being an identity, the split holds for every bag and every method whose estimate has the form $\sum_j\tilde\pi_j\tilde d_j$; it cannot be wrong. Its scientific content lies in what follows: the variance and the mean of each part (§T.4, §T.5), which depend on how the method obtains $\tilde\pi$ and $\tilde d$, and which parts each method's interval represents (§T.6, §T.7).

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

## T.5 The budget

Two properties give the budget. First, the mean squared error of any error $X$ is its variance plus its squared mean: $\mathbb E[X^2]=\operatorname{Var}(X)+(\mathbb E X)^2$, which follows from the definition $\operatorname{Var}(X)=\mathbb E[X^2]-(\mathbb E X)^2$. Second, the variance of a sum of uncorrelated parts is the sum of their variances (Property B of §T.4 with all weights equal to $1$). Applied to $\hat A-A(h,U)=e_1^U+e_1^V+e_2+e_P+e_3^b$ and grouping the parts by source gives

$$
\operatorname{MSE}(\hat A)\approx\underbrace{\operatorname{Var}(e_3^b)+\operatorname{Var}(e_1^U)}_{\text{from }U,\ \propto1/n}+\underbrace{\operatorname{Var}(e_2)+\operatorname{Var}(e_1^V)+b_q^2}_{\text{from }V,\ \text{independent of }n}+\underbrace{\operatorname{Var}(e_P)}_{\text{from the pool},\ \text{independent of }n},
$$

where $b_q$ is the quantifier's systematic bias propagated to accuracy (contained in $e_1^V$, written separately because it does not average out over draws of $V$). The approximation is first order and drops the within-group correlations of §T.4. The point of the budget is the scaling: as $n$ grows, only the first group shrinks.

The form of the budget is general, since it follows from the identity of §T.2; the size and the mean of each part are not. They depend on how the method obtains $\tilde\pi$ and $\tilde d$: the variances above are those of a method that uses the frequencies of $V$ and a quantifier such as ACC; for a method with a prior on $C$ (BayCAP) or a different quantifier, the same parts have other variances and may have a nonzero mean.

## T.6 Which parts each method represents

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
- **When first order fails.** The first-order approximation of §T.4 (Property C) needs small errors. It can fail with small $\lvert V_j\rvert$, $n$ or $P_j$ (rare classes), where the dropped products are no longer negligible; with an ill-conditioned surrogate, where $S_{11}-S_{10}$ is close to $0$, ACC is strongly curved and $\hat\pi$ can have heavy tails that the delta method underestimates; and near the vertices of the simplex, where quantifiers clip $\hat\pi$, which is not a smooth operation. In the toy of Appendix A the coverage rule is least accurate at large $r$, where the normal approximation for $e_2$ is coarsest.
- **"Represents a part" is an assumption** [new hypothesis]. The coverage rule of §T.7 assumes that each represented part is simulated at its true size. For BayCAP the spread along $e_1$ and $e_2$ is a posterior width, not a sampling variance, and it can be displaced by the prior ([[ext2_baycap]]). A failure of the rule can therefore come either from a wrong $\sigma_{\text{in}}$ or from a missing part; telling them apart needs a comparison of each method's own spread with $\sqrt{\sum_{\text{represented}}\sigma_k^2}$ before the comparison of its coverage with $\eta$.
- **Measuring $e_1^V$ through a large bag from the pool** [new hypothesis]. The approximation of $\pi^\infty(V)$ in E4.1 uses a bag of $10^5$ instances drawn from the pool, which inherits the surrogate's rates on the pool; the measured $e_1^V$ then contains the surrogate's pool part set aside in §T.4, which may not be second order on small pools.
- **$e_2$ and $e_P$ are not separately identifiable, and $e_P$ is one draw per pool** [new hypothesis]. Only their sum $\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)$ is observable; their split relies on the variances of §T.4. Within a dataset $e_P$ is a single realisation, so its variance is seen only across datasets, or across outer seeds if these re-draw the pool (to be checked in the splits of `main`).

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

## T.11 A bound on the prevalence part

§T.11–§T.15 come from a discussion of what the budget can be used for (2026-11-04), not from an experiment of the main section. All their results are new hypotheses; candidate experiments will be attached to them.

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

## T.12 The shortest interval, and floors on the width

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

## T.13 The variances for the two families, and the bias of BayCAP's prior

*Intuition.* The split of §T.2 is the same for every method, but the size and the mean of its parts depend on how $\tilde\pi$ and $\tilde d$ are obtained: frequencies of $V$ and a surrogate quantifier (SQS family), or a posterior with a prior and $h$ itself as quantifier (BayCAP family).

**What is the same for every method.** $e_3^b$ and $e_P$ depend only on the estimand and the protocol (§T.3), so their variances of §T.4 hold for every method.

**SQS family.** $e_2$ uses the frequencies $\tilde d_j=k_j/\lvert V_j\rvert$, with mean $d_j$ and variance $d_j(1-d_j)/\lvert V_j\rvert$ (§T.4). $e_1$ comes from the surrogate's rates $S$ through the quantifier (binary ACC: §T.4).

**BayCAP family, the recalls.** With the `"uniform"` prior, each column of $C$ has the prior $\operatorname{Dir}(\mathbf 1_K)$. Two properties are used.

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

- *Spread.* When $\lvert V_j\rvert\gg K$, $\bar d_j\approx\tilde d_j$ and the posterior variance is about $\tilde d_j(1-\tilde d_j)/\lvert V_j\rvert$, the sampling variance of the SQS family with $\tilde d_j$ in place of $d_j$. The meaning differs: the posterior variance is the uncertainty on $d_j$ given this $V$, the sampling variance the scatter of $\tilde d_j$ over draws of $V$. For large samples the two agree (Bernstein–von Mises theorem), so a posterior interval for $d_j$ covers it in about $95\%$ of the draws of $V$: this is the sense in which BayCAP represents $e_2$ (§T.6).
- *Mean: the bias of the prior.* For BayCAP, $e_2$ averaged over the draws is $\sum_j\pi^Q_j(\bar d_j-d_j)$. By linearity of the expectation and $\mathbb E[k_j]=\lvert V_j\rvert d_j$,

$$
\mathbb E[\bar d_j]-d_j=\frac{\lvert V_j\rvert d_j+1}{\lvert V_j\rvert+K}-d_j=\frac{\lvert V_j\rvert d_j+1-\lvert V_j\rvert d_j-Kd_j}{\lvert V_j\rvert+K}=\frac{1-Kd_j}{\lvert V_j\rvert+K}.
$$

  This bias does not average out over draws of $V$, and the posterior does not widen for it. It is small for $K=2$ and grows with $K$ and with small $\lvert V_j\rvert$, as the large-$K$ displacement found in [[ext2_baycap]]. Over draws of $V$, $\bar d_j$ has variance $\operatorname{Var}(k_j)/(\lvert V_j\rvert+K)^2=\lvert V_j\rvert d_j(1-d_j)/(\lvert V_j\rvert+K)^2$, slightly smaller than that of $\tilde d_j$: the prior trades variance for bias. So, for BayCAP, $\mathbb E[e_2]=\sum_j\pi^Q_j(1-Kd_j)/(\lvert V_j\rvert+K)$ and $\operatorname{Var}(e_2)=\sum_j(\pi^Q_j)^2\lvert V_j\rvert d_j(1-d_j)/(\lvert V_j\rvert+K)^2$.
- *Other priors.* With a prior $\operatorname{Dir}(\alpha_{\cdot j})$ on column $j$ (the `"empirical-bayes"` prior of `methods/baycap.py`), the same steps give $\operatorname{Beta}(k_j+\alpha_{jj},\lvert V_j\rvert-k_j+\alpha_{\text{off},j})$ with $\alpha_{\text{off},j}=\sum_{i\ne j}\alpha_{ij}$, and, writing $\alpha_{\cdot j}=\alpha_{jj}+\alpha_{\text{off},j}$, the bias $(\alpha_{jj}-d_j\alpha_{\cdot j})/(\lvert V_j\rvert+\alpha_{\cdot j})$. It vanishes when the prior mean $\alpha_{jj}/\alpha_{\cdot j}$ equals $d_j$, which is what a prior centred on the typical recall aims at. The hierarchical priors learn their parameters and are not covered by this closed form.

**BayCAP family, the prevalence part (binary).** BayCAP estimates $\pi^Q$ from the predictions of $h$ itself, so its $e_1$ is that of ACC with $h$ as the surrogate: $S_{10}=C_{10}=1-d_0$ and $S_{11}=C_{11}=d_1$, hence $S_{11}-S_{10}=d_1+d_0-1$. Write $g=(d_1-d_0)/(d_1+d_0-1)$ and $\kappa=g^2$. To first order (§T.4, Property C), and ignoring the priors:

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

So for BayCAP the prevalence part is the luck of the bag and the error of $V$, scaled by $\kappa$: it vanishes when $d_1=d_0$ and grows without limit as $d_1+d_0\to1$, i.e. as $h$ approaches a random classifier. The within-source correlations of §T.4, which the budget drops, are here structural, because the quantifier and the classifier are the same. These are frequentist variances of BayCAP's estimate to first order; the priors and the information of the target bag on $C$ change them.

## T.14 Label-free plug-ins, self-diagnosis and completed intervals

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

**Plug-ins for the BayCAP family.** The plug-ins become averages over the posterior draws, which also account for the uncertainty on $(\pi^Q,C)$: $\hat\sigma_{3b}^2=\frac1m\sum_r\frac1n\sum_j\pi^{(r)}_jd^{(r)}_j(1-d^{(r)}_j)$, which is the variance that the realised-table predictive adds by sampling a table for each draw. The spread along $e_2$ is already in the posterior (§T.13); its prior bias is not, but its plug-in is simple: $\frac{1-K\tilde d_j}{\lvert V_j\rvert+K}=\frac{k_j+1}{\lvert V_j\rvert+K}-\frac{k_j}{\lvert V_j\rvert}=\bar d_j-\tilde d_j$ (to check, bring both fractions to the denominator $\lvert V_j\rvert(\lvert V_j\rvert+K)$: the numerator is $\lvert V_j\rvert(k_j+1)-k_j(\lvert V_j\rvert+K)=\lvert V_j\rvert-Kk_j$). The estimated bias is the gap between the posterior mean and the frequency of $V$. For $e_1$, the binary formulas of §T.13 with $\bar d$ plugged in.

**Self-diagnosis: each method's coverage, without labels.** A method's interval gives its own $\sigma_{\text{in}}\approx$ half-width$/1.96$ (§T.7). The missing parts that have a plug-in give an estimate $\hat\eta=\hat\sigma^2_{\text{missing}}/\sigma_{\text{in}}^2$, and the coverage rule of §T.7 a predicted coverage $2\Phi(1.96/\sqrt{1+\hat\eta})-1$. Per method: SQS-H misses $e_1^U$, $e_1^V$, $e_2$, all with plug-ins; RQBS misses $e_1^V$ and $e_2$; the population-pushforward BayCAP misses $e_3^b$; the realised-table BayCAP misses none of the estimable parts. The parts without a plug-in, $e_P$ (it needs $P_j$ and the knowledge of the pool) and $b_q^2$ (a bias that variances cannot reveal), are non-negative additions to $\sigma^2_{\text{out}}$; since the coverage rule decreases as $\eta$ grows, the predicted coverage is then an *upper bound* on the coverage, to first order and as long as the within-source correlations, which can have either sign, are negligible. For BayCAP they are not (§T.13), and the prediction must include them.

**Break-even bag size.** SQS-H represents $e_3^b$ and misses $e_2$, so it is reliable only while $\hat\sigma_2^2$ is small compared with $\hat\sigma_{3b}^2$. Setting the two plug-ins equal and solving for $n$:

$$
\frac1n\sum_j\hat\pi_j\tilde d_j(1-\tilde d_j)=\sum_j\hat\pi_j^2\frac{\tilde d_j(1-\tilde d_j)}{\lvert V_j\rvert}
\quad\Longrightarrow\quad
n^*=\frac{\sum_j\hat\pi_j\tilde d_j(1-\tilde d_j)}{\sum_j\hat\pi_j^2\tilde d_j(1-\tilde d_j)/\lvert V_j\rvert}.
$$

This is the bag size at which the design ratio $r$ of §T.7 equals $1$, estimated without labels; at $n=n^*$ the oracle-$\pi$ SQS-H has coverage about $2\Phi(1.96/\sqrt2)-1\approx0.83$. It gives a practical rule: SQS-type intervals are not to be trusted for bags much larger than $n^*$.

**Budget-completed intervals.** The same plug-ins correct an interval for the parts it misses: $\hat A\pm1.96\sqrt{\sigma_{\text{in}}^2+\hat\sigma^2_{\text{missing}}}$, for example RQBS widened by $\hat\sigma_{1V}^2+\hat\sigma_2^2$, a closed-form approximation of SQS-H with $U$ and $V$ resampling (§T.8), and the population-pushforward BayCAP widened by $\hat\sigma_{3b}^2$, a closed-form approximation of the realised-table BayCAP. For BayCAP the centre can also be moved by the estimated prior bias, $-\sum_j\hat\pi_j(\bar d_j-\tilde d_j)$. These corrections are post-processing of stored outputs. They do not represent $e_P$ and $b_q$, so under the paper's protocol they should still under-cover on small pools, which is itself a test of the pool term.

## T.15 Intervals with a finite-sample guarantee

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
