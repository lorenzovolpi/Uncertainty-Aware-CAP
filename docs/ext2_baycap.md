# Extension 2: Does BayCAP describe the right accuracy?

## Claim

BayCAP's interval describes the accuracy that the classifier has *on average* over all target bags with a given prevalence (the population table $C\operatorname{diag}(\pi^Q)$). The paper scores that interval against the accuracy *of the bag at hand*, which is computed from its realised contingency table (`true_cts`). The two quantities differ, because on a finite bag the classifier is right on a random number of instances. We want to find out whether this mismatch makes BayCAP's intervals miscalibrated on the paper's datasets. If it does, we want to fix it with the *realised-table predictive*, which turns each posterior draw into a plausible table of the bag itself. The fix needs no target labels and no tuning. This matters because a CAP interval is useful only if it covers the quantity it is judged on, and BayCAP is the paper's Bayesian method. E2.1 showed that most of BayCAP's miscalibration is not a too-narrow interval but an interval centred on the wrong value, which the predictive cannot repair; the extension therefore also asks where this displacement comes from.

## Status

| Item | Status |
|---|---|
| `cap-ml` `ACC_BayesCAP` (population pushforward) | [implemented] |
| `methods/baycap.py:BayCAP` with `estimand` $\in\{$population, realised$\}$ | [implemented, not evaluated] |
| E2.1, approximate check on stored `main` results (`tables/estimand_gap.py`, work plan §II.10) | [run], analysis below |
| E2.2, causes of the displacement, on stored results (work-plan card WP-3; `tables/displacement.py`, `plots/displacement.py`) | [implemented, not run] |
| Candidate experiments E2.c1, E2.c2, E2.c4 | [proposal], not scheduled |
| The three candidate causes of the displacement (prior on $C$, conditioning of $C$, difficulty of the problem) | [new hypothesis] |

The paper-related material of earlier versions (contributions, hypotheses, paper outline, figure plan) has been removed for now; the last version that contains it is commit `62d3ca0`.

# Experiments

All experiments score every interval against the realised table of each target bag, as `main` does. They use the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), the $100$ APP bags of size $1000$ of `main` with the same seeds, and target labels only for scoring.

E2.1 and the exact run of candidate E2.c4 (once promoted) form a gate: together they decide whether the realised-table predictive changes BayCAP's intervals enough to be worth studying; the decision rule is at the end of this section. E2.2 explains the displacement that E2.1 found.

## E2.1: approximate check on stored results

**Why.** In a binary toy simulation (§T.3, toy result), BayCAP's interval covers the population accuracy at the nominal rate, but its coverage of the realised accuracy falls to $0.63$ when the validation set has $5000$ instances per class. The realised-table predictive brings it back to $0.96$. The paper's datasets differ in size, in number of classes and in classifier quality, so the toy says nothing about how large the effect is there. `main` already stores BayCAP's accuracy samples for every pair, so a first answer for accuracy costs no runs.

- **Design.**
  - **Inputs:** stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (accuracy samples `estim_acc_ranges`, mean table `estim_cts`, realised table `true_cts`), all $15$ datasets and $5$ classifiers ($75$ pairs).
  - **Varied:** the pushforward, population vs approximate realised-table predictive (§T.4).
  - **Measures:** coverage, amplitude and Winkler score of both intervals, for accuracy only; the ratio $\hat\sigma_g/\hat\sigma_p$ between the standard deviation the predictive adds and the standard deviation of BayCAP's samples; the median number of validation instances per class $n^s_j=\lvert V\rvert\pi^V_j$.
  - **Estimand:** population table for BayCAP and realised table for the predictive, both scored against the realised table.
  - **Cost:** none, no new runs.
- **Representation.** A per-pair table (rows: dataset, classifier; columns: coverage, amplitude and Winkler score for both intervals, $\hat\sigma_g/\hat\sigma_p$, median $n^s_j$), produced by `tables/estimand_gap.py` as LaTeX and PDF. A scatter plot is still to be added: BayCAP coverage on the horizontal axis, predictive coverage on the vertical axis, one point per pair, coloured by $\hat\sigma_g/\hat\sigma_p$, with lines at $0.95$ on both axes. Points that move up from below $0.95$ to the line are pairs the predictive repairs. Points stuck near $(0,0)$ are displaced intervals.
- **Analysis.** Run on 2026-10-14 (`tables/estimand_gap.py`, seed $0$, recorded in §II.10 of `docs/work_plan.md`):
  - $53$ of the $75$ pairs under-cover (coverage below $0.90$), so BayCAP's intervals miss the realised accuracy much more often than $5\%$ of the time;
  - the approximate predictive brings only $3$ of them within $[0.93,0.97]$ with a lower Winkler score, although it lowers the Winkler score on $56$ pairs;
  - the ratio $\hat\sigma_g/\hat\sigma_p$ is below $0.3$ on $24$ pairs, where the predictive barely widens the interval, and ranges from $0.07$ (`poker_hand`) to $1.50$ (`dry-bean`);
  - on `isolet` and `letter` ($K=26$) BayCAP's coverage is $0.00$ for every classifier, and the predictive leaves it at $0.00$–$0.02$; on `hand_digits` ($K=10$) it is at most $0.11$; on these datasets BayCAP's estimate is biased low (LR: $-0.18$ on `isolet`, $-0.07$ on `letter`).

  What this shows: BayCAP under-covers the realised accuracy on most pairs, but in most cases the cause is not an interval that is too narrow around the right value. Widening the interval by the bag-level fluctuation, which is all the predictive can do, rarely repairs coverage. On many pairs, and clearly on the large-$K$ datasets, the interval sits away from the true value (§T.5). Under the provisional decision rule below, E2.1 is neither a clear pass (far fewer than one pair in five is repaired) nor a clear fail (the ratio is small on about a third of the pairs, not on almost all). The pass/fail decision is not taken yet.

## E2.2: causes of the displacement

**Why.** E2.1 found that BayCAP's interval misses the realised accuracy on most pairs because it is centred on the wrong value, and that widening it does not help. Three candidate causes could produce this, and each predicts a different pattern (§T.5–§T.8): (1) the uniform prior on the columns of $C$ pulls the recalls towards $1/K$, which shows up in the recall term of the error and grows with $K/n^s_j$; (2) an ill-conditioned $C$ amplifies systematic errors into a wrong $\hat\pi$, which shows up in the prevalence term and grows with the condition number; (3) the classifier performs poorly on the problem even on i.i.d. data. All three can be measured on stored data at no cost, and the result tells how much the exact predictive run (E2.c4) can be expected to change.

- **Design.**
  - **Inputs:** stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (mean table `estim_cts`, realised table `true_cts`, accuracy samples `estim_acc_ranges`), all $75$ pairs; the stored validation posteriors of each classifier (`V_posteriors`) and the validation labels (`validation_y`), from which the validation counts $f_{ij}$ are rebuilt.
  - **Varied:** nothing (analysis). The pairs differ in $K$, in $n^s_j$ and in classifier.
  - **Measures:** vanilla accuracy only. Per bag, then averaged over the $100$ bags of each pair: coverage, amplitude and Winkler score of BayCAP; the shift $\hat A-A$ of the point estimate from the realised accuracy; its recall term and prevalence term (§T.5), which add up exactly to the shift; the prior part of the recall term, $\sum_j\pi_j\left(\frac{f_{jj}+1}{n^s_j+K}-\frac{f_{jj}}{n^s_j}\right)$, computed on the actual $V$; the mean absolute value of the prevalence term, since its sign depends on the bag; the prevalence error $\lVert\hat\pi-\pi\rVert_1$. Per pair: the condition number $\kappa_0(\hat C)$ of the rate matrix of $h$ on $V$, restricted to vectors summing to zero, and the absolute amplification $1/\sigma_{\min}$ on the same vectors, the only one of the two that is informative when $K=2$ (§T.7); the accuracy of $h$ on $V$ (§T.8); $K$, and the minimum and median $n^s_j$.
  - **Estimand:** population table (BayCAP as in `main`), scored against the realised table of each bag. The shift, its terms and the prior part use the bag's true class proportions and recalls, so they are an analysis of the error, not a method.
  - **Cost:** none, no new runs.
- **Representation.**
  - **Table:** one row per (dataset, classifier) pair, grouped by dataset and ordered by $K$, as LaTeX and a landscape PDF like `tables/estimand_gap.py`, so that the diagnostics and the errors of a pair are read side by side. Dataset-level descriptors: $K$, minimum and median $n^s_j$. Three column groups: *diagnostics*, $\kappa_0(\hat C)$, $1/\sigma_{\min}$, accuracy of $h$ on $V$; *interval*, coverage, amplitude, Winkler score; *error*, shift, recall term, prior part of the recall term, prevalence term (mean and mean absolute value), $\lVert\hat\pi-\pi\rVert_1$.
  - **Scatter for cause 1:** recall term against its prior part, one point per pair, coloured by $K$, with the line $y=x$; points on the line are pairs whose recall error is the prior's.
  - **Scatter for causes 2 and 3:** $1/\sigma_{\min}$ on the horizontal axis (log scale; pairs with a singular $\hat C$ are counted, not drawn), accuracy of $h$ on $V$ on the vertical axis, one point per pair, marker by classifier; two panels, coloured by coverage and by shift.
  - **How the causes are read:** cause 1 from how much of the recall term its prior part explains; cause 2 from the prevalence term and $\lVert\hat\pi-\pi\rVert_1$ against $1/\sigma_{\min}$ and $\kappa_0(\hat C)$, and from under-coverage ordered along the horizontal axis; cause 3 from under-coverage ordered along the vertical axis instead; comparing the five classifiers within a dataset separates the causes from $K$. The evidence is correlational: the prior is not changed, so effects that also grow with $K/n^s_j$ cannot be ruled out.
- **Analysis.** Not run.

## Decision rule of the gate (provisional)

- *Pass* if, in E2.1 and confirmed by the exact run (E2.c4, once promoted), at least one pair in five under-covers (coverage below $0.90$) and the predictive brings it within $[0.93,0.97]$ with a lower Winkler score.
- *Fail* if $\hat\sigma_g/\hat\sigma_p<0.3$ on almost every pair (the coverage then changes by about one percentage point or less), or if BayCAP over-covers and the predictive worsens the Winkler score everywhere.

All pairs count, including those whose interval is displaced (coverage near $0$ for both pushforwards), which no pushforward can repair (decision of 2026-10-20).

# Candidate experiments

Ideas that came up while working on E2.1 and E2.2. They are not scheduled and are numbered E2.c<n> (c for candidate); one becomes an experiment of the main section only when the user decides so, and then it gets the next number E2.<n> and a full specification.

- **E2.c1, diagnosis against the size of $V$.** *Why:* the toy predicts that BayCAP's coverage of the realised accuracy decreases as the number of validation instances per class grows (§T.3). *What it would show:* coverage and amplitude of BayCAP against the median $n^s_j$, one point per pair, with the displaced pairs marked. *Needs:* the E2.1 table only, no runs.
- **E2.c2, cause of the miscalibration in width.** *Why:* two effects act in opposite directions on the width of the interval, the missing bag-level fluctuation (too narrow) and the fixed class counts of APP bags (too wide, §T.6). *What it would show:* a $2\times2$ table, pushforward (population / predictive) × bag protocol (APP / i.i.d. class counts), with coverage and amplitude in each cell. *Needs:* BayCAP reruns on the `main` bags, storing the table draws and the predicted counts, plus i.i.d. bags with the same prevalences.
- **E2.c3** was absorbed into E2.2.
- **E2.c4, exact predictive on all pairs.** *Why:* E2.1 approximates the predictive (one $\bar\rho$ per bag, a normal in place of the binomial sum, §T.4); the exact run validates the approximation and gives the confirmation the decision rule asks for. *What it would show:* for each pair, coverage, amplitude and Winkler score of the population and the realised pushforward, next to the coverage E2.1 predicted. *Needs:* `BayCAP` with `estimand="population"` and `estimand="realised"` and the same `random_state`, on all $75$ pairs and the $100$ `main` bags; one NUTS run per bag shared by the two pushforwards, about $7500$ runs; the table draws and $n^t$ stored in a sidecar file.

# Theory

This part explains what E2.1 and E2.2 need and nothing more. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that an instance of true class $\omega_j$ is predicted as $\omega_i$, and $\pi^Q$ is the vector of class prevalences of the target bag $U$, with $N^Q=\lvert U\rvert$ instances and $K$ classes. The population table is $T=C\operatorname{diag}(\pi^Q)$: entry $(i,j)$ is the probability that a random target instance has true class $\omega_j$ and is predicted as $\omega_i$. Accuracy is the sum of its diagonal, $\sum_j C_{jj}\pi^Q_j$.

## T.1 BayCAP in brief

*Intuition:* BayCAP lists many plausible values of $(\pi^Q,C)$, computes the accuracy each one implies, and reports the range of the middle $95\%$.

BayCAP uses two kinds of counts. From the labelled validation set $V$ it uses $f_{ij}$, the number of validation instances of true class $\omega_j$ predicted as $\omega_i$, with $n^s_j=\sum_i f_{ij}$ instances of class $\omega_j$. From the unlabelled target bag it uses $n^t_i$, the number of target instances predicted as $\omega_i$. Two likelihood terms say how probable these counts are for given $(\pi^Q,C)$:

$$
\mathcal L_V(C)=\prod_{j=1}^K \operatorname{Mult}\!\left(f_{\cdot j}\mid n^s_j,\,C_{:j}\right),
\qquad
\mathcal L_U(\pi^Q,C)=\operatorname{Mult}\!\left(n^t\mid N^Q,\,C\pi^Q\right).
$$

In words: within each true class of $V$, the predictions are like draws from the probabilities in column $j$ of $C$; in $U$, each prediction is like a draw from the probabilities $C\pi^Q$ (first pick a class with probability $\pi^Q_j$, then a prediction with probability $C_{ij}$). The priors are a Dirichlet distribution on $\pi^Q$ (uniform by default, `prior`) and a uniform Dirichlet $\operatorname{Dir}(\mathbf 1)$ on each column of $C$ (fixed in the code). A sampler (NUTS) draws $R$ pairs $(\pi^{(r)},C^{(r)})$ from the posterior. Each draw gives a table $T^{(r)}=C^{(r)}\operatorname{diag}(\pi^{(r)})$ and an accuracy $A(T^{(r)})$. The point estimate is the mean of these accuracies, and the interval runs from their $2.5\%$ to their $97.5\%$ percentile.

*Implementation note:* in `cap-ml` and `quapy`, `p_c_cond_y` is indexed `[y, c]`, so it stores $C^\top$, and the tables returned by `ACC_BayesCAP` have true classes on rows.

## T.2 Population accuracy and realised accuracy

*Intuition:* even if we knew $\pi^Q$ and $C$ exactly, we would not know the accuracy on *this* bag, because the classifier is right on a random number of its instances.

- The **population accuracy** $\sum_j C_{jj}\pi^Q_j$ is the accuracy the classifier has on average over all bags with prevalence $\pi^Q$. BayCAP's posterior is about $(\pi^Q,C)$, so its interval is an interval for this quantity.
- The **realised accuracy** is the fraction of instances of this bag that the classifier gets right. It is computed from the bag's realised table (its true labels), and `main` scores against it (`true_cts`).

The bag contains $n_j$ instances of class $\omega_j$, and each one is classified correctly with probability $C_{jj}$, independently of the others. The number of correct predictions in class $\omega_j$ is therefore a binomial count, with mean $n_jC_{jj}$ and variance $n_jC_{jj}(1-C_{jj})$. Adding over the classes and dividing by $N^Q$, the realised accuracy has mean equal to the population accuracy and variance

$$
\frac{1}{(N^Q)^2}\sum_j n_j\,C_{jj}(1-C_{jj}).
$$

*Example.* Binary, $n_0=n_1=500$, $C_{00}=0.9$, $C_{11}=0.7$. The population accuracy is $0.8$. The standard deviation of the realised accuracy is $\sqrt{500\cdot0.9\cdot0.1+500\cdot0.7\cdot0.3}/1000\approx0.012$, so on a typical bag the realised accuracy lies anywhere in $[0.776,0.824]$.

BayCAP's interval contains no term for this fluctuation. When $V$ is small, the uncertainty about $C$ is large and hides it. When $V$ is large, the interval becomes narrower than the fluctuation, and it misses the realised accuracy more often than $5\%$ of the time. This is the *estimand gap*.

## T.3 The realised-table predictive

*Intuition:* we cannot see the true labels of the bag, but we can see its predictions; for each prediction we guess a plausible true class, and the guessed labels give a plausible realised table.

Take a target instance predicted as $\omega_i$. If $(\pi^Q,C)$ were known, Bayes' rule would give the probability that its true class is $\omega_j$:

$$
\rho_{ij}=P(Y=\omega_j\mid\hat Y=\omega_i)=\frac{C_{ij}\,\pi^Q_j}{\sum_l C_{il}\,\pi^Q_l}.
$$

The numerator is entry $(i,j)$ of the population table and the denominator is the sum of row $i$, so $\rho_{i\cdot}$ is row $i$ of the table rescaled to sum to one. On the diagonal, $\rho_{ii}$ is the precision of class $\omega_i$. For each posterior draw $r$, the predictive computes $\rho^{(r)}$ from $(\pi^{(r)},C^{(r)})$ and assigns true classes to the $n^t_i$ instances predicted as $\omega_i$ by one multinomial draw, $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, i.e. by $n^t_i$ independent picks with probabilities $\rho^{(r)}_{i\cdot}$. The resulting count table, divided by $N^Q$, is a sample $\tilde T^{(r)}$ of the realised table, and $A(\tilde T^{(r)})$ is a sample of the realised accuracy. The posterior is unchanged; only the last step changes, at the cost of one multinomial draw per sample. The samples now contain both the uncertainty about $(\pi^Q,C)$ and the bag-level fluctuation of T.2.

*Toy check* (`scripts/toys/ext2_estimand_gap.py`, toy result, binary, not BayCAP on real data): $C_{10}=0.1$, $C_{11}=0.7$, $\lvert U\rvert=1000$, $\pi^Q_1=0.5$ with fixed class counts, $n_s$ validation instances per class, uniform priors, $3000$ bags, Monte Carlo error about $\pm0.01$.

| $n_s$ per class | population interval vs population acc. | population interval vs realised acc. | realised-table predictive vs realised acc. | width (population / predictive) |
|---|---|---|---|---|
| $100$ | $0.94$ | $0.92$ | $0.94$ | $0.136$ / $0.147$ |
| $500$ | $0.95$ | $0.85$ | $0.94$ | $0.061$ / $0.081$ |
| $5000$ | $0.97$ | $0.63$ | $0.96$ | $0.027$ / $0.059$ |

With few validation instances the uncertainty about $C$ dominates and the population interval is nearly right also for the realised accuracy; with many, the estimand gap dominates and its coverage drops to $0.63$; the predictive stays near $0.95$ at every $n_s$, with wider intervals.

## T.4 The approximation of E2.1

*Intuition:* `main` did not store the posterior tables, so E2.1 cannot redo the predictive exactly; for accuracy, it adds to each stored accuracy sample a random error of the right size.

For accuracy only the diagonal matters: of the $n^t_i$ instances predicted as $\omega_i$, each is correct with probability $\rho^{(r)}_{ii}$, so the predictive accuracy of draw $r$ is a sum of binomial counts divided by $N^Q$,

$$
\tilde A^{(r)}=\frac{1}{N^Q}\sum_i \operatorname{Bin}\!\left(n^t_i,\rho^{(r)}_{ii}\right),
\qquad
\sigma_g^2=\frac{1}{(N^Q)^2}\sum_i n^t_i\,\rho^{(r)}_{ii}\left(1-\rho^{(r)}_{ii}\right),
$$

with mean close to the population accuracy of the draw and variance $\sigma_g^2$. E2.1 makes two simplifications: it uses one $\bar\rho$ per bag (the stored mean table `estim_cts`, each row rescaled to one) instead of one per draw, and it replaces the sum of binomials with a normal distribution, which is accurate when the counts are large. Each approximate predictive sample is then the stored accuracy sample plus independent noise $\mathcal N(0,\hat\sigma_g^2)$. Both $n^t$ and $\bar\rho$ use no target labels. The ratio $\hat\sigma_g/\hat\sigma_p$, with $\hat\sigma_p$ the standard deviation of the stored samples, measures how much the predictive can widen the interval: under normality, adding independent noise multiplies the width by $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$, so a ratio of $0.3$ widens it by about $4\%$. The approximation does not apply to macro-F1 or macro-K, whose off-diagonal cells matter; the exact run (E2.c4) would check it.

## T.5 Displacement and the prior on $C$ (cause 1)

*Intuition:* if an interval is centred on the wrong value, making it wider helps only when it becomes wide enough to reach the true value, and then it is too wide to be useful.

**Where the error goes.** Accuracy is the sum of the diagonal of the table, $A=\sum_j\pi_jr_j$, where $r_j=C_{jj}$ is the recall of class $\omega_j$: accuracy is the recalls averaged with weights $\pi_j$. BayCAP's point estimate has the same form, $\hat A=\sum_j\hat\pi_j\hat r_j$, with $\hat\pi_j$ and $\hat r_j$ read from its mean table. The difference splits exactly into two parts:

$$
\hat A-A=\underbrace{\sum_j\pi_j(\hat r_j-r_j)}_{\text{recall term}}+\underbrace{\sum_j(\hat\pi_j-\pi_j)\,\hat r_j}_{\text{prevalence term}},
$$

where $\pi$ and $r$ are the class proportions and realised recalls of the bag. The recall term is the error from estimating how good the classifier is; the prevalence term is the error from estimating how many instances of each class the bag contains. Accuracy is linear in the table, so the mean of BayCAP's accuracy samples is exactly the accuracy of its mean table, and the split applies to the point estimate as it is.

**The prior as imaginary counts.** A column $C_{:j}$ lists the probabilities that an instance of class $\omega_j$ is predicted as each of the $K$ classes. It works like a die with $K$ faces: each validation instance of class $\omega_j$ is a roll, and $f_{ij}$ counts how often face $\omega_i$ came out. The prior $\operatorname{Dir}(\mathbf 1)$ says that every composition of the column is equally likely. A Dirichlet prior has a convenient property: after observing the counts, the posterior is again a Dirichlet whose parameters are the prior's plus the counts. The prior acts exactly like imaginary rolls added before the data, here one roll on each of the $K$ faces. The mean of a face is its count divided by the total count, so the recall becomes

$$
\tilde r_j=\frac{f_{jj}+1}{n^s_j+K}
$$

instead of the plain $f_{jj}/n^s_j$; with $K=2$ this is Laplace's rule for a coin, $(h+1)/(n+2)$. The difference $\tilde r_j-f_{jj}/n^s_j$ is the prior part of the recall error, the quantity E2.2 measures. Averaged over possible validation sets, where $f_{jj}$ is on average $n^s_jr_j$, it equals

$$
\mathbb E[\tilde r_j]-r_j=\frac{n^s_jr_j+1}{n^s_j+K}-r_j=\frac{1-Kr_j}{n^s_j+K},
$$

which is negative whenever the classifier beats guessing ($r_j>1/K$): every recall is pulled towards $1/K$. For a good classifier the pull is about $-K/n^s_j$: negligible for binary problems, large when $K$ is comparable to the number of validation instances per class.

**Why the data do not undo it.** Bayes' rule combines prior and data once, $\text{posterior}\propto\text{prior}\times\text{likelihood}$; the sampler converges to this combination, not to the true $C$. The prior fades only when the real counts $n^s_j$ greatly outnumber the $K$ imaginary ones. Moreover, "every column equally likely" is not neutral about the recall: under $\operatorname{Dir}(\mathbf 1)$ a single entry of the column follows a $\operatorname{Beta}(1,K-1)$ distribution, with mean $1/K$ and $P(r_j>0.9)=0.1^{K-1}$, so with $26$ classes the prior considers a recall above $0.9$ practically impossible. The target bag cannot correct this: its predicted counts depend on $C$ and $\pi^Q$ only through the $K$ numbers $C\pi^Q$, far too few to pin down the $K^2$ entries of $C$.

**Why the interval is displaced and not just wide.** The posterior standard deviation of a recall is about $\sqrt{\tilde r_j(1-\tilde r_j)/(n^s_j+K+1)}$, the usual standard error of a proportion with the imaginary counts included. The shift shrinks like $K/n^s_j$, the standard deviation only like $1/\sqrt{n^s_j}$, so with large $K$ the shift is several standard deviations. Averaging into accuracy makes it worse: the random errors of the $K$ recalls partly cancel, so the standard deviation of the accuracy is smaller than that of each recall, while the prior pushes every recall in the same direction, so the shifts add up. The result is a narrow interval in the wrong place, and the predictive, which only adds spread around the same centre, cannot repair it.

Other symmetric priors with less total weight, such as $\operatorname{Dir}(1/K)$ (one imaginary count per column instead of $K$), are studied in [[ext10_baycap_priors_tempering|Extension 10]].

**The prior on $\pi^Q$ is not a cause.** `main` draws the target prevalences uniformly over all possible prevalence vectors (quapy's `UPP`), which is exactly BayCAP's default prior $\operatorname{Dir}(\mathbf 1)$ on $\pi^Q$. When the prior matches how the bags are generated, the interval for $\pi^Q$ covers at the nominal rate on average over the bags, so this prior alone cannot explain the under-coverage.

## T.6 Other effects on the width

- **Fixed class counts (too wide).** $\mathcal L_U$ treats the class of each target instance as random, while APP bags have fixed class counts. The model therefore expects more variation in $n^t$ than there is, trusts the counts too little, and makes the interval wider. In the toy, at $n_s=5000$, coverage of the population accuracy is $0.97$ on APP bags and $0.94$ on bags with random class counts, so the effect is small there. Candidate E2.c2 would measure it.
- **Uncertainty on $C$ (modelled).** A finite $V$ leaves real uncertainty about $C$; $\mathcal L_V$ represents it, and it dominates when $V$ has few instances per class.

## T.7 Conditioning of $C$ (cause 2)

*Intuition:* if the classifier behaves almost the same on two classes, its predictions say little about how many instances of each class the bag contains, and every small error in $C$ or in the counts becomes a large error in $\hat\pi$.

**A binary example** (illustrative numbers). With $\mathrm{TPR}=C_{11}$ and $\mathrm{FPR}=C_{10}$, the fraction of the bag predicted positive is $q=\mathrm{FPR}+(\mathrm{TPR}-\mathrm{FPR})\,\pi_1$, so $\pi_1=(q-\mathrm{FPR})/(\mathrm{TPR}-\mathrm{FPR})$. With $\mathrm{TPR}=0.9$ and $\mathrm{FPR}=0.1$ the denominator is $0.8$; with $\mathrm{TPR}=0.55$ and $\mathrm{FPR}=0.45$ it is $0.1$, and the chance error of $q$ on a bag of $1000$ instances, about $\pm0.016$, becomes about $\pm0.16$ on $\pi_1$. Any error in $q$, $\mathrm{TPR}$ or $\mathrm{FPR}$ is multiplied by $1/(\mathrm{TPR}-\mathrm{FPR})$.

**The condition number.** For $K$ classes the role of $\mathrm{TPR}-\mathrm{FPR}$ is played by the singular values of $C$, which measure how much $C$ stretches or shrinks vectors in each direction. Singular values are used rather than eigenvalues because $C$ is not symmetric: its eigenvalues can be complex and do not measure stretching. The condition number $\kappa=\sigma_{\max}/\sigma_{\min}$ bounds how much relative errors grow when $\pi$ is recovered from $q=C\pi$:

$$
\frac{\lVert\delta\pi\rVert}{\lVert\pi\rVert}\lesssim\kappa\left(\frac{\lVert\delta q\rVert}{\lVert q\rVert}+\frac{\lVert\delta C\rVert}{\lVert C\rVert}\right),
$$

covering both chance errors in the counts and systematic errors in $C$, such as the prior's. Prevalences sum to one, so every error in $\pi$ or in $q$ sums to zero, and $C$ maps such vectors to vectors that also sum to zero. E2.2 therefore uses $\kappa_0$, the condition number computed only over vectors that sum to zero; $\kappa_0\ge1$, with $\kappa_0=1$ for a perfect classifier.

**The absolute amplification** [new hypothesis, 2026-10-20]. With $K=2$ the vectors that sum to zero form a single direction, so $C$ acts on them as one number, $\mathrm{TPR}-\mathrm{FPR}$, and $\kappa_0=1$ for every binary classifier, however poor: a ratio of the largest to the smallest stretching is uninformative when there is only one stretching. The binary example above uses instead the factor $1/(\mathrm{TPR}-\mathrm{FPR})$, which multiplies absolute errors. Its general form is $1/\sigma_{\min}$, with $\sigma_{\min}$ the smallest singular value of $C$ on the vectors that sum to zero: an error $\delta q$ in the predicted proportions becomes an error of up to $\lVert\delta q\rVert/\sigma_{\min}$ in $\hat\pi$. E2.2 reports both: $1/\sigma_{\min}$ compares pairs of any $K$, including binary ones, while $\kappa_0$ tells, for $K>2$, whether the amplification concerns only some directions.

**Limits of the measure.**

- It is not specific to accuracy. If two classes are completely confused but have the same recall, $C$ is singular ($\kappa_0=\infty$) and their prevalences cannot be told apart, yet the accuracy is still exactly identifiable, because exchanging prevalence between two classes with the same recall does not change it.
- $\hat C$ is estimated on $V$: a column of a rare class rests on few instances, so its $\kappa_0$ is noisy; $n^s_j$ is reported next to it.
- It tends to grow with $K$, because there are more directions in which two columns can be close; comparing the classifiers within a dataset avoids mixing it with $K$, which also drives cause 1.

**The prior and the conditioning** [new hypothesis]. The imaginary counts pull every column towards the flat column $(1/K,\dots,1/K)$, which makes the columns more alike and amplifies errors more. $\kappa_0$ cannot measure this: when all classes have the same $n^s_j$, the prior multiplies every singular value on the vectors that sum to zero by the same factor $n^s_j/(n^s_j+K)$, so their ratio does not change. E2.2 does not measure this effect.

## T.8 Difficulty of the problem on i.i.d. data (cause 3)

*Intuition:* a classifier that is often wrong even without any shift carries little information about the classes.

The difficulty of the problem is measured by the accuracy of $h$ on $V$, which is drawn from the source distribution without any prior shift. No cross-validation is needed: BayCAP evaluates the classifier $h$ that is already trained, and its predictions on $V$ are stored; cross-validating on $V$ would train other classifiers on less data. Difficulty and conditioning are related but different: a weak classifier often has similar columns, but a $26$-class classifier with modest recalls can still separate the classes well. In a correct model, low accuracy alone widens the interval rather than displacing it, so if displacement follows the accuracy on $V$ but not $\kappa_0$, it points to a mechanism not described here.

# Implementation

Components:

- **`tables/estimand_gap.py`** (E2.1): per-pair table, gate summary, suggested pilot datasets, LaTeX and PDF output. It must run where the `main` results are stored.
- **`tables/displacement.py`** (E2.2, work-plan card WP-3): reuses `load_results` and `booktabs_clines` of `estimand_gap.py`; it rebuilds $f_{ij}$ as `BayCAP.fit` does, from the stored validation posteriors (`PretrainInfo.load_posteriors`) and the validation labels (`PretrainInfo.load_dataset_bundle`), computes the measures of E2.2 per bag and per pair, and writes the per-pair CSV and the table (LaTeX and landscape PDF) under `<OUT_DIR>/uncawe/tables/displacement/`. $\kappa_0$ and $1/\sigma_{\min}$ come from the singular values of $Q^\top\hat CQ$, with $Q$ an orthonormal basis of the vectors summing to zero; both are infinite when $\hat C$ is singular on them. It also reports, per pair, the largest gap between the mean of the accuracy samples and the accuracy of the mean table, which is zero up to rounding because accuracy is linear.
- **`plots/displacement.py`** (E2.2): both scatters, from the per-pair table, under `<OUT_DIR>/uncawe/plots/main/displacement/`.
- **`methods/baycap.py:BayCAP`** (for E2.c4): a copy of `cap-ml` `ACC_BayesCAP` (`fit`, `model`, `sample_posterior`, `predict_ct_range`, `predict_ct`; copied from commit `bb20aa5`) with the hyperparameter `estimand` $\in\{$`"population"`, `"realised"`$\}$, default `"population"`.
  - `"population"` returns the posterior tables $C^{(r)}\operatorname{diag}(\pi^{(r)})$, stored transposed as in `cap-ml`, exactly as `ACC_BayesCAP`.
  - `"realised"` returns realised-table predictive samples: for each draw and each predicted class $\omega_i$ (a column in the `cap` layout), the true-class counts are drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$ and the table is divided by $N^Q$.
  - NUTS keeps `PRNGKey(random_state)`; the multinomial draws use a NumPy `Generator` seeded from `random_state`, so both values of `estimand` with the same `random_state` share the posterior draws.
  - The realised pushforward is a module-level function (`realised_tables`), reusable on any table draws.
  - Method names in experiments: `BayesCAP` (population, as in `main`) and `BayesCAP-real` (realised).
- **Experiment `calibration`** (E2.c4) [proposal]: writes to `cap.env["OUT_DIR"]/uncawe/calibration` in the standard result schema.

Tests:

- `BayCAP(estimand="population")` reproduces `ACC_BayesCAP` exactly for the same `random_state`;
- with the same `random_state`, both values of `estimand` use identical NUTS draws;
- realised tables have predicted-class margins equal to $n^t/N^Q$ and, with $C=I$, return the observed counts exactly;
- for a fixed draw, the mean of many realised count tables has row $i$ equal to $n^t_i\,\rho^{(r)}_{i\cdot}$;
- a fixed `random_state` gives identical outputs; the layout is checked against a hand-computed $2\times2$ example (guards against transposed tables);
- E2.2: the recall and prevalence terms sum exactly to the shift on a hand-computed table; the prior part equals $\frac{f_{jj}+1}{n^s_j+K}-\frac{f_{jj}}{n^s_j}$ on a hand-computed validation table; $\kappa_0$ and $1/\sigma_{\min}$ match hand-computed $2\times2$ ($\kappa_0=1$, $1/\sigma_{\min}=1/(\mathrm{TPR}-\mathrm{FPR})$) and $3\times3$ examples and are infinite for singular ones; the recall and prevalence terms add up to the shift also on a bag with an absent class; $\hat C$ is in the paper's layout (`tests/test_displacement.py`).

Reproducibility and comparability:

- `main` and `binform` and their output paths are unchanged; E2.1 and E2.2 only read stored results;
- E2.c4 would use the same bags, seeds and NUTS settings as `main`, so its `BayesCAP` rows should agree with `main` statistically; bitwise agreement is not expected across JAX platforms;
- every row records the prior, the seeds and the pushforward (population or predictive).

# Appendix

## A. Toy simulation

`uv run python scripts/toys/ext2_estimand_gap.py` (defaults: $3000$ bags, seed $0$, about $90$ s) is seeded and prints the table of §T.3, including the bags with random class counts used in §T.6. It computes the exact BayCAP posterior for a binary problem by sampling $C$ from its validation posterior and weighting a grid over $\pi^Q_1$ by $\mathcal L_U$; it does not use NUTS or `cap-ml`.

## B. Relation to other extensions

- **[[ext10_baycap_priors_tempering|Extension 10]]**: priors (on $\pi^Q$ and on $C$) and tempering of BayCAP (temperature $\tau$); it compares tempering with the realised-table predictive and studies priors on $C$ that avoid the shift of §T.5, such as $\operatorname{Dir}(1/K)$.
- **[[ext1_metrics|Extension 1]]**: supplies the metrics.
- **[[ext3_uncertainty_budget|Extension 3]]**: the estimand gap is its finite-$U$ term; its regime map varies $\lvert U\rvert$ and $\lvert V\rvert$.
- **[[extensions_to_evaluate#B. Transductive BayCAP and the link to CBPE [new]|Proposal B]]**: keeps the soft-posterior (CBPE mixture) variant of the realised-table predictive.
- **[[baycap_kdey|BayCAP-KDEy]]**: also targets a population table, so the estimand gap applies to it. Out of scope here.
- **[[optional_extensions]]**: §1 covers post-hoc interval rescaling.

## C. Out of scope

- tempering, informative priors, and the choice of prior on $C$ ([[ext10_baycap_priors_tempering|Extension 10]]);
- the soft-posterior variant of the realised-table predictive (Proposal B);
- post-hoc interval recalibration;
- violations of PPS.
