# Extension 2: Does BayCAP describe the right accuracy?

## Claim

BayCAP's interval describes the accuracy that the classifier has *on average* over all target bags with a given prevalence (the population table $C\operatorname{diag}(\pi^Q)$). The paper scores that interval against the accuracy *of the bag at hand*, which is computed from its realised contingency table (`true_cts`). The two quantities differ, because on a finite bag the classifier is right on a random number of instances. We want to find out whether this mismatch makes BayCAP's intervals miscalibrated on the paper's datasets. If it does, we want to fix it with the *realised-table predictive*, which turns each posterior draw into a plausible table of the bag itself. The fix needs no target labels and no tuning. This matters because a CAP interval is useful only if it covers the quantity it is judged on, and BayCAP is the paper's Bayesian method.

## Status

| Item | Status |
|---|---|
| `cap-ml` `ACC_BayesCAP` (population pushforward) | [implemented] |
| `methods/baycap.py:BayCAP` with `estimand` $\in\{$population, realised$\}$ | [implemented, not evaluated] |
| Q0 Step A, approximate check on stored `main` results (`tables/estimand_gap.py`, work-plan card WP-2) | [run], analysis below; choice of the Step B datasets pending |
| Q0 Step B, exact pilot (work-plan card WP-3) | [proposal], scheduled |
| Candidate experiments Q1, Q2, Q5 | [proposal], not scheduled |
| Prior on $C$ as the cause of the large-$K$ failures | [new hypothesis] |

The paper-related material of earlier versions (contributions, hypotheses, paper outline, figure plan) has been removed for now; the last version that contains it is commit `62d3ca0`.

# Experiments

All experiments score every interval against the realised table of each target bag, as `main` does. They use the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), the $100$ APP bags of size $1000$ of `main` with the same seeds, and target labels only for scoring.

## Q0: is the realised-table predictive worth it?

**Why.** In a binary toy simulation (§T.3, toy result), BayCAP's interval covers the population accuracy at the nominal rate, but its coverage of the realised accuracy falls to $0.63$ when the validation set has $5000$ instances per class. The realised-table predictive brings it back to $0.96$. The paper's datasets differ in size, in number of classes and in classifier quality, so the toy says nothing about how large the effect is there. Q0 decides whether the predictive changes BayCAP's intervals enough to be worth studying. If it does not, the extension stops at the diagnosis.

### Step A: approximate check on stored results

- **Design.** Inputs: stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (accuracy samples `estim_acc_ranges`, mean table `estim_cts`, realised table `true_cts`), all $15$ datasets and $5$ classifiers ($75$ pairs). Varied: the pushforward, population vs approximate realised-table predictive (§T.4). Measures: coverage, amplitude and Winkler score of both intervals, for accuracy only; the ratio $\hat\sigma_g/\hat\sigma_p$ between the standard deviation the predictive adds and the standard deviation of BayCAP's samples; the median number of validation instances per class $n^s_j=\lvert V\rvert\pi^V_j$. Estimand: population table for BayCAP and realised table for the predictive, both scored against the realised table. Cost: none, no new runs.
- **Representation.** A per-pair table (rows: dataset, classifier; columns: coverage, amplitude and Winkler score for both intervals, $\hat\sigma_g/\hat\sigma_p$, median $n^s_j$), produced by `tables/estimand_gap.py` as LaTeX and PDF. A scatter plot is still to be added: BayCAP coverage on the horizontal axis, predictive coverage on the vertical axis, one point per pair, coloured by $\hat\sigma_g/\hat\sigma_p$, with lines at $0.95$ on both axes. Points that move up from below $0.95$ to the line are pairs the predictive repairs. Points stuck near $(0,0)$ are displaced intervals.
- **Analysis.** Run on 2026-10-14 (`tables/estimand_gap.py`, seed $0$, recorded in the WP-2 log of `docs/work_plan.md`):
  - $53$ of the $75$ pairs under-cover (coverage below $0.90$), so BayCAP's intervals miss the realised accuracy much more often than $5\%$ of the time;
  - the approximate predictive brings only $3$ of them within $[0.93,0.97]$ with a lower Winkler score, although it lowers the Winkler score on $56$ pairs;
  - the ratio $\hat\sigma_g/\hat\sigma_p$ is below $0.3$ on $24$ pairs, where the predictive barely widens the interval, and ranges from $0.07$ (`poker_hand`) to $1.50$ (`dry-bean`);
  - on `isolet` and `letter` ($K=26$) BayCAP's coverage is $0.00$ for every classifier, and the predictive leaves it at $0.00$–$0.02$; on `hand_digits` ($K=10$) it is at most $0.11$; on these datasets BayCAP's estimate is biased low (LR: $-0.18$ on `isolet`, $-0.07$ on `letter`).

  What this shows: BayCAP under-covers the realised accuracy on most pairs, but in most cases the cause is not an interval that is too narrow around the right value. Widening the interval by the bag-level fluctuation, which is all the predictive can do, rarely repairs coverage. On many pairs, and clearly on the large-$K$ datasets, the interval sits away from the true value (§T.5). Under the provisional decision rule below, Step A is neither a clear pass (far fewer than one pair in five is repaired) nor a clear fail (the ratio is small on about a third of the pairs, not on almost all). The pass/fail decision is not taken yet.

### Step B: exact pilot

- **Design.** Inputs: the `main` bags of $2$–$3$ datasets chosen from Step A (largest ratio, smallest ratio, one multiclass; suggested so far: `dry-bean`, $1.50$, and `poker_hand`, $0.07$), classifier LR. Varied: the pushforward only. `BayesCAP` and `BayesCAP-real` are `BayCAP` with `estimand="population"` and `estimand="realised"` and the same `random_state`, so they share the posterior draws (§T.3). Metrics: vanilla accuracy, macro-F1, macro-K. Measures: the paper's measures, plus the error of the Step A approximation on accuracy. Estimand: as in Step A. Cost: one NUTS run per bag shared by the two methods ($2$–$3$ datasets × $100$ bags).
- **Representation.** A table with one row per (dataset, metric) and, for `BayesCAP` and `BayesCAP-real`, columns for coverage, amplitude and Winkler score; for accuracy, an extra column with the coverage that Step A predicted for the predictive, to show how good the approximation is.
- **Analysis.** Not run.

### Decision rule (provisional, to be revised after Step B)

- *Pass* if, in Step A and confirmed by Step B, at least one pair in five under-covers (coverage below $0.90$) and the predictive brings it within $[0.93,0.97]$ with a lower Winkler score.
- *Fail* if $\hat\sigma_g/\hat\sigma_p<0.3$ on almost every pair (the coverage then changes by about one percentage point or less), or if BayCAP over-covers and the predictive worsens the Winkler score everywhere.

*Open point.* The rule counts as under-covering the pairs whose interval is displaced (coverage near $0$ for both pushforwards), which no pushforward can repair. Step A shows that these pairs are many. Whether to exclude them from the rule is not decided.

# Candidate experiments

Ideas that came up while working on Q0. They are not scheduled; one becomes an experiment of the main section only when the user decides so, and it gets a full specification then.

- **Q1, diagnosis against the size of $V$.** *Why:* the toy predicts that BayCAP's coverage of the realised accuracy decreases as the number of validation instances per class grows (§T.3). *What it would show:* coverage and amplitude of BayCAP against the median $n^s_j$, one point per pair, with the displaced pairs marked. *Needs:* the Step A table only, no runs.
- **Q2, cause of the miscalibration in width.** *Why:* two effects act in opposite directions on the width of the interval, the missing bag-level fluctuation (too narrow) and the fixed class counts of APP bags (too wide, §T.6). *What it would show:* a $2\times2$ table, pushforward (population / predictive) × bag protocol (APP / i.i.d. class counts), with coverage and amplitude in each cell. *Needs:* BayCAP reruns on the `main` bags, storing the table draws and the predicted counts, plus i.i.d. bags with the same prevalences.
- **Q5, decomposition of BayCAP's bias.** *Why:* Step A found intervals displaced on large-$K$ datasets, and the uniform prior on $C$ predicts a shift of the recalls of size $(1-Kr_j)/(n^s_j+K)$ (§T.5). *What it would show:* per dataset, ordered by $K$, the bias of the point estimate split exactly into a recall term and a prevalence term, $\hat A-A=\sum_j\pi_j(\hat r_j-r_j)+\sum_j(\hat\pi_j-\pi_j)\,\hat r_j$, with $\pi$ the class proportions of the bag and $r$ its realised recalls, next to the recall term predicted by the prior. The prediction uses target labels (pooled recalls), so it is an analysis, not a method. A first computation outside the repository on 2026-10-14 matched the prediction on the large-$K$ datasets and also found biases the prior does not explain (mostly in the prevalence term on weak classifiers such as `wine-quality` and `poker_hand`; a positive recall term on `mammographic`); its numbers are not reported until Q5 reproduces them with a script in the repository. *Needs:* stored `main` results only, next to `tables/estimand_gap.py`.

# Theory

This part explains what Q0 needs and nothing more. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that an instance of true class $\omega_j$ is predicted as $\omega_i$, and $\pi^Q$ is the vector of class prevalences of the target bag $U$, with $N^Q=\lvert U\rvert$ instances and $K$ classes. The population table is $T=C\operatorname{diag}(\pi^Q)$: entry $(i,j)$ is the probability that a random target instance has true class $\omega_j$ and is predicted as $\omega_i$. Accuracy is the sum of its diagonal, $\sum_j C_{jj}\pi^Q_j$.

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

## T.4 The approximation of Step A

*Intuition:* `main` did not store the posterior tables, so Step A cannot redo the predictive exactly; for accuracy, it adds to each stored accuracy sample a random error of the right size.

For accuracy only the diagonal matters: of the $n^t_i$ instances predicted as $\omega_i$, each is correct with probability $\rho^{(r)}_{ii}$, so the predictive accuracy of draw $r$ is a sum of binomial counts divided by $N^Q$,

$$
\tilde A^{(r)}=\frac{1}{N^Q}\sum_i \operatorname{Bin}\!\left(n^t_i,\rho^{(r)}_{ii}\right),
\qquad
\sigma_g^2=\frac{1}{(N^Q)^2}\sum_i n^t_i\,\rho^{(r)}_{ii}\left(1-\rho^{(r)}_{ii}\right),
$$

with mean close to the population accuracy of the draw and variance $\sigma_g^2$. Step A makes two simplifications: it uses one $\bar\rho$ per bag (the stored mean table `estim_cts`, each row rescaled to one) instead of one per draw, and it replaces the sum of binomials with a normal distribution, which is accurate when the counts are large. Each approximate predictive sample is then the stored accuracy sample plus independent noise $\mathcal N(0,\hat\sigma_g^2)$. Both $n^t$ and $\bar\rho$ use no target labels. The ratio $\hat\sigma_g/\hat\sigma_p$, with $\hat\sigma_p$ the standard deviation of the stored samples, measures how much the predictive can widen the interval: under normality, adding independent noise multiplies the width by $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$, so a ratio of $0.3$ widens it by about $4\%$. The approximation does not apply to macro-F1 or macro-K, whose off-diagonal cells matter; Step B checks it.

## T.5 Why widening cannot repair a displaced interval

*Intuition:* if an interval is centred on the wrong value, making it wider helps only when it becomes wide enough to reach the true value, and then it is too wide to be useful.

The predictive only adds spread around the same centre. If BayCAP's point estimate is far from the realised accuracy compared with the width of the interval, the coverage stays near $0$, which is what Step A finds on `isolet` and `letter`. One possible cause is the prior on $C$ [new hypothesis]. A uniform Dirichlet on a column of $C$ behaves like one extra, imaginary validation instance in each of its $K$ cells. With $f_{jj}$ correct predictions out of $n^s_j$ validation instances of class $\omega_j$, and ignoring the little that $U$ says about $C$, the estimated recall becomes

$$
\tilde r_j=\frac{f_{jj}+1}{n^s_j+K},
\qquad
\mathbb E[\tilde r_j]-r_j=\frac{1-Kr_j}{n^s_j+K},
$$

which pulls every recall towards $1/K$. For a good classifier ($r_j\approx1$) the shift is about $-K/n^s_j$: negligible in binary problems, large when $K$ is not small compared with the number of validation instances per class, as in $26$-class datasets. The posterior is narrow at the same time, so the interval is displaced rather than too narrow. Candidate experiment Q5 would test this; the choice of a prior that avoids the shift belongs to [[ext10_baycap_priors_tempering|Extension 10]].

## T.6 Other effects on the width

- **Fixed class counts (too wide).** $\mathcal L_U$ treats the class of each target instance as random, while APP bags have fixed class counts. The model therefore expects more variation in $n^t$ than there is, trusts the counts too little, and makes the interval wider. In the toy, at $n_s=5000$, coverage of the population accuracy is $0.97$ on APP bags and $0.94$ on bags with random class counts, so the effect is small there. Candidate Q2 would measure it.
- **Uncertainty on $C$ (modelled).** A finite $V$ leaves real uncertainty about $C$; $\mathcal L_V$ represents it, and it dominates when $V$ has few instances per class.

# Implementation

Components for Q0:

- **`methods/baycap.py:BayCAP`** (for Step B): a copy of `cap-ml` `ACC_BayesCAP` (`fit`, `model`, `sample_posterior`, `predict_ct_range`, `predict_ct`; copied from commit `bb20aa5`) with the hyperparameter `estimand` $\in\{$`"population"`, `"realised"`$\}$, default `"population"`.
  - `"population"` returns the posterior tables $C^{(r)}\operatorname{diag}(\pi^{(r)})$, stored transposed as in `cap-ml`, exactly as `ACC_BayesCAP`.
  - `"realised"` returns realised-table predictive samples: for each draw and each predicted class $\omega_i$ (a column in the `cap` layout), the true-class counts are drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$ and the table is divided by $N^Q$.
  - NUTS keeps `PRNGKey(random_state)`; the multinomial draws use a NumPy `Generator` seeded from `random_state`, so both values of `estimand` with the same `random_state` share the posterior draws.
  - The realised pushforward is a module-level function (`realised_tables`), reusable on any table draws.
  - Method names in experiments: `BayesCAP` (population, as in `main`) and `BayesCAP-real` (realised).
- **`tables/estimand_gap.py`** (Step A): per-pair table, gate summary, suggested pilot datasets, LaTeX and PDF output. It must run where the `main` results are stored.
- **Experiment `calibration`** (Step B, configuration `q0`) [proposal]: writes to `cap.env["OUT_DIR"]/uncawe/calibration` in the standard result schema.

Tests:

- `BayCAP(estimand="population")` reproduces `ACC_BayesCAP` exactly for the same `random_state`;
- with the same `random_state`, both values of `estimand` use identical NUTS draws;
- realised tables have predicted-class margins equal to $n^t/N^Q$ and, with $C=I$, return the observed counts exactly;
- for a fixed draw, the mean of many realised count tables has row $i$ equal to $n^t_i\,\rho^{(r)}_{i\cdot}$;
- a fixed `random_state` gives identical outputs; the layout is checked against a hand-computed $2\times2$ example (guards against transposed tables).

Reproducibility and comparability:

- `main` and `binform` and their output paths are unchanged; Step A only reads stored results;
- Step B uses the same bags, seeds and NUTS settings as `main`, so its `BayesCAP` rows should agree with `main` statistically; bitwise agreement is not expected across JAX platforms;
- every row records the prior, the seeds and the pushforward (population or predictive).

# Appendix

## A. Toy simulation

`uv run python scripts/toys/ext2_estimand_gap.py` (defaults: $3000$ bags, seed $0$, about $90$ s) is seeded and prints the table of §T.3, including the bags with random class counts used in §T.6. It computes the exact BayCAP posterior for a binary problem by sampling $C$ from its validation posterior and weighting a grid over $\pi^Q_1$ by $\mathcal L_U$; it does not use NUTS or `cap-ml`.

## B. Relation to other extensions

- **[[ext10_baycap_priors_tempering|Extension 10]]**: priors (on $\pi^Q$ and on $C$) and tempering of BayCAP (temperature $\tau$); it compares tempering with the realised-table predictive and studies a prior on $C$ that avoids the shift of §T.5.
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
