# Extension 2: A deeper study of BayCAP (estimand, sources of miscalibration, realised-table predictive)

## 0. Summary

**Question.** What does BayCAP's interval actually estimate, and why is it miscalibrated with respect to the quantity that CAP is scored on? A preliminary gate (Q0) first decides whether the estimand gap is large enough on the paper's datasets for the realised-table predictive to be worth studying.

**Main claim (new hypothesis, supported by a toy simulation and by a first analysis of stored `main` results).** BayCAP's interval describes the *population* contingency table $C\operatorname{diag}(\pi^Q)$, while the evaluation scores the *realised* table of the target sample. Four sources decide the direction of its miscalibration: the fixed class counts of the APP protocol (too wide), the estimand gap (too narrow), the uncertainty on $C$ from a finite $V$ (modelled), and the $\operatorname{Dir}(\mathbf 1)$ prior on the columns of $C$, which shrinks recalls towards $1/K$ and displaces the interval on datasets with many classes and few validation instances per class (biased). A realised-table posterior predictive removes the estimand gap without tuning; it cannot remove a bias.

**Candidate method (Calibrated BayCAP).** The realised-table posterior predictive: for each posterior draw, sample the true classes behind the observed predictions and apply the metric to that table. Its comparison with tempering, and every choice of prior or temperature, including the remedy for the bias of the prior on $C$, belong to [[ext10_baycap_priors_tempering|Extension 10]].

**Status.**

| Item | Status |
|---|---|
| `cap-ml` `ACC_BayesCAP`: BayesianCC posterior, population pushforward, $\operatorname{Dir}(\mathbf 1)$ prior on each column of $C$ | [implemented] |
| `methods/baycap.py:BayCAP` with `estimand` $\in\{$population, realised$\}$ (copy of `ACC_BayesCAP`) | [implemented, not evaluated] |
| `tables/estimand_gap.py`: Q0 Step A on stored `main` results (work-plan card WP-2) | [implemented, in review] |
| Toy simulation behind §3 (`scripts/toys/ext2_estimand_gap.py`) | [implemented] |
| Q0 Step B (exact pilot), Q1, Q2, Q5 | [proposal] |
| Source (iv), bias of the prior on $C$ | [new hypothesis] |
| All hypotheses (H0, H2, H6) | [new hypothesis] |

**Reader's guide.** Part I explains BayCAP, the two estimands, and the four sources of miscalibration. Part II states the contributions, hypotheses, and how they map onto the extended paper. Part III specifies the experiments and the components. Tempering and priors are in [[ext10_baycap_priors_tempering|Extension 10]].

# Part I: Understanding

## 1. BayCAP in brief

We follow the notation of the reference paper (§5). BayCAP uses the BayesianCC model. Let $f_{ij}$ be the number of validation instances with predicted class $\omega_i$ and true class $\omega_j$, $n^s_j=\sum_i f_{ij}$, and let $n^t_i$ be the number of target instances predicted as $\omega_i$, with $N^Q=\lvert U\rvert$. The two likelihood terms are

$$
\mathcal L_V(C)=\prod_{j=1}^K \operatorname{Mult}\!\left(f_{\cdot j}\mid n^s_j,\,C_{:j}\right),
\qquad
\mathcal L_U(\pi^Q,C)=\operatorname{Mult}\!\left(n^t\mid N^Q,\,C\pi^Q\right),
$$

with priors $\pi^Q\sim\operatorname{Dir}(\alpha)$ and $C_{:j}\sim\operatorname{Dir}(\mathbf 1)$. The paper also lists a prior on $\pi^P$; the implementations condition on the counts $n^s$, so $\pi^P$ factors out. Every posterior draw yields $T^{(r)}=C^{(r)}\operatorname{diag}(\pi^{Q(r)})$ and a metric sample $A(T^{(r)})$; the point estimate is the mean of the samples and the interval their percentiles.

In both implementations the prior on $\pi^Q$ is configurable (`prior`, uniform by default), while the prior on $C$ is fixed to $\operatorname{Dir}(\mathbf 1)$ per column (`src/uncawe/methods/baycap.py`, copied from `cap-ml`).

*Implementation note:* in `cap-ml` and `quapy`, `p_c_cond_y` is indexed `[y, c]`, so it stores $C^\top$, and the tables returned by `ACC_BayesCAP` follow the `cap.error` layout (true classes on rows).

## 2. Population table vs realised table

*Intuition.* Even if we knew $\pi^Q$ and $C$ exactly, we would not know the accuracy on *this* bag: the classifier is right on a random number of its instances.

**Two different accuracies.** For a target bag $U$ with prevalence $\pi^Q$, two quantities can be called "the accuracy of $h$ on $U$":

- the **population accuracy** $\sum_j C_{jj}\pi^Q_j$, i.e. $A(C\operatorname{diag}\pi^Q)$: the accuracy $h$ has *on average* over all bags with prevalence $\pi^Q$. It is a property of the distribution $Q$;
- the **realised accuracy** $\frac{1}{N^Q}\sum_{x\in U}\mathbf 1[h(x)=y_x]$: the fraction of instances of *this* bag that $h$ classifies correctly. It is computed from the realised contingency table of $U$, i.e. from its true labels, and is what `main` uses as the true value (`true_cts`).

They differ even when $\pi^Q$ and $C$ are known exactly. The bag contains $n_j$ instances of class $\omega_j$, and each one is classified correctly with probability $C_{jj}$, so the number of correct predictions in class $\omega_j$ is $\operatorname{Bin}(n_j,C_{jj})$. The realised accuracy therefore fluctuates around the population accuracy. Example: binary, $n_0=n_1=500$, $C_{00}=0.9$, $C_{11}=0.7$. The population accuracy is $0.8$, while the realised accuracy has standard deviation $\sqrt{500\cdot0.9\cdot0.1+500\cdot0.7\cdot0.3}/1000\approx0.012$, so it typically lies anywhere in $[0.776,0.824]$. This spread is comparable to the width of BayCAP's interval when $V$ is large (about $0.027$ in the toy of §3), which is why the gap matters.

BayCAP's posterior describes $(\pi^Q,C)$, so its interval is for the population accuracy. The evaluation scores the realised accuracy. The interval is thus too narrow by exactly this fluctuation (source (ii) in §3).

**Targeting the realised table.** The labels of $U$ are unknown, but its predictions are observed: $n^t_i$ instances are predicted as $\omega_i$. If we knew $(\pi^Q,C)$, we could ask, for one of these instances, how likely it is that its true class is $\omega_j$. By Bayes' rule, since $Q(\hat Y=\omega_i,Y=\omega_j)=C_{ij}\pi^Q_j$ is entry $(i,j)$ of the population table,

$$
\rho_{ij}=P(Y=\omega_j\mid\hat Y=\omega_i)=\frac{Q(\hat Y=\omega_i,\,Y=\omega_j)}{Q(\hat Y=\omega_i)}=\frac{C_{ij}\,\pi^Q_j}{\sum_l C_{il}\,\pi^Q_l}.
$$

Row $i$ of the population table, normalised to sum to one, gives these probabilities. For $i=j$, $\rho_{ii}$ is the precision of class $\omega_i$. Assigning a true class to every target instance according to $\rho$ produces one plausible realised table, consistent with the predictions actually observed. Since $(\pi^Q,C)$ is not known, the **realised-table posterior predictive** repeats this for every posterior draw $(\pi^{(r)},C^{(r)})$, using $\rho^{(r)}_{ij}=C^{(r)}_{ij}\pi^{(r)}_j/\sum_l C^{(r)}_{il}\pi^{(r)}_l$. The resulting samples combine the uncertainty about $(\pi^Q,C)$ with the bag-level randomness of the realised table.

Row $i$ of the sampled count table is drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, the table is divided by $N^Q$ to give $\tilde T^{(r)}$, and the metric sample is $A(\tilde T^{(r)})$. The posterior is unchanged; only the pushforward changes, at the cost of one multinomial draw per sample. This is the crisp-prediction variant of [[extensions_to_evaluate#B. Transductive BayCAP and the link to CBPE [new]|proposal B]].

## 3. Why BayCAP can be miscalibrated

Four sources of error decide whether BayCAP's intervals are too wide, too narrow, or displaced with respect to the realised table. Sources (i)–(iii) act on the width of the interval; source (iv) acts on its location.

**(i) Fixed class counts: too wide.** $\mathcal L_U$ treats the target bag as built one instance at a time: first the class, then the prediction. The predicted counts then vary for two reasons: how many instances fall in each class, and whether the classifier is right on each. APP fixes the class counts ($n_j\approx N^Q\pi^Q_j$, as `sampling_index` draws exact per-class counts), so the first reason does not exist. The model overstates the variance of the predicted counts by

$$
\operatorname{Var}_{\text{model}}(n^t)-\operatorname{Var}_{\text{APP}}(n^t)
=N^Q\sum_j\pi^Q_j\,(C_{:j}-C\pi^Q)(C_{:j}-C\pi^Q)^\top\;\succeq\;0,
$$

trusts the counts too little, and widens the posterior of $\pi^Q$.

**(ii) Estimand gap: too narrow.** The realised accuracy fluctuates around the population accuracy even with $\pi^Q$ and $C$ known, with variance of order

$$
\frac{1}{(N^Q)^2}\sum_j n_j\,C_{jj}(1-C_{jj}).
$$

The population interval has no term for this. Its share of the error grows as the uncertainty on $C$ shrinks.

**(iii) Uncertainty on $C$ from finite $V$: modelled.** It is real error, captured by $\mathcal L_V$, and dominates when $V$ has few instances per class.

**(iv) Prior on $C$: biased.** *Intuition.* The uniform prior on each column of $C$ acts as one extra validation instance in every cell of the column, so it adds $K$ imaginary instances per class, spread evenly over the predicted classes; with many classes and few validation instances per class, these pull every recall towards $1/K$.

Under $\operatorname{Dir}(\mathbf 1)$, and ignoring the little information that $U$ carries about $C$, the posterior mean of the recall $r_j=C_{jj}$ given $n^s_j$ validation instances of class $\omega_j$ is

$$
\tilde r_j=\frac{f_{jj}+1}{n^s_j+K},
\qquad
\mathbb E[\tilde r_j]-r_j=\frac{1-Kr_j}{n^s_j+K}.
$$

For a good classifier ($r_j\approx1$) the shift is about $-K/n^s_j$: negligible for binary problems, large when $K$ is not small relative to $n^s_j$. The posterior is narrow at the same time, so the interval sits away from the true value instead of being too narrow around it. Neither the realised-table predictive nor any widening can fix this. The choice of prior on $C$ that avoids the bias is studied in [[ext10_baycap_priors_tempering|Extension 10]].

The error of BayCAP's point estimate $\hat A=\sum_j\hat\pi_j\hat r_j$ with respect to the realised accuracy $A=\sum_j\pi_j r_j$ (with $\pi$ the bag's class proportions and $r$ its realised recalls) splits exactly into a recall term, where source (iv) acts, and a prevalence term:

$$
\hat A-A=\underbrace{\sum_j\pi_j(\hat r_j-r_j)}_{\text{recall term}}+\underbrace{\sum_j(\hat\pi_j-\pi_j)\,\hat r_j}_{\text{prevalence term}}.
$$

**Evidence so far.** In Q0 Step A (`tables/estimand_gap.py`, stored `main` results, seed $0$), BayCAP's coverage of the realised accuracy is $0.00$ for all five classifiers on `isolet` and `letter` ($K=26$), and at most $0.11$ on `hand_digits` ($K=10$); the approximate realised-table predictive leaves it at $0.00$–$0.02$ on `isolet` and `letter`. An interval that a wider pushforward cannot rescue is displaced, not too narrow, which is what source (iv) predicts. A first decomposition of the bias on these pairs, computed outside the repository on 2026-10-14, matched the prediction above; its numbers are not reported here until Q5 reproduces them with a script in the repository.

**Other biases (open).** The same first decomposition showed positive biases that source (iv) does not explain: on weak classifiers (e.g. `wine-quality`, `poker_hand`) the bias is mostly in the prevalence term, consistent with a near-singular $C$ that identifies $\pi^Q$ poorly; on `mammographic` the recall term is positive, which a prior shrinking towards $1/K=0.5$ from above cannot produce. Their causes are not investigated; Q5 reports them.

**Toy check of (i)–(iii)** (`scripts/toys/ext2_estimand_gap.py`, binary, not BayCAP on real data): $C_{10}=0.1$, $C_{11}=0.7$, $\lvert U\rvert=1000$, $\pi^Q_1=0.5$ with fixed counts, $n_s$ validation instances per class, uniform priors, $3000$ bags, Monte Carlo error about $\pm0.01$.

| $n_s$ per class | population interval vs population acc. | population interval vs realised acc. | realised-table predictive vs realised acc. | width (population / predictive) |
|---|---|---|---|---|
| $100$ | $0.94$ | $0.92$ | $0.94$ | $0.136$ / $0.147$ |
| $500$ | $0.95$ | $0.85$ | $0.94$ | $0.061$ / $0.081$ |
| $5000$ | $0.97$ | $0.63$ | $0.96$ | $0.027$ / $0.059$ |

What the toy shows:

- with few validation instances per class, source (iii) dominates and coverage is near nominal;
- with many, source (ii) dominates and the population interval under-covers badly ($0.63$);
- source (i) is a residual: against the population accuracy at $n_s=5000$, coverage is $0.97$ on APP bags and $0.94$ on i.i.d. bags;
- the realised-table predictive restores near-nominal coverage at every $n_s$, with wider intervals.

The toy is binary, so source (iv) is negligible in it. What is not known: whether the same holds for real classifiers, multiclass problems, and the paper's datasets. The paper reports that BayCAP "generally achieves high coverage, but often at the cost of wide intervals"; this is consistent with the toy if the results are dominated by datasets with few validation instances per class, but the toy does not prove it. Experiment Q1 settles it.

# Part II: For the paper

## 4. Contributions

1. **Diagnosis.** BayCAP's interval targets the population table, while CAP is evaluated on the realised table of the target sample; the resulting estimand gap, together with the fixed-count protocol and the uncertainty on $C$, explains the direction of BayCAP's miscalibration in width.
2. **Calibrated BayCAP.** The realised-table posterior predictive (§2), a correction that needs neither target labels nor tuning. Its comparison with the best calibration that tempering can achieve is in [[ext10_baycap_priors_tempering|Extension 10]].
3. **The prior on $C$ is not uninformative.** The uniform prior on the columns of $C$ biases BayCAP on datasets with many classes and few validation instances per class, which explains its near-zero coverage there; the remedy is in [[ext10_baycap_priors_tempering|Extension 10]].

## 5. Hypotheses

Gate:

- **H0 (the gap is material).** On a non-negligible share of (dataset, classifier) pairs, the fluctuation of the realised accuracy around the population accuracy is comparable to the spread of BayCAP's posterior, and the realised-table predictive improves the Winkler score. *Confirmed if* Q0 passes its decision rule. *Refuted if* it fails; then the predictive is not presented as a method, Extension 2 reduces to the diagnosis (Q1, Q5), and [[ext10_baycap_priors_tempering|Extension 10]] compares tempering with BayCAP at $\tau=1$ only. H2 is tested only if H0 is confirmed.

Primary:

- **H2 (estimand gap drives miscalibration in width).** Coverage of BayCAP at $\tau=1$ decreases as validation instances per class increase, and falls below nominal on datasets with large $V$ per class. *Confirmed if* Q1 shows this trend on the pairs not affected by source (iv) and Q2(a) shows that the realised-table predictive removes most of their under-coverage. *Refuted if* BayCAP over-covers uniformly; then source (i) or the uncertainty on $C$ dominate, and tempering with $\tau<1$ ([[ext10_baycap_priors_tempering|Extension 10]]) is the relevant correction.
- **H6 (the prior on $C$ biases BayCAP).** On datasets with $K\ge7$, BayCAP's accuracy bias is almost entirely in the recall term, and the recall term matches the shift predicted by the $\operatorname{Dir}(\mathbf 1)$ prior, $\sum_j\pi_j(1-Kr_j)/(n^s_j+K)$. *Confirmed if* in Q5 the predicted and observed recall terms agree, for every classifier, within the error of the pooled-recall approximation, and the prevalence term is small on these pairs. *Refuted if* the recall term does not scale like $K/n^s_j$ across datasets; then the bias comes from the likelihood or from the sampler, not from the prior.

## 6. Paper outline

Where the extension enters the extended paper, and the one claim each piece must support:

| Paper location | Content | Claim | Evidence |
|---|---|---|---|
| §5 (BayCAP), new subsection "Target estimand" | population vs realised table, sources (i)–(iv) | BayCAP's interval targets a different quantity from the one CAP evaluates, and this determines the direction of its miscalibration. | derivation of §3; Q0; Q1; Q2 |
| §5, new subsection "Calibrated BayCAP" | realised-table predictive | BayCAP can be corrected for the estimand gap without target labels or tuning. | Q0; Q2(a); comparison with tempering in [[ext10_baycap_priors_tempering|Extension 10]] |
| §5, short paragraph "Prior on $C$" | shift of the recalls under $\operatorname{Dir}(\mathbf 1)$ | BayCAP's failures on large-$K$ datasets come from its prior on $C$, not from its approach. | §3 (iv); Q5; remedy in [[ext10_baycap_priors_tempering|Extension 10]] |
| §6 (experiments), new subsection "Calibration of BayCAP" | F1, T1, T4 (see §6.1); optionally F0 in §5 | as above | Part III |

Wording if results are mixed: if H0 is refuted, the "Target estimand" subsection becomes a short remark stating that the gap exists but is negligible on the benchmark, and the predictive is not presented as a method; if H2 is refuted, the "Target estimand" subsection becomes a short remark and tempering ([[ext10_baycap_priors_tempering|Extension 10]]) becomes the calibration method; if H6 is refuted, the "Prior on $C$" paragraph is dropped and the large-$K$ failures are reported as an open problem.

### 6.1 Figures and tables

All use Acc; macro-F1 and macro-K versions go to the appendix. Conventions follow the existing paper figures: one marker per classifier, and a dashed line at the nominal coverage $0.95$. Figures and tables of the tempering and prior studies (F2, T2, T3, T5) are in [[ext10_baycap_priors_tempering|Extension 10]].

| Item | Question | Content | What it must show |
|---|---|---|---|
| F0 (optional) | toy, §3 | coverage vs $n_s$ per class for the population interval and the realised-table predictive (from `scripts/toys/ext2_estimand_gap.py`) | the prediction of H2, on the same axes as F1 |
| F1 | Q1 | two panels, coverage and amplitude of BayCAP at $\tau=1$ vs median validation instances per class $\lvert V\rvert\pi^V_j$ (log scale); one point per (dataset, classifier); colour binary vs multiclass; pairs flagged by Q5 as biased drawn with a hollow marker | whether coverage decreases with validation instances per class and crosses $0.95$ (H2) |
| T1 | Q2(a,b) | $2\times2$ table, pushforward (population / predictive) $\times$ bag protocol (APP / i.i.d.), cells mean coverage (amplitude); one block for few and one for many validation instances per class | the size of source (ii) (rows) and source (i) (columns) |
| T4 | Q5 | per dataset (rows ordered by $K$), mean over classifiers of bias, recall term, prevalence term, and predicted recall term; $K$ and median $n^s_j$ as index columns | that the recall term follows the prediction of source (iv) on large-$K$ datasets (H6) |

Appendix and internal material: per-pair table of Q0 (BayCAP vs predictive, ratio $\hat\sigma_g/\hat\sigma_p$; F1 may reuse the ratio as colour); per-dataset table for Q1; per-pair version of T4; `plots/amplitude_vs_coverage.py` with BayCAP and the predictive.

# Part III: For the implementation

## 7. Experiments

Common to all: same bags and seeds across conditions, $100$ bags per configuration as in the paper, measures as in the paper (AE, coverage, amplitude, Winkler at $\alpha=0.05$) on the metrics of [[ext1_metrics]], always scored against the realised table of each bag. Dataset subset: binary and multiclass datasets of `main` that cover small and large $\lvert V\rvert$ per class, well- and ill-conditioned $C$, and at least one large-$K$ dataset. All five classifiers. Order and gates: Q0 (gate) → Q1 → Q2; Q2 runs only if Q0 passes; Q5 is independent of the gate.

### Q0: is the realised-table predictive worth it?

- **Goal / hypothesis:** decide, before the rest of the extension, whether the estimand gap is large enough on the paper's datasets for the realised-table predictive to change BayCAP's intervals; H0. Step A approximates the answer on every (dataset, classifier) pair at no run cost; Step B checks it exactly on a few datasets.
- **Inputs:** Step A: stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (`estim_acc_ranges`, `estim_cts`, `true_cts`), all datasets and classifiers. Step B: 2–3 datasets chosen from Step A (largest ratio $\hat\sigma_g/\hat\sigma_p$, smallest ratio, one multiclass), LR only (pilot), the $100$ `main` bags with the same seeds.
- **Varied:** pushforward (population vs realised-table predictive); nothing else.
- **Outputs:** Step A: per (dataset, classifier), coverage, amplitude and Winkler score for BayCAP and for the approximate predictive, the ratio $\hat\sigma_g/\hat\sigma_p$, and the median number of validation instances per class $\lvert V\rvert\pi^V_j$. Step B: metric samples of `BayesCAP` and `BayesCAP-real` (§8) in the standard result schema, for Acc, macro-F1 and macro-K.
- **Measures:** the paper's measures, always against the realised table of each bag; in Step B also the error of the Step A approximation on Acc.
- **Interpretation:** see the decision rule below.
- **Visualization:** none in the paper; the Step A table goes to the appendix as internal material.
- **Cost:** Step A none; Step B 2–3 datasets × LR × $100$ bags, two NUTS runs per bag (one per `estimand` value, identical draws).

**Step A: approximation on stored results (Acc only).** `main` stores BayCAP's accuracy samples but not its table draws, so the exact predictive cannot be computed from it. For accuracy it can be approximated without building any realised table. Given draw $r$, the predictive accuracy sample is

$$
\tilde A^{(r)}=\frac{1}{N^Q}\sum_i \operatorname{Bin}\!\left(n^t_i,\rho^{(r)}_{ii}\right):
$$

of the $n^t_i$ target instances predicted as $\omega_i$, each is correct with probability $\rho^{(r)}_{ii}$, the precision of class $\omega_i$ under draw $r$, and the off-diagonal cells do not affect accuracy. Its conditional mean is $\sum_i n^t_i\rho^{(r)}_{ii}/N^Q$, close to the stored population accuracy of the draw, and its conditional variance is

$$
\sigma_g^2=\frac{1}{(N^Q)^2}\sum_i n^t_i\,\rho^{(r)}_{ii}\left(1-\rho^{(r)}_{ii}\right).
$$

Step A replaces $\rho^{(r)}$ with a single $\bar\rho$, the mean table `estim_cts` (paper layout) with each row normalised to one, and the sum of binomials with a normal: each approximate predictive sample is the stored accuracy sample plus independent noise $\mathcal N(0,\hat\sigma_g^2)$. Both $n^t$ (the predicted counts) and $\bar\rho$ are label-free; the labels in `true_cts` are used only for scoring. The ratio $\hat\sigma_g/\hat\sigma_p$, with $\hat\sigma_p$ the standard deviation of the stored samples, measures how much the predictive can widen the interval. The approximation ignores the variation of $\rho$ across draws and the non-normality of the binomial sum, and it does not apply to F1 or K, whose off-diagonal cells matter. Step A must run where the `main` results are stored.

**Step B: exact check.** `BayesCAP` and `BayesCAP-real` are the `BayCAP` class of §8 with `estimand="population"` and `estimand="realised"` and the same `random_state`, so they share the posterior draws and differ only in the pushforward: for `BayesCAP-real`, row $i$ of each paper-layout table is drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$ (§2).

**Decision rule** (thresholds are proposals, to be revised after the pilot):

- *pass* if, in Step A and confirmed by Step B, at least one pair in five under-covers (coverage below $0.90$) and the predictive brings it within $[0.93,0.97]$ with a lower Winkler score;
- *fail* if $\hat\sigma_g/\hat\sigma_p<0.3$ on almost every pair (under normality the coverage then changes by about one percentage point or less), or if BayCAP over-covers and the predictive worsens the Winkler score everywhere.

*Open point.* Step A also counts as under-covering the pairs whose interval is displaced by source (iv) (coverage near $0$ for both pushforwards), which no pushforward can bring within the band. Whether to exclude them from the rule is not decided.

**Reuse.** Q1 reuses the per-pair coverage table of Step A; the `BayCAP` class built for Step B is the component that Q2(a) and [[ext10_baycap_priors_tempering|Extension 10]] need, so passing the gate leaves no throwaway code.

### Q1: diagnosis

- **Goal / hypothesis:** direction of BayCAP's miscalibration at $\tau=1$; H2.
- **Inputs:** existing `main` results for `BayesCAP` (metric samples, true tables), all datasets and classifiers.
- **Varied:** nothing (analysis only).
- **Outputs:** per (dataset, classifier): coverage, amplitude, minimum and median validation instances per class ($\lvert V\rvert\,\pi^V_j$), $K$, classifier accuracy on $V$, and the Q5 flag for biased pairs.
- **Measures:** coverage and amplitude vs validation instances per class; rank correlation, computed with and without the pairs flagged by Q5.
- **Interpretation:** coverage decreasing with instances per class and below nominal at the high end supports H2; uniform over-coverage refutes it.
- **Gate:** decides which datasets enter the subset (both regimes must be represented) and informs the $\tau$ grid of [[ext10_baycap_priors_tempering|Extension 10]].
- **Visualization:** F1; per-dataset table in the appendix.
- **Cost:** none.

### Q2: cause

- **Goal / hypothesis:** measure how much of the miscalibration in width found in Q1 comes from sources (ii) and (i) of §3; H2. Source (iii) is modelled and needs no separate test; source (iv) is measured by Q5.
- **Inputs:** BayCAP rerun on the `main` bags of the subset, with the same PRNG key, storing the individual table draws and the predicted counts $n^t$ of every bag (`main` stores neither).
- **Design:** (a) and (b) form a $2\times2$ factorial, pushforward $\times$ bag protocol, on the same prevalences, classifier, and $V$.
- **Outputs:** table draws (sidecar `.npz`) for every variant, so that the predictive can be applied to them and [[ext10_baycap_priors_tempering|Extension 10]] can reuse the $\tau=1$ draws; metric samples.
- **Visualization:** T1.
- **Cost:** (a) no extra NUTS run, only one multinomial draw per posterior sample; (b) one extra BayCAP run per bag.

**(a) Pushforward: isolates source (ii).** The same posterior draws $(\pi^{(r)},C^{(r)})$ are turned into metric samples in two ways: the population pushforward $A(C^{(r)}\operatorname{diag}\pi^{(r)})$ (BayCAP as in the paper) and the realised-table predictive $A(\tilde T^{(r)})$ of §2. The posterior is identical, so the only difference is whether the bag-level fluctuation of the realised table is included. The change in coverage between the two is the effect of the estimand gap. Under H2, the gain is large on datasets with many validation instances per class and small on datasets with few.

**(b) Bag protocol: isolates source (i).** The bags are generated in two ways with the same prevalences $\pi^Q_i$: APP, where the class counts of bag $i$ are fixed to $N^Q\pi^Q_i$ (rounded) as in `main`, and i.i.d., where the class counts are first drawn as $\operatorname{Mult}(N^Q,\pi^Q_i)$ and a bag with those counts is then sampled. i.i.d. bags match BayCAP's target likelihood exactly, so the coverage difference between APP and i.i.d. bags is the effect of the fixed counts. Under H2 this difference is small: in the toy of §3, $0.97$ vs $0.94$ against the population accuracy. Evaluating both pushforwards on both protocols completes the $2\times2$ table and shows whether the two effects add up.

**Interpretation.** H2 is supported if (a) gives a large coverage gain on large-$V$ datasets and (b) a small difference. If instead (b) shows a large effect, the fixed-count likelihood is the main problem and tempering with $\tau<1$, not the predictive, is the relevant correction.

### Q5: bias decomposition

- **Goal / hypothesis:** measure the bias of BayCAP's point estimate and split it into the recall and prevalence terms of §3; H6.
- **Inputs:** stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (`estim_cts`, `true_cts`, `val_prev`), all datasets and classifiers; $\lvert V\rvert$ from the pretrained validation posteriors, as in `tables/estimand_gap.py` (`validation_size`).
- **Varied:** nothing (analysis only).
- **Outputs:** per (dataset, classifier): mean over bags of the bias $\hat A-A$, the recall term, the prevalence term, and the recall term predicted by source (iv), $\sum_j\pi_j(1-K\bar r_j)/(n^s_j+K)$, with $\bar r_j$ the recall of class $\omega_j$ pooled over the $100$ bags and $n^s_j=\lvert V\rvert\pi^V_j$; $K$, median $n^s_j$, mean recall; a flag for pairs whose bias exceeds the posterior standard deviation.
- **Measures:** difference and correlation between observed and predicted recall terms, by range of $K$; share of the bias in each term.
- **Interpretation:** see H6. Pairs with a large prevalence term or an unexplained recall term are listed as open (§3, "Other biases").
- **Visualization:** T4; a scatter of observed vs predicted recall term coloured by $K$ (appendix).
- **Cost:** none.
- **Note:** $\bar r_j$ uses target labels; the prediction is an analysis of the bias, not a method. The exact split uses the mean table `estim_cts`, so it decomposes the bias of the point estimate, not of each draw. The analysis belongs next to `tables/estimand_gap.py`, which already loads the same results and $\lvert V\rvert$.

## 8. Components, tests, and reproducibility

Components:

- **`methods/baycap.py:BayCAP`** (built first, for Q0 Step B): a copy of `cap-ml` `ACC_BayesCAP` (`fit`, `model`, `sample_posterior`, `predict_ct_range`, `predict_ct`; copied from commit `bb20aa5`) with the hyperparameter `estimand` $\in\{$`"population"`, `"realised"`$\}$, default `"population"`.
  - `"population"` returns the posterior tables $C^{(r)}\operatorname{diag}(\pi^{(r)})$, stored transposed as in `cap-ml`, exactly as `ACC_BayesCAP`.
  - `"realised"` returns realised-table predictive samples: for each draw and each predicted class $\omega_i$ (a column in the `cap` layout), the true-class counts are drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, with $\rho^{(r)}_{i\cdot}$ the draw's entries for predicted class $\omega_i$ normalised to one, and the table is divided by $N^Q$.
  - NUTS keeps `PRNGKey(random_state)`; the multinomial draws use a NumPy `Generator` seeded from `random_state`, so both values of `estimand` with the same `random_state` share the posterior draws.
  - The realised pushforward is a plain module-level function (`realised_tables`), reusable on any table draws.
  - Method names in experiments: `BayesCAP` (population, as in `main`) and `BayesCAP-real` (realised).
- **`tables/estimand_gap.py`** (Q0 Step A): per-pair table, gate summary, suggested pilot datasets, LaTeX and PDF output.
- **Q5 analysis** [proposal]: the bias split of §3 and the prediction of source (iv), in `tables/` next to `estimand_gap.py`.

Tests:

- `BayCAP(estimand="population")` reproduces `ACC_BayesCAP` exactly for the same `random_state`;
- with the same `random_state`, both values of `estimand` use identical NUTS draws;
- realised tables have predicted-class margins equal to $n^t/N^Q$ and, with $C=I$, return the observed counts exactly;
- for a fixed draw, the mean of many realised count tables has row $i$ equal to $n^t_i\,\rho^{(r)}_{i\cdot}$;
- fixed `random_state` gives identical outputs; the layout is checked against a hand-computed $2\times2$ example (guards against transposed tables);
- Q5: the recall and prevalence terms sum exactly to the bias on a hand-computed table; with $\operatorname{Dir}(\mathbf 1)$ and a known validation table, the predicted shift equals $(1-Kr_j)/(n^s_j+K)$.

Reproducibility and comparability:

- new experiment `calibration`, writing to `cap.env["OUT_DIR"]/uncawe/calibration`, shared with [[ext10_baycap_priors_tempering|Extension 10]]; `main` and `binform` and their output paths are unchanged;
- every row records prior, $\tau$, seeds, pushforward (population or predictive), and bag protocol (APP or i.i.d.);
- `main` stores only metric samples and mean tables, so Q2 needs BayCAP reruns; Q0 Step A, Q1 and Q5 do not;
- every configuration is fitted on all of $V$, so all rows are directly comparable with `main`;
- a pilot on two datasets estimates the NUTS cost before the full grid.

# Appendix

## A. Toy simulations

`uv run python scripts/toys/ext2_estimand_gap.py` (§3; defaults: $3000$ bags, seed $0$, about $90$ s) is seeded and prints the table used in this note. It also prints the i.i.d.-bag rows used for source (i). It computes the exact BayCAP posterior for a binary problem by sampling $C$ from its validation posterior and weighting a grid over $\pi^Q_1$ by $\mathcal L_U$; it does not use NUTS or `cap-ml`. The toy of the prior on $\pi^Q$ moved to [[ext10_baycap_priors_tempering|Extension 10]].

## B. Relation to other extensions

- **[[ext10_baycap_priors_tempering|Extension 10]]**: priors (on $\pi^Q$ and on $C$) and tempering of BayCAP; it compares tempering with the realised-table predictive of this note, and studies the remedy for source (iv).
- **[[ext1_metrics|Extension 1]]**: supplies the metrics.
- **[[ext3_uncertainty_budget|Extension 3]]**: supplies the $\lvert U\rvert$ axis and the uncertainty budget behind §3 (source (ii) is its finite-$U$ term); its regime map tests H2 within datasets by varying $\lvert U\rvert$ and $\lvert V\rvert$.
- **[[extensions_to_evaluate]] B**: keeps the soft-posterior (CBPE mixture) variant of the realised-table predictive.
- **[[baycap_kdey|BayCAP-KDEy]]**: also targets a population table, so source (ii) applies to it. Out of scope here.
- **[[optional_extensions]]**: §1 covers post-hoc interval rescaling.

## C. Out of scope

- tempering, informative priors, and the choice of prior on $C$ ([[ext10_baycap_priors_tempering|Extension 10]]);
- the soft-posterior variant of the realised-table predictive (proposal B);
- post-hoc interval recalibration;
- violations of PPS.
