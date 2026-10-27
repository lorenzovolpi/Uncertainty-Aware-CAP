# Extension 2: Does BayCAP describe the right accuracy?

## Claim

BayCAP's interval describes the accuracy that the classifier has *on average* over all target bags with a given prevalence (the population table $C\operatorname{diag}(\pi^Q)$). The paper scores that interval against the accuracy *of the bag at hand*, which is computed from its realised contingency table (`true_cts`). The two quantities differ, because on a finite bag the classifier is right on a random number of instances. We want to find out whether this mismatch makes BayCAP's intervals miscalibrated on the paper's datasets. If it does, we want to fix it with the *realised-table predictive*, which turns each posterior draw into a plausible table of the bag itself. The fix needs no target labels and no tuning. This matters because a CAP interval is useful only if it covers the quantity it is judged on, and BayCAP is the paper's Bayesian method. E2.1 showed that most of BayCAP's miscalibration is not a too-narrow interval but an interval centred on the wrong value, which the predictive cannot repair; the extension therefore also asks where this displacement comes from. E2.2 found that the largest displacements come from the uniform prior on the columns of $C$, which pulls every recall towards $1/K$; E2.4 tests a prior learned from the validation data instead, which pulls each recall towards the typical recall of the same classifier.

## Status

| Item | Status |
|---|---|
| `cap-ml` `ACC_BayesCAP` (population pushforward) | [implemented] |
| `methods/baycap.py:BayCAP` with `estimand` $\in\{$population, realised$\}$ | [implemented], evaluated by E2.3 as `BayesCAP-real` |
| E2.1, approximate check on stored `main` results (`tables/estimand_gap.py`, work plan §II.10) | [run], analysis below |
| E2.2, causes of the displacement, on stored results (work-plan card WP-3; `tables/displacement.py`, `plots/displacement.py`) | [run], analysis below |
| E2.3, the exact realised-table predictive as the method `BayesCAP-real` of `main`, and the E2.2 analysis repeated on it (work-plan card WP-4; `tables/pushforward.py`, `plots/pushforward.py`) | [run], analysis below |
| `BayCAP` with the hyperparameter `c_prior` $\in\{$uniform, empirical Bayes$\}$ for the prior on the columns of $C$ (work-plan card WP-5) | implemented (work plan §II.5), not evaluated |
| E2.4, BayCAP with an empirical-Bayes prior on the recalls, with a zero-cost check of its effect and the NUTS diagnostics of all BayCAP methods (work-plan card WP-6) | [proposal], scheduled; not run |
| Candidate experiments E2.c1, E2.c2 | [proposal], not scheduled |
| The three candidate causes of the displacement (prior on $C$, conditioning of $C$, difficulty of the problem) | [new hypothesis], examined by E2.2 |
| Cause 4, a difference between $V$ and the test pool shared by all bags | [new hypothesis], not tested |

The paper-related material of earlier versions (contributions, hypotheses, paper outline, figure plan) has been removed for now; the last version that contains it is commit `62d3ca0`.

# Experiments

All experiments score every interval against the realised table of each target bag, as `main` does. They use the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), the $100$ APP bags of size $1000$ of `main` with the same seeds, and target labels only for scoring.

E2.1 estimates, without new runs, how the realised-table predictive would change BayCAP's intervals; E2.2 explains the displacement that E2.1 found; E2.3 runs the exact predictive in `main` and repeats both analyses on it; E2.4 replaces the prior on $C$ that E2.2 found responsible for the largest displacements, with both pushforwards.

## E2.1: approximate check on stored results

**Why.** In a binary toy simulation (§T.3, toy result), BayCAP's interval covers the population accuracy at the nominal rate, but its coverage of the realised accuracy falls to $0.63$ when the validation set has $5000$ instances per class. The realised-table predictive brings it back to $0.96$. The paper's datasets differ in size, in number of classes and in classifier quality, so the toy says nothing about how large the effect is there. `main` already stores BayCAP's accuracy samples for every pair, so a first answer for accuracy costs no runs.

- **Design.**
  - **Inputs:** stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (accuracy samples `estim_acc_ranges`, mean table `estim_cts`, realised table `true_cts`), all $15$ datasets and $5$ classifiers ($75$ pairs).
  - **Varied:** the pushforward, population vs approximate realised-table predictive (§T.4).
  - **Measures:** coverage, amplitude and Winkler score of both intervals, for accuracy only; the ratio $\hat\sigma_g/\hat\sigma_p$ between the standard deviation the predictive adds and the standard deviation of BayCAP's samples; the median number of validation instances per class $n^s_j=\lvert V\rvert\pi^V_j$.
  - **Estimand:** population table for BayCAP and realised table for the predictive, both scored against the realised table.
  - **Cost:** none, no new runs.
- **Representation.** A per-pair table (rows: dataset, classifier; columns: coverage, amplitude and Winkler score for both intervals, $\hat\sigma_g/\hat\sigma_p$, median $n^s_j$), produced by `tables/estimand_gap.py` as LaTeX and PDF. A scatter plot is still to be added: BayCAP coverage on the horizontal axis, predictive coverage on the vertical axis, one point per pair, coloured by $\hat\sigma_g/\hat\sigma_p$, with lines at $0.95$ on both axes. Points that move up from below $0.95$ to the line are pairs the predictive repairs. Points stuck near $(0,0)$ are displaced intervals.
- **Analysis.** First run on 2026-10-14 on the old `BayesCAP` results, generated again on 2026-10-22 on `BayesCAP` recomputed on CPU (`tables/estimand_gap.py`, seed $0$, recorded in §II.10 of `docs/work_plan.md`); the numbers below are those of the CPU results, and differ from the first run only in the count of repaired pairs ($3$ before) and in the largest ratio ($1.50$ before, `dry-bean`):
  - $53$ of the $75$ pairs under-cover (coverage below $0.90$), so BayCAP's intervals miss the realised accuracy much more often than $5\%$ of the time;
  - the approximate predictive brings only $5$ of them within $[0.93,0.97]$ with a lower Winkler score (`spambase` SVM, `mammographic` kNN, `chess` SVM, `nursery` kNN and RF), although it lowers the Winkler score on $56$ pairs;
  - the ratio $\hat\sigma_g/\hat\sigma_p$ is below $0.3$ on $24$ pairs, where the predictive barely widens the interval, and ranges from $0.07$ (`poker_hand` LR) to $1.78$ (`letter` RF), with about $1.45$–$1.50$ on `dry-bean`;
  - on `isolet` and `letter` ($K=26$) BayCAP's coverage is $0.00$ for every classifier, and the predictive leaves it at $0.00$–$0.02$; on `hand_digits` ($K=10$) BayCAP's coverage is at most $0.10$ and the predictive's at most $0.40$; on these datasets BayCAP's estimate is biased low (LR: $-0.18$ on `isolet`, $-0.07$ on `letter`).

  What this shows: BayCAP under-covers the realised accuracy on most pairs, but in most cases the cause is not an interval that is too narrow around the right value. Widening the interval by the bag-level fluctuation, which is all the predictive can do, rarely repairs coverage. On many pairs, and clearly on the large-$K$ datasets, the interval sits away from the true value (§T.5).

## E2.2: causes of the displacement

**Why.** E2.1 found that BayCAP's interval misses the realised accuracy on most pairs because it is centred on the wrong value, and that widening it does not help. Three candidate causes could produce this, and each predicts a different pattern (§T.5–§T.8): (1) the uniform prior on the columns of $C$ pulls the recalls towards $1/K$, which shows up in the recall term of the error and grows with $K/n^s_j$; (2) an ill-conditioned $C$ amplifies systematic errors into a wrong $\hat\pi$, which shows up in the prevalence term and grows with the condition number; (3) the classifier performs poorly on the problem even on i.i.d. data. All three can be measured on stored data at no cost, and the result tells how much the exact predictive run (E2.3) can be expected to change.

- **Design.**
  - **Inputs:** stored `main` results for `BayesCAP` at $\lvert U\rvert=1000$ (mean table `estim_cts`, realised table `true_cts`, accuracy samples `estim_acc_ranges`), all $75$ pairs; the stored validation posteriors of each classifier (`V_posteriors`) and the validation labels (`validation_y`), from which the validation counts $f_{ij}$ are rebuilt.
  - **Varied:** nothing (analysis). The pairs differ in $K$, in $n^s_j$ and in classifier.
  - **Measures:** vanilla accuracy only. Per bag, then averaged over the $100$ bags of each pair: coverage, amplitude and Winkler score of BayCAP; the shift $\hat A-A$ of the point estimate from the realised accuracy; its recall term and prevalence term (§T.5), which add up exactly to the shift; the prior part of the recall term, $\sum_j\pi_j\left(\frac{f_{jj}+1}{n^s_j+K}-\frac{f_{jj}}{n^s_j}\right)$, computed on the actual $V$; the mean absolute value of the prevalence term, since its sign depends on the bag; the prevalence error $\lVert\hat\pi-\pi\rVert_1$. Per pair: the coverage predicted from the mean shift and the mean amplitude alone (see the analysis); the condition number $\kappa_0(\hat C)$ of the rate matrix of $h$ on $V$, restricted to vectors summing to zero, and the absolute amplification $1/\sigma_{\min}$ on the same vectors, the only one of the two that is informative when $K=2$ (§T.7); the accuracy of $h$ on $V$ (§T.8); $K$, and the minimum and median $n^s_j$.
  - **Estimand:** population table (BayCAP as in `main`), scored against the realised table of each bag. The shift, its terms and the prior part use the bag's true class proportions and recalls, so they are an analysis of the error, not a method.
  - **Cost:** none, no new runs.
- **Representation.**
  - **Table:** one row per (dataset, classifier) pair, grouped by dataset and ordered by $K$, as LaTeX and a landscape PDF like `tables/estimand_gap.py`, so that the diagnostics and the errors of a pair are read side by side. Dataset-level descriptors: $K$, minimum and median $n^s_j$. Three column groups: *diagnostics*, $\kappa_0(\hat C)$, $1/\sigma_{\min}$, accuracy of $h$ on $V$; *interval*, coverage, predicted coverage $\mathcal C_s$, amplitude, Winkler score; *error*, shift, recall term, prior part of the recall term, prevalence term (mean and mean absolute value), $\lVert\hat\pi-\pi\rVert_1$.
  - **Scatter for cause 1:** recall term against its prior part, one point per pair, coloured by $K$, with the line $y=x$; points on the line are pairs whose recall error is the prior's.
  - **Scatter for causes 2 and 3:** $1/\sigma_{\min}$ on the horizontal axis (log scale; pairs with a singular $\hat C$ are counted, not drawn), accuracy of $h$ on $V$ on the vertical axis, one point per pair, marker by classifier; two panels, coloured by coverage and by shift.
  - **How the causes are read:** cause 1 from how much of the recall term its prior part explains; cause 2 from the prevalence term and $\lVert\hat\pi-\pi\rVert_1$ against $1/\sigma_{\min}$ and $\kappa_0(\hat C)$, and from under-coverage ordered along the horizontal axis; cause 3 from under-coverage ordered along the vertical axis instead; comparing the five classifiers within a dataset separates the causes from $K$. The evidence is correlational: the prior is not changed, so effects that also grow with $K/n^s_j$ cannot be ruled out.
- **Analysis.** First run on 2026-10-20 on all $75$ pairs on the old `BayesCAP` results; the table was generated again on 2026-10-22 on `BayesCAP` recomputed on CPU, with the predicted coverage $\mathcal C_s$ (`tables/displacement.py`, recorded in §II.11 of `docs/work_plan.md`), and the numbers below are those of the CPU results, in `<OUT_DIR>/uncawe/tables/displacement/displacement.csv`. The two scatters were drawn again on the CPU results on 2026-10-23 (`plots/displacement.py`, commit `975f2ac`); the CPU results change the quoted numbers by at most a few hundredths and do not change any conclusion. Notation: $R$ recall term, $R_0$ its prior part, $P$ prevalence term; $R-R_0$, the part of the recall term that the prior does not explain, and $s=\text{shift}/(\text{amplitude}/2)$, the shift measured in half-widths of the interval, are differences and ratios of columns of the per-pair CSV.
  - **Coverage is a matter of displacement.** The predicted coverage $\mathcal C_s$ answers the question: what coverage would BayCAP have if its only fault were the shift? It is obtained in four steps.
    1. *What a correct interval looks like.* Across the bags of a pair, the error $\hat A-A$ of the point estimate varies. If BayCAP described its uncertainty correctly, this error would follow a bell curve (a normal distribution) centred at $0$ with some standard deviation $\sigma$, and a $95\%$ interval $\hat A\pm h$ would have half-width $h=1.96\,\sigma$, because $95\%$ of a normal distribution lies within $1.96$ standard deviations of its centre.
    2. *Add only the shift.* Keep the width, and so $\sigma=h/1.96$, but move the centre of the bell curve to the mean shift $\delta=\overline{\hat A-A}$. Measured in half-widths, the shift is $s=\delta/h$.
    3. *When the interval covers.* $A$ is inside $\hat A\pm h$ exactly when the error lies between $-h$ and $h$. Subtracting the centre $\delta$ and dividing by $\sigma$ turns the error into a standard normal variable, and the two limits into $(-h-\delta)/\sigma=-1.96(1+s)$ and $(h-\delta)/\sigma=1.96(1-s)$.
    4. *The probability.* With $\Phi$ the cumulative distribution function of the standard normal (the probability of falling below a value), the probability that the error falls between the two limits is $\mathcal C_s=\Phi\big(1.96(1-s)\big)-\Phi\big(-1.96(1+s)\big)$. It gives $0.95$ for $s=0$, about $0.83$ for $s=\pm0.5$, about $0.50$ for $s=\pm1$ (the true value on an end of the interval) and $0.025$ for $s=\pm2$.

    How to read it: coverage close to $\mathcal C_s$ means the width is right and the shift explains the whole loss; coverage below $\mathcal C_s$ means the interval is also too narrow for the spread of the error; coverage above $\mathcal C_s$ means that the shift is not the same in every bag (the formula assumes it is), or that the interval is wider than it needs to be. The formula assumes a symmetric interval and a normal error, and uses the mean amplitude of the pair; it is a reading aid, not a model of BayCAP. On `main` it follows the observed coverage closely: correlation $0.98$ over the $75$ pairs; e.g. `spambase` RF, $s=-0.89$, $\mathcal C_s=0.58$, observed $0.53$; `mammographic` MLP, $s=1.60$, $0.12$ against $0.10$; `dry-bean` LR, $s=-1.32$, $0.27$ against $0.29$. So the question is mostly what moves the centre.
  - **Cause 1, the prior on $C$, explains the largest displacements.** On `dry-bean`, `hand_digits`, `letter` and `isolet` ($K\ge7$, $\kappa_0\approx1$–$2$, $P\approx0$) the shift equals the recall term and the recall term equals its prior part: the points lie on $y=x$ in the cause-1 scatter. The coverage there is $0.00$–$0.33$, because the intervals are narrow ($0.014$–$0.049$) and $\lvert s\rvert$ ranges from $1.2$ to $8.9$. The size follows §T.5: `isolet` LR, $n^s_j\approx105$ and recall on $V$ $0.95$, gives $R_0=-0.181$ against a shift of $-0.178$; `letter`, with about $2.5$ times more validation instances per class, has shifts of $-0.065$ to $-0.084$. On `chess` ($K=15$) the recall term also follows the prior part, and the better classifiers cover worse (RF, accuracy on $V$ $0.62$, coverage $0.25$; LR, $0.33$, $0.88$): higher recalls are farther from $1/K$, so the prior pulls them more, and their intervals are narrower.
  - **Cause 2, conditioning, displaces only together with a systematic error.** Bad conditioning alone becomes width: `connect-4` LR ($\kappa_0=88.6$, amplitude $0.48$) and `shuttle` LR ($\kappa_0\approx3\cdot10^4$, amplitude $0.43$) cover at $0.98$ and $0.92$. It displaces where the prevalence error has the same sign in every bag ($\lvert P\rvert\approx\lvert\bar P\rvert$): `german` ($1/\sigma_{\min}=2.7$–$4.9$, $\lVert\hat\pi-\pi\rVert_1=0.20$–$0.44$, $P$ is $0.110$ of the $0.158$ shift of SVM, coverage $0.18$) and `wine-quality` ($\kappa_0=\infty$ for LR and SVM, LR $\bar P=0.202$, $\lvert P\rvert=0.203$, coverage $0.35$). Where $P$ changes sign across bags (`wine-quality` kNN, $\bar P=-0.034$, $\lvert P\rvert=0.059$) coverage holds ($1.00$). Why the prevalence error is systematic on a singular $\hat C$ is not explained: in the directions $\hat C$ cannot identify, the posterior falls back on the $\operatorname{Dir}(\mathbf 1)$ prior on $\pi^Q$, which matches UPP and should not bias the error on average over the bags.
  - **Cause 3, difficulty, does not act on its own.** A low accuracy on $V$ goes with under-coverage only through the conditioning (`german`, `wine-quality`, `poker_hand`); in the conditioning scatter the low-coverage pairs sit at $1/\sigma_{\min}\approx1$ and high accuracy on $V$, and on `chess` the relation is reversed. `poker_hand` (minimum $n^s_j=3$, $\kappa_0=\infty$ for every classifier, $\lVert\hat\pi-\pi\rVert_1=0.82$–$1.18$) covers at $0.68$–$0.88$ only because its intervals are very wide (up to $0.75$): they cover without informing.
  - **A recall error not explained by the prior remains on binary datasets.** On `mammographic` all five classifiers have a positive shift ($+0.034$ to $+0.071$) that is all recall term, with $P\approx0$ and $R_0\approx-0.005$ of the opposite sign ($R-R_0=+0.04$ to $+0.08$); coverage is $0.10$–$0.65$ except kNN ($0.89$). The same profile, on single classifiers, appears on `tictactoe` kNN ($R-R_0=+0.04$, coverage $0.66$), `semeion` LR and MLP ($+0.04$, covered by wide intervals) and `spambase` RF ($-0.01$ on an interval of width $0.029$, coverage $0.53$). The decomposition leaves two explanations, which E2.2 does not separate: the step in which the target bag moves the posterior of $C$, and a difference between the recalls of $h$ on $V$ and on the test pool from which the bags are drawn (cause 4 [new hypothesis]: the bags are drawn with replacement from a pool of $249$ instances on `mammographic`, so a sampling difference between $V$ and the pool is shared by all bags and all classifiers, and BayCAP's population interval does not account for the pool's own sampling error).
  - **Width falls short on a few pairs.** On $10$ pairs the observed coverage is more than $0.05$ below $\mathcal C_s$ (`spambase` SVM and RF, `connect-4` MLP, `nursery` SVM and MLP, `poker_hand` SVM, MLP and RF, `chess` LR and SVM; $9$ on the old results, with `chess` MLP in place of `spambase` RF), so the interval is also too narrow for the spread across bags. Candidate explanations, not tested: the bag-level fluctuation of the realised table (§T.2, the subject of E2.1 and E2.3), skewed posteriors near an accuracy of $1$ (`nursery` SVM and MLP, accuracy on $V$ $0.99$ and $0.999$), and NUTS sampling quality ($1000$ draws after $500$ warm-up iterations; multimodality when $\hat C$ is singular). Where $P$ changes sign across bags (`german`, `semeion`, `wine-quality` LR) the observed coverage is above $\mathcal C_s$, as expected, since the shift is not the same in every bag.

  What this shows: the under-coverage of BayCAP on `main` has two main sources. With many classes and few validation instances per class, the $\operatorname{Dir}(\mathbf 1)$ prior on $C$ pulls the recalls towards $1/K$ and moves the narrow interval away from the truth; this accounts for every pair with coverage near $0$. With a weak, badly conditioned classifier, a systematic error in the estimated prevalences displaces the interval when it is not wide enough. On binary datasets a smaller recall error remains that neither explains. The evidence is correlational (the prior was not changed). The realised-table predictive cannot repair any of the three, which is consistent with E2.1.

  The analysis of the coverage error stops here (decision of 2026-10-21); what it leaves open is recorded as a guess, not tested:
  - **the recall error on binary datasets** most likely comes from cause 4: the recalls of $h$ on $V$ and on the small test pool differ by chance, and the difference is repeated in every bag and for every classifier of the dataset, while the population interval does not account for the pool's own sampling error; the alternative is the step in which the target bag moves the posterior of $C$;
  - **the systematic prevalence error on a singular $\hat C$** (`wine-quality` LR and SVM) is guessed to come from the prior on $C$ rather than the prior on $\pi^Q$: when $h$ never predicts some class, the counts of the bag do not say how to attribute the predictions between that class and the classes it is confused with, and the imaginary counts of the $\operatorname{Dir}(\mathbf 1)$ prior on $C$ decide the attribution, in the same direction in every bag;
  - **the width that falls short on a few pairs** is guessed to be mainly the bag-level fluctuation of the realised table, which the population interval leaves out and the realised-table predictive adds; skewed posteriors near an accuracy of $1$ and NUTS sampling quality may contribute.

## E2.3: the exact realised-table predictive in `main`

**Why.** E2.1 approximated the realised-table predictive for accuracy (one $\bar\rho$ per bag, a normal in place of the sum of binomials, §T.4) and found that it would repair the coverage of only $5$ of the $53$ under-covering pairs ($3$ on the old `BayesCAP` results); the exact predictive tells whether the approximation is right. E2.2 found that most of the under-coverage of the population interval is displacement, and that on $10$ pairs the interval is also too narrow, possibly because it leaves out the bag-level fluctuation that the predictive adds. The realised pushforward changes the width and can also move the centre of the interval (§T.3), so repeating the E2.2 analysis on it tells whether it changes any of the causes of the displacement, or only the width. Running it in `main`, as a method like the others, stores its results next to those of every other method, with the same bags and the same NUTS draws as the stored `BayesCAP`.

- **Design.**
  - **Inputs:** the `main` results of `BayesCAP` (population pushforward, stored) and of `BayesCAP-real` (realised pushforward, added to `main`: `methods/baycap.py:BayCAP` with `estimand="realised"`, the priors and NUTS settings of `BayesCAP`, $500$ warm-up iterations and $1000$ draws, and the default seed, so that its posterior draws are those of `BayesCAP` on the same platform and JAX version; the stored `BayesCAP` results were computed on another platform, probably the GPU, and are recomputed on CPU by the user's decision of 2026-10-22, so that the two pushforwards share their draws); the $100$ `main` bags of size $1000$ (UPP) of all $75$ pairs; the validation counts $f_{ij}$ as in E2.2; the per-pair CSV of E2.1 for the approximate coverage.
  - **Varied:** the pushforward, population (`BayesCAP`) or realised (`BayesCAP-real`); data, bags, priors, sampler and seeds are held fixed.
  - **Measures:** vanilla accuracy only. For both pushforwards: coverage, amplitude and Winkler score, and the predicted coverage $\mathcal C_s$; the error measures of E2.2 from each pushforward's mean table (shift, recall term, prior part, prevalence term and its mean absolute value, $\lVert\hat\pi-\pi\rVert_1$); for the realised pushforward, also the coverage that E2.1 predicted with the approximation. For the Winkler score, a paired test of the difference between the two pushforwards on each pair: the two are scored on the same $100$ bags with the same posterior draws, so for each bag $b$ the difference $d_b=\mathcal W_b^{\text{real}}-\mathcal W_b^{\text{pop}}$ is computed, and the test asks whether their mean $\bar d$, which is the gap between the two Winkler columns, is further from $0$ than the variation over the bags allows. It is a paired bootstrap: the $100$ differences are resampled with replacement $10{,}000$ times (seeded), and the mean of each resample is recomputed; the $2.5\%$ and $97.5\%$ quantiles of these means give a $95\%$ interval for $\bar d$, and the two-sided p-value is the share of resampled means, shifted to be centred on $0$, that are at least as far from $0$ as $\bar d$. Two-sided means it detects both a decrease and an increase. The test is on the mean because the Winkler score is a mean over bags, in which a few large misses rightly weigh more than many small width costs; a rank test (Wilcoxon signed-rank) was tried first and set aside, because it tests the typical per-bag difference, which on $26$ pairs has the opposite sign of the mean (many covered bags pay the extra width, a few missed bags gain a lot). The median of the $d_b$ is stored as a description of the typical bag. The p-values are adjusted with Holm's method over the $75$ pairs, so that testing many pairs does not produce significant results by chance. All bags of a pair share the same $V$ and test pool, so a significant difference holds for these bags, not necessarily for other validation sets. The diagnostics of E2.2 ($\kappa_0(\hat C)$, $1/\sigma_{\min}$, accuracy on $V$) do not depend on the pushforward and are carried over.
  - **Estimand:** population table for `BayesCAP`, realised table for `BayesCAP-real`; both scored against the realised table of each bag. The error measures use the bag's true class proportions and recalls, so they are an analysis, not a method.
  - **Cost:** one rerun of `main` on `bcuda`, which computes only `BayesCAP-real` because the results of every other method exist: one NUTS run per bag, $75\times100=7500$ runs. `main` stores the samples of all three metrics (vanilla accuracy, macro-F1, macro-K) for `BayesCAP-real`, as for every method.
- **Representation.**
  - **Comparison table:** one row per pair, grouped by dataset and ordered by $K$: coverage, amplitude and Winkler score of both pushforwards, the $95\%$ bootstrap interval of the mean Winkler difference and its Holm-adjusted p-value, and the approximate coverage of E2.1 next to the exact one.
  - **E2.2 table of the realised pushforward:** the table of E2.2, same layout (diagnostics, interval, error), computed on `BayesCAP-real`.
  - **Scatter, coverage:** coverage of the population interval on the horizontal axis, of the predictive on the vertical axis, one point per pair, coloured by $\hat\sigma_g/\hat\sigma_p$ from E2.1, with lines at $0.95$ on both axes: points that move up to the line are pairs the predictive repairs, points near $(0,0)$ stay displaced.
  - **Scatter, approximation:** exact against approximate coverage of the predictive, one point per pair, with the line $y=x$.
  - **The E2.2 scatters of the realised pushforward**, next to those of the population one.
- **Analysis.** Run on `bcuda`: `BayesCAP-real` stored for all $75$ pairs by the `main` run that ended on 2026-10-22, `BayesCAP` recomputed on CPU by the user afterwards (§I.2), then the E2.1, E2.2 and E2.3 tables and the two E2.3 scatters generated on 2026-10-22, and the comparison table generated again with the Winkler test on 2026-10-23 (recorded in the log of work-plan card WP-4). The E2.2 scatters of both pushforwards were drawn on 2026-10-23 (`plots/displacement.py`, commit `975f2ac`). Results in `<OUT_DIR>/uncawe/tables/pushforward/pushforward.csv`, `<OUT_DIR>/uncawe/tables/displacement/BayesCAP-real/displacement.csv`, `<OUT_DIR>/uncawe/plots/main/pushforward/` and `<OUT_DIR>/uncawe/plots/main/displacement/BayesCAP-real/`. Notation as in E2.2; "population" is `BayesCAP`, "realised" is `BayesCAP-real`.
  - **The realised interval is wider, by exactly the bag-level spread.** On every pair the realised interval is wider than the population one, by a factor from $1.00$ to $2.12$ (median $1.11$). If the bag-level fluctuation is independent of the uncertainty on $(\pi^Q,C)$, the two spreads add as variances, $\sigma_{\text{real}}^2\approx\sigma_p^2+\sigma_g^2$, so the width should grow by the factor $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$, with $\hat\sigma_g/\hat\sigma_p$ from E2.1 (§T.4). The observed factor is within $\pm5\%$ of this value on every pair (median ratio $1.005$, correlation $0.998$): the realised pushforward adds the bag-level spread and nothing else to the width. The largest factors are on `letter` ($2.08$–$2.12$ except LR, $1.79$), `dry-bean` ($1.78$–$1.80$) and `hand_digits` ($1.75$–$1.78$), where many validation instances make $\hat\sigma_p$ small; the smallest on `connect-4` LR, `shuttle` LR and `poker_hand` LR ($\le1.01$), whose population intervals are already wide.
  - **The centre does not move.** The shift of the realised point estimate equals that of the population one within $0.009$ on every pair (correlation $0.9998$; the largest differences on `isolet`, where the shift is about $-0.17$). This confirms the hypothesis of §T.3: BayCAP chooses $\pi^Q$ so that the predicted shares $q$ match the observed ones, so replacing $q_i$ with $n^t_i/N^Q$ barely changes the mean. The diagnostics and the prior part $R_0$ are the same for the two pushforwards, since they depend only on $V$ and on the bags, and the split of the shift into $R$ and $P$ is the same within $0.007$ on every dataset except `poker_hand`. There the recall term of the realised table is about $0$ for every classifier (LR: $0.017$ for the population table, $0.000$ for the realised one) and the prevalence term carries the whole shift ($0.220\to0.238$). A possible reason, not checked: the column sums of a realised table are $\sum_in^t_i\rho_{i\cdot}$, the prevalences that the observed predictions imply, not the posterior $\pi$, and with $3$ validation instances in the smallest class the two can differ; the shift is the same, so this moves error between the two terms without changing it. The E2.2 scatters of the two pushforwards show the same picture: in the cause-1 scatter the points stay on or near $y=x$, except the `poker_hand` points, which drop from $R\approx0.01$–$0.02$ to $R\approx0$, and the `isolet` points, which move slightly towards $0$; the conditioning scatter keeps every point in place (its axes do not depend on the pushforward), and only the colours of the pairs whose coverage rises change.
  - **Coverage rises where the interval was only too narrow.** Mean coverage goes from $0.60$ to $0.68$; the pairs with coverage at least $0.95$ from $12$ to $28$; the pairs below $0.5$ from $26$ to $18$. No pair loses coverage (the largest drop is $0.01$, within the noise of $100$ bags), and $16$ pairs go from below $0.95$ to at least $0.95$: the binary pairs with a small shift (`spambase` LR, kNN, SVM and MLP, `semeion` LR, `tictactoe` LR and MLP, `mammographic` kNN, `german` MLP), all five `nursery` classifiers, `shuttle` kNN and `connect-4` SVM. Among the $53$ pairs below $0.90$, $12$ reach $0.90$ and $4$ fall within $[0.93,0.97]$ with a lower Winkler score (E2.1's criterion; $5$ for the approximation).
  - **Displaced intervals improve only in part, or not at all.** Coverage rises most where the interval was displaced by about one half-width and the bag-level spread is large ($\hat\sigma_g/\hat\sigma_p\approx1.5$): `dry-bean` from $0.21$–$0.33$ to $0.57$–$0.74$, `chess` RF from $0.25$ to $0.69$ and MLP from $0.51$ to $0.74$, `hand_digits` LR from $0.10$ to $0.42$. The shift is unchanged, but the half-width grows, so $\lvert s\rvert$ roughly halves (`dry-bean`: $1.22$–$1.59\to0.66$–$0.87$). Where the shift is several half-widths, coverage stays near $0$: `letter` and `isolet` ($0.00$–$0.01$; $\lvert s\rvert$ still $2.1$–$5.5$) and `hand_digits` kNN, SVM and MLP ($0.00$–$0.02$). Where the prevalence error has a fixed sign and $\hat\sigma_g/\hat\sigma_p<0.3$, coverage does not move: `german` LR, SVM and RF ($0.54\to0.56$, $0.18\to0.19$, $0.68\to0.68$), `wine-quality` LR, SVM and MLP ($0.35\to0.36$, $0.68$, $0.61$), `poker_hand` ($0.68$–$0.88\to0.70$–$0.88$). The binary recall error of `mammographic` stays as well: LR, SVM and MLP go from $0.48$, $0.48$, $0.10$ to $0.54$, $0.60$, $0.17$, with the same shift.
  - **The Winkler score improves significantly exactly where the interval was displaced.** The paired bootstrap of the mean difference (Holm-adjusted, $0.05$) finds a significantly lower Winkler score on $31$ pairs, a significantly higher one on $9$, and no significant difference on $35$ ($22$ with a lower mean, $13$ with a higher one). The $31$ decreases are all of `dry-bean`, `hand_digits`, `letter` and `isolet`, `chess` RF and MLP, `mammographic` LR, SVM, MLP and RF, `spambase` RF, `tictactoe` kNN, `shuttle` RF and `german` LR and SVM; on `letter` and `isolet` the score falls by $0.58$–$0.78$ ($12$–$31\%$) although coverage stays at $0$, because the intervals miss by less. On $4$ of them (`spambase` RF, `chess` MLP, `mammographic` RF, `tictactoe` kNN) the median per-bag difference is positive: most bags pay the extra width, and a few missed bags gain much more. The $9$ increases are pairs that already covered (population coverage $0.92$–$1.00$: `semeion` SVM and RF, `shuttle` SVM and MLP, `wine-quality` kNN, `connect-4` kNN, `tictactoe` MLP and RF, `spambase` kNN), and are small, $+0.004$ to $+0.011$. The $35$ pairs without a significant difference are of three kinds: pairs whose coverage the realised interval repairs, where the penalties saved on the few newly covered bags about balance the width added on every bag (all of `nursery`, `chess` LR, kNN and SVM, `connect-4` MLP and SVM, `spambase` SVM, `mammographic` kNN, `tictactoe` LR), pairs whose interval barely widens (`poker_hand`, `wine-quality` LR, SVM, MLP and RF, `german` kNN and RF, `shuttle` LR and kNN), and pairs that already covered, where the cost of the extra width is not significant (`spambase` LR and MLP, `semeion` LR, kNN and MLP, `connect-4` LR and RF, `german` MLP, `tictactoe` SVM). With $10{,}000$ resamples the smallest raw p-value is $10^{-4}$, which Holm turns into $0.0075$ for $75$ pairs; this does not change any conclusion at $0.05$.
  - **The E2.1 approximation is accurate and slightly conservative.** The exact and approximate coverages have correlation $0.998$; the exact one is higher by $0.008$ on average, and the mean absolute difference is $0.013$. Only $3$ pairs differ by more than $0.05$, all with the exact coverage higher: `chess` RF ($0.69$ against $0.57$), `dry-bean` LR ($0.73$ against $0.66$), `mammographic` RF ($0.78$ against $0.72$). One $\bar\rho$ per bag and a normal in place of the sum of binomials are enough to predict the effect of the realised pushforward on accuracy.
  - **The width shortfall of E2.2 is mostly the bag-level spread.** Coverage still follows $\mathcal C_s$ (correlation $0.978$, against $0.979$ for the population interval). Of the $10$ pairs where the population interval is more than $0.05$ below $\mathcal C_s$, $7$ are not below it with the realised interval (`spambase` SVM and RF, `connect-4` MLP, `nursery` SVM and MLP, `chess` LR and SVM). This supports the guess of E2.2 that the missing width is mainly the bag-level fluctuation. The $3$ `poker_hand` pairs (SVM, MLP, RF) remain below, which fits the other two candidates of E2.2 (few validation instances per class, NUTS sampling quality), not tested here. The $5$ `hand_digits` pairs fall newly below $\mathcal C_s$, by small amounts (e.g. kNN, $0.02$ against $0.09$): there $s$ is $1.0$–$1.8$, where $\mathcal C_s$ depends on the tails of the assumed normal error, and a guess, not checked, is that the real errors have lighter tails than the normal.

  What this shows: the realised-table predictive is the right pushforward for the quantity that `main` scores, and it costs nothing in coverage. It adds to the interval exactly the bag-level spread, which repairs the pairs whose population interval was centred correctly but too narrow, and lowers the Winkler score significantly wherever the interval was displaced; where it raises the score, the increase is small and confined to pairs that already covered. It does not move the centre, so it cannot repair displacement: the causes found by E2.2 (the prior on $C$ for large $K$, a fixed-sign prevalence error with a badly conditioned $\hat C$, the residual recall error on binary datasets) are the same under both pushforwards, and `letter`, `isolet`, `hand_digits`, `german`, `wine-quality` and `mammographic` remain under-covered. The approximation of E2.1 predicts all of this within about $0.01$ of coverage.

## E2.4: BayCAP with an empirical-Bayes prior on the recalls

**Why.** E2.2 found that the $\operatorname{Dir}(\mathbf 1)$ prior on the columns of $C$ explains the largest displacements: on `dry-bean`, `hand_digits`, `letter` and `isolet` the shift is almost entirely the prior part $R_0$ of the recall term, because the prior pulls every recall towards $1/K$ (§T.5). E2.3 showed that the realised pushforward widens the interval but does not move its centre, so these pairs stay displaced: the fix has to act on the prior. An empirical-Bayes prior (§T.9) gives the recalls of a classifier a common $\operatorname{Beta}$ prior whose centre $\hat m$ (the typical recall) and strength $\hat\nu$ (how alike the recalls are) are estimated from the validation counts; each recall is then pulled towards the typical recall of the same classifier, up for some classes and down for others, so the pulls cancel on accuracy on average instead of adding up. The prior uses only the labels of $V$, like the rest of BayCAP, and changes only the concentration parameters of the Dirichlet prior on the columns of $C$, so BayesianCC is otherwise unchanged. The experiment first predicts the effect of the new prior in closed form, from the step in which $V$ updates the recalls (the check), then measures it with NUTS (the variant), and the check explains the variant.

- **Design.**
  - **Inputs:** the $15$ datasets, $5$ classifiers and $100$ UPP bags of size $1000$ of `main` ($75$ pairs); the validation counts $f_{ij}$, rebuilt as in E2.2; the results of four methods of `main`, all built from `methods/baycap.py:BayCAP` with the prior on $\pi^Q$, the NUTS settings ($500$ warm-up iterations, $1000$ draws) and the seed of `BayesCAP`: `BayesCAP` (population pushforward, $\operatorname{Dir}(\mathbf 1)$ prior on $C$), `BayesCAP-real` (realised pushforward, $\operatorname{Dir}(\mathbf 1)$), `BayesCAP-EB` (population, empirical Bayes) and `BayesCAP-EB-real` (realised, empirical Bayes). `BayesCAP` is built from `BayCAP` instead of `cap-ml`'s `ACC_BayesCAP` (user's decision of 2026-10-26), so that it can store the NUTS diagnostics; `BayesCAP` and `BayesCAP-real` are run again to store them, and their draws, and so all their stored measures, must be unchanged.
  - **Varied:** the prior on the columns of $C$, $\operatorname{Dir}(\mathbf 1)$ or empirical Bayes, and the pushforward, population or realised, each method with its own run; data, bags, prior on $\pi^Q$, sampler settings and seed are held fixed, so with the same prior the two pushforwards share their NUTS draws (as found for E2.3). The empirical-Bayes prior gives column $j$ of $C$ the Dirichlet parameters $\alpha_{jj}=\hat\nu\hat m$ and $\alpha_{ij}=\hat\nu(1-\hat m)/(K-1)$ for $i\ne j$, where $(\hat m,\hat\nu)$ maximise the beta-binomial likelihood of the diagonal counts of $V$, within $\nu\le10^4$ and $m\in[10^{-3},1-10^{-3}]$ (§T.9).
  - **Measures:** vanilla accuracy only, per pair.
    - *Check (no runs).* Three estimates of each recall from $V$ alone: the plain recall $f_{jj}/n^s_j$; the posterior mean under $\operatorname{Dir}(\mathbf 1)$, $\frac{f_{jj}+1}{n^s_j+K}$; the posterior mean under empirical Bayes, $\frac{f_{jj}+\hat\nu\hat m}{n^s_j+\hat\nu}$. For each estimate $\tilde r_j$ and each bag $b$, with true class proportions $\pi^{(b)}$ and realised recalls $r^{(b)}_j$, the recall error $\Delta^{(b)}=\sum_j\pi^{(b)}_j(\tilde r_j-r^{(b)}_j)$; per pair, its mean over the bags (the displacement it predicts) and the mean of $\lvert\Delta^{(b)}\rvert$ (the typical size of the error). The $\operatorname{Dir}(\mathbf 1)$ version validates the check against the recall term $R$ of `BayesCAP`, since the check leaves out the step in which the target bag updates $C$. Also $\hat m$, $\hat\nu$, whether the fit reached a bound, and the standard deviation that empirical Bayes leaves out of the interval, $\sum_j\pi_j(1-w_j)\,\operatorname{sd}(\hat m)$ with $w_j=n^s_j/(n^s_j+\hat\nu)$ and $\pi$ uniform, $\operatorname{sd}(\hat m)$ from the curvature of the log-likelihood at its maximum (§T.9).
    - *Variant.* For the four methods: coverage, amplitude, Winkler score and predicted coverage $\mathcal C_s$; the error measures of E2.2 (shift, recall term $R$, prior part $R_0$, prevalence term $P$ and its mean absolute value, $\lVert\hat\pi-\pi\rVert_1$). Within each pushforward, the paired bootstrap test of the mean Winkler difference of E2.3 between the empirical-Bayes method and its $\operatorname{Dir}(\mathbf 1)$ counterpart, Holm-adjusted over the $75$ pairs. The change of the recall term, $R$ of `BayesCAP-EB` minus $R$ of `BayesCAP`, against the change the check predicts, the difference between the mean errors of the empirical-Bayes and $\operatorname{Dir}(\mathbf 1)$ estimates.
    - *NUTS diagnostics.* For the four methods, per bag: the number of divergent transitions, the largest split-$\hat R$ (one chain split in two halves) and the smallest effective sample size over the parameters of $\pi^Q$ and $C$; per pair, the share of bags with divergences, the largest split-$\hat R$ and the smallest effective sample size.
  - **Estimand:** population table for `BayesCAP` and `BayesCAP-EB`, realised table for `BayesCAP-real` and `BayesCAP-EB-real`; all four scored against the realised table of each bag (`true_cts`). $(\hat m,\hat\nu)$ use only the labels of $V$, so the empirical-Bayes variants are methods. The check and the error measures use the bags' true class proportions and recalls, so they are an analysis, not a method.
  - **Cost:** four runs of `main` on `bcuda` with `JAX=cpu`, one per method, each $75\times100=7500$ NUTS runs: `BayesCAP-EB` and `BayesCAP-EB-real`, then `BayesCAP` and `BayesCAP-real` after their stored results have been checked against a recomputation and deleted. The recomputation of `BayesCAP` on CPU took about $46$ minutes (work plan §I.2), so each run is expected to take about as long. The check costs nothing.
- **Representation.**
  - **Comparison table:** one row per pair, grouped by dataset and ordered by $K$: coverage, amplitude and Winkler score of the four methods; for each pushforward, the $95\%$ bootstrap interval of the mean Winkler difference (empirical Bayes minus $\operatorname{Dir}(\mathbf 1)$) and its Holm-adjusted p-value.
  - **Prior table:** one row per pair, same order: $K$, minimum and median $n^s_j$, $\hat m$, $\hat\nu$ (marked when at a bound), the left-out standard deviation; the mean and mean absolute recall error of the three estimates of the check; $R$, $R_0$ and $P$ of `BayesCAP` and `BayesCAP-EB`; the NUTS diagnostics of the four methods.
  - **Scatter, check against variant:** the change of the recall term predicted by the check on the horizontal axis, the observed one on the vertical axis, one point per pair, coloured by $K$, with the line $y=x$; a second panel with the $\operatorname{Dir}(\mathbf 1)$ check against $R$ of `BayesCAP`, which validates the check.
  - **Scatter, coverage:** coverage with the $\operatorname{Dir}(\mathbf 1)$ prior on the horizontal axis, with the empirical-Bayes prior on the vertical axis, one point per pair, one panel per pushforward, with lines at $0.95$ on both axes.
  - **The E2.2 table and scatters** of `BayesCAP-EB` and `BayesCAP-EB-real`, next to those of `BayesCAP` and `BayesCAP-real`.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up while working on E2.1 and E2.2. They are not scheduled and are numbered E2.c<n> (c for candidate); one becomes an experiment of the main section only when the user decides so, and then it gets the next number E2.<n> and a full specification.

- **E2.c1, diagnosis against the size of $V$.** *Why:* the toy predicts that BayCAP's coverage of the realised accuracy decreases as the number of validation instances per class grows (§T.3). *What it would show:* coverage and amplitude of BayCAP against the median $n^s_j$, one point per pair, with the displaced pairs marked. *Needs:* the E2.1 table only, no runs.
- **E2.c2, cause of the miscalibration in width.** *Why:* two effects act in opposite directions on the width of the interval, the missing bag-level fluctuation (too narrow) and the fixed class counts of APP bags (too wide, §T.6). *What it would show:* a $2\times2$ table, pushforward (population / predictive) × bag protocol (APP / i.i.d. class counts), with coverage and amplitude in each cell. *Needs:* BayCAP reruns on the `main` bags, storing the table draws and the predicted counts, plus i.i.d. bags with the same prevalences.
- **E2.c3** was absorbed into E2.2.
- **E2.c4** became E2.3.

# Theory

This part explains what the experiments need and nothing more. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that an instance of true class $\omega_j$ is predicted as $\omega_i$, and $\pi^Q$ is the vector of class prevalences of the target bag $U$, with $N^Q=\lvert U\rvert$ instances and $K$ classes. The population table is $T=C\operatorname{diag}(\pi^Q)$: entry $(i,j)$ is the probability that a random target instance has true class $\omega_j$ and is predicted as $\omega_i$. Accuracy is the sum of its diagonal, $\sum_j C_{jj}\pi^Q_j$.

## T.1 BayCAP in brief

*Intuition:* BayCAP lists many plausible values of $(\pi^Q,C)$, computes the accuracy each one implies, and reports the range of the middle $95\%$.

BayCAP uses two kinds of counts. From the labelled validation set $V$ it uses $f_{ij}$, the number of validation instances of true class $\omega_j$ predicted as $\omega_i$, with $n^s_j=\sum_i f_{ij}$ instances of class $\omega_j$. From the unlabelled target bag it uses $n^t_i$, the number of target instances predicted as $\omega_i$.

**The multinomial distribution.** Roll a die with $K$ faces $n$ times, independently, with probability $p_i$ of face $i$, and count how often each face comes out, $x_1,\dots,x_K$. By independence, one particular sequence of rolls with these counts has probability $\prod_ip_i^{x_i}$, and there are $n!/\prod_ix_i!$ sequences with the same counts (the ways of choosing which rolls give which face), so

$$
\operatorname{Mult}(x\mid n,p)=\frac{n!}{\prod_ix_i!}\prod_ip_i^{x_i}.
$$

**The two likelihoods.** Within class $\omega_j$ of $V$, each instance is predicted as $\omega_i$ with probability $C_{ij}$, independently of the others: the predictions are rolls of a die with the probabilities of column $j$ of $C$. In $U$, a random instance is predicted as $\omega_i$ with probability $\sum_jC_{ij}\pi^Q_j$, by the law of total probability (the probability of an outcome is the sum, over the ways it can happen, of the probability of each way: here, true class $\omega_j$ with probability $\pi^Q_j$, then prediction $\omega_i$ with probability $C_{ij}$); in vector form, the predicted shares are $C\pi^Q$. The two likelihood terms say how probable the observed counts are for given $(\pi^Q,C)$:

$$
\mathcal L_V(C)=\prod_{j=1}^K \operatorname{Mult}\!\left(f_{\cdot j}\mid n^s_j,\,C_{:j}\right),
\qquad
\mathcal L_U(\pi^Q,C)=\operatorname{Mult}\!\left(n^t\mid N^Q,\,C\pi^Q\right).
$$

**The priors.** A Dirichlet distribution $\operatorname{Dir}(\alpha)$ is a distribution over probability vectors $p$ (non-negative, summing to one), with density proportional to $\prod_ip_i^{\alpha_i-1}$. With all $\alpha_i=1$ the density is constant: every probability vector is equally likely. BayCAP puts a Dirichlet prior on $\pi^Q$ (uniform by default, `prior`) and the uniform $\operatorname{Dir}(\mathbf 1)$ on each column of $C$ (fixed in the code).

**The posterior.** Bayes' rule combines prior and data: the posterior is proportional to prior times likelihood,

$$
p(\pi^Q,C\mid V,U)\propto\operatorname{Dir}(\pi^Q;\alpha)\prod_j\operatorname{Dir}(C_{:j};\mathbf 1)\;\mathcal L_V(C)\,\mathcal L_U(\pi^Q,C).
$$

It has no closed form, so a Markov chain Monte Carlo sampler (NUTS) draws $R$ pairs $(\pi^{(r)},C^{(r)})$ whose histogram approximates it; by the law of large numbers, averages over many draws approximate averages over the posterior. Each draw gives a table $T^{(r)}=C^{(r)}\operatorname{diag}(\pi^{(r)})$ and an accuracy $A(T^{(r)})$. The point estimate is the mean of these accuracies, and the interval runs from their $2.5\%$ to their $97.5\%$ percentile.

*Implementation note:* in `cap-ml` and `quapy`, `p_c_cond_y` is indexed `[y, c]`, so it stores $C^\top$, and the tables returned by `ACC_BayesCAP` have true classes on rows.

## T.2 Population accuracy and realised accuracy

*Intuition:* even if we knew $\pi^Q$ and $C$ exactly, we would not know the accuracy on *this* bag, because the classifier is right on a random number of its instances.

- The **population accuracy** $\sum_j C_{jj}\pi^Q_j$ is the accuracy the classifier has on average over all bags with prevalence $\pi^Q$. BayCAP's posterior is about $(\pi^Q,C)$, so its interval is an interval for this quantity.
- The **realised accuracy** is the fraction of instances of this bag that the classifier gets right. It is computed from the bag's realised table (its true labels), and `main` scores against it (`true_cts`).

**One instance.** Let $B$ be $1$ if an instance of class $\omega_j$ is classified correctly and $0$ otherwise. It is $1$ with probability $p=C_{jj}$. Its mean is $\mathbb E[B]=1\cdot p+0\cdot(1-p)=p$; since $B^2=B$, also $\mathbb E[B^2]=p$; the variance is the mean of the square minus the square of the mean, $\operatorname{Var}(B)=p-p^2=p(1-p)$.

**One class.** The bag contains $n_j$ instances of class $\omega_j$, each classified correctly independently of the others. The number of correct predictions $c_j$ is the sum of their $n_j$ indicators, a binomial count. The mean of a sum is the sum of the means, and the variance of a sum of independent variables is the sum of their variances, so $\mathbb E[c_j]=n_jC_{jj}$ and $\operatorname{Var}(c_j)=n_jC_{jj}(1-C_{jj})$.

**The bag.** The realised accuracy is $A_{\text{real}}=\frac1{N^Q}\sum_jc_j$. The counts of different classes are independent, so the same two rules apply again, together with $\operatorname{Var}(aX)=a^2\operatorname{Var}(X)$ for a constant $a$ (here $a=1/N^Q$). With $n_j=N^Q\pi^Q_j$,

$$
\mathbb E[A_{\text{real}}]=\frac1{N^Q}\sum_jn_jC_{jj}=\sum_jC_{jj}\pi^Q_j,
\qquad
\operatorname{Var}(A_{\text{real}})=\frac{1}{(N^Q)^2}\sum_j n_j\,C_{jj}(1-C_{jj}):
$$

the mean is the population accuracy, and the variance is the fluctuation around it.

*Example.* Binary, $n_0=n_1=500$, $C_{00}=0.9$, $C_{11}=0.7$. The population accuracy is $0.8$. The standard deviation of the realised accuracy is $\sqrt{500\cdot0.9\cdot0.1+500\cdot0.7\cdot0.3}/1000\approx0.012$, so on a typical bag the realised accuracy lies anywhere in $[0.776,0.824]$.

BayCAP's interval contains no term for this fluctuation. When $V$ is small, the uncertainty about $C$ is large and hides it. When $V$ is large, the interval becomes narrower than the fluctuation, and it misses the realised accuracy more often than $5\%$ of the time. This is the *estimand gap*.

## T.3 The realised-table predictive

*Intuition:* we cannot see the true labels of the bag, but we can see its predictions; for each prediction we guess a plausible true class, and the guessed labels give a plausible realised table.

**The probability of the true class given the prediction.** The definition of conditional probability is $P(A\mid B)=P(A\cap B)/P(B)$. Take a target instance predicted as $\omega_i$. If $(\pi^Q,C)$ were known, the probability of "true class $\omega_j$ and prediction $\omega_i$" would be entry $(i,j)$ of the population table, $C_{ij}\pi^Q_j$ (T.1), and the probability of "prediction $\omega_i$" would be the sum of row $i$, $q_i=\sum_lC_{il}\pi^Q_l$ (the law of total probability of T.1). So

$$
\rho_{ij}=P(Y=\omega_j\mid\hat Y=\omega_i)=\frac{C_{ij}\,\pi^Q_j}{\sum_l C_{il}\,\pi^Q_l}.
$$

Row $i$ of $\rho$ is row $i$ of the table rescaled to sum to one. On the diagonal, $\rho_{ii}$ is the precision of class $\omega_i$.

**Sampling a realised table.** For each posterior draw $r$, the predictive computes $\rho^{(r)}$ from $(\pi^{(r)},C^{(r)})$ and assigns true classes to the $n^t_i$ instances predicted as $\omega_i$ by one multinomial draw, $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, i.e. by $n^t_i$ independent picks with probabilities $\rho^{(r)}_{i\cdot}$. The resulting count table, divided by $N^Q$, is a sample $\tilde T^{(r)}$ of the realised table, and $A(\tilde T^{(r)})$ is a sample of the realised accuracy. The posterior is unchanged; only the last step changes, at the cost of one multinomial draw per sample. The samples now contain both the uncertainty about $(\pi^Q,C)$ and the bag-level fluctuation of T.2.

**Where the predictive puts the centre.** Of the $n^t_i$ instances predicted as $\omega_i$, each is correct with probability $\rho_{ii}$, so for one draw the realised accuracy samples have mean $\sum_i\frac{n^t_i}{N^Q}\rho_{ii}$. The population accuracy of the same draw can be written in the same form: multiplying the definitions, $q_i\rho_{ii}=\sum_lC_{il}\pi^Q_l\cdot\frac{C_{ii}\pi^Q_i}{\sum_lC_{il}\pi^Q_l}=C_{ii}\pi^Q_i$, so $\sum_jC_{jj}\pi^Q_j=\sum_iq_i\rho_{ii}$, where $q_i$ is the share of the bag that the draw expects to be predicted as $\omega_i$. The two means differ by $\sum_i\big(\frac{n^t_i}{N^Q}-q_i\big)\rho_{ii}$, i.e. only where the observed shares $n^t_i/N^Q$ differ from $q_i$. BayCAP chooses $\pi^Q$ so that $q$ matches the observed shares as well as the model allows, so the centre is expected to move little and the predictive mostly changes the width [new hypothesis, confirmed by E2.3: the two centres differ by at most $0.009$]. In particular, the pull of the prior on $C$ towards $1/K$ acts on $C^{(r)}$, and so on $\rho^{(r)}$, and is expected to remain.

*Toy check* (`scripts/toys/ext2_estimand_gap.py`, toy result, binary, not BayCAP on real data): $C_{10}=0.1$, $C_{11}=0.7$, $\lvert U\rvert=1000$, $\pi^Q_1=0.5$ with fixed class counts, $n_s$ validation instances per class, uniform priors, $3000$ bags, Monte Carlo error about $\pm0.01$.

| $n_s$ per class | population interval vs population acc. | population interval vs realised acc. | realised-table predictive vs realised acc. | width (population / predictive) |
|---|---|---|---|---|
| $100$ | $0.94$ | $0.92$ | $0.94$ | $0.136$ / $0.147$ |
| $500$ | $0.95$ | $0.85$ | $0.94$ | $0.061$ / $0.081$ |
| $5000$ | $0.97$ | $0.63$ | $0.96$ | $0.027$ / $0.059$ |

With few validation instances the uncertainty about $C$ dominates and the population interval is nearly right also for the realised accuracy; with many, the estimand gap dominates and its coverage drops to $0.63$; the predictive stays near $0.95$ at every $n_s$, with wider intervals.

## T.4 The approximation of E2.1

*Intuition:* `main` did not store the posterior tables, so E2.1 cannot redo the predictive exactly; for accuracy, it adds to each stored accuracy sample a random error of the right size.

**The exact accuracy of one draw.** For accuracy only the diagonal matters. One cell of a multinomial is binomial (merge all the other faces of the die into one), so, of the $n^t_i$ instances predicted as $\omega_i$, the number assigned to the true class $\omega_i$ is $\operatorname{Bin}(n^t_i,\rho^{(r)}_{ii})$, with mean $n^t_i\rho^{(r)}_{ii}$ and variance $n^t_i\rho^{(r)}_{ii}(1-\rho^{(r)}_{ii})$ (T.2). The draws for different predicted classes are independent, so their variances add, and dividing by $N^Q$ divides the variance by $(N^Q)^2$:

$$
\tilde A^{(r)}=\frac{1}{N^Q}\sum_i \operatorname{Bin}\!\left(n^t_i,\rho^{(r)}_{ii}\right),
\qquad
\sigma_g^2=\frac{1}{(N^Q)^2}\sum_i n^t_i\,\rho^{(r)}_{ii}\left(1-\rho^{(r)}_{ii}\right),
$$

with mean close to the population accuracy of the draw (T.3) and variance $\sigma_g^2$.

**The two simplifications.** E2.1 uses one $\bar\rho$ per bag (the stored mean table `estim_cts`, each row rescaled to one) instead of one per draw. And it replaces the sum of binomials with a normal distribution: by the central limit theorem, a sum of many independent indicators is approximately normal, so the approximation is accurate when the counts are large. Each approximate predictive sample is then the stored accuracy sample plus independent noise $\mathcal N(0,\hat\sigma_g^2)$. Both $n^t$ and $\bar\rho$ use no target labels.

**How much wider.** Let $\hat\sigma_p$ be the standard deviation of the stored samples. Adding independent noise adds its variance, so the predictive samples have variance $\hat\sigma_p^2+\hat\sigma_g^2$. For normal samples the $95\%$ interval is the mean $\pm1.96$ standard deviations, so its width is proportional to the standard deviation, and the width is multiplied by

$$
\frac{\sqrt{\hat\sigma_p^2+\hat\sigma_g^2}}{\hat\sigma_p}=\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}.
$$

The ratio $\hat\sigma_g/\hat\sigma_p$ therefore measures how much the predictive can widen the interval: a ratio of $0.3$ widens it by about $4\%$ ($\sqrt{1.09}\approx1.044$). The approximation does not apply to macro-F1 or macro-K, whose off-diagonal cells matter; `main` stores the macro-F1 and macro-K samples of `BayesCAP-real` from E2.3, which would allow checking it.

## T.5 Displacement and the prior on $C$ (cause 1)

*Intuition:* if an interval is centred on the wrong value, making it wider helps only when it becomes wide enough to reach the true value, and then it is too wide to be useful.

**Where the error goes.** Accuracy is the sum of the diagonal of the table, $A=\sum_j\pi_jr_j$, where $r_j=C_{jj}$ is the recall of class $\omega_j$: accuracy is the recalls averaged with weights $\pi_j$. BayCAP's point estimate has the same form, $\hat A=\sum_j\hat\pi_j\hat r_j$, with $\hat\pi_j$ and $\hat r_j$ read from its mean table. Adding and subtracting $\sum_j\pi_j\hat r_j$,

$$
\hat A-A=\Big(\sum_j\pi_j\hat r_j-\sum_j\pi_jr_j\Big)+\Big(\sum_j\hat\pi_j\hat r_j-\sum_j\pi_j\hat r_j\Big),
$$

so the difference splits exactly into two parts:

$$
\hat A-A=\underbrace{\sum_j\pi_j(\hat r_j-r_j)}_{\text{recall term}}+\underbrace{\sum_j(\hat\pi_j-\pi_j)\,\hat r_j}_{\text{prevalence term}},
$$

where $\pi$ and $r$ are the class proportions and realised recalls of the bag. The recall term is the error from estimating how good the classifier is; the prevalence term is the error from estimating how many instances of each class the bag contains. The split applies to the point estimate as it is, because the mean is linear: the accuracy of each draw is the sum of the diagonal of its table, and the mean of a sum is the sum of the means, so the mean of BayCAP's accuracy samples is exactly the accuracy of its mean table.

**The prior as imaginary counts.** A column $C_{:j}$ lists the probabilities that an instance of class $\omega_j$ is predicted as each of the $K$ classes. It works like a die with $K$ faces: each validation instance of class $\omega_j$ is a roll, and $f_{ij}$ counts how often face $\omega_i$ came out. The prior $\operatorname{Dir}(\mathbf 1)$ says that every composition of the column is equally likely. Multiplying the prior density, proportional to $\prod_iC_{ij}^{1-1}$, by the likelihood of the column, proportional to $\prod_iC_{ij}^{f_{ij}}$ (T.1), gives a posterior proportional to $\prod_iC_{ij}^{f_{ij}+1-1}$, which is again a Dirichlet, $\operatorname{Dir}(f_{\cdot j}+\mathbf 1)$: the posterior parameters are the prior's plus the counts (*conjugacy*). The prior acts exactly like imaginary rolls added before the data, here one roll on each of the $K$ faces. The mean of a $\operatorname{Dir}(\beta)$ vector is $\beta_i/\sum_l\beta_l$ (T.9 derives it), a face's count divided by the total count, so the recall becomes

$$
\tilde r_j=\frac{f_{jj}+1}{n^s_j+K}
$$

instead of the plain $f_{jj}/n^s_j$; with $K=2$ this is Laplace's rule for a coin, $(h+1)/(n+2)$. The difference $\tilde r_j-f_{jj}/n^s_j$ is the prior part of the recall error, the quantity E2.2 measures. Averaged over possible validation sets, where $f_{jj}$ is on average $n^s_jr_j$ (T.2), it equals

$$
\mathbb E[\tilde r_j]-r_j=\frac{n^s_jr_j+1}{n^s_j+K}-r_j=\frac{n^s_jr_j+1-(n^s_j+K)r_j}{n^s_j+K}=\frac{1-Kr_j}{n^s_j+K},
$$

which is negative whenever the classifier beats guessing ($r_j>1/K$): every recall is pulled towards $1/K$. For a good classifier the pull is about $-K/n^s_j$: negligible for binary problems, large when $K$ is comparable to the number of validation instances per class.

**Why the data do not undo it.** Bayes' rule combines prior and data once, $\text{posterior}\propto\text{prior}\times\text{likelihood}$; the sampler converges to this combination, not to the true $C$. The prior fades only when the real counts $n^s_j$ greatly outnumber the $K$ imaginary ones. Moreover, "every column equally likely" is not neutral about the recall. One entry of a $\operatorname{Dir}(\alpha)$ vector follows a $\operatorname{Beta}(\alpha_j,\sum_{i\ne j}\alpha_i)$ distribution (T.9), so under $\operatorname{Dir}(\mathbf 1)$ the recall is $\operatorname{Beta}(1,K-1)$, with density $(K-1)(1-r)^{K-2}$ and mean $1/(1+(K-1))=1/K$. Integrating the density from $x$ to $1$ gives $P(r_j>x)=(1-x)^{K-1}$, so $P(r_j>0.9)=0.1^{K-1}$: with $26$ classes the prior considers a recall above $0.9$ practically impossible. The target bag cannot correct this: its predicted counts depend on $C$ and $\pi^Q$ only through the predicted shares $C\pi^Q$, $K$ numbers summing to one, far too few to pin down the $K^2$ entries of $C$.

**Why the interval is displaced and not just wide.** By the variance of a Beta distribution (T.9), $ab/\big((a+b)^2(a+b+1)\big)$ with $a=f_{jj}+1$ and $b=n^s_j-f_{jj}+K-1$, so $a+b=n^s_j+K$, the posterior standard deviation of a recall is $\sqrt{\tilde r_j(1-\tilde r_j)/(n^s_j+K+1)}$, the usual standard error of a proportion with the imaginary counts included. The shift shrinks like $K/n^s_j$, the standard deviation only like $1/\sqrt{n^s_j}$, so with large $K$ the shift is several standard deviations. Averaging into accuracy makes it worse. If the random errors $\varepsilon_j$ of the recalls are independent with standard deviation $\sigma$, the error of the accuracy, $\sum_j\pi_j\varepsilon_j$, has variance $\sum_j\pi_j^2\sigma^2$, which is smaller than $\sigma^2$ because $\sum_j\pi_j^2<1$ when more than one class is present ($1/K$ for equal prevalences): the random errors partly cancel. The prior instead pushes every recall in the same direction, so with shifts $\delta_j$ of the same sign the shift of the accuracy, $\sum_j\pi_j\delta_j$, is a weighted average of them and does not shrink. The result is a narrow interval in the wrong place, and the predictive, which only adds spread around the same centre, cannot repair it.

Other symmetric priors with less total weight, such as $\operatorname{Dir}(1/K)$ (one imaginary count per column instead of $K$), are studied in [[ext5_baycap_priors_tempering|Extension 5]]. E2.4 tests instead a prior whose centre and weight are estimated from $V$ (§T.9): $\operatorname{Dir}(1/K)$ puts most of its mass near the edges of the simplex, which is hard to justify as a description of the classifier, while a prior estimated from the data says what $V$ says.

**The prior on $\pi^Q$ is not a cause.** `main` draws the target prevalences uniformly over all possible prevalence vectors (quapy's `UPP`), which is exactly BayCAP's default prior $\operatorname{Dir}(\mathbf 1)$ on $\pi^Q$. When the prior matches how the bags are generated, the interval for $\pi^Q$ covers at the nominal rate on average over the bags, so this prior alone cannot explain the under-coverage.

## T.6 Other effects on the width

- **Fixed class counts (too wide).** $\mathcal L_U$ treats the class of each target instance as random, while APP bags have fixed class counts. With random classes, $n^t_i$ is a binomial count with $N^Q$ tries and probability $q_i$, so its variance is $N^Qq_i(1-q_i)$. With fixed counts $n_j=N^Q\pi_j$, $n^t_i$ is a sum of independent binomials, one per class, $\operatorname{Bin}(n_j,C_{ij})$, with variance $\sum_jn_jC_{ij}(1-C_{ij})$. Using $\sum_j\pi_jC_{ij}=q_i$, the difference is
  $$
  N^Q\Big[q_i(1-q_i)-\sum_j\pi_jC_{ij}(1-C_{ij})\Big]=N^Q\Big[\sum_j\pi_jC_{ij}^2-q_i^2\Big]\ge0,
  $$
  $N^Q$ times the variance of $C_{ij}$ across the classes, weighted by $\pi$. The model therefore expects more variation in $n^t$ than there is, trusts the counts too little, and makes the interval wider. In the toy, at $n_s=5000$, coverage of the population accuracy is $0.97$ on APP bags and $0.94$ on bags with random class counts, so the effect is small there. Candidate E2.c2 would measure it.
- **Uncertainty on $C$ (modelled).** A finite $V$ leaves real uncertainty about $C$; $\mathcal L_V$ represents it, and it dominates when $V$ has few instances per class.

## T.7 Conditioning of $C$ (cause 2)

*Intuition:* if the classifier behaves almost the same on two classes, its predictions say little about how many instances of each class the bag contains, and every small error in $C$ or in the counts becomes a large error in $\hat\pi$.

**A binary example** (illustrative numbers). With $\mathrm{TPR}=C_{11}$ and $\mathrm{FPR}=C_{10}$, the fraction of the bag predicted positive is $q=\mathrm{FPR}\,(1-\pi_1)+\mathrm{TPR}\,\pi_1=\mathrm{FPR}+(\mathrm{TPR}-\mathrm{FPR})\,\pi_1$ (law of total probability), so $\pi_1=(q-\mathrm{FPR})/(\mathrm{TPR}-\mathrm{FPR})$. With $\mathrm{TPR}=0.9$ and $\mathrm{FPR}=0.1$ the denominator is $0.8$; with $\mathrm{TPR}=0.55$ and $\mathrm{FPR}=0.45$ it is $0.1$, and the chance error of $q$ on a bag of $1000$ instances, about $\pm0.016$ (the standard deviation of a proportion, $\sqrt{q(1-q)/1000}\approx\sqrt{0.25/1000}$), becomes about $\pm0.16$ on $\pi_1$. Any error in $q$, $\mathrm{TPR}$ or $\mathrm{FPR}$ is multiplied by $1/(\mathrm{TPR}-\mathrm{FPR})$.

**Singular values.** For $K$ classes the role of $\mathrm{TPR}-\mathrm{FPR}$ is played by the singular values of $C$. They measure how much $C$ stretches or shrinks vectors: for every vector $v$, $\sigma_{\min}\lVert v\rVert\le\lVert Cv\rVert\le\sigma_{\max}\lVert v\rVert$, and both bounds are reached in some direction. Singular values are used rather than eigenvalues because $C$ is not symmetric: its eigenvalues can be complex and do not measure stretching.

**The condition number.** Suppose the predicted shares carry an error, $q+\delta q=C(\pi+\delta\pi)$. Since $q=C\pi$, subtracting gives $\delta q=C\,\delta\pi$, so $\lVert\delta q\rVert\ge\sigma_{\min}\lVert\delta\pi\rVert$ and $\lVert\delta\pi\rVert\le\lVert\delta q\rVert/\sigma_{\min}$. Also $\lVert q\rVert=\lVert C\pi\rVert\le\sigma_{\max}\lVert\pi\rVert$, i.e. $1/\lVert\pi\rVert\le\sigma_{\max}/\lVert q\rVert$. Multiplying the two inequalities,

$$
\frac{\lVert\delta\pi\rVert}{\lVert\pi\rVert}\le\kappa\,\frac{\lVert\delta q\rVert}{\lVert q\rVert},
\qquad
\kappa=\frac{\sigma_{\max}}{\sigma_{\min}}:
$$

the condition number $\kappa$ bounds how much relative errors grow when $\pi$ is recovered from $q$. An error $\delta C$ in the matrix adds, to first order, a term of the same form (a standard result of numerical linear algebra), so

$$
\frac{\lVert\delta\pi\rVert}{\lVert\pi\rVert}\lesssim\kappa\left(\frac{\lVert\delta q\rVert}{\lVert q\rVert}+\frac{\lVert\delta C\rVert}{\lVert C\rVert}\right),
$$

covering both chance errors in the counts and systematic errors in $C$, such as the prior's.

**Restricting to vectors that sum to zero.** Prevalences sum to one, so every error in $\pi$ or in $q$ sums to zero. The columns of $C$ sum to one, so for a vector $v$ that sums to zero the entries of $Cv$ sum to $\sum_i\sum_jC_{ij}v_j=\sum_jv_j\sum_iC_{ij}=\sum_jv_j=0$: $C$ maps such vectors to vectors that also sum to zero. E2.2 therefore uses $\kappa_0$, the condition number computed only over vectors that sum to zero; $\kappa_0\ge1$, with $\kappa_0=1$ for a perfect classifier.

**The absolute amplification** [new hypothesis, 2026-10-20]. With $K=2$ the vectors that sum to zero form a single direction, so $C$ acts on them as one number, $\mathrm{TPR}-\mathrm{FPR}$, and $\kappa_0=1$ for every binary classifier, however poor: a ratio of the largest to the smallest stretching is uninformative when there is only one stretching. The binary example above uses instead the factor $1/(\mathrm{TPR}-\mathrm{FPR})$, which multiplies absolute errors. Its general form is $1/\sigma_{\min}$, with $\sigma_{\min}$ the smallest singular value of $C$ on the vectors that sum to zero: by the first inequality above, an error $\delta q$ in the predicted proportions becomes an error of up to $\lVert\delta q\rVert/\sigma_{\min}$ in $\hat\pi$. E2.2 reports both: $1/\sigma_{\min}$ compares pairs of any $K$, including binary ones, while $\kappa_0$ tells, for $K>2$, whether the amplification concerns only some directions.

**Limits of the measure.**

- It is not specific to accuracy. If two classes are completely confused but have the same recall, $C$ is singular ($\kappa_0=\infty$) and their prevalences cannot be told apart, yet the accuracy is still exactly identifiable, because exchanging prevalence between two classes with the same recall does not change it.
- $\hat C$ is estimated on $V$: a column of a rare class rests on few instances, so its $\kappa_0$ is noisy; $n^s_j$ is reported next to it.
- It tends to grow with $K$, because there are more directions in which two columns can be close; comparing the classifiers within a dataset avoids mixing it with $K$, which also drives cause 1.

**The prior and the conditioning** [new hypothesis]. The imaginary counts pull every column towards the flat column $(1/K,\dots,1/K)$, which makes the columns more alike and amplifies errors more. $\kappa_0$ cannot measure this. When all classes have the same $n^s_j=n$, the posterior-mean column is $\frac{f_{\cdot j}+\mathbf 1}{n+K}=\frac{n}{n+K}\hat C_{:j}+\frac{1}{n+K}\mathbf 1$, so the posterior-mean matrix is $\frac{n}{n+K}\hat C+\frac1{n+K}\mathbf 1\mathbf 1^\top$. On a vector $v$ that sums to zero, $\mathbf 1\mathbf 1^\top v=\mathbf 1(\mathbf 1^\top v)=\mathbf 0$, so the matrix acts as $\frac{n}{n+K}\hat C$: the prior multiplies every singular value on these vectors by the same factor $n/(n+K)$, and their ratio does not change. E2.2 does not measure this effect.

## T.8 Difficulty of the problem on i.i.d. data (cause 3)

*Intuition:* a classifier that is often wrong even without any shift carries little information about the classes.

The difficulty of the problem is measured by the accuracy of $h$ on $V$, which is drawn from the source distribution without any prior shift. No cross-validation is needed: BayCAP evaluates the classifier $h$ that is already trained, and its predictions on $V$ are stored; cross-validating on $V$ would train other classifiers on less data. Difficulty and conditioning are related but different: a weak classifier often has similar columns, but a $26$-class classifier with modest recalls can still separate the classes well. In a correct model, low accuracy alone widens the interval rather than displacing it, so if displacement follows the accuracy on $V$ but not $\kappa_0$, it points to a mechanism not described here.

## T.9 An empirical-Bayes prior on the recalls (E2.4)

*Intuition:* the $K$ classes are judged by the same classifier, so they tell us what a typical recall of that classifier looks like. If $25$ classes have a recall around $0.95$, the $26$th is very likely around $0.95$ as well, not around $1/26$. Empirical Bayes estimates the typical recall and how much the recalls differ from the validation data, and pulls each recall towards the typical one: strongly when the class has few validation instances, weakly when it has many.

The numerical examples of this section are illustrative, computed by hand, not results. We write $n_j=n^s_j$ and $f_j=f_{jj}$ for the validation instances of class $\omega_j$ and the correctly classified ones, and $r_j=C_{jj}$ for the recall.

**The model for the validation counts.** Each class is like a coin with unknown probability of heads $r_j$, tossed $n_j$ times. The randomness is in the instances, not in the classifier: $h$ is fixed, and a given instance is always classified the same way. What is random is which instances end up in $V$.

- *One instance is one toss.* An instance $X$ of class $\omega_j$ is a draw from the class-conditional distribution $P(X\mid Y=\omega_j)$ of the source. Its success indicator $B=\mathbb 1\{h(X)=\omega_j\}$ is a Bernoulli variable with $P(B=1)=P(h(X)=\omega_j\mid Y=\omega_j)=C_{jj}=r_j$: the recall is the share of the class that $h$ classifies correctly, and a random draw from the class lands in that share with probability $r_j$.
- *The tosses are independent and identical.* The instances of $V$ are drawn independently from the source, so the $n_j$ indicators $B_1,\dots,B_{n_j}$ of class $\omega_j$ are independent, each with probability $r_j$. This needs $h$ to be trained on data other than $V$, as in the paper's protocol; if $h$ had seen $V$, the indicators would depend on $V$ through $h$.
- *Counting the successes gives the binomial.* $f_j=\sum_kB_k$. By independence, a particular sequence with $f$ successes and $n_j-f$ failures has probability $r_j^f(1-r_j)^{n_j-f}$, and there are $\binom{n_j}{f}$ ways to choose which positions are the successes, so

$$
f_j\mid r_j\sim\operatorname{Binomial}(n_j,r_j),\qquad P(f_j\mid r_j)=\binom{n_j}{f_j}r_j^{f_j}(1-r_j)^{n_j-f_j},
$$

with mean $n_jr_j$ and variance $n_jr_j(1-r_j)$, $n_j$ times those of one toss.

Two remarks. In an i.i.d. sample $n_j$ is itself random; the model conditions on it (how many of these $n_j$ instances are classified correctly), which is legitimate because $n_j$ carries no information on $r_j$. And $V$ is drawn once, so its $f_j$ is one number: the binomial describes how $f_j$ would vary over the validation sets we could have drawn, and this variation is the uncertainty on $r_j$. Recording the predicted class instead of "correct or not" turns each toss into the roll of a $K$-faced die with probabilities $C_{:j}$, and the column of counts into $\operatorname{Multinomial}(n_j,C_{:j})$, the likelihood $\mathcal L_V$ of BayCAP (§T.1); the binomial is that multinomial with all the wrong faces merged into one.

**The plain recall is the maximum-likelihood estimate.** The value of $r_j$ that makes the observed count most probable maximises the log-likelihood $\ell(r_j)=f_j\ln r_j+(n_j-f_j)\ln(1-r_j)$ (the binomial coefficient does not depend on $r_j$). Setting the derivative to zero,

$$
\frac{f_j}{r_j}-\frac{n_j-f_j}{1-r_j}=0\iff r_j=\frac{f_j}{n_j}=\hat r_j .
$$

Since $\mathbb E[f_j]=n_jr_j$ and $\operatorname{Var}(f_j)=n_jr_j(1-r_j)$, the plain recall is unbiased, $\mathbb E[\hat r_j]=r_j$, with variance $r_j(1-r_j)/n_j$: right on average, but noisy when $n_j$ is small, and with no statement of its own uncertainty ($f_j=n_j$ gives $\hat r_j=1$ "for sure").

**A Beta prior and the posterior mean.** A Bayesian describes the uncertainty on $r_j$ by a distribution on $[0,1]$. The Beta distribution has density $p(r)=r^{a-1}(1-r)^{b-1}/B(a,b)$, with $B(a,b)=\int_0^1r^{a-1}(1-r)^{b-1}dr=\Gamma(a)\Gamma(b)/\Gamma(a+b)$. Using $\Gamma(x+1)=x\,\Gamma(x)$, its mean and variance are

$$
\mathbb E[r]=\frac{B(a+1,b)}{B(a,b)}=\frac{a}{a+b},\qquad \operatorname{Var}[r]=\frac{ab}{(a+b)^2(a+b+1)} .
$$

We write $a=\nu m$ and $b=\nu(1-m)$: the **centre** $m=\mathbb E[r]$ is the recall the prior expects, and the **strength** $\nu=a+b$ sets how concentrated the prior is, $\operatorname{Var}[r]=m(1-m)/(\nu+1)$. By Bayes' theorem the posterior is proportional to likelihood times prior,

$$
p(r_j\mid f_j)\propto r_j^{f_j}(1-r_j)^{n_j-f_j}\cdot r_j^{a-1}(1-r_j)^{b-1}=r_j^{f_j+a-1}(1-r_j)^{n_j-f_j+b-1},
$$

which is again a Beta, $\operatorname{Beta}(f_j+a,\,n_j-f_j+b)$ (conjugacy). The update adds the observed successes to $a$ and the observed failures to $b$: the prior acts as $\nu$ imaginary validation instances, a share $m$ of which are correct. As a single estimate we take the posterior mean, which minimises the expected squared error, since $\mathbb E[(r_j-c)^2\mid f_j]=\operatorname{Var}(r_j\mid f_j)+(\mathbb E[r_j\mid f_j]-c)^2$ is smallest at $c=\mathbb E[r_j\mid f_j]$:

$$
\tilde r_j=\frac{f_j+\nu m}{n_j+\nu}=w_j\,\hat r_j+(1-w_j)\,m,\qquad w_j=\frac{n_j}{n_j+\nu}.
$$

It is a weighted average of what $V$ says and what the prior says, with the shares of real and imaginary instances as weights. For a fixed true recall, averaging over the possible validation sets ($\mathbb E[f_j]=n_jr_j$), its bias points towards $m$:

$$
\mathbb E[\tilde r_j]-r_j=\frac{\nu}{n_j+\nu}\,(m-r_j).
$$

**The uniform prior is the case $m=1/K$, $\nu=K$.** BayCAP puts $\operatorname{Dir}(\mathbf 1)$ on the whole column $C_{:j}$. A Dirichlet vector can be built as $p_i=G_i/\sum_lG_l$ from independent $G_i\sim\operatorname{Gamma}(\alpha_i,1)$; the sum of the other entries, $G_{-j}=\sum_{i\ne j}G_i$, is $\operatorname{Gamma}(\sum_{i\ne j}\alpha_i,1)$, and $G_j/(G_j+G_{-j})$ is Beta-distributed. The same construction gives the *aggregation property*: a sum of independent $\operatorname{Gamma}(\alpha_i,1)$ variables is $\operatorname{Gamma}(\sum_i\alpha_i,1)$, so if the entries of a $\operatorname{Dir}(\alpha)$ vector are summed in groups, the group sums form a Dirichlet vector whose parameters are the sums of the $\alpha_i$ in each group. Its mean follows as well: $\mathbb E[p_i]=\alpha_i/\sum_l\alpha_l$, the mean $a/(a+b)$ of the $\operatorname{Beta}(\alpha_i,\sum_{l\ne i}\alpha_l)$ below. So one entry of a $\operatorname{Dir}(\alpha)$ vector is $\operatorname{Beta}(\alpha_j,\sum_{i\ne j}\alpha_i)$, and under $\operatorname{Dir}(\mathbf 1)$ the recall is $\operatorname{Beta}(1,K-1)$: $m=1/K$ and $\nu=K$, chosen without looking at the data. The formulas above then give $\tilde r_j=\frac{f_j+1}{n_j+K}$ and the bias $\frac{K(1/K-r_j)}{n_j+K}$ of §T.5. As a check against the stored results, `isolet` LR ($K=26$, $n_j\approx105$, recall on $V$ $0.95$) gives $26(0.038-0.95)/131\approx-0.181$, the $R_0$ that E2.2 reports.

**Choosing $m$ and $\nu$ from the data.** Empirical Bayes assumes that the $K$ recalls of one classifier are themselves draws from a common Beta, as if the classes were coins from the same factory:

$$
r_j\sim\operatorname{Beta}\big(\nu m,\nu(1-m)\big)\ \text{independently},\qquad f_j\mid r_j\sim\operatorname{Binomial}(n_j,r_j),\qquad j=1,\dots,K.
$$

The assumption is that the classes are exchangeable: before looking at $V$, there is no reason to expect a particular class to be harder than another. The recalls are not observed, but the counts are, and the probability of a count under $(m,\nu)$ averages the binomial over all possible recalls (with $a=\nu m$, $b=\nu(1-m)$):

$$
P(f_j\mid m,\nu)=\int_0^1\binom{n_j}{f_j}r^{f_j}(1-r)^{n_j-f_j}\frac{r^{a-1}(1-r)^{b-1}}{B(a,b)}\,dr=\binom{n_j}{f_j}\frac{B(f_j+a,\,n_j-f_j+b)}{B(a,b)},
$$

the beta-binomial distribution: the integral is the definition of $B$ with the exponents $f_j+a$ and $n_j-f_j+b$. The classes are independent, so, dropping the binomial coefficients, the log-likelihood of $(m,\nu)$ is

$$
\ell(m,\nu)=\sum_{j=1}^K\Big[\ln B\big(f_j+\nu m,\,n_j-f_j+\nu(1-m)\big)-\ln B\big(\nu m,\,\nu(1-m)\big)\Big],
$$

and $(\hat m,\hat\nu)$ is its maximum. There is no closed form: the maximum is found numerically over $(\operatorname{logit}m,\ln\nu)$, which can take any real value, with $\ln B$ computed from log-Gamma functions. The cost is negligible.

**Why rare classes count less.** Under the model, the plain recalls have mean and variance (laws of total expectation and total variance, with $\mathbb E[r_j(1-r_j)]=m(1-m)-\operatorname{Var}(r_j)=m(1-m)\frac{\nu}{\nu+1}$)

$$
\mathbb E[\hat r_j]=m,\qquad \operatorname{Var}(\hat r_j)=\underbrace{\mathbb E\Big[\tfrac{r_j(1-r_j)}{n_j}\Big]}_{\text{binomial noise}}+\underbrace{\operatorname{Var}(r_j)}_{\text{true spread}}=\frac{m(1-m)}{\nu+1}\cdot\frac{n_j+\nu}{n_j}.
$$

A class with few instances has a noisier $\hat r_j$. Approximating each $\hat r_j$ as normal with these moments, the log-likelihood becomes a sum of squares $-\frac12\sum_j(\hat r_j-m)^2/\operatorname{Var}(\hat r_j)$ plus terms in $\nu$, and its maximum in $m$ is a weighted mean, $\hat m\approx\sum_jw_j\hat r_j/\sum_jw_j$, with the weights $w_j=n_j/(n_j+\nu)$ of the posterior mean. The exact maximum does the same automatically. A simpler method of moments (the plain average of the $\hat r_j$ for $m$, and the spread across classes minus the average binomial noise for the variance) gives every class the same weight; in an illustrative example with $n_j=200,200,200,5$ and $f_j=190,192,188,3$, it puts $m$ at $0.86$, dragged down by the class with $5$ instances, while the weighted mean with a large $\nu$ is $573/605\approx0.947$. Since the pairs of `main` have very unequal $n^s_j$ (down to $3$ on `poker_hand`), E2.4 uses maximum likelihood.

**The limit cases.** If the counts vary no more than binomial noise alone explains, the likelihood keeps increasing with $\nu$: as $\nu\to\infty$ the prior collapses on $m$, the beta-binomial becomes $\operatorname{Binomial}(n_j,m)$, the maximum is the pooled recall $\hat m=\sum_jf_j/\sum_jn_j$, and every class gets it. If the classes truly differ a lot, $\hat\nu$ is small, $w_j\approx1$, and each class keeps its plain recall. With $K=2$ two counts determine two parameters, and $\hat\nu$ often lands on one of the two limits; at $K=2$ the uniform prior was already harmless (E2.2: $R_0$ between $-0.001$ and $-0.006$). If every recall on $V$ is $1$, $\hat m\to1$. The fit is therefore bounded, $\nu\le10^4$ and $m\in[10^{-3},1-10^{-3}]$, and E2.4 reports the pairs where it reaches a bound.

**From the Beta to a prior on the whole column.** BayCAP needs a prior on each column of $C$, not only on its diagonal entry, because it estimates $\pi^Q$ from $q=C\pi^Q$. By the aggregation property above, the Dirichlet with

$$
\alpha_{jj}=\hat\nu\hat m,\qquad \alpha_{ij}=\frac{\hat\nu(1-\hat m)}{K-1}\quad(i\ne j)
$$

has diagonal entry $C_{jj}\sim\operatorname{Beta}(\hat\nu\hat m,\hat\nu(1-\hat m))$, since the other parameters sum to $\hat\nu(1-\hat m)$. Only the concentration parameters change; the likelihoods of $V$ and of the target counts, the prior on $\pi^Q$ and the sampler are those of BayesianCC. By conjugacy, $V$ turns the prior into $\operatorname{Dir}(f_{\cdot j}+\alpha_{\cdot j})$, whose diagonal mean is $\tilde r_j=\frac{f_j+\hat\nu\hat m}{n_j+\hat\nu}$; the target bag then updates $C$ and $\pi^Q$ together, as before. Two consequences:

- *The spread of the mistakes is fixed by $(\hat m,\hat\nu)$.* In a Dirichlet column, $C_{jj}$ is independent of how the rest of the column, $1-C_{jj}$, is shared among the wrong classes, and that share follows $\operatorname{Dir}(\alpha_{ij},\,i\ne j)$, here with all parameters equal to $\beta=\hat\nu(1-\hat m)/(K-1)$. A small $\beta$ lets the mistakes concentrate on the few classes that $V$ shows to be confused, while the uniform prior adds one imaginary mistake to every wrong class, which flattens the column (illustrative: with $104$ instances, $99$ correct and $5$ mistakes on $2$ of $25$ wrong classes, $\operatorname{Dir}(\mathbf 1)$ gives the $23$ never-confused classes a total of $23/130\approx0.18$ of the column). Choosing $\beta$ independently of $(\hat m,\hat\nu)$ would leave the Dirichlet family and change BayesianCC.
- *Small concentrations.* For an accurate classifier $\beta<1$ (illustrative: $\hat\nu=41$, $\hat m=0.95$ give $\beta\approx0.08$); a cell never filled on $V$ then has a posterior density that grows without bound at $0$. In the unconstrained coordinates where NUTS samples, the density stays finite but has a long tail, which can slow the sampler down. This is why E2.4 records the NUTS diagnostics.

**What happens to the shift.** With the uniform prior every recall above $1/K$ is pulled down, and the pulls add up in $A=\sum_j\pi_jr_j$. With the empirical-Bayes prior the bias of class $\omega_j$ is $\frac{\hat\nu}{n_j+\hat\nu}(\hat m-r_j)$: positive for the classes below the typical recall and negative for those above it. Under the model, $\mathbb E[m-r_j]=0$, so the biases cancel on accuracy on average over the classes; under UPP the prevalences are drawn independently of the classifier, so this holds on average over the bags. Two limits remain. The classes may not be exchangeable: a truly hard class is pulled up towards the others, and bags dominated by it are displaced (pooling reduces the total squared error over the classes, not the error of every class). And causes 2 and 4 of E2.2 are not addressed.

**Using $V$ twice.** $V$ chooses $(\hat m,\hat\nu)$ and then updates $C$ again, while the posterior treats $(\hat m,\hat\nu)$ as known constants. A fully Bayesian model would average over them, $p(r_j\mid V)=\int p(r_j\mid V,m,\nu)\,p(m,\nu\mid V)\,dm\,d\nu$, and by the law of total variance

$$
\operatorname{Var}(r_j\mid V)=\underbrace{\mathbb E\big[\operatorname{Var}(r_j\mid V,m,\nu)\mid V\big]}_{\text{kept by empirical Bayes}}+\underbrace{\operatorname{Var}\big(\mathbb E[r_j\mid V,m,\nu]\mid V\big)}_{\text{left out}},
$$

where the left-out term is about $(1-w_j)^2\operatorname{Var}(\hat m)$ if the uncertainty on $\nu$ is ignored. The error of $\hat m$ is shared by all the classes, so on accuracy it does not average out: the left-out standard deviation of the accuracy is about $\sum_j\pi_j(1-w_j)\operatorname{sd}(\hat m)$. $\operatorname{sd}(\hat m)$ is estimated from the curvature of $\ell$ at its maximum. For one parameter $\theta$, the log-likelihood near its maximum is close to a parabola, $\ell(\theta)\approx\ell(\hat\theta)+\frac12\ell''(\hat\theta)(\theta-\hat\theta)^2$, which is the logarithm of a normal density with variance $-1/\ell''(\hat\theta)$: a sharply peaked likelihood (large $-\ell''$) means a small uncertainty, and $\operatorname{sd}(\hat\theta)\approx1/\sqrt{-\ell''(\hat\theta)}$. With two parameters, $-\ell''$ becomes the matrix of second derivatives (the *observed information*) and the variances are the diagonal of its inverse; E2.4 computes it in the coordinates $(\operatorname{logit}m,\ln\nu)$. The standard deviation of $\hat m$ then follows by the delta method: if $\hat m=g(\hat\theta)$ with $\hat\theta$ close to the truth, a first-order expansion $g(\hat\theta)\approx g(\theta)+g'(\theta)(\hat\theta-\theta)$ gives $\operatorname{sd}(\hat m)\approx\lvert g'(\hat\theta)\rvert\operatorname{sd}(\hat\theta)$; for $m=1/(1+e^{-\theta})$, the inverse of the logit, $g'(\theta)=m(1-m)$, so $\operatorname{sd}(\hat m)=\hat m(1-\hat m)\operatorname{sd}(\operatorname{logit}\hat m)$. In an illustrative case like `isolet` ($K=26$, $n_j=104$, $\hat\nu=41$, spread of the recalls $0.04$) it is about $0.002$, against about $0.0036$ for the standard deviation that the uncertainty on $C$ gives the accuracy, so the $C$ part of the width would be about $13\%$ too small; the uncertainty on $\pi^Q$ makes the effect on the whole interval smaller. Because $V$ is drawn once, the error of $\hat m$ is the same in every bag of a pair: if it matters, it appears as a small displacement, not as random width. A second, smaller effect: each class enters $\hat m$, so the centre it is pulled towards has been moved slightly towards itself, and the pull is a little too weak; the effect is of order $1/K$. E2.4 reports the left-out standard deviation per pair. The fully Bayesian version, which samples $(m,\nu)$ with NUTS, would change the model of BayesianCC and is not pursued.

**What the check can and cannot predict.** The check of E2.4 computes $\tilde r_j$ from $V$ alone and compares it with the realised recalls of each bag. It predicts the recall term of the shift, not the coverage (it has no interval), and it leaves out the step in which the target bag updates $C$, the effect of the mistakes on $\hat\pi$ (the prevalence term), and the left-out variance. The variant measures all of these, and the comparison of the two tells which of them matter.

# Implementation

Components:

- **`tables/estimand_gap.py`** (E2.1): per-pair table, gate summary, suggested pilot datasets, LaTeX and PDF output. It must run where the `main` results are stored.
- **`tables/displacement.py`** (E2.2, work-plan card WP-3): reuses `load_results` and `booktabs_clines` of `estimand_gap.py`; it rebuilds $f_{ij}$ as `BayCAP.fit` does, from the stored validation posteriors (`PretrainInfo.load_posteriors`) and the validation labels (`PretrainInfo.load_dataset_bundle`), computes the measures of E2.2 per bag and per pair, and writes the per-pair CSV and the table (LaTeX and landscape PDF) under `<OUT_DIR>/uncawe/tables/displacement/`. $\kappa_0$ and $1/\sigma_{\min}$ come from the singular values of $Q^\top\hat CQ$, with $Q$ an orthonormal basis of the vectors summing to zero; both are infinite when $\hat C$ is singular on them. Per pair it also computes the predicted coverage $\mathcal C_s$ from the mean shift and the mean amplitude (`predicted_coverage`, with $z$ the normal quantile of `CONFIDENCE_LEVEL`), and prints its correlation with the coverage; and it reports the largest gap between the mean of the accuracy samples and the accuracy of the mean table, which is zero up to rounding because accuracy is linear.
- **`plots/displacement.py`** (E2.2): both scatters, from the per-pair table, under `<OUT_DIR>/uncawe/plots/main/displacement/`.
- **`methods/baycap.py:BayCAP`** (for E2.3): a copy of `cap-ml` `ACC_BayesCAP` (`fit`, `model`, `sample_posterior`, `predict_ct_range`, `predict_ct`; copied from commit `bb20aa5`) with the hyperparameter `estimand` $\in\{$`"population"`, `"realised"`$\}$, default `"population"`.
  - `"population"` returns the posterior tables $C^{(r)}\operatorname{diag}(\pi^{(r)})$, stored transposed as in `cap-ml`, exactly as `ACC_BayesCAP`.
  - `"realised"` returns realised-table predictive samples: for each draw and each predicted class $\omega_i$ (a column in the `cap` layout), the true-class counts are drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$ and the table is divided by $N^Q$.
  - NUTS keeps `PRNGKey(random_state)`; the multinomial draws use a NumPy `Generator` seeded from `random_state`, so both values of `estimand` with the same `random_state` share the posterior draws.
  - The realised pushforward is a module-level function (`realised_tables`), reusable on any table draws.
  - Method names in experiments: `BayesCAP` (population, as in `main`) and `BayesCAP-real` (realised).
- **`BayesCAP-real` in `main`** (E2.3, work-plan card WP-4): a factory entry in `experiments/methods.py` (`METHOD_CLASS` and `gen_methods`), `BayCAP(acc, estimand="realised", num_samples=BOOTSTRAP_ITERS, num_warmup=BAYES_WARMUPS)`, run with the other BayCAP methods (`BAYES_N_JOBS`); results under `<OUT_DIR>/uncawe/main/` like every method. It is excluded from the tables and ranks of `main` (`tables/main.py:EXCLUDED_METHODS`, also used by `tables/rank.py`) and from the default methods of its plots (`plots/common.py:DEFAULT_METHODS`), so the paper's tables and ranks are unchanged; only the E2 tables and plots use it. `main` takes `--datasets`, `--classifiers` and `--methods`, so the pilot can compute `BayesCAP-real` alone on a few pairs.
- **`tables/displacement.py` and `plots/displacement.py`** (for E2.3): take the method as `--method` (default `BayesCAP`); for any other method the outputs go to a subfolder named after the method (e.g. `<OUT_DIR>/uncawe/tables/displacement/BayesCAP-real/`), so the stored E2.2 outputs keep their paths and are not overwritten.
- **`tables/pushforward.py`** (E2.3): the comparison table, one row per pair where both pushforwards are stored (so it also works on the pilot), with the approximate coverage and $\hat\sigma_g/\hat\sigma_p$ read from the E2.1 CSV, which must exist; as a check, it reports the largest gap between its population coverage and the one stored by E2.1. CSV, LaTeX and PDF under `<OUT_DIR>/uncawe/tables/pushforward/`.
- **`plots/pushforward.py`** (E2.3): the two coverage scatters, coloured by $\hat\sigma_g/\hat\sigma_p$ on a log scale, under `<OUT_DIR>/uncawe/plots/main/pushforward/`.
- **`methods/baycap.py:BayCAP` with `c_prior`** (for E2.4, work-plan card WP-5, implemented, not evaluated): the hyperparameter `c_prior` $\in\{$`"uniform"`, `"empirical-bayes"`$\}$, default `"uniform"`, which keeps the current model. With `"empirical-bayes"`, `fit` estimates $(\hat m,\hat\nu)$ from the diagonal and the class totals of the validation table by maximising the beta-binomial log-likelihood of §T.9 over $(\operatorname{logit}m,\ln\nu)$, within the bounds of §T.9, and stores them, whether a bound was reached, and $\operatorname{sd}(\hat m)$ as fitted attributes; `model` uses the concentration matrix $\alpha$ of §T.9 in place of the matrix of ones. In the `cap` layout a column of $C$ is a row of `p_c_cond_y`, so $\alpha$ is built transposed (it is symmetric, so the two layouts coincide). The functions `beta_binomial_loglik`, `fit_beta_binomial` (L-BFGS-B with the analytic gradient, several starting values of $\ln\nu$, and $\nu\ge10^{-3}$ as a numerical safeguard) and `eb_concentration` are module-level; the fit is stored as `eb_fit`, the concentrations as `c_alpha`.
- **E2.4 in `main`** (work-plan card WP-6, planned): `BayesCAP` built as `BayCAP(acc, estimand="population")` instead of `ACC_BayesCAP`; the methods `BayesCAP-EB` and `BayesCAP-EB-real` (`c_prior="empirical-bayes"`, population and realised), results only like `BayesCAP-real`; `BayCAP` exposes the NUTS diagnostics of each `predict_ct_range` call (divergences, largest split-$\hat R$, smallest effective sample size), which the evaluation of contingency methods in `experiments/main.py` adds to the result frame as new per-bag columns. A table module and a plot module for E2.4 reuse the bootstrap test of `tables/pushforward.py` and the validation counts and decomposition of `tables/displacement.py`; `tables/displacement.py` and `plots/displacement.py` already take `--method`.

Tests:

- `BayCAP(estimand="population")` reproduces `ACC_BayesCAP` exactly for the same `random_state`;
- with the same `random_state`, both values of `estimand` use identical NUTS draws;
- realised tables have predicted-class margins equal to $n^t/N^Q$ and, with $C=I$, return the observed counts exactly;
- for a fixed draw, the mean of many realised count tables has row $i$ equal to $n^t_i\,\rho^{(r)}_{i\cdot}$;
- a fixed `random_state` gives identical outputs; the layout is checked against a hand-computed $2\times2$ example (guards against transposed tables);
- E2.2: the recall and prevalence terms sum exactly to the shift on a hand-computed table; the prior part equals $\frac{f_{jj}+1}{n^s_j+K}-\frac{f_{jj}}{n^s_j}$ on a hand-computed validation table; $\kappa_0$ and $1/\sigma_{\min}$ match hand-computed $2\times2$ ($\kappa_0=1$, $1/\sigma_{\min}=1/(\mathrm{TPR}-\mathrm{FPR})$) and $3\times3$ examples and are infinite for singular ones; the recall and prevalence terms add up to the shift also on a bag with an absent class; $\hat C$ is in the paper's layout; the predicted coverage is $0.95$ with no shift, symmetric in the sign of the shift, about $0.5$ at one half-width, and undefined for a zero amplitude; the method argument reads the results of the given method and keeps the E2.2 output folders for `BayesCAP` (`tests/test_displacement.py`);
- E2.3: the factory builds `BayesCAP-real` with the priors, NUTS settings and seed of `BayesCAP`, and the tables, ranks and default plots of `main` exclude it (`tests/test_methods.py`); the filters of `main` restrict the pairs and methods and send `BayesCAP-real` to the BayCAP pool (`tests/test_main_protocol.py`); the comparison table keeps the pairs stored for both methods, requires every pair in the E2.1 CSV, and has the expected layout, and both figures build (`tests/test_pushforward.py`).
- E2.4, `c_prior` (WP-5, `tests/test_baycap.py`): the log-likelihood matches `scipy.stats.betabinom`; the fit recovers $(m,\nu)$ on synthetic counts with $400$ classes; when all classes share one recall, the fit reaches the upper bound of $\nu$ and gives $\hat m=\sum_jf_j/\sum_jn_j$ up to $10^{-4}$ (the pooled recall is the maximum only as $\nu\to\infty$); with all recalls $1$ it reaches the bound of $m$; the concentration matrix has diagonal $\hat\nu\hat m$ and off-diagonal sums $\hat\nu(1-\hat m)$, in the `cap` layout; `c_prior="uniform"` reproduces the current draws and `ACC_BayesCAP` for a fixed `random_state`; with `"empirical-bayes"`, `fit` stores the fit of the validation counts and both estimands return valid tables.
- E2.4, the rest (WP-6, planned): `BayesCAP` built from `BayCAP` reproduces `ACC_BayesCAP`; the diagnostics are stored for every bag of the four BayCAP methods; the check gives $\tilde r_j=\frac{f_j+1}{n_j+K}$ and the empirical-Bayes $\tilde r_j$ on hand-computed counts.

Reproducibility and comparability:

- `main` and `binform` and their output paths are unchanged; E2.1 and E2.2 only read stored results;
- E2.3 adds the rows of `BayesCAP-real` to `main` and leaves the stored results of every other method untouched; with the same bags, seeds and NUTS settings, its posterior draws equal those of the stored `BayesCAP` on the same platform and JAX version, while bitwise agreement is not expected across JAX platforms, in which case the two pushforwards would be compared on different but equally distributed draws; a check on 5 pairs (`scripts/bench/xla_vs_stored.py`, 2026-10-22) found the stored `BayesCAP` results different from a CPU recomputation (coverage changes up to $0.05$, on `wine-quality`) and `BayesCAP-real` identical, so `BayesCAP` is recomputed on CPU before E2.3 and the old results are kept in `<OUT_DIR>/uncawe/main_22_10_26/`; E2.1 and E2.2 were computed on the old results;
- E2.4 (planned) adds the rows of `BayesCAP-EB` and `BayesCAP-EB-real` and reruns `BayesCAP` and `BayesCAP-real` to store the NUTS diagnostics; before the stored results of `BayesCAP` are deleted, `scripts/bench/xla_vs_stored.py` must show that `BayCAP(estimand="population")` reproduces them, so that the rerun changes no stored measure and adds only the diagnostic columns; the paper's tables are then unaffected;
- every row records the prior, the seeds and the pushforward (population or predictive).

# Appendix

## A. Toy simulation

`uv run python scripts/toys/ext2_estimand_gap.py` (defaults: $3000$ bags, seed $0$, about $90$ s) is seeded and prints the table of §T.3, including the bags with random class counts used in §T.6. It computes the exact BayCAP posterior for a binary problem by sampling $C$ from its validation posterior and weighting a grid over $\pi^Q_1$ by $\mathcal L_U$; it does not use NUTS or `cap-ml`.

## B. Relation to other extensions

- **[[ext5_baycap_priors_tempering|Extension 5]]**: priors (on $\pi^Q$ and on $C$) and tempering of BayCAP (temperature $\tau$); it compares tempering with the realised-table predictive and studies priors on $C$ that avoid the shift of §T.5, such as $\operatorname{Dir}(1/K)$; the empirical-Bayes prior of E2.4 is tested here, because it is the fix suggested by E2.2.
- **[[ext1_metrics|Extension 1]]**: supplies the metrics.
- **[[ext4_uncertainty_budget|Extension 4]]**: the estimand gap is its finite-$U$ term; its regime map varies $\lvert U\rvert$ and $\lvert V\rvert$.
- **[[extensions_to_evaluate#B. Transductive BayCAP and the link to CBPE [new]|Proposal B]]**: keeps the soft-posterior (CBPE mixture) variant of the realised-table predictive.
- **[[ext3_posterior_propagation|Extension 3]]**: BayCAP-KDEy also targets a population table, so the estimand gap applies to it; Extension 3 applies the realised-table predictive of §T.3 to it (E3.1).
- **[[optional_extensions]]**: §1 covers post-hoc interval rescaling.

## C. Out of scope

- tempering, informative priors, and priors on $C$ other than the empirical-Bayes prior of E2.4 ([[ext5_baycap_priors_tempering|Extension 5]]); the fully hierarchical version of that prior, which would change the model of BayesianCC;
- the soft-posterior variant of the realised-table predictive (Proposal B);
- post-hoc interval recalibration;
- violations of PPS.
