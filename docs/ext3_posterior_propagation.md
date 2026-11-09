# Extension 3: Propagating a Bayesian prevalence posterior to classifier performance

## Claim

A Bayesian quantifier returns a posterior over the target prevalence $\pi^Q$, not just one estimate. This extension adds CAP methods that turn such a posterior into an interval on classifier performance, and compares them with the methods of the paper. **BayCAP with a fixed $\hat C$** takes the posterior of a Bayesian quantifier and pushes each draw $\pi^{(r)}$ through the rate matrix of the classifier measured on the validation set, $\hat C\operatorname{diag}(\pi^{(r)})$, either as a population table or, with the realised-table predictive of [[ext2_baycap|Extension 2]], as a plausible realised table of the target bag. Two quantifiers are considered: the Bayesian KDEy of [KDE-Bayes](papers/kde_bayes.pdf) (BayesKDEy, giving **BayCAP-KDEy**), which estimates from the validation set how the outputs of a classifier are spread within each class, and QuaPy's BayesianMAPLS, a Bayesian version of the EM quantifier EMQ (giving **BayCAP-EMQ**), which reads the outputs as calibrated probabilities. Each quantifier can read the outputs of the classifier $h$ whose performance we estimate, or those of a separate MLP trained on the validation set. **SQS-B** does what the other SQS methods do, but draws the prevalences of its validation bags from the posterior of a Bayesian quantifier instead of from a point estimate or a bootstrap. The methods differ from existing ones in one component at a time (the quantifier, the classifier it reads, the handling of $C$, the source of the prevalences), so their comparison with BayCAP and RQBS tells which component matters. The uncertainty budget of [[ext4_uncertainty_budget|Extension 4]] predicts which sources of error each method represents, and so where it should cover well; Extension 4 also reuses the same implementations in its regime map over $\lvert U\rvert$ and $\lvert V\rvert$. The first experiment, E3.1, evaluates the four combinations of quantifier and input classifier, with both pushforwards, on the protocol of `main`. The second, E3.2, makes the rate matrix of BayCAP-KDEy random: it splits each class of the validation set by the prediction of $h$, so that the class densities of BayesKDEy become mixtures of cell densities whose weights are the columns of $C$, and samples $C$ jointly with $\pi$, with the Hf prior of Extension 2, so that the target bag updates both.

## Status

| Item | Status |
|---|---|
| `methods/baycap_kdey.py:BayCAPKDEy` with `c_prior="fixed"` (BayesKDEy) and `methods/baycap_mapls.py:BayCAPMAPLS` (BayesianMAPLS): fixed $\hat C$, $h$ or an MLP as input classifier, population or realised pushforward (work plan §II.4); one class, `methods/baycap_fixed.py:BayCAPFixed`, until the split of WP-13, which kept the draws | [implemented], evaluated by E3.1 |
| E3.1, BayCAP with a fixed $\hat C$: quantifier × input classifier × pushforward, on the `main` protocol | run and analysed (card WP-8 of [[work_plan]], done): BayCAP-KDEy with the realised pushforward is the best of the eight methods |
| SQS-B | [proposal], not implemented |
| BayCAP with a fixed $\hat C$ and the posterior of BayesianCC | [proposal], not implemented |
| E3.2, BayCAP-KDEy with a random $C$: the joint posterior of $(\pi,C)$ with a kernel likelihood and the Hf prior (§T.10–§T.13), with the split of `BayCAPFixed` into `BayCAPKDEy` and `BayCAPMAPLS` | run and analysed (card WP-13 of [[work_plan]], done): no improvement over E3.1 overall; it helps where the error is $e_2$ (`mammographic`, `semeion`) and fails elsewhere; not carried forward (user's decision of 2026-11-09) |
| A random rate matrix for BayesKDEy (options 1–2 of Appendix B); BayesianMAPLS with a uniform prior or a random calibration map (Appendix E) | [new hypothesis] |
| The predictions of §T.9 and the moving centre of §T.8 | checked by E3.1: P1–P3, P5, P6 hold, P8 only for BayesKDEy, P7 does not; the centre barely moves |
| The predictions P9–P14 of §T.13 | checked by E3.2: P9 in part, P11 only for `mlp`, P14 hold; P10 and P12 do not; P13 shows the opposite pattern |
| Candidate experiments E3.c1–E3.c4, E3.c6–E3.c8 (E3.c5 became E3.2) | [proposal], not scheduled |

This document replaces the former Extensions 8 (`docs/sqs_b.md`) and 9 (`docs/baycap_kdey.md`); their last version is in commit `391220c`, and the material that was set aside is in [[optional_extensions]] §8.

# Experiments

All experiments use the protocol of `main`: the $15$ datasets and $5$ classifiers ($75$ pairs), the $100$ UPP bags of size $1000$ per pair with the seeds of `main`, the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), and target labels only for scoring. Every interval is scored against the realised table of each bag (`true_cts`). Each experiment has its own NUTS runs (user's decision of 2026-10-27): draws are not shared across experiments.

## E3.1: BayCAP with a fixed $\hat C$: quantifier × input classifier × pushforward, on the `main` protocol

**Why.** BayCAP-KDEy is implemented but has never been evaluated, and it differs from BayCAP in three components at once: the quantifier (BayesKDEy instead of BayesianCC), the classifier whose outputs the quantifier reads (a separate MLP instead of $h$), and the rate matrix (fixed at its validation estimate $\hat C$ instead of sampled). E3.1 keeps the rate matrix fixed and crosses the other two components, so that their effects can be told apart. The quantifier is BayesKDEy, which estimates the distribution of the outputs within each class from $V$ (§T.3), or BayesianMAPLS, which takes it from the outputs themselves, read as calibrated probabilities (§T.4). The input classifier is $h$ or a separate MLP trained on $V$. Each of the four cells is run with both pushforwards. The theory makes predictions that E3.1 can check (§T.9). First, the population interval is narrow, because a fixed $\hat C$ carries the prevalence uncertainty into accuracy only through the differences between the recalls (§T.5), and it leaves out the luck of the bag (§T.6); it should under-cover on most pairs. Second, the realised pushforward should widen it by the factor $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$ found by E2.3, which here should be large. Third, unlike in BayCAP, the realised pushforward can also move the centre, because neither quantifier fits $\pi^Q$ to the predictions of $h$, and it should move less when the quantifier reads the outputs of $h$ (§T.8). Fourth, $\hat C$ has no prior, so the pull of the recalls towards $1/K$ that displaces BayCAP on the large-$K$ datasets (E2.2) should be absent. Fifth, BayCAP-EMQ should be worse where its input classifier is poorly calibrated, and its prior, which is not uniform (§T.4), may displace it on the bags with a class prevalence near $0$. The answers tell which quantifier and which input classifier to use for SQS-B (E3.c1) and for a random $C$ (E3.c3, E3.c4 and E3.2, formerly E3.c5), and how these methods compare with those of `main`.

- **Design.**
  - **Inputs:** the $75$ pairs and $100$ bags of `main`; the stored `main` results of every other method (`DoC`, `QuAcc_NxN`, `O-LEAP`, `L-CBPE`, `RQBS`, `PQ`, `BayesCAP` recomputed on CPU, `BayesCAP-real`, and `BayesCAP-Hf-real`, the best BayCAP prior of E2.8 in [[ext2_baycap]], realised only); the validation counts $f_{ij}$ of $h$, rebuilt as in E2.2; the results of eight new methods of `main`, each with its own run: `BayCAP-<Q>-<s>` (population) and `BayCAP-<Q>-<s>-real` (realised) for the quantifier $Q\in\{$`KDEy`, `EMQ`$\}$ and the input classifier $s\in\{$`h`, `mlp`$\}$, e.g. `BayCAP-EMQ-h-real`.
  - **Varied:** three factors, fully crossed ($2\times2\times2=8$ methods).
    - *Quantifier:* `KDEy`, QuaPy's `BayesianKDEy` with the Gaussian kernel, bandwidth $b=0.1$ and shrinkage $\lambda=0$ (QuaPy's defaults; user's decision of 2026-10-28; fixed, not selected, so the results hold for this configuration); or `EMQ`, QuaPy's `BayesianMAPLS` as QuaPy has it (user's decision of 2026-10-28): `prior="uniform"`, whose effective prior on $\pi$ is nonetheless not uniform (§T.4; the alternative is E3.c6), the source prevalence $\pi^P$ taken from the labels of $V$ (`exact_train_prev=True`), and the chain started at the EM estimate (`mapls_chain_init=True`).
    - *Input classifier:* `h`, the posteriors of $h$ on $V$ and on $U$: $h$ was trained on $L$, so its outputs on $V$ are already out of sample and the quantifier is fitted on all of $V$ without cross-fitting; or `mlp`, a separate MLP (`MLPClassifier()` with its default settings, as in the KDEy-based methods of `main`, and a fixed `random_state`) trained on $V$, whose validation outputs are cross-fitted ($5$ folds, QuaPy's default) and whose target outputs come from the MLP trained on all of $V$. On the $15$ pairs where $h$ is itself an MLP, the two levels are two MLPs trained on different data ($L$ and $V$); the contrast between the levels is clearest on the other $60$ pairs.
    - *Pushforward:* population or realised.
    - *Held fixed:* data, bags and seeds; temperature $\tau=1$ (tempering is studied in [[ext5_baycap_priors_tempering|Extension 5]]); NUTS with $500$ warm-up iterations and $1000$ draws, as for `BayesCAP`; within a cell, the same NUTS seed and MLP `random_state` for the population and the realised method, so that on the same platform the two runs give the same prevalence draws.
  - **Measures:** for all eight methods, all three metrics of `main` (vanilla accuracy, macro-F1, macro-K): AE, coverage, amplitude, Winkler score. For vanilla accuracy only:
    - the predicted coverage $\mathcal C_s$ of E2.2, and the split of the shift into recall term $R$ and prevalence term $P$ (E2.2, §T.5 of [[ext2_baycap]]), with prior part $R_0=0$, since $\hat C$ has no prior;
    - for the four realised methods, the widening factor of the realised interval against $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$, with $\bar\sigma_g$ and $\sigma_p$ computed from the draws as in §T.8, and the movement of the centre, the realised point estimate minus the population one;
    - the label-free mismatch $\lVert n^t/N^Q-\hat C\bar\pi\rVert_1$ between the shares of the bag that $h$ predicts as each class and the shares that the posterior-mean prevalence $\bar\pi$ implies (§T.8), and the spread of the recalls of $h$ on $V$, $\operatorname{sd}_j(\hat C_{jj})$ (§T.5);
    - the calibration of each input classifier on $V$: the expected calibration error (ECE) of its top-label confidence, with $15$ equal-width bins, computed on the validation outputs the quantifier is fitted on (cross-fitted for `mlp`);
    - for all eight methods, the coverage in bins of the smallest true class prevalence of the bag, $\min_j\pi^Q_j$, which tests the effect of the prior of §T.4.

    Paired bootstrap tests of the mean Winkler difference, as in E2.3 ($10{,}000$ seeded resamples of the $100$ per-bag differences, Holm-adjusted over the $75$ pairs, separately for each test): in each cell, the realised method against the population one ($4$ tests) against `BayesCAP-real` ($4$ tests) and against `BayesCAP-Hf-real` ($4$ tests); among the realised methods, `h` against `mlp` for each quantifier ($2$ tests), and `KDEy` against `EMQ` for each input classifier ($2$ tests). The tests against `BayesCAP-Hf-real` are reported on all pairs, also those where NUTS is unreliable for it in E2.8, with no flag.
  - **Estimand:** population table for the four methods without `-real`, realised table for the four with `-real`; all scored against the realised table of each bag, as every method of `main`. The split of the shift and the bins of $\min_j\pi^Q_j$ use the bags' true class proportions, so they are analyses, not methods; $n^t$, $\bar\pi$ and the ECE on $V$ use no target labels.
  - **Cost:** eight runs of `main` on `bcuda` with `JAX=cpu`, one per method, each $75\times100=7500$ NUTS runs, plus, per pair and cell, one fit of the quantifier (and of the MLP for `mlp`). A pilot on two datasets times the four cells. For both quantifiers the per-instance terms of the likelihood (the kernel densities of BayesKDEy, the outputs of the input classifier for BayesianMAPLS) are computed once before sampling, so each NUTS step costs about $N^QK$ operations.
- **Representation.**
  - **Comparison table, realised:** one row per pair, grouped by dataset and ordered by $K$: for the four realised methods, `BayesCAP-real`, `BayesCAP-Hf-real` and `RQBS`, coverage, amplitude and Winkler score; the widening factor, observed and predicted; the movement of the centre and the mismatch; the ECE of the two input classifiers; the bootstrap intervals and Holm-adjusted p-values of the Winkler tests. CSV, LaTeX and landscape PDF, like `tables/pushforward.py`.
  - **Comparison table, population:** the same for the four population methods and `BayesCAP`, without the widening factor and the centre.
  - **Summary of the design:** for each metric, a $2\times2$ table per pushforward (quantifier on rows, input classifier on columns) of the mean Winkler rank, mean coverage and mean amplitude over the $75$ pairs, also split by the classifier $h$.
  - **Rank table:** mean Winkler and AE ranks of all methods of `main` with the eight new methods, per metric (`tables/rank.py` with the new methods included).
  - **Decomposition table:** the table of E2.2 (diagnostics, interval, error) for the eight new methods, with `tables/displacement.py --method`.
  - **Scatter, coverage:** coverage of `BayesCAP-real` on the horizontal axis, of each realised method on the vertical axis, one panel per cell and one point per pair, coloured by $K$, with lines at $0.95$ on both axes; a second figure with the population versions against `BayesCAP`; a third with the realised methods against `BayesCAP-Hf-real`.
  - **Scatter, width:** observed against predicted widening factor, one panel per cell, one point per pair, with the line $y=x$, coloured by the mismatch.
  - **Scatter, centre:** movement of the centre against the mismatch, one panel per cell, one point per pair.
  - **Calibration:** coverage of each method against the ECE of its input classifier, one panel per cell, one point per pair, coloured by the classifier $h$; and coverage against the bins of $\min_j\pi^Q_j$, one line per realised method.
- **Analysis.** Run on `bcuda` by the user (`main --classic`, one run per method, JAX on CPU, `cap-ml` `bb20aa5`, branch `wp-8/fixed-c`; the dates and commits of the runs were not recorded here): the eight methods stored for all $75$ pairs and the three metrics, $100$ bags each. The E3.1 tables, the E2.2 decompositions of the eight methods and the figures `coverage`, `coverage_population`, `coverage_hf` and `width` were generated on 2026-11-05 and copied locally with `make pull` (work-plan card WP-8). The figures `centre`, `calibration` and `min_prev` and the rank table with the eight methods (`tables/rank.py --include-e31`, after the fix of 2026-11-05 to its columns) were generated on 2026-11-05; the numbers below are read from the CSVs the figures draw from and from the rank table. Results in `<OUT_DIR>/uncawe/main/` (the eight methods), `<OUT_DIR>/uncawe/tables/baycap_fixed/` (`baycap_fixed.csv`, `baycap_fixed_design.csv`, `baycap_fixed_min_prev.csv`), `<OUT_DIR>/uncawe/tables/displacement/<method>/`, `<OUT_DIR>/uncawe/tables/rank/extended/` and `<OUT_DIR>/uncawe/plots/baycap_fixed/`. The summaries are those of `tables/baycap_fixed.py:analysis_summary`, over the $75$ pairs and in three groups of $25$ ($K=2$, $3\le K\le7$, $K\ge8$); single pairs, means per dataset and per classifier, counts and Spearman correlations are read from the stored CSVs. Vanilla accuracy unless stated. The cells are named `KDEy-h`, `KDEy-mlp`, `EMQ-h`, `EMQ-mlp`; "realised" and "population" are the two pushforwards; "uniform" is `BayesCAP-real`, "Hf" `BayesCAP-Hf-real`. Tests are significant at $0.05$ after the Holm adjustment over the $75$ pairs; "lower" means a lower Winkler score than the reference, i.e. better. Two caveats apply to every count over pairs. The five pairs of a dataset share its $V$ and its test pool, so for any effect that depends on $V$ (the rates $\hat C$, the kernel densities, the MLP trained on $V$) E3.1 has about $15$ independent units, not $75$; and a coverage over $100$ bags has a standard deviation of about $\sqrt{0.95\cdot0.05/100}\approx0.022$ at nominal coverage (if the bags were independent), so differences of a few hundredths on one pair are noise.
  - **NUTS is reliable on every pair.** No pair has a largest split-$\hat R$ above $1.1$ in any cell (largest $1.04$); divergent transitions occur only for `KDEy-h`, on $1$–$2$ bags of $4$ pairs (`shuttle` LR, `wine-quality` LR and SVM, `poker_hand` MLP; at most $0.05$ divergent transitions per bag on average); the median over the pairs of the smallest effective sample size of $\pi$ is $244$–$271$ of $1000$ draws (smallest $32$, `KDEy-h`; $116$–$194$ in the other cells). So, unlike E2.8, no pair is left out.
  - **P1 holds: the population interval under-covers.** Mean coverage $0.33$ (`KDEy-h`), $0.31$ (`KDEy-mlp`), $0.26$ (`EMQ-h`), $0.25$ (`EMQ-mlp`), against $0.60$ for `BayesCAP`; at most $3$ pairs per cell reach $0.90$. The mean amplitude is $0.021$–$0.036$, against $0.118$ for `BayesCAP`. Coverage grows with the spread of the recalls of $h$ on $V$, as §T.5 predicts: Spearman correlation between $\operatorname{sd}_j(\hat C_{jj})$ and coverage $0.74$, $0.69$, $0.57$, $0.60$.
  - **P2 holds: the realised pushforward widens the interval by the predicted factor.** The median over the pairs of the observed widening factor is $3.3$, $3.4$, $3.0$, $3.4$, against a predicted $3.1$, $3.0$, $2.9$, $3.2$; the median ratio of observed to predicted is $1.02$–$1.05$, and it is within $10\%$ on $61$–$69$ of the $75$ pairs. Both factors are medians over the bags of a pair, because $\sigma_p=0$ when the recalls are equal (§T.5). That the width matches its prediction shows that the method computes the sources it represents ($e_1^U$ and $e_3$) as §T.8 says; it says nothing about the sources it leaves out, which decide coverage (P4). The factor is largest with two classes (median $4.6$–$5.7$), where $\sigma_p=\lvert\hat d_1-\hat d_0\rvert\operatorname{sd}(\pi_1)$ is small, and smallest for $3\le K\le7$ ($1.4$–$1.6$). Mean coverage rises from $0.25$–$0.33$ to $0.72$, $0.74$, $0.61$, $0.64$, a gain of $0.36$–$0.42$, against $0.08$ from `BayesCAP` to `BayesCAP-real`; the realised method is lower than the population one on $63$, $70$, $56$, $65$ pairs and higher on $0$, $0$, $7$, $3$.
  - **The centre barely moves, and P8 holds only for KDEy.** The mean absolute movement of the centre is $0.005$, $0.005$, $0.007$, $0.005$ (medians $0.002$–$0.004$), small against the mean realised amplitudes $0.045$–$0.057$; it follows the mismatch (Spearman correlation $0.72$, $0.65$, $0.62$, $0.48$). The mean mismatch is smaller when the quantifier reads $h$ for BayesKDEy ($0.040$ against $0.053$) but not for BayesianMAPLS ($0.074$ against $0.056$).
  - **P3 holds: the displacement of BayCAP on large-$K$ datasets is gone.** On `dry-bean`, `hand_digits`, `letter` and `isolet` the mean coverage of the four realised methods per dataset is $0.91$–$0.92$, $0.86$–$0.88$, $0.91$–$0.95$ and $0.70$–$0.88$, against $0.68$, $0.12$, $0.00$ and $0.00$ for uniform and $0.92$–$0.94$ for Hf. Their recall term is between $-0.004$ and $+0.002$, against $-0.014$, $-0.021$, $-0.075$ and $-0.167$ for uniform, whose prior part has the same size. Over the $25$ pairs with $K\ge8$, `KDEy-mlp` realised has coverage $0.85$ and Winkler score $0.185$ (Hf $0.93$, $0.115$; uniform $0.35$, $1.82$).
  - **P4: the fixed-$\hat C$ methods fail where the error comes from $V$.** On `mammographic` ($\lvert V\rvert=291$, $141$–$150$ instances per class, recalls almost equal, $\operatorname{sd}_j(\hat C_{jj})=0.01$) the coverage of the four realised methods is $0.01$–$0.06$ and that of `RQBS` $0.03$, against $0.62$ for uniform: the recall term is $+0.054$ to $+0.056$ in every cell (mean over the five classifiers; $+0.043$ to $+0.069$ per pair for `KDEy-h`), i.e. the recalls of $h$ measured on $V$ are higher than those on the test pool, and the half-amplitude is about $0.023$, so the shift is about $4.7$ standard deviations of the interval ($0.023/1.96\approx0.012$). The shift is plausible as one draw of $e_2$: a recall near $0.82$ measured on about $145$ instances has standard deviation $\sqrt{0.82\cdot0.18/145}\approx0.032$, and for two classes with $\pi$ uniform on the simplex ($E[\pi_j^2]=1/3$) the recall term $\sum_j\pi_j(\hat d_j-d_j)$ has standard deviation about $\sqrt{2\cdot\frac13\cdot0.032^2}\approx0.026$, so $+0.055$ is about two of them. The five classifiers are measured on the same $V$, so their recall terms are not independent: an easy $V$ makes every recall too high together. So `mammographic` is one realisation of $e_2$ shared by five pairs, not five failures; one split cannot tell it from a systematic difference between $V$ and the test pool; uniform has the same recall term ($+0.052$) but covers because its interval is about twice as wide ($0.10$–$0.11$ against $0.044$–$0.048$), from its posterior on $C$. On `wine-quality` the shift is mostly in the prevalence term, $0.057$–$0.079$ (uniform $0.083$): a bias of the quantifiers, shared by BayesianCC, so not specific to the kernel densities of BayesKDEy; coverage is $0.18$–$0.40$, against $0.72$ (uniform) and $0.82$ (Hf), which cover only because their intervals are wider. On `poker_hand` (smallest validation class of $3$ instances) BayesKDEy covers $0.64$–$0.66$ and BayesianMAPLS $0.31$–$0.37$ (Hf $0.95$). The binary datasets `semeion` and `tictactoe` are also at or below $0.75$ in every cell; on `semeion` the coverage of `KDEy-h` is $0.10$–$0.12$ for LR and MLP and $0.92$–$0.94$ for the other three classifiers. The E2.2 decomposition places the failure of LR and MLP in the recall term ($+0.030$ and $+0.028$, against a half-amplitude of about $0.02$; prevalence term $+0.008$ and $+0.005$), while for kNN, SVM and RF both terms are within $\pm0.006$. `semeion` has a minority class of $56$ validation instances, so its recall has a standard deviation of $0.040$–$0.053$ for a recall between $0.9$ and $0.8$ ($\sqrt{0.9\cdot0.1/56}$ and $\sqrt{0.8\cdot0.2/56}$; the per-class recalls are not in the stored CSVs), its contribution to the recall term has a standard deviation of about $0.023$–$0.031$ (times $\sqrt{E[\pi_j^2]}=\sqrt{1/3}$), and a recall term of $0.03$ is about one to one and a half of them: it is consistent with $e_2$ from the small class, here with a sign that differs between classifiers, unlike `mammographic`. `BayesCAP-real` has the same recall terms ($+0.031$, $+0.030$) and covers $0.97$–$0.99$ with intervals three times as wide; `KDEy-mlp` covers $0.31$ and $0.22$ on the two pairs, with recall terms $+0.017$ and $+0.019$.
  - **P6 holds: BayesianMAPLS depends on the calibration of its input classifier, and BayesKDEy is better.** Mean realised coverage $0.61$ (`EMQ-h`) and $0.64$ (`EMQ-mlp`) against $0.72$ and $0.74$ for BayesKDEy; mean Winkler score $0.73$ and $0.74$ against $0.43$ and $0.39$. BayesKDEy is significantly lower than BayesianMAPLS on $27$ pairs and higher on $10$ with $s=h$, lower on $24$ and higher on $6$ with $s$ an MLP. The evidence for calibration as the cause is the paired contrast: on the same pair and with the same input outputs, the two quantifiers differ only in how they read the outputs, so the dataset does not confound their difference. With $s=h$ the gap in coverage (BayesianMAPLS minus BayesKDEy) is largest for RF ($-0.22$) and kNN ($-0.21$), the classifiers whose outputs on $V$ are the least calibrated (median ECE $0.069$ and $0.047$, against $0.018$ for LR and SVM and $0.031$ for MLP), and it is $+0.05$ for LR, $-0.06$ for SVM, $-0.11$ for MLP. The correlation across pairs is weaker evidence: the Spearman correlation between the ECE of the input classifier and the difference of Winkler scores (BayesianMAPLS minus BayesKDEy) is $0.31$ with $s=h$, and in the `calibration` figure the pairs with low coverage occur at every ECE, also below $0.05$; with $s$ an MLP it is $0.56$, but the MLP is trained on $V$ only, so the five pairs of a dataset share one ECE, the correlation is over $15$ datasets and mixes in their difficulty, and the trend rests on three of them: `connect-4` and `german` (ECE $0.142$, the highest), where `EMQ-mlp` covers $0.24$ and $0.26$ against $0.91$ and $0.79$ for `KDEy-mlp`, and `tictactoe` ($0.106$). The pairs where BayesKDEy itself covers little at low ECE (`mammographic`, `semeion` LR and MLP, `tictactoe`) fail for another reason, the error from $V$ (P4).
  - **P7 does not hold: no sign of the prior of BayesianMAPLS near the edges.** The realised coverage of the `EMQ` methods is highest on the bags with $\min_j\pi^Q_j<0.01$ ($0.74$ and $0.84$, $2255$ bags) and lowest on those with $\min_j\pi^Q_j\ge0.2$ ($0.49$ and $0.46$, $1595$ bags); BayesKDEy, whose prior is uniform, has the same trend ($0.84$ to $0.59$ and $0.63$), and the gap between the two quantifiers is about the same at both ends with $s=h$ ($-0.10$). The bins mix datasets, so the trend follows $K$: $\min_j\pi^Q_j\ge0.2$ is possible only for $K\le5$ (if every class has at least $0.2$, then $0.2K\le1$), so the last bin holds the binary and small-$K$ bags, where the fixed-$\hat C$ methods do worst, and the first bin mostly bags of large-$K$ datasets, where they do best. The informative quantity is the gap between the quantifiers within each bin: with $s=h$ it is $-0.10$ in the first and the last bin and $-0.13$ to $-0.15$ in between, while a prior that pulls the draws away from the edges would make it grow towards $[0,0.01)$. So the data show no effect of the prior, but the test is weak; a version that bins $\min_j\pi^Q_j$ within each dataset would hold $K$ fixed. With $N^Q=1000$ the logistic-normal prior may simply be dominated by the likelihood.
  - **The input classifier matters little.** Between $s=h$ and $s$ an MLP, the mean realised coverage changes by $0.02$ (BayesKDEy) and $0.03$ (BayesianMAPLS); `KDEy-h` is significantly lower than `KDEy-mlp` on $4$ pairs and higher on $13$, `EMQ-h` lower than `EMQ-mlp` on $16$ and higher on $28$; mean AE $0.027$ (`KDEy-h`) and $0.024$ (`KDEy-mlp`). On the classifiers with poorly calibrated outputs the MLP helps BayesianMAPLS (above), not BayesKDEy.
  - **For BayesKDEy, the MLP helps when $h$ is a weak classifier.** The advantage of the realised `KDEy-mlp` over the realised `KDEy-h` is modest and concentrated: no significant difference on $58$ of the $75$ pairs, a lower Winkler score on $49$, a mean difference of Winkler scores $W_h-W_{mlp}$ of $+0.04$ ($0.425$ against $0.386$) but a median between $-0.001$ and $+0.034$ per classifier. By classifier of $h$, the mean difference is $+0.160$ for LR, $+0.058$ for SVM, $+0.013$ for kNN, $+0.010$ for RF and $-0.042$ for MLP (significant pairs, $h$ better / MLP better: $0/3$, $0/2$, $0/2$, $2/1$, $2/5$); by dataset, it comes mostly from `poker_hand` ($+0.49$), `german` ($+0.18$) and `semeion` ($+0.13$), and it reverses on `wine-quality` ($-0.39$). The difference is in the prevalence: the mean $\lVert\hat\pi-\pi^Q\rVert_1$ of the E2.2 decompositions is $0.194$ with $s=h$ and $0.155$ with $s$ an MLP for LR, $0.151$ and $0.148$ for SVM, $0.157$ and $0.151$ for kNN, $0.126$ and $0.143$ for RF, $0.139$ and $0.151$ for MLP. Calibration does not explain it: on the pairs where the MLP helps most its outputs are less calibrated than those of $h$ (ECE on $V$ $0.142$ against $0.066$ on `german` SVM, $0.062$ against $0.031$ on `poker_hand` RF, $0.011$ against $0.008$ on `semeion` LR), and BayesKDEy, which estimates the densities of the outputs within each class from $V$, is unchanged by a one-to-one recalibration of the outputs (up to the fit of the kernel width $b$). Nor does the fixed $\hat C$, which enters only after $\pi$ is drawn and is the same for both input classifiers; the mismatch with the predictions of $h$ is smaller with $s=h$ ($0.040$ against $0.053$), which does not make the intervals better. A likely explanation (new hypothesis, not tested): BayesKDEy identifies $\pi$ better when the densities of the outputs within each class overlap less, i.e. when the input classifier discriminates better; the MLP trained on $V$ (cross-fitted outputs) discriminates better than a linear LR, and no better than $h$ when $h$ is itself a strong non-linear classifier trained on $L$. The hypothesis predicts that $W_h-W_{mlp}$ grows with the difference in discrimination on $V$ between the MLP and $h$, and not with the difference in ECE; checking it needs the cross-fitted outputs of the MLP on $V$, which `main` does not store. It does not explain `wine-quality`, where the prevalence term is large for every quantifier.
  - **P5: the realised BayesKDEy methods are better than `RQBS`, which represents the same terms.** `RQBS` has mean coverage $0.69$, Winkler score $0.45$, AE $0.027$; the realised `KDEy-h` and `KDEy-mlp` have a lower Winkler score than `RQBS` on $55$ pairs each (not tested), the `EMQ` methods on $33$ and $44$. Both kinds fail on `mammographic`, where both use the rates of $V$.
  - **Against BayCAP.** Against uniform, the realised methods are lower on $41$, $45$, $29$, $36$ pairs and higher on $15$, $14$, $29$, $28$: they win on the large-$K$ datasets and lose on the binary ones (mean Winkler score for $K=2$: $0.46$, $0.38$, $0.67$, $0.90$ against $0.28$). Against Hf they are lower on $21$, $24$, $11$, $17$ pairs and higher on $19$, $15$, $33$, $32$. `KDEy-mlp` realised has a lower median Winkler score than Hf ($0.086$ against $0.120$) and a higher mean ($0.386$ against $0.233$), in every group of $K$ ($0.38$, $0.59$, $0.19$ against $0.28$, $0.31$, $0.12$): its intervals are narrower (mean amplitude $0.053$ against $0.126$), which pays on most pairs and costs a lot on the pairs of the previous points; mean coverage $0.74$ against $0.89$, mean AE $0.024$ against $0.035$.
  - **The ranks agree, and show where the means mislead.** The rank table ranks $15$ methods on each of the $15$ datasets of a classifier (the five methods of the paper, `BayesCAP`, Hf and the eight of E3.1; rank $1$ is the best, by the mean score over the bags of the dataset) and averages the ranks over the datasets; a dagger marks a method whose ranks are not significantly different from those of the best method of the row (two-sided Wilcoxon test over the $15$ datasets, $0.05$, not adjusted). Ranks are relative, so dropping the population methods would lower every mean rank but not change the order of the others on a dataset. For the Winkler score of vanilla accuracy, the mean ranks over the five classifiers are $3.81$ (`KDEy-mlp` realised; best for LR $3.47$, SVM $3.67$, tied with Hf for MLP $4.07$), $4.48$ (Hf), $4.55$ (`KDEy-h` realised; best for kNN $3.87$ and RF $3.53$), $5.71$ (`RQBS`), $5.73$ and $7.01$ (`EMQ-mlp`, `EMQ-h` realised), $8.03$ (`BayesCAP`), and $10.0$–$10.9$ for the four population methods. Over the $15$ rows of Winkler score (five classifiers, three metrics), `KDEy-mlp` realised is the best on $10$ (one tie) and never significantly worse than the best; `KDEy-h` realised is the best on $3$ and significantly worse than the best only for LR (vanilla accuracy $5.67$, macro-F1 $5.07$), the classifier where the MLP input helps (above); Hf is the best on $2$ and never significantly worse than the best; `EMQ-h` realised is the best on $1$ (macro-K, SVM); `RQBS` is within the dagger on $3$. So Hf and the realised BayCAP-KDEy cannot be separated by the ranks, while the mean Winkler score favours Hf ($0.23$ against $0.39$): the realised BayCAP-KDEy wins on most datasets by a little and loses on a few by a lot (P4). For AE the realised BayCAP-KDEy is the best on all $15$ rows (`KDEy-mlp` on $10$, `KDEy-h` on $5$; vanilla accuracy: mean ranks $4.16$ and $5.09$, against $7.20$ for `RQBS`, $7.49$ for `O-LEAP` and $8.49$ for Hf); its population methods rank lower ($7.1$–$8.9$ for vanilla accuracy), since the realised point estimate, the mean of the realised draws, moves slightly towards the predictions of $h$ (§T.8), which lowers the mean AE of BayesKDEy from $0.030$ to $0.027$ (`KDEy-h`) and from $0.028$ to $0.024$ (`KDEy-mlp`), but not that of BayesianMAPLS ($0.034$ and $0.033$ in both pushforwards).
  - **BayCAP-KDEy with the realised pushforward is the best method of E3.1.** The realised pushforward is better than the population one in every cell (P2), without exception for BayesKDEy ($0$ pairs higher) and on most pairs for BayesianMAPLS ($7$ and $3$ pairs higher); so, by the user's decision of 2026-11-05, only the realised methods are considered from now on. Among the four realised methods BayesKDEy is better than BayesianMAPLS with either input classifier (P6): mean coverage $0.72$–$0.74$ against $0.61$–$0.64$, mean Winkler score $0.43$ and $0.39$ against $0.73$ and $0.74$, and significantly lower on $27$ and $24$ pairs against $10$ and $6$ higher. Between its two input classifiers the difference is small, in favour of the MLP on average and of $h$ when $h$ is a strong classifier (previous points). Against the methods of the paper, the realised BayCAP-KDEy is lower than `RQBS` on $55$ pairs, and lower than uniform on $41$–$45$ pairs and higher on $14$–$15$; against Hf it is lower on $21$–$24$ pairs and higher on $15$–$19$, with a lower median Winkler score ($0.086$ for `KDEy-mlp`, against $0.120$) but a higher mean ($0.39$ against $0.23$) and a lower coverage ($0.74$ against $0.89$), because its intervals are much narrower ($0.053$ against $0.126$) and fail badly where the error comes from $V$ (P4). Its point estimate is also at least as accurate as those of Hf and `RQBS` (mean AE $0.024$–$0.027$, against $0.035$ and $0.027$), and it has the best AE rank of all methods for every classifier and metric. By rank it is the best method on the Winkler score too, level with Hf; by the mean Winkler score Hf stays ahead.
  - **The other metrics agree.** For macro-F1 the mean realised coverage is $0.75$, $0.77$, $0.64$, $0.65$ (population $0.26$–$0.42$). For macro-K the population interval is almost a point (mean amplitude $0.003$–$0.006$, coverage $0.04$–$0.08$) and the realised one covers $0.81$, $0.81$, $0.74$, $0.75$.

  What this shows: with a fixed $\hat C$, the theory of §T.5–§T.8 describes the intervals closely: the population interval is narrow where the recalls are similar, the realised pushforward widens it by $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$, the centre stays put, and without a prior on $C$ the displacement of BayCAP on the large-$K$ datasets disappears. The realised methods represent only $e_1^U$ and $e_3$, so they cover well where $V$ is large and the quantifier is right ($K\ge8$) and fail where the error comes from $V$: the rates of $h$ on `mammographic` (one realisation of $e_2$ shared by its five pairs) and on `semeion` LR and MLP (a small class on $V$), the quantifier on `poker_hand` ($e_2$ and $e_1^V$, with $3$ validation instances in the smallest class), and a bias of every quantifier on `wine-quality`, shared by BayesianCC. Of the two quantifiers BayesKDEy is the better one, and BayesianMAPLS is hurt by miscalibrated inputs, as P6 predicts (shown by the paired contrast, more weakly by the correlation across pairs); its prior is not visible, in a test confounded with $K$ (P7). The input classifier matters little for BayesKDEy: an MLP trained on $V$ helps when $h$ is a weak classifier (LR), probably because its outputs discriminate the classes better (new hypothesis), not because of calibration or of the fixed $\hat C$. All in all, BayCAP-KDEy with the realised pushforward is the best of the eight methods: narrower intervals than BayCAP, a better Winkler score on most pairs, the best mean Winkler rank (level with Hf by the Wilcoxon tests) and the best AE rank, but a lower coverage and a higher mean Winkler score than Hf, since it leaves out the error that comes from $V$; only the realised methods are considered from now on. For the candidates: the quantifier of SQS-B (E3.c1) can be BayesKDEy, with either input classifier; `mammographic` and `semeion` are the test cases of a random $C$ at the bridge (E3.c3) and `poker_hand` of the error of the kernel densities (E3.c4); on `wine-quality` a random $C$ or random kernel densities could only widen the interval, as Hf does, not remove the bias shared by all quantifiers; the results do not motivate a uniform prior for BayesianMAPLS (E3.c6), while its sensitivity to calibration is what a random calibration map (E3.c7) would represent.

## E3.2: BayCAP-KDEy with a random $C$: the joint posterior of $(\pi,C)$ with a kernel likelihood and the Hf prior

**Why.** E3.1 found that BayCAP-KDEy with the realised pushforward is the best of its eight methods, but that it fails where the error comes from $V$: it holds both the rate matrix $\hat C$ and the kernel densities fixed, so it leaves out the error of the rates of $h$ measured on $V$ ($e_2$; `mammographic`, `semeion` LR and MLP) and part of the error of the quantifier fitted on $V$ ($e_1^V$; `poker_hand`). `BayesCAP-Hf-real` ([[ext2_baycap]] E2.8) represents both through its joint posterior of $(\pi,C)$, and covers more ($0.89$ against $0.74$ on average) with wider intervals. E3.2 joins the two. It splits each class of $V$ into cells by the prediction of $h$ and estimates a kernel density in each cell; the class density of BayesKDEy is then the mixture of the cell densities, with the rates of $h$ as weights (§T.10). It makes these weights, i.e. the columns of $C$, random, with the Hf prior of Extension 2 and the validation counts as in BayCAP (§T.11), and samples them with $\pi$ by NUTS, so that the target bag updates $C$ as in BayCAP, but through the outputs of the input classifier instead of the counts of the predictions alone (§T.13). It is the reduced form of the full joint posterior of the prevalence and of one weight per validation instance (option 3 of Appendix B, the former candidate E3.c5): the cells keep the unknowns of BayCAP, whatever $\lvert V\rvert$. Each new method differs from an existing one in one component: from the realised BayCAP-KDEy of E3.1 with the same input classifier, in the rate matrix (random, with the Hf prior and updated by the bag, instead of fixed at $\hat C$); from `BayesCAP-Hf-real`, in the likelihood of the target bag (kernel densities of the outputs within the cells instead of the counts of the predictions).

- **Design.**
  - **Inputs:** the $75$ pairs and $100$ bags of `main`; the stored `main` results of `BayCAP-KDEy-h-real` and `BayCAP-KDEy-mlp-real` (E3.1), `BayesCAP-Hf-real`, `BayesCAP-real`, `RQBS` and the other methods of `main`; the validation counts $f_{ij}$ of $h$, rebuilt as in E2.2; the results of two new methods of `main`, each with its own run: `BayCAP-KDEy-Hf-h-real` and `BayCAP-KDEy-Hf-mlp-real`.
  - **Varied:** the input classifier, together with the form of the likelihood it requires (§T.10): `h`, the outputs of $h$, with the likelihood of the outputs alone, since the prediction of $h$ is a function of its outputs; or `mlp`, the MLP of E3.1, with the joint likelihood of the outputs of the MLP and the predictions of $h$ (user's decision of 2026-11-06). *Held fixed:* the realised pushforward (user's decision of 2026-11-05); the Hf prior on $C$ with centred shares, as in `BayesCAP-Hf-real` (§T.11; user's decision of 2026-11-06); the kernel of E3.1 (Gaussian, $b=0.1$, no shrinkage); the convention for empty cells of §T.12 (decision of 2026-11-06); $\pi\sim\operatorname{Dir}(\mathbf 1)$; temperature $\tau=1$; NUTS with $500$ warm-up iterations and $1000$ draws and the seed of `BayesCAP`; the MLP and its `random_state` as in E3.1; data and bags.
  - **Measures:** for both methods, all three metrics of `main` (vanilla accuracy, macro-F1, macro-K): AE, coverage, amplitude, Winkler score. For vanilla accuracy only:
    - the split of the shift into recall term $R$ and prevalence term $P$ (E2.2 of [[ext2_baycap]]);
    - paired bootstrap tests of the mean Winkler difference, as in E3.1 ($10{,}000$ seeded resamples of the $100$ per-bag differences, Holm-adjusted over the $75$ pairs, separately for each test), against the realised BayCAP-KDEy of E3.1 with the same input classifier and against `BayesCAP-Hf-real`;
    - the standardised shift of each pair, the shift divided by the standard deviation of the accuracy samples, averaged over the bags: since all bags of a pair share one $V$, the error from $V$ is one draw per pair, and an interval that represents it should leave standardised shifts that look like draws of a standard normal across pairs (§T.13);
    - the NUTS diagnostics of `main` (divergent transitions, largest split-$\hat R$, smallest effective sample size) on $\pi$, $C$ and the Hf hyperparameters, and the Hf posterior summaries ($\ln\nu$, $\ln\kappa$, $P(\kappa<1)$);
    - label-free, per cell $(k,j)$: the posterior standard deviation of $C_{kj}$ divided by the standard deviation that $V$ alone gives it, $\sqrt{\hat C_{kj}(1-\hat C_{kj})/(n^s_j+1)}$, against the count $f_{kj}$: a ratio below $1$ says that the bag adds information on that cell (§T.13); the mismatch $\lVert n^t/N^Q-\bar C\bar\pi\rVert_1$ of §T.8 with the posterior means $\bar C$ and $\bar\pi$; and, per pair, the number of empty cells and of cells with a borrowed density (§T.12).
  - **Estimand:** realised table for both methods, scored against the realised table of each bag (`true_cts`), as every method of `main`. The split of the shift and the standardised shift use the bags' true tables, so they are analyses, not methods; the diagnostics, the ratios of standard deviations, the mismatch and the counts of empty cells use no target labels.
  - **Cost:** two runs of `main` on `bcuda` with `JAX=cpu`, one per method, each $75\times100=7500$ NUTS runs on $K-1+K(K-1)$ unknowns plus the Hf hyperparameters (the size of `BayesCAP-Hf-real`), plus, per bag, the kernel values of the $N^Q$ target outputs at the $\lvert V\rvert$ validation outputs, and, per pair, the fit of the cell densities (and of the MLP for `mlp`). A pilot on `mammographic`, `poker_hand` and `wine-quality` (user's decision of 2026-11-06), the test cases of $e_2$, of $e_1^V$ with the sparsest validation table, and of a bias shared by all quantifiers, precedes the full run.
- **Representation.**
  - **Comparison table:** one row per pair, grouped by dataset and ordered by $K$: coverage, amplitude and Winkler score of the two new methods, of `BayCAP-KDEy-h-real`, `BayCAP-KDEy-mlp-real`, `BayesCAP-Hf-real` and `RQBS`; for each new method, the $95\%$ bootstrap intervals of the mean Winkler differences against its two references and their Holm-adjusted p-values; the standardised shift.
  - **Rank table:** mean Winkler and AE ranks of all methods of `main` with the two new methods, per metric (`tables/rank.py`).
  - **Decomposition table:** the table of E2.2 for the two new methods (`tables/displacement.py --method`).
  - **Scatter, coverage:** coverage of the E3.1 method with the same input classifier on the horizontal axis and of the new method on the vertical axis, one panel per input classifier; the same with `BayesCAP-Hf-real` on the horizontal axis; one point per pair, coloured by $K$, with lines at $0.95$.
  - **Scatter, information on $C$:** the ratio of standard deviations of each cell against $f_{kj}$ (logarithmic axis), one panel per input classifier, one point per cell and pair, with the line $y=1$.
  - **Diagnostics table:** per pair and method, the NUTS diagnostics and the Hf summaries, with the number of empty and borrowed cells.
- **Analysis.** Run on the $75$ pairs: pilot on 2026-11-06 (from `b9fe044`), full run on 2026-11-06 and 2026-11-07 (`BayCAP-KDEy-Hf-mlp-real` from `70b2140`, `BayCAP-KDEy-Hf-h-real` from `847e508`), on `bcuda` with `JAX=cpu`; tables and plots generated on 2026-11-09 from `a215a14`. The scores of vanilla accuracy come from the per-pair CSV of E3.2 (`<OUT_DIR>/uncawe/tables/baycap_kdey_hf/baycap_kdey_hf.csv`) and its per-cell CSV, those of macro-F1 and macro-K from the tables of `main` generated on 2026-11-09. The ranks among the six methods below, the head-to-head counts, the sign and Wilcoxon tests and the split into two groups of datasets were computed from the per-pair CSV, outside the analysis script. The rank table with `--include-e32` and the decomposition table were not generated; the split of the shift into $R$ and $P$ is in the per-pair CSV. Every interval targets the realised table and is scored against `true_cts`. Short names: `Hf-h` and `Hf-mlp` for the two methods of E3.2, `KDEy-h` and `KDEy-mlp` for the realised BayCAP-KDEy of E3.1, Hf for `BayesCAP-Hf-real`.
  - **Overall, E3.2 does not improve on E3.1.** Mean AE $0.029$ (`Hf-h`) and $0.027$ (`Hf-mlp`), against $0.027$ (`KDEy-h`), $0.024$ (`KDEy-mlp`), $0.035$ (Hf) and $0.027$ (`RQBS`); mean coverage $0.79$ and $0.77$, against $0.72$, $0.74$, $0.89$, $0.69$; mean amplitude $0.068$ and $0.061$, against $0.057$, $0.053$, $0.126$, $0.054$; mean Winkler score $0.41$ and $0.40$, against $0.43$, $0.39$, $0.23$, $0.45$, and median $0.082$ and $0.105$, against $0.088$, $0.085$, $0.120$, $0.113$. Ranked among these six methods on each pair (rank $1$ the best; coverage by its distance from $0.95$) and averaged over the $75$ pairs, the AE ranks are $3.13$, $2.89$, $3.29$, $2.73$, $4.72$, $4.23$, the amplitude ranks $4.60$, $3.75$, $2.24$, $1.88$, $5.99$, $2.55$, the coverage ranks $3.35$, $3.73$, $3.67$, $3.43$, $2.41$, $4.41$, and the Winkler ranks $3.17$, $2.81$, $3.63$, $3.08$, $3.88$, $4.43$. In the paired bootstrap tests (Holm-adjusted, $0.05$), `Hf-h` has a lower Winkler score than `KDEy-h` on $16$ pairs and a higher one on $13$; `Hf-mlp` than `KDEy-mlp` on $10$ and $15$; `Hf-h` than Hf on $16$ and $14$; `Hf-mlp` than Hf on $16$ and $18$.
  - **The better Winkler rank of `Hf-mlp` weighs little.** Head to head, `Hf-mlp` has a lower Winkler score than `KDEy-mlp` on $44$ pairs and a higher one on $31$, but the median difference is $-0.002$, the mean $+0.015$, and the Wilcoxon signed-rank test over the pairs gives $p=0.91$. By dataset (mean difference over its five pairs) it is lower on $10$ of the $15$ datasets (sign test $p=0.30$, about the number of independent units, since the pairs of a dataset share $V$ and the test pool); on $9$ of the $10$ the gain is between $0.002$ and $0.071$, and only on `mammographic` is it large ($-0.61$). The losses are large: `shuttle` $+0.63$ (median relative increase $+420\%$), `german` $+0.25$, `semeion` $+0.08$. A mean rank counts how often a method is ahead, not by how much, and it depends on the other methods ranked; so it reflects many very small wins. On the other measures the two are level: AE lower on $39$ pairs and higher on $36$, coverage closer to $0.95$ on $35$ and farther on $39$.
  - **Two regimes: BayCAP-KDEy where $V$ represents the target, Hf where the error comes from $V$.** On the four datasets where the error comes from $V$ or from a bias of the quantifiers (`mammographic`, `wine-quality`, `poker_hand`, `german`; $20$ pairs), the mean Winkler score is $1.28$ (`Hf-h`), $1.10$ (`Hf-mlp`), $1.28$ (`KDEy-h`), $1.20$ (`KDEy-mlp`), $0.58$ (Hf), $1.29$ (`RQBS`), and the mean coverage $0.46$, $0.49$, $0.42$, $0.41$, $0.75$, $0.36$: Hf is the only method that comes near covering, with intervals twice as wide ($0.224$ against $0.087$–$0.114$). On the other eleven datasets ($55$ pairs) the mean Winkler score is $0.097$, $0.145$, $0.114$, $0.089$, $0.108$, $0.140$, the mean coverage $0.91$, $0.87$, $0.84$, $0.86$, $0.94$, $0.81$, and the mean amplitude $0.051$, $0.046$, $0.043$, $0.041$, $0.090$, $0.043$; there Hf has the worst Winkler rank of the six ($4.49$), and `KDEy-mlp` the lowest mean Winkler score and amplitude. So the lead of Hf in mean Winkler score ($0.23$ against $0.39$) comes entirely from the four datasets, and its median Winkler score is one of the highest.
  - **P9 holds in part: the standardised shift falls where the error is $e_2$, but not to a standard normal.** On `mammographic` it falls from $3.5$–$6.2$ (`KDEy-h`) and $3.3$–$6.0$ (`KDEy-mlp`) to $0.6$–$3.2$ (`Hf-h`) and $1.6$–$2.9$ (`Hf-mlp`), and the mean coverage rises from $0.03$ and $0.02$ to $0.28$ and $0.38$ (Hf $0.56$, standardised shift $1.2$–$2.9$); the recall term stays ($+0.049$ and $+0.046$, against $+0.054$ to $+0.056$ in E3.1), but the intervals are wider ($0.074$–$0.078$ against $0.044$–$0.048$). On `semeion` LR and MLP it falls from $3.6$ (`KDEy-h`) and $2.1$, $2.6$ (`KDEy-mlp`) to $2.8$, $1.1$ (`Hf-h`) and $1.1$, $-1.2$ (`Hf-mlp`), with coverage $0.22$, $0.70$ and $0.57$, $0.69$, against $0.12$, $0.10$ and $0.31$, $0.22$ (Hf $0.91$ on both). Across the $75$ pairs, the standard deviation of the standardised shifts is $1.65$ (`Hf-h`) and $1.38$ (`Hf-mlp`), against $1.77$ and $1.73$ in E3.1 and $0.96$ for Hf, and the share of pairs with $\lvert z\rvert>2$ is $0.16$ and $0.20$, against $0.21$, $0.19$ and $0.09$: closer to draws of a standard normal than E3.1, far less than Hf.
  - **P10 does not hold: `poker_hand` does not improve.** Mean coverage $0.57$ (`Hf-h`) and $0.65$ (`Hf-mlp`), against $0.64$ and $0.66$ in E3.1 and $0.95$ for Hf; mean Winkler score $1.65$ and $0.66$, against $1.20$ and $0.71$ ($0.32$ for Hf). The posterior of $C$ is that of $V$: the median ratio of standard deviations of §T.13 is $0.97$ and $0.98$.
  - **P11 holds for `mlp`, not for `h`.** On `wine-quality` `Hf-mlp` is unchanged (coverage $0.18$, Winkler score $2.59$, against $0.18$ and $2.57$ for `KDEy-mlp`), with a prevalence term of $+0.087$; `Hf-h` improves (coverage $0.55$ against $0.34$, Winkler score $1.24$ against $2.18$, prevalence term $+0.037$), but stays below Hf ($0.82$, $0.88$).
  - **P12 does not hold.** The intervals of E3.2 are narrower than those of Hf (median ratio of amplitudes over the pairs $0.71$ for `h`, $0.67$ for `mlp`), but not at a similar coverage ($0.79$ and $0.77$ against $0.89$).
  - **P13: the opposite pattern.** The median ratio of the posterior standard deviation of $C_{kj}$ to that from $V$ alone is $0.83$ and $0.81$ (`h`, `mlp`) on the cells with $1\le f_{kj}\le10$, $0.94$ and $0.92$ with $11$–$100$, $0.96$ and $0.94$ with $101$–$1000$, $0.99$ and $0.98$ above $1000$: the small cells narrow most, the large ones hardly. The denominator has no prior, so on the small cells the ratio mixes the information of the bag with the pooling of the Hf prior, which also narrows them; on the large cells, where the prior weighs little, the bag adds almost no information on $C$. By dataset the median ratio is lowest on `semeion` ($0.51$), `tictactoe` ($0.68$, $0.52$), `mammographic` ($0.77$, $0.70$), `hand_digits` and `isolet` ($0.74$–$0.77$), and near $1$ on `wine-quality`, `poker_hand`, `chess`, `connect-4` and `nursery` ($0.93$–$0.98$). With `mlp` the ratio is a little lower than with `h`, as predicted.
  - **P14 holds.** The mismatch $\lVert n^t/N^Q-\bar C\bar\pi\rVert_1$ has mean $0.027$ (`h`) and $0.025$ (`mlp`) and median $0.019$ and $0.018$, below $1/\sqrt{N^Q}\approx0.032$, against means $0.040$ and $0.053$ in E3.1; it is smaller than in E3.1 on $68$ (`h`) and $75$ (`mlp`) of the $75$ pairs.
  - **New failures, not explained.** `Hf-mlp` fails on `shuttle` (coverage $0.39$–$0.76$ against $0.73$–$0.94$ for `KDEy-mlp`, standardised shift $1.3$–$3.2$, recall term $+0.021$ and prevalence term $+0.025$), where `Hf-h` covers $0.55$–$1.00$; `Hf-h` fails on `german` (coverage $0.24$–$0.58$ against $0.34$–$0.86$ for `KDEy-h`, standardised shift between $-5.6$ and $+2.9$), where `Hf-mlp` covers $0.42$–$0.95$. Neither pair has borrowed class densities, and the divergences there are few ($0$–$11$ per bag). The two variants fail on different datasets, which the analysis does not explain.
  - **The sampler.** No bag failed. Some bag has divergent transitions on $59$ (`h`) and $58$ (`mlp`) pairs; the median over the pairs of the mean number per bag is $0.8$ and $0.6$, but it is about $430$ on `poker_hand` ($K=8$), $28$–$30$ on `wine-quality` ($K=5$) and $43$–$50$ with $K=26$; the largest split-$\hat R$ is above $1.05$ on $23$ pairs for each method. On $47$ of the $50$ pairs with $K\ge3$ the posterior puts most of its mass on $\kappa<1$, so the shares of the empty cells have Dirichlet parameters below $1$ (§T.11, §T.12). The divergences on `poker_hand` are the same with `h` and `mlp`, so they come from the part of the model the two share (the prior and the empty cells), not from the likelihood of the outputs alone. The scores of the pairs with many divergences are less reliable.
  - **The other metrics agree.** Mean Winkler score for macro-F1: $0.41$ (`Hf-h`), $0.35$ (`Hf-mlp`), against $0.38$, $0.33$ in E3.1, $0.27$ for Hf; mean AE $0.034$, $0.030$ against $0.029$, $0.027$ ($0.040$ for Hf). For macro-K, E3.2 is better than E3.1 (mean Winkler score $0.29$ and $0.30$ against $0.37$ and $0.38$; Hf $0.22$), at about the same AE ($0.033$, $0.032$ against $0.030$, $0.030$).

  What this shows: a random $C$, updated by the bag through the cell densities, helps where the error is in the rates of $h$ measured on a validation set of moderate size with two classes (`mammographic`, `semeion`), and it makes the predicted shares match those of the bag (P14), but it does not improve BayCAP-KDEy overall: on most datasets the gains are negligible, on `poker_hand` and on `wine-quality` with `mlp` there is no change, and there are new failures (`shuttle` with `mlp`, `german` with `h`). A likely explanation (new hypothesis, not tested): the cell densities $\hat\phi_{kj}$ stay fixed at their estimates on $V$, and within each cell they still tell the classes apart, so they identify $\pi$ largely whatever $C$ is; a random $C$ then adds little width (the ratio of standard deviations is near $1$ on the large cells), and the error of the shapes of the cell densities (the rest of $e_1^V$, E3.c4) and any bias of the quantifier stay out. Hf, whose bag reveals only the predicted shares $q=C\pi$, identifies $\pi$ through $C^{-1}q$, so all its uncertainty on $C$ reaches $\pi$; this is why it covers where the error comes from $V$, at twice the amplitude. With two classes the cells are the two sides of the threshold of $h$, so $C$ sets a coarse shape of the class densities that the bag can correct, which explains the gain on `mammographic`; there the bag has $N^Q=1000$ outputs against about $145$ validation instances per class, and the likelihood treats the cell densities as exact, which may explain why the standardised shift stays above $2$ (new hypothesis; a temperature $\tau>1$ or smaller bags would test it). By the user's decision of 2026-11-09, the methods of E3.2 are not carried forward: `BayCAP-KDEy-mlp-real` of E3.1 stays the BayCAP-KDEy method (most accurate centre, narrowest intervals, best on most pairs), and `BayesCAP-Hf-real` stays the reference for coverage and for the mean Winkler score, where the error comes from $V$; the two have different meanings, and a method that blends them is candidate E3.c8.

# Candidate experiments

Ideas that came up in the discussions of 2026-10-27, 2026-10-28 and 2026-11-09. They are not scheduled and are numbered E3.c<n>; one becomes an experiment of the main section only when the user decides so, and then it gets the next number E3.<n> and a full specification. E3.c5, the full joint posterior of prevalence and weights for BayesKDEy, became E3.2 on 2026-11-06, in the reduced form of §T.10.

- **E3.c1, SQS-B on a Bayesian quantifier.** *Why:* SQS-B draws one validation bag of size $\lvert U\rvert$ for each joint posterior draw $\pi^{(r)}$, with class counts $\approx\lvert U\rvert\pi^{(r)}_j$, and applies the metric to the table of $h$ on that bag. Against RQBS it changes only the source of the prevalences (posterior instead of bootstrap of $U$); against the realised BayCAP method of E3.1 with the same quantifier it changes only the bridge from prevalence to table: a bag simulated forwards from $V$, from true classes to predictions, instead of true classes sampled backwards given the observed predictions of $h$ on $U$. Given $\pi^{(r)}$, the accuracy of an SQS-B bag has mean $\hat d^\top\pi^{(r)}$ and the binomial variance of §T.6 computed with the recalls on $V$, so the budget predicts that SQS-B and the realised methods of E3.1 represent the same sources of error (§T.9). *Needs:* an SQS-B method with the conventions of `RQBS` in `main` (surrogate quantifier, use of $V$, rounding of the class counts, sampling with replacement), to be read from `cap-ml` before implementation; the quantifier and input classifier of one cell of E3.1, chosen from its results; its own NUTS runs.
- **E3.c2, BayCAP with a fixed $\hat C$ and the posterior of BayesianCC.** *Why:* it pushes the marginal draws $\pi^{(r)}$ of BayCAP's posterior through $\hat C$, in both pushforwards. Compared with the methods of E3.1 it changes only the posterior of $\pi^Q$ (BayesianCC on the predictions of $h$ instead of BayesKDEy or BayesianMAPLS on soft outputs); compared with `BayesCAP` on the same draws it changes only whether $C$ is random, so it measures how much the uncertainty on $C$ and the pull of its prior towards $1/K$ act at the bridge, separately from their effect on the posterior of $\pi^Q$. *Needs:* a pushforward option in `methods/baycap.py:BayCAP` that uses $\hat C$ in place of $C^{(r)}$; its own rerun of BayCAP.
- **E3.c3, a random $C$ at the bridge (option 1 of Appendix B).** *Why:* it pairs each draw $\pi^{(r)}$ with an independent draw $C^{(r)}_{\cdot j}\sim\operatorname{Dir}(f_{1j},\dots,f_{Kj})$ of each column of the rate matrix, which adds the error of $C$ estimated on $V$ (the term $e_2$ of Extension 4) without a new posterior. It is BayCAP's posterior on $C$ from $V$ alone, but with the prior $\operatorname{Dir}(\mathbf 0)$ instead of $\operatorname{Dir}(\mathbf 1)$, so it has no pull towards $1/K$. It applies to either quantifier. *Decision rule:* if it brings coverage near nominal on the pairs where the realised methods of E3.1 under-cover, the random $C$ at the bridge is enough; if the displacement remains, the missing error of the quantifier from $V$ (term $e_1^V$) matters, and E3.c4 (for BayesKDEy) or E3.c7 (for BayesianMAPLS) is the next candidate. *Needs:* a third pushforward in the class of E3.1; its own run; a convention for cells with $f_{ij}=0$, which stay at $0$ in every draw.
- **E3.c4, cut posterior with random kernel densities for BayesKDEy (option 2 of Appendix B)** [new hypothesis]. *Why:* random weights on the validation instances make both the kernel densities of BayesKDEy and $C$ random, coherently, and add the term $e_1^V$. *Needs:* $M$ NUTS runs per bag, one per draw of the weights, so a cost about $M$ times that of one method of E3.1; a choice of $M$.
- **E3.c6, BayesianMAPLS with a uniform prior** (Appendix E). *Why:* the NUTS model of QuaPy's `BayesianMAPLS` gives $\pi$ a logistic-normal prior, whose density goes to $0$ at the edges of the simplex (§T.4), while BayesKDEy uses the uniform prior $\operatorname{Dir}(\mathbf 1)$, which matches the UPP bags of `main`. So the quantifier factor of E3.1 mixes two differences, the likelihood and the prior. Rerunning the `EMQ` cells with a uniform prior separates them; it is worthwhile if E3.1 finds the `EMQ` methods displaced on the bags with a small $\min_j\pi^Q_j$. *Needs:* a NUTS model that samples $\pi$ from the Dirichlet directly, or that keeps the ILR coordinates and replaces the normal factor with the log-Jacobian of the transformation (Appendix E); its own runs of the `EMQ` methods.
- **E3.c7, BayesianMAPLS with a random calibration map** (Appendix E) [new hypothesis]. *Why:* BayesianMAPLS has no kernel densities or validation counts to randomise: its model of the classes is the calibration of the input classifier (§T.4). A small parametric recalibration of the outputs, one temperature and $K$ class biases (bias-corrected temperature scaling, Alexandari et al. 2020), with parameters $\theta$, can be sampled jointly with $\pi$: the labels of $V$ give a likelihood on $\theta$, and the bag gives the EM likelihood on $(\pi,\theta)$. It is the analogue of BayesianCC's joint posterior of $(\pi,C)$, with $K+1$ extra parameters instead of the $\lvert V\rvert$ of the full joint posterior of Appendix B (option 3), so it would add the calibration part of $e_1^V$ cheaply and coherently. The rate matrix of $h$ at the bridge stays a separate object (Appendix E). *Needs:* a NumPyro model of its own, a prior on $\theta$, its own runs; to be considered if E3.1 finds the `EMQ` methods sensitive to the calibration of their input classifier.
- **E3.c8, a blend of BayCAP-KDEy and BayCAP-Hf** [new hypothesis]. *Why:* E3.1 and E3.2 found two regimes. Where $V$ represents the target, `BayCAP-KDEy-mlp-real` has the most accurate centre and the narrowest intervals that still cover; where the error comes from $V$ or from a bias of the quantifiers (`mammographic`, `wine-quality`, `poker_hand`, `german`, `semeion` LR and MLP), only `BayesCAP-Hf-real` comes near covering, because it identifies $\pi$ through $C^{-1}$ and so carries its uncertainty on $C$ into $\pi$, at twice the amplitude. A method that switches between the two, or mixes their draws, according to a label-free signal that $V$ is unreliable could keep the narrow intervals of the first on most pairs and the coverage of the second on the hard ones. *Needs:* a label-free signal that predicts the hard pairs, for example the smallest validation class $\min_jn^s_j$, the mismatch of §T.8, or the disagreement between the two posteriors (the distance between their centres relative to their widths), checked first on the stored results of `main`, with no new runs; a form of the blend (a switch, a mixture of the two sets of draws with a weight given by the signal, or the union of the two intervals); no new NUTS runs if the per-bag samples of the two methods are stored, otherwise reruns of both.

# Theory

This part explains what E3.1 and E3.2 need, step by step; §T.10–§T.13 are for E3.2. We follow the notation of the reference paper. There are $K$ classes $\omega_1,\dots,\omega_K$. $h$ is the classifier whose performance we estimate. $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that $h$ predicts $\omega_i$ for an instance of true class $\omega_j$, so column $j$ of $C$ sums to one. $\pi^Q$ is the vector of class prevalences of the target bag $U$, which has $N^Q=\lvert U\rvert$ instances. $V$ is the labelled validation set, with $n^s_j$ instances of class $\omega_j$. A contingency table $T$ has predicted classes on rows and true classes on columns. $s$ is the input classifier of the quantifier, $h$ itself or a separate MLP, and $z=s(x)$ its output probabilities, a point of the simplex $\Delta^{K-1}$ (a vector of $K$ non-negative numbers summing to one).

## T.1 Prior probability shift and the population table

*Intuition:* under prior probability shift only the mix of classes changes; how $h$ behaves on instances of one class does not, so a rate matrix measured on $V$ is valid on $U$.

**The assumption.** Prior probability shift (PPS) says that the distribution of the features within each class is the same in the source distribution $P$ (training and validation) and in the target distribution $Q$: $P(X\mid Y=\omega_j)=Q(X\mid Y=\omega_j)$ for every $j$, while the prevalences $P(Y=\omega_j)$ and $Q(Y=\omega_j)=\pi^Q_j$ may differ.

**Why $C$ is the same on $V$ and on $U$.** $h$ is a fixed function. For an instance of class $\omega_j$, the event "$h$ predicts $\omega_i$" is the event that its features fall in the set $\{x:h(x)=\omega_i\}$. The probability of a set computed with $Q(X\mid Y=\omega_j)$ equals the probability computed with $P(X\mid Y=\omega_j)$, because the two distributions are equal. So $Q(\hat Y=\omega_i\mid Y=\omega_j)=P(\hat Y=\omega_i\mid Y=\omega_j)=C_{ij}$.

**The population table.** By the product rule, $P(A\cap B)=P(A\mid B)\,P(B)$, the probability that a random target instance has true class $\omega_j$ and is predicted as $\omega_i$ is

$$
T_{ij}=Q(\hat Y=\omega_i,\,Y=\omega_j)=Q(\hat Y=\omega_i\mid Y=\omega_j)\,Q(Y=\omega_j)=C_{ij}\,\pi^Q_j,
$$

so $T=C\operatorname{diag}(\pi^Q)$. The accuracy of this table is the probability that the prediction is right, the sum of its diagonal:

$$
A(T)=\sum_jT_{jj}=\sum_jC_{jj}\,\pi^Q_j=d^\top\pi^Q,
\qquad d_j=C_{jj},
$$

where $d_j$ is the recall of class $\omega_j$. Accuracy is the average of the recalls, weighted by the prevalences. This is the *population accuracy*: the accuracy $h$ has on average over all bags with prevalence $\pi^Q$.

## T.2 The rate matrix measured on $V$

*Intuition:* within each class of $V$, count how often $h$ predicts each class; the fractions estimate the column of $C$.

Let $f_{ij}$ be the number of validation instances of true class $\omega_j$ that $h$ predicts as $\omega_i$. All the methods of E3.1 use the fractions

$$
\hat C_{ij}=\frac{f_{ij}}{n^s_j}.
$$

**Why the fractions.** Each of the $n^s_j$ instances of class $\omega_j$ is predicted as $\omega_i$ with probability $C_{ij}$, independently of the others. The count $f_{ij}$ is then a sum of $n^s_j$ indicators (variables equal to $1$ when the event happens and $0$ otherwise), each with mean $C_{ij}$. The mean of a sum is the sum of the means, so $\mathbb E[f_{ij}]=n^s_jC_{ij}$ and $\mathbb E[\hat C_{ij}]=C_{ij}$: on average over validation sets, the fraction is right (it is *unbiased*). It is also the maximum-likelihood estimate, the value of $C_{\cdot j}$ under which the observed counts are most probable.

**How wrong it can be.** An indicator with mean $p$ has variance $p(1-p)$: its mean is $p$, the mean of its square is also $p$ (because $1^2=1$ and $0^2=0$), and the variance is the mean of the square minus the square of the mean, $p-p^2$. Variances of independent variables add, and multiplying a variable by a constant $c$ multiplies its variance by $c^2$. So the estimated recall $\hat d_j=\hat C_{jj}$ has variance

$$
\operatorname{Var}(\hat d_j)=\frac{1}{(n^s_j)^2}\,n^s_j\,d_j(1-d_j)=\frac{d_j(1-d_j)}{n^s_j}.
$$

The methods of E3.1 treat $\hat C$ as if it were the true $C$, so this error, the term $e_2$ of [[ext4_uncertainty_budget|Extension 4]], does not appear in its interval.

**No pull towards $1/K$.** BayCAP gives each column of $C$ the prior $\operatorname{Dir}(\mathbf 1)$, which acts like one imaginary validation instance per cell and pulls the estimated recall to $(f_{jj}+1)/(n^s_j+K)$, towards $1/K$ ([[ext2_baycap]] §T.5). $\hat C$ has no prior, so it has no such pull.

*Implementation note:* `QuantifierBayCAP.fit` (`methods/baycap_quantifier.py`, the base of `BayCAPKDEy` and `BayCAPMAPLS`) computes $\hat C$ with `confusion_matrix(..., normalize="true")`, which puts true classes on rows, so the code stores $\hat C^\top$.

## T.3 BayesKDEy

*Intuition:* BayesKDEy describes how the output probabilities of the input classifier are spread within each class, and asks which mix of classes best explains the outputs observed on the target bag; NUTS lists many plausible mixes.

**The input classifier.** Both quantifiers of E3.1 work with the outputs $z=s(x)$ of an input classifier, never with the crisp predictions of $h$. The fit needs outputs on $V$ that look like outputs on new data, i.e. produced by a classifier that has not seen those instances. With $s=h$ (level `h`) this holds already: $h$ was trained on $L$, so its outputs on $V$ are out of sample, and the quantifier uses all of $V$ as it is. With $s$ an MLP trained on $V$ (level `mlp`), the outputs of the validation instances are obtained by cross-fitting (QuaPy's default, $5$ folds): $V$ is split into $5$ parts, and each instance gets the output of an MLP trained on the other $4$; on the target bag, the MLP is trained on all of $V$.

**The mixture.** $z$ is a function of the features, so by the argument of §T.1 its distribution within class $\omega_j$ is the same in $P$ and $Q$; call its density $f_j(z)$. A target instance is drawn by first picking a class, $\omega_j$ with probability $\pi^Q_j$, then features from that class. By the law of total probability (the probability of an outcome is the sum, over the ways it can happen, of the probability of each way), the density of $z$ in the target is the mixture

$$
g(z)=\sum_j\pi^Q_j\,f_j(z).
$$

**Kernel density estimates.** The densities $f_j$ are unknown. A kernel density estimate (KDE) places a small bump of probability on each validation output of class $\omega_j$ and averages them:

$$
\hat f_j(z)=\frac{1}{n^s_j}\sum_{v\in V_j}K_b(z,z_v),
\qquad
K_b(z,z_v)=c_b\exp\!\left(-\frac{\lVert z-z_v\rVert^2}{2b^2}\right),
$$

where $K_b$ is a Gaussian bump of width $b$ (the bandwidth) centred at $z_v$ and $c_b$ makes it integrate to one. A small $b$ follows the data closely; a large $b$ smooths them. The Aitchison variant of KDE-Bayes first moves each output towards the centre of the simplex by a factor $\lambda\in[0,1)$ (shrinkage) and then applies a log-ratio transformation, which suits points near the edges of the simplex better; the structure of the model is the same.

**The configuration of E3.1.** The Gaussian kernel with $b=0.1$ and no shrinkage. Two consequences help to read the results. First, the Gaussian bump is defined on all of $\mathbb R^K$, so part of each bump falls outside the simplex, where no output can be; the lost part is largest for outputs near the edges, which are the most frequent for a confident classifier. Second, when the outputs take few distinct values the estimate is a sum of separate spikes: a kNN with $k=5$ (a classifier of `main`) gives outputs that are multiples of $0.2$, so two neighbouring outputs, e.g. $(1,0)$ and $(0.8,0.2)$, are at distance $0.2\sqrt2\approx0.28$, almost three bandwidths, and their bumps barely overlap. This matters for the level `h` with kNN (and, less, with RF), not for `mlp`.

**The likelihood.** Treat the instances of $U$ as independent. The density of independent observations is the product of their densities, so the likelihood of $\pi$ given the target outputs $z_1,\dots,z_{N^Q}$ is

$$
\mathcal L_U(\pi)=\prod_{i=1}^{N^Q}\sum_j\pi_j\,\hat f_j(z_i),
\qquad
\log\mathcal L_U(\pi)=\sum_{i=1}^{N^Q}\log\sum_j\pi_j\,\hat f_j(z_i).
$$

The values $\hat f_j(z_i)$ are computed once, before sampling (`BayesianKDEy.sample_from_posterior`).

**The prior and the posterior.** The prior on $\pi$ is a Dirichlet distribution, with density proportional to $\prod_j\pi_j^{\alpha_j-1}$ on the simplex; with $\alpha=\mathbf 1$ the density is constant, so every prevalence vector is equally likely a priori. Bayes' rule combines prior and likelihood:

$$
p(\pi\mid U)\propto p(\pi)\,\mathcal L_U(\pi)^{1/\tau},
$$

with temperature $\tau=1$ in E3.1 (a temperature $\tau>1$ flattens the likelihood and widens the posterior).

**Posterior draws.** The logarithm of a sum has no closed-form posterior, so NUTS, a Markov chain Monte Carlo sampler, produces draws $\pi^{(1)},\dots,\pi^{(R)}$ whose histogram approximates $p(\pi\mid U)$. By the law of large numbers, an average over many draws approximates the corresponding average over the posterior. Each draw is a whole vector on the simplex: the draws keep the dependence between classes (if the data cannot tell class $\omega_1$ from $\omega_2$, draws with a high $\pi_1$ have a low $\pi_2$).

**What the spread represents.** The densities $\hat f_j$ are held fixed. As $N^Q$ grows, the log-likelihood is a sum of more and more terms and the posterior concentrates on the prevalence that best fits the outputs; its spread comes only from having finitely many target instances, the term $e_1^U$ of Extension 4. The error of $\hat f_j$ itself, from the finite $V_j$ and the choice of $b$, moves the posterior without widening it; it is the term $e_1^V$, and BayesKDEy does not represent it.

## T.4 BayesianMAPLS, a Bayesian EM quantifier

*Intuition:* if the outputs of the input classifier are calibrated probabilities, they already say how likely each class is for each instance under the source prevalences; a change of prevalences rescales them, and the prevalences that best explain the target bag are those under which the rescaled outputs make the bag most probable.

QuaPy has no class named BayesianEMQ: its Bayesian version of the EM quantifier EMQ (Saerens, Latinne and Decaestecker, 2002) is `BayesianMAPLS` (`quapy/method/_bayesian.py`), adapted from MAPLS (Ye et al., 2024). We call the resulting CAP method BayCAP-EMQ.

**Calibration.** Let $\pi^P$ be the prevalences of the source distribution. The outputs of $s$ are *calibrated* if they are the true class probabilities given the features, $s_y(x)=P(Y=\omega_y\mid X=x)$ for every class $\omega_y$: among the instances to which $s$ gives probability $0.8$ for $\omega_y$, $80\%$ are of class $\omega_y$. QuaPy takes $\pi^P$ as the class proportions of the labels on which the quantifier is fitted, here those of $V$ (`exact_train_prev=True`).

**From the source to the target.** Bayes' rule, $P(A\mid B)=P(B\mid A)P(A)/P(B)$, applied to the class and the features gives the density of the features within class $\omega_y$:

$$
P(x\mid\omega_y)=\frac{P(\omega_y\mid x)\,P(x)}{\pi^P_y}.
$$

Under PPS (§T.1) this density is the same in the target, $Q(x\mid\omega_y)=P(x\mid\omega_y)$. By the law of total probability, as in §T.3, the density of the features in the target is

$$
Q(x)=\sum_y\pi^Q_y\,Q(x\mid\omega_y)=\sum_y\pi^Q_y\,\frac{P(\omega_y\mid x)\,P(x)}{\pi^P_y}=P(x)\sum_y\frac{\pi^Q_y}{\pi^P_y}\,P(\omega_y\mid x).
$$

The target density is the source density times a weighted sum of the class probabilities, with weights the ratios of the prevalences: a class that is more frequent in the target than in the source gets a weight above $1$.

**The likelihood.** Treat the instances of $U$ as independent, so that their joint density is the product of their densities, and replace $P(\omega_y\mid x)$ with the outputs $s_y(x)$:

$$
\mathcal L_U(\pi)=\prod_{i=1}^{N^Q}P(x_i)\sum_y\frac{\pi_y}{\pi^P_y}\,s_y(x_i),
\qquad
\log\mathcal L_U(\pi)=\sum_{i=1}^{N^Q}\log\sum_y\frac{\pi_y}{\pi^P_y}\,s_y(x_i)+\sum_{i=1}^{N^Q}\log P(x_i).
$$

The last sum does not depend on $\pi$, so it is a constant factor of the likelihood, and Bayes' rule for $\pi$ (§T.3) removes it when the posterior is normalised. This is QuaPy's `BayesianMAPLS._log_likelihood`.

**The same form as BayesKDEy.** Both log-likelihoods have the form $\sum_i\log\sum_y\pi_y\,g_y(x_i)$. For BayesKDEy, $g_y(x_i)=\hat f_y(z_i)$, the kernel density of the outputs within class $\omega_y$ estimated on $V$. For BayesianMAPLS, $g_y(x_i)=s_y(x_i)/\pi^P_y$, which by the first display equals $P(x_i\mid\omega_y)/P(x_i)$: the density within class $\omega_y$ divided by a factor that is the same for all classes, and so does not change which $\pi$ fits best. The two quantifiers differ only in where the densities within each class come from: BayesKDEy estimates them from the outputs on $V$; BayesianMAPLS reads them off the outputs, trusting their calibration. If $s$ is not calibrated, the $g_y$ are wrong and the posterior moves without widening; as for the kernel densities (§T.3), this error is part of $e_1^V$, and it is not represented. Outputs exactly equal to $0$, frequent for kNN and RF, say that a class is impossible for that instance, and the likelihood takes them literally.

**The EM estimate.** BayesianMAPLS starts its chain at a preliminary EM estimate (`mapls_chain_init=True`), the maximiser of the likelihood (in MAPLS, of the likelihood times a Dirichlet penalty). At a maximum of a function of $\pi$ under the constraint $\sum_y\pi_y=1$ (with all $\pi_y>0$), the method of Lagrange multipliers says that the derivative with respect to every $\pi_y$ takes the same value $\mu$. Write $D_i=\sum_l(\pi_l/\pi^P_l)s_l(x_i)$ for the inner sum; the derivative of $\log D_i$ with respect to $\pi_y$ is $(s_y(x_i)/\pi^P_y)/D_i$, so

$$
\frac{\partial\log\mathcal L_U}{\partial\pi_y}=\sum_i\frac{s_y(x_i)/\pi^P_y}{D_i}=\mu.
$$

Multiply both sides by $\pi_y$ and define $w_{iy}=(\pi_y/\pi^P_y)s_y(x_i)/D_i$, the output for class $\omega_y$ reweighted by the ratio of the prevalences and renormalised to sum to one over $y$; then $\sum_iw_{iy}=\mu\pi_y$. Summing over $y$, the left side is $\sum_i\sum_yw_{iy}=\sum_i1=N^Q$ and the right side is $\mu$, so $\mu=N^Q$ and

$$
\pi_y=\frac{1}{N^Q}\sum_{i=1}^{N^Q}w_{iy}.
$$

The estimate is the average of the reweighted outputs; the EM algorithm finds it by repeating this equation from a starting value, usually $\pi=\pi^P$.

**The prior of the QuaPy implementation.** The NUTS model does not sample $\pi$ directly. It samples the isometric log-ratio (ILR) coordinates $z\in\mathbb R^{K-1}$, which map one-to-one onto the interior of the simplex through $\pi=\operatorname{ilr}^{-1}(z)$, gives them a standard normal distribution $z\sim\mathcal N(\mathbf 0,I)$, and adds the log-density of $\operatorname{Dir}(\alpha)$ at $\pi$ as an extra factor. With `prior="uniform"`, $\alpha=\mathbf 1$ and that factor is the constant $\log\Gamma(K)$, which changes nothing. So the prior of $\pi$ is the distribution of $\operatorname{ilr}^{-1}(z)$ with $z$ standard normal, a *logistic-normal* distribution, not the uniform $\operatorname{Dir}(\mathbf 1)$. Its density follows from the change-of-variables rule: if $z=g(\pi)$ is one-to-one, the density of $\pi$ is the density of $z$ at $g(\pi)$ times the absolute value of the determinant of the derivative of $g$. For the ILR transformation that determinant is proportional to $1/\prod_j\pi_j$ (Aitchison, 1986), so

$$
p(\pi)\propto\frac{\varphi_{K-1}\big(\operatorname{ilr}\pi\big)}{\prod_j\pi_j},
$$

with $\varphi_{K-1}$ the standard normal density in $K-1$ dimensions. *Binary case, step by step.* With $\pi_1=p$ and $\pi_0=1-p$, $z=\pm\frac{1}{\sqrt2}\ln\frac{p}{1-p}$ (the sign depends on the basis), and its derivative is $\frac{1}{\sqrt2}\big(\frac1p+\frac{1}{1-p}\big)=\frac{1}{\sqrt2\,p(1-p)}$ in absolute value, so

$$
p(p)=\frac{\varphi(z)}{\sqrt2\,p(1-p)},
\qquad
\varphi(z)=\frac{1}{\sqrt{2\pi}}\exp\!\Big(-\frac{1}{4}\Big(\ln\frac{p}{1-p}\Big)^2\Big).
$$

As $p\to0$, $\ln\frac{p}{1-p}\to-\infty$ and $\varphi(z)$ goes to $0$ like $\exp(-\frac14(\ln p)^2)$, faster than $\frac1p=\exp(-\ln p)$ grows, because $(\ln p)^2$ grows faster than $\lvert\ln p\rvert$; so the density goes to $0$ at the edges, where the uniform prior has density $1$. The UPP bags of `main` have prevalences spread uniformly over the simplex, so the uniform prior is the one matched to the protocol, and it is the one BayesKDEy uses. The posterior is the prior times a likelihood made of $N^Q=1000$ terms, which outweighs the prior where the prior is moderate; near the edges the logarithm of this prior goes to $-\infty$, so it pushes the draws away from prevalences near $0$. The effect should be concentrated on the bags with a small $\min_j\pi^Q_j$ [new hypothesis]; E3.1 measures it, and E3.c6 would remove it (user's decision of 2026-10-28: E3.1 keeps the prior of QuaPy).

**What the spread represents.** As for BayesKDEy, the outputs and $\pi^P$ are held fixed, so the spread of the posterior comes only from the finite bag, the term $e_1^U$.

## T.5 The population pushforward and its closed form for accuracy

*Intuition:* with a fixed rate matrix, accuracy depends on the prevalence only through the differences between the recalls; if all classes are recognised equally well, the mix of classes does not matter and the interval shrinks to a point.

**The method.** For each draw, a method of E3.1 forms the table $T^{(r)}=\hat C\operatorname{diag}(\pi^{(r)})$ and the metric value $a^{(r)}=A(T^{(r)})$; the point estimate is the mean of the $a^{(r)}$, and the interval runs from their $2.5\%$ to their $97.5\%$ percentile. For vanilla accuracy, by §T.1,

$$
a^{(r)}=\sum_j\hat d_j\,\pi^{(r)}_j=\hat d^\top\pi^{(r)}.
$$

**Mean.** The mean is linear: $\mathbb E[\sum_jc_jX_j]=\sum_jc_j\mathbb E[X_j]$ for constants $c_j$. Over the posterior, with $\hat d$ fixed,

$$
\mathbb E[a]=\hat d^\top\mathbb E[\pi]=\hat d^\top\bar\pi,
$$

so the point estimate is the accuracy at the posterior-mean prevalence $\bar\pi$.

**Variance.** The variance is the mean squared distance from the mean, $\operatorname{Var}(X)=\mathbb E[(X-\mathbb E X)^2]$. Here $a-\mathbb E[a]=\sum_j\hat d_j(\pi_j-\bar\pi_j)$. Squaring a sum gives the sum of all products of pairs of its terms:

$$
\big(a-\mathbb E[a]\big)^2=\sum_j\sum_k\hat d_j\hat d_k(\pi_j-\bar\pi_j)(\pi_k-\bar\pi_k).
$$

Taking the mean of each product gives a covariance, $\operatorname{Cov}(\pi_j,\pi_k)=\mathbb E[(\pi_j-\bar\pi_j)(\pi_k-\bar\pi_k)]$, so

$$
\operatorname{Var}(a)=\sum_j\sum_k\hat d_j\hat d_k\operatorname{Cov}(\pi_j,\pi_k)=\hat d^\top\Sigma_\pi\,\hat d,
$$

with $\Sigma_\pi$ the matrix of posterior covariances.

**Only the differences between recalls count.** The covariance is linear in each argument, and the covariance of anything with a constant is $0$. The prevalences sum to one, so for every $j$

$$
\sum_k\operatorname{Cov}(\pi_j,\pi_k)=\operatorname{Cov}\Big(\pi_j,\sum_k\pi_k\Big)=\operatorname{Cov}(\pi_j,1)=0,
$$

i.e. $\Sigma_\pi\mathbf 1=\mathbf 0$. Then for any number $c$,

$$
(\hat d-c\mathbf 1)^\top\Sigma_\pi(\hat d-c\mathbf 1)=\hat d^\top\Sigma_\pi\hat d-2c\,\mathbf 1^\top\Sigma_\pi\hat d+c^2\,\mathbf 1^\top\Sigma_\pi\mathbf 1=\hat d^\top\Sigma_\pi\hat d,
$$

because the last two terms contain $\Sigma_\pi\mathbf 1=\mathbf 0$ ($\Sigma_\pi$ is symmetric, so $\mathbf 1^\top\Sigma_\pi=\mathbf 0^\top$ too). Choosing $c$ equal to the mean recall $\bar d$,

$$
\operatorname{Var}(a)=(\hat d-\bar d\mathbf 1)^\top\Sigma_\pi(\hat d-\bar d\mathbf 1).
$$

If all recalls are equal, $\hat d-\bar d\mathbf 1=\mathbf 0$ and the interval has width $0$, however uncertain the prevalence is.

**Binary case.** With $\pi_0=1-\pi_1$, $a=\hat d_0+(\hat d_1-\hat d_0)\pi_1$, so $\operatorname{Var}(a)=(\hat d_1-\hat d_0)^2\operatorname{Var}(\pi_1)$ and the standard deviation is $\lvert\hat d_1-\hat d_0\rvert\operatorname{sd}(\pi_1)$.

*Example (arithmetic, not a result).* $\hat d=(0.90,0.70)$ and a posterior with $\operatorname{sd}(\pi_1)=0.02$ give $\operatorname{sd}(a)=0.2\cdot0.02=0.004$. The luck of a balanced bag of $1000$ instances with these recalls has standard deviation about $0.012$ ([[ext2_baycap]] §T.2), three times larger.

**Nonlinear metrics.** Macro-F1 and macro-K are not linear in the table, so the mean of $A(T^{(r)})$ is not $A$ of the mean table, and there is no closed form: the metric is applied to every draw. E3.1 uses the closed form only to read the results for accuracy.

## T.6 Population accuracy and realised accuracy

*Intuition:* even with the right prevalence and the right rates, the accuracy on this particular bag is random, because $h$ is right on a random subset of its instances.

`main` scores every interval against the *realised accuracy*, the fraction of the bag's instances that $h$ classifies correctly. The bag has $n_j=N^Q\pi^Q_j$ instances of class $\omega_j$ (UPP fixes the class counts, up to rounding), drawn with replacement from the test pool. Each is classified correctly with probability $d_j$ (the recall on the pool), independently of the others, so by the argument of §T.2 the number of correct predictions $c_j$ in class $\omega_j$ is a binomial count with mean $n_jd_j$ and variance $n_jd_j(1-d_j)$. The counts of different classes are independent, so their variances add, and dividing by $N^Q$ divides the variance by $(N^Q)^2$:

$$
A_{\text{real}}=\frac{1}{N^Q}\sum_jc_j,
\qquad
\mathbb E[A_{\text{real}}]=\sum_j\pi^Q_jd_j,
\qquad
\operatorname{Var}(A_{\text{real}})=\frac{1}{N^Q}\sum_j\pi^Q_jd_j(1-d_j).
$$

The mean is the population accuracy; the variance is the term $e_3$ of Extension 4, and it shrinks as $1/N^Q$. The population interval of §T.5 contains no part of it. With the numbers of the example of §T.5, the population interval has half-width $1.96\cdot0.004\approx0.008$ while the realised accuracy varies by about $\pm1.96\cdot0.012\approx\pm0.024$ around the population accuracy, so the interval misses the realised accuracy about half of the time even if $\pi$ and $\hat C$ are right (by the rule of §T.9 with $\eta=(0.012/0.004)^2=9$, coverage $2\Phi(1.96/\sqrt{10})-1\approx0.46$).

## T.7 The realised pushforward

*Intuition:* we cannot see the true classes of the bag, but we see what $h$ predicts on it; for each prediction we guess a plausible true class, and the guesses give a plausible realised table.

**The probability of each true class given a prediction.** By the definition of conditional probability, $P(A\mid B)=P(A\cap B)/P(B)$. A target instance is predicted as $\omega_i$ with probability $q_i=\sum_lT_{il}=\sum_l\hat C_{il}\pi_l$ (the sum of row $i$ of the table), so

$$
\rho_{ij}=P(Y=\omega_j\mid\hat Y=\omega_i)=\frac{T_{ij}}{q_i}=\frac{\hat C_{ij}\,\pi_j}{\sum_l\hat C_{il}\,\pi_l}.
$$

Row $i$ of $\rho$ is row $i$ of the table rescaled to sum to one; on the diagonal, $\rho_{ii}$ is the precision of class $\omega_i$.

**Sampling a realised table.** Let $n^t_i$ be the number of instances of the bag that $h$ predicts as $\omega_i$; it uses the predictions of $h$ on $U$, not its labels. For each draw $\pi^{(r)}$, compute $\rho^{(r)}$ from $(\hat C,\pi^{(r)})$; then, for each predicted class $\omega_i$, assign true classes to its $n^t_i$ instances by $n^t_i$ independent picks with probabilities $\rho^{(r)}_{i\cdot}$, i.e. one multinomial draw $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$. The counts, divided by $N^Q$, form a table $\tilde T^{(r)}$ whose row sums are the observed shares $n^t_i/N^Q$, and $A(\tilde T^{(r)})$ is a sample of the realised metric. This is the realised-table predictive of [[ext2_baycap]] §T.3 with $\hat C$ in place of the random $C^{(r)}$; `methods/baycap.py:realised_tables` already implements it for any table draws.

**Accuracy of one draw.** Of the $n^t_i$ instances predicted as $\omega_i$, each is correct (true class $\omega_i$) with probability $\rho^{(r)}_{ii}$, so the number of correct ones is binomial (one cell of a multinomial is binomial), and

$$
\tilde a^{(r)}=\frac{1}{N^Q}\sum_i\operatorname{Bin}\!\left(n^t_i,\rho^{(r)}_{ii}\right).
$$

## T.8 Width and centre of the realised interval

*Intuition:* the realised samples vary for two reasons, because the draws of $\pi$ differ and because each draw adds the luck of the bag; the two spreads add up. The centre can move, because neither quantifier chooses $\pi$ to match the predictions of $h$.

Write $\tilde a$ for a realised sample: first a draw $\pi$, then the multinomial picks of §T.7.

**Centre.** The law of total expectation says that the mean of $\tilde a$ is the mean, over the draws, of its mean given the draw: $\mathbb E[\tilde a]=\mathbb E\big[m(\pi)\big]$, with, by §T.7 and the mean $np$ of a binomial,

$$
m(\pi)=\mathbb E[\tilde a\mid\pi]=\sum_i\frac{n^t_i}{N^Q}\,\rho_{ii}(\pi).
$$

The population accuracy of the same draw can be written in the same form. From the definition of $\rho_{ii}$, $q_i\rho_{ii}=\hat C_{ii}\pi_i$, so

$$
a(\pi)=\sum_i\hat C_{ii}\pi_i=\sum_iq_i(\pi)\,\rho_{ii}(\pi),
\qquad
m(\pi)-a(\pi)=\sum_i\Big(\frac{n^t_i}{N^Q}-q_i(\pi)\Big)\rho_{ii}(\pi).
$$

The two centres differ only where the observed shares $n^t_i/N^Q$ differ from the shares $q_i(\pi)$ that the draw expects. In BayCAP the posterior of $\pi$ is fitted to $n^t$ through $C$, so $q\approx n^t/N^Q$, and E2.3 found the two centres within $0.009$ on every pair. The quantifiers of E3.1 fit $\pi$ to the outputs of the input classifier $s$, not to the predictions of $h$, so nothing forces the expected shares $q(\pi)=\hat C\pi$ to match the observed ones: where the posterior of the quantifier and the predictions of $h$ disagree, the realised centre moves [new hypothesis]. With $s=h$ the predictions of $h$ are the classes of largest output among the outputs the quantifier reads, so the two sources agree more closely, and the mismatch and the movement should be smaller than with $s$ an MLP trained on $V$, whose outputs carry information that $h$ does not use [new hypothesis]. Under PPS with the right $\pi$ and $C$, $\mathbb E[n^t_i/N^Q]=q_i$, and the mismatch $\lVert n^t/N^Q-\hat C\bar\pi\rVert_1$ is only sampling noise, of order $1/\sqrt{N^Q}$; a larger mismatch says that $\bar\pi$ or $\hat C$ is wrong, and it can be computed without target labels. Whether the moved centre is closer to the realised accuracy is an open question that E3.1 answers: the realised version keeps the observed shares and takes from $(\hat C,\pi)$ only the precisions.

**Width.** The law of total variance says that the variance of $\tilde a$ is the mean of its variance within a draw plus the variance of its mean across draws:

$$
\operatorname{Var}(\tilde a)=\mathbb E\big[\operatorname{Var}(\tilde a\mid\pi)\big]+\operatorname{Var}\big(m(\pi)\big).
$$

Given $\pi$, the counts of different predicted classes come from separate multinomial draws, so they are independent, and the binomial variance $np(1-p)$ of §T.2 gives

$$
\sigma_g^2(\pi)=\operatorname{Var}(\tilde a\mid\pi)=\frac{1}{(N^Q)^2}\sum_in^t_i\,\rho_{ii}(\pi)\big(1-\rho_{ii}(\pi)\big).
$$

If $m(\pi)-a(\pi)$ is about the same for all draws (a constant shift does not change a variance), $\operatorname{Var}(m(\pi))\approx\operatorname{Var}(a(\pi))=\sigma_p^2$, the variance of the population samples of §T.5, and

$$
\operatorname{Var}(\tilde a)\approx\sigma_p^2+\bar\sigma_g^2,
\qquad
\bar\sigma_g^2=\mathbb E\big[\sigma_g^2(\pi)\big].
$$

For roughly normal samples the $95\%$ interval is the mean $\pm1.96$ standard deviations, so its width is proportional to the standard deviation, and the realised interval is wider than the population one by the factor

$$
\frac{\sqrt{\sigma_p^2+\bar\sigma_g^2}}{\sigma_p}=\sqrt{1+\big(\bar\sigma_g/\sigma_p\big)^2}.
$$

E2.3 found this factor within $\pm5\%$ on every pair for BayCAP. For the methods of E3.1, $\sigma_p$ is small (§T.5) while $\bar\sigma_g$ is the luck of the bag, about $0.012$ in the example, so the factor should be large (about $3.2$ in the example of §T.5). If $m(\pi)-a(\pi)$ varies with the draw, the factor departs from this value; E3.1 measures both.

## T.9 What the uncertainty budget predicts

*Intuition:* an interval is wide only along the sources of error it simulates; the sources it leaves out move its centre without widening it.

[[ext4_uncertainty_budget|Extension 4]] splits the error of a CAP estimate into four parts: $e_3$, the luck of the bag (§T.6); $e_1^U$, the error of the prevalence from seeing only $N^Q$ target instances; $e_1^V$, the error of the prevalence from fitting the quantifier on a finite $V$, including its bias; $e_2$, the error of the rates of $h$ measured on $V$ (§T.2). From §T.2–§T.8:

| Method | Represented | Missing |
|---|---|---|
| `BayCAP-<Q>-<s>` (population), the four cells of E3.1 | $e_1^U$ | $e_1^V$ (the error of the kernel densities for `KDEy`, of the calibration of $s$ for `EMQ`), $e_2$, $e_3$ |
| `BayCAP-<Q>-<s>-real`, the four cells of E3.1 | $e_1^U$, $e_3$ | $e_1^V$, $e_2$ |
| `BayesCAP` (population) | $e_1^U$, $e_1^V$, $e_2$ (through the posterior of $C$) | $e_3$ |
| `BayesCAP-real` | all | none |
| `RQBS`; SQS-B (E3.c1) | $e_1^U$, $e_3$ | $e_1^V$, $e_2$ |

With $\eta$ the ratio of the missing to the represented variance, Extension 4 §T.7 derives that a $95\%$ interval covers with probability about $2\Phi\big(1.96/\sqrt{1+\eta}\big)-1$, where $\Phi$ is the cumulative distribution function of the standard normal: $0.95$ when nothing is missing, and less the more is missing. The predictions for E3.1 [new hypotheses]:

- **P1, the population interval under-covers.** It misses $e_3$, and its represented part is small when the recalls are similar (§T.5); its coverage should be lowest on the pairs with the smallest spread of the recalls, $\operatorname{sd}_j(\hat d_j)$.
- **P2, the realised interval is much wider.** The widening factor follows $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$ and is larger than for BayCAP, so coverage rises more than from `BayesCAP` to `BayesCAP-real`.
- **P3, no prior displacement.** On `dry-bean`, `hand_digits`, `letter` and `isolet`, where the shift of `BayesCAP` equals the prior part $R_0$ of its recall term (E2.2), the recall term of the methods of E3.1 should be small, because $\hat C$ is unbiased (§T.2); what remains of the shift is the prevalence term, from the error of the quantifier, and the difference between the recalls on $V$ and on the test pool (cause 4 of E2.2).
- **P4, the missing terms from $V$ matter where $V$ is small.** Where the validation classes are small (e.g. `poker_hand`, with $3$ instances in the smallest class), $e_2$ and $e_1^V$ are large and neither version represents them; these are the pairs where E3.c3, E3.c4 and E3.c7 could help.
- **P5, the realised methods and `RQBS` represent the same terms.** Their coverage should be similar where their quantifiers are equally good; differences then come from the quantifier (BayesKDEy or BayesianMAPLS against PACC) or from the moved centre of §T.8.
- **P6, BayCAP-EMQ depends on calibration.** BayesianMAPLS takes the densities within each class from the outputs of $s$ (§T.4), so its methods should cover worse the larger the ECE of $s$ on $V$; with $s=h$ this should show on the classifiers with coarse or poorly calibrated outputs (kNN, RF), while BayesKDEy, which estimates the densities from $V$, should depend less on calibration.
- **P7, the prior of BayesianMAPLS displaces it near the edges.** The coverage of the `EMQ` methods should fall on the bags with a small $\min_j\pi^Q_j$, and that of the `KDEy` methods should not, or less (§T.4).
- **P8, reading $h$ moves the centre less.** The mismatch and the movement of the centre should be smaller with $s=h$ than with $s$ an MLP (§T.8).

## T.10 The cells of $V$ and the likelihood of E3.2

*Intuition:* split each class of $V$ by what $h$ predicts on it. The density of the outputs of a class is then a mix of the densities of its cells, and the shares of the mix are the rates of $h$. Making the shares random makes $C$ random, and the target bag can then inform both $\pi$ and $C$.

**The cells.** Let $V_{kj}$ be the validation instances of true class $\omega_j$ that $h$ predicts as $\omega_k$; it has $f_{kj}$ instances (§T.2), and $\sum_kf_{kj}=n^s_j$. The kernel density of cell $(k,j)$ is the average of the bumps of its outputs, with the kernel $K_b$ of §T.3:

$$
\hat\phi_{kj}(z)=\frac1{f_{kj}}\sum_{v\in V_{kj}}K_b(z,z_v).
$$

(We write $\hat\phi$ for the densities of the cells, to keep $f_{kj}$ for their counts.)

**The class density is a mixture of the cell densities.** The cells of class $\omega_j$ split $V_j$ into groups with no instance in common, so the sum over $V_j$ in the kernel density of §T.3 is the sum of the sums over the cells. Multiplying and dividing each of them by $f_{kj}$:

$$
\hat f_j(z)=\frac1{n^s_j}\sum_k\sum_{v\in V_{kj}}K_b(z,z_v)=\sum_k\frac{f_{kj}}{n^s_j}\cdot\frac1{f_{kj}}\sum_{v\in V_{kj}}K_b(z,z_v)=\sum_k\hat C_{kj}\,\hat\phi_{kj}(z).
$$

So the density of BayesKDEy is the mixture of the cell densities, with weights the column $j$ of $\hat C$. E3.2 replaces $\hat C$ with an unknown $C$:

$$
f_j(z)=\sum_kC_{kj}\,\hat\phi_{kj}(z).
$$

In words: an instance of class $\omega_j$ falls in cell $k$ (it is predicted $\omega_k$ by $h$) with probability $C_{kj}$, and its output then follows the density of the cell.

**The joint likelihood, for the input classifier `mlp`.** For a target instance we observe the output $z_u$ of the MLP and the prediction $\hat y_u$ of $h$. By the chain rule of probability, $P(A\cap B\cap D)=P(A)\,P(B\mid A)\,P(D\mid A\cap B)$, the density that its true class is $\omega_j$, that $h$ predicts $\omega_k$ and that its output is $z$ is $\pi_j\,C_{kj}\,\hat\phi_{kj}(z)$. The true class is not observed, so by the law of total probability we sum over $j$:

$$
p(z_u,\hat y_u=\omega_k\mid\pi,C)=\sum_j\pi_j\,C_{kj}\,\hat\phi_{kj}(z_u).
$$

The instances of $U$ are treated as independent, so the likelihood is the product of these densities and its logarithm is a sum over the bag:

$$
\log\mathcal L_U(\pi,C)=\sum_{u\in U}\log\sum_j\pi_j\,C_{\hat y_uj}\,\hat\phi_{\hat y_uj}(z_u).
$$

The prediction of $h$ is not a function of the output of the MLP, so it adds information. Summing the joint density over $k$ gives the density of the outputs alone, $\sum_j\pi_jf_j(z_u)$; the joint density refines it with the predictions of $h$, the information that BayCAP's likelihood uses.

**The likelihood of the outputs alone, for the input classifier `h`.** With $s=h$ the prediction is the class of the largest output, $\hat y_u=\arg\max_kz_{u,k}$, so it is known once $z_u$ is known. Given $\hat y_u=\omega_k$, the output lies in the region $R_k$ of the simplex where $\omega_k$ has the largest output, but the Gaussian bumps of a cell put part of their mass outside $R_k$, more for the outputs near its border. In the joint form above each cell density would then have to be divided by its mass inside the region, $c_{kj}=\int_{R_k}\hat\phi_{kj}(z)\,dz$, to be a density on $R_k$; without that division the likelihood would favour the cells whose outputs lie far from the borders [new hypothesis]. E3.2 avoids computing $c_{kj}$ by using the density of the outputs alone, $\sum_j\pi_jf_j(z)$:

$$
\log\mathcal L_U(\pi,C)=\sum_{u\in U}\log\sum_j\pi_j\sum_kC_{kj}\,\hat\phi_{kj}(z_u).
$$

Every cell density is here a full kernel density, normalised as in BayesKDEy, and the prediction of $h$ enters through $z_u$, which determines it.

**E3.1 is the special case $C=\hat C$, for `h`.** By the mixture identity above, with $C=\hat C$ the inner sum is $\hat f_j(z_u)$, and the log-likelihood is that of BayesKDEy (§T.3). So, with $s=h$, E3.2 differs from the realised BayCAP-KDEy of E3.1 only in that $C$ is random, with its prior and the update by the bag. With `mlp`, $C=\hat C$ gives the likelihood of BayesKDEy refined by the predictions of $h$, so the two methods differ also in the use of the predictions.

**Computation.** For each bag the kernel values $K_b(z_u,z_v)$ of the $N^Q$ target outputs at the $\lvert V\rvert$ validation outputs are averaged within each cell once, before sampling, giving the array $F_{u,kj}=\hat\phi_{kj}(z_u)$ of size $N^Q\times K\times K$. NUTS then evaluates $\sum_u\log\sum_j\pi_jC_{kj}F_{u,kj}$, with $k=\hat y_u$ for `mlp` and summed over $k$ for `h`. The unknowns are those of BayCAP-Hf, whatever $\lvert V\rvert$.

## T.11 The prior on $C$: the validation counts and the Hf prior

*Intuition:* $V$ says a lot about the rates of $h$ on the large classes and little on the small ones; the prior decides the rest. The Hf prior lets the classes of a classifier borrow from each other, instead of pulling each recall towards $1/K$.

**The validation counts as data.** As in BayCAP, the counts of column $j$, $\mathbf f_{\cdot j}=(f_{1j},\dots,f_{Kj})$, are observed: each of the $n^s_j$ instances of class $\omega_j$ falls in cell $k$ with probability $C_{kj}$, independently of the others, so the counts are multinomial, $P(\mathbf f_{\cdot j}\mid C_{\cdot j})\propto\prod_kC_{kj}^{f_{kj}}$. By Bayes' rule the joint posterior is

$$
p(\pi,C\mid U,V)\propto p(\pi)\,p(C)\prod_jP(\mathbf f_{\cdot j}\mid C_{\cdot j})\;\mathcal L_U(\pi,C).
$$

**Why not a plain Dirichlet prior.** Take the prior $C_{\cdot j}\sim\operatorname{Dir}(\alpha,\dots,\alpha)$, whose density is proportional to $\prod_kC_{kj}^{\alpha-1}$. Bayes' rule multiplies it by the multinomial probability, and the exponents add, so what $V$ alone says about the column is again a Dirichlet:

$$
C_{\cdot j}\mid V\sim\operatorname{Dir}(\mathbf f_{\cdot j}+\alpha).
$$

One coordinate of a Dirichlet has mean its parameter divided by the sum of the parameters, so the recall $C_{jj}$ has mean

$$
\frac{f_{jj}+\alpha}{n^s_j+K\alpha}=(1-\lambda_j)\,\hat d_j+\lambda_j\,\frac1K,\qquad\lambda_j=\frac{K\alpha}{n^s_j+K\alpha},
$$

as one checks by writing the right side over the common denominator $n^s_j+K\alpha$, with $\hat d_j=f_{jj}/n^s_j$. A single $\alpha$ sets both how strongly each recall is pulled towards $1/K$ and how much probability the empty cells receive. With $\alpha=1$, BayCAP's prior, the pull displaces the recalls of the small classes when $K$ is large (E2.2 of [[ext2_baycap]]). With a small $\alpha$ the pull vanishes, but an empty cell has the parameter $\alpha<1$, whose long tail in the coordinates of NUTS causes divergent transitions (§T.11 of [[ext2_baycap]]). With $\alpha=0$ an empty cell is $0$ in every draw: a class recalled perfectly on $V$ has recall $1$ with no uncertainty, and a target instance predicted as a class that $h$ never predicts on $V$ has likelihood $0$ under every $(\pi,C)$.

**The Hf prior.** BayCAP-Hf (§T.16 of [[ext2_baycap]]) writes column $j$ as a recall and the shares of the mistakes, $C_{jj}=r_j$ and $C_{kj}=(1-r_j)\,s_{kj}$ for $k\ne j$, with $\sum_{k\ne j}s_{kj}=1$. Its prior has two parts:

- the recalls, $r_j\mid m,\nu\sim\operatorname{Beta}(\nu m,\nu(1-m))$, independent across the classes given $(m,\nu)$, with $m\sim U(0,1)$, the typical recall of the classifier, and $\rho=1/(1+\nu)\sim U(0,1)$, the share of the largest possible variance that the recalls have;
- the shares, $s_{\cdot j}\mid\kappa\sim\operatorname{Dir}(\kappa\mathbf 1_{K-1})$, with $1/(1+\kappa)\sim U(0,1)$: a small $\kappa$ puts the mistakes of a class in few cells, a large $\kappa$ spreads them evenly; with $K=2$ there are no shares and no $\kappa$.

**Where the recall is pulled.** A Beta is a Dirichlet with two coordinates, and the count of correct predictions of class $\omega_j$ on $V$ is binomial, a multinomial with two cells. So, given $(m,\nu)$, the update of the previous paragraph applies with the parameters $(\nu m,\nu(1-m))$ in place of $(\alpha,\alpha)$: the recall from $V$ is $\operatorname{Beta}(\nu m+f_{jj},\,\nu(1-m)+n^s_j-f_{jj})$, whose mean is

$$
\frac{\nu m+f_{jj}}{\nu+n^s_j}=(1-\lambda_j)\,\hat d_j+\lambda_j\,m,\qquad\lambda_j=\frac{\nu}{\nu+n^s_j}.
$$

The recall is pulled towards the typical recall $m$ of the classifier, not towards $1/K$, and by an amount that $V$ decides: $\nu$ is large, and the pull strong, only when the recalls of the other classes are similar. A class with few validation instances borrows from the others; a large class keeps its own count. The shares do the same for the mistakes, with $\kappa$ learned from how the mistakes of the classes are spread on $V$.

**Centred shares.** E3.2 samples the shares in the centred form of `BayesCAP-Hf-real` (user's decision of 2026-11-06), so that its prior and its coordinates are those of its reference. Where $\kappa<1$ meets empty cells the shares have long tails and NUTS may diverge (§T.16 of [[ext2_baycap]]); the non-centred form of E2.9 (§T.17 of [[ext2_baycap]]) samples the same model in other coordinates, but it was run on a pilot only. In E3.2 the bag may pin some of these cells (§T.13) and soften the problem [new hypothesis]; the divergences are recorded per pair.

## T.12 Empty cells

*Intuition:* the Hf prior gives every cell a positive rate, also the cells with no validation instance; such a cell needs a density, and it borrows the density of the instances that most resemble it.

A cell with $f_{kj}=0$ has no outputs, so $\hat\phi_{kj}$ is not defined, while $C_{kj}>0$ in every draw of the Hf prior. E3.2 uses the following convention (decision of 2026-11-06) [new hypothesis: the borrowed density is a reasonable stand-in for the missing one]:

1. if $h$ predicts $\omega_k$ for some validation instance, $\hat\phi_{kj}$ is the kernel density $\hat\phi_{k\cdot}$ of all the validation instances predicted $\omega_k$, whatever their true class: with $s=h$ their outputs lie in the region $R_k$, as those of the cell would; with $s$ an MLP it is the available description of the outputs of the instances that $h$ predicts as $\omega_k$;
2. if $h$ never predicts $\omega_k$ on $V$, $\hat\phi_{kj}$ is the kernel density $\hat f_j$ of class $\omega_j$ (§T.3).

Every cell then has a density, and since every $C_{kj}$ is positive, every target instance has a positive likelihood under every $(\pi,C)$: the posterior is always defined. A borrowed density acts only through the weight $\pi_jC_{kj}$ of its cell, which stays small, since $V$ has no instance in the cell, unless the bag supports it.

## T.13 What the target bag says about $C$, and what the budget predicts for E3.2

*Intuition:* BayCAP learns from the bag only how often $h$ predicts each class. E3.2 also sees how the outputs of the bag are spread within each prediction, which can separate the instances that $h$ gets right from those it gets wrong.

**In BayCAP the bag fixes only $C\pi$.** The likelihood of BayCAP uses the counts $n^t$ of the predictions of $h$ on $U$, a multinomial with probabilities $q=C\pi$. The $K$ counts sum to $N^Q$, so they carry $K-1$ free numbers, while $(\pi,C)$ has $K-1+K(K-1)$. Many pairs $(\pi,C)$ give the same $q$, and between them only $V$ and the prior decide.

**In E3.2 the bag can fix both $\pi$ and $C$.** Take the target instances that $h$ predicts as $\omega_k$. By §T.10 their outputs follow, up to a normalising constant, the mixture $\sum_j\beta_{kj}\hat\phi_{kj}(z)$ with weights $\beta_{kj}=\pi_jC_{kj}$. If the densities $\hat\phi_{k1},\dots,\hat\phi_{kK}$ have different shapes, the mixture determines its weights, as a kernel quantifier does, so the bag determines the $K^2$ products $\beta_{kj}$. They give back both unknowns: the columns of $C$ sum to one, so $\sum_k\beta_{kj}=\pi_j\sum_kC_{kj}=\pi_j$, and then $C_{kj}=\beta_{kj}/\pi_j$. Where the shapes overlap, or a cell has few instances on $V$, the bag cannot split the weights, and $V$ and the prior decide again. Two consequences [new hypotheses]:

- with $s=h$, the outputs of $h$ on its mistakes point, by definition, to the wrong class, so $\hat\phi_{kj}$ and $\hat\phi_{kk}$ overlap where $h$ is confidently wrong, and the bag informs the off-diagonal cells weakly; an MLP trained on $V$ may tell apart the instances that $h$ gets right and wrong, and inform them more;
- fitting $(\pi,C)$ to the bag makes the expected shares of the predictions $q=C\pi$ close to the observed shares $n^t/N^Q$: in the joint likelihood through the factor $C_{\hat y_uj}$, with $s=h$ through the regions $R_k$. So, as in BayCAP (E2.3), the realised pushforward should not move the centre (§T.8).

The ratio of the posterior standard deviation of $C_{kj}$ to its standard deviation from $V$ alone measures this cell by cell. From $V$ alone (Appendix B), $C_{kj}$ has a Beta marginal with mean $\hat C_{kj}$ and variance $\hat C_{kj}(1-\hat C_{kj})/(n^s_j+1)$; a ratio below $1$ says that the bag has narrowed the cell.

**The budget.** In the notation of §T.9:

| Method | Represented | Missing |
|---|---|---|
| `BayCAP-KDEy-Hf-<s>-real` (E3.2) | $e_1^U$, $e_3$, $e_2$ (through the posterior of $C$), the part of $e_1^V$ that is the weights of the cells within each class | the shapes of the cell densities (the rest of $e_1^V$), any bias of the kernel densities (the bandwidth, a difference between $V$ and the test pool) |

The weights of the cells of class $\omega_j$ are the column $C_{\cdot j}$, so the uncertainty on $C$ is at once the error of the rates of $h$ ($e_2$) and an error of the class densities of the quantifier ($e_1^V$), coherently, as in option 2 of Appendix B.

**Predictions for E3.2** [new hypotheses]:

- **P9, coverage rises where the error is $e_2$.** On `mammographic` and on `semeion` LR and MLP, coverage and Winkler score improve over the realised BayCAP-KDEy of E3.1. All bags of a pair share one $V$, so the error from $V$ is a single draw per pair: an interval that represents it correctly can still under-cover on a pair whose $V$ gave a large draw, and the five pairs of `mammographic` share one such draw. So the test is the standardised shift: it should fall on these pairs and, across pairs, look like draws of a standard normal; a coverage of $0.95$ on each pair is not expected.
- **P10, `poker_hand` improves only in part.** Its smallest class has $3$ validation instances: the Hf prior makes its rates uncertain and pulls them towards the typical recall, but the shapes of its cell densities, built from $3$ outputs, stay fixed.
- **P11, `wine-quality` does not improve.** Its error is a bias shared by all quantifiers (E3.1), which a random $C$ widens but does not remove.
- **P12, against `BayesCAP-Hf-real`.** The two methods have the same prior and the same use of $V$; where the cells are well separated the bag informs $C$ more in E3.2, so its intervals should be narrower at a similar coverage, more with `mlp` than with `h`.
- **P13, the bag narrows the cells with many instances.** The ratio of standard deviations is below $1$ on the cells with a large $f_{kj}$ and near $1$ on the small ones, more often with `mlp` than with `h`.
- **P14, the predicted shares match the observed ones.** The mismatch $\lVert n^t/N^Q-\bar C\bar\pi\rVert_1$ is of the order of the sampling noise, $1/\sqrt{N^Q}$, and smaller than for the methods of E3.1.

# Implementation

Components (for E3.1):

- **One class for the eight methods.** `methods/baycap_fixed.py:BayCAPFixed` [implemented, then split by WP-13] generalised the former `methods/baycap_kdey.py:BayCAPKDEy` (no stored results, removed) with three parameters: `quantifier` $\in\{$`"kdey"`, `"emq"`$\}$, BayesKDEy or BayesianMAPLS; `input_classifier` $\in\{$`"h"`, `"mlp"`$\}$; `estimand` $\in\{$`"population"`, `"realised"`$\}$. The E3.1 results were produced by this class; since WP-13 the two quantifiers are `BayCAPKDEy` and `BayCAPMAPLS`, with the same draws (see the components for E3.2).
- **The NUTS models.** QuaPy builds `BayesianKDEy` and `BayesianMAPLS` only when `pystan` is installed, which is not the case in the environment of the project, so the two NUTS models and their preparation are copied from QuaPy's `quapy.method._bayesian` (commit `31f36f6`; user's decision of 2026-11-03): the same sites, priors, likelihoods, initialisation and seeds, so the same draws as QuaPy's `sample_from_posterior` (checked by a test). The kernel densities are QuaPy's (`KDEBase`), and so is the EM start of BayesianMAPLS (`mapls`). The class runs NUTS with `extra_fields=("diverging",)`, which leaves the draws unchanged, and returns the divergences, split-$\hat R$ and effective sample size of $\pi$.
- **Input classifier `h`.** No classifier is trained: the quantifier is fitted on the `posteriors` argument of `fit` (the outputs of $h$ on $V$) and the labels of $V$, and sampled on the `posteriors` argument of `predict_ct_range` (the outputs of $h$ on the bag); the class order of the posteriors of $h$ is that of `val.classes_`. With `"mlp"`, the validation outputs are cross-fitted on $5$ folds, as QuaPy's `fit` does, in the process of the worker (`n_jobs=1`), and the target outputs come from the MLP trained on all of $V$.
- **Realised pushforward.** `"realised"` takes $n^t$ from the `posteriors` argument of `predict_ct_range` (the arg-max of the outputs of $h$ on $U$) and applies `methods/baycap.py:realised_tables` to the population draws; the multinomial draws use a NumPy `Generator` seeded from `mcmc_seed`, as in `BayCAP`. The tables stay in the `cap-ml` layout (true classes on rows).
- **Reproducible MLP:** `BayCAPFixed` (now `QuantifierBayCAP`) builds `MLPClassifier(random_state=random_state)` [implemented], so that the population and realised methods of a cell share their prevalence draws and a rerun reproduces them; `BayCAPKDEy` built it without a `random_state` (an item of FIX-3 in the work plan, now solved). The change had no stored results to affect.
- **Configuration:** BayesKDEy with `kernel="gaussian"`, `bandwidth=0.1`, `shrinkage=0.0`; BayesianMAPLS with QuaPy's defaults (`prior="uniform"`, `exact_train_prev=True`, `mapls_chain_init=True`); for both, `temperature=1`, `num_warmup=500`, `num_samples=1000`, `mcmc_seed` as for `BayesCAP`.
- **Source prevalence of BayesianMAPLS with `h`:** the outputs of $h$ refer to the prevalences of $L$, on which $h$ was trained, while QuaPy sets $\pi^P$ to those of $V$. $L$ and $V$ are the two halves of a stratified split of the same training data (`cap.data.datasets.split_train`), so $\lvert\pi^L_j-\pi^V_j\rvert$ is only the rounding of the class counts, of order $1/\lvert V\rvert$, and QuaPy's behaviour is kept.
- **Methods in `main`:** eight factory entries `BayCAP-<Q>-<s>` and `BayCAP-<Q>-<s>-real` in `experiments/methods.py` (`FIXED_C_VARIANTS`, `METHOD_CLASS` and `gen_methods`), run in the BayCAP pool with `BAYES_ENV` (CPU, single-threaded XLA; the pool holds the methods of `BAYES_POOL_METHODS`); results only: excluded from the tables, ranks and default plots of `main`, like `BayesCAP-real`. Per bag, `main` stores the NUTS diagnostics of $\pi$ and `prev_mean` ($\bar\pi$), `acc_sd` ($\sigma_p$), `sigma_g` ($\bar\sigma_g$) and `val_ece` (the ECE of the input classifier on its validation outputs).
- **Tables and plots of E3.1:** `tables/baycap_fixed.py`, reusing the paired bootstrap of `tables/pushforward.py`, and `plots/baycap_fixed.py`; `tables/displacement.py --method` for the decomposition, with $R_0=0$ for the eight methods; `tables/rank.py --include-e31` for the rank table. The mismatch uses $n^t/N^Q$, the predicted margins of the stored realised table (no labels), $\hat C$ rebuilt from the validation counts, and the stored $\bar\pi$; the widening factor uses the stored $\sigma_p$ and $\bar\sigma_g$.

Tests:

- for the same seed, the draws equal those of QuaPy's `sample_from_posterior`, for both quantifiers, and the copied models have the log-density of §T.3 and §T.4;
- for each quantifier and input classifier, with the same seeds, both estimands use identical prevalence draws;
- with `input_classifier="h"`, the quantifier receives exactly the posteriors passed to `fit` and `predict_ct_range`, and no classifier is trained;
- with `quantifier="emq"`, the source prevalence equals the class proportions of $V$;
- realised tables have predicted-class margins equal to $n^t/N^Q$; with $\hat C=I$ they return the observed counts;
- for accuracy, the mean of the population samples equals $\hat d^\top\bar\pi$, and their variance equals $(\hat d-\bar d\mathbf 1)^\top\hat\Sigma_\pi(\hat d-\bar d\mathbf 1)$ on the draws;
- the factory builds the eight methods with the configuration of E3.1, and the tables, ranks and default plots of `main` exclude them.

Reproducibility and comparability:

- E3.1 adds the rows of eight methods to `main` and leaves the stored results of every other method untouched, as E2.3 did for `BayesCAP-real`; the paper's tables are unchanged;
- every row records the quantifier and its configuration, the input classifier, the seeds and the pushforward;
- the eight methods are comparable with the CPU results of `BayesCAP`, `BayesCAP-real` and `BayesCAP-Hf-real` (same bags, same platform); a pilot on two datasets precedes the full run.

Components (for E3.2), all [implemented] (card WP-13), not run:

- **Split of `BayCAPFixed`** (user's decision of 2026-11-06). `methods/baycap_fixed.py:BayCAPFixed` is split into `methods/baycap_kdey.py:BayCAPKDEy` (BayesKDEy) and `methods/baycap_mapls.py:BayCAPMAPLS` (BayesianMAPLS); the parameter `quantifier` disappears, and the code they share (input classifier and cross-fitting of the MLP, realised pushforward, NUTS diagnostics, ECE, moments of accuracy) moves to the base class `methods/baycap_quantifier.py:QuantifierBayCAP`; `BayCAPFixed` is removed. The eight factory entries of E3.1 keep their names and configurations and build the new classes. The name `BayCAPKDEy` was used by the class that `BayCAPFixed` replaced, which has no stored results.
- **The random $C$ in `BayCAPKDEy`.** A parameter `c_prior` $\in\{$`"fixed"`, `"hierarchical-f"`$\}$, default `"fixed"` (E3.1). With `"hierarchical-f"`, a NumPyro model of its own: $\pi\sim\operatorname{Dir}(\mathbf 1)$; the Hf sites and the construction of $C$ of `methods/baycap.py` (`hierarchical_rate_sites`, which `BayCAP` also uses, with `strength_from_rho`, `hierarchical_rates` and centred shares), shared, not copied (`kdey_hf_model`); the validation counts observed as multinomials per class, as in `BayCAP.model`; the log-likelihood of §T.10 as a `numpyro.factor`, computed with a log-sum-exp over the true classes (and the predicted classes for `h`) from the logarithms of $\pi_j$, $C_{kj}$ and $F_{u,kj}$ (`kdey_hf_loglik`).
- **Cell densities.** Per fit, one kernel density per non-empty cell $(k,j)$ on the validation outputs of the input classifier (cross-fitted for `mlp`), with the kernel of E3.1 (QuaPy's `KDEBase`, Gaussian, $b=0.1$), and the borrowed densities of §T.12 for the empty cells; per bag, the array of log-densities $\log F_{u,kj}$ of the target outputs, computed before sampling (`cell_log_densities`: all $K^2$ cells for `h`, the $K$ cells of the prediction of $h$ for `mlp`); each distinct density is fitted once, and `cell_source_` records which cells borrow.
- **Realised pushforward.** `methods/baycap.py:realised_tables` applied to the table draws $C^{(r)}\operatorname{diag}(\pi^{(r)})$, in the `cap-ml` layout (true classes on rows); the population pushforward stays available for tests, and only the realised one is run.
- **Diagnostics and summaries.** `nuts_diagnostics` on $\pi$, $C$ and the Hf hyperparameters; `hierarchical_summaries` and `kappa_summaries`; per bag, the posterior mean and standard deviation of $C$ (`c_mean`, `c_sd`), the mismatch with $\bar C$ (`mismatch`), the number of empty cells (`n_empty_cells`) and, among them, of those that take the class density (`n_class_density_cells`; the others take the density of the predicted class). A failed start of NUTS is handled as in `BayesCAP-Hf-real`.
- **Methods in `main`:** two factory entries, `BayCAP-KDEy-Hf-h-real` and `BayCAP-KDEy-Hf-mlp-real`, run in the BayCAP pool with `BAYES_ENV`; in the tables, ranks and coverage vs amplitude plot of `main` (user's request of 2026-11-06), not in its other default plots.
- **Tables and plots:** `tables/baycap_kdey_hf.py` (comparison, tests, diagnostics, and the per-cell CSV of the information on $C$), reusing the paired bootstrap of `tables/pushforward.py`, and `plots/baycap_kdey_hf.py` (the two coverage scatters and the scatter of the information on $C$); `tables/displacement.py --method` for the decomposition, with the prior part $R_0$ reported as NaN (the posterior mean of a recall given $V$ alone has no closed form under the Hf prior); `tables/rank.py --include-e32` for the rank table. The comparison table of the Representation is split into two LaTeX tables, the scores with the standardised shift and the four tests, which do not fit one page together; the scatter of the information on $C$ leaves out the cells with no ratio ($f_{kj}=0$ or $\hat C_{kj}\in\{0,1\}$).

Tests (for E3.2):

- after the split, `BayCAPKDEy` and `BayCAPMAPLS` give, for the same seeds, the tables, diagnostics and summaries of `BayCAPFixed` in the eight configurations of E3.1 (against a frozen copy of it, `tests/_reference_baycap_fixed.py`);
- the cells partition $V$, and $\sum_k\hat C_{kj}\hat\phi_{kj}$ equals the class density $\hat f_j$ of BayesKDEy;
- with $C=\hat C$, the log-likelihood of `h` equals that of BayesKDEy, and the joint log-likelihood of `mlp`, summed over the predicted classes, equals the log-likelihood of the outputs alone;
- the borrowed densities follow §T.12, and on a bag with a prediction that $h$ never makes on $V$ the log-posterior is finite;
- the Hf sites are those of `BayCAP` with `c_prior="hierarchical-f"`, and with $K=2$ there is no $\kappa$;
- the realised tables have predicted-class margins equal to $n^t/N^Q$;
- for the same seeds, two runs give the same draws.

Reproducibility and comparability (for E3.2):

- the split of `BayCAPFixed` changes no draw (first test above), so the stored results of E3.1 stay valid and its factory entries reproduce them;
- E3.2 adds the rows of two methods to `main` and leaves the stored results of every other method untouched;
- the two methods are comparable with the stored CPU results of E3.1 and of `BayesCAP-Hf-real` (same bags, same platform); a pilot on `mammographic`, `poker_hand` and `wine-quality` precedes the full run.

# Appendix

## A. Toy simulations

None yet.

## B. Making the rate matrix random for BayesKDEy (background for E3.c3, E3.c4 and E3.2)

*Intuition:* describe the uncertainty about each class-conditional distribution with random weights on the validation instances of that class; one set of weights gives, coherently, both the kernel densities of the quantifier and a rate matrix of $h$.

**The Bayesian bootstrap.** For each class, draw weights on its validation instances, $w_j=(w_{j,v})_{v\in V_j}\sim\operatorname{Dir}(1,\dots,1)$: non-negative, summing to one, uniform on the simplex. Each draw is a version of $V_j$ in which some instances count more and others less. This is Rubin's Bayesian bootstrap (1981), the posterior of a Dirichlet process with a prior that vanishes. One draw of the weights gives

$$
\hat f^{(w)}_j(z)=\sum_{v\in V_j}w_{j,v}K_b(z,z_v),
\qquad
C^{(w)}_{ij}=\sum_{v\in V_j}w_{j,v}\,\mathbf 1[h(v)=\omega_i],
$$

two functionals of the same random distribution of class $\omega_j$.

**The column of $C$ is a Dirichlet.** The *aggregation property* of the Dirichlet distribution says that if $(w_1,\dots,w_n)\sim\operatorname{Dir}(\alpha_1,\dots,\alpha_n)$ and the coordinates are split into groups, the group sums follow a Dirichlet whose parameters are the sums of the $\alpha$'s in each group. Grouping the instances of $V_j$ by the prediction of $h$, group $i$ has $f_{ij}$ instances, each with $\alpha=1$, so

$$
C^{(w)}_{\cdot j}\sim\operatorname{Dir}(f_{1j},\dots,f_{Kj}).
$$

This is BayCAP's posterior on the column from $V$ alone, with the prior $\operatorname{Dir}(\mathbf 0)$ in place of $\operatorname{Dir}(\mathbf 1)$; its mean is $f_{ij}/n^s_j=\hat C_{ij}$, so it adds spread without the pull towards $1/K$, and a cell with $f_{ij}=0$ is $0$ in every draw.

**Three ways to use it**, by increasing cost:

1. *Random $C$ at the bridge only* (E3.c3): keep the posterior of §T.3 and pair each $\pi^{(r)}$ with an independent $C^{(r)}_{\cdot j}\sim\operatorname{Dir}(f_{\cdot j})$. It adds $e_2$, not $e_1^V$, and costs no extra NUTS.
2. *Cut posterior* (E3.c4): for each of $M$ draws of the weights, recompute $\hat f^{(w)}_j$ and $C^{(w)}$, run NUTS on $\pi$ with $\hat f^{(w)}_j$, and pool the draws. The weights come from $V$ only and are not updated by $U$: this is modular Bayesian inference, not a full posterior. It adds $e_1^V$ and $e_2$, coherently, at about $M$ times the cost.
3. *Full joint posterior* (the former candidate E3.c5): NUTS samples $(\pi,w)$ together, with likelihood $\sum_{i\in U}\log\sum_j\pi_j\sum_{v\in V_j}w_{j,v}K_b(z_i,z_v)$, so $U$ also updates the weights, as it updates $C$ in BayesianCC. It has $\lvert V\rvert$ extra parameters, and since $U$ updates the weights only through the outputs of $s$, the predictions of $h$ on $U$ do not inform $C$. The likelihood depends on $(\pi,w)$ only through the products $\beta_v=\pi_{j(v)}w_{j(v),v}$, where $j(v)$ is the class of $v$: the target is modelled as one weighted kernel density over all of $V$, and $\pi_j$ is the total weight of class $\omega_j$. Two validation outputs closer than $b$ have almost the same bump, so the bag cannot tell their weights apart, and the posterior is flat along many directions, which makes NUTS slow. Grouping the instances of each class and giving one weight to each group removes these directions; by the aggregation property the group weights have the prior $\operatorname{Dir}$ of the group sizes, so only the re-weighting within a group is lost. Grouping by the prediction of $h$ makes the group weights the columns of $C$: this is the model of E3.2 (§T.10), with the validation counts and the Hf prior in place of the Bayesian bootstrap (§T.11).

## C. Relation to other extensions

- **[[ext1_metrics|Extension 1]]**: supplies the metrics; macro-F1 and macro-K have no closed form under the fixed $\hat C$ (§T.5).
- **[[ext2_baycap|Extension 2]]**: supplies the realised-table predictive (§T.7), the decomposition of the shift and the paired Winkler test (E2.2, E2.3), the finding that the prior on $C$ displaces BayCAP on large-$K$ datasets, which P3 tests from the other side, and the Hf prior on $C$ (E2.8) used by E3.2, whose reference is `BayesCAP-Hf-real`.
- **[[ext4_uncertainty_budget|Extension 4]]**: supplies the budget of §T.9; its regime map (E4.2a) includes the methods of this extension, with the same implementations, so this extension does not vary $\lvert U\rvert$ or $\lvert V\rvert$.
- **[[ext5_baycap_priors_tempering|Extension 5]]**: tempering, including the temperatures of BayesKDEy and BayesianMAPLS.
- **Extension 7** ([[extensions]] §7): tuning of the MLP input classifier and of the hyperparameters of BayesKDEy (kernel, $b$, $\lambda$) beyond the configuration fixed for E3.1.
- **[[optional_extensions]]**: §3 keeps Aitchison-SQS; §8 keeps the material set aside from the former Extensions 8 and 9.

## D. Out of scope

- tempering of BayesKDEy and BayesianMAPLS ($\tau=1$ throughout; Extension 5);
- varying $\lvert U\rvert$ and $\lvert V\rvert$ (Extension 4);
- time and memory measurements (optional, user's decision of 2026-10-27; [[optional_extensions]] §8);
- the sensitivity analyses and ablations of the former documents ([[optional_extensions]] §8);
- violations of PPS ([[optional_extensions]] §2).

## E. BayesianMAPLS: a uniform prior and a random calibration map (background for E3.c6–E3.c7)

*Intuition:* the prior of the QuaPy model can be made uniform by sampling the prevalences directly or by correcting for the change of variables; and since BayesianMAPLS models the classes through the calibration of its input classifier, the calibration is where its uncertainty from $V$ can be represented.

**Two ways to a uniform prior (E3.c6).** Both keep the likelihood of §T.4.

1. *Sample $\pi$ directly* from $\operatorname{Dir}(\alpha)$, as BayesKDEy does; with $\alpha=\mathbf 1$ the prior is uniform. The EM start of the chain is kept by passing the EM estimate as the initial value of $\pi$.
2. *Keep the ILR coordinates and correct the density.* By the change-of-variables rule of §T.4, the density of $z$ that corresponds to a density $p(\pi)$ is $p(\operatorname{ilr}^{-1}z)$ times the absolute determinant of the derivative of $\operatorname{ilr}^{-1}$, which is the reciprocal of the one of §T.4 and so is proportional to $\prod_j\pi_j$. The model gives $z$ a flat (improper) density in place of $\mathcal N(\mathbf 0,I)$ and adds the factor $\log\operatorname{Dir}(\pi;\alpha)+\sum_j\log\pi_j$ with $\pi=\operatorname{ilr}^{-1}(z)$. QuaPy's model adds the first term but replaces the second with the log-density of $\mathcal N(\mathbf 0,I)$, which is what makes its prior logistic-normal.

**Where the uncertainty from $V$ sits (E3.c7)** [new hypothesis]. In BayesKDEy the validation set enters through the kernel densities, a sum over the instances of $V$, which the Bayesian bootstrap of Appendix B can randomise. In BayesianMAPLS it enters only through $\pi^P$ and, for `mlp`, through the training of the MLP: the densities within each class are read off the outputs (§T.4), so there is no sum over $V$ to randomise. Its error from $V$ is the miscalibration of $s$. A parametric recalibration makes it explicit: the recalibrated outputs are

$$
s^\theta_y(x)=\frac{\exp\big(\log s_y(x)/t+\beta_y\big)}{\sum_l\exp\big(\log s_l(x)/t+\beta_l\big)},
\qquad
\theta=(t,\beta_1,\dots,\beta_K),
$$

a temperature $t>0$ that makes the outputs more or less confident and class biases $\beta_y$ that shift them towards some classes (bias-corrected temperature scaling, Alexandari et al. 2020; one bias can be fixed at $0$, since adding the same number to all of them changes nothing). The joint posterior of $(\pi,\theta)$ has two likelihoods: on $V$, $\sum_{v\in V}\log s^\theta_{y_v}(x_v)$, the log-probability of the observed labels under the recalibrated outputs, which involves $\theta$ only; on $U$, the likelihood of §T.4 with $s^\theta$ in place of $s$, which involves both. The labels of $V$ locate $\theta$, the bag also updates it, as it updates $C$ in BayesianCC, and the uncertainty about $\theta$ widens the posterior of $\pi$. With $K$ parameters added to the $K-1$ of $\pi$, NUTS stays cheap, unlike the $\lvert V\rvert$ weights of the full joint posterior of Appendix B (option 3).

**The bridge stays separate.** The recalibration changes the outputs that the quantifier reads, not the predictions of $h$, which are fixed. So it does not make the rate matrix $\hat C$ of $h$ random: with $s=h$ the two are computed from the same outputs but are different functions of them, and a random $C$ at the bridge still needs option 1 of Appendix B (E3.c3).
