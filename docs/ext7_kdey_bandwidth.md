# Extension 7: Selecting the bandwidth of BayesKDEy

## Claim

BayesKDEy estimates, for each class $\omega_j$, the density of the outputs of its input classifier from the validation set $V$ with a Gaussian kernel of bandwidth $b$. The bandwidth decides how sharp these class densities are, and so how narrow the posterior of $\pi^Q$ is and where it is centred. In `main` it is fixed at $b=0.1$ for every dataset, class and validation size. This extension asks whether a bandwidth selected from $V$ alone gives `BayCAP-KDEy-mlp-real` better intervals. E7.1 selected one bandwidth per class by the leave-one-out likelihood of the class densities: it fitted the densities better and made the intervals worse (pilot, closed). E7.2 selects one bandwidth per pair by the error it causes on the quantity the method estimates: $V$ is split in two halves, the method is fitted on one half, and the bandwidth is the value of a grid whose estimated accuracy is closest, on labelled bags drawn from the other half, to the accuracy that the true prevalences of the bags would give. It is kept deliberately tight (user's decision of 2026-11-19): one experiment at a time, no candidates for now.

## Status

| Item | Status |
|---|---|
| BayesKDEy with one bandwidth for all classes (`methods/bayes_kdey.py:BayesKDEy`, `bandwidth=0.1`), used by `BayCAP-KDEy-mlp-real` and `SQS-B` in `main` | [implemented], run and stored in `main` (E3.1, E3.3 of [[ext3_posterior_propagation]]) |
| Quantifier classes `BayesKDEy`, `BayesMAPLS` used by `BayCAPKDEy`, `BayCAPMAPLS` and `SQSB` (no change of results) | [implemented] (card WP-25, done) |
| Per-class bandwidths selected by leave-one-out likelihood on $V$, with densities normalised on the simplex (§T.4): `bandwidth="loo"` of `methods/bayes_kdey.py:BayesKDEy` | [implemented], evaluated by the pilot of E7.1, not carried forward (card WP-25, done) |
| E7.1 (`BayCAP-KDEy-mlp-real-loo` and `SQS-B-loo` against the stored methods with $b=0.1$) | [implemented], pilot run and analysed on 2026-11-19 (`mammographic`, `wine-quality`, 10 pairs); closed after the pilot, without the full run (user's decision of 2026-11-19): both methods worse on every metric; card WP-25, done ([[work_plan]] §II.29) |
| E7.2 (`BayCAP-KDEy-mlp-real-cal`: one bandwidth per pair selected on calibration bags drawn from half of $V$, against the stored method with $b=0.1$) | [implemented], pilot run and analysed on 2026-11-20 (`mammographic`, `wine-quality`, `poker_hand`, 15 pairs, with the NUTS check), full run not done (card WP-26, Implemented, not evaluated, [[work_plan]] §III.2 and §II.4) |
| Factorial grid-search study of $h$, the surrogate classifier and the quantifier (the earlier scope of Extension 7) | set aside on 2026-11-19, in [[optional_extensions]] §13 |

# Experiments

## E7.1: Per-class leave-one-out bandwidth for BayesKDEy

**Why.** The bandwidth $b=0.1$ of BayesKDEy is QuaPy's default and was never tuned in this project. A bandwidth that is too small makes each class density a set of narrow spikes around the validation outputs, so the posterior of $\pi^Q$ is too narrow; one that is too large blurs the classes together, so the posterior is wide and the classes are hard to tell apart (§T.1). The right value depends on how many validation outputs a class has and how spread they are, which differ across datasets and across the classes of one dataset. The leave-one-out likelihood picks, for each class, the bandwidth whose density best predicts validation outputs it has not seen (§T.2, §T.3); it uses only $V$, so it is a method, not a reference. Because `SQS-B` uses the posterior draws of `BayCAP-KDEy-mlp-real`, one selection serves both methods.

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs (datasets × the five classifiers of `main`, after FIX-8), $100$ target bags of size $1000$ per pair with the seeds of `main`; the stored `main` results of `BayCAP-KDEy-mlp-real` and `SQS-B` with $b=0.1$; the results of two new methods of `main`, `BayCAP-KDEy-mlp-real-loo` and `SQS-B-loo`, with their own run.
  - **Varied:** the bandwidth of BayesKDEy: one $b=0.1$ shared by all classes (stored) against one $\hat b_j$ per class, selected by leave-one-out likelihood on the cross-fitted validation outputs of the MLP of class $\omega_j$ over the grid $\mathcal G=\{10^{-2+0.1k}:k=0,\dots,20\}$ ($21$ values, log-spaced from $0.01$ to $1$, containing $0.1$; the range of the Gaussian-KDE grid of the BayesKDEy paper, `docs/papers/kde_bayes.pdf`, App. D.2, Table 4, which selects by quantification error instead), with the class densities normalised on the simplex (§T.4). *Held fixed:* the Gaussian kernel, no shrinkage ($\lambda=0$), the prior $\pi\sim\operatorname{Dir}(\mathbf 1)$, $\tau=1$, the MLP of E3.1 (`MLPClassifier()` with its default settings and the `random_state` of `BayCAP-KDEy-mlp`, validation outputs cross-fitted on $5$ folds), the fixed rate matrix $\hat C$ of $h$ on $V$, NUTS with $500$ warm-up iterations, $1000$ draws and the seeds of `BayCAP-KDEy-mlp-real`, and the construction of the bags of `SQS-B` (E3.3).
  - **Measures:**
    - for the four methods, on the three metrics of `main` (vanilla accuracy, macro-F1, macro-K): AE, coverage, amplitude and Winkler score at $\alpha=0.05$, per bag, per pair and averaged over the pairs;
    - per pair and method metric, the difference of the mean Winkler score between the LOO variant and the stored one, with paired bootstrap tests as in E3.1 ($10{,}000$ seeded resamples of the $100$ per-bag differences, Holm-adjusted over the $75$ pairs) and a Wilcoxon signed-rank test over the $75$ per-pair means;
    - per pair and class, label-free: the selected $\hat b_j$, the class size $n_j$ in $V$, whether $\hat b_j$ is at an edge of $\mathcal G$ ($0.01$ or $1$), and the leave-one-out log-likelihood per instance at $\hat b_j$ and at $0.1$.
  - **Estimand:** realised table for all four methods (the realised pushforward for `BayCAP-KDEy-mlp-real(-loo)`, a simulated bag of $V$ for `SQS-B(-loo)`), scored against the realised table of each target bag (`true_cts`), as every method of `main`. The selection of $\hat b_j$ uses no target labels and no target bag.
  - **Cost:** a pilot on the $10$ pairs of `mammographic` and `wine-quality` (user's decision of 2026-11-19), then one run of `main` on `bcuda` with `JAX=cpu` for the two new methods: $2\times75\times100=15{,}000$ NUTS runs, as `BayCAP-KDEy-mlp-real` and `SQS-B` in E3.1 and E3.3; per pair, one fit of the MLP and one leave-one-out selection, which costs $O(n_j^2)$ per class for the pairwise distances and $21$ sums of exponentials over them, negligible against NUTS.
- **Representation.**
  - **Summary table:** one row per method (the two stored and the two LOO variants) and one column group per metric, with the means over the pairs of AE, coverage, amplitude and Winkler score, and, for each LOO variant against its stored counterpart, the number of pairs with a lower and with a higher mean Winkler score (Holm-adjusted $p<0.05$) and the Wilcoxon $p$-value.
  - **Per-pair table:** for vanilla accuracy, the mean Winkler score and coverage of the four methods per pair, and the median $\hat b_j$ over the classes of the pair.
  - **Bandwidth plot:** per dataset, the selected $\hat b_j$ of each class (one point per class and classifier, $\log_{10}$ scale) against $n_j$, with the grid edges and $b=0.1$ marked.
  - **Effect plot:** per pair, the difference of the mean Winkler score (LOO minus stored) against the median of $\log_{10}(\hat b_j/0.1)$, for vanilla accuracy.
- **Analysis.** Pilot only (user's decision of 2026-11-19: E7.1 closed after the pilot, without the full run). Run on `bcuda` on 2026-11-19 (`main --classic` for the two LOO methods, commit `e4ad02c`, `cap-ml` `93ae246`, JAX on CPU) on the $10$ pairs of `mammographic` ($K=2$) and `wine-quality` ($K=5$), the three metrics, no failed cell; tables and plots generated at commit `72e430a` (`tables/kdey_loo.py`, `plots/kdey_loo.py`), under `<OUT_DIR>/uncawe/tables/kdey_loo/` and `<OUT_DIR>/uncawe/plots/kdey_loo/`. Every score is against the realised table of the target bag. These are the two datasets where E3.1 found the realised BayCAP-KDEy failing (the rates of $h$ on `mammographic`, a bias shared by the quantifiers on `wine-quality`), so the pilot is not representative of the $75$ pairs.
  - **The selection picks small bandwidths, inside the grid.** The median $\hat b_j$ is $0.035$ on `mammographic` and $0.025$ on `wine-quality` (range $0.02$–$0.063$), no class at an edge of the grid; the selection depends only on $V$ and the MLP, so it is the same for the five classifiers of a dataset. The densities fit the validation outputs better than with $b=0.1$: the mean leave-one-out log-likelihood per instance rises from $0.03$ to $0.19$ on `mammographic` and from $3.07$ to $4.79$ on `wine-quality` (`kdey_loo_classes.csv`).
  - **Both methods get worse on every metric.** Means over the $10$ pairs (`kdey_loo_summary.csv`):

    | Metric | Method | AE | Coverage | Amplitude | Winkler | Pairs worse / better (Holm) | Wilcoxon $p$ |
    |---|---|---|---|---|---|---|---|
    | vanilla accuracy | `BayCAP-KDEy-mlp-real` | 0.081 | 0.101 | 0.074 | 1.923 | – | – |
    | | `BayCAP-KDEy-mlp-real-loo` | 0.125 | 0.022 | 0.060 | 3.862 | 10 / 0 | 0.002 |
    | | `SQS-B` | 0.090 | 0.060 | 0.070 | 2.311 | – | – |
    | | `SQS-B-loo` | 0.126 | 0.025 | 0.058 | 3.950 | 5 / 0 | 0.064 |
    | macro-F1 | `BayCAP-KDEy-mlp-real` | 0.070 | 0.244 | 0.090 | 1.292 | – | – |
    | | `BayCAP-KDEy-mlp-real-loo` | 0.100 | 0.044 | 0.075 | 2.626 | 10 / 0 | 0.002 |
    | | `SQS-B` | 0.068 | 0.250 | 0.085 | 1.316 | – | – |
    | | `SQS-B-loo` | 0.093 | 0.108 | 0.074 | 2.416 | 10 / 0 | 0.002 |
    | macro-K | `BayCAP-KDEy-mlp-real` | 0.088 | 0.317 | 0.158 | 1.714 | – | – |
    | | `BayCAP-KDEy-mlp-real-loo` | 0.097 | 0.200 | 0.135 | 2.019 | 6 / 0 | 0.002 |
    | | `SQS-B` | 0.084 | 0.333 | 0.140 | 1.615 | – | – |
    | | `SQS-B-loo` | 0.086 | 0.270 | 0.129 | 1.783 | 10 / 0 | 0.002 |

    With $10$ pairs, $p=0.002$ is the smallest two-sided $p$-value of the Wilcoxon test.
  - **On `wine-quality` the centre moves, not only the width.** For `BayCAP-KDEy-mlp-real` on vanilla accuracy, the AE rises from $0.086$–$0.145$ to $0.172$–$0.228$ on the five pairs, the amplitude falls from $0.083$–$0.112$ to $0.063$–$0.083$, and the coverage from $0.05$–$0.31$ to $0.00$–$0.04$. On `mammographic` the change is small (AE $+0.003$, Winkler score $+0.13$ to $+0.18$) and `SQS-B` is unchanged: its error comes from the rates of $h$ on $V$ (E3.1), which the bandwidth does not touch (`kdey_loo_pairs.csv`).
  - **Conclusion.** A better fit of the class densities does not give better intervals (§T.6): the leave-one-out bandwidths make the posterior of $\pi^Q$ narrower and, on `wine-quality`, move it away from the target prevalence. New hypotheses, not tested: sharper class densities make the likelihood of a bag of $1000$ outputs more peaked, so the posterior is overconfident; on `wine-quality` the small classes get larger bandwidths ($0.04$–$0.063$) than the large ones ($0.02$), which changes how the outputs in the overlap regions are attributed to the classes. The `"loo"` option stays in the code, not carried forward. The bandwidth is to be selected by a criterion on the error of the CAP estimate instead (user's decision of 2026-11-19).

## E7.2: One bandwidth per pair, selected on calibration bags drawn from $V$

**Why.** E7.1 showed that the bandwidth that best fits the class densities is not the one that gives the best intervals: on `wine-quality` the leave-one-out bandwidths moved the centre of `BayCAP-KDEy-mlp-real` away from the target. A bandwidth should rather be judged by the error it causes on what the method estimates. That error can be measured without target labels on labelled bags drawn from $V$, as E5.3 did for the temperature: $V$ is split in two halves, the method is fitted on one and scored on bags drawn from the other, so that the bags are new data for the fitted densities (§T.9). The score keeps only the part of the error of the estimated accuracy that the bandwidth can change (§T.7), and it needs only the centre of the posterior, which the maximum-likelihood estimate of the same model gives at almost no cost (§T.8). One bandwidth is shared by the classes: a grid search per class would need $21^K$ fits, and with one bandwidth the normalisation issue of §T.4 does not arise. Only `BayCAP-KDEy-mlp-real` is studied; `SQS-B`, which shares its quantifier, is left out for now (user's decision of 2026-11-19).

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs (datasets × the five classifiers, after FIX-8), $100$ target bags of size $1000$ per pair with the seeds of `main`; the stored `main` results of `BayCAP-KDEy-mlp-real` ($b=0.1$); the results of one new method of `main`, `BayCAP-KDEy-mlp-real-cal`, with its own run. For each pair, inside the fit of the method: $V$ split per class into $V_{\text{fit}}$ and $V_{\text{cal}}$, half each (stratified, seeded by the `random_state` of the method, so the same split for the five classifiers of a dataset); on $V_{\text{fit}}$, the MLP (cross-fitted validation outputs on $5$ folds, then trained on all of $V_{\text{fit}}$), the kernel densities at each $b$ of the grid, and the recalls $\hat d^{\text{fit}}$ of $h$; $S=50$ calibration bags of size $1000$ drawn with replacement from $V_{\text{cal}}$ at prevalences drawn uniformly on the simplex ($\operatorname{Dir}(\mathbf 1)$, QuaPy's UPP, as `main` draws its target bags; seeded, the same for every $b$), each with its true prevalence $\pi_s$ (the class proportions of the bag) and the outputs of the MLP of $V_{\text{fit}}$ on its instances. On small datasets each instance of $V_{\text{cal}}$ appears several times in a bag (`mammographic`: about $145$ instances).
  - **Varied:** the bandwidth of BayesKDEy, shared by the classes: $b=0.1$ (stored) against $b_{\text{cal}}=\arg\min_{b\in\mathcal G}\frac1S\sum_{s=1}^S\big\lvert\hat d^{\text{fit}\top}(\hat\pi_s(b)-\pi_s)\big\rvert$ over the grid $\mathcal G=\{10^{-2+0.1k}:k=0,\dots,20\}$ of E7.1, where $\hat\pi_s(b)$ is the maximum-likelihood estimate of the BayesKDEy model (fitted on $V_{\text{fit}}$ at bandwidth $b$) on the outputs of bag $s$ (§T.7, §T.8); ties are broken towards the value closest to $0.1$ on the $\log_{10}$ scale. The deployed method is BayesKDEy refitted on the full $V$ at $b_{\text{cal}}$, as a practitioner would do (E5.3); with $b_{\text{cal}}=0.1$ it is the stored method. *Held fixed:* the Gaussian kernel, no shrinkage, the prior $\operatorname{Dir}(\mathbf 1)$, $\tau=1$, the MLP input, the fixed $\hat C$ of $h$ on the full $V$, the realised pushforward, NUTS with $500$ warm-up iterations, $1000$ draws and the seeds of `BayCAP-KDEy-mlp-real`.
  - **Measures:**
    - on the target bags, for `BayCAP-KDEy-mlp-real` and `BayCAP-KDEy-mlp-real-cal`, on the three metrics of `main`: AE, coverage, amplitude and Winkler score at $\alpha=0.05$, per bag, per pair and averaged over the pairs; per pair, paired bootstrap tests of the mean per-bag difference of AE and of the Winkler score (cal minus stored; $10{,}000$ seeded resamples of the $100$ bags, Holm over the $75$ pairs, per metric and score), and Wilcoxon signed-rank tests over the $75$ per-pair means;
    - per pair, label-free on the target: $b_{\text{cal}}$, the calibration criterion over the whole grid, its value at $b_{\text{cal}}$ and at $0.1$, and the Spearman correlation across the pairs between the calibration gain (criterion at $0.1$ minus at $b_{\text{cal}}$) and the test gain (mean AE of the stored method minus that of the cal method, vanilla accuracy);
    - pilot only, the check of §T.8: on the pilot pairs, the calibration criterion also computed with the NUTS posterior mean of $\pi$ in place of the maximum-likelihood estimate, for every $b$ and bag; per pair, both selected bandwidths, the number of grid steps between them, and the largest difference between the two criterion curves.
  - **Estimand:** realised table for both methods on the target bags, scored against the realised table of each bag (`true_cts`), as every method of `main`. The calibration criterion scores the population accuracy $\hat d^{\text{fit}\top}\hat\pi_s(b)$ against $\hat d^{\text{fit}\top}\pi_s$, the part of the error of the centre that depends on $b$ (§T.7); it uses only the labels of $V$, so $b_{\text{cal}}$ is a method, not a reference.
  - **Cost:** per pair, inside the fit: one MLP on $V_{\text{fit}}$, $21$ kernel-density fits, the log-densities of the instances of $V_{\text{cal}}$ under each class density at each $b$ (computed once per $b$ and indexed by the bags), $21\times50$ maximum-likelihood estimates by EM; then the target run as `BayCAP-KDEy-mlp-real`, $75\times100=7500$ NUTS runs. The pilot check adds $21\times50$ NUTS runs per pilot pair ($15\times1050=15{,}750$). A pilot first on `mammographic`, `wine-quality` and `poker_hand` with the five classifiers ($15$ pairs; user's decision of 2026-11-19).
- **Representation.**
  - **Summary table:** per metric, one row per method, with the means over the pairs of AE, coverage, amplitude and Winkler score; for the cal method, the pairs where its AE and its Winkler score are significantly (Holm) lower and higher than the stored ones and the two Wilcoxon $p$-values; also split by $K$ ($K=2$, $3\le K\le7$, $K\ge8$).
  - **Per-pair table:** per metric, one row per pair: $K$, $b_{\text{cal}}$, the calibration criterion at $0.1$ and at $b_{\text{cal}}$, the test AE, coverage and Winkler score of both methods, and the two differences (bold where Holm $p<0.05$).
  - **Bandwidth plot:** $b_{\text{cal}}$ per pair ($\log_{10}$ scale), one column per dataset ordered by $K$, markers by classifier, with $b=0.1$ and the edges of the grid marked.
  - **Gain plot:** per pair, the test gain against the calibration gain (vanilla accuracy), coloured by $K$.
  - **Pilot check table:** per pilot pair, $b_{\text{cal}}$ from the maximum-likelihood estimate and from the NUTS posterior mean, the steps between them and the largest difference of the criterion curves.
- **Analysis.** Pilot only so far; the full run is not done. Run on `bcuda` on 2026-11-19 (`main --classic --methods BayCAP-KDEy-mlp-real-cal --kdey-cal-check`, commit `520580c`, `cap-ml` `93ae246`, JAX on CPU, 16:01–20:41) on the $15$ pairs of `mammographic` ($K=2$), `wine-quality` ($K=5$) and `poker_hand` ($K=7$) with the five classifiers, the three metrics, no failed cell; divergent NUTS transitions on at most $1\%$ of the target bags of a pair. Tables generated locally on 2026-11-20 after `make pull-results` (`tables/kdey_cal.py --tables`, code identical to `devel` `7b465eb`), under `<OUT_DIR>/uncawe/tables/kdey_cal/`; the plots are not generated yet. Every test score is against the realised table of the target bag; the criterion uses only the labels of $V$. `mammographic` and `wine-quality` are datasets where E3.1 found the realised BayCAP-KDEy failing, so the pilot is not representative of the $75$ pairs.
  - **The selection moves away from $0.1$ on $13$ of $15$ pairs, inside the grid.** $b_{\text{cal}}=0.316$ on the five `mammographic` pairs ($5$ grid steps above $0.1$), $0.079$ or $0.1$ on `wine-quality` ($-1$ or $0$ steps), $0.050$ on four `poker_hand` pairs and $0.020$ on `poker_hand` with LR ($-3$ and $-7$ steps); no pair at an edge of the grid. The criterion curve is almost flat on `mammographic` (at $0.1$ and at $b_{\text{cal}}$: $0.0020$ and $0.0018$ with LR), shallow on `wine-quality`, and steep on `poker_hand` (with LR: $0.385$ at $0.1$, $0.224$ at $0.020$) (`kdey_cal_pairs.csv`, `kdey_cal_curves.csv`).
  - **The check of §T.8: the maximum-likelihood estimate selects the same bandwidth as the NUTS posterior mean on $14$ of $15$ pairs.** At the selected bandwidth the two curves differ by less than $0.002$ on `mammographic`, about $0.003$ on `wine-quality` and $0.004$–$0.012$ on `poker_hand`. They part at large bandwidths: above about $b=0.4$ the criterion with the posterior mean falls (on `poker_hand` with LR from $0.414$ to $0.214$), while the one with the maximum-likelihood estimate keeps rising on `poker_hand` and falls only slightly on `wine-quality`; the largest difference is $0.03$–$0.06$ on `wine-quality` and $0.21$–$0.41$ on `poker_hand`. On `poker_hand` with LR this decides the selection: the posterior-mean curve reaches $0.214$ at $b=1$ (the edge of the grid), just below its local minimum $0.220$ at $b=0.020$, where the maximum-likelihood selection stays ($17$ steps apart; `kdey_cal_check.csv`). New hypothesis, not tested: at large $b$ the class densities overlap, the likelihood of a bag is nearly flat, and the posterior mean is pulled towards the centre of the prior $\operatorname{Dir}(\mathbf 1)$, the uniform $\pi$; the calibration prevalences are drawn from the same $\operatorname{Dir}(\mathbf 1)$, so a centre near $1/K$ scores well on them, while the posterior, and so the interval, is wide. The criterion scores only the centre and does not see this; the maximum-likelihood estimate, which has no prior, is not affected. The failure mode anticipated in §T.8 (the maximum on the boundary of the simplex) does not change the selection on these pairs.
  - **EM does not always converge.** Of the $21\times50=1050$ EM fits of a pair, none reach the cap of $10^4$ steps on `mammographic`, $25$ on `wine-quality` and $177$ ($17\%$) on `poker_hand` (the same count for the five classifiers of a dataset, since the split, the MLP and the bags do not depend on $h$). The stored count does not say at which bandwidths; the unconverged fits enter the criterion with their last iterate.
  - **The calibrated bandwidth lowers the AE on every pair where it moves, and worsens no pair.** Means over the $15$ pairs (`kdey_cal_summary.csv`; lower/higher: pairs with Holm-adjusted $p<0.05$ in the paired bootstrap test of cal minus stored):

    | Metric | Method | AE | Coverage | Amplitude | Winkler | AE lower / higher | Winkler lower / higher | Wilcoxon $p$ (AE, Winkler) |
    |---|---|---|---|---|---|---|---|---|
    | vanilla accuracy | `BayCAP-KDEy-mlp-real` | 0.088 | 0.111 | 0.073 | 2.128 | – | – | – |
    | | `BayCAP-KDEy-mlp-real-cal` | 0.082 | 0.169 | 0.075 | 1.927 | 11 / 0 | 10 / 0 | 0.001, 0.001 |
    | macro-F1 | `BayCAP-KDEy-mlp-real` | 0.059 | 0.251 | 0.074 | 1.089 | – | – | – |
    | | `BayCAP-KDEy-mlp-real-cal` | 0.051 | 0.304 | 0.077 | 0.868 | 9 / 0 | 6 / 0 | 0.002, 0.055 |
    | macro-K | `BayCAP-KDEy-mlp-real` | 0.064 | 0.545 | 0.181 | 1.218 | – | – | – |
    | | `BayCAP-KDEy-mlp-real-cal` | 0.062 | 0.550 | 0.163 | 1.153 | 8 / 0 | 8 / 0 | 0.006, 0.001 |

    On the two `wine-quality` pairs with $b_{\text{cal}}=0.1$ the two methods coincide bag by bag, as designed. The gains are small against the error left: on vanilla accuracy the mean AE falls by $0.006$ on `mammographic` ($0.054\to0.048$), $0.003$ on `wine-quality` and $0.008$ on `poker_hand` (largest pair: `poker_hand` with LR, $-0.028$), and the coverage stays far below $0.95$ ($0.02\to0.15$, $0.18\to0.19$, $0.13\to0.17$). The largest change is on `mammographic` with macro-F1 (AE $0.082\to0.061$, coverage $0.03\to0.21$, Winkler $1.71\to1.07$). The amplitude changes little except on `poker_hand` with macro-K ($0.227\to0.173$).
  - **The calibration gain predicts the test gain within a dataset, not across datasets.** Spearman correlation between the calibration gain (criterion at $0.1$ minus at $b_{\text{cal}}$) and the test gain (vanilla accuracy): $0.18$ ($p=0.52$) over the $15$ pairs, $0.74$ ($p=0.014$) over the $10$ pairs with $3\le K\le7$. The scales differ between datasets: on `mammographic` the calibration gain is about $2\cdot10^{-4}$ and the test gain about $0.006$, $30$ times larger; on `poker_hand` the calibration gain is $0.05$–$0.16$ and the test gain $0.002$–$0.028$. New hypothesis, not tested: on `mammographic`, where E3.1 traced the error to the rates of $h$ on $V$, the larger bandwidth moves $\hat\pi$ in a direction that partly offsets the error of the rates on these target bags, the compensation that the criterion is built to leave out (§T.7), so the gain there would not come from a better prevalence estimate.
  - **Conclusion of the pilot.** The fast selection holds its main hypothesis (the maximum-likelihood estimate selects as the posterior mean, except where the posterior is dominated by the prior) and improves the centre and the Winkler score of `BayCAP-KDEy-mlp-real` on every pair it changes, by small amounts; the intervals still undercover. Open before the full run: at which bandwidths EM hits the cap of $10^4$ steps on `poker_hand`.

# Candidate experiments

None for now (user's decision of 2026-11-19).

# Theory

This part explains what E7.1 (§T.1–§T.6) and E7.2 (§T.7–§T.9) need, step by step. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$ and $T=C\operatorname{diag}(\pi)$ the population table, with predicted classes on rows and true classes on columns. The bandwidth enters only the posterior of $\pi^Q$; the bridge from $\pi$ to the table (the fixed $\hat C$ and the realised pushforward for BayCAP-KDEy, the forward bag for SQS-B) is that of [[ext3_posterior_propagation]] and is unchanged.

## T.1 BayesKDEy and the role of the bandwidth

**The data.** The input classifier (here an MLP trained on $V$) maps an instance $x$ to a vector of class probabilities $z=(z_1,\dots,z_K)$, with $z_k\ge0$ and $z_1+\dots+z_K=1$. On $V$ the outputs are cross-fitted (each instance is scored by an MLP that did not see it), so they behave like outputs on new data. Write $z_{j,1},\dots,z_{j,n_j}$ for the validation outputs of the $n_j$ instances of class $\omega_j$, and $z_1,\dots,z_N$ for the outputs on the $N$ instances of the target bag.

**The class densities.** Under PPS the distribution of $z$ within a class is the same in $V$ and in the target. BayesKDEy estimates it by a kernel density: it places a small Gaussian bump of width $b$ on each validation output of the class and averages them,

$$\hat p_j(z)=\frac{1}{n_j}\sum_{i=1}^{n_j}\varphi_b(z-z_{j,i}),\qquad \varphi_b(u)=\frac{1}{(2\pi b^2)^{d/2}}\exp\!\Big(-\frac{\lVert u\rVert^2}{2b^2}\Big),$$

where $\varphi_b$ is the density of a Gaussian with independent coordinates of standard deviation $b$ in $d$ dimensions (§T.4 discusses which $d$).

**The posterior.** An output of the target bag comes from class $\omega_j$ with probability $\pi_j$ and then has density $\hat p_j$, so its density is the mixture $\sum_j\pi_j\hat p_j(z)$. The outputs are independent given $\pi$, so the likelihood of the bag is the product over its instances, and with the prior $\operatorname{Dir}(\mathbf 1)$ and the temperature $\tau$ the posterior is

$$p(\pi\mid z_1,\dots,z_N)\propto\Big(\prod_{u=1}^N\sum_{j=1}^K\pi_j\hat p_j(z_u)\Big)^{1/\tau},$$

which NUTS samples (`methods/baycap_kdey.py:kdey_model`).

**What $b$ does.** Two extremes make the effect visible.

- If $b$ is very small, $\hat p_j(z)$ is large only very close to a validation output of class $\omega_j$. An output of the bag that lies near validation outputs of one class only is assigned to that class with near certainty, as if the classes were perfectly separated, and the posterior of $\pi$ is narrow. An output that lies near no validation output of any class has a tiny density under every class, and the classes it would most plausibly belong to are decided by the few nearest validation outputs.
- If $b$ is very large, every bump covers the whole simplex, so every $\hat p_j$ is close to the same flat function. The factor $\sum_j\pi_j\hat p_j(z_u)$ then barely depends on $\pi$, the likelihood is flat, and the posterior stays close to the prior: wide and centred near $1/K$.

Between the two, $b$ trades the detail of the class densities (small $b$) against their stability (large $b$). The best $b$ depends on $n_j$ (more points support a smaller $b$) and on how spread the outputs of the class are, which is why one value for every class and dataset is unlikely to be the best.

## T.2 Why the likelihood is computed leaving one out

**The goal.** We want the $b$ whose $\hat p_j$ is closest to the true density $p_j$ of the outputs of class $\omega_j$. A standard measure of closeness is the Kullback–Leibler divergence

$$\operatorname{KL}(p_j\,\Vert\,\hat p_j)=\int p_j(z)\ln\frac{p_j(z)}{\hat p_j(z)}\,dz=\underbrace{\int p_j(z)\ln p_j(z)\,dz}_{\text{does not depend on }b}-\int p_j(z)\ln\hat p_j(z)\,dz .$$

It is $0$ only when $\hat p_j=p_j$ and positive otherwise. The first term does not involve $\hat p_j$, so minimising the divergence over $b$ is the same as maximising the second integral, $\int p_j\ln\hat p_j=\mathbb E[\ln\hat p_j(Z)]$, the expected log-density that the estimate gives to a *new* output $Z$ of class $\omega_j$.

**Why not the ordinary likelihood.** The obvious estimate of $\mathbb E[\ln\hat p_j(Z)]$ is the average of $\ln\hat p_j$ over the validation outputs themselves. It fails because each $z_{j,i}$ was used to build $\hat p_j$: the sum defining $\hat p_j(z_{j,i})$ contains the term $l=i$, $\varphi_b(0)/n_j=1/\big(n_j(2\pi b^2)^{d/2}\big)$, which grows without limit as $b\to0$. The ordinary likelihood is therefore maximised by $b\to0$, the spiky density that memorises the data.

**Leaving one out.** The fix is to evaluate each output under the density built without it:

$$\hat p_{j,-i}(z_{j,i})=\frac{1}{n_j-1}\sum_{l\ne i}\varphi_b(z_{j,i}-z_{j,l}),\qquad \operatorname{LOO}_j(b)=\sum_{i=1}^{n_j}\ln\hat p_{j,-i}(z_{j,i}).$$

Since $z_{j,i}$ is independent of the other outputs, $z_{j,i}$ is a genuinely new point for $\hat p_{j,-i}$, so $\operatorname{LOO}_j(b)/n_j$ is an honest estimate of the expected log-density of a new output (for a density built on $n_j-1$ points instead of $n_j$, a negligible difference). The selected bandwidth is $\hat b_j=\arg\max_{b\in\mathcal G}\operatorname{LOO}_j(b)$.

**Computing it.** Scikit-learn can do the same with `GridSearchCV(KernelDensity(), cv=LeaveOneOut())`, which refits the density $n_j$ times per grid value and costs about $n_j^3$ operations per value. The formula above needs only the squared distances $D_{il}=\lVert z_{j,i}-z_{j,l}\rVert^2$, computed once; for each $b$, $\operatorname{LOO}_j(b)$ is a sum over the matrix of $\exp(-D_{il}/2b^2)$ without its diagonal. The two give the same numbers; the second costs about $n_j^2$ operations per grid value.

## T.3 One bandwidth per class

With one bandwidth $b_j$ per class, the total leave-one-out log-likelihood of the class-conditional model is $\sum_j\operatorname{LOO}_j(b_j)$. The term of class $\omega_j$ depends only on $b_j$, because $\hat p_{j,-i}$ is built from the outputs of class $\omega_j$ alone. Maximising a sum of terms that each depend on a different variable is the same as maximising each term separately, so the joint selection splits into $K$ searches over $\mathcal G$, one per class.

A class with $n_j=1$ has no leave-one-out density ($n_j-1=0$); it keeps $b_j=0.1$, and this is recorded. With few instances, $\hat b_j$ rests on few distances and is noisy; E7.1 reports $\hat b_j$ against $n_j$ for this reason.

## T.4 With per-class bandwidths the kernel must be normalised on the simplex

**The issue.** QuaPy's class densities are scikit-learn's `KernelDensity` on the $K$ columns of the outputs, so they use the Gaussian in $d=K$ dimensions. But the outputs are not spread in $K$ dimensions: they satisfy $z_1+\dots+z_K=1$, so they lie on a flat surface of dimension $K-1$ (a line for $K=2$, a plane for $K=3$).

**The factor.** Take orthonormal coordinates on that surface, $K-1$ of them, plus one coordinate perpendicular to it. Distances are unchanged by a change of orthonormal coordinates, and for two points on the surface the perpendicular coordinate is the same, so their difference has a perpendicular component of $0$. The $K$-dimensional Gaussian then factorises into the $(K-1)$-dimensional Gaussian along the surface times the one-dimensional Gaussian density at $0$ in the perpendicular direction:

$$\varphi_b^{(K)}(z-z')=\frac{1}{(2\pi b^2)^{1/2}}\,\varphi_b^{(K-1)}(z-z').$$

The $(K-1)$-dimensional Gaussian integrates to $1$ over the surface; the $K$-dimensional one, restricted to the surface, integrates to $c(b)=(2\pi b^2)^{-1/2}$. So QuaPy's $\hat p_j$ is the correct density on the surface multiplied by $c(b)$.

**With one shared $b$ the factor cancels.** Every $\hat p_j$ carries the same $c(b)$, so each factor of the likelihood becomes $c(b)\sum_j\pi_j\hat p_j(z_u)$ and the likelihood is multiplied by $c(b)^N$, a constant that does not depend on $\pi$. The posterior is proportional to the likelihood times the prior, so a constant factor disappears in the normalisation: the methods of `main` with $b=0.1$ are correct.

**With per-class $b_j$ it does not.** The factor of the bag becomes $\sum_j\pi_j\,c(b_j)\,\hat p_j(z_u)$: the class densities are no longer on the same scale, and a class with a smaller $b_j$ gets its density multiplied by a larger $c(b_j)$, by the ratio $c(b_j)/c(b_k)=b_k/b_j$ against class $\omega_k$. This acts like a change of the prevalences, $\pi_j\mapsto\pi_jc(b_j)$ up to normalisation, and pulls the posterior towards the classes with small bandwidths. The selection is biased in the same direction: with the $K$-dimensional Gaussian,

$$\operatorname{LOO}^{(K)}_j(b)=\operatorname{LOO}^{(K-1)}_j(b)-\frac{n_j}{2}\ln(2\pi b^2),$$

and the extra term grows as $b$ shrinks, so it favours small bandwidths: going from $b=0.1$ to $b=0.01$ adds $n_j\ln10\approx2.3\,n_j$ to the criterion (about $1150$ for $n_j=500$) for no reason related to the fit.

**Consequence for E7.1.** Both the selection and the class densities of the LOO methods use the Gaussian in $d=K-1$ dimensions, i.e. the log-density of `KernelDensity` plus $\tfrac12\ln(2\pi b_j^2)$. The methods with a shared $b=0.1$ keep QuaPy's densities, so the stored results stay valid.

**A caveat that remains.** The Gaussian bumps are not confined to the simplex: a bump centred near an edge or a vertex puts part of its mass outside, where no output can fall. The lost mass is larger for a larger $b_j$ and for outputs near the edges, which is where the outputs of a confident MLP lie. With one shared $b$ this loss is similar across classes; with per-class $b_j$ it can differ, leaving some of the scale difference that §T.4 removes. The BayesKDEy paper notes the same limitation of Gaussian kernels on compositional data and proposes the Aitchison and ILR kernels with shrinkage; they are not used here.

## T.5 Where the selection can hit the edges of the grid

Look at one term of $\operatorname{LOO}_j(b)$ when $b$ is small. Let $\delta_i$ be the distance from $z_{j,i}$ to its nearest other output of the class. For small $b$ the sum is dominated by that nearest output, so

$$\ln\hat p_{j,-i}(z_{j,i})\approx-\frac{\delta_i^2}{2b^2}-\frac{K-1}{2}\ln(2\pi b^2)-\ln(n_j-1).$$

If $\delta_i>0$, the first term goes to $-\infty$ faster than the second goes to $+\infty$, so the term goes to $-\infty$ as $b\to0$. For large $b$, every distance is negligible against $b$ and every term behaves like $-\frac{K-1}{2}\ln(2\pi b^2)$, which also goes to $-\infty$. So, as long as some outputs of the class are distinct, the maximum is at an interior $b$, of the order of the typical distance between neighbouring outputs.

Two situations push $\hat b_j$ to the edges of $\mathcal G$:

- **Lower edge.** A confident MLP maps most instances of a class to almost the same vertex of the simplex, so the $\delta_i$ are tiny (e.g. $10^{-4}$) and the best $b$ is far below $0.01$. If every output of the class had an exact duplicate, $\operatorname{LOO}_j(b)$ would even grow without limit as $b\to0$. The lower bound $0.01$ stops the selection there.
- **Upper edge.** A class with few, widely spread outputs can prefer the widest bumps.

E7.1 reports the share of classes at each edge. Many classes at the lower edge would mean that the outputs of the MLP are too concentrated for a Gaussian kernel to separate their spread from noise at this scale.

## T.6 What E7.1 can and cannot show

The leave-one-out likelihood selects the bandwidth that best fits the class densities. The CAP methods are scored on their intervals for the realised table. The two goals are related, since the posterior of $\pi^Q$ is built from the class densities, but they are not the same: a better density fit could still give a posterior that is too narrow or too wide for the target bag, and the error that comes from $V$ itself (the rates of $h$, Extension 4) does not depend on $b$. E7.1 therefore measures whether a label-safe density criterion improves the intervals; it does not search for the bandwidth that would give the best intervals, which would need a different, quantification-oriented criterion.

## T.7 The error that the bandwidth controls, and the calibration criterion

**The centre of the method.** With a fixed $\hat C$, the population accuracy of a draw $\pi$ is $\hat d^\top\pi=\sum_j\hat C_{jj}\pi_j$, where $\hat d_j=\hat C_{jj}$ is the recall of $h$ on class $\omega_j$ measured on $V$. The realised pushforward adds the luck of the bag around it but, as E3.1 found, barely moves the centre, so the point estimate of `BayCAP-KDEy-mlp-real` is close to $\hat d^\top\bar\pi$, with $\bar\pi$ the posterior mean.

**Three parts of the error.** Let $\pi$ be the true prevalence of a bag (its class proportions), $d$ the true recalls of $h$, and $A$ the accuracy of $h$ on the realised table of the bag. Adding and subtracting $\hat d^\top\pi$ and $d^\top\pi$,

$$\hat d^\top\bar\pi-A=\underbrace{\hat d^\top(\bar\pi-\pi)}_{\text{(i) quantifier}}+\underbrace{(\hat d-d)^\top\pi}_{\text{(ii) rates of }h\text{ on }V}+\underbrace{d^\top\pi-A}_{\text{(iii) luck of the bag}}.$$

The bandwidth enters only through $\bar\pi$, so only term (i) depends on $b$. Term (ii) depends on how far the recalls measured on $V$ are from the true ones, and term (iii) on which instances the bag happens to contain (Extension 4 calls them $e_2$ and $e_3$).

**Only the differences between recalls matter.** The prevalences sum to $1$, so $\sum_j(\bar\pi_j-\pi_j)=0$ and, for any number $c$, $c\mathbf 1^\top(\bar\pi-\pi)=0$. Hence

$$\hat d^\top(\bar\pi-\pi)=(\hat d-c\mathbf 1)^\top(\bar\pi-\pi)\quad\text{for every }c.$$

An error that moves prevalence between two classes of equal recall does not change the accuracy; one between a class of recall $0.9$ and one of recall $0.5$ counts with weight $0.4$. This is why a score on the prevalences, such as $\lVert\bar\pi-\pi\rVert_1$, would not target the accuracy: it weights every direction of the error equally (user's question of 2026-11-19).

**Why terms (ii) and (iii) are left out of the criterion.** On calibration bags drawn from $V_{\text{cal}}$, the accuracy $A_s$ is known, so the plain accuracy error $\lvert\hat d^{\text{fit}\top}\bar\pi_s-A_s\rvert$ could be used. But its term (ii) is $(\hat d^{\text{fit}}-d)^\top\pi_s$, where the recalls measured on $V_{\text{fit}}$ differ from the true ones by an amount that is about the same for every bag: call $\delta$ this common offset. Then each bag contributes $\lvert q_s(b)+\delta\rvert$, with $q_s(b)$ the term (i) of bag $s$, and the mean of $\lvert q_s(b)+\delta\rvert$ is smallest for a $b$ whose term (i) is close to $-\delta$ on average: the selection would choose a bandwidth whose quantifier error cancels the error of the recalls of $V_{\text{fit}}$. That error is a property of the half $V_{\text{fit}}$, not of the full $V$ on which the deployed method is fitted, so the cancellation would not carry over (the same reason as §T.9 of [[ext5_baycap_priors_tempering]]: the calibration bags cannot see the error that $V$ itself causes). Term (iii) does not depend on $b$ either; it only adds noise. The criterion therefore keeps term (i) alone, with the recalls of $V_{\text{fit}}$ and the true prevalences of the bags:

$$\mathrm{crit}(b)=\frac1S\sum_{s=1}^S\big\lvert\hat d^{\text{fit}\top}(\hat\pi_s(b)-\pi_s)\big\rvert,\qquad b_{\text{cal}}=\arg\min_{b\in\mathcal G}\mathrm{crit}(b).$$

In words: if the recalls of $h$ were exactly those measured, how much accuracy error would the quantifier cause at this bandwidth? The same construction gives a criterion for the other metrics, $\lvert A(\hat C\operatorname{diag}(\hat\pi_s))-A(\hat C\operatorname{diag}(\pi_s))\rvert$; E7.2 selects on vanilla accuracy only and reports the other metrics.

## T.8 The maximum-likelihood estimate in place of the posterior mean

**The approximation.** The criterion needs only the centre $\bar\pi_s(b)$ of the posterior of each bag, for $21$ bandwidths and $50$ bags per pair, about $1000$ posteriors per pair. With the uniform prior $\operatorname{Dir}(\mathbf 1)$, whose density is constant on the simplex, the posterior is proportional to the likelihood, so its highest point is the maximum-likelihood estimate $\hat\pi_s(b)$. For a bag of $N=1000$ outputs the posterior is concentrated: by the usual large-sample argument it is close to a Gaussian centred at $\hat\pi_s$ with a spread of order $1/\sqrt N$, so its mean is close to $\hat\pi_s$. E7.2 uses $\hat\pi_s(b)$ in the criterion. This is a new hypothesis, checked on the pilot (Measures).

**Where it can fail.** The argument needs the maximum inside the simplex. Bags drawn from $\operatorname{Dir}(\mathbf 1)$ often have a class with a very small share, and then the maximum can sit on the boundary ($\hat\pi_{s,j}=0$), while the posterior mean, an average over the simplex, stays strictly positive. The two then differ by about the spread of the posterior in that direction, and the selected bandwidths may differ. The pilot reports how often they do.

**Computing the estimate.** The log-likelihood $\ell(\pi)=\sum_u\ln\sum_j\pi_j\hat p_j(z_u)$ is a sum of logarithms of linear functions of $\pi$, so it is concave, and its maximum over the simplex (a convex set) is a single connected set of points with the same value. The EM algorithm reaches it: given the current $\pi$, it computes for each output the probabilities $r_{uj}=\pi_j\hat p_j(z_u)/\sum_k\pi_k\hat p_k(z_u)$ that it comes from each class, and sets the new $\pi_j$ to their average $\frac1N\sum_ur_{uj}$. Each step increases $\ell$, and the steps stop changing $\pi$ only at the maximum. The class densities $\hat p_j(z_u)$ are computed once per bandwidth for the instances of $V_{\text{cal}}$, so the cost is a few matrix products per bag.

## T.9 Calibrating on half of $V$

**Why the split.** If the calibration bags were drawn from the same instances on which the class densities are fitted, each output of a bag would sit on top of its own bump, and small bandwidths would look better than they are, for the reason that the ordinary likelihood prefers $b\to0$ (§T.2). Drawing the bags from $V_{\text{cal}}$, which the fitted densities have not seen, removes this.

**The bandwidth of half the data.** The deployed method is fitted on all of $V$, twice the data of the calibration fit, so the best bandwidth for it is somewhat smaller. A standard property of kernel density estimation gives the size of the effect. The mean integrated squared error of a kernel density on $n$ points in $d$ dimensions is approximately $A\,b^4+B/(nb^d)$, a bias term that grows with $b$ and a variance term that shrinks with $b$ and with $n$ ($A,B>0$ depend on the density and the kernel, not on $n$ and $b$). Setting its derivative with respect to $b$ to zero,

$$4A\,b^3-\frac{dB}{n\,b^{d+1}}=0\quad\Longrightarrow\quad b^{d+4}=\frac{dB}{4A\,n}\quad\Longrightarrow\quad b\propto n^{-1/(d+4)}.$$

Halving $n$ multiplies the best bandwidth by $2^{1/(d+4)}$; with $d=K-1$, this is $1.15$ for $K=2$, $1.09$ for $K=5$ and $1.02$ for $K=26$, less than one step of the grid ($10^{0.1}\approx1.26$). The bias is noted and not corrected: $b_{\text{cal}}$ may be about one step larger than the best bandwidth for the full $V$ on the binary datasets.

**Small $V_{\text{cal}}$.** A calibration bag has $1000$ instances drawn with replacement from $V_{\text{cal}}$, so on small datasets the same instances recur in many bags, and the $50$ bags carry less independent information than their size suggests. The criterion is still an average over prevalences that cover the simplex, which is what it is meant to probe.

# Implementation

Components:

- **Quantifier classes** (user's decision of 2026-11-19): BayesKDEy and BayesianMAPLS become quantifier classes of their own (`BayesKDEy`, `BayesMAPLS`, on a base `BayesianQuantifier`), used by `BayCAPKDEy`, `BayCAPMAPLS` and `SQSB`, which no longer borrows a BayCAP method; the restructure changes no result (checked by a golden test).
- **Per-class bandwidths:** the bandwidth is a hyperparameter of `BayesKDEy`, a positive number, `"scott"`, `"silverman"` or `"loo"`, passed through by `BayCAPKDEy` and `SQSB`. `"loo"` needs the Gaussian kernel and no shrinkage (and `c_prior="fixed"` in `BayCAPKDEy`), with a grid `bandwidth_grid` (default `LOO_GRID`, $\{10^{-2+0.1k}\}_{k=0}^{20}$). With `"loo"`, `_SelectedKernelDensities` selects $\hat b_j$ per class by the closed-form leave-one-out log-likelihood of §T.2 in $d=K-1$ dimensions (squared distances computed in chunks to bound memory on large classes), fits one `KernelDensity` per class with its $\hat b_j$, and returns log-densities corrected by $+\tfrac12\ln(2\pi\hat b_j^2)$ (§T.4); a class with $n_j<2$ keeps $0.1$. The selected bandwidths are stored (`bandwidths_`) and written per bag. Numeric bandwidths and `"scott"`/`"silverman"` keep QuaPy's densities unchanged.
- **E7.1 in `main`:** two factory entries in `experiments/methods.py:LOO_VARIANTS`, `BayCAP-KDEy-mlp-real-loo` and `SQS-B-loo` (with the configuration and seed of `BayCAP-KDEy-mlp-real`, so the same selection and draws), excluded from the tables, ranks and default plots of `main`; per bag, the extra columns of the selection (`bandwidths`, `class_sizes`, `bandwidth_edge`, `loo_selected`, `loo_default`); the tables `tables/kdey_loo.py` and the plots `plots/kdey_loo.py`.

- **E7.2, the selection:** `bandwidth="cal"` in `BayCAPKDEy` (with `c_prior="fixed"`, the Gaussian kernel and no shrinkage only), with the grid `bandwidth_grid` (default `LOO_GRID`), the number of calibration bags (default $50$) and the bag size (default $1000$). Its `fit` splits $V$ per class into halves with a generator seeded by `random_state`, fits the MLP on $V_{\text{fit}}$ (cross-fitted outputs), measures $\hat d^{\text{fit}}$, draws the calibration bags from $V_{\text{cal}}$ with QuaPy's UPP and its own seed, computes for each $b$ the log-densities of the instances of $V_{\text{cal}}$, the EM estimates of the bags (§T.8) and the criterion (§T.7), selects $b_{\text{cal}}$, and then fits the deployed BayesKDEy on the full $V$ at $b_{\text{cal}}$ exactly as the stored method does, so that the calibration consumes no random state of the deployed fit and $b_{\text{cal}}=0.1$ gives the draws of the stored method. The EM estimate runs on the instance counts of each bag (`methods/bayes_kdey.py:em_prevalences`). A diagnostic option, for the pilot only (`cal_nuts_check`, turned on by `main --kdey-cal-check`), also computes the criterion with the NUTS posterior mean (§T.8). Per bag, `main` stores $b_{\text{cal}}$, the criterion curve, its values at $b_{\text{cal}}$ and at $0.1$, the number of EM estimates that did not converge, $\lvert V\rvert$ and $\lvert V_{\text{cal}}\rvert$ (and, with the diagnostic, the NUTS curve and the bandwidth it selects).
- **E7.2 in `main`:** the factory entry `BayCAP-KDEy-mlp-real-cal` in `experiments/methods.py`, with the configuration and seed of `BayCAP-KDEy-mlp-real`, excluded from the tables, ranks and default plots of `main`; the tables `tables/kdey_cal.py` and the plots `plots/kdey_cal.py`, following [[work_plan]] Part V.

Tests:

- the closed-form leave-one-out log-likelihood equals the brute-force one (one `KernelDensity` per left-out point) on a small random set, with the $(K-1)$-dimensional normalisation;
- on simplex data with $K=2$ and $K=3$, the corrected class density integrates to $1$ over the simplex surface up to the mass outside the simplex (checked numerically on data far from the edges);
- the selection returns a known optimum on a toy set, and $0.01$ when the outputs of a class are all duplicated;
- with a numeric `bandwidth`, the log-densities and the posterior draws are identical to those of the current code (stored results stay comparable);
- `SQS-B-loo` and `BayCAP-KDEy-mlp-real-loo` share their prevalence draws for the same seeds, as `SQS-B` and `BayCAP-KDEy-mlp-real`;
- E7.2: the split is stratified, disjoint and covers $V$; the calibration bags are reproducible for a seed and the same for every $b$; the EM estimate equals the maximiser of the log-likelihood found by a generic optimiser on a toy mixture; the criterion is invariant to adding a constant to the recalls (§T.7); with a grid of the single value $0.1$, `BayCAP-KDEy-mlp-real-cal` gives exactly the draws and tables of `BayCAP-KDEy-mlp-real`; the selection uses no target bag.

Reproducibility and comparability:

- The selection is deterministic given the cross-fitted outputs, which are seeded as in E3.1; the seeds of NUTS and of the bags are those of the stored methods, so the LOO variants differ from them only through the bandwidths.
- The stored results of `main` are unchanged: the new methods add rows to `main`, and the shared-bandwidth path is not modified.
- E7.2: the calibration is seeded by the `random_state` of the method and runs before the deployed fit without consuming its random state, so on the pairs where $b_{\text{cal}}=0.1$ the cal method reproduces the stored method bag by bag, and elsewhere it differs from it only through the bandwidth.

# Appendix

## A. Toy simulations

None.

## B. Relation to other extensions

- **[[ext3_posterior_propagation|Extension 3]]**: E7.1 and E7.2 change only the bandwidth of the quantifier of `BayCAP-KDEy-mlp-real` (E3.1) and, in E7.1, of `SQS-B` (E3.3); the bridge from $\pi$ to the table is unchanged.
- **[[ext5_baycap_priors_tempering|Extension 5]], E5.3**: E7.2 reuses its calibration design (half of $V$, $50$ bags from $\operatorname{Dir}(\mathbf 1)$, the deployed method refitted on the full $V$) for the bandwidth instead of the temperature, with a criterion on the centre instead of the Winkler score.
- **[[ext5_baycap_priors_tempering|Extension 5]]**: the prior on $\pi^Q$ and the temperature $\tau$ are held at $\operatorname{Dir}(\mathbf 1)$ and $1$, the values kept by Extension 5.
- **[[ext4_uncertainty_budget|Extension 4]]**: the error coming from the rates of $h$ on $V$ does not depend on $b$ (§T.6).
- **History.** Extension 7 was a factorial grid-search study of $h$, the surrogate classifier and the quantifier; it was narrowed to the bandwidth of BayesKDEy on 2026-11-19 (user's decision), and the earlier scope is set aside in [[optional_extensions]] §13. E7.1 (per-class leave-one-out bandwidths) was closed after its pilot on 2026-11-19 and E7.2 added the same day (user's decisions).

## C. Out of scope

- the factorial optimisation of $h$, of surrogate classifiers and of other quantifiers ([[optional_extensions]] §13);
- selection of the bandwidth by the prevalence error, as in the BayesKDEy paper (§T.7 explains why E7.2 scores the accuracy instead), per-class bandwidths selected on calibration bags, the Aitchison and ILR kernels and shrinkage;
- `SQS-B` in E7.2 (user's decision of 2026-11-19, for now);
- the input classifier $h$ for BayesKDEy: only the MLP input of `main` is studied.
