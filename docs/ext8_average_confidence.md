# Extension 8: Baselines: Average Confidence and the rate matrix rescaled by a quantifier

## Claim

This extension adds baselines to the project, against which the methods under study are read. It holds two of them (scope widened by the user's decision of 2026-11-23): Average Confidence with a Poisson-binomial interval (E8.1, with a label-shift calibrator in E8.3), the simplest uncertainty-aware CAP method, and the rate matrix of $h$ measured on $V$ rescaled by the prevalences of a point quantifier (E8.2), the simplest CAP method built on quantification, with no uncertainty.

**Average Confidence.** Average Confidence (AC) estimates the accuracy of a classifier $h$ on an unlabelled bag as the mean of its confidences, the largest posterior probability of each instance. Kivimäki et al. ([JAIR 2025](papers/Kivimäki%20et%20al.%20-%202025%20-%20Confidence-based%20Estimators%20for%20Predictive%20Performance%20in%20Model%20Monitoring.pdf)) show that, if the confidences are calibrated on the target distribution, AC is unbiased and consistent, and that the number of correct predictions follows a Poisson-binomial distribution, from which they read a central confidence interval for the accuracy of the bag. AC is therefore the simplest uncertainty-aware CAP method: one mean and one interval, both in closed form. This extension adds it to the project as a baseline (user's decision of 2026-11-20), with the confidences recalibrated per bag by LasCal, a calibration method designed for label shift, as `L-CBPE` already does on the binary datasets. AC works for any number of classes, so it extends that baseline to all the datasets of `main`. It is a baseline, not a method under study: the interval represents only the luck of the bag given the calibrated confidences (§T.6), and this is accepted (user's decision of 2026-11-20). LasCal only rescales the logits by a temperature, which cannot represent the exact correction of the posteriors under PPS (§T.4); E8.3 replaces it with label-shift binning, a histogram calibration on $V$ whose frequencies are reweighted by the prevalences estimated on the bag (§T.9), in `AC` and in `L-CBPE` (user's decision of 2026-11-24).

**The rate matrix rescaled by a quantifier.** Under prior probability shift the rates $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ of $h$ do not change, so the contingency table of the target is $T=C\operatorname{diag}(\pi^Q)$ and only $\pi^Q$ is unknown. Measuring $\hat C$ on $V$ and replacing $\pi^Q$ by the estimate $\hat\pi^Q$ of a quantifier gives a point estimate of the table, $\hat C\operatorname{diag}(\hat\pi^Q)$, and from it of every metric (§T.7). This is the core that BayCAP and the fixed-$\hat C$ methods of [[ext3_posterior_propagation|Extension 3]] make uncertainty-aware, so its errors are the starting point of the research: how far a CAP estimate built from quantification goes with four standard quantifiers of QuaPy (KDEy, EMQ, ACC, PACC) in their base form, and how its error splits into the error of $\hat\pi^Q$ and the error of $\hat C$ (§T.8). It is a point-estimate baseline, not an uncertainty-aware method (user's decision of 2026-11-23).

## Status

| Item | Status |
|---|---|
| `L-CBPE`: `cap-ml`'s `CBPE(calib_method="lascal")`, AC with LasCal and the highest-density interval of the Poisson-binomial distribution, binary datasets only (`experiments/methods.py`) | [implemented], run and stored in `main` (binary datasets); excluded from the main tables, the rank table and the coverage vs amplitude plot |
| `AC`: AC with LasCal and the central interval of the Poisson-binomial distribution, any number of classes, vanilla accuracy only (`cap-ml`'s `cap.models.confidence.AC` since WP-32; `methods/ac.py` before) | [implemented] (card WP-29, merged `b17b9b0`, [[work_plan]] §II.31), run and stored in `main` (75 pairs, 2026-11-20) |
| E8.1 (`AC` against `L-CBPE` in a separate table, `tables/average_confidence.py`) | [implemented], card WP-29 (done), run (no pilot, user's decision) and analysed (below) |
| `cap-ml`'s `ContTableTransferCAP`: the table of $h$ on $V$ rescaled by the prevalences of a quantifier | in `cap-ml`, not used by `uncawe` |
| E8.2 (`CTT-KDEy`, `CTT-EMQ`, `CTT-ACC`, `CTT-PACC` and the reference `CTT-oracle`, point estimates, in an experiment module of their own) | [proposed] (user's decision of 2026-11-23); code on branch `wp-30/ctt` (`experiments/ctt.py`, `tables/ctt.py`, `plots/ctt.py`), card WP-30 (In progress), not run |
| `cap-ml`'s `AC` (moved from `methods/ac.py`, with a calibrator argument), `LasCalPerBag`, `LSBinning`, `CBPE(calib_method="lsb")` | in `cap-ml` `eefa550` (version 0.1.1.15), pinned by `uncawe` on branch `wp-32/ac-lsb` |
| E8.3 (`AC-LSB` and `L-CBPE-LSB` against the stored `AC` and `L-CBPE`, in `main`) | [run] (user's decision of 2026-11-24), card WP-32 (In progress); code merged into `devel` (`7daa9b7`: factory entries, `tables/ac_lsb.py`, `plots/ac_lsb.py`); full run on 2026-11-26 (`e0766c4`, all $75$ pairs, results in `main`); tables, plot and analysis pending |

# Experiments

## E8.1: AC with LasCal against L-CBPE

**Why.** `L-CBPE` is the only confidence-based baseline of `main`, and it runs only on the $5$ binary datasets because `cap-ml`'s `CBPE` is binary. AC is the same estimator written for any number of classes (§T.1), with the interval proposed by the AC paper (§T.3). On the binary datasets the two methods share the calibration, the point estimate and the Poisson-binomial distribution, and differ only in how the $95\%$ interval is read from it (central against highest-density, §T.3), so the comparison checks the new method against the stored one and measures what the choice of interval changes; on the multiclass datasets AC gives a confidence-based baseline where there was none. Both methods are kept out of the main tables, the rank table and the coverage vs amplitude plot, and compared in a table of their own (user's decision of 2026-11-20).

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs (the $15$ datasets × the five classifiers of `main`, after FIX-8), $100$ target bags of size $1000$ per pair with the seeds of `main`; the stored `main` results of `L-CBPE` (the $25$ pairs of the binary datasets); the results of the new method `AC` of `main`, with its own run.
  - **Varied:** the method: `AC` on the $75$ pairs and `L-CBPE` on the $25$ binary pairs. *Held fixed:* the posteriors of $h$ stored by `main` and its predictions $\hat y=\arg\max_k p(\omega_k\mid x)$ (`pretrain/data.py`); the LasCal configuration of `cap-ml`'s wrapper (posteriors turned into logits by $\log$, a grid of $100$ temperatures from $0.1$ to $20$, the label-shift calibration error of the `lascal` package, §T.5); the level $1-\alpha=0.95$.
  - **Measures:**
    - for both methods, on vanilla accuracy: AE, coverage, amplitude and Winkler score at $\alpha=0.05$, per bag, per pair and averaged over the pairs (over the $25$ binary pairs for the comparison, over the $75$ pairs for `AC` alone); macro-F1 and macro-K are not estimated by either method (NaN);
    - on the binary pairs, per bag: the difference between the point estimates of the two methods (expected to be zero up to rounding, §T.5), and the differences between their lower and upper bounds;
    - for `AC`, per pair: the number of bags where LasCal failed and the uncalibrated posteriors were used (§T.5), and the distribution of the selected temperature $\theta$.
  - **Estimand:** for both methods, the realised accuracy of each target bag, the trace of its realised table $A(T)=\sum_jT_{jj}$, scored against the realised table of the bag (`true_cts`), as every method of `main`. The Poisson-binomial distribution is conditional on the confidences of the bag, so it describes the realised accuracy, not the population accuracy $\sum_j\pi^Q_jC_{jj}$ (§T.2).
  - **Cost:** one run of `main` for `AC` on the $75$ pairs: per bag, one LasCal fit ($100$ temperatures, each with an estimate of the importance weights and of the calibration error on $V$ and on the bag) and one Poisson-binomial distribution of size $1001$ computed by FFT; no NUTS. `L-CBPE` is not rerun. Pilot to be decided with the card.
- **Representation.**
  - **Comparison table:** one row per pair (dataset × classifier) and one column group per score (AE, coverage, amplitude, Winkler score), with the columns `AC` and `L-CBPE` (empty on the multiclass pairs); two summary rows, the means over the $25$ binary pairs (both methods) and over the $75$ pairs (`AC` only). Method names in tables and plots: `AC` and `L-CBPE`.
  - **Check table:** on the binary pairs, the largest absolute difference of the point estimates and the mean differences of the lower and upper bounds (`AC` minus `L-CBPE`), and for `AC` on every pair the number of LasCal fallbacks.
- **Analysis.** Source: the full run of `AC` in `main` on `bcuda` (commit `f995d1d`, 2026-11-20, 75 pairs, 100 bags each, JAX on CPU) and the stored `L-CBPE` results; tables by `tables/average_confidence.py` (commit `74c83c4`), `<OUT_DIR>/uncawe/tables/average_confidence/average_confidence.csv` and `average_confidence_check.csv`. All scores are on vanilla accuracy, against the realised accuracy of each bag, at $1-\alpha=0.95$.
  - **Check against `L-CBPE`.** On 23 of the 25 binary pairs the point estimates of the two methods agree to at most $2.7\times10^{-15}$ on every bag, as §T.5 predicts. On `german` with kNN and with RF they differ on 4 bags of the 200 (3 and 1), by $2.8\times10^{-4}$ to $8.9\times10^{-4}$; in those bags `AC` selected mid-grid temperatures ($\theta=6.1$, $9.7$, $11.2$, $12.8$). *Hypothesis (not tested):* the label-shift calibration error has near-ties between neighbouring temperatures there, and a rounding difference between the two runs (the stored `L-CBPE` comes from an earlier run of `main`) selects a neighbouring $\theta$. The difference is two orders of magnitude below the AE of these pairs ($0.076$ and $0.093$).
  - **LasCal.** No bag of the 7,500 fell back to the uncalibrated posteriors. The selected temperature spans the whole grid. Its median over the bags is at the upper edge ($\theta\ge19.5$) on 5 pairs (kNN on `german`, `shuttle` and `poker_hand`; RF on `isolet` and `letter`) and at the lower edge ($\theta=0.1$) on 5 binary pairs (`semeion` with LR, SVM and MLP; `spambase` with MLP; `tictactoe` with SVM). A temperature of $20$ divides the logits by $20$, so the calibrated posteriors are much flatter than the raw ones and the confidences much lower. The two largest AE of the 75 pairs, $0.647$ on `isolet` with RF and $0.513$ on `letter` with RF, are on pairs whose median $\theta$ is at the upper edge, but kNN on `german` and `shuttle` also has $\theta$ at the edge with AE $0.076$ and $0.073$, so the edge alone does not explain a large error.
  - **Central against highest-density interval (binary pairs).** The two methods have the same distribution of $K/n$, so they differ only in how the interval is read from it. The central interval of `AC` is wider on every binary pair, by $0.00035$ to $0.00059$ (mean amplitude $0.0319$ against $0.0314$): its lower bound is lower by $0.0003$–$0.0005$ and its upper bound higher by at most $0.0002$, which is the long tail towards low accuracies of §T.3. Coverage is equal on 18 pairs and higher for `AC` on 7 (mean $0.216$ against $0.212$), and the Winkler score is lower for `AC` on all 25 pairs (mean $1.712$ against $1.722$). The choice of interval is therefore a second-order effect: the Winkler score changes by $0.6\%$ on average.
  - **Coverage of `AC`.** The interval covers the realised accuracy far less often than its level: mean coverage $0.216$ on the binary pairs and $0.176$ on the 75 pairs, at most $0.69$ on a pair (`dry-bean` with kNN), and $0$ on 18 pairs (all five classifiers on `mammographic` and `letter`; RF, SVM and MLP on `hand_digits`; LR, SVM, MLP and RF on `isolet`; kNN on `poker_hand`). The mean amplitude ($0.040$ over the 75 pairs) is below the mean AE ($0.108$), and on 57 of the 75 pairs the mean amplitude is below the mean AE. This is the expected behaviour of §T.6: the interval represents only the luck of the bag given the calibrated confidences, whose standard deviation is at most $1/(2\sqrt{1000})\approx0.016$ (§T.3), while the error of AC comes mostly from the calibration under prior shift, which the interval does not represent. As a result the Winkler score is dominated by the misses: its mean over the 75 pairs is $3.61$, against a mean amplitude of $0.040$.
  - **Not examined.** The hypothesis of §T.4, that the residual bias grows with $\lvert\pi^Q-\pi^P\rvert$, needs the AE per bag against the shift of the bag; the tables of E8.1 give only means per pair, so it is left as a candidate. `AC` is not compared with the methods of the main tables (E8.c3).

## E8.2: the rate matrix of $V$ rescaled by a point quantifier

**Why.** Every CAP method of the project that goes through quantification (BayCAP, the fixed-$\hat C$ methods of [[ext3_posterior_propagation|Extension 3]], the SQS methods) rests on the identity $T=C\operatorname{diag}(\pi^Q)$ under PPS (§T.7), and adds uncertainty to one or both factors. Its plainest version, with $C$ measured on $V$ and $\pi^Q$ estimated by one point quantifier, has never been measured on the protocol of `main`. Measuring it gives the starting point of the research (user's decision of 2026-11-23): how large the error of a quantification-based point estimate is with four standard quantifiers, how much of it comes from the quantifier and how much from $\hat C$ (§T.8), and, with the true prevalence of the bag in place of the estimate, the error that remains when quantification is perfect.

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs (the $15$ datasets × the five classifiers of `main`, after FIX-8), the posteriors of $h$ stored by `main` and its predictions $\hat y=\arg\max_kp(\omega_k\mid x)$, the validation set $V$ of `main`, $100$ target bags of size $1000$ per pair with the seeds of `main`. No stored result of `main` is used.
  - **Varied:** the source of the prevalences that rescale $\hat C$, five levels:
    - `CTT-KDEy`: QuaPy's `KDEyML` in its base form (Gaussian kernel, bandwidth $0.1$, no shrinkage, $5$-fold cross-fitting on $V$, `val_split=5`);
    - `CTT-EMQ`: QuaPy's `EMQ` in its base form (no recalibration, `calib=None`; the MLP trained on the whole $V$; the training prevalence is that of $V$, `exact_train_prev=True`);
    - `CTT-ACC`: QuaPy's `ACC` in its base form ($5$-fold cross-fitting of the misclassification rates on $V$, solver `minimize`, `norm="clip"`), the crisp counterpart of PACC (added by the user's decision of 2026-11-23, after the first run of the other levels);
    - `CTT-PACC`: QuaPy's `PACC` in its base form ($5$-fold cross-fitting of the misclassification rates on $V$, solver `minimize`, `norm="clip"`);
    - `CTT-oracle` (**reference**, it uses the target labels): the true prevalence of the bag, $\bar\pi_j=n_j/n$, with $n_j$ the number of instances of class $\omega_j$ in the bag.

    *Held fixed:* $\hat C$, the rates of $h$ on $V$ (§T.7), shared by the five levels; the classifier read by the four quantifiers, an MLP trained on $V$ on the features, seeded with the experiment seed (`MLP(random_state=qp.environ["_R_SEED"])`), as the quantifiers of `main` (`experiments/methods.py:kdey`, `pacc`; FIX-3) (user's decision of 2026-11-23).
  - **Measures:**
    - for each level, per bag, the point estimate of vanilla accuracy, macro-F1 and macro-K computed from the estimated table, and its absolute error (AE) against the realised value; per pair, the mean AE over the $100$ bags; the mean of the per-pair means over the binary pairs ($25$), the multiclass pairs ($50$) and all $75$ pairs;
    - for each quantifier, per bag, the quantification error $\lVert\hat\pi^Q-\bar\pi\rVert_1=\sum_j\lvert\hat\pi^Q_j-\bar\pi_j\rvert$, against the true prevalence of the bag; its mean per pair;
    - for vanilla accuracy, per bag, the two parts of the error of §T.8, the quantification part $e_\pi$ and the rate part $e_C$, with a check that $e_\pi+e_C=\hat A-A(T)$ and that the error of `CTT-oracle` is $e_C$; per pair, the mean and the root mean square (RMS) of each part;
    - per pair, the number of classes absent from $V$ and of bags where a quantifier failed (expected $0$; §Implementation).
  - **Estimand:** the four methods target the population table $C\operatorname{diag}(\pi^Q)$; `CTT-oracle`, with the prevalence of the bag, targets the expected realised table given the class counts of the bag (§T.7). All five are scored against the realised table of each bag (`true_cts`), as every method of `main`. Quantification errors are scored against the prevalence of the bag, the column sums of the realised table.
  - **Cost:** per pair, four quantifier fits on $V$ (one MLP each; KDEy, ACC and PACC with $5$ folds) and $100$ predictions each; the oracle needs no fit. No NUTS. Pilot to be decided with the card.
- **Representation.**
  - **AE tables:** one per metric (vanilla accuracy, macro-F1, macro-K); one row per pair (dataset × classifier, grouped by dataset), in a block for binary and one for multiclass datasets, each closed by the row of the means over its pairs, and a final row of the means over the $75$ pairs; the columns `CTT-KDEy`, `CTT-EMQ`, `CTT-ACC`, `CTT-PACC`, the point estimates of BayCAP-KDEy (`BayCAP-KDEy-mlp-real`) and BayCAP-Hf (`BayesCAP-Hf-real`) stored by `main` on the same bags (both with the realised-table predictive, scored against the realised table; user's decision of 2026-11-23), and, set apart as a reference, `CTT-oracle`. In each row the six methods are coloured from the green of the rank tables (smallest AE) to white (largest), linearly in the AE, and the smallest is in bold; the reference is not coloured and not compared.
  - **Split table (vanilla accuracy):** the same rows; the range $R$ of the recalls of $V$; per quantifier, the mean quantification error and the RMS of $e_\pi$; the RMS of $e_C$, shown once, since it is the same for every quantifier and equals the RMS of the error of `CTT-oracle` (§T.8).
  - **Check table:** per pair, the largest $\lvert e_\pi+e_C-(\hat A-A)\rvert$, the largest difference between the $e_C$ of a quantifier and that of `CTT-oracle`, the largest $\lvert e_\pi\rvert$ of `CTT-oracle`, the bags beyond the bound of §T.8, and the bags where a quantifier failed (all expected $0$).
  - **Plot:** per quantifier, the mean AE of vanilla accuracy against the mean quantification error, one point per pair, binary and multiclass pairs with different markers, one panel per quantifier, with the mean AE of `CTT-oracle` on the same pair as a second marker joined to the first, so that the vertical gap shows what quantification adds.
- **Analysis.** Not run.

## E8.3: AC and CBPE with label-shift binning in place of LasCal

**Why.** §T.4 shows that, for posteriors calibrated on the source, the exact correction under PPS reweights each class by $w_k=\pi^Q_k/\pi^P_k$, and that a temperature, the only thing LasCal changes, cannot represent it (a new hypothesis). In E8.1 the AE of `AC` is large (mean $0.108$ over the $75$ pairs) and its coverage far below its level, which §T.6 attributes to the error of the calibrated confidences. Label-shift binning (Podkopaev & Ramdas, UAI 2021) does what §T.4 asks for: it calibrates on $V$ with a histogram of the posteriors and reweights the frequencies of each bin by $\hat w$ estimated on the bag (§T.9). Replacing LasCal by it in `AC` and `L-CBPE`, with everything else fixed, measures how much of the error of the confidence-based baselines comes from the calibration under PPS. This experiment replaces the candidate E8.c2 (promoted by the user's decision of 2026-11-24), with label-shift binning as its calibrator and no oracle reference.

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs (the $15$ datasets × the five classifiers of `main`, after FIX-8), the posteriors of $h$ stored by `main` and its predictions $\hat y=\arg\max_kp(\omega_k\mid x)$, $V$, $100$ target bags of size $1000$ per pair with the seeds of `main`; the stored `main` results of `AC` ($75$ pairs, E8.1) and `L-CBPE` ($25$ binary pairs); the results of the new methods `AC-LSB` and `L-CBPE-LSB`, with their own run.
  - **Varied:** the calibrator: LasCal (stored `AC` and `L-CBPE`) against label-shift binning (`AC-LSB` on the $75$ pairs, `L-CBPE-LSB` on the $25$ binary pairs, §T.9). *Held fixed:* the confidence of an instance, the calibrated posterior of the class predicted by $h$ (§T.9; for LasCal it is the largest calibrated posterior, §T.5); the interval read from the Poisson-binomial distribution (central for `AC` and `AC-LSB`, highest-density for `L-CBPE` and `L-CBPE-LSB`, §T.3); the level $1-\alpha=0.95$; the configuration of label-shift binning (`cap-ml`'s `LSBinning` at its defaults: BBSE-soft, QuaPy's `BBSEsoft` at default arguments on the posteriors of $h$; the number of bins selected per pair on the grid $\{5,10,15,20,30,40,50,75,100,150,200,300\}$ by the leave-one-out Brier score on $V$).
  - **Measures:**
    - for the four methods, on vanilla accuracy: AE, coverage, amplitude and Winkler score at $\alpha=0.05$, per bag, per pair and averaged over the pairs (over the $25$ binary pairs for the four methods, over the $75$ pairs for `AC` and `AC-LSB`); macro-F1 and macro-K are not estimated (NaN);
    - per pair, the paired differences `AC-LSB` minus `AC` and `L-CBPE-LSB` minus `L-CBPE` of the four scores;
    - per bag, against the shift of the bag $\lVert\bar\pi-\pi^V\rVert_1$ ($\bar\pi$ the prevalence of the bag, $\pi^V$ that of $V$): the signed error $\hat A-A$ of `AC` and of `AC-LSB`, for the hypothesis of §T.4 that the residual bias of LasCal grows with the shift;
    - for `AC-LSB`, per pair: the selected number of bins and the effective number per class (tied posteriors merge edges, §T.9), whether the grid fell back to its smallest value, and, per bag, the number of instances whose class-wise values were all $0$ (made uniform, §T.9), the fraction of instances whose calibrated argmax differs from the prediction of $h$, and the quantification error of BBSE-soft $\lVert\hat\pi^Q-\bar\pi\rVert_1$;
    - check: on the binary pairs, the largest absolute difference of the point estimates of `AC-LSB` and `L-CBPE-LSB` (expected $0$ up to rounding, §T.9).
  - **Estimand:** for the four methods, the realised accuracy of each target bag, the trace of its realised table, scored against the realised table (`true_cts`), as every method of `main`: the Poisson-binomial distribution is conditional on the confidences of the bag (§T.2). The quantification error of BBSE-soft is scored against the prevalence of the bag, the column sums of the realised table.
  - **Cost:** one run of `main` for `AC-LSB` and `L-CBPE-LSB`: per pair, one fit of label-shift binning on $V$ ($12$ bin counts, each with $K$ binnings of $V$, and one BBSE-soft fit, no classifier to train); per bag, one BBSE-soft estimate, $K$ bin lookups and one Poisson-binomial distribution of size $1001$ by FFT; no NUTS, no LasCal. `AC` and `L-CBPE` are not rerun. Pilot to be decided with the card.
- **Representation.**
  - **Comparison table:** one row per pair (dataset × classifier) and one column group per score (AE, coverage, amplitude, Winkler score), with the columns `AC`, `AC-LSB`, `L-CBPE`, `L-CBPE-LSB` (the last two empty on the multiclass pairs); summary rows, the means over the $25$ binary pairs (four methods) and over the $75$ pairs (`AC` and `AC-LSB`). Method names in tables and plots: `AC`, `AC-LSB`, `L-CBPE`, `L-CBPE-LSB`.
  - **Diagnostics table:** per pair, the selected number of bins, the range of the effective numbers over the classes, the grid fallback, the mean over the bags of the degenerate instances and of the fraction with a changed argmax, and the mean quantification error of BBSE-soft; the check of `AC-LSB` against `L-CBPE-LSB` on the binary pairs.
  - **Plot:** the signed error of vanilla accuracy against the shift of the bag, one point per bag, one panel per method (`AC`, `AC-LSB`), binary and multiclass pairs with different colours.
- **Analysis.** Not run.

# Candidate experiments

- **E8.c1, AC with isotonic calibration on $V$ (AC$_c$).** *Why:* it is the setting of the experiments of the AC paper (isotonic regression fitted on held-out labelled data, §5.2.1), as opposed to its method, which leaves the calibrator open; with no label-shift correction it should show the bias of §T.4. *Needs:* an isotonic map of the correctness $\mathbb 1[\hat y=y]$ on the confidence, fitted on $V$ (our adaptation to $K>2$), and one run of `main`.
- **E8.c3, AC against the methods of the main tables.** *Why:* a baseline is useful mainly against the paper's methods, and E8.1 compares `AC` only with `L-CBPE`. *Needs:* the stored results of `MAIN_METHODS` and a table that adds `AC`, no new run beyond E8.1.

# Theory

## T.1 AC and its assumption

Let $h$ be a classifier with posteriors $p(\omega_k\mid x)$, $k=1,\dots,K$, and predictions $\hat y=h(x)=\arg\max_kp(\omega_k\mid x)$, as in `main`. For an instance $x$, define:

- the **confidence** $S=s(x)=\max_kp(\omega_k\mid x)$, the posterior probability of the predicted class;
- the **correctness** $C=\mathbb 1[\hat y=y]$, which is $1$ if the prediction is right and $0$ otherwise.

For a bag of $n$ instances $x_1,\dots,x_n$, AC estimates the accuracy as the mean of the confidences:

$$\hat A_{\text{AC}}=\frac1n\sum_{i=1}^ns_i.$$

The estimator uses only the confidences, so it is defined in the same way for any number of classes $K$; it was introduced for multiclass softmax classifiers (Hendrycks and Gimpel, 2017). The theory of the AC paper uses only the pair $(C,S)$, never the number of classes. Its assumption is that the confidences are **calibrated on the target distribution** $Q$:

$$P_Q(C=1\mid S=s)=s\quad\text{for every }s.$$

In words: among the instances to which $h$ gives confidence $s$, a fraction $s$ is classified correctly. For $K>2$ this is called top-label calibration, because it concerns only the probability of the predicted class.

## T.2 AC is unbiased under the assumption

Two properties are used.

- **Law of total expectation.** For two random variables $C$ and $S$, $\mathbb E[C]=\mathbb E\big[\mathbb E[C\mid S]\big]$: the mean of $C$ is the mean, over the values of $S$, of the mean of $C$ at each value.
- **Linearity of the expectation.** The mean of a sum is the sum of the means, and constants come out: $\mathbb E\big[\frac1n\sum_iS_i\big]=\frac1n\sum_i\mathbb E[S_i]$.

*One instance.* Since $C$ takes only the values $0$ and $1$, its conditional mean is a probability: $\mathbb E_Q[C\mid S=s]=P_Q(C=1\mid S=s)$, which equals $s$ by calibration. Then, by the law of total expectation,

$$\mathbb E_Q[C]=\mathbb E_Q\big[\mathbb E_Q[C\mid S]\big]=\mathbb E_Q[S].$$

The left side is the probability that a random instance of $Q$ is classified correctly, the population accuracy $A_Q$. So the mean confidence on $Q$ equals the accuracy on $Q$. (The AC paper proves the same result with Bayes' rule, Lemma 1.)

*A bag of i.i.d. instances.* By linearity, $\mathbb E_Q[\hat A_{\text{AC}}]=\frac1n\sum_i\mathbb E_Q[S_i]=A_Q$: AC is unbiased for the population accuracy (the AC paper's Theorem 2).

*The realised accuracy.* `main` scores against the accuracy of the bag itself, $K/n$ with $K=\sum_iC_i$ the number of correct predictions. Conditioning on the confidences of the bag instead of averaging over them, calibration gives $\mathbb E_Q[C_i\mid s_i]=s_i$ for each instance, and by linearity

$$\mathbb E_Q\Big[\frac Kn\,\Big|\,s_1,\dots,s_n\Big]=\frac1n\sum_is_i=\hat A_{\text{AC}}.$$

So AC is also the conditional mean of the realised accuracy given what we observe: it targets the realised accuracy of the bag, the estimand of `main`.

## T.3 The Poisson-binomial distribution and the central interval

**The model.** Given the confidences, the AC paper treats the correctness of each instance as an independent Bernoulli trial with success probability $s_i$:

$$C_i\mid s_i\sim\operatorname{Bernoulli}(s_i),\qquad C_1,\dots,C_n\text{ independent given }s_1,\dots,s_n.$$

The number of correct predictions $K=\sum_iC_i$ is then a sum of independent Bernoulli variables with different probabilities. Its distribution is called **Poisson-binomial**; when all $s_i$ are equal it reduces to the binomial. Its probability mass function is

$$P(K=k)=\sum_{A\in F_k}\prod_{i\in A}s_i\prod_{j\notin A}(1-s_j),$$

where $F_k$ is the set of all subsets of $\{1,\dots,n\}$ with $k$ elements: each subset $A$ is one way of having exactly the instances of $A$ correct, and its probability is the product of the probabilities of the independent trials. The sum has $\binom nk$ terms, so it is computed with the FFT algorithm of Hong (2013), exactly and in $O(n\log n)$.

**Mean and variance.** A Bernoulli($s_i$) variable has mean $s_i$ and variance $s_i(1-s_i)$. By linearity, $\mathbb E[K]=\sum_is_i$, which gives back AC for $K/n$. The variance of a sum of *independent* variables is the sum of their variances, and $\operatorname{Var}(aX)=a^2\operatorname{Var}(X)$, so

$$\operatorname{Var}\Big(\frac Kn\Big)=\frac1{n^2}\sum_{i=1}^ns_i(1-s_i)\le\frac1{n^2}\cdot\frac n4=\frac1{4n},$$

since $s(1-s)\le\frac14$ for $s\in[0,1]$ (its maximum is at $s=\frac12$). At $n=1000$, the standard deviation of $K/n$ is at most $1/(2\sqrt{1000})\approx0.016$, and much less when the confidences are close to $1$ or $0$: the interval is narrow by construction.

**The central interval.** The AC paper reads the $95\%$ interval for $K/n$ from the cumulative distribution function $F(k)=P(K\le k)$, as the range where $0.025\le F\le0.975$ (§4.2). Since $K$ is discrete, $F$ jumps, and this range is made precise as follows (our reading, chosen so that the interval holds at least $95\%$ of the probability):

$$k_{\text{low}}=\min\{k:F(k)\ge\tfrac\alpha2\},\qquad k_{\text{up}}=\min\{k:F(k)\ge1-\tfrac\alpha2\},\qquad[\,k_{\text{low}}/n,\;k_{\text{up}}/n\,].$$

The probability below $k_{\text{low}}$ is $F(k_{\text{low}}-1)<\frac\alpha2$ and the probability above $k_{\text{up}}$ is $1-F(k_{\text{up}})\le\frac\alpha2$, so the interval holds more than $1-\alpha$.

**Central against highest-density.** `L-CBPE` uses `cap-ml`'s `PoissonConfidenceInterval`, the highest-density interval (HDI) of the CBPE paper (Kivimäki et al., [2025](papers/Kivimäki%20et%20al.%20-%202025%20-%20Performance%20Estimation%20in%20Binary%20Classification%20Using%20Calibrated%20Confidence.pdf), Algorithm 3): the shortest interval holding at least $95\%$, built by removing the less probable end of the distribution until the next removal would exceed $\alpha$. The central interval leaves $\frac\alpha2$ in each tail instead. On a symmetric distribution the two coincide. When the confidences are mostly close to $1$, the distribution of $K/n$ piles up against its upper limit $1$ and has a longer tail towards low accuracies; the HDI then moves towards the most probable values and is slightly shorter, cutting more of the long tail.

**Why the central interval.** It is the interval of the AC paper (user's decision of 2026-11-20), and it is the one the Winkler score rewards. The Winkler score of an interval $[l,u]$ for a true value $y$ is

$$W=(u-l)+\frac2\alpha(l-y)\,\mathbb 1[y<l]+\frac2\alpha(y-u)\,\mathbb 1[y>u].$$

If $y$ is drawn from a distribution with cumulative distribution function $F$, the expected score as a function of $l$ is $-l+\frac2\alpha\,\mathbb E[(l-Y)\mathbb 1[Y<l]]$ plus terms without $l$. Its derivative in $l$ is $-1+\frac2\alpha F(l)$, because moving $l$ up by a small step shortens the interval by that step and raises the penalty by $\frac2\alpha$ times the step for each $y$ below $l$, which happens with probability $F(l)$. Setting the derivative to zero gives $F(l)=\frac\alpha2$. In the same way, the derivative in $u$ is $1-\frac2\alpha(1-F(u))$, zero at $F(u)=1-\frac\alpha2$. So the expected Winkler score is smallest at the central interval of the true distribution (the Winkler score is a proper scoring rule for it; Gneiting and Raftery, 2007), and an HDI of the true distribution scores worse whenever the distribution is skewed.

## T.4 Why a calibration fitted on the source does not hold under PPS

Let $P$ be the source distribution, from which $V$ is drawn, and $Q$ the target distribution of a bag. Under prior probability shift the class-conditional distributions do not change, $p_Q(x\mid\omega_j)=p_P(x\mid\omega_j)$, and only the prevalences do, from $\pi^P$ to $\pi^Q$. Since the confidence $s(x)$ and the prediction $h(x)$ are functions of $x$ alone, two class-wise quantities do not change either:

- the recall of class $\omega_j$, $C_{jj}=P(\hat Y=\omega_j\mid Y=\omega_j)$;
- the mean confidence on the instances of class $\omega_j$, $m_j=\mathbb E[S\mid Y=\omega_j]$.

**Accuracy and mean confidence on $Q$.** By the law of total probability over the true class, the accuracy is $A_Q=\sum_j\pi^Q_jC_{jj}$. By the law of total expectation over the true class, the mean confidence is $\mathbb E_Q[S]=\sum_j\pi^Q_jm_j$.

**The bias.** The expectation of AC on $Q$ is $\mathbb E_Q[S]$ (linearity), so its bias is

$$\mathbb E_Q[\hat A_{\text{AC}}]-A_Q=\sum_j\pi^Q_j\,(m_j-C_{jj}).$$

If the confidences are calibrated on $P$, §T.2 applied to $P$ gives $\mathbb E_P[S]=A_P$, that is $\sum_j\pi^P_j(m_j-C_{jj})=0$. Subtracting this zero from the bias,

$$\mathbb E_Q[\hat A_{\text{AC}}]-A_Q=\sum_j(\pi^Q_j-\pi^P_j)\,(m_j-C_{jj}).$$

The bias is zero when $\pi^Q=\pi^P$ and grows linearly with the shift. *Intuition:* the confidence does not know the true class. On a class with a low recall, $h$ still makes many confident predictions, wrong ones, so typically $m_j>C_{jj}$; moving prevalence towards such a class lowers the accuracy more than it lowers the mean confidence, and AC overestimates. Under the APP of `main`, where the target prevalences are drawn uniformly from the simplex, the shifts are large.

**The exact correction.** If the posteriors are calibrated on $P$ in the full sense, $p(\omega_k\mid x)=P_P(Y=\omega_k\mid x)$, Bayes' rule gives the posteriors on $Q$. Writing $P_Q(Y=\omega_k\mid x)\propto p_Q(x\mid\omega_k)\,\pi^Q_k=p_P(x\mid\omega_k)\,\pi^P_k\cdot\frac{\pi^Q_k}{\pi^P_k}\propto P_P(Y=\omega_k\mid x)\,\frac{\pi^Q_k}{\pi^P_k}$,

$$P_Q(Y=\omega_k\mid x)=\frac{w_k\,p(\omega_k\mid x)}{\sum_lw_l\,p(\omega_l\mid x)},\qquad w_k=\frac{\pi^Q_k}{\pi^P_k}.$$

The confidence calibrated on $Q$ is then this posterior evaluated at the predicted class, $P_Q(Y=h(x)\mid x)$; it needs $\pi^Q$, which is unknown. Label-shift binning (E8.3, §T.9) applies this correction to a histogram calibration fitted on $V$, with $\pi^Q$ estimated on the bag.

**A temperature cannot represent it (new hypothesis).** For $K=2$ write the posterior through its log-odds $z(x)=\log\frac{p(\omega_1\mid x)}{p(\omega_0\mid x)}$. Taking the log of the ratio of the two corrected posteriors, the correction above *adds a constant*: $z_Q(x)=z(x)+\log\frac{w_1}{w_0}$. A temperature $\theta$ *divides* the log-odds: $z_\theta(x)=z(x)/\theta$. The first shifts every instance towards the class that became more frequent; the second only makes all posteriors sharper or flatter, symmetrically around $\frac12$, and keeps every prediction. No single $\theta$ reproduces a shift of the log-odds on all instances, so a temperature-based recalibration such as LasCal (§T.5) can reduce the bias of §T.4 on average over the bag but not remove it; the residual should grow with $\lvert\pi^Q-\pi^P\rvert$. This is a hypothesis for the analysis of E8.1, not a result.

## T.5 LasCal and how AC uses it

LasCal (Popordanoska et al., NeurIPS 2024) is the calibration method that `L-CBPE` uses, through `cap-ml`'s wrapper (`cap/calibration/lascal.py`) of the `lascal` package (`lascal/post_hoc_calibration/calibrator.py:Calibrator.lascal`). As implemented, for each bag:

1. the posteriors of $V$ (with their labels) and of the bag are turned into logits by $z=\log p$;
2. for each temperature $\theta$ of a grid of $100$ values evenly spaced from $0.1$ to $20$, the logits of $V$ and of the bag are divided by $\theta$; the importance weights $\pi^Q_k/\pi^P_k$ are estimated from the scaled predictions on $V$ and on the bag (RLLS, the default `rlls-hard`); and a calibration error on the bag is estimated without its labels, from the labels of $V$ reweighted by the importance weights (class-wise, squared differences, $15$ bins with equal counts);
3. the $\theta$ with the smallest estimated error is kept, and the calibrated posteriors of the bag are the softmax of $z/\theta$.

So LasCal is a temperature scaling whose temperature is chosen per bag for the target. Three consequences for AC:

- **The predictions do not change.** Dividing all the logits of an instance by the same $\theta>0$ keeps their order, so the class with the largest calibrated posterior is the class with the largest original posterior, which is the prediction of $h$ in `main` ($\hat y=\arg\max_kp(\omega_k\mid x)$, `pretrain/data.py`). The confidence of AC, the largest calibrated posterior, is therefore the calibrated posterior of the predicted class, as §T.1 requires.
- **On binary data AC is `L-CBPE`.** For $K=2$, `cap-ml`'s `CBPE` takes $\max(\tilde s,1-\tilde s)$ of the calibrated posterior $\tilde s$ of the positive class, which is the largest calibrated posterior: the two methods have the same confidences, the same point estimate and the same Poisson-binomial distribution, and differ only in the interval (§T.3).
- **Fallback.** If the `lascal` package raises an error, `cap-ml`'s wrapper `cap/calibration/lascal.py` returns the uncalibrated posteriors without notice; the wrapper that `CBPE` uses, `cap/calib.py:LasCal`, would instead raise, because its fallback calls the module's own `softmax(preact, temp, biases)` with an `axis` argument. The package also has a silent failure of its own: if the estimated error is NaN at every temperature of the grid, it keeps its initial value $\theta=-1$ and returns $z/(-1)$, which reverses the order of the classes (for $K=2$ the largest posterior is unchanged, for $K>2$ it is not). `AC` treats both cases as a fallback to the uncalibrated posteriors, so that AC reduces to AC on raw confidences for that bag, and records it per bag.

## T.6 What the interval leaves out

The Poisson-binomial interval describes one source of error: given calibrated confidences, which instances of the bag happen to be classified correctly. It is the instance-level counterpart of the luck of the bag of [[ext4_uncertainty_budget|Extension 4]]. It leaves out:

- **the error of the calibrated confidences**, which the model treats as exact: the temperature chosen by LasCal on one bag of $1000$ instances, the importance weights estimated from the same bag, the finite $V$, and the part of the prior shift that a temperature cannot represent (§T.4). In the terms of Extension 4, the prevalence part $e_1$ and the recall part $e_2$ are not represented;
- **duplicates.** The bags of `main` are drawn with replacement from the test pool, so an instance can appear more than once, always with the same label. Two copies of an instance with confidence $s$ are counted by the model as two independent trials, with variance $2s(1-s)$, while their correctness is the same, so the variance of their sum is $\operatorname{Var}(2C)=4s(1-s)$. Duplicates make the interval narrower than it should be.

The AC paper names the first point itself: the confidences are "only point estimates of the empirical probabilities" (§6.2). So under-coverage under PPS is expected; it is accepted, since AC is a baseline (user's decision of 2026-11-20).

## T.7 The table under PPS and its point estimate (E8.2)

**The table factorises.** The contingency table of $h$ on the target distribution $Q$ has entries $T_{ij}=P_Q(\hat Y=\omega_i,\,Y=\omega_j)$, the probability that an instance has true class $\omega_j$ and predicted class $\omega_i$ (predicted classes on rows, true classes on columns). By the definition of conditional probability, $P(A,B)=P(A\mid B)\,P(B)$, so

$$T_{ij}=P_Q(\hat Y=\omega_i\mid Y=\omega_j)\;P_Q(Y=\omega_j).$$

Under prior probability shift the class-conditional distributions do not change, $p_Q(x\mid\omega_j)=p_P(x\mid\omega_j)$, and the prediction $\hat y=h(x)$ is a function of $x$ alone, so the first factor is the same on $P$ and on $Q$: it is the rate $C_{ij}$ of $h$. The second factor is the prevalence $\pi^Q_j$. In matrix form, $T=C\operatorname{diag}(\pi^Q)$: column $j$ of $C$ is multiplied by $\pi^Q_j$.

**The point estimate.** $C$ is estimated on $V$, where the labels are known: with $N^V_{ij}$ the number of instances of $V$ of true class $\omega_j$ predicted as $\omega_i$ and $n^V_j=\sum_iN^V_{ij}$ the number of instances of class $\omega_j$ in $V$,

$$\hat C_{ij}=\frac{N^V_{ij}}{n^V_j},$$

the fraction of the instances of class $\omega_j$ that $h$ predicts as $\omega_i$. A quantifier gives $\hat\pi^Q$ from the unlabelled bag. The estimated table is $\hat T=\hat C\operatorname{diag}(\hat\pi^Q)$, and every metric is computed from it as from a true table; for vanilla accuracy, the sum of the diagonal,

$$\hat A=\sum_j\hat T_{jj}=\sum_j\hat\pi^Q_j\,\hat C_{jj},$$

the recalls of $h$ on $V$ averaged with the estimated prevalences as weights. `cap-ml`'s `ContTableTransferCAP` computes the same table in another way: it normalises the table of $V$ to sum to $1$, $\hat T^V=\hat C\operatorname{diag}(\pi^V)$ with $\pi^V_j=n^V_j/\lvert V\rvert$, and multiplies column $j$ by $\hat\pi^Q_j/\pi^V_j$, which gives $\hat C\operatorname{diag}(\pi^V)\operatorname{diag}(\hat\pi^Q/\pi^V)=\hat C\operatorname{diag}(\hat\pi^Q)$. (The code stores the transpose, with true classes on rows, and rescales rows.)

**Population or realised table.** $\hat C\operatorname{diag}(\hat\pi^Q)$ estimates the population table $C\operatorname{diag}(\pi^Q)$ and carries no uncertainty. `main` scores against the realised table of the bag, $T^{\text{bag}}_{ij}=N_{ij}/n$, with $N_{ij}$ the counts of the bag. Given the class counts $n_j$ of the bag, each instance of class $\omega_j$ is predicted as $\omega_i$ with probability $C_{ij}$, independently, so $N_{ij}$ counts successes in $n_j$ trials and, by linearity of the expectation, $\mathbb E[N_{ij}\mid n_j]=n_jC_{ij}$. Dividing by $n$,

$$\mathbb E\big[T^{\text{bag}}_{ij}\,\big|\,n_1,\dots,n_K\big]=C_{ij}\,\bar\pi_j,\qquad\bar\pi_j=\frac{n_j}n.$$

So the reference `CTT-oracle`, $\hat C\operatorname{diag}(\bar\pi)$, estimates the expected realised table given the class counts of the bag: what remains of its error is the error of $\hat C$ and the luck of which instances of each class fall in the bag.

## T.8 The error splits into a quantification part and a rate part (E8.2)

Write the realised accuracy of the bag with the same two factors. With $\bar C_{jj}=N_{jj}/n_j$ the recall of $h$ in the bag (for $n_j>0$; a class absent from the bag has $\bar\pi_j=0$ and contributes nothing),

$$A(T^{\text{bag}})=\sum_j\frac{N_{jj}}n=\sum_j\frac{n_j}n\cdot\frac{N_{jj}}{n_j}=\sum_j\bar\pi_j\,\bar C_{jj}.$$

Subtract it from $\hat A=\sum_j\hat\pi^Q_j\hat C_{jj}$, and add and subtract $\sum_j\bar\pi_j\hat C_{jj}$:

$$\hat A-A(T^{\text{bag}})=\underbrace{\sum_j(\hat\pi^Q_j-\bar\pi_j)\,\hat C_{jj}}_{e_\pi}+\underbrace{\sum_j\bar\pi_j\,(\hat C_{jj}-\bar C_{jj})}_{e_C}.$$

- $e_\pi$, the **quantification part**, is the error of the prevalences weighted by the recalls of $V$; it is zero for `CTT-oracle`, whose error is therefore exactly $e_C$.
- $e_C$, the **rate part**, is the error of the recalls of $V$ against the recalls of the bag, weighted by the true prevalences; it is the same for every quantifier on the same bag, since it does not involve $\hat\pi^Q$.

So the difference between two quantifiers on a bag lies entirely in $e_\pi$.

**A bound on the quantification part.** Write $d_j=\hat\pi^Q_j-\bar\pi_j$. Both prevalence vectors sum to $1$, so $\sum_jd_j=0$, and for any constant $m$, $\sum_jd_j\hat C_{jj}=\sum_jd_j(\hat C_{jj}-m)$, since the added term is $m\sum_jd_j=0$. Choose $m$ halfway between the largest and the smallest recall, so that $\lvert\hat C_{jj}-m\rvert\le R/2$ for every $j$, with $R=\max_j\hat C_{jj}-\min_j\hat C_{jj}$ the range of the recalls. The absolute value of a sum is at most the sum of the absolute values (triangle inequality), so

$$\lvert e_\pi\rvert\le\sum_j\lvert d_j\rvert\,\lvert\hat C_{jj}-m\rvert\le\frac R2\,\lVert\hat\pi^Q-\bar\pi\rVert_1.$$

*Intuition:* moving prevalence from one class to another changes the accuracy only as much as the recalls of the two classes differ. If $h$ has the same recall on every class ($R=0$), the accuracy does not depend on the prevalences and a wrong $\hat\pi^Q$ costs nothing; the more the recalls differ, the more the quantifier matters. *Hypothesis (new, for the analysis of E8.2):* the ranking of the four quantifiers on the AE of vanilla accuracy follows their ranking on the quantification error mostly on the pairs with a large $R$, and on pairs with a small $R$ the AE of the four methods is close to that of `CTT-oracle`. Macro-F1 and macro-K are not linear in the table, so the split applies to vanilla accuracy only.

## T.9 Label-shift binning and the confidence of the predicted class (E8.3)

**Which confidence.** Let $\hat y(x)=h(x)$ be the prediction of $h$ and $\tilde p_k(x)$ any calibrated posteriors. The accuracy of $h$ on $Q$ is the probability that the prediction is correct, $A_Q=P_Q(Y=h(X))=\mathbb E_Q\big[\mathbb 1[Y=h(X)]\big]$. By the law of total expectation, conditioning on $X$,

$$A_Q=\mathbb E_Q\Big[\mathbb E_Q\big[\mathbb 1[Y=h(X)]\,\big|\,X\big]\Big]=\mathbb E_Q\big[P_Q(Y=h(X)\mid X)\big],$$

since the conditional mean of an indicator is a probability. So the confidence that AC averages must estimate $P_Q(Y=h(x)\mid x)$, the probability that **the prediction of $h$** is correct: $s_i=\tilde p_{\hat y_i}(x_i)$. The largest calibrated posterior, $\max_k\tilde p_k(x_i)$, estimates instead the probability that the class $\arg\max_k\tilde p_k(x_i)$ is correct, which is the accuracy of another classifier, the one that predicts with the calibrated posteriors. The two coincide when the calibration keeps the argmax, as a temperature does (§T.5), so `AC` and `L-CBPE` are unchanged; label-shift binning can change the argmax, so `AC-LSB` and `L-CBPE-LSB` take $\tilde p_{\hat y_i}(x_i)$.

**The bins.** Label-shift binning is defined by Podkopaev & Ramdas for $K=2$; for $K>2$ we use a class-wise instantiation (our adaptation, not the paper's method). For each class $\omega_k$, the values $h_k(x)=p(\omega_k\mid x)$ of the instances of $V$ are split into $M$ bins with about the same number of instances each (uniform-mass bins, with edges at the quantiles of the values). For a bin $b$ of class $k$, with $n_b$ instances of $V$ of which $c_{b,j}$ have true class $\omega_j$, the bin stores the vector of frequencies

$$f^{(k)}_{b,j}=\frac{c_{b,j}}{n_b},\qquad j=1,\dots,K.$$

Write $B$ for the event "$h_k(X)$ falls in bin $b$". Among the instances of $V$ in the bin, the fraction of class $\omega_j$ estimates the probability of that class given the bin on the source, $f^{(k)}_{b,j}\approx P_P(Y=\omega_j\mid B)$, by the law of large numbers. This is a histogram calibration: all the instances of a bin get the same calibrated value.

**The reweighting is Bayes' rule.** $B$ depends on $x$ alone, so under PPS its probability given the class does not change: $P_Q(B\mid Y=\omega_j)=P_P(B\mid Y=\omega_j)$. Three steps give the probability of the class given the bin on $Q$:

1. Bayes' rule on $Q$: $P_Q(Y=\omega_j\mid B)=\dfrac{P_Q(B\mid Y=\omega_j)\,\pi^Q_j}{P_Q(B)}$.
2. PPS, then Bayes' rule on $P$ for $P_P(B\mid Y=\omega_j)=\dfrac{P_P(Y=\omega_j\mid B)\,P_P(B)}{\pi^P_j}$: substituting, $P_Q(Y=\omega_j\mid B)=P_P(Y=\omega_j\mid B)\,\dfrac{\pi^Q_j}{\pi^P_j}\cdot\dfrac{P_P(B)}{P_Q(B)}$.
3. The last factor does not depend on $j$, and the probabilities of the $K$ classes given $B$ sum to $1$, so it is the normalising constant:

$$P_Q(Y=\omega_j\mid B)=\frac{P_P(Y=\omega_j\mid B)\,w_j}{\sum_lP_P(Y=\omega_l\mid B)\,w_l},\qquad w_j=\frac{\pi^Q_j}{\pi^P_j}.$$

This is the correction of §T.4, now exact for the bin and not only for posteriors calibrated in the full sense. With the frequencies in place of $P_P(Y\mid B)$ and $\hat w$ in place of $w$, the class-wise value of an instance $x$ for class $k$ is the $k$-th component of the reweighted vector of its bin:

$$o_k(x)=\frac{f^{(k)}_{b,k}\,\hat w_k}{\sum_lf^{(k)}_{b,l}\,\hat w_l},\qquad b\text{ the bin of }h_k(x).$$

*Intuition:* if a class became twice as frequent, its share in every bin doubles before the shares are normalised again; a temperature, by contrast, only sharpens or flattens the posteriors and cannot move them towards one class (§T.4).

**Combining the classes.** $o_k(x)$ estimates $P_Q(Y=\omega_k\mid h_k(X)\in b)$, and each class conditions on its own binning, so the $K$ values of an instance need not sum to $1$. They are renormalised, with a small $\epsilon=10^{-12}$ added to each:

$$\tilde p_k(x)=\frac{o_k(x)+\epsilon}{\sum_l\big(o_l(x)+\epsilon\big)}.$$

The renormalisation is a choice of the class-wise instantiation, not a consequence of the model. Without $\epsilon$, an instance whose $K$ bins all have frequency $0$ for their own class would give $0/0$; with it, such an instance gets the uniform posterior $1/K$, and every other instance changes by about $\epsilon$. For $K=2$, $h_1=1-h_0$, so the two binnings are mirror images up to ties, $o_0+o_1\approx1$, and the renormalisation changes little.

**The weights.** $\hat w$ comes from the prevalence $\hat\pi^Q$ estimated on the bag by BBSE-soft (Lipton et al., 2018) on the posteriors of $h$: on $V$ it estimates the matrix of the mean posteriors per true class, $M_{ij}=\mathbb E_P[p(\omega_i\mid X)\mid Y=\omega_j]$; under PPS these do not change, so by the law of total expectation the mean posterior on the bag is $\mu^Q_i=\sum_jM_{ij}\pi^Q_j$, a linear system solved for $\pi^Q$ (QuaPy's least squares on the simplex). Then $\hat w_j=\max(\hat\pi^Q_j,10^{-8})/\pi^V_j$, with $\pi^V$ the prevalence of $V$; the floor keeps every weight positive. Every class must appear in $V$, or $\pi^V_j=0$.

**The number of bins.** Few bins average the class frequencies over very different instances; many bins leave few instances per bin, and so noisy frequencies. $M$ is chosen on a grid by the leave-one-out Brier score on $V$, without reweighting (there is no shift on $V$, $w=1$). The Brier score of a probability vector $q$ for an instance of class $\omega_y$ is $\sum_k(q_k-\mathbb 1[k=y])^2$, the squared distance from the one-hot label. Leaving out instance $i$, of class $\omega_{y_i}$, changes only the frequencies of its own bins: for class $k$, with $b$ its bin,

$$f^{(k,-i)}_{b,k}=\frac{c_{b,k}-\mathbb 1[y_i=k]}{n_b-1},$$

so the criterion has a closed form; the edges are kept fixed (an approximation, since the quantiles of $V$ without $i$ can differ slightly). A bin with $n_b<2$ makes the leave-one-out frequency undefined, so that $M$ gets an infinite score. If every $M$ of the grid does, the smallest one is used, and the fallback is recorded. When the posteriors take few distinct values (kNN, RF), equal quantiles merge, so the effective number of bins of a class can be smaller than $M$. The Brier score is used rather than the log-loss because a bin frequency is often exactly $0$.

**CBPE.** For $K=2$, `cap-ml`'s `CBPE` turns the calibrated posterior $\tilde s$ of the positive class into $\max(\tilde s,1-\tilde s)$, the largest calibrated posterior (§T.5). With label-shift binning the confidence is already that of the predicted class, $\tilde p_{\hat y}$, and a value below $\frac12$ is the probability that the prediction of $h$ is correct; folding it would replace it with the probability that it is wrong. `L-CBPE-LSB` therefore uses $\tilde p_{\hat y}$ without folding, so on binary data it has the confidences, point estimate and Poisson-binomial distribution of `AC-LSB`, and differs from it only in the interval (§T.3).

**What changes and what does not (new hypothesis).** The interval is still the Poisson-binomial interval of §T.3, conditional on the confidences, so it still represents only the luck of the bag (§T.6): the error of $\hat\pi^Q$ estimated on one bag and of the frequencies estimated on $V$ is not represented. *Hypothesis:* label-shift binning reduces the bias of §T.4, so `AC-LSB` has a lower AE than `AC` on the pairs where the shift of the bags is large, and the gap of `AC` grows with the shift while that of `AC-LSB` grows less; its coverage stays far below $0.95$, since its amplitude is bounded as in §T.3.

# Implementation

## E8.1

Components:

- **Method** `AC`, a class in `src/uncawe/methods/` (to be listed in [[work_plan]] §II.12, code that should eventually move to `cap-ml`), with the interface of `cap-ml`'s `CBPE`: `fit` stores the posteriors and labels of $V$; on each bag it calls the `lascal` package as `cap-ml`'s wrapper does (§T.5), but keeps the selected temperature $\theta$ and a fallback flag, which the wrapper does not return; it takes the largest calibrated posterior of each instance, builds the Poisson-binomial distribution of $K/n$ (the `poibin` FFT as in `CBPE`) and returns its mean as the point estimate and the central interval of §T.3. Vanilla accuracy only; macro-F1 and macro-K are NaN.
- **Central interval:** a `ConfidenceInterval` of `cap-ml`'s interface built from the probability mass function of $K/n$ with the quantile rule of §T.3, so that the tables of `main` compute coverage, amplitude and Winkler score as for the other methods.
- **`main`:** the factory entry `AC` in `experiments/methods.py`, run on every dataset; per bag, the extra columns `lascal_temperature` and `lascal_fallback`. `AC` is not added to `tables/main.py:MAIN_METHODS`, so it stays out of the main tables and of the coverage vs amplitude plot (`plots/amplitude_vs_coverage.py`), and it is excluded from the rank table (`tables/rank.py`), as `L-CBPE` is.
- **Table module** for E8.1: the comparison table and the check table, CSVs and LaTeX documents, following [[work_plan]] Part V.

Tests:

- on binary data, `AC` gives the same confidences, point estimate and probability mass function as `cap-ml`'s `CBPE(calib_method="lascal")` on the same posteriors;
- the central interval equals the quantiles of a brute-force Poisson-binomial distribution on a small $n$, and holds at least $1-\alpha$ of the probability;
- on synthetic confidences calibrated by construction (labels drawn from Bernoulli($s_i$), seeded), the coverage of the central interval is close to $0.95$;
- dividing the logits by a temperature keeps the argmax, so the confidence is the calibrated posterior of the prediction;
- the fallback flag is set when the `lascal` package raises (mocked), and the uncalibrated confidences are used;
- LasCal gives the same output on two fits with the same inputs (determinism);
- macro-F1 and macro-K are NaN.

Reproducibility and comparability:

- `AC` adds a method to `main`; the stored results, including those of `L-CBPE`, are unchanged, and the bags are those of `main`.
- On the binary pairs, the point estimates of `AC` and of the stored `L-CBPE` should coincide bag by bag; the check table of E8.1 verifies it.

## E8.2

E8.2 has point estimates only, while `main` stores intervals and its tables compute coverage, amplitude and Winkler score from them (`experiments/main.py:_evaluate_contingency_method` calls `predict_ct_range` and `ci_from_cts`). So E8.2 has an experiment module of its own (user's suggestion of 2026-11-23), which reuses the protocol of `main` instead of adding methods to it.

Components:

- **Methods:** `cap-ml`'s `ContTableTransferCAP(acc, q)`, fitted on $V$ and the posteriors of $h$ on $V$, with `q` QuaPy's `KDEyML`, `EMQ` or `PACC` in their base form, each with `MLP(random_state=qp.environ["_R_SEED"])` as the classifier (the default arguments of QuaPy 0.2.4 listed in E8.2). The reference `CTT-oracle`: the table $\hat T^V$ of the same fit rescaled by $\bar\pi/\pi^V$, with $\bar\pi$ from the column sums of `true_cts` of the bag (in the code, the row sums of the transpose); used only by this module and never as a method.
- **Experiment module** (e.g. `experiments/ctt.py`): the pairs, the dataset bundle, $V$, the bags and the seeds of `main` (`experiments/main.py:_prepare_evaluation`, `test_prot`, `test_prot_true_cts`); per pair and level, the estimated table of each bag; per bag, the point estimate and the AE of each metric (`experiments/metrics.py:gen_acc_measure`), the estimated prevalence $\hat\pi^Q$ (`ctt_test_prev`), the prevalence of the bag, and for vanilla accuracy $e_\pi$ and $e_C$ (§T.8); the fit and prediction times. Results in `<OUT_DIR>/uncawe/ctt/`, one file per pair, level and metric, skipped if present, as in `main`. $\hat\pi^Q$ is read back as the row sums of the estimated table; since `vanilla_acc` divides the trace by the sum of the table and a quantifier's estimate sums to $1$ only up to about $10^{-11}$, $e_\pi$ is computed with $\hat\pi^Q$ normalised to sum to $1$, so that $e_\pi+e_C$ equals the stored $\hat A-A$ to rounding.
- **Table and plot modules** for the AE table, the split table and the plot of E8.2, following [[work_plan]] Part V.
- **Make targets** for the pilot and full run, the tables and the plot.

Tests:

- `ContTableTransferCAP` with a fixed prevalence equals $\hat C\operatorname{diag}(\hat\pi^Q)$ computed from the counts of $V$ (transpose handled), and with $\hat\pi^Q=\pi^V$ equals the table of $V$;
- `CTT-oracle` on a bag equals $\hat C\operatorname{diag}(\bar\pi)$, with $\bar\pi$ from the true table of the bag;
- $e_\pi+e_C=\hat A-A(T^{\text{bag}})$ to rounding, $e_\pi=0$ for the oracle, and $\lvert e_\pi\rvert\le\frac R2\lVert\hat\pi^Q-\bar\pi\rVert_1$, on synthetic tables;
- two fits with the same seed give the same estimates (determinism);
- a class absent from $V$ ($\pi^V_j=0$, a division by zero in `ContTableTransferCAP`) is detected and reported, not silently turned into NaN.

Reproducibility and comparability:

- E8.2 adds an experiment; `main` and its stored results are unchanged. The bags, $V$ and the posteriors of $h$ are those of `main`, so the AE of E8.2 is comparable bag by bag with the stored methods of `main`.
- The quantifiers are those of `main` in their classifier and seed (`kdey()`, `pacc()`), so `CTT-KDEy` uses the same KDEy as `O-LEAP` and `QuAcc_NxN`, and `CTT-PACC` the same PACC as the SQS methods; EMQ with an MLP is not used elsewhere in `main` (the EMQ of E3.1, `BayCAP-EMQ-*`, is QuaPy's Bayesian version).

## E8.3

The method code lives in `cap-ml` (user's decision of 2026-11-24), at commit `eefa550` (version 0.1.1.15): the interface `cap.calibration.base.BagCalibrator` (fitted once on $V$, applied to each bag, with per-bag diagnostics); `cap.calibration.LasCalPerBag`, the LasCal of `methods/ac.py` moved unchanged (selected temperature, flagged fallback); `cap.calibration.LSBinning` (§T.9); `cap.models.confidence.AC`, the `AC` of `methods/ac.py` with a `calibrator` argument (default `LasCalPerBag`) and the confidence of the predicted class (`predicted_class_confidences`), with `CentralPoissonInterval`; `CBPE(calib_method="lsb")`, and `CBPE(calib_method="lascal")` through `LasCalPerBag`.

Components in `uncawe`:

- **Pin bump** of `cap-ml` from `93ae246` to `eefa550` (`pyproject.toml`, `uv lock`). Between the two commits `cap-ml` changes only the calibration package, `cap/models/confidence.py` (`AC`, `CentralPoissonInterval`, the two `CBPE` changes), their tests, the README and the version.
- **`AC` from `cap-ml`:** `experiments/methods.py`, `experiments/main.py` and `tables/main.py` import `AC` and `CentralPoissonInterval` from `cap.models.confidence`; the per-bag columns come from `last_calibration_columns()` (for LasCal still `lascal_temperature` and `lascal_fallback`); `methods/ac.py` is removed and its entry leaves [[work_plan]] §II.12, with the `cap-ml` commit.
- **`main`:** factory entries `AC-LSB` (`AC(calibrator=LSBinning())`, every dataset) and `L-CBPE-LSB` (`CBPE(acc_name, calib_method="lsb")`, binary datasets, NaN elsewhere, as `L-CBPE`); per bag, the columns of `LSBinning.last_columns()` (`lsb_nbins`, `lsb_nbins_eff`, `lsb_grid_fallback`, `lsb_prevalence`, `lsb_n_degenerate`, `lsb_argmax_changed`); both in `tables/main.py:PMF_INTERVALS` (central and highest-density), outside the main tables, the rank tables and the coverage vs amplitude plot, as `AC` and `L-CBPE`.
- **Table and plot modules:** `tables/ac_lsb.py` (the comparison table, a table of the paired differences, and the diagnostics table with the check of `AC-LSB` against `L-CBPE-LSB`) and `plots/ac_lsb.py` (the signed error against the shift of the bag, with $\pi^V$ rebuilt from the stored `val_prev`, rounded to $4$ decimals), following [[work_plan]] Part V.

Tests (in `uncawe`; those of the calibrators and of `AC` are in `cap-ml`, `tests/test_ac.py` and `tests/test_ls_binning.py`):

- `cap-ml`'s `AC` with LasCal gives, bit for bit, the probability mass function, temperature, fallback flag and interval of the former `methods/ac.py` on seeded synthetic bags (80 bags with $K\in\{2,3,5,10\}$ in a one-off check of 2026-11-24; in the suite, $20$ bags and a fallback, against a frozen copy of `methods/ac.py`, `tests/_reference_ac.py`);
- the factory entries, the per-bag columns stored by `main` for vanilla accuracy only, and the NaN of `L-CBPE-LSB` on multiclass data;
- the tables on synthetic results.

Reproducibility and comparability:

- With `LasCalPerBag`, `AC` and `L-CBPE` give the same results as before wherever the `lascal` package works; on a failure of the package `L-CBPE` now falls back to the uncalibrated posteriors instead of raising. No stored bag of `AC` fell back (E8.1), and the stored `L-CBPE` results have no error, so the stored results of both stay valid and are not rerun.
- `AC-LSB` and `L-CBPE-LSB` add methods to `main`; the bags, $V$ and the posteriors of $h$ are those of `main`, so they are comparable bag by bag with the stored `AC` and `L-CBPE`. Label-shift binning is deterministic (no random component), so the methods take no `random_state`.

# Appendix

## A. Toy simulations

None.

## B. Relation to other extensions

- **[[ext4_uncertainty_budget|Extension 4]]**: the interval of AC represents only an instance-level luck of the bag; the prevalence and recall parts of the error split are left out (§T.6). AC estimates accuracy without a contingency table, so it does not enter the error split, as `DoC`.
- **[[ext1_metrics|Extension 1]]**: AC estimates vanilla accuracy only; the other metrics would need the table of CBPE, which is out of scope.
- **[[ext3_posterior_propagation|Extension 3]]**: the fixed-$\hat C$ methods of E3.1 push each posterior draw $\pi^{(r)}$ of a Bayesian quantifier through the same $\hat C$ of $V$; the methods of E8.2 are their counterpart with one point estimate of $\pi^Q$ and no uncertainty.
- **[[ext4_uncertainty_budget|Extension 4]]**: $e_\pi$ and $e_C$ of §T.8 are a two-part version of the error split of E4.1, with the true prevalence of the bag as weights of the rate part; $e_C$ joins the recall part and the luck of the bag of E4.1.

## C. Out of scope

- extensions of CBPE: a multiclass CBPE, and precision, recall and $F_1$ from its random confusion matrix (user's decision of 2026-11-20);
- the highest-density interval for AC (the central interval of the AC paper is used, user's decision of 2026-11-20);
- in E8.3, a reference with the true prevalence of the bag in place of $\hat\pi^Q$ in label-shift binning (user's decision of 2026-11-24);
- the other confidence-based estimators compared in the AC paper (DoC-Feat, ATC);
- for now, in E8.2: other quantifiers, the posteriors of $h$ as the input of the quantifiers, the no-adaptation reference $\hat\pi^Q=\pi^V$ and comparisons with the methods of `main` or of E3.1 other than the point estimates of BayCAP-KDEy and BayCAP-Hf in the AE tables (user's decisions of 2026-11-23).
