# Extension 8: Average Confidence as an uncertainty-aware baseline

## Claim

Average Confidence (AC) estimates the accuracy of a classifier $h$ on an unlabelled bag as the mean of its confidences, the largest posterior probability of each instance. Kivimäki et al. ([JAIR 2025](papers/Kivimäki%20et%20al.%20-%202025%20-%20Confidence-based%20Estimators%20for%20Predictive%20Performance%20in%20Model%20Monitoring.pdf)) show that, if the confidences are calibrated on the target distribution, AC is unbiased and consistent, and that the number of correct predictions follows a Poisson-binomial distribution, from which they read a central confidence interval for the accuracy of the bag. AC is therefore the simplest uncertainty-aware CAP method: one mean and one interval, both in closed form. This extension adds it to the project as a baseline (user's decision of 2026-11-20), with the confidences recalibrated per bag by LasCal, a calibration method designed for label shift, as `L-CBPE` already does on the binary datasets. AC works for any number of classes, so it extends that baseline to all the datasets of `main`. It is a baseline, not a method under study: the interval represents only the luck of the bag given the calibrated confidences (§T.6), and this is accepted (user's decision of 2026-11-20).

## Status

| Item | Status |
|---|---|
| `L-CBPE`: `cap-ml`'s `CBPE(calib_method="lascal")`, AC with LasCal and the highest-density interval of the Poisson-binomial distribution, binary datasets only (`experiments/methods.py`) | [implemented], run and stored in `main` (binary datasets); excluded from the main tables, the rank table and the coverage vs amplitude plot |
| `AC`: AC with LasCal and the central interval of the Poisson-binomial distribution, any number of classes, vanilla accuracy only (`methods/ac.py`) | [implemented] (card WP-29, branch `wp-29/average-confidence`), run and stored in `main` (75 pairs, 2026-11-20) |
| E8.1 (`AC` against `L-CBPE` in a separate table, `tables/average_confidence.py`) | [implemented], card WP-29, run (no pilot, user's decision) and analysed (below) |

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

# Candidate experiments

- **E8.c1, AC with isotonic calibration on $V$ (AC$_c$).** *Why:* it is the setting of the experiments of the AC paper (isotonic regression fitted on held-out labelled data, §5.2.1), as opposed to its method, which leaves the calibrator open; with no label-shift correction it should show the bias of §T.4. *Needs:* an isotonic map of the correctness $\mathbb 1[\hat y=y]$ on the confidence, fitted on $V$ (our adaptation to $K>2$), and one run of `main`.
- **E8.c2, AC with the confidences corrected by the prevalences.** *Why:* §T.4 shows that, for posteriors calibrated on the source, the exact correction under PPS is a reweighting by $\pi^Q_k/\pi^P_k$, which a temperature cannot represent; with the true $\pi^Q$ it is a reference (it uses target labels), with an estimate $\hat\pi^Q$ (e.g. EMQ) a method. *Needs:* posteriors calibrated on $V$, the reweighting of §T.4 and one run of `main` per variant.
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

The confidence calibrated on $Q$ is then this posterior evaluated at the predicted class, $P_Q(Y=h(x)\mid x)$; it needs $\pi^Q$, which is unknown (candidate E8.c2).

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

# Implementation

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

# Appendix

## A. Toy simulations

None.

## B. Relation to other extensions

- **[[ext4_uncertainty_budget|Extension 4]]**: the interval of AC represents only an instance-level luck of the bag; the prevalence and recall parts of the error split are left out (§T.6). AC estimates accuracy without a contingency table, so it does not enter the error split, as `DoC`.
- **[[ext1_metrics|Extension 1]]**: AC estimates vanilla accuracy only; the other metrics would need the table of CBPE, which is out of scope.

## C. Out of scope

- extensions of CBPE: a multiclass CBPE, and precision, recall and $F_1$ from its random confusion matrix (user's decision of 2026-11-20);
- the highest-density interval for AC (the central interval of the AC paper is used, user's decision of 2026-11-20);
- the other confidence-based estimators compared in the AC paper (DoC-Feat, ATC).
