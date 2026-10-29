# Extension 3: Propagating a Bayesian prevalence posterior to classifier performance

## Claim

A Bayesian quantifier returns a posterior over the target prevalence $\pi^Q$, not just one estimate. This extension adds CAP methods that turn such a posterior into an interval on classifier performance, and compares them with the methods of the paper. **BayCAP with a fixed $\hat C$** takes the posterior of a Bayesian quantifier and pushes each draw $\pi^{(r)}$ through the rate matrix of the classifier measured on the validation set, $\hat C\operatorname{diag}(\pi^{(r)})$, either as a population table or, with the realised-table predictive of [[ext2_baycap|Extension 2]], as a plausible realised table of the target bag. Two quantifiers are considered: the Bayesian KDEy of [KDE-Bayes](papers/kde_bayes.pdf) (BayesKDEy, giving **BayCAP-KDEy**), which estimates from the validation set how the outputs of a classifier are spread within each class, and QuaPy's BayesianMAPLS, a Bayesian version of the EM quantifier EMQ (giving **BayCAP-EMQ**), which reads the outputs as calibrated probabilities. Each quantifier can read the outputs of the classifier $h$ whose performance we estimate, or those of a separate MLP trained on the validation set. **SQS-B** does what the other SQS methods do, but draws the prevalences of its validation bags from the posterior of a Bayesian quantifier instead of from a point estimate or a bootstrap. The methods differ from existing ones in one component at a time (the quantifier, the classifier it reads, the handling of $C$, the source of the prevalences), so their comparison with BayCAP and RQBS tells which component matters. The uncertainty budget of [[ext4_uncertainty_budget|Extension 4]] predicts which sources of error each method represents, and so where it should cover well; Extension 4 also reuses the same implementations in its regime map over $\lvert U\rvert$ and $\lvert V\rvert$. The first experiment, E3.1, evaluates the four combinations of quantifier and input classifier, with both pushforwards, on the protocol of `main`.

## Status

| Item | Status |
|---|---|
| `methods/baycap_kdey.py:BayCAPKDEy`: BayesKDEy on an MLP, fixed $\hat C$, population pushforward (work plan §II.4) | [implemented], not evaluated |
| BayesianMAPLS as quantifier, $h$ as input classifier, realised pushforward | [proposal], needed by E3.1 |
| E3.1, BayCAP with a fixed $\hat C$: quantifier × input classifier × pushforward, on the `main` protocol | [proposal], scheduled (card WP-8 of [[work_plan]]), not run |
| SQS-B | [proposal], not implemented |
| BayCAP with a fixed $\hat C$ and the posterior of BayesianCC | [proposal], not implemented |
| A random rate matrix for BayesKDEy (options 1–3 of Appendix B); BayesianMAPLS with a uniform prior or a random calibration map (Appendix E) | [new hypothesis] |
| The predictions of §T.9 and the moving centre of §T.8 | [new hypothesis] |
| Candidate experiments E3.c1–E3.c7 | [proposal], not scheduled |

This document replaces the former Extensions 8 (`docs/sqs_b.md`) and 9 (`docs/baycap_kdey.md`); their last version is in commit `391220c`, and the material that was set aside is in [[optional_extensions]] §8.

# Experiments

All experiments use the protocol of `main`: the $15$ datasets and $5$ classifiers ($75$ pairs), the $100$ UPP bags of size $1000$ per pair with the seeds of `main`, the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), and target labels only for scoring. Every interval is scored against the realised table of each bag (`true_cts`). Each experiment has its own NUTS runs (user's decision of 2026-10-27): draws are not shared across experiments.

## E3.1: BayCAP with a fixed $\hat C$: quantifier × input classifier × pushforward, on the `main` protocol

**Why.** BayCAP-KDEy is implemented but has never been evaluated, and it differs from BayCAP in three components at once: the quantifier (BayesKDEy instead of BayesianCC), the classifier whose outputs the quantifier reads (a separate MLP instead of $h$), and the rate matrix (fixed at its validation estimate $\hat C$ instead of sampled). E3.1 keeps the rate matrix fixed and crosses the other two components, so that their effects can be told apart. The quantifier is BayesKDEy, which estimates the distribution of the outputs within each class from $V$ (§T.3), or BayesianMAPLS, which takes it from the outputs themselves, read as calibrated probabilities (§T.4). The input classifier is $h$ or a separate MLP trained on $V$. Each of the four cells is run with both pushforwards. The theory makes predictions that E3.1 can check (§T.9). First, the population interval is narrow, because a fixed $\hat C$ carries the prevalence uncertainty into accuracy only through the differences between the recalls (§T.5), and it leaves out the luck of the bag (§T.6); it should under-cover on most pairs. Second, the realised pushforward should widen it by the factor $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$ found by E2.3, which here should be large. Third, unlike in BayCAP, the realised pushforward can also move the centre, because neither quantifier fits $\pi^Q$ to the predictions of $h$, and it should move less when the quantifier reads the outputs of $h$ (§T.8). Fourth, $\hat C$ has no prior, so the pull of the recalls towards $1/K$ that displaces BayCAP on the large-$K$ datasets (E2.2) should be absent. Fifth, BayCAP-EMQ should be worse where its input classifier is poorly calibrated, and its prior, which is not uniform (§T.4), may displace it on the bags with a class prevalence near $0$. The answers tell which quantifier and which input classifier to use for SQS-B (E3.c1) and for a random $C$ (E3.c3–E3.c5), and how these methods compare with those of `main`.

- **Design.**
  - **Inputs:** the $75$ pairs and $100$ bags of `main`; the stored `main` results of every other method (`DoC`, `QuAcc_NxN`, `O-LEAP`, `L-CBPE`, `RQBS`, `PQ`, `BayesCAP` recomputed on CPU, `BayesCAP-real`); the validation counts $f_{ij}$ of $h$, rebuilt as in E2.2; the results of eight new methods of `main`, each with its own run: `BayCAP-<Q>-<s>` (population) and `BayCAP-<Q>-<s>-real` (realised) for the quantifier $Q\in\{$`KDEy`, `EMQ`$\}$ and the input classifier $s\in\{$`h`, `mlp`$\}$, e.g. `BayCAP-EMQ-h-real`.
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

    Paired bootstrap tests of the mean Winkler difference, as in E2.3 ($10{,}000$ seeded resamples of the $100$ per-bag differences, Holm-adjusted over the $75$ pairs, separately for each test): in each cell, the realised method against the population one ($4$ tests) and against `BayesCAP-real` ($4$ tests); among the realised methods, `h` against `mlp` for each quantifier ($2$ tests), and `KDEy` against `EMQ` for each input classifier ($2$ tests).
  - **Estimand:** population table for the four methods without `-real`, realised table for the four with `-real`; all scored against the realised table of each bag, as every method of `main`. The split of the shift and the bins of $\min_j\pi^Q_j$ use the bags' true class proportions, so they are analyses, not methods; $n^t$, $\bar\pi$ and the ECE on $V$ use no target labels.
  - **Cost:** eight runs of `main` on `bcuda` with `JAX=cpu`, one per method, each $75\times100=7500$ NUTS runs, plus, per pair and cell, one fit of the quantifier (and of the MLP for `mlp`). A pilot on two datasets times the four cells. For both quantifiers the per-instance terms of the likelihood (the kernel densities of BayesKDEy, the outputs of the input classifier for BayesianMAPLS) are computed once before sampling, so each NUTS step costs about $N^QK$ operations.
- **Representation.**
  - **Comparison table, realised:** one row per pair, grouped by dataset and ordered by $K$: for the four realised methods, `BayesCAP-real` and `RQBS`, coverage, amplitude and Winkler score; the widening factor, observed and predicted; the movement of the centre and the mismatch; the ECE of the two input classifiers; the bootstrap intervals and Holm-adjusted p-values of the Winkler tests. CSV, LaTeX and landscape PDF, like `tables/pushforward.py`.
  - **Comparison table, population:** the same for the four population methods and `BayesCAP`, without the widening factor and the centre.
  - **Summary of the design:** for each metric, a $2\times2$ table per pushforward (quantifier on rows, input classifier on columns) of the mean Winkler rank, mean coverage and mean amplitude over the $75$ pairs, also split by the classifier $h$.
  - **Rank table:** mean Winkler and AE ranks of all methods of `main` with the eight new methods, per metric (`tables/rank.py` with the new methods included).
  - **Decomposition table:** the table of E2.2 (diagnostics, interval, error) for the eight new methods, with `tables/displacement.py --method`.
  - **Scatter, coverage:** coverage of `BayesCAP-real` on the horizontal axis, of each realised method on the vertical axis, one panel per cell and one point per pair, coloured by $K$, with lines at $0.95$ on both axes; a second figure with the population versions against `BayesCAP`.
  - **Scatter, width:** observed against predicted widening factor, one panel per cell, one point per pair, with the line $y=x$, coloured by the mismatch.
  - **Scatter, centre:** movement of the centre against the mismatch, one panel per cell, one point per pair.
  - **Calibration:** coverage of each method against the ECE of its input classifier, one panel per cell, one point per pair, coloured by the classifier $h$; and coverage against the bins of $\min_j\pi^Q_j$, one line per realised method.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussions of 2026-10-27 and 2026-10-28. They are not scheduled and are numbered E3.c<n>; one becomes an experiment of the main section only when the user decides so, and then it gets the next number E3.<n> and a full specification.

- **E3.c1, SQS-B on a Bayesian quantifier.** *Why:* SQS-B draws one validation bag of size $\lvert U\rvert$ for each joint posterior draw $\pi^{(r)}$, with class counts $\approx\lvert U\rvert\pi^{(r)}_j$, and applies the metric to the table of $h$ on that bag. Against RQBS it changes only the source of the prevalences (posterior instead of bootstrap of $U$); against the realised BayCAP method of E3.1 with the same quantifier it changes only the bridge from prevalence to table: a bag simulated forwards from $V$, from true classes to predictions, instead of true classes sampled backwards given the observed predictions of $h$ on $U$. Given $\pi^{(r)}$, the accuracy of an SQS-B bag has mean $\hat d^\top\pi^{(r)}$ and the binomial variance of §T.6 computed with the recalls on $V$, so the budget predicts that SQS-B and the realised methods of E3.1 represent the same sources of error (§T.9). *Needs:* an SQS-B method with the conventions of `RQBS` in `main` (surrogate quantifier, use of $V$, rounding of the class counts, sampling with replacement), to be read from `cap-ml` before implementation; the quantifier and input classifier of one cell of E3.1, chosen from its results; its own NUTS runs.
- **E3.c2, BayCAP with a fixed $\hat C$ and the posterior of BayesianCC.** *Why:* it pushes the marginal draws $\pi^{(r)}$ of BayCAP's posterior through $\hat C$, in both pushforwards. Compared with the methods of E3.1 it changes only the posterior of $\pi^Q$ (BayesianCC on the predictions of $h$ instead of BayesKDEy or BayesianMAPLS on soft outputs); compared with `BayesCAP` on the same draws it changes only whether $C$ is random, so it measures how much the uncertainty on $C$ and the pull of its prior towards $1/K$ act at the bridge, separately from their effect on the posterior of $\pi^Q$. *Needs:* a pushforward option in `methods/baycap.py:BayCAP` that uses $\hat C$ in place of $C^{(r)}$; its own rerun of BayCAP.
- **E3.c3, a random $C$ at the bridge (option 1 of Appendix B).** *Why:* it pairs each draw $\pi^{(r)}$ with an independent draw $C^{(r)}_{\cdot j}\sim\operatorname{Dir}(f_{1j},\dots,f_{Kj})$ of each column of the rate matrix, which adds the error of $C$ estimated on $V$ (the term $e_2$ of Extension 4) without a new posterior. It is BayCAP's posterior on $C$ from $V$ alone, but with the prior $\operatorname{Dir}(\mathbf 0)$ instead of $\operatorname{Dir}(\mathbf 1)$, so it has no pull towards $1/K$. It applies to either quantifier. *Decision rule:* if it brings coverage near nominal on the pairs where the realised methods of E3.1 under-cover, the random $C$ at the bridge is enough; if the displacement remains, the missing error of the quantifier from $V$ (term $e_1^V$) matters, and E3.c4 (for BayesKDEy) or E3.c7 (for BayesianMAPLS) is the next candidate. *Needs:* a third pushforward in the class of E3.1; its own run; a convention for cells with $f_{ij}=0$, which stay at $0$ in every draw.
- **E3.c4, cut posterior with random kernel densities for BayesKDEy (option 2 of Appendix B)** [new hypothesis]. *Why:* random weights on the validation instances make both the kernel densities of BayesKDEy and $C$ random, coherently, and add the term $e_1^V$. *Needs:* $M$ NUTS runs per bag, one per draw of the weights, so a cost about $M$ times that of one method of E3.1; a choice of $M$.
- **E3.c5, full joint posterior of prevalence and weights for BayesKDEy (option 3 of Appendix B)** [new hypothesis]. *Why:* the analogue of BayCAP for BayesKDEy: the target bag also updates the weights, as it updates $C$ in BayesianCC. *Needs:* NUTS on $K-1+\lvert V\rvert$ parameters, which may be slow or mix poorly; to be considered only if E3.c4 is worthwhile.
- **E3.c6, BayesianMAPLS with a uniform prior** (Appendix E). *Why:* the NUTS model of QuaPy's `BayesianMAPLS` gives $\pi$ a logistic-normal prior, whose density goes to $0$ at the edges of the simplex (§T.4), while BayesKDEy uses the uniform prior $\operatorname{Dir}(\mathbf 1)$, which matches the UPP bags of `main`. So the quantifier factor of E3.1 mixes two differences, the likelihood and the prior. Rerunning the `EMQ` cells with a uniform prior separates them; it is worthwhile if E3.1 finds the `EMQ` methods displaced on the bags with a small $\min_j\pi^Q_j$. *Needs:* a NUTS model that samples $\pi$ from the Dirichlet directly, or that keeps the ILR coordinates and replaces the normal factor with the log-Jacobian of the transformation (Appendix E); its own runs of the `EMQ` methods.
- **E3.c7, BayesianMAPLS with a random calibration map** (Appendix E) [new hypothesis]. *Why:* BayesianMAPLS has no kernel densities or validation counts to randomise: its model of the classes is the calibration of the input classifier (§T.4). A small parametric recalibration of the outputs, one temperature and $K$ class biases (bias-corrected temperature scaling, Alexandari et al. 2020), with parameters $\theta$, can be sampled jointly with $\pi$: the labels of $V$ give a likelihood on $\theta$, and the bag gives the EM likelihood on $(\pi,\theta)$. It is the analogue of BayesianCC's joint posterior of $(\pi,C)$, with $K+1$ extra parameters instead of the $\lvert V\rvert$ of E3.c5, so it would add the calibration part of $e_1^V$ cheaply and coherently. The rate matrix of $h$ at the bridge stays a separate object (Appendix E). *Needs:* a NumPyro model of its own, a prior on $\theta$, its own runs; to be considered if E3.1 finds the `EMQ` methods sensitive to the calibration of their input classifier.

# Theory

This part explains what E3.1 needs, step by step. We follow the notation of the reference paper. There are $K$ classes $\omega_1,\dots,\omega_K$. $h$ is the classifier whose performance we estimate. $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that $h$ predicts $\omega_i$ for an instance of true class $\omega_j$, so column $j$ of $C$ sums to one. $\pi^Q$ is the vector of class prevalences of the target bag $U$, which has $N^Q=\lvert U\rvert$ instances. $V$ is the labelled validation set, with $n^s_j$ instances of class $\omega_j$. A contingency table $T$ has predicted classes on rows and true classes on columns. $s$ is the input classifier of the quantifier, $h$ itself or a separate MLP, and $z=s(x)$ its output probabilities, a point of the simplex $\Delta^{K-1}$ (a vector of $K$ non-negative numbers summing to one).

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

*Implementation note:* `BayCAPKDEy.fit` computes $\hat C$ with `confusion_matrix(..., normalize="true")`, which puts true classes on rows, so the code stores $\hat C^\top$.

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

# Implementation

Components (for E3.1):

- **One class for the eight methods.** Generalise `methods/baycap_kdey.py:BayCAPKDEy` (implemented, no stored results) with three parameters: `quantifier` $\in\{$`"kdey"`, `"emq"`$\}$, building QuaPy's `BayesianKDEy` or `BayesianMAPLS`; `input_classifier` $\in\{$`"h"`, `"mlp"`$\}$; `estimand` $\in\{$`"population"`, `"realised"`$\}$. The defaults (`"kdey"`, `"mlp"`, `"population"`) keep the current output up to the seed of the MLP (below). The name of the class and of the module may change with the generalisation; the card decides.
- **Input classifier `h`.** The quantifier skips QuaPy's classifier step: `fit` calls `aggregation_fit` on the `posteriors` argument (the outputs of $h$ on $V$) and the labels of $V$, and `predict_ct_range` calls `sample_from_posterior` on its `posteriors` argument (the outputs of $h$ on $U$); the class order of the posteriors of $h$ must be that of `val.classes_`. With `"mlp"`, the quantifier is fitted with `fit(*val.Xy)` as now, with QuaPy's $5$-fold cross-fitting.
- **Realised pushforward.** `"realised"` takes $n^t$ from the `posteriors` argument of `predict_ct_range` (the arg-max of the outputs of $h$ on $U$) and applies `methods/baycap.py:realised_tables` to the population draws; the multinomial draws use a NumPy `Generator` seeded from `mcmc_seed`, as in `BayCAP`. The tables stay in the `cap-ml` layout (true classes on rows).
- **Reproducible MLP:** `BayCAPKDEy` builds `MLPClassifier()` without a `random_state` (work plan Part IV); E3.1 needs a fixed one, so that the population and realised methods of a cell share their prevalence draws and a rerun reproduces them. The fix changes the output of `BayCAPKDEy`, which has no stored results.
- **Configuration:** BayesKDEy with `kernel="gaussian"`, `bandwidth=0.1`, `shrinkage=0.0`; BayesianMAPLS with QuaPy's defaults (`prior="uniform"`, `exact_train_prev=True`, `mapls_chain_init=True`); for both, `temperature=1`, `num_warmup=500`, `num_samples=1000`, `mcmc_seed` as for `BayesCAP`.
- **Source prevalence of BayesianMAPLS with `h`:** the outputs of $h$ refer to the prevalences of $L$, on which $h$ was trained, while QuaPy sets $\pi^P$ to those of $V$. Before the card, check that $V$ is a stratified split of the same data as $L$, so that the two are close; if they are not, the card decides whether to keep QuaPy's behaviour.
- **Methods in `main`:** eight factory entries `BayCAP-<Q>-<s>` and `BayCAP-<Q>-<s>-real` in `experiments/methods.py` (`METHOD_CLASS` and `gen_methods`), run in the BayCAP pool with `BAYES_ENV` (CPU, single-threaded XLA); results only: excluded from the tables, ranks and default plots of `main`, like `BayesCAP-real`.
- **Tables and plots of E3.1:** a comparison module reusing the paired bootstrap of `tables/pushforward.py`; `tables/displacement.py --method` for the decomposition, with $R_0=0$ for methods with a fixed $\hat C$; the mismatch and the widening factor need, per bag, $n^t$ and the population draws or their moments, which `main` must store for the new methods (or which are recomputed from the stored test posteriors of $h$); the ECE needs the validation outputs of each input classifier, stored at fit time.

Tests:

- with the defaults, the class reproduces the output of `BayCAPKDEy` for fixed seeds;
- for each quantifier and input classifier, with the same seeds, both estimands use identical prevalence draws;
- with `input_classifier="h"`, the quantifier receives exactly the posteriors passed to `fit` and `predict_ct_range`, and no classifier is trained;
- with `quantifier="emq"`, the source prevalence equals the class proportions of $V$;
- realised tables have predicted-class margins equal to $n^t/N^Q$; with $\hat C=I$ they return the observed counts;
- for accuracy, the mean of the population samples equals $\hat d^\top\bar\pi$, and their variance equals $(\hat d-\bar d\mathbf 1)^\top\hat\Sigma_\pi(\hat d-\bar d\mathbf 1)$ on the draws;
- the factory builds the eight methods with the configuration of E3.1, and the tables, ranks and default plots of `main` exclude them.

Reproducibility and comparability:

- E3.1 adds the rows of eight methods to `main` and leaves the stored results of every other method untouched, as E2.3 did for `BayesCAP-real`; the paper's tables are unchanged;
- every row records the quantifier and its configuration, the input classifier, the seeds and the pushforward;
- the eight methods are comparable with the CPU results of `BayesCAP` and `BayesCAP-real` (same bags, same platform); a pilot on two datasets precedes the full run.

# Appendix

## A. Toy simulations

None yet.

## B. Making the rate matrix random for BayesKDEy (background for E3.c3–E3.c5)

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
3. *Full joint posterior* (E3.c5): NUTS samples $(\pi,w)$ together, with likelihood $\sum_{i\in U}\log\sum_j\pi_j\sum_{v\in V_j}w_{j,v}K_b(z_i,z_v)$, so $U$ also updates the weights, as it updates $C$ in BayesianCC. It has $\lvert V\rvert$ extra parameters, and since $U$ updates the weights only through the outputs of $s$, the predictions of $h$ on $U$ do not inform $C$.

## C. Relation to other extensions

- **[[ext1_metrics|Extension 1]]**: supplies the metrics; macro-F1 and macro-K have no closed form under the fixed $\hat C$ (§T.5).
- **[[ext2_baycap|Extension 2]]**: supplies the realised-table predictive (§T.7), the decomposition of the shift and the paired Winkler test (E2.2, E2.3), and the finding that the prior on $C$ displaces BayCAP on large-$K$ datasets, which P3 tests from the other side.
- **[[ext4_uncertainty_budget|Extension 4]]**: supplies the budget of §T.9; its regime map (E4.2) includes the methods of this extension, with the same implementations, so this extension does not vary $\lvert U\rvert$ or $\lvert V\rvert$.
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

a temperature $t>0$ that makes the outputs more or less confident and class biases $\beta_y$ that shift them towards some classes (bias-corrected temperature scaling, Alexandari et al. 2020; one bias can be fixed at $0$, since adding the same number to all of them changes nothing). The joint posterior of $(\pi,\theta)$ has two likelihoods: on $V$, $\sum_{v\in V}\log s^\theta_{y_v}(x_v)$, the log-probability of the observed labels under the recalibrated outputs, which involves $\theta$ only; on $U$, the likelihood of §T.4 with $s^\theta$ in place of $s$, which involves both. The labels of $V$ locate $\theta$, the bag also updates it, as it updates $C$ in BayesianCC, and the uncertainty about $\theta$ widens the posterior of $\pi$. With $K$ parameters added to the $K-1$ of $\pi$, NUTS stays cheap, unlike the $\lvert V\rvert$ weights of E3.c5.

**The bridge stays separate.** The recalibration changes the outputs that the quantifier reads, not the predictions of $h$, which are fixed. So it does not make the rate matrix $\hat C$ of $h$ random: with $s=h$ the two are computed from the same outputs but are different functions of them, and a random $C$ at the bridge still needs option 1 of Appendix B (E3.c3).
