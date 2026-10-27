# Extension 3: Propagating a Bayesian prevalence posterior to classifier performance

## Claim

A Bayesian quantifier returns a posterior over the target prevalence $\pi^Q$, not just one estimate. This extension adds two CAP methods that turn such a posterior into an interval on classifier performance, and compares them with the methods of the paper. **BayCAP-KDEy** extends the Bayesian KDEy quantifier (BayesKDEy) of [KDE-Bayes](papers/kde_bayes.pdf): it pushes each posterior draw $\pi^{(r)}$ through the rate matrix of the classifier measured on the validation set, $\hat C\operatorname{diag}(\pi^{(r)})$, either as a population table or, with the realised-table predictive of [[ext2_baycap|Extension 2]], as a plausible realised table of the target bag. **SQS-B** does what the other SQS methods do, but draws the prevalences of its validation bags from the posterior of a Bayesian quantifier instead of from a point estimate or a bootstrap. The two methods differ from existing ones in one component each (the quantifier, the handling of $C$, the source of the prevalences), so their comparison with BayCAP and RQBS tells which component matters. The uncertainty budget of [[ext4_uncertainty_budget|Extension 4]] predicts which sources of error each method represents, and so where it should cover well; Extension 4 also reuses the same implementations in its regime map over $\lvert U\rvert$ and $\lvert V\rvert$. The first experiment, E3.1, evaluates BayCAP-KDEy on the protocol of `main`.

## Status

| Item | Status |
|---|---|
| `methods/baycap_kdey.py:BayCAPKDEy`: BayesKDEy posterior, fixed $\hat C$, population pushforward (work plan §II.4) | [implemented], not evaluated |
| Realised pushforward for BayCAP-KDEy (`estimand="realised"`) | [proposal], needed by E3.1 |
| E3.1, BayCAP-KDEy on the `main` protocol, population vs realised | [proposal], not scheduled (no card) |
| SQS-B | [proposal], not implemented |
| BayCAP with a fixed $\hat C$ (marginal posterior of $\pi^Q$ from BayesianCC) | [proposal], not implemented |
| A random rate matrix for BayesKDEy (options 1–3 of Appendix B) | [new hypothesis] |
| The predictions of §T.8 and the moving centre of §T.7 | [new hypothesis] |
| Candidate experiments E3.c1–E3.c5 | [proposal], not scheduled |

This document replaces the former Extensions 8 (`docs/sqs_b.md`) and 9 (`docs/baycap_kdey.md`); their last version is in commit `391220c`, and the material that was set aside is in [[optional_extensions]] §8.

# Experiments

All experiments use the protocol of `main`: the $15$ datasets and $5$ classifiers ($75$ pairs), the $100$ UPP bags of size $1000$ per pair with the seeds of `main`, the measures of the paper (AE, coverage, amplitude, Winkler score at $\alpha=0.05$), and target labels only for scoring. Every interval is scored against the realised table of each bag (`true_cts`). Each experiment has its own NUTS runs (user's decision of 2026-10-27): draws are not shared across experiments.

## E3.1: BayCAP-KDEy on the `main` protocol, population vs realised

**Why.** BayCAP-KDEy is implemented but has never been evaluated. It differs from BayCAP in two ways: the posterior of $\pi^Q$ comes from BayesKDEy, which reads the output probabilities of a surrogate classifier, instead of from BayesianCC, which reads the crisp predictions of $h$; and the rate matrix is fixed at its validation estimate $\hat C$ instead of being sampled. The theory makes four predictions that E3.1 can check (§T.8). First, the population interval is narrow, because a fixed $\hat C$ carries prevalence uncertainty into accuracy only through the differences between the recalls (§T.4), and it leaves out the luck of the bag (§T.5); it should under-cover on most pairs. Second, the realised pushforward should widen it by the factor $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$ found by E2.3, which here should be large. Third, unlike in BayCAP, the realised pushforward can also move the centre, because BayesKDEy does not fit $\pi^Q$ to the predictions of $h$ (§T.7). Fourth, $\hat C$ has no prior, so the pull of the recalls towards $1/K$ that displaces BayCAP on the large-$K$ datasets (E2.2) should be absent. The answers tell whether BayCAP-KDEy is worth completing with a random $C$ (E3.c3–E3.c5) and how it compares with the methods of `main`.

- **Design.**
  - **Inputs:** the $75$ pairs and $100$ bags of `main`; the stored `main` results of every other method (`DoC`, `QuAcc_NxN`, `O-LEAP`, `L-CBPE`, `RQBS`, `PQ`, `BayesCAP` recomputed on CPU, `BayesCAP-real`); the validation counts $f_{ij}$ of $h$, rebuilt as in E2.2; the results of two new methods of `main`, `BayCAP-KDEy` (`BayCAPKDEy` with `estimand="population"`) and `BayCAP-KDEy-real` (`estimand="realised"`), each with its own run.
  - **Varied:** the pushforward of BayCAP-KDEy, population or realised. Held fixed: data, bags and seeds; the BayesKDEy configuration, with an MLP surrogate as in the KDEy-based methods of `main`, temperature $\tau=1$ (tempering is studied in [[ext5_baycap_priors_tempering|Extension 5]]), the uniform prior $\operatorname{Dir}(\mathbf 1)$ on $\pi^Q$, NUTS with $500$ warm-up iterations and $1000$ draws as for `BayesCAP`, and the same NUTS seed for both methods, so that on the same platform the two runs give the same prevalence draws. Kernel, bandwidth $b$ and shrinkage $\lambda$ are still to be decided before the card is written: either the defaults of the code (Gaussian kernel, $b=0.1$, $\lambda=0$) or the shrinkage-regularised Aitchison kernel, the strongest variant in KDE-Bayes, with $(b,\lambda)$ fixed or selected on $V$ alone (KDE-Bayes selects them by the absolute error on Dirichlet bags drawn from a validation split, which uses no target labels).
  - **Measures:** for both methods, all three metrics of `main` (vanilla accuracy, macro-F1, macro-K): AE, coverage, amplitude, Winkler score. For vanilla accuracy only: the predicted coverage $\mathcal C_s$ of E2.2; the split of the shift into recall term $R$ and prevalence term $P$ (E2.2, §T.5 of [[ext2_baycap]]), with prior part $R_0=0$, since $\hat C$ has no prior; the widening factor of the realised interval against $\sqrt{1+(\hat\sigma_g/\hat\sigma_p)^2}$, with $\hat\sigma_g$ and $\hat\sigma_p$ computed from the draws as in §T.7; the movement of the centre, the realised point estimate minus the population one; the label-free mismatch $\lVert n^t/N^Q-\hat C\bar\pi\rVert_1$ between the shares of the bag that $h$ predicts as each class and the shares that the posterior-mean prevalence $\bar\pi$ implies (§T.7); the spread of the recalls of $h$ on $V$, $\operatorname{sd}_j(\hat C_{jj})$ (§T.4). Two paired bootstrap tests of the mean Winkler difference, as in E2.3 ($10{,}000$ seeded resamples of the $100$ per-bag differences, Holm-adjusted over the $75$ pairs): `BayCAP-KDEy-real` against `BayCAP-KDEy`, and `BayCAP-KDEy-real` against `BayesCAP-real`.
  - **Estimand:** population table for `BayCAP-KDEy`, realised table for `BayCAP-KDEy-real`; both scored against the realised table of each bag, as every method of `main`. The split of the shift uses the bags' true class proportions and recalls, so it is an analysis, not a method; $n^t$ and $\bar\pi$ use no target labels.
  - **Cost:** two runs of `main` on `bcuda` with `JAX=cpu`, one per method, each $75\times100=7500$ BayesKDEy NUTS runs, plus one fit of the surrogate and of the kernel densities per pair. A pilot on two datasets estimates the time; the KDE log-densities of the bag are computed once before sampling, so each NUTS step costs about $N^QK$ operations.
- **Representation.**
  - **Comparison table:** one row per pair, grouped by dataset and ordered by $K$: for `BayCAP-KDEy`, `BayCAP-KDEy-real`, `BayesCAP`, `BayesCAP-real` and `RQBS`, coverage, amplitude and Winkler score; the widening factor, observed and predicted; the movement of the centre and the mismatch; the bootstrap intervals and Holm-adjusted p-values of the two Winkler tests. CSV, LaTeX and landscape PDF, like `tables/pushforward.py`.
  - **Rank table:** mean Winkler and AE ranks of all methods of `main` with the two new methods, per metric (`tables/rank.py` with the new methods included).
  - **Decomposition table:** the table of E2.2 (diagnostics, interval, error) for both new methods, with `tables/displacement.py --method`.
  - **Scatter, coverage:** coverage of `BayesCAP-real` on the horizontal axis, of `BayCAP-KDEy-real` on the vertical axis, one point per pair, coloured by $K$, with lines at $0.95$ on both axes; a second panel with the population versions.
  - **Scatter, width:** observed against predicted widening factor, one point per pair, with the line $y=x$, coloured by the mismatch.
  - **Scatter, centre:** movement of the centre against the mismatch, one point per pair.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussion of 2026-10-27. They are not scheduled and are numbered E3.c<n>; one becomes an experiment of the main section only when the user decides so, and then it gets the next number E3.<n> and a full specification.

- **E3.c1, SQS-B on BayesKDEy.** *Why:* SQS-B draws one validation bag of size $\lvert U\rvert$ for each joint posterior draw $\pi^{(r)}$, with class counts $\approx\lvert U\rvert\pi^{(r)}_j$, and applies the metric to the table of $h$ on that bag. Against RQBS it changes only the source of the prevalences (posterior instead of bootstrap of $U$); against `BayCAP-KDEy-real` it changes only the bridge from prevalence to table: a bag simulated forwards from $V$, from true classes to predictions, instead of true classes sampled backwards given the observed predictions of $h$ on $U$. Given $\pi^{(r)}$, the accuracy of an SQS-B bag has mean $\hat d^\top\pi^{(r)}$ and the binomial variance of §T.5 computed with the recalls on $V$, so the budget predicts that SQS-B and `BayCAP-KDEy-real` represent the same sources of error (§T.8). *Needs:* an SQS-B method with the conventions of `RQBS` in `main` (surrogate quantifier, use of $V$, rounding of the class counts, sampling with replacement), to be read from `cap-ml` before implementation; the BayesKDEy configuration of E3.1; its own NUTS runs.
- **E3.c2, BayCAP with a fixed $\hat C$.** *Why:* it pushes the marginal draws $\pi^{(r)}$ of BayCAP's posterior through $\hat C$, in both pushforwards. Compared with BayCAP-KDEy it changes only the posterior of $\pi^Q$ (BayesianCC on the predictions of $h$ instead of BayesKDEy on a surrogate); compared with `BayesCAP` on the same draws it changes only whether $C$ is random, so it measures how much the uncertainty on $C$ and the pull of its prior towards $1/K$ act at the bridge, separately from their effect on the posterior of $\pi^Q$. *Needs:* a pushforward option in `methods/baycap.py:BayCAP` that uses $\hat C$ in place of $C^{(r)}$; its own rerun of BayCAP.
- **E3.c3, BayCAP-KDEy with a random $C$ (option 1 of Appendix B).** *Why:* it pairs each draw $\pi^{(r)}$ with an independent draw $C^{(r)}_{\cdot j}\sim\operatorname{Dir}(f_{1j},\dots,f_{Kj})$ of each column of the rate matrix, which adds the error of $C$ estimated on $V$ (the term $e_2$ of Extension 4) without a new posterior. It is BayCAP's posterior on $C$ from $V$ alone, but with the prior $\operatorname{Dir}(\mathbf 0)$ instead of $\operatorname{Dir}(\mathbf 1)$, so it has no pull towards $1/K$. *Decision rule:* if it brings coverage near nominal on the pairs where `BayCAP-KDEy-real` under-covers, the random $C$ at the bridge is enough; if the displacement remains, the missing error of the kernel densities (term $e_1^V$) matters, and E3.c4 is the next candidate. *Needs:* a third pushforward in `BayCAPKDEy`; its own run; a convention for cells with $f_{ij}=0$, which stay at $0$ in every draw.
- **E3.c4, cut posterior with random kernel densities (option 2 of Appendix B)** [new hypothesis]. *Why:* random weights on the validation instances make both the kernel densities of BayesKDEy and $C$ random, coherently, and add the term $e_1^V$. *Needs:* $M$ NUTS runs per bag, one per draw of the weights, so a cost about $M$ times that of E3.1; a choice of $M$.
- **E3.c5, full joint posterior of prevalence and weights (option 3 of Appendix B)** [new hypothesis]. *Why:* the analogue of BayCAP for BayesKDEy: the target bag also updates the weights, as it updates $C$ in BayesianCC. *Needs:* NUTS on $K-1+\lvert V\rvert$ parameters, which may be slow or mix poorly; to be considered only if E3.c4 is worthwhile.

# Theory

This part explains what E3.1 needs, step by step. We follow the notation of the reference paper. There are $K$ classes $\omega_1,\dots,\omega_K$. $h$ is the classifier whose performance we estimate. $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ is the probability that $h$ predicts $\omega_i$ for an instance of true class $\omega_j$, so column $j$ of $C$ sums to one. $\pi^Q$ is the vector of class prevalences of the target bag $U$, which has $N^Q=\lvert U\rvert$ instances. $V$ is the labelled validation set, with $n^s_j$ instances of class $\omega_j$. A contingency table $T$ has predicted classes on rows and true classes on columns.

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

Let $f_{ij}$ be the number of validation instances of true class $\omega_j$ that $h$ predicts as $\omega_i$. BayCAP-KDEy uses the fractions

$$
\hat C_{ij}=\frac{f_{ij}}{n^s_j}.
$$

**Why the fractions.** Each of the $n^s_j$ instances of class $\omega_j$ is predicted as $\omega_i$ with probability $C_{ij}$, independently of the others. The count $f_{ij}$ is then a sum of $n^s_j$ indicators (variables equal to $1$ when the event happens and $0$ otherwise), each with mean $C_{ij}$. The mean of a sum is the sum of the means, so $\mathbb E[f_{ij}]=n^s_jC_{ij}$ and $\mathbb E[\hat C_{ij}]=C_{ij}$: on average over validation sets, the fraction is right (it is *unbiased*). It is also the maximum-likelihood estimate, the value of $C_{\cdot j}$ under which the observed counts are most probable.

**How wrong it can be.** An indicator with mean $p$ has variance $p(1-p)$: its mean is $p$, the mean of its square is also $p$ (because $1^2=1$ and $0^2=0$), and the variance is the mean of the square minus the square of the mean, $p-p^2$. Variances of independent variables add, and multiplying a variable by a constant $c$ multiplies its variance by $c^2$. So the estimated recall $\hat d_j=\hat C_{jj}$ has variance

$$
\operatorname{Var}(\hat d_j)=\frac{1}{(n^s_j)^2}\,n^s_j\,d_j(1-d_j)=\frac{d_j(1-d_j)}{n^s_j}.
$$

BayCAP-KDEy treats $\hat C$ as if it were the true $C$, so this error, the term $e_2$ of [[ext4_uncertainty_budget|Extension 4]], does not appear in its interval.

**No pull towards $1/K$.** BayCAP gives each column of $C$ the prior $\operatorname{Dir}(\mathbf 1)$, which acts like one imaginary validation instance per cell and pulls the estimated recall to $(f_{jj}+1)/(n^s_j+K)$, towards $1/K$ ([[ext2_baycap]] §T.5). $\hat C$ has no prior, so it has no such pull.

*Implementation note:* `BayCAPKDEy.fit` computes $\hat C$ with `confusion_matrix(..., normalize="true")`, which puts true classes on rows, so the code stores $\hat C^\top$.

## T.3 BayesKDEy

*Intuition:* BayesKDEy describes how the output probabilities of a surrogate classifier are spread within each class, and asks which mix of classes best explains the outputs observed on the target bag; NUTS lists many plausible mixes.

**The surrogate.** BayesKDEy does not use the predictions of $h$. It trains its own classifier $s$ (an MLP in BayCAP-KDEy) on $V$ and works with its output probabilities $z=s(x)$, a point of the simplex $\Delta^{K-1}$ (a vector of $K$ non-negative numbers summing to one). The outputs of the validation instances are obtained by cross-fitting (QuaPy's default, $5$ folds): each instance gets the output of an $s$ trained without it, so the outputs look like those on new data; on the target bag, $s$ is trained on all of $V$.

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

## T.4 The population pushforward and its closed form for accuracy

*Intuition:* with a fixed rate matrix, accuracy depends on the prevalence only through the differences between the recalls; if all classes are recognised equally well, the mix of classes does not matter and the interval shrinks to a point.

**The method.** For each draw, BayCAP-KDEy forms the table $T^{(r)}=\hat C\operatorname{diag}(\pi^{(r)})$ and the metric value $a^{(r)}=A(T^{(r)})$; the point estimate is the mean of the $a^{(r)}$, and the interval runs from their $2.5\%$ to their $97.5\%$ percentile. For vanilla accuracy, by §T.1,

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

## T.5 Population accuracy and realised accuracy

*Intuition:* even with the right prevalence and the right rates, the accuracy on this particular bag is random, because $h$ is right on a random subset of its instances.

`main` scores every interval against the *realised accuracy*, the fraction of the bag's instances that $h$ classifies correctly. The bag has $n_j=N^Q\pi^Q_j$ instances of class $\omega_j$ (UPP fixes the class counts, up to rounding), drawn with replacement from the test pool. Each is classified correctly with probability $d_j$ (the recall on the pool), independently of the others, so by the argument of §T.2 the number of correct predictions $c_j$ in class $\omega_j$ is a binomial count with mean $n_jd_j$ and variance $n_jd_j(1-d_j)$. The counts of different classes are independent, so their variances add, and dividing by $N^Q$ divides the variance by $(N^Q)^2$:

$$
A_{\text{real}}=\frac{1}{N^Q}\sum_jc_j,
\qquad
\mathbb E[A_{\text{real}}]=\sum_j\pi^Q_jd_j,
\qquad
\operatorname{Var}(A_{\text{real}})=\frac{1}{N^Q}\sum_j\pi^Q_jd_j(1-d_j).
$$

The mean is the population accuracy; the variance is the term $e_3$ of Extension 4, and it shrinks as $1/N^Q$. The population interval of §T.4 contains no part of it. With the numbers of the example of §T.4, the population interval has half-width $1.96\cdot0.004\approx0.008$ while the realised accuracy varies by about $\pm1.96\cdot0.012\approx\pm0.024$ around the population accuracy, so the interval misses the realised accuracy about half of the time even if $\pi$ and $\hat C$ are right (by the rule of §T.8 with $\eta=(0.012/0.004)^2=9$, coverage $2\Phi(1.96/\sqrt{10})-1\approx0.46$).

## T.6 The realised pushforward

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

## T.7 Width and centre of the realised interval

*Intuition:* the realised samples vary for two reasons, because the draws of $\pi$ differ and because each draw adds the luck of the bag; the two spreads add up. The centre can move, because BayesKDEy does not choose $\pi$ to match the predictions of $h$.

Write $\tilde a$ for a realised sample: first a draw $\pi$, then the multinomial picks of §T.6.

**Centre.** The law of total expectation says that the mean of $\tilde a$ is the mean, over the draws, of its mean given the draw: $\mathbb E[\tilde a]=\mathbb E\big[m(\pi)\big]$, with, by §T.6 and the mean $np$ of a binomial,

$$
m(\pi)=\mathbb E[\tilde a\mid\pi]=\sum_i\frac{n^t_i}{N^Q}\,\rho_{ii}(\pi).
$$

The population accuracy of the same draw can be written in the same form. From the definition of $\rho_{ii}$, $q_i\rho_{ii}=\hat C_{ii}\pi_i$, so

$$
a(\pi)=\sum_i\hat C_{ii}\pi_i=\sum_iq_i(\pi)\,\rho_{ii}(\pi),
\qquad
m(\pi)-a(\pi)=\sum_i\Big(\frac{n^t_i}{N^Q}-q_i(\pi)\Big)\rho_{ii}(\pi).
$$

The two centres differ only where the observed shares $n^t_i/N^Q$ differ from the shares $q_i(\pi)$ that the draw expects. In BayCAP the posterior of $\pi$ is fitted to $n^t$ through $C$, so $q\approx n^t/N^Q$, and E2.3 found the two centres within $0.009$ on every pair. BayesKDEy fits $\pi$ to the outputs of the surrogate $s$, not to the predictions of $h$, so nothing forces the expected shares $q(\pi)=\hat C\pi$ to match the observed ones: where the posterior of BayesKDEy and the predictions of $h$ disagree, the realised centre moves [new hypothesis]. Under PPS with the right $\pi$ and $C$, $\mathbb E[n^t_i/N^Q]=q_i$, and the mismatch $\lVert n^t/N^Q-\hat C\bar\pi\rVert_1$ is only sampling noise, of order $1/\sqrt{N^Q}$; a larger mismatch says that $\bar\pi$ or $\hat C$ is wrong, and it can be computed without target labels. Whether the moved centre is closer to the realised accuracy is an open question that E3.1 answers: the realised version keeps the observed shares and takes from $(\hat C,\pi)$ only the precisions.

**Width.** The law of total variance says that the variance of $\tilde a$ is the mean of its variance within a draw plus the variance of its mean across draws:

$$
\operatorname{Var}(\tilde a)=\mathbb E\big[\operatorname{Var}(\tilde a\mid\pi)\big]+\operatorname{Var}\big(m(\pi)\big).
$$

Given $\pi$, the counts of different predicted classes come from separate multinomial draws, so they are independent, and the binomial variance $np(1-p)$ of §T.2 gives

$$
\sigma_g^2(\pi)=\operatorname{Var}(\tilde a\mid\pi)=\frac{1}{(N^Q)^2}\sum_in^t_i\,\rho_{ii}(\pi)\big(1-\rho_{ii}(\pi)\big).
$$

If $m(\pi)-a(\pi)$ is about the same for all draws (a constant shift does not change a variance), $\operatorname{Var}(m(\pi))\approx\operatorname{Var}(a(\pi))=\sigma_p^2$, the variance of the population samples of §T.4, and

$$
\operatorname{Var}(\tilde a)\approx\sigma_p^2+\bar\sigma_g^2,
\qquad
\bar\sigma_g^2=\mathbb E\big[\sigma_g^2(\pi)\big].
$$

For roughly normal samples the $95\%$ interval is the mean $\pm1.96$ standard deviations, so its width is proportional to the standard deviation, and the realised interval is wider than the population one by the factor

$$
\frac{\sqrt{\sigma_p^2+\bar\sigma_g^2}}{\sigma_p}=\sqrt{1+\big(\bar\sigma_g/\sigma_p\big)^2}.
$$

E2.3 found this factor within $\pm5\%$ on every pair for BayCAP. For BayCAP-KDEy, $\sigma_p$ is small (§T.4) while $\bar\sigma_g$ is the luck of the bag, about $0.012$ in the example, so the factor should be large (about $3.2$ in the example of §T.4). If $m(\pi)-a(\pi)$ varies with the draw, the factor departs from this value; E3.1 measures both.

## T.8 What the uncertainty budget predicts

*Intuition:* an interval is wide only along the sources of error it simulates; the sources it leaves out move its centre without widening it.

[[ext4_uncertainty_budget|Extension 4]] splits the error of a CAP estimate into four parts: $e_3$, the luck of the bag (§T.5); $e_1^U$, the error of the prevalence from seeing only $N^Q$ target instances; $e_1^V$, the error of the prevalence from fitting the quantifier on a finite $V$, including its bias; $e_2$, the error of the rates of $h$ measured on $V$ (§T.2). From §T.2–§T.7:

| Method | Represented | Missing |
|---|---|---|
| `BayCAP-KDEy` (population) | $e_1^U$ | $e_1^V$, $e_2$, $e_3$ |
| `BayCAP-KDEy-real` | $e_1^U$, $e_3$ | $e_1^V$, $e_2$ |
| `BayesCAP` (population) | $e_1^U$, $e_1^V$, $e_2$ (through the posterior of $C$) | $e_3$ |
| `BayesCAP-real` | all | none |
| `RQBS`; SQS-B (E3.c1) | $e_1^U$, $e_3$ | $e_1^V$, $e_2$ |

With $\eta$ the ratio of the missing to the represented variance, Extension 4 §3.5 derives that a $95\%$ interval covers with probability about $2\Phi\big(1.96/\sqrt{1+\eta}\big)-1$, where $\Phi$ is the cumulative distribution function of the standard normal: $0.95$ when nothing is missing, and less the more is missing. The predictions for E3.1 [new hypotheses]:

- **P1, the population interval under-covers.** It misses $e_3$, and its represented part is small when the recalls are similar (§T.4); its coverage should be lowest on the pairs with the smallest spread of the recalls, $\operatorname{sd}_j(\hat d_j)$.
- **P2, the realised interval is much wider.** The widening factor follows $\sqrt{1+(\bar\sigma_g/\sigma_p)^2}$ and is larger than for BayCAP, so coverage rises more than from `BayesCAP` to `BayesCAP-real`.
- **P3, no prior displacement.** On `dry-bean`, `hand_digits`, `letter` and `isolet`, where the shift of `BayesCAP` equals the prior part $R_0$ of its recall term (E2.2), the recall term of BayCAP-KDEy should be small, because $\hat C$ is unbiased (§T.2); what remains of the shift is the prevalence term, from the error of BayesKDEy, and the difference between the recalls on $V$ and on the test pool (cause 4 of E2.2).
- **P4, the missing terms from $V$ matter where $V$ is small.** Where the validation classes are small (e.g. `poker_hand`, with $3$ instances in the smallest class), $e_2$ and $e_1^V$ are large and neither version represents them; these are the pairs where E3.c3 and E3.c4 could help.
- **P5, `BayCAP-KDEy-real` and `RQBS` represent the same terms.** Their coverage should be similar where their quantifiers are equally good; differences then come from the quantifier (BayesKDEy against PACC) or from the moved centre of §T.7.

# Implementation

Components (for E3.1):

- **`methods/baycap_kdey.py:BayCAPKDEy` with `estimand`** $\in\{$`"population"`, `"realised"`$\}$, default `"population"`, which keeps the current output. `"realised"` takes $n^t$ from the `posteriors` argument of `predict_ct_range` (the predictions of $h$ on $U$, currently accepted but unused) and applies `methods/baycap.py:realised_tables` to the population draws; the multinomial draws use a NumPy `Generator` seeded from `mcmc_seed`, as in `BayCAP`. The tables stay in the `cap-ml` layout (true classes on rows).
- **Reproducible surrogate:** `BayCAPKDEy` builds `MLPClassifier()` without a `random_state` (work plan Part IV); E3.1 needs a fixed one, so that the two runs share their prevalence draws and a rerun reproduces them. The fix changes the output of `BayCAPKDEy`, which has no stored results.
- **BayesKDEy configuration:** kernel, bandwidth and shrinkage as decided for E3.1 (see its design); if $(b,\lambda)$ are selected on $V$, the selection uses only $V$ and is part of `fit`.
- **Methods in `main`:** factory entries `BayCAP-KDEy` and `BayCAP-KDEy-real` in `experiments/methods.py` (`METHOD_CLASS` and `gen_methods`), run in the BayCAP pool with `BAYES_ENV` (CPU, single-threaded XLA); results only: excluded from the tables, ranks and default plots of `main`, like `BayesCAP-real`.
- **Tables and plots of E3.1:** a comparison module reusing the paired bootstrap of `tables/pushforward.py`; `tables/displacement.py --method` for the decomposition, with $R_0=0$ for methods with a fixed $\hat C$; the mismatch and the widening factor need, per bag, $n^t$ and the population draws or their moments, which `main` must store for the two new methods (or which are recomputed from the stored test posteriors of $h$).

Tests:

- `estimand="population"` reproduces the current output of `BayCAPKDEy` for a fixed seed;
- with the same seed, both estimands use identical prevalence draws;
- realised tables have predicted-class margins equal to $n^t/N^Q$; with $\hat C=I$ they return the observed counts;
- for accuracy, the mean of the population samples equals $\hat d^\top\bar\pi$, and their variance equals $(\hat d-\bar d\mathbf 1)^\top\hat\Sigma_\pi(\hat d-\bar d\mathbf 1)$ on the draws;
- the factory builds both methods with the configuration of E3.1, and the tables, ranks and default plots of `main` exclude them.

Reproducibility and comparability:

- E3.1 adds the rows of two methods to `main` and leaves the stored results of every other method untouched, as E2.3 did for `BayesCAP-real`; the paper's tables are unchanged;
- every row records the BayesKDEy configuration, the seeds and the pushforward;
- the two methods are comparable with the CPU results of `BayesCAP` and `BayesCAP-real` (same bags, same platform); a pilot on two datasets precedes the full run.

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

- **[[ext1_metrics|Extension 1]]**: supplies the metrics; macro-F1 and macro-K have no closed form under the fixed $\hat C$ (§T.4).
- **[[ext2_baycap|Extension 2]]**: supplies the realised-table predictive (§T.6), the decomposition of the shift and the paired Winkler test (E2.2, E2.3), and the finding that the prior on $C$ displaces BayCAP on large-$K$ datasets, which P3 tests from the other side.
- **[[ext4_uncertainty_budget|Extension 4]]**: supplies the budget of §T.8; its regime map (E4.2) includes the methods of this extension, with the same implementations, so this extension does not vary $\lvert U\rvert$ or $\lvert V\rvert$.
- **[[ext5_baycap_priors_tempering|Extension 5]]**: tempering, including the temperature of BayesKDEy.
- **Extension 7** ([[extensions]] §7): tuning of the surrogate and of the hyperparameters of BayesKDEy beyond the configuration fixed for E3.1.
- **[[optional_extensions]]**: §3 keeps Aitchison-SQS; §8 keeps the material set aside from the former Extensions 8 and 9.

## D. Out of scope

- tempering of BayesKDEy ($\tau=1$ throughout; Extension 5);
- varying $\lvert U\rvert$ and $\lvert V\rvert$ (Extension 4);
- time and memory measurements (optional, user's decision of 2026-10-27; [[optional_extensions]] §8);
- the sensitivity analyses and ablations of the former documents ([[optional_extensions]] §8);
- violations of PPS ([[optional_extensions]] §2).
