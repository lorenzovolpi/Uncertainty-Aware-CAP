# Extension 5: Prevalence priors and tempering for BayCAP-Hf and BayCAP-KDEy

## Claim

The two Bayesian CAP methods kept by Extensions 2 and 3, BayCAP-Hf (`BayesCAP-Hf-real`, [[ext2_baycap|Extension 2]] E2.8) and BayCAP-KDEy with an MLP input and a fixed $\hat C$ (`BayCAP-KDEy-mlp-real`, [[ext3_posterior_propagation|Extension 3]] E3.1), both put a uniform prior on the target prevalence $\pi^Q$ and use their likelihoods at full strength ($\tau=1$). The paper names informative priors and posterior tempering as unexplored features of BayCAP. This extension measures two things, one experiment each: how much the knowledge of the distribution of the target prevalences improves the intervals of the two methods, when the prior on $\pi^Q$ is that distribution (E5.1); and how much tempering, which widens the posterior by giving the data less weight, improves them when the temperature is chosen without the labels of the target dataset, from the other datasets (E5.2). Theory predicts small effects in both cases (§T.2, §T.6), except for the prior with many classes and small bags: a prior on $\pi^Q$ weighs little against a bag of $1000$ instances, and neither a prior on $\pi^Q$ nor a wider posterior moves an interval displaced by the error that comes from $V$, which is where both methods fail. The extension is kept deliberately thin (user's decision of 2026-11-10): the prior on $C$, the comparison of tempering with the realised-table predictive and the selection of $\tau$ on validation bags of the earlier version are set aside or are candidates.

## Status

| Item | Status |
|---|---|
| BayCAP-Hf: `methods/baycap.py:BayCAP` with `c_prior="hierarchical-f"`, `estimand="realised"`; argument `prior` (Dirichlet prior on $\pi^Q$) | [implemented]; evaluated with the uniform prior only ([[ext2_baycap|Extension 2]] E2.8) |
| `temperature` argument of `BayCAP` (joint tempering of $\mathcal L_V$ and $\mathcal L_U$) | [proposal] |
| BayCAP-KDEy: `methods/baycap_kdey.py:BayCAPKDEy` with `c_prior="fixed"`, MLP input, `estimand="realised"`; arguments `prior` and `temperature` (the latter divides the log-likelihood of the bag) | [implemented]; evaluated with the uniform prior and $\tau=1$ only ([[ext3_posterior_propagation|Extension 3]] E3.1) |
| `binform` experiment: bags at prevalences drawn from a non-uniform $G$ (built from $\pi^V$), `connect-4` only, the old `BayesCAP` with two priors on $\pi^Q$, $\lvert U\rvert=1000$ | [implemented] ([[work_plan]] §II.2); to be extended for E5.1, with the $G$ of E5.1 |
| Toy simulation of §T.3 (`scripts/toys/ext5_prior_calibration.py`) | [implemented] |
| E5.1 (prior on $\pi^Q$), E5.2 (tempering) | [proposal], no card |
| Candidates E5.c1 (tempering of one likelihood term), E5.c2 (selection of $\tau$ on validation bags), E5.c3 (prior $\times$ temperature) | [proposal], not scheduled |
| Prior on $C$ ($\operatorname{Dir}(1/K)$, diagonal-aware prior), comparison of tempering with the realised-table predictive, the full grid of priors on $\pi^Q$, hypotheses H1, H4, H5, H7, H8 and the paper outline of the earlier version | set aside on 2026-11-10, in [[optional_extensions]] §12 |

**Renumbering of 2026-11-10** (user's decision to restructure the extension): the earlier E5.4 (its minimal check: priors on $\pi^Q$ under a non-uniform $G$) becomes E5.1; the earlier E5.1 (temperature sweep) and E5.2 (Calibrated BayCAP against tempering) become E5.2, on the two methods above and without the comparison with the predictive, which is now part of both methods; the label-free temperature selection of the earlier E5.5 is split: its selection across datasets is part of E5.2, its calibration on validation bags becomes the candidate E5.c2; the earlier E5.3 (prior on $C$) and the full study of E5.4 are set aside. The hypothesis IDs of the earlier version are not used any more.

# Experiments

**Common settings.** The two methods as they are in `main`: `BayesCAP-Hf-real` (the Hf prior on $C$, centred shares, the NUTS settings of E2.8) and `BayCAP-KDEy-mlp-real` (BayesKDEy at the Gaussian kernel, $b=0.1$, $\lambda=0$, the MLP input trained on $V$, the fixed $\hat C$ of $h$ on $V$), both with the realised pushforward. The $75$ (dataset, classifier) pairs of `main` ($15$ datasets, five classifiers) and $100$ bags per pair and configuration. Measures for vanilla accuracy, macro-F1 and macro-K: the absolute error (AE) of the point estimate, coverage, amplitude and Winkler score at $\alpha=0.05$. Every score is computed against the realised table of each bag (`true_cts`), the quantity that both methods target. The five pairs of a dataset share its $V$ and its test pool, so for effects that depend on $V$ there are about $15$ independent units, not $75$.

**Order.** E5.1 and E5.2 are independent; E5.2 needs the new argument of `BayCAP` only for BayCAP-Hf.

## E5.1: Knowledge of the prior on the prevalence

**Why.** Under the protocol of `main` the target prevalences are drawn uniformly on the simplex, so the uniform prior of both methods is already the well-specified one (§T.3), and the question "how much would knowing the prior help" cannot be asked there. When the prevalences come from a non-uniform distribution $G$ and the prior is $G$, the posterior is calibrated on average over the bags and is narrower than with the uniform prior; the gain depends on the prior weight $c/(c+N_{\text{eff}})$, where $N_{\text{eff}}$ is the number of perfectly labelled instances that the bag is worth (§T.2). With the concentration $c=3K$ of E5.1, the prediction is a small gain at $\lvert U\rvert=1000$ except where $N_{\text{eff}}$ is small (classes poorly separated by $h$ or by the MLP) or $K$ is large ($c=78$ at $K=26$, so $w\ge0.07$), a larger one at $\lvert U\rvert=100$ ($w\ge0.06$ at $K=2$, $w\ge0.44$ at $K=26$), and none on the pairs where the interval is displaced by the error from $V$ (§T.4). A prior with the wrong mean measures the cost of believing in wrong knowledge. $G$ is built so that it carries the same amount of information for every number of classes $K$ and differs from the uniform distribution on every dataset (user's decision of 2026-11-10, §T.3).

- **Design.**
  - **Inputs:** the $75$ pairs of `main` (pre-trained classifiers, $V$ and test pool of each dataset); for each pair and each bag size, $100$ bags drawn from the test pool at prevalences drawn from $G=\operatorname{Dir}(\lambda K\mu^\star)$, with $\lambda=3$. The mean $\mu^\star$ and a second mean $\mu^\dagger$, used by the misspecified prior, are fixed points of the simplex, one pair of points per dataset (shared by its five classifiers): both are drawn from $\operatorname{Dir}(\mathbf 1)$ with a seed fixed per dataset and recorded with the results, and drawn again until each is at $L_1$ distance at least $0.5$ from the centre $\mathbf 1/K$ and they are at $L_1$ distance at least $0.5$ from each other (§T.3). The requested prevalences and the bags are drawn with seeded generators, as in `binform`, so the bags are the same for every prior and method. A class with a small $\mu^\star_k$ has a parameter $\lambda K\mu^\star_k$ below $1$ and is often almost absent from the bags; this is part of what the well-specified prior knows.
  - **Varied:** the prior on $\pi^Q$: uniform, $\operatorname{Dir}(\mathbf 1)$ (the default); well specified, $\operatorname{Dir}(\lambda K\mu^\star)=G$; misspecified, $\operatorname{Dir}(\lambda K\mu^\dagger)$, with the same concentration as $G$ and a wrong mean; the bag size $\lvert U\rvert\in\{100,1000\}$; the method (BayCAP-Hf, BayCAP-KDEy). Held fixed: $\tau=1$, the bags, everything else as in `main`.
  - **Measures:** the common measures per bag, and per pair their means over the $100$ bags; per pair, the value of the knowledge, $\Delta W_{\text{know}}=W_{\text{uniform}}-W_{\text{well-specified}}$, the cost of wrong knowledge, $\Delta W_{\text{wrong}}=W_{\text{misspecified}}-W_{\text{uniform}}$ (both positive when they mean what their name says), and the ratio of the mean amplitude with the well-specified prior to that with the uniform prior; a paired Wilcoxon signed-rank test over the $75$ per-pair means for each of the two differences, per method and bag size.
  - **Estimand:** the realised table of each bag, for both methods and every prior, scored against it.
  - **Cost:** $3$ priors $\times$ $2$ bag sizes $\times$ $2$ methods on the $75$ pairs, $900$ runs of $100$ bags; the uniform prior at $\lvert U\rvert=1000$ cannot reuse the results of `main`, whose bags are drawn uniformly. BayCAP-KDEy samples $\pi$ alone and is cheap; BayCAP-Hf costs, per prior and bag size, about one run of `BayesCAP-Hf-real` in `main`. A pilot on three datasets with the five classifiers first.
- **Scope.** E5.1 measures how much; it has no decision rule.
- **Representation.**
  - **Summary table:** for each method and metric, one row per bag size and prior, with the means over the $75$ pairs of AE, coverage, amplitude and Winkler score, the number of pairs where the prior has a lower mean Winkler score than the uniform one, and the $p$-value of the paired test; also split by $K$ ($K=2$, $3\le K\le7$, $K\ge8$).
  - **Scatter:** per method, the Winkler score with the uniform prior (horizontal) against that with the well-specified and with the misspecified prior (vertical), one point per pair, coloured by $K$, one panel per bag size, with the diagonal.
- **Analysis.** Not run.

## E5.2: Tempering

**Why.** Tempering divides the log-likelihood by $\tau$, which is the same as pretending that the data are $\tau$ times smaller (§T.5): the centre of the posterior stays in place and its spread grows by about $\sqrt\tau$. BayCAP-KDEy has a single likelihood term, that of the bag, so tempering widens only the part of its interval that comes from $\pi$, and leaves out the error of the rates of $h$ on $V$ where E3.1 found it failing; where the luck of the bag dominates the interval, as on binary datasets, the interval barely widens (§T.6). BayCAP-Hf is tempered jointly, its validation term and its bag term both (user's decision of 2026-11-10), so the rates of $h$ become more uncertain too, which reaches the error from $V$ (§T.7). Tempering cannot move a displaced centre, so the question is whether the extra width buys enough coverage to lower the Winkler score, and with which $\tau$. A $\tau$ chosen with the target labels always looks at least as good as $\tau=1$, even when tempering does nothing (§T.8), so the tempered method is the one whose $\tau$ is chosen on the other datasets (leave one dataset out), and the $\tau$ chosen with the target labels is only a reference (user's decision of 2026-11-10).

- **Design.**
  - **Inputs:** the protocol of `main`: the $75$ pairs, $100$ bags of size $1000$ per pair at prevalences uniform on the simplex, with the seeds of `main`; the stored results of `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real` as the runs at $\tau=1$ (reused if the new argument reproduces them exactly, as tested; otherwise run again).
  - **Varied:** $\tau\in\{0.5,1,2,4,8\}$; BayCAP-Hf tempered jointly ($\mathcal L_V$ and $\mathcal L_U$), BayCAP-KDEy on its bag term. Held fixed: the uniform prior on $\pi^Q$, the bags, the NUTS settings, everything else as in `main`.
  - **Measures:** the common measures per bag and their per-pair means at every $\tau$; the leave-one-dataset-out $\tau$ (§T.8): for each held-out dataset $d$, per method and metric, $\tau_{-d}$ is the value with the lowest mean, over the $70$ pairs of the other $14$ datasets, of the per-pair mean Winkler score, and the measures of the pairs of $d$ are read at $\tau_{-d}$; per dataset, the mean over its five pairs of the per-pair mean Winkler score at $\tau=1$ and at $\tau_{-d}$; as references, per pair the oracle-best $\tau$, the one with the lowest mean Winkler score, and per method the global best $\tau$, the one with the lowest mean over all $75$ pairs; for BayCAP-KDEy, the observed widening of the amplitude with respect to $\tau=1$ against the prediction $\sqrt{(\tau+r^2)/(1+r^2)}$ with $r=\bar\sigma_g/\sigma_p$ stored per bag (§T.6); for BayCAP-Hf, the NUTS diagnostics at every $\tau$ (divergent draws, split-$\hat R$), since a sharper posterior ($\tau<1$) can make the sampler worse.
  - **Estimand:** the realised table of each bag, for both methods and every $\tau$, scored against it. The leave-one-dataset-out $\tau$ uses no label of the dataset it is applied to, so it is a method; the oracle-best and the global best $\tau$ are chosen with the target labels, so they are references, not methods.
  - **Cost:** $4$ new values of $\tau$ $\times$ $2$ methods on the $75$ pairs, $600$ runs of $100$ bags; for BayCAP-Hf each value costs about one run of `BayesCAP-Hf-real` in `main`. A pilot on three datasets with the five classifiers first.
- **Decision rule** (user's decision of 2026-11-10): for each method and for vanilla accuracy, tempering is retained if the leave-one-dataset-out $\tau$ lowers the per-dataset mean Winkler score with respect to $\tau=1$, by a paired Wilcoxon signed-rank test over the $15$ datasets with $p<0.05$; the datasets, not the pairs, are the units, because the five pairs of a dataset share $V$ and their $\tau$ is chosen together. Otherwise $\tau=1$ is kept, and the oracle-best $\tau$ is reported only as an upper bound on what tempering can give. If tempering is retained but $\tau_{-d}$ varies widely across the held-out datasets, the selection on validation bags (E5.c2) becomes worth its cost.
- **Representation.**
  - **Line plot:** per method and metric, the mean Winkler score, coverage and amplitude against $\tau$ (log scale), the mean over the pairs in bold and each pair as a faint line, one panel per measure; the same split by $K$ in a second figure.
  - **Table:** per method and metric, one column per $\tau$, with the means over the pairs of AE, coverage, amplitude and Winkler score, plus a column for the leave-one-dataset-out $\tau$, with the $p$-value of the test of the decision rule, and greyed reference columns for the global best $\tau$ and the oracle-best $\tau$ of each pair; the value of $\tau_{-d}$ for each held-out dataset, and the distribution of the oracle-best $\tau$ over the pairs.
  - **Scatter (BayCAP-KDEy):** observed against predicted widening, one point per pair and $\tau$, coloured by $K$.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussions; they are not scheduled. One becomes an experiment of the main section only when the user decides so.

- **E5.c1, tempering one likelihood term of BayCAP-Hf.** *Why:* joint tempering widens both the rates of $h$ and the prevalence; tempering $\mathcal L_U$ only leaves the validation counts at full weight, and $\mathcal L_V$ only widens the rates alone, so the two variants tell which term does the work, if joint tempering helps in E5.2. *Needs:* a flag of the `temperature` argument of `BayCAP` selecting the term, and two more sweeps of BayCAP-Hf on the protocol of E5.2.
- **E5.c2, selection of $\tau$ on validation bags.** *Why:* the leave-one-dataset-out $\tau$ of E5.2 is one value per method; if tempering helps but the best $\tau$ differs much between datasets, a $\tau$ chosen for each pair is needed, and the only label-free source for it is $V$. Bags drawn from the same $V$ on which the method is fitted cannot show the error from $V$ (§T.8), so $V$ must be split into a part that fits the method and a part from which the calibration bags are drawn, as in the earlier E5.5 ([[optional_extensions]] §12); the method is then fitted on less data and is not comparable with `main`. *Needs:* a stratified split of $V$, calibration bags drawn from the held-out part, one NUTS run per calibration bag and value of $\tau$, and the reference runs of both methods fitted on the reduced $V$. Worth it only under the condition stated in the decision rule of E5.2.
- **E5.c3, prior $\times$ temperature.** *Why:* tempering raises the weight of the prior on $\pi^Q$ (§T.5), so with a well-specified prior a $\tau>1$ might widen the interval where the data are wrong while the prior keeps it in place, and with a misspecified prior it would make the damage larger. *Needs:* the bags of E5.1 and a few values of $\tau$, run only if both E5.1 and E5.2 show an effect.

# Theory

Notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the classifier $h$, $d_j=C_{jj}$ its recall on class $\omega_j$, $\pi^Q$ the prevalence of the target bag $U$ of $N=\lvert U\rvert$ instances, and $T=C\operatorname{diag}(\pi^Q)$ the population table, with predicted classes on rows and true classes on columns. $\tau$ is the temperature.

## T.1 Where the prior and the temperature enter the two methods

**BayCAP-Hf.** The posterior is over $(\pi^Q,C)$ and the parameters $(m,\rho,\kappa)$ of the Hf prior on $C$ ([[ext2_baycap|Extension 2]] §T.16):

$$
p(\pi^Q,C,m,\rho,\kappa\mid V,U)\;\propto\;\underbrace{\mathcal L_V(C)}_{\text{counts of }V}\;\underbrace{\mathcal L_U(\pi^Q,C)}_{\text{predicted counts of }U}\;p(\pi^Q)\;p(C\mid m,\rho,\kappa)\;p(m,\rho,\kappa).
$$

$\mathcal L_V(C)$ is a product of multinomials: in each true class $\omega_j$ of $V$, the $n^s_j$ instances fall in the predicted classes with probabilities $C_{:j}$. $\mathcal L_U(\pi^Q,C)$ is the multinomial of the $N$ predictions of $h$ on $U$, with probabilities $C\pi^Q$. Joint tempering replaces the two likelihoods with $[\mathcal L_V\mathcal L_U]^{1/\tau}$ and leaves the priors unchanged.

**BayCAP-KDEy.** The rate matrix is held at $\hat C$, the rates of $h$ on $V$, and NUTS samples $\pi^Q$ alone, from

$$
p(\pi^Q\mid U)\;\propto\;\Big[\prod_{n=1}^N\sum_j\pi^Q_j\,\hat p_j(x_n)\Big]^{1/\tau}p(\pi^Q),
$$

where $x_n$ are the outputs of the MLP on the instances of $U$ and $\hat p_j$ the kernel density of those outputs within class $\omega_j$, estimated on $V$ ([[ext3_posterior_propagation|Extension 3]] §T.3). There is no validation term to temper: $V$ enters only through $\hat C$ and $\hat p_j$, which are fixed.

**The interval.** Both methods turn each posterior draw into a realised table of $U$ (the realised pushforward, [[ext2_baycap|Extension 2]] §T.3), and the interval is the percentile interval of the metric over these tables. In both methods the prior on $\pi^Q$ is $\operatorname{Dir}(\alpha)$, the argument `prior`, uniform ($\alpha=\mathbf 1$) by default. We write $\alpha=c\,\mu$, with $\mu$ the prior mean and $c=\sum_k\alpha_k$ its concentration; the uniform prior has $\mu=\mathbf 1/K$ and $c=K$.

## T.2 How much a prior on the prevalence weighs

**The simplest case.** Take a perfect and known classifier, so that the predicted counts $n_1,\dots,n_K$ of $U$ are its true class counts, multinomial with probabilities $\pi^Q$. The tempered likelihood is $\prod_k(\pi^Q_k)^{n_k/\tau}$ (T.5) and the prior density is proportional to $\prod_k(\pi^Q_k)^{c\mu_k-1}$. Multiplying the two, the posterior density is proportional to $\prod_k(\pi^Q_k)^{c\mu_k+n_k/\tau-1}$, which is again a Dirichlet (the Dirichlet is conjugate to the multinomial):

$$
\pi^Q\mid U\;\sim\;\operatorname{Dir}\!\Big(c\,\mu+\frac{n}{\tau}\Big).
$$

The mean of component $k$ of a $\operatorname{Dir}(a)$ is $a_k/\sum_la_l$. Here $\sum_la_l=c+N/\tau$, so

$$
\mathbb E[\pi^Q_k\mid U]=\frac{c\mu_k+n_k/\tau}{c+N/\tau}=w\,\mu_k+(1-w)\,\frac{n_k}{N},\qquad w=\frac{c}{c+N/\tau},
$$

where the second form follows by writing $c\mu_k=w(c+N/\tau)\mu_k$ and $n_k/\tau=(1-w)(c+N/\tau)\,n_k/N$. The **prior weight** $w$ is the share of the posterior mean that comes from the prior; the rest comes from the observed proportions. The variance of component $k$ of a $\operatorname{Dir}(a)$ is $\bar a_k(1-\bar a_k)/(\sum_la_l+1)$, with $\bar a_k$ its mean, so the posterior standard deviation is about

$$
\operatorname{sd}(\pi^Q_k\mid U)\approx\sqrt{\frac{\bar\pi_k(1-\bar\pi_k)}{c+N/\tau+1}} .
$$

*Intuition:* the prior acts as $c$ imaginary instances added to the $N/\tau$ real ones. With $c=10$ and $\tau=1$, $w=0.010$ at $N=1000$ and $w=0.091$ at $N=100$; at the same posterior mean, adding the prior shrinks the standard deviation by the factor $\sqrt{(N+1)/(N+c+1)}$, which is $0.995$ at $N=1000$ and $0.954$ at $N=100$.

**A real classifier is worth fewer instances.** With confusions, the predicted counts tell less about $\pi^Q$ than true labels would. In the binary case with recalls $d_1$ (class $\omega_1$) and $d_0$ (class $\omega_0$) and known, the probability that an instance is predicted as $\omega_1$ is

$$
q=d_1\pi_1+(1-d_0)(1-\pi_1)=(1-d_0)+\pi_1\,(d_1+d_0-1),
$$

so $\pi_1=(q-(1-d_0))/(d_1+d_0-1)$. The observed share $\hat q$ of predictions $\omega_1$ is a proportion of $N$ independent instances, with variance $q(1-q)/N$; the estimate $\hat\pi_1=(\hat q-(1-d_0))/(d_1+d_0-1)$ is a linear function of it, and $\operatorname{Var}(aX+b)=a^2\operatorname{Var}(X)$, so

$$
\operatorname{Var}(\hat\pi_1)=\frac{q(1-q)}{N\,(d_1+d_0-1)^2}.
$$

With true labels it would be $\pi_1(1-\pi_1)/N$. Equating the two defines the **effective size** of the bag, the number of true labels that it is worth:

$$
N_{\text{eff}}=N\,(d_1+d_0-1)^2\,\frac{\pi_1(1-\pi_1)}{q(1-q)} .
$$

For example, $d_1=d_0=0.8$ and $\pi_1=0.5$ give $q=0.5$ and $N_{\text{eff}}=0.36\,N$: a bag of $100$ is worth $36$ labels, and the prior weight of $c=10$ becomes $10/(10+36)=0.22$. With $K$ classes the same holds direction by direction in the simplex, and the directions that $C$ separates worst (small singular values) have the smallest effective size. In BayCAP-KDEy the outputs of the MLP replace the predictions, and $N_{\text{eff}}$ is larger the less the densities $\hat p_j$ overlap; in BayCAP-Hf the uncertainty on $C$ lowers it further. The prior on $\pi^Q$ matters only where $N_{\text{eff}}/\tau$ is not much larger than $c$.

## T.3 Well-specified prior and calibration on average

*Intuition:* a prior is "correct" when it describes how often each target situation actually occurs; then, and only then, Bayesian intervals are calibrated on average over those situations.

In every experiment, bag prevalences are produced by a known mechanism: $\pi^Q_i\sim G$, then a bag $U_i$ with that prevalence is sampled. A prior is **well specified** when $p(\pi^Q)=G$. Under the protocol of `main`, $G=\operatorname{Dir}(\mathbf 1)$, which is exactly the uniform prior of both methods: the paper already evaluates BayCAP with a well-specified prior on $\pi^Q$, and under that protocol any informative prior is misspecified by construction. This is why E5.1 draws its bags from a non-uniform $G$.

If the prior equals $G$ and the likelihood is the true sampling model, a $95\%$ credible interval contains the true value for exactly $95\%$ of the bags. The reason is Bayes' theorem: the posterior is then the actual conditional distribution of the true $\pi^Q$ among all bags that produce the same observed data. The guarantee is an average over bags drawn from $G$, not a property of each bag. A vague wrong prior usually keeps coverage and loses sharpness; a confident wrong prior can lose coverage.

Toy example (`scripts/toys/ext5_prior_calibration.py`): binary, perfect known classifier, $n\sim\operatorname{Bin}(20,\pi_1)$, Beta priors, $200{,}000$ bags, $95\%$ equal-tailed intervals.

| $G$ | prior | coverage | mean width | coverage for $\pi_1\in[0.4,0.6)$ |
|---|---|---|---|---|
| uniform | uniform (well specified) | $0.950$ | $0.33$ | $0.95$ |
| uniform | $\operatorname{Beta}(2,8)$ | $0.65$ | $0.31$ | $0.83$ |
| $\operatorname{Beta}(2,8)$ | $\operatorname{Beta}(2,8)$ (well specified) | $0.950$ | $0.26$ | $0.86$ |
| $\operatorname{Beta}(2,8)$ | uniform | $0.955$ | $0.31$ | $0.95$ |
| $\operatorname{Beta}(2,8)$ | $\operatorname{Beta}(8,2)$ | $0.26$ | $0.34$ | $0.81$ |

With a non-uniform $G$ the uniform prior keeps coverage but its intervals are about $20\%$ wider than those of the well-specified prior: that difference is the value of the prior information, and the formula of T.2 predicts it ($n=20$, $c=10$: $\sqrt{21/31}=0.82$). With bags of $100$ or $1000$ instances the same formula predicts a much smaller value, unless $N_{\text{eff}}$ is small.

The guarantee has three conditions, and each can fail in the two methods: the prior must equal $G$; the likelihood must be the true sampling model (the rates and densities are estimated on $V$ and then treated as exact); and the interval must target the quantity that is scored (it does, with the realised pushforward). A well-specified prior removes only the first cause of miscalibration.

**The choice of $G$ in E5.1.** The amount of information that $G=\operatorname{Dir}(c\,\mu)$ adds to the uniform prior depends on its concentration $c$ compared with $K$, the concentration of the uniform prior. The variance of component $k$ of a $\operatorname{Dir}(c\,\mu)$ is $\mu_k(1-\mu_k)/(c+1)$ (T.2). At the same mean, the variance of the uniform prior divided by that of $G$ is therefore

$$
\frac{\mu_k(1-\mu_k)/(K+1)}{\mu_k(1-\mu_k)/(c+1)}=\frac{c+1}{K+1}.
$$

With a fixed concentration, such as the $c=10$ of `binform`, this ratio is $11/3\approx3.7$ at $K=2$, $1$ at $K=10$ (no information about the spread) and $11/27\approx0.4$ at $K=26$ ($G$ more spread than the uniform distribution, its parameters $10/26\approx0.38$ below $1$, so its bags lie near the faces of the simplex). The knowledge would then mean different things on different datasets, and nothing at $K=10$. With $c=\lambda K$ the ratio is $(\lambda K+1)/(K+1)$, which is $2.3$ at $K=2$ and tends to $\lambda$ as $K$ grows: with $\lambda=3$, $G$ is about three times less spread than the uniform prior on every dataset, and at $K=3$ its concentration, $9$, is close to that of `binform`. The mean of `binform`, $\mu^\star_k=(1-\pi^V_k)/(K-1)$, is close to the centre $\mathbf 1/K$ when $V$ is nearly balanced, since then $\pi^V_k\approx1/K$ and $(1-1/K)/(K-1)=1/K$; and its "no shift" prior, centred on $\pi^V$, is then close to $G$ itself. E5.1 therefore draws its two means as fixed points of the simplex, required to be at $L_1$ distance at least $0.5$ from the centre and from each other, so that the well-specified prior differs from the uniform one and the misspecified prior from the well-specified one on every dataset. In the binary case these conditions read $\lvert\mu^\star_1-\tfrac12\rvert\ge\tfrac14$, $\lvert\mu^\dagger_1-\tfrac12\rvert\ge\tfrac14$ and $\lvert\mu^\star_1-\mu^\dagger_1\rvert\ge\tfrac14$, since the $L_1$ distance between two binary prevalence vectors is twice the difference of their first components.

## T.4 What a prior on the prevalence cannot do

Take BayCAP-KDEy, where $C$ is fixed at $\hat C$ with recalls $\hat d_j$. The point estimate of accuracy is close to $\sum_j\hat\pi_j\hat d_j$, and the realised accuracy of $U$ is $A=\sum_j\pi^Q_j\,c_j/n_j$, with $n_j$ the instances of class $\omega_j$ in $U$ and $c_j$ those that $h$ classifies correctly. Adding and subtracting $\sum_j\pi^Q_j\hat d_j$ and $\sum_j\pi^Q_jd_j$, with $d_j$ the recalls on the test distribution,

$$
\sum_j\hat\pi_j\hat d_j-A=\underbrace{\sum_j(\hat\pi_j-\pi^Q_j)\,\hat d_j}_{\text{prevalence term}}+\underbrace{\sum_j\pi^Q_j(\hat d_j-d_j)}_{\text{recall term}}+\underbrace{\sum_j\pi^Q_j\Big(d_j-\frac{c_j}{n_j}\Big)}_{\text{luck of the bag}},
$$

the error split of [[ext4_uncertainty_budget|Extension 4]] (its parts $e_1$, $e_2$, $e_3$). The prior on $\pi^Q$ changes only $\hat\pi$, so it acts only on the prevalence term. On the pairs where E3.1 found BayCAP-KDEy failing (`mammographic`, `semeion` with LR and MLP), the error is in the recall term, which is the same for every prior; on `wine-quality` it is in the prevalence term, but as a bias that every quantifier shares, and a prior centred on $G$ does not know the bias of the quantifier. BayCAP-Hf represents the recall term through its posterior on $C$, but the prior on $\pi^Q$ still changes only the prevalence part. So E5.1 is expected to show its gains, if any, on pairs that already cover.

## T.5 Tempering: the data counted $\tau$ times less

For a multinomial with counts $n_1,\dots,n_K$ and probabilities $p_1,\dots,p_K$, the likelihood is, apart from a constant coefficient, $\prod_ip_i^{n_i}$. Raising it to the power $1/\tau$,

$$
\Big(\prod_ip_i^{n_i}\Big)^{1/\tau}=\prod_ip_i^{n_i/\tau},
$$

the likelihood of the same proportions observed on $N/\tau$ instances. The kernel likelihood of BayCAP-KDEy is a product over the instances of the bag, so raising it to $1/\tau$ also counts each instance as $1/\tau$ of an instance.

**One recall, step by step.** With a uniform prior on a recall $d$ and $f$ correct predictions out of $n$ instances, the tempered posterior is $\operatorname{Beta}(f/\tau+1,(n-f)/\tau+1)$. Its mean is $(f/\tau+1)/(n/\tau+2)$, close to $f/n$ when $n/\tau$ is large: the centre stays in place. Its variance is $\bar d(1-\bar d)/(n/\tau+3)$, by the variance of the Beta distribution, so for $n/\tau$ large

$$
\operatorname{sd}(d\mid\text{data})\approx\sqrt{\frac{\tau\,\bar d(1-\bar d)}{n}}=\sqrt\tau\times\operatorname{sd}\text{ at }\tau=1 .
$$

**In general.** With many data the log-posterior is close to a quadratic function around its maximum (the normal approximation of the posterior), whose curvature is the Fisher information of the data, $I$; the posterior covariance is $I^{-1}$. Dividing the log-likelihood by $\tau$ divides the curvature by $\tau$, so the covariance becomes $\tau I^{-1}$ and every standard deviation grows by $\sqrt\tau$, while the maximum does not move. A value $\tau<1$ sharpens the posterior in the same way.

**The prior gains weight.** The prior is not tempered, so its imaginary instances are compared with $N/\tau$ real ones: the prior weight of T.2 grows from $c/(c+N)$ to $c/(c+N/\tau)$. With the uniform prior on $\pi^Q$ this pulls the posterior slightly towards the centre of the simplex; it is the reason for the candidate E5.c3.

## T.6 Tempering BayCAP-KDEy: how much the realised interval widens

**Two sources of spread.** Let $A$ be the realised accuracy drawn by the realised pushforward. Given a draw of $\pi$, the realised table is drawn around the population table, so $A$ has a conditional mean $\mu(\pi)=\sum_j\pi_j\hat d_j$ and a conditional variance $\sigma_g^2(\pi)$, the luck of the bag ([[ext3_posterior_propagation|Extension 3]] §T.8: it depends on $\pi$ only through the precisions of $h$, and is stored per bag as $\bar\sigma_g$). The law of total variance, $\operatorname{Var}(A)=\mathbb E[\operatorname{Var}(A\mid\pi)]+\operatorname{Var}(\mathbb E[A\mid\pi])$, gives

$$
\operatorname{Var}(A)=\bar\sigma_g^2+\sigma_p^2,
$$

where $\sigma_p^2$ is the posterior variance of $\mu(\pi)$, the spread of the population interval.

**Effect of $\tau$.** By T.5, tempering multiplies $\sigma_p^2$ by about $\tau$; $\bar\sigma_g^2$ depends on the observed predictions of $U$ and only weakly on $\pi$, so it stays about the same. The width of the interval is proportional to the standard deviation, so the widening with respect to $\tau=1$ is

$$
\frac{\text{width at }\tau}{\text{width at }1}\approx\sqrt{\frac{\tau\sigma_p^2+\bar\sigma_g^2}{\sigma_p^2+\bar\sigma_g^2}}=\sqrt{\frac{\tau+r^2}{1+r^2}},\qquad r=\frac{\bar\sigma_g}{\sigma_p},
$$

dividing numerator and denominator by $\sigma_p^2$. It lies between $1$ (the bag dominates, $r$ large) and $\sqrt\tau$ (the prevalence dominates, $r$ small).

**What the stored results predict.** E3.1 measured the widening of the realised over the population interval, $F=\sqrt{1+r^2}$, so $r^2=F^2-1$. With its medians for BayCAP-KDEy, $F=4.6$–$5.7$ at $K=2$ and $F=1.4$–$1.6$ at $3\le K\le7$ ([[ext3_posterior_propagation|Extension 3]], Analysis of E3.1, medians over the four realised methods), $\tau=4$ widens the interval by a factor $1.04$–$1.07$ at $K=2$ and $1.47$–$1.59$ at $3\le K\le7$ (derived from the formula, not measured). On binary datasets, where E3.1 found the failures from $V$, tempering BayCAP-KDEy is nearly inert. As a rough illustration, an interval displaced by $4.7$ of its standard deviations, as on `mammographic` in E3.1, needs about $2.4$ times its width to reach the target ($4.7/1.96$), which with the median $r^2$ of $K=2$ ($F=4.6$, $r^2=20.2$) requires $(\tau+20.2)/21.2\ge2.4^2$, i.e. $\tau\gtrsim100$.

## T.7 Tempering BayCAP-Hf jointly

Joint tempering divides the counts of $V$ and of $U$ by $\tau$ (T.5). The validation counts are what determine the rates of $h$, so the posterior standard deviation of each recall grows by about $\sqrt\tau$ where the data dominate the prior, and with it the part of the interval that represents the recall term of T.4, the error from $V$. This is the part that the fixed-$\hat C$ methods lack and that lets BayCAP-Hf cover more than BayCAP-KDEy (mean coverage $0.89$ against $0.74$ for vanilla accuracy in E3.1, with mean amplitude $0.126$ against $0.053$). The bag term $\bar\sigma_g^2$ is unchanged, as in T.6. Two side effects follow from the untempered prior: the hierarchical prior on the recalls gains weight with respect to the $n^s_j/\tau$ instances of each class, so the recalls are pooled more towards their common centre $m$; and the uniform prior on $\pi^Q$ gains weight as in T.5. Tempering moves no centre, so on pairs where the interval of BayCAP-Hf is displaced it helps only by widening; on pairs that over-cover, $\tau<1$ may narrow the interval without losing coverage.

## T.8 Choosing the temperature without the target labels

**Why the best $\tau$ in hindsight is optimistic.** For a pair $p$, the mean Winkler score $W_p(\tau)$ over its $100$ bags is the true score of the method plus a random error, because the bags are a random sample. Suppose tempering did nothing, so the five values $W_p(0.5),\dots,W_p(8)$ had the same true score and differed only by their errors. Their smallest value would still be below $W_p(1)$ in most pairs, since the minimum of five numbers is at most each of them, and below it with probability $4/5$ when the five errors are exchangeable. So the oracle-best $\tau$, chosen by looking at the scores of the pair, shows a gain even with no real effect. It is an upper bound, not evidence.

**Leave one dataset out.** For each dataset $d$ of the $15$:

1. take the results of the sweep on the other $14$ datasets ($70$ pairs);
2. for each $\tau$, average the per-pair mean Winkler scores over these $70$ pairs, and call $\tau_{-d}$ the value with the lowest average;
3. read the results of the five pairs of $d$ at $\tau_{-d}$.

No label of $d$ is used to choose $\tau_{-d}$, so on $d$ the selected method is evaluated as a practitioner would use it: with a temperature tuned on other problems, as a default hyperparameter is tuned on a benchmark. The whole dataset is held out, not one pair, because its five pairs share $V$ and its test pool; holding out one pair would let the other four pairs of the same dataset, with the same error from $V$, choose its $\tau$.

**The test.** For each dataset, let $\bar W_d(1)$ and $\bar W_d(\tau_{-d})$ be the means over its five pairs of the per-pair mean Winkler scores. The paired Wilcoxon signed-rank test ranks the $15$ differences $\bar W_d(1)-\bar W_d(\tau_{-d})$ by their absolute value and compares the sum of the ranks of the positive differences with what it would be if positive and negative signs were equally likely; a small $p$ means that the leave-one-dataset-out $\tau$ lowers the score more often and by more than chance would give. The datasets are the units because their errors from $V$ are independent, while those of the pairs of one dataset are not.

**Why validation bags cannot choose $\tau$ here.** Both methods use the rates of $h$ measured on $V$ (BayCAP-KDEy through $\hat C$, BayCAP-Hf through the counts of $V$). A bag drawn from $V$ has, on average over its draws, recalls equal to those of $V$, so the recall term of T.4 is zero on it: the error from $V$, which is what tempering should cover, does not occur on validation bags, and a selection on them would favour a $\tau$ that is too small. This is why the selection on validation bags (E5.c2) needs a part of $V$ that the method does not see.

# Implementation

Components (all [proposal]):

- **`temperature` argument of `methods/baycap.py:BayCAP`:** positive, default $1$; divides the log-likelihoods of the validation counts and of the predicted counts of the bag by $\tau$ (e.g. with `numpyro.handlers.scale(scale=1/τ)` around the two observed sites), for every `c_prior`; the priors are not scaled. With $\tau=1$ the model and the draws are unchanged.
- **E5.2 in `main`:** factory entries `BayesCAP-Hf-real-t<τ>` and `BayCAP-KDEy-mlp-real-t<τ>` for $\tau\in\{0.5,2,4,8\}$ in `experiments/methods.py`, excluded from the tables, ranks and default plots of `main`; the NUTS diagnostics of BayCAP-Hf stored as for `BayesCAP-Hf-real`; a table module (with the leave-one-dataset-out selection of §T.8 and the test of the decision rule) and a plot module for E5.2, following [[work_plan]] Part V.
- **E5.1 in `binform`:** options for the datasets (all those of `main`), the bag size ($\lvert U\rvert\in\{100,1000\}$, written in the output path) and the methods, with `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real` each with the three priors of E5.1 (uniform, well specified $\operatorname{Dir}(\lambda K\mu^\star)$, misspecified $\operatorname{Dir}(\lambda K\mu^\dagger)$); the bag-generating distribution $G=\operatorname{Dir}(\lambda K\mu^\star)$, with the two means drawn per dataset by the seeded rule of E5.1 and stored with the results; the default configuration (`connect-4`, its current methods and priors) is kept, so that its stored results stay reproducible; a table module and a plot module for E5.1.

Tests:

- `BayCAP` with `temperature=1` gives the same draws as without the argument, for every `c_prior` and the same `random_state`;
- with `c_prior="uniform"`, a large validation set and a large $\tau$, so that the bag term is weak, the posterior mean of each column of $C$ matches the conjugate value $(f_{\cdot j}/\tau+1)/(n^s_j/\tau+K)$ within Monte Carlo error;
- the two means of E5.1 are reproducible for a given seed, lie on the simplex, and satisfy the distance conditions; the concentrations are $3K\mu^\star$ and $3K\mu^\dagger$;
- the leave-one-dataset-out selection on a hand-built table of scores: $\tau_{-d}$ never depends on the scores of $d$;
- `binform` with its default arguments builds the same tasks and bags as before.

Reproducibility and comparability:

- E5.2 adds methods to `main` and changes none of the stored ones; its runs at $\tau=1$ are the stored `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real` only if the test above holds with the code that produced them.
- E5.1 uses bags drawn from its own $G$, not those of `main` nor those of `binform`, so its results are comparable across the priors and methods of E5.1 but not with the tables of `main`; its uniform prior is the reference within it. The stored `binform` results (`connect-4`, old `BayesCAP`) are unchanged.
- Both experiments write under `cap.env["OUT_DIR"]/uncawe`, in the directories of `main` and `binform`.

# Appendix

## A. Toy simulations

- `uv run python scripts/toys/ext5_prior_calibration.py` (§T.3; defaults: $200{,}000$ bags, seed $0$) is seeded and prints the table of §T.3.

## B. Relation to other extensions

- **[[ext2_baycap|Extension 2]]**: supplies BayCAP-Hf and the realised pushforward; the priors on $C$ (empirical-Bayes, hierarchical, Hf) are studied there, which is why the prior on $C$ of the earlier version of this extension is set aside.
- **[[ext3_posterior_propagation|Extension 3]]**: supplies BayCAP-KDEy with the fixed $\hat C$ and the decomposition $\operatorname{Var}(A)=\bar\sigma_g^2+\sigma_p^2$ used in §T.6; it keeps $\tau=1$, and tempering BayesKDEy belongs here (user's decision of 2026-10-27).
- **[[ext4_uncertainty_budget|Extension 4]]**: the error split used in §T.4.
- **History.** The extension was restructured on 2026-11-10 (Status, renumbering); the earlier text is in [[optional_extensions]] §12.

## C. Out of scope

- priors on $C$ other than those of [[ext2_baycap|Extension 2]], the comparison of tempering with the realised-table predictive, the full grid of priors on $\pi^Q$ (concentration and misspecification curves), and the label-free calibration of $\tau$ on held-out validation bags beyond the candidate E5.c2: set aside on 2026-11-10 ([[optional_extensions]] §12);
- the other Bayesian CAP methods (`BayesCAP`, `BayesCAP-real`, the EB and H priors, BayCAP-MAPLS, the methods of E3.2), not carried forward by Extensions 2 and 3;
- post-hoc interval recalibration ([[optional_extensions]] §1);
- violations of PPS.
