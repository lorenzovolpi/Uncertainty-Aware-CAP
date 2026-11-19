# Extension 7: Selecting the bandwidth of BayesKDEy

## Claim

BayesKDEy estimates, for each class $\omega_j$, the density of the outputs of its input classifier from the validation set $V$ with a Gaussian kernel of bandwidth $b$. The bandwidth decides how sharp these class densities are, and so how narrow the posterior of $\pi^Q$ is and where it is centred. In `main` it is fixed at $b=0.1$ for every dataset, class and validation size. This extension selects the bandwidth of each class from $V$ alone, by maximising the leave-one-out likelihood of the class density (a label-safe selection that uses no target bag), and measures whether the two methods built on the posterior of BayesKDEy, `BayCAP-KDEy-mlp-real` and `SQS-B`, give better intervals with the selected bandwidths than with $b=0.1$ (E7.1). It is kept deliberately tight (user's decision of 2026-11-19): one experiment, no candidates for now.

## Status

| Item | Status |
|---|---|
| BayesKDEy with one bandwidth for all classes (`methods/baycap_kdey.py:BayCAPKDEy`, `bandwidth=0.1`), used by `BayCAP-KDEy-mlp-real` and `SQS-B` in `main` | [implemented], run and stored in `main` (E3.1, E3.3 of [[ext3_posterior_propagation]]) |
| Quantifier classes `BayesKDEy`, `BayesMAPLS` used by `BayCAPKDEy`, `BayCAPMAPLS` and `SQSB` (no change of results) | [implemented] (card WP-25, done) |
| Per-class bandwidths selected by leave-one-out likelihood on $V$, with densities normalised on the simplex (§T.4): `bandwidth="loo"` of `methods/bayes_kdey.py:BayesKDEy` | [implemented], evaluated by the pilot of E7.1, not carried forward (card WP-25, done) |
| E7.1 (`BayCAP-KDEy-mlp-real-loo` and `SQS-B-loo` against the stored methods with $b=0.1$) | [implemented], pilot run and analysed on 2026-11-19 (`mammographic`, `wine-quality`, 10 pairs); closed after the pilot, without the full run (user's decision of 2026-11-19): both methods worse on every metric; card WP-25, done ([[work_plan]] §II.29) |
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

# Candidate experiments

None for now (user's decision of 2026-11-19).

# Theory

This part explains what E7.1 needs, step by step. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$ and $T=C\operatorname{diag}(\pi)$ the population table, with predicted classes on rows and true classes on columns. The bandwidth enters only the posterior of $\pi^Q$; the bridge from $\pi$ to the table (the fixed $\hat C$ and the realised pushforward for BayCAP-KDEy, the forward bag for SQS-B) is that of [[ext3_posterior_propagation]] and is unchanged.

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

# Implementation

Components:

- **Quantifier classes** (user's decision of 2026-11-19): BayesKDEy and BayesianMAPLS become quantifier classes of their own (`BayesKDEy`, `BayesMAPLS`, on a base `BayesianQuantifier`), used by `BayCAPKDEy`, `BayCAPMAPLS` and `SQSB`, which no longer borrows a BayCAP method; the restructure changes no result (checked by a golden test).
- **Per-class bandwidths:** the bandwidth is a hyperparameter of `BayesKDEy`, a positive number, `"scott"`, `"silverman"` or `"loo"`, passed through by `BayCAPKDEy` and `SQSB`. `"loo"` needs the Gaussian kernel and no shrinkage (and `c_prior="fixed"` in `BayCAPKDEy`), with a grid `bandwidth_grid` (default `LOO_GRID`, $\{10^{-2+0.1k}\}_{k=0}^{20}$). With `"loo"`, `_SelectedKernelDensities` selects $\hat b_j$ per class by the closed-form leave-one-out log-likelihood of §T.2 in $d=K-1$ dimensions (squared distances computed in chunks to bound memory on large classes), fits one `KernelDensity` per class with its $\hat b_j$, and returns log-densities corrected by $+\tfrac12\ln(2\pi\hat b_j^2)$ (§T.4); a class with $n_j<2$ keeps $0.1$. The selected bandwidths are stored (`bandwidths_`) and written per bag. Numeric bandwidths and `"scott"`/`"silverman"` keep QuaPy's densities unchanged.
- **E7.1 in `main`:** two factory entries in `experiments/methods.py:LOO_VARIANTS`, `BayCAP-KDEy-mlp-real-loo` and `SQS-B-loo` (with the configuration and seed of `BayCAP-KDEy-mlp-real`, so the same selection and draws), excluded from the tables, ranks and default plots of `main`; per bag, the extra columns of the selection (`bandwidths`, `class_sizes`, `bandwidth_edge`, `loo_selected`, `loo_default`); the tables `tables/kdey_loo.py` and the plots `plots/kdey_loo.py`.

Tests:

- the closed-form leave-one-out log-likelihood equals the brute-force one (one `KernelDensity` per left-out point) on a small random set, with the $(K-1)$-dimensional normalisation;
- on simplex data with $K=2$ and $K=3$, the corrected class density integrates to $1$ over the simplex surface up to the mass outside the simplex (checked numerically on data far from the edges);
- the selection returns a known optimum on a toy set, and $0.01$ when the outputs of a class are all duplicated;
- with a numeric `bandwidth`, the log-densities and the posterior draws are identical to those of the current code (stored results stay comparable);
- `SQS-B-loo` and `BayCAP-KDEy-mlp-real-loo` share their prevalence draws for the same seeds, as `SQS-B` and `BayCAP-KDEy-mlp-real`.

Reproducibility and comparability:

- The selection is deterministic given the cross-fitted outputs, which are seeded as in E3.1; the seeds of NUTS and of the bags are those of the stored methods, so the LOO variants differ from them only through the bandwidths.
- The stored results of `main` are unchanged: the new methods add rows to `main`, and the shared-bandwidth path is not modified.

# Appendix

## A. Toy simulations

None.

## B. Relation to other extensions

- **[[ext3_posterior_propagation|Extension 3]]**: E7.1 changes only the bandwidth of the quantifier of `BayCAP-KDEy-mlp-real` (E3.1) and `SQS-B` (E3.3); the bridge from $\pi$ to the table is unchanged.
- **[[ext5_baycap_priors_tempering|Extension 5]]**: the prior on $\pi^Q$ and the temperature $\tau$ are held at $\operatorname{Dir}(\mathbf 1)$ and $1$, the values kept by Extension 5.
- **[[ext4_uncertainty_budget|Extension 4]]**: the error coming from the rates of $h$ on $V$ does not depend on $b$ (§T.6).
- **History.** Extension 7 was a factorial grid-search study of $h$, the surrogate classifier and the quantifier; it was narrowed to the bandwidth of BayesKDEy on 2026-11-19 (user's decision), and the earlier scope is set aside in [[optional_extensions]] §13.

## C. Out of scope

- the factorial optimisation of $h$, of surrogate classifiers and of other quantifiers ([[optional_extensions]] §13);
- quantification-oriented selection of the bandwidth (by the error on bags drawn from $V$, as in the BayesKDEy paper), the Aitchison and ILR kernels and shrinkage;
- the input classifier $h$ for BayesKDEy: only the MLP input of `main` is studied.
