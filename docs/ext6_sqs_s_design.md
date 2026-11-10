# Extension 6: How SQS-S uses the validation set

## Claim

SQS-S estimates the accuracy of $h$ on a target bag $U$ by finding, among many labelled bags drawn from the validation set, those whose estimated prevalence is close to the estimated prevalence of $U$, and by reading off their accuracies. Two design choices decide how well this works, and the paper fixes both without studying them: the distance on the simplex that defines "close", and the split of the validation set $V$ between the data that train the quantifier and the data from which the bags are drawn. This extension measures, on the corrected SQS-S of the main protocol, whether a different distance selects better neighbourhoods (E6.1) and how the estimate and the interval change when the split moves away from the paper's $50\%/50\%$ (E6.2). It is kept deliberately thin (user's decision of 2026-11-09): one experiment per question, and everything else is a candidate or set aside.

## Status

| Item | Status |
|---|---|
| SQS-S (`PQ`, `cap-ml` `PrediQuant`) as in the paper: quantifier trained on $V_1$ only, bags and target estimated by the same surrogate pipeline | [implemented], fixed by FIX-1 + FIX-4 (`cap-ml` `e9e77cd`, [[work_plan]] §I.2), stored in `main`, re-run on 2026-11-09 |
| `predist`, the distance diagnostic ($L_1$, Hellinger, Jensen–Shannon; LR, KDEy(MLP), $8$ three-class datasets, $5$ test bags) | [implemented] ([[work_plan]] §II.3); the results computed before FIX-1 are kept in `<OUT_DIR>/uncawe/predist_pre_fix1/`; re-run with the fixed `PrediQuant` on 2026-11-09 (LR, KDEy(MLP), completed), superseded by E6.1, which uses PACC(MLP) |
| E6.1 (distances, PACC(MLP), five classifiers, summary tables) | [implemented], run and analysed on 2026-11-10 (card WP-16, done, [[work_plan]] §II.22): $L_1$ kept |
| Allocation $\rho$ as a parameter of `PrediQuant`, nested splits | [implemented], not evaluated (card WP-15, `cap-ml` `0f25e9e`, [[work_plan]] §II.21); `PQ` keeps `rho=None` |
| E6.2 (allocation $\rho$ on the `main` protocol) | [proposal], scheduled as card WP-17, after WP-15 |
| Candidates E6.c1 (cross-fitting), E6.c2 (sensitivity to $\gamma$), E6.c3 (kernel weights) | [proposal], not scheduled |
| Adaptive bandwidth, Aitchison geometry, factorial $\rho\times\lvert V\rvert\times\lvert U\rvert$, Neyman allocation across classes, label-free selection of $\rho$ | set aside on 2026-11-09, in [[optional_extensions]] §11 |

# Experiments

**Common settings.** SQS-S as in `main` after FIX-1 (§T.1): bags of size $\lvert U\rvert=1000$; $n=10{,}000$ candidate bags drawn with replacement from $V_b$ at prevalences drawn uniformly on the simplex; the $m=1000$ candidates closest to the target (a fraction $\gamma=0.1$) are selected; the point estimate is the mean of their accuracies and the interval the $[2.5\%,97.5\%]$ percentile interval. Every score is computed against the realised table of the target bag (`true_cts`).

**Order.** E6.1 first (it uses a short, cheap diagnostic), then E6.2.

## E6.1: Distances on the simplex

**Why.** SQS-S compares estimated prevalences with the $L_1$ distance. Hellinger and Jensen–Shannon weigh a difference in a rare class more than the same difference in a common class (§T.2), so near the edges of the simplex, where some classes are rare, they select different neighbourhoods. The stored `predist` results cannot answer the question: they were computed before FIX-1, when the bags and the target were estimated by two different estimators, and the choice of distance then only changed how the wrong space was measured (e.g. `connect-4`, LR, test bag 1: coverage $0$ for every distance).

- **Design.**
  - **Inputs:** `predist` with the fixed `PrediQuant`: the $8$ three-class datasets of `THREE_CLASS_DATASETS`, the five classifiers of `main` ($40$ pairs), $5$ test bags of size $1000$ per pair at fixed points of the simplex (the centre, the three vertices at $1-0.01$, and the random point farthest from them), drawn from the test pool with the seeds of `predist`; one fit of `PrediQuant` per pair, evaluated with the three distances on the same test bags. Quantifier: PACC(MLP), the quantifier of `PQ` in `main`, so that the conclusion applies directly to `PQ` (user's decision of 2026-11-10; the stored `predist` results use KDEy(MLP)).
  - **Varied:** the distance ($L_1$, Hellinger, Jensen–Shannon); held fixed: the fit, the candidate bags, $\gamma$, the test bags.
  - **Measures:** per test bag and distance, for vanilla accuracy: the absolute error of the point estimate, coverage, amplitude and Winkler score of the interval; the mean $L_1$ distance between the *true* prevalences of the selected bags and the true prevalence of the test bag (how well the selection works in true-prevalence space, §T.2); per distance, the means over the $200$ test bags, and per pair the means over its $5$ test bags.
  - **Estimand:** the realised table of each test bag, for every distance.
  - **Cost:** one `predist` run over the $40$ pairs, parallel over pairs.
- **Decision rule** (user's decision of 2026-11-09, "keep $L_1$ unless another distance is clearly better"; operationalised here): $L_1$ is kept unless another distance has a lower mean Winkler score over the $200$ test bags, a lower per-pair mean Winkler score on a majority of the $40$ pairs, and a paired Wilcoxon signed-rank test over the $40$ per-pair means with $p<0.05$.
- **Representation.**
  - **Summary table:** one row per distance, with the means over the $200$ test bags of the absolute error, coverage, amplitude, Winkler score and true-space distance of the selected bags, the number of pairs where the distance has the best Winkler score, and the $p$-value of the paired test against $L_1$.
  - **Per-region table:** the same means split by the position of the test bag (centre, vertices, random point), since the distances differ most near the vertices (§T.2).
  - **Ternary plots:** the existing per-bag plots of `predist` (selected bags in predicted and true prevalence space, one panel per distance).
- **Analysis.** Run on 2026-11-10 on `bcuda` (`predist` at commit `6cc1aeb`, `cap-ml` `637fdd5`; $40$ pairs, $200$ test bags, no failed pair; no pilot, at the user's request); tables at commit `54742e9`, under `<OUT_DIR>/uncawe/tables/predist/` (`predist_summary.csv`, `predist_region.csv`, `predist_pairs.csv`, `predist_pair_region.csv`, and the LaTeX documents); ternary plots under `<OUT_DIR>/uncawe/plots/predist/`. Every score is against the realised table of the test bag; the numbers below are means over the test bags unless stated otherwise.
  - **Outcome of the decision rule: $L_1$ is kept.** Hellinger and Jensen–Shannon have a lower mean Winkler score than $L_1$ ($0.820$ and $0.824$ against $0.861$), but their per-pair mean Winkler score is lower than that of $L_1$ on only $17$ and $16$ of the $40$ pairs, and the paired Wilcoxon test gives $p=0.63$ and $p=0.64$; two of the three conditions fail. $L_1$ has the lowest per-pair mean Winkler score on $23$ pairs, Hellinger on $15$, Jensen–Shannon on $2$. The absolute error is the same for the three distances ($0.0596$–$0.0598$), and so, within $0.015$, are coverage ($0.540$–$0.555$) and amplitude ($0.133$–$0.135$).

    | Distance | AE | Coverage | Amplitude | Winkler | True-space $L_1$ | Best pairs | Pairs $<L_1$ | $p$ vs $L_1$ |
    |---|---|---|---|---|---|---|---|---|
    | $L_1$ | 0.0596 | 0.540 | 0.1326 | 0.8612 | 0.3767 | 23 | – | – |
    | Hellinger | 0.0598 | 0.555 | 0.1354 | 0.8201 | 0.3829 | 15 | 17 | 0.627 |
    | Jensen–Shannon | 0.0597 | 0.550 | 0.1342 | 0.8244 | 0.3798 | 2 | 16 | 0.637 |

  - **Where the lower mean Winkler score of Hellinger comes from.** Split by region (`predist_region.csv`), the mean Winkler score of $L_1$ minus that of Hellinger is $0.004$ at the centre ($40$ bags), $0.011$ at the vertices ($120$ bags) and $0.168$ at the random point ($40$ bags). Multiplying by the number of bags, the random point contributes $0.168\times40=6.7$ of the total difference $0.041\times200=8.2$, about $80\%$. At the random point, $6$ bags (one per pair) contribute $6.1$ of these $6.7$: the four `page_block` pairs with LR, kNN, RF and SVM and `phishing` with kNN and MLP (`predist_pair_region.csv`); on the other $34$ pairs the difference at the random point is small and of either sign (over the $40$ pairs, $19$ positive and $20$ negative, median $-0.001$). These are bags where the quantifier misses the test prevalence. In the plot of `page_block`, SVM, bag 4, the estimated prevalence of the test bag is far from the true one; $L_1$ selects a compact set of bags around the estimate, Hellinger and Jensen–Shannon a set stretched along the edge of the simplex (the rare class weighs more, §T.2), which reaches closer to the true prevalence; the interval is wider (amplitude $0.270$ against $0.192$) and still misses the target, and the Winkler score drops from $4.51$ to $2.75$ only because the miss is smaller. The gain of the two distances is therefore a wider neighbourhood where the quantifier fails, not a better selection (a hypothesis drawn from the plots and the per-pair values, not tested).
  - **The selected bags are far from the test bag, for every distance.** The mean $L_1$ distance between the true prevalences of the selected bags and that of the test bag is $0.38$ overall, $0.24$ at the centre and $0.44$ at the vertices, almost the same for the three distances; the absolute error is $0.036$ at the centre and $0.070$ at the vertices, and the coverage $0.78$ and $0.44$ (`predist_region.csv`, $L_1$). At the vertices this is not caused by the quantifier: the $L_1$ error of the estimated prevalence of the test bag is at most $0.045$ on $5$ of the $8$ datasets (`phishing`, `mhr`, `molecular`, `connect-4`, `waveform-v1`), $0.096$ on `academic-success`, and large only on `page_block` ($0.19$) and `cmc` ($0.50$) (`predist_pair_region.csv`, means over classifiers and vertices). Yet the selected bags are on average $0.33$ away from that estimate in estimated-prevalence space: the neighbourhood itself is large. In the plot of `phishing`, LR, bag 1, the estimate is almost exact, and the $1000$ selected bags fill the corner of the simplex with $p_0$ down to about $0.6$; the interval misses (coverage $0$, absolute error $0.082$).
  - **A hypothesis for the size of the neighbourhood at the vertices.** SQS-S keeps a fixed fraction $\gamma=0.1$ of candidates drawn uniformly on the simplex, so the neighbourhood of a target at a vertex must cover $10\%$ of the simplex. For prevalences uniform on the simplex of three classes, the set $\{p_k\ge1-t\}$ is a triangle similar to the simplex with sides scaled by $t$, so its area, and its probability, is $t^2$ of the whole. The $L_1$ distance from the vertex $e_k$ is $\sum_j\lvert p_j-(e_k)_j\rvert=(1-p_k)+\sum_{j\ne k}p_j=2(1-p_k)$. The fraction $\gamma$ closest to the vertex is therefore $\{p_k\ge1-\sqrt\gamma\}$, which for $\gamma=0.1$ reaches $p_k=1-0.316=0.684$, an $L_1$ distance of $0.63$. Inside it, $s=1-p_k$ has density $2s/t^2$ on $[0,t]$ (its distribution function is $s^2/t^2$), so its mean is $\int_0^t s\cdot2s/t^2\,ds=2t/3=0.211$, and the mean $L_1$ distance is $2\times0.211=0.42$, close to the observed $0.44$ in true-prevalence space. The neighbourhood also lies entirely on one side of a target at a vertex, so when the accuracy of $h$ changes with the prevalence, the mean accuracy of the selected bags is pulled towards the inside of the simplex and the interval moves away from the target. This would explain most of the under-coverage at the vertices, and it depends on $\gamma$ and on the candidate design, not on the distance: it bears on E6.c2 rather than on E6.1.
  - **A quantifier that compresses the simplex.** On `cmc` the quantifier misses the vertices (error $0.50$). In the plot of `cmc`, LR, bag 1, the estimated prevalences of the selected bags are packed near the vertex while their true prevalences spread over a large region (true-space distance $0.58$ at the vertices of `cmc`, against $0.08$ in estimated space): this is the second condition of §T.2 failing, the quantifier maps distant prevalences to nearby estimates. The distance cannot repair it, since all three distances work on the estimates.
  - **Conclusion.** The choice among the three distances barely matters on the corrected SQS-S with PACC(MLP); $L_1$ is kept, as in the paper and in `PQ`. The quality of the selection is limited by the size of the neighbourhood, set by $\gamma$, at the edges of the simplex, and by the quantifier where it compresses or misses the target prevalence; both are outside the scope of E6.1.

## E6.2: Allocation of the validation set

**Why.** SQS-S splits $V$ into $V_q$, which trains the quantifier, and $V_b$, from which the bags are drawn, with $\lvert V_q\rvert=\rho\lvert V\rvert$ and the paper's $\rho=0.5$. A larger $\rho$ gives a better quantifier, so the selected bags are closer to the target in true prevalence; a smaller $\rho$ gives a larger $V_b$, so the recalls of $h$ inside the bags are closer to its true recalls. The first error shrinks and the second grows as $\rho$ increases (§T.3); the second is shared by all the bags and therefore does not widen the interval, so it is a source of under-coverage. Where the balance lies is not known, and it may differ from $0.5$.

- **Design.**
  - **Inputs:** the `main` protocol: the $75$ pairs (datasets × the five classifiers), $100$ APP target bags of size $1000$ per pair with the seeds of `main`; SQS-S with PACC(MLP) as in `PQ`, $n=10{,}000$ candidates, $\gamma=0.1$, $L_1$ distance, $m=1000$.
  - **Varied:** $\rho\in\{0.1,0.3,0.5,0.7,0.9\}$, with nested splits (§T.4): for each class, one seeded random order of its validation instances, and $V_q(\rho)$ the first fraction $\rho$ of each class, so that $V_q(0.1)\subset V_q(0.3)\subset\dots\subset V_q(0.9)$. Held fixed: $\lvert V\rvert$, $\lvert U\rvert$, the target bags, $n$, $\gamma$ and $m$. No outer seeds (user's decision of 2026-11-09, §T.4): the $75$ pairs are the replicates.
  - **Measures:** per target bag and level: the quantification error on $U$, $\operatorname{AE}(\hat\pi^Q,\pi^Q)=\frac1K\sum_j\lvert\hat\pi^Q_j-\pi^Q_j\rvert$, with $\pi^Q$ the true prevalence of the bag; the absolute error of the point estimate, coverage, amplitude and Winkler score, for vanilla accuracy, macro-F1 and macro-K; the class support of the selected bags, for each class $\omega_j$ the number of distinct instances of $\omega_j$ in $V_b$ that appear in at least one of the $m$ selected bags divided by the number of instances of $\omega_j$ in $V_b$ (§T.3), summarised by its minimum over the classes. Per pair and level, the means over the $100$ bags.
  - **Estimand:** the realised table of each target bag, for every level.
  - **Cost:** $5$ runs of SQS-S on the $75$ pairs (the re-run of `PQ` on 2026-11-09 took $47$ minutes on `bcuda`, so about $4$ hours in all); the level $0.5$ is run too, since the nested split differs from the split of `PQ` (§T.4). A pilot on $3$ datasets (e.g. `mammographic`, `wine-quality` and a large dataset) with the five classifiers first.
- **Scope.** E6.2 is an analysis only (user's decision of 2026-11-10): it describes how the measures change with $\rho$ and has no decision rule; the allocation of SQS-S in `PQ` and in later experiments stays $\rho=0.5$.
- **Representation.**
  - **Table:** for each metric, one row per measure and one column per level, with the mean over the $75$ pairs; the cells that differ from $\rho=0.5$ with a Holm-adjusted $p<0.05$ are marked.
  - **Line plot:** each measure (quantification error, absolute error, coverage, amplitude, Winkler score, class support) against $\rho$, the mean over the pairs in bold and each pair as a faint line, one panel per measure, vanilla accuracy; the same for macro-F1 and macro-K in separate figures.
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussions; they are not scheduled. One becomes an experiment of the main section only when the user decides so.

- **E6.c1, cross-fitting.** *Why:* the trade-off of E6.2 exists because each validation instance serves either the quantifier or the bags; swapping the roles of $V_q$ and $V_b$ and pooling the candidate bags of the two halves uses every instance for both, which could remove most of the trade-off. *Needs:* a new variant of `PrediQuant` (two quantifiers, each estimating the bags drawn from the half it did not see, and two estimates of the target), run on the `main` protocol at $\rho=0.5$.
- **E6.c2, sensitivity to $\gamma$.** *Why:* after FIX-1, SQS-S has wider intervals than SQS-H (`RQBS`) in the tables of `main` generated on 2026-11-09; $\gamma$, the fraction of candidates kept, sets the size of the neighbourhood and so, directly, the spread of the selected prevalences and the width of the interval. E6.1 adds a second reason: at the vertices the selected bags are on average $0.33$ away from the estimated prevalence of the test bag, and $0.44$ from its true prevalence, even where the quantifier is accurate, and the coverage there is $0.44$; with a fixed fraction $\gamma$ of candidates drawn uniformly on the simplex, the neighbourhood of a target at a vertex is large and lies on one side of it (Analysis of E6.1), so a smaller $\gamma$ might reduce this bias at the edges of the simplex at the cost of fewer bags. *Needs:* almost nothing: as `predist` does for the distances, one fit per pair re-scored at several $\gamma$ with the number of candidates fixed.
- **E6.c3, kernel weights.** *Why:* the hard selection gives every one of the $m$ closest bags the same weight and every other bag none; weighting each bag by a decreasing function of its distance (a kernel) would turn SQS-S into a local average on the simplex, and is the only item of the former geometric study that would yield a new rule rather than a comparison. *Needs:* a new method (weighted mean and weighted percentiles of the bag accuracies) and a choice of bandwidth.

# Theory

This part explains what the experiments need, step by step. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$, $d_j=C_{jj}$ its recall on class $\omega_j$ (its accuracy within that class), and $T=C\operatorname{diag}(\pi)$ the population table, with predicted classes on rows and true classes on columns.

## T.1 SQS-S, step by step, and what it estimates

The labelled validation set $V$ is split, class by class, into two parts $V_q$ and $V_b$, with $\lvert V_q\rvert=\rho\lvert V\rvert$ and $\lvert V_b\rvert=(1-\rho)\lvert V\rvert$; the paper and `PQ` use $\rho=0.5$.

1. A quantifier $q$ (in `main`, an MLP followed by the PACC adjustment) is trained on $V_q$ only.
2. From $V_b$, $n$ candidate bags $\sigma_1,\dots,\sigma_n$ of size $\lvert U\rvert$ are drawn with replacement, each at a prevalence drawn uniformly on the simplex. Since their labels are known, each has a known accuracy $A(\sigma_b)$, computed from the predictions of $h$ on the bag, and the quantifier gives its estimated prevalence $\hat\pi_b=q(\sigma_b)$.
3. For the unlabelled target bag $U$, the same quantifier gives $\hat\pi^Q=q(U)$.
4. The $m=\gamma n$ candidates with the smallest distance $\operatorname{dist}(\hat\pi_b,\hat\pi^Q)$ are selected; the estimate is the mean of their accuracies and the interval the percentile interval of their accuracies.

The target is the accuracy of $h$ on the bag $U$ itself, the realised accuracy $A(h,U)=\sum_j\pi^Q_j\,c_j/n_j$, where $n_j$ is the number of instances of class $\omega_j$ in $U$, $c_j$ the number of them that $h$ classifies correctly, and $\pi^Q_j=n_j/\lvert U\rvert$. Each candidate bag is itself a bag of size $\lvert U\rvert$, so its accuracy is a realised accuracy of the same kind: SQS-S targets the realised table of the target bag, not the population table $C\operatorname{diag}(\pi^Q)$.

## T.2 Why matching works, and where the distance matters

**Matching needs one estimator.** Write $f(\pi)$ for the average estimate $q$ gives to a bag of true prevalence $\pi$ drawn from the validation distribution. Under prior probability shift the instances of each class have the same distribution in $V$ and in the target, so a target bag with true prevalence $\pi$ and a candidate bag with the same $\pi$ receive, on average, the same estimate $f(\pi)$, even if $q$ is biased ($f(\pi)\neq\pi$). Selecting candidates with $\hat\pi_b$ close to $\hat\pi^Q$ therefore selects candidates whose true prevalence is close to that of the target, provided two conditions hold: the bags and the target go through the *same* estimator (otherwise the two sides have different functions $f$ and the bias no longer cancels; this was the second problem fixed by FIX-1), and $f$ separates different prevalences (if $f$ maps two distant prevalences to nearby estimates, matching cannot tell them apart). The measure "true-space distance of the selected bags" of E6.1 checks the outcome of both conditions directly.

**What the distance changes.** For two prevalence vectors $p$ and $p'$:

$$
L_1(p,p')=\sum_j\lvert p_j-p'_j\rvert,\qquad
H(p,p')=\sqrt{\tfrac12\sum_j\big(\sqrt{p_j}-\sqrt{p'_j}\big)^2}.
$$

To see how $H$ treats a small difference $\delta_j=p'_j-p_j$, expand the square root around $p_j$: for small $\delta_j$, $\sqrt{p_j+\delta_j}\approx\sqrt{p_j}+\delta_j/(2\sqrt{p_j})$ (first-order Taylor expansion, valid when $\lvert\delta_j\rvert$ is small compared with $p_j$). Then

$$
\big(\sqrt{p'_j}-\sqrt{p_j}\big)^2\approx\frac{\delta_j^2}{4p_j},
\qquad\text{so}\qquad
H(p,p')^2\approx\frac18\sum_j\frac{\delta_j^2}{p_j}.
$$

Each squared difference is divided by $p_j$: the same absolute difference costs more in a rare class than in a common one. For small differences the Jensen–Shannon divergence has the same form, $\tfrac18\sum_j\delta_j^2/p_j$ (from the same expansion applied to its logarithms), so it ranks nearby candidates like Hellinger. $L_1$ weighs all classes equally. Near the centre of the simplex all $p_j$ are similar, the weights are similar, and the three distances rank the candidates almost in the same order; near a vertex, where some $p_j$ are close to $0$, Hellinger and Jensen–Shannon insist on matching the rare classes in relative terms and tolerate larger absolute differences in the dominant class. This is why E6.1 reports the results by region of the simplex.

## T.3 The allocation trade-off

To see how $\rho$ acts, split the error of the SQS-S point estimate into three parts, as in the error split of [[ext4_uncertainty_budget|Extension 4]].

**Setting.** Let $\hat d^{(b)}_j$ be the recall of $h$ on the instances of class $\omega_j$ in $V_b$, and let $n_{b,j}$ be their number. A candidate bag of prevalence $\pi$ drawn from $V_b$ has, on average over its draws, accuracy $\sum_j\pi_j\hat d^{(b)}_j$, because each of its class-$\omega_j$ instances is a random draw from the class-$\omega_j$ instances of $V_b$, which $h$ classifies correctly with frequency $\hat d^{(b)}_j$ (linearity of the expectation). The selected bags have true prevalences spread around some $\pi^Q+\varepsilon$, where $\varepsilon$ is the error that the quantifier makes, translated into true-prevalence space (T.2). So the point estimate is, up to the spread of the bags,

$$
\hat A\approx\sum_j(\pi^Q_j+\varepsilon_j)\,\hat d^{(b)}_j .
$$

**The split.** Subtract the realised accuracy $A(h,U)=\sum_j\pi^Q_jc_j/n_j$ and add and subtract $\sum_j\pi^Q_j\hat d^{(b)}_j$ and $\sum_j\pi^Q_jd_j$:

$$
\hat A-A(h,U)\approx\underbrace{\sum_j\varepsilon_j\hat d^{(b)}_j}_{\text{prevalence part}}
+\underbrace{\sum_j\pi^Q_j\big(\hat d^{(b)}_j-d_j\big)}_{\text{recall part}}
+\underbrace{\sum_j\pi^Q_j\big(d_j-c_j/n_j\big)}_{\text{luck of the bag}} .
$$

**Recall part, and its dependence on $\rho$.** $\hat d^{(b)}_j$ is the mean of $n_{b,j}$ indicators "instance correctly classified", each equal to $1$ with probability $d_j$. The variance of one indicator is $d_j(1-d_j)$, and the variance of the mean of $n$ independent variables with variance $\sigma^2$ is $\sigma^2/n$. With $n_{b,j}=(1-\rho)\lvert V_j\rvert$, where $\lvert V_j\rvert$ is the number of class-$\omega_j$ instances in $V$,

$$
\operatorname{Var}\big(\hat d^{(b)}_j\big)=\frac{d_j(1-d_j)}{(1-\rho)\lvert V_j\rvert}.
$$

The recalls of different classes are computed on different instances, so they are independent, and the variance of a sum of independent terms is the sum of their variances; with $\operatorname{Var}(aX)=a^2\operatorname{Var}(X)$,

$$
\operatorname{Var}(\text{recall part})=\sum_j(\pi^Q_j)^2\,\frac{d_j(1-d_j)}{(1-\rho)\lvert V_j\rvert},
$$

which grows as $\rho$ grows: the factor $1/(1-\rho)$ is $1.1$ at $\rho=0.1$, $2$ at $\rho=0.5$ and $10$ at $\rho=0.9$. Crucially, every candidate bag is drawn from the same $V_b$, so all of them share the same $\hat d^{(b)}_j$: the recall part moves the whole set of selected accuracies, and therefore the whole interval, by the same amount. It is not reflected in the width of the interval, so when it is large the interval misses the target: this is a source of under-coverage that grows with $\rho$.

**Prevalence part, and its dependence on $\rho$.** The quantifier is trained on $V_q$, of size $\rho\lvert V\rvert$. With PACC, the adjustment is estimated from averages over the instances of each class in $V_q$, and by the same rule $\operatorname{Var}=\sigma^2/n$ the error of these averages shrinks as $\rho\lvert V\rvert$ grows; the MLP itself also improves with more training data. So $\varepsilon$, and with it the prevalence part, tends to shrink as $\rho$ grows. There is no closed form for its size, which depends on the dataset, on $h$ and on the MLP.

**Luck of the bag.** It does not depend on $\rho$: it is the difference between the expected and the realised accuracy of the target bag itself. The candidate bags reproduce it in their spread: a candidate bag of size $\lvert U\rvert$ drawn with replacement has, around its average, the same kind of variation, so this part is reflected in the width of the interval.

**The trade-off.** As $\rho$ increases, the prevalence part tends to decrease and the recall part increases, by a factor $1/(1-\rho)$ in variance. The best $\rho$ is where the two changes balance, and it depends on $\lvert V\rvert$, on how hard the quantification is and on the recalls of $h$; E6.2 measures it. The quantification error on $U$ of E6.2 tracks the first part, and the coverage tracks mostly the second.

**Class support.** When $V_b$ has few instances of a class, the candidate bags re-use the same instances many times. If a bag needs $k$ instances of class $\omega_j$ and draws them with replacement from $n_{b,j}$ instances, a given instance is missed by one draw with probability $1-1/n_{b,j}$, and by all $k$ draws, which are independent, with probability $(1-1/n_{b,j})^k$. By linearity of the expectation, summing over the $n_{b,j}$ instances, the expected number of distinct instances in the bag is

$$
n_{b,j}\Big(1-\big(1-1/n_{b,j}\big)^k\Big),
$$

which is close to $k$ when $n_{b,j}\gg k$ and close to $n_{b,j}$ when $n_{b,j}\ll k$. With $k=\pi_j\lvert U\rvert$ up to $1000$ and $n_{b,j}$ possibly in the tens on small datasets, the bags of SQS-S can be made of a few instances repeated many times; their accuracies are then always computed on the same instances, which is what makes the recall part shared. The class support of E6.2 measures how much of $V_b$ the selected bags actually use.

## T.4 Nested splits and why the pairs are the replicates

E6.2 compares levels of $\rho$ on the same pair. For a score $X$ (e.g. the mean Winkler score of a pair) at two levels $a$ and $b$, the difference $\Delta=X_a-X_b$ has variance

$$
\operatorname{Var}(\Delta)=\operatorname{Var}(X_a)+\operatorname{Var}(X_b)-2\operatorname{Cov}(X_a,X_b),
$$

by the rule for the variance of a difference. Part of each $X$ is noise from which instances happened to fall into $V_q$. If the two levels used independent splits, this noise would be independent between them, the covariance would be $0$, and the difference would carry the noise of both. With nested splits, $V_q(a)\subset V_q(b)$ for $a<b$, so the two levels share most of their instances, the noise is largely common, the covariance is large and positive, and it cancels in $\Delta$: the difference reflects the effect of $\rho$ rather than the luck of the split. The price is that the nested split at $\rho=0.5$ is a different random split from the one of `PQ`, so the stored `PQ` results cannot serve as the level $0.5$.

The question of E6.2 is about the average effect over the pairs, $\bar\Delta=\frac1{75}\sum_p\Delta_p$. The $75$ pairs use $75$ independent splits, so their split noise is independent, and the variance of a mean of $75$ independent terms is the variance of one term divided by $75$. Repeating each pair with several outer seeds would only be needed for claims about a single pair (e.g. "the best $\rho$ on `mammographic` is $0.3$"), which E6.2 does not make.

# Implementation

Components:

- **Allocation in `cap-ml`:** `PrediQuant` gets a parameter `rho` (default `None`). With `rho=None`, the split is the current stratified `train_test_split` at $50\%/50\%$, so `PQ` and its stored results are unchanged (user's decision of 2026-11-09, option (a)). With `rho` in $(0,1)$, the nested split of E6.2: for each class, a permutation of its validation indices seeded by `random_state`, and $V_q$ the first $\operatorname{round}(\rho\,\lvert V_j\rvert)$ of each class. `PrediQuant` also keeps, after each prediction, the estimate $\hat\pi^Q$ and the indices of the selected bags, so that the quantification error and the class support can be stored. Followed by a pin bump in UNCAWE.
- **E6.2 in `main`:** five factory entries `PQ-rho<ρ>` in `experiments/methods.py`, excluded from the tables, ranks and default plots of `main`; per bag, the extra columns $\hat\pi^Q$ and the class support of the selected bags; a table module and a plot module for E6.2, following [[work_plan]] Part V.
- **E6.1 in `predist`:** `--classifiers` accepting several classifiers, with all (dataset, classifier) pairs in one parallel pool, as `main` does; PACC(MLP) in place of KDEy(MLP) as the quantifier of `PrediQuant`; the true-space distance of the selected bags stored per test bag and distance; a summary table module for the tables of E6.1.

Tests:

- with `rho=None`, the split is the one of `train_test_split` (same indices as before);
- with `rho` set, $V_q$ and $V_b$ are disjoint, cover $V$, have $\operatorname{round}(\rho\lvert V_j\rvert)$ instances of class $\omega_j$ in $V_q$, and are nested across levels;
- the quantifier never sees $V_b$, and a target equal to a candidate bag gets the bag's estimate (as `tests/test_prediquant.py` of `cap-ml`);
- the class support of a set of bags whose instances are known by construction;
- `predist` builds one task per (dataset, classifier) pair.

Reproducibility and comparability:

- `PQ` and all stored results of `main` are unchanged by `rho=None`; E6.2 adds the rows of five methods to `main`. Small datasets may have classes with fewer than $5$ instances in $V_q$ at $\rho=0.1$, which PACC's internal $5$-fold cross-validation cannot handle; such pairs are recorded as failed and counted, not silently dropped.
- E6.1 writes the `predist` schema, with a new schema version for the change of quantifier and the new column; its run replaces the stored `predist` results (LR, KDEy(MLP), re-run on 2026-11-09), which are not comparable with it.
- Pin bumps follow the rule of [[work_plan]] Part V (`cap-ml` pin).

# Appendix

## A. Toy simulations

None.

## B. Relation to other extensions

- **[[ext4_uncertainty_budget|Extension 4]]**: its error split has the three parts of §T.3 (prevalence, recalls, luck of the bag); it subsamples $\lvert V\rvert$ at the fixed allocation of `PQ`, while E6.2 varies the allocation at fixed $\lvert V\rvert$.
- **Extension 7** ([[extensions]] §7): when the quantifier of SQS-S is tuned, the search must stay inside $V_q$, so that tuning does not use $V_b$.
- **History.** The extension merges the former Extensions 4 and 6 (2026-10-27); it was restructured into this document on 2026-11-09, after FIX-1, which showed that the stored `predist` results compared two different estimators.

## C. Out of scope

- adaptive bandwidth, Aitchison geometry, the factorial study of $\rho$, $\lvert V\rvert$ and $\lvert U\rvert$, the Neyman allocation of $V$ across classes, and the label-free selection of $\rho$: set aside on 2026-11-09 ([[optional_extensions]] §11);
- separate results for binary and multiclass datasets (user's decision of 2026-11-09: not required);
- SQS-H and the other CAP methods: only SQS-S is studied here.
