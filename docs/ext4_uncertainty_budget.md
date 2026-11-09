# Extension 4: The error split of CAP under PPS

## Claim

A quantification-based CAP method estimates accuracy as "estimated prevalence of each class times estimated accuracy of $h$ within that class". Its error on a target bag therefore splits exactly into three parts: the error of the prevalences, the error of the recalls of $h$, and the luck of the bag, i.e. the gap between the realised accuracy of the bag and the accuracy expected from its prevalences. This extension measures the three parts, to see which one drives the error of each method of the main table in the paper's setting, how each part changes with the size $\lvert U\rvert$ of the target bag and the size $\lvert V\rvert$ of the validation set, and how much the error falls when a part is removed by replacing an estimated component with its true value (an oracle). The extension is descriptive: it measures the parts and does not model their variances. Throughout, the recalls of $h$ on the test pool stand for its population recalls (§T.1).

## Status

| Item | Status |
|---|---|
| Methods of the main table (`tables/main.py:MAIN_METHODS`): `DoC`, `QuAcc_NxN`, `O-LEAP`, `RQBS` (SQS-H), `PQ` (SQS-S), `BayesCAP-real` (BayCAP), `BayesCAP-Hf-real` (BayCAP-Hf), `BayCAP-KDEy-mlp-real` (BayCAP-KDEy, MLP input, fixed $\hat C$) | [implemented], stored in `main` at $\lvert U\rvert=1000$ |
| FIX-1 (SQS-S fits its quantifier on all of $V$; [[work_plan]] §I.2, card FIX-1 + FIX-4) | fixed in `cap-ml` `e9e77cd` (2026-11-09) and pinned in `uncawe` on the branch of FIX-1 + FIX-4 (only `PrediQuant` changes from the previous pin), `PQ` re-run in `main` on 2026-11-09, merged into `devel` (`fcdabf2`); the gate for every experiment of this extension, met (user's confirmation of 2026-11-09) |
| Error split with the pool recalls (§T.1, §T.2) and oracle $2\times2$ (§T.5) | implemented on the branch of WP-14 (`error_split.py`, `tables/error_split.py`, `plots/error_split.py`), not run |
| Fixed and bag parts within a cell (§T.4) | [proposal], not implemented |
| E4.1 (split on stored results), E4.3 (oracle $2\times2$ on stored results) | card WP-14 ([[work_plan]] §III.2), in progress: code and tests written, not run |
| E4.2 (parts across $\lvert U\rvert$ and $\lvert V\rvert$) | [proposal], not implemented; the FIX-1 gate is met |
| Bounds on the parts: on $e_1$ (brought back from the earlier §T.11), on the cross term $x$, on $e_2$ and $e_3$ (§T.6) | derived; tightness not measured |
| Candidates E4.c11 (oracles fed into the methods), E4.c12 (tightness of the bounds) | [proposal], not scheduled |
| Variance model of the parts and everything built on it, pool term, protocol without replacement, toy simulation of the budget | set aside on 2026-11-09, in [[optional_extensions]] §10 |

# Experiments

**Restructuring of 2026-11-09.** The user narrowed the extension to the error split, its behaviour across $\lvert U\rvert$ and $\lvert V\rvert$, and oracle substitutions, and restructured its experiments. The experiments and candidates of the previous version map as follows (the old ones are kept in [[optional_extensions]] §10):

| Old | New |
|---|---|
| — | E4.1 (new) |
| E4.1a, the parts across $\lvert U\rvert$ and $\lvert V\rvert$ | E4.2, restricted to two Bayesian methods, without the coverage rule and with the pool term merged into $e_2$ |
| oracle ladder of E4.1a | E4.3, a $2\times2$ design on stored results |
| E4.1c, the bound on the prevalence part | candidate E4.c12 (2026-11-09), extended to the bounds of §T.6 |
| E4.1b, E4.2a–E4.2f, E4.3a, E4.3b, candidates E4.c1, E4.c2 | set aside ([[optional_extensions]] §10) |

New candidates are numbered from E4.c11, since E4.c3–E4.c10 were used before.

**Gate.** No experiment of this extension runs before FIX-1 is resolved in `uncawe`: the `cap-ml` commit with the fix pinned, the effect of the new pin on the other methods taken from `cap-ml` checked, and `PQ` re-run in `main` (user's decision of 2026-11-09). The gate is met (user's confirmation of 2026-11-09): the fix is pinned (`cap-ml` `e9e77cd`, in which only `PrediQuant` changes from the previous pin, so no other method of `main` is affected), `PQ` was re-run in `main` on 2026-11-09, and the change is merged into `devel` (`fcdabf2`). E4.2 does not use `PQ`; the gate applies to it too, so that all the results of the extension come from the same `cap-ml`.

**Common settings.** Methods: the methods of the main table whose estimate comes from an estimated contingency table, i.e. all except `DoC`, which estimates accuracy directly (§T.2). Every split is computed against the realised table of each bag (`true_cts`), with the true prevalence of the bag and the recalls of $h$ on the test pool; it uses target labels, so it is an analysis, not a method. The Bayesian methods are taken with the realised-table predictive; for the split only their mean table matters (§T.2).

**Order.** E4.1 and E4.3 first, on the stored results of `main`, at no run cost; then E4.2 (pilot, then full run).

## E4.1: The error split on stored results

**Why.** In the paper's setting ($\lvert U\rvert=1000$, the full $V$) it is not known which part of the error dominates for each method: a method may be wrong mostly because of its prevalences, mostly because of the recalls of $h$ measured on $V$, or mostly because of the luck of the bag, which no estimate can remove. The answer says where each method can be improved and is the reference for reading E4.2 and E4.3.

- **Design.**
  - **Inputs:** the stored results of `main` (vanilla accuracy, $\lvert U\rvert=1000$, $100$ APP bags per pair, $75$ pairs), with `PQ` re-run after FIX-1, for `QuAcc_NxN`, `O-LEAP`, `RQBS`, `PQ`, `BayesCAP-real`, `BayesCAP-Hf-real`, `BayCAP-KDEy-mlp-real`: the mean estimated table `estim_cts` and the realised table `true_cts` of each bag; the predictions and labels of $h$ on the test pool, from the pretrain files.
  - **Varied:** dataset, classifier, method; nothing else.
  - **Measures:** per bag and method, $\tilde\pi$ and $\tilde d$ from `estim_cts`, the true prevalence $\pi^Q$ from `true_cts`, the pool recalls $d^{\text{pool}}$, and the parts $e_1$, $e_2$, $e_3$ of §T.2, with a check that $e_1+e_2+e_3=\hat A-A(h,U)$; per pair and method, the mean and the root mean square (RMS) of each part and of the total error over the $100$ bags, and the part with the largest RMS; per block of datasets (binary, multiclass), the RMS averaged over the pairs and the share of pairs on which each part has the largest RMS. Bags where NUTS failed are dropped and counted; a class with an empty estimated column ($\tilde\pi_j=0$) gets $\tilde d_j:=d^{\text{pool}}_j$, so its whole contribution goes to $e_1$, and such bags are counted (user's decision of 2026-11-09); a class absent from the bag ($n_j=0$) has weight $\pi^Q_j=0$ and contributes $0$ to $e_3$.
  - **Estimand:** the realised table of each bag, for every method; the split uses the true prevalence of the bag and the labels of the pool (an analysis).
  - **Cost:** none.
- **Representation.**
  - **Table A, RMS (user's decision of 2026-11-09):** one row per pair (dataset × classifier, grouped by dataset), in a block for binary and one for multiclass datasets, each closed by a row with the mean over its pairs and a row with the share of pairs in which each part is the largest. Column groups: $e_1$ (one column per method), $e_2$ (one per method), $e_3$ (a single column, since the luck of the bag uses no estimate and is the same for every method on the same bag) and the total error (one per method). In each row of the $e_1$, $e_2$ and total groups, colours run from the worst (red) to the best (green) method, as in the tables of E3.1, and the best is in bold; $e_3$ is not coloured. The total-error cell of each method carries a superscript $^{1}$, $^{2}$ or $^{3}$ naming its part with the largest RMS. A note recalls that the mean squares of the parts do not add up to the mean square of the total: $\overline{(e_1+e_2+e_3)^2}=\overline{e_1^2}+\overline{e_2^2}+\overline{e_3^2}+2(\overline{e_1e_2}+\overline{e_1e_3}+\overline{e_2e_3})$, with the bar the mean over the bags, and the cross products can be negative.
  - **Table B, signed means (user's decision of 2026-11-09):** the same rows, blocks and column groups without the total, with the mean of $e_1$, $e_2$ and $e_3$ over the bags (a mean far from $0$ is a systematic error); not coloured, since a large mean is bad whatever its sign.
  - **Bar plot:** mean square of each part, stacked, one bar per method, one panel per dataset.
- **Analysis.** Not run.

## E4.2: The parts across $\lvert U\rvert$ and $\lvert V\rvert$

**Why.** Each part has its own sources (§T.3): the luck of the bag depends on $U$ only; recalls estimated from $V$ alone depend on $V$ only; the prevalences, and in BayCAP-Hf also the recalls, depend on both. Varying the two sizes separates, within each part, what the bag contributes from what $V$ contributes (§T.4), and shows which part dominates when the bag or $V$ is small or large. The two Bayesian methods differ in one respect that matters here: BayCAP-KDEy keeps the recalls of $V$ fixed, while BayCAP-Hf updates them with the bag, so their $e_2$ should respond differently to $\lvert U\rvert$.

- **Design.**
  - **Inputs:** a subset of the datasets of `main` (binary and multiclass, small and large $\lvert V_j\rvert$, well- and ill-conditioned $C$, one large-$K$ dataset; the list is fixed with the pilot) and their test pools; classifiers LR and RF; methods `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real`, with the settings of `main`.
  - **Varied:**
    - $\lvert U\rvert\in\{100,300,1000,3000,10000\}$, including the paper's $1000$; bags drawn with replacement from the test pool, as in `main` (bags larger than the pool repeat instances);
    - $\lvert V\rvert$ level: the full $V$ and a stratified subsample of $1/4$ (and $1/16$ on the largest datasets), nested ($V_{1/16}\subset V_{1/4}\subset V$) and seeded; the methods are refitted on each subsample, $h$ is not retrained, and the split $L/V$ is unchanged;
    - $V$ seed: $S_V$ seeds of the subsamples (the full $V$ has one);
    - prevalence: three fixed vectors per dataset: uniform $\mathbf 1/K$, one skewed vector (a seeded draw from $\operatorname{Dir}(\mathbf 1)$), one near-vertex vector ($0.9$ on the rarest validation class, the rest uniform);
    - $R$ bags per cell (prevalence, $\lvert U\rvert$, $\lvert V\rvert$ level, $V$ seed);
    - held fixed: the pool, $h$, the methods' settings. $S_V$ and $R$ are fixed with the pilot (proposal: $S_V=3$, $R=30$).
  - **Measures:** per bag, the parts $e_1$, $e_2$, $e_3$ (§T.2); per cell, the fixed part and the bag part of each (§T.4); against $\lvert U\rvert$ and $\lvert V\rvert$ level, the RMS over the cells of the fixed parts and over the bags of the bag parts, and the fitted log–log slope of each RMS against $\lvert U\rvert$ (descriptive, with no predicted value).
  - **Estimand:** the realised table of each bag; the split uses the true prevalence of the bag and the labels of the pool (an analysis).
  - **Cost:** about $D\times2\times5\times L_V\times S_V\times3\times R$ NUTS runs per method, for $D$ datasets and $L_V$ levels of $\lvert V\rvert$ (the full $V$ has one seed, so this is an upper value); with the proposal, $D=6$, $L_V=2$: about $32\,000$ runs per method, about $4$ times a full run of `main` for one method. Reference times on `bcuda` at $\lvert U\rvert=1000$: a full run of `main` for `BayesCAP` ($7\,500$ NUTS runs) took about $46$ minutes ([[work_plan]] §I.2), one of `BayCAP-KDEy-Hf-mlp-real` ($6\,000$ runs) about $2$ hours (WP-13). The likelihood of BayCAP-Hf uses the counts of the predictions of $h$ on the bag, so its cost should barely depend on $\lvert U\rvert$; that of BayCAP-KDEy has one term per instance of the bag, so its cost should grow about linearly with $\lvert U\rvert$, about $10$ times at $\lvert U\rvert=10000$ (expectations, to be measured by the pilot).
- **Representation.**
  - **Log–log plot:** RMS of each part against $\lvert U\rvert$, one panel per method and $\lvert V\rvert$ level, one line per dataset (LR), solid for the fixed part and dashed for the bag part.
  - **Table:** per method, dataset and $\lvert V\rvert$ level, the RMS of the fixed and bag parts of $e_1$, $e_2$, $e_3$ at each $\lvert U\rvert$.
- **Analysis.** Not run.

## E4.3: Oracle substitutions on stored results

**Why.** E4.1 says how large each part is, but the parts are not independent: replacing the estimated prevalences with the true ones removes $e_1$, while replacing the estimated recalls with the pool ones removes $e_2$ and also changes the prevalence error, through a cross term (§T.5). A $2\times2$ design, which replaces each component alone and both together, measures how much the error falls when each component is known and whether the two components interact. It is computed after estimation, from each method's mean table, so it needs no run; feeding the oracles into the methods is candidate E4.c11.

- **Design.**
  - **Inputs:** as E4.1.
  - **Varied:** prevalence, estimated $\tilde\pi$ or true $\pi^Q$; recalls, estimated $\tilde d$ or pool $d^{\text{pool}}$; dataset, classifier, method. The three cells with an oracle are references, never methods.
  - **Measures:** per bag, method and cell, the estimate $\sum_j\pi_jd_j$ with the chosen $\pi$ and $d$, and its absolute error against the realised accuracy $A(h,U)$ and against the pool accuracy $\sum_j\pi^Q_jd^{\text{pool}}_j$; per pair, method and cell, the mean absolute error (AE); per method, the reduction of AE due to each oracle alone and to both, and the interaction (the cross term $x$ of §T.5, its mean and RMS).
  - **Estimand:** the realised table of each bag; the pool accuracy as a reference, which removes $e_3$.
  - **Cost:** none.
- **Representation.**
  - **Table, against the realised accuracy:** the rows and blocks of table A of E4.1; column groups for the four cells $(\tilde\pi,\tilde d)$, $(\pi^Q,\tilde d)$, $(\tilde\pi,d^{\text{pool}})$ and $(\pi^Q,d^{\text{pool}})$, each with the mean AE of every method, coloured in each row and group from the worst (red) to the best (green) method, with the best in bold; a last group with the mean and the RMS of the cross term $x$, not coloured.
  - **Table, against the pool accuracy:** the same layout, with every cell scored against $\sum_j\pi^Q_jd^{\text{pool}}_j$ (the cell $(\pi^Q,d^{\text{pool}})$ is then $0$).
- **Analysis.** Not run.

# Candidate experiments

Ideas that came up in the discussions; they are not scheduled. One becomes an experiment of the main section only when the user decides so.

- **E4.c11, oracles fed into the methods.** *Why:* E4.3 substitutes the true components after estimation, which leaves the other component of each estimate unchanged; a method that receives the true prevalence or the pool recalls as input may change its other component too (with $\pi^Q$ known, the posterior of $C$ of BayCAP-Hf changes), and only a run with the oracle inside the method gives an interval, hence a coverage. *Needs:* oracle variants of the methods (true $\pi^Q$; recalls fixed at $d^{\text{pool}}$), run on the protocol of `main`; to limit the cost, a few methods only (user's remark of 2026-11-09), e.g. `RQBS` and one Bayesian method.
- **E4.c12, tightness of the bounds.** *Why:* the bounds of §T.6 hold always; their ratio to the parts they bound says whether the error of the quantifier times the spread of the recalls (for $e_1$), or the product of the two errors (for the cross term $x$), is a good proxy of the error of accuracy, and how much of a part comes from the size of the per-class errors rather than from their arrangement across the classes. It is the former E4.1c, extended to $x$, $e_2$ and $e_3$. *Needs:* the per-bag quantities of E4.1 and E4.3 ($\tilde\pi$, $\tilde d$, $\pi^Q$, $d^{\text{pool}}$, $c_j/n_j$), no runs; per bag and method, each part, its bound and their ratio; per pair, the spread of the recalls $\max\tilde d-\min\tilde d$ and the median ratios; across pairs, the rank correlation between the accuracy error and the quantifier's AE, with and without the factor of the spread of the recalls.

# Theory

This part explains what the experiments need, step by step. We follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ for the target classifier $h$, $d=\operatorname{diag}(C)$ the vector of its recalls (the accuracy of $h$ within each class), and the population table $T=C\operatorname{diag}(\pi)$, with predicted classes on rows and true classes on columns. Numerical examples are illustrative, computed by hand.

## T.1 Setting, protocol, estimand, and the pool recalls

**Three versions of the recalls.** $d_j$ is the population recall of $h$ on class $\omega_j$: the probability that $h$ classifies correctly a new instance of that class. Two finite samples estimate it. The validation set $V$, with $\lvert V_j\rvert$ instances of class $\omega_j$, gives $\tilde d_j$, the share of them that $h$ classifies correctly. The labelled test pool from which the target bags are drawn, with $P_j$ instances of class $\omega_j$, has its own share $d^{\text{pool}}_j$.

**Protocol.** The paper's APP draws a bag $U$ of size $n=\lvert U\rvert$ from the pool, with class counts $n_j=n\pi^Q_j$ (rounded), and draws the instances of each class *with replacement*. Each instance of class $\omega_j$ in the bag is then a uniformly random instance of the pool's class $\omega_j$, correctly classified with probability $d^{\text{pool}}_j$, independently of the others. Write $c_j$ for the number of correctly classified instances of class $\omega_j$ in the bag.

**Estimand.** Every estimate is scored against the **realised accuracy** of the bag, computed from its realised table (`true_cts` in `main`):

$$
A(h,U)=\frac1n\sum_jc_j=\sum_j\frac{n_j}{n}\cdot\frac{c_j}{n_j}=\sum_j\pi^Q_j\,\frac{c_j}{n_j}.
$$

The second equality multiplies and divides each term by $n_j$; the third uses $n_j/n=\pi^Q_j$. In words, the realised accuracy is the prevalence of each class times the share of that class that $h$ gets right *in this bag*.

**Assumption: the pool recalls stand for the population recalls** (user's decision of 2026-11-09). The population recalls $d$ are unknown; the experiments use $d^{\text{pool}}$ in their place. This is also the natural reference under the protocol above: since the bags are drawn from the pool, the share $c_j/n_j$ varies from bag to bag around $d^{\text{pool}}_j$, not around $d_j$. Caveats:

- $e_2$ of §T.2 is then the gap between the recalls on $V$ and on the pool. $V$ and the pool are both finite samples, so this gap contains the error of both, and $e_2$ does not vanish as $\lvert V\rvert$ grows: it levels off at the error of the pool, which matters on small pools (e.g. `mammographic`, about $124$ instances per class);
- statements about the population accuracy $d^\top\pi^Q$, rather than the realised one, are out of scope; the separate study of the error of the pool (cause 4 of [[ext2_baycap|Extension 2]]) is set aside in [[optional_extensions]] §10.

## T.2 The error split

*Intuition.* Every quantification-based estimate has the form "estimated prevalence times estimated per-class accuracy"; its error is wrong prevalence, plus wrong per-class accuracy, plus the luck of this particular bag.

**Accuracy as a sum over classes.** The table $T=C\operatorname{diag}(\pi)$ has entries $T_{ij}=C_{ij}\pi_j$: the probability that an instance is of class $\omega_j$ and is predicted as $\omega_i$. Accuracy is the probability that the prediction is right, i.e. the sum of the diagonal entries (the trace): $\sum_jT_{jj}=\sum_jC_{jj}\pi_j=d^\top\pi$.

**Prevalences and recalls from any estimated table.** A method that estimates a table $\hat T$ (normalised to sum to $1$) gives an estimated prevalence and estimated recalls, read from it: the prevalence of class $\omega_j$ is the sum of column $j$, and the recall is the diagonal entry divided by that sum,

$$
\tilde\pi_j=\sum_i\hat T_{ij},\qquad \tilde d_j=\frac{\hat T_{jj}}{\tilde\pi_j}.
$$

Then $\tilde\pi_j\tilde d_j=\hat T_{jj}$, so $\hat A=\sum_j\tilde\pi_j\tilde d_j=\operatorname{tr}\hat T$: the estimate of accuracy is exactly the sum of the products. This holds for every method of the main table that estimates a table: the SQS methods (`RQBS`, `PQ`), whose $\tilde d$ is measured on $V$; the Bayesian methods, whose $\hat T$ is the mean of their table draws; `O-LEAP` and `QuAcc_NxN`, whose $\hat T$ is their own estimate. It does not hold for `DoC`, which estimates accuracy without a table. (In the code, `estim_cts` stores the transpose, true classes on rows, so $\tilde\pi$ is read from its row sums.)

For the Bayesian methods, $\hat T$ is the mean table over the draws. Accuracy is a linear function of the table (the trace), and the mean of a linear function is the linear function of the mean (linearity of the expectation, here of the average over draws); so the mean of the accuracies of the draws equals the accuracy of the mean table, and $\hat A$ is the method's point estimate. The split of the mean table is not the mean of the splits of the draws, since the mean of a product is not the product of the means; we split the mean table, so that each bag has one $\tilde\pi$ and one $\tilde d$.

**Derivation.** Adding and subtracting the same quantity does not change a value. We add and subtract two intermediate quantities: $\sum_j\pi^Q_j\tilde d_j$, the right prevalence with the estimated recalls, and $\sum_j\pi^Q_jd^{\text{pool}}_j$, the right prevalence with the pool recalls:

$$
\hat A-A(h,U)=\Big(\sum_j\tilde\pi_j\tilde d_j-\sum_j\pi^Q_j\tilde d_j\Big)+\Big(\sum_j\pi^Q_j\tilde d_j-\sum_j\pi^Q_jd^{\text{pool}}_j\Big)+\Big(\sum_j\pi^Q_jd^{\text{pool}}_j-A(h,U)\Big).
$$

Each intermediate quantity appears once with $+$ and once with $-$, so they cancel and the right-hand side is again $\hat A-A(h,U)$. Collecting terms, and writing $A(h,U)=\sum_j\pi^Q_jc_j/n_j$ (§T.1), the identity is exact for every bag:

$$
\hat A-A(h,U)=\underbrace{\sum_j(\tilde\pi_j-\pi^Q_j)\,\tilde d_j}_{e_1\ \text{prevalence}}+\underbrace{\sum_j\pi^Q_j(\tilde d_j-d^{\text{pool}}_j)}_{e_2\ \text{recalls}}+\underbrace{\sum_j\pi^Q_j\Big(d^{\text{pool}}_j-\frac{c_j}{n_j}\Big)}_{e_3\ \text{luck of the bag}}.
$$

In words: $e_1$ is the error from the wrong prevalence, weighted by the estimated recalls; $e_2$ is the error from the wrong recalls, weighted by the right prevalence; $e_3$ is the error that remains even with the right prevalence and the pool recalls, because $h$ gets right a share $c_j/n_j$ of the instances of the bag that differs from $d^{\text{pool}}_j$. All three are computable in an experiment, since the true prevalence of the bag and the labels of the pool are known.

The order of the add-and-subtract steps is a choice: the second-order cross term $\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d^{\text{pool}}_j)$ is included in $e_1$ (because $e_1$ uses $\tilde d$, not $d^{\text{pool}}$); the other order would move it into $e_2$. E4.3 measures it (§T.5).

**Worked example** (illustrative, computed by hand). Take $\pi^Q=(0.5,0.5)$, $\tilde\pi=(0.45,0.55)$, $d^{\text{pool}}=(0.90,0.70)$, $\tilde d=(0.88,0.73)$ and a bag in which $h$ gets right the shares $c_j/n_j=(0.89,0.69)$, so that $A(h,U)=0.5\cdot0.89+0.5\cdot0.69=0.79$. Then $\hat A=0.45\cdot0.88+0.55\cdot0.73=0.7975$ and $\hat A-A(h,U)=0.0075$. The parts are $e_1=(-0.05)(0.88)+(0.05)(0.73)=-0.0075$, $e_2=0.5(-0.02)+0.5(0.03)=0.005$ and $e_3=0.5(0.01)+0.5(0.01)=0.01$, and indeed $-0.0075+0.005+0.01=0.0075$.

**What the split does and does not say.** Being an identity, the split cannot be wrong; its content lies in the sizes and signs of the parts, which depend on how each method obtains $\tilde\pi$ and $\tilde d$ (§T.3) and which the experiments measure. A part whose mean over many bags is far from $0$ is a systematic error; a part with a large RMS and a mean near $0$ is noise.

## T.3 Sources of each part

*Intuition.* Each part is driven by the target bag $U$, by the validation set $V$, or by both. A part driven by $U$ gets smaller as the bag grows; a part driven by $V$ alone does not change with the size of the bag.

**The luck of the bag, $e_3$.** It depends on $U$ only. Within class $\omega_j$, $c_j/n_j$ is the share of correct predictions among $n_j$ instances, each correct with probability $d^{\text{pool}}_j$ independently of the others (§T.1). By the law of large numbers, the share of successes in $n_j$ independent trials gets closer to the probability of success as $n_j$ grows; so $e_3$ gets smaller as $\lvert U\rvert$ grows. Its mean over the bags is $0$, because the expected share of successes is the probability of success. It does not depend on $V$ or on the method: it is the same for every method on the same bag.

**The recalls, $e_2$.** It depends on how the method obtains $\tilde d$:

- `RQBS`, `PQ` and `BayCAP-KDEy-mlp-real` (fixed $\hat C$) take $\tilde d$ from $V$ alone; given $V$ and the prevalence of the bag, $e_2$ is fixed: it does not change from bag to bag and does not shrink with $\lvert U\rvert$. It depends on $\lvert V\rvert$, and on the pool through $d^{\text{pool}}$ (§T.1);
- `BayesCAP-real` and `BayesCAP-Hf-real` estimate the recalls jointly with the prevalence, from the counts of $V$ and the predictions of $h$ on the bag, so $\tilde d$ also depends on $U$, and on the prior of $C$ (a systematic pull, [[ext2_baycap|Extension 2]]);
- `O-LEAP` and `QuAcc_NxN` estimate the table with their own models, fitted on $V$ and applied to the bag; for them $\tilde d$ is not a share measured on $V$, and the sources of $e_2$ are measured, not predicted.

**The prevalences, $e_1$.** Every method estimates $\tilde\pi$ from the bag with a model fitted on $V$, so $e_1$ has a part due to the finite bag, which gets smaller as $\lvert U\rvert$ grows, and a part due to the finite $V$ (and to any systematic bias of the quantifier), which does not.

| Part | `RQBS`, `PQ`, BayCAP-KDEy | BayCAP, BayCAP-Hf | `O-LEAP`, `QuAcc_NxN` |
|---|---|---|---|
| $e_1$ | $U$ and $V$ | $U$, $V$ and prior | $U$ and $V$ |
| $e_2$ | $V$ (and pool) | $U$, $V$, prior (and pool) | measured |
| $e_3$ | $U$ | $U$ | $U$ |

E4.2 separates the contributions of $U$ and $V$ for the two methods that differ in the source of $e_2$: BayCAP-KDEy and BayCAP-Hf.

## T.4 Fixed part and bag part within a cell

*Intuition.* If many bags are drawn with the same prevalence, the same size and the same $V$, whatever their errors have in common comes from $V$ (and the pool), and whatever differs between them comes from the bags.

**Definition.** In E4.2, a *cell* is a combination of prevalence, $\lvert U\rvert$, $\lvert V\rvert$ level and $V$ seed, with $R$ bags $b=1,\dots,R$. For any part $e$ (one of $e_1$, $e_2$, $e_3$) with value $e^{(b)}$ on bag $b$, define its mean over the cell and the deviation of each bag from that mean:

$$
\bar e=\frac1R\sum_{b=1}^Re^{(b)},\qquad e^{(b)}=\underbrace{\bar e}_{\text{fixed part}}+\underbrace{\big(e^{(b)}-\bar e\big)}_{\text{bag part}}.
$$

The split is exact (add and subtract $\bar e$), and the bag parts of a cell sum to $0$ by construction, since $\sum_b(e^{(b)}-\bar e)=R\bar e-R\bar e=0$.

**Reading.** The fixed part is shared by the bags of a cell: it varies only across cells, i.e. across $V$ seeds, prevalences and sizes. The bag part measures how much the part changes from one bag to another with everything else held fixed. From §T.3: the fixed part of $e_3$ should be close to $0$ and its bag part should shrink as $\lvert U\rvert$ grows; for BayCAP-KDEy, $e_2$ should have no bag part at all (its $\tilde d$ is fixed given $V$) and a fixed part that does not change with $\lvert U\rvert$; for BayCAP-Hf, $e_2$ may have a bag part, since its recalls are updated by each bag.

**A caveat.** $\bar e$ is the mean of $R$ bags, not of infinitely many, so the fixed part carries a share of the bag-to-bag noise, which gets smaller as $R$ grows; the fixed part of a part with a large bag part is therefore measured less precisely.

## T.5 The oracle $2\times2$

*Intuition.* Replacing an estimated component with its true value removes the corresponding error, but the two components multiply in the estimate, so replacing one also changes how much the error of the other counts.

**The four cells.** With $\pi\in\{\tilde\pi,\pi^Q\}$ and $d\in\{\tilde d,d^{\text{pool}}\}$, the four estimates are $\hat A_{\pi,d}=\sum_j\pi_jd_j$. Write $x=\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d^{\text{pool}}_j)$ for the cross term. Subtracting $A(h,U)$ and using the split of §T.2:

- **$(\tilde\pi,\tilde d)$**, the method: the error is $e_1+e_2+e_3$.
- **$(\pi^Q,\tilde d)$**: the prevalence error is gone; adding and subtracting $\sum_j\pi^Q_jd^{\text{pool}}_j$ gives $\sum_j\pi^Q_j\tilde d_j-A(h,U)=e_2+e_3$.
- **$(\pi^Q,d^{\text{pool}})$**: both errors are gone; $\sum_j\pi^Q_jd^{\text{pool}}_j-A(h,U)=e_3$.
- **$(\tilde\pi,d^{\text{pool}})$**: the recall error is gone. Adding and subtracting $\sum_j\pi^Q_jd^{\text{pool}}_j$ gives $\sum_j(\tilde\pi_j-\pi^Q_j)d^{\text{pool}}_j+e_3$. Writing $d^{\text{pool}}_j=\tilde d_j-(\tilde d_j-d^{\text{pool}}_j)$ in the first sum and separating it into two sums gives $\sum_j(\tilde\pi_j-\pi^Q_j)\tilde d_j-\sum_j(\tilde\pi_j-\pi^Q_j)(\tilde d_j-d^{\text{pool}}_j)=e_1-x$. So the error is $e_1-x+e_3$.

**Main effects and interaction.** In terms of the signed error, knowing the prevalence removes $e_1$ when the recalls are estimated (first two cells) and $e_1-x$ when the recalls are the pool ones (last two cells). The two removals differ by $x$, which is therefore the interaction of the two components: how much the prevalence error depends on whether the recalls are right. If $x$ is small compared with $e_1$, the two components can be studied separately. E4.3 reports the mean absolute error of each cell, which is what a user sees; the signed relations above explain the differences between the cells.

**Against the pool accuracy.** Scoring the four cells against $\sum_j\pi^Q_jd^{\text{pool}}_j$ instead of $A(h,U)$ removes $e_3$ from every cell (the last cell then has error $0$), which separates the error that a better estimate could remove from the luck of the bag, which it cannot.

**Post-estimation oracles.** The cells keep each method's other component as it was estimated. For a method whose two components are estimated jointly, as BayCAP and BayCAP-Hf, knowing one would change the other; that version is candidate E4.c11.

## T.6 Bounds on the parts

*Intuition.* A wrong prevalence changes the estimated accuracy only because the classes have different recalls: giving too much weight to a class matters only if that class is easier or harder for $h$ than the others. So the damage is at most "how much weight was moved" times "how different the recalls are". The same argument bounds the cross term of §T.5, and the two other parts are averages, so they lie between their smallest and largest per-class terms.

This section comes from §T.11 of the earlier version of the extension (set aside in [[optional_extensions]] §10 and brought back on 2026-11-09 at the user's request), adapted to the split of the mean table (§T.2); the bounds on $x$, $e_2$ and $e_3$ are added here. The bounds are mathematical facts; how tight they are on real data is not known (candidate E4.c12).

**A lemma.** Let $u$ be a vector whose entries sum to $0$, and $a$ any vector with the same number of entries. Write $\lVert u\rVert_1=\sum_j\lvert u_j\rvert$ for the $L_1$ norm of $u$. Then

$$
\Big\lvert\sum_ju_ja_j\Big\rvert\le\tfrac12\lVert u\rVert_1\cdot\Big(\max_ja_j-\min_ja_j\Big).
$$

*Derivation.*

1. Split the entries into those with $u_j>0$ and those with $u_j<0$. Since the $u_j$ sum to $0$, the positive ones and the negative ones have the same total size: $\sum_{u_j>0}u_j=\sum_{u_j<0}\lvert u_j\rvert$. The two totals add up to $\lVert u\rVert_1$, so each is $\frac12\lVert u\rVert_1$.
2. Then $\sum_ju_ja_j=\sum_{u_j>0}u_ja_j-\sum_{u_j<0}\lvert u_j\rvert a_j$.
3. In the first sum every $a_j$ is at most $\max a$; in the second every $a_j$ is at least $\min a$. Replacing them can only increase the first sum and decrease what is subtracted, so $\sum_ju_ja_j\le\frac12\lVert u\rVert_1\max a-\frac12\lVert u\rVert_1\min a$.
4. Exchanging the roles of $\max$ and $\min$ gives $\sum_ju_ja_j\ge-\frac12\lVert u\rVert_1(\max a-\min a)$. Together, the claim.

The bound is tight: equality holds when all the weight of $u$ moves from the entry with the smallest $a_j$ to the entry with the largest.

**The prevalence part.** Take $u=\tilde\pi-\pi^Q$, which sums to $1-1=0$ because both are prevalence vectors, and $a=\tilde d$. The lemma gives

$$
\lvert e_1\rvert=\Big\lvert\sum_j(\tilde\pi_j-\pi^Q_j)\,\tilde d_j\Big\rvert\le\tfrac12\lVert\tilde\pi-\pi^Q\rVert_1\cdot\Big(\max_j\tilde d_j-\min_j\tilde d_j\Big).
$$

In the binary case $\frac12\lVert u\rVert_1=\lvert\tilde\pi_1-\pi^Q_1\rvert$ and $e_1=(\tilde\pi_1-\pi^Q_1)(\tilde d_1-\tilde d_0)$ (since $u_0=-u_1$), so the bound is an equality. The quantification literature reports the absolute error $\mathrm{AE}=\lVert\tilde\pi-\pi^Q\rVert_1/K$ of a quantifier, so $\lvert e_1\rvert\le\frac K2\,\mathrm{AE}\,(\max\tilde d-\min\tilde d)$: the error of the quantifier, times the spread of the recalls, bounds the error of accuracy that it causes. If all the recalls are equal, a wrong prevalence costs nothing, which is the condition of Eq. 2 of the paper. Since the split uses one $\tilde\pi$ and one $\tilde d$ per bag, read from the method's mean table (§T.2), the bound holds for every method that estimates a table, `O-LEAP` and `QuAcc_NxN` included; it does not hold for `DoC`. The same derivation bounds the difference between the accuracies of any two prevalence vectors $\pi,\pi'$ with the same recalls: $\lvert\sum_j(\pi_j-\pi'_j)\tilde d_j\rvert\le\frac12\lVert\pi-\pi'\rVert_1(\max\tilde d-\min\tilde d)$.

**The cross term of §T.5.** Take the same $u=\tilde\pi-\pi^Q$ and $a=\delta$, with $\delta_j=\tilde d_j-d^{\text{pool}}_j$ the error of the recall of class $\omega_j$. The lemma gives

$$
\lvert x\rvert=\Big\lvert\sum_j(\tilde\pi_j-\pi^Q_j)\,\delta_j\Big\rvert\le\tfrac12\lVert\tilde\pi-\pi^Q\rVert_1\cdot\Big(\max_j\delta_j-\min_j\delta_j\Big)\le\lVert\tilde\pi-\pi^Q\rVert_1\cdot\max_j\lvert\delta_j\rvert,
$$

where the last step uses $\max\delta-\min\delta\le2\max_j\lvert\delta_j\rvert$. So the interaction of the oracle $2\times2$ is the product of two errors, the prevalence error and the spread of the recall errors: it is small whenever either is small, and it is zero when all the recalls are wrong by the same amount.

**The recall part and the luck of the bag.** Both are averages with the weights $\pi^Q_j$, which are non-negative and sum to $1$: $e_2=\sum_j\pi^Q_j\delta_j$ and $e_3=\sum_j\pi^Q_j\varepsilon_j$, with $\varepsilon_j=d^{\text{pool}}_j-c_j/n_j$ the luck of the bag in class $\omega_j$. A weighted average with such weights lies between the smallest and the largest of the averaged values (replacing every value by the largest can only increase the sum, and the weights then sum to $1$; likewise for the smallest), so

$$
\min_j\delta_j\le e_2\le\max_j\delta_j,\qquad \min_j\varepsilon_j\le e_3\le\max_j\varepsilon_j,
$$

and in particular $\lvert e_2\rvert\le\max_j\lvert\delta_j\rvert$ and $\lvert e_3\rvert\le\max_j\lvert\varepsilon_j\rvert$. Unlike $e_1$, these parts do not cancel when their per-class errors are equal: a recall error common to all classes passes entirely into $e_2$.

**What the bounds are for.** They turn per-class errors, which are what a quantifier or a validation set gets wrong, into errors of accuracy, without any assumption on how the errors are distributed. Per bag they are not computable without labels, since $\pi^Q$ and $d^{\text{pool}}$ are unknown; on the stored results, where both are known, they say how much of each part is explained by the size of the per-class errors and how much by their arrangement across the classes.

# Implementation

Components:

- error split (`src/uncawe/error_split.py`, new, pure numpy; card WP-14): $\tilde\pi$ and $\tilde d$ from a stored `estim_cts` (transposed in storage: true classes on rows; normalised to sum to $1$), $\pi^Q$ and the shares $c_j/n_j$ from `true_cts`, $d^{\text{pool}}$ from the predictions and labels of $h$ on the test pool (pretrain files); the parts $e_1$, $e_2$, $e_3$ (§T.2), the cross term $x$ and the four oracle estimates (§T.5); the fixed and bag parts within a cell (§T.4);
- analyses of E4.1 and E4.3 on the stored results of `main` (card WP-14): the tables of both in `tables/error_split.py` (a per-bag CSV, tables A and B of E4.1, the two tables of E4.3), the bar plot of E4.1 in `plots/error_split.py`, following [[work_plan]] Part V;
- E4.2: a new study `target_size`, with bags of fixed prevalence at any $\lvert U\rvert$ drawn with replacement from the pool (per-bag seeds from the dataset, prevalence, $\lvert U\rvert$, $\lvert V\rvert$ level, $V$ seed and bag index), nested seeded stratified subsampling of $V$ with the methods refitted on each subsample and $h$ not retrained, and the methods `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real` built as in `main`.

Tests:

- $e_1+e_2+e_3=\hat A-A(h,U)$ exactly on random tables;
- with $\tilde\pi=\pi^Q$ and $\tilde d=d^{\text{pool}}$, $e_1=e_2=0$; the four oracle errors equal $e_1+e_2+e_3$, $e_2+e_3$, $e_1-x+e_3$ and $e_3$;
- reading $\tilde\pi$ and $\tilde d$ from a stored (transposed) table gives the column sums and recalls of the paper's orientation;
- the bag parts of a cell sum to $0$;
- the fixed-prevalence sampler returns bags with the requested class counts and size; nested subsamples satisfy $V_{1/16}\subset V_{1/4}\subset V$ and are stratified.

Reproducibility and comparability:

- E4.1 and E4.3 read stored results only and change no result; they run after the gate (FIX-1), on the `main` results of the pinned `cap-ml`;
- E4.2 writes to `cap.env["OUT_DIR"]/uncawe/target_size`; `main` and its outputs are unchanged; every row records $\lvert U\rvert$, the $\lvert V\rvert$ level, the $V$ seed, the prevalence, the bag seed, `config_hash` and `schema_version`; its bags have fixed prevalences rather than APP ones, so its cells at $\lvert U\rvert=1000$ and the full $V$ are comparable with `main` in distribution, not bag by bag.

# Appendix

## A. Relation to other extensions

- **[[ext2_baycap|Extension 2]]**: its estimand gap is the term $e_3$; the systematic pull of the prior of $C$ shows up in the mean of $e_2$ for BayCAP and BayCAP-Hf. Its cause 4 (the error of the pool) is merged into $e_2$ here (§T.1); its separate study is set aside ([[optional_extensions]] §10).
- **[[ext3_posterior_propagation|Extension 3]]**: BayCAP-KDEy (fixed $\hat C$, MLP input, realised pushforward), its best method of E3.1, is one of the methods of E4.1–E4.3, with the implementation of E3.1.
- **Extension 6** ([[extensions]] §6): varies the allocation $\rho$ at fixed $\lvert V\rvert$; here $\lvert V\rvert$ is subsampled at fixed allocation.

## B. Out of scope

- the variance model of the parts and everything built on it (coverage rule, regime map, width floors, plug-ins, completed and guaranteed intervals), the parts of BayCAP with joint estimation, the pool term and the protocol without replacement, and the toy simulation of the budget: set aside on 2026-11-09 ([[optional_extensions]] §10);
- the methods not in the main table (other BayCAP priors, population pushforwards, the other methods of Extension 3, `L-CBPE`): they may be shown in separate experiments, not here;
- violations of PPS ([[optional_extensions]] §2); the validation allocation $\rho$ (Extension 6).
