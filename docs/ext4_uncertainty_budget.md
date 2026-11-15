# Extension 4: The error split of CAP under PPS

## Claim

A quantification-based CAP method estimates accuracy as "estimated prevalence of each class times estimated accuracy of $h$ within that class". Its error on a target bag therefore splits exactly into three parts: the error of the prevalences, the error of the recalls of $h$, and the luck of the bag, i.e. the gap between the realised accuracy of the bag and the accuracy expected from its prevalences. This extension measures the three parts, to see which one drives the error of each method of the main table in the paper's setting, how each part changes with the size $\lvert U\rvert$ of the target bag and the size $\lvert V\rvert$ of the validation set, and how much the error falls when a part is removed by replacing an estimated component with its true value (an oracle). The extension is descriptive: it measures the parts and does not model their variances. Throughout, the recalls of $h$ on the test pool stand for its population recalls (§T.1).

## Status

| Item | Status |
|---|---|
| Methods of the main table (`tables/main.py:MAIN_METHODS`): `DoC`, `QuAcc_NxN`, `O-LEAP`, `SQS-H` (SQS-H as in the paper, FIX-2), `SQS-Hbs` (SQS-H with a bootstrap of the prevalence, formerly `RQBS`), `PQ` (SQS-S), `BayesCAP-real` (BayCAP), `BayesCAP-Hf-real` (BayCAP-Hf), `BayCAP-KDEy-mlp-real` (BayCAP-KDEy, MLP input, fixed $\hat C$) | [implemented], stored in `main` at $\lvert U\rvert=1000$ |
| FIX-1 (SQS-S fits its quantifier on all of $V$; [[work_plan]] §I.2, card FIX-1 + FIX-4) | fixed in `cap-ml` `e9e77cd` (2026-11-09) and pinned in `uncawe` on the branch of FIX-1 + FIX-4 (only `PrediQuant` changes from the previous pin), `PQ` re-run in `main` on 2026-11-09, merged into `devel` (`fcdabf2`); the gate for every experiment of this extension, met (user's confirmation of 2026-11-09) |
| Error split with the pool recalls (§T.1, §T.2) and oracle $2\times2$ (§T.5) | [implemented] by WP-14 (`error_split.py`, `tables/error_split.py`, `plots/error_split.py`; [[work_plan]] §II.20), run on the stored results of `main` (2026-11-10) |
| Fixed and bag parts within a cell (§T.4) | [implemented] by WP-21 (`error_split.py:cell_parts`, `fixed_se`, `over_seeds`; `tables/target_size.py`), run on E4.2b (2026-11-15) |
| E4.1 (split on stored results), E4.3 (oracle $2\times2$ on stored results) | run and analysed on 2026-11-10 (`PQ` re-run in `main` with `cap-ml` `87b4d9b`, which stores its table); card WP-14 done ([[work_plan]] §II.20, §III.3) |
| E4.2a (parts across $\lvert U\rvert$, full $V$), E4.2b (parts across $\lvert V\rvert$ with $S_V$ seeds, $\lvert U\rvert=1000$); split of E4.2 into two parts on 2026-11-11 | E4.2b [implemented] (study `target_size`), full run on 2026-11-15 (pilot skipped, user's decision) and analysed, card WP-21 done (merged `7fc4dc6`; [[work_plan]] §II.26, §III.3); E4.2a [implemented] (study `target_size`, `--u-grid`), not run (card WP-22, In progress; [[work_plan]] §III.2); the FIX-1 gate is met |
| Bounds on the parts: on $e_1$ (brought back from the earlier §T.11), on the cross term $x$, on $e_2$ and $e_3$ (§T.6) | derived; tightness not measured |
| Candidates E4.c11 (oracles fed into the methods), E4.c12 (tightness of the bounds), E4.c13 (interaction of $\lvert U\rvert$ and $\lvert V\rvert$) | [proposal], not scheduled |
| Variance model of the parts and everything built on it, pool term, protocol without replacement, toy simulation of the budget | set aside on 2026-11-09, in [[optional_extensions]] §10 |

# Experiments

**Restructuring of 2026-11-09.** The user narrowed the extension to the error split, its behaviour across $\lvert U\rvert$ and $\lvert V\rvert$, and oracle substitutions, and restructured its experiments. The experiments and candidates of the previous version map as follows (the old ones are kept in [[optional_extensions]] §10):

| Old | New |
|---|---|
| — | E4.1 (new) |
| E4.1a, the parts across $\lvert U\rvert$ and $\lvert V\rvert$ | E4.2, restricted to two Bayesian methods, without the coverage rule and with the pool term merged into $e_2$; split on 2026-11-11 into E4.2a ($\lvert U\rvert$) and E4.2b ($\lvert V\rvert$) |
| oracle ladder of E4.1a | E4.3, a $2\times2$ design on stored results |
| E4.1c, the bound on the prevalence part | candidate E4.c12 (2026-11-09), extended to the bounds of §T.6 |
| E4.1b, E4.2a–E4.2f, E4.3a, E4.3b, candidates E4.c1, E4.c2 | set aside ([[optional_extensions]] §10), where the old experiment numbers are prefixed with `s` since 2026-11-11 (E4.s1b, E4.s2a–E4.s2f, E4.s3a, E4.s3b), so that E4.2a and E4.2b name the parts of the current E4.2 |

New candidates are numbered from E4.c11, since E4.c3–E4.c10 were used before. In this table the column "Old" keeps the names of that version (E4.1a, E4.2a–E4.2f, …), which are E4.s1a, E4.s2a–E4.s2f, … in [[optional_extensions]] §10.

**Gate.** No experiment of this extension runs before FIX-1 is resolved in `uncawe`: the `cap-ml` commit with the fix pinned, the effect of the new pin on the other methods taken from `cap-ml` checked, and `PQ` re-run in `main` (user's decision of 2026-11-09). The gate is met (user's confirmation of 2026-11-09): the fix is pinned (`cap-ml` `e9e77cd`, in which only `PrediQuant` changes from the previous pin, so no other method of `main` is affected), `PQ` was re-run in `main` on 2026-11-09, and the change is merged into `devel` (`fcdabf2`). E4.2 (both parts) does not use `PQ`; the gate applies to it too, so that all the results of the extension come from the same `cap-ml`.

**Common settings.** Methods: the methods of the main table whose estimate comes from an estimated contingency table, i.e. all except `DoC`, which estimates accuracy directly (§T.2). Every split is computed against the realised table of each bag (`true_cts`), with the true prevalence of the bag and the recalls of $h$ on the test pool; it uses target labels, so it is an analysis, not a method. The Bayesian methods are taken with the realised-table predictive; for the split only their mean table matters (§T.2).

**Order.** E4.1 and E4.3 first, on the stored results of `main`, at no run cost; then E4.2b (card WP-21: pilot, which fixes $R$ and $S_V$, then full run), first since it tests the hypothesis of E4.1 at the cost per run of `main`, and E4.2a (card WP-22: pilot on one dataset to measure the cost at large $\lvert U\rvert$, then full run).

## E4.1: The error split on stored results

**Why.** In the paper's setting ($\lvert U\rvert=1000$, the full $V$) it is not known which part of the error dominates for each method: a method may be wrong mostly because of its prevalences, mostly because of the recalls of $h$ measured on $V$, or mostly because of the luck of the bag, which no estimate can remove. The answer says where each method can be improved and is the reference for reading E4.2 and E4.3.

- **Design.**
  - **Inputs:** the stored results of `main` (vanilla accuracy, $\lvert U\rvert=1000$, $100$ APP bags per pair, $75$ pairs), with `PQ` re-run after FIX-1 and again with `cap-ml` `87b4d9b`, which stores its table (card WP-14), for `QuAcc_NxN`, `O-LEAP`, `RQBS`, `PQ`, `BayesCAP-real`, `BayesCAP-Hf-real`, `BayCAP-KDEy-mlp-real`: the mean estimated table `estim_cts` and the realised table `true_cts` of each bag; the predictions and labels of $h$ on the test pool, from the pretrain files.
  - **Varied:** dataset, classifier, method; nothing else.
  - **Measures:** per bag and method, $\tilde\pi$ and $\tilde d$ from `estim_cts`, the true prevalence $\pi^Q$ from `true_cts`, the pool recalls $d^{\text{pool}}$, and the parts $e_1$, $e_2$, $e_3$ of §T.2, with a check that $e_1+e_2+e_3=\hat A-A(h,U)$; per pair and method, the mean and the root mean square (RMS) of each part and of the total error over the $100$ bags, and the part with the largest RMS; per block of datasets (binary, multiclass), the RMS averaged over the pairs and the share of pairs on which each part has the largest RMS. Bags where NUTS failed are dropped and counted; a class with an empty estimated column ($\tilde\pi_j=0$) gets $\tilde d_j:=d^{\text{pool}}_j$, so its whole contribution goes to $e_1$, and such bags are counted (user's decision of 2026-11-09); a class absent from the bag ($n_j=0$) has weight $\pi^Q_j=0$ and contributes $0$ to $e_3$.
  - **Estimand:** the realised table of each bag, for every method; the split uses the true prevalence of the bag and the labels of the pool (an analysis).
  - **Cost:** none.
- **Representation.**
  - **Table A, RMS (user's decision of 2026-11-09):** one row per pair (dataset × classifier, grouped by dataset), in a block for binary and one for multiclass datasets, each closed by a row with the mean over its pairs and a row with the share of pairs in which each part is the largest. Column groups: $e_1$ (one column per method), $e_2$ (one per method), $e_3$ (a single column, since the luck of the bag uses no estimate and is the same for every method on the same bag) and the total error (one per method). In each row of the $e_1$, $e_2$ and total groups, colours run from the worst (red) to the best (green) method, as in the tables of E3.1, and the best is in bold; $e_3$ is not coloured. The total-error cell of each method carries a superscript $^{1}$, $^{2}$ or $^{3}$ naming its part with the largest RMS. A note recalls that the mean squares of the parts do not add up to the mean square of the total: $\overline{(e_1+e_2+e_3)^2}=\overline{e_1^2}+\overline{e_2^2}+\overline{e_3^2}+2(\overline{e_1e_2}+\overline{e_1e_3}+\overline{e_2e_3})$, with the bar the mean over the bags, and the cross products can be negative.
  - **Table B, signed means (user's decision of 2026-11-09):** the same rows, blocks and column groups without the total, with the mean of $e_1$, $e_2$ and $e_3$ over the bags (a mean far from $0$ is a systematic error); not coloured, since a large mean is bad whatever its sign.
  - **Bar plot:** mean square of each part, stacked, on a linear scale, one bar per method, one panel per dataset, without `QuAcc_NxN` (`O-LEAP` stays as the baseline); `QuAcc_NxN`, whose errors are one to two orders of magnitude larger and would flatten the other bars, in a separate figure with one bar per dataset (user's decision of 2026-11-10).
- **Analysis.** Source: the tables and the bar plots of `tables/error_split.py` and `plots/error_split.py`, run on `bcuda` on 2026-11-10 from commit `24b6776` (bar plots from `45a69b5`), on the stored results of `main` with `PQ` re-run with `cap-ml` `87b4d9b`; the per-pair values are in `<OUT_DIR>/uncawe/tables/error_split/error_split.csv`. Values below are means over the pairs of a block ($25$ binary pairs, $50$ multiclass pairs) of per-pair quantities over the $100$ bags. After FIX-2 + FIX-3 (`QuAcc_NxN`, `O-LEAP`, `PQ` and `RQBS`, renamed `SQS-Hbs`, re-run with seeded quantifiers; `SQS-H` added) and FIX-8 (`poker_hand` with $K=7$, every method re-run on it), the tables and bar plots were generated again on 2026-11-13 from `50f17df`; the numbers below are updated, with the value before these fixes in brackets where it changed ("before"). The readings do not change. The identity $e_1+e_2+e_3=\hat A-A$ holds on every bag within $10^{-14}$ for every method except `O-LEAP` (within $3.3\cdot10^{-11}$; before, $8.1\cdot10^{-11}$) and `QuAcc_NxN`, whose stored estimate differs from the trace of its stored mean table by up to $1.1\cdot10^{-6}$ ($26$ pairs; before, $1.5\cdot10^{-6}$ on $11$); no bag is dropped (no NUTS failure); bags with at least one empty estimated column: `QuAcc_NxN` 839, `SQS-H` 2385, `SQS-Hbs` 730, BayCAP-KDEy 11, none for the other methods (before: `QuAcc_NxN` 838, `RQBS` 910).
  - **The luck of the bag is a common floor.** $e_3$ has RMS $0.0099$ (binary) and $0.0089$ (multiclass) and mean $\approx0$, as §T.3 predicts. Where the methods are accurate it is a large part of their error: its mean square is $72$–$74\%$ of the mean square of the total error on `spambase` for every method except `PQ` ($38\%$) and `QuAcc_NxN` ($8\%$), $65$–$72\%$ on `hand_digits` except `PQ` ($56\%$), `BayesCAP-real` ($5\%$) and `QuAcc_NxN` ($0\%$), $89\%$ for BayCAP-KDEy on `letter` (before: $72$–$75\%$ on `spambase`); there little is left that a better estimate of the table could remove.
  - **Binary datasets: the recalls dominate, and their error is systematic.** $e_2$ is the part with the largest RMS on $44$–$88\%$ of the binary pairs, depending on the method (e.g. `SQS-Hbs`: $e_1$ RMS $0.0074$, $e_2$ $0.0287$, the same for `SQS-H`; BayCAP-KDEy: $0.0071$ and $0.0201$; before, `RQBS` $0.0068$ and $0.0287$). For every method $\lvert\text{mean of }e_2\rvert/\text{RMS of }e_2$ is $0.77$–$0.87$ (median over the pairs; before, $0.77$–$0.88$), so $e_2$ is mostly an offset that repeats on every bag of a pair; §T.3 predicts this for the methods that take $\tilde d$ from $V$ (`SQS-H`, `SQS-Hbs`, `PQ`, BayCAP-KDEy), but it holds for `O-LEAP` and the BayesCAP variants too. The offset has the same sign for every method: the mean of $e_2$ is $+0.014$ to $+0.020$ (`QuAcc_NxN` $+0.053$), mostly from `mammographic` ($+0.051$ to $+0.059$ for every method, $\lvert V\rvert=291$, the smallest validation set) and `german` (up to $+0.024$, $\lvert V\rvert=350$). It is smaller for RF and kNN ($+0.009$ to $+0.017$) than for LR, SVM and MLP ($+0.013$ to $+0.026$). Since methods with different estimators of the recalls share it, the hypothesis is that it comes from the finite $V$: on these validation sets the recalls of $h$ are higher than on the pool (for a recall near $0.8$ measured on about $145$ instances the standard error is $\sqrt{0.8\cdot0.2/145}\approx0.033$, so a gap of $0.058$ is within reach of a single $V$). E4.2b, which varies $\lvert V\rvert$, tests it: the gap between the recalls of $h$ on $V$ and on the pool is within $1.7$ standard deviations of what two finite samples of these sizes give, so the hypothesis holds, in the form "finite $V$ and finite pool" (E4.2b, Analysis).
  - **Multiclass datasets: the prevalences weigh more.** $e_1$ is the part with the largest RMS on $28$–$52\%$ of the multiclass pairs, and its RMS exceeds that of $e_2$ for every method except `QuAcc_NxN` (e.g. `SQS-Hbs`: $e_1$ $0.0293$, $e_2$ $0.0097$; `SQS-H`: $0.0293$ and $0.0093$; BayCAP-KDEy: $0.0287$ and $0.0108$; `PQ`: $0.0345$ and $0.0162$; before: $30$–$52\%$; `RQBS` $0.0242$ and $0.0091$, BayCAP-KDEy $0.0237$ and $0.0107$, `PQ` $0.0347$ and $0.0159$). The RMS of $e_1$ grows mostly through `poker_hand`, where after FIX-8 $e_1$ carries almost the whole error of every method (RMS $0.08$–$0.17$, against $e_2$ at most $0.009$ except `O-LEAP`, $0.067$, and `QuAcc_NxN`). `BayesCAP-real` has a systematic negative $e_2$ that grows with $K$ (mean $-0.074$ on `letter`, $-0.167$ on `isolet`, $K=26$): it underestimates the recalls, consistent with the pull of the prior on $C$ ([[ext2_baycap|Extension 2]]); `BayesCAP-Hf-real` removes it (multiclass mean of $e_2$: $+0.001$; RMS $0.0088$ against $0.0355$).
  - **`QuAcc_NxN`** has the largest parts by one to two orders of magnitude (multiclass RMS: $e_1$ $0.076$, $e_2$ $0.250$, total $0.229$; before, $0.071$, $0.241$, $0.227$), and they partly cancel: the mean of the cross term $x$ is $-0.074$ (binary) and $-0.129$ (multiclass; before, $-0.068$ and $-0.126$), against $\lvert\text{mean of }x\rvert\le0.0042$ for every other method, and the mean square of its total error is well below the sum of the mean squares of its parts. Its table cannot be read as prevalences times recalls that are each close to the truth.

## E4.2: The parts across $\lvert U\rvert$ and $\lvert V\rvert$

**Why.** Each part has its own sources (§T.3): the luck of the bag depends on $U$ only; recalls estimated from $V$ alone depend on $V$ only; the prevalences, and in BayCAP-Hf also the recalls, depend on both. Varying the two sizes separates, within each part, what the bag contributes from what $V$ contributes (§T.4), and shows which part dominates when the bag or $V$ is small or large. The two Bayesian methods differ in one respect that matters here: BayCAP-KDEy keeps $\hat C$ fixed at the rates of $V$, while BayCAP-Hf updates $C$ with the bag, so their $e_2$ should respond differently to $\lvert U\rvert$; with the realised-table predictive the recalls of both are also reweighted by the predictions of $h$ on the bag (§T.3, corrected after E4.2b). The two sizes are varied in two parts, one size at a time (user's decision of 2026-11-11): E4.2a varies $\lvert U\rvert$ with the full $V$, E4.2b varies $\lvert V\rvert$ at the paper's $\lvert U\rvert=1000$. The interaction of the two sizes (whether a large bag makes up for a small $V$) is candidate E4.c13.

**Common settings.**

- **Datasets:** six datasets of `main` and their test pools (user's decision of 2026-11-11), chosen from the results of E4.1 (`tables/error_split/error_split.csv`): `mammographic` ($K=2$, $\lvert V\rvert=291$), the binary offset case; `german` ($K=2$, $350$), small and ill-conditioned, with an offset of $e_2$ for BayCAP-Hf only ($+0.021$ to $+0.029$, against $-0.000$ to $+0.006$ for BayCAP-KDEy); `spambase` ($K=2$, $1610$), a binary control without offset, whose subsamples ($\approx400$ and $\approx100$) reach the size of the small binary validation sets; `wine-quality` ($K=5$, $2262$), ill-conditioned, where $e_1$ dominates; `dry-bean` ($K=7$, $4764$), well-conditioned, with every part near the floor; `letter` ($K=26$, $7000$), large $K$ (chosen over `isolet`, where BayCAP-Hf diverges, E2.8 of [[ext2_baycap]], and `poker_hand`, which depends on FIX-8). Classifiers LR and RF.
- **Methods:** `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real`, with the settings of `main`.
- **Prevalence:** three fixed vectors per dataset: uniform $\mathbf 1/K$, one skewed vector (a seeded draw from $\operatorname{Dir}(\mathbf 1)$), one near-vertex vector ($0.9$ on the rarest validation class, the rest uniform).
- **Bags:** $R$ bags per cell, drawn with replacement from the test pool, as in `main` (bags larger than the pool repeat instances); the class counts are $\lvert U\rvert\pi_j$ rounded by largest remainder, the same for every bag of a prevalence; $R$ is fixed with the pilot (proposal: $R=30$). The bags depend on the dataset, the prevalence, $\lvert U\rvert$ and the bag index only, so every $\lvert V\rvert$ level and $V$ seed of a prevalence and size sees the same $R$ bags (user's decision of 2026-11-15): $e_3$ is then the same across the $V$ cells, and the differences between the fixed parts of two $V$ cells carry no bag-to-bag noise (§T.4, the caveat).
- **Held fixed:** the pool, $h$, the split $L/V$, the methods' settings.
- **Shared cell:** the cell with the full $V$ and $\lvert U\rvert=1000$ belongs to both parts and is run once.

### E4.2a: The parts across $\lvert U\rvert$

**Why.** With $V$ fixed, whatever changes with $\lvert U\rvert$ comes from the bag: the bag parts should shrink as the bag grows (§T.3), including the bag part of $e_2$ of BayCAP-KDEy, which comes from the weights $w$ of the realised-table predictive; its fixed part should move only through the error of the prevalence (§T.3, corrected after E4.2b). It shows which part dominates on small and on large bags.

- **Design.**
  - **Inputs:** the common settings; the full $V$ of each dataset.
  - **Varied:** $\lvert U\rvert\in\{100,300,1000,3000,10000\}$, including the paper's $1000$; prevalence; dataset, classifier, method. A cell is a combination of prevalence and $\lvert U\rvert$ (§T.4). Held fixed: $V$ (the full one, a single draw).
  - **Measures:** per bag, the parts $e_1$, $e_2$, $e_3$ (§T.2); per cell, the fixed part and the bag part of each (§T.4); against $\lvert U\rvert$, the RMS over the cells of the fixed parts and over the bags of the bag parts, and the fitted log–log slope of each RMS against $\lvert U\rvert$ (descriptive, with no predicted value).
  - **Estimand:** the realised table of each bag; the split uses the true prevalence of the bag and the labels of the pool (an analysis).
  - **Cost:** $D\times2\times5\times3\times R$ NUTS runs per method, for $D$ datasets; with $D=6$ and $R=30$, $5\,400$ runs per method. Reference times on `bcuda` at $\lvert U\rvert=1000$: a full run of `main` for `BayesCAP` ($7\,500$ NUTS runs) took about $46$ minutes ([[work_plan]] §I.2), one of `BayCAP-KDEy-Hf-mlp-real` ($6\,000$ runs) about $2$ hours (WP-13). The likelihood of BayCAP-Hf uses the counts of the predictions of $h$ on the bag, so its cost should barely depend on $\lvert U\rvert$; that of BayCAP-KDEy has one term per instance of the bag, so its cost should grow about linearly with $\lvert U\rvert$, about $10$ times at $\lvert U\rvert=10000$ (expectations, to be measured by the pilot).
- **Representation.**
  - **Log–log plot:** RMS of each part against $\lvert U\rvert$, one panel per method, one line per dataset (LR), solid for the fixed part and dashed for the bag part.
  - **Table:** per method and dataset, the RMS of the fixed and bag parts of $e_1$, $e_2$, $e_3$ at each $\lvert U\rvert$.
- **Analysis.** Not run.

### E4.2b: The parts across $\lvert V\rvert$

**Why.** E4.1 found on the binary datasets an offset of $e_2$ shared by every method, largest on the smallest validation sets, and ascribed it, as a hypothesis, to the finite $V$. The full $V$ cannot be drawn again without changing the split $L/V$, so the offset of the full $V$ cannot be re-measured on other validation sets; what can be measured is how the error that comes from $V$ grows as $V$ shrinks. A single subsample gives one draw of that error, which mixes the effect of the size of $V$ with the luck of that subsample; several seeds per level give its spread, and their mean should reproduce the full-$V$ value (§T.4, across $V$ seeds). If the spread at the full $V$, extrapolated from the subsamples, is comparable with the offset, the hypothesis is supported. The bag is held at the paper's $\lvert U\rvert=1000$, so the results are comparable with `main` and BayCAP-KDEy stays at its cost in `main`.

- **Design.**
  - **Inputs:** the common settings; nested seeded stratified subsamples of $V$ ($V_{1/16}\subset V_{1/4}\subset V$), with the methods refitted on each subsample and $h$ not retrained.
  - **Varied:** $\lvert V\rvert$ level: the full $V$ and subsamples of $1/4$ and $1/16$, each subsampled level only where every class keeps at least $10$ validation instances (the threshold of FIX-8, applied to both levels by the user's decision of 2026-11-15; the counts are checked before the pilot, `--counts`); $V$ seed: $S_V$ seeds for each subsampled level (the full $V$ has one); prevalence; dataset, classifier, method. A cell is a combination of prevalence, $\lvert V\rvert$ level and $V$ seed (§T.4). Held fixed: $\lvert U\rvert=1000$. $S_V$ is fixed with the pilot (proposal: $S_V=3$).
  - **Measures:** per bag, the parts $e_1$, $e_2$, $e_3$ (§T.2); per cell, the fixed part and the bag part of each (§T.4); per dataset, classifier, method, prevalence and subsampled level, the mean and the standard deviation over the $V$ seeds of each fixed part; against the $\lvert V\rvert$ level, the RMS over the cells of the fixed parts and over the bags of the bag parts, and the fitted log–log slope of the RMS of the fixed parts against $\lvert V\rvert$ (descriptive).
  - **Estimand:** as E4.2a.
  - **Cost:** $D\times2\times3\times S_V\times R$ NUTS runs per method for each subsampled level (the full-$V$ cell is shared with E4.2a); with $D=6$, $S_V=3$, $R=30$ and the level $1/4$ only, $3\,240$ runs per method, at the cost per run of `main`.
- **Representation.**
  - **Log–log plot:** RMS of the fixed part of each part against $\lvert V\rvert$, one panel per method, one line per dataset (LR), one point per seed and the line through the means over the seeds.
  - **Table:** per method, dataset and $\lvert V\rvert$ level, the RMS of the fixed and bag parts of $e_1$, $e_2$, $e_3$, and the mean and the standard deviation over the seeds of the fixed part of $e_2$, next to its value at the full $V$.
- **Analysis.** Source: full run on `bcuda` on 2026-11-15 from `598d404` (card WP-21; $R=30$, $S_V=3$, the proposals of the design, since the pilot was skipped by the user's decision); tables and plots from `14dcdee`: `<OUT_DIR>/uncawe/tables/target_size/` (`target_size_{levels,seeds,cells,slopes,recalls,drift,diagnostics}.csv`, table `target_size_levels_u1000`) and `<OUT_DIR>/uncawe/plots/target_size/v_size/` (LR and RF). The level $1/16$ runs on `spambase`, `dry-bean` and `letter` only (on the other three a class keeps fewer than $10$ instances). Values are RMS over the cells of a (method, pair, level) unless stated; "pair" is dataset and classifier.
  - **Checks.** $396$ cells ($198$ per method), each with $R=30$ bags and the same `config_hash`; the identity $e_1+e_2+e_3=\hat A-A$ holds within $7.3\cdot10^{-16}$; no bag is dropped and none has an empty estimated column; every bag has the same realised table in all the cells that share it. No NUTS run failed. BayCAP-Hf has at least one divergence on $477$ of $1080$ bags at the full $V$, $1360$ of $3240$ at $1/4$ and $752$ of $1620$ at $1/16$ (all bags of `mammographic` and `wine-quality` at the full $V$, as in `main`, where BayCAP-Hf also diverges on these pairs); BayCAP-KDEy on $4$ bags in all. As in `main`, divergent runs are kept, so the values of BayCAP-Hf carry this caveat. The standard error $s/\sqrt R$ of the fixed part of $e_2$ (§T.4) is at most $0.0052$ (BayCAP-KDEy, `wine-quality`, RF, $1/4$) and at most $0.0021$ in every other row, well below the fixed parts discussed below: $R=30$ is enough. As designed, the parts of $e_3$ do not change with the level (fixed part $0.0005$–$0.0024$, bag part $0.0057$–$0.0149$), since the bags are shared.
  - **The recall part grows as $V$ shrinks, for both methods.** The RMS of the fixed part of $e_2$ grows from the full $V$ to the smallest level in $21$ of the $24$ (method, pair) series (e.g. `letter`, LR: BayCAP-Hf $0.064$, $0.097$, $0.161$ at the full $V$, $1/4$, $1/16$; BayCAP-KDEy $0.015$, $0.048$, $0.115$; `dry-bean`, LR: BayCAP-Hf $0.0066$, $0.0159$, $0.0277$; BayCAP-KDEy $0.0049$, $0.0096$, $0.0159$). The median log–log slope against $\lvert V\rvert$ is $-0.37$ (BayCAP-Hf) and $-0.44$ (BayCAP-KDEy), over two or three levels (descriptive); a share counted on $n$ instances has a standard deviation proportional to $1/\sqrt n$, i.e. a slope of $-0.5$, and the slopes are flatter because the fixed part also contains the error of the full $V$ and of the pool, which do not shrink. The exceptions are `mammographic` with RF (BayCAP-Hf $+0.45$, BayCAP-KDEy $+0.04$) and `wine-quality` with RF for BayCAP-KDEy ($+0.23$, see below). The bag part of $e_2$ stays small (at most $0.0065$ for BayCAP-Hf, $0.028$ for BayCAP-KDEy), so on a small $V$ the error of the recalls is an offset shared by all the bags of a cell: on `letter` at $1/16$ (LR, BayCAP-Hf) the fixed part of the total error has RMS $0.173$, its bag part $0.014$.
  - **The binary offset of E4.1 is the gap between two finite samples.** On `mammographic` the recalls of $h$ on $V$ exceed those on the pool for both classes and both classifiers, and the gap, weighted by the prevalence of the bag ($\sum_j\pi_j(d^V_j-d^{\text{pool}}_j)$, `target_size_recalls.csv`), is $0.039$–$0.058$ over the three prevalences; the fixed part of $e_2$ at the full $V$ carries it ($0.051$–$0.057$ for BayCAP-KDEy, $0.034$–$0.052$ for BayCAP-Hf). If $V$ ($291$ instances) and the pool ($249$) were two independent samples of one population, the standard deviation of the gap would be $\sqrt{\sum_j\pi_j^2\bar d_j(1-\bar d_j)(1/n_j+1/m_j)}=0.033$–$0.045$, with $n_j$ and $m_j$ the instances of class $\omega_j$ in $V$ and in the pool and $\bar d_j$ its recall on both together (the variance of a difference of independent shares is the sum of their variances); so the gap is $0.9$–$1.7$ standard deviations, within what finite samples give. Over the $36$ rows (12 pairs, 3 prevalences) the ratio is at most $1.73$ in absolute value. The hypothesis of E4.1 holds, with one refinement: $e_2$ is measured against the pool, which on the small binary datasets is as small as $V$ (`german`: $350$ and $300$), so about half of the variance of the offset comes from the pool, not from $V$. The subsamples agree without adding precision: at $1/4$ the spread of the fixed part of $e_2$ over the three seeds is $0.011$–$0.039$ (BayCAP-KDEy) against $0.037$–$0.052$ from the formula of §T.4, and with three seeds a standard deviation is too imprecise to extrapolate to the full $V$.
  - **On a small $V$, shrinking $V$ also shifts $e_2$.** §T.4 predicts that, for recalls counted on $V'$, the mean over the seeds reproduces the full-$V$ value. On the three datasets that reach $1/16$ (`target_size_drift.csv`), the RMS of the drift of the mean over the seeds from the full-$V$ value is $0.020$ (BayCAP-KDEy) and $0.016$ (BayCAP-Hf) at $1/4$, close to the spread over the seeds ($0.015$, $0.014$) and to the formula of §T.4 ($0.018$); at $1/16$ it is $0.046$ for both, against a spread over the seeds of $0.016$ and $0.013$ and a formula of $0.040$. At $1/16$ the seeds agree with each other more than the formula predicts, and their mean moves away from the full-$V$ value: with $11$–$40$ validation instances in the smallest class the recalls of the methods are not the shares counted on $V'$, and the shift is systematic. The largest drift is on `letter` at the near-vertex prevalence ($0.9$ on the rarest class, $16$ instances at $1/16$): $+0.17$ for both methods with LR, $+0.08$ for BayCAP-KDEy with RF; on `spambase` it is $+0.007$ to $+0.024$ and on `dry-bean` $-0.024$ to $0.000$.
  - **The recalls of BayCAP-KDEy depend on the bag.** §T.3 places BayCAP-KDEy among the methods whose $\tilde d$ comes from $V$ alone, so that $e_2$ has no bag part. Its bag part is not $0$ (up to $0.028$, and larger than that of BayCAP-Hf in $28$ of the $30$ rows of the table, e.g. `letter` LR full $V$: $0.0067$ against $0.0035$). The cause is the realised-table predictive of `BayCAP-KDEy-mlp-real` (`methods/baycap.py:realised_tables`): it keeps the predicted counts of $h$ on the bag and redraws the true classes behind each predicted class with probabilities $\propto C_{ij}\pi_j$, so the recalls read from its mean table depend on the estimated prevalence and on the bag, not on $\hat C$ alone. Where the prevalence is well estimated the two coincide (`mammographic`: fixed part of $e_2$ $0.0574$ against a gap between $V$ and the pool of $0.0575$, LR, uniform); where it is not, $e_1$ and $e_2$ move together: on `wine-quality` with RF at the near-vertex prevalence and the full $V$, the fixed parts are $e_1=-0.409$ and $e_2=+0.569$ (total $0.161$), while the gap between $V$ and the pool is $-0.036$. On `german` the same mechanism explains why the offset of E4.1 is found for BayCAP-Hf only: at the near-vertex prevalence BayCAP-Hf carries the gap ($0.076$ and $0.060$ against $0.084$ and $0.069$, LR and RF) and BayCAP-KDEy does not ($0.003$, $-0.004$). The reading of §T.3 for BayCAP-KDEy held for its population table, not for the realised-table variant; §T.3 was corrected accordingly on 2026-11-15 (the realised-table predictive reweights the recalls of $\hat C$ by the ratio of the observed to the expected predictions of $h$).
  - **The prevalence part does not come from the finite $V$ where it is large.** On `wine-quality` with LR the RMS of the fixed part of $e_1$ is $0.214$ and $0.194$ (BayCAP-Hf, full $V$ and $1/4$), $0.096$ and $0.140$ (BayCAP-KDEy); on `german` it is flat for BayCAP-Hf ($0.055$ and $0.051$ with LR, $0.080$ and $0.080$ with RF) and grows for BayCAP-KDEy ($0.023$ to $0.056$, $0.032$ to $0.081$). Elsewhere it stays below $0.03$ at every level.

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
- **Analysis.** Source: the tables and the bar plots of `tables/error_split.py` and `plots/error_split.py`, run on `bcuda` on 2026-11-10 from commit `24b6776` (bar plots from `45a69b5`), on the stored results of `main` with `PQ` re-run with `cap-ml` `87b4d9b`; the per-pair values are in `<OUT_DIR>/uncawe/tables/error_split/error_split.csv`. Values below are means over the pairs of a block ($25$ binary pairs, $50$ multiclass pairs) of per-pair quantities over the $100$ bags. After FIX-2 + FIX-3 (`QuAcc_NxN`, `O-LEAP`, `PQ` and `RQBS`, renamed `SQS-Hbs`, re-run with seeded quantifiers; `SQS-H` added) and FIX-8 (`poker_hand` with $K=7$, every method re-run on it), the tables and bar plots were generated again on 2026-11-13 from `50f17df`; the numbers below are updated, with the value before these fixes in brackets where it changed ("before"). The readings do not change.
  - **Mean AE against the realised accuracy, per cell** (estimated $(\tilde\pi,\tilde d)$; true prevalence $(\pi^Q,\tilde d)$; pool recalls $(\tilde\pi,d^{\text{pool}})$; both $(\pi^Q,d^{\text{pool}})$):

    | block | method | $(\tilde\pi,\tilde d)$ | $(\pi^Q,\tilde d)$ | $(\tilde\pi,d^{\text{pool}})$ | $(\pi^Q,d^{\text{pool}})$ |
    |---|---|---|---|---|---|
    | binary | `O-LEAP` | 0.0277 | 0.0252 | 0.0115 | 0.0077 |
    | binary | `SQS-H` | 0.0305 | 0.0279 | 0.0118 | 0.0077 |
    | binary | `SQS-Hbs` | 0.0305 | 0.0279 | 0.0117 | 0.0077 |
    | binary | `PQ` | 0.0311 | 0.0287 | 0.0124 | 0.0077 |
    | binary | `BayesCAP-real` | 0.0390 | 0.0243 | 0.0241 | 0.0077 |
    | binary | `BayesCAP-Hf-real` | 0.0394 | 0.0248 | 0.0240 | 0.0077 |
    | binary | BayCAP-KDEy | 0.0238 | 0.0193 | 0.0119 | 0.0077 |
    | binary | `QuAcc_NxN` | 0.0667 | 0.0860 | 0.0424 | 0.0077 |
    | multiclass | `O-LEAP` | 0.0333 | 0.0178 | 0.0280 | 0.0070 |
    | multiclass | `SQS-H` | 0.0318 | 0.0107 | 0.0266 | 0.0070 |
    | multiclass | `SQS-Hbs` | 0.0318 | 0.0109 | 0.0267 | 0.0070 |
    | multiclass | `PQ` | 0.0346 | 0.0158 | 0.0283 | 0.0070 |
    | multiclass | `BayesCAP-real` | 0.0662 | 0.0358 | 0.0371 | 0.0070 |
    | multiclass | `BayesCAP-Hf-real` | 0.0338 | 0.0101 | 0.0305 | 0.0070 |
    | multiclass | BayCAP-KDEy | 0.0299 | 0.0104 | 0.0266 | 0.0070 |
    | multiclass | `QuAcc_NxN` | 0.2108 | 0.2249 | 0.1150 | 0.0070 |

    Before the fixes: binary `O-LEAP` $0.0268$, $0.0254$, $0.0111$; `RQBS` $0.0298$, $0.0278$, $0.0115$; `PQ` $0.0304$, $0.0287$, $0.0122$; `QuAcc_NxN` $0.0732$, $0.0999$, $0.0424$; multiclass `O-LEAP` $0.0316$, $0.0139$, $0.0261$; `RQBS` $0.0255$, $0.0103$, $0.0225$; `PQ` $0.0352$, $0.0156$, $0.0285$; `BayesCAP-real` $0.0656$, $0.0357$, $0.0365$; `BayesCAP-Hf-real` $0.0329$, $0.0100$, $0.0296$; BayCAP-KDEy $0.0248$, $0.0103$, $0.0222$; `QuAcc_NxN` $0.2076$, $0.2162$, $0.1158$ (first three cells; the binary BayCAP rows and the cell with both oracles are unchanged).

  - **The bottleneck changes with the block.** On binary datasets the pool recalls cut the AE by $50$–$61\%$ for every method except the BayesCAP variants (e.g. `SQS-Hbs` $0.0305\to0.0117$) while the true prevalence barely helps ($0.0305\to0.0279$); on multiclass datasets it is the reverse: the true prevalence brings the best methods to about $0.010$ (`SQS-H` $0.0318\to0.0107$, `SQS-Hbs` $0.0318\to0.0109$, BayCAP-KDEy $0.0299\to0.0104$, `BayesCAP-Hf-real` $0.0338\to0.0101$; before, `RQBS` $0.0298\to0.0115$ and $0.0278$, $0.0255\to0.0103$, BayCAP-KDEy $0.0248\to0.0103$, `BayesCAP-Hf-real` $0.0329\to0.0100$), close to the floor $0.0070$ of the cell with both oracles, while the pool recalls save little. This agrees with E4.1: on binary datasets the room for improvement is in the recalls measured on $V$, on multiclass datasets in the quantifier.
  - **The BayesCAP variants on binary datasets** gain about equally from either oracle ($0.039\to0.024$ with each): their prior affects both components.
  - **The two components barely interact, except in `QuAcc_NxN`.** For every other method $\lvert\text{mean of }x\rvert\le0.0042$, so substituting one component does not change the error due to the other. For `QuAcc_NxN` the true prevalence makes the estimate worse ($0.067\to0.086$ binary, $0.211\to0.225$ multiclass; before, $0.073\to0.100$ and $0.208\to0.216$): its prevalence error offsets part of its recall error (E4.1).
  - **Against the pool accuracy**, which removes $e_3$, the AE of the estimated cell is at most $0.002$ lower than against the realised accuracy (e.g. `SQS-Hbs` binary $0.0291$ against $0.0305$; multiclass $0.0298$ against $0.0318$; before, `RQBS` $0.0282$ against $0.0298$ and $0.0234$ against $0.0255$): in aggregate the luck of the bag is small next to the errors of the methods, and matters only where these are already small (E4.1).

# Candidate experiments

Ideas that came up in the discussions; they are not scheduled. One becomes an experiment of the main section only when the user decides so.

- **E4.c11, oracles fed into the methods.** *Why:* E4.3 substitutes the true components after estimation, which leaves the other component of each estimate unchanged; a method that receives the true prevalence or the pool recalls as input may change its other component too (with $\pi^Q$ known, the posterior of $C$ of BayCAP-Hf changes), and only a run with the oracle inside the method gives an interval, hence a coverage. *Needs:* oracle variants of the methods (true $\pi^Q$; recalls fixed at $d^{\text{pool}}$), run on the protocol of `main`; to limit the cost, a few methods only (user's remark of 2026-11-09), e.g. `RQBS` and one Bayesian method.
- **E4.c12, tightness of the bounds.** *Why:* the bounds of §T.6 hold always; their ratio to the parts they bound says whether the error of the quantifier times the spread of the recalls (for $e_1$), or the product of the two errors (for the cross term $x$), is a good proxy of the error of accuracy, and how much of a part comes from the size of the per-class errors rather than from their arrangement across the classes. It is the former E4.1c, extended to $x$, $e_2$ and $e_3$. *Needs:* the per-bag quantities of E4.1 and E4.3 ($\tilde\pi$, $\tilde d$, $\pi^Q$, $d^{\text{pool}}$, $c_j/n_j$), no runs; per bag and method, each part, its bound and their ratio; per pair, the spread of the recalls $\max\tilde d-\min\tilde d$ and the median ratios; across pairs, the rank correlation between the accuracy error and the quantifier's AE, with and without the factor of the spread of the recalls.
- **E4.c13, interaction of $\lvert U\rvert$ and $\lvert V\rvert$.** *Why:* E4.2a varies $\lvert U\rvert$ with the full $V$ and E4.2b varies $\lvert V\rvert$ at $\lvert U\rvert=1000$, so neither sees whether the error that comes from $V$ depends on the size of the bag. For BayCAP-KDEy it does not for $e_2$, whose recalls depend on $V$ only (§T.3); for BayCAP-Hf it may, since its recalls are updated by the counts of the bag, which grow with $\lvert U\rvert$ while the information from $V$ stays the same, so a large bag may partly make up for a small $V$; $e_1$ depends on both for both methods. *Needs:* the cells of E4.2b at the subsampled level, with the same $V$ seeds, at one more size, e.g. $\lvert U\rvert=10000$: $3\,240$ more runs per method with the proposal of E4.2 ($D=6$, $S_V=3$, $R=30$), cheap for BayCAP-Hf and about $10$ times the cost per run for BayCAP-KDEy, so possibly BayCAP-Hf only (discussion of 2026-11-11).

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

Then $\tilde\pi_j\tilde d_j=\hat T_{jj}$, so $\hat A=\sum_j\tilde\pi_j\tilde d_j=\operatorname{tr}\hat T$: the estimate of accuracy is exactly the sum of the products. This holds for every method of the main table that estimates a table: the SQS methods (`RQBS`, `PQ`), whose $\tilde d$ is measured on $V$ (for `PQ`, $\hat T$ is the mean of the tables of the $M$ candidate bags of $V_2$ it selects, all of the same size, so $\operatorname{tr}\hat T=\frac1M\sum_m\operatorname{tr}T_m$, the mean accuracy of the selected bags, which is its estimate; `cap-ml` stores this table from `87b4d9b`, version 0.1.1.10); the Bayesian methods, whose $\hat T$ is the mean of their table draws; `O-LEAP` and `QuAcc_NxN`, whose $\hat T$ is their own estimate. It does not hold for `DoC`, which estimates accuracy without a table. (In the code, `estim_cts` stores the transpose, true classes on rows, so $\tilde\pi$ is read from its row sums.)

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

- `SQS-H`, `SQS-Hbs`, `PQ` and the population table of BayCAP-KDEy (`BayCAP-KDEy-mlp`, fixed $\hat C$) take $\tilde d$ from $V$ alone; given $V$ and the prevalence of the bag, $e_2$ is fixed: it does not change from bag to bag and does not shrink with $\lvert U\rvert$. It depends on $\lvert V\rvert$, and on the pool through $d^{\text{pool}}$ (§T.1). For `PQ` the prevalence $\tilde\pi$ is the mean true prevalence of the selected bags, not the estimate of its quantifier, so its $e_1$ measures the error of the selection of the bags; its $\tilde d$ is pooled over these bags, drawn from $V_2$, half of $V$;
- `BayCAP-KDEy-mlp-real`, the realised-table predictive of BayCAP-KDEy, keeps $\hat C$ fixed but reads recalls that also depend on the bag and on the estimated prevalence (below, *the realised-table predictive*): $e_2$ has a bag part and moves with $e_1$ (corrected on 2026-11-15, after E4.2b; before, this method was listed with the previous ones);
- `BayesCAP-real` and `BayesCAP-Hf-real` estimate the recalls jointly with the prevalence, from the counts of $V$ and the predictions of $h$ on the bag, so $\tilde d$ also depends on $U$, and on the prior of $C$ (a systematic pull, [[ext2_baycap|Extension 2]]); being realised-table predictives too, their recalls are also reweighted as below;
- `O-LEAP` and `QuAcc_NxN` estimate the table with their own models, fitted on $V$ and applied to the bag; for them $\tilde d$ is not a share measured on $V$, and the sources of $e_2$ are measured, not predicted.

**The realised-table predictive.** *Intuition.* A realised table must contain exactly the predictions that $h$ made on the bag. When the estimated prevalence and $\hat C$ predict fewer predictions of some class than $h$ actually made, the predictive pushes more of the table into that predicted class; the recalls read from the table change accordingly, even if $\hat C$ is held fixed.

*Model.* Take one posterior draw, with prevalence $\pi$ and the fixed rate matrix $\hat C$ (for BayCAP-KDEy, the rates of $h$ on $V$). Let $q_i=n^t_i/N$ be the share of the $N$ instances of the bag that $h$ predicts as $\omega_i$ (observed), and $p_i=\sum_l\hat C_{il}\pi_l$ the share that the draw expects (the probability of predicting $\omega_i$ under the draw). The predictive (`methods/baycap.py:realised_tables`) keeps the $n^t_i$ instances predicted as $\omega_i$ and draws their true classes as $\operatorname{Mult}(n^t_i,\rho_{i\cdot})$, with $\rho_{ij}=\hat C_{ij}\pi_j/p_i$, the probability that an instance predicted as $\omega_i$ is of class $\omega_j$ under the draw (Bayes' rule).

*Mean table of the draw.* The mean of a multinomial count is the number of trials times the probability, so the mean share of the bag predicted as $\omega_i$ and of class $\omega_j$ is

$$
\mathbb E[T_{ij}]=q_i\rho_{ij}=\hat C_{ij}\pi_j\,w_i,\qquad w_i=\frac{q_i}{p_i},
$$

where $w_i$ compares the observed share of predictions of $\omega_i$ with the expected one ($w_i>1$: $h$ predicted $\omega_i$ more often than the draw expects).

*Prevalence and recall read from it* (§T.2: column sum and diagonal over column sum):

$$
\tilde\pi_j=\sum_i\hat C_{ij}\pi_jw_i=\pi_j\sum_i\hat C_{ij}w_i,\qquad \tilde d_j=\frac{\hat C_{jj}\pi_jw_j}{\pi_j\sum_i\hat C_{ij}w_i}=\frac{\hat C_{jj}w_j}{\sum_i\hat C_{ij}w_i}.
$$

*When it reduces to $\hat C$.* The entries of column $j$ of $\hat C$ are the probabilities of the $K$ predictions for an instance of class $\omega_j$, so $\sum_i\hat C_{ij}=1$. If every $w_i$ equals the same number $c$, then $\sum_i\hat C_{ij}w_i=c$ and $\tilde d_j=\hat C_{jj}c/c=\hat C_{jj}$: the recall is the share counted on $V$, and $\tilde\pi_j=c\,\pi_j$ with $c=1$ since both $q$ and $p$ sum to $1$. So the realised predictive reads the recalls of $V$ exactly when the observed predictions match those that the draw expects. Otherwise $\tilde d_j$ is $\hat C_{jj}$ reweighted: it rises when $w_j$ is larger than the average of the $w_i$ over the predicted classes where the instances of $\omega_j$ go, and falls when it is smaller.

*Why the weights differ from $1$.* (a) The bag is finite: $q$ fluctuates from bag to bag around its expected value, so $\tilde d$ has a bag part that shrinks as $\lvert U\rvert$ grows. (b) The prevalence of the draw is wrong: $p$ then differs from the predicted shares of the bag even on an infinite bag, and the same error that makes $e_1$ moves $\tilde d$, so $e_1$ and $e_2$ move together. (c) $\hat C$ differs from the rates on the pool, which also shifts $p$. The stored mean table averages $\mathbb E[T_{ij}]$ over the draws (plus the noise of the multinomial draws, of mean $0$), so $\tilde d_j$ has the same form with the products $\pi_jw_i$ averaged over the draws. E4.2b found all three effects (Analysis): a bag part of $e_2$ of BayCAP-KDEy up to $0.028$, values of $e_2$ equal to the gap between $V$ and the pool where the prevalence is well estimated (`mammographic`), and opposite fixed parts of $e_1$ and $e_2$ where it is not (`wine-quality`, RF).

**The prevalences, $e_1$.** Every method estimates $\tilde\pi$ from the bag with a model fitted on $V$, so $e_1$ has a part due to the finite bag, which gets smaller as $\lvert U\rvert$ grows, and a part due to the finite $V$ (and to any systematic bias of the quantifier), which does not.

| Part | `SQS-H`, `SQS-Hbs`, `PQ`, BayCAP-KDEy (population) | BayCAP-KDEy (realised) | BayCAP, BayCAP-Hf (realised) | `O-LEAP`, `QuAcc_NxN` |
|---|---|---|---|---|
| $e_1$ | $U$ and $V$ | $U$ and $V$ | $U$, $V$ and prior | $U$ and $V$ |
| $e_2$ | $V$ (and pool) | $V$, and $U$ and $\tilde\pi$ through $w$ (and pool) | $U$, $V$, prior, $w$ (and pool) | measured |
| $e_3$ | $U$ | $U$ | $U$ | $U$ |

E4.2 separates the contributions of $U$ and $V$ for two methods that differ in the source of their recalls: BayCAP-KDEy, whose $\hat C$ is fixed at the rates of $V$, and BayCAP-Hf, which updates $C$ with the bag; both are taken with the realised-table predictive, so the recalls of both are reweighted by $w$.

## T.4 Fixed part and bag part within a cell

*Intuition.* If many bags are drawn with the same prevalence, the same size and the same $V$, whatever their errors have in common comes from $V$ (and the pool), and whatever differs between them comes from the bags.

**Definition.** In E4.2, a *cell* is a combination of prevalence, $\lvert U\rvert$, $\lvert V\rvert$ level and $V$ seed, with $R$ bags $b=1,\dots,R$ (in E4.2a the $V$ is always the full one; in E4.2b $\lvert U\rvert=1000$). For any part $e$ (one of $e_1$, $e_2$, $e_3$) with value $e^{(b)}$ on bag $b$, define its mean over the cell and the deviation of each bag from that mean:

$$
\bar e=\frac1R\sum_{b=1}^Re^{(b)},\qquad e^{(b)}=\underbrace{\bar e}_{\text{fixed part}}+\underbrace{\big(e^{(b)}-\bar e\big)}_{\text{bag part}}.
$$

The split is exact (add and subtract $\bar e$), and the bag parts of a cell sum to $0$ by construction, since $\sum_b(e^{(b)}-\bar e)=R\bar e-R\bar e=0$.

**Reading.** The fixed part is shared by the bags of a cell: it varies only across cells, i.e. across $V$ seeds, prevalences and sizes. The bag part measures how much the part changes from one bag to another with everything else held fixed. From §T.3: the fixed part of $e_3$ should be close to $0$ and its bag part should shrink as $\lvert U\rvert$ grows; for the population table of BayCAP-KDEy, $e_2$ would have no bag part at all (its $\tilde d$ is fixed given $V$) and a fixed part that does not change with $\lvert U\rvert$; with the realised-table predictive, used in E4.2, $e_2$ of BayCAP-KDEy has a bag part through the weights $w$ (§T.3), which shrinks as $\lvert U\rvert$ grows, and a fixed part that moves with the error of the prevalence; for BayCAP-Hf, $e_2$ has a bag part also because its recalls are updated by each bag.

**Across $V$ seeds (E4.2b).** *Intuition.* A subsample of $V$ inherits whatever the full $V$ gets wrong, and adds an error of its own that changes from one subsample to another; drawing several subsamples separates the two.

Take a method whose recalls are shares counted on the validation set, as the population table of BayCAP-KDEy with its fixed $\hat C$ (§T.3; with the realised-table predictive the shares are reweighted by $w$, so what follows is a reference, not a prediction, for the methods of E4.2). Write $\tilde d^V_j$ for the recall of class $\omega_j$ on the full $V$, with $n_j$ validation instances of that class, and $\tilde d^{V'}_j$ for the recall on a stratified subsample $V'$ with $n'_j$ of them. Then:

1. *Split.* Adding and subtracting $\tilde d^V_j$,
$$
\tilde d^{V'}_j-d^{\text{pool}}_j=\underbrace{\big(\tilde d^{V'}_j-\tilde d^V_j\big)}_{\text{error of the subsample}}+\underbrace{\big(\tilde d^V_j-d^{\text{pool}}_j\big)}_{\text{error of the full }V}.
$$
2. *The mean over subsamples.* Given $V$, the $n'_j$ instances of class $\omega_j$ are drawn without replacement from its $n_j$ instances, each instance with the same chance. A share counted on a sample drawn this way has expected value equal to the share in the set it is drawn from (each instance is in the sample with probability $n'_j/n_j$, so by linearity of the expectation the expected number of correct ones is $(n'_j/n_j)$ times the correct ones in $V$). So $\mathbb E\big[\tilde d^{V'}_j\mid V\big]=\tilde d^V_j$: the error of the subsample has mean $0$, and the mean over many seeds of the recall error is the error of the full $V$.
3. *The spread over subsamples.* The variance of a share counted on a sample drawn without replacement is that of the binomial times the finite-population correction:
$$
\operatorname{Var}\big(\tilde d^{V'}_j\mid V\big)=\frac{\tilde d^V_j\big(1-\tilde d^V_j\big)}{n'_j}\cdot\frac{n_j-n'_j}{n_j-1}.
$$
The first factor is the variance of the share of successes in $n'_j$ independent trials; the second shrinks it because, without replacement, the sample cannot stray far from the set it exhausts (it is $0$ when $n'_j=n_j$).

So, in E4.2b, the mean over the $V$ seeds of the fixed part of $e_2$ estimates the fixed part at the full $V$, up to the error of averaging $S_V$ seeds, and the standard deviation over the seeds measures the error that a validation set of size $n'_j$ adds. One seed gives one draw of that error, which cannot be told apart from the luck of that subsample. For BayCAP-Hf ($\tilde d$ also depends on the bag and on the prior) and for $e_1$ (the quantifier is refitted on $V'$) there is no such formula, and the seeds measure the spread directly.

**A caveat.** $\bar e$ is the mean of $R$ bags, not of infinitely many, so the fixed part carries a share of the bag-to-bag noise, which gets smaller as $R$ grows; the fixed part of a part with a large bag part is therefore measured less precisely. Its standard error is $s/\sqrt R$, with $s$ the standard deviation of the $R$ bag parts of the cell: the mean of $R$ values has the variance of one value divided by $R$ (the variance of a sum of independent terms is the sum of their variances, and dividing by $R$ divides the variance by $R^2$). In E4.2 the cells of a prevalence and size that differ only in $\lvert V\rvert$ level or $V$ seed share the same $R$ bags (common settings). Then $e_3$, which uses no estimate, has the same fixed part in all of them, and for a part whose value on a bag depends on $V$ only (as $e_2$ of the population table of BayCAP-KDEy) the difference between two $V$ cells contains no bag-to-bag noise at all; for the other parts, the share of bag noise that the $R$ bags put in $\bar e$ is partly common to the $V$ cells, so it inflates the spread over the seeds less than independent bags would.

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
- E4.2a and E4.2b: a new study `target_size`, with two grids (E4.2a: $\lvert U\rvert$ with the full $V$; E4.2b: $\lvert V\rvert$ level and $V$ seed at $\lvert U\rvert=1000$; the shared cell run once), with bags of fixed prevalence at any $\lvert U\rvert$ drawn with replacement from the pool (per-bag seeds from the dataset, prevalence, $\lvert U\rvert$ and bag index, shared by every $V$ cell), nested seeded stratified subsampling of $V$ with the methods refitted on each subsample and $h$ not retrained, and the methods `BayesCAP-Hf-real` and `BayCAP-KDEy-mlp-real` built as in `main`. Implemented for E4.2b by WP-21 (`experiments/target_size.py`, any $\lvert U\rvert$ through `--sizes`; tables `tables/target_size.py`, plot `plots/target_size.py`; [[work_plan]] §III.2).

Tests:

- $e_1+e_2+e_3=\hat A-A(h,U)$ exactly on random tables;
- with $\tilde\pi=\pi^Q$ and $\tilde d=d^{\text{pool}}$, $e_1=e_2=0$; the four oracle errors equal $e_1+e_2+e_3$, $e_2+e_3$, $e_1-x+e_3$ and $e_3$;
- reading $\tilde\pi$ and $\tilde d$ from a stored (transposed) table gives the column sums and recalls of the paper's orientation;
- the bag parts of a cell sum to $0$;
- the fixed-prevalence sampler returns bags with the requested class counts and size; nested subsamples satisfy $V_{1/16}\subset V_{1/4}\subset V$ and are stratified.

Reproducibility and comparability:

- E4.1 and E4.3 read stored results only and change no result; they run after the gate (FIX-1), on the `main` results of the pinned `cap-ml`;
- E4.2a and E4.2b write to `cap.env["OUT_DIR"]/uncawe/target_size`; `main` and its outputs are unchanged; every row records $\lvert U\rvert$, the $\lvert V\rvert$ level, the $V$ seed, the prevalence, the bag seed, `config_hash` and `schema_version`; its bags have fixed prevalences rather than APP ones, so its cells at $\lvert U\rvert=1000$ and the full $V$ are comparable with `main` in distribution, not bag by bag.

# Appendix

## A. Relation to other extensions

- **[[ext2_baycap|Extension 2]]**: its estimand gap is the term $e_3$; the systematic pull of the prior of $C$ shows up in the mean of $e_2$ for BayCAP and BayCAP-Hf. Its cause 4 (the error of the pool) is merged into $e_2$ here (§T.1); its separate study is set aside ([[optional_extensions]] §10).
- **[[ext3_posterior_propagation|Extension 3]]**: BayCAP-KDEy (fixed $\hat C$, MLP input, realised pushforward), its best method of E3.1, is one of the methods of E4.1–E4.3, with the implementation of E3.1.
- **Extension 6** ([[extensions]] §6): varies the allocation $\rho$ at fixed $\lvert V\rvert$; here $\lvert V\rvert$ is subsampled at fixed allocation.

## B. Out of scope

- the variance model of the parts and everything built on it (coverage rule, regime map, width floors, plug-ins, completed and guaranteed intervals), the parts of BayCAP with joint estimation, the pool term and the protocol without replacement, and the toy simulation of the budget: set aside on 2026-11-09 ([[optional_extensions]] §10);
- the methods not in the main table (other BayCAP priors, population pushforwards, the other methods of Extension 3, `L-CBPE`): they may be shown in separate experiments, not here;
- violations of PPS ([[optional_extensions]] §2); the validation allocation $\rho$ (Extension 6).
