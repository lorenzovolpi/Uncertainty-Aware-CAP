Before deciding anything about Ext. 6, there is a problem: I think the stored 6.1 results can't answer the distance question, because SQS-S compares the target and the validation bags with two different estimators. I found this by reading the code and checked it against one plot, but haven't measured how big it is.

## 6.1: what exists and what I found

**What is stored.** `predist` ran on 2026-05-20, and its results are on bcuda in `uncawe-devel/output/uncawe/predist/`.
- **Setup:** LR on the eight 3-class datasets, with KDEy(MLP) as the quantifier.
- **Bags:** 5 target bags per dataset, of size $|U|=1000$, at fixed points of the simplex (centre, the three corners, one more point).
- **Distances:** $L_1$, Hellinger and Jensen–Shannon, all on the same fitted `PrediQuant`.

That gives $8\times5=40$ bags per distance. Only per-bag ternary plots were produced; no summary table exists. So "completing" it would mean one table per distance (absolute error, coverage, width, Winkler score). With 40 bags at hand-picked points, that table could only show large differences.

**The problem (suspected bug, verified by code reading only).** In cap-ml, `PrediQuant.fit` runs:

- for each candidate validation bag $\sigma_b$: `sigma_pred_prevs = q.aggregate(P_i)`, where `P_i` are the posteriors of the classifier $h$ under evaluation;
- for the target: `q.predict(X)`, which uses the quantifier's own MLP.

The quantifier was fitted with its own MLP, not with $h$: `KDEyML(MLP())` in `predist`, `PACC(MLP())` for `PQ` in `main`. Without `reuse_h`, its adjustment parameters belong to the MLP's posterior space. So:

$$
\hat\pi_b=\text{aggregate}_{\text{MLP-fitted}}\big(h(\sigma_b)\big),\qquad \hat\pi^Q=\text{aggregate}_{\text{MLP-fitted}}\big(\text{MLP}(U)\big).
$$

These two estimators have different biases. The nearest-neighbour step $d(\hat\pi_b,\hat\pi^Q)\le$ threshold then compares numbers that live in different spaces.

The connect-4 plot for sample 01 (target near the $p_0$ corner) shows exactly this. The *predicted* prevalences of the selected bags sit on the target. Their *true* prevalences are spread around $(0.3,0.5,0.2)$, far from the corner. Result: coverage 0 and absolute error about $0.44$–$0.50$ for all three distances. On waveform-v1 at the centre everything looks fine, probably because LR and the MLP agree there.

As long as this is in place, choosing among distances means choosing how to measure in the wrong space. The same problem would affect `PQ` in `main`, and so the paper's SQS-S numbers.

**Proposal for 6.1, kept thin:**
1. Check the problem first. A cheap numerical test: on the same bags, compare `q.aggregate(h posteriors)` with `q.predict(X)`. I won't run it until you ask.
2. If it's confirmed, treat it as a new Part IV fix next to FIX-1. Both change the same `fit` method, so they would naturally be one fix.
3. After the fix, re-run `predist` (cheap), extended to all five classifiers and with one summary table. Decision rule: keep $L_1$ unless another distance is clearly better.
4. Move kernel weights, adaptive bandwidth, Aitchison/ilr and $\gamma$ sensitivity to `optional_extensions.md`.

## 6.2: core experiment and side aspects to decide

The core: vary only $\rho$, with $|V_q|=\rho|V|$ and $|V_b|=(1-\rho)|V|$. Hold fixed $|V|$, $|U|$, the target bags, the $n=10{,}000$ candidates and the $m=1000$ selected bags. Score against the realised table (`true_cts`), as `main` does. SQS-S draws bags of size $|U|$, so it naturally targets realised accuracy.

| # | Side aspect | My recommendation |
|---|---|---|
| a | FIX-1, removing the overlap $V_q\cap V_b$ | **Required.** Without it, $\rho$ means nothing |
| b | The mismatch above: what does the quantifier fit on $V_q$? (i) reuse $h$, so only the adjustment is fitted on $V_q$; or (ii) its own MLP, trained on $V_q$ | **Must decide.** It changes what $\rho$ trades off: (i) only the estimate of the adjustment parameters, (ii) also the MLP's training. I lean to (i): it's consistent by construction and needs no extra classifier |
| c | Grid for $\rho$ | $\{0.1,0.3,0.5,0.7,0.9\}$, with $0.5$ the paper's baseline |
| d | Repeated stratified splits (outer seeds) | **Include**, about 5 seeds; needed to separate the effect of $\rho$ from the particular split |
| e | Measures | Keep: quantification error on $U$; point error, coverage and width. Keep one diagnostic: distinct instances per class in the selected bags. Drop: training and inference cost, and the separate proximity in estimated/true space (the class-support diagnostic and the quantification error already cover it) |
| f | Quantifier | PACC(MLP), as in `main`, for comparability |
| g | Datasets and classifiers | The `main` protocol. Pilot on a subset |
| h | Cross-fitting (swap $V_q$ and $V_b$, pool the bags) | Candidate, E6.c1. It could remove the trade-off, but it's a new method |
| i | $\rho\times|V|\times|U|$ factorial | Move to optional; Ext. 4 already varies $|V|$ at fixed $\rho$ |
| j | Neyman allocation across classes, $|V_j|\propto\sqrt{d_j(1-d_j)}$ | Move to optional; it changes how $V$ is built, so it is a different protocol |
| k | Label-free selection of $\rho$ | Move to optional. The experiment is descriptive; if one $\rho$ wins, adopt it |

That leaves Ext. 6 with one fix (FIX-1 plus the mismatch), one small 6.1 re-run with a decision rule, and one 6.2 experiment.

## Decisions needed from you

1. Should I run the numerical check of the mismatch now?
2. For (b): should the quantifier reuse $h$, or keep its own MLP?
3. Do you agree to set aside 6.1's kernel, bandwidth, Aitchison and $\gamma$ items, and 6.2's (h)–(k)?

I haven't edited the repository. I saved the problem and your "keep it thin" direction to session memory.