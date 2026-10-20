# UNCAWE work plan

This document records the state of the UNCAWE implementation and what to implement next. It is not a schedule of the extensions: everything in [[extensions]] is a research proposal, and an extension enters this plan only when the user adds a corresponding card to the Part III board. The document is the tracker of all implementation and experimentation work of the project.

The document has five parts:

- **Part I:** what is implemented and reported in the [reference paper](papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf);
- **Part II:** what is implemented in the code but not reported in the paper;
- **Part III:** the work tracker: a board of scheduled, in-progress and implemented-but-not-evaluated work, chosen by the user;
- **Part IV:** pending fixes and known divergences between the paper and the code, which are prerequisites rather than next steps;
- **Part V:** implementation conventions for experiments, methods, results, tables and plots.

Status labels: **[paper]** implemented and reported in the paper; **[code-only]** implemented, not in the paper; **[fix]** pending correction. Work in progress is tracked by the statuses of the Part III board. Code paths are relative to `src/uncawe/` unless stated otherwise. Code names and paper names differ; the mapping is in §I.2.

Notation follows the paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ and $T=C\operatorname{diag}(\pi)$, with predicted classes on rows and true classes on columns. `cap-ml` and the code store the transpose (true classes on rows).

# Part I: Implemented and reported in the paper

Everything in this part is **[paper]**. It is produced by the `main` experiment (`experiments/main.py`, entry point `uncawe.main`).

## I.1 Protocol

- **Datasets.** UCI datasets accessible through QuaPy: the 5 largest binary datasets among a list of native binary datasets and the 10 largest multiclass datasets (`experiments/datasets.py:get_filtered_datasets`).
- **Splits.** Each dataset $D$ is split with stratification into a training part (70%) and a test pool $U$ (30%); the training part is split again into $L$ (50%), used to train the classifier $h$, and $V$ (50%), used to train the CAP methods. The splits come from the `cap-ml` fetchers (`fetch_UCIBinaryDataset`, `fetch_UCIMulticlassDataset`) and are shared by all methods.
- **Classifiers.** LR, kNN, SVM (RBF kernel), MLP and RF from scikit-learn with default hyperparameters (`pretrain/classic.py`). Their posteriors on $V$ and $U$ are computed once and stored under `<CAP_DATA>/pretrain/<domain>/` (`pretrain/data.py:PretrainInfo`).
- **Target bags.** 100 bags with $\lvert U_i\rvert=1000$, drawn with replacement from $U$ at prevalence vectors sampled uniformly from the simplex. The paper calls this APP; the code uses QuaPy's `UPP` with `random_state=qp.environ["_R_SEED"]=0` (`pretrain/data.py:DatasetBundle.with_test_data`).
- **Estimand.** Every method is scored against the realised contingency table of each target bag (`true_cts`), not against the population table $C\operatorname{diag}(\pi^Q)$.

## I.2 Methods

| Code name (`experiments/methods.py`) | Paper name | Class | Interval | Configuration |
|---|---|---|---|---|
| `DoC` | DoC | `BootstrapDirectCAP(DoC)` | bootstrap of the target bag | 1000 bootstrap draws |
| `QuAcc_NxN` | QuAcc (QuAcc-K×K) | `BootstrapCTCAP(QuAccNxN)` | bootstrap of the target bag | KDEy(MLP); extra features `add_maxconf`, `add_negentropy`, `add_maxinfsoft`; 1000 draws |
| `O-LEAP` | LEAP (O-LEAP) | `BootstrapCTCAP(O_LEAP)` | bootstrap of the target bag | KDEy(MLP); 1000 draws |
| `L-CBPE` | L-CBPE | `CBPE(calib_method="lascal")` | Poisson-binomial | binary datasets only |
| `RQBS` | SQS-H | `RQBS` | percentile of validation bags | PACC(MLP) surrogate; $m=1000$ bags of size 1000 |
| `PQ` | SQS-S | `PrediQuant` | percentile of selected validation bags | PACC(MLP); $n=10{,}000$ candidates, $\gamma=0.1$, $L_1$ distance; $m=1000$ |
| `BayesCAP` | BayCAP | `ACC_BayesCAP` | posterior credible interval | NUTS, 500 warm-up iterations, 1000 draws, uniform Dirichlet priors |

All methods return 1000 performance samples per bag. The point estimate is their mean, and the interval is the empirical $[\alpha/2,1-\alpha/2]$ percentile interval with $1-\alpha=0.95$ (`CONFIDENCE_LEVEL`). The table above describes the code; where it differs from the paper's definitions is listed in Part IV (FIX-1 and FIX-2).

## I.3 Evaluation

- **Point estimate:** absolute error $\operatorname{AE}=\lvert A(h,U_i)-\hat A(h,U_i)\rvert$ (`acc_err`).
- **Interval:** coverage, amplitude and Winkler score $W_i=(u_i-\ell_i)+\frac{2}{\alpha}\big(\max\{0,\ell_i-a_i\}+\max\{0,a_i-u_i\}\big)$ at $\alpha=0.05$, computed after the run from the stored samples.
- **Metric:** vanilla accuracy only.

## I.4 Paper artefacts and the code that produces them

| Paper artefact | Code | Output |
|---|---|---|
| Table 1: average ranks of AE and $W$ per classifier, Wilcoxon $\alpha=0.05$, L-CBPE excluded | `tables/rank.py` | `<OUT_DIR>/uncawe/tables/rank/` |
| Appendix Table 2: mean AE and $W$ per dataset and classifier | `tables/main.py` | `<OUT_DIR>/uncawe/tables/main/` |
| Figure: coverage vs amplitude, one facet per method | `plots/amplitude_vs_coverage.py` | `<OUT_DIR>/uncawe/plots/main/amplitude_vs_coverage/` |

## I.5 Reproduction

```bash
uv run python -m uncawe.pretrain.classic
uv run python -m uncawe.main --classic
uv run python -m uncawe.tables.main --classic --all
uv run python -m uncawe.tables.rank --classic --all
uv run python -m uncawe.plots.amplitude_vs_coverage --domain classic
```

# Part II: Implemented but not reported in the paper

Everything in this part is **[code-only]**.

## II.1 Additional metrics in `main`

`main` also evaluates macro-F1 (F1 of the positive class on binary datasets) and macro-K (macro-averaged informedness; `k_bin` on binary datasets) (`experiments/metrics.py`). Results are stored under `<OUT_DIR>/uncawe/main/<domain>/<acc_name>/`, and `tables/main.py` and `tables/rank.py` produce one table per metric. The paper reports neither metric. Contingency-table methods fit once and evaluate all three metrics on the same table samples; direct methods (`DoC`) are refitted per metric.

## II.2 The `binform` experiment

`experiments/binform.py` (entry point `uncawe.binform`) compares two BayCAP priors on `connect-4` (3 classes) with the five classifiers. Its 100 target bags have size 1000 and prevalences drawn from $\operatorname{Dir}(q)$ with $q_i=c(1-p_i)/(K-1)$, where $p$ is the validation prevalence and $c=10$ (`TEST_PRIOR_C`). Methods: `O-LEAP`, `RQBS`, `BayesCAP_uniform` and `BayesCAP_transformed_val`, whose prior is the same transformation with $c=3$ (`BAYESCAP_PRIOR_C`). It writes the result schema of `main`. Tables: `tables/binform.py`; figure of the bag-generating and prior distributions on the ternary simplex: `plots/binform_simplex.py`.

## II.3 The `predist` experiment

`experiments/predist.py` (with `predist_config.py`, `predist_io.py`, `predist_sampling.py`; entry point `uncawe.predist`) is a diagnostic of how the distance used by `PrediQuant` ($L_1$, Hellinger, Jensen–Shannon) changes the selected neighbourhood of validation bags. It uses the 3-class datasets in `THREE_CLASS_DATASETS`, one classifier (LR by default), KDEy as the surrogate quantifier, and 3–5 test prevalences at chosen points of the simplex. It has its own result schema (`RESULT_SCHEMA_VERSION = 3`), with atomic incremental writes, and ternary plots per sample (`plots/predist.py`). It is not a comparison between CAP methods.

## II.4 The `BayCAPKDEy` method

`methods/baycap_kdey.py:BayCAPKDEy` implements the BayCAP-KDEy proposal of [[baycap_kdey]]. It pushes joint `BayesianKDEy` prevalence draws $\pi^{(r)}$ (internal MLP surrogate, kernel in $\{$Gaussian, Aitchison, ILR$\}$) through the fixed rate matrix $C^h$ measured on $V$, giving $T^{(r)}=C^h\operatorname{diag}(\pi^{(r)})$. Its estimand is the population table; uncertainty on $C^h$ and finite-$U$ fluctuation are not represented. It has unit tests (`tests/test_baycap_kdey.py`) and has not been evaluated in any experiment.

## II.5 The `BayCAP` method with population or realised-table estimand

`methods/baycap.py:BayCAP` copies the BayesianCC model, priors and NUTS sampler of `cap-ml` `ACC_BayesCAP` (commit `bb20aa5` of `devel`) and adds the hyperparameter `estimand`. With `"population"` (default) it returns the population tables $C^{(r)}\operatorname{diag}(\pi^{(r)})$, identical to `ACC_BayesCAP` for the same `random_state`. With `"realised"` it returns the realised-table posterior predictive: for each draw and each predicted class $\omega_i$, the true classes of the $n^t_i$ target instances predicted as $\omega_i$ are drawn as $\operatorname{Mult}(n^t_i,\rho^{(r)}_{i\cdot})$, with $\rho^{(r)}_{ij}=C^{(r)}_{ij}\pi^{(r)}_j/\sum_l C^{(r)}_{il}\pi^{(r)}_l$ (module-level function `realised_tables`). Both estimands represent the uncertainty on $\pi^Q$ and $C$; only `"realised"` also represents the finite-sample fluctuation of the target table, which is what the experiments score. Both share the NUTS draws for the same `random_state`; the multinomial draws use a NumPy generator seeded with `random_state`. Method names: `BayesCAP` and `BayesCAP-real` (labels in `tables/common.py` and `plots/common.py`). It has unit tests (`tests/test_baycap.py`) and has not been evaluated in any experiment. Source: WP-1.

## II.6 Additional plots

- `plots/ae_vs_winkler.py`: AE against Winkler score per method, one facet per method, markers per classifier.
- `plots/rank_heatmap.py`: heatmaps (dataset × classifier/method) of the mean per-bag rank of each method, for AE and Winkler score.

Both read `main` results for vanilla accuracy.

## II.7 Disabled variants

The CBPE variants `B-CBPE` and `BE-CBPE` were run in earlier versions. Only `plots/common.py` still mentions them (colours and a commented-out method list); `experiments/methods.py` has no factory for them.

## II.8 Toy simulations

`scripts/toys/ext2_estimand_gap.py`, `scripts/toys/ext10_prior_calibration.py` and `scripts/toys/ext3_budget.py` are seeded toy simulations that support [[ext2_baycap]], [[ext10_baycap_priors_tempering]] and [[ext3_uncertainty_budget]]. Their numbers are toy results, not experiment results.

## II.9 Shared infrastructure and tests

- `experiments/execution.py`: task runner (serial or `joblib`), outcome codes, persistence, worker shutdown.
- `experiments/records.py`: `EvaluationContext` and `build_result_frame`, the result schema shared by `main` and `binform`.
- `experiments/metrics.py`, `experiments/methods.py`, `experiments/datasets.py`: metric, method and dataset factories.
- `main.py`, `binform.py`, `predist.py`, `config.py`: compatibility shims over `experiments/`.
- `tests/`: protocol tests for `main`, `binform` and `predist`, and tests of the runner, the result frame, `BayCAPKDEy` and `BayCAP` (`uv run python -m unittest discover -s tests`; `tests/` has no `__init__.py`, so plain `python -m unittest` finds no tests).
- `Makefile` (repository root): runs experiments on `bcuda` (default, `WHERE=remote`) or on the local machine (`WHERE=local`). `make init` creates `REMOTE_DIR` (default `/home/volpi/uncawe-devel`) and copies `env.json`, `sorted_sets.json` and `output/` from the old folder `/home/volpi/uncawe` without overwriting; `make setup` pushes `src/`, `scripts/`, `tests/`, `pyproject.toml`, `uv.lock` with rsync and runs `uv sync --locked`; `make run-<name>` starts one experiment in a detached tmux session `uncawe-<name>` (`GPU=<index>` sets `CUDA_VISIBLE_DEVICES`, `JAX=cpu|cuda` sets `JAX_PLATFORMS`, `ARGS=...` appends arguments); `make status`, `logs`, `attach`, `stop` manage runs; `make pull` copies tables and plots back; `make fetch-data` copies pretrain files and datasets from `bcuda` to the local `cap.env` paths; `make pull-results` and `make push-results` merge result files of `main`, `binform`, `predist` between the two machines with `--ignore-existing`. Local runs need an `env.json` entry for the local hostname with `OUT_DIR`; `env.json` is tracked in the repository, with the entries of every machine.

# Part III: Work tracker

This part tracks all implementation and experimentation work of the project: what is scheduled, what is being worked on, and which experiments are implemented but not yet run. The board in §III.1 gives the state at a glance; each card has an entry in §III.2 with its full description.

## III.1 Board

| ID | Title | Kind | Status | Depends on | Updated |
|---|---|---|---|---|---|
| WP-2 | Estimand-gap check of BayCAP on stored `main` results | analysis | In progress | none | 2026-10-14 |
| WP-3 | Causes of BayCAP's displacement on stored `main` results | analysis | Scheduled | none | 2026-10-20 |

Kinds: **method** (a CAP method or a component of one), **experiment** (an experiment module together with its tables and plots), **analysis** (tables, plots or post-hoc scores on stored results, with no new runs), **infra** (shared infrastructure), **fix** (a Part IV entry, with ID `FIX-<n>`, once the user schedules it).

Statuses, in board order:

1. **Scheduled:** selected by the user, work not started.
2. **In progress:** code is being written.
3. **Implemented, not evaluated:** experiments only. The code, tests and pilot run are done, but the full experiment has not been run, so no stored results exist yet.
4. **Done:** the card leaves the board.
   - A **method**, **infra** or **fix** card is done when its code and tests are merged; its description moves to Part II (methods, infra) or the Part IV entry is removed and the change is noted in Part I or II (fixes). A method that no experiment has evaluated yet is described in Part II as "not evaluated", like `BayCAPKDEy` (§II.4).
   - An **experiment** or **analysis** card is done when the full run has produced stored results under `<OUT_DIR>/uncawe/<experiment>/`; its description moves to Part II (or to Part I if the paper reports it), with the result location, the `cap-ml` commit used and the date of the run.

Rules:

- Only the user adds cards, removes cards, or schedules Part IV fixes. Status changes follow the work and are made in the same change as the code they describe; the `Updated` column holds the date (YYYY-MM-DD) of the last status change.
- Rows are ordered by status, then by ID. Card IDs (`WP-<n>`) and fix IDs (`FIX-<n>`) are never reused.
- A card may cite an extension as its source, but its entry must stand alone: it restates the parts of the extension it needs and stays valid if the extension changes, is cut, or is moved to [[optional_extensions]].
- A card names the Part IV fixes and the other cards it depends on.

## III.2 Entries

Entry template:

```markdown
### WP-<n> <title>

- **Kind / status:** <method | experiment | analysis | infra> / <as in the board>
- **Source:** <extension link, or "new">
- **Goal:** <what the card adds and why>
- **Estimand:** <population table C diag(π) | realised table of the target sample>
- **Scope:** <methods / experiment / tables / plots touched or added>
- **Depends on:** <`FIX-<n>` fixes, `WP-<n>` cards>
- **Comparability:** <effect on existing results; which outputs must be recomputed>
- **Deviations from Part V:** <none | list>
- **Done when:** <observable completion criterion>
- **Log:** <dated notes on progress: pilot run, problems, decisions>
```

### WP-2 Estimand-gap check of BayCAP on stored `main` results

- **Kind / status:** analysis / In progress
- **Source:** [[ext2_baycap]] (E2.1)
- **Goal:** estimate, without new runs, on which (dataset, classifier) pairs a realised-table posterior predictive would change BayCAP's accuracy intervals, and choose the datasets of WP-3. `main` stores BayCAP's accuracy samples but not its table draws, so the predictive is approximated for accuracy only. Given draw $r$, the predictive accuracy is $\frac{1}{N^Q}\sum_i\operatorname{Bin}(n^t_i,\rho^{(r)}_{ii})$, with $\rho^{(r)}_{ii}$ the precision of class $\omega_i$, conditional variance $\sigma_g^2=\sum_i n^t_i\rho^{(r)}_{ii}(1-\rho^{(r)}_{ii})/(N^Q)^2$. The analysis uses one $\bar\rho$ per bag, the stored mean table `estim_cts` (paper layout, predicted classes on rows) with each row normalised to one, and replaces the binomial sum with a normal: each approximate predictive sample is the stored accuracy sample plus independent $\mathcal N(0,\hat\sigma_g^2)$ noise. $n^t$ (predicted counts) and $\bar\rho$ are label-free; labels are used only for scoring.
- **Estimand:** realised table of the target sample (scored against `true_cts`).
- **Scope:** an analysis script and table on `main` results for `BayesCAP`, vanilla accuracy, $\lvert U\rvert=1000$, all datasets and classifiers. Per pair: coverage, amplitude and Winkler score ($\alpha=0.05$) of BayCAP and of the approximate predictive; the ratio $\hat\sigma_g/\hat\sigma_p$, with $\hat\sigma_p$ the standard deviation of the stored samples; the median number of validation instances per class $\lvert V\rvert\pi^V_j$. Seeded noise. No new experiment.
- **Depends on:** access to the `main` results on the remote `OUT_DIR` (or a `make pull-results`).
- **Comparability:** read-only on stored results; nothing is recomputed.
- **Deviations from Part V:** none.
- **Done when:** the per-pair table is stored under `<OUT_DIR>/uncawe/tables/` and the datasets for WP-3 are chosen (largest ratio, smallest ratio, one multiclass), recorded in the log.
- **Log:** 2026-10-14: card added. 2026-10-14: started on branch `wp-2/estimand-gap-check`; `main` results available locally (`OUT_DIR=./output`). 2026-10-14: `tables/estimand_gap.py` and `tests/test_estimand_gap.py` committed on the branch and pushed (commit `8ba027e`), PR #1 into `devel` opened, pending approval; per-pair table written to `<OUT_DIR>/uncawe/tables/estimand_gap/` (local run, seed 0). Gate counts (75 pairs): 53 under-cover (coverage below $0.90$), 3 brought within $[0.93,0.97]$ with lower Winkler, 24 with $\hat\sigma_g/\hat\sigma_p<0.3$, 56 with lower Winkler under the approximation. Suggested pilot datasets (LR): `dry-bean` (largest ratio, $1.50$), `poker_hand` (smallest, $0.07$); not yet chosen. Finding: on large-$K$ datasets BayCAP is biased low (LR: `isolet` $-0.18$, `letter` $-0.07$), consistent with the uniform Dirichlet prior on the columns of $C$ shrinking recalls towards $1/K$; the predictive cannot correct a bias.

### WP-3 Causes of BayCAP's displacement on stored `main` results

- **Kind / status:** analysis / Scheduled
- **Source:** [[ext2_baycap]] (E2.2)
- **Goal:** explain why BayCAP's accuracy intervals are displaced on most pairs. Per (dataset, classifier) pair and per bag, split the error of BayCAP's point estimate into a recall term and a prevalence term, $\hat A-A=\sum_j\pi_j(\hat r_j-r_j)+\sum_j(\hat\pi_j-\pi_j)\hat r_j$. Measure the prior part of the recall term on the actual validation counts, $\sum_j\pi_j\left(\frac{f_{jj}+1}{n^s_j+K}-\frac{f_{jj}}{n^s_j}\right)$. Measure the conditioning of $\hat C$ with $\kappa_0$, the condition number on vectors summing to zero, and the difficulty of the problem with the accuracy of $h$ on $V$.
- **Estimand:** population table (BayCAP as in `main`), scored against the realised table (`true_cts`); the decomposition uses target labels and is an analysis.
- **Scope:** a script in `src/uncawe/tables/` next to `estimand_gap.py`; vanilla accuracy, $\lvert U\rvert=1000$, all 75 pairs. Outputs:
  - Table A (diagnostics), one row per pair: $K$, minimum and median $n^s_j$, $\kappa_0(\hat C)$, accuracy on $V$;
  - Table B (errors), one row per pair: coverage, amplitude, Winkler score, shift, recall term, prior part, prevalence term (mean and mean absolute value), $\lVert\hat\pi-\pi\rVert_1$;
  - both tables as LaTeX and PDF;
  - two scatter plots;
  - tests on hand-computed tables, including a singular $C$.
- **Depends on:** stored `main` results and pretrain validation posteriors and labels (`make pull-results`, `make fetch-data`).
- **Comparability:** read-only on stored results.
- **Deviations from Part V:** none.
- **Done when:** the tables and plots are stored under `<OUT_DIR>/uncawe/tables/` and the E2.2 analysis is written in [[ext2_baycap]].
- **Log:** 2026-10-20: card added, replacing the earlier WP-3 (exact pilot of the realised-table predictive), which is discarded; the exact run is candidate E2.c4 of [[ext2_baycap]].

# Part IV: Pending fixes and known divergences

Every entry is **[fix]**. Each one states where it is fixed (UNCAWE or upstream `cap-ml`) and what it does to existing results.

## IV.1 (FIX-1) SQS-S fits the quantifier on all of $V$

- **Divergence.** The paper's SQS-S (§4.2) splits $V$ with stratification into $V_1\cup V_2$ with $\lvert V_1\rvert\approx\lvert V_2\rvert$, trains the quantifier on $V_1$, and draws the candidate bags from $V_2$. `cap-ml` `PrediQuant` makes the same 50/50 split but fits the quantifier on all of $V$, so the quantifier-training and bag-sampling sets overlap.
- **Where:** upstream in `cap-ml`, followed by a pin bump (FIX-4).
- **Comparability:** `PQ` results in `main` must be recomputed. Old and new `PQ` results are not comparable, and the paper's SQS-S figures come from the old version.

## IV.2 (FIX-2) SQS-H is implemented as a bootstrap variant

- **Divergence.** The paper's SQS-H (§4.1) computes a single estimate $\hat\pi^Q=q(U)$ and draws $m$ bags from $V$ at that prevalence. `RQBS` wraps $q$ in `AggregativeBootstrap`, so each of the $m$ bags has its own bootstrap prevalence $\hat\pi^{*(r)}$. Both fit $q$ on all of $V$, as the paper prescribes for SQS-H.
- **Decision needed:** either document `RQBS` as the implemented variant of SQS-H (in the paper's revision and in §I.2), or add a point-estimate SQS-H and keep `RQBS` as a variant.
- **Comparability:** documenting changes nothing. A new point-estimate SQS-H must be added under a new method name, not by changing `RQBS`.

## IV.3 (FIX-3) Unseeded randomness

- `RQBS.predict_ct_range` calls `sampling_index` without a `random_state` (`cap-ml`, `cap/models/confidence.py`), so its bags depend on the global NumPy state.
- `BayCAPKDEy` builds `MLPClassifier()` without a `random_state` (`methods/baycap_kdey.py`); `mcmc_seed` seeds only NUTS.
- **Where:** `cap-ml` for `RQBS`, UNCAWE for `BayCAPKDEy`.
- **Comparability:** seeding changes the draws. `RQBS` results become reproducible but differ from the stored ones within Monte Carlo error.

## IV.4 (FIX-4) `cap-ml` is pinned to a branch

`pyproject.toml` takes `cap-ml` from `rev = "devel"`, and `uv.lock` is gitignored, so a change on `devel` can silently change results. Fix: pin a commit (or track `uv.lock`) and record the commit used for each set of stored results.

## IV.5 (FIX-5) Interval scores are implemented three times

Coverage, amplitude and Winkler score are computed separately in `tables/main.py` (`get_acc_*`), `plots/common.py` (`interval_metrics`, `winkler_score`) and `experiments/predist.py` (`_interval_metrics`). Fix: one shared module used by all three; the existing tables and plots must not change.

## IV.6 (FIX-6) To verify: smoothed metric used as ground truth

`build_result_frame` (`experiments/records.py`) computes `true_accs` with the `acc_fn` returned by `gen_acc_measure`, which is the *selection* metric: for macro-F1 it is `smooth(f1_macro)` (or `smooth(f1)`), which adds $\varepsilon=10^{-5}$ to every cell. `get_evaluation_acc` (unsmoothed) exists but is not used for the ground truth, and the tables and plots also recompute the truth with `acc_from_ct(..., type="selection")`. Vanilla accuracy and macro-K are unaffected. To check: whether the difference matters for macro-F1 on bags with absent classes, and which version should be the reference.

## IV.7 (FIX-7) README

`README.md` is in Italian and documents only `main`, its tables and the coverage–amplitude plot. It does not cover `binform`, `predist`, `BayCAPKDEy`, the extra plots or the tests.

# Part V: Implementation conventions

Each convention is labelled **[general]** (applies to every experiment) or **[specific: \<experiment\>]**, and as *current practice* (what the code does) or *rule* (required for new code). New code follows these conventions; any deviation is stated in the entry of the Part III card that introduces it. A change to a convention states its effect on reproducibility and on comparability with stored results.

## V.1 Experiments

- **[general] Rule. One module per experiment.** `experiments/<name>.py`, with a module docstring that states scope, purpose, target bags, methods and outputs (as in `experiments/main.py`). A shim under `uncawe/<name>.py` is optional; existing shims stay.
- **[general] Current practice. Output location.** `<OUT_DIR>/uncawe/<experiment>/<domain>/<acc_name>/<dataset>/<classifier_full_name>/<method>.parquet`, built with `util.local_path`. `OUT_DIR` comes from `cap.env`, possibly overridden by `env.json` (`env.py`). Logs go to `<OUT_DIR>/<experiment>.log`.
- **[general] Rule. Frozen experiments.** `main`, `binform` and `predist` keep their method lists, seeds and output paths. New methods are not added to `gen_methods()` or `METHOD_CLASS`: `all_results_exist` and `get_existing_dataset_names` derive completeness from `get_method_names()`, so a new name would mark every dataset as incomplete, trigger re-runs and change the rank tables. New comparisons go in a new experiment directory.
- **[general] Current practice. Seeds.** `qp.environ["_R_SEED"] = 0` seeds the target bags and the validation splits. **Rule:** every random choice in new code (bag sampling, surrogate classifiers, MCMC) takes an explicit seed derived from the experiment seed; no reliance on global RNG state.
- **[general] Rule. Shared target bags.** Target bags depend only on the seed, the dataset, the bag index and the protocol, never on the method, so every method in an experiment is evaluated on the same bags.
- **[general] Rule. Label safety.** Target labels are used only for scoring. Variants that use them (true prevalences, oracle priors, oracle-best hyperparameters) are reported as references, never as methods.
- **[general] Current practice. Incremental runs.** A (dataset, classifier, method, metric) cell whose file exists is skipped; deleting a file re-runs that cell.
- **[general] Current practice. Scheduling.** Methods are split into two pools: regular methods with `cap.env["N_JOBS"]` workers, NumPyro methods (BayCAP and variants) with `BAYES_N_JOBS = 1`. JAX defaults to `JAX_PLATFORMS=cuda`, `CUDA_VISIBLE_DEVICES=0`.
- **[general] Current practice. Shared constants** (`experiments/methods.py`): `SAMPLE_SIZE = 1000` (bag size), `BOOTSTRAP_ITERS = 1000` (performance samples per bag), `BAYES_WARMUPS = 500`, `CONFIDENCE_LEVEL = 0.95`. **Rule:** new experiments reuse them unless the varied factor is one of them.
- **[general] Rule. Branches.** Each card is developed on its own branch from `devel`, named `wp-<n>/<slug>` for cards and `fix-<n>/<slug>` for fixes (e.g. `wp-1/baycap-estimand`), and merged into `devel` through a pull request. Changes that belong to no card or scheduled fix are committed directly to `devel`, without a pull request. The PR does not move the card: the card stays on the board until the user approves and merges the PR, and only then moves as described in §III.1.
- **[general] Rule. Execution.** Experiments run through the `Makefile`, never from a hand-edited copy of the code: on `bcuda` by default, locally with `WHERE=local`. Each run log `<OUT_DIR>/runs/<name>_<timestamp>.out` starts with the git commit and `git describe --dirty` of the code that ran (remote runs push the working tree first), the host, the command, the GPU and the JAX platform; the environment is the one pinned by `uv.lock` (`uv run --locked`). Raw results stay where they are produced and are merged with `make pull-results` / `make push-results`.
- **[general] Current practice. Environment pins.** `pyproject.toml` constrains `numpyro<0.22`: numpyro 0.22.0 makes `ACC_BayesCAP` fail during NUTS (`TracerArrayConversionError` in the `Multinomial` support check, because the total count is traced), so every BayesCAP task would end in an error. The stored results were produced with numpyro 0.21.0 and jax 0.9.2 (old `bcuda` environment), which the current lock reproduces; the pin therefore restores comparability with them.
- **[general] Current practice. JAX platform for NUTS.** BayesCAP defaults to `JAX_PLATFORMS=cuda`, but its model is small and runs a single sequential chain, so it is latency-bound. In a toy timing (`scripts/bench/baycap_platform.py`: `ACC_BayesCAP.sample_posterior`, 500 warm-up iterations, 1000 draws, synthetic counts with seed 0, $K\in\{2,10,26\}$, compilation excluded), one bag took 0.5–1.8 s on the local CPU (ARM, 20 cores) and 3.2–8.9 s on the local GB10 GPU; on `bcuda` it took 1.6–3.8 s on CPU and 6.4–15.4 s on a shared, busy A40. Running with `JAX=cpu` is therefore preferred. Draws are seeded, but CPU and GPU floating point differ, so results computed on different platforms agree statistically, not bit for bit; the run log records the platform.
- **[general] Rule. Pilot first.** New experiments are run on a small configuration (few datasets, LR, few bags) before the full grid; NUTS cost is estimated from the pilot.
- **[specific: main] Current practice.** 100 `UPP` bags from $U$. `DoC` (the only method requiring a validation protocol) is fitted on $V_1$ (60% of $V$) and receives 100 `UPP` bags drawn from $V_2$ as its validation protocol (`util.split_validation`, ratio 0.6); all other methods are fitted on all of $V$. This split is unrelated to the SQS-S split.
- **[specific: binform] Current practice.** Bags drawn at Dirichlet prevalences with per-bag seeds `seed + sample_id`; prior constant $c$ as in §II.2.
- **[specific: predist] Current practice.** Own schema and own incremental logic (`predist_io.py`); results are reloaded only if `schema_version` is at least `RESULT_SCHEMA_VERSION`.

## V.2 Methods

- **[general] Current practice. API.** Methods follow the `cap-ml` API. Contingency-table methods subclass `CAPContingencyTable` (with `CTCAPWithConfidence` for intervals) and implement `fit(V, posteriors)`, `predict_ct_range(X, posteriors)` and `predict_ct(X, posteriors)`; they are fitted once, and `switch(acc_fn)` then `ci_from_cts` evaluate each metric on the same table samples. Direct methods (`CAPDirect`) are refitted per metric with `switch_and_fit`.
- **[general] Rule. Location.** New methods live in `uncawe.methods`, one class per module, each with its own test module in `tests/`. Only generic bug fixes go upstream to `cap-ml`, followed by a pin bump (FIX-4). Helpers shared by several methods are plain functions, not base classes.
- **[general] Rule. Parameters.** Every factor that an experiment varies is an explicit constructor argument, and every class takes a `random_state` that seeds all internal randomness. Bag-sampling methods take the bag size from the target sample unless it is fixed explicitly.
- **[general] Rule. Estimand.** Each method's docstring states whether it targets the population table $C\operatorname{diag}(\pi^Q)$ or the realised table of the target sample, and which uncertainty sources it represents.
- **[general] Rule. Table orientation.** Inside methods, tables follow `cap-ml`: true classes on rows, i.e. $\operatorname{diag}(\pi)C^\top$ for the paper's $C\operatorname{diag}(\pi)$. Docstrings mention the transpose where it matters (as in `BayCAPKDEy.predict_ct_range`).
- **[general] Rule. Bayesian quantifiers.** Methods that consume a Bayesian quantifier use joint draws of shape $(m,K)$ on the simplex, never per-class marginal intervals, and validate them. A posterior temperature is called `temperature` in code and $\tau$ in documents.
- **[general] Rule. Naming.** Code names stay as in §I.2. New methods get a code name and, in their docstring, the paper or proposal name. LaTeX labels go in `tables/common.py:METHOD_LATEX_LABELS`, plot labels and colours in `plots/common.py`.

## V.3 Results

- **[general] Current practice. Schema** (`experiments/records.py`, `results.py:ResultsDataFrame`). One row per target bag, with columns `shifts`, `true_cts`, `estim_cts`, `estim_accs`, `estim_acc_ranges`, `acc_err`, `classifier`, `method`, `dataset`, `collection`, `n_classes`, `acc_name`, `train_prev`, `val_prev`, `t_train`, `t_test_ave`. Contingency tables are stored flattened (`_pack_ct`), with true classes on rows; `estim_cts` is the mean of the table samples (`None` for direct methods).
- **[general] Rule. Store samples, score later.** Results keep the full performance samples (`estim_acc_ranges`), so coverage at any nominal level, Winkler score and other interval scores are computed after the run and never require re-runs.
- **[general] Rule. Schema changes.** New columns are optional so that old files stay readable; experiments with their own schema carry a `schema_version` column, bumped on every change.
- **[general] Current practice. Loading.** `results.py:Results.load(base_dir, domain, acc_name, dataset, filter_classifiers, filter_methods)` globs the Parquet files and concatenates them.
- **[general] Rule. Never commit outputs.** Results, tables, plots and logs stay out of the repository. Environment-specific paths (`env.json`) are committed, so that every machine runs with the same tracked configuration; this changes no result and no comparability.

## V.4 Tables

- **[general] Current practice.** LaTeX tables built with the pandas `Styler`, written as `tabular` fragments plus a standalone document under `<OUT_DIR>/uncawe/tables/<table>/` (`main`, `rank`, `binform`). CLI flags: `--classic|--text|--image`, `--tables` (fragments and document), `--pdf` (compile only, needs `pdflatex`), `--all`, `--metrics`, `--table-dir`.
- **[general] Current practice. Content.** Mean (and optionally std, `--no-std` to hide) per dataset × classifier × method, best value in bold, colour gradient between best and worst, and a $^\dagger$ marker for methods not significantly different from the best under a paired Wilcoxon signed-rank test.
- **[specific: main tables] Current practice.** Metrics `acc_err` and `winkler` (`METRICS`); Wilcoxon over per-bag values at $\alpha=0.01$.
- **[specific: rank tables] Current practice.** Methods ranked per dataset and classifier by mean value, ranks averaged over datasets; Wilcoxon over per-dataset ranks at $\alpha=0.05$; `L-CBPE` excluded (`EXCLUDED_METHODS`) because it runs only on binary datasets.
- **[general] Rule.** Reference variants that use target labels appear as separate, unranked columns, excluded from bold and $^\dagger$ marking. Interval scores are computed from the stored samples with the shared implementation (once FIX-5 is done).

## V.5 Plots

- **[general] Current practice.** Matplotlib/seaborn, with `MPLCONFIGDIR` set to a temporary directory. Method colours, plot labels, classifier markers, font scale and facet layout come from `plots/common.py`. Output goes to `<OUT_DIR>/uncawe/plots/<experiment>/<plot_name>/` (`plots/predist.py`: `<OUT_DIR>/uncawe/plots/predist/<dataset>/`) as png plus svg (`--no-svg` disables it); `plots/amplitude_vs_coverage.py` also writes pdf (`--no-pdf`). `--add-stem` appends a suffix to the file name where supported.
- **[general] Current practice.** Plots of `main` default to vanilla accuracy (`ACC_NAME`) and the methods in `DEFAULT_METHODS` (all except `L-CBPE`), and draw reference lines at the nominal level $0.95$ where relevant.
- **[general] Rule.** One module per figure in `plots/`, reading stored results only (never re-running methods); new methods are added to the shared colour and label maps rather than defined locally.
