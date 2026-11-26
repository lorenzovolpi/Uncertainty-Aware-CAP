# Tables and figures of the extended paper

This document lists the tables and figures that the extended version of the paper shows (user's decision of 2026-11-24). The list itself is `src/uncawe/paper.py:ARTEFACTS`, with the command that produces each artefact in `COMMANDS`; this document describes it and must be kept in sync with it.

**Rule.** When a change touches the code of any artefact below (methods shown, labels and macros, colours, layout), all the artefacts are regenerated together and collected again, so that the paper never mixes outputs of different versions of the code.

## How to regenerate

1. Commit the change on `devel`.
2. `make run-paper` runs, on `bcuda`, every command of the list below in order (it stops at the first failure) and then collects the artefacts (`python -m uncawe.paper --generate --collect`).
3. `make pull PULL_DIRS=paper` copies the folder to the local machine.

`uv run python -m uncawe.paper --list` prints the commands and the source and destination of every file; `--collect` alone collects again from outputs already generated.

One table or figure at a time: `make run-paper-<item>` runs the command of that item on `bcuda` and copies only its files into the paper folder, overwriting them and keeping the others (`python -m uncawe.paper --generate --collect --only <item>`); then `make pull PULL_DIRS=paper`. The items (`src/uncawe/paper.py:ITEMS`), one per command of the list: `table-main`, `table-rank`, `table-tau-select`, `table-coverage-amplitude`, `table-binary`, `plot-avc`, `plot-shift-profile`, `plot-binform-side-prior`, `plot-pq-rho`, `plot-target-size-scores` (e.g. `make run-paper-table-main`).

## The paper folder

`<OUT_DIR>/uncawe/paper/` holds:

- `tables/`: the LaTeX tabulars, to `\input` in the paper;
- `figures/`: each figure as pdf and png;
- `macros.tex`: the macros of the method names (`tables/common.py:METHOD_LATEX_MACROS`, e.g. `\bcappop`, `\bcapkdey`, `\sqshbs`), which the tabulars use; the paper inputs it in its preamble.

The two subfolders are emptied at every collection, so they hold exactly the artefacts of the list; a missing source stops the collection before anything is copied. Files keep the name of their source, except the two coverage vs amplitude figures, renamed `amplitude_vs_coverage` and `amplitude_vs_coverage_zoom`.

## Tables

Every table and figure scores the realised table of each bag (`true_cts`), as `main` does.

| Artefact | Files in `paper/tables/` | Command | Source |
|---|---|---|---|
| Main table: mean AE and Winkler score $\mathcal W$ per classifier and dataset, without std; baselines (since 2026-11-26 also AC, i.e. `AC-LSB`, on vanilla accuracy only) and the other methods of `tables/main.py:MAIN_METHODS`; rows of wins, ties and losses against LEAP and of the average rank | `main_classic_<acc>.tex`, one per accuracy measure (vanilla accuracy, macro-F1, macro-K) | `make run-tables-main ARGS="--tables --no-std"` | `tables/main/tabular/` |
| Rank tables: mean rank per classifier, one table per score | `main_classic_<acc>_rank_<score>.tex`, $3\times2$ (score `acc_err` or `winkler`) | `make run-tables-rank ARGS="--tables"` | `tables/rank/tabular/` |
| $\tau$ chosen per pair on calibration bags (E5.3), compact: $\tau_{\text{cal}}$, oracle-best $\tau^\star$ (a reference), AE and $\mathcal W$ at $\tau=1$ and $\tau_{\text{cal}}$, for the bases of BayCAP-pop, BayCAP-sample, BayCAP-Hf and BayCAP-KDEy (`BayesCAP`, `BayesCAP-real`, `BayesCAP-Hf-real`, `BayCAP-KDEy-h-inst-cb-keep`; `tables/tau_select.py:COMPACT_BASES`, user's decision of 2026-11-26), whose $\tau_{\text{cal}}$ columns are the four default methods | `tau_select_compact_<acc>.tex`, one per accuracy measure | `make run-tables-tau-select ARGS="--tables"` | `tables/tau_select/tabular/` |
| Coverage $\mathcal C$ and amplitude $\mathcal A$: mean over the bags per classifier and dataset, for the methods of the main table without BayCAP-pop; hard colours (green or white): coverage green within $0.10$ of the nominal level $0.95$, amplitude green below $0.10$ (user's decisions of 2026-11-25) | `main_classic_<acc>_coverage_amplitude.tex`, one per accuracy measure | `make run-tables-coverage-amplitude ARGS="--tables"` | `tables/coverage_amplitude/tabular/` |
| Binary datasets (user's request of 2026-11-26): the main table restricted to the 5 binary datasets (`uci_binary`), mean AE and $\mathcal W$ per classifier and dataset, without std, for DoC, QuAcc, LEAP, AC (`AC-LSB`) and CBPE (`L-CBPE-LSB`, binary only) under "Baseline methods" and BayCAP-KDEy (`BayCAP-KDEy-h-inst-cb-tcal-keep`) under "Our methods"; rows of wins, ties and losses against LEAP and of the average rank; vanilla accuracy only (AC and CBPE estimate no other measure); `tables/binary.py` | `binary_classic_vanilla_accuracy.tex` | `make run-tables-binary ARGS="--tables --no-std"` | `tables/binary/tabular/` |

## Figures

| Artefact | Files in `paper/figures/` (pdf, png) | Command | Source |
|---|---|---|---|
| Coverage vs amplitude, one facet per method of the main table without SQS-B and BayCAP-pop (vanilla accuracy): per pair the mean coverage and amplitude over its bags, the KDE regions holding $50\%$, $70\%$ and $99\%$ of the pairs (bandwidth $0.75\times$ Scott's rule), a box with the means over the pairs of $\mathcal W$, amplitude and coverage; `poker_hand` with LR left out (user's decision of 2026-11-25); linear axes, and axes zoomed on amplitude $0.05$ and coverage $0.95$ | `amplitude_vs_coverage`, `amplitude_vs_coverage_zoom` | `make run-plot-avc` (both) | `plots/main/amplitude_vs_coverage/classic_vanilla_accuracy`, `classic_vanilla_accuracy_zoom` |
| Shift profile: AE and $\mathcal W$ by within-pair shift decile (vanilla accuracy), all 15 datasets, without SQS-B; per method and decile the geometric mean over the datasets on a logarithmic axis, with bars of $\pm1$ standard error over the datasets after removing each dataset's average level (user's decision of 2026-11-25) | `shift_profile_pooled_classic_vanilla_accuracy` | `make run-plot-shift-profile ARGS="--figures pooled --exclude-methods SQS-B"` | `plots/main/shift_profile/` |
| E5.5, side priors: per-pair violins at the edge and at the centre of the simplex, all pairs (vanilla accuracy), without SQS-B (user's decision of 2026-11-25); the references are the SQS methods with KDEy, without the PACC ones (user's decision of 2026-11-26) | `binform_side_prior_pair_violins_edge_all_vanilla_accuracy`, `binform_side_prior_pair_violins_centre_all_vanilla_accuracy` | `make run-plot-binform-side-prior ARGS="--figures pair-violins --groups all --metrics vanilla_accuracy --exclude-methods SQS-B SQS-H SQS-Hbs PQ"`, after `make run-tables-binform-side-prior ARGS="--tables"` | `plots/binform_side_prior/` |
| E6.2, allocation $\rho$ of SQS-S with KDEy (`PQ-KDEy-rho*`, since 2026-11-26, user's decision): AE and $\mathcal W$ against $\rho$ (vanilla accuracy) | `pq_rho_ae_w_vanilla_accuracy` | `make run-plot-pq-rho ARGS="--metrics vanilla_accuracy --quantifier kdey"`, after `make run-tables-pq-rho ARGS="--tables --quantifier kdey"` | `plots/pq_rho/kdey/` |
| E4.4, target size: AE, $\mathcal W$, amplitude and coverage across $\lvert U\rvert$ and across $\lvert V\rvert$, all datasets (poker_hand included), without SQS-B and without the SQS methods with PACC (their KDEy versions are drawn, user's decision of 2026-11-26); geometric mean over the datasets (coverage: mean, line at $0.95$), linear vertical axes, bars of $\pm1$ standard error over the datasets after removing each dataset's average level, as the shift profile (user's decision of 2026-11-25) | `target_size_u-scores`, `target_size_v-scores` | `make run-plot-target-size-scores ARGS="--exclude-methods SQS-B SQS-H SQS-Hbs PQ"` | `plots/target_size/u_scores/`, `plots/target_size/v_scores/` |

## Inputs the list does not regenerate

The commands above read stored results and, for some figures, the CSVs of other tables, which `make run-paper` does not regenerate: `plots/pq_rho.py` reads the per-pair CSV of `tables/pq_rho.py`, and the tables of E5.3 read the calibration runs of `tau_select` and the stored runs of `main` and E5.2. Regenerate those first when a change touches them.
