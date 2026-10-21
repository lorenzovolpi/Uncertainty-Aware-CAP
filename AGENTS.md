# UNCAWE project instructions

## Scientific context

UNCAWE studies uncertainty-aware prediction of classifier performance under prior probability shift.

The reference paper is:

- `docs/papers/02__Uncertainty_Aware_Classifier_Accuracy_Prediction______QCDS_2026_.pdf`

Before discussing extensions of the paper, read:

- `docs/extensions.md`

and all the in-depth extension documents it points to.

Distinguish clearly between:

- methods implemented in the current codebase;
- proposals documented but not implemented;
- new hypotheses introduced during the conversation.

All methods and ideas stated in `docs/extensions.md` are currently research proposal, not established implemented methods/ideas.

Use english when producing content for documents relative to scientific context.
Use $...$ and $$...$$ for math formulas in every response and production of text regarding the scientific context.
In documents, follow the notation of the reference paper: $C_{ij}=P(\hat Y=\omega_i\mid Y=\omega_j)$ and $T=C\operatorname{diag}(\pi)$, so contingency tables have predicted classes on rows and true classes on columns. Note that `cap.error` and the code store the transpose (true classes on rows); mention this only where implementation details matter.
$T$ denotes contingency tables only; write the temperature of BayCAP or of a Bayesian quantifier as $\tau$ (in the code, the `temperature` parameter).
Every method, interval, and evaluation must state whether it targets the population table $C\operatorname{diag}(\pi)$ or the realised table of the target sample; `main` scores the realised table (`true_cts`).
Variants that use target labels (oracle priors, oracle-best hyperparameters, true prevalences) are reported as references, never as methods.
In Markdown documents, do not hard-wrap text: write each paragraph and each list item on a single line, and use blank lines only to separate blocks (paragraphs, lists, tables, display math). Display-math blocks may span multiple lines.

## Documents

- Every number in a document comes from a seeded script in the repository (toy simulations in `scripts/toys/`) or from stored experiment results; label toy results as such.
- When an in-depth extension document changes its scope, update its summary in `docs/extensions.md`. Move set-aside material to `docs/optional_extensions.md` instead of deleting it. A change to an extension does not require editing `docs/work_plan.md` unless the user asks for it.
- Work on an extension is experiment-driven. In-depth extension documents (e.g. `docs/ext2_baycap.md`) follow this structure:
  - **Claim:** an intuitive, clear and concise statement of what the extension tries to do and why;
  - **Status:** a short table of what is implemented, run, scheduled or proposed;
  - **Experiments:** only the experiments the user has decided on, each with the template below, plus any decision rule;
  - **Candidate experiments:** ideas for experiments that come up in the discussion, each with a short "why" and "what it needs"; an experiment moves to the main section only when the user decides so;
  - **Theory:** the explanations behind the experiments of the main section, driven by them and without digressions, unless the user asks to add an explanation about a discussion or an argument on the extension;
  - **Implementation:** components, tests, reproducibility and comparability for the experiments of the main section;
  - **Appendix:** toy simulations, relation to other extensions, out of scope.
- Every experiment of the main section, in every extension, follows this template (as in `docs/ext2_baycap.md`, E2.1):

  ```markdown
  ## E<extension>.<n>: <title>

  **Why.** <what justifies the experiment>

  - **Design.**
    - **Inputs:** <data, stored results, datasets, classifiers, bags>
    - **Varied:** <the factors that change, and what is held fixed>
    - **Measures:** <what is computed, for which metrics>
    - **Estimand:** <population table or realised table, for each method, and what it is scored against>
    - **Cost:** <runs needed and their size, or "none">
  - **Representation.** <tables and plots of the results>
  - **Analysis.** <once run: the results, with their source, and what they show; before: "Not run.">
  ```
- Experiments are named `E<extension>.<n>`, with `<n>` a progressive number within the extension (e.g. E2.1, E2.2); each experiment gets its own number, also when several experiments serve one goal. Candidate experiments are named `E<extension>.c<n>` (e.g. E2.c1); when the user moves one into the main section, it is renamed with the next free `E<extension>.<n>`, and every reference to it is updated. Numbers are never reused.
- Tackle one experiment at a time.
- Explain statistics at a first-year-undergraduate level, so that the documents are broadly comprehensible.
- Extension documents contain no paper material (contributions, paper outline, figure plan) for now; the paper will be addressed later.
- Only `docs/ext2_baycap.md` follows this structure so far; the other extension documents keep their old structure until the user reviews them.

## Work plan

`docs/work_plan.md` is the tracker of all implementation and experimentation work of the project. It has five parts: I implemented and reported in the paper; II implemented but not in the paper; III work tracker (board and card entries); IV pending fixes and known paper/code divergences; V implementation conventions.

- Only the user adds or removes Part III cards and schedules Part IV fixes on the board. Propose candidate cards in the conversation, never insert them in the document.
- Keep the Part III board in sync with the work: when you start a card, set it to "In progress"; when an experiment's code, tests and pilot run are done but the full run is not, set it to "Implemented, not evaluated". Runs happen only when the user asks for them (see Development), so a card whose code and tests are done stays "In progress" until the user has asked for the pilot run and it is done. Update the board's `Updated` date and the entry's `Log` in the same change as the code.
- When a card's branch is merged, fill the `Outcome` field of its entry: the merge commit, and the decisions, deviations, problems and follow-ups that came up during implementation and are not already in the card. When the card moves to Part II or Part I, carry its `Outcome` into that entry.
- "Implemented, not evaluated" applies only to experiments and analyses. A method, infra or fix card is done when its code and tests are complete.
- When a card is done, remove it from the board and from §III.2: a method or infra card moves to Part II (a method not yet evaluated by any experiment is marked "not evaluated"); an experiment or analysis card moves to Part II (or Part I if the paper reports it) only after the full run has stored results, recording the result location, the `cap-ml` commit and the date; a fix removes its Part IV entry and notes the change in Part I or II.
- Part III cards stand alone: they may cite an extension as their source, but they restate what they need and stay valid if the extension changes or is cut. Use the entry template in §III.2.
- Parts I, II and IV describe the code as it is. Update them in the same change that alters the code, and cite the file paths involved.
- New code follows the conventions of Part V. State any deviation in the card's entry. When a convention changes, record its effect on reproducibility and on comparability with stored results.
- Label every convention as general (applies to every experiment) or specific (name the experiment), and as current practice or rule for new code.
- Prerequisite fixes are described in Part IV, not as Part III cards; once scheduled they appear on the board with their `FIX-<n>` ID; Part III cards have IDs `WP-<n>`.
- §II.12 lists the code that should eventually move to `cap-ml` (owned by the user); for now, the methods in `src/uncawe/methods/`. Keep it in sync: add an entry when such code is added, update it when the code changes, and remove it, recording the `cap-ml` commit, once the code has moved.

## Development

- Use Python 3.12 or newer.
- Run commands through `uv run`.
- Preserve the existing experimental protocol unless a change is requested.
- Experimental outputs live outside the repository under `cap.env["OUT_DIR"]/uncawe`.
- Do not commit generated results, plots, or tables.
- Before changing an experiment, identify its effect on reproducibility and comparability with existing results.
- After implementing code, you may run the unit tests, but do not run experiments, analyses, table or plot scripts, or any other code (locally or on `bcuda`, including `make run-*`) unless the user explicitly asks for that run.
- Each card or scheduled fix is developed on its own branch from `devel` (`wp-<n>/<slug>` or `fix-<n>/<slug>`). When the work is done, the user reviews the diff from `devel`; after the user approves, the branch is optionally rebased on `devel` and merged with `git merge --no-ff`, with a merge commit message that starts with the card or fix ID (e.g. `WP-3: ...`). No pull requests.
- Changes to the repository that do not belong to a card or a scheduled fix on the Part III board of `docs/work_plan.md` are committed directly to `devel` unless the user says otherwise.

## Conversation conventions

- A user message that starts with "[guideline]" states a general guideline: add it to this file, in the section where it belongs.
