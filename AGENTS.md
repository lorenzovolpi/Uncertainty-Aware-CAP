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
- In-depth extension documents (e.g. `docs/ext2_baycap.md`) follow this structure:
  - **Summary:** question, main claim, candidate methods, status table, reader's guide;
  - **Part I, Understanding:** the needed background and concepts, each with a one-line intuition, and the reasoning behind the hypotheses;
  - **Part II, For the paper:** contributions, hypotheses with "confirmed if / refuted if", and a paper outline mapping each claim to its evidence and to the figures and tables;
  - **Part III, For the implementation:** one specification per experiment with fixed fields (goal / hypothesis, inputs, varied factors, outputs, measures, interpretation, visualization, cost), then components, tests, and reproducibility;
  - **Appendix:** toy simulations, relation to other extensions, out of scope.

## Work plan

`docs/work_plan.md` is the tracker of all implementation and experimentation work of the project. It has five parts: I implemented and reported in the paper; II implemented but not in the paper; III work tracker (board and card entries); IV pending fixes and known paper/code divergences; V implementation conventions.

- Only the user adds or removes Part III cards and schedules Part IV fixes on the board. Propose candidate cards in the conversation, never insert them in the document.
- Keep the Part III board in sync with the work: when you start a card, set it to "In progress"; when an experiment's code, tests and pilot run are done but the full run is not, set it to "Implemented, not evaluated". Update the board's `Updated` date and the entry's `Log` in the same change as the code.
- "Implemented, not evaluated" applies only to experiments and analyses. A method, infra or fix card is done when its code and tests are complete.
- When a card is done, remove it from the board and from §III.2: a method or infra card moves to Part II (a method not yet evaluated by any experiment is marked "not evaluated"); an experiment or analysis card moves to Part II (or Part I if the paper reports it) only after the full run has stored results, recording the result location, the `cap-ml` commit and the date; a fix removes its Part IV entry and notes the change in Part I or II.
- Part III cards stand alone: they may cite an extension as their source, but they restate what they need and stay valid if the extension changes or is cut. Use the entry template in §III.2.
- Parts I, II and IV describe the code as it is. Update them in the same change that alters the code, and cite the file paths involved.
- New code follows the conventions of Part V. State any deviation in the card's entry. When a convention changes, record its effect on reproducibility and on comparability with stored results.
- Label every convention as general (applies to every experiment) or specific (name the experiment), and as current practice or rule for new code.
- Prerequisite fixes are described in Part IV, not as Part III cards; once scheduled they appear on the board with their `FIX-<n>` ID; Part III cards have IDs `WP-<n>`.

## Development

- Use Python 3.12 or newer.
- Run commands through `uv run`.
- Preserve the existing experimental protocol unless a change is requested.
- Experimental outputs live outside the repository under `cap.env["OUT_DIR"]/uncawe`.
- Do not commit generated results, plots, tables, or environment-specific paths.
- Before changing an experiment, identify its effect on reproducibility and comparability with existing results.
