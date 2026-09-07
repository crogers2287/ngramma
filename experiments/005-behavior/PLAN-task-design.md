# Experiment 005 task-design proposal

Find repeatable **content errors in actual unconstrained generated answers** before searching for any row edit. Experiment 004 established finite row sensitivity, not behavioral improvement. These tasks provide programmatic ground truth without a teacher. None is asserted to be an observed failure of this checkpoint yet.

## Bounded discovery curriculum

`tasks.py` supplies 16 development tasks: four per family, with two A and two B answers in each family. All answers require selecting the correct substantive result; formatting alone cannot determine it.

| Family | Candidate error mechanism | Oracle and contrast |
|---|---|---|
| Decimal versus software-version comparison | Applying component or digit-length comparison to decimal values, or decimal rules to versions | `Decimal` numeric comparison versus integer component tuples. Pair the same displayed numbers under different semantics. |
| Signed arithmetic | Losing a minus sign when subtracting a negative or multiplying negatives | Python integer operations, explicit parentheses, two subtraction and two multiplication items. |
| State replacement | Keeping stale state or treating a later assignment as an addition | Sequential `set`/`add` interpreter; distractor omits the last replacement. |
| Missing versus zero | Treating an explicit zero as an absent field | Dictionary membership, not truthiness; half the records are empty and half contain zero. |

These are familiar reasoning distinctions worth probing, not literature-backed claims about this model's error rates. Every family balances answer labels. Decimal/version mode, arithmetic operation, and missing-field presence also appear with both correct labels, avoiding a trivial subtype-to-label rule. Development decimal examples intentionally share number pairs across semantic modes; they are paired contrasts, not independent samples.

The script also generates 32 distinct held-out inputs (eight per family) and eight easy regression controls (two per family, balanced labels). Input domains are separate; generation is deterministic with a fixed seed and no external files. No previously sealed family or its answers is accessed. Neither holdout nor controls are teacher targets. The development hints state the relevant rule without an item-specific result and are for diagnosing a reminder advantage, not for deployment.

## API and grading contract

Use `generate_tasks('dev')`, `generate_tasks('holdout')`, or `generate_tasks('controls')`. Each `Task` exposes `id`, `family`, `split`, `prompt`, `answer`, `hint`, and `hint_prompt`. `to_dict()` produces a JSON-compatible record. CLI output is JSONL. Keep `answer`, `semantic_input`, and other oracle fields out of the actual model prompt.

`grade(task, generated_text)` consumes the **entire actual decoded completion**. It permits surrounding whitespace and otherwise accepts exactly `A` or `B`. Empty answers, explanations, punctuation, answer prefixes, multiple choices, reasoning tags, and lowercase answers fail and are labeled `malformed` separately from `wrong_choice`. Do not extract the first convenient letter, discard reasoning content, or stop generation when a valid answer first appears. The actual generated text must remain available for audit. A normal EOS is fine; fix a short generation cap before all runs and report cap exhaustion separately.

Use ordinary decoding, with no grammar, allowed-token list, forced answer, or A/B logit comparison. Fix chat serialization, model identity, EOS settings, generation cap, and greedy decoding before scoring. Greedy decoding is still unconstrained decoding. For a reasoning-enabled model, establish the supported no-thinking serialization once rather than postprocessing reasoning away; malformed responses do not establish the content mistake this curriculum seeks.

Prompts are short (under 50 whitespace-delimited words); this is **not a token-count guarantee**. Tokenize the exact serialized prompts on the pinned tokenizer before running, prefer fewer than 60 model tokens, and report actual lengths including chat overhead. Hints add tokens. The earlier 32-token exact-prefill qualification does not automatically qualify longer or autoregressive execution; validate the intended generation path and zero overlay first. Do not truncate prompts or silently change a subset after inspecting answers.

## Discovery and progression gates

1. Score all 16 unhinted development tasks and eight controls with the same fixed unrestricted generation protocol. Retain every completion and distinguish incorrect choices from malformed output. Repeat the full unhinted development batch in a fresh process. No row search if errors are not reproducible.
2. Run the 16 hinted development contrasts in fresh state. Repeat any proposed hint rescue under the same protocol. A candidate family needs at least two reproducible, well-formed unhinted content errors rescued by its hint, and no hinted loss on its previously correct development items. This is a pilot selection gate, not statistical proof of a broad reminder advantage. If none qualifies, report no target found; do not switch objectives to answer formatting or use the holdout to manufacture one.
3. Select one family and freeze its development set, objective, generated-answer grader, eligible existing rows, search budget, and regression controls before finite row-edit search. Baseline-correct controls remain preservation checks; do not optimize their answers or drop controls that an edit breaks. A narrow candidate must improve at least two development answers under repeat runs with no loss on baseline-correct controls before accessing held-out inputs. Root's experiment plan should fix the search and promotion thresholds before execution.
4. Lock one overlay and evaluate all 32 held-out inputs plus controls once with the same baseline/overlay protocol. Report paired task outcomes and family-level counts, including losses outside the target family. If evaluation fails, retain it; using it to revise the overlay turns that partition into development data and requires a new untouched evaluation partition. A small pass is a within-family pilot signal, not evidence that the model is reliably smarter in general.

A second fresh greedy run is a reproducibility check, not an independent statistical sample. Sixteen related development prompts and eight holdout variants per family give little power for broad generalization. The holdout shares task families, rule hints, templates, and authoring logic with development; numeric disjointness does not make it a sealed unseen-family evaluation. Balanced labels limit position shortcuts but do not eliminate template learning or prompt-specific row effects. Future claims of reliable improvement need new families/templates, larger locked evaluations, multiple repeatable edits, and preservation tests beyond these eight controls.

## Suggested initial target and model-free validation

Start discovery with the paired decimal/version comparisons: the same surface numbers require different content decisions, and a reminder can isolate confusion about the comparison rule. Do not assume that family fails or select it over a family with stronger preregistered development evidence. Signed arithmetic and state replacement are substantive alternatives; missing-versus-zero is also a useful preservation challenge.

`python experiments/005-behavior/tasks.py --self-test` runs five embedded model-free tests covering deterministic/disjoint inputs, family counts and label balance, independently checked oracle examples, strict grading, and bounded prompt word lengths. `python experiments/005-behavior/tasks.py --split dev` exports the development JSONL. No model process, teacher call, row edit, or actual task evaluation was performed while implementing this curriculum.
