# Proposed bounded finite row-edit search

**Proceed only if the fixed target gate passes.** The initial malformed completions are an assay problem, not evidence of useful content-error targets. Two genuine wrong choices in different families do not satisfy the requirement for two repeatable hint-rescued content errors in one family. The current follow-up may identify such a family or establish that none qualifies. Do not force a candidate search in the latter case.

This design fixes a small forward-only intervention search before candidate scores. It makes no gradient, training, or “rule stored in a row” claim. No model, actual row-vector, or held-out-answer evaluation was performed in writing it.

## Lock the assay and target first

Choose one explicit serialization/system-instruction condition using the already declared assay diagnostics, then rerun the entire relevant baseline and controls under that condition. If a formatting-only instruction makes answers usable, that is an assay change; its gain cannot be credited to a memory edit. Do not choose different system prompts per task or candidate. Freeze the exact prompt bytes/tokens, unrestricted greedy settings, generation cap, EOS treatment, complete-text grader, and target/preservation membership.

Require at least two repeatable **well-formed content errors within one family** that its fixed procedural hint corrects on repeated runs, with no hinted loss on that family's baseline-correct development items. If multiple families qualify, select by: most repeatable hint rescues; then fewest malformed ordinary completions; then the already published family order. Freeze this ranking before reading the follow-up scores. If no family qualifies, publish the assay result and stop this search.

Candidate evaluation uses unhinted task prompts under the locked assay. Hints are absent at candidate inference. Keep hints and formatting-only diagnostic arms out of the candidate objective. The target objective is actual generated-answer accuracy, not correct-token probability, logits, local scalar responses, or agreement with hinted output text. Targets remain programmatic oracle answers.

## Select existing rows without candidate-score feedback

Use observed token-to-row mappings from the locked **development prompts only**. Prefer one exact, content-bearing trigram wholly inside the shared target instruction, with all three tokens inside that instruction span. For counting, the fixed “Count overlapping occurrences” phrase is a hypothesis to inspect, not a guaranteed three-token unit. Tokenizer boundaries must determine the actual trigram. Exclude chat delimiters, assistant prefixes, answer labels/options, the generic A/B suffix, and trigrams crossing content/format boundaries.

Require that the chosen trigram is accessed in every target development prompt; if no eligible common trigram exists, stop and revise the search plan before observing candidate scores rather than quietly targeting formatting. Among eligible trigrams use a deterministic rule: fewest address collisions with preservation prompts, then earliest position in the common content instruction, then lexicographic token-ID order. Record all eligible candidates and the chosen rule. Row lookup traces and baseline answers may inform target/coverage selection; actual row perturbation results must not.

Edit only the existing trigram-head rows associated with that one chosen trigram, at most eight distinct global addresses. Confirm the table's actual head/order layout rather than assuming an address range. Deduplicate address aliases; a global row gets one replacement at every occurrence. Record head, n-gram order, row ID, triggering token IDs, target occurrence positions, control collisions, original-row byte hash, and decoded FP64 RMS. Reject nonfinite anchors or zero-RMS anchors for this design. Do not invent a new row or train a global delimiter.

Control collisions do not necessarily invalidate a content row: a preservation case may legitimately share the instruction. Prefer zero unrelated-control collisions, report every remaining collision, and preserve the corresponding correct answers. Absence of direct row overlap also does not guarantee absence of broader behavior changes. No holdout tokens or row mappings should influence this selection.

## Freeze exactly 12 candidate overlays

Let `a_r` be the original decoded FP32 vector of row `r`, and `rho_r = RMS_FP64(a_r)`. Use the same selected row set in every candidate. For each of three named raw directions `v_r`, define the row direction:

```text
u_r = v_r / RMS_FP64(v_r)
d_r = FP32(rho_r * u_r)
replacement_r = FP32(a_r + FP32(epsilon) * d_r)
```

The last expression must use the same audited FP32 multiply-then-add behavior as the row-patch utility, not silently switch to FP64 addition or fused arithmetic. Record realized replacement RMS and changed-coordinate counts after export; nominal epsilon alone is insufficient. Reuse the exact same baseline anchors for every overlay.

| Direction | Fixed construction | Interpretation |
|---|---|---|
| Seeded Rademacher | For each row coordinate, derive ±1 from SHA256 of a fixed seed, global row ID, and coordinate index using an explicitly specified UTF-8 message/bit rule. | Deterministic generic direction; no quality claim. |
| Alternating | Coordinate 0 is +1, then alternate −1/+1. | Generic control direction matching the earlier response experiment's structure. |
| Hint-row contrast | Mean of a predeclared set of same-head content-hint row vectors, minus the target anchor; normalize as above. | Hypothesis that a structural row displacement may transfer useful behavior. It is not a representation of a rule or a qualified gradient. |

Suggested fully specified random rule: `SHA256("ngramma005:seed=5005:row=" + decimal_row_id + ":coord=" + decimal_coordinate)`; choose +1 when the low bit of the first digest byte is 1, otherwise −1. Log this string format and seed in the search manifest.

For the hint contrast, use only trigrams wholly inside the previously fixed procedural hint, excluding chat scaffolding, the appended task, and any instance answer. For each target row's head, take the arithmetic mean of **unique** qualifying same-head global rows, computed in FP64. Repeated occurrences do not add weight. Exclude the target row itself. Record contributing row IDs/token spans and source hashes. Do not mix vectors across heads, because their coordinate roles are not established as interchangeable. Do not choose hint trigrams or subsets based on edit scores. If the set is empty or the resulting direction has zero/nonfinite RMS, mark the direction unavailable and shrink the candidate count; do not substitute an adaptive direction to fill the budget.

For each available direction use `epsilon ∈ {−2^-8, +2^-8, −2^-6, +2^-6}`: three directions × four signed magnitudes = **12 candidates maximum**. No blends, per-row sign fitting, coordinate search, changed magnitudes, or additional seeds after scoring. All selected rows move together under a candidate's global sign and magnitude. Hash the full candidate manifest and exported overlays before the first candidate evaluation. Preserve exact duplicate/no-change overlays in the manifest but score a unique overlay only once.

These magnitudes are a bounded exploratory choice informed by earlier observed finite responses; they are not proven safe or sufficient for new rows. Editing up to eight rows is a multirow intervention: a success cannot identify which row or direction component caused the gain. A later ablation would be a separate preregistered experiment.

## Controls, selection, and compute accounting

Before candidate scoring, require unmodified and original-row replacement controls to agree on complete generated outputs under the qualified generation path. Repeat the baseline tasks within a process to test state reset and across fresh processes to test repeatability. Record EOS/cap termination and raw completion bytes. Do not count a cap-truncated answer as proven compliant stopping; apply the same frozen rule to all conditions.

Every unique candidate receives the same target development tasks and the same preservation suite, in a fixed order, with fresh state per job and one immutable overlay per process. The preservation suite should include baseline-correct target items, baseline-correct original families under the stabilized assay, and the declared easy controls. If counting is targeted, add its four predeclared counting controls. Do not select a smaller control set after observing damage. Record baseline-wrong/malformed preservation items too, although they cannot provide a “correct answer retained” check.

Reject a candidate that loses any baseline-correct preservation answer, turns a previously well-formed response malformed, fails a resource/control check, or does not correct at least two targeted baseline errors. Rank surviving candidates by target correct-answer count, then fewer changed rows, then smaller absolute epsilon, then fixed direction order above and fixed sign order negative before positive. This is a discrete pilot objective, not a significance test. Do not use held-out results, token probabilities, or prettier explanations to break ties.

Confirm only the selected winner with a repeat of the locked baseline and candidate batches in fresh processes before accessing holdout. If it fails confirmation, report no confirmed candidate for this round; do not use confirmation failures to start an unplanned search. Count baseline/zero controls, all 12 candidate attempts including failures, and confirmation as compute. They are additional processes beyond the candidate count. A process resource failure is not an incorrect model answer and must not silently drop a task from the denominator.

## One locked evaluation and honest conclusions

Freeze one confirmed overlay hash before evaluating any held-out inputs. Run the original and candidate on the complete declared held-out partition and fixed controls with identical unrestricted generation. Report paired gains/losses, malformed and cap-terminated counts, target/non-target family breakdowns, and row/prompt overlap. Baseline-correct holdout preservation losses block promotion. Do not retune and rescore on the same partition; a failed holdout remains a failed locked test.

The same-head hint contrast is worth retaining as one of three candidates, alongside the two generic directions, because it is an explicit falsifiable hypothesis at low extra search cost. It is not clearly preferable a priori. A successful candidate would show that one finite existing-row intervention improved this small generated-answer test under fixed conditions. Numeric-input holdouts that share templates/rules remain within-family tests. Neither successful hint rescues nor a small candidate win establishes a reliable general improvement in intelligence, a useful optimizer, or a deployable overlay. Broader claims require larger untouched evaluations across families, independent row edits, and stronger preservation coverage.
