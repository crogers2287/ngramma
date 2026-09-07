# Experiment 006: retain an instruction-following reminder in existing rows

This is a new objective, fixed before any candidate scores. Experiment 005
rejected its content-reminder target: neither counting nor state replacement
had the required rescues. It did establish a repeatable instruction-following
contrast. On the original 16 development prompts, a uniform response-format
system instruction changed complete correct answers from 1/16 to 14/16. All
seven originally correct development/control items remained correct. This is
a narrow response-contract capability, not new arithmetic knowledge.

## Fixed objective and data

Use experiment 005's original 16 development prompts and eight controls,
without the stronger system instruction. Keep the exact no-thinking chat
serialization, four-token cap, EOG requirement, and unconstrained greedy
decoding. The output must be exactly the programmatically correct A/B choice,
apart from surrounding whitespace. Never extract a letter from an explanation
or count a cap-truncated prefix as success. Score malformed, incorrect, and
unfinished answers separately.

The stronger formatting instruction is guidance for diagnosis and one
candidate direction only. It is absent during candidate and primary holdout
inference. This experiment explicitly permits targeting the existing rows for
the requested response format, unlike experiment 005's content-row search.
It still excludes chat delimiters and assistant scaffolding. No teacher output,
new embedding row, adapter, or backbone change is used.

## Rows and finite candidates

Tokenize the original shared instruction `Answer only A or B.` using the pinned
runtime. Among its trigrams that occur in every development/control prompt,
choose the earliest triple whose three tokens are wholly inside the instruction.
Use its eight existing trigram-head rows; record every occurrence and collision.
No other table or model value changes. Apply a row's replacement at every lookup.

Use the three directions and FP32 arithmetic specified by experiment 005's
`SEARCH-DESIGN.md`: fixed SHA256 Rademacher signs, alternating signs, and a
same-head hint-row mean minus the target anchor, each normalized to the original
row RMS. For the third direction, the donor span is the already tested format
system instruction, not the state-replacement hint. Include every unique
eligible trigram row wholly within that span, excluding selected target rows.
Use signed magnitudes `{-2^-8,+2^-8,-2^-6,+2^-6}`: at most twelve candidates.
Unavailable directions shrink the budget. Hash all recipes and overlays before
the first candidate generation. Keep raw rows/overlays local; publish recipes,
hashes, outcomes, and reproduction code.

This is derivative-free candidate selection in the actual quantized engine.
No surrogate derivative or full-sequence gradient is claimed. All candidate
answers use live native attention/routing with fresh sequence state.

## Controls, selection, and held-out evaluation

Require original-row replacements to reproduce all 24 unmodified task outputs
and selected logits. Verify intended multirow lookup interception on a captured
task before attributing any candidate effect. Reuse the already repeated
unmodified baseline only when the execution profile and recorded scores agree.

Every candidate receives all 24 original tasks in fixed order. Reject any loss
of a baseline-correct answer or newly malformed response on a previously
well-formed item. A candidate must fix at least two development failures and
increase complete correct answers. Rank survivors by development success count,
then fewer rows, smaller absolute magnitude, direction order Rademacher /
alternating / hint contrast, and negative before positive. No logit-only win.

Confirm the one selected winner with fresh baseline/candidate runs and repeats
within a process. If confirmation fails, do not advance another candidate in
this round. Lock its overlay hash, then evaluate the original and candidate on
all 32 previously untested within-family holdout inputs plus eight controls.
Report paired gains/losses and require no baseline-correct preservation losses.
A failed holdout remains failed; do not retune on it. No broader reliability or
unseen-family claim follows from this small template-sharing test.

## Resources and loader qualification

Use one CPU model process at a time, eight threads, context 128, batch/microbatch
32, f32 caches, flash attention off, 80 GiB RSS cap, 24 GiB host reserve, and
900 seconds per process. Batch tasks per immutable overlay. No GPU, service,
model-file, or production configuration changes.

The full model-shard SHA256 checks are expensive. An optional, separately
recorded local loader extension may compute the three independent shard hashes
concurrently. It must retain all canonical-path, size, and complete-file digest
checks, pass negative authentication tests, and reproduce the unmodified and
zero-overlay generation controls. Record its source/binary hash in every run;
it must not alter inference operations or cache unchecked authentication.
