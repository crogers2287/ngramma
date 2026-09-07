# Experiment 005: search for a useful existing-row edit

This protocol precedes the behavior runs. Experiments 003/004 qualified a
narrow forward and finite interventions, not a serving gradient or an improved
model. This experiment uses actual engine generations and programmatic answers.

## Discovery before optimization

Use four short, objectively graded A/B task families, balanced by answer and
described in `PLAN-task-design.md`. Run ordinary and procedural-hint prompts
through a fixed non-thinking chat template, with unconstrained greedy decoding.
No grammar, forced answer token, teacher call, or production service is used.
Repeat any promising ordinary/hinted contrast with freshly cleared state.
Keep every failure and malformed/truncated completion in the results.

Select a family only if at least two well-formed ordinary-prompt errors repeat
and are rescued by its hint without losses on previously correct items. If all
families succeed, expand discovery explicitly, before any memory optimization.
Do not quietly count wording variants as independent problem families.

## Candidate search

The baseline and a replacement-with-original-values control must agree. Check
the extended lens against the previous exact prefill fixture and repeat tasks
within one process to catch state leakage. Generation uses the actual engine's
live attention and expert choices; no replica attention approximation is used.

Choose content-associated existing trigram rows used across the selected
development prompts. Avoid universal chat delimiters. Record collisions,
coverage, original-row hashes, direction seeds, magnitudes, and model identity.
The initial row budget is at most eight rows (one observed trigram's heads).
Keep all other model weights and table rows fixed. Never write model files.

Because the serving path has no qualified backward, the first optimizer is
finite candidate search in the actual quantized engine. Predeclare the candidate
directions and magnitudes in a separate search record **before** observing their
scores. Limit the first search to twelve candidate overlays. This is a small
derivative-free adaptation experiment, not full-sequence gradient training.

Select only on discovery/development generations and diagnostics. Require fewer
actual wrong answers, not merely higher correct-token probability. Reject any
candidate that damages the development preservation controls. Freeze its hash
before evaluating held-out inputs. Keep the original sixth pilot family sealed.

## Evaluation and claims

Evaluate the chosen candidate and the original on new inputs, including answer
reversals, plus unrelated controls. A small within-family gain is exploratory
evidence, not established broad intelligence or unseen-family generalization.
Report exact correct/wrong/malformed counts, baseline/candidate paired changes,
prompt and row overlap, termination, hashes, and compute. If no candidate passes,
publish the negative result and do not promote any overlay.

## Resource limits

Run one isolated CPU model process at a time: eight threads, 128-token context,
32-token batches, f32 caches, flash attention off, 64 MiB n-gram cache, no GPU.
Each process is limited to 900 seconds, 64 GiB RSS, and a 24 GiB available-RAM
reserve. Batch multiple freshly reset tasks per immutable candidate to amortize
the native loader's full-shard authentication. Store raw local outputs under
ignored `.local/005`; publish small records and reproduction code, not weights.

### Resource amendment before any task outcome

The first monitored process stopped before readiness at 70,225,825,792 bytes
RSS (65.4 GiB), with 126,394,576,896 bytes of host memory still available. Its
record is retained as `discovery-allocation-stop.json`. The mapped model alone
exceeds the proposed 64 GiB limit. Raise the engine RSS cap to **80 GiB** while
keeping the 24 GiB reserve, CPU/thread settings, and 900-second limit. This
changes the resource admission bound, not the tasks, scoring, or search gate.
