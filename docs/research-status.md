# Progress against the handoff

The project objective comes from [handoff.md](../handoff.md): improve unseen ordinary-prompt tasks through sparse changes to existing native memory rows, with the rest of the model frozen and the teacher absent at inference. The first experiment is reported in [REPORT.md](../REPORT.md).

The publication includes an installable offline response explorer, a model-free
overlay inspector, row-patch utilities, evidence checkers, and annotated
experimental source. It is not the complete system specified by the handoff.
Host orchestration and production activation are outside this repository's
current research scope.

| Handoff milestone | Current evidence | Status and remaining exit criteria |
|---|---|---|
| M0 — Audit and runnable baseline | Actual checkpoint/table metadata and hashes; actual-model diagnostic and synthetic tool runs; pinned source revisions | Partial. Exact model distribution/license and full dependency lock still need independent audit. Fresh portable build unqualified. |
| M1 — Native lookup and no-op overlay | Three CPU sequences, 39 tokens; 624 matching row accesses per traced condition; empty and original-row overlays exact; invalid overlays rejected | Narrow CPU checks passed. Broader packed/batched/multi-turn coverage and GPU parity remain. |
| M2 — Narrow differentiable edit | Exact unmodified native forward on 39 tokens; experiment 004 measures 19 edits of one existing row and validates a separate smooth local PLE derivative against its own finite differences | Full serving-gradient and useful finite-update qualification remain open. The native reference has no backward; these edits were not learned. |
| M3 — Verified curriculum and selection | Five evaluated families, 47 episodes including repeats; independent state replay; observed-row index and 512 provisional candidates | Partial. No repeatable reminder advantage, verified teacher corrections, or influence/retention-qualified row selection. |
| M4 — Complete short-sequence training | Shared-row sequence reference and loss/gate excerpts exist | Unqualified. No actual-model training, full-sequence gradient qualification, development improvement, or forward/backward resource benchmark. |
| M5 — Locked evaluation | Planned in the handoff | Not run. No locked learned candidate, training-seed comparison, sealed evaluation, or regression evidence. |
| M6 — Controlled serving | Not part of the present research publication | Not attempted. No learned overlay, cache/rollback qualification, or deployment claim. |

## Differences from the proposed experiment

The handoff proposes a 50–100-family engineering pilot and a larger substantive study. Experiment 001 evaluated only five families, each with four variants. Its results cannot establish the power or generality of the proposed study. The pilot's sixth family stayed outside the reminder loop; its answers are not included in the public archive.

The handoff's initial 128–256-token training range was a planning default. The sequence reference imposes a 128-token ceiling; experiment 003's validated capture profile permits up to 32 tokens in one prefill, and its full comparisons cover three sequences of 10, 17, and 12 tokens. No completed training sequence was optimized.

The initial reminder text was authored as part of the test harness. It was not obtained through the proposed teacher-correction pipeline. No teacher model was called. The stored partition name `train` is a planned split label, not evidence of actual training.

The handoff's proposed promotion margins are a five-percentage-point target gain and two-percentage-point preservation margin. The prototype's local release scaffolding used different exploratory defaults. Neither was exercised on a learned candidate. Final thresholds, family counts, and compute budgets must be fixed before any sealed evaluation; the publication does not silently adopt the prototype's defaults as the project's protocol.

The handoff's desired deliverable—an immutable overlay that improves unseen tasks—is **not yet achieved**. The first report records a prerequisite failure rather than marking that objective complete.

## Next research milestone

[Experiment 005](../experiments/005-behavior/REPORT.md) adds actual generated
answers and repeatability checks. A uniform format instruction changes the
small development assay from 1/16 contract-compliant correct answers to 14/16;
13 original failures were malformed completions under a four-token cap, so
this is not a general accuracy comparison. The remaining state-replacement
errors repeat, but their fixed content hint does not rescue them. Counting
also shows no hint benefit. No content target qualified, so no candidate row
search or held-out evaluation was run. The demonstrated instruction-following
contrast is a separate, narrower candidate for a future edit experiment.

[Experiment 004](../experiments/004-row-response/REPORT.md) makes the numerical
distinction concrete. Small changes to all 160 values of one row vanish at the
Q8_0 activation quantizer. The first changed ladder sample alters two FP16
scales, while all integer codes remain unchanged. A separate smooth derivative
passes its own finite differences but predicts the wrong sign for one measured
positive local step. Full-engine controls test selected finite updates with
fresh state and live routing. These are prefill response measurements, not
generated-answer scores or task improvements.

The [workbench](../examples/memory-edit-workbench/README.md) and overlay inspector
run without model dependencies; the wheel was installed and both commands
tested in a clean environment without NumPy or Torch. This qualifies those
portable tools, not a fresh build of the complete native inference stack.

[Experiment 003](../experiments/003-attention/REPORT.md) resolves the measured
CPU forward discrepancies on three short fixtures: all 48 residuals and every
output logit match bit-for-bit over 39 tested tokens. Rotary arithmetic, padded
attention width, stride-dependent CPU matrix dispatch, and PLE scalar arithmetic
were material. The original differentiable replica has not
thereby acquired a qualified gradient: these native operations have no backward
implementation, so M2 and M4 remain unqualified.

The first wider Unicode/chat control failed: 0.148-nat maximum selected-token error
and a new first difference in memory-bearing layer 1. Its exact memory gather
does not establish exact downstream memory processing. This retained control
limited the original ten-token success. A controlled change to the PLE scalar
expression then restored the complete 17-token forward, including matching
byte hashes for intermediates and logits. Final EOS/repeat and original
regression runs also pass with matching byte hashes. Wider execution modes remain open.

Expand coverage across independent rows, directions, and prompts, and test
whether a proposed surrogate recommends useful finite updates in the serving
engine. A local smooth derivative check alone does not qualify that surrogate.
In independent task work, find a reminder advantage that repeats under a fixed
protocol. These remain prerequisites for M2/M3.
