# Progress against the handoff

The project objective comes from [handoff.md](../handoff.md): improve unseen ordinary-prompt tasks through sparse changes to existing native memory rows, with the rest of the model frozen and the teacher absent at inference. The first experiment is reported in [REPORT.md](../REPORT.md).

The current publication is a report, evidence archive, replay checker, and annotated experimental source. It is not the complete system specified by the handoff. Host orchestration and production activation are outside this repository's current research scope.

| Handoff milestone | Current evidence | Status and remaining exit criteria |
|---|---|---|
| M0 — Audit and runnable baseline | Actual checkpoint/table metadata and hashes; actual-model diagnostic and synthetic tool runs; pinned source revisions | Partial. Exact model distribution/license and full dependency lock still need independent audit. Fresh portable build unqualified. |
| M1 — Native lookup and no-op overlay | Three CPU sequences, 39 tokens; 624 matching row accesses per traced condition; empty and original-row overlays exact; invalid overlays rejected | Narrow CPU checks passed. Broader packed/batched/multi-turn coverage and GPU parity remain. |
| M2 — Narrow differentiable edit | A manual perturbation reached the intended row and changed logits; native forward-only reference now matches all 48 layers on the original short fixture | Blocked at differentiable-forward and gradient qualification. The native reference has no backward implementation; the perturbation was not learned. |
| M3 — Verified curriculum and selection | Five evaluated families, 47 episodes including repeats; independent state replay; observed-row index and 512 provisional candidates | Partial. No repeatable reminder advantage, verified teacher corrections, or influence/retention-qualified row selection. |
| M4 — Complete short-sequence training | Shared-row sequence reference and loss/gate excerpts exist | Unqualified. No actual-model training, finite-difference qualification, development improvement, or forward/backward resource benchmark. |
| M5 — Locked evaluation | Planned in the handoff | Not run. No locked learned candidate, training-seed comparison, sealed evaluation, or regression evidence. |
| M6 — Controlled serving | Not part of the present research publication | Not attempted. No learned overlay, cache/rollback qualification, or deployment claim. |

## Differences from the proposed experiment

The handoff proposes a 50–100-family engineering pilot and a larger substantive study. Experiment 001 evaluated only five families, each with four variants. Its results cannot establish the power or generality of the proposed study. The pilot's sixth family stayed outside the reminder loop; its answers are not included in the public archive.

The handoff's initial 128–256-token training range was a planning default. The sequence reference imposes a 128-token ceiling; the full numerical comparison covered only ten tokens. No completed training sequence was optimized.

The initial reminder text was authored as part of the test harness. It was not obtained through the proposed teacher-correction pipeline. No teacher model was called. The stored partition name `train` is a planned split label, not evidence of actual training.

The handoff's proposed promotion margins are a five-percentage-point target gain and two-percentage-point preservation margin. The prototype's local release scaffolding used different exploratory defaults. Neither was exercised on a learned candidate. Final thresholds, family counts, and compute budgets must be fixed before any sealed evaluation; the publication does not silently adopt the prototype's defaults as the project's protocol.

The handoff's desired deliverable—an immutable overlay that improves unseen tasks—is **not yet achieved**. The first report records a prerequisite failure rather than marking that objective complete.

## Next research milestone

[Experiment 003](../experiments/003-attention/REPORT.md) resolves the original
ten-token CPU forward discrepancy: all 48 residuals and every output logit now
match bit-for-bit. Rotary arithmetic, padded attention width, and stride-dependent
CPU matrix dispatch were material. The original differentiable replica has not
thereby acquired a qualified gradient: these native operations have no backward
implementation, so M2 and M4 remain unqualified.

The first wider Unicode/chat test fails: 0.148-nat maximum selected-token error
and a new first difference in memory-bearing layer 1. Its exact memory gather
does not establish exact downstream memory processing. This retained control
limits the original ten-token success; broader forward agreement remains open.

Expand short-sequence coverage, then compare a differentiable implementation
against the exact reference and validate directional gradients. Distinguish
stable routing from routing/quantization boundaries. In independent task work,
find a reminder advantage that repeats under a fixed protocol. These remain
prerequisites for M2/M3.
