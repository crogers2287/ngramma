# Progress against the handoff

The project objective comes from [handoff.md](../handoff.md): improve unseen ordinary-prompt tasks through sparse changes to existing native memory rows, with the rest of the model frozen and the teacher absent at inference. The first experiment is reported in [REPORT.md](../REPORT.md).

The current publication is a report, evidence archive, replay checker, and annotated experimental source. It is not the complete system specified by the handoff. Host orchestration and production activation are outside this repository's current research scope.

| Handoff milestone | Current evidence | Status and remaining exit criteria |
|---|---|---|
| M0 — Audit and runnable baseline | Actual checkpoint/table metadata and hashes; actual-model diagnostic and synthetic tool runs; pinned source revisions | Partial. Exact model distribution/license and full dependency lock still need independent audit. Fresh portable build unqualified. |
| M1 — Native lookup and no-op overlay | Three CPU sequences, 39 tokens; 624 matching row accesses per traced condition; empty and original-row overlays exact; invalid overlays rejected | Narrow CPU checks passed. Broader packed/batched/multi-turn coverage and GPU parity remain. |
| M2 — Narrow differentiable edit | A manual perturbation reached the intended row and changed logits | Blocked at forward agreement. The perturbation was not learned and does not meet M2's gradient-validation exit criterion. |
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

[Experiment 002](../experiments/002-parity/REPORT.md) adds a configurable CPU
forward harness and exact first-mixer diagnostics. Its latest full-model control
still fails (8/10 top tokens; 2.704-nat maximum selected-token log-probability
error). Native operations have no backward implementation; M2 and M4 remain
unqualified. Consult that report for current measurements rather than treating
lower component error as model-level success.

Resolve the complete first-layer numerical path and expand full forward checks. Then validate directional gradients in stable-routing cases and investigate routing/quantization boundaries. In independent task work, find a reminder advantage that repeats under a fixed protocol. These are prerequisites for M2/M3, not a request to skip directly to broad training.
