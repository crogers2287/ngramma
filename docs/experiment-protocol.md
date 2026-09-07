# Experiment protocol and next-run requirements

The mission and complete proposed method are in [handoff.md](../handoff.md). Experiment 001 tested prerequisites and a small baseline pilot. This document records the performed protocol separately from future requirements.

## Performed protocol

The exact checkpoint and runtime identity were recorded before overlay comparisons. The original table was read-only; the tested interventions were an empty overlay, replacement of 152 rows with their decoded original values, and a one-component manual perturbation in one existing row. CPU checks used no speculative draft, f32 caches, and the exact token sequences in [compatibility-cases.json](../data/compatibility-cases.json).

The full replica comparison used one ten-token sequence. At each position, the engine's top token selected the log-probability being compared. Thresholds were below 0.02 nats maximum selected-token error and below 1% maximum relative RMS across captured intermediates. Both full references failed; no real-model gradient run followed. Exact-input component diagnostics were recorded separately from the complete sequence comparison.

For the tool pilot, five scenario families generated four variants each. Every variant received ordinary and hinted attempts, with one corrected retry after the sole ordinary failure. Success was graded by mock state, inventory inspection and tool errors. The initial runner used temperature zero, seed 1234, no thinking, a 256-token output limit per response, and at most eight response rounds. It allowed prompt-cache reuse and generated tool-call IDs.

Confirmation repeated the failed task three times per condition, in alternating order, with cache reuse disabled and canonical tool-call IDs. These settings differ from the initial pilot. Both conditions passed all repeats; attribution to any one changed setting is not possible. The public replay recomputes saved actions and grades; it does not sample the model again.

The pilot's reminder text was a fixed harness instruction, not a teacher-produced correction. Expected states and evaluated tasks are public. Their stored split labels do not make them sealed or training data.

## Requirements before learning and evaluation

1. Record a clean build, dependency lock, precise checkpoint distribution/license, and broader forward agreement. Establish actual-model directional-gradient evidence before admitting the sequence optimizer.
2. Fix generation/cache/tool-ID conditions, find repeatable ordinary-versus-guided differences across independent development families, and build a separately representative preservation population.
3. Verify a replaceable teacher's model access and permitted intended use before producing training corrections; independently replay or test every accepted correction. Keep ordinary student inputs free of the successful reminder.
4. Predeclare row selection and budgets, training/resource limits, and family-level statistics. Include collision-sharing preservation examples and an alternative row-selection control. Keep sealed answers outside curriculum and selection.
5. Lock a candidate before unseen evaluation. Compare actual generated task outcomes under identical constraints, report regressions and uncertainty, and retain failed/inconclusive experiments without promotion.

The handoff's five-percentage-point gain and two-percentage-point preservation margin remain planning defaults. Determine sample size and finalize thresholds before the sealed run. No outcome in experiment 001 satisfies the central skill-transfer criterion.
