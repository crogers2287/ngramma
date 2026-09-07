# Experiment 002: locate the remaining forward mismatch

Status: first diagnostic cycle completed; full-model numerical gate still fails.
Layers 0–2 match bit-for-bit on the original fixture. The next cycle starts at
layer 3. Historical experiment 001 evidence stays unchanged.

The first experiment matched native memory reads, but its full sequence reference
still differed from inference. This experiment first locates the earliest
remaining discrepancy in a complete layer, then tests proposed corrections
against the full model. It does not begin by optimizing rows.

1. Reconstruct a runnable, configurable research harness and capture layer-0
   intermediate tensors from the recorded CPU engine using the same ten tokens.
2. Compare every hyper-connection stage and the recurrent/expert stages, both
   with exact reference inputs and with chained replica inputs. Measure whether
   activation rounding or reduction order amplifies the earliest difference.
3. Test each correction separately. Keep raw tensor captures local; publish
   scripts, input IDs, source identities, numerical summaries, and failed trials.
4. Repeat full 48-layer comparisons and expand short-sequence coverage only if
   component evidence supports a correction. Retain the existing 0.02-nat and
   1%-RMS diagnostic gates; do not relax them to obtain a pass.
5. Run actual-model directional gradient checks only after forward qualification.
   A surrogate gradient and a lower loss are not sufficient evidence.

The initial resource inventory permits CPU diagnostics while existing GPU
services remain untouched. Run at most one full-model diagnostic at a time,
with four replica threads and eight engine threads. Keep decoded-weight caches
bounded, inspect available RAM before runs, and cap each diagnostic at 15 minutes.
Local raw artifacts are excluded from Git. Source and auditable summaries are
committed to this repository after each verified step.

Independent AI code reviews assist diagnosis; their outputs are not student
training examples. No teacher corrections or sealed-task answers enter this
experiment. The handoff's behavioral hypothesis remains the research objective.
