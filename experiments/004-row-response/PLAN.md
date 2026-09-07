# Experiment 004: one row, measured response

This plan is recorded before inspecting perturbation results. It follows
experiment 003's exact unmodified CPU forwards. It is a numerical experiment,
not a training run or a capability benchmark.

## Fixed intervention

- Fixture: experiment 003's 17-token Unicode/chat sequence, fresh prefill.
- Select the existing trigram row at token index 6, head 8. Apply the same edit
  at **every occurrence** of that global address. Record all occurrences.
- Direction: coordinates alternate +1, -1 starting with +1, multiplied by the
  original decoded row's FP64-computed RMS and rounded to FP32. No search for a
  direction that improves the result.
- Signed epsilon ladder: zero and both signs of 2^-20, 2^-18, 2^-16, 2^-14,
  2^-12, 2^-10, 2^-8, 2^-6, 2^-4. Each overlay replaces one existing row with
  FP32(original + epsilon * direction); checkpoint files remain read-only.
- Local scalar: PLE output at [token 6, stream 0, coordinate 0]. Full-engine
  scalar: final-position logit margin of the unmodified top token over its
  runner-up. Token IDs are fixed from the baseline before edited forwards.

## Controls and measurements

First require the gathered baseline and the native local PLE output to match
the engine capture exactly. Compare a smooth FP64 PLE derivative against
central differences of that same smooth function. This validates only the
local smooth reference, not a derivative of the quantized serving engine.

For every epsilon, record changed key/value projection elements, PLE output
RMS change, local scalar response, and complete actual activation-encoding
bytes where the engine dispatch has been established. Include scale bytes;
integer codes alone are insufficient to establish a plateau.

Choose at most three positive magnitudes for full engine checks by this rule:
the largest magnitude where both signs leave both projections unchanged
(if one exists); the first magnitude where either sign changes a projection;
and its next larger magnitude. If no plateau exists, use the first changed
magnitude, its next larger magnitude, and the largest ladder magnitude.
Deduplicate. Run both signs and a zero overlay in separate fresh processes.
Run the unmodified engine once too. Actual engine routing and attention
selection remain live. No replica attention-support substitution is used.

Every engine overlay must pass the existing loader's complete shard checks.
Require exact zero-overlay equivalence. Check gathered replacements, PLE
outputs, final logits, and the predeclared margin; retain all selected results,
including negative or unchanged responses. An effect on logits is not a useful
answer improvement.

## Reuse and bounds

Ship a small reusable row-patch utility, a runnable response probe, saved
measurements, and a standard-library HTML explorer with a clearly labeled
synthetic example. Readers can inspect results without downloading weights.

One CPU model process at a time; 4 replica / 8 engine threads; 2 GiB decoded
weight cache; 900-second per-process limit. No service or GPU changes. Raw
captures, original row values, local manifests, and overlay binaries stay in
ignored `.local/004`. Preserve original `handoff.md` and failed controls.
