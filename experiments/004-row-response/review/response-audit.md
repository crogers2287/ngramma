# Independent audit of experiment 004 local response and engine-check design

Reviewed `PLAN.md`, `probe.py`, `row_patch.py`, `engine_check.py`, `capture_reference.py`, and saved `response.json`. This review ran no models and changed none of those sources. Full-engine checks were still running at review time; this note does not attest to their eventual results.

**No blocker found for the measured local response or the selected full-engine checks.** The experiment measures a finite response of one existing row in one fixed direction. It neither implements nor qualifies a serving gradient.

## Isolation and numerical controls

The probe fixes token index 6, head 8, alternating direction scaled by the original row RMS, and the epsilon ladder before edited forwards. It checks all offline gathered values against the engine's gathered memory, obtains all occurrences by global address, and applies the replacement at each occurrence. The saved fixture has exactly one occurrence, `[6,8]`. `RowPatch` freezes its anchor/replacement arrays, checks the gathered anchor, exports exactly one row, refuses overwriting an existing overlay directory, and binds the model identity and payload hashes. The engine driver checks overlay hashes again and checks gathered replacements across the entire memory tensor, not only the selected location.

The local PLE uses the captured layer-0 residual, the corrected FP32 scale coefficient, and the qualified native forward operations. Its zero output must equal the reconstructed engine PLE output. The full-engine driver independently requires the earlier layer-0 residual to stay equal, checks the full PLE output hash against each local result, verifies a fresh unmodified baseline, and checks zero-overlay equivalence. Every selected overlay runs in a fresh process through the normal engine attention and routing paths. The driver's unchanged-projection/changed-logits check is a useful explicit falsification of isolation.

The smooth control is an explicitly different function: FP64 PLE with dequantized weights and no activation quantization. Its scalar, perturbation direction, and central differences match its own autograd evaluation. The derivative is `-0.00011579135462728632`. Relative central-difference error ranges from about `3.04e-11` in the central part of the small-step ladder to `1.00e-5` at the largest step. This validates the smooth control's directional differentiation; it does not validate that derivative for the native quantized forward. The local scalar is not the final logit margin, so these two responses must not be compared as if they had the same objective.

## What the saved response establishes

All 160 replacement coordinates change at every nonzero epsilon, including the smallest `2^-20`. Thus the unchanged native result through the sampled `±2^-12` steps is not explained by the FP32 row replacement swallowing the edits. Complete replayed activation bytes, key/value projection values, and the local output remain unchanged at those sampled points. This establishes sampled unchanged responses, not proof that every intermediate real-valued epsilon lies on one connected plateau.

The first changed magnitude in the prescribed ladder is `2^-10`. Each sign changes exactly **two scale bytes and zero integer-code bytes**, while key/value projections change. Byte counts are not block counts; the saved summary alone does not identify how many distinct FP16 scales changed. The accounting checks: total changed activation bytes equal changed scale plus changed code bytes for every point.

At `-2^-10`, the local scalar change is `2.561137080192566e-8`; at `+2^-10`, it is `2.7008354663848877e-8`. Both are positive. A negative smooth derivative predicts a positive change for a negative epsilon, so it is incorrect to say both sides contradict the smooth gradient's sign. The positive-epsilon side contradicts that first-order sign prediction. The native symmetric finite-step secant is `+7.152557373046875e-7`, versus the negative smooth derivative. Neither that secant nor its sign is evidence of derivative convergence. At `2^-8`, the native symmetric secant is `+7.94529914855957e-5` and integer codes also change.

Selection matches the predeclared rule: largest sampled unchanged magnitude `2^-12`, first changed magnitude `2^-10`, next larger magnitude `2^-8`, both signs plus zero. There is no response-based direction or scalar selection in the inspected code. Gate-sign changes are zero throughout, but the probe does not record clamp-branch crossings or downstream routing selections. Do not claim those were measured or held invariant. The engine uses live routing; that is different from documenting which routes changed.

## Activation semantics and evidence scope

Captured PLE weights are checked as Q8_0 in ordinary CPU_Mapped buffers. The isolated encoder calls the actual linked CPU activation trait. Initial llamafile dispatch rejects F32 activations for Q8_0 weights; conversion then creates the ordinary Q8_0 blocks consumed by either the second llamafile attempt or fallback dots. Each block includes FP16 scale bytes and 32 integer codes, with no auxiliary sums. Both PLE projections receive the same concatenated activation vector and therefore share that encoding.

`response.json` correctly labels this as dispatched-encoder replay, not graph workspace capture. I independently checked that its activation library hash matches the isolated build record, and that the activation and native forward build records bind identical CPU/base library hashes. No model execution was needed for those checks.

## Small evidence hardening items before final publication

1. `engine_check.py:112–118` assigns `logits_bitwise_equal_to_original` using NumPy numeric equality. That treats positive and negative zero as equal. Before publishing literal byte equality, compare each saved logits SHA256 to the original SHA256; the capture summaries already contain the necessary hashes. PLE equality already uses hashes. This does not show that any actual signed-zero mismatch occurred.
2. The response records the activation library hash but omits its companion build record (`probe.py:219–220`). Preserve that record with the public evidence or a sanitized provenance companion, binding the hash as checked above. This can be done after execution without changing the experiment.
3. The engine driver hashes its runtime libraries and scripts across the whole run but does not enforce one lens hash across all captures or against the original (`engine_check.py:131–138`). Each capture does record its lens hash. Audit that all selected records have the same lens identity and configuration before aggregating them. The frozen-source run gives no reason to expect a mismatch; this is a missing assertion, not an observed mixed-engine result.

The saved interpretation's `finite_step`, `training_admitted=false`, and explicit denial of task improvement or serving-gradient qualification are appropriate. Keep the first-change statement scoped to this ladder, row, direction, fixture, and CPU configuration. Full-engine conclusions require the remaining independent engine records; a logit response alone is not an answer-quality improvement.

## Follow-up: individual scale transitions replayed from saved inputs

The isolated `scale_transition.py` replay reads only the captured PLE activation and three saved overlays. All three complete activation hashes reproduce `response.json`. `scale-transition-final.json` binds the script, wrapper, capture metadata/data, response, spec, overlays, activation library/build record, and CPU library; the earlier `scale-transition.json` and `scale-transition-maxima.json` are retained as preliminary audit controls. The final record explicitly includes baseline and both edited encoding hashes.

At token 6, both signs change two distinct scales, in global blocks 40 and 41 (head 8, local blocks 0 and 1). Each changes exactly one scale byte. All integer codes remain unchanged. Block 40's FP16 bits move `0778 → 0779` for the negative edit and `0778 → 0777` for the positive edit. Block 41 moves `06d2 → 06d3` for **both** signs. Each step is one FP16 ULP, `5.960464477539063e-8`. The JSON includes complete before/after scale values and bits.

The saved unedited block 41 has equal positive absolute maxima at local coordinates 0 and 31. The alternating positive edit raises coordinate 0; the negative edit raises coordinate 31. Consequently its maximum magnitude, and here its stored scale, increase for both directions. Block 40 has one positive maximizer at coordinate 13 and moves in opposite directions. These are direct saved-input/encoder observations. The resulting encoded perturbations are not negatives of each other.

**Inference:** this shared scale increase makes a same-sign local scalar response compatible with the opposite raw row edits. The activation-only audit does not determine how much each changed scale contributes through projection weights and subsequent nonlinear operations; it does not prove that block 41 alone causes the measured scalar sign. No model forward or shard read was performed for this follow-up.
