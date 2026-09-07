# Layer-0 numerical review

The existing exact-input component checks do not establish chained parity. This review found a concrete amplification mechanism on the recorded activations: one-ULP input changes can cross activation-quantization thresholds and produce changes many orders larger. It did not run a model forward, locate the actual first crossing in the replica, or establish how much of the recorded 0.0014566 relative RMS error this mechanism explains.

## Fixture-only results

`fixture_arithmetic.py` reads the saved layer-0 activation tensors and calls the existing CPU activation-quantization helper. It does not instantiate a model or access weights. Results are in `fixture-arithmetic.json`.

| Probe | Result |
|---|---|
| First HC mixed activation, every element moved one ULP toward positive infinity, Q8_0 roundtrip | Input max change 4.7684e-7; 64 rounded values change, maximum 0.000968933 |
| Second (FFN) HC mixed activation, same perturbation | Input max change 2.3842e-7; two rounded values change, maximum 0.011062622 |
| Both activations, one ULP toward negative infinity | No rounded value changes |
| Final residual reconstructed from captured embedding, attention and FFN outputs, fitting two scalar coefficients per token/stream | Maximum error 9.3132e-10; inferred scatter coefficients range 0.000219789–0.7130863 |
| Initial embedding squared-mean reduction: FP32 torch mean versus FP64 accumulation of FP32 squares, cast back | 28 of 40 means differ; maximum difference 1.8190e-12 |

The one-ULP perturbation is a sensitivity experiment, not the measured replica error direction. The fitted scatter coefficients are not captured injection logits; this demonstrates consistency of the saved output with scalar HC scattering, not parity of replica injection projections. The normalized-embedding comparison in the JSON uses explicitly illustrative epsilon 1e-6.

## Concrete source differences and unresolved boundaries

1. **RMSNorm arithmetic differs before the first quantized projection.** The actual replica's `vendor/engraft/replica/layers.py:46–52` computes FP32 squared mean and `torch.rsqrt`. Engine `ggml/src/ggml-cpu/ops.cpp:3975–3982` accumulates FP32 products in `ggml_float` (double, defined in `ggml/src/ggml-cpu/vec.h:15`), casts the mean to float, then uses `1.0f/sqrtf(mean + eps)`. This is a real operation-order/precision difference; the fixture confirms unequal mean values. A candidate local reference correction is FP64 accumulation of already-FP32 squares, FP32 mean, FP32 sqrt followed by reciprocal, then the two multiplications in engine order. Calling the engine normalization kernel would be a stronger diagnostic reference than assuming PyTorch sqrt matches the compiled kernel.

2. **Activation rounding does not reproduce the quantized dot product.** `reference/flash_memory/activation_reference.py:36–42` substitutes dequantized rounded activations and calls the original float `mm`/`mv`. `reference/engine/dequant.cpp:16–34` performs only quantization/dequantization. For example, engine `ggml/src/ggml-cpu/quants.c:451–480` expresses the generic Q8_0 dot product as integer block sums multiplied by block-scale products, then float accumulation. Dequantized float GEMM has different product rounding and reduction order. The active architecture-specific kernel must be identified before claiming bit parity with that generic implementation. A forward-only wrapper calling the matching engine dot kernel on original quantized rows would isolate this remaining difference.

3. **HC stream collapse uses different reduction expressions, but the new fixture rules it out at exact input.** Replica `layers.py:101–104` uses `gated.mean(dim=1)`; engine `src/models/qwen4exp.cpp:383–394` adds streams in explicit order and then scales. Follow-up captured `hc_norm` and `hc_gate` values reproduce both engine `hc_mixed` outputs bit-identically using either torch mean or explicit left-associated sums. There is no demonstrated collapse-order error on these fixtures.

4. **Injection and the second mixer are unqualified by the published component checks.** `reference/scripts/layer0_probe.py:33–52` checks first `hc_mix` output but discards the returned injection, then supplies exact attention input and exact second mixer output to attention and MoE respectively. It never checks either injection projection, either HC combine, or the FFN HC mixer with exact post-attention residual. Engine already labels `hc_norm`, `hc_gate`, `hc_inject`, and `hc_combine` in `qwen4exp.cpp:373–421`. The original recorded fixture lacks those intermediates; a follow-up capture enabled the exact-input checks below. Scalar combine formulas themselves agree: replica `layers.py:111–120`, engine `qwen4exp.cpp:404–423`.

5. **Recurrent arithmetic/state needs a separate, lower-priority qualification.** Replica `layers.py:302–333` always iterates the autoregressive recurrence. Engine `src/models/delta-net-base.cpp:425–447` selects autoregressive, chunked, or fused implementation using token count and runtime flags; `:527–565` returns/stores final state. The current 10-token attention check compares only output, not final recurrent state or convolution history. `reference/flash_memory/sequence.py:68–81` starts every layer with a fresh `LayerState`, consistent with evaluating a whole fresh sequence. There is no evidence here of stale-state reuse causing layer-0 drift. Chunk-size 1 versus 10 on identical fresh inputs, plus comparison of `new_state`, `state_predelta`, convolution history, and per-position attention output, would distinguish numerical recurrence differences from state handling errors.

## Quantization-hook coverage review

The hook supports only `aten.mm` and `aten.mv` and looks up the weight's starting data pointer (`activation_reference.py:24–42`). The inspected direct weighted products use these operators and unshifted weight/transposed views. No demonstrated current coverage omission was found. Copies, nonzero-offset views, `addmm`, and disk-loaded expert arrays could bypass this registry if introduced. Inherited `vendor/engraft/replica/weights.py:100–109` can load persisted experts through `np.load`, bypassing `EngineWeights.decode` registration, but the current `EngineWeights` constructor does not enable a disk cache, so this is not an explanation of the recorded run. Log operator, stable tensor name/type, registry hit, input shape, and quantized result at every weighted product rather than relying solely on a total call count.

## Follow-up: exact HC combines and nonlinear kernels

`hc_exact_fixture.py` reads the expanded follow-up capture. It uses captured residuals, attention/FFN outputs, and injection logits, never weights. Results are in `hc-exact-fixture.json`.

| Exact-input probe | Maximum error versus engine | Unequal elements |
|---|---:|---:|
| Attention HC combine using torch sigmoid | 7.4506e-9 | 4,969 |
| Attention HC combine using scalar libm expf sigmoid | 0 | 0 |
| FFN HC combine using torch sigmoid | 1.8626e-9 | 2,100 |
| FFN HC combine using scalar libm expf sigmoid | 0 | 0 |
| Both HC collapses using torch mean | 0 | 0 |

Engine `ggml/src/ggml-cpu/vec.h:936` implements scalar sigmoid as `1.f / (1.f + expf(-x))`. A scalar libm implementation with explicit FP32 operations reproduces both captured HC combines exactly. Torch sigmoid does not. The discrepancy is tiny before any downstream quantization; these results do not establish its contribution to the complete layer drift.

Engine SiLU differs as well: `ggml/src/ggml-cpu/ops.cpp:2628–2630` invokes `ggml_vec_silu_f32`, whose architecture-specific loops are in `ggml/src/ggml-cpu/vec.cpp:380–417`. That symbol is exported by the existing CPU library and callable through ctypes. On a synthetic 4,096-element FP32 grid from -16 to 16, its output differs from torch SiLU at 1,012 elements, maximum 4.7684e-7. This is kernel evidence, not a measurement of actual HC low-rank projection activations. Use the native vector function with the same row width as the engine: SIMD/tail treatment can depend on width.

The smallest useful forward-only bridge candidates are: (a) native rowwise `ggml_vec_silu_f32`; (b) sigmoid via the compiled engine scalar expression, which is fixture-qualified here; (c) engine RMSNorm or a clone retaining FP32 square, double accumulation, FP32 mean, reciprocal sqrtf, and ordered scale/weight multiplies. These can replace one primitive at a time before implementing a larger graph or quantized matmul bridge. None supplies an admitted backward derivative.

## Minimal causal probe order

1. Capture both engine HC slots including norm, gate, inject and combine; retain exact pre-quantization inputs, quantized block bytes/scales, and outputs for each named weight operation.
2. Run the chained layer while stopping at the first differing quantized byte/scale. Compare raw FP32 difference and rounded activation difference separately. Stable routing alone does not constrain expert activations or softmax weights.
3. Patch only RMSNorm arithmetic and explicit HC reduction in an isolated reference; repeat the same sequence. If a difference remains before any activation-rounding divergence, replace that matmul with the actual engine dot kernel for diagnosis.
4. Compare both HC combines from exact recorded residual, block output, and injection; then test the second mixer and MoE with replica inputs. This avoids attributing all unmeasured paths to the attention block.
5. After layer-0 qualification, check final recurrent state and chunked sequence continuation before broad full-model parity. Do not admit training from the fixture sensitivity test alone.

All work in this review was read-only outside the review directory. No model weights were loaded, GPU operations invoked, services changed, or commits created.

## Follow-up: recurrence after native elementwise and matrix primitives

The subsequent `operator-probe-native-all.json` reports exact first HC stages, exact second HC stages from exact inputs, and exact combines, while recurrent attention still differs (maximum 0.00059408, relative RMS 0.00015759). That localizes the remaining measured difference inside the attention block under this primitive configuration. It does not isolate an individual operation. Whole-layer relative RMS in that artifact is 0.00094779.

The smallest additional native primitive supported by source evidence is **the fused gated delta net operation**, preceded by **SSM convolution** if the raw-convolution capture differs. Replacing only elementwise functions and weight matmuls leaves these composite operations numerically different:

| Stage | Engine | Replica | Consequence |
|---|---|---|---|
| Query scale | Fused GDN dots state with unscaled q, then multiplies dot result by `1/sqrtf(D)` (`ggml/src/ggml-cpu/ops.cpp:11073–11077`) | Scales q before dot (`vendor/engraft/replica/layers.py:317–330`) | Different multiplication/reduction order |
| State–key and state–query contractions | `ggml_vec_dot_f32` (`ops.cpp:11059–11077`) | Torch einsum, lowering to batched contractions | Native weight-matmul hook does not replace these contractions |
| Outer-product state update | `ggml_vec_mad_f32` (`ops.cpp:11067–11069`); SIMD FMA at `ggml/src/ggml-cpu/vec.h:416–425` | Separate multiplication and addition | Different rounding, even with identical q/k/v/g/beta |
| Decay | Fused kernel calls scalar `expf` (`ops.cpp:11053`) | Native exp graph when intercepted | Both represent exp, but preserving the composite native kernel avoids assumptions about equivalent dispatched implementations |
| Convolution | Ordered scalar FP32 `sumf += s*c` (`ops.cpp:9781–9789`) | Explicit tensor product followed by in-place add (`layers.py:264–272`) | Same mathematical tap order; compiler contraction may still differ. Need raw-output fixture comparison before attributing error |

This is the default engine path: `src/llama-context.cpp:232–234` initializes fused GDN AR/CH enabled and automatic selection disabled; `src/models/delta-net-base.cpp:425–447` selects the fused operation for a multi-token sequence when CH is enabled. Capture/probe configuration should still record actual flags. If a run overrides the fused choice, match its chunked graph rather than silently substituting the fused operation.

**Native GDN bridge contract:** q/k/v have GGML dimensions `[D,H,T,1]` (contiguous Torch `[T,H,D]`), g and beta `[1,H,T,1]`, and initial state `[D,D,H,1]`. The kernel stores state as contiguous rows indexed `state[value,key]` (`ops.cpp:11037–11038`), whereas the replica stores `[head,key,value]`. Transpose the last two Torch state dimensions before the call and transpose the returned state back. Use K=1 for a final state without snapshots. Return both the entire output sequence and the updated state. The `ggml_gated_delta_net` primitive preserves reduction, FMA, scale placement, and decay together; a local Python reordering of q scaling alone cannot guarantee that result.

Softplus and the post-recurrence output gate do not show a remaining structural mismatch: engine `ggml/src/ggml-cpu/unary-ops.cpp:80–82` uses x>20 passthrough and `logf(1+expf(x))`, which the new native softplus primitive calls. `src/models/qwen4exp.cpp:683–692` performs weighted RMSNorm followed by multiplication by sigmoid(z), matching the replica ordering. Capture `alpha`, `a_softplus`, `gate`, `beta_sigmoid`, `conv_output_raw`, `q_conv_predelta`, `k_conv_predelta`, `v_conv_predelta`, `attn_output`, `new_state`, and `final_output` to identify the first divergence in that order.

Native hook review: `src/ngramma_runtime/native_forward.py` intercepts registered weight `mm`/`mv`, plus selected unary operations and named normalization functions. The three recurrence einsums remain outside its native matmul path. Its weak-reference registry checks array liveness; torch tensors made with `from_numpy` retain the backing array, so ordinary weight transposes remain valid registrations. No evidence of a weak-reference liveness failure was found. Registry misses should remain observable. A whole native graph is still sensitive to tensor shape/layout and available fused kernels; batching expert calls into new shapes must be qualified rather than presumed equivalent.

## Follow-up: CPU matrix buffer dispatch

The later attention diagnostic reports that Q4_0 QKV projection is already nonexact from exact input (maximum 9.54e-6, relative RMS approximately 1.58e-7). Before attributing downstream recurrence error, match the engine's matrix buffer type and packed kernel.

The native bridge `src/ngramma_runtime/native/ggml_forward.cpp:94–99` initially assigns original weight bytes directly to `weights->data`. This leaves `weights->buffer` and `weights->extra` unset. That is not equivalent to a model-loaded CPU tensor when the model selects an extra buffer type:

1. `src/llama-model.cpp:1058–1084` puts available extra CPU buffer types before the ordinary CPU buffer; `:2700` defaults `use_extra_bufts` to true.
2. `ggml/src/ggml-cpu/repack.cpp:4573–4577` selects Q4_0 8x8 packing when AVX2 is available and the output dimension is divisible by eight. Other architectures/types have other selections; inspect the actual tensor's buffer type.
3. Repack-buffer initialization attaches tensor traits (`repack.cpp:4726–4729`); `ggml_backend_tensor_set` invokes its repacking upload callback (`:4733–4742`). Assigning original bytes directly bypasses both.
4. CPU `ggml_graph_compute` already checks extra-buffer dispatch (`ggml/src/ggml-cpu/ggml-cpu.c:1918–1920`); repack selection requires the weight buffer type and extra traits (`repack.cpp:4810–4814`). Switching only from direct graph compute to backend graph compute cannot repair missing weight metadata and packing.
5. Packed matmul uses GEMM for input-row chunks longer than three and GEMV for remaining rows (`repack.cpp:4239–4252`). Preserve the reference token batch and tensor shape. A collection of single-row dots need not reproduce a ten-row graph.

**Concrete bridge change to test:** record the reference buffer type for each weight. Allocate the weight in a separate GGML context through `ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, selected_buft)`, then upload original bytes with `ggml_backend_tensor_set(weights, raw, 0, ggml_nbytes(weights))`. Keep the existing host F32 input/output and original batch dimensions. Existing graph planning and compute can then see the packed traits and their workspace requirements. Confirm support before uploading; unsupported types must use their recorded ordinary buffer, not be forced into CPU_REPACK. Discover available CPU extra types through the registered `ggml_backend_dev_get_extra_bufts` procedure when avoiding private repack headers.

This is a source-supported mechanism and a concrete discrimination test, not proof that the saved reference used CPU_REPACK. Capture `ggml_backend_buffer_name(weight->buffer)` or the buffer-type name for QKV and gate weights before declaring repacking the cause. A plain-buffer reference would require investigation of another dispatch difference.
