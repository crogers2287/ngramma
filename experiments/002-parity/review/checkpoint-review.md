# Checkpoint review: native repacking and remaining MoE error

Read-only review; no model execution. The saved repack operator result now reports exact attention, exact second HC mixer, FFN maximum error 8.9407e-8, and final layer relative RMS 5.6871e-8 (maximum 2.9802e-8). The ancestor capture confirms QKV and shared-expert Q4_0 weights use CPU_REPACK; expert IQ3_S/Q2_0 and HC weights use CPU_Mapped. These findings supersede the earlier unconfirmed-buffer hypothesis.

## Material boundary risk

`src/ngramma_runtime/native_forward.py` validates CPU FP32 in `unary`, but its registered matrix path and patched convolution/GDN functions pass NumPy pointers to C++ `float*` parameters without equivalent dtype and complete shape validation. Half-precision buffers can be smaller than the C++ read size; wrong state or history shape can likewise cause invalid reads. Add explicit CPU/FP32 and no-gradient checks to all native entry paths, validate convolution history `[K-1,C]`, matching channel widths, and GDN q/k/v/g/beta/state dimensions. This is a misuse boundary risk, not an explanation of the observed all-FP32 diagnostic.

The C++ repack selection correctly probes extra-buffer operation support, allocates through that buffer type, and uploads through the tensor setter. No material error was found in that path for the tested Q4_0 tensor. Choosing available buffer types remains a diagnostic policy: reproduce the captured buffer identity when qualifying a particular serving configuration.

## Remaining MoE: two concrete uncovered operations

1. **Router-weight denominator.** Replica `vendor/engraft/replica/layers.py:420` sums the selected weights in FP32. Engine `src/llama-graph.cpp:2082` uses `ggml_sum_rows`; its non-Accelerate CPU implementation accumulates in double before casting (`ggml/src/ggml-cpu/vec.h:1495–1503`). Capture `ffn_moe_weights_sum` and `ffn_moe_weights_norm`. A controlled candidate is `weights.double().sum(-1, keepdim=True).float()` followed by the existing clamp and divide.
2. **Shared-expert scalar gate.** `layers.py:439` calls `x @ gate_inp_shexp`, which lowers to `aten.mv(matrix=x, vector=weight)`. The adapter checks the matrix data pointer for a registered weight, so this vector-weight form bypasses the native bridge. It is the likely identity of the single recorded unregistered `mm`/`mv`, to be confirmed by logging tensor shapes/name. Capture `shared_expert_gate`, `shared_expert_gate_sigmoid`, and `ffn_shexp_gated`. Explicitly calling the native dot on a one-output matrix, or presenting `weight.reshape(1,-1).T` to a shape-aware hook and squeezing the output, avoids this omission.

**Expert aggregation itself already uses the same order.** Replica `layers.py:424–434` adds weighted expert outputs in selected-expert order; engine `llama-graph.cpp:2283–2286` does the same. The replica's initial zero add is numerically neutral for finite nonzero values. There is no vectorized `torch.sum` of expert outputs in the inspected implementation.

**MUL_MAT_ID is lower priority for this fixture.** Actual expert weights use plain CPU_Mapped IQ3_S (type 22) and Q2_0 (type 42). Their dot traits specify `nrows=1` (`ggml-cpu.c:397–402`, `:555–560`). Plain CPU MUL_MAT_ID dispatch calls that same per-row `vec_dot(..., nrc=1)` (`ggml-cpu.c:1683–1719`); single-row native MUL_MAT also uses those dot traits. This supports equivalence for exact weights/inputs, but is not a replacement for capturing `ffn_moe_down` and `ffn_moe_weighted`. Packed expert tensors would require separate qualification; repack MUL_MAT_ID has its own grouped-row path.

## Report check

The experiment report's stated pre-repack layer errors, attention error, full-model failure, 151-second forward timing, and sensitivity disclaimer agree with their cited saved diagnostics. Its statement that a repack operator trial is the next test and its results table were stale once `operator-probe-native-repack.json` finished. Add that result while labeling the existing full-model metrics as the earlier raw-buffer/native-recurrent configuration. The completed layer-0 result does not establish full-model agreement, gradients, or trained-row benefit. Historical top-level experiment 001 reporting need not be rewritten.

## Resolution note: subsequent fixes and qualification

The findings above are a **pre-fix review snapshot**, not a description of unresolved current adapter defects. CPU/FP32 and shape guards have since been implemented for native entry paths. Convolution accepts valid empty and partial history: dilation-one calls explicitly prepend missing zeros before entering the native kernel, and dilated calls preserve the original fallback semantics. Overlong or incompatible history remains rejected.

The updated adapter fixtures passed **21 tests** against the new local sum-enabled bridge. They cover registered vector-weight projection, dtype rejection before native calls, gradient-enabled entry refusal, malformed convolution/GDN shapes, empty/partial history and dilated fallback, context restoration, and row-sum dimensions/precision. No checkpoint was loaded by these tests.

`operator-probe-native-sums.json` now reports **bit-identical values for every captured first-layer comparison**, following native routing-weight reduction and registered shared-gate vector projection. This resolves the measured layer-0 residual discussed above for the original ten-token CPU fixture.

At the time of this note, the full sum-enabled run is still underway. Its reported intermediate progress has layers 0, 1, and 2 bit-identical; the first differing layer is layer 3, the first full-attention layer. These are progress observations, not a completed full-model or logit-parity result. Full-model agreement, gradient qualification, and trained-row benefit remain unestablished.
