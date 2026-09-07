# Validated RoPE configuration for the current lens

Read the first GGUF shard's metadata only. Its SHA-256 matches the identity record: `a2d357ac63cd8f7aab93a2284cb0e01a5506a4a5f183cf2f3f47323cd20ca8db`. No tensor weights or inference were read/run. `rope-config-audit.json` records the extracted fields and resolved configuration.

The only architecture metadata keys containing `rope` or `context` are:

```json
{
  "qwen4exp.context_length": 262144,
  "qwen4exp.rope.dimension_sections": [11, 11, 10, 0],
  "qwen4exp.rope.freq_base": 10000000.0,
  "qwen4exp.rope.dimension_count": 64
}
```

The current capture command (`scripts/capture_reference.py:36–40`) passes `-fa off`, F32 caches, and `-c 128`, but **no RoPE or YaRN overrides**. The lens constructs ordinary `common_params` and initializes through the normal common loader (`src/ngramma_runtime/native/lens.cpp:44–51`). The original RoPE context is therefore **262144, not the allocated context 128**.

Resolved native call parameters:

| Parameter | Value | Source of resolution |
|---|---:|---|
| mode | 40 (IMRoPE) | qwen4exp architecture mapping, `src/llama-model.cpp:2963–2965`; constant `ggml/include/ggml.h:254` |
| n_dims | 64 | GGUF metadata |
| sections | 11, 11, 10, 0 | GGUF metadata |
| frequency base | 10000000 | GGUF metadata; no context override |
| original context | 262144 | `src/llama-model.cpp:1300–1301` and `src/llama-context.cpp:135–137` fallback to training context |
| frequency scale | 1 | Missing scaling factor means 1; `llama-model.cpp:1314–1319` |
| extrapolation factor | 0 | Missing scaling type defaults to linear (`llama-model.cpp:1307–1310`); unset extrapolation factor resolves to 0 for non-YaRN (`llama-context.cpp:172–173`) |
| attention factor | 1 | `src/llama-hparams.h:138,149`; resolution `llama-context.cpp:114,213` |
| beta_fast | 32 | `llama-hparams.h:150`; resolution `llama-context.cpp:115` |
| beta_slow | 1 | `llama-hparams.h:151`; resolution `llama-context.cpp:116` |

`common/common.h:465–471` defines the CLI sentinels (base/scale 0, YaRN factors/betas -1, original context 0), and `common/common.cpp:1739–1745` forwards them into context initialization. Since ext_factor is zero, YaRN correction interpolation is inactive; nevertheless pass the resolved original context and beta values rather than substituting probe dimensions.

## Loader validation contract

For the current short, plain-text diagnostic, use a deliberately bounded loader:

1. Require the pinned checkpoint identity and `general.architecture == "qwen4exp"`; validate metadata against the four extracted values above. Keep this artifact-specific check separate from generic GGUF parsing.
2. Detect any `qwen4exp.rope.scaling.*`, `qwen4exp.rope.scale_linear`, frequency-factor tensors, or nondefault CLI/context overrides. This artifact has none. Reject these cases until their source-derived resolution is implemented and fixture-qualified; do not silently overwrite them with the current constants.
3. Build the resolved configuration explicitly from metadata plus the pinned-engine defaults listed above. Store the resolved values and source/metadata identity in the probe result. `hp.rope_dim`, sections, and base alone are insufficient to reconstruct the full native operator contract.
4. Require CPU FP32 `[T,H,D]`, even `n_dims <= D`, finite parameters, valid four-section values, and integral I32-range text positions. For this text-only probe, replicate the same positions into the four consecutive position axes required by IMRoPE. Reject nonintegral positions instead of silently truncating the replica's FP64 arange representation.
5. Compare native output against captured post-RoPE Q/K from exact `Qcur_normed`/`Kcur_normed`. Metadata/default resolution is validated by source; only this fixture comparison can validate arithmetic agreement of the new bridge.

A generic implementation can later reproduce the engine's full precedence: explicit context override; else metadata; else engine default. Note the non-obvious rules: absent scaling type defaults to **linear**, scaling factor is inverted, type none forces frequency scale to 1, and original context defaults to training metadata independent of allocated inference context. These must not be inferred from common RoPE conventions.

The baseline's reported dense-causal probability support across all 24 heads and 256 cache slots eliminates a missing-visible-key explanation on this fixture. It does not remove the RoPE difference or qualify a ten-key contraction as numerically identical to the engine's masked 256-slot attention computation.
