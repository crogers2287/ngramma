# PLE numerical audit for the wider fixture

The supplied wider-fixture observation is exact layer 0 and memory gather, followed by a small first mismatch at memory-bearing layer 1 (maximum 1.49e-8) and larger downstream drift. This review performed source inspection and synthetic scalar arithmetic only. It did not read model weights, run inference, or establish the actual first differing PLE operation.

## First candidate: scale the PLE dot by multiplication

The two expressions differ in their floating-point operations:

- Replica `vendor/engraft/replica/layers.py:480`: `(key * query).sum(-1) / (n_embd ** 0.5)`.
- Engine `src/models/qwen4exp.cpp:1673–1674`: native sum followed by `ggml_scale(..., 1.0f / sqrtf((float)n_embd))`.

For n_embd=2560, Python's double square root is 50.59644256269407; the rounded FP32 sqrtf value is 50.59644317626953. The engine's FP32 reciprocal is **0.019764235243201256**. Replacing a divide by a double-derived scalar with multiplication by that FP32 reciprocal is not guaranteed to preserve rounded values.

Synthetic evidence from the installed CPU Torch/libm environment:

| Test | Result |
|---|---|
| 100,003 FP32 values from -4000 to 4000: Torch division by `2560**.5` versus multiplication by FP32 `1/sqrtf(2560)` | 6,508 differing values; maximum 7.62939453125e-6 |
| 100,003 positive FP32 values logarithmically spanning 1e-6 to 1e5: Torch sqrt versus scalar libm sqrtf | Zero differing values |

This demonstrates an available scale discrepancy and gives less support to sqrt as the first candidate. Neither result attributes the actual Unicode fixture mismatch. The scalar sample was synthetic and does not assert its values or frequency reflect actual PLE summed dots.

**Minimal correction to probe:** retain the already-native sum, compute the denominator and reciprocal in FP32, and multiply:

```python
scale = np.float32(1.0) / np.sqrt(np.float32(n_embd))
s = (key * query).sum(dim=-1) * float(scale)
```

For a strict native reference, obtain the scale through the same compiled `1.0f/sqrtf` expression or expose a native scale helper. Do not change sigmoid, convolution, and other operations at the same time before checking this one difference.

## Other PLE operations

Key/value projections and grouped norms use the existing native matrix/norm machinery. Sum_rows is now native. Sign, absolute value, and clamp are pointwise; signed sqrt and sigmoid follow the same structural order (`qwen4exp.cpp:1676–1678`; replica `layers.py:481–482`). A native sqrt operation is a reasonable exact-forward extension if the actual magnitude capture differs after the scaled dot is exact, but the scalar survey does not justify blaming it first.

The dilated convolution is not the engine's fused SSM convolution. Engine `qwen4exp.cpp:1709–1735` constructs one shifted input and one converted per-channel weight for each tap, performs a separate `ggml_mul`, then adds the tap results in increasing tap order. The replica's `causal_depthwise_conv` (`layers.py:253–268`) also multiplies taps separately and adds them in increasing order. Therefore the fused-SSM-convolution FMA concern from recurrent attention does not automatically apply here. The replica starts from zero and skips unavailable history; the engine explicitly pads zero history and starts from its first tap product. For finite ordinary values this has matching arithmetic apart from signed-zero details. Test exact normalized convolution input and kernel before proposing a different convolution backend.

Native SiLU sees the same channel-row width after the tap sum. The final residual parenthesization is aligned: `hidden + (gated + conv_out)` in both implementations (`qwen4exp.cpp:1740`; replica `layers.py:496`). The current native wrapper deliberately keeps dilation-three convolution on the original fallback and accepts valid empty/partial history. The prior synthetic history tests establish indexing semantics, not actual-model bit parity at this stage.

## Smallest useful component capture

Start with the native PLE input memory, hidden state, raw key/value projections, and grouped normalized key/query; confirm these are exact. Then capture the following chain:

1. Unscaled summed key–query product and the scaled scalar s.
2. Clamped absolute magnitude, its square root, and `ple_gate`.
3. `ple_gated_value` and the grouped normalized convolution input.
4. Raw tap sum before SiLU and `ple_conv_out`; capture individual tap products only if their common input is exact but the raw sum differs.
5. PLE residual output before the attention HC mixer.

Existing engine labels expose `ple_gate`, `ple_gated_value`, and `ple_conv_out`; the earlier scalars and raw tap sum need ancestor inspection or additional diagnostic labels. Compare both the normal chain and each stage from exact captured input. If changing only the scalar scale expression makes the first differing gate and PLE residual exact, that is evidence for this fixture's cause. If scaled s already matches, leave this hypothesis rejected for that fixture and proceed to magnitude/convolution isolation.

No runtime changes or model jobs were performed for this audit.
