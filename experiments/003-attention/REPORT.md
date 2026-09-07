# Experiment 003: reproducing full-attention arithmetic

**The original ten-token sequence now matches the CPU engine bit-for-bit through
all 48 layers and every output logit.** Selected-token log-probability error
falls from 0.905555 nats to zero. The work addresses a compatibility prerequisite of the
[original handoff](../../handoff.md); it does not train memory rows.

## Full-model result

The corrected native reference passes both unchanged numerical gates: maximum
selected-token log-probability error strictly below 0.02 nats, and every measured
intermediate relative RMS strictly below 0.01. PLE input, all 48 residuals, and
the final logits are identical on the original fixture; top-token agreement is
10/10. This is agreement with the reference model, not improved task accuracy.

The run records all 12 full-attention calls, 24 rotary calls, 48 shared-gate
vector projections, zero uncovered matrix/vector calls, and no source changes
during execution. Forward time after initialization was 89.14 seconds; peak
process RSS was 14.98 GiB. These measurements share host/OS caches and are not
controlled throughput benchmarks or estimates of backward cost.

The first wider `unicode-chat` test does **not** pass: 17 tokens yield 16/17
top-token agreement, maximum selected-token error **0.148039 nats**, and final
layer relative RMS **0.290736**. Memory gather and layer 0 are exact. The first
difference appears at memory-bearing layer 1, with maximum absolute error
1.49e-8, then grows downstream. This failed control is retained in
`full-unicode-chat.json`; it prevents claiming broad short-sequence agreement.
The next component test isolates the memory gate's scale operation. Source and
synthetic evidence identify it as a candidate, not yet a measured explanation.

## Component findings

Experiment 002 first diverged at layer 3. Starting from the engine's saved
layer-2 residual, the new probe finds exact HC mixing, query/key/value
projections, query/key normalization, and gate inputs. It then isolates rotary
positions, score computation, softmax, and value aggregation. All controls use
the same ten token IDs and checkpoint identity.

| Layer-3 condition | Maximum rotary-query error | Maximum QK score error | Maximum pre-gate value error | Maximum projected-output error |
|---|---:|---:|---:|---:|
| Experiment-002 native condition | 2.38419e-6 | 1.22070e-4 | 4.64916e-6 | 1.07706e-4 |
| Native rotary only | 0 | 1.22070e-4 | 4.64916e-6 | 1.07706e-4 |
| Native padded attention only | 2.38419e-6 | 3.05176e-5 | 9.53674e-7 | 0 |
| Both, contiguous query copy | 0 | 3.05176e-5 | 9.53674e-7 | 0 |
| Both, original query layout | 0 | 0 | 0 | 0 |

These are chained component measurements, not behavioral scores. Zero means
every compared float32 element is identical. The table's third row groups
native QK, masked softmax, padded cache width, and value multiplication; it
does not identify an individual contribution from each of those changes.

### Rotary arithmetic

The replica computed frequencies and trigonometric functions in float64, then
cast to float32. The engine uses float32 frequency progression and native
trigonometric operations. Calling the actual IMROPE operator eliminates the
query and key differences. Correcting rotary positions alone leaves the
projected attention error unchanged on this fixture.

The exact metadata and engine defaults are recorded in the
[RoPE configuration audit](review/rope-config-loader.md): 64 rotated dimensions,
sections `[11,11,10,0]`, base `10000000`, original context `262144`, frequency
scale 1, extension factor 0, attention factor 1, and betas 32/1. The capture's
small allocated context does not replace the original training context.

### Cache width and matrix dispatch

The engine evaluates attention over a 256-slot cache even for ten tokens. The
old replica used ten keys. Captured positive probability support matches dense
causal support for the tested sequence. The corrected probe retains the padded
width and uses native scaled/masked softmax and value multiplication.

Using the same matrix operator was still insufficient. The engine's queries
are physically stored as `[tokens, heads, dimensions]`; a logical permutation
does not make them contiguous as `[heads, tokens, dimensions]`. Copying to that
contiguous layout enables a different CPU GEMM branch. Preserving the original
layout instead selects the engine's row-dot path.

On exact saved rotary inputs, the contiguous helper differs at **1,400 of 2,400
valid score elements**. Both a direct native dot-product fixture and the new
permuted-query helper match **all 2,400**. Given those exact scores, the existing
native softmax and value helpers also become exact. See the
[dispatch isolation](review/score-strides-resolution.md) and its saved JSON.

The output projection in the contiguous controls already rounded to the exact
reference output despite upstream differences. That is why a final projection
alone cannot establish internal attention agreement. Conversely, experiment
002 showed how tiny differences can be amplified by activation quantization.

## Scope and evidence

The corrected component run snapshots Python/C++ source identities before model
initialization, records the native binary's build provenance, and reports no
source changes during execution. Earlier component ablations record their probe
script and native build; their Python wrapper's input-validation guards were
tightened subsequently. Their numerical branches are unchanged, and the final
fully recorded run establishes the exactness claim.

Two expanded engine captures reproduce the historical logits hash unchanged.
The full runner checks captured dense-causal probability support at all twelve
full-attention layers before running. It then recomputes attention values and
normal expert routing. It never substitutes captured hidden states or logits
into the full forward. The cache layout is fixture-derived; this is not a
general implementation of sparse QSA selection.

The native reference is forward-only and refuses gradient-enabled use. It
supports fresh, short, plain-text sequences on the tested CPU engine. No cached
prefixes, multimodal positioning, GPU execution, 200k-context inference,
backward computation, teacher examples, or learned memory overlay are qualified.
No inference service or original model file was changed.

After independent review, subsequent full probes also bind the capture's
recorded model/configuration to the supplied identity, reject nonfinite
comparisons, hash every compared intermediate, and snapshot the external
ENGRAFT/GGUF Python sources. The initial ten-token run remains preserved; the
additional provenance run uses a distinct filename. These checks authenticate
recorded small artifacts, not the original multi-gigabyte model files, whose
independent hash verification remains a separate prerequisite.

Source, ablations, corrected measurements, and reproduction commands are
committed in this directory. Raw model-derived tensors and binaries remain
local. The [reproduction instructions](REPRODUCE.md) describe model-dependent
reruns. `python scripts/verify_attention_experiment.py` checks saved metric
consistency and fixed-threshold decisions without rerunning model inference.

## Next gate

An exact forward-only reference provides a target for a differentiable sequence
implementation. It supplies no derivative. The next experiment must check
directional finite differences on existing memory rows, distinguish stable
routing from routing/quantization discontinuities, and compare the actual
training forward to this reference. Numerical correctness still cannot establish
the handoff's behavioral hypothesis: improvement on unseen ordinary-prompt tasks
requires a verified curriculum, learned candidate, and independent evaluation.
