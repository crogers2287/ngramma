# Experiment 002: locating numerical disagreement

**The first hyper-connection stages now match exactly on the original ten-token
CPU fixture, but full-model agreement still fails. No rows were trained.**

The objective remains the [handoff](../../handoff.md): make guided capabilities
more reliable through changes to selected existing memory rows. This experiment
works on the prerequisite that the optimization computation must agree with
inference. Historical experiment 001 files remain unchanged.

## Measurements so far

All comparisons use the same checkpoint identity and ten tokens as experiment
001. Relative RMS is a ratio: 0.001 corresponds to 0.1%. Exact-input component
checks replace only a component's input with saved engine values; chained checks
propagate the replica's own preceding output.

| Forward implementation | First HC mix relative RMS | Complete layer 0 relative RMS | Full-model selected-token log-probability error | Top-token agreement |
|---|---:|---:|---:|---:|
| Historical activation-roundtrip control | 7.03e-8 | 0.00145665 | 1.80264 nats (experiment 001) | 9/10 |
| Native unary/norm and raw-buffer matrix operations | 0 | 0.000947788 | Not run | Not run |
| Also native convolution and fused recurrence | 0 | 0.000721832 | 2.70391 nats | 8/10 |

The latest full control finishes with relative RMS 0.442664 at layer 47. It fails
both unchanged exploratory gates: selected-token log-probability error below
0.02 nats and intermediate relative RMS below 0.01. A lower first-layer error is
not evidence that downstream agreement improves.

The native bridge is **forward-only**. It is a diagnostic instrument, not an
autograd implementation, sequence trainer, or accepted improvement to Flash Next.
The adapter explicitly refuses gradient-enabled use. No teacher corrections,
learned overlay, unseen-task improvement, or gradient qualification occurred.

## What the additional captures establish

The initial control reproduces the historical complete layer-0 error. Expanded
captures include both HC normalizations, gates, injections, and combines.
Native normalization, nonlinear operations, and matrix calls make every first
HC stage bit-identical; the second mixer and both combines are also exact when
given exact reference inputs. This closes gaps in the earlier component probe,
which did not test these complete paths.

The deeper attention trace locates its first residual error in Q4_0 QKV and gate
projections. QKV relative RMS is 1.58e-7 from exact input. Alpha, beta, softplus,
and decay are exact. The final attention projection differs by 1.02222e-5 at its
worst element. The remaining small difference grows in the following mixer and
expert computation.

Fixture-only tests show that one-ULP changes can cross activation-quantization
thresholds: a maximum 4.77e-7 input perturbation changed 64 rounded values by up
to 0.000968933. This demonstrates sensitivity, not a measurement of how much of
the actual chained error those crossings explain. Native sigmoid reproduces the
captured combines exactly where Torch sigmoid does not. See the
[independent code review and fixture records](review/numerical-review.md).

Source inspection identifies CPU weight repacking as a concrete remaining
dispatch difference. The original native bridge sets raw weight bytes; a loaded
engine can attach CPU_REPACK buffers and choose packed GEMM/GEMV kernels. The
new optional repack entry point reports the buffer it selects. A direct engine
capture now confirms that `blk.0.attn_qkv.weight` uses CPU_REPACK, while the HC
weights use CPU_Mapped. The expanded lens preserves the historical logits
bit-for-bit. A matching operator/full-model trial is the next discrimination
test; the buffer difference alone does not establish its error contribution.

## Evidence and reproduction

The JSON files alongside this report preserve controls and failed trials:
`operator-probe.json`, `operator-probe-native-all.json`,
`operator-probe-native-recurrent.json`, `attention-native-recurrent.json`, and
`full-native-recurrent.json`. They contain numeric summaries, source/library
identities, and qualification status. The initial full control's source-hash
scope is explicitly marked: source files evolved during that run; its loaded
native library is pinned by its binary hash. Later probes snapshot source files
before initialization and include build records. Raw model-derived tensors stay
local and are not redistributed.

[Runtime instructions](../../docs/runtime.md) describe the optional package and
matching dependencies. `scripts/capture_reference.py` captures CPU tensors in a
new directory. `operator_probe.py` compares the first layer, `attention_probe.py`
isolates its recurrent attention, and `full_probe.py` compares all 48 layers and
logits. The optional lens records ancestor weight-buffer names without exporting
weight bytes. Compilation and synthetic tests do not qualify a fresh full-model
build.

Each model process runs separately on CPU, with a bounded decoded-weight cache,
four replica threads, and a 900-second process limit. Existing inference services
are not modified. The full native-recurrent control took approximately 151 seconds
for the forward after initialization; this is inference diagnosis, not a training
speed measurement. Its process peak RSS and initialization time are recorded in
the result JSON. Training remains closed pending numerical and gradient evidence.
