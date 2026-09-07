# Local forward diagnostic harness

`ngramma_runtime` adapts the original `flash_memory` environment, artifact,
read-only table, bounded weight decoding, short-sequence forward, memory guard,
and activation-reference code. Historical `reference/` and `data/` files remain
unchanged. This package does not download weights, build or launch services,
call teachers, train, or deploy overlays. The recorded forward-parity failure
remains a closed training gate; successful imports are not numerical qualification.

Install the source with `python -m pip install -e '.[research,test]'`. The base
package has no numerical dependencies, so configuration and artifact helpers
can also be installed alone. NumPy and PyTorch version ranges describe supported
installation candidates, not a reproduced historical dependency lock. Record
exact installed versions, source revisions, and native-library hashes for each run.

Supply matching local dependencies before importing any model classes:

```sh
export NGRAMMA_ENGINE_SOURCE=/path/to/reconstructed/llama.cpp
export NGRAMMA_ENGRAFT_SOURCE=/path/to/vendor
export NGRAMMA_RUNTIME=/path/to/local/runtime
```

The engine path must contain `gguf-py/gguf/__init__.py`; an unrelated pip GGUF
reader is deliberately refused. ENGRAFT may be a parent containing `engraft/`,
the package directory itself, or (if the variable is omitted) an independently
installed package. Use the pinned ENGRAFT revision and local patch described in
`REPRODUCIBILITY.md`. Configuration checks existence and import origins; it does
**not** authenticate source revisions or model hashes. Independently verify those
against the evidence manifest before admitting any model comparison.

`NGRAMMA_RUNTIME` contains `libflash-memory-dequant.so` built against the matching
engine. `NGRAMMA_DEQUANT_LIBRARY` can override its exact path. Both the dequantization
and activation-quantization entry points must come from the matching bridge.
No native binaries are distributed. The original build has not been qualified
as a clean portable build. Linux `/proc` and a CPU-capable PyTorch installation
are required for model forward probes.

```python
from pathlib import Path
from ngramma_runtime import RuntimeConfig, configure

cfg = configure(RuntimeConfig.from_env())
cfg.validate()  # checks paths, without loading model weights or shared libraries
from ngramma_runtime.table import ModelTable
from ngramma_runtime.weights import EngineWeights
from ngramma_runtime.sequence import SequenceReplica
from ngramma_runtime.activation_reference import ActivationReference
from engraft.replica.hparams import Hparams
import torch

# Explicit matching shard lists; never paths inferred from the archived manifest.
model_paths = [Path('/path/to/model-shard.gguf')]
table_paths = [Path('/path/to/table.gguf')]
table = ModelTable(table_paths)
try:
    weights = EngineWeights(model_paths, ram_cache_bytes=1 << 30)
    # The first table/metadata shard carries architecture fields; a weight
    # shard may omit them. n_vocab can be absent from that metadata shard.
    hp = Hparams.from_gguf(table.readers[0])
    hp.n_vocab = weights.shape("output.weight")[1]
    replica = SequenceReplica(hp, weights, table, max_tokens=32)
    with torch.no_grad():
        capture = {}
        logits = replica.full([1, 2, 3], capture=capture)
finally:
    table.close()
```

Supply every required model/table shard in the example lists and put the shard
carrying architecture metadata first in `table_paths`. `weights.shape()` returns
GGML dimensions, so `output.weight` has vocabulary size at index 1. This is the
pinned `Hparams.from_gguf(reader)` API; it does not populate vocabulary size when
the metadata reader has no output tensor.

Replace example token IDs with independently verified tokenizer outputs. Each full
forward starts fresh state; the sequence must fit the checkpoint's sparse
attention top-k limit. `capture` holds PLE input, layer outputs, and routing;
retaining every tensor increases RAM. `routing_source` may explicitly impose
saved routes for diagnosis and must be reported as such. `rows` is an optional
object exposing `gather(tokens)` for forward perturbation probes; no optimizer
or shared trainable row implementation is included.

`SequenceReplica.full()` requires `torch.no_grad()` and rejects activation
checkpointing. `ActivationReference(weights)` is an optional context inside
`torch.no_grad()` for investigating CPU activation rounding. It is not a gradient
estimator and does not imply parity has passed. The inherited ENGRAFT replica
methods are upstream interfaces, not an admitted training surface.

Memory guards run once per layer, defaulting to 64 GiB maximum RSS and 24 GiB
available host RAM reserve. Set `NGRAMMA_MAX_RSS_GIB` and
`NGRAMMA_MIN_AVAILABLE_GIB` explicitly for a diagnostic host. These checks are
not hard OS allocation limits and cannot prevent allocations between checks.
Never infer that a complete model fits from sparse row size.

Run lightweight validation with `python -m pytest tests/test_runtime_config.py`.
These tests use no checkpoint, GPU, or real bridge. Base module imports do not
modify `sys.path` or load NumPy, Torch, GGUF, or ENGRAFT; requesting model classes
loads the optional dependencies. Configure once before those imports; use a
fresh process for a different engine or ENGRAFT source.

## Optional native CPU primitive bridge

From the repository root, build against **matching, already-built** engine
libraries and headers. The helper requires a C++17 compiler and Linux shared
libraries `libggml-cpu.so` and `libggml-base.so`. It compiles only this research
bridge and does not rebuild or start the engine:

```sh
python scripts/build_forward_bridge.py \
  --engine-source "$NGRAMMA_ENGINE_SOURCE" \
  --runtime "$NGRAMMA_RUNTIME" \
  --output .local/002/libngramma-forward.so
export NGRAMMA_FORWARD_LIBRARY="$PWD/.local/002/libngramma-forward.so"
python -m pytest tests/test_native_bridge.py -q
```

These fixtures load the native library and small synthetic CPU arrays. They
check matrix multiplication, unary row operations, convolution windows, state
layout, and split-sequence recurrence. No model artifacts are read. With
`NGRAMMA_FORWARD_LIBRARY` unset the fixtures skip; a configured missing or
unloadable library fails. The helper embeds the supplied runtime location as an
RPATH, so moving its libraries requires rebuilding or an explicitly configured
loader environment. Headers and libraries must be verified as the same engine
revision; neither compilation nor synthetic tests establish that identity.

For a model diagnostic, replace `EngineWeights` in the earlier example with
`NativeEngineWeights`, then enter `NativeForward` after `torch.no_grad()`:

```python
import os
from ngramma_runtime.native_forward import NativeEngineWeights, NativeForward

weights = NativeEngineWeights(model_paths, ram_cache_bytes=1 << 30)
# Reuse hp and an OPEN table from the earlier example's try block.
replica = SequenceReplica(hp, weights, table, max_tokens=32)
with torch.no_grad():
    with NativeForward(weights, os.environ["NGRAMMA_FORWARD_LIBRARY"],
                       primitives=True, matmul=True, recurrent=True,
                       repack=True, reductions=True,
                       threads=2) as native:
        capture = {}
        logits = replica.full([1, 2, 3], capture=capture)
print(dict(native.calls))
print(native.library_sha256)
```

Place this replacement inside the earlier `try` block, before `table.close()`;
the example is not a standalone script. `NativeEngineWeights` retains original
quantized buffers to run native matrix multiplication and can consume additional
RAM beyond decoded-cache accounting. Its matrices must be used without
unsupported shape transformations. `NativeForward` defaults to unary primitives
and matrix multiplication enabled, recurrent substitution disabled. Set each
switch explicitly and record call/fallback counts when comparing conditions.
Set `repack=True` to select available loader-style CPU extra buffers for native
matrices. The bridge reports actual buffer selections in `native.calls`; raw
matrix calls remain the default control. Repacking can select a different
kernel and rounding order. Compare its buffer choice and token batching with
the engine capture before treating the two as equivalent. The adjacent
`.so.build.json` records source, headers, linked libraries, and compiler details;
the Python adapter checks its library hash when that record exists.
Set `reductions=True` for native last-axis row sums and the corrected native
projection of registered one-dimensional weights, used by shared-expert gates.
Both corrections are grouped in this diagnostic condition; compare component
captures before attributing an effect to either one separately. The underlying
row-sum kernel accumulates into double and returns float32. Short convolution
history is left-padded for native dilation-1 calls; empty history is valid at
sequence start.
Dilation other than 1 falls back to the original convolution. Unregistered
matrix operations remain on PyTorch. This is selective operator substitution,
not complete engine execution, and does not establish model parity.

The context temporarily patches ENGRAFT functions process-wide. Use an isolated,
single diagnostic process; do not run concurrent replicas or overlap this context
with `ActivationReference`. It restores those functions on exit. **There is no
native backward implementation, straight-through estimator, or admitted training
path.** Inputs must be CPU float32 tensors, and entry requires `torch.no_grad()`.
Historical failed qualification remains separate from any improved forward-only
comparison.

The C ABI uses contiguous caller-owned arrays, status `0` for success and `-1`
for guarded failures; `ngramma_last_error()` reports the current thread's message.
Callers must supply sufficiently sized buffers. Upstream GGML assertions can
abort the process and cannot be converted into Python exceptions by this bridge.

| Entry point | Layout and semantics |
| --- | --- |
| `ngramma_matmul` | Source GGML weights `[n,k]`, F32 input `[m,k]`, F32 output `[m,n]`; source quantization block size must divide `k`. |
| `ngramma_matmul_repack` | Same ABI; chooses supported CPU extra buffers, uploads original bytes through the backend, reports the selection via `ngramma_last_buffer_type`. |
| `ngramma_unary` | F32 `[rows,width]`; op 0 RMSNorm, 1 SiLU, 2 sigmoid, 3 L2Norm, 4 softmax, 5 exp, 6 softplus. Normalization and softmax operate per row. |
| `ngramma_sum_rows` | F32 `[rows,width]` to F32 `[rows]`, retaining engine accumulation precision. |
| `ngramma_ssm_conv` | Full history/input `[channels,tokens+kernel-1]`, weights `[channels,kernel]`, output `[tokens,channels]`; one sequence, dilation 1, no implicit padding or activation. |
| `ngramma_gdn` | Unscaled q/k `[tokens,key_heads,dim]`, v/output `[tokens,value_heads,dim]`, log-decay g and sigmoid beta `[tokens,value_heads]`; one sequence, scalar gates, final state only. |
| `ngramma_rope_multi` | F32 values/output `[tokens,heads,dim]`, I32 positions `[4,tokens]`, four I32 sections; explicit IMROPE settings, no frequency-factor tensor. |
| `ngramma_batched_matmul` | F32 weights `[weight_heads,n,k]`, inputs `[input_heads,m,k]`, output `[input_heads,m,n]`; contiguous head-group broadcasting. |
| `ngramma_attention_scores` | Keys `[key_heads,key_slots,dim]`, queries `[tokens,query_heads,dim]`, output `[query_heads,tokens,key_slots]`; preserves the engine's permuted query strides. |
| `ngramma_softmax_ext` | Scores/output `[heads,tokens,key_slots]`, additive mask `[tokens,key_slots]`; native scale and mask, no ALiBi, sinks, or softcap. |

GDN native state is `[value_heads,value_dim,key_dim]`, transposed relative to the
replica's conceptual `[key_dim,value_dim]` state. The Python adapter handles this
transpose. Value-head count must be divisible by key-head count; native head
mapping is `value_head % key_heads`. GGML computes `exp(g)` internally and applies
`1/sqrt(dim)` after the state/query dot product. Output and new-state destinations
must not overlap. These details are tested independently because different
layouts or scaling order can create numerical divergence.

Experiment 003 adds `AttentionOps` and `NativeAttentionReference`. The latter
temporarily replaces the complete attention function in both ENGRAFT modules.
It requires a fresh sequence and independently checked dense-causal probability
support at every full-attention layer of that exact capture. It uses captured
cache width but recomputes every attention value and keeps normal expert routing.
It does not implement general QSA selection, multimodal positions, or reusable
prefix caches. A contiguous query copy changes CPU kernel dispatch, so the score
helper deliberately preserves the engine's physical `[tokens,heads,dim]` layout.

Use [experiment 003's commands](../experiments/003-attention/REPRODUCE.md) for this
composite path. The full runner binds the capture's model label and execution
configuration, verifies metadata/logit hashes, records each compared intermediate
hash, and snapshots both local and external ENGRAFT/GGUF Python sources. Its
current capture profile admits one complete prefill of at most 32 tokens. The
lower-level sequence ceiling of 128 is not a claim that every such sequence has
been tested. Model-file authenticity still requires separate shard verification.

On the original ten-token fixture, all 48 layer outputs and final logits match
bit-for-bit. This native result supplies a forward reference; it has no backward
path and does not qualify the original differentiable replica for training.

The final experiment-003 command also enters `NativePleScale`. This context
changes only the PLE gate's summed-dot scaling: FP32 reciprocal multiplication
replaces division by the Python-derived square root. It requires one fresh
PLE invocation and restores the original functions on exit. The independent
component probe verifies the old transcription against upstream before comparing
the scalar correction. This leaves all weights, n-gram rows, convolution,
normalization, sigmoid, and addition order unchanged. Its full-run switch is
`--native-ple-scale`; omit it to reproduce the retained wider-fixture failures.

The final full runner records logical float32 byte hashes for all compared
intermediates and logits, in addition to numeric errors. This distinguishes
bitwise agreement from ordinary numeric equality, including signed-zero cases.
Successful forward hashes still do not supply a backward implementation.

Original integration code is covered by the repository MIT license. ENGRAFT
interfaces derive from Copyright 2026 fulvian, Apache-2.0; see `NOTICE`,
`licenses/ENGRAFT-Apache-2.0.txt`, and `licenses/ENGRAFT-NOTICE.txt`. ENGRAFT itself
is supplied separately. Engine licenses remain in `licenses/llama-cpp-MIT.txt`.
