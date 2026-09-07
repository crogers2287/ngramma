# PLE activation encoding: observed path and isolated replay

The retained Unicode PLE capture identifies `blk.1.ple_key.weight` and `blk.1.ple_value.weight` as GGML type 8 (`Q8_0`), both in `CPU_Mapped` buffers. Their GGML shapes are `[2560,10240,1,1]` and `[2560,2560,1,1]`. This is observed layer-1 metadata, not a claim about every layer or another machine. The full-bound capture also exposes those same two projection weights.

## Actual dispatch

- `src/llama-model.cpp:1058–1084` places supported extra CPU buffers before ordinary CPU buffers. However, `ggml/src/ggml-cpu/repack.cpp:4695–4721` has Q8_0 repack branches for NEON/RISC-V, with no x86 branch. Ordinary buffers observed for these weights are therefore consistent with this engine's x86 dispatch. Do not infer repack solely from quantization type or model-wide settings.
- `ggml/src/ggml-cpu/ggml-cpu.c:471–480` selects Q8_0 activations, `quantize_row_q8_0`, and Q8_0 × Q8_0 dots for Q8_0 weights.
- An initial llamafile attempt occurs before encoding (`ggml-cpu.c:1495–1514`), but the Q8_0 weight branch explicitly rejects F32 activations (`llamafile/sgemm.cpp:4041–4043`). Thus this attempt cannot bypass activation quantization. The CPU function then invokes the dispatched `from_float` into its work buffer (`ggml-cpu.c:1518–1555`). After its barrier, a second llamafile attempt consumes this same encoded buffer (`:1564–1587`). Successful second-attempt GEMM and fallback vector dots therefore share the encoding.
- The x86 dispatched quantizer (`arch/x86/quants.c:302–352`) computes `d=amax/127`, stores `d` in FP16, multiplies inputs by `127/amax`, then uses nearest rounding. The reference quantizer's reciprocal arithmetic and half-tie behavior should not be substituted.

The actual PLE input is the head-concatenated 2560-coordinate vector (`src/models/qwen4exp.cpp:1629–1644`), used by both projections (`:1657–1658`). Encode that complete vector at every edited-row occurrence. The 160-coordinate head segment consists of five aligned 32-coordinate blocks; head 8 occupies blocks 40–44. Its encoding is independent of other heads' blocks, but full-vector hashes preserve the experimental input contract.

## What the bytes mean

`ggml/src/ggml-common.h:251–256` defines Q8_0 as exactly 34 bytes: a two-byte FP16 scale followed by 32 signed integer codes. There are **no auxiliary sums** in this format. A 2560-coordinate row has 80 blocks and 2720 encoded bytes. Preserve the raw scale bits, not a rounded textual scale. Report complete-byte, scale-byte, and code-byte equality separately. Equal integer codes alone do not establish a plateau; different encoded scale bits can change the projection.

`src/ngramma_runtime/native/activation_probe.cpp` and `activation_encoding.py` provide an isolated encoder replay. The C entry calls the linked runtime's `ggml_get_type_traits_cpu(Q8_0).vec_dot_type` and that activation type's `from_float`, checking the expected 32-element/34-byte layout. It supports ordinary Q8_0 only. The Python entry accepts finite native-endian FP32 arrays `[T,K]`, returns unsigned bytes `[T,K/32*34]`, and exposes layout metadata including `auxiliary_sums=None`.

This helper is **not direct graph workspace instrumentation**. Its source-grounded equivalence depends on captured ordinary Q8_0 weight buffers, exact captured FP32 projection inputs, the same linked CPU library, and unchanged process floating-point settings. A literal in-engine observation would copy the PLE matmul work buffer immediately after the conversion barrier (`ggml-cpu.c:1564`), before the second llamafile branch, once per operation and outside competing thread writes; record weight name, source tensor shape/strides, activation type, row byte count, and all bytes. The graph tensor callback alone cannot expose this internal workspace. No running engine was modified for this work.

## Validation and provenance

Seventeen synthetic tests pass with the isolated library configured: known scale/code bytes, all-zero blocks, scale-only changes, full-row versus individual-block and individual-row partition invariance, noncontiguous input copying, unsupported types, nonfinite/dtype/shape rejection before native calls, and C boundary checks. These tests load no checkpoints and execute no models. They do not establish observed full-engine workspace equality or a deployment gradient.

`build_activation_probe.py` builds a separate shared object and companion build record hashing its source, builder, public GGML headers, linked CPU/base libraries, and output library, with compiler identity and sanitized build arguments. The response experiment must bind this record and its library hash to the capture/projection-library identities, hash the encoder wrapper as well, and retain input float hashes and full encoded output hashes. Do not call a reference-quantizer or decoded-float comparison complete-byte equivalence.

Usage: `ActivationEncoding(library).encode(8, ple_input)`; block views are `encoded.reshape(T,80,34)`, scale bytes `[..., :2]`, and code bytes `[..., 2:]`. Model-free test command: `PYTHONPATH=src NGRAMMA_ACTIVATION_LIBRARY=<probe.so> python3 -m pytest -q tests/test_activation_encoding.py`.
