# Attention-score stride resolution

Saved-activation experiments isolate the residual score mismatch to the query layout supplied to the native matrix bridge. No model or weights were loaded.

The actual post-RoPE query is physically `[T,H,D]`. Engine `src/llama-graph.cpp:2554` permutes its GGML view from `[D,H,T]` to `[D,T,H]`, retaining token stride 24576 bytes and head stride 1024 bytes on this fixture. The first bridge copied the permuted values into contiguous `[H,T,D]` instead.

CPU dispatch explicitly distinguishes those layouts: `ggml/src/ggml-cpu/ggml-cpu.c:1499–1527` considers the llamafile matrix path only when the second operand is contiguous. Otherwise the standard row-dot path handles the strided query. The routine does not use matrix precision op-param 0 to choose this branch; preserving the physical layout is the necessary distinction here.

| Exact captured Q/K input | Valid score differences / 2,400 | Maximum error |
|---|---:|---:|
| Direct native `ggml_vec_dot_f32` for every pair | 0 | 0 |
| Original contiguous batched matrix bridge | 1,400 | 3.0517578125e-5 |
| New `ngramma_attention_scores` preserving query strides | 0 | 0 |

After exact native row dots, the existing native masked softmax and PV multiplication also reproduce captured probabilities and pregate attention values bit-identically over the padded 256-slot window. Thus this fixture does not require a different PV kernel once QK is exact.

The new C ABI accepts contiguous keys `[Hw,N,D]`, queries `[T,H,D]`, and output `[H,T,N]`, followed by D,N,T,Hw,H,threads. It constructs the query permutation inside the graph without copying and sets F32 matmul precision to match the engine. It was built into a new local library, preserving the previous library for comparison.

`tests/test_native_attention_scores.py` passed seven synthetic tests: native row-dot equality for two layouts at one/four threads, independent mathematical head-index checks, and malformed-argument rejection. `score_dispatch_fixture.py` reproduces the captured-activation comparison; its new-function results are in `score-strides-fixture.json`. These qualify the bounded CPU operation and this capture; they do not establish completed full-model or gradient parity.
