# Architecture audit for experiment 001

This is a bounded audit of the actual tested artifact set, not a complete independent architecture reimplementation. See [model-identity.json](../data/model-identity.json) for exact metadata and hashes.

The model has 48 backbone layers, width 2,560, four hyper-connection streams, 512 experts with ten selected, and a full-attention interval of four. Its table is the joined `per_layer_token_embd.weight`, stored as IQ4_NL, with physical dimensions 320,001,536 × 160. Metadata identifies the native memory layer with zero-based index 1.

At each token position, eight bigram and eight trigram heads produce addresses from token IDs and model-specific integer multipliers, per-head offsets, and vocabulary sizes. The offline implementation uses unsigned 64-bit arithmetic and the model's EOS/insufficient-history behavior. Indices are absolute within this single pinned joined table; the checkpoint identity and table geometry provide their scope. This is not a generic multi-table compound-key implementation.

The memory reader gathers 16 rows per position. Learned normalization, gating, projection/convolution, and residual operations then incorporate that memory into the backbone. These parameters are frozen in the proposed row-only experiment. The exact source paths are [the engine model translation unit](../reference/engine/qwen4exp-memory.cpp) and [sequence reference](../reference/flash_memory/sequence.py); the ENGRAFT patch records the PLE residual-addition-order correction.

Overlay interception occurs in the disk-backed row gather. The exported vectors replace decoded row values before downstream computation. Deltas are converted to replacements by adding the original decoded anchor. The loader binds the export to the original shard hashes and joined-table geometry. [Loader](../reference/engine/memory-overlay.h), [gather](../reference/engine/llama-ple-disk-memory.cpp), [table reader](../reference/flash_memory/table.py).

## Verified coverage and gaps

The three CPU sequences exercise ordinary tokens, Unicode/chat/whitespace, EOS/repetition, and different chunk boundaries. All 624 tested row accesses per traced condition agreed with the offline calculation. Empty/original-row overlays were bit-identical in this setting.

The handoff requests broader packed-example, independent-batch, surrounding-tokenization, and multi-turn coverage. Those are not all established by 39 diagnostic tokens. Published pilot transcripts were reconstructed using the engine's chat template and tokenizer; their usage index is not a complete generated-token trace.

The differentiable reference recomputes short history, gates, recurrent computation and expert decisions. Its complete forward path still disagrees with the CPU engine. The short-sequence ceiling does not qualify longer sparse/compressed-attention behavior. Neither GPU overlay agreement nor real-model gradients have passed.
