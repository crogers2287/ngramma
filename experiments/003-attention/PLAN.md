# Experiment 003: the first full-attention boundary

Start from commit `e5b6d7b4d88aad8b9681ddacfa48c31c4cf7992d`.
Experiment 002 matched layers 0–2 exactly on the ten-token CPU fixture. The
first discrepancy was layer 3; the final selected-token log-probability error
was 0.905555 nats. The original handoff and earlier evidence remain unchanged.

1. Capture layer-3 projections, Q/K normalization and rotary outputs, attention
   scores/probabilities, gating, and output from the same CPU engine. Check that
   enabling these captures preserves the historical logits.
2. Compare stages from exact saved inputs and from the complete preceding chain.
   Distinguish numerical kernels from masking, cache layout, and head mapping.
3. Implement only evidence-supported forward corrections; retain failed controls
   and hash the actual source and libraries used before model execution.
4. Repeat all 48 layers. Keep the strict 0.02-nat and 1%-relative-RMS gates. If
   both pass, expand short-sequence coverage before considering any gradients.

Use one model process at a time on CPU, four replica/eight engine threads,
bounded caches, and a 900-second limit per run. Raw model tensors remain local.
Existing inference services and model weights are unchanged. No teacher training
examples, sealed test answers, optimization, or learned overlay are part of this
forward diagnostic cycle. Commit the source, numeric summaries, and report.
