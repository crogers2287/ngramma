# Ngramma experiment 001: editing existing n-gram rows in Flash Next

*An exploratory compatibility study and numerical-parity failure report — September 7, 2026*

This is the first experimental report for the research program in [handoff.md](handoff.md). The handoff defines the desired teacher-guided, existing-row-only improvement system; this report records the narrower work actually performed. [Milestone status](docs/research-status.md) maps the results back to that specification.

## Research question and outcome

Can a capability that Flash Next demonstrates under procedural guidance become more reliable without that guidance, by editing only existing n-gram memory rows?

We implemented an inference overlay and a short-sequence training reference, then tested their compatibility before optimization. The overlay passed identity, address, unchanged-row, and deliberate-perturbation checks on the actual checkpoint. The differentiable reference failed full-model numerical agreement even after correcting activation quantization and normalization discrepancies. Training therefore did not proceed. A small tool-use pilot also failed to establish a repeatable reminder-dependent weakness.

**The result is a working mechanism for applying row edits, plus an unresolved prerequisite for learning useful edits. It is not evidence that row editing improves capabilities, nor evidence that the idea cannot work.**

## Proposed intervention

The intended model is:

$$
\mathrm{FlashNext}_{\mathrm{edited}} = (W, E_0 + \Delta E).
$$

Here, the backbone $W$ and original table $E_0$ remain unchanged. Only selected existing rows receive trainable deltas. Tokenization, ordinary embeddings, attention weights, experts, router weights, memory projections/gates, and the output head are frozen. Router decisions must still respond normally to changed hidden states.

The proposed learning loop is: find repeatable failures, obtain independently verified corrections, select influential rows, optimize shared row deltas, then evaluate actual generation on unseen families and regression tasks. A teacher would participate only during improvement cycles. **No teacher was called in this experiment, and no correction examples were used to optimize rows.**

The implementation includes FP32 normalized row deltas, assistant-target loss masking, a retention term against the original student, and an edit-size penalty. Every selected-row occurrence shares its delta, including prompt positions. The short-sequence reference recomputes history and routing; it does not reuse detached baseline prefix states. These are implementation properties, not validated training results. The actual-model gradient qualification was not run after forward agreement failed.

## Experimental setup

The tested checkpoint is identified by the local artifact name `Qwen3.8-Flash-Next-SAFE-Q4DENSE-HC8`. This is a specific quantized GGUF artifact set, not a claim that all Flash Next conversions share these properties. Shard, table, tokenizer, and chat-template hashes are in [model-identity.json](data/model-identity.json).

| Property | Value read from the tested artifacts |
|---|---|
| Architecture metadata prefix | `qwen4exp` |
| Backbone layers / embedding width | 48 / 2,560 |
| Experts / selected experts | 512 / 10 |
| Hyper-connection streams | 4; the filename's `HC8` is not the metadata value |
| Joined table | `per_layer_token_embd.weight` |
| Storage | IQ4_NL, type 20; 28,800,138,240 bytes |
| Physical table shape | 320,001,536 rows × 160 values |
| Lookups per token position | 16: eight bigram heads and eight trigram heads |
| Original three-shard size | 69,857,022,656 bytes |

Addresses use token IDs, model-specific multipliers and head ranges, unsigned integer behavior, and sequence-boundary rules. A row's observed phrases are usages, not an exhaustive semantic label: different n-grams can collide. Physical storage includes padding; candidate addresses must respect actual head ranges.

The overlay loader uses a versioned, checksummed binary export with absolute row replacements computed from original decoded anchors plus deltas. It checks model shard identity, table geometry, row bounds, duplicate indices, nonfinite values, and payload integrity. Original GGUF bytes are unchanged.

The overlay comparisons ran on **CPU with f32 key/value caches**, eight inference threads, and the recorded quantized weights. f32 caches do not turn quantized matrix products into full-precision ones. The replica used four PyTorch threads, one interop thread, and short sequences. The tool pilot ran on two RTX 3090 GPUs, q8_0 caches, one 8,192-token slot, ten experts, no draft model, temperature zero, seed 1234, and thinking disabled. No GPU overlay parity result is claimed.

ENGRAFT's pinned Python implementation was a useful starting point, but its documented engine fork is separate. We implemented our own overlay and diagnostic hooks and adapted the reader to the joined table. The pinned checkpoint also uses Q2_0 (type 42), which the installed generic GGUF Python package did not support; weight decoding used the matching engine's routines. [ENGRAFT engine requirements](https://github.com/fulvian/engraft-ngram/blob/028129c9c5c50fddb09b1503f6ccae07349b9831/engine/README.md).

## Experiment 1: does the overlay preserve and change the intended computation?

We compared normal table reads, the disk-backed path, an empty overlay, and 152 original-row replacements. The test used three sequences: 10 ordinary tokens, 17 Unicode/whitespace/chat-delimiter tokens, and 12 EOS/repetition tokens. Their chunk sizes were 32, 5, and 1. Exact token IDs are published in [compatibility-cases.json](data/compatibility-cases.json).

| Check | Result | Evidence |
|---|---|---|
| Disk path versus original logits | Bit-identical on all three sequences | [Engine compatibility](data/engine-compatibility.json) |
| Empty overlay versus original logits | Bit-identical on all three sequences | Same artifact |
| Original decoded rows versus original logits | Bit-identical on all three sequences | Same artifact |
| Offline versus engine addresses | All 624 accesses matched in each of the three traced conditions | Same artifact |
| Invalid-overlay rejection | All six malformed/incompatible cases rejected | [Negative checks](data/overlay-negative-checks.json) |
| Final loader build | Original-row smoke test remained exact; metadata override rejected | [Final smoke check](data/final-engine-smoke-check.json) |

A separate manual perturbation changed the first component of row 167282821 by 10% of that row's RMS scale. The sequence accessed it once. The intended gathered values matched the overlay exactly, other gathered values stayed unchanged, and the maximum logit change was 5.288765. [Perturbation result](data/perturbation-check.json), [export procedure](reference/scripts/export_diagnostics.py).

This confirms that an edit reaches the model. It does not measure task improvement, locality beyond this sequence, or regression safety. The unchanged controls cover only 39 tokens and CPU inference.

## Experiment 2: does the differentiable replica match the engine?

The initial replica decoded the same table values but used ordinary floating-point matrix inputs. We compared full logits and intermediate tensors on the same 10-token sequence. The selected token at each position was the original engine's top token; log-probability error is measured for that token.

The diagnostic criteria were maximum selected-token log-probability error below 0.02 nats and maximum intermediate relative RMS error below 1%. These are experimental gates, not established universal tolerances.

| Full-sequence metric | Initial reference | Reference after fixes |
|---|---:|---:|
| Top-token agreement | 8/10 | 9/10 |
| Maximum absolute logit error | 6.708081 | 5.773980 |
| Mean absolute logit error | 0.637531 | 0.562454 |
| Maximum selected-token log-probability error, nats | 1.580705 | 1.802639 |
| Layer 0 relative RMS error | 1.6248% | 0.1457% |
| Layer 47 relative RMS error | 45.4163% | 42.5776% |
| Forward agreement gate | Failed | Failed |

Sources: [initial reference](data/replica-parity.json), [reference after fixes](data/replica-quantized-parity.json).

Top-token agreement improved, while the worst selected-token probability discrepancy increased. Reporting only the former would conceal the unresolved mismatch.

### Discrepancies identified

The CPU engine quantizes matrix inputs according to each weight format's dot-product type. Plain PyTorch matrix products omitted that step. Calling the engine's activation quantizer reduced the isolated layer-0 mixer's relative RMS error from 1.4359% to approximately 0.000007%, when supplied the exact engine input. [Activation diagnosis](data/activation-quantization-diagnosis.json).

The normalization formula also differed. The original reference used:

$$x / \sqrt{\sum x^2 + \epsilon}.$$

The pinned ggml implementation uses:

$$x / \max(\sqrt{\sum x^2}, \epsilon).$$

The local reference was changed to the latter, and the order of PLE residual additions was aligned. [Recorded patch](reference/patches/engraft-local.patch).

With exact component inputs, the resulting layer-0 mixer, recurrent attention, and expert block had maximum absolute errors of approximately 9.54e-7, 1.02e-5, and 1.19e-7 respectively. The complete first layer still differed by 0.1457% relative RMS, even with the same expert choices. [Layer-0 diagnosis](data/layer0-quantized-diagnosis.json).

Component agreement is conditional on supplying the engine's exact input. It does not establish agreement when components are chained. Further rounding, quantized dot-product, and recurrent-computation differences remain to be isolated; their individual contributions are not established.

![Full-sequence relative RMS disagreement by layer](figures/replica-drift.png)

### Consequences for training

The engine activation-quantization reference is explicitly forward-only. A separate straight-through estimator exists as an experimental diagnostic, but it is not an exact derivative of rounding and was not admitted to the trainer. Toy gradient plumbing tests cannot establish actual-model gradient validity.

The initial 10-token forward took 262.1 seconds and peaked at 14.90 GiB RSS; the later forward took 100.0 seconds and peaked at 15.01 GiB. Cache and concurrent host conditions differed. These are observations, not a controlled speed comparison, and neither is a forward/backward training step.

Because the full forward gate failed, no real-model directional finite-difference qualification or trained overlay followed. [Recorded training status](data/training-admission.json).

## Experiment 3: finding a reminder-dependent failure

The pilot contained five evaluated scenario families, four variants each. Tasks required inventory lookup, multiple-target actions, conditional state changes, room disambiguation, or correctly handling an absent target. A sixth family was excluded from the reminder/correction loop. The words “train” and “development” in stored task records denote planned partitions; the published records are evaluation-only and were not used for training.

All actions used an in-process mock inventory. Success required the expected final state, a prior inventory inspection, and no recorded tool errors. The verifier did not comprehensively grade natural-language claims, clarification behavior, or general answer quality. A wrong state was enough to catch the observed false completion.

| Condition | Successful episodes | Interpretation |
|---|---:|---|
| Ordinary task | 19/20 | Initial pilot |
| Added procedural reminder | 20/20 | Same 20 tasks, paired by task |
| Error feedback and retry | 1/1 | Only the observed failed task |
| Controlled repeat, ordinary task | 3/3 | One task repeated, not three independent problems |
| Controlled repeat, reminder | 3/3 | Same task and alternating condition order |

Sources: [pilot](data/pilot/summary.json), [confirmation](data/confirmation/summary.json). All 47 episode records and the 20 evaluated task definitions are published for replay.

In the failed episode, the request was to turn on both office lamps. The model inspected inventory, changed the desk lamp, left the ceiling lamp off, then claimed both were on. The state verifier rejected it. [Failure transcript](data/pilot/episodes/multi_target-2-unassisted.json).

That failure did not recur in the controlled repeats. Those repeats disabled prompt-cache reuse and replaced tool-call IDs with canonical IDs, alternating the condition order. The initial pilot allowed cache reuse and server-generated IDs. Because multiple settings changed and the sample was small, the cause of the difference is unresolved. There is no established five-percentage-point general improvement, and no repeatable reminder advantage has been demonstrated.

Reconstructed transcripts covered 16,420 tokens and 25,102 distinct observed rows, yielding 512 provisional trigram candidates. These are frequency-based candidates, not gradient- or ablation-qualified targets. Retention coverage and delimiter-dominated rows need examination. The reconstruction uses the engine's tokenizer/template and includes a generation-header suffix; it is not an exact record of every generated token ID. [Usage summary](data/row-usage-summary.json).

## What remains open

1. Isolate the first divergence in the complete layer-0 path, then verify full short-sequence logits and intermediate states on more than one sequence.
2. Qualify directional gradients against actual inference changes, distinguishing smooth regions, quantization effects, and routing discontinuities.
3. Find development task families with a reproducible reminder/correction advantage under fixed generation and cache conditions.
4. Only then optimize small shared row budgets and evaluate actual generation on new family-level held-out tasks, including unrelated contexts that access edited rows.
5. Qualify the deployment inference path, regression behavior, and throughput before calling any overlay an improvement.

The central capability-reliability hypothesis remains open. This report establishes a narrow overlay mechanism and documents why the first sequence-training attempt did not qualify to proceed.
