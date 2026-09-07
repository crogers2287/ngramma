# Greedy generation lens protocol

Build the repository lens with a configured engine source and matching runtime:

```sh
python scripts/build_lens.py --engine-source "$NGRAMMA_ENGINE_SOURCE" \
  --runtime "$NGRAMMA_RUNTIME" --output .local/005/flash-memory-lens
NGRAMMA_ENGINE_SOURCE="$NGRAMMA_ENGINE_SOURCE" python -m pytest -q tests/test_lens_generation_protocol.py
```

The tests compile only pure protocol helpers and synthetic logits. They do not
link the engine, initialize a model, or establish inference parity. The adjacent
binary `.build.json` records source, compiler, binary, and runtime hashes.

The lens accepts one JSON object per stdin line after its `ready` JSON response.
The process retains its loaded model. Each generation job clears the engine
memory before prefill, including recurrent and attention state. Jobs without
`generate` retain the existing capture protocol. Runtime command-line options
select the model, context, threads and hardware as before.

```json
{"tokens":[1,2,3],"generate":{"max_new_tokens":4,"context_tokens":128,"compact":true,"top_k":5,"score_tokens":[4,5]}}
```

Token IDs above are illustrative and must be replaced with model-specific IDs.
Alternatively supply `text`; the existing tokenizer parses special tokens and
does not automatically add BOS or a chat template. Supply a fully formatted
prompt or explicit IDs. `tokenize_only` and `generate` cannot be combined.

Generation is unconstrained argmax over every vocabulary token. There is no
sampler, grammar, answer forcing, repetition penalty, or requested-score bias.
Ties choose the lowest token ID. Nonfinite logits fail the job. `top_k` controls
reported first-step scores only; it does not restrict token selection.

| Generation option | Contract |
| --- | --- |
| `max_new_tokens` | Required integer, 1–1024; includes terminal EOG if selected. |
| `context_tokens` | Integer, at most the loaded context and 32768; defaults to that bound. Prompt must leave room for one new token. |
| `compact` | Boolean, defaults to true. |
| `top_k` | Integer, 0–100 and at most vocabulary size; default min(5, vocabulary size). |
| `score_tokens` | Up to 256 unique vocabulary IDs, reported in requested order. |

Unknown generation options are rejected. Integer fields exclude booleans and
floating-point values. The top-level `chunk_size` defaults to the loaded batch
size and must be an integer between one and that size. Generation truncates its
budget to available context, without discarding or sliding prompt tokens.

The success JSON has schema `ngramma.greedy-generation/v1` and these main fields:

| Field | Meaning |
| --- | --- |
| `prompt_tokens`, `generated_tokens` | Exact token IDs; generated IDs include a selected terminal EOG. |
| `text` | Concatenated generated pieces, excluding only terminal EOG. Non-EOG special/control token spellings remain visible for strict grading. Invalid UTF-8 bytes are replaced during JSON serialization. |
| `text_bytes_hex` | Exact pre-serialization generated text bytes, including any incomplete UTF-8 sequence. |
| `first_step` | `greedy_token_id`, `top_logits`, `requested_logits`; score entries contain `token_id` and raw, unnormalized `logit`. |
| `stop_reason` | `eog`, `max_new_tokens`, or `context_limit`. |
| `eog_token`, `stopped_on_eog` | Terminal EOG ID or null, and corresponding boolean. |
| `generated_count`, `effective_new_token_budget` | Number selected and context-limited upper bound. |
| `prefill_calls`, `decode_calls`, `seconds` | Work counters and elapsed generation time. Final selected token need not be decoded when stopping. |

Results also record `greedy`, `temperature`, `tie_break`, `grammar`,
`fresh_state`, `model_reused`, requested limits, chunk size and vocabulary size.
`seconds` is nondeterministic; repeated-job comparisons should compare tokens,
text and scores rather than the entire metadata object. Job failures return an
error JSON through the existing lens loop. A caller can impose its own wall-time
limit; this protocol does not implement a generation deadline.

Compact jobs perform no tensor or full-logit writes. An optional top-level
`output_dir` writes only `generation.json`, matching the stdout result. Existing
files in that directory are not removed. Noncompact jobs require `output_dir`
and write `prefill-logits.f32` (prompt positions × vocabulary),
`generation-logits.f32` (selected steps × vocabulary), `tokens.json`,
`tensors.json`, and requested `capture` tensors. Logit files are native float32
in position-major order. Captures include prefill and subsequent decode calls.
