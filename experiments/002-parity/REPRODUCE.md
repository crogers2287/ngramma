# Reproduce the CPU diagnostics

The commands below require the exact local checkpoint, patched ENGRAFT source,
engine headers, and already-built runtime described in the root
[reproduction notes](../../REPRODUCIBILITY.md). They do not download those
dependencies or qualify a clean engine build. Follow the
[runtime setup](../../docs/runtime.md) first.

Create a local model identity JSON by copying the published identity and adding
the actual local `path` to each shard entry. Verify each shard's hash and the
tokenizer/template identity independently. The probes read the manifest's
identity label; they do not rehash approximately 70 GB of weights on every run.
Set a task-specific shell variable such as `NGRAMMA_MANIFEST` to this local file.
Keep it and raw outputs under an ignored local directory.

```sh
export PYTHONPATH=src
python scripts/build_forward_bridge.py --output .local/002/forward.so
python scripts/build_lens.py \
  --engine-source "$NGRAMMA_ENGINE_SOURCE" --runtime "$NGRAMMA_RUNTIME" \
  --output .local/002/lens
python -c 'import json,pathlib; pathlib.Path(".local/002/tokens.json").write_text(json.dumps([9419,11,1814,0,220,17,478,220,17,283]))'
```

Capture each layer, memory input, and the first-layer components. The output
directory must not already exist. The lens appends weight-buffer descriptions to
tensor metadata; they are not weight values. The reference is CPU-only, f32 K/V,
ten tokens in one batch, and the engine's default repacking enabled.

```sh
python scripts/capture_reference.py \
  --manifest "$NGRAMMA_MANIFEST" --runtime "$NGRAMMA_RUNTIME" \
  --lens .local/002/lens --tokens .local/002/tokens.json \
  --output .local/002/reference --capture l_last- --capture ple_embd \
  --capture hc_init --capture hc_norm-0 --capture hc_gate-0 \
  --capture hc_mixed-0 --capture hc_inject-0 --capture hc_combine-0 \
  --capture linear_attn_out-0 --capture ffn_out-0 --capture ffn_moe_topk-0
```

Run at most one model probe at a time. These commands retain the default
four-thread replica and bounded caches, with an external 900-second limit.

```sh
timeout 900 python experiments/002-parity/operator_probe.py \
  --manifest "$NGRAMMA_MANIFEST" --reference .local/002/reference \
  --native-library .local/002/forward.so --native-recurrent --native-repack --native-reductions \
  --output .local/002/operator-result.json
timeout 900 python experiments/002-parity/full_probe.py \
  --manifest "$NGRAMMA_MANIFEST" --reference .local/002/reference \
  --native-library .local/002/forward.so --native-recurrent --native-repack --native-reductions \
  --output .local/002/full-result.json
```

Omit `--native-reductions` for the repack-only control. Also omit
`--native-repack` for the raw-buffer native control. Omit the native-library
and native flags for the historical activation-roundtrip control. Save each
trial under a different filename. Rebuilding the library writes an adjacent
build-identity record; keep it with the library. Avoid changing runtime sources
while a probe is running.

For the attention-only trace, make a separate reference capture with the
following prefixes in addition to `hc_mixed-0`: `linear_attn_qkv_mixed-0`, `z-0`,
`beta-0`, `beta_sigmoid-0`, `alpha-0`, `a_softplus-0`, `gate-0`,
`conv_output_raw-0`, `conv_output_silu-0`, `q_conv_predelta-0`,
`k_conv_predelta-0`, `v_conv_predelta-0`, `attn_output-0`, `new_state-0`,
`final_output-0`, and `linear_attn_out-0`. Run `attention_probe.py` with that
reference, manifest, native library, recurrent/repack flags, and a new output.

JSON summaries can be reviewed and published after checking provenance and
removing local paths. Raw logits, activations, and checkpoints stay local. None
of these commands trains, evaluates a learned overlay, or supplies backward
derivatives. A numerical pass on one sequence would still require broader
coverage before gradient qualification.
