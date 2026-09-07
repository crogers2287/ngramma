# Reproduce experiment 003

Use the exact checkpoint, ENGRAFT source, engine build, and dependency identities
documented in [experiment 002](../002-parity/REPRODUCE.md). The original handoff is
unchanged. These commands run CPU forward diagnostics and require local weights;
they do not download artifacts, train rows, or start an inference service.

Set `NGRAMMA_ENGINE_SOURCE`, `NGRAMMA_ENGRAFT_SOURCE`, `NGRAMMA_RUNTIME`, and
`NGRAMMA_MANIFEST` as described there. The manifest must contain actual local
shard paths. Independently verify the model hashes first: probes record the
manifest identity without rehashing the entire checkpoint on every run.

```sh
export PYTHONPATH=src
python scripts/build_forward_bridge.py --output .local/003/attention.so
python scripts/build_lens.py --engine-source "$NGRAMMA_ENGINE_SOURCE" \
  --runtime "$NGRAMMA_RUNTIME" --output .local/003/lens
python -c 'import json,pathlib; pathlib.Path(".local/003/tokens.json").write_text(json.dumps([9419,11,1814,0,220,17,478,220,17,283]))'
```

Capture the ten-token sequence in one prefill. The helper selects CPU, F32 K/V,
flash attention off, and loader repacking on. Its output directory must be new.

```sh
python scripts/capture_reference.py --manifest "$NGRAMMA_MANIFEST" \
  --runtime "$NGRAMMA_RUNTIME" --lens .local/003/lens \
  --tokens .local/003/tokens.json --output .local/003/reference \
  --capture ple_embd --capture l_last --capture kq \
  --capture hc_mixed-3 --capture hc_inject-3 \
  --capture Qcur --capture Kcur --capture Vcur \
  --capture gate_reshaped-3 --capture gate_sigmoid-3 \
  --capture attn_pregate-3 --capture attn_gated-3 --capture attn_output-3
```

The reported layer-component and full-model runs used separate captures with
different prefix sets. Both reproduce the historical logits bit-for-bit. The
combined prefix set above supplies both probes; its metadata hash will differ.

Run these sequentially. Preserve each trial in a separate file and retain the
library's adjacent build record. Do not edit runtime sources during a run.

```sh
timeout 900 python experiments/003-attention/probe.py \
  --manifest "$NGRAMMA_MANIFEST" --reference .local/003/reference \
  --native-library .local/003/attention.so \
  --native-rope --native-attention --native-strides \
  --output .local/003/component-result.json
timeout 900 python experiments/003-attention/full_probe.py \
  --manifest "$NGRAMMA_MANIFEST" --reference .local/003/reference \
  --native-library .local/003/attention.so \
  --native-recurrent --native-repack --native-reductions --native-attention \
  --output .local/003/full-result.json
```

For component ablations, omit `--native-strides` to measure contiguous query
copies. Also omit `--native-rope` or `--native-attention` to isolate those
components; `--native-strides` requires `--native-attention`. With all three
omitted, the probe also compares the original complete layer 3. Numerical
branches from the earlier component scripts remain recoverable through Git.

The full runner checks every full-attention layer's captured probability support
against dense causal support. It refuses missing captures, different tokens, or
a selector that excluded a visible key. Native attention then recomputes Q/K/V,
scores, probabilities, and outputs; it does not substitute captured activation
values. Cache width comes from the capture. This validates a specific fixture,
not a general implementation of QSA. Cached-prefix and multimodal execution are
unsupported by this reference context.

The additional `unicode-chat` and `eos-repeat` token lists are in
`data/compatibility-cases.json`. Extract one list at a time, capture with
`--chunk-size 32`, and repeat the full command. These are fresh full-prefill
tests; experiment 001's chunk sizes of 5 and 1 are a separate execution condition
and are not qualified by these full-prefill comparisons.

The raw-activation dispatch fixture needs no model execution once captures exist:

```sh
python experiments/003-attention/review/score_dispatch_fixture.py \
  .local/003/reference "$NGRAMMA_RUNTIME/libggml-cpu.so" \
  .local/003/attention.so --output .local/003/score-dispatch.json
export NGRAMMA_FORWARD_LIBRARY="$PWD/.local/003/attention.so"
python -m pytest -q tests
```

The score fixture's dot-product loop isolates the engine kernel numerically; it
is not a benchmark. Raw activations, logits, and binaries stay local. Published
summaries and source hashes permit auditing the reported experiment, but cannot
independently reproduce comparisons without those artifacts. No backward
implementation or learned-overlay qualification is provided by these commands.
