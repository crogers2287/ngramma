# Reproduce the row-response experiment

## Explore the saved evidence in under a minute

From the repository root, with Python 3.10 or newer:

```sh
PYTHONPATH=src python3 -m ngramma_runtime.response_report \
  experiments/004-row-response/response.json \
  --html .local/row-response.html
python3 scripts/verify_row_response.py --local-only
```

Open `.local/row-response.html` in a browser. Everything runs offline; no model,
GPU, API key, NumPy, or Torch is needed for these commands. The checker audits
saved-record consistency and recorded hashes; it does not repeat inference.
The [synthetic example](../../examples/memory-edit-workbench/README.md) describes
the JSON contract for producing reports from another memory-edit implementation.

## Repeat the actual experiment

This requires the exact experiment003 model shards, matched patched engine,
ENGRAFT sources, and native forward bridge. See
[the runtime setup](../../docs/runtime.md) and
[experiment003 reproduction](../003-attention/REPRODUCE.md). This repository does
not include the model weights or prebuilt native libraries. Generic upstream
llama.cpp does not supply this experiment's overlay hook.

Set `NGRAMMA_ENGINE_SOURCE`, `NGRAMMA_ENGRAFT_SOURCE`, and `NGRAMMA_RUNTIME` as
documented there. Use a **local unsanitized model identity manifest**, whose
hashes and shard paths match the actual worker. The published identity uses
redacted paths and cannot be passed directly to the overlay loader.

Build the isolated activation encoder:

```sh
python3 scripts/build_activation_probe.py \
  --engine-source "$NGRAMMA_ENGINE_SOURCE" \
  --runtime "$NGRAMMA_RUNTIME" \
  --output .local/reproduction/libngramma-activation.so
```

Capture the 17-token Unicode/chat fixture using the matched lens and the
experiment003 CPU settings. Include capture prefixes `ple_embd`, `l_last`,
`kq_soft_max`, `ple_gate`, `ple_gated_value`, and `ple_conv_out`; retain its
`capture-summary.json`. The reference is one complete prefill with empty state,
not an incremental/chunked generation fixture. The probe explicitly verifies
the preregistered tokens.

Run the local PLE response in a new directory:

```sh
PYTHONPATH=src timeout 900 python3 experiments/004-row-response/probe.py \
  --manifest /path/to/local-model-identity.json \
  --reference /path/to/unicode-ple-reference \
  --native-library /path/to/qualified-forward-bridge.so \
  --activation-library .local/reproduction/libngramma-activation.so \
  --local .local/reproduction/row-probe \
  --output .local/reproduction/response.json
```

The probe writes the fixed row/direction/scalar specification before edited
forwards, checks exact baseline PLE agreement, then writes all 19 observations
and experimental overlay files. It refuses prior output paths. The captured
layer0 residual is valid here because memory enters only at layer1; the later
full-engine check independently verifies that earlier residual stays unchanged.

Run the preselected full-engine controls:

```sh
PYTHONPATH=src python3 experiments/004-row-response/engine_check.py \
  --response .local/reproduction/response.json \
  --manifest /path/to/local-model-identity.json \
  --reference /path/to/unicode-ple-reference \
  --runtime "$NGRAMMA_RUNTIME" --lens /path/to/matching-ngramma-lens \
  --local .local/reproduction/row-probe \
  --output .local/reproduction/response-with-engine.json
```

Each selected overlay runs in a fresh process, with normal routing and attention
selection. The helper sets the overlay explicitly and clears inherited overlay
and trace variables. Every overlay load verifies complete model shards; on a
CPU without SHA acceleration this costs more than the short forward itself.
There is a 900-second timeout per engine process and progress is written after
each condition. Keep one model job active at a time. Raw captures, original row
values, local manifests, and model-specific overlay binaries belong in ignored
local storage, not a public commit.

The activation bytes are a replay of the actual dispatched Q8_0 encoder using
the exact gathered inputs and matching CPU library. They are not a capture of
the engine's work buffer. The
[source audit](review/activation-encoding-audit.md) establishes that dispatch for
these two non-repacked PLE projections. The portable encoder rejects other
weight types; do not generalize its layout to repacked weights or Q8_K.
