# Memory edit workbench

Generate an offline interactive report without a model, NumPy, or a native library:

```sh
PYTHONPATH=src python -m ngramma_runtime.response_report \
  examples/memory-edit-workbench/demo.json \
  --html .local/memory-edit-workbench.html
```

Open the generated HTML in a browser. The accessible observation table is primary;
checkboxes toggle SVG series. Everything is embedded, with no external scripts,
assets, or network requests. The JSON remains unchanged. The supplied fixture is
**entirely synthetic** and the report prominently labels it as such.

## Inspect an overlay without a model

An install supplies `ngramma-report` and `ngramma-inspect`. Both commands run
without research dependencies. From a checkout:

```sh
PYTHONPATH=src python3 -m ngramma_runtime.overlay_inspect /path/to/rows.fml \
  --identity /path/to/local-model-identity.json
```

This checks file structure, checksums, row ranges, finite values, and agreement
with the supplied manifest. It does not open model files or authenticate their
contents; the engine must still perform that check before loading the overlay.
The output lists row IDs and hashes, without dumping row vectors or model paths.

## Data contract

Use `schema: "ngramma.row-response/v1"`. `demo.json` is a complete example.
Required top-level fields are `evidence_kind` (`synthetic_demo` or `measured`),
`title`, `experiment`, `interpretation`, and `points`.

`experiment` contains nonempty `row_id` and `direction` strings, with optional
`notes`. Give compound row identity and direction normalization/units explicitly.
`interpretation.response_kind` is `finite_step` for perturbation observations;
it may be `smooth_derivative` for a separately supported derivative study, which
still requires evidence beyond the plotted samples. Optional `notes` explain the
method. Extra fields are preserved in the input and accepted for provenance,
activation hashes, derivative checks, and detailed row identities, but are not
automatically verified or displayed by this renderer.

Each point contains:

| Field | Meaning |
| --- | --- |
| `epsilon` | Finite signed step along the recorded direction; must be unique. |
| `ple_output_rms` | Nonnegative **absolute RMS difference** from the unedited native PLE output, not relative RMS. |
| `key_changed_count` | Nonnegative integer count of changed numerical scalar elements in key projection outputs across the full sequence. |
| `value_changed_count` | Same count for value projection outputs. |
| `engine_logit_margin_delta` | Optional finite signed change in a full-engine logit margin, relative to unedited inference. Omit it or use `null` where unmeasured. Record the compared token pair and position in the experiment notes. |

A zero-epsilon observation is mandatory. Its RMS/count changes, and its engine
margin delta if supplied, must be exactly zero. All numeric observations must be
finite; booleans are not numbers. Duplicate epsilon values must be resolved in
the producing experiment, not silently averaged by the report.

The plot uses evenly spaced sorted epsilon observations so tiny signed steps stay
visible. Connecting lines only aid comparison; they neither interpolate measured
evidence nor establish a derivative. Missing engine observations break their plot
line and remain explicitly missing in the table.

Measured-response labels **do not establish improved capability**. Quantized paths
can show plateaus or jumps; finite changes do not qualify smooth gradients,
backward execution, training, or deployment. The report checks the data contract,
not source authenticity, model identity, or whether supplied measurements were
actually collected. Those remain part of the producing experiment's evidence.

Optional quantizer fields `activation_changed_bytes`,
`activation_scale_changed_bytes`, and `activation_code_changed_bytes` add a
separate accessible table and toggleable plot. Each supplied count must be a
nonnegative integer; if all three are present, total must equal scale plus code.
Any supplied zero-epsilon counts must be zero. These counts describe changed
representation bytes, not their numerical magnitude: changed scale bytes can
alter decoded values even when the integer codes are unchanged. The toy fixture
omits these fields; measured experiments can supply them without changing the
primary observation table.
