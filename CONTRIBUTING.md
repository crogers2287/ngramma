# Build on Ngramma

The current useful pieces are an offline response explorer, checked single-row
patch construction, activation-encoding diagnostics, and real-engine controls.
They help answer whether a memory intervention reaches inference. We have not
released an optimizer or an overlay that improves task performance.

## Three useful contributions

1. **Bring another inference backend.** Produce the
   [response JSON contract](examples/memory-edit-workbench/README.md) from its
   real memory lookup and inference operations. Establish a zero-edit control
   before comparing nonzero edits. The HTML explorer does not depend on the
   particular backend, model, quantizer, or a teacher API.
2. **Extend the numerical evidence.** Test independent rows, directions,
   repeated occurrences, and prompts. Record the exact encoded activation
   representation, including scales and auxiliary fields. Keep the original
   fixture, failed trials, and predeclared selection rule in the result.
3. **Find a verifiable behavior to improve.** Start with short tool-selection,
   extraction, arithmetic, or code-repair tasks. Establish a repeatable reminder
   advantage on development cases and grade actual generated actions or answers.
   Split underlying problem families before generating wording variants.

## A backend-neutral report producer

Your backend can compute each observation independently of this repository's
native bridge. Record an epsilon0 baseline, absolute RMS change of memory
outputs, and counts of changed key/value projection elements over the complete
sequence. Add a fixed final logit-margin change only when it was measured in a
full-engine run. Leave unmeasured engine values absent or null; zero means a
measured zero response.

```python
import json
from ngramma_runtime.response_report import render_html

# Use your measured data, following the documented schema.
report = json.loads(open("my-response.json", encoding="utf-8").read())
html = render_html(report)
with open("my-response.html", "w", encoding="utf-8") as output:
    output.write(html)
```

The parser checks the data contract; it cannot establish that your producer
actually ran a model. Label synthetic examples `synthetic_demo`. Preserve
method, sources, input hashes, settings, and acceptance checks alongside measured
records. The experiment004 checker has additional protocol-specific assertions;
it is not a universal validator for arbitrary experiments.

## A row-patch integration

`RowPatch.from_direction(table, row_id, direction, epsilon)` reads the original
decoded row and constructs its FP32 replacement. `patch.apply(addresses,
gathered)` returns a copy with that address replaced at every occurrence. The
current implementation requires160-value rows and a `table.read_global(rows)`
interface. It does not alter addresses, add rows, change routing weights, or
update the original model files.

`patch.export(directory, identity, table, provenance)` writes an experimental
`rows.fml` for the patched engine. Export checks the original anchor and the
model identity's self-checksum. Actual shard/table compatibility and complete
shard authentication occur in the engine loader. A self-consistent manifest
alone is not proof that model files match it. Keep worker-specific manifests,
original row vectors, and model-dependent binary overlays in local storage.

## Review and verification

Run these from the repository root:

```sh
python3 scripts/verify_results.py
python3 scripts/verify_row_response.py
python3 -m pytest -q tests
```

Install the `test` extra for pytest; the `memory` extra enables NumPy row-patch
tests. Native tests require separately built matching libraries and explicit
environment variables. A skipped native test is not a native pass. Install the
`research` extra and configure the pinned sources only for actual model work.

When adding or updating published evidence, stage the intended files, run
`python3 scripts/update_manifest.py`, and stage `data/SHA256SUMS.json` too.
The manifest covers tracked/indexed files. Preserve `handoff.md` as the original
design record and distinguish proposals, diagnostics, and measured capability
results in the report.
