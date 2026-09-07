# Published evidence

| Files | Meaning |
|---|---|
| `model-identity.json`, `engine-build.json` | Checkpoint and historical build provenance, with local paths redacted |
| `engine-compatibility.json`, `compatibility-cases.json` | Three sequences, unchanged-output comparisons, 624 row accesses per traced condition |
| `overlay-negative-checks.json`, `final-engine-smoke-check.json` | Invalid input rejection and final loader smoke test |
| `perturbation-check.json` | A manual one-row change reached inference; no improvement claim |
| `replica-parity.json`, `replica-quantized-parity.json` | Initial and corrected full forward comparison on one 10-token sequence |
| `activation-quantization-diagnosis.json`, `layer0-quantized-diagnosis.json` | Isolated-component diagnostics; exact-input results do not imply chained agreement |
| `training-admission.json` | Recorded refusal to train before forward/gradient qualification |
| `pilot/`, `confirmation/` | 41 initial and six repeat episodes, their summaries, and 20 synthetic evaluated tasks |
| `row-usage-summary.json` | Counts from reconstructed transcripts; no row weights or approved selection |
| `publication-provenance.json` | Original artifact hashes, public hashes, and declared transformations |
| `source-reconstruction-check.json` | Public-parent patches reproduce the recorded engine source changes and the modified ENGRAFT layer file |
| `historical-unit-tests.log` | Original 22-test guard/plumbing run; separate from publication verification |
| `SHA256SUMS.json` | Public evidence and reference-source integrity manifest |

The identifiers in mock-tool episodes refer to synthetic devices. Tool-call IDs are retained because canonical versus generated IDs differed between the pilot and repeat protocol. Raw response envelopes were removed; cache/timing/fingerprint metadata remains available. All numerical summaries preserve the recorded values.

Original `identity_sha256` and embedded source hashes are historical identifiers. Path redaction and projection change the bytes of public JSON files; use `SHA256SUMS.json` for their current file hashes. Hashes protect archive consistency and identify provenance; they do not independently establish the truth of measurements.

No model weights, decoded memory rows, trained overlays, private conversation logs, or operational configuration are included. This archive is the evidence available for external inspection; complete tensor-level reproduction requires a new model run.
