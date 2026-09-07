# Full-probe evidence review before wider fixtures

Read-only review of `experiments/003-attention/full_probe.py` and `src/ngramma_runtime/attention_reference.py` during the first complete native-attention control. No model execution and no runtime edits.

No captured hidden states or probability magnitudes are injected into the recomputed attention. `verified_layouts` uses reference probabilities only to reject incompatible support and obtain cache width; the adapter recomputes all projections, Q/K rotation, scores, softmax, and values. Exact full-forward equality would therefore be meaningful baseline forward evidence, not an artifact of replaying captured activations.

Before wider fixtures, tighten these evidence boundaries:

1. Bind the capture identity to the supplied checkpoint manifest. The current probe records the manifest identity but does not compare it to `capture-summary.json` or another reference identity record. Matching tokens and tensor shapes alone do not authenticate which model/configuration produced a capture. Check checkpoint identity, cache types, flash-attention setting, and absence of unrecorded RoPE overrides.
2. Hash each layer/PLE tensor actually compared. The result hashes metadata, tokens, and logits; verified attention layouts additionally hash probabilities. The metadata names raw tensor files but does not itself bind their contents. Add per-compared-tensor hashes to make intermediate-parity evidence reproducible and tamper-evident.
3. Include loaded external ENGRAFT Python files in the source snapshot, especially layers, model, hparams, and weights. The current snapshot covers the local runtime package and wrapper scripts but not those executed external source modules. Hash their resolved `__file__` paths under sanitized labels, and detect changes during the run as for local sources.
4. Explicitly reject nonfinite actual and expected intermediate values rather than relying on downstream maxima and thresholds. The actual exact/finite fixture is unaffected, but NaN reduction/comparison behavior should not influence future qualification status.

The fixture-specific restrictions are appropriately narrow: fresh sequence only, no cached prefixes, sequence length at most 128, dense-causal support at every full-attention layer, matching head counts, and complete call count. These do not establish sparse selection behavior under edited hidden states, multimodal positions, different cache formats, or general continuation/chunking. Broader fixtures should retain independent support verification and record their own identities rather than inheriting the original sequence's admission.

`score_dispatch_fixture.py` now records script, native CPU/attention library, adjacent build record, reference metadata/tokens, and all consumed tensor hashes. Its rerun is saved separately as `score-strides-qualified.json`; previous JSON controls remain unchanged. Native stride-preserving scores and downstream masked softmax/PV remain bit-identical on the saved layer-3 activation fixture.
