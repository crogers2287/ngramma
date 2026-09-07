# Reproduction scope

There are two distinct activities: checking published evidence and rerunning the model experiment.

## Checking the evidence without a model

Run `python3 scripts/verify_results.py` from the repository root. It uses the standard library and:

1. Validates the SHA-256 manifest for published evidence and reference source.
2. Replays saved assistant tool calls from fresh mock states, compares tool results and final grades, and recomputes all pilot/confirmation totals.
3. Checks sequence lengths, address counts, unchanged-overlay results, perturbation status, and forward-parity decisions for consistency with the saved numerical artifacts.
4. Prints the numerical comparison and confirms that the recorded training gate remains closed.

These checks verify the archive and recompute task grading. The public archive contains numerical summaries, not full logits, tensor dumps, table rows, or model shards. It cannot independently recompute the tensor comparisons without rerunning inference.

The figure is generated directly from the `l_last-0` through `l_last-47` relative RMS entries in both replica JSON files. Reshaped diagnostic aliases are excluded. `python3 scripts/plot_results.py` writes a PNG and an SVG; Matplotlib is its only direct optional dependency.

## What is pinned

`data/model-identity.json` preserves shard and table checksums, tokenizer/template hashes, architecture metadata, quantization counts, and the original runtime identity. Absolute paths have been replaced with placeholders. Its `identity_sha256` identifies the original manifest, **not** the redacted JSON document.

`data/publication-provenance.json` maps selected original artifacts to public files with both hashes. JSON records preserve numeric values while redacting paths; episode records replace raw API envelopes with relevant usage/timing/fingerprint metadata. Published task definitions contain only the five evaluated families. Expected states are public, so these tasks must not be treated as sealed in future experiments.

`data/engine-build.json` contains the historical commands with path placeholders and the source/runtime hashes. It records an existing-build relink procedure, not a portable build script. `data/final-engine-smoke-check.json` records the final loader build; its hash can differ from earlier checkpoint identity/runtime entries. Do not silently combine these as one binary.

## Engine source reconstruction

The recorded engine commit `ed7f3ed8a60e84c5bc9041dcc3a4a207f6d6d559` is a local commit. Its public parent is:

```text
https://github.com/LaurentZuijdwijk/llama.cpp
5e085d123eead2e89b5c19f824fccb05727da6a2
```

In a separate source checkout, the patch order is:

```sh
git checkout 5e085d123eead2e89b5c19f824fccb05727da6a2
git apply /path/to/this-repo/reference/patches/engine-local-commit.patch
git apply /path/to/this-repo/reference/patches/engine-working-tree.patch
```

The first patch contains source changes from the public parent to the recorded local commit, with operational Markdown notes omitted. The second preserves the working-tree changes present in the research engine source. They include prior engine changes because those are part of provenance; this study does not claim to have evaluated each independently. Applying patches reconstructs source content, not the original Git commit identity or a bit-identical binary. The source reconstruction was checked against the experiment's files; see [the check record](data/source-reconstruction-check.json).

The generated overlay translation units, loader, diagnostic runner, and dequantization bridge are supplied in `reference/engine/`. They require integration into the matching engine build. The original experiment relinked existing object files; a fresh portable build has **not** been qualified in this publication.

ENGRAFT Python source is pinned at `028129c9c5c50fddb09b1503f6ccae07349b9831`. Apply `reference/patches/engraft-local.patch` to that checkout for the recorded L2-normalization and PLE-addition changes. This project does not supply or claim to reproduce ENGRAFT's separately documented engine fork.

## Requirements for an independent model rerun

Acquire the exact matching checkpoint artifacts separately and check every hash. Use the matching engine's GGUF reader and dequantization routines, including its Q2_0 type support. Supply the runtime and path configuration required by the archived reference scripts. Record compiler, dependencies, kernel/backend, generation settings, cache configuration, and all changes from the historical build.

Repeat unchanged-overlay and address checks before perturbations. Reproduce full forward comparisons before attempting gradients. GPU overlay compatibility, full-sequence backward correctness, longer-sequence attention behavior, and learned task improvement have not been established here.

**This release is an auditable research snapshot with source excerpts, not a turnkey training package.** A clean, independently validated build and complete model rerun remain future work.
