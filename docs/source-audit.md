# Source and evidence audit

Audit date: September 7, 2026. This document distinguishes actual inspected artifacts from research leads in [handoff.md](../handoff.md). Unverified leads do not support the numerical claims in experiment 001.

| Source or claim | What was established | Limit |
|---|---|---|
| [ENGRAFT repository](https://github.com/fulvian/engraft-ngram/tree/028129c9c5c50fddb09b1503f6ccae07349b9831) | Public revision resolved; Python source and Apache-2.0 license inspected | Its reported fact-grafting results are separate experiments, not reproduced here. |
| [ENGRAFT engine requirements](https://github.com/fulvian/engraft-ngram/blob/028129c9c5c50fddb09b1503f6ccae07349b9831/engine/README.md) | Documents a separate `fork-ple` engine and diagnostic interface | That engine is not distributed by the inspected ENGRAFT repository. This experiment implemented different hooks. |
| [Public llama.cpp source parent](https://github.com/LaurentZuijdwijk/llama.cpp/tree/5e085d123eead2e89b5c19f824fccb05727da6a2) | Public revision resolved; MIT license retained | Recorded experiment commit `ed7f3ed…` is local. Source-only patches reconstruct the changed files; a clean build is not qualified. |
| Actual student checkpoint | Three GGUF shard hashes, tokenizer/template hashes, table hash/layout, architecture metadata and quantization formats recorded | Artifact filenames are not a verified upstream repository revision. Exact model distribution provenance and applicable license remain unaudited in this publication. |
| Native n-gram mechanism | Actual joined IQ4_NL table, 16-head token-ID addressing and inference interception observed | This verifies the tested artifact set only. It does not validate every model-name or parameter-count claim in prior discussion. |
| Teacher | No calls; original prototype kept teacher use disabled | No usable teacher API model ID, feature set, account access, or applicable training-use permission is established by this report. |
| Handoff's Reddit posts, upstream model URL, and other background links | Retained as starting points in the design document | Not independently audited for this publication and not used as evidence for reported outcomes. |

## Revisions and source reconstruction

ENGRAFT: `028129c9c5c50fddb09b1503f6ccae07349b9831`.

Public engine parent: `5e085d123eead2e89b5c19f824fccb05727da6a2`.

Recorded local engine commit: `ed7f3ed8a60e84c5bc9041dcc3a4a207f6d6d559`, plus the recorded working-tree patch and overlay translation units.

The local commit was not available through either inspected public fork at publication time. The source-only patch was exported from the local Git objects. Applying the two engine patches reconstructed the 12 affected source files exactly; applying the ENGRAFT patch reconstructed the modified layer implementation. [Check record](../data/source-reconstruction-check.json).

Original and public-projection hashes are in [publication-provenance.json](../data/publication-provenance.json). Historical runtime hashes identify the measured build, but do not substitute for a reproducible clean build or a dependency lock. No model or runtime binaries are distributed.

## Corrected or unsupported assumptions

The tested GGUF metadata uses `qwen4exp`; do not infer a different architecture string from external descriptions. The filename includes `HC8`, but the actual hyper-connection count is four. The memory is a joined tensor rather than ENGRAFT's documented per-head split layout. The generic installed GGUF reader lacked the checkpoint's Q2_0 type.

Full-precision key/value caches do not disable weight/input quantization in matrix products. Matching row addresses and decoded table values did not imply matching backbone logits. After component corrections, complete model agreement still failed.

Generalization, useful learned memory, teacher-guided training, and accepted overlays remain unsupported by this experiment. The failure is specific to this checkpoint/engine/reference configuration, not a refutation of ENGRAFT's separate claims.
