# Reference source from the experiment

These files preserve the relevant research implementation for inspection. They are **archival source excerpts**, not an installable package or a qualified trainer. `data/publication-provenance.json` records byte-identical source hashes and any published transformations.

| Directory | Contents |
|---|---|
| `engine/` | Checked row-overlay loader, generated engine translation units, JSONL diagnostic runner, and exact engine dequantization/activation-quantization bridge |
| `flash_memory/` | Joined-table reader, overlay format/export, shared sequence reference, weight reader, forward-only activation quantization, experimental gradient surrogate, and analysis/evaluation utilities |
| `scripts/` | Original CPU compatibility, diagnostic export, malformed-overlay, replica, and component probes |
| `patches/` | Engine source provenance and ENGRAFT normalization/addition fixes |

The Python excerpts import an experiment-specific `environment` module and the pinned ENGRAFT and GGUF implementations. That path configuration, orchestration, and serving launchers are intentionally not distributed. Some excerpts expect artifacts and runtime binaries that are not in the public archive. A reader must integrate these into a separately configured experiment to execute them; the repository's runnable entry point is `scripts/verify_results.py`.

`activation_reference.py` refuses gradient-enabled use. `experimental_gradient.py` is an unqualified straight-through surrogate; it was not used to train the model. `sequence.py` contains the evidence gate that rejected this checkpoint. Do not interpret the presence of a gradient implementation as a validated method.

The engine source changes are MIT-licensed with the ggml authors' notice preserved. ENGRAFT patch context and derivatives retain Apache-2.0 attribution. See the root [NOTICE](../NOTICE), [licenses](../licenses/), and [reproduction notes](../REPRODUCIBILITY.md).
