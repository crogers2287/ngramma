# Independent code audit and disposition

A read-only Claude CLI review examined the frozen candidate preparation,
selection, lookup audit, overlay export, and authentication code while the
first finite candidate was running. It found no reason to stop that process
and proposed five checks. This was a code review, not a source of training
examples, task answers, or evidence that an overlay improves the model.

The executing search, engine binary, recipes, and loader extension remained
unchanged. An additional artifact audit checked the actual inputs before later
candidate scores were available. The separate saved-evidence checker now
enforces the bindings below without changing the selection rule.

| Review concern | Inspection and disposition |
|---|---|
| Duplicate target rows could conflict. | The actual eight global IDs are distinct and sorted. Heads occupy separate global ranges. `MultiRowPatch` also rejects duplicate IDs. The proposed failure was already guarded in export; the independent input audit now records the actual IDs' uniqueness. |
| Zero qualification was insufficiently bound to the active recipe/baseline in `search.main`. | Valid hardening gap in that entry point. The independent audit verified the recipe's three input hashes, original baseline hash, zero record and lookup hashes, and exact zero-overlay hash. Saved-evidence verification repeats those checks. None of the actual artifacts was stale. |
| Search partition was implicit in generator defaults. | The exact recorded requests contain 16 development tasks, eight controls, two tokenizer checks, and only the first candidate's diagnostic witness. Input auditing and replay explicitly enforce the 24 scored IDs and exclude holdout IDs. No held-out model answers were used for candidate selection. |
| Small changes might disappear when the overlay is stored. | The premise did not apply to this exporter: FML stores absolute FP32 row replacements, with no BF16/int8 export conversion. The captured finite lookup contains exactly the intended 1,280 changed values. Later quantized activations can still absorb or distort changes; actual generated answers decide admission. A partial search never selects a winner. |
| The zero summary did not link negative authentication tests; scorer did not itself check full-hash markers. | The runner checks complete-shard verification on every overlay process. Eleven synthetic native authentication tests passed and were rerun into `verification-tests.json`. Saved-evidence verification now binds that test source to the loader build and checks every candidate's complete-file, no-cache marker. Empty path and manifest vectors match the original loader's vacuous behavior; this experiment always supplies three shards and rejects count mismatch. |

The omitted "fewer rows" ranking field is immaterial here: every candidate
changes the same eight rows. Ties use the remaining predeclared criteria.

`input-audit.json` records inspection of all thirteen actual FML exports and the
nonzero lookup witness. `verification-tests.json` describes synthetic files,
including same-size corruption of each shard, wrong path/size/digest/count,
missing files, and mutation between repeated checks. It does not claim the
original model files were modified during testing.

The audit does not establish inference parity for another engine build or GPU
profile. The measured profile and hashes remain explicit in each raw result.
