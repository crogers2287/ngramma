# Reproduce the instruction-memory experiment

## Check saved evidence without a model

Python 3.10+ and its standard library are sufficient:

```sh
python3 scripts/verify_results.py
python3 experiments/005-behavior/score_results.py
python3 experiments/006-instruction-memory/verify_results.py
```

For a commit published before all twelve candidates finish, add
`--allow-partial` to the last command. Partial results cannot select a winner.
The checker reconstructs exact requests, grades actual complete answers with
programmatic oracles, detects preservation losses, and reapplies the fixed
admission/ranking rule. It checks bindings among the baseline, prepared
candidates, zero and finite lookup controls, authentication tests, and raw
candidate records. The repository manifest detects changes to published files.

This is an audit of saved evidence. It does not rerun inference, read the
original model files, or recompute the ignored raw tensor captures.

To regenerate the outcome figure, install the existing optional figure
dependencies and run:

```sh
python3 -m pip install -r requirements-figures.txt
python3 experiments/006-instruction-memory/plot_results.py \
  --output-prefix .local/reproduce006/outcomes
```

Use `--allow-partial` for an unfinished-search commit. The stronger-format row
is a different prompt baseline; it is never counted as a memory-edit result.

## What an actual rerun needs

Read [runtime.md](../../docs/runtime.md), the preceding
[generation reproduction](../005-behavior/REPRODUCE.md), and
[source reconstruction limits](../../REPRODUCIBILITY.md). Supply the actual
checkpoint, full local model identity, matching patched libraries, and a
qualified lens build. The public identity has redacted paths and is unsuitable
for authentication. CPU/GPU profiles and different library builds are separate
qualifications; compiling successfully does not qualify their math.

The measured profile uses CPU, eight threads, context 128, batch/microbatch 32,
f32 attention caches, flash attention off, repacking on, and a four-token
unconstrained greedy answer budget. Each job clears sequence state. At most one
model process runs at a time. The driver enforces 80 GiB RSS, 24 GiB available
host memory, and 900 seconds per native process, including model authentication.

## Build and qualify the optional authentication extension

The original loader hashes the complete model files serially on every overlay
load. This optional Linux/GCC-ABI-specific extension hashes independent shards
with at most three workers. It preserves canonical paths, exact sizes, and full
SHA256 checks; it never caches a prior authentication result.

```sh
python3 scripts/build_parallel_verify.py \
  --engine-source /path/to/pinned-engine \
  --runtime /path/to/matching-runtime \
  --output .local/reproduce006/libngramma-verify.so

NGRAMMA_ENGINE_SOURCE=/path/to/pinned-engine \
  python3 -m pytest -q tests/test_parallel_verify.py
```

The builder checks the required `libllama.so` call site and writes a build
record beside the library. The runner requires that record and matching library
hashes when `--verification-library` is supplied. Our measured build and test
results are in `parallel-verify-build.json` and `verification-tests.json`.
Synthetic negative tests must pass before a real overlay run. No original model
file needs to be modified to test rejection of corruption.

The extension is qualified here by original-row replacement: all 24 baseline
generations and selected scores match, as do all logits on the previous
17-token reference. A zero overlay whose values equal the decoded original
rows must also produce an exact native gather. `zero-validation.json` records
the measured qualification; `qualify_zero.py` recomputes it when the original
local captures are available. A changed build requires a new qualification.

## Prepare and run a finite candidate

`prepare_candidates.py` reads only original table rows and previously captured
token IDs. It finds the earliest fully contained trigram in `Answer only A or
B.`, selects its eight existing trigram-head rows, and writes thirteen local
FMLs: one zero control and twelve immutable finite edits. Its outputs include
private original anchors/directions; do not publish those weight arrays.

On the same qualified identity and baseline artifacts:

```sh
PYTHONPATH=src NGRAMMA_ENGINE_SOURCE=/path/to/pinned-engine \
NGRAMMA_RUNTIME=/path/to/matching-runtime \
python3 experiments/006-instruction-memory/prepare_candidates.py \
  --manifest /path/to/full-local-identity.json \
  --local .local/reproduce006/candidates \
  --output .local/reproduce006/candidates.json

python3 experiments/005-behavior/prepare_jobs.py --controls \
  --output .local/reproduce006/jobs.json

python3 experiments/005-behavior/run_batch.py \
  --manifest /path/to/full-local-identity.json \
  --runtime /path/to/matching-runtime \
  --lens /path/to/qualified-lens \
  --verification-library .local/reproduce006/libngramma-verify.so \
  --overlay .local/reproduce006/candidates/rademacher-minus-8/rows.fml \
  --jobs .local/reproduce006/jobs.json \
  --local .local/reproduce006/rademacher-minus-8-run \
  --output .local/reproduce006/rademacher-minus-8.json
```

Use new output paths. The candidate command is an example of generation,
not an automatic qualification or improvement claim. First run the no-overlay
and zero controls for the new local setup. For the lookup witness, add a second
copy of the first task with `compact: false`, `capture: ["ple_embd"]`, and a new
`output_dir`; check it with `check_lookup.py`. This captures actual prompt and
decoded-history row values, including every unselected row.

The published `search.py` is the fixed experiment driver, not a generic trainer.
It uses the published recipe/zero evidence, evaluates candidates in declared
order, resumes only matching complete records, and refuses to continue after
a changed source, binary, recipe, or overlay. Its first candidate requires the
extra lookup witness. `audit_inputs.py` checks all actual local FMLs against
the public recipe. A new model identity or different baseline requires a new
experiment namespace and qualification records, not overwriting old results
or reusing this study's pass flags.

## Selection and interpretation

Read [PLAN.md](PLAN.md) and [CONFIRMATION.md](CONFIRMATION.md) before running the
model. The search uses only 16 development tasks and eight controls. Every
candidate must fix at least two development failures without losing a
previously correct answer or breaking a previously complete A/B response.
Higher total scores cannot hide paired regressions.

Only the selected winner may receive fresh confirmation and then the 32
previously unevaluated within-family holdout inputs. A search with no winner
must leave that holdout unevaluated. These four-token response-contract tests
are not a general model-accuracy benchmark. No sequence gradient, teacher
correction training, or broad intelligence improvement is established by a
finite candidate search.
