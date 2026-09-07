# Reproduce the behavior assay

## Replay the published evidence without a model

From the repository root, Python 3.10+ and its standard library are sufficient:

```sh
python3 experiments/005-behavior/score_results.py
python3 experiments/005-behavior/tasks.py --self-test
python3 -m unittest discover -s experiments/005-behavior -p test_counting_tasks.py
```

The scorer checks the exact generated job lists, actual recorded completions,
termination, and selected logits across repeats. It recomputes outcomes against
programmatic oracles. It does not rerun inference or authenticate the historical
origin of a response merely because its JSON is internally consistent.

## Run the real CPU model

Use the checkpoint identity and matching patched engine described by
[the runtime documentation](../../docs/runtime.md) and
[reproducibility limits](../../REPRODUCIBILITY.md). The public redacted identity
cannot authenticate local weights. Supply a locally pinned complete identity.
The new lens is linked against the same libraries; it adds greedy generation
without changing engine math. `lens-build.json` records the measured binary,
source, compiler, and linked-library hashes. Fresh external builds remain a
separate qualification task.

```sh
python3 scripts/build_lens.py \
  --engine-source /path/to/pinned-engine \
  --runtime /path/to/matching-runtime --output .local/reproduce005/lens

python3 experiments/005-behavior/prepare_jobs.py --controls \
  --output .local/reproduce005/discovery-jobs.json
python3 experiments/005-behavior/prepare_jobs.py --followup \
  --output .local/reproduce005/followup-jobs.json
python3 experiments/005-behavior/prepare_jobs.py --state-confirmation \
  --output .local/reproduce005/state-jobs.json

python3 experiments/005-behavior/run_batch.py \
  --manifest /path/to/local-model-identity.json \
  --runtime /path/to/matching-runtime --lens .local/reproduce005/lens \
  --jobs .local/reproduce005/discovery-jobs.json \
  --local .local/reproduce005/discovery-run \
  --output .local/reproduce005/discovery.json
```

Repeat the last command sequentially for `followup-jobs.json` and
`state-jobs.json`, with new output/run paths. Run one model process at a time.
The driver uses CPU only, eight threads, context 128, batch/microbatch 32,
f32 caches, flash attention off, and a fixed four-token unconstrained greedy
completion budget. It monitors the child process, with an 80 GiB RSS cap,
24 GiB available-RAM reserve, and 900-second wall-clock bound. Do not mistake
the first attempt's retained 64 GiB allocation stop for a model answer.

Every job clears the model's sequence memory. No baseline prefix state is
reused. The loaded weights may be reused across jobs. The driver clears inherited
overlay/trace environment variables. An explicit `--overlay` is supported for
future controlled interventions; the native loader must authenticate the full
model shards before admitting it.

Read [generation-protocol.md](../../docs/generation-protocol.md) for JSON-line
requests, complete-text/EOG handling, optional compact outputs, and diagnostic
captures. Non-EOG control tokens remain visible in generated text. A/B score
requests are diagnostics; they never restrict the greedy vocabulary.

The single-message no-thinking chat serialization is copied from the actual
artifact's template branch. The follow-up's uniform system instruction is an
explicit prompt-protocol change. These are different baselines; do not combine
their scores or attribute the prompt improvement to a memory edit.

No original pilot sealed family or new held-out model answer is used in these
discovery commands. Do not run `--split holdout` until one candidate is locked.
