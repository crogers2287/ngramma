# Ngramma

**Research into making Flash Next more reliable by refining its existing n-gram memory.**

The project begins with [handoff.md](handoff.md), its original research specification. The central question is: **can a task the model already solves with a useful reminder become more reliable without that reminder after training a sparse subset of its existing memory rows?**

The intended result is the original frozen model plus a small, versioned memory overlay. A teacher diagnoses failures and supplies independently verified corrections during improvement cycles. A numerical optimizer changes selected existing rows; the backbone, tokenizer, memory reader, and gates stay frozen. Ordinary inference and the primary evaluation use no teacher or extra lesson text.

**Latest result:** the CPU forward reference now matches the real engine
bit-for-bit on three short fixtures (39 tokens), through all 48 layers and every
output logit. This fixes an important testing prerequisite. **No learned memory
improvement has been demonstrated.** [Read experiment 003](experiments/003-attention/REPORT.md).

This repository follows that research program independently of any particular host or orchestration setup. The handoff is preserved as the design record. Its proposed components and milestones are not claims of completed implementation.

## First experiment: September 7, 2026

**Existing-row overlays worked in the tested CPU engine, but the sequence-training replica failed numerical agreement. No learned improvement was demonstrated.** The first experiment reached overlay interception and numerical diagnostics; it did not qualify a training run.

| Question | Observed result |
|---|---|
| Can the engine load an overlay without changing baseline outputs? | Yes: empty and original-row controls gave bit-identical logits over 39 CPU-tested tokens. |
| Does offline addressing match the engine? | Yes: 624 row accesses per traced condition matched. |
| Does an actual row change reach inference? | Yes: a diagnostic perturbation changed the intended gathered value and logits. It was not an improvement test. |
| Does the differentiable replica match inference? | No: the corrected reference agreed on the top token at 9/10 positions; maximum selected-token log-probability error was 1.803 nats. |
| Did reminders expose a repeatable weakness? | Not established: 19/20 versus 20/20 initially; the failed case then passed 3/3 both with and without a reminder. |
| Was a learned overlay released? | No. Forward agreement and real-model gradient checks remain unresolved. |

[Read the findings](REPORT.md) · [Progress against the handoff](docs/research-status.md) · [Inspect the evidence](data/README.md) · [Reproduction limits](REPRODUCIBILITY.md) · [Reference source](reference/README.md)

![Relative RMS disagreement across 48 layers, before and after the diagnostic fixes](figures/replica-drift.png)

The graph shows one 10-token sequence. Lower component errors did not eliminate disagreement through the full model.

## Check the published evidence

Python 3.10 or newer; no model, GPU, API key, or third-party Python package is needed:

```sh
git clone https://github.com/crogers2287/ngramma.git
cd ngramma
python3 scripts/verify_results.py
```

The command checks file hashes, replays all 47 saved mock-tool episodes against the original state verifier, recomputes the pilot and confirmation counts, and checks the numerical report's internal consistency. **It does not rerun model inference or independently reproduce saved tensor comparisons.**

To regenerate the figure:

```sh
python3 -m pip install -r requirements-figures.txt
python3 scripts/plot_results.py
```

## Research status

[Experiment 003](experiments/003-attention/REPORT.md) now matches the CPU engine
bit-for-bit through all 48 layers and every output logit on the original,
Unicode/chat, and repeated-EOS fixtures. Native rotary arithmetic, padded
attention, preservation of query memory layout, and a memory-gate scalar
correction resolve the measured discrepancies. Original-sequence error falls
from experiment 002's 0.906 nats to zero. The wider failed controls remain
published alongside the corrections. The
[forward diagnostic harness](docs/runtime.md) has no backward
implementation; no rows have been trained.

The experimental prototype recorded 22 guard and plumbing tests passing; these were not real-model gradient validation. The checker in this publication is a separate evidence audit. No teacher model was called and no selected rows were optimized on correction examples. Only a manually specified diagnostic perturbation was evaluated.

Broader execution coverage and directional-gradient qualification remain
prerequisites to training. A development-only task search is also needed to find
a repeatable reminder advantage before testing the behavioral hypothesis. See
the [source audit](docs/source-audit.md), [architecture audit](docs/architecture-audit.md),
and [experiment protocol](docs/experiment-protocol.md).

## Attribution

This work builds on [ENGRAFT](https://github.com/fulvian/engraft-ngram/tree/028129c9c5c50fddb09b1503f6ccae07349b9831) and a [llama.cpp fork](https://github.com/LaurentZuijdwijk/llama.cpp/tree/5e085d123eead2e89b5c19f824fccb05727da6a2). Our checkpoint layout, engine, and experimental objective differ from ENGRAFT's reported fact-grafting experiment; our parity failure is specific to the configuration tested here.

The implementation and report were prepared with AI assistance. Results are tied to saved artifacts; the central hypothesis remains untested. See [NOTICE](NOTICE) for source attribution and [CITATION.cff](CITATION.cff) to cite this research snapshot.
