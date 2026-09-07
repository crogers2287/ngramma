# Independent audit and one bounded harder discovery proposal

This is a proposal, not an evaluated family. No model process, holdout evaluation, teacher call, or sealed-family access occurred. The frozen task generator, batch runner, and plans were not changed.

## Audit of the current frozen curriculum and runner

No oracle error was found. Decimal comparison uses decimal arithmetic rather than binary float rounding; versions use integer components; signed arithmetic uses explicit parentheses; state updates implement replacement; missing-field checks use membership rather than truthiness. All families and key subtypes counterbalance answer positions. Five embedded synthetic tests pass. The original tasks may simply be too easy for the checkpoint; that is an empirical discovery outcome rather than justification to count malformed answers as content mistakes.

Two items should be resolved explicitly between model processes:

- `run_batch.py` rejects a job only when `response.get('ok') is False`. A JSON response with missing `ok`, missing completion text, or the wrong schema could be retained in an apparently completed batch. Require literal success and expected generation schema; verify returned prompt tokens, complete text, generation budget, greedy/unconstrained settings, and fresh-state metadata before scoring. These are evidence-hardening checks, not evidence that the current native lens misbehaved. Source inspection of the lens shows a memory clear per generation job and unrestricted vocabulary argmax. The driver itself does not construct or grade prompts, so the jobs builder must keep answer/oracle fields out of tokenized prompts and bind the exact chat serialization.
- `PLAN-task-design.md` requires at least two reproducible well-formed content errors with repeated hint rescue; root `PLAN.md` says hint benefit only strengthens selection. Freeze one authoritative gate before selecting a family. If a family is pursued without hint rescue, report that changed rationale explicitly; do not later claim the reminder-benefit prerequisite passed.

Strict whole-completion grading is appropriate, but a well-formed `A` cut off at a generation cap is not evidence that the model would stop there. Report cap termination independently; use the fixed completion/EOS rule consistently, with no stopping on the first answer letter. Fresh repeats are reproducibility checks, not independent statistical samples. The root plan's generation qualification and original-row overlay controls remain necessary before optimization.

## Proposed family: exact overlapping substring counts

If all first 16 tasks succeed, add **one** eight-item development family, `overlap_count`. It asks for a numeric property of a supplied symbolic string, not a formatting trick or a teacher's judgment. The challenge is tracking every starting position, including overlaps and interrupted runs. Inputs use only literal ASCII A/B, patterns have length 3 or 4, and strings have 24–26 symbols.

Use this shared unhinted instruction exactly:

```text
Count overlapping occurrences of {pattern} in {string}.
A: {choice_a}
B: {choice_b}
Answer only A or B.
```

The separate procedural hint is:

```text
Check every starting position. After a match, advance one character, not the pattern length.
```

The hint provides an algorithm, not the instance's answer or a solved example. Preserve the fixed main instruction across development examples to permit a later search for content-associated existing n-gram rows. Similar text is not proof of shared model rows: observe actual tokenization and row addresses, then exclude universal delimiters and measure control collisions. Do not target the A/B answer suffix merely because it is shared.

### Eight proposed development items

| ID | String | Pattern | Choice A | Choice B | Oracle answer |
|---|---|---|---:|---:|---|
| 1 | `ABABABABABABAABABABABABABA` | `ABA` | 12 | 6 | A |
| 2 | `AAAAAAAAAAAAABAAAAAAAAAAA` | `AAA` | 21 | 20 | B |
| 3 | `BAABAABAABAABBAABAABAABAAB` | `BAAB` | 4 | 8 | B |
| 4 | `ABABBAABABABAABABBAABABABA` | `ABA` | 8 | 9 | A |
| 5 | `AAAAABAAAAAABAAAAABAAAAAA` | `AAA` | 6 | 14 | B |
| 6 | `AABAABAABAABAAABAABAABAABA` | `AABA` | 8 | 9 | A |
| 7 | `BABABABABABABBABABABABABA` | `BAB` | 11 | 6 | A |
| 8 | `AAAABAAAABAAAABAAAABAAAA` | `AAA` | 11 | 10 | B |

Labels are balanced four/four. Half the distractors are the non-overlapping count, and half are one above the correct count. Therefore neither choosing the larger number nor always counting without overlaps solves all tasks. Each distractor class also has balanced answer labels. Do not disclose this construction to the model.

The table's counts were checked using two independent standard-library forms, with no model:

```python
import re

def oracle(text, pattern):
    if not pattern:
        raise ValueError('Empty patterns are excluded')
    count = sum(text.startswith(pattern, i)
                for i in range(len(text) - len(pattern) + 1))
    assert count == len(re.findall(r'(?=' + re.escape(pattern) + r')', text))
    return count
```

Both correctly handle overlapping matches; `str.count(pattern)` alone does not and serves only as one distractor. A production generator should retain matching start indices as audit metadata, never in the model prompt, and test boundary matches, absent patterns, one-character shifts, and interrupted runs. Keep using the current whole-completion A/B grader and unconstrained actual generation.

## Bounds, held-out design, and progression

Tokenize exact chat-serialized prompts before running: ASCII length is not tokenizer length, and repeated strings may tokenize unevenly. Prefer fewer than 60 prompt tokens where possible; reject or uniformly shorten the preregistered family before evaluating if it breaches the fixed context/generation budget. Do not truncate individual inputs after inspecting model errors. Preserve the original 16-task success result and label this as an explicit discovery extension.

Run only these eight unhinted items and their eight hinted contrasts first. Repeat prospective content-error/hint-rescue cases and unhinted controls in fresh state. Keep the original eight easy controls plus four newly fixed counting controls covering zero matches, one match, nonoverlapping matches, and an obvious two-character overlap. They must be baseline-correct preservation checks, not optimized targets. If this family also produces no reproducible well-formed content errors, report no usable target after this extension; do not silently lengthen strings or keep inventing variants until a failure appears.

Before any row search, lock a new held-out generator seed, input-length range, pattern set, answer placement, duplicate/exact-reversal rejection rule, and evaluation count. One reasonable bounded holdout is 16 newly generated strings with the same alphabet, 24–30 symbols, patterns of length 3–4, positive and zero counts, and balanced correct labels/relative choice magnitude. Reserve distinct strings and their A/B symbol complements from development. Include a fixed paraphrase subset to distinguish shared-instruction dependence from transfer. Do not evaluate or inspect model answers on these inputs during discovery or search.

This is still a single synthetic counting family. It can establish a narrow generated-answer improvement if a locked row edit passes paired holdout and preservation checks. It cannot establish general string reasoning, robust mathematics, unseen-family transfer, or that the model is reliably smarter overall. Shared prompts, shared patterns, and related run structures create correlated examples; eight items do not constitute eight independent capabilities. A later broader claim needs new families and stronger locked evaluation.
