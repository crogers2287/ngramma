# Behavior discovery: the usable signal is instruction following

**A stronger response-format instruction made this small assay usable. The
tested content reminders did not repair the repeatable reasoning errors. No
memory edit was searched or learned in this experiment.**

The frozen model answered programmatically graded questions with unrestricted
greedy decoding. We requested exactly A or B, allowed four generated tokens,
and required the entire completion to be a correct choice ending at EOG. An
explanation or an unfinished answer failed this narrow contract. These scores
are not general model accuracy and do not establish that an explanation would
eventually reach a wrong substantive answer.

## Recorded outcomes

| Condition | Correct | Wrong choice | Malformed | Total |
|---|---:|---:|---:|---:|
| Original development prompts | 1 | 2 | 13 | 16 |
| Original prompts, fresh-process repeat | 1 | 2 | 13 | 16 |
| Original prompts plus content hints | 6 | 0 | 10 | 16 |
| Development prompts plus uniform format instruction | 14 | 2 | 0 | 16 |
| Easy controls plus format instruction | 7 | 1 | 0 | 8 |
| Counting, with format instruction | 4 | 4 | 0 | 8 |
| Counting plus content hint and format instruction | 4 | 4 | 0 | 8 |

The format instruction was:

> Reply with exactly one character: A or B. Do not explain your answer.

It passed the predeclared formatting gate: 24/24 original development/control
answers were complete A/B choices, exceeding the required 22. The original
prompts already asked for A/B; the additional system instruction supplied a
stronger reminder. Its improvement belongs to prompting, not memory editing.
All seven initially correct development/control answers stayed correct.

The two remaining development errors both involved replacing an earlier value
with a later assignment. For example, `x=2; set 5; add 3; set 3; add 4` ends
at 7, but the model chose the distractor 12. That distractor corresponds to
omitting the second assignment; the answer alone does not reveal the model's
internal reasoning. The fixed reminder about replacement did not correct
either error. All four state questions returned A ordinarily and with the hint,
including repeated runs, so each condition scored 2/4.

Counting also returned A on every one of its eight ordinary items, eight hinted
items, and four easy controls. Its 4/8 development score therefore gives no
evidence of useful counting behavior or hint rescue.

## What was established

The original 16 development answers and their recorded first-step scores were
identical across fresh processes. All 24 format-condition baseline answers and
scores repeated exactly. State ordinary/hinted repeats within one process also
matched exactly after other jobs, with state cleared each time.

The extended lens reproduced the earlier 17-token reference logits and tensor
metadata byte-for-byte through its unchanged capture path. Actual generation
now uses the real engine's attention/routing, no grammar, and no allowed-token
filter. Requested A/B logits are diagnostics only. All non-EOG control-token
spellings remain visible to the grader.

The content-search gate required two repeatable wrong choices in one family
that its fixed hint rescues. **Zero qualified.** The proposed 12-candidate
content search was therefore not run. No row was selected from held-out data,
no holdout model answer was evaluated, and the original pilot's sixth family
remains outside this work.

The useful follow-up hypothesis is narrower: can a small existing-row edit
retain the demonstrated benefit of the stronger **format instruction** when
only the original user instruction is present? That would test instruction
following, not establish new arithmetic or counting ability. It needs its own
declared objective and unseen-input controls; this report does not silently
relabel malformed answers as reasoning mistakes.

## Resources and reusable artifacts

The three completed CPU batches took 199.75, 760.20, and 416.63 seconds: 136
generated episodes, including repeats, and eight tokenization-only jobs. Peak
RSS was approximately 66.94 GiB. A first process was stopped before readiness
at the initial 64 GiB RSS bound; the retained record and pre-outcome amendment
raise that bound to 80 GiB while keeping a 24 GiB host reserve. No GPU or
production service was used.

The repository includes the bounded reusable generation runner, deterministic
task/oracle generators, exact prompt replay, stronger response validation,
synthetic failure tests, and a checked eight-row overlay exporter. The latter
was tested with synthetic rows; this experiment did not apply it to the model.

Reproduce with [REPRODUCE.md](REPRODUCE.md). Exact aggregate outcomes are in
[validation.json](validation.json), recomputed from [discovery.json](discovery.json),
[followup.json](followup.json), and [state-confirmation.json](state-confirmation.json).
The plans and failed allocation attempt are retained alongside the results.
