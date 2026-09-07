# Discovery follow-up, fixed before its outcomes

The first complete run (`discovery.json`) returned only three well-formed
choices on 16 development tasks: one correct and two wrong. Thirteen began
explanations or added a prefix. Six of eight controls were correct; two were
malformed. All malformed responses remain failures in that protocol. No row
search or held-out evaluation has occurred.

This leaves too few content decisions to find the intended reminder contrast.
Keep the original tasks and four-token generation cap. Test one uniform system
instruction for the response format, without giving any content rule:

> Reply with exactly one character: A or B. Do not explain your answer.

This is an explicit change to the prompt protocol, not a model improvement.
The content hints remain a separate condition. No grammar, restricted token
list, answer prefill, or stopping on A/B is introduced. The original prompt-only
results will not be pooled into the revised baseline.

The fixed follow-up batch contains:

1. A fresh-process repeat of the original 16 ordinary tasks.
2. The original 16 hinted tasks, without the new system instruction.
3. All 16 ordinary tasks plus eight controls with the format instruction.
4. The eight previously specified counting tasks, eight hinted contrasts, and
   four counting controls with the same format instruction.

Adopt the revised protocol for any content search only if at least **22 of 24**
original ordinary/control answers are whole A/B choices that terminate at EOG.
Use this formatting threshold regardless of correctness. Otherwise reject the
revised protocol and report that this bounded assay did not establish a usable
content target. Counting is an explicit extension because the original batch
had no eligible two-error family; its inputs were fixed before this follow-up.

Any proposed target still needs two reproducible wrong choices rescued by its
hint, without losses on previously correct items. Repeat proposed rescues and
the relevant preservation controls before row selection. No holdout input is
evaluated in this follow-up. Source validation hardening between runs does not
change model computation; replay every saved response through the stronger
validator as well.
