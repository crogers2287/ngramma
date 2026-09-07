# State-replacement target confirmation

The revised format condition passed its fixed gate: 24/24 answers were whole
A/B choices terminating at EOG. Development content accuracy was 14/16; both
wrong choices belonged to `state_override` (`dev-state-1` and `dev-state-3`).
Seven of eight easy controls were correct. These are prompt changes, not row
edits. Counting returned A on every ordinary/hinted item: 4/8 in both conditions,
with no hint rescue. It does not qualify for the proposed search.

Before accessing rows or searching, run `state_confirmation_jobs()`:

1. Repeat the complete 16-task revised baseline and eight controls in a fresh
   process, with the exact same format instruction and four-token cap.
2. Run all four state tasks with their already fixed procedural hint.
3. Repeat those four hinted tasks, then the four ordinary state tasks, within
   the same process after other tasks. Compare tokens, text, termination, and
   selected logits to catch stale state or non-repeatable responses.

The target qualifies only if both previously wrong state choices are still
wrong ordinarily and both become correct with the hint in each repeat, with no
loss on baseline-correct state tasks. Otherwise the predeclared content search
stops with no eligible target. No held-out inputs are used.

Tokenize the common instruction substring ` In order: set x to` and the fixed
hint through the same runtime. Tokenization has no target-answer information.
If the gate passes, use the deterministic row-selection and 12-candidate bounds
in `SEARCH-DESIGN.md`; no row or direction has yet been scored.
