# Fixed continuation after candidate selection

This spells out the existing PLAN's repeat/holdout procedure while the twelve
candidate search is still running. It does not add candidates, change the
selection rule, or use held-out model responses.

If no candidate passes, stop this experiment without evaluating holdout. If
one is selected, only that winner may proceed. A failed confirmation ends
this round; do not substitute the runner-up.

1. Record the completed search, recipe, winner record, overlay, lens, runtime,
   and verification-library hashes. Require the model-free evidence checker
   and local overlay/input audit to pass before launching another model.
2. In a new unmodified model process, run the same 24 tasks twice, with fresh
   sequence state for each job. In another new process, run that same two-pass
   batch with the winner. The two tokenizer checks precede each batch. Compare
   actual generated tokens, complete decoded bytes/text, stopping status, and
   first-position requested/top scores. Require each condition's two passes to
   match exactly, the fresh original to match discovery, and the fresh winner
   to match its recorded search result. Reapply the unchanged admission rule.
3. Only a passing confirmation may write a locked candidate record. Bind it
   to all confirmation input/output hashes. Generate the 32 existing
   within-family holdout inputs plus eight original controls only after this
   lock. The holdout generator's oracle is programmatic; its answers never go
   into a model prompt.
4. Run this 40-task batch once on the unmodified model and once on the locked
   overlay, in separate fresh processes with the same CPU profile. Score the
   whole generated answer and require EOG, just as in development. Report
   paired gains and losses, correct/incorrect/malformed/unfinished counts by
   partition and family, and actual completion text.
5. A narrow positive result requires at least one new holdout success, no loss
   of a baseline-correct holdout/control answer, and no new malformation or
   nontermination of a previously well-formed terminated answer. Zero gain is
   not an improvement. Any failed condition blocks promotion in this round.
   Do not retune or select another candidate using this holdout.

The last rule makes "improves unseen tasks" concrete; it is not a claim of
statistical significance or general reasoning improvement. The holdout shares
templates and task families with development. Even a pass would justify only
a versioned experimental overlay for this response contract, pending a larger
independent-family and unrelated-task study.

All native processes retain the existing 900-second, 80-GiB-RSS, 24-GiB-reserve
limits. A timeout is an incomplete execution, not a failed model answer.
