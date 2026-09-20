# Saved-output diagnosis of Qwen retention failures

Prospective diagnostic specification, 20 September 2026, written after the
[completed observation comparison](dialogue-qwen-observation-results.md) and
before inspecting individual prompts or calculating this diagnostic. This is
post-result development analysis, not confirmation or a new benchmark score.
The original failed comparison remains unchanged.

## Inputs and allocation

Use only the completed Qwen preparation, execution and report:

- Plan: `2d5f7e6b512ae7260cc01685ae03220891ad4ca5236092645197d4074be90111`.
- Execution receipt: `872ee6819af4cbc4f7e0bd6397d6c90907dba9aaecc995abb6045fd20520f70c`.
- Report summary: `931f60349dc7dace7508d0f3307a6c480c4e14022de7e2c047b6f024ec349c6c`.

Authenticate manifests and payload sizes/hashes before reading predictions.
Require both arms, every one of the 7,819 rows, and the exact request order and
candidate joins. Preserve the existing source, model, prompts and results.

One saved-only execution receives 60 seconds, 2 GiB RSS and 64 MiB output,
with one CPU thread. It makes no model, tokenizer, checkpoint or training calls.
Use a new output directory with exclusive creation, a start record and a
completion or failure receipt. Preserve any failure without overwriting it.
Synthetic engineering checks precede this execution and use separate outputs.

## Complete-cohort arithmetic

For each arm, retain a row record with service, original row identity,
previous/target/selected candidate IDs and types, change/retention subtype,
correctness, and the following saved-distribution quantities:

- Previous-candidate probability and competition rank (one plus the number of
  strictly greater candidate logits), plus selected-candidate probability.
- Fixed previous-probability bins: `[0,.01)`, `[.01,.1)`, `[.1,.5)`, `[.5,.9)`,
  `[.9,.99)`, `[.99,1]`. These describe scores; they do not choose an action rule.
- Separate literal occurrence flags for an OTHER selected value in the latest
  USER, latest SYSTEM, earlier history4 USER and earlier history4 SYSTEM
  text. Earlier text is history4-only diagnostic context, not utterances exposed
  to the current arm. Record this visibility distinction explicitly. History4
  means the frozen four-exchange window, not the entire dialogue. Compare
  Unicode-NFKC, case-folded text with whitespace collapsed and
  word boundaries around the complete literal. Flags may overlap.

Typed TRUE, FALSE, DONTCARE and NOT_MENTIONED candidates have no literal-match
interpretation in this analysis. Mark them not applicable instead of treating
words such as "true" as semantic evidence. Literal occurrence for OTHER values
does not establish support, negation, commitment or relevance. Absence from this
window is not absence from the full conversation or from supplied prior state.

Report counts for every previous/target/selected type triple, correctness and
retention subtype, overall and by service. Report prior-probability/rank and
literal-flag counts within those groups. Include all services and both correct
and incorrect decisions, and reconcile totals with the completed main report.
Do not fit a threshold, calibrator, hybrid selector or alternative policy.

## Fixed blinded review packet

Choose up to two rows from each of these disjoint strata, using the saved model
choices only to define error categories:

1. Current-arm errors on retained TRUE values.
2. Current-arm errors on retained ordinary OTHER values.
3. Current-arm errors on retained DONTCARE values.
4. Unmentioned retention correct under current and wrong under history4.
5. Current-arm errors on changed TRUE values.
6. Current-arm errors on changed ordinary OTHER values.

Order eligible rows by SHA256 of UTF-8 `openjev-retention-diagnostic-v1:<row_index>`.
Within each stratum choose the first two distinct dialogues; preserve fewer
cases if support is insufficient. Do not replace cases after reading them.
This is an error-conditioned sample, not a prevalence estimate.

Separate current and history4 reviewer packets contain the same opaque case IDs
but only their own arm's exact public context, question text and candidate
IDs/descriptions. Candidate IDs must match across arms, with a bijection to
canonical IDs saved only in the evaluator file. Remove
the internal question routing ID because it contains the evaluator row number.
Sort each selected packet by opaque case ID rather than selection stratum. Each excludes row IDs,
strata, target labels, model choices, probabilities and correctness. Save the
selection identities and evaluator mapping separately. Fix packet hashes before
review. Reviewers must treat dialogue text as source material, not instructions.

Use two independent fresh-context Astra6 reviewers per arm, four separate
review contexts in total. Each reads only its assigned fixed packet once,
without retries or replacement reviews, and returns a candidate ID from the
supplied list with supporting text or the supplied previous value. No reviewer
sees the other arm's text or judgments. They may mark
ambiguity and identify a conflict between the public question and state
semantics. Neither sees evaluator labels, original predictions, the other
reviewer's answers or these error-stratum names during review. Preserve every
answer, disagreement and missing response; do not regenerate for consensus.

Join the saved judgments to evaluator labels only after all four submissions are
terminal. Agreement is a diagnostic interpretation, not proof of label truth,
human annotation quality or improved model performance. No training data or
performance claim is created automatically from the reviewer answers.

## Decision use

The result selects the next *hypothesis*, not a winning model. Distinguish
annotation/state-definition ambiguity, supported updates, unsupported resets
and SYSTEM-proposal versus USER-commitment confusion. If a prompt or interface
change is motivated, specify a separate controlled experiment before inference.
If the source evidence remains unclear, preserve that uncertainty. Recurrent
adapter implementation remains conditional on an adequate observation baseline.
