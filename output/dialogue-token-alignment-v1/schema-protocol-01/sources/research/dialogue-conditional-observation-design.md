# Conditional observation decoding with a known previous value

Prospective source-only design. No rows were extracted, predictions inspected,
encoders called or models fitted for this note. This changes the scientific
question; it cannot satisfy or relabel the failed V2, token-training or kernel
qualification rules. A failed speed cell is not a reason by itself to launch
new fits. The closed token fits remain unresumed and unscored.

The [V2 results](dialogue-copy-v2-results.md) leave observation interpretation
and excessive retention unresolved. The [one-step analysis](dialogue-selective-one-step-analysis.md)
shows that matched scalar and selective updates coincide at a one-hot prior;
it does not explain separately trained models. This proposal asks a narrower
question: **with the previous categorical value supplied correctly, can
candidate-conditioned token evidence decode the next value better than
slot-conditioned evidence?** It tests conditional observation decoding, not
deployable memory or full dialogue state tracking.

## One nine-fit comparison

Train three arms with three paired seeds, fixed prospectively before fitting:

- **Mean:** the existing normalized, chunk-weighted sentence vector.
- **Slot:** attention over existing contextual token vectors using `[query; query]`.
- **Candidate:** the same attention using `[query; candidate]`.

Reuse the authenticated pooled, token, schema and lexical caches. Slot and
candidate use the existing attention formula, chunk priors, token support and
64-dimensional attention projections described in the
[token design](dialogue-token-evidence-design.md). No new encoder, tokenizer,
LLM, text template or lexical rule is introduced. Candidate versus slot is the
primary contrast. Mean is a secondary, smaller control; its missing attention
parameters must not be disguised as equal active capacity.

All arms use the same shared candidate scorer and unrestricted masked softmax
over the current supplied candidates. Supply the previous gold value as a
one-hot candidate feature, with zero entropy. This is an explicitly privileged
diagnostic input in both training and evaluation. There is no retain/write
mixture, learned recurrent state, sequence rollout or backpropagation through
earlier turns. Current gold is only the supervised target. No operation label,
future annotation or model prediction becomes an input.

Implement a pure conditional scorer, not either existing recurrent `_advance`
branch. The existing readout branch resets belief to NONE; the scalar branch
would apply the excluded retain/write mixture. Reuse the common projections,
feature network and writer tensors, explicitly construct the supplied previous
one-hot and zero entropy, then return masked writer softmax directly. Do not
instantiate an unused departure head or silently discard the supplied prior.

The remaining inputs are identical: the CURRENT cached preceding-SYSTEM/current-USER
context, current query/candidate embeddings and all ten inherited public lexical
features. In particular, `literal_current` and `literal_previous` retain their
original causal prefix-derived register history. Thus this is not a history-free
language probe. Earlier text and hidden neural state are not added. Candidate
permutation must permute the previous-value indicator and outputs consistently.

Within each seed, copy all common scorer initialization tensors across all
three arms and both attention projections across slot/candidate. Use identical
eligible rows, epoch orders, objective weights and final-only evaluation; no
best seed, checkpoint or arm selection. Report registered and active parameters
separately where needed.

## Exact prospective row admission

Use only the original authenticated train/dev packet and cache metadata.
The packet's `time` is the index in its complete public USER-turn sequence,
including USER turns without scored annotations. For each scored current row
at index `t > 0`, require exactly one scored row for the same query at `t-1` in
the same dialogue and split. Same query means exact service/slot identity,
validated against the catalog, not merely a reused integer array index. Two
consecutive scored records separated by an unscored USER turn do not qualify.
No inferred previous NONE value is supplied at the first turn or across a gap.

Resolve the previous label through its own catalog to its exact canonical
candidate ID, then look up that ID in the CURRENT candidate list. Reserved
`reserved:NOT_MENTIONED` and `reserved:DONTCARE` remain distinct from literal
`value:...` IDs. Never transfer a numeric label index between catalogs, match
embeddings approximately, case-fold an ID or substitute the current gold.
If candidate order or membership differs, retain the row whenever the exact
previous ID exists in the current list; otherwise exclude it and report its
full count. Duplicate or inconsistent IDs/labels are validation failures, not
silent exclusions. The current target must validate against its own catalog.

Publish the admission ledger by split and seen/unseen panel: all scored current
rows, first-turn rows, rows lacking an adjacent same-query scored predecessor,
eligible adjacent rows, absent-previous-ID exclusions and final admitted rows,
with dialogue/query counts. After adjacency is established, absence of the
previous candidate is the only extra coverage exclusion. Do not filter by
correctness, confidence, mention presence, loss, prior model results or subgroup
performance. This annotation-selected diagnostic cohort is not an online actor
update mask and cannot support all-turn rollout claims.

## Outcomes and controls

Derive changed versus retained state by exact previous/current candidate IDs.
Report unmentioned retention, assigned retention, first assignment, revision
and clear separately as well as combined changed state. Report accuracy, NLL
and Brier on all admitted rows and these groups for seen/unseen services.
The single primary estimand is candidate-minus-slot changed-state NLL on
unseen services: take the mean over all admitted changed-state query-time rows
for each fit, subtract the paired slot value, then average those three seed
differences. Negative favors candidate conditioning. Retain each paired seed
difference. Seen-service results, accuracy, retention, and equal-dialogue means
within each group are fixed secondary outputs, not alternative primaries.
Report scored-row, distinct-query and dialogue counts. Empty groups are
undefined, never zero or a pass. No efficacy cutoff is set here.

Report current TRUE, FALSE and DONTCARE targets separately, including their
changed/retained breakdown. Boolean membership follows the existing canonical
true/false ontology definition; DONTCARE is the reserved ID. The published full
unseen panel has 524 TRUE and 179 DONTCARE targets and zero FALSE targets.
Those are not the new adjacent-panel denominators: recount after admission.
This subset cannot establish general polarity or negation performance.

Two deterministic controls use exactly the same admitted rows: carry the
supplied correct previous value, and the unchanged public literal-mention/carry
reference at the current turn. Report their accuracy and counts; do not pretend
they provide learned probabilities or finite probabilistic scores. Carry's
retention advantage is an intentional privileged baseline. No method gets a
denominator conditioned on its own previous correctness.

## Interpretation and execution prerequisite

A candidate-over-slot gain would support useful candidate-specific extraction
within this conditional probe. A gain only over mean supports a broader
observation-adapter explanation. Failure would not identify recurrence as the
cause: local-context sufficiency, training and scorer limitations remain open.
Even success does not establish usable recurrent accuracy, calibration, unknown
candidate handling, RL, a world model or a new attention principle. Development
has been repeatedly exposed; the official test remains untouched.

Frozen-V2 inference-only gate surgery is insufficient causal training evidence.
Forcing writes or changing the update uses factors trained jointly with the old
recurrence; propagating modified states also changes the distribution of the
belief/entropy inputs seen by its scorer. A one-step intervention on the original
trajectory can describe repairs and harms, but cannot establish the performance
of a trained replacement. Supplying gold priors to frozen V2 would introduce a
further input-distribution change. This probe instead trains both compared
decoders under the same explicitly privileged condition.

Before any data-backed model run or fitting, a separate metadata-only cost design
must bind eligible membership, token lengths and shared-cache identities. Then
prospectively freeze the objective, train-only weighting rule, batch construction,
epochs/update budget, seed list, optimizer, numerical checks, whole-run wall and
storage caps, source/runtime closure and exclusive output paths. Retain all nine
fits or the complete failed prefix, with no retry, resume or partial-result
selection. Charge cache gathering, validation, monitoring, optimization, reporting
and the inherited cache preparation cost explicitly. Removing sequential training
does not establish a runtime saving; none is claimed here. This note authorizes
no execution and does not relax any earlier gate.
