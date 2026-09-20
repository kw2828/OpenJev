# Conditional observation probe: prospective training protocol

Status: prospective specification. No training plan is frozen and no scientific
fit is authorized by this document alone. The completed, independently checked
[metadata preparation](dialogue-conditional-preparation-protocol.md) must first
bind the eligible rows and cache addresses. Then freeze this protocol, all
executable source, inputs, runtime and exact operation counts before one run.
This diagnostic does not reopen or satisfy any earlier failed experiment.

The question is whether candidate-conditioned token attention extracts more
useful current evidence than slot-conditioned attention when both receive the
correct previous categorical value. This is privileged conditional decoding,
not a rollout model or an architecture-novelty claim. The
[design](dialogue-conditional-observation-design.md) fixes the cohort semantics.

## Models, inputs and pairing

Train mean pooling, slot attention and candidate attention at seeds
**5301, 5302 and 5303**, for exactly **nine final fits**. Use the standalone
`DialogueConditionalObservation` scorer with input width 384, projection width
64 and hidden width 64. The token attention width is 64. There is no departure
head, recurrent state, carry mixture, future context or predicted prior.
The previous gold candidate is an exact supported one-hot actor input. Current
gold, transition class and panel/category flags are supervision or evaluation
metadata, never scorer inputs.

All arms receive identical current query/candidate embeddings and ten public
lexical features, including inherited literal-register history. Mean receives
the existing normalized pooled context. Slot and candidate receive identical
raw token states, positive chunk priors and masks. Slot uses `[query;query]`;
candidate uses `[query;candidate]`. Both execute the same batch-by-candidate
attention layout. Do not silently collapse duplicated slot queries to a cheaper
kernel, prune candidates, truncate tokens or substitute an optimized closed
qualification implementation. Mean is a smaller secondary control.

Construct fresh decoders for each seed. Explicitly copy every common scorer
tensor from the seed's canonical mean initializer into all three arms; copy
both attention tensors from the slot initializer into candidate. Hash the
initial common tensors and attention tensors to verify pairing. Do not load
trained V2 or token-study weights. Registered parameter counts and the known
zero-entropy weights/common-shift bias are disclosed; these counts are not
effective functional dimensions.

## Admitted-row objective and schedule

Use every admitted training row and every admitted development row from the
new metadata receipt. No model-dependent filtering or class subsampling.
Batch **256 query-time rows**, including the final shorter batch. Within each
seed, a NumPy `default_rng(seed)` produces twenty complete permutations of
the canonical admitted training-row order. All three arms use the identical
saved permutations for that seed. Evaluation uses canonical row order.

Train **20 epochs** with AdamW, learning rate **0.001**, weight decay **0.0001**,
gradient-norm clipping at **1.0**, float32 CPU execution, four intra-op threads,
one inter-op thread and deterministic Torch algorithms. Use no learning-rate
schedule, checkpoint selection, dropout or early stopping. Preserve all final
fits and all three paired seed comparisons.

Recompute three training strata from the admitted rows: unmentioned retention,
assigned retention, and changed state (first assignment, revision or clear).
Require all three to have nonzero training support. With training-row count
`N` and stratum counts `n_k`, assign `w_k = N / (3*n_k)`. For each batch take
the arithmetic mean of `w_k * -log_p[current_target]`, using the scorer's
returned log-softmax directly. Do not renormalize weights within a batch or
inherit the old full-cohort counts. This gives the three strata equal aggregate
training weight and prevents plentiful unchanged rows from defining the whole
learning objective. This weighting is fixed before cohort counts are inspected.
Report the unweighted evaluation metrics specified below.

Visit arms in a balanced fixed order: mean/slot/candidate at 5301,
slot/candidate/mean at 5302, candidate/mean/slot at 5303. The freeze must record
`20 * ceil(N/256)` updates per fit and nine times that total. A fit uses a fresh
optimizer. No warm-start, resume, replacement seed or partial-fit selection.

## Numerical and coverage checks

Verify the complete metadata ledger, manifest, source/runtime identities and
opaque parent cache hashes before numeric feature loading. Load existing float
arrays read-only. Check finite actor inputs, exact supported previous one-hot,
positive valid token priors summing to one within 2e-6, and valid candidate
support. Each scored row must receive finite supported log probabilities,
exact negative infinity off support and raw probability mass within 2e-6 of
one. Never repair output by flooring probabilities or renormalizing a failed
distribution. Reject nonfinite target NLL, training loss or gradients.

Retain a ledger of every update's exact row identities, input shapes, work
counts, normalization checks, loss and timing. Count actual optimizer attempts
and returns. The same seed's three arms must have identical row membership
and ordering. Retain evaluation row IDs and every final candidate log
probability for every fit, including masked entries. Match them one-to-one
with the common admitted-development ledger. Incomplete coverage cannot pass.

The current-target join must remain outside actor construction. Synthetic
tests must verify current-target/category changes cannot alter the same row's
actor, whereas the explicitly supplied previous target does. Preserve candidate
permutation, padding, row independence and initializer-copy invariants.

## Fixed analysis and continuation

The sole primary is candidate-minus-slot **unseen changed-state NLL**. For
each fit, average `-log_p[target]` over every admitted unseen changed-state
query-time row. Subtract the slot value within each paired seed, then average
the three differences. Negative favors candidate conditioning. Use saved log
probabilities directly, without exponentiation followed by a probability floor.
Report absolute and relative changes and all three individual differences.

Fixed secondary outputs are mean-pooling comparisons; seen/unseen all-row,
changed and retained accuracy/NLL/Brier; each of the five transition groups;
and current TRUE/FALSE/DONTCARE with changed/retained breakdowns. Report row,
distinct-query and dialogue denominators, plus equal-dialogue means within
each group. Empty groups are undefined, never zero. The absence of unseen
FALSE support cannot support general boolean or negation claims.

Carry the supplied correct previous value and the unchanged public literal
carry as deterministic references on exactly the same rows. The latter may be
read from the authenticated `literal_current` feature after verifying its
exact one-hot validity and inherited semantics. Report their accuracy only;
do not assign finite probability losses to wrong deterministic predictions.

Continuation requires all nine fits and every numerical, coverage and cost
check to complete, nonempty primary support, and strictly negative primary
NLL differences at **all three paired seeds**. A nonnegative pair closes the
probe under this rule. No alternative panel, aggregation, mean-only win,
favorable category or selected seed can rescue it. A pass permits designing a
separate memory experiment; it does not establish statistical significance,
practical importance, deployment quality or a novel architecture. Report all
secondary regressions and effect sizes even when this diagnostic rule passes.

## Complete cost and terminal states

Use one exclusive run directory after a separate published plan freeze. The
entire nine-fit execution has a **3,600-second wall cap**, a **6 GiB sampled
process-lifetime RSS ceiling** and **512 MiB output cap**. Do not start while
another known heavy local experiment is active. These are execution ceilings,
not forecasts or proof that this recipe will finish.

Charge authentication, cache loading, batch gathering/padding, validation,
all model and optimizer work, evaluation, counters, checkpoint/prediction I/O
and payload hashing to the run. Report fit/update timing separately with exact
scope. Attribute inherited pooled/token/lexical preparation time and bytes
separately, rather than claiming the shared caches were free. Total process RSS
does not establish per-arm memory consumption. Test/preflight costs are reported
separately and are not benchmark latency measurements.

On any failure or cap breach, preserve completed and partial artifacts and a
failure receipt. No retry, resumption, cap extension, reduced epoch count or
replacement output path under this protocol. Do not score partial development
predictions for a scientific comparison. Only after all nine fits complete may
a saved-output-only reporter compute the specified analyses. Publish the full
result, failed criteria and independent audit. Official test remains untouched.
