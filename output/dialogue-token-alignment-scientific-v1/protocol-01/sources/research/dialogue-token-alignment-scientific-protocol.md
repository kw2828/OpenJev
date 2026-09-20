# Token alignment scientific campaign

## Resource decision made before quality evaluation

The previous [cost screen](dialogue-token-alignment-capacity-results.md) remains
**NOT ADMITTED under its original rule**. Its 77.27-minute projection exceeded
the self-imposed 48-minute admission threshold and 60-minute study ceiling.
Its [original scientific protocol](dialogue-token-alignment-study-protocol.md)
remains unexecuted. No task-quality result or fitted checkpoint was produced.

This is a separately identified campaign, `dialogue-token-alignment-scientific-v1`.
After observing that cost, allocate a hard **7,200-second local CPU budget**
before observing any quality result. There is no additional cloud API spending.
The 4,635.96-second projection leaves approximately 43 minutes of headroom,
although actual data gathering, checkpoint serialization and final hashing can
differ from the synthetic measurement. This explicitly revises the resource
allocation. It does not reinterpret the failed screen as a pass, restart a
partial fit, change a scientific threshold, or claim prospective cost admission.

Preserve the old screen, failed metadata freeze, correction, receipts and
published statements as historical evidence. Publish this protocol and the new
source/input/order freeze before any fitting. This campaign gets one exclusive
training attempt. Do not extend its budget or resume it after failure.

## Scientific question

Does bidirectional token alignment improve actual branch decisions beyond
both an input/parameter-matched token-mean comparison and the existing
candidate-attention scorer? The architecture adapts established
align/compare/aggregate attention. This is a development experiment supporting
a stronger observation model; it is not itself a new recurrent architecture,
biological-wiring result, calibrated filter or RL method.

Train `flat_stratum`, `token_mean`, and `token_aligned` at seeds 6201, 6202 and
6203, all from fresh weights. Rotate arm order by seed exactly as specified in
the shared implementation. Copy the complete new-model initializer between
mean and aligned arms. Copy only the ten named compatible query/candidate
projection, feature, head and branch-gate tensors from the baseline. Preserve
its own turn projection and candidate attention. Report the 124,482 new-model
and 173,506 baseline registered parameter counts, their shared 75,138 subset,
and the inactive entropy column/common softmax shifts. No equal-capacity claim
is made between the baseline and the new arms.

Use the exact completed schema cache of all 53 TRAIN query schemas and all
307 supplied candidate strings. Every candidate string already contains its
full service/slot question; do not add text, paraphrases or target selection.
The new models receive the same cached dialogue/schema tokens, pooled vectors,
lexical fields, public candidate-type flags and correct previous value. Their
only intervention is bidirectional soft alignment versus opposite-sequence
prior-weighted means. Both keep shared projection, comparison and aggregation
networks and flat candidate-plus-branch-gate normalization.

## Data and fixed training recipe

Reuse the original typed study's exact original-TRAIN membership: 29,211 fit
rows and 13,599 evaluation rows. Entire dialogues containing held-out services
are excluded from fitting. Primary evaluation has 7,819 rows from six held-out
services: 578 changed and 7,241 retained. These are historically exposed TRAIN
development examples. Official DEV inference and TEST access are excluded.

Copy the exact orders from `capacity-protocol-02`, whose plan SHA256 is
`3b28aac097d9c8c4f3eba38d9ee7ab50f3ed9786cd36249e4772005022f20eea`.
Do not regenerate or select permutations. Use 20 epochs, effective batch 256,
row microbatch 32, CPU float32, deterministic operations, four intra-op threads
and one inter-op thread. AdamW uses learning rate .001 and weight decay .0001.
Use ordinary three-stratum weights from fit counts 17,666 / 9,246 / 2,299;
there is no rare-type balancing.

Sum each microbatch's weighted negative log likelihood and divide by the
actual effective-batch row count, including the short final batch. Accumulate
gradients, clip their norm at 1, and update once per effective batch. All
candidate logits feed one supported-candidate normalization. Every row,
candidate and token stays in the comparison. No token truncation, label-based
update masks, early stopping for quality, checkpoint selection, alternate
seeds, or omitted arms.

Each fit has 2,300 optimizer updates, 18,260 training microbatches and 425 final
evaluation microbatches. The campaign totals 20,700 optimizer updates,
164,340 training microbatches, 3,825 evaluation microbatches, 5,257,980 training
row visits and 122,391 evaluation row visits. Save all nine final checkpoints
and all final 13,599-by-12 padded log-probability arrays. Save exact update row
identities, normalization checks, work counts and timings. Training losses are
numerical execution records. No evaluation-quality metrics are computed by the
training runner or inspected between fits.

## One continuation rule, unchanged from the original proposal

Decode and score final predictions only after all nine fits and their saved
artifacts pass authentication. Compare alignment separately with **both**
controls on the same 578 primary changed rows. Require all six conditions
against each control:

1. Mean row-weighted changed accuracy across three seeds increases by at least
   2 percentage points.
2. Mean wrong-selected-branch rate decreases by at least 2 percentage points.
3. Neither metric worsens at any paired seed.
4. Mean error on the 7,241 primary retained rows increases by at most .5 points.
5. Mean supported TRUE false-positive rate increases by at most .5 points.
6. Mean supported DONTCARE false-positive rate increases by at most .5 points.

Use exact count fractions for thresholds. Branch means the branch of the
selected candidate: NONE, DONTCARE or concrete. Do not substitute the argmax of
summed branch probabilities. A supported false-positive denominator contains
all primary rows offering that type whose current target is not that type.
Expanded for per-seed consistency, the rule contains 22 checks; technical
completion is a prerequisite, not an additional scientific win.

Report all fits, row/equal-service/equal-dialogue accuracy, raw NLL and Brier,
branch-versus-within-branch loss, actual wrong-branch/wrong-value partitions,
paired repairs and new mistakes, and per-service results. Keep previous-value
and literal-register accuracy references. TRUE and DONTCARE change recall
remain descriptive: support is 29 and five rows, with no FALSE changes or
clears. Repeated seeds are not new independent examples. Lower NLL or Brier
alone does not establish calibration or decision improvement.

## Limits, failure and reporting

The entire training attempt is capped at **7,200 seconds**, **6 GiB process
peak RSS**, and **512 MiB output**. Count authentication, gathering, padding,
training, evaluation, serialization and hashing. Record process load and
runtime; do not claim uncontended hardware timing. Report the original frozen
feature preparation and 4.995-second schema preparation separately from the
training total.

On failure, stop and retain the original error, completed fits, partial active
weights, any returned evaluation prefix, counters and artifacts. Do not score
an incomplete campaign. Do not resume, replace seeds, reduce support, extend
the ceiling or alter the rule to obtain a pass. The exclusive source freeze
is metadata-only; any technical correction before fitting must remain visible
and be published as a separate artifact.

A saved-output-only reporter and independent reader will verify results.
Neither may call the model or encoder. A passing engineering continuation rule
would justify further investigation, not an ICLR novelty claim, deployment
claim, or fresh confirmation. The full OpenJev research goal remains unresolved
until a useful architecture improvement survives stronger independent evidence.
