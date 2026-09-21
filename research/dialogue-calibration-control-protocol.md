# A held-out temperature control for observation learning

21 September 2026. This protocol follows the completed V2 comparison and
saved-probability diagnostic. DEV has already informed this research direction.
It is development evidence, not untouched confirmation or a new calibration
method. The original V2 result remains a failure with six of seven conditions
passed. None of its sources, weights, saved predictions or decisions change.

## Fixed data and models

Use all four final V2 arms and seeds 6901, 6902 and 6903, in the original
seed-outer order. Restore the authenticated final memory state and, for
trainable arms, final encoder state. Frozen arms use the original pinned MiniLM
weights. No optimizer or model-weight update is permitted. Retain eager
float32 MPS encoding, float32 CPU scalar recurrence, original chunking/pooling,
candidate order, lexical variants and autonomous state initialized to NONE at
each complete dialogue. Future label availability cannot reset or mask state.

Select exactly **512** eligible source TRAIN dialogues, before obtaining any
new model scores. Eligibility means at least one categorical endpoint under
the existing supplied-schema task contract. It uses endpoint identity/query
metadata, not target values, transition outcomes or model scores. Remove all
2,017 actually fitted dialogue IDs and every whole-dialogue normalized-text
group containing any fitted or evaluated DEV dialogue. Use the existing
ordered-speaker NFKC/casefold/whitespace identity. Within each remaining group,
retain the smallest `SHA256("openjev-calibration-v1:" + dialogue_id)`, breaking
hash ties by dialogue ID. Select the 512 smallest remaining hash/ID pairs.
Stop if fewer than 512 eligible groups remain; do not replace the sample size.
Keep this order for every checkpoint. Record excluded counts, group digests,
selected IDs, services and complete workload geometry in the preparation.

This excludes direct normalized-text duplicates, not paraphrases or unknown
pretraining exposure. These dialogues are excluded from the twelve weight
fits, not guaranteed untouched by all prior research. Preserve their source
split as TRAIN and record `analysis_role=calibration` separately. Keep labels
outside actor payloads. Fresh packet indices and lexical arrays must preserve
the exact original public-text and candidate contracts. Official TEST remains
unopened. Report calibration-service overlap with the seen/unseen DEV panels.

## Qualification before calibration inference

Freeze source/test/protocol hashes and external input pins after synthetic
qualification. Authenticate V2 completion, all 64 original sources, the saved
diagnostic/audit, original preparation, materialized data and model assets
before decoding task inputs or loading weights. New outputs use exclusive
directories. Stop and preserve a failure; no automatic retry, replacement
sample, checkpoint substitution or budget extension.

Preparation has 300 suspend-inclusive seconds, 8 GiB RSS and 512 MiB output;
tokenization is allowed, pretrained neural inference is not. Record all actual
selected geometry before a timing experiment.

The replay qualification has 600 suspend-inclusive seconds, 8 GiB RSS,
8 GiB MPS driver memory and 256 MiB output. For every checkpoint, replay two
complete DEV dialogues: the first in original prepared actor order and the
dialogue with greatest original `padded_attention_positions` (ties by ID).
If these coincide, use the next-ranked workload dialogue. Preserve source DEV
and record `analysis_role=qualification`. Compare every saved endpoint, with
identical supported candidate IDs, finite float32 logs, mass within 2e-6,
unchanged canonical first-argmax choices, maximum supported log difference
at most 1e-5 and probability difference at most 1e-6. These checks do not
establish equivalence of internal states that V2 did not save; monitor every
new recurrent update under the unchanged invariants. Verify restored encoder
digests against each final-fit receipt and record memory digests before/after.

Record model-loading and complete synchronized dialogue costs separately,
including transfers, endpoint collection and output writing. Project full
calibration cost conservatively from all selected workload geometry, using
the maximum paid per-dialogue cost divided by each of encoder calls, padded
attention positions and real question updates across all replay cases; take
the largest resulting total and add all twelve observed loading costs and
preparation time. Precisely, for each work measure k compute
`max_replay(case_seconds / case_work[k]) * (12 * sum_512(calibration_work[k]))`;
take the maximum over k, then add the twelve loads and preparation. This
projects all 6,144 forwards, not one model's 512. Qualification admits inference only when projection is at
most 1,800 seconds. This is a screening estimate, not a runtime guarantee.

The separately launched complete inference allocation is **3,600**
suspend-inclusive seconds, 8 GiB RSS, 8 GiB MPS driver memory and 512 MiB output.
All 6,144 full-dialogue forwards must finish. Charge every checkpoint load,
tokenization, transfer, synchronization and artifact write; record exact
encoder work and state invariants. Save raw endpoint log probabilities and
canonical identities. Do not fit temperatures or score DEV during inference.
Use the native suspend-inclusive clock and a parent watchdog for neural phases.

## Scalar fitting and complete reporting

Fit one inverse temperature beta per checkpoint, on the same 512-dialogue
calibration cohort, minimizing unweighted endpoint NLL. Supported raw logs
are promoted to float64. Normalize `beta * log_prob` by stable masked
log-sum-exp at **every beta, including one**. Bounds are `[0.125, 8]`.
The derivative is the mean expected log score minus target log score.
Choose a boundary if its derivative implies a boundary optimum; otherwise use
exactly 64 derivative bisections. If all rows are exactly uniform over support,
choose beta 1. Report boundaries, derivatives, normalization-only drift and
fitted calibration loss. No floors, clipping, per-service fitting, learned gate
or DEV-based temperature selection is allowed.

Apply the fitted scalar only to existing final output distributions. It never
feeds recurrent state. Report raw, normalization-only beta=1 and calibrated
scores for every arm/seed, all/seen/unseen panels and three strata; preserve
the original five transition-bin and candidate-type breakdowns where supported.
Require identical canonical selected IDs and ties at all endpoints. Use the
original endpoint-micro NLL and candidate-sum Brier definitions and equal-seed
means. Do not treat optimization seeds as independent evaluation samples.
Fit/report reading has a separate 300-second, 4-GiB, 128-MiB allocation and makes
zero new model calls.

The primary comparison is calibrated trainable_numbers against **calibrated**
frozen_numbers. Apply the same seven original conditions in
[the scoring contract](dialogue-observation-learning-scoring.md) to this new
comparison, explicitly preserving the old raw failure. Five decision-based
conditions must be unchanged; only NLL and Brier can change. Four additional
unseen-service mean safeguards require trainable NLL to strictly improve over
its own raw output, trainable Brier not to worsen, frozen NLL not to worsen and
frozen Brier not to worsen. All eleven must hold for calibration-control
continuation. Report every paired seed difference and seen-service proper
scores, including regressions. No confidence interval, general calibration
guarantee or architectural attribution follows from these development results.

Passing supports an observation-model baseline with ordinary output
calibration. It does not establish new recurrence, biological wiring or a
world model. Failure means this fixed global-temperature procedure did not
resolve the tradeoff; it is not evidence that a more complex architecture is
necessary. Further mechanisms need their own matched experiment.

Prior art: [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).
Calibration under shift remains a separate problem:
[Ovadia et al., NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html).
