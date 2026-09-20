# Does row-uniform training improve on fixed loss-weight correction?

Prospective six-fit development study, 20 September 2026. No new fit has run.
Implementation, synthetic checks, metadata freeze and CPU coordination must
complete before launch. Earlier studies retain their original outcomes,
including token alignment's **FAIL, 18/22**. A positive result here would support
an objective choice, not establish a new architecture or complete the broader
research goal.

## Question and fixed comparison

The [fixed correction](dialogue-weight-prior-results.md) improves retention but
misses more real changes in every fitted model. It changes only the readout.
At an ideal weighted-cross-entropy optimum, removing the known loss weights
recovers the unweighted posterior. Finite capacity, optimization and
regularization need not realize that equivalence. This study asks whether
training with ordinary row-uniform cross-entropy improves the prespecified
changed/retained operating point over that fixed correction.

Use the unchanged token-aligned model for two training arms:

| Arm | Training loss |
|---|---|
| stratum | Existing weights N/(3 × FIT stratum count) |
| uniform | Ordinary per-row cross-entropy, every weight one |

There are three paired seeds, **6201, 6202, 6203**, and six fresh fits. Fresh
means new optimization from initial tensors, never old fitted checkpoints.
Reuse the historical initializer draw sequence, then copy the aligned
initializer into independent models with no parameter-storage aliasing. Copy
all model tensors exactly within each pair. Do not introduce another head,
change the model, update the encoder, or add recurrence.

Within seed 6201 train stratum then uniform; for 6202 uniform then stratum;
for 6203 stratum then uniform. Both arms use the byte-identical existing row
orders, actor inputs, batches and update counts. Reusing these seeds aids
pairing and does not create independent data replicates.

## Inputs and optimizer

Inherit the exact 29,211 fitting and 13,599 evaluation rows, the held-out-service
exclusions, frozen MiniLM context/schema caches and public candidate types
from the completed token-alignment campaign. Primary support stays 7,819
held-out-service rows: 578 changed and 7,241 retained, including 4,032
unmentioned and 3,209 assigned retentions. The correct previous value remains
a privileged evaluator input. There is no official DEV inference or TEST
access and no autonomous recurrent rollout.

The inherited plan SHA256 is
`e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1`.
Its metadata-freeze completion is
`8a2fb2659efed6ac187e92095eef3b3214beacf4ddcbcd7101664c9ef4c57e5f`.
These differ from the historical training completion
`e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c`.
Authenticate their proper roles and every inherited cache/order/source
identity before use. No historical weights are loaded.

Keep 20 epochs, effective batch 256, microbatch 32, float32, deterministic
operations, AdamW learning rate 0.001, weight decay 0.0001, gradient clipping
1.0, four Torch threads and one interop thread. Hidden and projection widths
remain 64, with 124,482 registered trainable parameters per model. Each fit performs 2,300
updates. Divide each accumulated effective-batch loss by its actual row count,
including the tail batch; do not average microbatch means equally.

The runner saves every final model and all evaluation distributions before
quality scoring. No checkpoint selection, early stopping based on outcomes,
temperature search, correction-strength fitting, replacement seed or retry.
Preserve journals, work counts, normalization checks, initialization hashes,
per-fit timings, live-resource observations and any partial terminal state.

## Four fixed readouts per seed

The saved-only report computes all four cells from the two trained models:

| Trained objective | Unweighted target readout | Stratum-weighted target readout |
|---|---|---|
| stratum | Normalize log p minus log w | Original normalized log p |
| uniform | Original normalized log p | Normalize log p plus log w |

Use the same hypothetical candidate transition and analytic float64 weights
specified in the [correction diagnostic](dialogue-weight-prior-diagnostic.md).
No current label enters construction. Preserve supported candidates, negative
infinity padding, log-space normalization and first-index exact tie handling.
The primary comparison is **uniform original versus fresh stratum corrected**.
The secondary fixed comparison is **uniform reweighted versus stratum original**.
No readout or coefficient is chosen afterward.

Keep all three historical corrected families visible as fixed external
development references: flat, token mean and aligned. Their summary SHA256 is
`97e6388f89020628854b3274b8c6d05f8524ea69e7a31931738840ff1bbbd961`,
receipt `a0b5209b81a81f33bcebb833b3badc8e7fcf4144ceda5eb2f64bd03adeb67975`,
and independent audit `b1f802b477fddfc3c11bf304588753780536d580ec36037acba04290bc96d771`.
Report historical normalized original families too for context. Historical
controls share data/seeds but are not new contemporaneous fits or equal-capacity
models. Their timings must retain their original collection conditions.

## Fixed decision rules

Compute the following using full-precision primary held-out-service metrics.
Means give each of the three optimizer seeds equal weight. Percentage-point
limits below are absolute differences, not relative percentage changes.
These are development continuation requirements, not significance tests.

**Objective evidence requires all seven checks:**

1. Uniform original improves mean changed accuracy by at least **2 pp** over
   fresh stratum corrected under row weighting.
2. The same **2 pp** minimum holds under equal-service weighting.
3. Mean retained error does not increase under row weighting.
4. Mean retained error does not increase under equal-service weighting.
5. Mean overall NLL does not increase under row weighting.
6. Mean overall NLL does not increase under equal-service weighting.
7. In at least **two common paired seeds**, changed accuracy strictly improves
   and retained error does not increase under **both** weightings. Separate
   favorable seed subsets do not satisfy this joint check.

**Practical continuation additionally requires all six checks:**

1. Uniform original exceeds historical **corrected flat** mean overall row
   accuracy by at least **0.25 pp**.
2. The same **0.25 pp** minimum holds for equal-service overall accuracy.
3. Mean overall row NLL is no worse than corrected flat.
4. Mean overall equal-service NLL is no worse than corrected flat.
5. Mean primary TRUE false-positive rate is at most **0.5 pp** above corrected
   flat. This uses the inherited 2,039 eligible rows and row aggregation, then
   equal-seed averaging; it is not an equal-service false-positive rule.
6. In at least **two common same-numbered seeds**, overall accuracy strictly
   improves and overall NLL does not increase against corrected flat under
   **both** row and equal-service weighting.

All thirteen checks are required for continuation. Keep objective and practical
decisions separate in the report. A passing objective result with a failed
practical result is not sufficient. Partial completion, missing or nonfinite
required metrics cannot pass. Do not retroactively apply this new rule to an
earlier study or change the original 18/22 outcome.

The secondary reweighted comparison is descriptive and cannot substitute for
failed primary checks. Report every seed, all/seen/held-out panels, per-service
effects, changed/retained subdivisions, branch/value errors, NLL, Brier, rare
recall/false positives and paired repair/harm tables. Preserve undefined
zero-support rates. Do not exclude small or unfavorable services after results.

## Execution allocation and validation

The six-fit training phase has a **6,000-second whole-run cap**, **6 GiB
process-lifetime peak-RSS limit**, and **512 MiB output limit**, including input
authentication, model initialization, training, evaluation, saving and
finalization. Require a separate CPU coordination check before launch; this
note is not an active resource reservation. No other OpenJev numerical job
may overlap it.

Historical aligned fit times were 760.896, 759.475 and 763.863 seconds. Doubling
their sum projects **4,568.47 seconds, or 76.14 minutes**, before shared
authentication/finalization. The cap provides about 31% headroom. This is an
estimate using the same model and workload, not a measured guarantee for the
new run. No additional encoder or speculative capacity campaign is authorized
by this protocol.

Before launch, synthetic tests must cover independent but identical initial
tensors, exact row-order identity, uniform CE equivalence, both objectives'
microbatch accumulation including tails, public actor-input independence from
targets, four-readout identities, decision-rule boundaries, incomplete/corrupt
run rejection and failure preservation. Freeze source, tests, this protocol,
runtime, input identities and orders in an exclusive metadata directory.
Publish that freeze before training. Metadata preparation is bounded to
60 seconds, 1 GiB process-lifetime peak RSS and 128 MiB output, with no model
forward calls or feature-array decoding.

After all six fits complete, the saved-only report and independent primary
check each receive 60 seconds, 1 GiB peak RSS and 64 MiB output. The independent
check reconstructs all twelve readouts across five primary strata, both paired
comparisons and all thirteen decisions, using separate correction arithmetic.
It authenticates selected saved inputs; it does not replay training or claim
independent verification of every descriptive panel. Publish the
complete outcome, figure and receipts even if continuation fails. Positive
development evidence would still need untouched confirmation and a separate
experiment demonstrating a new architecture's benefit.

For the main report's technical checks, the pinned prepared row metadata,
split and integer schema index/offset metadata may be read to reconstruct
work counts. Authenticate each before decoding. The main report may hash
saved weights for identity but must not deserialize them or load floating-point
feature caches. The independent primary check reads neither schema metadata
nor weights, orders or training journals. Both make zero model or encoder calls.
