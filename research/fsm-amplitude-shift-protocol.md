# Frozen-weight FSM amplitude-shift confirmation: draft protocol

**Prospective, unregistered and unrun.** Proposed version `fsm-amplitude-shift-v1`.
This document specifies a new 300 mV contrast, not permission to read its arrays.
The closed [reference comparison](fsm-author-nllfr-factorial-results.md) satisfies
the prerequisite in the [proposal](fsm-amplitude-shift-proposal.md). Registration
still requires qualified adapters, an independent auditor, fixed resource caps,
all source/model/input hashes and a published pre-access commit. No existing
scientific source or checkpoint changes.

## Fixed question and roster

Does the previously selected tanh-feedback model retain its advantage over the
previously strongest eligible control under a larger applied-input amplitude?
The candidate is `tanh_feedback-lr0.0003`; the fixed reference is
`tanh_output_only-lr0.001`. Both use all seeds **9201, 9202, 9203**. Bind their
selection to the closed exposed-DEV audit, not to new 300 mV scores. Every other
declared control remains visible even if it outperforms either named family.

The canonical deployment list has **28 instances and 18 families**, in this order:

| Zero-based positions | Frozen deployment, in exact internal order | Instances |
| --- | --- | ---: |
| 0–8 | VARX order 32, 64, 96; within each order penalties 1e-6, 1e-3, 0.1 | 9 |
| 9 | Original native VARX32 backbone | 1 |
| 10–21 | For each seed 9201, 9202, 9203: affine output-only at 1e-4, affine feedback at 1e-4, tanh output-only at 1e-3, tanh feedback at 3e-4 | 12 |
| 22–24 | Existing folded affine-feedback coefficients, seeds 9201, 9202, 9203 | 3 |
| 25 | Completed author BLA28 and its qualified causal initializer | 1 |
| 26–27 | Completed new author NL-LFR checkpoint, with fixed 16- then 64-direction context policies | 2 |

The two NL-LFR deployments share the completed 51,065-iteration checkpoint.
The old capped checkpoint is excluded from this roster. There is no new fit,
optimizer, calibration, coefficient folding, rate/seed search, normalizer update,
checkpoint replacement, early checkpoint selection or reserve-driven rescue.
Use the already qualified folded coefficients. A future study of old diagnostics
would need a separately frozen roster and must not alter these denominators.

## Admission and information boundary

Before decoding, authenticate the whole opaque archive against the exact byte
size and SHA256 in registration. Bind all deployment files, normalizers, source
snapshots, qualified runtime identities, parent registrations, original process
closures and agreeing audits. Reject drift before the first array/model load.
Record all admitted hashes again at closure. No mutable external model service.

Use only the new [restricted reader](../src/openjev/research/fsm_shift_data.py).
Its original [qualification receipt](../output/fsm-shift-engineering-v1/reader-qualification-01/receipt.json)
records 41 fabricated checks plus Ruff, source `199982be6f23…`, test
`5f73c1897bd8…`, and no changes to the earlier four-member reader. The new reader
requires explicit snapshot expectations and decodes only `u_300mV_train` and
`y_300mV_train`: native float64 `[8192,3,6,2]`, finite, 6,400 Hz. Known other
member names may occur in the ZIP directory; their array headers and values
remain closed, including every official-test member. No download is authorized
by this draft.

The record order is realization 0 through 5, then period 0 and 1 within each
realization. Record IDs are `300mV-realization-r-period-p`; all twelve are
confirmation records. Do not concatenate, wrap, average periods, resample,
impute or remove records. Realizations 0–2 and 3–5 form two fixed six-record
reporting groups because their lower-amplitude counterparts supplied FIT and
exposed DEV, respectively. Neither group permits new fitting.

For each record use exactly 32 starts `s=256k`, `k=0,...,31`:

- Arrived output context: `y[s:s+100]`.
- Past applied-input context: `u[s+1:s+100]`, omitting unavailable `u[s]`.
- Forecast forcing: `u[s+100:s+228]`.
- Target, extracted for scoring after forecast return: `y[s+100:s+228]`.

`public_window(record,s)` returns only the first three arrays; `target_window`
is a separate scorer API. Initializers receive only the two context arrays.
The reader necessarily materializes the admitted output member for schema and
finiteness checks; target-after-prediction means separate target extraction and
no target argument, closure, or state accessible to prediction. Do not claim
that future-output bytes were globally unread at archive admission.

Run CPU float64 with the same registered thread/runtime settings for all models.
Residual/VARX models retain the original FIT normalizer. Author models retain
their own embedded FIT normalizers and final-coordinate causal initialization.
Use the original common FIT output mean/scale only for shared scoring. Every
request begins with fresh state; no future forcing enters state estimation.
All models receive C100 although their actual history consumption differs.

## Forecasts, evidence and arithmetic

Retain **10,752 forecast slots = 28 × 12 × 32**, and **336 record-instance slots**.
Use canonical record, start, then deployment order for forecast attempts.
Save each of the 384 legal public-input banks once, with exact identity links
from every attempt. Preserve physical and standardized forecasts, separately
obtained standardized targets, starts, forecast-start/final state, and all
author context-solver traces, statuses, accepted-step/work counts and initial
seeds. Frozen weight/buffer values must match before and after every deployment's
evaluation. No retained mutable state crosses requests.

For a complete record-instance bank, let `e` be standardized prediction minus
standardized target, using the common FIT output statistics separately on both.
Primary RMSE is `sqrt(sum(e**2)/(32*128*3))`. Also retain each channel's RMSE,
native-unit channel RMSE obtained by multiplying by that channel's common FIT
scale, and descriptive late-H64 RMSE from forecast positions 64–127. No mixed
native-unit aggregate is used. An instance score is the equal mean of its
twelve record RMSEs; a three-seed family score is the equal mean of its three
instance scores. Per-record family scores average the three seed RMSEs. A group
score averages its six record-family scores. No successful-subset average.

Known numerical forecast/score/solver failures remain explicit failed slots
with original errors and available inputs, states, traces and raw returned
arrays. Continue only where the registered numerical-failure classifier allows;
arbitrary schema, source, I/O or programming errors are fatal. On a process cap,
retain completed artifacts and mark remaining scheduled slots `not_run` from the
frozen ledger. Missing values are never zero. Finite capped/stalled NL context
solves are valid forecasts under the held policy, not convergence certificates.
Keep valid quality scores even when timing fails.

For each of three seeds, compare folded and unfused affine-feedback predictions
on all 384 requests at `atol=rtol=1e-9`. These are comparisons of scheduled banks,
not extra forecast calls. Retain every comparison, including finite mismatches;
failed/missing operands fail parity and make folded timing unavailable.

The independent auditor authenticates closure and inventories before numerical
loads, then reconstructs all available forecasts from legal inputs before
loading their target values. Require output and state parity at
`atol=rtol=1e-8`; author floating diagnostics use `atol=1e-10, rtol=1e-8`, with
exact identity, categorical status, trace branch and work-count agreement.
Compare finite/nonfinite patterns explicitly on retained failed arrays without
promoting them. Independently recompute scores, gates, folded parity and timing
aggregates. It does not refit, optimize or replay clock timings.

## Current matched costs and exact order

After the full forecast/parity phase closes, retain **28 warmup slots and 672
timed slots**, with 24 timed requests per deployment. Warm up each deployment
once, in canonical order, on realization 0/period 0/start 0. A known failed
deployment or failed warmup keeps all of its later cost slots explicitly
unavailable; it is never replaced or retried.

Let record index `r=2*realization+period`, from 0 through 11. Timing slots
`j=2r` and `2r+1` use that record's starts 0 and 7936, respectively. For slot j,
left-rotate the 28-instance canonical roster by `floor(j/2) mod 28`, then reverse
the entire rotated list iff j is odd. Persist the expanded schedule before any
timing, and record slot, position, instance and request identity on every call.

This paired reversal gives each primary candidate/reference seed pair twelve
orders in each direction. The pairs occupy base positions (12,13), (16,17),
(20,21); cuts 0–11 never split them, so they stay adjacent in all 24 slots.
Every model receives the same 24 requests. Position exposure across all 28
models is not balanced, and reversal is paired with first/last request endpoint.
This is deterministic matched timing, not randomized ordering or exclusive-host
measurement. Report these limitations with all samples.

Measure batch-one complete requests: fresh input copies, normalization, causal
initialization, every GN/Jacobian/line-search/SVD operation, rollout, validation
and physical output conversion. Include deadline checks uniformly. Exclude disk
I/O, target extraction, scoring and publication. No cached states, factorizations
or substitute inference path. Retain component times and actual solver work.
An instance latency is its median of 24 positive finite timed samples; family
latency is the equal mean of three instance medians where seeded. Historical
latencies cannot replace these measurements.

Report deployed numeric weights/coefficient buffers, required inference
normalizers, one-stream state, and numeric policy/metadata bytes separately and
as their sum. Do not double-count a shared tensor within one deployment; count
each deployment independently. Scoring-only statistics, Python objects and
request/workspace buffers are excluded from persistent storage and disclosed
separately. Report transient peak-workspace scope rather than hiding GN work.

## Eight frozen confirmation conditions

All eight are required for `CONFIRMATION_PASS`; otherwise the result is
`DO_NOT_CONFIRM`, with separate engineering and coverage status. Let C/R denote
the candidate and the fixed tanh output-only reference, never a reserve-selected
replacement. Criteria 3–6 are the four quality conditions.

1. **Complete roster:** all 10,752 forecasts, 336 record scores, 28 warmups and
   672 timing slots are valid; all 28 deployments retain admitted identities.
2. **Folded parity:** all three seeds pass every scheduled folded/unfused
   comparison at the stated tolerance. Missing comparisons fail.
3. **Mean improvement:** R's complete mean is strictly positive and
   `mean(C) <= 0.95*mean(R)`. Zero tying zero cannot claim relative improvement.
4. **Paired seeds:** for each of the three fixed seeds, C's complete equal-record
   mean is strictly lower than R's same-seed mean.
5. **Record harm:** on every record, C's seed-mean RMSE is at most
   `1.02 * R's seed-mean RMSE`. A zero reference requires zero candidate error.
6. **Both realization groups:** C's group mean is strictly lower than R's in
   both groups 0–2 and 3–5. This explicitly replaces the development
   two-amplitude criterion; there is only one confirmation amplitude.
7. **Latency:** C's equal-seed mean median request latency is at most
   `1.10 * R's`, using only complete current timings.
8. **Storage:** C's equal-seed mean persistent numeric bytes are no greater
   than R's; also display every instance's breakdown.

Report quality and cost conditions even when another condition fails, using
unavailable rather than invented values for incomplete operands. Other controls'
failures block completeness without erasing their valid scores. A named-contrast
pass does not establish that C beats every control. No post-result threshold,
reference, group, rate, seed or solver-policy change.

## Remaining pre-access work and claim limits

The runner interface should accept only a frozen registration and exclusive
output directory. Registration must carry the exact ordered deployment/record/
start/timing ledgers, all sources and model/normalizer pins, the original snapshot
descriptor, closed parent audit/process identities, tolerances, rules and caps.
The independent auditor takes the closed study, original process receipt and
its separate qualified-source freeze; it has no model-selection API.

**Wall, per-call and RSS limits are deliberately unset.** Establish them from
fabricated full-geometry throughput and peak memory for the slowest declared
initializer, with a documented allowance, then freeze them before admission.
Until these values and the adapters/auditor are qualified, registration and
empirical launch are blocked. Use one supervised attempt with preserved failures,
no retries, and no threshold or budget expansion after numerical access.

The [publication correction](fsm-author-publication-correction.md) documents
prior opaque copying of vendor 300 mV train and test example bytes. Do not claim
a universally pristine reserve, infer numerical training contamination from
copying alone, or omit the limitation. New packages must prune raw archives and
all vendor examples before enumeration or hashing. Official test stays closed.
Repeated periods/windows and matched excitations are correlated. Known future
inputs make this conditional forecasting on one plant. Passing would support
this fixed-weight amplitude contrast, not stability, autonomous control,
independent environments, biological novelty or new architectural novelty.
