# Observation learning: prospective clock correction

This is a separate technical correction and fresh twelve-fit attempt. It is not
a continuation or successful reinterpretation of the
[failed first attempt](dialogue-observation-learning-timing-failure.md).
No scientific execution or result is established by this document.

## Scientific recipe is unchanged

The [original scientific protocol](dialogue-observation-learning-protocol.md)
and [scoring contract](dialogue-observation-learning-scoring.md) retain every
cohort, public observation, target, model, optimizer, seed, epoch, batch order,
final-only evaluation, metric and seven-condition continuation rule. The only
superseded provisions concern implementation version, source closure and timing
semantics, as specified below. Official TEST remains excluded; DEV remains
exposed development evidence.

Run all twelve fits from fresh initialization in the same order: seeds 6901,
6902, 6903, with frozen_original, frozen_numbers, trainable_original and
trainable_numbers within each seed. Use the same prepared data and pretrained
MiniLM checkpoint, scalar-memory architecture, 20 epochs, effective batch 32,
learning rates, loss weights and three paired epoch orders. There is no new
architecture, data selection, hyperparameter search or additional neural work.

Do not read, score or reuse the failed attempt's predictions, labels copied into
its evaluator ledger, or trained weights. Reuse only the independently qualified
preparation and cost-pilot metadata, along with the original untrained assets.
The new allocation binds the failed attempt's publication manifest
`41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f`.
That manifest preserves 27 files and the five complete fits plus the partial
sixth. Its metadata and byte hashes can be checked without decoding predictions.

Previously spent work remains separately attributed: 201,700 completed-fit
training dialogue visits and 6,400 optimizer updates, plus 32,545 completed
dialogue visits and 1,032 updates in the interrupted fit. Its failed phase
recorded 39,211.373 civil seconds and 8,176.152 parent monotonic seconds; neither
is included in the new attempt's allocation or misreported as successful work.

## One suspend-inclusive deadline

On macOS use `mach_continuous_time`, converted through checked
`mach_timebase_info`; Linux's supported equivalent is `CLOCK_BOOTTIME`.
Unavailable, invalid, regressing or failed clock reads fail closed. Do not fall
back to `mach_absolute_time`, `CLOCK_MONOTONIC`, `perf_counter` or civil time for
the deadline. The qualified scientific device remains MPS on this Mac; Linux
clock support does not authorize a scientific device fallback.

The parent captures its start before spawning the worker and sets an absolute
deadline at start plus the allocation. The worker authenticates the exclusive
parent launch receipt, exact command, process identities, clock backend, plan,
output path and cap before any model operation. It inherits that absolute
deadline. Worker initialization and launch-handshake time consume the parent
budget; neither fit boundaries nor a new process reset it.

Every worker budget check uses this clock. The supervisor uses short polling
waits and checks the deadline after a wake, after child exit and after cleanup.
The clock cannot cause a sleeping laptop to execute code. If the deadline has
expired by the time it wakes, the attempt fails, including an apparent exit 0.
Equality with the deadline is expired. A one-shot awake-time signal, if used,
is a secondary failure mechanism and cannot authorize success.

Both parent and worker record clock backend, integer start, finish, deadline
and elapsed nanoseconds. Root `wall_seconds` is elapsed nanoseconds divided by
one billion. The worker start and finish must lie inside the parent interval;
both share the same absolute deadline. The parent's final measurement, taken
after child exit and process-group cleanup, must be strictly before that
deadline, with child exit 0, no recorded error or timeout, and no surviving
process group. Terminal JSON publication follows that measurement and is
excluded from this measured interval. The supervisor itself must then exit
successfully. Missing or inconsistent timing evidence rejects the run before
quality scoring.

Civil start/end timestamps are provenance only. They must be finite valid
timestamps, but wall-clock adjustments in either direction cannot cause a
pass or failure. They are not compared with elapsed-clock durations.
Per-update, fit, evaluation and checkpoint timing retains the original
performance-counter instrumentation and explicitly means awake execution time.
These nested measurements are not added to whole-attempt elapsed time and do
not support isolated hardware latency claims when other work overlaps.

Failure cleanup must still work when the native clock cannot be read. Preserve
the original error; use bounded termination escalation and record whether the
process group disappeared. Mark unavailable timing as unavailable, never zero
or a fallback estimate. Cleanup grace does not extend successful execution.
If a worker completion was written but its final check fails, retain it as
invalid evidence alongside the failure receipt. A failed parent terminal also
prevents acceptance of any worker completion.

## Qualification, freeze and allocation

Before training, qualify injected-clock suspend jumps, civil jumps both ways,
deadline equality, late exit 0, missing terminal evidence, backend mismatch,
clock failure, cleanup, and late completion invalidation. Include a bounded
native-clock/no-op process smoke. Do not induce system sleep or alter global
power settings. Synthetic tests and native clock access do not establish real
sleep/wake integration or model efficacy.

The new source closure contains the unchanged original 53 sources plus eleven
new files: V2 worker and tests, V2 reporter and tests, V2 supervisor and tests,
suspend-clock helper and tests, this protocol, and the V2 independent saved-only
auditor and its tests. Freeze all 64 hashes and source snapshots, the complete
runtime, unchanged prepared orders/evaluator identities, allocation and failure
lineage before execution. No source may change during the attempt.

The new allocation retains the original limits: 28,800 suspend-inclusive
seconds, 8 GiB process RSS, 8 GiB sampled MPS driver memory and 2 GiB output.
Metadata freeze has a separate 300-second, 8 GiB RSS and 512 MiB output cap.
The prior 7.02-hour estimate remains a scheduling heuristic, not a guarantee.
All actual work, including this correction's qualification, is separately
recorded. No old training or prediction artifact is a cache for this attempt.

Use new exclusive allocation, freeze, run and supervisor paths under
`output/dialogue-observation-learning-v2`. Coordinate shared compute before
launch. Any resource, state, provenance or timing failure stops the whole
attempt. There is no automatic retry, resume, seed replacement, trimmed cohort,
quality-based selection or budget extension. A scientific comparison requires
all twelve fits and successful production and independent saved-only audits.

On this Mac, launch the supervisor through `/usr/bin/caffeinate -i` and record
that outer command. This creates an idle-sleep assertion only for the utility's
lifetime and releases it on exit. It does not change global power settings,
keep the display on, or guarantee against lid/manual sleep. Qualify the same
wrapper with a no-op child first. Suspend-inclusive acceptance remains mandatory
even with this assertion.

Publication rules remain unchanged: publish source, aggregate results and
receipts; keep reversible data, labels, individual predictions and weights
local unless separately designated for publication. The source dialogue data
retain CC BY-SA 4.0 terms. A future pass would support only this exposed-data
observation-learning comparison, not biological wiring or architectural novelty.

Primary clock references: [Apple clock declarations](https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/mach_time.h),
[Linux clock documentation](https://man7.org/linux/man-pages/man2/clock_gettime.2.html).
