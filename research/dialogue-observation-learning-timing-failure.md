# Observation-learning attempt: timing failure

The first twelve-fit scientific attempt **failed technically and cannot support
a quality comparison**. Five fits completed; `frozen_numbers-6902` stopped after
1,032 of 1,280 optimizer updates. The remaining six fits did not start. No saved
predictions were decoded for quality, and no completed fit was selected.

## What happened

The frozen allocation was 28,800 seconds. On this Mac, Python's `monotonic` and
`perf_counter` use `mach_absolute_time`, which does not advance during system
sleep. The parent supervisor and worker budget used those clocks. System sleep
was observed in the machine's power log, including a wake at 20:30:46 PDT on
September 20. This confirms sleep occurred; it does not attribute the entire
clock difference to sleep or exclude civil-clock adjustments.

At termination, the receipts recorded:

| Clock or limit | Seconds |
| --- | ---: |
| Frozen allocation | 28,800.000 |
| Civil time between parent launch and terminal receipt | 39,211.373 |
| Parent monotonic elapsed time | 8,176.152 |
| Worker performance-counter elapsed time | 8,174.969 |

The frozen reporter and independent checker require civil elapsed time to be
nonnegative and no greater than parent elapsed time, which must stay within the
allocation. That condition could no longer pass. The prose's generic "wall"
label and the different clocks were inconsistent; we have not reinterpreted
the limit or changed the old acceptance check after seeing this failure.

The root agent checked the worker identity and sent `SIGALRM` to PID 14919.
The worker's existing timeout handler wrote `failed.json`. Session 90741 exited
1, and process group 14919 was independently confirmed absent. The parent's
`timed_out: false` is retained: its own watchdog did not expire. The root stop,
worker failure and parent terminal receipt describe different parts of the same
event, not an automatically enforced parent timeout.

## Preserved evidence

- [Failure manifest](../output/dialogue-observation-learning-v1/failed-scientific-publication-01/manifest.json)
  binds all 27 run files, totaling 253,766,070 bytes, and the process receipts.
  SHA-256: `41384aeab0d108992906f9f8d0ffbe341751bf7d061288aa98a7441c96f4476f`.
- [Worker failure](../output/dialogue-observation-learning-v1/scientific-run-01/failed.json)
  preserves completed and interrupted operation counts, including an encoder
  dispatch that had not returned.
- [Root stop request](../output/dialogue-observation-learning-v1/civil-time-stop-01.json),
  [signal acknowledgment](../output/dialogue-observation-learning-v1/civil-time-stop-01-sent.json)
  and [actual parent terminal](../output/dialogue-observation-learning-v1/scientific-process-01.terminal.json)
  preserve the intervention and exit state.
- All 53 frozen source hashes still match. The original plan, allocation,
  progress snapshots and five fit completion receipts remain unchanged.

Weights, individual predictions, evaluator labels and reversible tokens remain
local under the existing publication rule. Byte hashing is not quality scoring.
There is no root completion receipt and no scientific result or figure for this
attempt. It will not be resumed, extended or scored as a five-fit comparison.

## Prospective repair

A separate version must use one explicit suspend-inclusive elapsed clock for
the worker budget, parent deadline and final acceptance. Civil timestamps remain
provenance only; clock adjustments cannot determine success. Apple provides
[`mach_continuous_time`](https://developer.apple.com/documentation/kernel/1646199-mach_continuous_time)
for elapsed time that includes sleep; the contrast with
[`mach_absolute_time`](https://developer.apple.com/documentation/kernel/1462446-mach_absolute_time)
matters here.

Synthetic clock and process-lifecycle checks must pass before a new freeze.
Any new scientific attempt requires separate paths, a published correction,
fresh initialization of all twelve fits, the same scientific recipe and no
reuse of this attempt's checkpoints. A repaired clock can detect an expired
deadline after wake; it cannot execute training while the laptop is asleep.
