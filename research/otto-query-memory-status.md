# Query-written memory study status

**All 108 trajectories are collected; training and evaluation code pass qualification.**
No new learned-model performance result exists. The earlier protected-readout
FAIL 15/29 is unchanged.

The new [protocol](otto-query-memory-protocol.md) compares query-written memory
with ordinary recurrent training, instantaneous memory and simple last-error
correction. It uses 54 TRAIN, 18 DEV and 36 untouched TEST collector paths.
All model constants, fit seeds, training schedules and continuation conditions
are fixed before collection. The TEST phase requires successful DEV production,
an independently passed gate and the DEV auditor's original successful closure.

The composed model, full-census projector/loss, schedule-aware metrics and new
collector pass **311 fabricated tests and lint**. The first attempt's tests all
passed; its three mechanical lint findings were corrected and the complete
qualification rerun. Sources and logs remain in the
[qualification record](../output/otto-query-memory-study-engineering-v1/attempt-02/receipt.json).
Independent source reviews covered collection isolation, data leakage, complete
episode denominators, metrics and phase separation. Passing these checks is not
evidence of an architecture advantage.

The [collection plan](../output/otto-query-memory-v1/collection-plan-01.json) has
SHA256 `624b7fcebcdaa4df9d3b2079127ec40b9a0e3e043864034fd7869296cc1a2645`.
It binds source/runtime/teacher identities, the
[scoped seed reservation](../output/otto-query-memory-v1/seed-reservation-02.json),
and qualification before any native call. The metadata-only planning attempt
with the wrong interpreter was rejected before collection; the required native
interpreter produced the plan. Both attempts are retained.

The full training/evaluation suite now passes **528 fabricated tests and lint**.
It verifies complete-episode updates, all eight evaluation views, independent
metrics, schedule-derived computation counts and the boundary before DEV access.
Both qualification attempts are preserved in the
[training engineering record](otto-query-memory-training-engineering.md).

The [original collection receipt](../output/otto-query-memory-v1/collection-01/receipt.json)
records all 108 trajectories: 54 TRAIN, 18 DEV and 36 TEST. Its
[original supervisor](../output/otto-query-memory-v1/collection-native-01.terminal.json)
completed with exit code zero, no timeout, a reaped worker and an absent worker
process group. Collection took 2,154.57 seconds including supervisor cleanup.
All 17 payload hashes were authenticated before the capacity planner used only
TRAIN episode lengths. No collected arrays or checkpoints have been decoded for
fitting or evaluation. Raw collection payloads remain local pending an evidence
release; the receipts identify their exact bytes.

Training remains unadmitted. The
[capacity plan](../output/otto-query-memory-v1/capacity-plan-01.json) is frozen
before its synthetic timing run. Training still requires that check to pass and
a separate published source-bound training plan. No scientific retries or
replacement seeds are allowed.

The period-eight comparison changes observations on fixed collected paths.
It does not establish autonomous search quality, teacher-call savings or an ICLR
contribution. Those claims require subsequent experiments if this screen passes.

[Related work and limits on novelty claims](otto-query-memory-related-work.md)
compare this mechanism with fast-weight delta updates, Gated Delta Networks and
Titans. The registered experiment remains unchanged.
