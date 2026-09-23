# Query-written memory study status

**Fresh collection is running; training and evaluation code pass qualification.**
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

The original collection worker and supervisor were verified live at 78 of 108
completed trajectories on September 23, 2026. This is a progress snapshot, not a
completed collection receipt. No collected arrays or checkpoints have been
decoded for fitting or evaluation.

Training remains unadmitted. It needs a successful original collection closure,
the separate fabricated training-capacity check and a published source-bound
training plan. No scientific retries or replacement seeds are allowed.

The period-eight comparison changes observations on fixed collected paths.
It does not establish autonomous search quality, teacher-call savings or an ICLR
contribution. Those claims require subsequent experiments if this screen passes.
