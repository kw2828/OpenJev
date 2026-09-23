# Query-written memory study status

**Training is running on the completed 108-trajectory collection.**
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
TRAIN episode lengths. TRAIN arrays are now decoded inside the admitted worker;
DEV access is guarded by the completed checkpoint/TRAIN-prediction barrier,
and this producer cannot decode TEST. All 17 raw collection payloads are now in
the [collection evidence release](https://github.com/kw2828/OpenJev/releases/tag/otto-query-memory-collection-v1),
with their original receipts, 132 frozen source records and a dependency map.
Packaging used opaque bytes only. GitHub's four asset digests and sizes match
the local files; the [verification record](../output/otto-query-memory-collection-publication-v1/release-verification-01.json)
preserves those checks. Native teacher weights, kernels and the installed
runtime remain external, so this is not a self-contained native reproduction.

The [capacity check](../output/otto-query-memory-v1/capacity-01/summary.json)
completed all five fabricated updates and passed its rule. The projected time
is 14,316.69 seconds, below the 16,200-second admission threshold. The genuine
supervisor completed in 2.78 seconds with exit code zero and an absent worker
group. This estimate is not a runtime guarantee or an effectiveness result.

The [training plan](../output/otto-query-memory-v1/training-plan-01.json) is
frozen before fitting. It binds both completed phases, all qualified sources,
18 fits across three seeds and the unchanged 21,600-second hard limit. The
[original training worker](../output/otto-query-memory-v1/training-native-01.launch.json)
is now running. It completed its first three pretraining epochs at the first
live check; both worker and supervisor were present. Launch and progress do not
establish completion or model effectiveness. No scientific retries, replacement
seeds, shortened schedules or budget extensions are allowed.

The period-eight comparison changes observations on fixed collected paths.
It does not establish autonomous search quality, teacher-call savings or an ICLR
contribution. Those claims require subsequent experiments if this screen passes.

[Related work and limits on novelty claims](otto-query-memory-related-work.md)
compare this mechanism with fast-weight delta updates, Gated Delta Networks and
Titans. The registered experiment remains unchanged.

The [conditional TEST evaluator and independent audit](otto-query-memory-test-engineering.md)
now pass 105 new fabricated tests and lint, with independent source review.
They preserve all 48 P4/P8 views and require the completed, independently passed
DEV gate before any TEST decode. These checks leave the live training sources
unchanged and do not admit TEST execution or establish effectiveness.

The [saved-audit renderer](../scripts/render_otto_query_memory.py) passes
16 fabricated tests and its scoped lint check. Its synthetic eight-panel chart
was visually inspected. It retains every method, seed and failed condition;
empirical rendering still requires the original independent audit and process
closures. The [qualification record](../output/otto-query-memory-render-engineering-v1/renderer-qualified.json)
retains the initial combined lint attempt, including the separate archive
packager's style finding. No empirical result chart has been generated.
