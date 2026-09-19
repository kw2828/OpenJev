# Reacher cache engineering rehearsal

The first retained rehearsal completed **15 fits and 60 controller rows**. Its saved-output audit replayed **3,300 native transitions with zero discrepancy**, plus 147 supplied-physics observer transitions. This validates the reduced fixture's integration, not model effectiveness.

The [fixture](fixture-description.json) used four engineering training episodes, one epoch and two updates per fit, GRU width 4, MLP width 7, two prediction episodes and one control case per row. All five architectures and three initialization pairs were retained. All 15 actual classes were restored before fresh evaluation. The only substituted audit boundaries were historical lineage and the inherited-training corpus identity; all four fixture training episodes were still replayed.

Execution took 28.743451 seconds, saved-output audit validation 3.011025 seconds, and the [launcher](supervision-completed.json) recorded 32.659091 seconds overall. These nested fixture timings are not isolated latency or full-study estimates. See the [coverage and accounting](summary.json).

[The archive manifest](bundle-manifest.json) binds every original file in `attempt-01`, the closed launcher and terminal files, all 70 plan-bound source snapshots and the [fixture helper](source-snapshot/tests/reacher_cache_fixture.py). Every uncompressed archive member was reopened and checked against its path, length and SHA-256. The [receipt](receipt.json) records the archive identity and unchanged input checks. The archive remains local at `output/reacher-cache-rehearsal-v1/publication-v1/reacher-cache-rehearsal-v1.tar.gz`; no upload is claimed.

Raw fixture gate and utility outputs remain losslessly preserved inside the archive. They are **engineering outputs, not scientific results**. Packaging made no model, training or native calls and did not rerun tests. External historical dependencies are not recursively included, so this is a verification bundle rather than a standalone production-lineage reproduction. Original source licenses remain applicable; packaging adds no separate data or third-party license claim.
