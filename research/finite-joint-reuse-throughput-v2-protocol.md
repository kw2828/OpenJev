# Integrated throughput: corrected qualification assertion

This is a new registration after the original throughput qualification stopped
with 77 passing tests and one failed test. Its original native process closed
after 3.106916791 seconds. No timed benchmark or saved-output audit started.
Preserve all 88 original source files, their snapshots, registration, logs and
failure receipts unchanged. Do not rerun the original registration.

The failure is confined to a test selector: checking whether `round` occurs in
a condition name also selects `rounded_median_gt1.05`. At a supplied ratio of
exactly 1.05, the production gate correctly passes the nine per-pair conditions
and fails the three strict median conditions. The test wrongly mixed these
groups. The replacement test selects names ending in `_ratio_gt1` and asserts
there are exactly nine. No production gate or training behavior is changed.

All experimental requirements in the [original protocol](finite-joint-reuse-throughput-protocol.md)
remain in force: one namespace-943201 dataset, identical seed 943301, all three
models, two implementations, 32 prefix and 64 joint updates, one warmup pair
and three measured pairs per arm, the same ordering, exact initial/prefix
states and symmetric final tolerance 1e-7. Retain all 24 fits. Every measured
ratio must exceed 1 and all three median ratios must exceed 1.05. Keep all
original phase, producer, fit, memory and output bounds.

Use a separately named auditor solely to bind the new registration location;
its numerical and evidence checks remain unchanged. Use a separately named
test file with the selector correction and a new orchestrator that binds the
complete failed prerequisite before executing any phase. The new source
closure contains the original 88 sources plus these four new files. Freeze
this closure once, then qualify, run and audit under separate native processes.
Any failure stops this registration with its original evidence retained.

This is the first opportunity to run the timed workload, not a replacement of
unfavorable timing measurements. It provides no task-effectiveness result or
scientific admission. The earlier equal-update feasibility failure also stays
closed with its original counts and threshold.
