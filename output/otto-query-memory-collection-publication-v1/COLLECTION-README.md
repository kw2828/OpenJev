# OpenJev query-memory collection evidence

This archive preserves **108 fixed collector paths**, split into 54 TRAIN,
18 DEV and 36 TEST paths, with a complete teacher-score census. Collection took
zero optimizer steps. It is a collection evidence release, not a learned-model
result or permission to advance to TEST evaluation.

All 17 original collection payloads are included byte-for-byte, together with
the original plan, receipt, process closure, direct admission inputs and the
132 source records pinned by that plan. `MANIFEST.json` records member hashes
and sizes. `SHA256SUMS` is published alongside the archive for download checks.
Packaging reads the arrays and trajectory journals only as opaque bytes; it
does not decode numerical arrays, parse trajectories or run models.

Files use repository-relative paths. The original data are under
`output/otto-query-memory-v1/collection-01/`. Historical receipts retain their
original absolute paths as provenance. A relocated rerun needs its own explicit
plan and fresh output paths; editing these original receipts would break their
identity.

This archive is **not a self-contained native reproduction**. `DEPENDENCIES.json`
identifies the original runtime and eight external native inputs, including
teacher weights, extracted tensors and observation kernels, with exact hashes.
Their bytes and the installed runtime are not bundled. Upstream retrieval and
attribution are described in `research/otto-learned-reference-and-symmetry.md`.

The environment and reference teacher come from OTTO, not from a new OpenJev
architecture. Root `LICENSE` and both OTTO MIT notices under `third_party/otto/`
are included. Preserve those notices when redistributing the associated source.

The running study compares teacher-score prediction on these fixed paths.
Its frozen DEV continuation rule must pass independent audit and original
process closure before TEST is evaluated. Data preservation alone establishes
no accuracy, calibration, autonomous utility, teacher-call saving or architecture
novelty. The full-census teacher computation remains a paid cost of the study.
