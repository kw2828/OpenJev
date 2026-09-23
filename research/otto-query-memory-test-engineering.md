# Conditional query-memory TEST evaluation

The new TEST evaluator and independent saved-output auditor pass **105 fabricated
tests and lint**. This is implementation qualification, not a scientific TEST
result. The original training worker is still running; TEST remains unopened.
The [registered protocol](otto-query-memory-protocol.md) is unchanged.

## Admission and evaluation

The evaluator requires the completed TRAIN/DEV producer, its successful original
supervisor, all 13 independently audited DEV conditions, and the DEV auditor's
successful original supervisor. It authenticates their sources, inputs and
saved payloads before numerical imports. A failed DEV condition leaves TEST
unused. No TEST plan or scientific execution is admitted by these engineering
checks alone.

If admitted, the evaluator uses all 18 frozen final checkpoints and produces
48 views: eight methods, three fit seeds and both period-four and period-eight
observations. Only genuinely scheduled teacher answers enter model inputs.
It preserves exact query outputs, all parameter witnesses, the no-write control,
complete trajectory support and charged work. It performs no optimization,
teacher calls, simulator calls or TRAIN/DEV rescoring. Its original hard limit
is 1,800 seconds, 4 GiB peak RSS and 2 GiB output.

The independent auditor reconstructs six metric scopes, all 25 TEST conditions
and work counts directly from saved outputs. The common nonquery scopes allow
P4/P8 comparison on the same steps; unlike the actual-period scopes, they do
not contain age breakdowns. The auditor checks all 48 predictions and the 18
source checkpoints without executing a model. Its original hard limit remains
600 seconds, 2 GiB peak RSS and 256 MiB output. A worker result is conditional
on its own original supervisor closing successfully.

## Preserved qualification

The [bounded runner](../output/otto-query-memory-test-engineering-v1/run_engineering.py)
binds 44 files: 39 unchanged files from the prior successful training
qualification, four new evaluator/auditor files, and itself. The new tests cover
failed admission, mismatched checkpoints, unseen-score isolation, canonical
state and padding, complete P4/P8 support, threshold failures, operation counts,
and preservation of failures. No empirical arrays or trained checkpoints were
decoded.

[Attempt 01](../output/otto-query-memory-test-engineering-v1/attempt-01/receipt.json)
passed: 105 tests in 2.70 seconds, 3.22 seconds including command startup, and
lint in 0.08 seconds. Both child processes exited with code zero, were reaped
and left no process group. Before/after source hashes match. The receipt SHA256
is `196d69becaacf5ad028f486389db6f72e0ea5b05f49a8cc567f4d09646cd94d8`.

The fabricated scalar-audit probe measures six maximum-length paths under each
observation period. Twice the slower per-row rate extrapolated to all 3,780,864
possible rows is 37.37 seconds; adding the fixed 120-second reserve gives
157.37 seconds. This passes the capacity rule, but is a heuristic rather than
a timing guarantee. The 600-second scientific audit limit is unchanged.

Independent source reviews found no actionable issues in admission, evaluation,
metric arithmetic, checkpoint parity or work accounting. The
[review record](../output/otto-query-memory-v1/test-engineering-review-01.json)
identifies the reviewed files and limits. Numerical inference and gradients are
not independently replayed by the saved-output auditor; its documentation
retains that limitation.

The underlying comparison is still teacher-score imitation on fixed collector
paths. Qualification does not establish autonomous utility, calibrated outcome
probabilities, teacher savings or an architecture contribution.
