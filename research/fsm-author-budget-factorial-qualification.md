# State-estimation budget: arithmetic qualified

The separate 16/64-step state estimator and its independent replay passed
**71 fabricated tests plus lint**. This prepares the [four-condition comparison](fsm-author-budget-factorial-draft.md)
of training budget and inference budget. No new DEV evaluation or performance
result is available. The full evaluator and its process/audit admission still
need implementation and qualification.

| Original qualification | Tests | Evidence |
| --- | ---: | --- |
| Production state estimator | 29 | [Receipt](../output/fsm-author-budget-factorial-engineering-v1/producer-core-qualification-01/receipt.json) |
| Independent numerical replay | 31 | [Receipt](../output/fsm-author-budget-factorial-engineering-v1/independent-math-qualification-01/receipt.json) |
| Direct comparison between implementations | 11 | [Receipt](../output/fsm-author-budget-factorial-engineering-v1/cross-parity-qualification-01/receipt.json) |

The original 16-step implementation is unchanged. Its new counterpart accepts
only 16 or 64 outer directions and reproduces the original arithmetic at 16.
The independent counterpart separately reproduces its own held predecessor.
Fabricated capped solves check the first-16 path, retained state boundary and
actual work counts; early stops, rejected trials, causal inputs and failure
handling are also covered.

Six direct comparisons use physical inputs with nonzero means, unequal scales,
nonlinear feedback, and both short and C100/H128 requests, including a
rank-deficient 28-state model. They compare forecasts and retained states at
`atol=rtol=1e-8`, complete context diagnostics at `atol=1e-10, rtol=1e-8`, and
branch choices/work counts exactly. These fixtures validate implementation
agreement, not the usefulness of 64 directions on the measured benchmark.

The larger-budget training run is governed by its [separate protocol](fsm-author-nllfr-budget-protocol.md).
The four-condition evaluation must wait for that original process and its
independent FIT audit to close, then freeze exact checkpoint identities. It
will retain both checkpoints and both inference budgets, including failures or
worse forecasts. The old capped fit remains an incomplete diagnostic.

[Source review and qualification closure](../output/fsm-author-budget-factorial-engineering-v1/qualification-and-source-review.json)
bind the source snapshots, original commands and logs. No training source,
old numerical artifact, candidate weight or reserved measurement was changed
by this preparation.
