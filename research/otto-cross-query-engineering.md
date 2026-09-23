# Cross-query recurrence: implementation and cost qualification

**The four-model comparison is feasible under its planned training budget. No
new trajectory or task-performance result is available yet.** The proposed model
keeps recurrent state across planner queries and corrects that state using the
newly observed forecast error. Three ordinary GRU controls test whether the
explicit correction adds anything beyond memory and access to the same error.

The [prospective protocol](otto-cross-query-forecast-protocol.md) fixes fresh
cases, three fit seeds, complete histories, sampled training targets and all
53 continuation conditions. The [previous experiment](otto-sampled-forecast-results.md)
remains a negative result: recursive score feedback lost to an ordinary direct
GRU. This follow-up keeps a direct offset from the last actual query instead.

## What passed

The model, collector, chronological data, metrics, training runner and saved-output
auditor passed **157 fabricated tests**: 30, 22, 12, 73, 10 and 10 respectively.
These cover state isolation,
hidden-label exclusion, query replacement, chunk boundaries, complete episode
tails, sampled weights, independent-case counting and the continuation rule.
An initial collector test invocation failed because its module search path was
missing; that attempt is retained alongside the corrected environment run.

| Family | Parameters | Full synthetic episode-batch time |
|---|---:|---:|
| Error-corrected recurrent state | 5,978 | 0.262244 s |
| Ordinary GRU with explicit error | 5,996 | 0.287776 s |
| Persistent direct GRU | 5,862 | 0.218831 s |
| Reset-at-query direct GRU | 5,862 | 0.210288 s |

Each timing covers six complete 2,188-step synthetic histories, 69 chronological
chunks, 38 chunks with backward computation, 31 forward-only chunks, gradient
clipping and one Adam update. Parameters stay fixed until all chunks finish.
State crosses chunk boundaries; gradients stop every 32 steps. The timing also
includes operation guards and durable logging. Model and optimizer construction
are recorded separately. There is one observation per family, not repeated
latency benchmarking.

The four times sum to **0.979140 seconds**. The predeclared estimate
`1.5 * 3 * 720 * total + 120` is **3,292.41 seconds (54.87 minutes)**, below the
5,400-second admission threshold within a 7,200-second fitting allocation.
The multiplier supplies 50% headroom and the added 120 seconds covers planned
setup, final rescoring, evaluation and closure. This estimate is not a runtime
guarantee. First-update recurrent gradients were zero because the readout starts
at zero; the output heads changed as expected. No learning-quality claim follows.

The original synthetic worker and supervisor completed successfully. A separate
saved-file check passed 1,501 checks covering file hashes, all 729 logged calls,
complete histories, costs and the fixed admission calculation. It did not rerun
models. Its first invocation used the wrong terminal filename and is preserved
as a failed audit preparation attempt, with no synthetic work repeated.

## Evidence and next step

- [Model tests](../output/otto-cross-query-forecast-v1/qualification-model-01/receipt.json),
  [collector tests](../output/otto-cross-query-forecast-v1/qualification-collector-02/receipt.json),
  [data tests](../output/otto-cross-query-forecast-v1/qualification-data-01/receipt.json),
  [metric tests](../output/otto-cross-query-forecast-v1/qualification-metrics-01/receipt.json),
  [training tests](../output/otto-cross-query-forecast-v1/qualification-training-01/receipt.json),
  [audit tests](../output/otto-cross-query-forecast-v1/qualification-audit-02/receipt.json).
- [Frozen capacity plan](../output/otto-cross-query-forecast-v1/capacity-plan-01.json),
  [all four measurements](../output/otto-cross-query-forecast-v1/capacity-01/summary.json),
  [worker receipt](../output/otto-cross-query-forecast-v1/capacity-01/receipt.json),
  [original supervisor](../output/otto-cross-query-forecast-v1/capacity-supervisor-01.terminal.json),
  [saved-file check](../output/otto-cross-query-forecast-v1/capacity-audit-02/receipt.json).

The training runner and independent results auditor are now qualified. The next
step is collecting the fresh 54 TRAIN and 36 VALID paths. All twelve fits must close
before VALID is decoded. A forecast improvement would still need a separate
fresh autonomous-control comparison that includes all computation costs and
tests an unseen sensing regime. There is
no new architecture, biological-learning or recurrent-world-model advantage
established by these engineering checks.
