# Two-observation control: implementation status

The completed [memory comparison](../../research/reacher-geometry-memory.md) passed its 25 continuation checks. This follow-up asks whether two recent valid observations and the intervening issued actions explain the persistent model's advantage. **The follow-up has not been trained or scored.**

The [prospective design](../reacher-geometry-memory-v1/history-control-design.md) was written before the preceding comparison's results were reviewed. It keeps the original GRU64 parameter schema, data, initial tensors, minibatch orders, loss and update count. The new model reconstructs its state from a bounded public history at every real observation. Every reconstruction pays for all twelve observation updates and eleven transitions, including padding.

## Reviewed components

| Component | Synthetic checks | Review |
| --- | ---: | --- |
| History model and causal buffers | 49 | [Model review](integration-review.md) |
| Paired, resumable training | 63 | [Training review](training-review.md) |
| Geometry-scored CEM controller | 24 | [Controller review](control-review.md) |
| Prospective scope and 25 checks | 60 | [Protocol review](protocol-review.md) |
| Independent public-history reconstruction | 76 | [History audit review](public-audit-review.md) |
| Complete paired-result arithmetic | 37 | [Results review](results-review.md) |
| Episode recording and partial failures | 23 | [Episode review](episode-review.md) |
| Fit serialization and partial updates | 33 | [Fitting review](fitting-review.md) |
| Explicit random streams and history separation | 48 | [Stream review](streams-review.md) |
| Saved training state and update accounting | 86 | [Training audit review](training-audit-review.md) |

These component checks include hand-built histories, tiny training fixtures and fake episode orchestration. They are not a whole-study rehearsal or an empirical performance result. Each adjacent validation JSON records the author's test report and the separate static review. The history auditor checks saved public evidence without recomputing learned hidden states.

## Actual training integration

A retained [tiny integration](training-integration-attempt-01/completed.json) completed all three fits, serialization and independent saved-training audits in 0.96 seconds. Each fit made four real optimizer updates. The fits intentionally reused the same synthetic seed410 tensors and orders; they are not independent training replicates. All saved files were hash-checked and all 90 frozen scientific sources remained unchanged. No native control or scientific evaluation occurred. [Driver](integrate_training.py) · [Process exit](training-integration-attempt-01/process.json).

## Proposed experiment

Train three new history models from the three authenticated original initializations. Retain the three persistent and three single-observation GRUs as paired references. Evaluate all nine models and five reference controllers on the same fresh 64 cases in full sensing, six-step gaps and ten-step gaps: 42 rows in total.

The proposed rule requires persistent memory to improve mean cost by at least 3% against both learned controls on both gap panels, lose no paired fit comparison, degrade full-sensing means by no more than 2%, and pass the declared zero-action competence checks. All 25 checks must pass. A failed superiority check does not prove equivalence or show that short history explains the earlier result.

Before scientific fitting, finish the [study integration](study-integration-map.md), authenticate the inherited inputs and all source dependencies, verify fresh random streams, and complete a retained rehearsal and capacity measurement. Freeze the executable protocol and measured limits separately from the current prospective definitions. The existing 90 frozen scientific sources remain unchanged.

This tests a stronger conventional control. It does not by itself test biological wiring, Bayesian inference, a new architecture or transfer to another environment.
