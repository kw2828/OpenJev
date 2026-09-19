# Two-observation control: implementation status

The completed [memory comparison](../../research/reacher-geometry-memory.md) passed its 25 continuation checks. This follow-up asks whether two recent valid observations and the intervening issued actions explain the persistent model's advantage. **The scientific follow-up has not been trained or scored.**

The [prospective design](../reacher-geometry-memory-v1/history-control-design.md) was written before the preceding comparison's results were reviewed. It keeps the original GRU64 parameter schema, data, initial tensors, minibatch orders, loss and update count. The new model reconstructs its state from a bounded public history at every real observation. Every reconstruction pays for all twelve observation updates and eleven transitions, including padding.

## Reviewed components

| Component | Synthetic checks | Review |
| --- | ---: | --- |
| History model and causal buffers | 49 | [Model review](integration-review.md) |
| Paired, resumable training | 63 | [Training review](training-review.md) |
| Geometry-scored CEM controller | 24 | [Controller review](control-review.md) |
| Prospective scope and 25 checks | 64 | [Protocol review](protocol-review.md), [engineering metadata correction](protocol-engineering-width-validation.json) |
| Independent public-history reconstruction | 76 | [History audit review](public-audit-review.md) |
| Complete paired-result arithmetic | 37 | [Results review](results-review.md) |
| Episode recording and partial failures | 23 | [Episode review](episode-review.md) |
| Fit serialization and partial updates | 33 | [Fitting review](fitting-review.md) |
| Explicit random streams and history separation | 48 | [Stream review](streams-review.md) |
| Saved training state and update accounting | 86 | [Training audit review](training-audit-review.md) |
| Complete execution and terminal failures | 45 | [Runner review](runner-review.md) |
| Preparation, full inheritance and rehearsal setup | 43 | [Preparation and fixture review](experiment-fixture-review.md) |
| Complete saved-study audit and failures | 37 | [Enclosing audit review](enclosing-auditor-review.md) |

These component checks include hand-built histories, tiny training fixtures and fake episode orchestration. They are not a whole-study rehearsal or an empirical performance result. Each adjacent validation JSON records the author's test report and the separate static review. The history auditor checks saved public evidence without recomputing learned hidden states.

## Actual training integration

A retained [tiny integration](training-integration-attempt-01/completed.json) completed all three fits, serialization and independent saved-training audits in 0.96 seconds. Each fit made four real optimizer updates. The fits intentionally reused the same synthetic seed410 tensors and orders; they are not independent training replicates. All saved files were hash-checked and all 90 frozen scientific sources remained unchanged. No native control or scientific evaluation occurred. [Driver](integrate_training.py) · [Process exit](training-integration-attempt-01/process.json).

## Complete engineering rehearsal

The [retained rehearsal](../reacher-two-observation-rehearsal-v1/attempt-01/review.md) completed preparation, three tiny fits, all 42 control rows and the independent saved-output audit in 113.98 seconds. All three subprocesses exited successfully. The audit replayed 1,232,886 simulator transitions plus 4,851 observer transitions with zero error. All 8,119 execution payloads and 116 source files passed hash checks. The [qualification receipt](../reacher-two-observation-rehearsal-v1/attempt-01/qualification.json) binds the complete locally retained artifacts and actual process exits.

This used four-dimensional models, one case per panel and two updates per new fit. Its diagnostic gate passed 14 of 25 checks and failed overall; that result does not test scientific superiority. Capacity qualification and a separate scientific freeze remain pending. The [as-run driver](run_rehearsal.py) is preserved unchanged.

## Full-size capacity measurement

The [second capacity attempt](capacity-launch-attempt02.json) completed 24 real updates at the full model and batch size, nine learned-controller rows and five references in 719.44 seconds. Its original saved-output audit stopped after 11.89 seconds on a mismatch between two work-count field names. The [failure review](../reacher-two-observation-capacity-v1/attempt-02-audit-failure-review.json) preserves both process exits and verifies all 2,888 measurement payloads, totaling 6.49 GB.

The separate [saved-only audit repair](../reacher-two-observation-capacity-v1/audit-repair-root-verification.json) completed in 456.28 seconds. All 14 rows passed, including 44,800 executed, 26,247,168 candidate and 9,600 selected simulator transitions with zero replay error. No model inference or training was repeated. Original measurements, source versions and failed audit remain retained. Independent sizing and the scientific freeze are separate next steps. Large engineering payloads are retained locally; these repository receipts bind their membership and hashes.

The [first attempt](../reacher-two-observation-capacity-v1/attempt-01-failure-review.json) stopped on an incorrect import before any numerical work; its failure is preserved. The import repair passed 36 focused checks, including real resolution of every delayed import. Neither capacity attempt is an efficacy experiment.

## Proposed experiment

Train three new history models from the three authenticated original initializations. Retain the three persistent and three single-observation GRUs as paired references. Evaluate all nine models and five reference controllers on the same fresh 64 cases in full sensing, six-step gaps and ten-step gaps: 42 rows in total.

The proposed rule requires persistent memory to improve mean cost by at least 3% against both learned controls on both gap panels, lose no paired fit comparison, degrade full-sensing means by no more than 2%, and pass the declared zero-action competence checks. All 25 checks must pass. A failed superiority check does not prove equivalence or show that short history explains the earlier result.

The [study integration](study-integration-map.md) and retained rehearsal are complete. Before scientific fitting, measure full-size capacity, then freeze the executable protocol, authenticated inputs, fresh random streams and measured limits separately from the current prospective definitions. The existing 90 frozen scientific sources remain unchanged.

This tests a stronger conventional control. It does not by itself test biological wiring, Bayesian inference, a new architecture or transfer to another environment.
