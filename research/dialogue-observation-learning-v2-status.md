# Observation learning: clock-corrected comparison

**The corrected twelve-fit experiment is now training from fresh initialization.** It asks
whether learning the text encoder helps autonomous state tracking beyond a
simple number-word correction. The four variants and three paired seeds are
unchanged. There is no accuracy result or new-architecture claim.

The [first attempt](dialogue-observation-learning-timing-failure.md) remains
failed and unscored. Its five fitted models are not reused. The
[prospective correction](dialogue-observation-learning-protocol-v2.md) replaces
the timing and supervision machinery while preserving the complete scientific
recipe, data, scoring functions and seven continuation conditions.

## Qualification and freeze

The clock, supervisor, worker, reporter and independent auditor have passing
evidence for **154 test cases**. Initial lint and synthetic-fixture failures
remain in their receipts with the targeted corrections. A real no-op process
also passed under the temporary idle-sleep assertion. No real sleep was induced.
[Qualification records](../output/dialogue-observation-learning-v2/README.md).

Source review binds **64 files**, including all unchanged 53 original sources.
Twenty-one training, data and scoring functions have identical ASTs to V1.
Metadata freeze completed in **1.881 seconds**, making zero model calls; the
supervisor recorded **2.001 seconds**, exit 0 and no surviving process group.
Independent byte/metadata auditing passed all source, payload, timing and
scientific-lineage checks. Reversible data and duplicate source snapshots remain
local; manifests bind the complete 69-payload closure.

- [Allocation](../output/dialogue-observation-learning-v2/scientific-allocation-01.json):
  `4a5cd6c226069b8e31aeb47a03363e331f7a09a8fa1d07663d398975d48eaa9a`.
- [Frozen plan](../output/dialogue-observation-learning-v2/scientific-freeze-01/plan.json):
  `295b4f6857b4f9d625ed1133e0cf0785db40b53f4537c235cdb609f168c98885`.
- [Freeze completion](../output/dialogue-observation-learning-v2/scientific-freeze-01/completed.json):
  `a3fab5f79dd8724ae31fa9400794e4a5ff305f4d0e6795e7bda1129b149e63c6`.
- [Independent freeze audit](../output/dialogue-observation-learning-v2/scientific-freeze-review-01/result-01/receipt.json):
  `88f041f05286aba5591961d9f3f0c39221a34f9ead7ddb5d98e9bf9b32dc64af`.
- [Actual freeze process check](../output/dialogue-observation-learning-v2/freeze-terminal-review-01.json):
  `c937dc33943161f6dd7fdb6efa9a347e841b8cf3eaab2b7a5d51e8e26e983516`.

## Live execution

The source and corrected plan were published in `6a93965` and `faf345d` before
launch. Session **61998**, worker process group **41434**, is running. **Five
of twelve fits have completed** all 1,280 updates and their final DEV pass:
all four variants for seed 6901 and `frozen_original-6902`. Their work counts,
state checks and payload hashes match the frozen plan. All three frozen
encoders are unchanged; both
trainable encoders' fingerprints have changed. This confirms execution, not
useful learning. The sixth fit, `frozen_numbers-6902`, is training.
All 64 source hashes still match.
The worker has authenticated and
inherited the parent's absolute native-clock deadline. This is execution
progress only; predictions have not been decoded for task-quality scoring.

- [Actual supervisor launch](../output/dialogue-observation-learning-v2/scientific-process-01.launch.json):
  `eefba879336be7ebb3f211174091014f1405741872a80f9934fb461c81c4ada9`.
- [Worker start](../output/dialogue-observation-learning-v2/scientific-run-01/started.json):
  `7caf0c297f1966d08bc90b8e2b0891155c9f1fa35159a7ba4c008e4729f5ebdf`.
- [First live snapshot](../output/dialogue-observation-learning-v2/scientific-progress-01.json)
  distinguishes observed execution from completed fits or scientific validity.
- [First completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-02.json)
  binds the [fit receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_original-6901/completed.json)
  and verifies execution metadata without reading predictions, weights or labels.
- [Second completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-03.json)
  binds both completed fits, including the [number-corrected frozen-encoder receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_numbers-6901/completed.json),
  and records the start of training with a trainable encoder.
- [Third completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-04.json)
  binds all three completed fits and the [first trainable-encoder receipt](../output/dialogue-observation-learning-v2/scientific-run-01/trainable_original-6901/completed.json).
  The independent metadata review also verified its counts, state checks, raw
  payload hashes and encoder change without decoding predictions or weights.
- [First complete-seed snapshot](../output/dialogue-observation-learning-v2/scientific-progress-05.json)
  binds all four completed variants, including the [number-corrected trainable-encoder receipt](../output/dialogue-observation-learning-v2/scientific-run-01/trainable_numbers-6901/completed.json),
  and records the second seed's start. One completed seed is not the required
  three-seed scientific comparison; its quality has not been scored.
- [Fifth completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-06.json)
  binds all five completed fits and the [second-seed frozen-original receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_original-6902/completed.json).
  Independent metadata review passed all 15 execution checks, including the
  unchanged frozen encoder and all live and frozen source hashes. The snapshot
  records the sixth fit's start without opening task-quality metrics.
- [Compute context](../output/dialogue-observation-learning-v2/compute-prelaunch-01.json)
  records released prior worker groups and background demo services. These are
  not isolated hardware latency measurements.

The full execution has an eight-hour suspend-inclusive deadline. A temporary
idle-sleep assertion accompanies the process; manual/lid sleep can still cause
a timeout. All twelve fits must complete and pass both saved-result audits
before quality is reported. Partial completion is not a scientific comparison.
Official DEV remains exposed development evidence; official TEST is untouched.

The [V2 figure generator](../output/dialogue-observation-learning-v2/figure-01/README.md)
is ready, with seven input/lifecycle rejection checks and a visually checked
synthetic render. It requires both completed audits, shows every fit and all
seven conditions, and preserves scientific failures. Invented fixture plots
remain local and are not benchmark results.

macOS also [confirmed the active idle-sleep assertion](../output/dialogue-observation-learning-v2/runtime-idle-assertion-01.json).
The helper is a child of the supervisor on this host; its release will be
checked after the actual supervisor exit. This observation does not guarantee
against manual or lid sleep.
