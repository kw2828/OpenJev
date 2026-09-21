# Observation learning: clock-corrected comparison

**All twelve fresh fits completed. The scientific continuation rule failed 6/7.**
Training the text encoder improves unseen-service macro accuracy from 76.46%
to 79.34% beyond a simple number-word correction, but log loss worsens from
0.705 to 0.827. The independent arithmetic audit agrees. The four variants,
three paired seeds and ordinary scalar memory are unchanged.
[Complete results and figures](dialogue-observation-learning-v2-results.md).

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

## Completed execution and audits

The source and corrected plan were published in `6a93965` and `faf345d` before
launch. Session **61998** exited 0 after **17,063.176 seconds** of
suspend-inclusive parent elapsed time, within its 28,800-second cap. All twelve
fits completed 1,280 updates and a final DEV pass each. Aggregate work is
15,360 optimizer updates and 747,948 evaluated endpoints across the twelve fits.
All 64 live and frozen sources match. The worker, supervisor, process group
and temporary idle-sleep helper are absent.

All six frozen encoders are unchanged; all six trainable encoder fingerprints
changed. Independent metadata checks passed for every fit. These establish
execution; the separate saved-result audits establish the reported arithmetic.

- [Whole-study completion](../output/dialogue-observation-learning-v2/scientific-run-01/completed.json):
  `454ba05d600d8d8a69726e2e58ee8d337040c2be64c0aa31f4f6854b87eb28f5`.
- [Supervisor terminal](../output/dialogue-observation-learning-v2/scientific-process-01.terminal.json):
  `7fe9b5e9fae7a83bdd04dceea559a849d78dcd346b97247b4b38d72be5d2b9ef`.
- [Actual exit and process/helper release](../output/dialogue-observation-learning-v2/scientific-terminal-review-01.json):
  `d1ab331064edd8f63e2772755ddef67875d9f91c773a6f574ddcfee632e43b40`.
- [Complete scorer](../output/dialogue-observation-learning-v2/report-01/receipt.json):
  `4983c8a96db377ebb70f92323c63ef600758eb22a59d84ddd808d08d0b314765`;
  9.458 seconds, zero model calls, scientific FAIL 6/7.
- [Independent saved-result audit](../output/dialogue-observation-learning-v2/result-audit-01/result-02/receipt.json):
  `9af7f9e2a2bd689aa497e78b2ba5c4f2ca3baec2e6500f73ef2942854d3142a2`;
  1.312 seconds, 144 fit cells, 24 literal-reference cells and 1,608 scalar
  comparisons, with agreement on all seven scientific conditions.
- [Actual report, audit and figure exits](../output/dialogue-observation-learning-v2/saved-result-terminal-review-01.json)
  record exit 0 and verify their complete saved-file membership and hashes.

The [first audit launcher failed](../output/dialogue-observation-learning-v2/result-audit-01/result-01/failed.json)
before entering the auditor body: resolving the virtual-environment interpreter
selected base Python, which lacks NumPy. No saved input was decoded.
The [exact invocation](../output/dialogue-observation-learning-v2/result-audit-01/result-01/invocation.json)
and [prospective correction](../output/dialogue-observation-learning-v2/result-audit-01/launcher-correction-01.json)
are preserved. Direct invocation through `.venv/bin/python` then completed in a
new directory with unchanged source, input hashes and limits. No training was
repeated, and this correction does not change the scientific failure.

## Execution history

The following immutable snapshots record progress before complete-study scoring.

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
- [Sixth completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-07.json)
  binds all six completed fits and the [second-seed frozen-numbers receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_numbers-6902/completed.json).
  Independent metadata review passed all 15 execution checks, including work
  counts, state checks, unchanged encoder and payload hashes. The seventh fit
  has started; this is halfway through the
  required fit count, not a completed scientific comparison.
- [Seventh completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-08.json)
  binds all seven completed fits and the [second-seed trainable-original receipt](../output/dialogue-observation-learning-v2/scientific-run-01/trainable_original-6902/completed.json).
  Independent metadata review passed all 15 execution checks, including the
  changed trainable encoder, payload hashes and all live and frozen source
  hashes. The eighth fit has started; task quality remains unscored.
- [Second complete-seed snapshot](../output/dialogue-observation-learning-v2/scientific-progress-09.json)
  binds all eight completed fits and the [second-seed trainable-numbers receipt](../output/dialogue-observation-learning-v2/scientific-run-01/trainable_numbers-6902/completed.json).
  Independent metadata review passed all 15 execution checks. Both first seeds
  now have all four variants; the third seed's first fit is training. These
  completion checks establish execution only, with task quality still unscored.
- [Ninth completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-10.json)
  binds all nine completed fits and the [third-seed frozen-original receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_original-6903/completed.json).
  Independent metadata review passed all 15 execution checks, including the
  unchanged frozen encoder and all live and frozen source hashes. The tenth
  fit is training; task quality remains unscored.
- [Tenth completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-11.json)
  binds all ten completed fits and the [third-seed frozen-numbers receipt](../output/dialogue-observation-learning-v2/scientific-run-01/frozen_numbers-6903/completed.json).
  Independent metadata review passed all 15 execution checks. All six
  frozen-encoder fits are now complete. The third seed's first trainable-encoder
  fit is running; the full comparison still requires both remaining fits.
- [Eleventh completed-fit snapshot](../output/dialogue-observation-learning-v2/scientific-progress-12.json)
  binds all eleven completed fits and the [third-seed trainable-original receipt](../output/dialogue-observation-learning-v2/scientific-run-01/trainable_original-6903/completed.json).
  Independent metadata review passed all 15 execution checks. Work counts,
  state checks, encoder change and raw payload hashes match the frozen plan.
  The final fit has started. Task quality remains unscored until all twelve
  fits and the complete saved-result audits finish.
- [Compute context](../output/dialogue-observation-learning-v2/compute-prelaunch-01.json)
  records released prior worker groups and background demo services. These are
  not isolated hardware latency measurements.

The complete execution met its eight-hour suspend-inclusive deadline. All twelve
fits and both saved-result audits are complete. Official DEV remains exposed
development evidence; official TEST is untouched. This is an observation-learning
comparison, with no established connectome, recurrent-architecture or world-model
advantage.

The qualified [V2 figure generator](../output/dialogue-observation-learning-v2/figure-01/README.md)
produced [actual result figures](../output/dialogue-observation-learning-v2/figure-01/render-01/receipt.json)
after authenticating both completed audits. They show every fit and all seven
conditions, preserving the scientific failure. The actual PNGs passed
[visual inspection](../output/dialogue-observation-learning-v2/figure-01/render-01-visual-qa.json).
Invented qualification plots remain local and are not benchmark results.

macOS [confirmed the active idle-sleep assertion during training](../output/dialogue-observation-learning-v2/runtime-idle-assertion-01.json).
Its release was verified after the actual supervisor exit, as recorded above.
