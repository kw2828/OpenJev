# Fixed training coverage collection stopped at a sampling quota

**All 72 planned trajectories were collected, but the proposed training mixture was not admitted.** Three required groups had too few pre-action states. The fixed sampling rule stopped the run after 52,072 native steps. No continuation training or held-out evaluation ran, and this attempt provides no architecture-performance result.

## What happened

The [prospective protocol](otto-coverage-protocol.md) fixes three unchanged width-eight collectors, 24 fresh training cases, and a 5,589-row training mixture: 2,799 original teacher rows and 2,790 student rows. It preserves the original six sensing/initial-hit stratum sizes, splits student quotas across all three collectors, and samples row IDs without replacement by a fixed hash. Rows are ranked by fixed IDs, without filtering or ranking by outcomes; episode length determines how many pre-action rows are available. An insufficient quota must stop collection rather than admit more seeds or duplicate rows.

There were 52,072 pre-action states overall, but most were outside the deficient stratum. For sensing length 3 and initial hit 2, the four cases per collector ended too quickly to supply the required rows:

| Collector seed | Available rows | Required rows | Shortfall |
| ---: | ---: | ---: | ---: |
| 10101 | 35 | 122 | 87 |
| 10102 | 23 | 121 | 98 |
| 10103 | 31 | 121 | 90 |

The other 15 collector/stratum groups met their count requirements. Their surplus cannot fill these gaps under the frozen rule. No replacement, extra episode, resampling or quota relaxation was performed.

The preserved execution has 49 found and 23 horizon-censored trajectories, each with its final public update recorded. These are descriptive training-collection counts, not an independent success-rate estimate. The worker returned 52,072/52,072 native steps and scalar readouts, 74/74 resets and 3/3 checkpoint loads, with no pending operation. The original supervisor exited with code 1 after **131.56 seconds**, without timeout, and confirmed the process group had ended.

## Evidence and limits

The metadata review independently checks fixed cohort membership, episode chronology, quota counts, recorded operation totals and process closure. It does **not** read the detailed transition/work journals, rerun the simulator, execute model inference, reconstruct numerical posterior/branch values, or replace the planned complete-collection audit. That audit requires a completed mixture and was not run. No mixture arrays, selected-row ledger or successful completion receipt were produced.

All original failure files and raw traces are retained in the [evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-coverage-collection-v1). Inspect the [frozen plan](../output/otto-coverage-v1/collection-plan-01.json), [original supervisor terminal](../output/otto-coverage-v1/collection-process-01.terminal.json), [worker failure](../output/otto-coverage-v1/collection-01/failed.json) and [metadata review](../output/otto-coverage-v1/failure-review-01/receipt.json).

The engineering checks passed before execution: 53 selector fixtures, 12 collector fixtures and 42 auditor fixtures. After two formatting fixes in the auditor tests, the 13 affected fixtures and lint passed again. These checks establish implementation behavior, not sufficient coverage or better learned control.

## Consequence for the next experiment

This collection design under-supplied one required stratum despite a large total state count. It does not test whether training on student states improves control, and does not revise the previous capacity or Bellman results. Stage 2 remains unexecuted and unadmitted. A subsequent collection design would need fresh seeds and a separately frozen allocation; this attempt and its failed quota remain unchanged.

The [implementation review](../output/otto-coverage-v1/stage2-implementation-review.md) records how a future matched comparison would bind each dataset to every target refresh and prevent placeholder return labels from reaching optimization. It is planning only, contingent on independently verified training data.
