# Does persistent memory beat two observations and recent actions?

**The stronger comparison failed its continuation rule: 24 of 25 checks
passed.** Persistent GRU reduced mean native control cost by **4.44%** with
six-step sensing gaps and **2.94%** with ten-step gaps versus a separately
trained two-observation model. The frozen rule required at least 3% in both.
All three paired fits improved on both gap panels, but the shifted mean missed
the required margin. The threshold and failed result remain unchanged.

![All nine models and five references on all three sensing panels](../evidence/reacher-two-observation-study-v1/report/native-costs.png)

This narrows the earlier [positive memory result](reacher-geometry-memory.md).
Persistent memory remains better than the one-observation control, but its
advantage is much smaller against two observations and intervening commands.
The failed superiority margin does not prove equivalence or show that two
observations fully explain the earlier gain. Supplied-physics references still
outperform all three learned families. No biological or architectural novelty
is established.

## All family and reference means

Lower cost is better. Each learned-family entry averages all three fits; every
row uses the same 64 fresh initial conditions and disturbance innovations.

| Controller | Full sensing | Six-step gaps | Ten-step gaps |
|---|---:|---:|---:|
| Persistent GRU | 5.29152 | 5.41583 | 5.89316 |
| Two observations + intervening actions | 5.35185 | 5.66769 | 6.07162 |
| One observation + subsequent actions | 7.28927 | 7.80196 | 8.10416 |
| True-state supplied physics | 4.47874 | 4.47874 | 4.47874 |
| Public particle filter + supplied physics | 4.59850 | 4.63607 | 4.77363 |
| Public kinematics + supplied physics | 4.50167 | 4.52210 | 4.64364 |
| Zero command | 11.05255 | 11.05255 | 11.05255 |
| Uniform commands | 42.57883 | 42.57883 | 42.57883 |

Persistent cost was 30.58%/27.28% below the one-observation control on the gap
panels. These are descriptive means from this new cohort, separate from the
31.9%/26.0% result of the preceding study. All 42 row means are available in
[CSV](../evidence/reacher-two-observation-study-v1/report/all-rows.csv), with
individual episode costs and every criterion in the
[completed audit](../evidence/reacher-two-observation-study-v1/audit/summary.json).

The sole failure was the ten-step mean comparison:
`5.89315642123704 <= 0.97 * 6.071618747407715` is false. Its right side is
`5.889470184985484`. All 12 paired-fit non-worsening checks, both full-sensing
checks, seven competence checks and the other three gap mean checks passed.
Passing 24 checks does not qualify the overall result.

## What was matched and what was paid

The [published protocol](../evidence/reacher-two-observation-study-v1/protocol/plan.json)
trained three fresh history controls from the original paired initial weights,
using the same 768 training episodes, minibatch orders, objective and 1,152
updates per fit. It retained all six persistent/one-observation models.
All nine models and five references ran on all three sensing panels, producing
2,688 episodes and 134,400 native decisions. There were no Astra calls, new
training episodes, replacement cases or post-result fitting.

The history control reconstructs state from the last two valid observations
and the intervening issued actions at each real decision. Its padded kernels
pay for twelve observation updates and eleven transitions. It does not retain
an older learned state between decisions. Every learned controller uses the
same geometry score and CEM256 search budget; equal search budgets and parameter
counts do not match total computation. Supplied-physics controllers receive
additional model knowledge, and the true-state reference additionally receives
hidden velocity.

Actual execution took **2,772.48 seconds**, including **836.72 seconds** for
the three new fits. The separate audit took **1,356.93 seconds**. These are
shared-host measurements, not isolated deployment latency. Full work counts
and per-row times are retained in the audit. New training made 3,456 optimizer
updates; evaluation made none.

## Verification and limits

Both original processes exited successfully. The saved-output audit checked
all 42 rows, public history construction, candidate scores, actions and work
accounting. It independently replayed 134,400 executed, 78,741,504 candidate
and 28,800 selected simulator transitions, plus public observer transitions,
with zero recorded replay discrepancy. All 116 frozen scientific source files
remained unchanged.

The auditor did not rerun neural inference, gradients or optimizer updates.
It checked saved learned tensors and training arithmetic. This distinction
matters: successful artifact verification is not a numerical reproduction of
every fit. Three paired fits and one environment do not establish statistical
equivalence, generalization to a new task, or biological superiority.

- [Original process exits](../output/reacher-two-observation-study-v1/process/completed.json)
- [Audit receipt and execution hashes](../evidence/reacher-two-observation-study-v1/audit/receipt.json)
- [Independent saved-result review](../output/reacher-two-observation-study-v1/results-review.md)
- [Figure receipt](../evidence/reacher-two-observation-study-v1/report/receipt.json)
- [Implementation, preparation and engineering history](../output/reacher-two-observation-control-v1/README.md)

[Download all nine fitted models and results](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-two-observation-v1).
The bundle includes the fit logs, frozen source/protocol and audited results.
Its [300-member verification](../output/reacher-two-observation-study-v1/model-release-v1/verification.json)
reopens and hashes every archived file. The complete 20.07 GB execution payload
is retained locally and hash-bound by the audit; raw execution traces are
explicitly excluded from this smaller model release.

The next proposed task asks whether a persistent change in motor strength is
identifiable from public observations **and** useful for control. The
[adaptation review](reacher-fast-adaptation-source-review.md) and
[prospective task design](../output/persistent-dynamics-qualification-v1/design-review.md)
require a conventional identification baseline before another neural fit.
That is a separate qualification, not a repair or extension of this failed rule.
