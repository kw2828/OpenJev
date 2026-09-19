# Changing motor strength: engineering result

**The six-controller pipeline works, but this case does not justify a learned adaptation study.** Rolling identification improved post-change native cost by only **0.21%** versus nominal physics, while total cost was **0.87% worse**. The same-observer reference with the actual current gain improved post-change cost by **1.70%**. This is one reused engineering initial condition, not a scientific qualification or architecture result.

![All six controllers, native costs and the parameter used for planning](../output/persistent-dynamics-qualification-v1/control-engineering-results.png)

## What ran

One seed-410 initial condition, six controllers, 200 actions each. The arm receives a new public target at actions 0, 50, 100 and 150. Hidden motor strength changes from 0.7 to 1.3 at action 80. All controllers share initial state, targets, actuator-noise arrays and initial search innovations. No controller sees future targets, noise or the gain-switch schedule.

The five planning controllers use the same 256-candidate, 12-step CEM configuration. The rolling estimator scores 21 fixed candidate gains against its last 20 observed transitions. The frozen estimator stops updating after transition 40. Both receive only public angles and issued commands. Current-gain and true-state references have separately declared privileges. These finite-search references are not optimal control bounds.

This is a custom tracking task using native Reacher physics. There was no model training, connectome comparison or new scientific seed cohort. The run was fixed before execution and completed once, without retry.

## All results

Native cost per action, lower is better. The post-change interval is actions 80-199; the first complete new-target interval after the switch is 100-149.

| Controller | Whole episode | After gain change | First new-target interval |
|---|---:|---:|---:|
| Nominal gain | 0.113498 | 0.104591 | 0.056778 |
| Rolling estimate | 0.114488 | 0.104374 | 0.064468 |
| Frozen estimate | 0.118485 | 0.111294 | 0.057278 |
| Known current gain, public observer | 0.133947 | 0.102810 | 0.045301 |
| Known current state and gain | 0.142544 | 0.125210 | 0.123920 |
| Zero command | 0.178585 | 0.172680 | 0.179576 |

Rolling identification beats the stale estimate by **6.22%** after the switch, but its advantage over nominal physics is negligible. During the first complete new-target interval it is **13.54% worse** than nominal. Its mean absolute gain error over all post-change planning roots is **0.16125**, and it already overestimates gain before the hidden change. The shorter roots-110-through-129 average of 0.09 must not substitute for the full trace or the proposed requirement at every offset.

The privileged state reference also performs worse than nominal in this case. That does not show that accurate state is harmful. Different roots and dynamics produce different finite-search trajectories; the planner's adequacy remains unresolved. Horizon, state estimation and local search behavior are candidate explanations, not established causes.

## Verification and cost

All six rows completed before any numerical interpretation. Independent replay checked **1,200 executed transitions, 2,987,520 candidate transitions, 1,000 selected advances and 5,040 identification transitions**. Native state and identifier replay errors were zero. Geometry arithmetic differed by at most `2.384185791015625e-7`, within its declared tolerance; recorded float32 score arithmetic and selection were checked exactly.

Execution took **50.887 seconds**, full replay **49.386 seconds**, and the enclosing process **101.268 seconds**. Each planning row took 9.80-10.25 seconds for 200 actions including setup, native control, copying, trace writes and hashing. These are instrumented shared-host measurements, not isolated deployment latency. Internal timers overlap and must not be added to the outer totals.

An independent saved-output review authenticated all **4,054 run files**, fourteen live/source-snapshot files, and the unchanged 116-source and 136-source maps from the earlier closed studies. The four new component suites passed 107 focused tests: planning bank 36, policy 14, audit 38 and rollout 19. Earlier wrapper and estimator suites contribute another 64 tests. Test counts describe engineering coverage, not scientific efficacy.

[Saved results and limits](../output/persistent-dynamics-qualification-v1/control-results-review-01/results_review.md) · [Complete arithmetic and hashes](../output/persistent-dynamics-qualification-v1/control-results-review-01/control-results-review.json) · [Run driver](../output/persistent-dynamics-qualification-v1/run_control_engineering.py) · [Independent audit review](../output/persistent-dynamics-qualification-v1/control-audit-review.md) · [Failure-handling review](../output/persistent-dynamics-qualification-v1/rollout-review.md).

[Download complete engineering evidence](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-tracking-engineering-v1), including every raw decision trace, input, source snapshot and replay receipt.

## Next decision

[Independent readiness review](../output/persistent-dynamics-qualification-v1/next-decision.md).

Keep the four-fresh-case screen and neural training unlaunched while diagnosing planner and observer behavior on these saved traces. The prospective 5% current-gain and 3% rolling-estimator margins remain unchanged; no scientific gate was evaluated here. Do not retune this engineering case into an architecture success or reopen the earlier negative studies.

A useful next mechanism must outperform nominal physics and conventional identification, preserve ordinary tracking, and pay for its online computation. Fast-weight world models remain a conditional follow-up. [Primary-paper review and proposed comparisons](reacher-fast-adaptation-source-review.md) · [Prospective qualification criteria](../output/persistent-dynamics-qualification-v1/qualification-protocol-review.md).
