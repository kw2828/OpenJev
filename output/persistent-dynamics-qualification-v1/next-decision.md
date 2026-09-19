# Engineering-to-screen decision

**Recommendation: hold the four-fresh-case screen.** The completed six-role engineering loop is reproducible, but this fixed case does not yet demonstrate that the current controller makes sufficiently useful use of gain knowledge or online identification. Preserve the proposed screen and its thresholds unchanged. Do not promote this case into scientific evidence, tune against it, or start a neural adaptation comparison from these results.

This review reads completed saved artifacts only. It makes no simulator, model, controller or RNG calls. The case reuses engineering seed 410, fixed targets and one gain change from 0.7 to 1.3 at root 80. It covers neither the reverse change nor fresh cases, nominal tracking or the ordinary competence panel. There is no scientific qualification decision for this run.

## Completion and evidence boundary

The outer process completed with exit code 0 in 101.2683 seconds. Execution took 50.8867 seconds and the saved-output audit phase 49.3862 seconds. All six row audits completed. They checked 1,200 real transitions, 2,987,520 candidate transitions, 1,000 selected-action transitions and 5,040 gain-identification transitions. Native control, candidate state and identification replay errors were zero; maximum independent geometry arithmetic error was `2.384185791015625e-07`.

I checked the aggregate execution digest, all six audit digests, all six row completion digests, their episode-file digests, and the gain-observation file digests used below. I then independently recomputed the cost windows from the saved native transition rewards. This is a targeted results review, not a second full-member or native-replay audit.

| Bound artifact | SHA-256 |
|---|---|
| [Outer completion](control-process-01/completed.json) | `9ce680d4e038fef3276e865059cc17f835d18d5dcc4fd13398668a4931f123b7` |
| [Aggregate completion](control-engineering-01/completed.json) | `064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050` |
| [Execution completion](control-engineering-01/execution-completed.json) | `4efa30de325f17f76753179a0e423726099deec8ea31a72551c5663efdbc9312` |

The aggregate completion binds the six audits and row completions. The [earlier source review](control-audit-review.md) documents the numerical and information-boundary checks. This review does not depend on the separate aggregate reviewer's unfinished memo.

## What the one case shows

All costs below are **native cost per action**, lower is better. POST is actions `[80,200)`; MOVE is the first complete new-goal interval after the switch, `[100,150)`. Full covers `[0,200)`. These are the previously proposed windows, not selected favorable intervals.

| Controller | Full | POST | MOVE | Whole row seconds |
|---|---:|---:|---:|---:|
| Public nominal | 0.11349797 | 0.10459064 | 0.05677829 | 9.9146 |
| Public rolling ID | 0.11448848 | 0.10437350 | 0.06446815 | 9.8003 |
| Public ID then freeze | 0.11848501 | 0.11129425 | 0.05727800 | 10.2497 |
| Public state, true current gain | 0.13394715 | 0.10280964 | 0.04530133 | 9.9972 |
| True state and current gain | 0.14254395 | 0.12520961 | 0.12392000 | 10.1537 |
| Zero command | 0.17858525 | 0.17267980 | 0.17957590 | 0.5461 |

The same-observer true-gain reference improves POST cost by **1.70%** versus nominal, below the proposed 5% requirement. Its 7.62% improvement versus frozen ID cannot replace the required nominal comparison. It does improve MOVE cost by 20.21% versus nominal, so these results do not establish that gain information is entirely irrelevant or that only a settled tail is being measured.

Rolling ID improves POST cost by only **0.208%**, or **0.00021714 per action**, versus nominal. Both fall below the proposed 3% and 0.001 materiality requirements. Its 6.22% POST improvement versus frozen ID does not rescue this comparison. MOVE cost is **13.54% worse** than nominal and **12.55% worse** than frozen ID. Full-episode cost is 0.87% worse than nominal. Reporting only POST versus frozen ID would obscure the central weakness.

There is a real tradeoff in the recorded components. Rolling ID lowers POST distance cost from 0.09147267 to 0.08932766, but increases actuator cost from 0.01311797 to 0.01504584. Its small total improvement cannot be described as an unqualified control win. In MOVE, both distance and actuator costs are higher than nominal.

At the predeclared planning roots 110 through 129, rolling ID's average absolute gain error is **0.09**, but five individual offsets exceed 0.1: root 125 has error 0.15 and roots 126 through 129 have error 0.20. The 20-root average alone therefore hides failure of the proposed per-offset recovery condition in this case. The estimate's movement to the top of the grid is a diagnostic observation, not permission to widen the grid, smooth the estimate or relax the criterion after seeing it.

The true-state/current-gain reference is **19.71% worse** than public nominal on POST. Both true-gain references also have substantially worse pre-switch cost than public nominal. These finite-horizon, finite-search controllers are not monotone utility upper bounds: more accurate information need not improve their closed-loop trajectories. Because the true-state reference is also affected, the outcome cannot be attributed to public velocity estimation alone. Candidate-search variation, short-horizon planning, objective approximation and closed-loop path dependence remain possible explanations; this review establishes none of them causally. Audited replay consistency rules out neither a weak controller nor a task whose useful adaptation effect is too small.

The roughly ten-second planned rows show no obvious engineering runtime obstacle. Rolling ID's row happened to finish 1.15% faster than nominal, despite its additional identification work. That single shared-host timing is not evidence that identification is free or faster, and cannot establish the proposed platform overhead condition.

## Decision and next boundary

The engineering machinery is ready to preserve a larger experiment, but the scientific motivation is not ready to consume its fresh cases. Keep this run as a completed engineering result with the unfavorable comparisons intact. Do not count it as the four-case screen or claim that the task has failed across seeds.

The next useful step is a bounded diagnosis of the already saved controller and observer evidence: check the pre-switch reference weakness and the identification drift against the existing causal/state/scoring contracts. A demonstrated implementation error would justify a separately retained repair and engineering rerun. An algorithmic change, such as a different horizon, observer, objective, grid or excitation process, needs a distinct prospective design and new engineering attempt; it must not silently modify the current screen or its thresholds.

If no concrete defect is found, stop this controller/task recipe rather than using repeated fresh cases to search for a favorable result. The unchanged four-case screen would still be a valid explicit test of this fixed recipe if separately chosen, but these engineering results provide no positive reason to advance it now. The current evidence supports neither a learned fast-adaptation mechanism nor a claim that persistent hidden dynamics alone creates a valuable benchmark.
