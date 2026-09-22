# Teacher-cost uncertainty: analytical budget check

The range-only Hoeffding bound is too conservative to require confident action rankings in a small pilot seeking gaps of a few dozen moves. With the current task horizon of **2,188**, four actions and per-anchor familywise `alpha=0.05`, obtaining a radius of at most **10 moves** requires **524,275 replicate vectors at one anchor**. Each vector contains four continuations, totaling **2,097,100 rollouts**. This is an analytical projection, not a measured gap, label-quality result or admitted sampling allocation.

The [implemented reducer](otto-teacher-cost-component.md) reports

```text
r(N) = sqrt(2 * 2187^2 * log(240) / N)
N_min(delta) = ceil(2 * 2187^2 * log(240) / delta^2)
```

| Desired radius, moves | Minimum replicate vectors | Continuation rollouts at four actions |
| ---: | ---: | ---: |
| 1 | 52,427,453 | 209,709,812 |
| 5 | 2,097,099 | 8,388,396 |
| 10 | 524,275 | 2,097,100 |
| 25 | 83,884 | 335,536 |
| 50 | 20,971 | 83,884 |
| 100 | 5,243 | 20,972 |

At 16, 64, 256 and 1,024 replicate vectors, the radii are respectively **1,810.17, 905.09, 452.54 and 226.27 moves**. Common randomness can reduce observed paired variance, but this range-only bound does not use that variance reduction. Shortening the horizon to obtain narrower bounds would change the learning objective.

These sample sizes target a radius only. Strict observed resolution requires the observed absolute mean gap to exceed that radius. On the simultaneous coverage event, a true gap greater than twice the radius is sufficient for strict correct resolution. True gaps and detection power are unknown here. The guarantee remains conditional on the sampling assumptions in the component documentation and covers one fixed anchor, not selected anchors or policy outcomes.

This changes the prospective design before any labels are collected: retain fixed-budget continuous paired-cost targets for every planned anchor, report their uncertainty, and judge usefulness through a separately frozen autonomous comparison. Unresolved range-only intervals do not by themselves show that such targets are useless. Excluding uncertain anchors would change the training distribution. If confident ordering becomes necessary, a variance-adaptive interval must be selected and qualified prospectively; descriptive standard error alone is not a confidence guarantee. No existing empirical pass/fail rule is changed.

The [machine-readable calculation](../output/otto-teacher-cost-engineering-v1/analytic-budget-01.json) records each radius at `N_min` and `N_min-1`, continuation counts and worst-case step counts. It checks the integer sample-count boundary for every row. The calculation used no empirical records, models, simulators, training or label collection. Original tool chunk: `9b1117`, exit 0. Independent mathematical review agreed with the formula and all sample counts.

Real collection remains unadmitted. The [conditional learning follow-up](otto-action-cost-followup.md) still depends on the original spatial-control audit and on qualifying a public-only rollout sampler with complete accounting.
