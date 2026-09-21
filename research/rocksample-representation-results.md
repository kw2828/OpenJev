# RockSample: factorized filtering improves reward, but longer memory is not yet justified

21 September 2026. The eight-controller comparison completed **256 episodes on
eight fresh maps**. The factorized full-history controller averaged **42.8125**
native reward versus **36.8750** for the fixed-particle controller on the same
cases. However, it did not sufficiently beat recent history or the simpler
quality-only controller, and its independent-rock approximation violated native
occupancy bounds. **8/13 conditions passed; continuation failed. No learned
memory pilot is admitted.** This is a classical state-estimation experiment,
with no trained recurrent model, connectome or new architecture result.

![All eight map means, measured controller costs and paired differences](../output/rocksample-representations-v1/figure-01/representations.png)

## What changed

The preceding [control screen](rocksample-policy-value-results.md) failed its
continuation rule. A source-only diagnostic then found that its fixed 256-map
bank assigned zero support to **125 of 1,210 labeled rock-location pairs**.
Weighting those fixed hypotheses cannot introduce an absent location.

This experiment adds a factorized filter with support for every possible
location of each rock. It also removes cross-rock exclusion: separate rock
beliefs can overlap. The paired comparison therefore changes both support and
the joint-state approximation. Its gain cannot be attributed to support alone.
The planner, native task, action costs and sensing limits are unchanged.

The [protocol](rocksample-representation-protocol.md), implementation and source
manifest were committed as `9a1f323` before the run. The raw POBAX
RockSample(11,11) interface used constructor seeds 13001-13008 and reset seeds
23001-23004, with all eight controllers on each case. Arm order rotated. Reset
and transition keys were paired, but differing actions produce differing
observation histories. Each episode retained the native 1,000-step horizon and
64-check cap. No replacement seeds or extensions were used.

Public controllers receive positions, actions and noisy check readings, with
the same persistent public sampled-cell depletion rule. They do not receive
rewards, hidden locations, hidden qualities or simulator seeds. Recent-128
retains checks from the last 128 primitive transitions; latest retains one
check per rock. Both retain and chronologically replay all sample events.
Quality-only infers initial qualities while keeping locations uniform.

Two additional-information references separate localization from quality
inference. **Known map** receives true locations, initializes every quality
probability at 0.5, and infers qualities from noisy public checks. **Privileged**
also receives true initial qualities. Neither is a fair public-input competitor
or an optimal-planning oracle.

## All results

Values average four resets within each map, then average the eight maps.
Undiscounted native reward is primary and is the planner's forecast target.
Gamma-0.99 return is a secondary measure of when rewards arrive.

| Controller | Native reward | Gamma-0.99 return | Steps | Checks | Controller seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Immediate exit | 10.0000 | 9.6569 | 4.50 | 0.00 | 0.00000264 |
| Factorized full history | 42.8125 | 7.6170 | 344.06 | 43.50 | 3.19277 |
| Factorized recent-128 | 38.4375 | 6.9296 | 338.97 | 51.25 | 2.69787 |
| Factorized latest check | 28.1250 | 5.9315 | 288.69 | 34.13 | 2.24932 |
| Fixed-particle full history | 36.8750 | 2.0343 | 409.44 | 62.75 | 0.64523 |
| Quality only | 40.6250 | 13.1349 | 204.63 | 12.63 | 1.72390 |
| Known map, qualities inferred | 77.5000 | 15.7640 | 181.78 | 56.50 | 0.35469 |
| Known map and qualities | 77.5000 | 61.6935 | 41.53 | 0.00 | 0.14353 |

All 256 episodes exited naturally. Controller time includes representation
construction, filtering, replay, diagnostics and planning. It excludes
environment calls, reset, trace I/O, imports, JIT and bookkeeping. These are
whole-episode CPU measurements from one run, not deployment latency or optimized
implementations. Identical planner limits do not imply equal actual compute.

Full factorized history improves reward over fixed particles on all eight maps,
but takes about **4.95 times** their controller time. Its smaller gain over
quality-only costs about **1.85 times** as much controller time and reverses on
discounted return. Full history averages 5.90625 good, 2.625 bad and 72.03125
empty samples per episode. Both map-informed references collect 6.75 good rocks
with no bad or empty samples. Known-map achieves the same raw reward as the
privileged reference, but requires many more checks and steps. This supports
further diagnosis of localization and sensing cost; it does not isolate either
as the sole cause of the public controllers' shortfall.

The prior study used different maps and reset seeds. The increase from its
absolute reward levels is **not** a treatment effect. Only comparisons within
this complete paired cohort support the contrasts above.

## Every continuation condition

| Condition | Observed | Decision |
| --- | ---: | --- |
| Full minus recent-128 reward at least +5 | +4.3750 | Fail |
| Full beats recent-128 on at least 6/8 maps | 5/8 | Fail |
| Full minus latest reward at least +5 | +14.6875 | Pass |
| Full beats latest on at least 6/8 maps | 8/8 | Pass |
| Full minus quality-only reward at least +5 | +2.1875 | Fail |
| Full beats quality-only on at least 6/8 maps | 4/8 | Fail |
| Full minus fixed-particle reward at least +5 | +5.9375 | Pass |
| Full beats fixed particles on at least 6/8 maps | 8/8 | Pass |
| Full minus exit reward at least +5 | +32.8125 | Pass |
| Known-map minus exit reward at least +20 | +67.5000 | Pass |
| Privileged minus exit reward at least +20 | +67.5000 | Pass |
| Complete valid cohort | 256/256 | Pass |
| Every observed full planning state respects occupancy and reward bounds | 6 overfull decisions | Fail |

The last condition checks all **2,958 actual full-history planning states**.
Maximum cell occupancy was **1.046110**, exceeding one, and minimum expected
sample reward was **-10.461100**, below the native -10 lower bound. These values
are deliberately not clipped. Recent-128 also had two overfull decisions;
latest had none among its 2,102 planning decisions.

The diagnostics do not establish coherence of the full joint posterior and do
not examine every hypothetical sensing leaf. The violations expose a real
approximation defect, but do not show that fixing it would recover the failed
performance margins. Clipping a forecast is not a coherent belief update.
All thirteen requirements were fixed before execution. They are exploratory
practical thresholds, not significance tests. Eight maps do not establish
general benchmark performance. This interface hides locations and should not
be compared directly with standard RockSample results supplying the map.

## Verification and reproducibility

The run completed **58,035 native transitions, 14,634 decisions and 8,344
checks**, exiting 0 in **372.118930 suspend-inclusive seconds**. Peak worker RSS
was **630,243,328 bytes**. No training updates or external model calls occurred.
The prelaunch qualification passed **267 synthetic tests**. Those tests cover
implementation behavior, public-input isolation and accounting, not efficacy.

An independent saved-output auditor agrees on **293,561 scalar comparisons**,
with maximum absolute difference **1.42e-14**. It reconstructs action and
depletion joins, rewards, discounts, work counts, recorded diagnostics,
construction costs, all map means and all thirteen conditions. A separate
read-only arithmetic review independently agrees on the cohort, returns,
diagnostic aggregates and continuation decision. Neither reruns the simulator
or Bayesian inference, establishes planner optimality, or independently measures
timing. Hypothetical sensing-leaf coherence remains unaudited.

- [Frozen plan and 22 source hashes](../output/rocksample-representations-v1/plan-01.json)
  and [synthetic qualification](../output/rocksample-representations-v1/engineering-01/qualification.json).
- [Static support diagnostic and prior-only timing](../output/rocksample-representations-v1/engineering-01/prior-timing-and-support.json).
- [All episodes](../output/rocksample-representations-v1/run-01/episodes.jsonl),
  [transitions](../output/rocksample-representations-v1/run-01/transitions.jsonl),
  [decisions](../output/rocksample-representations-v1/run-01/decisions.jsonl) and
  [all aggregates and conditions](../output/rocksample-representations-v1/run-01/summary.json).
- [Run receipt](../output/rocksample-representations-v1/run-01/receipt.json), SHA-256
  `ba59ee5b9bd1d9958bb1d2b3bf3c32e6846e2f2e3fc3ee9bc0986cb990494d02`,
  and [supervisor terminal](../output/rocksample-representations-v1/run-process-01.terminal.json).
- [Independent auditor](../scripts/audit_rocksample_representations.py),
  [audit receipt](../output/rocksample-representations-v1/audit-01/receipt.json)
  and [execution witnesses](../output/rocksample-representations-v1/execution-witness.json).
- [Standalone figure PDF](../output/rocksample-representations-v1/figure-01/representations.pdf),
  [plotted values](../output/rocksample-representations-v1/figure-01/plotted-values.json)
  and [figure source](../scripts/plot_rocksample_representations.py).

## Research consequence

The support-limited particle bank is an inadequate sole comparator for a new
model. Recent history and quality-only remain necessary controls. The present
results do not justify distilling the factorized controller into a learned
recurrent architecture, relaxing the thresholds, or claiming a biological
learning advantage.

A separately specified coherent location model could address the revealed
approximation defect, but improvement is unproven. Another source-reviewed
candidate is [olfactory search](olfactory-search-opportunity.md), which has
released classical and neural reference policies and a direct connection to
insect navigation. Its reproducible raw-observation interface and local runtime
still need qualification. That source review is not a benchmark result or
permission to bypass the failed RockSample continuation rule.
