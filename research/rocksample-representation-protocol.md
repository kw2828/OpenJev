# RockSample representation and information diagnostic

21 September 2026. The previous policy-value screen remains **FAIL 3/10**.
This is a separate, prospective comparison on fresh maps. It asks whether the
state representation or absent map information prevents useful control. It does
not rerun or repair the prior cohort, train an architecture or establish novelty.

## Motivation and controls

The original fixed 256-map bank has no support for 125 of 1,210 labeled
rock-location pairs. This static property is independent of actual maps or
rewards. High effective sample size does not restore those missing possibilities.
The existing public factorized filter covers all 110 locations for each rock,
but ignores exclusion between different rocks. Comparing these representations
therefore changes both support and joint approximation, not support alone.

The former privileged arm also combined known map and known qualities. Add a
map-known, quality-unknown reference to test whether the shared planner can use
noisy checks to collect reward when localization is supplied. This is explicitly
a different information contract, not a public-input competitor.

Eight arms complete every case:

1. **exit:** unchanged direct east exit.
2. **factorized_full:** complete public history, independent joint distributions
   over each rock's location and quality, covering all 110 prior locations.
3. **factorized_recent128:** the same factorized representation, retaining only
   checks from the most recent 128 primitive transitions.
4. **factorized_latest:** the same representation, newest check per rock.
5. **particle_full:** unchanged full-history 256-map mixture, seed 530001.
6. **quality:** unchanged location-uniform quality-only model.
7. **known_map:** supplied true labeled rock positions, unknown qualities with
   probability 0.5 each; exact conditional quality filtering using public checks.
8. **privileged:** supplied true map and initial qualities, unchanged reference.

Every arm retains the public sampled-cell ledger. Recent/latest reconstruction
retains all samples in their original chronological positions. No public arm
receives realized rewards, true map, qualities, simulator state or environment
seeds. Known-map receives coordinates only; privileged additionally receives
initial quality bits. Beliefs and histories reset every episode.

## Fixed task, planner and budget

Use the already qualified, unchanged raw POBAX RockSample(11,11), pinned upstream
commit and isolated CPU runtime. Eight constructor seeds **13001-13008** and
four reset seeds **23001-23004** produce 32 cases per arm, **256 episodes** total.
Transition keys use `440000 + map_index*4 + reset_index`, shared across arms in
each case. Rotate arm order by case index modulo eight. Different actions yield
different histories. Do not substitute seeds after observing results.

Reuse the frozen policy-value episode loop and route planner. The native horizon
is 1,000 primitive steps and the check cap is 64. Four exploitation tours compete
with direct exit and the fixed sensing bundles. Sensing must improve the
undiscounted forecast by strictly more than 0.05. No forced checking, new planner,
particle expansion, resampling, reward input or sensor modification.

Only an adapter exposes the existing factorized filter to the shared planner.
Its sampling forecast is the sum of good-location mass minus bad-location mass,
times 10. Independence can assign summed occupancy above one to a cell. Record
maximum occupancy, number of overfull cells and sampling-forecast extrema at
every actual planning decision. Do not clip or repair them. These diagnostics
do not establish a globally coherent posterior or inspect every hypothetical
sensing leaf. The filter remains approximate even when their bounds hold.

One native run: **1,800 suspend-inclusive seconds, 8 GiB peak RSS, 512 MiB output,
192,000 primitive transitions**. The step budget is intentionally below the
256,000 worst-case total of the cohort; hitting it fails completion and does not
authorize an extension. Preserve incomplete outputs and actual process status.
Synthetic engineering qualification is bounded separately to 600 execution
seconds, with no native episodes or model calls.

Report raw reward, gamma-0.99 reward, steps, checks, sample outcomes, exits,
planner work, filter/planner time and all representation diagnostics. Charge
per-episode representation construction to filter/controller time. Initial shared
particle-bank construction belongs to whole-run cost. Recorded controller cost
excludes simulator, reset, I/O, imports and JIT; whole-run elapsed includes them.
Shared search limits do not match actual computation. No speed claim is targeted.

## Continuation and stop rule

Give each map equal weight. All **13** conditions are required before considering
a learned memory pilot:

- Factorized full beats each of its own recent-128 and latest controls, quality
  only and particle full by at least **5 reward units**, and has a strictly
  positive difference on at least **6/8 maps** against each. Eight conditions.
- Factorized full exceeds direct exit by at least **5**. One condition.
- Known-map and privileged each exceed direct exit by at least **20**. Two
  conditions. These qualify information use and informed planner capacity.
- All 256 cases complete with valid action, episode and process accounting.
- At every actual factorized-full planning decision, summed occupancy is at
  most one and expected sample rewards remain in [-10,10]. This rejects an
  observed one-cell contradiction; it does not prove exact Bayesian coherence.

Thresholds are exploratory practical limits, not significance tests. Known-map
success alone cannot admit recurrence. If latest/recent memory explains the gain,
report that. If all unknown-map arms fail while known-map succeeds, localization
or the public-information contract remains unresolved. If known-map also fails,
this planner has not established useful sensor-based control. No learned pilot,
threshold revision, selective map report or outcome-dependent extension follows
a failed screen. Audit/reporting read saved artifacts only.
