# OTTO 53x53: larger-setting odor-memory opportunity pilot

21 September 2026. **Prospective protocol; no efficacy result.** This bounded
pilot asks whether older odor observations improve control beyond a recent
window and a shared visitation ledger. It trains no model and establishes no
connectome, recurrent-architecture or ICLR contribution. The preceding OpenJev
RockSample and 19x19 OTTO failures remain unchanged. This setting is selected
after the short searches in the 19x19 pilot, so it is an exploratory development
study, not independent confirmation of a learned architecture.

## Source, task and execution route

Use unchanged OTTO-benchmark commit
[`a6aaef6507cffd2aff79291c1019f506f616bbef`](https://github.com/auroreloisy/otto-benchmark/tree/a6aaef6507cffd2aff79291c1019f506f616bbef).
The [source review](olfactory-search-opportunity.md) and
[source inventory](../output/otto-source-review-v1/receipt.json) document the
inspected interface. Bind the upstream files, local adapter/runner, protocol,
tests and isolated runtime in an externally pinned plan before execution.

Use the official isotropic-53x53 parameters: two dimensions,
`lambda_over_dx=3`, `R_dt=2`, Euclidean sensing. Keep automatic grid/hit-category
selection and require it to produce **N=53 and Nhits=4**. Failure of that check
stops execution; do not silently substitute dimensions. The simulator uses
`draw_source=True`: a source is sampled once per case and a search ends when
the agent reaches it. The fixed horizon is **2,188 primitive moves**.

The isolated runtime pins NumPy 2.5.3 and SciPy 1.18.1 for the simulator and
heuristics. This does not qualify the released TensorFlow neural policies, Gym wrapper,
PBVI policy files or official evaluator. Do not download weights or train.

## Public information and memory controls

Every actor receives the same task rules, initial hit, public position, issued
actions, subsequent hit categories and episode boundary. Position is obtainable
from the known initial center and deterministic moves. No actor receives the
sampled source, random seed/stream, simulator posterior, simulator hit map or
future detections.

The actor's `SourceTracking(draw_source=False, initial_hit=public_initial_hit)`
object is separate from the simulator. It is a policy-facing public model;
never call its `step` method to generate observations. Its belief and entropy
must be maintained solely from the actor's permitted history. Full-history
posterior comparisons with the simulator are evaluator-only diagnostics.

All arms retain a permanent mask of cells visited without finding the source.
They also retain the initial-hit-conditioned prior. Thus this is a test of
**odor evidence beyond a common visitation ledger**, not a comparison against
an actor with no memory at all. Zero hits are informative and count as ordinary
completed observations. Category 3 contains the saturated Poisson tail (three or more detections).

At decision time after `t` completed moves, define four belief constructions:

- **full:** initial-hit-conditioned prior and every completed odor observation.
- **recent32:** reconstruct from that initial prior, the permanent visited mask,
  and only observations with transition indices `max(0,t-32)` through `t-1`.
- **recent8:** the same construction with indices `max(0,t-8)` through `t-1`.
- **initial:** initial-hit-conditioned prior and the permanent visited mask,
  with no subsequent odor likelihoods.

The initial hit is included exactly once. Reconstruct recent beliefs from the
initial prior, never from a full posterior at the start of the window. Apply
retained likelihoods at their recorded public arrival coordinates, exclude
known non-source cells and normalize. Do not floor probabilities, insert a
true source into support or silently repair invalid mass. Invalid or nonfinite
beliefs stop the run. All states reset at episode boundaries.

## Eight arms and fixed paired cohort

Use these arms in this base order:

1. `space_full`
2. `space_recent32`
3. `space_recent8`
4. `space_initial`
5. `info_full`
6. `info_recent32`
7. `info_recent8`
8. `info_initial`

`space` uses upstream space-aware infotaxis, policy 1; `info` uses upstream
infotaxis, policy 0. Both use one-step enumeration, unchanged objectives and
the upstream first-action-within-tolerance tie rule. No tuned lookahead,
additional policy family or outcome-based policy selection.

There are eight blocks `b=0..7`. Each block contains four cases in each initial-hit
category 1, 2 and 3. For `hit` in `{1,2,3}` and `j=0..3`:

```text
case_index = b*12 + (hit-1)*4 + j
seed = 610001 + case_index
```

All eight arms run all **96 cases, 768 episodes**. Rotate the base arm order
left by `case_index % 8`. Use separate reproducible source-draw and observation
channels, paired across arms within each case. Pin their derivation in the
runner/plan. Setting NumPy's global seed alone is insufficient because upstream
constructs fresh unseeded `RandomState` objects. Qualify the explicit stream
integration. Different actions can produce different observations even under
the same paired random inputs. No replacement seeds or adaptive extra cases.

Initial-hit strata are equally represented for coverage, but aggregate metrics
use the exact **upstream initial-hit probability formula**. For each
`h in {1,2,3}`, compute its unnormalized mass as upstream `_initial_hit` does:

```text
r = arange(1, int(1000 * lambda_over_dx))
mass[h] = max(0, sum(PoissonCategory(mean_hits(r), h)
                    * (volume_ball(r+0.5) - volume_ball(r-0.5))))
weight[h] = mass[h] / (mass[1] + mass[2] + mass[3])
```

Use upstream sensor and volume functions and retain full floating-point weights
in the output. Apply these weights to stratum means, including within each
twelve-case block. The aggregate is not the unweighted mean of the balanced
cohort. Report all three strata, all blocks and every episode.

## Qualification before the cohort

Run structural qualification first. The scientific cohort starts only after
all declared qualification cases pass. Use separate fixture identifiers and
random streams; no qualification outcome selects cohort seeds or controllers.
Qualification is bounded to **2,000 native steps** within the total allocation.
It must establish:

- Expected 53x53/four-category geometry, action IDs and legal boundary moves;
  all three initial-hit priors and correct saturated-tail likelihoods.
- Source-found termination, terminal-hit handling, reset separation, and no
  action after termination. Source discovery on step 2,188 counts as success.
- Repeated seeded replay and arm-order-independent source/observation channels;
  no unseeded draws escape the declared random-stream integration.
- Public-filter normalization, finite nonnegative probabilities, permanent
  non-found cell exclusion, exact recent-window boundaries and no double use
  of the initial hit. Full public/native posterior maximum absolute difference
  must be at most `1e-10`; deterministic replay must preserve public outputs
  and selected actions exactly.
- Isolation of source and simulator posterior from actor inputs, including
  during heuristic lookahead; private hypothetical beliefs cannot mutate the
  real actor or another action branch. Both heuristic implementations and
  their tie handling must match the pinned source on qualified fixtures.
- Complete attempted/returned move accounting and cumulative resource checks.

Record each qualification result before the first cohort episode. Passing
qualification establishes interface correctness, not controller effectiveness.

## Metrics, costs and official-evaluator distinction

For each sampled-source episode record actual moves, success, initial hit,
case/block/arm identity, and capped time `min(T,2188)`. A source not found by the
horizon incurs **2,188**, not its last successful prefix length. Log
`agent_stuck` as a diagnostic; do not terminate early on that flag. Report
weighted mean capped search time and weighted failure probability. Successful-
episode mean time is descriptive only and must retain its success denominator.

The official evaluator instead requires `draw_source=False`, accumulates
survival-weighted expected time, stops using probability/stuck conditions, and
normalizes its found-time PDF by found mass. This sampled-source pilot is **not
a reproduction of its estimator or published score**. Do not equate its
probability-weighted failures with this pilot's sampled failure fraction.

Retain per-step public action/observation, action scores, posterior hash,
filter-assimilation/reconstruction time and policy time. Measure public
inference from packet assimilation through action selection, including ledger
updates, belief reconstruction and any per-actor initialization. Record cold
initialization separately and include it in total controller cost. Each actor
constructs its own likelihood table, included in that initialization cost.
Each simulator also constructs a table; simulator construction is included in
whole-run time. Record stepping time separately, including its resource-check
wrapper overhead. Evaluator-only parity checks and trace writing are excluded
from controller time and included in whole-run time. Do not describe replay-from-scratch cost as
optimized streaming deployment, or equal action enumeration as equal compute.

## Prospective opportunity rule

The primary controller is **space-aware infotaxis**. All of these requirements
must pass, using unrounded weighted metrics and no numerical allowance:

1. `space_full` weighted success is at least **95%**.
2. Its mean capped time is at least **10% lower** than `space_recent32`.
3. The same absolute improvement is at least **two primitive steps**.
4. The weighted block time improvement is strictly positive in at least **6/8**
   fixed blocks. Ties are not wins.
5. Its weighted failure probability is no higher than `space_recent32`.
6. Its weighted failure probability is no higher than `space_recent8`.
7. Its mean capped time is strictly lower than `space_recent8`.
8. Its mean capped time is strictly lower than `space_initial`.
9. All **768** episodes and their action/process accounting are complete.
10. Every required structural qualification check passes.

For requirement 2, improvement is `(recent32_mean - full_mean) / recent32_mean`;
the denominator must be positive. Publish the corresponding infotaxis
comparisons descriptively without changing the primary controller. These
practical thresholds are exploratory, not statistical significance tests.

One supervised run has limits of **1,800 suspend-inclusive seconds, 4 GiB peak
RSS, 512 MiB output and 1,690,000 native steps**, including qualification. The
complete cohort's worst-case move count is 1,680,384; qualification adds at most
2,000. A resource or qualification failure stops the run and preserves partial
artifacts. No extensions, selective scoring, substituted cases or threshold
changes follow a failure. Successful worker completion and a successful actual
supervisor terminal are both required before saved-output scoring is accepted.

A pass establishes a classical odor-memory opportunity on this development
cohort only. Any later learned comparison needs separately frozen training,
evaluation and shift cohorts, GRU/recent-history/classical references and total
compute accounting. Failure means this fixed task/control screen does not admit
that learning pilot. Neither outcome establishes biological-wiring superiority.

## Exact differences from the completed 19x19 pilot

The same frozen `otto_public.py` and `otto_memory.py` modules are reused without
edits. The runner is a separate file, so the small-study implementation,
receipts and failed continuation rule remain reproducible. The sensor kernel
now uses `mu(d) = 2 * K0(d/3) / log(6)` at nonzero distance. Initial-hit weights
use radii 1 through 2,999 and all three positive hit strata. The last category
is the Poisson tail for at least three detections. Geometry, likelihoods,
initial priors and action semantics must pass native checks for this setting.

The horizon 2,188 is the upstream evaluator's automatic horizon formula for
these physical parameters, used here as a fixed sampled-source cap. Its use does
not make the sampled-source metric equal to the official evaluator's metric.

Qualification uses seeds 420001 and 420011-420013, distinct from the cohort.
The repeated/forced-hit replay takes 28 north moves, then 28 west moves, then
moves directly to the evaluator-known sampled source; it is a semantics fixture,
not a public policy. Each replay has at most 256 steps. A separate source at
(52,52) checks a forced category-3 reading, blocked boundary movement and found
termination. Every actual native step, including replay references, counts
against the 2,000-step qualification and 1,690,000-step total limits. No model
fits, larger search allocation or additional scenarios are admitted by this
protocol itself.
