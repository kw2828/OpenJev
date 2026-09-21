# Autonomous compact odor memory on fresh paired searches

21 September 2026. Prospective test of the fixed, untrained DCT memory from the
[saved-path comparison](otto-spectral-memory-results.md). That comparison found
better imitation on long histories but worse case-weighted imitation across all
histories, numerical-fill sensitivity and greater computation than full Bayes.
This study tests the missing question: what happens when each controller chooses
its own actions? It does not revise the original memory-opportunity failure,
train a model or establish biological, recurrent-world-model or ICLR novelty.

## Six unchanged actors, two fixed environments

Use the pinned upstream OTTO source and public seeded adapter already qualified
in the [53x53 study](otto-large-memory-results.md). Keep two dimensions, Euclidean
sensing, an explicit **53x53 grid**, **four hit categories**, **R_dt=2** and a
**2,188-move horizon** in both environments. The baseline sensing length is
**lambda_over_dx=3**; the shifted length is **4**. The latter is a fixed-grid
sensing shift, not the automatically sized upstream lambda4 benchmark. Saturated
category 3 includes three or more detections.

Rebuild the supplied kernel, conditioned priors and initial-hit mixture separately
for each environment. Every actor knows the applicable observation model; this
is not sensor-model identification or robustness to an unknown parameter. Preserve
the actual numerical kernel, including cancellation zeros, without floors or
analytic-tail substitutions. Hold the score function, tie rule, grid, action costs
and horizon fixed. Do not choose a shift after inspecting new outcomes.

Run all six arms:

1. `full_bayes`: full history in probability space.
2. `exact_log`: full-grid additive log evidence.
3. `recent32`: initial prior, permanent visited ledger and latest 32 readings.
4. `recent32_hard`: latest 32 soft readings plus every historical hard exclusion.
5. `dct16_neutral`: 256 coefficients with zero-log finite extension.
6. `dct16_nearest`: the same rank with category-specific distance-one extension.

Reuse the previously frozen actor implementations. A compressed actor retains
its coefficients and one persistent hard-support mask, without a dense posterior
cache or old-observation list. The exact prior is included once at decoding.
All controllers use the same pinned space-aware action-score mathematics and
first-action-within-1e-10 tie rule. No learned gate, fallback policy, controller
selection or rank tuning is introduced.

Actors receive only the current public position, hit, step, terminal flag and
legal actions, plus the supplied model. Sampled source, simulator posterior,
random streams and seed IDs remain evaluator-only. Apply every nonterminal
observation, including a final horizon-censored reading. Never encode the found
sentinel as odor or request another action after finding the source. Reset every
actor and random stream for every episode.

## Fresh paired cohorts

Each environment has **96 cases**, grouped into eight blocks. Each block contains
four cases in each initial-hit category. For block `b=0..7`, hit `h=1..3` and
replicate `j=0..3`:

```text
case = 12*b + 4*(h-1) + j
baseline seed = 630001 + case
shifted seed = 640001 + case
```

All six actors run every case: **1,152 autonomous episodes**. Rotate arm order
left by `(environment_index*96 + case) % 6`. Seeds pair source locations and
the public adapter's random-number channels within each environment. Different
actions generally produce different observations; no fixed-path forcing occurs.
The numeric seed ranges occur in unrelated robot protocols, but have not been
used for prior OTTO cases. They are fresh for this environment and controller
comparison, not a claim of globally unique integer seeds.

There are no replacement seeds, retries, extra episodes or outcome-based
subset selection. Success means reaching the sampled source. Every unsuccessful
episode runs to the horizon and contributes **2,188 moves** to capped search time.
Do not average search time only over successful episodes. Preserve every episode,
transition, source-pairing record and resource failure.

Aggregate each initial-hit stratum equally over its 32 cases, then apply that
environment's qualified initial-hit mixture. Compute each block similarly from
its four cases per stratum. Preserve unweighted counts, all three strata, all
eight blocks and all paired episode differences. Never pool environments or
finite fills to rescue a failed requirement.

## Prospective decisions

The following are practical engineering tolerances informed by the earlier
development comparison, not statistical noninferiority margins or novelty
criteria. For **each DCT fill in each environment**, require all of:

- Full Bayes weighted success is at least 95%, and the candidate's weighted
  success is no lower than either full Bayes or exact additive memory.
- Candidate weighted capped time is at most **105% of each full/exact reference**.
- Candidate weighted capped time is strictly lower than **both recent controls**,
  with a positive weighted paired gain against `recent32_hard` in at least
  **six of the eight blocks**.
- Candidate evolving array payload, including its support mask, is at most
  **20% of full Bayes**, and its complete mean controller cost is at most
  **150% of full Bayes**.

Call compact autonomous control viable only if all four fill/environment
combinations pass. Even that does not show a utility-versus-compute improvement:
the tolerance allows modestly worse utility and more computation in exchange
for less evolving state.

Therefore separately report **utility_compute_advantage**. For each fill in
both environments, require capped time and complete controller cost each no
higher than full Bayes, at least one strictly lower, and no success regression.
The overall flag requires both fills to satisfy both environments. No favorable
choice of fill, environment, scalarized reward or weighting may replace this rule.
Report every component and both flags, including failures. Neither flag admits
a learned campaign or establishes an architecture contribution by itself.

## Cost and numerical qualification

Time actor initialization, observation updates, full-grid decoding and action
planning separately. Include all four in complete controller cost. Measure
shared spectral-model construction once per environment and charge its time
divided by 96 to every exact-log and DCT episode, as if each controller processed
its own 96-case workload. Report that unamortized setup time too. Simulator and
public-kernel construction are separate from controller cost. Record both whole
episode and per-decision costs; actor order is rotated, but this remains one
descriptive timing pass on a shared CPU rather than a deployment benchmark.

Count evolving arrays, local immutable priors, shared model/kernel tables and
observation rings. Disclose dense transform and planning workspaces and Python
overhead exclusions. Whole-process RSS includes every actor and reporting object;
do not present the 20% state condition as an 80% total-memory saving.

Before any cohort episodes, qualify both regimes using separate fixed seeds:
650001 and 650010+h for baseline, 660001 and 660010+h for shift. Check geometry,
all numerical likelihood categories, the initial-hit mixture and each conditioned
prior against independent formulas within declared numerical tolerances. Check
seeded source/hit replay, public chronology, boundary moves, zero/saturated
readings, terminal behavior and unchanged global NumPy RNG. On these fixtures,
full Bayes, exact log and **both full-rank q=53 fills** must agree with native
filtering and the upstream action rule. Qualification actions or forced sources
are evaluator fixtures, never efficacy episodes. Maximum qualification allocation:
**2,048 native calls** across both regimes. Preserve incremental checks on failure.

During cohorts, compare full/exact decoded public beliefs against each actor's
own simulator belief as an evaluator diagnostic. Require finite normalized
beliefs, valid actions and causal public packets for all arms. No numerical
repair, replacement controller or source-aware action is permitted. Qualify
the new runner's summary weighting, caps, fixed decision rules, arm rotation,
terminal handling and publication boundaries with synthetic tests before launch.

## Freeze, execution and reporting

Bind the protocol, actor/runner/test/math/adapter/clock/supervisor sources,
upstream commit and source files, and isolated Python/NumPy/SciPy versions in a
committed plan before the first native invocation. One exclusive worker run
under the existing native suspend-inclusive supervisor. Limits are **900 seconds**,
**4 GiB RSS**, **512 MiB output**, and at most **2,522,624 native calls** including
qualification. The last limit equals 1,152 times 2,188 plus 2,048. The time and
output limits may stop a bad cohort before that worst-case call allocation.

Require both a completed worker receipt and an actual completed supervisor
terminal with exit zero and the process group gone. A timeout or failed
qualification remains failed with its partial outputs. No budget extension or
rerun under this protocol. A separately frozen technical repair may address
an implementation defect, without changing a scientific outcome.

A saved-output verifier must authenticate process/source/payload bindings and
independently recompute complete episode coverage, public movement/found joins,
recorded score-to-action choices, source pairing, component costs, per-regime
weighted outcomes, block contrasts and both fixed decisions. It must not execute
new episodes or import the actor/runner. Its statistical/reporting verification
does not independently establish posterior correctness or measured timing truth.
Publish all six arms under both environments, including failures and censoring.

For an illustrative replay, preselect **case zero in each environment**, with
all six arms shown. Render saved positions only, without new simulation or
outcome-based episode selection. Any subsampling of animation frames must be
disclosed; source markers are viewer-only annotations. An incomplete execution
does not earn a completed-comparison animation.

The original learned-memory admission remains false. This experiment addresses
autonomous consequences of an implemented representation. Architectural novelty,
biological learning, learned dynamics and transfer to physical robots remain
separate requirements of the broader research objective.
