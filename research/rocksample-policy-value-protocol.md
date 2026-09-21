# RockSample classical policy-value screen

21 September 2026. This separate experiment asks whether old public information
improves task return under a shared classical controller. The preceding
prediction diagnostic remains **FAIL 4/6** and does not admit a learned model.
No architecture is trained here. A passing control screen would be new evidence
about useful memory, not a retroactive pass or a novelty claim.

## Task and cohort

Use unchanged POBAX `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`, raw
RockSample(11,11) with its native TimeLimit wrapper and 1,000-step horizon,
Gymnax 0.0.9/JAX 0.6.2 isolated CPU runtime. Eight fresh constructor seeds
12001-12008, each with four reset seeds 22001-22004. All six arms complete all 32
cases once. The policy's action histories differ; pairing initial/transition
random keys is not a claim of identical histories. No replacement seeds or
outcome-dependent extensions.

Actor inputs are public one-hot position, signed check readings, past actions,
remaining step/check budget and the shared sampled-cell ledger. No public arm
receives realized reward, map coordinates, hidden qualities or constructor seeds.
Task constants and a fixed random bank of hypothetical maps are allowed.
Sampling a cell guarantees it cannot subsequently provide positive reward;
every arm retains that fact indefinitely. Map beliefs reset between episodes,
even though the simulator constructor map persists.

## Six arms

1. **exit:** direct east exit, raw-return reference.
2. **full:** complete public history in a finite mixture of 256 distinct-map
   hypotheses, sampled from a fixed prior using seed 530001. Conditional qualities
   use exact Bernoulli updates within each map. Map weights use each check's
   pre-update likelihood. No resampling or proposal adaptation.
3. **recent128:** same hypothesis bank and filtering rules, retaining checks from
   the most recent 128 primitive transitions. Replay every sample chronologically,
   including older samples, so it never forgets public depletion.
4. **latest:** same bank, retaining only the newest check of each rock and every
   sample, replayed in chronological order.
5. **quality:** 11 initial-quality probabilities and the same depletion ledger.
   Location remains uniform; likelihoods average the sensor rule over the grid
   and known depletion. It discards location-quality correlations and is an
   explicitly approximate, location-blind control.
6. **privileged:** one hypothesis containing the true map and initial qualities,
   updated through known depletion, with the same planner. This additional
   information is an explicitly labeled planner-capacity reference, not a fair
   public-input baseline or an optimal oracle.

The finite map mixture respects location exclusion but remains approximate:
a high effective sample size does not prove coverage of the actual unknown map.
Record ESS, maximum weight and surviving support at every planning decision.
An inadequate fixed particle bank cannot be enlarged after observing outcomes.

## Shared finite-horizon planner

Plan **undiscounted native return**, accounting for every primitive move, sample,
check and eventual exit. Report discounted return at 0.99 only as a secondary
metric. Checks/movement have no native reward penalty. All non-exit arms share:

- Four fixed reflected row-serpentine exploitation tours over the public board.
  Ignore already sampled and nonpositive-expected-reward targets; choose the best
  feasible prefix including eventual exit. Coordinate value is 10 times total
  good-location mass minus bad-location mass, never quality confidence alone.
- Direct exit competes with exploitation. An exploitation decision commits only
  to its first move-to-cell-and-sample macro, then replans.
- Sensing at the current coordinate and four interior corners. At each distinct
  feasible vantage, rank rocks by predictive entropy, ties by index. For the
  top four indices (a,b,c,d), examine exactly [a], [a,b], [a,b,c,d], [a,a], and
  [a,a,a,a]. Enumerate sequential predictive outcomes exactly under the chosen
  belief approximation; evaluate the same exploitation/exit continuation at
  every leaf. Maximum 25 bundles and 210 leaves before feasibility pruning.
- Sensing must improve expected return by strictly more than 0.05. Non-sensing
  wins ties. Commit the selected move-and-check bundle, then replan.
- A shared maximum of 64 checks per episode; no compulsory sensing quota.
  Remaining task time and sensing allowance constrain every macro.

Equal candidate allocations are not equal actual computation. Measure filter
reconstruction/assimilation and planning time, primitive steps, plans, branches,
samples, positive/negative samples, exits and timeouts. Record all raw transitions
and decisions. Never interpret from-scratch replay cost as an optimized deployment
implementation. No speed superiority claim is targeted by this screen.

## Qualification, acceptance and limits

Before native scoring, qualify exact tiny-map filtering and sequential branch
probabilities, sample depletion without observing reward, clone independence,
zero-probability branches, primitive horizon/exit accounting, diffuse occupancy,
known-good exploitation, redundant sensing, multi-check value and the common
ledger beyond 128 transitions. A bounded synthetic timing check can decide whether
the fixed cohort fits its allocation; it cannot establish task efficacy.

All continuation requirements must hold, using equal-map means:

- Full raw return exceeds each of recent128, latest and quality by at least 5
  reward units, and beats each on at least 6/8 paired map means.
- Full exceeds direct exit by at least 5 reward units.
- The privileged reference exceeds direct exit by at least 20 reward units,
  establishing that this planner can collect useful rewards when informed.
- At least 90% of full-arm planning decisions retain ESS of at least 8. This is a
  predeclared finite-particle adequacy limit, not proof of posterior correctness.
- All32 cases per arm complete with valid accounting, no illegal action,
  unhandled impossible observation or unreturned transition.

These practical thresholds are exploratory and do not establish statistical
significance. If quality-only or latest-reading memory explains the gain, no
location-inference or larger recurrent-model claim follows. Failure of privileged
headroom is a controller limitation, not proof that history has no value. No
learning pilot follows a failed control screen.

One fixed native run: 1,800 suspend-inclusive seconds, 8 GiB peak RSS, 512 MiB output,
at most 192,000 primitive transitions. Engineering qualification has its own
separate 600-second allocation and uses declared synthetic or injected cases.
Retain incomplete/failed outputs. Reporter/auditor use only saved artifacts.
