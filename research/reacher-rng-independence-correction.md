# Reacher correction: random streams were not independent across roles

Status: protocol defect confirmed on 2026-09-18. The reward-residual v1 attempt
was interrupted before completion. Its partial evaluation cannot support an
efficacy or continuation claim. The earlier world-model v1 records remain
unchanged, but their claim of independent hidden disturbances and evaluation
streams requires this correction.

This diagnosis used frozen source code, random-number arithmetic and terminal
receipts. No model, planner, simulator or new-data calls were made. No partial
reward-residual performance arrays, plots or metrics were inspected to decide
whether to stop or retain checkpoints.

## The collision

Both runners separated base seeds by 100,000, then added offsets of 100,000 or
200,000 for prediction and control. Those additions erased the apparent
separation. The actual generator seeds overlap:

| Concrete role | World-model v1 | Reward-residual v1 |
|---|---:|---:|
| Control actuator noise, episode `i` | `64,700,001 + i` | `66,700,001 + i` |
| Prediction exploration, episode `i` | `64,700,001 + i` | `66,700,001 + i` |
| Candidate bank, decision `t` | `64,700,001 + t` | `66,700,001 + t` |
| Uniform-action reference | `64,900,001` | `66,900,001` |
| Bootstrap resampling | `64,900,001` | `66,900,001` |

World-model v1 has two further cross-role collisions: prediction actuator noise
uses `64,600,001 + i`, also the training exploration seed; prediction sensor
schedule uses `64,500,001 + i`, also the training actuator-noise seed. Training
and control schedules also reuse their phase seeds. Reward-residual v1 copies
the original training files; it does not regenerate training with its new base
seeds, so those two training-to-prediction collisions are not inherited by its
fresh prediction cohort.

The responsible frozen code is in
[the original runner](../scripts/reacher_world_model_study.py),
[the residual runner](../scripts/reacher_reward_residual_study.py) and
[the environment wrapper](../src/openjev/research/robotics_reacher.py).
`numpy.random.default_rng` initialized with the same integer has the same
initial stream. Different array shapes, scales or calls do not establish
independence.

There is a concrete counterexample despite the overwritten candidate anchors.
At control decision zero, the 12-step horizon and three-step action blocks use
four Gaussian pairs per candidate. Candidates 0 through 6 are overwritten;
candidate 7 is retained. In case 0, its four raw pairs are the same standardized
Gaussian values as episode 0's hidden actuator noise at steps 28 through 31.
The candidate standard deviation is .25 and the actuator standard deviation
is .05. Therefore:

```text
candidate_bank[case=0, candidate=7, horizon_step=[0, 3, 6, 9]]
    == float32(clip(5 * actuator_noise[episode=0, step=28:32], -1, 1))
```

A pure RNG check at seed `66,700,001` found maximum raw numerical difference
`1.39e-17` and exactly zero difference after candidate clipping and float32
conversion. At decision `t=i`, candidate-bank randomness also shares episode
`i`'s disturbance stream, creating cross-case dependence. This does not mean
every candidate or decision contains its own next disturbance. It is sufficient
to falsify the intended independent-stream construction.

The exploration collector first draws its mixture mode, changing consumption
but not the underlying seed. For the same illustrative seed, its first two
standardized exploration values equal flattened actuator-noise values 1 and 2.
The uniform/bootstrap overlap uses different draw operations, but still reuses
the generator rather than defining an independent stream. None of these facts
establishes the direction or magnitude of performance bias.

The residual auditor checked freshness within each named role relative to the
parent experiment. It did not compare concrete streams across different roles
within an experiment. That validation was insufficient. Successful native replay
checks reproduction of recorded physics and noise, not statistical independence
of those noise streams from other experiment inputs.

## Terminal state and what remains usable

The stopped attempt's frozen plan is
[reward-residual v1](../evidence/reacher-reward-residual-v1/protocol/plan.json),
SHA-256 `df9929ea6320e24ba32ec8e8fdd84f88f8435ed03db1bea655ec31d82345e0fd`.
The orchestrating process reported SIGINT termination with exit code 130. Its
local `runs/reacher-reward-residual-v1/execution/failed.json` records:

- `KeyboardInterrupt()` after **359.1389642080758 seconds**.
- Phase `control`, ordinary panel, particle reference, step 40.
- Failure-receipt SHA-256
  `83b3eb55e6d39226f3ce8b03d55068444d947a634bbd023318aedfc30784839e`.

There is no execution `completed.json` or successful audit receipt. All six
fits had finished before evaluation, as recorded by `all-fits-completed.json`:
`free-271`, `residual-271`, `residual-283`, `free-283`, `free-293`,
`residual-293`. That boundary's six fit-receipt hashes were checked. Its elapsed
time, 210.28938762494363 seconds, includes pre-fit preparation and is not a
standalone training-cost total. These are completion/provenance facts, not
quality results.

The **original 768 training episodes remain reusable**. Within that training
cohort, reset, schedule, exploration and actuator-noise streams occupy disjoint
seed ranges. The demonstrated defect couples those streams to other roles in
old evaluation, not to one another within training. The residual fits used that
unchanged corpus and completed before evaluation. No evaluation feedback selected
or trained them. Their paired final checkpoints can therefore be evaluated in
a separately declared experiment without claiming that the stopped v1 trial
completed successfully.

Preserve both v1 protocols, frozen implementations, audit records, checkpoints,
partial outputs and failure receipts. Recorded old metrics are still descriptive
of their actual execution, but are not evidence from the intended independent
hidden-disturbance benchmark. Do not use the stopped trial's partial metrics
for promotion, seed replacement, model selection or adjustment of thresholds.

## Required stream contract for a new evaluation

Assign random generators by **role and concrete use**, rather than by arithmetic
offsets of nearby base seeds. The new
[stream helper](../src/openjev/research/reacher_random_streams.py) hashes the
study namespace and each role into distinct 48-bit seed blocks, reserving
`2**20` values per role for the legacy helpers' offsets. It expands the actual
episode/decision allocations and rejects collisions in both integer seeds and
initial generator states, including the particle filter's spawned children.
Nine regression checks reproduce the old defect and exercise the new checks.

The [v2 engineering preview](../evidence/reacher-reward-residual-control-v2-preflight/seed-preview.json)
contains 692 root allocations and 820 actual generator identities, with no
overlap against the original training or either old evaluation allocation.
Constructing this manifest draws no random samples. This is an implementation
preflight, not a frozen evaluation or efficacy result; the exact source and
manifest must still be bound into the new protocol before scored calls.

The concrete manifest must cover:

- Native Gymnasium/NumPy reset RNG, including random initial state and target.
- Hidden actuator-noise generator for every episode.
- Exploration mixture choice and held exploratory commands.
- Sensor blackout phase/schedule draws.
- Candidate banks at every decision, including all cases represented by a bank.
- Particle-filter initial particles, modeled process noise and resampling;
  its three spawned child streams must remain separately identified.
- Uniform-reference actions.
- Bootstrap resampling.

Check the complete set for collisions **across roles, cohorts and all allocated
case/decision indices**, including concrete derived integers and initial bit
generator identities, not merely top-level names or base seeds. New evaluation
streams must also be disjoint from inherited training and both earlier evaluation
allocations. If no random draws occur for a role, record that explicitly instead
of inventing evidence of a consumed stream. Add regression tests reproducing the
old collision and rejecting it before environment, model or planner execution.

Intentional pairing is an explicit exception: control arms and sensing panels
share the same case reset/disturbance streams and candidate banks; ordinary and
shifted schedules may share the declared phase draw. All fits see the same
prediction corpus. Bootstrap draws may be shared across declared paired
comparisons. Each intentional reuse must name the common experimental unit;
none permits actuator noise to share a stream with a candidate bank, exploratory
command or observation schedule. A hash or seed manifest guards against accidental
reuse; it does not prove mathematical independence of finite pseudorandom streams.

## Proposed v2: evaluate every existing final fit

Freeze a new **evaluation-only recovery** with all six v1 final checkpoints,
their exact hashes, paired initialization/order evidence, residual-mode settings
and original training provenance. No retraining, warm start, alternative epoch,
replacement seed, model pruning or selection is part of this recovery. The
choice to retain every fit is based on the completed-fit boundary and the
identified protocol defect, not its partial performance.

The [fit-inheritance receipt](../evidence/reacher-reward-residual-v1/fit-inheritance.json)
authenticates all six final fits, paired initial tensors and minibatch orders.
It reads no held-out efficacy metrics. Their summed fitting wall time is
209.1519727089908 seconds, already included in the stopped attempt's total.
The [invalid-attempt receipt](../evidence/reacher-reward-residual-v1/invalid.json)
and [original failure receipt](../evidence/reacher-reward-residual-v1/execution-failed.json)
preserve its terminal state.

Generate a fresh common prediction cohort and paired control cases using the
new stream contract. Retain both arms, all three seeds, all sensing panels,
reset diagnostics, reference controllers, candidate-budget matching and the
seventeen previously stated useful-effect criteria. Freeze exact inputs,
source/runtime hashes, reporting rules, cap and stopping policy before evaluating
any recovered checkpoint. The source-bound audit must verify the stream manifest
as well as records, arithmetic, full model coverage and native replay.

Report inherited fitting cost separately from the new evaluation's preparation,
prediction, planning/control and audit costs. Also retain the 359.14-second
invalid v1 attempt as prior-attempt cost; do not double-count its included fitting
time when presenting totals. A new evaluation cannot be represented as a cheaper
end-to-end training result by omitting inherited work.

This is a new prospective evaluation of already trained models. It does **not**
resume, retry or rescue the old frozen no-resume attempt, which remains stopped
and invalid for efficacy. A future completed v2 can support only its own stated
comparison and must still satisfy its independent memory-competence requirements
before a connectome claim is considered.
