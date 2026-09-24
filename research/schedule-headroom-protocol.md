# Public-history schedule headroom diagnostic

This separately registered diagnostic follows the reliability-memory failure
at 8/13 conditions. It changes no previous result and trains no model. Its
purpose is to decide whether a learned approximation to a richer public-history
filter is worth pursuing, or whether the learned world model is the bottleneck.
An exact finite mixture is established Bayesian inference, not a new architecture.

## Population and information

Generate five fresh independent datasets, namespaces 439260924 through 439260928.
Each contains 512 independently generated episodes, with 32 actions plus reset.
Retain all episodes, including early found events. The physical transition,
destination hazard and centered decision costs are the unchanged eight-state
finite world. Every episode independently draws one of four equally likely
families: static noise 0.12, static 0.30, a change 0.12 to 0.30, or the reverse.
For either change, the first changed event is uniform on indices 8 through 24.

Per-case SeedSequence([namespace, case index]) spawns four PCG64 streams in
order: episode actions, fork actions, schedule, event draws. Precommit all 32
public actions and each boundary's eight fork actions before any events. Sample
events from the path-conditional state mixture; no hidden physical-state labels
are sampled or provided to models. Reset is an odor under a uniform state prior,
without a transition or found hazard. Include the first found in event loss;
thereafter labels are found, predictive law is one-hot found and costs are zero.

An exact filter receives a fixed prior over 36 complete schedules: two static
schedules with probability 1/4 each, and 17 change locations per direction,
each with probability 1/68. It conditions joint schedule/state mass on public
actions and observations only. Time is available, but case ID, family, actual
switch time, future observations and the realized private noise path are not.
The supplied prior is correct for this new population. It is explicit additional
diagnostic knowledge: the old trainable controls were fitted on 0.12/0.48 noise.
This is not a fair architecture comparison or robustness to an unknown prior.

## Ten fixed references

For every cohort, reuse its original frozen learned T/O/h/C backbone and all
six original final reliability models in order, with no checkpoint selection:
unchanged, global scalar, static three-mode bank, Markov bank, recurrent bank,
and GRU-reset bank. The parent registration and publication are pinned, and
all 35 original backbone/model files are copied byte-for-byte before the first
new scientific episode is generated. No optimizer or training call is allowed.

Add four references:

- `learned_exact`: the 36-schedule filter with that cohort's learned fields.
- `true_exact`: the same 36-schedule prior with supplied true world laws.
- `learned_static2`: a two-static-schedule filter at 0.12/0.30 with equal prior
  and the learned fields. It cannot represent a change.
- `true_static2`: the same two-static-schedule approximation with true laws.

For learned emissions use O(epsilon)=(1-q)O_base+q/4, where
q=(epsilon-0.12)/0.63. This supplied corruption family is unchanged from the
parent mechanism. The true-law version equals the actual emission matrix.
Maintain correlations between schedule hypotheses and state; average conditional
action costs before selecting an action. Forks marginalize schedule and propagate
the state with surviving transition matrices only, without any new observations.

Exact filters store 288 joint-mass entries, versus 16 for static-two, 8 for
unchanged/global, 24 for old banks and 28 for GRU banks. These counts exclude
static model tables, terminal flags and temporary work. Same original stored
size does not imply equal effective capacity: the GRU-reset control's 48 hidden
matrix weights always multiply zero, while its Bayesian posterior persists.

## Estimand and interpretation

The private-path reference conditions on public history and the actual realized
noise schedule. For each boundary t=8..32 and each blind horizon H4/H8, compute
its conditional expected cost for the model-selected action minus its minimum
conditional expected cost. Sum these 50 nonnegative regrets within an episode
and divide by **50 regardless of survival**. Absorbed boundaries contribute zero.
Average over all 512 episodes, then equally over the five cohorts.

This fixed-denominator additive endpoint differs deliberately from the parent's
average over alive boundaries. Its loss at a boundary does not depend on future
episode survival. The true-law public-history filter minimizes expected cost
for this endpoint under the declared population prior, in expectation. It need
not beat another rule on every finite sample, cohort or conditional family.
The private-path oracle has extra information and remains a privileged risk
reference, not an attainable public-information floor.

The learned exact filter performs exact inference in its specified learned
model; that model may be wrong. Its contrast with true-law filtering combines
transition, hazard, emission and cost-readout mismatch. It does not isolate
physics, identify learned coordinates with true latent states, or prove that
any single learned component caused the gap.

Report primary regret, separate H4/H8 regret, immediate post-observation regret
with divisor 25, actual-event NLL including first found, event and found counts,
every per-episode primary regret, and all four family means/support counts.
Require at least 64 episodes per family in every dataset; otherwise fail the
diagnostic without replacing data. Histories use random actions. Counterfactual
decisions do not change those histories, so this is not autonomous control,
gameplay, robotics transfer or an RL result. No calibration or significance
claim follows from these regret measurements.

## Precommitted interpretation rule

For each of `learned_exact` and `true_exact`, compare with every one of the six
frozen controls and `learned_static2`. Each candidate has 15 conditions:

1. Positive equal-cohort control regret and at least 10% lower candidate regret
   for each of the seven comparisons: seven conditions.
2. Strict improvement in at least four of five paired cohorts for each of those
   seven comparisons: seven conditions.
3. On static-0.12 family alone, candidate equal-cohort mean regret no greater
   than 1.05 times unchanged plus 1e-6: one preservation condition. This is an
   application requirement, not a consequence of Bayes optimality for the mixture.

Separately compare `true_exact` to `true_static2` using the same 10% mean and
four-of-five cohort rules: two history-opportunity conditions. Use the following
fixed order, never promote an arm or relax a threshold after results:

- If learned exact passes all 15, label `LEARNED_MODEL_HEADROOM`. A future
  approximation study may be worth designing with matched prior/training access.
- Otherwise, if true exact passes all 15 and both history conditions, label
  `MODEL_MISMATCH_HEADROOM`. Investigate the learned world model before another
  reliability GRU.
- Otherwise, if true exact passes all 15, label
  `TRUE_MODEL_ADVANTAGE_WITHOUT_SCHEDULE_MEMORY`. Supplied world laws help, but
  the required history-specific benefit over true static-two is not established.
- Otherwise label `NO_REGISTERED_HEADROOM` and stop this adaptation line for
  this population, history length and requirements. A private-oracle gap alone
  cannot admit another architecture attempt.

These are pilot screens conditional on five fixed trained backbones, not
statistical hypothesis tests, novel architecture claims or a revised verdict
on the earlier failure. All ten arms, 50 rows and 32 conditions remain visible.

## Execution and independent checks

Develop and peer-review tests using fabricated fixtures and separate engineering
namespaces. Freeze the protocol, full source closure, parent input hashes and
runtime in an exclusive registration. Qualification runs all four new test files
and lint, then 64 episodes at namespace 949599 through all ten references using
parent cohort zero. Save the complete probe but score no effectiveness metric.
Twice the observed generation-plus-inference time scaled by 5*(512/64) must
be below 450 seconds, half the scientific cap. This projection excludes output
serialization; the remaining cap is reserved for serialization and validation.

Original native caps are qualification 180 seconds, scientific run 900 seconds,
and independent audit 300 seconds. Complete inference calls are individually
bounded below 120 seconds. Timings cover filtering plus all eight blind-fork
steps. They exclude model construction, conversions around torch, persistence
and metric calculations; native phase times include surrounding work. These
are measured calls on one machine, not equal FLOPs or matched-compute training.
Rotate the ten-arm evaluation order by cohort. Preserve every output and closure;
no restart, seed replacement, budget extension or frozen-source edit is allowed.

The independent audit imports no producer, learner or world generator. It
reconstructs true private-path targets, verifies copied fields/model buffers,
recomputes the four exact-reference outputs from public evidence, checks saved
fork values, and calculates all metrics/conditions independently. It does not
replay old neural gates, sampling, optimization or training. Exact-reference
reconstruction is explicitly an audit computation, not a second scientific run.
Publication follows clean original closure of qualification, run and audit.

Established prior work already combines uncertainty and recurrence, including
[Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html)
and [Switching Recurrent Kalman Networks](https://arxiv.org/abs/2111.08291).
The present diagnostic supplies finite hypotheses rather than proposing a new
neural implementation of those papers. Any later mechanism must earn a benefit
over strong controls with the same disclosed information.
