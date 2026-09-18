# Reacher objective ablation: historical protocol draft

Superseded by the [frozen comparison](reacher-objective-ablation.md). The text
below records the proposal before the engineering checks and protocol freeze.

**UNFROZEN, UNRUN. September 18, 2026.** This document proposes a bounded next
experiment. No fitting seeds, generator states, loss multipliers or run budget
have been generated or selected. No new model has been trained or scored.
Numbers below are proposed settings, not an executable or frozen protocol.

The [completed search study](reacher-adaptive-search.md) supports CEM256 with
the existing residual GRU. The question here is whether a training objective
improves native control with that architecture and planner held fixed. This
does not test a connectome, establish a new architecture, or reproduce JEPA.
See the [learning roadmap](connectome-learning-program.md) for later stages.

## Fixed arms and training

Use `GRUResidualRewardWorldModel(hidden_size=64, dt=0.02, noise_std=0.05,
residual_reward=True)` for every arm. Inputs remain public angle cosine/sine,
target position, validity, measurement age and issued commands. Native reward
is a training target only. Velocities, applied-action noise and simulator state
never enter the model.

| Arm | Objective | Additional training state |
|---|---|---|
| Anchor | Existing `sequence_loss` | None |
| Raw | Anchor + fixed `lambda_raw * raw_endpoint_loss` | Hidden-to-hidden predictor |
| Latent | Anchor + fixed `lambda_latent * latent_consistency_loss` | Same-size predictor and frozen EMA teacher |

Reuse the authenticated 768 complete, 50-transition training episodes, copied
bit-for-bit with their native provenance. Bind the original corpus through the
[training protocol](../evidence/reacher-reward-residual-v1/protocol/plan.json),
its source records, and the
[corrected inheritance record](../evidence/reacher-reward-residual-control-v2/protocol/plan.json).
Do not warm-start from earlier fitted weights. Use three fresh fitting-seed
identities, giving nine fits. For each identity, construct one initial student
and copy its tensors into all three arms. Save canonical tensor hashes, not
just seed labels. Use a separate predictor generator and copy identical initial
predictor tensors into Raw and Latent. Teacher construction and predictor
initialization must not consume the student or minibatch generators.

Match the [existing training loop](../scripts/reacher_reward_residual_study.py):
48 epochs, batch size 32, 1,152 successful optimizer updates per fit, Adam at
0.001, no scheduler or checkpoint selection. Freeze Adam's remaining settings
explicitly. Save the same 48 epoch permutations for each paired triple and
verify their hashes. Interleave arm order cyclically across the three fitting
identities, independent of results. Use deterministic CPU float32 with the
pinned robotics runtime and two threads, subject to a synthetic capacity check.

Keep anchor settings unchanged: rollout horizon 5, rollout weight 0.5, reward
MSE weight 4, and the existing GRU-zero KL term/settings. Do not turn its
five-step loss into a seven-step loss. Clip the combined trainable parameter
gradient norm at 10 in every arm, rejecting nonfinite gradients. Report
backbone/predictor norms and clipping frequency because auxiliary-head
gradients affect that combined norm. There is no teacher in the optimizer and
the student's shared decoder appears in exactly one optimizer parameter group.

## Endpoint targets and training-only scale choice

Use the implemented [Raw](../src/openjev/research/reacher_raw_endpoint.py) and
[Latent](../src/openjev/research/reacher_latent_consistency.py) auxiliaries with
horizons `(1, 3, 7)`. Roots and endpoints must both contain observed public
measurements; intermediate missing angles are discarded. Prefix-valid masks
must prevent crossing episode boundaries, while retaining the final valid
endpoint. Average per-dimension MSE within each nonempty horizon, then average
those horizons equally. Complete training episodes keep the anchor's existing
mask semantics; synthetic padding fixtures cannot silently change that loss.

Seven transitions bridge six missing packets. Ten missing packets require
eleven transitions, which is deliberately absent from this first training
objective. Ten-gap evaluation tests generalization beyond its maximum span.
Adding horizon 11 would require a later matched Raw/Latent treatment, not a
repair after inspecting shifted results.

Raw adds a 64-to-64 predictor, then the existing observation decoder, targeting
four observed cosine/sine values. Latent targets the detached EMA teacher's
64-dimensional state after assimilation of the public endpoint. Both add 4,160
trainable predictor parameters. Their target scales, gradient paths and compute
differ; the raw decoder receives additional gradients.

Proposed scaling rule, to resolve before freeze: at initialization, use the
first four complete batches in the authenticated training-file order. For
each of the three paired initializations, measure separate anchor and unweighted
auxiliary gradient norms over the shared observation-update and transition
parameters only. Pool the 12 norms for each loss and set, for each auxiliary,
`lambda = clip(0.5 * median(anchor_norm) / median(aux_norm), 0.01, 100)`.
Use one resulting multiplier per objective across all three fits, fixed for all
updates. Reject zero or nonfinite norms; record any bound hit without retuning.
Calibration makes no optimizer or teacher update, clears all gradients, and
must leave initial tensors unchanged. Charge every calibration pass separately.
This balances initial backbone gradient scale approximately, not throughout
training. Freeze this derivation rule before calibration; record its numerical
outputs before fitting or drawing fresh evaluation data. No evaluation return,
old winning checkpoint, loss-weight sweep or seed selection enters this choice.

## EMA, checkpoints and actual restoration

Initialize the teacher as an exact copy of its paired student. Keep it in eval
mode with gradients disabled. Use momentum 0.99; variance and covariance
regularization weights remain zero. Teacher targets process public histories
under no-grad. After each successful optimizer step, update teacher parameters
exactly once and copy all buffers exactly. No EMA update occurs during
calibration, evaluation, failed batches or forward-only calls.

Save student initialization, initial predictor, final student, final auxiliary
state, final Adam state, generator states, epoch/batch cursor, successful-update
count and EMA-update count. An auxiliary checkpoint must contain the actual
EMA teacher; recreating a teacher from the final student is not restoration.
Save named optimizer-group membership and all scalar configuration alongside
tensors: model class, width, timestep, residual flag/noise scale, horizons,
EMA/regularizer/loss weights, anchor settings, clipping, runtime and source
hashes. Assert residual settings on both student and teacher; the current
latent compatibility check does not cover those Python scalars.

After all nine fits finish, discard in-memory models and reconstruct fresh
instances from saved configuration and tensors before evaluation. Strictly
validate tensor membership and canonical hashes. Prediction/control uses only
the restored student; teacher and predictor remain training artifacts. Test
complete optimizer/EMA restoration on synthetic fixtures, but do not allow
resuming, replacing or extending a failed scored attempt. Save such attempts
with their partial state and phase counters.

## Fixed planning and paired evaluation

Keep the completed CEM256 implementation and reward scoring fixed: four batches
of 64, top-eight diagonal proposals, standard-deviation floor 0.001, paid final
mean, globally best evaluated candidate, three-step action blocks, horizon
`min(12, 50-t)`, no terminal value. Preserve reward clipping and record raw as
well as clipped predictions. All arms receive the same full-horizon initial
innovations and later standard-normal innovations for each case/step; adapted
proposals may differ because their models score differently.

Proposed scope: 64 fresh paired cases under full, six-gap and ten-gap sensing,
plus reset variants for both gap panels. This is 45 learned rows across nine
fits, with all four established reference controllers in each intact panel
(12 reference rows). References retain their original access/budgets and are
not equal-compute learned comparators. Add 96 fresh mixed-policy prediction
episodes for saved one-step reward/angle and matched endpoint diagnostics.
Prediction metrics explain behavior; they do not replace the control gate.
All nine restored students must be authenticated before fresh evaluation draws.

Report per-fit native episode cost (negative summed native reward), paired
objective differences, full-sensing regression, reset effects, reward clipping,
held-out prediction errors and all
compute. Use episode-paired bootstrap intervals descriptively, conditional on
these three fits. Publish every fit; no best-seed selection. This remains a
development experiment. A later confirmation cohort and second environment
are required for a broader claim.

## History-reset intervention

At each prespecified gap's last visible packet, `t = gap_start - 1`, reset
hidden state to `model.initial` immediately **before** assimilating that packet.
Assimilate the same sanitized current packet exactly once. Thus both intact
and reset agents retain current angles, target, validity and age, while the
reset hidden state cannot retain earlier history. The trigger comes from the
frozen sensing schedule, never from future measurements. Reject a missing
precursor rather than moving the intervention after observing results.

Reset neither physical state, time limit, actuator noise, weights nor the
planner's random streams. Do not replay the preceding command or execute the
next command early. Apply at both gap precursors; memory can rebuild during
each subsequent gap. Closed-loop pairs share initial conditions, schedules and
exogenous noise, but their actions and later observations may diverge. Only a
separate recorded-history diagnostic can hold later packets/actions identical.

Record exact reset times and the public packet used to reconstruct state.
Synthetic tests must show that different prefixes ending in the same packet
produce the same reset state, current information survives, and no future data
or intact-branch state is changed. A reset penalty shows sensitivity to earlier
hidden state under this intervention. It does not alone establish an advantage
over a separately trained current-observation model, because reset can shift
the state distribution. That comparator is required before a stronger useful-
memory or topology claim.

## Compute, freshness and continuation

Count sample-level work, not just Python call counts. For a 50-transition
episode, each auxiliary adds 51 student assimilations, 50 prefix advances,
329 imagined advances and 142 predictor evaluations. Raw adds 142 extra
decoder calls; Latent adds 51 teacher assimilations and 50 teacher advances.
Every standard advance already executes its native observation/reward heads.
These are additional to the anchor's work. Record forward/backward, EMA,
calibration, setup, restoration, proposal fitting, serialization and final
hashing costs. The current latent component also computes covariance terms
and CPU float64 SVD collapse diagnostics every batch despite zero regularizer
weights; retain and charge these explicitly. Report training and deployment
costs separately. Update matching does not imply compute matching; an explicit
compute-matched Raw comparison remains necessary for any efficiency claim.

Reserve a new scored namespace, provisionally
`reacher-objective-ablation-v1-scored`; leave all numeric seeds and generator
states unset in this draft. Separate student initialization, predictor,
minibatch order, prediction collection, control reset/schedule/noise, CEM
innovations and bootstrap roles. Declare intended pairing explicitly. Before
freeze, enumerate all actual generator states against original training and
all previous evaluations, including `reacher-search-v1-scored`, plus full role
coverage of every engineering namespace. Engineering checks must never draw
from the proposed scored namespace. Reusing the authenticated training corpus
is intentional; its original collection is charged once in cumulative costs.

Proposed continuation gate: complete, independently audited coverage; latent
family mean native cost at least 5% below **both** Anchor and Raw on ordinary
and shifted panels; no paired latent fit worse than either comparator on
either panel. Require latent-family reset cost at least 5% above intact cost
on both panels, with every paired fit's reset effect positive. Separately
require every latent fit to beat zero-command cost by at least 10% on both
panels, and require known-state physics to beat zero by at least 10% on the
ordinary panel.
Lower latent loss or noncollapsed features cannot pass this gate. If Raw
explains the improvement, attribute it to the longer objective rather than
latent targets. Passing supports a follow-up current-observation baseline and
confirmation study, not biological or ICLR novelty by itself.

Before any freeze: implement a separate training/evaluation runner and saved-
output auditor; verify paired tensor/order identity, zero-weight anchor parity,
optimizer ownership, EMA count/gradient isolation, loss-scale calibration,
checkpoint restore, mask/boundary semantics, reset invariants, stream exclusions,
all cost counters and partial-failure preservation using synthetic fixtures.
Measure random-weight capacity, then set one explicit whole-run cap covering
all phases through final hashing and a separate audit budget. No cap is chosen
here. Freeze all source/runtime/data identities, remaining settings, seeds and
decision rules only after these checks. No retries, replacement seeds,
checkpoint selection or budget extensions after the scored run starts.
