# Prospective two-observation/action-history control

Written 2026-09-19 03:01:37 UTC (2026-09-18 20:01:37 America/Los_Angeles), before reviewing any results of the current geometry-memory comparison. This is a conditional design, not a frozen protocol, implementation, model run or scientific result. No current-study execution or audit output was read. Advance only if that whole comparison passes its declared gate. Otherwise retain its outcome and do not use this proposal to move the goalposts.

## Recommendation

Train a **bounded reconstruction GRU** that rebuilds its decision state from the older of its last two valid public angle observations and every issued command since that observation. Start the reconstruction from zero at every real packet. Keep the original GRU transition, observation update, heads, geometry scorer and planner. This is a strong known-history baseline, not a novel architecture.

This is cleaner than making a history MLP the first comparator: it preserves the existing 36,805-parameter GRU64 tensor schema and imagined dynamics. It tests whether a short, explicit public history explains the benefit attributed to persistent learned state. The existing three-real-packet component already provides the relevant phase and reconstruction pattern, but three consecutive packets can all be missing. The new boundary follows **valid observations**, not a fixed three-packet window.

A flat command-history MLP has an avoidable shift confound. A six-step blackout requires at most seven commands since the older valid observation; ten-step blackouts require eleven. With lag-specific flattened weights, four older command slots are never active in the training schedule. Their weights would receive no data gradient and become newly active only during the shift. Shared recurrent reconstruction uses the same trained operations at every lag. Longer reconstruction remains a genuine distribution shift; it simply avoids introducing entirely untrained lag-specific parameter blocks.

## Exact causal information

At real decision index `t`, the public packet is float32:

`p_t = [cos(q0), cos(q1), sin(q0), sin(q1), target_x, target_y, valid, age_seconds]`.

There are decisions `t=0..49`, commands `a_0..a_49`, and recorded observations `p_0..p_50`; `dt=.02`. The command `a_j` is the already clipped float32 command issued between observations `j` and `j+1`. Do not use applied actions, actuator disturbances, native velocity, fingertip displacement or reward as model inputs. The static target is public even during blackouts. Future schedule entries are not available to the model.

Let `v_t` be the newest index `<=t` whose packet is valid, and `u_t` the preceding valid index if one exists. Define the reconstruction anchor `b_t=u_t` when two observations exist, otherwise `b_t=v_t=0`. Its only permitted evidence is:

- The actually observed four-angle features at `u_t` and `v_t`, with their real indices and a Boolean saying whether the older observation exists. Startup absence is different from a missing measurement.
- The intervening sanitized public packet timeline from `b_t` through `t`, including actual validity and age. By definition it contains no additional valid observation between the last two.
- Every chronological issued command `a_b_t, ..., a_(t-1)`. Include commands between the older and newer observations and after the newer observation. Do not include `a_t`, which is not yet selected.

For a gap starting at `g` with length `L`, the last missing decision is `t=g+L-1`, the newest visible measurement is `g-1`, and the older measurement is `g-2`. Thus `t-b_t=L+1`: at most seven commands for `L=6` and eleven for `L=10`. At the first visible packet after the gap, the interval between the newest two valid measurements is also `L+1` commands. The declared schedules therefore require **at most twelve real packets and eleven issued commands**, not ten commands. This bound is specific to the declared maximum gap, not a general partially observed robotics guarantee.

Maintain fixed-size, left-padded arrays `real_packets[B,12,8]`, `real_actions[B,11,2]` and a separate `present[B,12]` startup mask. After each new real observation, remove or zero every entry older than `b_t`; never retain it as an alternate encoder input. Store integer real indices and validate their continuity. Padding packets/actions must be zero. Zero issued commands remain present and must not be mistaken for padding. Reject an unsupported history exceeding eleven commands rather than silently truncating or fabricating a measurement.

The first packet must be valid and have age zero. Later valid packets have age zero; missing packets have `age=(t-v_t)*dt`. Use the existing tolerance `rtol=1e-5, atol=1e-6` for float32 age checks, and integer indices for ordering. Sanitize unavailable angle entries before finite-value validation so poisoned missing placeholders cannot convey information. The actual packet remains unchanged after sanitization; cached or predicted angles never turn `valid=0` into a measurement. Reject changes to the static public target.

## Reconstruction and imagination are separate

For each real assimilation, ignore preceding hidden vectors and predicted angles. Initialize the original GRU state to zero and replay the twelve padded slots in order:

1. Run the unchanged observation assimilation on a real slot, retaining its result only if that slot is present. Its normal validity gate skips hidden-state correction on missing measurements.
2. Between present neighboring slots, run the unchanged transition with the corresponding issued command. Mask absent startup edges without treating them as real transitions.
3. Return the reconstructed hidden vector and the actual sanitized current packet as the planner root.

A simple vectorized implementation may execute all twelve observation updates and eleven transitions for every sample, even when padding masks most of them. Freeze that choice and charge all executed calls, including the original observation/reward heads and analytic action-cost helper reached by each transition. Do not count only the unmasked work or detach the reconstruction from training gradients to make the cost look smaller.

Candidate planning starts from independent copies of this root. Repeated imaginary advances may retain private recurrent state, but never append imagined packets or actions to the persistent real-evidence buffer. After selection, re-advance the chosen action once from the real root. Its pending issued command is appended to the real buffer only when the next actual packet arrives. Require episode-start depth zero or exactly one selected advance before real assimilation; reject a multi-step candidate terminal. The caller and saved native trace still must prove that the pending command was actually issued.

No decision is made at `t=50`. The final observation can be a loss/replay target; it does not authorize a fifty-first action or extend the planning horizon. Reconstruction must reset independently at every episode and case.

## Complementary finite-difference velocity option

The transparent public estimate is an **interval-average angular rate**, not current native velocity. For each joint, obtain `theta=atan2(sin,cos)` from the two valid observations and compute:

`delta = ((theta_new-theta_old+pi) mod (2*pi)) - pi`

`omega_interval = delta / ((v_t-u_t)*dt)`.

Specify the branch interval `[-pi,pi)`, including the exact pi tie, before implementation. With only one valid observation, return rate zero together with `older_available=0`; never present that zero as a measured stationary state. Keep the measured interval, newest-observation age and availability mask. Do not divide by time since the newest observation or by a hardcoded one-step interval across a blackout.

This estimate aliases motion exceeding pi between observations, including possible elbow soft-limit crossings. Noise and acceleration make it differ from instantaneous velocity, and it becomes increasingly stale after the newest measurement. Issued commands do not reveal realized actuator noise. Constant-rate extrapolation `theta_now_est=theta_new+omega_interval*age` would add another approximation and must not be relabeled a current measurement. The raw trigonometric history baseline avoids choosing an angular unwrapping branch, though its observations remain physically ambiguous too.

A velocity-feature MLP is useful as a separately named transparent comparator, especially if an explicit state estimate is the product goal. It is a less clean first test of persistent memory because it bundles rate estimation, feature compression and a changed imagined transition architecture. The already declared public-kinematic physics reference is not a substitute: it also knows the native dynamics. Do not claim biological or learned-state necessity merely by beating a finite-difference baseline.

## Matched fitting and evaluation

The minimal incremental version adds **three newly trained history-control fits**. Reuse all three authenticated original GRU initial tensor payloads, training episodes and complete minibatch orders from the cache parent, together with its exact Adam settings. Train the new class from those original initial weights with reconstruction active at **every training real boundary**. Do not initialize from a fitted persistent checkpoint, install a deployment-only reset, transplant a history encoder after training, or reuse an optimizer state from a completed fit.

Retain all three already trained persistent-GRU and cached-GRU controls as paired references. This is nine evaluated models with three fresh fits, not nine fresh fits or an independent training replication. The initial tensors, data, order, loss and update count match within each pair; final weights and real-assimilation semantics differ. If a fully fresh nine-fit replication is preferred, choose that before any new fit or evaluation and use three newly frozen paired initialization/order streams. Do not choose between those designs from results.

Use the unchanged 768 public training episodes, 48 epochs, batch32 and 1,152 updates per fit; unchanged `sequence_loss`, valid-angle masks, native executed reward targets, five-step rollout term and residual expected-action-cost skip. The existing rollout loss includes only valid roots/endpoints under its frozen mask; do not silently broaden targets for the new arm. Equal examples and updates do not equal training compute because reconstruction repeats neural work. Report it explicitly.

All three GRU classes can inherit the same constructor, parameter names, initial tensor payload and 36,805-parameter count. No added position embedding, learned command-slot matrix, velocity head or target privilege is necessary. Actual class and reconstruction semantics must be bound in new trainer/checkpoint/control metadata; compatible state-dict names do not permit substituting it into a frozen trainer registry.

Evaluate every model on the same **fresh** 64 cases in full sensing, six-step gaps and ten-step gaps, with all five supplied-physics/floor references retained: 27 learned rows plus15 references,42 total rows and134,400 executed native transitions. Pair environment resets, hidden disturbances, public schedules and planner innovations across rows; do not join new-model means to old evaluation results. Use the same geometry score, retained original reward-head work, CEM256, horizon12, block3, per-step clipping, paid mean, global-best rule and terminal truncation. Later CEM proposals are score-adaptive and can differ despite shared innovations.

The primary future contrast is persistent versus bounded two-observation reconstruction. Cached GRU answers whether the larger explicit history improves on a single retained measurement. Preserve all pairs, panels and failures. Freeze numerical continuation margins and competence checks before fitting or drawing the new evaluation cohort; do not select them from the current run. Reusing historical training inputs is intentional matching, whereas evaluation streams must be newly separated from every scored and engineering stream.

## Costs, acceptance boundaries and useful failure

Separate historical fit costs from the three new fits. Charge all twelve-slot reconstruction calls, padding work, command buffers, copying, geometry, original heads, selected action advances, native steps and trace I/O. Candidate budget matching is not equal total compute. Report reconstruction time/sample counts and complete row wall time; a faster persistent controller may benefit simply from avoiding repeated public-prefix computation. Do not label that a superior information representation without the utility comparison.

Before any scored work, synthetic checks should establish exact equality for histories with identical last-two-observation suffixes but different older pasts; poison immunity; first/second observation transitions; visible packets immediately after gaps; command alignment; the eleven-command boundary; circular wrap/alias examples for the optional rate estimator; no candidate mutation of real buffers; trained-gradient flow through reconstruction; actual-class restoration; and terminal50 behavior. The existing three-packet implementation is a template, not an already valid two-observation implementation. Prepare new additive files, independent saved-output audit and capacity measurement; keep the current90 sources immutable.

If the bounded history control removes the persistent advantage, report that this simple information restriction was sufficient for the tested task and training configuration; failure of a superiority gate is not proof of equivalence. If persistence remains better across the predeclared pairs and scenario shift, the justified claim is advantage beyond this bounded public-history/control recipe. It still does not identify a latent velocity coordinate, Bayesian inference, biological topology or a novel architecture. A second environment and independently repeated training remain separate requirements.

Source basis, read-only: `robotics_reacher.py::packet/ReacherEpisode`; `reacher_world_models.py::GRUWorldModel/sequence_loss`; `reacher_cached_observation.py::_PublicBoundary`; `reacher_bounded_history.py::BoundedThreePacketGRUWorldModel`; `reacher_cache_protocol.py::settings/schedule`; `reacher_kinematic_control.py::KinematicObserver`. No current geometry-memory result was used.
