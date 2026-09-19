# Explicit public-memory controls

Prospective design only. No implementation, new training, seed selection or protocol freeze. This note explains the next comparison; it makes no new performance claim.

The completed study establishes an advantage over the trained current-packet and three-packet GRUs. It does not identify the useful information. The packet MLP is close in mean utility and wins one paired fit in both gap panels. Keep that strong comparator when testing whether remembering angles, rather than a richer learned recurrent state, explains the gain.

## Source constraint that changes the design

The frozen [GRU](../../src/openjev/research/reacher_world_models.py) sanitizes unavailable angles and gates its observation update on actual measurement validity. Its transition consumes the action, goal, validity and age, together with hidden state. The [current-packet GRU](../../src/openjev/research/reacher_observation_baseline.py) resets hidden state before that gated update, so a missing packet leaves zero hidden. The [bounded GRU](../../src/openjev/research/reacher_bounded_history.py) rebuilds from three actual public packets and two issued commands; it cannot retain an angle measurement older than that window.

Simply inserting cached angles into an external packet does not work: sanitization removes them when validity is zero. Setting validity to one would falsely label a past measurement as current. Both must be avoided.

## Last-valid-angle control: exact GRU weight schema

Maintain an explicit cache of the latest **actually observed** cosine/sine angles and its real decision index. Cache updates come only from sanitized public packets with validity one. Require a valid first packet, as in this task; reject unsupported missing-start episodes. At each real decision:

1. Validate the incoming packet and static target. Verify age against elapsed real decisions since the last actual measurement. Discard hidden state and all predicted angles from the preceding real decision.
2. Form an internal eight-feature encoder input `[cached cos/sin (4), current goal (2), actual current validity (1), actual current age (1)]`. This is explicitly a cached-feature vector, not a new sensor packet. On a visible step, the cache is updated first and contains the current measured angles.
3. Compute `observation_update(features, zero_hidden)` at **every** real root. Keep the actual sanitized packet separately as `state.packet`; never change its validity, age or training-target fields. The original action-conditioned transition, readouts and analytical actuator-cost skip remain unchanged.

This preserves all GRU parameter names/shapes and its **36,805 parameters at hidden size 64**. It permits exactly paired initial tensors. It changes the assimilation semantics, so the actual class/configuration must bind that change; compatible tensor shapes do not authorize reusing a trained checkpoint as this comparator.

**Required gate control:** pair the cached GRU with a zero-filled, reset-and-encode GRU that also runs `observation_update(features, zero_hidden)` on every root, using the sanitized current packet as its features. Otherwise cache availability is confounded with removing the frozen current-GRU validity gate and adding nonlinear processing on missing steps. Train this control with its own declared semantics; do not reinterpret the previous gated model's results.

The cache is explicit memory that can span a long blackout. It is not current-observation-only or a finite-time-window controller. No learned hidden state or predicted angle crosses a real assimilation boundary. Ordinary recurrence remains available *within* each candidate's imagined rollout.

## Two-valid-measurement control: explicit motion features

Retain the last two actual angle measurements and their decision indices. With two measurements, calculate each joint's interval-average angular velocity as `wrap(angle_latest - angle_previous) / ((index_latest - index_previous) * dt)`. Use the actual interval, including skipped measurements, rather than a one-step denominator. Before two measurements exist, use zero velocity and an explicit unavailable flag. During a blackout, do not refresh these values from predictions; retain the measurement interval and expose the growing current age separately.

Use 12 encoder features: the cache-only eight, plus two velocities scaled by known `dt`, the measured interval in seconds, and a two-measurement-available flag. Shortest-angle differencing has multi-turn ambiguity and gives interval-average, potentially stale velocity, not true native `qvel`. No dynamics model, realized disturbance, native state, hidden-angle labels or inferred future observation enters this controller. Do not add nominal action-driven propagation to this arm: that would introduce another mechanism already present in the supplied-physics reference.

A 12-input observation GRU adds `3 * 64 * 4 = 768` parameters: **37,573 total**, 2.09% above the existing GRU. For the motion ablation, give the cache-only and current-only controls the same 12-input constructor, zeroing unavailable feature columns under an explicit feature contract. They then have identical parameter shapes and initialization. Keep shared layers equal; do not narrow hidden size merely to match the earlier total, because that also changes transition/readout capacity. For an 8-versus-12-feature initialization comparison, explicitly share unchanged tensors and draw extra input columns separately: the same constructor seed alone does not preserve later layers after a larger input matrix consumes more draws. Report the additional encoder work and actual wall time.

An eight-slot constant-velocity angle extrapolator could retain the old weight schema, but would conceal the motion feature and lose velocity information when measurement age is zero. It is a different, weaker control and should not substitute for the explicit motion comparison.

## Keep a strong feedforward comparison

Apply the same current/cache/motion feature contracts to a reset-at-real-boundary MLP. Its first imagined advance consumes the root features plus the proposed action. Later imagined advances can use predicted packets, flagged unobserved, but may never update the real measurement cache. Every new real packet discards those imagined estimates.

| Feature comparison | GRU | Feedforward comparator |
| --- | --- | --- |
| Current versus cached angles, 8 features | Hidden 64, 36,805 parameters | Width 107, existing 36,599 parameters |
| Cache versus motion, 12 features | Hidden 64, 37,573 parameters | Width 108, 37,697 parameters, approximately 0.33% more |

These are parameter comparisons, not compute matches. The 12-feature MLP count follows its unchanged two-layer encoder/readouts with a 14-input first layer (features plus action). During imagination, measured velocity/interval and their existing two-measurement-available flag remain attached to the real root; current measurement validity stays zero and age increases. The availability flag means two past measurements exist, not that a fresh measurement occurred. Model-predicted angles remain separate and never create another velocity sample; no thirteenth input feature is implied.

Prioritize the 8-feature cache comparison first, including the persistent GRU anchor, reset-and-encode current GRU, cached GRU, current MLP and cached MLP. Add the matched 12-feature motion comparisons only if the first comparison leaves a relevant unexplained gap. Keep the original gated current GRU as historical context; add it as a fresh diagnostic arm only if measuring the gate change itself becomes a separate question.

## Minimum engineering and interpretation requirements

- Rebuild identical hidden/root features from different older histories whenever their declared cache, current packet and feature inputs match. Poison old hidden/predicted packets and missing-angle placeholders to test that they cannot affect a real root.
- Candidate expansion copies cache tensors; `advance` never changes actual measurement buffers. Permit real assimilation only at episode start or after exactly one selected issued-action advance, never from a multi-step candidate terminal state.
- Test wrapped angles, irregular measurement intervals, first/second observation, six/ten-step blackouts, age disagreement, terminal boundaries, static target and cache immutability. Preserve causal training prefixes and the **original actual-validity loss masks**, including observed rollout roots; cached features are never new labels.
- In a future separately frozen study, pair data, initial shared tensors, minibatch order, updates, CEM proposals and exogenous evaluation noise. Charge cache/feature construction, every encoder call, state copying and full training/deployment time. Authenticate actual class, feature order/scaling, cache rules and initialization in restored checkpoints.
- If a cached MLP or cached GRU closes the gap, report simple retention as sufficient for this task. A motion-feature gain supports useful information in two public measurements, not a learned world-model discovery. Residual gains after both controls motivate further study, but still do not establish biological topology or transfer to a second environment.
