# Public classical comparator for persistent Reacher dynamics

**Recommend the finite-window gain bank first, subject to an explicit public-observer identification check.** It is transparent, cheap relative to CEM256, and independently replayable. It is not a joint optimal state/parameter estimator or a calibrated posterior. A failure caused by velocity-estimation bias would justify a separately specified augmented-state UKF, not a claim that the task requires a neural architecture.

This is a prospective interface review. No simulation, inference, fitting, RNG, or current-study outcome inspection was performed.

## Minimal public interface

```python
reset(packet0) -> belief
observe(previous_belief, issued_command, next_packet, *, decision_index) -> belief, receipt
planning_root(belief) -> copied_angles, copied_velocity, gain_estimate, current_goal
snapshot(belief) -> auditable_public_history_and_estimator_state
```

Construction receives an independently instantiated, fingerprinted **nominal XML model**, declared timestep/frame skip, fixed gain grid/window/initialization settings, and known noise specification. Never obtain the nominal model by copying the live plant: its gear can already contain the hidden gain. No `MjData`, actual velocity, reward, applied noise, parameter schedule, switch flag, or plant object enters this interface.

Start with full sensing and exactly one `observe` call per actual issued command. Packet `t+1` closes transition `t`; there is no estimator update during candidate imagination. Initial gain is 1 and velocity is zero. The first packet establishes principal angles; later packets maintain continuous branches using shortest observed increments. Flag possible angular aliasing instead of repairing it from native angles. Elbow soft-limit physics makes preserving the continuous branch consequential.

At root `t`, use the backward difference after one observed interval, then the specified three-point endpoint derivative. **Save this root estimate before seeing packet `t+1`.** For each of 21 fixed gains from 0.5 to 1.5, predict the completed command's next four cosine/sine channels from that same saved root. After the real packet arrives, append its squared residual to that gain's 20-transition ring. Choose from the sum over completed transitions only. Never recompute the start velocity using the endpoint being scored.

Each nominal prediction resets private integration state, installs public angle/velocity and the old public target, calls `mj_forward`, and executes the same two RK4 substeps of 0.01 seconds. Gear is `200*g`; the issued action and declared nominal disturbance approximation are held over both substeps. Save all four predicted channels and residuals. The approximation to clipped stochastic actuation must be explicit: zero-disturbance issued-action propagation differs from its conditional mean near saturation. Do not silently absorb that difference into a gain claim.

Goal changes update only the current public goal. They must not reset the velocity, residual window, or gain, and cannot signal a hidden dynamics change. Planning uses the current goal throughout its horizon, no future goal/switch preview. Selected actions are acknowledged by the real wrapper; imagined successors never overwrite the observer.

## Is the proposed diagnostic sufficient?

It can establish a useful baseline if it passes both identification and control checks. Nonzero command energy alone is insufficient: gain and latent velocity can explain similar one-step displacements. Three-point differentiation assumes locally smooth motion; action changes and hidden torque noise create endpoint-velocity error that a gain fit can absorb. More residuals do not remove systematic bias. The 20-transition window also deliberately mixes regimes immediately after a switch.

Keep the proposed alternating signed pulses, coasts, and zero-command/zero-noise negative control, plus an independent switch sequence. Compare parameter error using audit-only labels, without feeding them back. Retain all unexcited cases. The **public-state/true-gain oracle** isolates gain knowledge; the additional true-state/true-gain reference exposes velocity confounding. Neither is an optimal-control bound.

One prospective tie refinement: retain the previous gain on an exactly flat residual bank, initialized at 1, instead of selecting the lower endpoint. If the existing earliest-minimum rule is retained, explicitly label flat windows non-identifying; their arbitrary estimate is not evidence of parameter recovery.

Minimum independent engineering checks:

- Changing only future packets or an unissued candidate cannot change current state/residuals.
- Saved prior velocity excludes the measured endpoint; command `t` is scored against packet `t+1` exactly once.
- Noise-free, excited native traces recover each declared gain with the actual public observer; the zero-input/noise control is gain-indistinguishable.
- Wrap crossings preserve public branches; target changes preserve estimator state; unexpected missing packets are rejected in this full-sensing version.
- Candidate order/batch duplication leaves real history unchanged, including after failure; independent model copies prevent gear contamination.
- Saved predictions replay with the declared reset/substep semantics. Count identification simulations, selected predictions, copies and updates in total cost.

## UKF as the stronger fallback

An augmented-state UKF would jointly estimate two angles, two velocities, and positive gain, using public cosine/sine measurements and known commands. It reduces the need to treat differentiated velocity as exact, but introduces covariance, gain-process-noise, initialization, and numerical-floor choices. Gain-scaled actuator noise and clipping must be modeled consistently. Five latent dimensions require 11 sigma points in the basic formulation; augmenting two actuator-noise dimensions requires 15. All propagation is paid work.

A UKF should use the same nominal model and planner, a common gain/velocity prior independent of the hidden case, wrapped angular treatment, and private branch state. Freeze its settings before qualification outcomes. Do not tune process noise until it wins. Implement it only if the simple baseline fails a diagnosed state/parameter-estimation check; do not run both as an open-ended search for a favorable task.

Reviewed design SHA256: `97b85a599e7e14a5e9e30373ab932752c3aaf857a01ab5d5a0429208ee65d7a1`. Existing nominal planner SHA256: `31332b0226a27a44602548bd8a8679bf0c4456d6f9f4ca505dfdebff28aab101`. New dynamic-target/gain integration must be additive; the old observer's static-target contract and planner's model fingerprint must remain unchanged.
