# Qualify a persistent-dynamics task before another architecture experiment

September 19, 2026. **Prospective design only: no implementation, seed allocation, simulator calls, fitting or result selection.** The normalized-innovation pilot failed its 21/29 continuation rule. This qualification would establish whether an independently motivated task rewards identifying changing dynamics; it does not reopen that pilot or promote its mechanism.

**Recommendation: a separate 200-decision Reacher tracking task with one hidden scalar motor gain, repeated public goals, and one unannounced gain change.** Start with full observations. Do not bundle gain, damping, missing sensors, new neural objectives and adaptive planning into the first qualification.

## What existing evidence and code permit

The [completed clean-Pendulum qualification](../../research/robotics-pendulum-qualification.md) already stopped that expansion: its two-angle, nominal-gain controller nearly matched the supplied-physics reference, and reference cost after the switch was only about 4.3 of 299. Identifying gain did not establish that gain identification mattered for control. Do not rerun that unchanged task or make it harder after seeing its scores.

The existing [Reacher episode](../../src/openjev/research/robotics_reacher.py) has 50 decisions, two RK4 substeps of 0.01 seconds each, a static target, full native reward weights 1/1, and public packets `[cos(q0),cos(q1),sin(q0),sin(q1),target_x,target_y,valid,age_seconds]`. The installed XML has motor gear 200, arm joint damping 1 and joint armature 1. The native reward is computed after stepping from fingertip distance and the squared normalized applied action. These facts were read from source, not measured in a new environment.

The [public kinematic observer](../../src/openjev/research/reacher_kinematic_control.py) knows nominal physics and infers interval-average velocity from two actual measurements. It rejects target changes and is not a gain estimator. The [geometry physics planner](../../src/openjev/research/reacher_geometry_physics.py) copies and fingerprints its model, restarts nominal candidate branches from supplied state, and uses approximate FK geometry plus expected clipped-action cost. Its configurable episode length can exceed 50; its existing outer runners, record replay and learned models cannot silently be extended.

The [roadmap](../../research/connectome-learning-program.md) and [mechanism note](../../research/reacher-architecture-mechanism-options.md) correctly require both identifiable persistent physics and meaningful post-change control. Neither supplies evidence that the new task below passes.

## One concrete plant change

Use one shared multiplier `g` on both native motor gears: `gear[:,0] = 200*g`. Keep masses, armature, damping, limits, integrator and reward weights unchanged. At the wrapper boundary:

```
u_t = clip(float32(clip(command_t, -1, 1)) + epsilon_t, -1, 1)
epsilon_t ~ independent Normal(0, 0.05^2) per actuator and decision
motor joint torque contribution = 200*g_t*u_t
native reward = -native fingertip distance - sum(u_t^2)
```

The gain therefore changes control authority without directly changing the normalized action penalty. Do not instead multiply the argument passed to native `step` and call its gain-dependent action penalty an unchanged reward. Gear also scales physical disturbance torque, which is a real confound to report. Zero commands with nonzero actuator noise are not an exact unidentifiability condition; use zero commands **and zero noise** for that engineering diagnostic. MuJoCo documents transmission gear as part of the mapping from actuator force to joint force. Its joint damping is a different velocity-dependent force, so vary only one first. [Official actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model), [gear reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-general-gear).

Candidate qualification settings, to freeze only after engineering review:

- 200 decisions / 4 seconds, no state reset at the change. All 201 public packets visible initially; no hidden velocity, applied noise, reward, gain or native-state fields enter a public controller.
- A new reachable public target at decisions 0, 50, 100 and 150, sampled by the same declared target distribution for every controller. Do not select targets by controller performance. No future target preview; all planners use the current public target throughout each candidate horizon.
- Five fully crossed regimes: `g=1`, `g=0.7`, `g=1.3`, `0.7->1.3`, `1.3->0.7`. For switched cases, change at decision 80 or 120, balanced across cases and independent of reset, target, action-noise and proposal streams. The controller receives neither the change flag nor its time. Current-parameter references know only the gain now, not a future schedule.
- Use 16 reset/target/noise cases in every regime. A goal change at 100 or 150 requires fresh movement after either switch time. Report costs by motion interval and switch direction, not just episode total. These fixed discrete settings qualify a task; they do not test unseen continuous parameter generalization.
- Separately retain the ordinary 50-step, fixed-target, `g=1` task on the same 16 case identities as a competence check. All qualification variants get identical public information and the same initial observer rule.

Changing targets has a precise event boundary: step with the target visible at decision `t`, calculate native reward against that target, then install any next public target before packet `t+1`. Save pre-event transition state/reward target and post-event decision state/packet target separately at these three boundaries. Otherwise an auditor can accidentally grade an action against a target the controller had not yet received. Announced target changes must not be treated as hidden-plant change labels.

This is a **custom tracking task using native Reacher physics**, not an unchanged Gymnasium Reacher-v5 score. Its horizon and public target process are deliberate changes that address control relevance, not evidence favoring a particular memory mechanism.

## Minimal classical identification and control contrasts

Use a transparent supplied-physics scalar grid estimator first, not the failed neural gate. Candidate gains are a fixed 21-point grid from 0.5 to 1.5. For every completed real action, start each candidate model from the same public-angle/velocity estimate and compare its nominal one-step cosine/sine prediction with the next actually visible packet. Maintain summed squared residuals over the last 20 completed transitions; choose the earliest minimum. The gain grid and all nominal simulations are explicit model-class privileges and paid work. This is finite-window prediction-error identification, not a calibrated Bayesian posterior.

To reduce the timing error of an interval-average velocity, define the common public observer explicitly: unwrap successive *observed* increments with shortest signed circular differences, use zero velocity at startup, one backward difference at the second packet, then the three-point backward endpoint derivative `(3*d_latest-d_previous)/(2*dt)`. Preserve the continuous branch through elbow soft-limit crossings. This remains an approximation under acceleration and hidden torque noise; never substitute audit velocities into the identification residuals. A separate engineering comparison against recorded velocities can expose its error but cannot repair its inputs. If this simple estimator is too biased to identify gain, stop and repair the reference under a separate specification before claiming a hard adaptation task.

Six control rows are sufficient for initial qualification:

| Row | State input | Dynamics used in planning | What it resolves |
|---|---|---|---|
| Public nominal | Common public observer | `g=1` | Whether ignoring gain already suffices |
| Public rolling identification | Same observer | Latest 20-transition estimate | Conventional adaptation comparator |
| Public estimate then freeze | Same observer | Same estimator until decision 40, then retain that value | Whether online correction after the change matters |
| Public current-parameter oracle | Same observer | True current `g` | Gain knowledge isolated from state knowledge |
| Privileged state + parameter | Actual current `q,qdot` and public target | True current `g` | Supplied-physics competence with both hidden quantities revealed |
| Zero command | Public interface | None | Native-cost floor for competence checks |

The parameter-only oracle is essential: comparing only against true state plus true gain would confound parameter identification with velocity estimation. Neither oracle sees future disturbances, future target values or future gain changes, and finite-search references are not optimal bounds.

Give every planned row the same existing geometry score, clipping, CEM256, horizon 12, action block 3, paid mean and earliest-global-tie rules. Pair initial innovations and noise across rows; subsequent CEM proposals differ because scores differ. Charge identification, state reconstruction, candidate simulations, selected advance, copies, validation and trace I/O. Equal candidate count does not imply equal wall time. Candidate branches cannot update an estimator. Do not remove the classical method's update work from its reported decision cost.

## Excitation and explicit stop conditions

First use a small separate, predeclared public-action diagnostic with alternating signed pulses on each joint and coast segments, plus the zero-command/noise-free negative control. This validates gain observability and command alignment without selecting a controller. For the noisy control trajectories, report issued-command energy, clipping, public angular movement, grid residual separation, and the number/age of post-change informative observations. Do not delete unexcited cases. A 4-transition version of the **same saved-residual estimator** is a useful descriptive short-history check: if it identifies gain as well as 20 transitions, do not claim the task needs long memory.

Suggested practical admission rules, not existing results or selected thresholds:

1. On both gain values and switch directions, the public estimator reaches mean absolute gain error at most 0.1 within 30 completed post-change transitions and maintains it in the next 20. Require this on both a fixed excited diagnostic and the actual tracking controller's histories. Audit-only gain is used for grading, never updates.
2. In ordinary 50-step and nominal tracking controls, the privileged reference beats zero cost by at least 10%; identification adds at most 5% cost versus public nominal. Failure signals deficient task/controller competence.
3. On each switched direction, the public current-gain oracle reduces *post-change* native cost by at least 5% versus nominal and versus the frozen estimate. This establishes a useful value of parameter information under the same state observer. Also report absolute distance/action-cost differences; do not accept a large ratio from a negligible denominator.
4. Public rolling identification reduces post-change cost by at least 3% against both nominal and frozen-estimate controls in each direction, without worsening pre-change cost by more than 5%. Retain all cases and separately show the first required movement after the switch. A gain-error pass without a cost pass is insufficient.

Predeclare these or another agreed numerical rule before any qualification outcome. An estimator failure is not an architecture opportunity by itself. If nominal control is already competitive, or adaptation matters only in a selected gain/case, stop the broad persistent-context claim. Full-sensing qualification precedes a separate missing-sensor stress panel; no need to add it here. Even a pass authorizes a newly specified learning comparison, not normalized gating, connectome structure or novelty claims.

## Additive integration, without changing frozen sources

Prefer four new files plus focused tests: `robotics_reacher_persistent.py` for the 200-step/event/gain wrapper and replay; `reacher_gain_identification.py` for the public observer, gain bank and update trace; `qualify_reacher_persistent_dynamics.py` for the explicit six-row orchestration; and `audit_reacher_persistent_dynamics.py` for saved-data replay/criteria. These are proposed paths, not implemented files. Reuse unchanged packet construction, geometry math and adaptive-search kernels where their contracts fit.

Do not subclass the old episode then override its `HORIZON` global, mutate old observer targets, or change the frozen physics adapter's private model after its fingerprint. Construct independent gain-specific model copies for the grid and oracle; a new adapter selects a declared fixed model for each planning call. Preserve every model identity and exact gain/event timeline in audit-only records. No hidden parameter can enter a public model constructor, serialized state or packet. Native replay must restore the scheduled model parameter as well as integration state, since the latter alone does not encode a changed MjModel.

The maximum proposed main control coverage is `16 cases * 5 regimes * 6 rows * 200 = 96,000` executed decisions, plus 4,800 ordinary decisions. CEM nominal work is much larger, and gain-grid identification adds work. Measure one representative full-size engineering row and its complete replay before choosing execution/audit caps or storage limits. This note allocates no seeds, budget or launch authority. Keep the existing 136 pilot and 116 scientific source files byte-identical.

Reacher is the better immediate integration target because its scored control pipeline and public-packet boundary already exist. Pendulum is a later second-task possibility only with an independently motivated tracking/disturbance process that passes the same identification-plus-control test. Its old stopped clean-gain task is not a confirmation environment.

The next useful evidence would be native cost plus parameter recovery from the classical qualification, followed by a fresh paired learning experiment against that classical method and ordinary GRU/history controls. Generic online identification or fast-weight adaptation is already established; a contribution would need a specific improved learning mechanism and convincing matched evidence.
