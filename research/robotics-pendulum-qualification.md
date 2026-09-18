# Does this robotics task require persistent memory?

This is a supplied-physics qualification of an observation-only Pendulum task,
not a trained world model or a connectome result. It follows the
[connectome and robotics literature review](connectome-world-model-robotics-reading.md).

The question comes before architecture selection: can a short observation
window recover enough hidden state to control the system? In this noiseless
simulator, two angle observations determine angular velocity. A third can
identify constant actuator gain if the action is informative and neither
torque nor speed is saturated. A sophisticated memory model should not receive
credit for solving a task that this calculation already solves.

## Frozen comparison

The [protocol](../evidence/robotics-pendulum-qualification-v1/protocol/plan.json)
was prepared before the scored execution. Its SHA-256 is
`fd369d82e16793615f484a349af628e453e985b562d1efc0d4dc5fecec0cc58a`.
Implementation commit: `aa9352b`.

- 32 initial states, 200 steps each, six controllers, two panels: 384 episodes.
- Hidden actuator gains of 0.65, 0.8, 1.0, 1.2 and 1.35. The second panel
  changes gain to `2 - gain` halfway through an episode, without announcing it.
- Observation-based controllers receive only float32 angle cosine/sine and
  their issued command histories. All planners are supplied the same physical
  equations. The reference additionally receives true current velocity and gain.
- Compare current observation, velocity with nominal gain, full-history gain
  estimation, last-three-observation gain estimation, known-state reference and
  a uniform-action floor.
- Identical two initial probe commands and identical candidate action sequences
  for every planner. Both probes count toward episode cost.
- 64 candidate plans, a 32-step horizon and four-step action blocks. Execute
  the first action, then replan. No fitting, tuning, warm starts or terminal value.
- Separate 256-case excited-action and zero-action observability diagnostics.
- Ten-minute execution cap, no retries, replacement cases or extensions.

Lower native episode cost is better. The known-state controller uses finite
random-shooting planning, so it is neither optimal nor an upper bound on return.
Paired episode bootstrap intervals describe these cases; they do not measure
training-seed uncertainty. Gains are not distributed equally across the 32
cases, so the audit also reports each gain cell.

Short-history qualification requires all six frozen checks: within 5% of
reference mean stationary cost, at least 10% better than current-observation
control, reference at least 20% better than uniform, accurate velocity and gain
recovery, and no gain identification from zero commands. A pass rules out a
large long-memory/connectome sweep on this clean task. A failure calls for
diagnosis, not a biological-architecture claim.

The saved-output audit independently replays every physical transition through
the installed Gymnasium implementation, verifies recorded observations and
estimates, and recomputes costs and paired intervals. No new planner decisions
are made during the audit. Raw trajectories stay under the local run directory;
the public protocol and audit bind their hashes.

## Reproduce

Use the runtime recorded in the protocol and the exact frozen source files.
The runner rejects source or runtime changes, existing output directories and
a mismatched external plan hash.

```bash
python scripts/qualify_robotics_pendulum.py run \
  --plan evidence/robotics-pendulum-qualification-v1/protocol/plan.json \
  --expected-plan-sha256 fd369d82e16793615f484a349af628e453e985b562d1efc0d4dc5fecec0cc58a \
  --out runs/robotics-pendulum-qualification-v1/execution
python scripts/qualify_robotics_pendulum.py audit \
  --plan evidence/robotics-pendulum-qualification-v1/protocol/plan.json \
  --expected-plan-sha256 fd369d82e16793615f484a349af628e453e985b562d1efc0d4dc5fecec0cc58a \
  --execution runs/robotics-pendulum-qualification-v1/execution \
  --out evidence/robotics-pendulum-qualification-v1/audit
```

The protocol binds the original platform. A different machine needs a separate
protocol and output location; its measurements should remain a separate run.
