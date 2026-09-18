# Can a known actuator penalty improve learned planning?

Status: frozen for one scored execution after 114 engineering checks passed.
No completed performance result yet.
It leaves the [negative nine-fit experiment](reacher-world-model-pilot.md)
unchanged. No scored training has run for this treatment.

[Frozen protocol](../evidence/reacher-reward-residual-v1/protocol/plan.json)
SHA-256: `df9929ea6320e24ba32ec8e8fdd84f88f8435ed03db1bea655ec31d82345e0fd`.
Implementation commit: `3a9ed3d`. Fit seeds are 271, 283 and 293; the protocol
specifies interleaved treatment order and every independent evaluation stream.

## Why this comparison

The previous GRU largely chose zero commands. Its loss was still falling when
the fixed training budget ended. Supplied-physics planning improved control,
but learned prediction did not translate into useful actions. These observations
motivate a competence test before any biological-topology comparison.

The native reward includes a known squared actuator penalty. Instead of making
the network learn that term, compare:

1. The existing GRU, which predicts total executed reward.
2. An identical GRU whose reward prediction is its learned residual minus
   `E[sum(clip(clip(command) + Gaussian noise, -1, 1)^2)]`.

The treatment uses the issued command and known noise standard deviation only.
It receives no realized disturbance, applied-action, hidden-state or separate
distance-reward labels. The loss remains prediction error against native total
reward. This is a conventional use of known reward structure, not a new RL
algorithm or evidence for a recurrent-memory advantage.

## Intended matched comparison

Use three fresh paired initialization seeds and six final fits. Keep the same
768-episode training corpus, GRU size, initialization, optimizer, minibatch
orders, loss weights, public observations and model-predictive planner within
each pair. Both arms receive 48 epochs, 1,152 updates at batch size 32. There is
no warm start, early stopping or best-checkpoint selection. Equal parameter
counts and update counts do not make wall time equal: charge the analytical
term's computation to both training and deployed decision timing.

Use 96 fresh exploratory prediction episodes and 64 fresh paired control cases
per sensing panel. Preserve full sensing, six-step gaps and ten-step shifted
gaps, plus zero-command and supplied-physics references. Retain memory-reset
interventions as diagnostics. Complete every fit before control evaluation.
Freeze exact fit, environment, noise, scheduling and candidate-bank seeds,
source hashes, all scoring rules and a single 30-minute cooperative execution
cap in a new protocol before the scored run. Preserve failed attempts; no
replacement seeds, resumes, cap extensions or evaluation-based retuning.

The prospective useful-effect rule is:

- Every residual fit reduces ordinary and shifted episode cost by at least
  10% relative to zero commands, with a competent supplied-physics reference.
- Residual family mean cost is at least 5% below the matched free-head GRU in
  both ordinary and shifted panels; each paired fit improves in both.
- Mean held-out reward MSE falls at least 10% versus the free-head comparator.
- Mean one-step angle MSE is no more than 5% worse than the matched comparator.

Report all control, prediction, reset and timing results even if these criteria
fail. These are development continuation criteria, not significance tests.
If longer training alone makes the free-head model effective, report that
without assigning its gain to the analytical term. If the treatment qualifies,
the next study must still compare memory against a strong history model at
the same training budget before a connectome experiment is justified.

## Engineering evidence

The new [component](../src/openjev/research/reacher_reward_residual.py) subclasses
the frozen GRU without changing its parameter names, count, order or seeded
initialization. `residual_reward=False` reproduces the original forward pass,
loss and gradients. Store `noise_std` and `residual_reward` in the experiment
configuration; the compatible weight dictionary does not contain them.

All **114 engineering checks** passed: 35 component checks, 32 unchanged
world-model checks, five runner checks and 42 independent audit checks. They
include numerical integration of the clipped-Gaussian moment, gradient checks,
clipping order, initialization identity, masked-input behavior, malformed
evidence rejection and a complete tiny train/control/native-replay fixture.
Its synthetic cases are separate from the scored cohort. Analytical evaluation
uses float64 on CPU. These tests establish engineering readiness, not control
effectiveness.

The run clock starts at the runner function's entry and includes input
validation, training-data copying, every fit, evaluation and final artifact
hashing. Python imports and final completion-receipt writing are outside that
clock. The cap is cooperative, checked between operations. All-fit and
evaluation-start receipts record the phase boundary; they are not an independent
process observer. Timings share a host with another chess study, and batch
amortized latency is not an isolated single-request benchmark.
