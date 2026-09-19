# Action-conditioned associative memory: offline dynamics pilot

Prospective protocol, September 19, 2026. This is a new exploratory experiment,
not a reopening of the closed clean-Pendulum or Reacher protocols. No result is
implied by this document.

## Question

Does advancing an associative memory with actions before correcting it with
observations improve short-horizon dynamics forecasts? Compare learned row
transport against scalar decay, a standard GRU, a separately trained GRU using
only the last 16 observations, and a learned diagonal filter. Add a strong
16-observation direct ridge predictor and persistence reference.

The mechanism combines established predict/correct and delta-memory ideas.
[Action-Conditional Recurrent Kalman Networks](https://arxiv.org/abs/2010.10201),
[HiP-RSSM](https://openreview.net/forum?id=ds8yZOUsea), and
[Multi Time Scale World Models](https://github.com/ALRhub/MTS3)
are relevant prior work. This is not an exact reproduction of those models,
a new biological principle, an RL method, or evidence of architectural novelty.

## Data and boundaries

Use the numeric `ts_002_50x2000_w_grad.npz` archive from the official
[HiP-RSSM repository](https://github.com/ALRhub/HiP-RSSM/tree/f4adef0ce5ed4b37c9847973df320f72965bd330).
The preparation receipt pins its path, bytes and source revision. This is a
**simulated PyBullet mobile robot**, not a real robot experiment. Recorded
applied torques condition forecasts; they are not authenticated issued commands.
This offline conditional prediction task does not establish a deployable causal
world model. The source supplies no terrain/run labels sufficient to claim a
terrain-disjoint test or independent raw simulation runs.

Retain complete trajectories before splitting. Train on source IDs 0-29;
development on 41-49; test on 30-40, preserving the publisher's test IDs.
No source trajectory crosses partitions. Exact duplicate trajectories are
checked. Each has 1,750 frames. Inputs are xyz position and sine/cosine of
three Euler angles; the slope channel is excluded. Applied torques at index t
condition the forecast for frame t+1, following the upstream loader. Normalize
both observation and action channels using training trajectories only.

Take 16 nonoverlapping 42-frame windows per trajectory, with integer starts
equally spaced from 0 to 1708. This gives 480 train, 144 development and 176
test windows. Windows from the same trajectory are not independent experimental
replicates. Each window has 32 context observations followed by 10 targets.
Models see no target observations during forecasting. Forecast step h may use
recorded actions only through that step, not subsequent controls.

The source configuration reports 500 Hz without per-frame timestamps. Thus
these are nominal 20 ms forecasts after 62 ms context, a short-horizon diagnostic,
not long-horizon planning. Persistence and direct ridge are essential controls.

The source repository does not provide a clear dataset license. Keep downloaded
and normalized data local; publish code, provenance, aggregate results and our
own model weights, not upstream data/code. This is a new protocol on its data,
not a comparable reproduction of the paper's reported benchmark score.

## Models and training

- Transported delta: a 16 by 4 matrix state, unit-norm learned keys,
  learned values, sigmoid write strength, scalar retention, action drive,
  and action-conditioned convex mixing of learned row-stochastic maps.
- Decay delta: identical common-module initialization, retention, drive,
  write and readout, with row mixing absent and not computed.
- GRU: 64-state observation and action updates.
- History16: a separately trained identical GRU reconstructed from zero using
  only the last 16 context observations and the 15 intervening actions.
- Diagonal filter: 64 mean and 64 variance entries, learned diagonal transition,
  process noise, observation encoding and noise, and scalar Kalman-like updates.

The first four have 64 recurrent-state floats. Equal state size is not equal
parameter count or compute. Report actual counts; the filter has extra variance
state. History16 also needs its observation/action buffer in an online service.
No predicted observation is fed back as a measured observation.

Train all five families for 20 epochs, batch size 32, Adam learning rate .003,
gradient norm cap 1, in CPU float32 with one thread. Use paired seeds
101, 202 and 303 and identical batches/targets per seed. Loss is mean squared
error over all 10 future observations and 9 normalized channels. Every fit
receives 300 optimizer updates. Use the final checkpoint, without development
selection or test-based revisions. Retain all losses and initial/final weights.
Development is descriptive. Test arrays are opened for scoring only after all
15 neural fits are terminal. All state resets at window boundaries.

The direct ridge reference uses the last 16 observations, corresponding past
actions, and known future actions only up to each prediction step. Fit separate
linear outputs for all 10 horizons with penalty 1 on non-intercept weights,
using training windows only. No hyperparameter selection. It is deterministic
and does not need three identical fits. Hold-last repeats the root observation.

## Endpoints and continuation

Primary endpoint: mean normalized squared error over the 10-step forecast,
all channels and all held-out windows. Also report errors at every horizon,
each fit, and per original trajectory. Normalized errors are not directly
comparable to paper metrics in physical units.

The transported candidate merits a larger, separately frozen experiment only
if it lowers mean primary error by at least 5% against every other neural
family, ridge16 and hold-last, improves over every paired neural fit and both references across
all three seeds, and has median complete-window predictor latency no more
than 1.5 times the GRU. These are practical pilot thresholds, not significance
tests. A result against only the decay ablation is insufficient.

Timing measures one-window initialization, context assimilation and ten-step
prediction on CPU, with three warmups and 100 observations per fitted model.
Report all samples, median and p95. Data preparation, normalization and service
overhead are outside this predictor timing; training wall time is separate.
This is not a streaming per-action latency claim. No native policy return,
confidence calibration, conformal coverage or topology advantage is scored.

Before training, hash-bind this protocol, implementation, tests and prepared
data. Use an exclusive output directory. Preserve any failed attempt without
silently retrying or replacing seeds. Reports use saved predictions only.
