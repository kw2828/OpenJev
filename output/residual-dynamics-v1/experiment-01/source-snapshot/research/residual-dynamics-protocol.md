# Residual dynamics with causal online correction

Prospective local protocol, September 19, 2026. A new experiment after the
[failed action-filter pilot](action-filter-results.md), whose sources and
results remain unchanged. The [diagnosis](action-filter-diagnosis.md) identified
strong contraction and poor training fit. This study tests a stronger common
forecasting baseline before attributing gains to a new memory architecture.

## Mechanism and prior work

A GRU predicts observation increments with an exact persistence skip:
`next = current + delta_scale * readout(next_hidden)`. The readout starts at zero.
Observation updates use orientation and the actual previous-to-current
observation difference. Action updates use orientation, the last difference
and the complete next block of applied torques. Absolute xyz coordinates
remain in the prediction accumulator but are excluded from learned inputs.
This makes the learned dynamics invariant to position offsets, rather than
allowing them to memorize absolute progress along a training trajectory.
It also removes explicit position-dependent terrain information.

The same saved GRU is compared with three analytic residual corrections:

1. A scalar-feature online bias correction.
2. Recursive least squares (RLS) on explicit orientation, difference and action
   features, normalized to unit length with an appended intercept (56 features).
3. RLS on the learned action-prior hidden features, normalized similarly
   (65 features). This is the candidate, called latent RLS below.

Each head starts from zero weights and identity inverse precision, with no
forgetting. Update only after a real context transition, using its uncorrected
forecast error in delta-scale units. At the forecast root, freeze the head and
its covariance. The correction changes only the private predicted observation
and difference carried into the next step. Predicted observations never become
adaptation targets. Every window starts with an empty head and zero recurrent
state. There is no gradient-based update during deployment.

This is an established family of mechanisms, not a novel invention.
[ALPaCA](https://arxiv.org/abs/1807.08912) learns features and Bayesian priors
for online regression; our features are trained for ordinary forecasting and
the RLS prior is fixed. We do not reproduce ALPaCA's meta-training or claim
calibrated posterior uncertainty. [Dynamics meta-adaptation](https://arxiv.org/abs/1803.11347)
and [MTS3](https://arxiv.org/abs/2310.18534) are further close precedents.
JEPA, biological connectivity and RL are not introduced in this comparison.

## Data

Use the numeric archives from official HiP-RSSM revision
`f4adef0ce5ed4b37c9847973df320f72965bd330`, identified by exact paths and hashes
in the preparation script and receipt. These are simulated PyBullet trajectories,
with no per-frame timestamps. The paper/configuration support nominal 500 Hz.

- Train: old `sin2/ts_002_50x2000_w_grad.npz`, trajectory IDs 0-29.
- Development: IDs 41-49 from that archive.
- Exclude IDs 30-40, which were already exposed as tests in the old study.
- New test panels: all 10 trajectories from each of the plain and zigzag
  `sin_infer` archives. They have not previously been scored in this project.

Integrity inspection found no exact xyz/Euler pose-frame overlap between the
new archives and the old full archive, or between the new archives. The other
old `sin2/ts_002_50x2000.npz` was discovered to duplicate the old data and is
not a fresh test. Filenames and distinct frames do not prove collection-seed,
terrain or physical-regime independence; do not claim those properties.

Use xyz and sine/cosine of three Euler angles; never use the slope channel,
velocities, joint positions, joint velocities or reaction forces. Normalize
using all frames of training trajectories only. Delta output scales are
training standard deviations of ten-raw-step observation differences, divided
by observation scales, with a floor of .001. No test statistics enter scaling.

Sample observations every ten raw frames (nominal 20 ms). Each action retains
the ordered ten-by-four applied-torque values between consecutive sampled
observations, flattened to 40 dimensions. Applied torques are not authenticated
issued commands. This remains conditional offline prediction, not a demonstrated
deployable causal model or robot controller.

Each window contains 32 context observations (620 ms between endpoints) and
25 future observations (500 ms). Train on starts 0,50,...,1150 in each training
trajectory: 720 overlapping windows. Development uses starts 0,594,1189:
27 nonoverlapping windows. Each new test trajectory uses 16 nonoverlapping
starts, integer linspace from 0 through 9439: 160 windows per panel. Window
errors share parent trajectories and are not independent experimental replicates.

Downloaded and normalized data remain local because the upstream repository
does not supply an explicit data license. Publish provenance, own code and
weights, and aggregate/error measurements, not source observations or targets.

## Training and controls

Train GRU64 and a separate identical GRU reconstructed from only the last
16 context observations. Use seeds 411, 512, 613, identical initialization and
batch order within each pair, 30 epochs, batch size 32, Adam at .001, gradient
norm cap 1, CPU float32 with one thread. All six fits use the final checkpoint,
with no development selection. Each gets 690 updates (4,140 total).
Loss averages normalized observation squared error over all 25 future steps
and all nine channels. No online correction is used during slow-weight training.
All six neural fits and the ridge fit finish before opening test arrays to score.

Compare no correction, bias RLS, public-feature RLS and latent-feature RLS on
each of the same three full-context weights. The three separately trained
history16 models are an additional control. They cannot read earlier observations
or the action leading into their first observation. This is six trained models
and fifteen evaluated neural configurations, not fifteen trained models.

Add direct ridge16: separate linear residual prediction at each horizon, unit
penalty, unpenalized intercept, trained only on training windows. Features are
the last 16 observations with xyz relative to the current root, their intervening
actions, and future actions only through the predicted endpoint. The dual linear
solve is an algebraic implementation choice. Add hold-last as a second reference.

Equal backbone weights do not match total state or computation. RLS adds
`d*d + d*9` matrix entries, plus an update counter: bias 10, public 3,640, latent 4,810
entries before the counter. Count observation accumulators and the GRU state
separately; history16 also requires an online input buffer. No equal-memory claim.

## Endpoints, cost and continuation

Primary error is mean normalized MSE over all 25 future observations and 9
channels, separately for each test archive. Also retain all horizon errors,
fit errors, and errors grouped by source trajectory. Do not compare these
numbers with the old 20 ms experiment or claim to reproduce the source paper's
benchmark metrics.

Continue the latent-RLS hypothesis only if:

- It lowers family-mean primary error by at least 10% against each of the four
  other neural configurations, ridge16 and hold-last, on both panels (12 checks).
- Every candidate fit improves over the corresponding paired neural fit and
  both references, on both panels (36 checks).
- Its median complete-window predictor latency is at most 2 times the same
  backbone without adaptation (1 check).

All 49 checks must pass. These are practical pilot thresholds, not statistical
significance tests or a definition of architectural novelty. A bias-only or
public-feature win must not be attributed to learned latent features.

Measure one-window initialization, context reconstruction, every online update
and 25-step private forecast. Use three warmups and 50 measured cases per
fit/configuration/panel; pool 300 samples per configuration for the gate.
Report medians and p95. Exclude data preparation, normalization, service work
and checkpoint loading; report training and ridge fitting costs separately.
This is complete-window latency, not streaming per-action latency.

Hash-bind sources, tests, protocol and prepared data before any training.
Use exclusive output directories, preserve failure receipts and all attempts,
and do not retry, replace seeds or extend training after inspecting results.
Saved-output reporting must not make new prediction, training or simulator calls.
