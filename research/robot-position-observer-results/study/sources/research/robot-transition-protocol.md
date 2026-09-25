# Bounded robot transition: development protocol v1

This is a new development experiment following the closed
[joint-coupling failure](robot-coupling-results.md), not a revision of its
20/55 result. The same two development recordings have already been exposed.
Nothing in this protocol opens internal CONFIRM2 or official TEST.

## Question and prior art

Does a bounded, input- and state-scheduled transition improve long forecasts
over an unrestricted GRU residual, and does persistent scheduling help beyond
instantaneous or constant mixing? This qualifies a baseline mechanism. It does
not establish architectural novelty, connectome benefits, robotics control,
RL improvements, or an ICLR submission result.

[ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf) already learns
recurrently scheduled linear transitions and prefix initialization. Its stable
variant uses a shared similarity transform and diagonal transitions. This
implementation uses two dense operators contractive in one learned diagonal
metric, allowing noncommuting transitions. That difference is a hypothesis,
not a novelty claim. [Recurrent equilibrium networks](https://arxiv.org/abs/2104.05942)
provide stronger contraction guarantees that do not transfer to this model.

The saved-output diagnostic fixed bins 1-8, 9-32, 33-64 and 65-128 before reading
forecasts. It finds early error as well as later drift. It is descriptive and
uses the already-exposed data; it is not an independent validation result.

## Data and causality

Reuse only the authenticated saved measurements from `robot-coupling-study-v1`:
FIT7 and DEV2, the old FIT-only normalizers and the FIT-only linear initializer.
The new registration pins every file and all imported scientific source files.
Recompute normalizers from FIT7 and require exact equality with the old values.
Do not decode raw MAT files or the concatenated prepared benchmark arrays.

The full data roster, units and per-recording causal filtering remain in the
[parent protocol](robot-coupling-protocol.md). Each recording has 3,636 filtered
samples at 10 Hz. Use all six positions and total measured torques. These are
realized torques, not authenticated issued commands. This remains offline
conditional forecasting even though each target uses only causally preceding
inputs.

Observe q/u at indices 0..31. Starting from q[31], first future torque u[31]
predicts q[32]. Future torque u[31+k] predicts q[32+k]; later inputs cannot
affect earlier predictions. No target positions enter conditioning or rollout.
Training and evaluation horizon are both 128. Evaluation starts at 64+160k,
22 disjoint context-plus-forecast windows per DEV recording. These are the
same DEV windows as the prior study.

## Models

All learned models use float32. Set all CPU/BLAS thread controls and Torch
threads to one. Normalize in float64 using FIT statistics.

**Bounded LPV, recurrent scheduler.** Latent x=[q6,v6] starts from the last
observed q and last observed one-step q difference. The auxiliary coordinates
are free learned latent coordinates after initialization, not measured physical
velocities. A GRU with 8 hidden values takes [predicted q6,current torque6].
Condition its hidden state using observed [q[t],u[t]] at t=1..30. The next
GRU update sees q[31],u[31] before making the first prediction. Its two-way
softmax mixes two affine experts:

```
D = diag(exp(3*tanh(d)))
K[j] = 0.9999 * W[j] / max(1, spectral_norm(W[j]))
h_next = GRU(concat(q, u), h)
g = softmax(gate(h_next))
x_next = inverse(D) * sum_j g[j]*(K[j]*D*x + B[j]*u + b[j])
q_next = x_next[:6]
```

Compute both spectral norms once per complete model forward, retaining
gradients. Include their cost in measured inference. The fixed diagonal
scaling starts at [1,1,1,1,1,1,10,10,10,10,10,10] but remains trainable via d.
Initialize W[j]=.999I+.003(R[j]-R[j].T)/sqrt(12), B[j]=.01N(0,1), b=0,
and the gate output weights/biases to zero. A local seed controls initialization;
shared expert and scheduler components are paired between relevant controls.

For bounded inputs and finite weights, in exact arithmetic,
||D*x_next|| <= .9999||D*x|| + max_j(||B[j]||*||u||+||b[j]||).
This bounds forced trajectories. It does not prove incremental contraction
of the nonlinear scheduler, bounded gradients, prediction accuracy, or robot
safety. No unrestricted AR2 bypass or output repair is allowed.

**Instant scheduler.** Same stored 1,014 parameters, resetting the GRU hidden
state before every step. The 192 hidden-to-hidden weights are inactive and
must be disclosed. It retains 12 latent values versus recurrent's 20.

**Constant mixture.** Fix g=(.5,.5), omit scheduler/gate entirely. It has 468
parameters and 12 latent values. This ablation has less useful capacity;
the experiment cannot attribute every difference solely to memory.

**GRU residual.** The qualified 1,296-parameter parent implementation, with
trainable FIT-only AR2 initialization, GRU10 and initially zero residual head.
Refit from scratch with the new exposure and identical sampled windows.

## Strong causal regression controls

Fit two conventional float64 direct ridge banks, penalties 1 and 100, including
the intercept in the penalty. Training windows use FIT starts at 64+32k,
context32 and horizon128. At target horizon h=1..128, use the last16 observed
positions, last16 past torques ending at context index30, and only future_u[:h].
Each horizon predicts six outputs with its own coefficients. A shared Gram
matrix and 128 indexed Cholesky solves implement the same independent ridge
solutions. No adaptive jitter, changed penalty, extra fitting or rescue.

Each bank stores 445,440 coefficients (3,563,520 bytes), with 16 bytes of scalar
metadata, 1,536 bytes of required history and 192 bytes of normalization counted
separately. Do not store a padded bank and report the smaller triangular count.
Verify horizon-wise suffix invariance before fitting. This is a conventional
baseline, not a new world model. Also report unchanged frozen linear and
persistence controls. The old noncausal direct ridge is background only and
does not enter this rule.

## Fixed optimization and selection

Three seeds: 8101,8102,8103. Two learning rates: .001,.003. Four families,
24 fitting attempts in seed/rate/family order. Every attempt starts anew.
For each seed, precompute 4,096 batches of 16 FIT windows using PCG64(seed+520000).
Sample recordings uniformly, then valid starts uniformly after the 64-row skip.
All families and both rates use the same windows for that seed.

Adam betas(.9,.999), epsilon1e-8; gradient norm clip1. No teacher forcing during
forecast, no curriculum, early stopping, validation checkpoint selection or
auxiliary loss. Uniform mean squared normalized position error over all128
forecast steps and six joints. This is eight times the forecast-target exposure
of the old 1,024-update/horizon64 study; gains over the old study cannot be
attributed to architecture alone. The fresh controls receive the same exposure.

Save initial/final parameters, every loss and preclip gradient norm, optimizer
arrays, batch indices, failures and elapsed time. Any nonfinite loss, gradient,
parameter or Adam state fails that attempt without repair. A programming/source
error is campaign-fatal. Fit cap600s including construction/preservation; whole
study cap7200s. Preserve failures and do not restart inside this registration.

Attempt all24 fits before loading saved DEV arrays in this run. The DEV files
were previously exposed, so this barrier does not restore independence. Select
one rate per family by pooled H128 normalized RMSE over both DEV files and all
three individual fits; a failed/incomplete recipe is ineligible. Break ties by
lower rate. Select ridge penalty by pooled DEV H128 error; tie by arm name.
Never select a favorable seed or average predictions into an ensemble.

## Advancement rule

The recurrent LPV candidate must satisfy every condition:

- All four selected families and a causal ridge bank are eligible.
- On each DEV file, mean H128 RMSE is at least5% below constant LPV, instant LPV,
  fresh GRU, frozen linear and persistence.
- On each DEV file and each paired seed, H128 RMSE is at least2% below each
  trained control (constant LPV, instant LPV, fresh GRU).
- On each DEV file, mean H128 RMSE is within5% of selected causal ridge.
- On each DEV file, no joint's mean physical RMSE exceeds fresh GRU by10%.
- Median full-request latency is at most twice fresh GRU, and persistent numeric
  storage is no larger. Report the tradeoff; the latency allowance is not a
  speedup claim.

Report H64 and H128 plus per-joint degree errors for every recipe. H64 is
descriptive; selection/rule use H128. All comparisons use means of individual
fit errors, not an ensemble. Report exactly45 Boolean conditions, including
costs. Missing evidence fails the relevant condition.

Measure batch1 normalization, conversion, context conditioning, prediction128,
denormalization and finite checks: 3 warmups,20 repetitions, every selected fit
and all four references. Family latency is median of3 per-fit medians. Count
parameters, state, buffers and normalization; report input/output arrays
separately. Python object overhead, temporary workspace and model/disk load are
excluded explicitly. Parameter count is not a proxy for measured speed.

A pass only qualifies a separately designed confirmation protocol. A failure
keeps confirmation closed and is published unchanged. Neither outcome alone
supports a novel-architecture, biological-learning or scientific benchmark claim.
