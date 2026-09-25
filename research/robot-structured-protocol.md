# Structured robot transition: development protocol v1

This follows the closed [bounded-transition study](robot-transition-results.md).
Its recurrent scheduler failed 18 of 45 conditions. The simpler current-state
scheduler was more accurate than GRU10 on average but slower. This experiment
tests a smaller transition and a stronger GRU32 control. It does not revise the
earlier outcome. The same two DEV recordings have already informed model design;
internal CONFIRM2 and official TEST remain closed.

## Question and prior art

Can a current-state scheduler driving four Householder reflections preserve
forecast quality while reducing complete-request latency and persistent numeric
storage by at least 20% relative to matched bounded dense transitions?

Input-dependent products of Householder matrices are established prior art,
including [DeltaProduct](https://arxiv.org/abs/2502.10297),
[orthogonal recurrent networks](https://proceedings.mlr.press/v70/mhammedi17a.html),
and [RUM](https://aclanthology.org/Q19-1008/). Learned scheduling of linear
transitions is also established in
[ReLiNet](https://www.ijcai.org/proceedings/2023/385).
This is a test of a concrete accuracy/compute tradeoff, not a claim that this
combination is a novel architecture, a biological model, RLCD, or a robot policy.
State-dependent scheduling prevents a claim of parallel linear-scan training.

A saved-checkpoint diagnostic fixed either matrix mixing, forcing mixing, or
both at equal weights. Each intervention worsened all six selected
seed/recording totals. Exact original forecasts replayed first. This motivates
preserving both paths; it does not establish the necessity of either path after
retraining, independent validation, or additive causal attribution. The new
registration pins that diagnostic and its process record before any new fit.

## Data and inherited controls

Use exactly the saved FIT7 and DEV2 arrays authenticated by the two parent
registrations. No raw MAT decoding or concatenated prepared data. The
[parent data protocol](robot-coupling-protocol.md) defines per-recording causal
filtering, 10 Hz sampling, all six position channels in degrees and total
measured torque. Recompute FIT-only normalizers and require exact equality with
both parents. Measured torque is not an authenticated issued action: this is
offline conditional forecasting, not a causal control world-model benchmark.

Context32 observes q/u[0..31]. Starting from q[31], u[31] predicts q[32]; future
torque k predicts position k+1. No future target enters conditioning or rollout.
Horizon128 is used for training and evaluation; H64 is descriptive only. Each
DEV recording has 22 disjoint context-plus-forecast windows starting at 64+160k.

Reuse the exact three parent batch files, 4,096 updates of 16 windows, seeds
8101,8102,8103 with window offset520000. Verify them against the deterministic
parent sampler. Reuse the FIT-only linear initializer and both causal direct
ridge banks, penalties1 and100. Do not refit these references.

Retain all six parent `lpv_instant` fits as `legacy_instant`: both learning rates
and all three seeds. Copy their initial/final weights, Adam state, trace and
fit receipt byte-for-byte, joined to the original closed manifest and audit.
Load them using the original implementation. No legacy refit. Preserve array
memory layout when loading checkpoints and the linear initializer.

## New models

All learned parameters and states use float32. Normalize and compute reported
errors using float64. Only explicit caller state persists; no hidden cached
trajectory. Set Torch and all five inherited CPU/BLAS thread controls to one.

The four transition models initialize x=[q_last, q_last-q_previous]. These
12 coordinates are learned latent values after initialization, not physical
velocities. They use D=diag(exp(3*tanh(d))), initially six ones and six tens,
and paired forcing weights B[j]=.01*N(0,1), biases0. A two-way gate reads the
predicted current q and current torque. It does not retain historical gate state.

**Compact reset-GRU gate.** This computes the parent GRU12->8 with its previous
hidden state set to zero, followed by Linear8->2 and softmax. Remove the 192
inactive hidden weights and fold the reset/update bias pairs, leaving338
parameters. Inference is algebraically equivalent to the original gate, tested
with float64 and float32 tolerances. Folded biases change optimizer coordinates:
the new Adam trajectory is not claimed equivalent to the old training.

**Householder candidate,630 parameters.** Two expert prototypes each contain
four 12-dimensional reflection vectors and12 decay values. Each vector has one
coordinate fixed at1, anchors(0,0,6,6), with11 free tanh coordinates. For g in
the two-way simplex, interpolate the prepared vectors and decays:

```
v[k] = sum_j g[j] * insert_anchor(1, tanh(raw_vector[j,k]))
rho = sum_j g[j] * .9999 * sigmoid(raw_decay[j])
z = D*x
for k in 0..3:
    z = z - 2*v[k]*(dot(v[k],z)/dot(v[k],v[k]))
x_next = inverse(D) * (rho*z + sum_j g[j]*(B[j]*u+b[j]))
q_next = x_next[:6]
```

Prepare tanh prototype vectors and sigmoid decays once per complete forward;
retain their gradients. Anchors give denominators at least1 without repairs.
Initialize independent reflection parameters in identical pairs using local
seed `seed ^ 0x484F5553`; initialize decays to.999. The operator is therefore
.999I in exact arithmetic. Four reflections restrict rank(Q-I) to at most4;
this is less expressive than an arbitrary dense transition. The gate spans
only a one-dimensional mixing family.

**Bounded dense,806 parameters.** Same compact gate, D, B and biases. Two
12x12 matrices K[j]=.9999*W[j]/max(1,spectral_norm(W[j])); mix and apply them
before the same forcing term. Compute both norms once per forward, including
their measured cost. Initialize W[j]=(.999/.9999)I.

**Unbounded dense,806 parameters.** Identical construction except K[j]=.9999W[j]
without spectral normalization. **Dense MLP,590 parameters.** Bounded dense
transitions with a tanh12->8->2 gate,122 parameters, instead of compact reset GRU.
All gates have initially zero output heads. All four new models therefore start
at .999I in real arithmetic, not necessarily bitwise across implementations.
The archived legacy control has a different initial operator and optimizer
coordinates; it is a performance anchor, not a single-variable ablation.

For bounded inputs and finite parameters, bounded variants satisfy
||D*x_next|| <= .9999||D*x|| + max_j(||B[j]||*||u||+||b[j]||)
in exact arithmetic. This bounds forced trajectories, not the nonlinear
state-dependent Jacobian, incremental contraction, gradients or robot safety.
No output clipping, adaptive repairs or unconstrained residual bypass.

**GRU32 residual,5,916 parameters.** Widen the qualified GRU residual from10 to32
hidden values, retaining its FIT-only trainable AR2 initializer and zero residual
head. Conditioning and recurrence otherwise retain the parent equations. It has
50 persistent state values versus12 for each transition model. This is a stronger
capacity control, not a parameter-matched model.

## Fitting, failures and selection

Fit five new families x two rates(.001,.003) x three seeds =30 fresh attempts.
Each gets4,096 Adam updates, betas(.9,.999), epsilon1e-8, gradient clip1, batch16,
and normalized position MSE over the full128-step forecast. Training windows
are paired exactly with the parent and across new families. Execution order is
seed, rate, then householder/bounded/unbounded/MLP/GRU32. Per-fit cap900 seconds;
whole campaign cap10,800 seconds. No restart, rescue, extra fit or seed promotion.

Retain every initial/final checkpoint, optimizer state, trace and failure.
Translate only the exact documented nonfinite transition-output guard to the
existing numerical fit failure. Schema/programming failures stop the campaign.
Close all30 new attempts and copy all6 archived fits before loading DEV for
this campaign. This barrier does not undo previous DEV exposure.

For each family choose one rate by pooled H128 standardized RMSE over both DEV
recordings and all three seeds, ties to lower rate. All six seed/recording rows
must be valid for eligibility. Reselect the legacy rate from its unchanged six
fits. Choose a ridge penalty by pooled DEV error, ties by arm name. No ensemble
or per-seed rate selection. Save all160 H64/H128 metric rows, including four
reference models. Nonfinite or missing evidence fails relevant criteria.

## Prospective continuation rule

All61 conditions must pass. One requires all six family recipes and causal
ridge to be eligible. On each DEV recording, the Householder candidate must:

- Have mean H128 RMSE at most1.02 times each of the five trained controls:
  bounded dense, unbounded dense, dense MLP, GRU32 and archived instant LPV.
- Have each paired-seed RMSE at most1.05 times each of those five controls.
- Beat frozen linear and persistence mean RMSE by at least5%, and stay within
  5% of the selected causal ridge bank.
- Have mean physical RMSE on each of six joints no more than10% worse than GRU32.

These are29 conditions per recording. Two final conditions require complete
request latency and total persistent numeric storage each at most80% of bounded
dense. Means refer to errors of individual fits, not averaged predictions.
The criterion tests accuracy preservation plus cost reduction, not superiority
on every accuracy metric. It is deliberately stricter than parameter counting.

Retiming uses the current host for all selected new and archived models and all
references: batch1, first DEV window,3 warmups,20 repetitions. Include input
normalization, conversion, context conditioning, horizon128, denormalization,
outer parameter validation and finite checks. Family latency is median of the
three per-fit medians. Clear all model gradients before timing. Count parameter,
state, buffer and normalizer bytes; report request9216B and output6144B separately.
Python objects, temporary workspace, disk/model loading and optimizer storage
are outside the deployment count. Report raw timings and runtime/host identity.

Four reflection steps may be slower than tiny optimized dense kernels despite
fewer arithmetic operations. Measure eager CPU execution without a Rust or
compiled-kernel claim. The storage target compares2760B against3464B before any
undisclosed persistent additions; every actual buffer is counted.

A pass only qualifies a new confirmation protocol. A failure leaves confirmation
closed. Publish the original outcome and all candidate recipes. Neither outcome
by itself establishes architectural novelty, calibrated decision probabilities,
connectome benefits, robotic control, RL gains or an ICLR-ready result.
