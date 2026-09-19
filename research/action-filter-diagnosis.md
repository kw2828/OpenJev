# Action-filter diagnosis and one residual-dynamics follow-up

19 September 2026. Read-only diagnosis of the completed failed pilot. No new model forward, optimization, environment or RNG call was made. Initial/final tensors were loaded with `weights_only=True`; final checkpoint hashes were checked against fit receipts. Training-loss arrays and the already published report supplied the numerical evidence. No test/development arrays or saved prediction arrays were opened. Fit receipts contain development summaries, which were visible while checking their schema; they are not used below. All previous outcomes and frozen sources remain unchanged.

**The immediate problem is an underfitting, strongly contractive absolute predictor. Fix the common dynamics/readout parameterization before attributing anything to fast-weight adaptation.** This diagnosis supports a new exploratory recipe, not continuation of the failed 25-check rule or a new architectural claim.

## Verified source facts and training evidence

The frozen [cell](../src/openjev/research/action_filter_models.py), SHA `9f421020059f0f04db1032c00a2530e1fc9a33fb093c4da04c6993a342f2b5ed`, forecasts as

```text
T(a) = w0(a) I + w1(a) P1 + w2(a) P2       # nonnegative row-stochastic
M_next = rho(a) T(a) M + (1-rho(a)) tanh(B a)
y_next = W vec(M_next) + b                  # absolute normalized observation
```

The decay arm substitutes `T=I`. Both normalize the learned four-dimensional value and sixteen-dimensional key to unit norm. The write is `M += beta k (v-M^T k)^T`. There is no direct path carrying the last measured observation into the forecast, no additive near-identity state increment, and no observation-dependent nonlinear transition during an open-loop forecast. All four value columns share the same row transport. Zero normalized action means the training-mean torque, not necessarily zero physical torque; its drive is exactly zero because `B` has no bias.

For two roots receiving identical future actions, the drive cancels:

```text
Delta M_h = (rho_(h-1) T_(h-1)) ... (rho_0 T_0) Delta M_0
max_abs(Delta M_h) <= product(rho_t) * max_abs(Delta M_0)
```

This is an exact forecast sensitivity bound. It is not a measured distribution of real action trajectories. A further row-span contraction is bounded by the Dobrushin coefficient `d(T)=0.5*max_ij sum_k |T_ik-T_jk|`. At initialization both learned maps are uniform, so `d(T)=w0`: the proposed transport begins as particularly strong averaging.

Static final-checkpoint results at **zero normalized action repeated ten times**, without running a model:

| Paired seed | Initial rho | Final rho | Final rho^10 | Final (rho*d(T))^10 | Last-epoch training MSE |
|---|---:|---:|---:|---:|---:|
| 101 | 0.406112 | 0.700624 | 0.028500 | 0.000314 | 0.617019 |
| 202 | 0.547989 | 0.798255 | 0.105055 | 0.003988 | 0.531077 |
| 303 | 0.554709 | 0.804790 | 0.113979 | 0.0000327 | 0.546826 |

The initial ten-step row-span bounds were approximately `9.95e-9`, `2.60e-7`, and `3.99e-9`. Training raises retention, but strong structural forgetting remains at this reference action. A decoder can amplify surviving directions; this bound does not prove the cause of every prediction error.

The last-epoch loss is the mean of all fifteen minibatch losses, each measured before its update. It is not a new final-checkpoint training evaluation. Corresponding ranges across three fits are **0.4003-0.5839** for decay, **0.01858-0.02016** for GRU, **0.01860-0.02018** for history16, and **0.1813-0.2021** for the diagonal filter. The candidate therefore already fits training windows poorly; its problem is not merely a held-out generalization gap. None of these logs establishes convergence.

The [published outcome](action-filter-results.md) also makes persistence essential: ten forecast steps represent nominally 20 ms, hold-last MSE is 0.025337, and ridge16 reaches 0.019255. The candidate's failed result is unchanged.

Checkpoint identities for the static transport calculations: `101=c4188fc74c5c9a9cd05cf20f50695c3b0e2c8f21c1bdadc3cae54896c0c7c4b3`, `202=5c209229d9a7c97e2be0ce953da76b3f43e10c33f79ef509789d81b1af346fae`, `303=3cc0100456d3f1e8a8ae9ae50bbdcef90425d33e0148b1ccb875e13e3651730c`. Files are under `output/action-filter-v1/experiment-01/run-01/`; loss files use the same fit stems plus `-losses.npy`.

## What is plausible, but not yet established

The model repeatedly erases root information while being asked to reproduce nearly unchanged absolute coordinates. This is a poor starting bias for a 500 Hz forecast. Four-dimensional normalized values additionally remove their radial magnitude. Keys and gates can still encode observation magnitude, so this is **not** proof that the whole encoder has only three degrees of freedom or cannot represent nine channels. A no-normalization comparison has not been run. The shared row operator and small parameter count may also limit dynamics, and 300 updates may be insufficient. These explanations remain hypotheses rather than post-hoc causal findings.

[Gated DeltaNet, v3](https://arxiv.org/html/2412.06464v3#S3.SS1) already combines selective delta writes and learned decay, including a pure-delta limit near retention one. It does not imply that arbitrary half-life initialization or row averaging is appropriate for physical trajectories. [Mamba, v2](https://arxiv.org/html/2312.00752v2#S2) frames recurrent transitions through discretized continuous dynamics; near-identity evolution at small time steps is ordinary state-space reasoning. Neither residual dynamics nor a persistence skip is a new principle.

## Concrete next model and matched controls

Build an additive residual predictor, leaving the frozen cell untouched. All new learned arms receive the same observation-residual output path:

```text
h_prior = h + epsilon * F(h, a)             # or matrix-shaped equivalent
y_pred_next = y_pred_current + D(h_prior, a)
```

At each real observation, set the prediction accumulator to that actual normalized observation and apply a learned correction to `h`. During forecasting, advance only private latent state and the prediction accumulator. The accumulator is a predicted quantity with explicit provenance, never a fabricated measurement. Initialize the increment head to zero so every arm initially implements exact hold-last; initialize the residual state step near identity, for example epsilon=0.01 per sample as a **prospective engineering choice**, not a fitted physical time constant. Use identical choices in all controls, report state/Jacobian norms and finite failures, and do not assert global stability of a nonlinear residual network.

For the delta pair, retain normalized keys but use amplitude-preserving learned values. A shared 8-by-8 matrix is a reasonable fixed 64-float redesign. Make row transport a small residual term `epsilon*(T(a)-I)M`, alongside an identical action-conditioned residual drive. The decay control removes only this transport term and does not compute it. Both receive the same values, timescale, observation correction, decoder and output accumulator. Compare these **new common-parameterization** arms with residual GRU64, separately trained history16, ridge16 and hold-last. Changing several shared ingredients does not isolate which caused the old failure; the new transport/decay contrast isolates only transport within the stronger recipe.

Then the smallest meaningful online-plasticity intervention is a rank-four transition adapter, **on a prechosen residual backbone**, not a confidence gate:

```text
W_effective = W + U4 P V4^T,     P in R^(4x4), P_start=0
```

Obtain fixed factors from the trained transition matrix, or explicitly train their parameterization. Do not leave arbitrary trainable `U,V` untrained behind an always-zero adapter and then claim a trained adaptation space. For a real observed endpoint, take one bounded gradient step on `P` using the preceding saved root, actual intervening actions, and current public prediction error. Replay that same transition under the updated `P`, then assimilate the returned observation once. There are no adaptation targets during imagination. In the offline window pilot, adaptation uses **context observations only**; all forecast targets remain inaccessible until scoring.

Keep slow weights frozen during deployment, reset `P` at declared episode/window boundaries, and copy it into every forecast branch. Compare the same saved backbone with adaptation disabled and with output-only adaptation of matched dimension/update opportunity. Count all gradient, replay, projection, state and copying work. This tests whether correcting dynamics helps beyond an output correction, rather than rewarding extra computation. Step size, radius and regularization must be fixed from training-only development before any new confirmation; no old test is fresh again.

This is close prior art, particularly [Adapting Neural Robot Dynamics on the Fly, v2](https://arxiv.org/html/2604.04039v2#S4.SS2), which adapts low-dimensional neural dynamics for MPC. A first-order sixteen-parameter implementation would be a simpler comparator, not a reproduction or novel invention. On this dataset, recorded applied torques still do not establish issued-command causality, and changing hidden state does not itself establish changing dynamics. Fast adaptation may have nothing useful to identify.

**Decision:** train the stronger residual/skip models in one bounded development comparison before promoting online adaptation. Require a useful result beyond ridge and persistence, retain all seeds and the trained short-history control, and report the old failure intact. If the stronger ordinary predictors already explain the data, stop the fast-weight claim on this recipe. This is model development on exposed evidence, not a fresh generalization result, a connectome result, or a reason to extend the old test until it passes.
