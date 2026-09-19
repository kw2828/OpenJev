# Adaptation-aware pose dynamics: development protocol

September 19, 2026. This is a new, bounded experiment on exposed development
archives. It cannot rescue the previous failed experiments or establish novel
architecture, calibrated uncertainty, real-robot control or fresh generalization.

## Hypothesis

Learning a feature space through the complete forecasting loss after a context
regression update may make that update more useful than adapting features that
were trained only for static prediction. This is closely related to
[ALPaCA](https://arxiv.org/abs/1807.08912),
[Learning to Adapt](https://arxiv.org/abs/1803.11347), and
[Adapting Neural Robot Dynamics on the Fly](https://arxiv.org/abs/2604.04039).
The regression mechanism itself is established prior art. This study tests
whether the geometry-correct, multistep training recipe produces useful gains
beyond simpler local regression and static prediction.

The [saved-output diagnosis](pose-adaptation-diagnosis.md) found stronger motion
changes in the development archives and growing forecast-correction errors.
It also identified the previous CV16 starting-motion estimate as a limitation.
Every newly trained arm here starts with CV1, so adaptation is not credited
merely for changing that estimate.

## Inputs, state and causal labels

Use the unchanged residual-dynamics-v1/data-01 bindings: 720 training windows
from the old archive, and 160 windows from ten parents in each plain and zigzag
archive. Both evaluation archives are already exposed development data. The
old 27-window development split is not used for parameter or checkpoint choice.

Recover physical xyz and SO(3) rotations from the stored observations exactly
as in pose-transport-v1. The Euler convention Rz(yaw)Ry(pitch)Rx(roll) is inferred
from Bullet documentation, not authenticated collection code. Each model step
represents nominally 20 ms, with all ten ordered four-torque samples retained
as a 40-dimensional action. Future recorded applied torques are provided for
offline conditional prediction; this does not establish controller access to
issued commands or a causal control result.

Context fitting receives exactly observations 0 through 31 and action blocks
0 through 30. Forecasting receives only the fitted state and actions 31 through
55. Future poses never enter the forecast API. CV1 starts from actual motion
between observations 30 and 31, spanning one interval.

Support rows are t=1 through 30, exactly 30 completed transitions. Each feature
uses pose t, backward motion from t-1 to t and action block t. Its target uses
the next measured pose t+1. Spatial displacement/angular increments are first
subtracted in the same world frame, then expressed in body frame R_t:

```
dp_t = p_t - p_(t-1)
w_t  = Log(R_t R_(t-1)^T)
y_t  = [R_t^T(dp_(t+1)-dp_t), R_t^T(w_(t+1)-w_t)] / change_scales
```

These targets are changes in per-step increments, not SI accelerations.
Four component-pooled RMS scalars, one per vector type, are computed only from training windows using
the frozen training_scales helper, with its 1e-5 floor. No per-coordinate
sine/cosine standardization enters the output or physical loss.

## Architecture and controls

Learned features use body-frame vertical, body motion and the current ordered
torque block: 49 inputs, a 48-unit tanh layer and 12 tanh outputs. Normalize
the twelve outputs by their L2 norm with epsilon 1e-8, then append an intercept
of one. A trainable zero-initialized 13-by-6 matrix is the global prior W0.
The public-feature control applies tanh and the same normalization to all 49
public inputs, then appends an intercept, producing 50 features and a 50-by-6
trainable prior. There is no learned precision or uncertainty claim.

With fixed unit ridge precision, fit one context posterior:

```
Wpost = W0 + (Phi^T Phi + I)^(-1) Phi^T (Y - Phi W0)
```

Use a float64 Cholesky solve and convert the result to float32. The outer
training loss differentiates through the solve and all 25 future steps.
Normalized features bound the normal matrix condition number by 61 in exact
arithmetic: each of 30 rows has squared norm at most two, and the minimum
eigenvalue is at least one. This is conditioning of a regression solve, not
stability or calibrated predictive uncertainty.

At a forecast step, query the frozen weights using the private predicted state
and current action. Rotate the six scaled corrections into world coordinates,
update displacement/angular increments, and advance position and orientation
using Exp(w_world)R. Never fit on a prediction or update the posterior during
imagination. Reset fast state for every window.

Train four arms for three seeds each:

- **meta:** learned features and prior, trained with context adaptation.
- **static:** identical learned-feature architecture and paired initial weights,
  trained without adaptation.
- **public:** fixed public features; train its prior through adapted rollouts.
- **gru:** freshly train the previous body-coordinate GRU with CV1 initialization.
  This is a stronger static recurrent control, not an inherited old checkpoint.

Evaluate two additional zero-fit ablations: **meta_prior**, the same meta
checkpoint with adaptation disabled; and **static_adapt**, the static-trained
checkpoint with the identical context solve enabled. All precision and feature
normalization choices are shared. Differences in public/GRU feature capacity
and parameter count are reported, not described as parameter matched.

Retain hold, world CV1 and CV16, rooted LS16, proper SE(3) body twist16, and
torque-conditioned direct residual ridge16. Refit the latter on training data
only before any development-panel forecast. Use the frozen reference code from
the previous study. CV1 uses two observations over one interval; CV16 and
body16 use sixteen observations spanning fifteen intervals.

## Fixed budget and objective

Seeds 1101, 1202, 1303; 12 new fits, 30 epochs, batch 32, 690 updates per fit
and 8,280 total. Paired seeds share batch order. Adam learning rate .001,
gradient norm cap one, float32 models, one CPU thread, deterministic operations.
Select the final checkpoint only. No retries, replacement seeds, extra epochs,
hyperparameter sweeps or development-driven selection.

Train on all 25 forecast steps with the fixed physical loss
||p_pred-p_true||^2/(0.1 meter)^2 + angle(R_pred^T R_true)^2/(0.1 radian)^2.
Report position RMSE in meters and geodesic rotation RMSE in radians separately.
Pool squared errors across fits/windows/horizons before square roots.

All fits and ridge fitting finish before development forecasts begin. Evaluate
24 configurations per panel: eighteen neural/configuration rows and six
references. Measure 20 full batch-one forecasts per row after three warmups.
Include fitting the context posterior and all 25 rollout steps. Each neural
family has 120 timing samples; each reference has 40. Loading, normalization
and service overhead are excluded. Save initial/final weights, all losses,
predictions, targets and timing samples with hashes and a completion record.

## Seventeen grouped continuation requirements

For each of four panel/physical-endpoint combinations, require:

1. At least 10% pooled RMSE improvement over every designated control.
2. Every one of the three paired-fit MSE comparisons is nonworse.
3. At least eight of ten parent MSEs are nonworse against every control.
4. Every leave-one-parent-out pooled MSE is strictly lower than every control.

Controls are the other five neural configurations and all six fixed references.
The final requirement is pooled median complete-window meta latency at most
1.5 times the newly trained GRU. All 17 groups must pass. Preserve all 661
underlying comparisons; grouping changes presentation, not stringency.

The independent auditor uses saved outputs only and independently checks data,
initial-weight pairing, reused-checkpoint identities and numeric reductions.
A failure stays failed. A pass would justify confirmation on new data, not
establish a novel mechanism or alter any earlier failed result. Raw/prepared
data, targets and source-derived forecasts stay local because upstream data
licensing is not explicit; our code, weights and numeric error reports can be
published.
