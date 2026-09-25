# Closed-checkpoint mixing diagnostic

This is a post hoc intervention on three unchanged selected instantaneous LPV
fits, not a trained ablation or a new advancement test. Both already-exposed DEV
recordings, all 22 windows per recording, all seeds and all six joints remain in
diagnostic.json. The original recurrent candidate still fails its rule, with
27/45 conditions passed.

All six baseline forecasts replay exactly before intervention. No model is fit,
no raw MAT is opened, and CONFIRM/TEST remain closed. Dynamic gates are recomputed
from each intervention's own evolving state and current torque.

| Trained-model intervention | DEV1 H128 mean RMSE | DEV2 H128 mean RMSE |
|---|---:|---:|
| Original dynamic mixing of both terms | 0.625821 | 0.679592 |
| Fix matrix mix, retain dynamic input/bias mix | 0.984091 | 0.996928 |
| Retain dynamic matrix mix, fix input/bias mix | 1.095859 | 1.147125 |
| Fix both mixtures | 0.967446 | 1.002712 |

The original function beats each intervention on all six seed-recording totals.
It also wins 35/36 joint comparisons against fixed matrix mixing and 36/36 against
each other intervention. Both parts of the trained shared gate therefore matter
to these forecasts. Dynamic forcing alone does not preserve the gain. Fixing only
forcing is worse than fixing both, showing interactions rather than an additive
error attribution. Retraining a simpler architecture could produce a different
result; these interventions do not establish its achievable error.

The original first-forecast expert-one weight has mean 0.312-0.665 across the six
seed-recording groups, with within-group standard deviation 0.207-0.275. The
trained gate is neither uniformly 0.5 nor constant over these contexts. The two
bounded-coordinate matrices differ by Frobenius norm 0.590-0.656; their measured
float32 spectral norms are approximately 0.9999. These are two distinct learned
operators, not a single transition with only variable additive forcing.

The JSON field physical_coordinate_spectral_norms denotes D^-1 M D in the
original FIT-standardized latent coordinates, not physical degree coordinates.
Its values above one do not contradict the common transformed-coordinate bound.
No full nonlinear Jacobian, incremental-contraction or physical-causality claim
is made. The reported trained bound is an exact-arithmetic design with ordinary
float32 norm roundoff, not a new numerical stability certificate.

The narrow follow-up suggested by this diagnostic is to preserve a shared gate
over both matrix and forcing experts while testing a compact feedforward
scheduler under a fresh training protocol. A matrix-only or forcing-only
trained control could test necessity, but this result does not authorize one
or supply a new quality gate. It provides no evidence for persistent scheduler
memory or biological wiring. Full cumulative H8/H32/H64/H128 and per-joint
metrics, gate statistics, source/input hashes and the original analysis process
record are retained alongside this note.
