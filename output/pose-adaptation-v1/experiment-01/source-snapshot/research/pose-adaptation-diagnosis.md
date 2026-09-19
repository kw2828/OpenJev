# Pose adaptation: shift, stale motion, and growing correction error

September 19, 2026. Post hoc saved-output diagnosis of `pose-transport-v1`,
which remains **failed at 345/541**. The ordinary body-frame model is the fixed
learned baseline throughout this note. Both plain and zigzag panels are exposed
development data. No new predictions, fits, random draws, or simulator calls
were made. No previous source, data, receipt, or gate was changed.

The evidence supports two problems together: the longer source archives have
larger changes in motion and occasional extreme torques, while the learned
forecast has a weak recent-motion starting point and substantial horizon-dependent
error. It does not identify a single causal explanation or justify scaling
transported memory. A global correction offset would explain only a minority
of the observed squared error.

## Distribution comparison

Prepared observations were unnormalized in float64 using the bound training
statistics. Rotations were reconstructed independently with
`Rz(yaw) Ry(pitch) Rx(roll)`. For adjacent nominal 20 ms observations, define
`dp[t]=p[t+1]-p[t]` and `w[t]=Log(R[t+1] R[t]^T)` in world coordinates.
The following RMS is `sqrt(mean(sum(vector**2)))` over all selected window
intervals. Changes are differences between consecutive increments, not a new
acceleration forecast. Divide displacement changes by `.02**2` to express
them as acceleration under the nominal timing assumption.

| Dataset | Windows | Displacement RMS, m/step | Angular increment RMS, rad/step | Displacement-change RMS, m/step² | Angular-change RMS, rad/step² |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training, old IDs 0-29 | 720 | 0.02993 | 0.01389 | 0.000623 | 0.001541 |
| Original development, old IDs 41-49 | 27 | 0.03002 | 0.01356 | 0.000622 | 0.001554 |
| Exposed plain | 160 | 0.03942 | 0.01027 | 0.001338 | 0.003495 |
| Exposed zigzag | 160 | 0.03376 | 0.01607 | 0.001710 | 0.003084 |

The original development split closely resembles training. Relative to training,
plain/zigzag displacement-change RMS is 2.15/2.75 times larger and angular-change
RMS is 2.27/2.00 times larger. Maximum displacement changes rise from 0.00204
in training to 0.02682/0.03939; maximum angular changes rise from 0.00521 to
0.08388/0.05010. These are distribution differences, not proof of distinct
physical regimes or hidden parameters.

| Dataset | Four-wheel normalized torque RMS range | Overall normalized torque range | Per-wheel samples outside training range |
| --- | ---: | ---: | ---: |
| Training | 0.982-1.048 | -6.744 to 6.244 | Reference |
| Original development | 0.983-1.005 | -6.190 to 5.446 | 0% |
| Exposed plain | 1.179-1.261 | -48.062 to 65.730 | 0.098-0.169% |
| Exposed zigzag | 1.054-1.436 | -14.115 to 58.986 | 0.075-0.515% |

Ranges compare each wheel with its own training minimum/maximum. Each exposed
panel contributes 89,600 raw torque samples per wheel: 160 windows × 56 blocks
× 10 raw samples. The means/RMS remain much less shifted than the extreme
values. These are recorded applied torques, not verified issued commands.
Training windows overlap, and all statistics here are window-weighted; they
are not independent-sample estimates or significance tests.

## Saved forecast errors grow differently

Values below come from authenticated saved squared-error arrays. Body entries
pool all three fits before the square root; references are counted once.
Each cell is **position RMSE in meters / rotation RMSE in radians**.

| Panel and forecast | 20 ms | 100 ms | 200 ms | 500 ms |
| --- | ---: | ---: | ---: | ---: |
| Plain body GRU | 0.00254 / 0.00172 | 0.01190 / 0.02434 | 0.02499 / 0.07692 | 0.10631 / 0.27145 |
| Plain CV1 | 0.00057 / 0.00136 | 0.00562 / 0.00588 | 0.02007 / 0.01472 | 0.11375 / 0.06513 |
| Plain body16 | 0.00152 / 0.00180 | 0.00918 / 0.00958 | 0.02251 / 0.02366 | 0.08662 / 0.08988 |
| Zigzag body GRU | 0.00487 / 0.00733 | 0.02580 / 0.03617 | 0.05713 / 0.07738 | 0.19003 / 0.21162 |
| Zigzag CV1 | 0.00094 / 0.00248 | 0.01086 / 0.02230 | 0.03730 / 0.06888 | 0.18925 / 0.33475 |
| Zigzag body16 | 0.00389 / 0.00853 | 0.02437 / 0.05103 | 0.06127 / 0.12325 | 0.25401 / 0.43815 |

The body model helps long-horizon zigzag rotation relative to both references,
but plain rotation deteriorates sharply. Plain position improves over CV1 at
500 ms but still loses to body16. This is not uniformly harmful correction or
uniformly inadequate motion history.

There is a concrete source-level limitation: immediately before imagination,
the frozen model replaces its latest observed displacement/angular increment
with the secant over observations 16 and 31. This is the declared CV16 skip,
not a newly discovered implementation defect. CV1's much smaller first-step
position errors show that the latest measured motion is informative. A future
adaptation comparison should isolate this initialization choice before claiming
that an online learning mechanism is responsible for an improvement.

To assess systematic signed error, saved prediction-minus-target position
errors and spatial rotation-error vectors `Log(R_pred R_target^T)` were rotated
into each window's observed root frame. At 500 ms, the pooled body means are:

| Panel | Mean position error, root-frame xyz (m) | Mean rotation-error vector (rad) | Squared pooled-mean share of position / rotation MSE |
| --- | --- | --- | ---: |
| Plain | [-0.01850, -0.00022, -0.00786] | [-0.00043, 0.03247, 0.00064] | 3.57% / 1.43% |
| Zigzag | [0.00717, -0.00952, -0.05756] | [0.00482, 0.10002, 0.00219] | 9.57% / 22.40% |

All three zigzag fits share the negative root-z mean (-0.0535 to -0.0622 m)
and positive root-y rotation-error mean (0.0879 to 0.1103 rad). That supports
a repeatable signed component, not a claim that a constant bias correction
would solve the task. Plain rotation's large error has little global signed
bias, so conditional errors and dispersion dominate this decomposition.

## Are recent observations sufficient for adaptation?

They contain useful local information, particularly for translation, but the
evidence does not establish a stable local dynamics law. As a descriptive
check, compare the mean of the last three observed increment changes with
the mean of the next five actual increment changes. Pearson correlations
across windows are:

| Panel | Translational x / y / z | Angular x / y / z |
| --- | --- | --- |
| Training | 0.536 / -0.455 / 0.885 | -0.492 / 0.840 / -0.457 |
| Original development | 0.508 / -0.334 / 0.918 | -0.340 / 0.740 / -0.400 |
| Exposed plain | 0.813 / -0.587 / 0.905 | -0.553 / 0.080 / -0.296 |
| Exposed zigzag | 0.786 / 0.557 / 0.799 | -0.289 / 0.584 / 0.037 |

These correlations use actual context and target poses, not new forecasts.
They are post hoc and depend on the sampled windows. The changing lateral
sign and weak plain angular-y correlation caution against assuming that a
recent acceleration estimate transfers unchanged for 500 ms. Neither rare
torque extremes nor these correlations establish the cause of prediction error.

The narrow next mechanism question is whether causal recent-transition
residuals can regulate a fixed body model's correction beyond a strong recent-
motion baseline. Keep CV1, body16, and a simple public-feature local correction
as controls. Separate a latest-motion initialization change from online
adaptation, and freeze any fast update before private rollouts. Use current
panels only for development. If cheap recent-motion/local-linear controls
explain the improvement, do not attribute it to a new memory architecture.

## Bindings and limits

Before calculation, the protocol-bound data files and all eleven scientific
source hashes were verified; used predictions/targets were checked against
the producer manifest, and report outputs against the audit receipt. No model
weights were loaded. The calculations used NumPy reductions, finite differences,
and pose-error arithmetic only. The following SHA-256 identities bind the
inputs; the protocol/receipt maps bind the individual saved row files.

| Input | SHA-256 |
| --- | --- |
| Frozen protocol | `ac59480f11927a5aa0fd50af6eb4635d4ebb7600d7618ab0201aac99e8bc61c3` |
| Producer completion | `c940616fa1e0970e4ca8e31694c64c68dfbdf7a622532f177cbe4e4148f34fdf` |
| Audit receipt | `09ad35d9805170eca1649147058847a898061624898b84b640c85513c99bde07` |
| Saved error arrays | `90a33f143a1bd7af20fc36ded2a52d315ba72ce88bf2a20894a74855ed8d1785` |
| Data manifest | `7471d4ec0920dd1c2c0d8d9b35949e9d157d6fb674a4229ca574a6b925faaa5b` |
| Training arrays | `f6b049365f931d1ead0806f53883ae35a5fccb63b5e6fd75df5d782833bfe9b4` |
| Original development arrays | `30d259ef22f231e271a0c44c4fbfe0b25a01ad3ad576c67df91731c9f665f563` |
| Exposed plain arrays | `1a402646581edbd6152c51db090f401e357c744db9d2e8c10ce2a37ec0f077f6` |
| Exposed zigzag arrays | `7ad0ad932c480add80952ad22a176039f6ffa1319106f6d3a3d0b7abf5373407` |
| Normalization | `0e07ba075c29987455cedb11971b95ce92f9509bee99b2795439ab4daf11beec` |
| Body model source | `e8c3cbb651341aee8f28d336acc26de03a6905ad21eb0a2e3daa2968eb149050` |
| Trainer source | `760a3808ea2fe9fbb1bdf239bc147a52c0603523d093f22ab9ce892101f4042a` |
| Geometry source | `d93e93911888a6e99f905474182e33851e7ecc05f5defd74b76b715ad905873a` |

This is a simulated, exposed-data diagnostic, not a new gate or causal finding.
The timing is nominal, the Euler convention is inferred, and archive independence
does not establish terrain or collection-seed independence. Raw observations,
torques, targets and predictions remain local. No added metric changes either
completed study's outcome.
