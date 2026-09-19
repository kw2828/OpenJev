# Differential-drive feedback benchmark: primary-source review

Status at review: source review only. No simulator, model, training, or benchmark calls were made for this review. No upstream implementation or checkpoint was executed or copied into OpenJev. A subsequent independent simulation followed a separate frozen protocol and [failed its opportunity screen](drive-qualification-results.md).

Reviewed on 2026-09-19: [Do Better Imagined Rollouts Mean Better Robot Control?](https://arxiv.org/html/2609.02811v1), including appendices, and its [official repository](https://github.com/rdharini2001/Robot_World_Model/tree/abce1f92a26b2309ad2467a6d0b9933b8d89b50e), pinned to commit `abce1f92a26b2309ad2467a6d0b9933b8d89b50e`. The paper compares dead reckoning, EKF, and GRU/SSM residual observers anchored to either estimator. Its useful evaluation distinction is measurement-corrected replay versus measurement-free prediction versus actual feedback control. Neither forecast error nor longer-outage training consistently determines closed-loop rankings.

## Released task and equations

State is planar pose `(x, y, theta)`, command is `(v, omega)`, and the sampling interval is `dt = 0.1 s`. The released transition uses forward Euler, despite its function docstring calling it exact:

```text
x_next = x + dt * v * cos(theta)
y_next = y + dt * v * sin(theta)
theta_next = wrap(theta + dt * omega)
v_odom = slip * v + Gaussian(0, sigma_v²)
omega_odom = omega + gyro_bias + Gaussian(0, sigma_omega²)
```

Known landmarks provide noisy Euclidean range and heading-relative bearing. A landmark is observed only when active, within 6 m, and outside the sensing blackout. Missing values carry explicit masks. These are the actual [motion and sensor formulas](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/dynamics.py#L14-L85).

The path is `(3 sin(2 pi t/T), 1.5 sin(4 pi t/T))`, with `T = 22 s` for 220 evaluation steps. Pure pursuit advances a monotone path index while its current point lies within the 0.8 m lookahead. It uses `v = 1 m/s` and `omega = 2 v sin(alpha)/0.8`, clipped to `[0, 2.5] m/s` and `[-3, 3] rad/s`. [Path and schedule](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/sim.py#L69-L103), [controller](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/controllers.py#L7-L28).

Nominal evaluation specifies slip `0.92`, gyro bias `0.06`, velocity/rate noise `0.03`, range/bearing noise `0.12/0.06`, six of eight landmarks, and 14-step blackouts every 45 steps with randomized phase. Four separate six-level sweeps vary blackout length, range/bearing noise, landmark count, or gyro bias. These 24 conditions are not a factorial grid. [Exact configurations](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/experiments.py#L52-L70).

## Two concrete qualification blockers

1. **Prediction and measurement timestamps differ.** The [released loop](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/sim.py#L133-L142) forms current-command odometry, senses the pre-motion pose, calls the observer, stores that estimate against the pre-motion truth, then advances the plant. The [EKF predicts before updating](https://github.com/rdharini2001/Robot_World_Model/blob/abce1f92a26b2309ad2467a6d0b9933b8d89b50e/src/ssm_obs/ekf.py#L71-L75). Therefore the motion prediction and measurement do not describe the same timestamp. This static finding does not quantify its effect on the paper's results.
2. **Issued commands are an exact-state shortcut.** The plant is noiseless and starts at a known pose; slip and gyro bias affect only odometry. Integrating issued commands with the declared transition reproduces the plant exactly. The published observer interface does not receive those commands, but a controller can retain them. This prevents using the unchanged setup to establish a necessary memory advantage under the usual public-command boundary.

An independent implementation must declare its own event contract: assimilate packet `t`, issue command `t`, advance the plant, receive interval odometry and packet `t+1`, then predict/update the next estimate. Unknown physical actuation disturbances would remove the command-integration shortcut, but would constitute a declared task change. Do not claim reproduction of the paper's scores.

## Controls and decision boundary

Require command-only integration, pose-only EKF, a parameter-augmented EKF, and finite-window weighted least squares before another neural fit. For the published odometry equation, an augmented state `(x,y,theta,log(slip),gyro_bias)` can propagate with corrected rates `v_odom/slip` and `omega_odom-gyro_bias`. A window estimator can jointly fit poses and nuisance parameters from public odometry and landmark residuals. Their equations must match any newly declared physical disturbance model.

Freeze covariance assumptions, window length, regularization, and insufficient-excitation fallback using training conditions. The published EKF receives condition-specific noise scales, so noise-information access must be matched explicitly. Count retained priors as memory. Private forecasts must not receive future realized odometry or landmarks. Evaluate closed-loop tracking, divergence, and recovery; keep offline prediction diagnostics separate.

The next mechanism question is whether learned history improves causal state/parameter updates and subsequent control beyond these controls. An output residual alone does not establish that mechanism. No architecture novelty or performance claim follows from this review.

## Reuse and resources

The repository contains compact NumPy/PyTorch simulation, small learned estimators, and checkpoints. No `LICENSE` file was found in the pinned tree, and GitHub's repository metadata reported no detected license. The paper's CC BY 4.0 label does not establish licensing of repository code or weights. Any follow-up should use an independently written implementation of the attributed mathematical setup, without copying upstream implementation or checkpoints.

CPU suitability is plausible from the low-dimensional equations and small networks, but no local runtime was measured. No new protocol was frozen and no experimental outcome was produced.
