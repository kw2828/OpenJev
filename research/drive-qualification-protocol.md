# Differential-drive control: necessary opportunity screen

This independently implemented simulation qualifies a possible research venue.
It is not a neural-model result or reproduction of the cited paper's scores.
The [source review](differential-drive-source-review.md) identified timestamp
misalignment and an exact command-integration shortcut in the released task.
This version aligns endpoint packets and applies unknown physical actuator
gains and actual velocity disturbances. It uses a clocked tracking controller,
rather than copying the released pure-pursuit implementation.

## Fixed task and public information

Euler unicycle motion uses dt=0.1 seconds and 300 actions. Actual interval
velocities are `v=exp(a)*command_v+eta_v` and
`w=exp(c)*command_w+eta_w`, with independent velocity disturbances of standard
deviation 0.03. Odometry reports `exp(kappa)*v+epsilon_v` and
`w+b+epsilon_w`, again with independent standard deviations 0.03.
The four hidden parameters are two physical log gains, log odometer scale and
gyro bias. Sensor calibration stays fixed within each episode, including
physical-gain switches. This avoids making an indistinguishable simultaneous
gain/calibration change into an alleged learnable deficit.

Six identified landmarks lie on an explicit radius-five regular hexagon. Range
and bearing noise standard deviations are 0.12 m and 0.06 rad; visibility has
a six-metre limit and excludes ranges below 0.1 m. Missing values are NaN with
an explicit mask. Initial pose `(0,0,pi/4)` is known to every method. Observers
receive only commands, interval odometry and endpoint packets. Every method
has the same declared noise scales and landmark map. Private random streams,
future packets, true pose and parameters remain outside public observers.

The target is the time-indexed lemniscate
`(3*sin(.22*t), 1.5*sin(.44*t))`. The common controller adds 1.2 times position
error to target velocity, projects it onto estimated heading for forward speed,
and adds three times heading error to target angular velocity. Commands are
clipped to speed [0,1.5] and turn rate [-3,3]. It cannot obtain free progress by
stopping at the path intersection. Every controller, including privileged
references, uses this same nominal command law; oracle gains are not secretly
used to invert actuator dynamics.

## Paired conditions and controls

Twelve seeds, 24101 through 24112, generate six named child streams each.
Initial physical log gains are N(0,.15), clipped to [-.3,.3]; log odometer scale
is N(0,.1), clipped to [-.2,.2]; bias is N(0,.08), clipped to [-.16,.16]. A new
independent pair of physical gains follows the same law. A private switch step
is uniformly selected from 120 through 180. Its distribution is public; its
realization is never an observer input. The public filters use fixed parameter
diffusion rather than an oracle switch detector. A positive screen would still
need a change-point-aware classical comparison before neural claims.

The three panels share each seed's full exogenous arrays: stationary physics;
one physical-gain switch; the same switch with longer landmark blackouts.
Blackouts occupy 14 or 28 steps per 45-step cycle with one seeded phase.
Sensor values and masks differ when controllers visit different states, but
exogenous noise at each time/landmark is paired. Arrays are saved before any
scored episode. All seven methods run all three panels and twelve seeds:

1. Nominal command integration.
2. Uncorrected odometry integration.
3. Pose EKF with nominal sensor calibration.
4. Calibration EKF with pose, odometer scale and gyro bias.
5. Joint EKF with pose, both actuator gains and both calibration parameters.
6. Parameter oracle: the joint observer receives the true current parameters,
   but still estimates pose from exactly the same public sensing interface.
7. Pose oracle: exact pose feeds the same controller after each real step.

Joint EKF explicitly includes the two interval velocity disturbances as
temporary latent variables. It conditions them and the persistent state on
odometry, propagates the conditioned velocity into pose, and then updates with
endpoint landmarks. This handles the correlation between actual motion noise
and measured odometry. Persistent parameter prior standard deviations are
(.2,.2,.15,.12); random-walk deviations per step are (.015,.015,.003,.003).
The calibration-only filter uses the last two values. Parameter oracle has
zero parameter uncertainty and diffusion. Joseph covariance updates, wrapped
bearing residuals and analytic Jacobians are checked before execution.

## Decision rule and limitations

**This is only a necessary first screen. Passing does not authorize a neural
efficacy claim.** A later qualification would need a command-aware moving-window
MAP estimator, change-point-aware identification and evidence of recoverable
information beyond detection delay. That work is unnecessary if the simpler
joint filter already closes the useful parameter-information gap here.

Primary control diagnostic is late time-indexed tracking MSE in square metres.
For stationary runs, late means actions 150-299. For switched runs it starts
30 actions after the private change, leaving at least 90 scored actions.
These evaluator-only windows exclude the immediate undetectable surprise;
they never trigger observer resets. Overall tracking MSE, localization error,
heading error, command energy, maximum tracking error and divergence are also
reported. Divergence means tracking distance above three metres at any step.

Pool by the arithmetic mean of the twelve per-episode late MSE values, giving
each seed equal weight despite differing late-window lengths. A paired win
requires strictly lower oracle MSE, not equality.

Require the following **in each of the two switched panels** before investing
in the next baseline stage: the same-observer parameter oracle lowers pooled
late tracking MSE by at least 10% and at least 0.001 square metres versus joint
EKF, improves at least 9/12 paired seeds, and neither arm diverges. A pose-oracle
gap alone does not pass: it may be irreducible observation uncertainty. These
are prospective practical screens, not confidence tests or an acceptance
criterion selected from observed results. Report every panel and method.

If either switched panel fails, close this fixed configuration for the proposed
recurrent-transition comparison; do not alter gains, blackout lengths, controller
or thresholds after seeing its results. No neural fits, RL calls, teacher calls
or biological graph changes belong to this qualification.

Freeze source/tests and protocol before the 252 scored episodes, retain every
trajectory and failure, and audit event alignment, public controls, exogenous
pairing and metrics from saved arrays. Development tests use seeds outside the
scored range. Simulator steps in scored execution, observer wall time and
reporting work are accounted separately. Synthetic simulation supports no
real-robot performance claim.
