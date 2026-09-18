# Learned world models for reaching with missing sensors

This pilot asks whether a trained recurrent world model improves control when
joint observations disappear. It follows the [literature review](connectome-world-model-robotics-reading.md)
and the [closed Pendulum qualification](robotics-pendulum-qualification.md).
It is a conventional baseline experiment, not evidence for biological wiring
or a new reinforcement-learning algorithm.

## Task and information

Use Gymnasium Reacher-v5 with its native 50-step horizon, reset distribution,
physical model and distance-plus-action reward. Both reward weights are set
explicitly to 1, matching the [version 1.3.0 constructor](https://github.com/Farama-Foundation/Gymnasium/blob/v1.3.0/gymnasium/envs/mujoco/reacher_v5.py).
The prose documentation says the action weight is 0.1; the installed and tagged
code say 1. The code is authoritative for this experiment.

The modified observation interface exposes angle cosine/sine, a static target,
a validity bit and sensor age. It removes velocities and fingertip-to-target
displacement, including indirect clues during observation gaps. Missing angles
are zero placeholders with validity false. Two gaps last six decisions during
training and ordinary evaluation, and ten during shifted evaluation. The
native time step is 0.02 seconds.

Independent Gaussian actuator noise, standard deviation 0.05, is added after
command clipping and before applied-action clipping. The native reward uses
the applied action. Its value is a training target, never a deployed model
input. This is a documented partial-observation variant, not an unmodified
Reacher leaderboard result.

## Models and training

Compare an autoregressive history MLP, a GRU world model and a small Gaussian
recurrent state-space model. The first retains a 12-step window of measurements,
estimates and actions. Predicted estimates can feed subsequent predictions
with validity false; its effective memory is not strictly limited to 12 steps.
It is deliberately stronger than a window that forgets everything during
imagination. The GRU and RSSM separately assimilate real observations and
advance state under candidate actions. RSSM evaluation uses distribution means,
so this does not test uncertainty-aware planning.

All three receive the same 768 exploratory episodes, 12 epochs, 32 episodes per
batch, Adam at 0.001, and three paired seeds. There are nine fits, each with 288
updates. Train on observed next angles and executed reward, plus a five-step
open-loop prediction loss. Missing angle targets are masked. No hidden velocity,
simulator-state target or expert action label enters the learning loss.

Half the collection episodes use a supplied-physics inverse-kinematics/PD
controller with exploration noise. The remainder use two scales of held random
actions. This provides shared behavior coverage; the collector's privileged
information is disclosed rather than attributed to model learning. Prediction
evaluation uses 96 separate episodes. Only final checkpoints are evaluated.

Parameter counts differ: history 32,645, GRU 36,805, RSSM 40,709. Equal data and
updates are not equal computation or capacity. Report full decision timing and
these counts; do not call this an exactly capacity-matched architecture test.

## Control and continuation rule

Evaluate every fit on 32 paired untouched cases under full angle sensing,
ordinary gaps and longer gaps. Use identical candidate action plans, a 12-step
horizon and 64 candidates. Predicted rewards are clipped to the physical range
[-2.5, 0]; there is no terminal value. This is learned-model predictive control,
not online policy-gradient training.

References use the native physical model with either true current state or a
32-particle estimator that receives only public history. The particle estimator
uses a known noise distribution, approximate observation weighting and angle
reanchoring. It is not an exact Bayesian filter. Both plan under zero future
disturbance and have additional model knowledge. Zero and uniform actions are
floors. GRU/RSSM interventions clear learned memory at blackout onset.

Advance a recurrent kind only if the known-state reference is competent, every
fit improves on zero-action control by at least 10%, mean cost improves over
the history model by at least 5% in ordinary and shifted panels, memory reset
increases mean cost by at least 5% in both, and every fit beats angle persistence
by at least 10% on masked one-step prediction error. All cases and fit seeds
remain in the report. Episode bootstrap intervals are descriptive and do not
replace training-seed uncertainty.

The complete run has a cooperative 30-minute cap and no retries, replacement seeds,
extensions, checkpoint selection or test-based tuning. All fits complete before
control evaluation. Checks occur between optimizer batches, prediction roots,
control steps and native-MPC cases; an individual native operation is not
preempted. An over-cap execution cannot report success. Failure preserves its
evidence and does not justify a
connectome sweep. A later topology experiment needs its own protocol and
matched biological, rewired and conventional dynamics.

This comparison follows the need for strong recurrent controls emphasized by
[Ni et al.](https://arxiv.org/abs/2110.05038) and the distinction between filtering
and control illustrated by [Recurrent Kalman Networks](https://arxiv.org/abs/1905.07357).
Neither this small RSSM nor the particle estimator reproduces those systems.

## Runtime and evidence

The [isolated dependency lock](robotics-requirements.txt) leaves the live chess
runtime unchanged. The frozen protocol binds source files, package versions,
native task source and XML. The audit uses saved trajectories, predictions,
candidate scores and weights. It replays physical transitions and recomputes
metrics without new training or planner decisions. On-policy prediction errors
are reported separately because a planner can exploit errors that are rare in
the exploratory prediction set.

Status: implementation and engineering verification. No scored results yet.
