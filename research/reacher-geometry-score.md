# Reacher: does reward scoring hide useful recurrent predictions?

Status: implementation and engineering validation. No scored result yet.

The [explicit-cache study](reacher-cache-ablation.md) failed its recurrence gate.
The cached MLP controlled the arm better, although the persistent GRU predicted
missing angles more accurately. This follow-up tests a specific explanation:
the learned reward head may prevent the planner from using those predictions.

## One intervention, no new training

Reuse all three persistent GRUs and all three cached MLPs from the completed
cache study. Keep their weights, transitions, public observations and CEM256
planner unchanged. Compare their original learned rewards with a geometric
score decoded from each model's predicted angles. Both modes still compute
the original reward head; geometric scoring adds measured work.

The geometric score is negative target distance minus the expected cost of
the noisy, clipped actuator command. It is approximate: forward kinematics
from projected angles does not reproduce MuJoCo's cached-body reward exactly,
and distance at the predicted angle is not expected distance under uncertainty.

## Fixed comparison

First inspect action rankings at 48 preselected roots from exposed cache-study
trajectories. All models receive the same actual public history. A shared set
of 64 initial sequences and the 12 planner selections forms a finite union.
Replay each unique sequence under four common, fresh actuator-noise branches.
These diagnostic results are descriptive and cannot change the experiment.

Then run 64 fresh paired cases for every model and scoring mode under full
sensing, six-step gaps and ten-step gaps. Keep known-state physics, particle
filter, public kinematic, zero-action and uniform-action references. This is
51 controller rows and 163,200 native control transitions.

The 25-check primary rule requires at least 3% lower mean cost for geometric
GRU scoring on both gap panels, no worse mean cost in any paired fit, at most
2% degradation under full sensing, and the prespecified competence margins
against zero action. All checks must pass. Cached-MLP comparisons and exposed
diagnostics cannot rescue a failed gate.

## What the result could support

A pass would establish a scoring improvement for these saved recurrent models.
It would not establish a new architecture, biological superiority or an ICLR
contribution. If the cached MLP remains better, that remains the stronger
control baseline. A failure would rule out this particular fixed intervention
under the stated margins, without proving that memory is useless.

Every fit, case and attempted computation is retained. The implementation
records candidate predictions, both reward components, actual actions,
checkpoints, native states and timing. A separate auditor reconstructs scores,
search decisions, native replay and qualification from saved outputs; it does
not repeat neural inference or training. Capacity and full-pipeline rehearsal
must finish before the scored protocol is frozen.
