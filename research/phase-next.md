# Beat the small nonlinear dynamics baseline first

Prospective direction after `phase-study-v1`, 24 September 2026. This note
registers no new fit and admits no official benchmark test.

The [completed phase comparison](phase-results.md) fails 18 of its 21
conditions. The energy-dependent rotation recipe is closed. Do not promote
its favorable individual seed or change its criteria. The official Silverbox
TEST remains unopened under this result.

There is useful evidence for the next design: a seven-parameter cubic AR2
model, refined through its own simulated predictions, reaches approximately
0.94 mV error on both development sequences. It outperforms the 18-parameter
phase and readout models substantially. GRU16 improves further. The next
candidate therefore needs to explain what it adds beyond a compact nonlinear
transition, not merely beyond linear memory or a memoryless output map.

My next research question is whether sparse coupling between recurrent state
blocks transfers more efficiently across interacting physical subsystems.
This is a hypothesis to qualify, not an established advantage. A useful
comparison would include a compact polynomial state-space model, a small GRU,
a published coupled oscillator and an appropriate stable recurrent model.
[coRNN](https://arxiv.org/abs/2010.00951) already learns nonlinear coupled
oscillators; [recurrent equilibrium networks](https://arxiv.org/abs/2104.05942)
provide stability and robustness constructions. Either is prior art and a
potential control, not a mechanism to relabel as our invention.

The [Industrial Robot identification benchmark](https://www.nonlinearbenchmark.org/benchmarks/industrial-robot)
is a candidate source to inspect next. Its six-joint measured robot has
coupled motion, backlash and pose-dependent forces. The source also describes
temperature-dependent friction as potentially time varying; this does not
prove that the released sequences contain a useful adaptation signal.
Read its current problem description and partition contract before numerical
access, then use an allowed development split to establish baseline headroom.
No industrial-robot data has been loaded or fitted for this study.

Before another architecture run:

1. Define one mechanism and the specific prediction or control problem it is
   intended to solve. Qualify the released observation/action interface and
   train/development/test boundaries first.
2. Retain compact nonlinear dynamics and GRU controls. Compare true subsystem
   coupling with dense, random sparse and degree-matched rewired coupling at
   matched parameters, state and training exposure. An anatomical connectome
   requires a defensible mapping to the task; otherwise call the graph sparse
   or structured, not biological.
3. Measure input-driven free-run rollout error and measured inference cost. Report any
   supervised initialization, observation feedback and reset policy for every
   model. A later control claim requires its own closed-loop evaluation.
4. Freeze a continuation rule before measurements. Cross-system transfer and
   scenario shift must use untouched data; this study's DEV cannot become a
   new test set. Conformal calibration or RL belongs only after a predictive
   mechanism beats the relevant simple controls.

This narrows the architecture search while retaining the failed candidate and
its successful comparators. No connectome advantage, calibration improvement,
robot-control result or ICLR-ready contribution has been established here.
