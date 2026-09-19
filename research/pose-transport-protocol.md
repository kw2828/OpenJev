# Pose transport: frozen development comparison

September 19, 2026. This is a new training recipe on already exposed robot
forecasting data. It cannot confirm generalization, rescue the failed
residual-dynamics-v1 rule, or establish paper-level novelty on its own.

## Question and matched intervention

Does rotating recurrent memory as the robot body turns improve a predictor
beyond an otherwise identical GRU in body coordinates?

All learned models predict changes in per-step world displacement and spatial
rotation increments, with a constant-motion output skip. They start every
forecast with the same last16 secant estimate, using observations16 and31,
which span15 intervals. A zero-initialized readout exactly reproduces valid
world constant-velocity/constant-angular-rate extrapolation before training.

The main pair shares a 48-scalar hidden state, two GRU cells, a six-dimensional
linear readout, motion scales, initial tensors, minibatch orders and optimizer:

- **body:** GRU consumes body-frame displacement/rotation increments, unit world
  vertical expressed in the body frame, and the ordered40 torque block.
- **transport:** same architecture, but its48 hidden scalars are reshaped to16
  row vectors. Moving from body orientation R_old to R_new maps those vectors
  by right multiplication with R_old^T R_new. The map is also applied when
  correcting a predicted endpoint to the next actual observation.
- **history16:** independently trained transport model receiving only the final
  16 context observations and their15 intervening action blocks.
- **world:** same hidden width, output skip and objective, with world-coordinate
  motion features/corrections and all9 rotation-matrix entries as orientation
  input. This gives more orientation information and more parameters than the
  body pair. Its comparison is not a pure coordinate-only ablation.

Standard GRU nonlinearities are not vector-neuron equivariant operations.
Transport is a hypothesis about memory coordinates, not proof of full SO(3)
equivariance, biological learning or a new algorithm. Body-coordinate inputs
and fixed vertical imply a yaw/translation coordinate symmetry, not immunity
to terrain, contact or gravity changes. No test-time weight adaptation occurs.

## Data and physical geometry

Use the exact data bindings from residual-dynamics-v1/data-01:720 training
windows from old source trajectories0..29 and160 windows from10 parent
trajectories in each sin and zigzag archive. The latter are exposed development
panels. Do not call them untouched tests. Do not fit on their targets or use
them for checkpoint selection. Old development windows are unused in this run.

Undo the training-only normalization of the stored float32 observations in
float64. Recover Euler angles with atan2(sin,cos), then construct
R=Rz(yaw)Ry(pitch)Rx(roll), mapping body coordinates to world coordinates.
This matches the documented Bullet convention, but the archive generator and
original quaternions are absent, so the collection convention is an assumption.

Position has units of meters and rotations are valid SO(3) matrices. Advance
orientation by Exp(world rotation increment)R. Rotational distance is the
principal SO(3) geodesic angle, at most pi. Neither interpolation nor future
observations enter history features. Scales for displacement, angular increment
and their changes are four scalar training-window RMS values, floored at1e-5.
No independently normalized sine/cosine coordinate is an output or loss term.

Context32 and horizon25 use nominal20ms steps and500ms forecasts. Each action
retains all ten ordered raw4-torque samples. These are recorded applied torques,
not authenticated issued commands. This is conditional offline prediction on
simulated data, not closed-loop or real-robot evidence. Raw/prepared data and
target/prediction arrays remain local because upstream licensing is not explicit.

## Fixed references

Include hold-last; world constant velocity/rotation using1 and16 observations;
root-anchored least-squares velocity/rotation over all16 observations; and
constant body twist from the SE(3) logarithm of T16^-1 T31, including the correct
rotation-translation coupling. Classical rotations always stay on SO(3).

The sixth reference is a direct linear residual ridge predictor, separately
fit for each forecast horizon, around valid world CV16. Inputs are root-relative
last16 positions and spatial rotation vectors, past15 action blocks, and future
action blocks only through the current endpoint. Targets are position and
root-relative rotation-vector residuals. Unit ridge penalty, unpenalized
intercept, exact float64 dual solve. Fit before any development-panel forecast.

## Training and evaluation budget

Twelve fits: four variants for seeds701,802,903. Same initialization seed and
batch permutations per paired fit,30 epochs, batch32,23 updates/epoch,
690 updates/fit and8,280 total. Adam lr.001, gradient norm cap1, float32,
one CPU thread, deterministic Torch operations. Use the final checkpoint;
no retries, replacement seeds, sweeps or development-selected checkpoints.

Train on the mean over all25 future steps of
||p_pred-p_true||^2/(0.1 meter)^2 + angle(R_pred^T R_true)^2/(0.1 radian)^2.
These explicit engineering weights do not equate meters with radians.
Report the two physical errors separately, and retain the fixed composite.

After all fits and ridge training complete, evaluate all18 configurations on
each development panel. Each neural family pools3 fits. Save all predictions,
targets, updates, initial/final weights and per-fit metrics. Measure20 complete
batch1 windows per row after3 warmups, including context reconstruction and all
25 forecast steps. Neural families have120 pooled timing samples; references40.
Exclude loading, normalization and service overhead. Report all reference costs.

## Prospective development continuation rule

For each panel, each of the other three neural families and six fixed references,
and each physical endpoint, require all of the following:

1. Transport pooled RMSE is at least10% lower.
2. Each of its three fits has nonworse MSE than the paired learned control or
   the fixed reference.
3. At least8/10 parent trajectories have nonworse pooled MSE.
4. Every leave-one-parent-out comparison has strictly lower pooled MSE.

Finally require pooled median full-window transport latency at most1.5 times
body latency. There are 541 individual checks: 36 endpoint/control/panel comparisons with
15 checks each, plus the latency check. All must pass, with no omissions or
substituted metrics.
The independent auditor recomputes all errors from sealed saved outputs. It
does not call models, training, simulation, or random generators.

A pass would justify confirmation on new trajectories and a separate online
adaptation experiment. A failure remains failed. No previous result is amended.

## Related work and claims

Geometry-preserving dynamics already have substantial prior art, including
[port-Hamiltonian networks on Lie groups](https://arxiv.org/abs/2401.09520).
Vector-valued latent representations are established by
[Vector Neurons](https://research.google/pubs/vector-neurons-a-general-framework-for-so3-equivariant-networks/),
and hidden-state transport relates to
[Flow Equivariant Recurrent Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2025/file/e637029c42aa593850eeebf46616444d-Paper-Conference.pdf).
This experiment is not a reproduction of those methods. It tests one narrow
memory intervention on a known forecasting problem with stronger controls.
