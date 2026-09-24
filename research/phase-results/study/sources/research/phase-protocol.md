# Small nonlinear oscillatory memory on measured hardware

Prospective version `phase-study-v1`, 24 September 2026. Commit the registration
and all scientific sources before numerical measurement loading, fitting or DEV
rollouts. This is a fresh development mechanism test, not a repair to the
[rejected sensor screen](sensor-screen-results.md) or closed robot studies.

## Question and scientific boundary

Can state-dependent phase improve an extremely small recurrent model's
free-running prediction compared with a fixed phase and an equally small
nonlinear readout? The [Silverbox benchmark](https://www.nonlinearbenchmark.org/benchmarks/silverbox)
is a measured electronic Duffing-like oscillator with cubic feedback. Its
input-output dynamics motivate this question without hiding an available sensor.
The model receives voltage inputs and predicts voltage outputs, not decisions
or simulated control rewards. Its parameters are fixed after offline training.

A phase mechanism is not automatically a new architecture contribution.
[coRNN](https://arxiv.org/abs/2010.00951) already uses coupled oscillatory
recurrence; [ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf) and
[learned LPV scheduling](https://arxiv.org/abs/2204.04060) address related
state-dependent dynamics. [Polynomial nonlinear state-space identification](https://doi.org/10.1016/j.automatica.2010.01.001)
is established. [Recurrent equilibrium networks](https://arxiv.org/abs/2104.05942)
provide stronger stability guarantees that this cell does not inherit.
A useful result here would qualify a specific mechanism for further testing,
not establish long-memory necessity, online adaptation, biological learning,
robot control, calibration, a world model for planning or a state-of-the-art score.

## Data identity and reserved tests

Use `SilverboxFiles/SNLS80mV.csv` from the official linked
[archive](https://drive.google.com/file/d/17iS-6oBUUgrmiAcrZoG9S5sOaljZnDSy/view).
It has header `V1,V2,`, 131,072 data records and one blank trailing line.
V1 is input and V2 is output, sampled at 610.35 Hz. Values are in volts.

- ZIP SHA256: `2398dda503ac4c30f46c7a1ba09d8a8864d001cc71bbeb59f207a054cc592d8c`
- CSV SHA256: `ae62d5a91230c10f76e6dd02c8a4fac3c9d4d8a95fbf50e87cb0c4885003e0f1`
- Official v1.0 loader SHA256: `cf6749f3e1c34df5686ee4a1e2ec0909ad5c04fd8efefd64c18674cc2f61ee0b`

The [official loader](https://github.com/MaartenSchoukens/nonlinear_benchmarks/blob/v1.0/nonlinear_benchmarks/benchmarks.py)
uses raw zero-based rows `[40650:105712]` for TRAIN/validation. Its separate
multisine TEST is `[105712:127400]`; full arrow TEST is `[100:40575]`, and its
non-extrapolation subset is `[100:32100]`. Those tests are all reserved and
numerically unparsed. The arrow subset is not an independent test population.

Use fixed subsets of official TRAIN, tied to published multisine realizations:

| Partition | Raw indices, stop exclusive | Records | Use |
| --- | --- | ---: | --- |
| FIT | 40650:83946 | 43,296 | Initial records through fifth complete realization |
| DEV A | 84446:92638 | 8,192 | Sixth complete realization |
| DEV B | 93138:101330 | 8,192 | Seventh complete realization |

Other rows are unconverted. No missing-value imputation or exclusion is allowed.
Nonfinite or malformed values inside an admitted partition stop execution.
Compute scalar means and population std for u and y from all FIT records only;
standardize using those four immutable float64 values. Load DEV numerically only
after all fits finish. DEV outputs enter scoring only, never any predictor,
normalizer, initialization, loss, checkpoint choice or optimizer.

These are development results on one physical system, not independent datasets
or the official benchmark test protocol. The official first-50-output
initialization allowance is not used here. We use input-only zero-state rollout
and a common 512-sample warmup on each internal DEV sequence. Do not compare the
result numerically with published official TEST scores without rerunning the
separately registered official procedure.

The data source pages do not establish an explicit redistribution license.
The loader's BSD software license is separate. Retain raw ZIP/CSV and numerical
FIT/DEV arrays locally; publish derived results, independently authored code,
parameter checkpoints, hashes, links and receipts. No commercial-rights claim.

## Three controlled oscillator models

Each model has two two-dimensional blocks z, four float32 recurrent state
scalars, and an explicit zero initial state. Input u_t advances the state and
then produces yhat_t at the same index:

```
z_j^+ = r_j R(theta_j) z_j + b_j u_t
yhat_t = sum_j c_j dot z_j^+ + d u_t + e
r_j = 0.9999 sigmoid(a_j)
omega_j = pi sigmoid(w_j)
```

`R(theta)` is the two-dimensional rotation with rows `[cos,-sin]` and `[sin,cos]`.
All models learn the two radii, two base angles, four input weights, four linear
readout weights, feedthrough and bias: 14 parameters.

- **fixed_phase, 14 parameters:** theta_j = omega_j.
- **energy_phase, 18 parameters:**
  theta_j = omega_j + 0.5 tanh(k_j) tanh(softplus(s_j) ||z_j||^2).
  This changes recurrence using state energy before the input update.
- **nonlinear_readout, 18 parameters:** keep fixed phase; add four learned
  readout weights on `z_j^+ ||z_j^+||^2`. Nonlinearity acts only on the output.

For each paired seed, all common initial tensors are identical. Initial radii
are (0.99,0.95), base angles (0.1pi,0.3pi), input and readout weights local-seeded
N(0,0.1^2), feedthrough/bias zero. Energy k is zero, softplus(s)=1; extra readout
weights are zero. All three models initially compute the same function. The
energy-scale gradient is initially zero while the phase-coefficient gradient can
be nonzero. Do not mistake that expected initial behavior for a broken path.

For finite parameters, rotation preserves norm and r<1, giving a bounded-input,
bounded-state inequality `||z^+|| <= r_max ||z|| + ||b|| |u|`. This does not imply
incremental stability, bounded training gradients or agreement between different
floating-point implementations for every parameter choice. Test the selected
numerical audit tolerance before loading real measurements.

## Strong controls

**GRU16:** PyTorch GRU with one input and 16 hidden units, then a learned linear
readout. Default torch initialization under a local paired seed. Initial hidden
state is zero; neither the GRU nor any phase model accepts measured outputs.

**Cubic AR2:** seven coefficients for features in this exact order:
`[yprev, yprev2, u_now, u_prev, yprev^2, yprev^3, 1]`. Initialize with ridge on
FIT one-step observed pairs starting at FIT row 2; this supervised initialization
is disclosed. Every simulated rollout uses its own two previous predictions,
never measured outputs. Initial previous outputs and input are zero. Preserve
both the frozen ridge coefficients and a separately simulation-refined model.
No output clipping, reset or teacher-forced scored rollout is allowed.

**Static cubic:** ridge on `[1,u,u^2,u^3]` using every FIT row.
**FIR128 and FIR512:** ridge on `[1,u_t,u_(t-1),...,u_(t-L+1)]`; fit only rows
with all L actual FIT input positions available. At DEV startup left-pad the
input history with zeros. The common 512 warmup fills both finite input buffers.
All ridge fits use float64 normal equations, penalty 1e-6 on every coefficient
including intercept, and an ordinary solve. No hyperparameter search or jitter.
FIR/static inference uses float64. Frozen AR2 inference casts to float32, as
for its trained counterpart, while its original float64 ridge fit is saved.

## Training exposure, execution and evidence

Three fixed seeds: 7301,7302,7303. Train five families (the three phase models,
GRU16 and refined AR2), giving 15 fits. For each seed generate one shared
2,048 x 32 int64 array of sequence starts using NumPy default_rng(seed+410000),
uniform from zero through FIT_length - 256 inclusive. Reuse those exact start
positions across all five families. Each sequence has 256 input samples and is
reset to zero state. Optimize normalized MSE over positions 64:256. Truncated
training sequences and long DEV simulation are distinct and both disclosed.

Use Adam for 2048 updates, batch 32, betas (0.9, 0.999), epsilon 1e-8, learning rate 0.003
for phase/GRU and 0.0001 for AR2. The smaller AR2 rate is predeclared for its direct
dynamics coefficients, not selected after data inspection. Clip global gradient
norm at 1.0 and reject nonfinite loss/gradients/parameters. Keep the final update;
no best-DEV or best-training checkpoint selection. The frozen AR2 row protects
against refinement making the classical control worse.

This matches input exposure and gradient-update counts, not computation or
trainable parameter count across all families. The two 18-parameter phase arms
are parameter matched. Record optimizer-loop time, including final Adam slot extraction, and original
whole-run time separately. Model/Adam construction, tensor conversion and file
writes are outside the former and inside the latter. Report actual persistent
parameter/state bytes plus 32 bytes for four float64 normalizers. Include FIR
input queues and AR2 previous outputs/input. Do not infer speed from parameter
count. Python/native workspace, Adam state, training arrays and audit artifacts
are separate from deployed logical state. No neural deployment cache or input
archive persists beyond the declared state.

Execute on CPU with float32 neural parameters/state, one numerical thread and
deterministic PyTorch algorithms. Each optimizer-loop cap 600 seconds; original study
cap 2400 seconds, enforced internally and by an external original-process wrapper.
Qualification and independent audit caps are 300 and 600 seconds. The synthetic
throughput probe used no benchmark values and estimated roughly 11-15 seconds per
512 phase updates; it determines feasibility, not scientific settings after data.

Save full initial/final named tensors and Adam slots, every loss and gradient
norm, shared training-window arrays, FIT and DEV records with raw row indices,
complete 8192 normalized predictions per model/DEV, four ridge coefficient vectors,
normalizers, source snapshots, results, resources and a SHA256 manifest.
Failures preserve the original log, completed fits and the failing model/partial
training trace where Python catches the failure. A hard process timeout may leave
only completed artifacts. Do not retry, resume, silently drop a model or change
comparison criteria. Divergence is an execution failure and does not qualify
an architecture win against a broken comparator.

## Frozen 21-condition rule

Each complete DEV sequence is simulated once from zero state, with no measured
output feedback. Score 7680 samples after the common 512 warmup. Report RMSE and
MAE in mV, using FIT y_std to undo normalization. No bootstrap or significance
claim from three optimization seeds. They share the same physical observations.

One global condition requires every complete prediction array to be finite.
For each of DEV A and DEV B require all ten conditions:

1. Mean energy_phase RMSE <= 0.9 times mean fixed_phase RMSE.
2. Mean energy_phase RMSE <= 0.9 times mean nonlinear_readout RMSE.
3. Each of three paired seeds has energy_phase RMSE no higher than fixed_phase
   (three separate conditions).
4. Each of three paired seeds has energy_phase RMSE no higher than nonlinear_readout
   (three separate conditions).
5. Mean energy_phase RMSE <= 1.05 times the minimum of mean GRU16, mean refined
   AR2, frozen AR2, static cubic, FIR128 and FIR512 RMSE.
6. Mean energy_phase RMSE <= 0.1 times FIT target std, expressed in mV.

All 21 conditions: `ADVANCE_PHASE_MECHANISM`. Any failed condition:
`DO_NOT_ADVANCE_PHASE_MECHANISM`. A pass authorizes designing a separately frozen
official-test/second-system experiment. It is not official-test admission here.
A failure preserves every fit and score; no threshold or baseline is weakened.

## Qualification and independent audit

Freeze nine files: phase core, Silverbox parser, producer, independent auditor,
their four test files and this protocol. The registration pins their exact bytes,
CSV, numerical environment and the original closed successful fabricated-test
receipt and log. Commit registration and sources before the first fit.

Fabricated tests cover input/output alignment, causal prefixes, state carry,
matched initial functions, gradient paths, training-window/optimizer shapes,
per-seed gates, strong-baseline veto, parser boundaries and failure retention.
An 8192-step nonzero-phase/readout NumPy parity witness precedes registration.
The fixed replay tolerance is rtol=atol=1e-4 on normalized predictions; do not
loosen it after seeing learned checkpoints.

The independent auditor imports neither producer nor phase/parser modules.
It authenticates the closed original process, source/registration/Git pins and
file manifest before decoding evidence. It independently parses only admitted
FIT/DEV rows, computes normalizers and ridge coefficients, checks training-window
identity, common phase initialization, all completed trace/Adam shapes and step
counts, and replays all 38 final model/DEV predictions with independent NumPy
phase, GRU, AR2, static and FIR equations. All 135 manifest-listed files and 504
saved arrays must be covered by its final roster (any accounting correction is
allowed only before registration). Float32 recurrence outputs are stored as
float64; storage conversion does not increase inference precision. Raw data,
row IDs and shared windows compare exactly. Normalizer scalar checks use
separately declared tighter tolerances in auditor source.

Bandlimited inputs can make FIR coefficients poorly determined even when the
ridge solution's predictions are numerically stable. A fabricated bandlimited
probe demonstrated this before measurement loading. Therefore, the independent
Cholesky ridge solve is checked against the producer's saved solution using a
regularized normal-equation backward-error certificate, bounded by
`64 * gamma_d`, where `gamma_d = d*u/(1-d*u)`, `d` is coefficient count and
`u = eps_float64/2`. Also compare their FIT and both DEV predictions using the
fixed normalized `rtol=atol=1e-4` tolerance. Retain coefficient differences as
diagnostics, not as an ill-conditioning-sensitive pass criterion. These twelve
solution-prediction comparisons supplement all 38 saved prediction replays;
they do not replace them or alter the ridge penalty, scientific gates or fits.

The audit verifies saved numerical inference, data boundaries, metrics, state
accounting and all 21 criteria. It does not independently rerun 30,720 gradient
updates or prove every optimizer transition. Receipts and traces document those
updates; the report must preserve that distinction. Audit failure never becomes
a scientific success or permission to optimize on official tests.
