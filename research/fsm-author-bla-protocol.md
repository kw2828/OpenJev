# FIT-only author BLA28 comparison

This prospective study fits one conventional reference using the authors'
published frequency-domain method. It asks whether the previously selected
recurrent residual still improves conditional forecasts after adding a stronger
linear baseline. The existing development data are exposed. This is neither
confirmation nor a new architecture, and it is not reproduction of the authors'
periodic test score.

## Admission and sources

The machine-readable registration pins this protocol, all executed wrapper
sources, the existing restricted measurement reader, an isolated dependency
lock, the exact author implementation and prerequisite qualification receipts.
Run only after CPU float64 import/source preflight, synthetic causal initializer,
synthetic author fitting/simulation/serialization, and fabricated evaluation
checks pass. Keep every failed qualification attempt.

Use the original hashed FSM archive and its existing lazy reader. Decode only
`u_100mV_train`, `y_100mV_train`, `u_200mV_train`, `y_200mV_train`. The native
shape is `(8192,3,6,2)`. FIT is realizations 0..2 at both amplitudes, both periods;
DEV is 3..5. Preserve periods. All 300 mV and official-test measurement members
and headers stay closed. The authors' supplied weights and eager benchmark
loader are forbidden.

Only the twelve FIT records reach `create_data_object`, pooled as 100 mV's
triplet followed by 200 mV's triplet, shape `(8192,3,6,2)`. Author normalization
uses these FIT values, then its implementation averages periods. Evaluation
scores every DEV period separately using the unchanged parent FIT scales.
These normalizers have distinct roles even if their values agree numerically.

## One frozen fit

Use `freq-statespace` commit `a79e8c567b018a6c9462528fc1e10b77fd19b3e2`,
with the isolated `research/fsm_author/uv.lock`. CPU and float64/complex128;
record environment/thread settings without claiming CPU means single-threaded.
Sampling rate 6400 Hz, excited indices 1..3839, subspace order 28 and dimension
29, `input_output_mode=False`, `freq_weighting=False`. Fit the pooled
nonparametric frequency response, with no amplitude weighting or DEV selection.

Refine once with BFGS `rtol=1e-3`, `atol=1e-5`, at most 5000 iterations, and no
periodic performance logging. A supervisor enforces 1800 seconds from the
original child launch, including imports, data admission, fitting and evaluation.
Retain the original process/log/exit or timeout. No retry, pole repair, clipping,
alternative optimizer, checkpoint selection or model substitution is allowed.

Save subspace and final model archives, numeric matrices and normalizers,
frequency target and independent frequency errors, solver history/timings,
author stop flag and spectral radius. Require finite matrices and trace and no
increase in independent FIT frequency error. The author stop flag means the
BFGS Cauchy small-change condition, not certified stationarity or an optimum.
If the iteration cap is reached first, retain finite diagnostic forecasts but
label the reference incomplete. A timeout or exception is a failed attempt,
not evidence that this conventional method is ineffective.

## Causal forecasts and cost

For each of twelve DEV records, use starts 0,256,...,7936, context 100, horizon
128. Input u[k] corresponds to y[k]. Fit a latent state at the second observed
sample using the 99 available paired observations/inputs, fixed least-squares
cutoff 1e-12 and minimum-norm rule. Propagate through all context pairs, then
forecast output before state update. No future target, first missing input,
periodic tail warmup, cached request state or additional period is available.

Use the qualified NumPy implementation on exported author matrices. Save all
384 forecasts, targets, final states, rank/singular-value and context-residual
diagnostics. Nonfinite or missing records make the reference incomplete; rank
deficiency alone does not change the declared estimator. No rank-driven retuning.
The metric is per-record square root of mean squared FIT-standardized error
over 32 requests, 128 steps and three channels, then equal mean across records.
Also report channel and native-unit error. This matches the parent study's
metric and is not the authors' periodic NRMSE.

After one untimed full-request warmup, time one complete request at the first
and last start of every DEV record: 24 measurements. Include slicing, model
validation, normalization, observability construction and least-squares solve,
context propagation, rollout, diagnostics and physical output conversion.
Report the median. Matrix caching/JIT are not added in this implementation.
Count all numeric model/normalizer/sample-time arrays, 28 request-state doubles
and the C/H/rcond numeric scalars. Exclude transient workspace and interpreter
overhead explicitly. Timing comparisons to previously retained controls are
descriptive measurements from different processes, not a matched runtime win.

## Frozen comparison and next step

Keep all three `tanh_feedback-lr0.0003` seeds 9201,9202,9203 unchanged. Use their
retained forecasts from the audited stronger-linear-control study. Verify file
hashes, record/start/target joins and recompute all compared errors independently.
Compare this BLA, VARX96/ridge1e-6, tanh output-only and the fixed candidate.
Do not rerank/select a new candidate.

A completed reference allows a development continuation check: the fixed
candidate's equal-seed/equal-record mean must be at least 5% below the strongest
of the new BLA and all previously complete controls; each candidate seed must
beat that reference, no record's seed-mean error may be over 2% worse, and both
amplitudes must improve. Report all conditions, including failures. No new
accuracy/compute Pareto claim is allowed from this separate runtime.

Passing only supports this next reference check. The author nonlinear NL-LFR
still needs separate causal-initializer qualification and its own frozen fit.
Reserved 300 mV access and a second measured environment remain separate,
unexecuted studies. The current residual construction is established prior art.

Sources and exact API distinctions: [author review](fsm-author-reference-qualification.md),
[runtime contract](fsm-author-runtime-contract.md), [causal contract](fsm-author-causal-contract.md),
[evaluation contract](fsm-author-evaluation-contract.md).
