# Independent audit contract for the FIT-only author NL-LFR

**Prospective audit design, not a registration or an audit result.** No measured
arrays, weights, inference or fitting were used to write this note. The producer,
supervisor, evaluator, exact output roster and numerical tolerances must be
pinned before their empirical execution. This audit must not call the producer's
scoring/selection functions, the author's simulator or optimizer, or retrain a
model. It independently reconstructs specified calculations from retained data.

## Admit a closed original process before numerical reads

Verify the registration and prefit commit, complete source/snapshot roster,
locked Python/packages, installed author source at `a79e8c56`, CPU float64,
`PYTHONHASHSEED=0`, seed 42 and the original qualification receipts/logs. Bind
the original launch command, process identity, terminal exit, elapsed/cap/RSS
fields, log hash and retained output inventory. Check inputs and source pins
again at closure. An output directory without its original terminal receipt
is not an admitted study; an audit may not run against a live fit.

The proposed measured cap is one **18,000-second, 32-GiB** original child, with
at most 10,000 recorded BFGS iterations. A timeout or memory stop remains
incomplete, with its partial files and original log preserved. Do not require a
final checkpoint that the terminated author call never returned, manufacture a
return object, continue from another process, or certify a partial result as a
completed reference. Audit agreement and scientific completion are distinct.

Bind our own closed BLA's final ZIP/numeric export, fit/process/audit/registration
and its normalizer. Supplied author weights are not an allowed substitute. Bind
the frozen residual/linear-control registrations, original audits and complete
comparison inventories, including all three unchanged candidate checkpoints.
All 300 mV and official-test members/headers remain outside decode interfaces.

## Model, initialization and optimizer evidence

Require exact numeric shapes for eight live matrices and six NN arrays:
`A[28,28], Bu[28,3], Cy[3,28], Dyu[3,3], Bw[28,8], Cz[16,28], Dyw[3,8],
Dzu[16,3]`, weights `[64,16],[64,64],[8,64]`, biases `[64],[64],[8]`.
All are finite float64, with two hidden ReLUs and identity output activation.
There are **7,473 trainable scalars**; the twelve normalizer values, scalar
sampling time and original `_bla` are not trainable. Check positive finite
scales and `ts=1/6400`. Preserve raw returned exports, ZIP and trace before
validating them, so invalid returned values do not erase failure evidence.

Check initial live linear matrices/statistics exactly against the admitted BLA
export. For a model-construction-free check of the random initialization, pin
the full-size synthetic qualification's initial numeric export: the four
coupling matrices and six NN arrays should match it exactly under the same
runtime, seed, hash seed and sigma `1e-4`. These draws do not depend on BLA or
measurement values. Check the retained original BLA and all normalization
fields remain unchanged in the final model. Numeric/ZIP equivalence needs an
explicitly scoped serialization check; hashing two files alone does not prove it.

Require trace lengths equal the raw iteration count, bounded by 10,000, with
finite scalar losses/times and the original stop flag. Record initial/final
parameter changes without requiring change for engineering success. A recorded
optimizer iteration is not necessarily an accepted weight update. The source
performs an extra discarded warmup step; retain its cost in the outer duration.
The author flag is a Cauchy small-change test, not a stationarity or optimum
certificate. Do not assert monotone trace values or equality of the last trace
entry and returned-model loss without establishing the author's auxiliary-value
timing convention. Reconstruct the returned-model loss independently instead.

## Independent full-spectrum FIT objective

The exact FIT roster is 100/200 mV realizations 0..2, periods 0..1: twelve
records arranged `[8192,3,6,2]`. FIT normalization precedes period averaging.
Retain normalized period-mean `u,y[8192,3,6]`, complex period-mean
`U,Y[4097,3,6]`, frequency indices 1..3839, and the training initial-state
evidence. Check `rfft(u)` against U and `rfft(y)` against Y at the frozen FFT
tolerance, acknowledging different averaging/reduction order.

Reconstruct the **original BLA's** periodic state independently at every RFFT
bin, including DC and Nyquist:

```text
X[k] = solve(exp(2*pi*i*k/8192)*I - A_bla, B_bla @ U[k])
x_period = irfft(X, n=8192, axis=0)
x0 = x_period[-820]
u_extended = concatenate(u[-820:], u)
```

Use general dense solves, not the diagonal simplification from the fabricated
fixture. Fail on singular/nonfinite solves without pseudoinverse or jitter.
Compare this x0 with the retained training-state evidence. Starting from x0,
simulate each initial/final NL-LFR through all **9,012** normalized inputs using
an independent NumPy recurrence. Remove only the first 820 outputs, then compute

```text
Y_hat = rfft(y_hat[820:], axis=0)
objective = sum(abs(Y - Y_hat)**2) / (4097*6).
```

This sums channels and bins; there is no extra division by three, one-half,
FFT-length normalization or doubled interior-frequency weight. It uses all
4,097 bins, not just the excited BLA lines. Match saved initial/final objectives
at predeclared tolerances and retain any worsening/failure honestly. This
reconstructs the native reported squared-residual objective, not our forecast
RMSE or necessarily the optimizer's internal half-squared scalar convention.

Saved normalized FIT tensors permit independent objective reconstruction but
not independent proof of raw-record assignment or preprocessing. That lineage
remains source/admission-attested unless an explicitly authorized independent
reader rederives it from the same four allowed archive members. State which
scope was executed. Never imply that a source hash alone validates numeric
normalization or that FIT statistics were independently refitted when they were not.

## Independently reconstruct 384 causal forecast requests

The fit producer may stop before DEV evaluation. Admit the separately closed,
frozen evaluator before this stage; incomplete or absent evaluation is not
evidence of a passed comparison. Require twelve DEV records with starts
`0,256,...,7936`, 32 requests per record, C100/H128 and no period crossing.
Compare starts and common-normalized targets exactly with previously audited
banks. Every request must have an attempted identity and explicit outcome.

Implement the equations and analytic sensitivities independently, following the
[causal contract](fsm-nllfr-causal-contract.md). Estimate x at the second
observation using final `A,Bu,Cy,Dyu`; never reuse the original BLA trajectory,
the unavailable first input, the future input sequence or any future target in
the solve. Reproduce the fixed 16-direction/8-trial algorithm and its derivative
at ReLU zero, damped augmented LS, relative cutoff, scaled gradient and strict
Armijo acceptance. Nonfinite line-search trials alone may be rejected; malformed
inputs, initial/Jacobian/solve failure must not become an alternative seed.

Verify per-request seed/solved-start/forecast states, initial/final objectives,
termination, accepted steps, rank/singular diagnostics and complete trial trace.
`trial_attempts` counts proposals; `trajectory_evaluations` counts actual rollout
calls. Retain failed requests and finite ZERO_JACOBIAN/STALLED/ITERATION_CAP
statuses separately. Numerical branch differences in an independent solver may
change a local solution: use only predeclared tolerances, record the discrepancy
and stop agreement, never loosen tolerances or select whichever replay is better.

Forecast output before state update from x at s+100 using u at s+100. Compare
physical forecasts and final states, then independently normalize with the old
benchmark scales and recompute record MSE/RMSE, per-channel and native-unit
errors. Aggregate record RMSEs equally, then neural seeds equally. Physical
normalization, author normalization and benchmark scoring normalization are
distinct. A missing/failed request must not silently disappear from an average.

## Comparison, costs and verdict

The fixed candidate is `tanh_feedback-lr0.0003`, seeds 9201..9203; no checkpoint,
rate or seed reselection. Retain all complete prior controls and the newly
completed NL-LFR. Select the strongest noncandidate by equal-record/equal-seed
mean error, with a frozen lexical tie break independent of timing. Recompute
the existing four continuation checks: 5% lower candidate mean; every candidate
seed improves against the paired stochastic-control seed or same single
reference; no record mean over 2% worse; both amplitudes improve. One author
seed is one common reference, not three paired replications. An incomplete
new reference blocks an affirmative claim about surviving that reference check.

Validate retained timing samples/medians arithmetically without rebenchmarking.
Full requests must charge normalization, seed solve, all context derivatives
and rejected trials, diagnostics, H128 forecast, validation and denormalization.
Distinguish warmup, compilation, fitting, artifact preservation and evaluation
duration. Charge every retained final matrix/network/statistic, original BLA or
factorization if deployed, solver constants and 28-state buffer; transient
Jacobian/SVD workspace is separate. Prior-process latency is descriptive only.

The audit should emit separate engineering agreement, fit completion, forecast
coverage and comparison outcomes, exact input/source pins and dynamic output
inventory. A negative or incomplete scientific outcome can still have honest
audit agreement. No claim of independent gradient/optimizer replay, robust
state identification, learned stability, novel architecture, official-score
reproduction or untouched transfer follows from this audit.
