# Author NL-LFR training/inference budget comparison

This prospective protocol governs one four-condition comparison. Exact sources,
checkpoint identities and successful original FIT-audit closures must be
registered before evaluation. The larger-budget FIT-only process is a separate
study; this protocol never restarts or alters it.

The original author NL-LFR reached its 10,000-iteration training cap, while
378 of 384 causal context solves reached their 16-direction inference cap.
Neither count proves that more work will improve forecasting. This experiment
separates the two budgets before drawing conclusions about the recurrent
candidate's apparent advantage.

## Four fixed conditions

| Checkpoint | Context directions | Role |
| --- | ---: | --- |
| Original 10,000-iteration fit | 16 | Replay the audited diagnostic |
| Original 10,000-iteration fit | 64 | Isolate inference-budget change at fixed weights |
| Fresh larger-budget fit | 16 | Isolate training-budget change under original inference policy |
| Fresh larger-budget fit | 64 | Measure the joint change and its interaction |

Name the new checkpoint by its actual returned iteration count, not an assumed
100,000 steps. Its registered limit is 100,000. Do not train another candidate,
change checkpoints, choose a best request, fall back between conditions, or
select the inference budget after seeing errors. Both new-checkpoint policies
enter the strongest-control comparison if independently eligible.

Admission requires the original successful process closures, full inventories,
source identities and independent audits for both fits. The larger-budget fit
must preserve the original initialization and inputs, reproduce the first
10,000 loss entries exactly, and return finite validated artifacts. A trace
prefix mismatch stops this budget-attribution experiment before DEV evaluation;
it is not permission to change the checkpoint or the comparison.

A finite new fit that reaches its larger cap may be evaluated as an incomplete
diagnostic. Exact prefix and source/input identity support the budget comparison
even if that cap is reached, but only `FIT_ONLY_COMPLETE` can support a completed
nonlinear reference. The original checkpoint remains incomplete in both cells.

## Causal request and numerical policy

Use the same twelve exposed DEV records and starts `0,256,...,7936`, with context
100 and horizon 128. Reuse authenticated saved physical request arrays from the
original diagnostic rather than materializing the combined measurement archive.
Use the original common FIT scoring normalizer and exactly the same saved
standardized targets. Standardize each fresh physical prediction, subtract the
cached standardized target directly, and score that difference. This differs
slightly in rounding order from the historical physical-subtraction-first scalar
score. Preserve historical reports; compute all new cells and unchanged historical
control banks with the same standardized-difference formula in this study. Exact
old-cell parity concerns prediction/target arrays, retained states and diagnostics,
not bitwise identity of differently ordered scalar arithmetic. No new recording,
300 mV or official-test member is decoded.
Opaque hashing needed to authenticate parent evidence is reported separately.

Each call receives 100 observed outputs, their 99 available paired inputs, and
128 future inputs. The state estimator receives only the observed context.
Future inputs enter only the forecast; future targets enter only scoring after
the prediction returns. The unavailable first input is never fabricated. The
caller materializes fresh legal input copies inside the whole-request timer.

Use a separate qualified implementation with an explicit integer iteration
argument restricted to 16 or 64. Do not edit or monkeypatch the original module.
Only the outer direction limit changes. Keep the same per-checkpoint final-linear
minimum-norm seed, normalization, Jacobians, eight backtracking trials, damping
`1e-3`, Armijo `1e-4`, gradient tolerance `1e-8`, scale floor `1e-8`, least-squares
cutoff `1e-12`, final diagnostic SVD and output-before-update rollout.

The 64-step call starts fresh from the same linear seed, not from the saved
16-step solution. Require identical seeds and identical first-16 direction/trial
traces within each checkpoint. If the 16-step call stops early, the 64-step call
must stop identically. A longer successful solve cannot increase its fitted
context objective because the existing acceptance rule remains unchanged; it
can still worsen the held-out future forecast. Retain that outcome normally.

Before measurement use, fabricated tests must prove that each new implementation
reproduces its own held 16-step predecessor bitwise. Independent-versus-production
forecast/state replay retains the original `atol=rtol=1e-8` tolerances. Complete
context diagnostics, including their state/singular-value arrays and objectives,
retain the held audit's `atol=1e-10, rtol=1e-8` comparison.
Cover nonlinear/rank-deficient cases, early stops, rejected nonfinite trials,
work caps, batch and order independence, target/future-input isolation and
structural errors. Qualify independent 64-step replay without production numeric
imports. Preserve all original source and qualification evidence.

## Evidence and cost

Attempt all 1,536 forecasts and retain all 48 record-condition slots. Save
predictions, exact starts/targets, legal physical inputs, linear/solved/forecast/
end states, full context traces, objective values, stop reasons and actual work
counts. Preserve numerical failures explicitly; one missing forecast makes its
record and cell incomplete. Structural/source/I/O failure stops the original
process. No average over only successful requests is a primary score.

Primary error is the original equal-record mean of per-record FIT-standardized
RMSE over 32 requests, 128 future steps and three output channels. Also report
every record and both amplitudes. Keep the original candidate and earlier
controls' forecast banks unchanged. The original-checkpoint/16 cell must reproduce
the previous audited predictions, states, targets and context statuses exactly;
any difference is an implementation discrepancy, not a new result.

Collect current-host latency after the fit process closes, with no concurrent
training from this task. One fresh warmup per condition, then the same first and
last request of each of twelve records: 24 timed calls per condition, 96 total.
Use the canonical cell order `old16, old64, new16, new64` for warmups on the
first record at start zero. Order timing slots by the fixed record roster, first
start zero then start 7936 for each record. Rotate the four-condition order across
those 24 slots using slot index modulo four. Each condition occupies each order
position six times. The same explicit order is used regardless of results. Attempt
all timed slots even when a warmup fails, retaining each success/failure explicitly.

Time input slicing/copying, normalization, seed inference, all Jacobian and trial
work, final diagnostic SVD, rollout and output denormalization. Do not reuse a
state, factorization or forecast across calls. Disk loading and evidence saving
are excluded, while fresh legal request materialization is included. Report the
cache-based request adapter used here. Historical request times use a different
adapter/occasion and are descriptive only; the four new cells share one adapter.

Report median full-request time, per-call timings, work counts, numeric parameter
and state bytes and explicit failure counts. The larger inference budget does
not change parameter storage. Timing does not establish a speed comparison with
the historical candidate unless that candidate is separately measured under the
same conditions. Do not claim a deployment frontier from unmatched old timings.

One original evaluator is capped at 3,600 seconds and 32 GiB sampled child RSS.
All four cells share that limit, including warmups and evidence preservation.
No restart, rescue or substitution is part of the comparison. A separate original
audit replays all forecasts and state-estimation traces using independent NumPy
arithmetic; it recomputes saved timing summaries without retiming models.

## Interpretation and continuation

Report training effects at both inference budgets, inference effects at both
checkpoints, and their difference-in-differences interaction. Include absolute
and relative changes with all four denominators visible. A checkpoint change
also changes its final-linear seed, so a training contrast does not isolate
learned dynamics from state initialization. A lower context objective does not
establish lower forecasting error or a better learned representation.

Preserve the original four candidate continuation rules against the strongest
eligible control: at least 5% lower equal-record mean, every candidate seed
improves, no record seed-mean over 2% worse, and both amplitudes improve. Both
new-checkpoint cells with complete forecasts, valid parity and complete training
are eligible accuracy controls. A timing failure must not remove a strong accurate
control from that ranking. Overall matrix completion additionally requires all
1,536 forecasts, 48 record slots, four warmups and 96 timed slots to be valid and
all parity checks to agree. Reference completion and affirmative candidate
continuation require that matrix completion plus the new fit being complete.
Missing cost evidence therefore blocks continuation without discarding valid
accuracy scores. Finite capped or stalled context solves remain scoreable; their
convergence is not a new completion requirement. The original capped fits remain
diagnostic and cannot be promoted by improved inference alone.

This is one author-method adaptation on exposed DEV, with one author-model seed
and a previously selected candidate. It is not untouched generalization,
architecture superiority, biological learning, or a new optimizer. If the
candidate survives the stronger comparison, an untouched scenario and second
environment remain necessary. If the conventional reference closes the gap,
report it and use that evidence to choose the next learning experiment.
