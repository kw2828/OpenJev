# FIT-only author NL-LFR comparison

This prospective comparison asks whether the fixed recurrent residual survives
a conventional nonlinear control. It uses the author's full nonlinear feedback
model, restricted to our existing FIT partition, followed by causal short-context
state estimation. Development data have already been exposed. This is an
author-method adaptation, not a reproduction of the published periodic score,
an untouched test, or a new architecture.

## Sources and qualification

The registration pins this protocol, the producer, evaluator and supervisor,
the inference implementation and tests, dependency lock, existing restricted
reader, author revision, our own closed BLA checkpoint, and original prerequisite
receipts. The author revision is `a79e8c567b018a6c9462528fc1e10b77fd19b3e2`.
Use only the isolated `research/fsm_author` environment, CPU float64, and its
qualified installed package/source identities. Preserve GPL-3.0-or-later notices
and separate CC BY 4.0 measurement attribution.

The fabricated model/context suite passed 66 tests. The single full-size
synthetic run qualified 7,473 parameters, N8192, six realizations, two periods,
explicit-axis simulation/export parity, fresh-process initialization and archive
roundtrips. Its two recorded BFGS steps did not improve loss. It established
runtime feasibility, not optimizer effectiveness. Keep both original receipts
and all earlier failed qualification attempts. Producer/evaluator/supervisor
qualification and independent source review must close before measurement use.

## Data boundary and one fit

Decode only the four existing 100/200 mV training archive members through the
qualified lazy reader. Those members contain both exposed FIT and DEV slices;
the reader materializes both. Only realizations 0..2, both periods and both
amplitudes, enter fitting. All 300 mV and official-test members and headers stay
closed. Do not use the author's eager loader or supplied trained weights.

Assemble the twelve FIT records as `(8192,3,6,2)`, with the 100 mV triplet first.
Author normalization precedes period averaging. Preserve raw FIT tensors and
normalized period means/spectra for an independent preprocessing check. The old
benchmark FIT normalizer remains the separate scoring coordinate system.

Load our own completed pooled BLA28 archive, verify its numeric arrays exactly
against the closed export, and do not refit it. Connect the author's seed-42
network: 16 nonlinear inputs, 8 nonlinear outputs, two width-64 ReLU hidden
layers with biases and identity final activation; coupling scale `1e-4`.
Set `PYTHONHASHSEED=0` before interpreter startup because the author's seed
function also uses Python's randomized tag hash. Check all ten random network/
coupling arrays against the qualified full-size synthetic initialization.
Keep sampling time as a static Python float. Verify exactly 7,473 trainable
scalars; normalization and original BLA are not trainable.

Use unweighted BFGS with `rtol=1e-3`, `atol=1e-5`, at most 10,000 recorded
iterations, `print_every=-1`, and offset820. Jointly train the dynamic and neural
arrays using all 4,097 RFFT bins. The original BLA supplies periodic FIT initial
states; prepend the last 820 FIT inputs and simulate 9,012 steps. This is legitimate
periodic training context and is never available to a DEV request. Preserve
the source's loss and endpoint weighting without replacing them with H128 MSE.

One supervisor limits the original fit child to 18,000 seconds and 32 GiB RSS,
including admission, compilation, optimization and artifact preservation.
The source also performs one discarded warmup optimizer step; its cost remains
inside this limit. Poll RSS and preserve the process log/exit and final observed
resource use. No restart, width reduction, optimizer switch, pole repair,
clipping, checkpoint search or DEV selection is permitted. A timeout/memory
stop retains partial artifacts and is incomplete, not a negative accuracy result.

Save initial/final ZIP and raw numeric exports, training-state evidence,
normalization, loss history, iteration times and the raw author stop flag before
validation. Independently reconstruct the initial/final native FIT objective.
Require finite parameters/evidence and a nonworsening reconstructed objective.
The author's Cauchy small-change flag permits a completed fit status; it is not
a stationarity or optimum certificate. A finite iteration-capped model may be
evaluated diagnostically but cannot become a completed conventional reference.

## Fixed causal evaluation

Evaluate only after the original fit process has closed successfully and its
artifact inventory authenticates. Use a separate original evaluator process,
limited to 3,600 seconds / 32 GiB with no restart. No weight updates occur in it.
Request roster: all twelve exposed DEV records, starts 0,256,...,7936, context 100,
horizon 128. Forecasts and metrics use the same periods and targets as the closed
parent studies. Future targets are read only after each request returns.

Use the [qualified causal solver](fsm-nllfr-causal-contract.md): estimate state
at the second observed output from 99 paired observations/inputs, using the final
model's linear minimum-norm seed, then at most 16 damped Gauss-Newton directions
and 8 trials per direction. Damping 1e-3, Armijo 1e-4, gradient tolerance 1e-8,
column-scale floor 1e-8 and least-squares cutoff 1e-12 remain fixed. Advance 99
nonlinear transitions before the first forecast. No future inputs enter this
state solve; no periodic tail, missing first input or cached state is available.

Attempt all 384 requests. Retain physical contexts/future inputs, normalized
forecasts/targets, seed/solved-start/forecast-start/end states, complete trial
traces, objectives, ranks/singular values, termination and actual work counts.
State magnitude diagnostics cover these retained endpoints, not every internal
trajectory. Finite stalled/capped estimates remain scoreable with their true
status. Numerical failure is explicit; structural/schema/I/O failures terminate
the original evaluator rather than masquerading as ordinary poor forecasts.
Never substitute another seed or a BLA forecast. A partial record has no primary
record score, and missing records cannot silently disappear from a mean.

Primary metric: per-record root mean squared FIT-standardized error over 32
requests,128 steps and3 channels, then equal mean across 12 records. Also retain
channel and physical-unit RMSE. It is not the author's periodic NRMSE.

After one untimed full-request warmup, time first/last requests of every record,
24 measurements. Charge slicing, normalization, linear seed, all context Jacobian
and rejected-trial calls, final diagnostic SVD, H128 rollout and denormalization.
Report initialization/rollout components and median full-request time. Charge
all 19 final numeric arrays,28 state doubles and9 C/H/solver-policy numeric scalars.
The deployed export retains no original BLA/factorization/cache. Report transient
solver workspace separately. Historical timings are descriptive, not a matched
accuracy/compute frontier.

## Independent audit and continuation

Follow the [audit contract](fsm-nllfr-audit-contract.md) after original process
closure. Authenticate sources/inputs/outputs first, then reconstruct raw FIT
normalization, spectra and the original BLA periodic initial states without
calling the author simulator/optimizer. Independently simulate initial/final
NL-LFR objectives and all 384 causal requests. Compare exact initial arrays and
exact parent starts/targets. Numeric tolerances are fixed in the registration:
objective `atol=1e-7,rtol=1e-8`; forecast/state `atol=rtol=1e-8`; context loss
`atol=1e-10,rtol=1e-8`; preprocessing/FFT `atol=rtol=1e-10`. Branch divergence
in the independent state solver is an audit discrepancy, not permission to
change tolerances or select a better local solution.

Keep `tanh_feedback-lr0.0003`, seeds 9201/9202/9203, and its forecast banks fixed.
Retain every complete earlier control, the completed BLA and a completed NL-LFR
in the strongest-control pool; choose minimum equal-record/equal-seed error,
lexical tie break. Recompute the same four development continuation conditions:
candidate mean at least 5% lower; every seed improves; no record seed-mean more
than 2% worse; both amplitudes improve. Use matched seeds for a stochastic control
and the same single author reference for each candidate seed. Any incomplete
reference blocks an affirmative survival claim for this reference check.

Our candidate was selected from 36 fits on exposed DEV; the author reference is
one seed with a different training objective, search budget and initializer.
The comparison does not isolate architecture or establish superiority over the
published method. A remaining gain would justify an untouched scenario check
and a second environment under separate protocols. A weak author adaptation
could instead motivate matching its training objective to the conditional task,
with equal additional training for controls. Neither follow-up is part of this
fit. Connectome, novel learning mechanism and paper-readiness claims remain open.
