# Larger-budget author NL-LFR: FIT-only study

This is a new attempt prompted by the closed original reference reaching its
10,000-iteration limit. It tests whether additional optimization from the same
initialization resolves that limit. It does not extend the original process,
change the recurrent candidate, or authorize another development evaluation.
The prior capped result and its independent audit remain unchanged.

## Fixed question and scope

Run the original author NL-LFR once from its exact saved **initial** model, using
100,000 recorded iterations instead of 10,000. The vendor solver does not retain
the optimizer state in the saved model, so loading the old final checkpoint
would be a different experiment. This attempt starts a fresh BFGS optimizer
from the original initialization. The original final checkpoint is not an input.

The experiment is FIT-only. Load the audited parent `fit-data.npz` and the parent
initial ZIP/numeric export, original BLA export, and solver trace. Do not decode
the combined measurement archive, DEV arrays, 300 mV arrays, official-test arrays,
or vendor example measurements. Metadata admission may hash existing prerequisite
files as opaque bytes to authenticate the closed parent evidence; that is not
array decoding. The earlier publication-scope correction remains applicable.

The recurrent candidate `tanh_feedback-lr0.0003` and its three existing seeds stay
fixed. There is no model selection, new candidate fit, DEV evaluation, threshold
adjustment, or untouched-test call in this registration. A later causal comparison
requires a separate prospective protocol and qualification.

## Admission and unchanged recipe

Before arrays are loaded, authenticate the parent registration, original fit and
evaluation process closures, complete artifact inventories, independent PASS
audit and its successful original-process closure. Require that the parent is
the audited 10,000-iteration capped attempt. Bind each new input to its exact
role in that audited inventory. Pin the new protocol, producer, supervisor,
tests, qualification and source review, the held parent helpers, dependency
lock, and runtime identity in the new registration.

Use the same isolated CPU float64 runtime and environment as the original run,
including `PYTHONHASHSEED=0`. Preserve the author's GPL-3.0-or-later source and
the measurements' CC BY 4.0 attribution. No vendor implementation is changed.

Load the original initial ZIP. Compare all 19 live numeric arrays to the original
initial export and all 9 nested BLA arrays to its original export. Verify static
architecture/activations and Python-float sample intervals. Rebuild the author
data object from the cached raw FIT tensors of shape `(8192,3,6,2)` and compare
normalization, period means, RFFT spectra, frequency indices, and training initial
states to the audited cache. Retain all checks before optimizer entry.

The model remains 28 states, 16 nonlinear inputs, 8 nonlinear outputs, two hidden
layers of width 64 with ReLU, biases and identity output, with the original
seed-42 coupling/network arrays. Require the exact 14-leaf, 7,473-scalar trainable
partition. The original BLA and normalization are not trained.

Call the unchanged vendor optimizer with BFGS `rtol=1e-3`, `atol=1e-5`, unweighted
frequency loss, offset `None` resolving to the original 820 inputs, and
`print_every=-1`. All 4,097 RFFT bins, six realizations, original periodic FIT
initial states and 9,012-step simulation remain identical. Only `max_iter`
changes to 100,000. The discarded vendor warmup step is still performed and
charged. Do not substitute the causal H128 evaluation objective.

## Budget, preservation and attribution

Use one original supervised process, capped at 18,000 seconds and 32 GiB sampled
child RSS. Admission, compilation, warmup, optimization and evidence preservation
count toward the wall limit. Preserve timestamps, log, observed exit code, peak
polled RSS, source/input identities and complete output inventory. A timeout,
memory stop, numerical failure or process failure remains incomplete. Do not
restart, resume, switch optimizers, change dimensions, repair poles, clip states,
select checkpoints or relax a failed check.

Preserve the returned final ZIP, raw arrays, loss/time traces and raw author stop
flag before validation. Check exact serialization roundtrip, unchanged original
BLA/normalizer and finite evidence. Independently reconstruct the native initial
and final FIT objectives using the original fixed tolerances
(`atol=1e-7`, `rtol=1e-8`); require a nonworsening final objective.

Require exact float64 equality of the first 10,000 recorded loss values against
the original trace. Retain compared length, first mismatch and maximum absolute
difference. An insufficient or divergent prefix is an attribution failure, even
if the final loss is lower. It must not be described as a budget-only result.
The vendor trace records one scalar per solver step and may repeat previous
accepted-state losses during rejected trials. It is not a record of every trial
objective, intermediate parameter state or accepted-step count.

`FIT_ONLY_COMPLETE` requires successful process/artifact closure, exact initial
and input identity, exact trace prefix and the author's stopping flag. The flag
means its small-change criterion fired, not that a stationary point or global
optimum was certified. A capped fit or prefix mismatch is
`FIT_ONLY_INCOMPLETE`; retain the separate optimizer status and attribution flag.
Neither status is `REFERENCE_COMPLETE` and neither establishes DEV performance.

## Qualification and independent audit

Before measured fitting, qualify the new admission/status/supervision code using
fabricated tests. Perform one small fabricated vendor run comparison with two
and four recorded steps to check budget-prefix consistency and preservation of
the optimizer start through ZIP serialization. It is a wiring check, not an
effectiveness result. Keep the prior full-size 7,473-parameter synthetic runtime
qualification as the geometry/feasibility evidence. Freeze exact source hashes,
test logs, qualification receipt and source review before registration/launch.

After process closure, use a separately qualified auditor. It may reuse only the
held independent auditor's pinned NumPy math helpers, not production simulation,
optimizer, measurement decoding or DEV replay. Authenticate registration,
sources, parent artifacts, process closure and complete output inventory first.
Reconstruct FIT normalization, spectra, original BLA periodic initial states and
initial/final native objectives; check initial/roundtrip arrays, trace prefix,
status derivation and zero evaluation calls. Preprocessing/FFT tolerances remain
`atol=rtol=1e-10`. Preserve discrepancies without tolerance changes or refits.

An audit PASS establishes internal agreement for this FIT-only attempt. Any
claim about stronger control accuracy, architecture, connectomes, a learning
mechanism or paper readiness remains outside this study. Once it closes, use
the observed optimizer status to decide whether a separately registered causal
evaluation is justified.
