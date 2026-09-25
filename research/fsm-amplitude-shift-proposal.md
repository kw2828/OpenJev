# Proposed frozen-weight 300 mV confirmation

**Conditional, unregistered and unrun.** Prefer this amplitude-shift check before
additional sensitivity training. The [four-condition NL-LFR audit](fsm-author-nllfr-factorial-results.md) has
closed with agreement, complete eligible references and 4/4 continuation checks.
That satisfies the prior reference condition; this separate shift experiment
still requires qualified sources and a published registration before access.
This proposal follows [the residual next steps](fsm-residual-next.md); it
authorizes no decoding, prediction, fitting or model selection.

## Fixed comparison

Freeze these **28 model/policy instances across 18 families** by original
checkpoint, normalization, source and audit hashes before access:

| Models | Instances |
| --- | ---: |
| Affine output-only and feedback, each rate 1e-4, seeds 9201/9202/9203 | 6 |
| Tanh output-only at 1e-3 and candidate tanh feedback at 3e-4, same seeds | 6 |
| Existing VARX orders 32/64/96 crossed with penalties 1e-6/1e-3/0.1 | 9 |
| Existing folded affine-feedback coefficients, all three seeds | 3 |
| Original native VARX32 backbone | 1 |
| Completed author BLA28 with its qualified causal initializer | 1 |
| Completed new author NL-LFR checkpoint, both fixed 16/64-direction policies | 2 |

No refitting, calibration, normalizer updates, new rates, seed removal or
checkpoint replacement. Preserve each author's embedded FIT normalization;
use the original common FIT normalizer for cross-model scoring. Supply the same
C100 interface while acknowledging that models consume different history lengths.

The candidate remains `tanh_feedback-lr0.0003`. Freeze the strongest eligible
control identity from the closed exposed-DEV audit:
`tanh_output_only-lr0.001`. Bind its terminal audit and original checkpoints at
registration. Never reselect it on 300 mV. Report every other fixed control too.
The two old capped NL-LFR policies may be a separately declared diagnostic
appendix, but are omitted from this minimum roster and cannot become completed
references. Adding them requires freezing their separate counts before access.

## Information and scoring

Admit only `u_300mV_train` and `y_300mV_train` from the already identified source
archive. These names do not permit training. Require native float64 shape
`[8192,3,6,2]`, 6,400 Hz, finite values, realizations 0..5 and periods 0..1.
Keep twelve chronological records separate. No wrapping, averaging periods,
resampling or imputation; schema mismatch stops the attempt. Every official
`*_test` member and header stays closed.

For starts `0,256,...,7936`, supply `y[s:s+100]`, `u[s+1:s+100]` and
`u[s+100:s+228]`; predict `y[s+100:s+228]`. The unavailable first input is
dropped. Initializers receive no future input; prediction receives no future
output. Read targets only after prediction returns. Future applied inputs are
supplied forcing, not a claim of autonomous or closed-loop control.

Retain **10,752 forecast attempts and 336 record-instance slots**, including
failures. Score standardized prediction minus standardized target using the
same frozen arithmetic for every model, then equal-record RMSE and equal-seed
family means. Report every record, channel, seed, standardized and native-unit
channel error, late-H64 error descriptively, and numerical/context-solver status.
Never average only successful requests. Report realization groups 0..2 and 3..5
separately: their lower-amplitude counterparts were FIT and exposed DEV.
Repeated periods, windows and matched excitations are correlated.

## Proposed frozen decision and cost

Require all declared forecasts, folded/unfused parity checks and costs complete;
then require candidate mean at least 5% below the fixed reference, every candidate
seed strictly better than its paired reference seed (or the same deterministic
reference), and no record seed-mean more than 2% worse.
There is only one confirmation amplitude. Explicitly replace the development
two-amplitude check at registration with improvement in **both fixed realization
groups**, rather than inventing a second amplitude or choosing a favorable group.
Also retain the candidate latency ceiling of 1.10 times tanh output-only and
storage no greater than that family. All conditions are required. Another
control outperforming the candidate must remain visible even if this named
contrast passes; that pass would not establish superiority over every control.

Re-time all 28 instances together after training has closed: one fresh warmup
each, then first/last request per record, 24 samples each (**672 timed calls**).
Freeze canonical instance order with paired rotation/reversal before execution.
Include fresh copies, normalization, causal state inference, every GN/Jacobian/
line-search/SVD operation, rollout and physical output conversion. No cached
state/factorization or fallback. Keep failures and actual solver work. Historical
timings are descriptive; the new matched timings govern the cost criterion.
Count deployed weights/coefficient buffers, required normalizers, one-stream
state and numeric policy metadata; report transient workspace separately.

## Before numerical access

Use a new restricted reader, leaving the qualified four-member reader unchanged.
Fabricated tests must prove exact two-member decoding, rejection before decoding
on source/model drift, admitted-array shape/dtype checks, poisoned official-test members remaining unopened,
causal alignment, frozen-state ownership and preserved failures. Qualify all
adapters and independent replay on fabricated inputs before registration. Bind the
closed parent audits, all 28 deployments, original scales, exact record/start
roster, metrics, decision rules and deterministic timing order. Freeze a single
process wall/RSS budget from fabricated throughput, with no retry or rescue.

The [publication correction](fsm-author-publication-correction.md) records prior
opaque copying of vendor 300 mV train **and test** example bytes. Do not describe
this as a universally pristine reserve or infer that copying proves numerical
training contamination. State the narrower documented numerical-use boundary
and its historical limitation. New publication must exclude raw archives and
all vendor examples before enumeration/hashing. A pass would support this
fixed-weight amplitude-shift contrast on one plant, not independent environments,
stability, biological novelty or permission for subsequent reserve-driven tuning.
