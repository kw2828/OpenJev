# Frozen stronger linear controls for the FSM residual result

Version `fsm-linear-controls-study-v1`. Freeze this protocol, source hashes and
all parent artifact identities before new empirical solves or predictions.
The preceding [residual study](fsm-residual-results.md) passes on exposed DEV.
This comparison asks whether its fixed candidate still improves error when
longer linear memory is available, and measures a correctly folded linear
deployment. It is development, not untouched confirmation or novel architecture.

## Fixed models and data

Keep all twelve selected parent residual checkpoints unchanged: affine
output-only and feedback at rate 1e-4, tanh output-only at 1e-3 and tanh feedback
at 3e-4, each with seeds 9201, 9202 and 9203. No further training, rate/seed
selection, checkpoint replacement or candidate retuning is allowed.

The candidate is fixed as `tanh_feedback-lr0.0003`. Retain the original native
VARX32 backbone and frozen FIT normalizer. Fit exactly nine new conventional
VARX models: **orders 32, 64, 96 crossed with penalties 1e-6, 1e-3, 0.1**.
All fit inside C100 with its 99 aligned past inputs. Larger orders have more
parameters and a longer observed history; the point is a stronger reference,
not a capacity-matched mechanism claim.

The unchanged guarded adapter admits only the four 100/200 mV estimation
members of source archive SHA-256
`bdf6004da1342a8746e51580a57b5ddb8ac400ac5caa48368719c33cbb0ef505`.
At each amplitude, realizations 0-2 are FIT and 3-5 are exposed DEV; both periods
remain separate. There are twelve FIT and twelve DEV records, each 8,192 by
three samples/channels at 6,400 Hz. No 300 mV or official-test member or array
header is decoded. Original data attribution and transformations remain as
[documented](fsm-residual-results/DATA_LICENSE.txt).

The adapter decodes FIT and DEV within the admitted members, but only FIT
records enter sufficient statistics. Every new solve closes before any new
DEV forecast or scoring. There is no new normalization fit, period wrapping,
period averaging or concatenation. Future input u[k] predicts y[k]. All
requests provide C100 observed outputs, 99 aligned past inputs and H128 future
inputs; future outputs are never arguments to prediction.

## Longer-memory linear fitting

For each order, construct the unregularized mean Gram and cross-product once,
using the same canonical record/feature ordering as the qualified existing
VARX solver: chronological last-p outputs, current input, chronological last-p
inputs, intercept. Retain both statistics, their hashes, exact FIT identities
and `12*(8192-p)` rows. Apply each positive penalty only to non-intercept
diagonal entries. Use float64 primal Cholesky with no jitter, pole repair,
clipping, normalization refit or alternative solve.

Qualify equality with the existing solver on fabricated records for all nine
recipes before registration. Each order's statistics construction/write and
each solve/checkpoint write have separate 120-second caps, checked on return.
These caps detect overruns; they do not interrupt a native library call.
Charge shared statistics once and each solve separately. Retain the full outer
study duration too. A failed order creates three explicit failed solve slots;
a failed solve retains its error and any produced coefficients. No retries.

Store unregularized statistics so the independent auditor can verify the
solution's backward error:

`||A W^T - cross||_F / (||A||_F ||W||_F + ||cross||_F) <= 1e-10`,

where `A` is mean Gram plus the declared non-intercept penalty. A zero
denominator uses the absolute residual. This checks the algebraic solution;
the FIT-only lineage of the statistics remains source/receipt-attested unless
raw data are independently decoded. It is not an independent raw-data refit.

## Folding affine feedback

For each selected affine-feedback seed, add its weight matrix and bias to
the matching frozen backbone coefficients. Validate exact CPU float64 shapes,
finite tensors, the affine-feedback architecture and backbone identity. Return
a plain VARX model holding only the merged coefficients and parent provenance;
the parent's order/penalty/fit-row fields describe the original backbone fit,
not a new ridge fit of the merged model.

Re-evaluate every unchanged parent model. Compare each folded forecast against
its freshly evaluated unfused counterpart on every DEV record/window at
`atol=rtol=1e-9`, before folded timing. Retain every finite parity forecast bank,
including mismatches. If conversion or parity fails, preserve the error and
mark its evaluation/cost slots unavailable; do not repair or time a substitute.
The complete saved roster is **nine new linear + twelve unchanged residual +
three folded affine + one native backbone = 25 evaluations**, or fifteen
families. Every model is re-timed in this run.

## Scoring and cost

Use the unchanged residual evaluator: all twelve DEV records with starts
0,256,...,7936, C100/H128, saving prediction/target/start banks. Primary error
is the equal-record mean RMSE in original FIT-standardized output units,
then the equal-seed mean where applicable. Retain channel errors in both
standardized and native units. Periods, windows and matched excitations are
correlated; no confidence intervals or independent-replication counts are inferred.

Choose the single strongest control globally from **all fourteen noncandidate
families**, by complete finite mean DEV error with lexical family-name ties.
Its identity stays fixed for all record, seed and amplitude comparisons.
Timing failures do not replace an accurate control. The candidate is the fixed
parent selection; no new candidate selection occurs. Reference selection and
reporting on exposed DEV remain descriptive.

CPU float64 and one PyTorch thread apply. Native BLAS threading is not forced;
there is no exclusive-machine timing claim. After one warmup, time the first
and last fixed request per record, 24 samples per model. Include normalization,
copying/conversion, conditioning, all forecast steps, validation and
denormalization; exclude loading/disk I/O. Report each model's median and
equal-seed mean median. Preserve all samples and failures.

Numeric storage includes retained coefficients/weights, one stream's lag state,
96-byte normalizer and logical metadata. Requests, temporary workspace and
Python/string overhead are excluded. Folded models retain 588 float64
coefficients, 192 float64 state scalars and 24 metadata bytes: **6,360 bytes**
including normalization. Their original unfused model/checkpoint is provenance,
not required runtime storage. All deployments' actual numerical state is counted.

## Frozen continuation rule

All nine conditions are required for a development pass:

1. All nine linear fits complete with finite qualified solutions.
2. All 25 evaluation slots are complete and have finite scores and costs.
3. All three folded models pass the declared forecast equivalence check.
4. The fixed candidate has at least 5% lower mean error than the globally
   strongest control.
5. Every candidate seed has lower error than that same control, using its
   matching seed when learned/folded and its single score when native linear.
6. No candidate record seed mean is more than 2% worse than that control.
7. Candidate mean error improves at both amplitudes against that control.
8. Candidate mean seed-median latency is at most 1.10 times the unchanged
   tanh output-only family, freshly timed here.
9. Candidate numeric storage is no larger than tanh output-only.

Report fitting, parity, quality and costs separately. A failed complete-search
condition does not erase a useful individual outcome. Passing would justify
further reference qualification and a frozen untouched-shift test; it would not
establish a neural latency frontier, new architecture, stability, causal physical
identification, closed-loop robotics performance or ICLR readiness.

The authors' BLA28/NL-LFR reproduction remains a separate prerequisite for broad
competitiveness claims. Their periodic warmup must not supply future request
inputs to initialization. No author-trained model or author-control performance
is imported into this registered comparison.
