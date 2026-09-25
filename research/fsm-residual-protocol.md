# Frozen linear-backbone residual development comparison

Version `fsm-residual-study-v1`. Freeze this protocol, implementation and parent
artifact hashes before any new empirical fit. The preceding
[correction pilot](fsm-correction-results.md) failed. Its best VARX model is the
starting point here, selected using already exposed DEV records. This is a
development search, not untouched confirmation or an official benchmark score.

## Question and controls

Can a small nonlinear correction improve the accurate linear predictor, and
does feeding that correction back improve forecasts beyond correcting only
the reported output? Use the complete factorial:

| Residual | Output-only | Feedback |
|---|---|---|
| Affine, 588 trainable scalars | Linear trajectory advances independently | Corrected output enters the next lag state |
| Tanh, 4,779 trainable scalars | Linear trajectory advances independently | Corrected output enters the next lag state |

Affine feedback is an adapted linear recurrence. It controls for improving the
training objective from one-step ridge to multistep simulation. Tanh heads have
more parameters; this does not isolate nonlinearity from capacity. Parameter
counts match only within each placement pair. The construction is established
prior art, discussed in the [literature note](fsm-residual-prior-art.md).

## Frozen parents and information

Reuse the unchanged float64 order-32, ridge-1e-6 coefficients and FIT normalizer
from `fsm-correction-study-v1`. Do not refit either, cast them to float32, repair
poles, clip forecasts or fall back to a different model. The registration pins
the original coefficient file, fit receipt, normalizer, registration, audit,
closure, summary and observed process completion. Copy these into this run.

The unchanged `fsm_data` adapter reads only the four 100/200 mV estimation
members of the same archive, SHA-256
`bdf6004da1342a8746e51580a57b5ddb8ac400ac5caa48368719c33cbb0ef505`.
The original CC BY 4.0 attribution and transformation notice apply. No official
test or 300 mV member or array header is decoded. At each admitted amplitude,
realizations 0,1,2 are FIT and 3,4,5 are exposed DEV; both periods remain separate.
These are twelve FIT and twelve DEV records, with correlated periods/windows.

The adapter necessarily decodes both partitions within the permitted members.
Only FIT arrays enter training. Every declared fit must close before any new
DEV prediction, scoring or learning-rate selection. No period concatenation,
end-of-period wraparound, interpolation or new normalization fit is allowed.

Each request supplies C100 outputs, the aligned 99 context inputs, and H128
future inputs. Both heads retain only the final 32 observed outputs and inputs.
Input u[k] predicts y[k]. The first context input is omitted as before. Future
outputs are never supplied to prediction. This is conditional forecasting,
not closed-loop control, arbitrary-state recovery or a biological network.

## Model and fitting budget

Per step, concatenate chronological last-32 outputs, current input and
chronological last-32 inputs: 195 standardized features. The frozen linear
projection includes its intercept. An affine residual is 195 to 3. The tanh
residual is 195 to 24 to 3, with one tanh activation and biases. Both add the
residual in standardized output units, before denormalization.

Zero the final residual weights and biases exactly. Tanh hidden weights use
the declared seed. Across placement modes and rates within a head type/seed,
initial tensors must match. Affine initial functions are identical across
seeds too; their training variation comes from sampled batches. Check the
initial forecast against native VARX on each fit's first scheduled FIT batch,
using absolute and relative tolerances both 1e-9. Hidden gradients at zero
output initialization may vanish; feedback/output-head gradients may differ.

Cross four architectures with rates **1e-4, 3e-4, 1e-3** and seeds
**9201, 9202, 9203**: **36 fits, 73,728 accepted updates if all complete**.
Every fit gets exactly 2,048 Adam updates, batch 16, no weight decay, clipping
of the total gradient norm at 1, and full free-running H128 mean squared error.
No teacher-forced future values, auxiliary loss, warm-started residual,
checkpoint selection, early stopping on DEV or post-result extension.

Use CPU float64 and one PyTorch thread. Each seed has one saved schedule of
uniformly drawn FIT record/start indices from the qualified local RNG routine
with seed 100000 + model seed. All twelve recipes for a seed share that schedule.
Rotate recipe execution order by the seed index. Retain initial/final weights,
final Adam states and every update's loss, gradient norm and elapsed time.

Each fit has a **240-second cap**, including its initial parity check, batches,
training and trace writes; check it again after the last update. Model
construction and checkpoint serialization are outside that per-fit interval
but inside the outer study interval. Preserve partial states and errors for any
failure. A failed fit gets an explicit unrun evaluation slot, never a silent
restart or replacement. Complete the remaining declared attempts.

## Scoring, selection and costs

Retain all 36 learned-model evaluation slots and one current native VARX
evaluation: **37 slots**. Every complete slot has twelve records and 32 fixed
starts per record, 0,256,...,7936. Save predictions, targets and starts. Primary
error is record RMSE across all 32 requests, 128 steps and three FIT-standardized
outputs, then the equal-record and equal-seed arithmetic mean. Also retain
per-channel errors in both standardized and source-native output units.
These correlated windows are not independent replications.

Select one rate per architecture from complete fits and finite, complete
forecast scores, using the three-seed mean. Break exact ties by lower numeric
rate. **Cost failures do not replace the best-accuracy rate.** No seed-specific
rate, checkpoint or cost-based selection. Then choose one strongest affine
architecture/rate globally, and one strongest reference globally from that
affine model, selected tanh output-only and native VARX. Cross-architecture
exact ties use the lexicographic recipe name. Their identities stay fixed
through all seed, record and amplitude comparisons.

Rate selection and reporting use the same exposed DEV data. The pooled seed
comparison is descriptive, not an unbiased selected-model efficacy estimate.
The two affine architectures give the affine reference six candidate recipes;
the tanh-feedback candidate has three. Retain every recipe, including failures.

Warm up the first fixed request, then time the first and last request of every
record, 24 samples per model. Include normalization, copies/conversion,
conditioning, full rollout, validation and denormalization. Report every sample,
each median and the mean seed median. Model/disk loading are excluded. No
exclusive-machine or controlled BLAS-thread claim. Re-time native VARX now;
the previous GRU result remains historical context, not a fresh cost control.

Storage includes every residual parameter, frozen coefficient, one stream's
192 float64 lag values, 96-byte normalization, and logical numeric metadata:
24 bytes for tanh, 16 for affine, 24 for native VARX. Requests, temporary
workspace, gradients cleared before evaluation and Python/string overhead are
excluded. Affine feedback could be algebraically folded into a native linear
recurrence. Its unfused PyTorch latency is not its minimum deployable cost;
this experiment cannot establish a neural runtime/frontier advantage over a
qualified folded control. The cost gate compares the matched tanh pair only.

## Fixed continuation rule

All nine conditions are required for a development pass:

1. All 36 fits finish every declared update.
2. All 37 evaluation slots are complete, finite and have all declared costs.
3. All initial forecast identity checks pass and the backbone remains unchanged.
4. Selected tanh feedback has at least 5% lower mean error than the single
   strongest reference selected above.
5. Each of its three seeds beats the same-seed selected tanh output-only and
   strongest affine reference, and the native VARX error.
6. Its seed-mean error for no record exceeds the strongest reference by 2%.
7. It lowers mean error at both amplitudes versus that same strongest reference.
8. Its mean seed-median request latency is at most 1.10 times selected tanh output-only.
9. Its persistent numeric storage is no larger than selected tanh output-only.

Report quality, reliability and cost outcomes separately even if the aggregate
rule fails. A single unstable declared recipe fails the complete-search rule;
it does not erase a useful selected-model quality result. Passing would justify
designing a stronger comparison and untouched test. It would not establish
convergence, architectural novelty, physical identification, stability, robotics
control performance, benchmark SOTA or ICLR readiness.
