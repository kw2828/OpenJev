# FIT-only BLA28 comparison: evaluation contract

**Independent source and metadata review; no empirical execution in this note.**
This specifies compatibility with the completed [linear-controls study](fsm-linear-controls-results.md).
It is not a new registration, an author-model result or permission to open held-out members.
The measured producer must freeze its own fit, failure and comparison rules before execution.

## Identities to preserve

The [frozen registration](fsm-linear-controls-registration.json) has SHA-256
`996e492b7db79754fec9ac79239563431076673fe378ca696d240ca07d53c364` and was committed at
[`1a5539074bccc815493debbedc9a4d7d3ae83820`](https://github.com/kw2828/OpenJev/commit/1a5539074bccc815493debbedc9a4d7d3ae83820).
Its `source_sha256` and `parent_artifacts` maps are the exact source and checkpoint roster, not merely descriptive filenames.
The [protocol](fsm-linear-controls-protocol.md) hash is
`c0036cc506e7f342c469b0b997c01f730e0bb26b6184dc3e80b33fec0e6dffe8`.

| Required artifact | Recorded SHA-256 |
|---|---|
| Benchmark scoring normalizer: `output/fsm-correction-study-v1/normalizer.npz` | `46cf8fdc80747a5c701f86d92531839bf18a5ac67d94f91537bdb6b8ded601f9` |
| Frozen VARX32 backbone: `output/fsm-correction-study-v1/varx32-ridge1e-06/model.npz` | `b5790fb407e229229a47043b494d9e6a3ba01f957d483b0886ac211ccd6a76cc` |
| Candidate seed 9201: `output/fsm-residual-study-v1/tanh_feedback-lr0.0003-9201/final.pt` | `4aa6ff03ae9b320294b46ad3c19878016d1428e76e476a6ba19be4eb653d6c1c` |
| Candidate seed 9202: `output/fsm-residual-study-v1/tanh_feedback-lr0.0003-9202/final.pt` | `31f95f8d2f43a06658e13c2b406ea3791f5d890cdb02a9f929581e58e3c548a1` |
| Candidate seed 9203: `output/fsm-residual-study-v1/tanh_feedback-lr0.0003-9203/final.pt` | `3be4e2fd47c1c4bcba481ffe8696facf85b37f0718db03002219ccfe7c3eb0c1` |

These hashes were read from registration metadata; the array/checkpoint payloads were not opened for this review.
Preserve all three candidate seeds and their fixed rate. The strongest existing overall control is the fixed tanh output-only family at 1e-3; the strongest fresh linear control is VARX96/ridge 1e-6.
Their reported identities are historical DEV selections, not permission to select another candidate after seeing BLA28.
The existing registration also binds all twelve selected residual checkpoints. Declare any smaller retained comparison roster prospectively and keep the full historical table linked.

## Data boundary and canonical ordering

The admitted source archive is pinned to
`bdf6004da1342a8746e51580a57b5ddb8ac400ac5caa48368719c33cbb0ef505`.
Only `u_100mV_train`, `y_100mV_train`, `u_200mV_train`, `y_200mV_train` may be decoded.
Each is float64 `[8192,3,6,2]`, with axes sample, channel, realization, period and sampling frequency 6400 Hz.
Central-directory member names may be read, but no 300 mV or official-test array/header may be decoded.

- FIT uses realizations 0,1,2 at both amplitudes, both periods: twelve records, 98,304 samples per channel.
- DEV uses realizations 3,4,5 at both amplitudes, both periods: twelve separate records.
- Canonical record order is amplitude `100mV` then `200mV`, realization ascending, period 0 then 1. IDs are `100mV-realization-3-period-0`, etc.
- Build author FIT tensors `[8192,3,6,2]` by placing the 100 mV FIT triplet before the 200 mV FIT triplet on the realization axis. Preserve the period axis and orthogonal triplets.
- Author-method period averaging is confined to its FIT preprocessing. Evaluation never averages periods, wraps them, concatenates them in time, or uses DEV to estimate normalization or model parameters.

The guarded [adapter](../src/openjev/research/fsm_data.py), source hash
`4edad31e866ba88ed3c1e43b92f0cbc37d23fa454c2fab39cb8cf8cd8e5b6d90`, defines the record ordering and window contract.
Authentication of sources, original parent closures and opaque input pins must precede decoding; close the author fit and save its fixed model before DEV forecast/scoring.

## Keep model and scoring normalizers separate

The benchmark normalizer has exactly four float64 vectors of length three: `u_mean`, `u_scale`, `y_mean`, `y_scale`, with positive scales.
They are the population mean and standard deviation (`ddof=0`) of all twelve complete FIT records, concatenated in canonical order. Reuse the pinned values exactly; do not refit or estimate scales per request, amplitude, DEV record or window.

The author package instead names its FIT-only vectors `u_mean`, `u_std`, `y_mean`, `y_std`, computed across axes `(0,2,3)` before period averaging.
Although the admitted FIT population is the same, storage names and numerical reduction order differ. Do not assume bitwise equality or substitute one set after fitting.
Use the author's own scales to normalize observed context and inputs for its fitted matrices; denormalize its predicted outputs to physical units first.
Then score every model with the unchanged benchmark normalizer:

```text
prediction_score = (prediction_physical - benchmark_y_mean) / benchmark_y_scale
target_score     = (target_physical     - benchmark_y_mean) / benchmark_y_scale
```

If using public author `simulate`, pass physical future inputs because that wrapper normalizes internally. If using the pure exported-matrix rollout, pass author-normalized future inputs. Never normalize twice or normalize latent states.
Actual numeric scale entries are intentionally not reproduced here because this review does not decode NPZ payloads; the full hash and exact field contract above identify them.

## Exact request and state timing

Each period uses the 32 starts `0,256,...,7936`. For start `s`, the only model inputs are:

```text
y_context = y[s:s+100]          # [100,3]
u_context = u[s+1:s+100]        # [99,3], paired with y_context[1:]
future_u  = u[s+100:s+228]      # [128,3]
target    = y[s+100:s+228]      # scoring only, never a predictor argument
```

For BLA28, omit the first observed output from the state solve because its matching input is unavailable.
Estimate `x[s+1]` from the 99 paired observations with stacked `C A^j`, subtracting known forcing and direct feedthrough.
Use the qualified fixed `rcond=1e-12` minimum-norm solve, then advance all 99 known inputs to `x[s+100]`.
Read `y[k]=C x[k]+D u[k]` before advancing `x[k+1]=A x[k]+B u[k]` for each future input.
Retain rank, singular values and residual diagnostics; rank deficiency is disclosed, not repaired with jitter or a different cutoff.
No periodic tail, unavailable `u[s]`, additional context, future output or pole clipping is allowed.

The [pure initializer](fsm_author/src/openjev_fsm_author/linear_context.py) hash is
`342f0656de39f84fb0070f9253c05ac767a8d348519ff608c130bdbdeeb62267`.
Its [52 fabricated checks](fsm_author/tests/test_linear_context.py) and [independent derivation](fsm-author-causal-contract.md) qualify this positional interface, not author-baseline efficacy.
For public author simulation use explicit `u[128,3,batch]`, `x0[28,batch]` and `offset=None`, including batch one. Its returned state trajectory is pre-update; the last stored state is not the final post-transition state.

## Exact metric reduction and evidence

The [frozen metric helper](../src/openjev/research/fsm_study.py) has source hash
`8ec26ffefedaa4b212fc7eb8b1cb794ce2b4e2106d1f487b49acc9a5ae6e2df5`.
For each record, save float64 forecast/target arrays `[32,128,3]` and int64 starts. With score-space error `e`:

```text
record_mse          = mean(e**2)                         # all 32*128*3 entries
record_rmse         = sqrt(record_mse)
channel_rmse[c]     = sqrt(mean(e[:,:,c]**2))
native_channel_rmse = channel_rmse * benchmark_y_scale
instance_score      = mean(record_rmse over 12 records)
candidate_score     = mean(instance_score over 3 fixed seeds)
```

Do not replace the headline score with pooled-record RMSE, average per-window RMSE, average channel RMSE or the authors' periodic NRMSE.
Report each candidate seed against the same single BLA fit, each record's seed mean, and each amplitude's six-record mean. These windows and periods are correlated, not independent replication units.
Keep invalid forecasts, metric overflow, incomplete records and failed timing slots explicit. Never omit a failed slot to improve an average or replace a stronger-error reference because its timing failed.
An independent audit should reconstruct metrics from saved arrays, join identical targets/starts across compared models, and verify frozen checkpoint/matrix/normalizer identities. State clearly whether inference and raw FIT preprocessing were independently replayed or merely source/receipt-attested.

## Comparable complete-request cost

The [current evaluator](../src/openjev/research/fsm_residual_study.py), source hash
`29a4eb8d99c643c90ab88c82381b03032ea6a4d72cb3225d1e3dd0a5915f7a99`, times batch-one physical-unit requests.
It runs one untimed warmup at the first DEV record/start, then starts 0 and 7936 for each record in canonical order: 24 retained samples per instance.
Report each median, then the arithmetic mean of seed medians for learned families. Retiming on the same current host is required for a new direct latency comparison; historical serial medians are context, not matched present-day measurements.

Include input slicing/normalization, owned conversions, context state inference, all 128 forecast steps, prediction-internal validation and output denormalization. The existing evaluator's final physical-output finiteness check follows its stopped timer; disclose that boundary rather than claiming it is timed.
For this initializer, rebuilding observability and solving least squares are request costs. No request-specific state or future-dependent cache may be reused. Any later persistent factorization optimization needs separately declared storage and parity qualification.
For JAX calls, synchronize completed outputs before stopping the clock; report compilation/warmup separately and retain an outer fit/process clock in addition to author optimizer timings.
Use CPU float64, retain the existing single PyTorch thread for frozen controls, and record native BLAS/JAX thread settings. CPU selection alone does not establish single-thread or exclusive-machine execution. Small serial timing differences are not architectural speed claims.

BLA28 contains 961 matrix entries (`784+84+84+9`), 28 latent state values and twelve author-normalizer values. Charge their actual retained float64 bytes plus retained numeric metadata such as sampling period; distinguish benchmark-only scoring scales from prediction-resident normalization. Charge every additional cached factorization if introduced.
Report temporary least-squares workspace separately from persistent state. Do not count original neural checkpoint/optimizer provenance as required folded-model deployment memory, or exclude a retained author model copy without explanation.

## Boundaries for the new producer

Pin the [author runtime](fsm-author-runtime-contract.md), its resolved environment, FIT recipe, deadlines and final-checkpoint policy before fitting. Record the raw author stop flag and iteration cap separately; neither an exit-zero process nor that flag alone proves convergence, stable dynamics or a global optimum.
Preserve initial/final exports, normalization, FIT identities, process failures and all evaluation attempts. Do not load supplied author weights, select a BLA checkpoint on DEV, silently retry, or expand data access after a weak result.
The 2026 notebook's FIT-restricted BLA28 is an author-method reference on our conditional task, not reproduction of its published score or the older paper's optimizer. NL-LFR remains a separately qualified method.
All compared DEV records are already exposed. A useful BLA result does not convert them into untouched confirmation, authorize 300 mV/test access, or establish novelty or closed-loop control performance.
