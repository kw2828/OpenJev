# TRAIN-only sensor-memory usefulness screen

Version `sensor-screen-v1`, 24 September 2026. Freeze source and registration in
Git before any real-data regression fit or chronological replay. This is a
benchmark qualification exercise, not a novel architecture experiment or a
held-out performance claim. No RL training occurs in this screen.

## Why screen first

The [supplied-law consolidation baseline](measurement-results.md) succeeded,
but a fixed grid, known kernel and competitive DCT control do not establish a
learned-memory advantage. A possible next setting is real sensor calibration.
Before investing in a learned recurrent sketch, test whether static calibration
already explains the data and whether adaptation offers repeatable headroom.
A strong single-sensor quadratic is mandatory. Good calibration fit alone is
not proof that the target was derived from that sensor or evidence of leakage.

[UCI Air Quality](https://archive.ics.uci.edu/dataset/360/air+quality) provides
sensor responses and reference-analyzer targets. The official CSV contains
9,357 dated rows from 2004-03-10 18:00 through 2005-04-04 14:00 and 114 blank
trailing records. This differs from the web page's stated count and date range.
All dated records are hourly, without duplicate or missing timestamps. Missing
measurement sentinel -200 becomes NaN; no imputation, backfill, interpolation
or silent row removal is allowed. The timezone is unspecified. Use naive
calendar hours, not an inferred UTC or DST conversion.

Official source ZIP:
https://archive.ics.uci.edu/static/public/360/air%2Bquality.zip

- ZIP SHA256: `d4a64013fb385288a8a48d9d193ca7079b2e1bbddf6f8d458feb8c08ab2b8a2a`
- CSV SHA256: `13277ae5d8581e80b7be09d47c7d3d06fe9b8e957078f2cf6e859f955e62f996`

The source page has a CC BY 4.0 license section and an older research-only /
no-commercial-use sentence. Preserve both in the retrieval receipt; do not
claim unrestricted commercial rights. Retain the raw download locally. Publish
hashes, source links and this research screen's derived results, not the raw
CSV or retained numeric TRAIN arrays, without resolving that discrepancy.

## Data boundary and causal interface

Only March through June 2004 is TRAIN and numerically accessible in this run.
Other dates may be inspected for schema, chronology and missingness only.
The parser exposes a dedicated `load_train` API; its separate explicitly
registered evaluation API is never called. Save all admitted TRAIN rows with
original 1-based source row IDs, naive integer-hour timestamps, NaNs and masks.

Target is `C6H6(GT)`, in its original benzene units. Public input order is:
`PT08.S1(CO)`, `PT08.S2(NMHC)`, `PT08.S3(NOx)`, `PT08.S4(NO2)`,
`PT08.S5(O3)`, `T`, `RH`. Include all seven. Exclude other ground-truth analyzer
columns and AH. Do not remove S2 or switch targets after seeing this screen.

The screen starts T0 = 2004-05-01 00:00 and ends 2004-07-01 00:00 exclusive.
A label for time t becomes available at t+24 hours. This delay is an artificial
experimental constraint, not a claim about real analyzer latency.

FIT includes exactly rows with all seven inputs and target finite and
`t + 24 < T0`. The strict inequality leaves the Apr30 00 label for the first
screen step. Compute all seven input means and population standard deviations
on these same FIT rows only. Keep these normalizers fixed. Nonpositive scales
or failure of a required solve is an execution failure; no adaptive jitter,
fallback solver or silent refit. Target population standard deviation is a
qualification scale only; target regression values remain in original units.

Process every integer screen hour, including missing/gap hours, in this order:

1. Multiply each adaptive precision and information state by its fixed lambda.
2. Reveal only the label from 24 hours earlier. Assimilate it only if finite
   and its own queued seven-dimensional input is complete.
3. Predict the current input if all seven coordinates are finite.
4. Overwrite that ring slot with the current raw public input, including NaNs.

Initialize the 24-slot ring from public inputs Apr30 00..23 without labels.
Each adaptive model owns its queue. The feed returns only current x and due y;
no model can request old inputs from an external archive. NaNs encode queue
missingness, and hour modulo 24 determines the slot. Source indices in audit
logs are harness metadata, not model-accessible storage.

All nine methods use the same complete-input prediction mask. Score only rows
with complete input and finite target, after the chronological replay finishes.
Unreleased labels may be used for scoring only, never fitting or normalization.

## Nine fixed controls

Four static ridge regressors with designs over normalized input z:

| Method | Columns | Coefficients |
| --- | --- | ---: |
| s2linear | 1, z1 (S2 is zero-based input 1) | 2 |
| s2quadratic | 1, z1, z1 squared | 3 |
| linear7 | intercept and all seven z | 8 |
| quadratic7 | intercept, seven z, all products zi*zj for i<=j in lexicographic order | 36 |

Every static model uses `A = Phi.T @ Phi + 1e-6 I`, `b = Phi.T @ y` and
`beta = solve(A,b)`. Penalize the intercept too. No model or ridge selection.

Four adaptive controls initialize from the corresponding FIT A,b and use
`linear7` or `quadratic7`, each with lambda 1 or 0.995. At each calendar hour
multiply both A and b by lambda, including the first hour. Then add
`phi phi.T` and `phi*y` only for an eligible newly released label. Predict with
`phi_now.T @ solve(A,b)`. These are conventional precision-form recursive
least-squares updates. They are not calibrated Bayesian uncertainty estimates.

The ninth method is persistence: initialize to the last finite target strictly
released before T0, regardless of that row's input validity. Update on every
finite due label before predicting. It does not require a historical input
queue. Report it but do not count it as evidence that RLS adaptation helps.

## Frozen continuation rule

Report RMSE and MAE separately for May and June. Six conditions must all pass:

1. At least 512 complete release-eligible FIT rows.
2. At least 256 scored May rows.
3. At least 256 scored June rows.
4. Best static May RMSE exceeds 0.01 times FIT target population std.
5. Best static June RMSE exceeds that same threshold.
6. One same RLS configuration has RMSE <=0.9 times the best static RMSE in
   both May and June. Best static is the minimum over all four static models,
   independently per month. Persistence is excluded from this condition.

All six pass: `QUALIFIES_LEARNED_MEMORY_SCREEN`. Otherwise:
`REJECT_BENZENE_MEMORY_BENCHMARK`. Qualification authorizes designing a separate
learned-memory comparison, not opening held-out labels or claiming a result on
them. Failure stops development on this benzene-memory benchmark. Do not
weaken the baseline, drop a sensor or move the target to manufacture headroom.
A different dataset or task requires a fresh independently frozen design.
There are no confidence intervals or generalization claims in this TRAIN screen.

## State, evidence and process

This screen has no 1 KB cap and does not compare architectures at matched bytes.
Report actual logical persistent array/scalar storage per method: all normalizers,
dense A,b for RLS, 24x7 float64 raw-input queue, decay scalar, clock and one-byte
design tag. Each RLS queue costs 1,344 bytes before its model state. Static
regressors retain coefficients and the seven means/scales plus the tag.
Persistence retains target, clock and tag. Python/native workspace, runtime,
fit-only matrices and audit artifacts are excluded and explicitly identified.
No RSS, latency, posterior-calibration or packed-symmetric-storage claim.
The earlier [prospective sketch budget](measurement-next.md) omitted pending
inputs and is superseded as a feasible 24-hour deployment budget by this note.

Freeze seven files: parser, producer, independent auditor, their three test
files and this protocol. Qualification uses fabricated data only, including
future-label poisoning, delay-boundary and missing-hour checks. Pin its original
closed successful process receipt in registration. Commit registration and
sources before fitting. Run the registered study once, in a single-thread
numerical environment with a 300-second internal cap. Preserve the original
process log and exit receipt, including any failure. Do not silently rerun,
resume, replace evidence or change the six-condition gate after results.

Save registration, exact source snapshots, admitted TRAIN arrays, predictions,
initial and final states, assimilated/revealed indices, results, logical storage
and a SHA256 file manifest. The independent auditor authenticates the original
closed process first, parses only TRAIN from the pinned raw source independently,
and reconstructs equations and gate without importing the producer or parser.
Compare finite floats with rtol=1e-7 and atol=1e-8, integer/mask/queue data exactly,
with NaNs matched. Verify source pins before and after. A numerical or provenance
mismatch is audit failure, never an alternative result. Audit numerical replay
is disclosed; it is not an additional optimization or study run.

## Boundary of a future claim

[Frequent Directions](https://arxiv.org/abs/1501.01711) already compresses streaming
matrices; [Learning-Augmented Frequent Directions, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/6de668dab370194fa304a08be5aacd85-Paper-Conference.pdf)
already learns or predicts useful sketch directions. [ALPaCA](https://arxiv.org/abs/1807.08912)
learns features and Bayesian priors for online regression. A decision-trained
retention rule would need demonstrable decision value at matched total storage,
including pending-input queues, and independent chronological validation.
A positive conventional screen alone cannot establish that contribution.
