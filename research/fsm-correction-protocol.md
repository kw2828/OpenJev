# Multivariate recurrent-correction development pilot

Version `fsm-correction-study-v1`, frozen before measurement decoding or empirical
fitting. This is a new, trained-from-scratch mechanism comparison, not transfer
of the failed robot observer. The objective is to test whether decoder-tied
selective correction improves forecasts beyond dense or fixed correction.
Established block-coordinate inference is prior art, not a novelty claim.

## Data and information boundary

Use the authors' [CubeSpec Fine Steering Mirror data](https://github.com/merijnfloren/fsm-benchmark-data)
at commit `539a12fef384b086a8562b500498b2fa3899ef70`. The archive has
19,532,226 bytes, Git blob `1b0650c8cd5b89282eec130034c6c9f0d0971f30`, and
SHA-256 `bdf6004da1342a8746e51580a57b5ddb8ac400ac5caa48368719c33cbb0ef505`.
Its CC BY 4.0 license applies independently of repository code licenses. Cite
Floren et al., *Data-driven state-space identification and nonlinearity assessment
of the CubeSpec Fine Steering Mirror*, ISMA-USD 2024, when using this dataset.
Opaque bytes and source metadata were checked before registration. Measurements
were not decoded for choosing this design.

The three inputs are applied piezo-actuator voltages and the three outputs are
measured displacements. Sampling is 6,400 Hz. Decode only `u_100mV_train`,
`y_100mV_train`, `u_200mV_train`, `y_200mV_train`, each expected to be float64 with
axes `(8192 samples, 3 channels, 6 realizations, 2 periods)`. An admission
mismatch stops the run rather than silently changing the contract. The archive's
300 mV and official-test arrays remain undecoded, including their array headers.

At each admitted amplitude, use realization indices 0,1,2 for FIT and 3,4,5 for
DEV. Each period is a separate chronological record: twelve FIT and twelve DEV
records. Keep orthogonal realization triplets intact. Compute per-channel input
and output means and population standard deviations only over all FIT samples.
No period averaging, wraparound, resampling, interpolation or missing-row removal.

Every request has 100 observed outputs, 99 aligned context inputs and 128 supplied
future inputs. Initialize from `y[s]`; transition context input `u[s+t]` pairs
with `y[s+t]` for t=1..99. Forecast input `u[s+100+h]` predicts output at that same
index. All models omit `u[s]`, receive no future outputs, and cannot use inputs
later than their output horizon. These are conditional forecasts, not closed-loop
control or arbitrary-initial-state recovery. The source contains steady periods.

## Models and training

Seven neural families, each with seeds 9101/9102/9103, make **21 fits**:

- Dense decoder-gradient correction.
- Selective correction of the largest own-gradient state block.
- Rewired selection: cyclic successor of that block, using its own gradient.
- Fixed block 0, fixed block 1, fixed block 2.
- Conventional autoregressive GRU with teacher-forced observed context and its
  own predictions after the observation boundary.

The six correction variants share identical initialization within each seed,
72 latent values in three blocks, the same transition and linear decoder, and
auxiliary heads of width 96 at horizons 1,8,32. They each contain 37,020 parameters.
All latent blocks advance through a dense GRU. Sparse correction does not imply
sparse transition work. The decoder norm is prepared once per conditioning call;
fixed routing omits selection arithmetic and receives its actual measured cost.

The conventional GRU consumes current input and previous output. It has 37,668
parameters and a 75-value caller state, including the last output. Its auxiliary
heads read the same latent-block/input-prefix shapes; they do not directly read
the packed last-output value. This baseline is slightly larger, not falsely
labeled parameter-identical. All models retain their auxiliary parameters when
storage is reported, even though forecasting does not execute those heads.

Use CPU float32, one PyTorch thread, Adam at 0.001, batch 8, exactly 1,024 updates,
and global gradient-norm clipping at 1. The loss is standardized autonomous
128-step MSE plus 0.1 times mean endpoint MSE over the three auxiliary heads.
No learning-rate selection, checkpoint selection, early stopping based on DEV,
pretrained weights or empirical restarts. Each seed has a saved common schedule
of uniformly sampled FIT records and valid within-period starts. Every family
uses that same schedule. Rotate family execution order by seed index.

Each neural fit has a 180-second cap including training/batch construction and
trace output. Check the cap after the last update too. A failure retains its
partial weights, optimizer, trace and error; it is not repaired or silently
removed. Save initial and final states. Clear gradients before inference.
The fabricated throughput probe suggested about 37 ms per update for the
selective family. That estimate is not an empirical speed or quality result.

Fit **nine conventional VARX controls** in float64: orders 8/16/32 crossed with
ridge penalties 1e-6/0.001/0.1. Fit all within-period FIT rows using mean-Gram
ridge, an unpenalized intercept and one Cholesky solve per model. Features are
chronological past outputs, current input, chronological past inputs and an
intercept. Rollouts consume only predicted future outputs. No pole projection,
jitter, stability repair or DEV fitting. Retain every declared configuration.
All neural and VARX fits finish before any DEV predictions.

These controls do **not** reproduce the authors' 28-state BLA or nonlinear LFR.
Published weights saw our DEV realizations, and their periodic initialization
uses end-of-record inputs. Those weights/initialization are inadmissible here.
A later strong-reference comparison must refit on our FIT data and estimate
state from the permitted prefix. Passing this pilot is not benchmark SOTA.

## Evaluation and costs

For each DEV record, request starts are 0,256,...,7936: 32 fixed windows.
Retain every prediction and target, with record/seed identity. Primary error is
RMSE across all windows, horizons and three FIT-standardized output channels,
then the arithmetic mean of record RMSEs and seed means. Also retain per-channel
RMSEs. Periods, windows and channels are correlated; they are not independent
replications. There is one DEV orthogonal triplet per amplitude. This small
pilot reports paired descriptive results, not inflated sample-size significance.

For each model, warm up the first declared request and time the first and last
window of every DEV record, sequentially: 24 complete requests. Include input
normalization, conversion, prefix conditioning, routing/preparation, full rollout
and output denormalization. Report the median and all individual observations;
family cost is mean seed median. This is warm CPU inference from in-memory
records, excluding model loading and disk I/O. Evaluation is separate from fitting.

Persistent numeric storage counts all retained parameters, one stream's state,
and the shared 96-byte normalization values; VARX also reports its 24-byte logical
numeric metadata. Request/output arrays, temporary workspace and Python object
overhead are excluded and stated. Report training time separately. Extra routing
diagnostic replay is charged to evaluation time, not represented as online work.
No new Rust-versus-Python comparison is made by this study.

## Fixed continuation rule

All nine conditions must pass:

1. All 30 declared fits complete; every neural fit reaches 1,024 updates.
2. All 30 evaluations have the exact records and seeds, with finite errors and costs.
3. The selective family is complete across all three seeds.
4. Its equal-record mean RMSE is at least 5% below the strongest complete declared control family, including the conventional GRU and all VARX variants.
5. Its mean seed-median request latency is at most 1.10 times dense correction.
6. Its persistent numeric storage is at most 1.10 times dense correction.
7. No record's seed-mean RMSE is more than 2% worse than dense correction.
8. Every selective seed has lower equal-record RMSE than its paired dense seed.
9. In each selective seed, at least two blocks each receive at least 5% of nonzero-gradient DEV context corrections. A fixed route is not adaptive evidence.

Unavailable comparisons remain failed conditions. A pass only admits designing
a stricter comparison with the strong BLA/LFR references and untouched shift
measurements. A failure does not authorize opening the reserved tests or claiming
routing efficacy. Preserve negative results and distinguish implementation
correctness, developmental efficacy, novelty and generalization.
