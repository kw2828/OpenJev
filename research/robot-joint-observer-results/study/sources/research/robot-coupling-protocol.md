# Does recurrent residual coupling help measured robot prediction?

Prospective development protocol, 24 September 2026. This is a new experiment
after the rejected [phase-memory mechanism](phase-results.md). Nothing from
that experiment's closed recipe or official TEST is reopened. The machine
registration records exact source, data and environment hashes before the
first numerical robot-data load. This document is frozen with that registration.

## Question and limits

Can a small recurrent residual on the mechanical joint chain outperform the
same model with rewired connections or instantaneous messages, while remaining
competitive with conventional dynamics and recurrent predictors?

The common linear AR2 predictor already couples all six joints densely.
Consequently this isolates the topology and history of **residual messages**,
not the value of interjoint coupling in the entire system. The graph is a
mechanical-chain prior, not an anatomical connectome. The cell has no proven
stability or biological-learning guarantee.

This is conditional offline forward prediction. Inputs include future
**realized measured motor torques**, not authenticated issued commands. No
planner, physical deployment, closed-loop control, or reinforcement-learning
claim follows from prediction accuracy here.

## Source and partitions

Use the [Industrial Robot benchmark](https://www.nonlinearbenchmark.org/benchmarks/industrial-robot)
and its [original data release](https://doi.org/10.26204/data/5). The current
[problem description, v2.0](https://www.dfki.de/fileadmin/user_upload/import/15661_Robot_Identification_Benchmark_Description.pdf)
corrects the older linear baseline and coefficient-of-determination report.
This pilot deliberately uses different preprocessing and internal partitions;
its scores must not be ranked against the published prepared-data benchmark.

Each raw file contains three trajectories, each repeated twice. Keep the whole
file together. Use the original script's lexical filename order, which is not
strict time order because one minute field lacks a leading zero:

| Partition | Recording suffixes on 2021-12-15 | Access in this run |
| --- | --- | --- |
| FIT | 20H_29M, 20H_38M, 20H_45M, 21H_11M, 21H_32M, 21H_3M, 21H_41M | Allowed |
| DEV | 21H_54M, 22H_10M | After all fit attempts end |
| Internal confirmation | 22H_41M, 22H_50M | Closed |
| Official TEST | 22H_58M | Closed |

The complete raw archive SHA256 is
`9011509cf901dcb50a8de1e4a5a0fcf6cb8bed2f64945ba0fbbfb8e0b54886fe`.
Every allowed MAT byte stream must match its registered size and SHA256 before
numerical decoding. The official TEST is unsupported by the loader; this
registration contains no confirmation or TEST file descriptor. This grouping
prevents copies of a trajectory crossing partitions but does not create
independent robots or establish a distribution-shift result.

Original archives, metadata, transfer errors and hashes are retained locally
under `output/robot-data-engineering-v1`. Release metadata names both MIT and
CC BY-SA 4.0 without a clear per-file assignment. Do not redistribute raw
measurements as if they were uniformly MIT licensed. Attribute derived results
to the original dataset.

## Causal observations and windows

Decode only measured position, total measured torque, time and recording-valid
status. Position uses secondary encoders for joints 1-3 and motor encoders for
joints 4-6, in degrees. All six total motor torques are in Nm. Exclude reference
trajectories, reference torques, feedback-only torque and measured velocities
from the predictor.

Each recording independently receives a fourth-order 4 Hz Butterworth lowpass
at 250 Hz, implemented as forward SOS filtering with first-sample steady-state
initialization. Select indices 0,25,50,... to get 3,636 points at nominal 10 Hz.
Never use the official concatenated `filtfilt` output for these internal splits.
No filter state crosses a file, and no centered difference or future output is
used to construct a predictor input.

Compute six position and six torque means and population standard deviations
from FIT only after omitting each recording's first 64 decimated rows, in
float64. A zero or nonfinite scale fails the run. These 24
normalizer scalars cost 192 bytes. Neural models use float32; ridge fitting and
physical-unit metric accumulation use float64.

Each forecast conditions on C=32 observed positions and torques. Its first
prediction is q[C], using torque u[C-1]. Recurrent conditioning processes
transitions t=1,...,C-2, then returns q[C-1], q[C-2], u[C-2] and the accumulated
hidden state. Future input is u[C-1:C-1+H]; target is q[C:C+H]. No measured
future position is fed back during prediction. Reset every window and file.

Window starts begin at decimated index 64. Training H=64 (6.4 seconds), DEV
H=128 (12.8 seconds). DEV starts use stride C+H=160, giving 22 complete windows
per recording. Report both first-64 and full-128 errors from the same forecast;
these are correlated horizons, not independent replicates.

## Fixed models and training

The common initializer is a FIT-only ridge regression with penalty 1.0 on all
coefficients, including intercept. Its 25 features are q_t, q_(t-1), u_t,
u_(t-1), and one. Rows stay within their source recording. Output is q_(t+1),
giving 150 coefficients. Save the float64 ridge solution and the float32
initialization. Retain the unfine-tuned ridge as a reference.

| Trained family | Parameters | Explicit recurrent values | Purpose |
| --- | ---: | ---: | --- |
| Chain memory | 266 | 28 | Candidate |
| Rewired memory | 266 | 28 | Same degree and size, different residual topology |
| Chain instantaneous | 266 | 18 | Same encoders, no persistent edge memory |
| GRU residual, hidden width 10 | 1,296 | 28 | Conventional recurrent control |
| Quadratic AR2 | 1,950 | 18 | Conventional nonlinear dynamics control |

For graph arms, ten directed edges connect the six nodes. Physical node order
is [0,1,2,3,4,5]; the rewired order is [0,1,3,2,4,5]. Endpoint/interior degrees
remain equal and external joint identities stay fixed. Each edge encodes
[q_receiver, delta_q_receiver, q_sender, delta_q_sender, u_receiver, u_sender]
with a tanh affine map. Memory updates are lambda*m+(1-lambda)*encoded, starting
with lambda=0.9. The instantaneous control drops lambda*m but retains the same
encoder and decay parameters. Incoming messages are mean-aggregated. A local
tanh residual uses q, delta_q and u per joint. Zero initial local/edge gains
make every graph arm exactly the common AR2 predictor. Encoders become trainable
through the loss as gains move away from zero. Each graph stores 160 additional
bytes of edge-index buffers.

The GRU receives [q, previous_q, u, previous_u] and predicts a residual through
a zero-initialized output head. Its common AR2 base is also trainable. It retains
ten hidden values, giving the same total state count as the memory graph.

Quadratic AR2 uses the 24 nonconstant AR2 features, their 300 upper-triangle
quadratic products, and an intercept. A separate FIT ridge with penalty 1.0
initializes its 325-by-6 map. Both the frozen polynomial and simulation-refined
polynomial are retained. Do not clip predictions or repair divergent runs.

Train all five families at learning rates 0.0001 and 0.001, each with seeds
8101,8102,8103: 30 fixed attempts. Use 1,024 updates, batches of 16, Adam with
betas (0.9,0.999), epsilon 1e-8, and global gradient norm clip 1.0. The loss is
mean squared normalized position error over the full 64-step own-prediction
rollout. Draw the recording uniformly, then a valid start uniformly. Save exact
recording/start arrays for every seed and reuse them across every family and
learning rate. No checkpoint selection, early-success stop or extra recipe.

Use one CPU thread. Each fit has a 600-second cap and the entire run a
3,600-second cap. Preserve initial weights, last/final weights, optimizer,
loss trace, elapsed time and status for every attempted fit. Numerical failure
ends that attempt and makes its entire family/rate recipe ineligible, while
remaining predeclared attempts continue. Source, file-I/O or whole-run cap errors
stop the campaign. No silent retries or continuation of interrupted attempts.

Additional references are last observed position, causal least-squares velocity
from the last five observed positions, and a large direct-history ridge. The
latter reads the last 16 positions ending at C-1, last 16 past torques ending at
C-2, all 128 available future input torques beginning at C-1, and an intercept.
It predicts all 768 output values directly from 961 features. Fit on complete
FIT windows with stride 32 and penalties 1 and 100. It is a high-resource
quality reference, not a parameter- or state-matched comparator. Charge its
weights and input buffer. Every family has access to the same allowed observed
prefix and future torque information; this control accesses the future torque
sequence jointly rather than through a unidirectional rollout.

## Selection, metrics and continuation

Finish all training attempts before decoding DEV. Score every completed model
on both DEV recordings, all six joints, both horizons, without cherry-picking
windows. Report physical RMSE by joint and equal-joint standardized RMSE
sqrt(mean(normalized squared error)). Seed results remain visible.

Select one learning rate per family using pooled full-128 standardized RMSE,
the square root of total standardized squared error divided by its scalar
count over both DEV recordings and all three seeds. A recipe is eligible only if all three
fits and both-file predictions are finite and complete. Ties choose the first
declared rate. Select direct-history penalty by the same pooled DEV metric.
Nonfinite predictions or squared-error overflow are evaluation failures, never
eligible scores. Selection makes this development evidence, not an unbiased
final test.

Advance chain memory to a separately frozen confirmation only if all apply:

1. Every required family has a complete, finite selected recipe; all frozen
   references and selected direct-history predictions are finite.
2. On **each** DEV recording, its mean full-128 error over seeds is at least
   10% lower than each selected compact control: rewired, instantaneous, GRU,
   refined quadratic, frozen linear, frozen quadratic, persistence and velocity.
3. On each recording and each paired seed, it is at least 5% better than each
   of the other four trained families using their independently selected rates.
4. On each recording and each joint, its mean physical RMSE over seeds is no
   more than 10% worse than the selected GRU.
5. On each recording, its mean full-128 error is within 5% of the much larger
   selected direct-history ridge.

All conditions are conjunctive. Report the complete pass/fail breakdown.
Neither a favorable seed nor the shorter horizon can override a failed rule.
Do not open confirmation on a failure, and do not relabel DEV as TEST. A passing
DEV rule merely admits later confirmation, not publication readiness.

Measure full conditioning plus 128-step forecast on a single stream, including
normalization, casting and inverse normalization, with three warmups and 20
timed repetitions. Report median and p95 wall latency, parameter bytes, persistent
state, graph buffers and normalizers. Timing excludes disk/model loading and
is conditional on the registered CPU/runtime. Report context/input/output and
temporary workspace separately where measurable. Parameter savings alone are
not a speedup claim; training throughput is not inference latency.

## Prior art and interpretation

[ReLiNet](https://www.ijcai.org/proceedings/2023/0385.pdf) and
[sparse regularized system identification](https://arxiv.org/abs/2403.03827)
already study this robot benchmark. [Graph-network physics](https://arxiv.org/abs/1806.01242)
and [TRACE](https://arxiv.org/abs/2609.02991) precede physical graph models and
persistent interaction-edge memory. [coRNN](https://arxiv.org/abs/2010.00951)
and [recurrent equilibrium networks](https://arxiv.org/abs/2104.05942) already
study oscillator recurrence and stable recurrent models. This pilot is not a
reproduction or comprehensive comparison of all these methods. A pass would
motivate stronger published baselines and a second environment. A failure
closes this recipe. Neither outcome establishes architectural novelty by itself.
