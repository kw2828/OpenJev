# Persistent memory corrected at real queries

This prospective experiment follows the closed [sampled forecast comparison](otto-sampled-forecast-results.md).
Its residual score-feedback GRU lost to the direct GRU. That result stays closed.
Here the single new hypothesis is that carrying memory across actual planner
queries, then correcting it with an observed prediction error, helps forecast
the next three action-score vectors. We do not accumulate predicted scores.
This is a supervised forecast screen, not autonomous control or a new RL result.

## Four matched predictors

All families consume the same 31 public features and only teacher scores at
absolute steps 0, 4, 8, and so on. Hidden state starts at zero for each episode.
Every real query returns its supplied score vector exactly. At other steps a
centered readout of the current hidden state predicts an offset from the last
actual query vector. Scores use the fixed scale 64. No skipped teacher label,
legal-action target, future observation or evaluation result enters the model.

| Family | Parameters | State at a later query |
|---|---:|---|
| `innovation` | 5,978 | Forecast with the old anchor, then add a learned correction from the observed error |
| `innovation_gru` | 5,996 | Forecast with the old anchor, then feed the error and new anchor through an ordinary GRU |
| `persistent_direct` | 5,862 | Process the new anchor in an ordinary GRU while retaining memory |
| `reset_direct` | 5,862 | Reset memory, then process the new anchor in an ordinary GRU |

The candidate has a 35-input, 29-unit GRU and four-score readout. At a later
query it first processes the public features with the old anchor, producing
`h_minus` and the prequery prediction. Let `e` be the centered difference
between the actual new score and that prediction, divided by 64. Update
`h_plus = h_minus + tanh(B e)`, with a bias-free 4-to-29 matrix B. The first
query initializes the ordinary core from its actual score and has no invented
prediction error. The readout and B start at zero. B is trained by backpropagation;
this is a state correction, not an online update to model weights.

The explicit-error control uses 40 inputs and 28 hidden units: public features,
anchor, four error coordinates and a query flag. Its later query performs the
same prequery forecast followed by a second ordinary GRU update with the actual
new anchor, error and query flag. The extra update is included in time and work.
The other two controls use the same base dimensions as the candidate. All
initializations use local seeds without changing the caller's random generator.

State is owned by the caller and never shared across episodes. Query predictions
are excluded from the loss. Padding cannot advance state or enter computation.
Nonquery score slots are poisoned in qualification to verify that they are not
read. Finite checks precede saturated error corrections.

## Fresh, complete data

Collect 54 TRAIN paths and 36 VALID paths, with analytic, always-neural and
period-four-held-Q controllers for every originating case. There are nine TRAIN
cases and six VALID cases at each sensing length 3 and 4. Three controller paths
from one case are not three independent cases. Keep the unchanged 2,188-step
horizon, hit stratification, rotating controller order, paired source and indexed
observation streams, public filter and original eight-view TensorFlow teacher.
Retain every path through discovery or horizon, including its final update.

TRAIN case seeds are 251000001-251000009 and 252000001-252000009. VALID case seeds
are 253000001-253000006 and 254000001-254000006. Fit seeds are 255000001,
255000002 and 255000003. Independent TRAIN selection seeds are
256000001-256000054 in global TRAIN episode order. The prior shorter proposal
collided with incidental historical metadata and was rejected before execution;
both seed reviews remain saved. These longer seeds cleared the recorded scoped
review before the collector was written.

For an episode of length T, let W=ceil(T/4), k=min(8,W), and M=T-W. Draw k
disjoint period-four windows uniformly without replacement, including short
and query-only tails in the population. Keep the unchanged PCG64 selection
helper. Selection sees only episode length and its declared seed.

TRAIN annotations are the union of every period-four query row and every row
in the sampled windows. Reuse already returned deployment scores. Reconstruct
missing labels only after the complete path using the unchanged public replay,
verifying every posterior witness and final update. Annotation cannot affect
the path or refresh the deployed controller's cache. Every query score may enter
the chronological model. Only selected nonquery rows supply training targets.
Incidental labels elsewhere remain recorded but cannot enter the objective.
VALID retains the complete teacher-score census with actions fixed before any
annotation-only call.

## Chronological fitting

Each of 12 fits runs 80 epochs with Adam, learning rate 0.003, zero weight decay,
CPU float32 and gradient norm clipping at 5. Each epoch permutes all 54 episodes
using the fit's local seeded generator. The same seed supplies identical episode
orders across families. Batch size is six complete episodes: nine optimizer
updates per epoch, 720 per fit and 8,640 across all fits. No checkpoint selection,
early stopping, hyperparameter changes or replacement fits are allowed.

Process each episode batch chronologically in chunks of 32 steps. Parameters
remain fixed throughout the batch. Accumulate each chunk's loss gradients,
detach the numerical hidden state between chunks, then clip once and make one
optimizer update after the complete episode batch. Memory crosses both chunk
and query boundaries; gradients cross neither chunk boundary. This is truncated
backpropagation, not full-episode backpropagation. Zero-target chunks may run
without autograd because their final state is detached before the next chunk.
Their forward computation still counts. Zero-target batches still receive an
ordinary zero-gradient optimizer update for every parameter.

Center predictions and targets over legal actions, divide scores by 64, and
average squared error over those actions. Every selected nonquery row receives
weight W/(k*54*M), or zero when M=0. Multiply a batch loss sum by 54 divided by
its actual number of episodes. Do not renormalize realized sampled weights.
All 54 episodes, including zero-support cases, remain in the denominator.

Save the exact chronological inputs, separate selected targets and weights,
orders, costs and final checkpoints. Final TRAIN loss is a fresh rescore of the
selected rows using the final weights. Finish all 12 final fits before decoding
VALID arrays. Evaluate complete VALID histories using actual query inputs only.
No optimizer update or model choice follows VALID exposure.

## Primary comparison and continuation rule

The primary rows are nonquery steps at absolute step 5 or later, after the first
real correction at step 4. Steps 1-3 contribute only to full-trajectory descriptive
metrics. Within a setting or age group, each episode has equal weight; its
eligible rows divide that weight equally. Episodes with no eligible rows retain
their denominator share and contribute zero. Report support alongside every
metric. Preserve the inherited near-minimum rule: eligible scores whose float32
difference from the legal minimum is strictly less than float32(1e-10) form the
near-minimum set. Choose its first action index for the prediction; agreement
accepts any action in the teacher's near-minimum set. Report first-argmin matching
separately. Raw gap is the teacher score of the predicted action minus its best
legal score.

Advance only if all **53 conditions** pass, without selecting a fit or setting:

1. One technical condition: complete collection, all final fits, finite saved
   evidence, original supervisors and independent saved-output audit close.
2. Six support conditions: at least four distinct originating VALID cases supply
   postcorrection rows for each of three ages in each setting.
3. Twelve hold comparisons: for every fit seed and setting, candidate primary
   agreement is at least hold agreement and raw gap is at most 80% of hold gap.
4. Eighteen age comparisons: candidate postcorrection gap is no larger than hold
   for every fit seed, setting and age 1, 2 or 3.
5. Twelve learned-control comparisons: the three-fit candidate mean in each
   setting has agreement at least each control's mean and gap at most 90% of
   each control's mean. Compare all three controls separately.
6. Four full-trajectory conditions: candidate mean agreement and raw gap do not
   regress versus the reset control in either setting.

This is a development continuation rule, not a significance test. Preserve all
per-fit and per-case values. A pass would require a separate fresh autonomous
comparison with total computation paid before any utility or novelty claim.

## Resource admission before collection

First qualify the four predictors on fabricated data. Then run one bounded
synthetic capacity check at six episodes of the maximum 2,188 steps, chunks of
32, period-four queries and eight uniformly selected loss windows per episode.
Each family performs one complete chronological batch with gradient accumulation,
clipping and one optimizer update. Use no recorded trajectory, teacher, native
environment or evaluation data in this qualification.

Before measuring, fix the admission calculation as
`1.5 * 3 * 720 * sum(four full-batch seconds) + 120 <= 5400` seconds.
This includes 50% training-time headroom and 120 seconds for later rescoring,
evaluation and closure, within a 7,200-second fitting allocation. It is a
planning screen, not a timing guarantee. The synthetic supervisor has a
120-second cap, CPU1, 4 GiB RSS and 128 MiB output. If it fails, preserve that
attempt and revise only engineering in a separately qualified implementation;
do not shrink the scientific comparison or collect the cohort speculatively.

Collection allocation: **7,200 seconds, CPU1, 4 GiB RSS, 2 GiB output**.
Maximum physical teacher calls are 10,278 analytic TRAIN, 39,384 neural TRAIN,
10,278 held TRAIN and 78,768 VALID, totaling **138,708**. Bounds remain 90 native
resets, 196,920 moves, 54 public replay resets, 118,152 replay updates and 144
teacher-policy bindings. Setup, I/O, annotation and cleanup count.

Training allocation: **7,200 seconds, CPU1, 4 GiB RSS, 2 GiB output**, all 12
fits, final TRAIN rescoring and complete VALID evaluation included. Saved-output
audit allocation: **120 seconds, CPU1, 2 GiB RSS, 128 MiB output**, with no new
model, teacher, environment or optimizer calls. Before collection, hold the
entire collector, model, training, metrics and audit implementation, fabricated
qualifications and this protocol. Each phase binds its original inputs and
supervisor and uses a fresh exclusive output directory. Preserve all partial
files and failure receipts. No retry, resumption, seed substitution, time
extension or fitting on incomplete collection is permitted.

## Relation to existing work

Prediction followed by observation-dependent latent correction has substantial
prior art, including [Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html).
Error-based recurrent memory updates also appear in
[Gated Delta Networks](https://arxiv.org/abs/2412.06464). This small GRU correction
is not a replication of either architecture and introduces no uncertainty model.
Its narrower empirical question is whether explicit query-error correction buys
anything beyond ordinary persistent recurrence and an explicit-error GRU at
similar parameter counts. The ordinary controls are central to the result.
