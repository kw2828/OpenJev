# Sampled TRAIN annotations, complete VALID score forecasts

This is a new annotation design and fresh cohort for the four-family score
propagation question. The earlier 67/90 collection remains failed and closed;
its TRAIN arrays are not fitted or reused. The exact-cache diagnostic failed
its opportunity rule, so no cache is used here. No original allocation is
resumed, extended or assigned replacement seeds.

## Scientific comparison

Preserve the mechanism, four model families, parameter counts, 31 public
features, fixed score scale 64, period-four anchors, legal-action loss,
80-epoch Adam recipe, final-checkpoint policy and all 45 continuation conditions
of [the original score-forecast protocol](otto-score-forecast-protocol.md).
All twelve fits use the same sampled TRAIN windows and complete VALID census.
This tests prediction over three steps between queries. It does not establish
persistent memory, autonomous control, biological wiring or novelty.

The only learning-design change is TRAIN window sampling. VALID metrics and
all 45 conditions keep their original complete-trajectory meaning. Fit seeds
are 235001, 235002 and 235003. No tuning or seed/checkpoint selection follows
collection or observed results. All final models must close before VALID
arrays are decoded by the learner.

## Complete fresh trajectories

Collect 54 TRAIN paths, nine paired cases at each sensing length 3 and 4 under
analytic, always-neural and period-four-held-Q control. TRAIN seeds are
23100001-23100009 and 23200001-23200009. Collect 36 VALID paths, six paired
cases per setting, seeds 23300001-23300006 and 23400001-23400006. These are
18 independent TRAIN cases and 12 independent VALID cases, each with three
controller paths. The fresh-seed review found no exact matches in 5,424 scoped
source and prior protocol/seed/manifest files before the new sources were written.

Keep the original hit stratification, rotating arm order, paired source and
indexed observation streams, 53-by-53 public filter, original TensorFlow CPU
teacher with eight-way symmetry averaging, in-bounds action selection and
2,188-move horizon. Retain every trajectory through discovery or its horizon,
including the final public update. Sampled annotations cannot affect a path.

During TRAIN, call the teacher only when the collector deploys it: every
neural action and every fourth held-Q action. Analytic control uses no live
teacher calls. Record all public features, actions and observations. At path
completion, before deferred scores, sample its windows as specified below.
Reconstruct the public belief through every original observation, verifying
every saved posterior hash and mass, including the final update. Score missing
selected rows using the unchanged upstream policy over that public replay.
Reuse already returned deployed scores. Never refresh deployed memory during
annotation. There are no additional native trajectories.

VALID retains exactly one physical teacher call per preaction state. At
skipped decisions, select the action before annotation, as before. Keep complete
teacher scores and all public witnesses for the full evaluation census.

## Uniform TRAIN windows and weights

For a complete episode of length T, use disjoint windows beginning at
0, 4, 8, ... . Let W=ceil(T/4), k=min(8,W), and M=T-W be its full nonquery-row
count. Select k windows without replacement from all W windows, including
short and query-only tails in that population. Tails are neither forced into
the sample nor replaced. Use NumPy PCG64 with seed 23600001 plus the global
TRAIN episode index (0-53); sort selected offsets before label reconstruction.
Each episode has its own independent declared seed. Selection receives only
T and its seed, never scores, losses, outcomes or feature values.

Keep every row of each selected window. Only its first teacher vector becomes
a model input; later vectors are loss targets. Incidental deployed labels
outside selected windows remain recorded but cannot enter fitting. Unscored
flat rows have zero placeholders and an explicit false label mask, not invented
teacher answers. Store deterministic selections and the exact sampled arrays
used by every model.

Each selected nonquery row has weight W/(k*54*M), with zero weight when M=0.
Within the row, average loss over its legal actions. Do not normalize the
realized sampled weights. The expectation equals the original finite-population
episode-balanced loss at any fixed parameter vector; that does not make a
training-selected model's final loss an unbiased estimate of generalization.
All 54 episodes stay in the denominator, including zero-support cases.

Shuffle the same selected windows across all families for a given fit seed.
Every selected window is exposed once per epoch. Multiply each minibatch's
weighted loss sum by selected_total_windows / actual_batch_windows. The W/k
factor appears only in row weights. Retain query-only minibatches and ordinary
zero-gradient Adam steps. All other optimizer settings remain unchanged.

## Prospective resource allocation

One new collection supervisor: **7,200 seconds, CPU1, 4 GiB RSS, 2 GiB output**.
Hard bounds are 90 native resets, 196,920 native moves, 54 public replay resets
and 118,152 public replay updates. At most 144 teacher-policy bindings are
permitted, including one optional deferred binding per TRAIN episode.

| Teacher work | Maximum calls |
|---|---:|
| TRAIN neural deployment: 18*2,188 | 39,384 |
| TRAIN held deployment: 18*547 | 9,846 |
| TRAIN analytic selected-window labels: 18*8*4 | 576 |
| TRAIN held selected missing labels: 18*8*3 | 432 |
| TRAIN total | 50,238 |
| Complete VALID census: 36*2,188 | 78,768 |
| Total physical teacher calls | 129,006 |

The old closed receipt's observed mean teacher-score and native-step costs
project about 5,117 seconds at those hard count maxima, before other overhead.
The 7,200-second cap provides planning headroom; this is not a worst-case timing
guarantee. It is a prospective limit for changed data collection on fresh cases,
not more time for the failed run. Setup, deferred public replay, annotations,
journal I/O and final closure all remain paid.

Keep the original separate **600-second fitting/forecast allocation**, CPU1,
4 GiB RSS and 2 GiB output. At most 432 sampled TRAIN windows produce 14 batches
per epoch and 1,120 updates per fit, or 13,440 across twelve fits. Keep the
separate **120-second saved-output audit**, CPU1, 2 GiB RSS and 128 MiB output.
The audit must verify sampled-data lineage, all final fits, full VALID metrics
and all 45 conditions without new model, teacher or environment calls.

Freeze sources, fabricated qualifications, exact inputs and original parent
limits before each execution. Save attempted/returned work, pending operations
and every partial file on failure. Use exclusive paths; no retries, replacement
cases, resumption, time extensions or fitting on partial collection. A successful
forecast screen would still require a separate fresh autonomous comparison and
fully paid utility-versus-compute evidence before any architecture claim.
