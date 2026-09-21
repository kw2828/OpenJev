# Does predicted state help select the next observation's evidence?

21 September 2026. New development pilot, specified before selecting examples
or running this model. Earlier experiments and their failed rules remain intact.

The public Qwen RLCD example is an inference optimization, already assessed in
[our source review](qwen-parallel-source-review.md). Shared context can reduce
serving cost; it does not establish better learned state. The completed
[calibration control](dialogue-calibration-runtime-v2-results.md) improved the
trained model's scores but failed its complete rule. Neither result establishes
a novel architecture advantage. This experiment returns to the learning goal.

## Mechanism and controls

Cache the same frozen, pretrained MiniLM token features for every arm. At each
public turn, the model's previous categorical belief defines an expectation of
candidate embeddings. All supported candidates participate, including NONE and
DONTCARE. A weighted embedding summary can collapse different distributions;
it is not a lossless belief representation. No previous gold state, annotation
availability mask or future turn enters the actor.

The same unchanged normalized scalar memory then updates its belief from the
observation. Every head already receives its own previous probability and
entropy. The comparison changes where that state enters token pooling:

| Arm | Attention query conditioner | Extra key/value token |
| --- | --- | --- |
| `schema_attention` | Uniform candidate mean | Uniform candidate mean |
| `belief_query` | Previous belief-weighted candidate mean | Uniform candidate mean |
| `state_token` | Uniform candidate mean | Previous belief-weighted candidate mean |
| `pooled` | No additional attention | No additional attention |

The first three share parameter shapes, projections, key/value lengths and
forward work. Both state-bearing arms retain gradients through their own
previous predictions. Their backward paths differ, so equal forward geometry
does not mean identical training compute. The fourth arm is a cheaper capacity
reference. Its unused attention parameters are reported separately from active
parameters. All four receive identical fresh scalar weights per paired seed.
The attention output projection starts at zero. All four apply the same final
normalization to the observation, giving identical initial outputs in the
synthetic float32 check. This extra normalization is also applied to the new
pooled control; it is a numerical difference from the older pooled implementation.

This tests attention placement after a fixed encoder, not recurrence inside
the backbone, biological wiring, JEPA, reinforcement learning or a world model.
The state-token control is one appended key/value in this pooling layer; it
does not represent every encoder architecture that accepts previous state.
SOM-DST already conditions its encoder on prior state; Feedback Transformer
also supplies earlier recurrent feedback. General state conditioning is not
the novelty claim. [SOM-DST](https://aclanthology.org/2020.acl-main.53/),
[Feedback Transformer](https://arxiv.org/abs/2002.09402).

## Fixed pilot

- Use the existing prepared SGD TRAIN and DEV actors, separate target records
  and number-aware lexical features. Official TEST remains unopened. These
  development data have informed earlier research; no untouched-data claim.
- Select 128 TRAIN and 128 DEV dialogues by ascending SHA-256 of
  `openjev-belief-pooling-pilot-v1:<split>:<dialogue_id>`. Break ties by dialogue
  ID. Selection uses public identity only, with no outcome filtering.
- Seeds 7101 and 7102; seed outer, then pooled, schema_attention, belief_query,
  state_token. Fit every arm. Three complete epochs, paired hash-permuted orders,
  effective batch eight dialogues, full autonomous streams, endpoint-uniform
  NLL, AdamW learning rate 0.001, weight decay 0.01, gradient norm clip 1.
  No early stopping, best-checkpoint selection or tuning on DEV.
- Fixed pretrained `sentence-transformers/all-MiniLM-L6-v2` revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, local files only. Encoder float32
  MPS/eval/no-grad; attention and memory float32 CPU. Use unchanged 254-content
  token chunks and batches of 32. All valid chunk token states including
  CLS/SEP are retained for attention, without padding/truncation. Original
  content-length-weighted chunk pooling and final normalization remain exact
  for schema vectors and the pooled turn residual. The new attention treats
  retained token positions equally before learned weighting.
- Metadata/source freeze before any pilot input or model evaluation. Bind the
  inherited source/data/model manifests, new sources and installed runtime.
  Exclusive run outputs, final weights and every evaluated endpoint saved.
  No retries, replacement seeds or resource extensions within this pilot.
- Preparation: 120 seconds, 2 GiB RSS, 32 MiB output. Execution: native-clock
  supervisor limit 1,800 seconds, 8 GiB process RSS, 8 GiB sampled MPS driver
  allocation, 2 GiB output. Charge loading, caching, transfers, all fits and
  publication. On failure retain partial evidence and do not score it as a
  complete comparison. Shared caching is a development convenience; it is not
  a serving-latency comparison with previous trainable-encoder studies.

## Fixed readout and continuation

Only after all eight fits succeed, compute DEV accuracy, NLL and multiclass
Brier from saved endpoint probabilities. Report all four arms and both seeds;
the primary population is the complete selected DEV cohort. Report seen/unseen
services, the three transition strata, changed subtypes and recovery after a
previous model error descriptively. Missing subgroup support is reported,
never replaced or treated as success. Accuracy's primary aggregate is the
equal mean of unmentioned retention, assigned retention and changed accuracy.
Proper scores average endpoints; Brier sums candidate squared errors.

Advance to a larger matched experiment only if `belief_query`, against **each**
of `pooled`, `schema_attention` and `state_token`, satisfies all of the following:

1. Mean paired macro accuracy improves by at least 0.01, with a strictly
   positive macro improvement in each seed.
2. Mean endpoint NLL and Brier are each no worse.
3. Mean unmentioned-retention and assigned-retention error are each no worse.

These are six checks per comparator, eighteen in total (the positive-each-seed
condition is one check). All required strata must have support, all scores
must be finite, and the full fixed work and provenance must validate. A pass
admits a larger development experiment only; it does not establish novelty,
statistical significance, transfer, or ICLR readiness. A failure closes this
pilot recipe. Do not quietly expand epochs or select a different seed.
