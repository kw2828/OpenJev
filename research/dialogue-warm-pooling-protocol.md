# Can belief-guided observation help an already trained memory?

21 September 2026. Prospective development study, written before new caching,
qualification or adaptation. The [cold pilot](dialogue-belief-pooling-pilot-results.md)
failed its complete rule and scored zero changed-state accuracy in all eight
fits. That recipe remains closed. This separate study starts from the three
completed trained checkpoints and uses the original full cohort and balanced
objective. It does not attribute the cold pilot's failure to any one cause.

## Mechanism and paired controls

Use the same four observation-pooling placements as the cold pilot:

| Arm | Attention query conditioner | Extra key/value token |
| --- | --- | --- |
| `pooled` | None | None |
| `schema_attention` | Uniform candidate mean | Uniform candidate mean |
| `belief_query` | Previous predicted belief-weighted candidate mean | Uniform candidate mean |
| `state_token` | Uniform candidate mean | Previous predicted belief-weighted candidate mean |

The three attention arms have identical parameter dimensions, token lengths and
forward geometry. All four retain the scalar head's own previous probability
and entropy features. The attention output projection is initialized to zero.
The new residual is `pooled + (normalize(pooled + correction) - normalize(pooled))`.
At zero it preserves the stored pooled vector exactly, including float32 norm
rounding. Away from zero its norm is not constrained to exactly one. The original
shared turn projection is evaluated once, then a bias-free projection of the
observation delta is added before tanh. Keep the original one-dialogue batch
layout with all its independent queries. These choices preserve the original
scalar arithmetic at initialization; full-corpus parity is still required.

Belief summaries include every supported candidate, including reserved states.
They can map different distributions to the same vector. No gold prior,
annotation-availability mask or future turn enters the actor. State-bearing arms
retain gradients through autonomous previous predictions. Backward paths differ,
so identical forward geometry is not a claim of equal training cost. The pooled
arm has unused registered attention parameters, reported separately from active
and gradient-bearing parameters.

This is attention-placement head continuation using a frozen, task-trained
encoder. It is not recurrence inside the backbone, a biological connectome,
reinforcement learning, a world model, or exact Bayesian inference. General
prior-state conditioning already appears in [SOM-DST](https://aclanthology.org/2020.acl-main.53/)
and [Feedback Transformer](https://arxiv.org/abs/2002.09402). An eventual positive
result would need stronger controls and external confirmation before a novelty
or transfer claim.

## Immutable starting points and initialization qualification

Restore all three original final `trainable_numbers` checkpoints, seeds 6901,
6902 and 6903, from `output/dialogue-observation-learning-v2/scientific-run-01`.
Authenticate the original complete campaign, source closure, preparation,
supervisor, checkpoint tensors and saved prediction identities. Checkpoints
contain memory and encoder weights but no optimizer state. This is weight
warm-start with a fresh optimizer, not exact training resumption.

| Seed | Checkpoint SHA-256 |
| --- | --- |
| 6901 | `9a6306eaa3b557213bbce0b3102879bc8a52c2003a3c48f2486fb48fc43527f6` |
| 6902 | `332f7a857273bca21050c08a0075c06e4121499eb7cf03a20ae3439dacd4ac7b` |
| 6903 | `28caabb194995f8e77ad038202c90babaf60beb5dd2aa0820812727b8969bac0` |

For each seed, cache that trained encoder using original per-dialogue text order,
254-content-token chunks and batches of 32. Preserve CLS/SEP and exclude PAD.
Use the original content-length-weighted chunk pooling and final normalization.
Store pooled vectors for every dialogue-local text and valid raw token states
only for public-turn texts. Do not globally rebatch or substitute the pretrained
encoder cache. Encoder float32 MPS, eager attention, eval, no dropout and no
gradients. Pooling and scalar memory float32 CPU with one thread. Local pinned
MiniLM assets only, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Before constructing **any optimizer**, qualify all five paths for all three
seeds on all 2,363 DEV dialogues and 62,329 original endpoints: untouched cached
scalar and each of the four zero-adapter wrappers. Save all fifteen complete
prediction files. Against each original saved prediction require:

- Maximum supported log-probability error at most `1e-5`.
- Maximum candidate probability error at most `1e-6`.
- Zero changes in canonical selected candidate ID and exact top-tie masks.
- Finite supported logs, exact negative-infinity padding, and maximum probability
  mass error at most `2e-6`, checked for replay and reference.
- Identical full row identities and unchanged weights/encoder state.

Any qualification failure stops the complete study before adaptation. Do not
relax tolerances, change batching, replace a seed or retry within this study.
Successful qualification is implementation evidence, not efficacy evidence.

## Fixed full-cohort continuation

Use every original prepared TRAIN dialogue (2,017; 51,741 endpoints) and every
original DEV dialogue (2,363; 62,329 endpoints), preserving number-aware lexical
features, actor/target separation and autonomous full streams. These data and
checkpoints have informed earlier development. There is no untouched DEV claim.
Official TEST remains unopened. No new calibration split or temperature fitting
is part of this study; all reported probabilities are raw model outputs.

For each seed, pair all four initial states and the same frozen trained-encoder
cache. Run seed outer, then pooled, schema_attention, belief_query, state_token.
Five complete epochs, effective batch 32 dialogues, microbatch one complete
dialogue, full backpropagation through the public stream. Epoch order sorts by
SHA-256 of `dialogue-warm-pooling-v1:order:<seed>:<epoch>:<dialogue_id>`, with
dialogue ID as tie-breaker. Epoch indices start at zero.

Use fresh AdamW with memory learning rate `1e-4`, attention learning rate `1e-3`,
weight decay `1e-4`, and global gradient norm clip 1. Preserve original stratum
weights in order unmentioned retention, assigned retention, changed:
`[0.5271410232899322, 1.2034749842997696, 3.675831202046036]`. The loss denominator
is the actual number of scored endpoints in the effective batch, including its
short final batch. The encoder stays frozen. No extra training, selection of
checkpoints, early stopping, seed replacement or tuning on DEV.

Save every final state, optimizer-update receipt and DEV endpoint prediction.
Expected work: twelve fits, 320 updates per fit, 3,840 total updates, 121,020
training-dialogue visits, 28,356 adapted evaluation-dialogue visits, plus 35,445
initial qualification-dialogue visits. No quality scoring until the entire
campaign has completed successfully. The untouched control is each seed's
qualified original scalar output; it incurs no adaptation.

## Limits and receipts

Freeze metadata, complete source closure, runtime, input/checkpoint hashes and
this protocol before any new model calls. Exclusive outputs, preserved failures
and unchanged inherited frozen sources are required. Metadata freeze: 120
seconds, 2 GiB RSS, 32 MiB output. Execution: native suspend-inclusive supervisor,
7,200 seconds total, 16 GiB process RSS, 8 GiB sampled MPS driver allocation,
32 GiB output. All loading, cache construction, qualification, fits, transfers,
prediction serialization and worker publication count toward the execution cap.
The wall cap is an allocation, not a forecast based on the smaller cold pilot.

Original per-dialogue metadata bounds all three complete raw-token caches at
20.85 GiB, plus 0.75 GiB for pooled vectors and small offsets/receipts; storing raw
public-turn tokens only is smaller. Memory mapping reduces resident memory, not allocated disk. Check the
complete next cache size against remaining output capacity before constructing it.
Failure or timeout retains partial receipts and receives no complete-study score.
No extension or restart inside this protocol. Validate native parent completion,
reaping, absent child process group, no cleanup errors and successful timing.

## Readout and continuation rule

Use the existing original metric definitions: equal mean of unmentioned
retention, assigned retention and changed accuracy for macro accuracy;
endpoint-mean NLL; candidate-summed multiclass Brier. Primary population is
unseen-service DEV. Also report seen/all panels, each stratum, changed subtypes
and recovery following the model's own previous errors. Missing required support
is a failure, not an omitted condition. Use equal weighting across the three
paired seeds, exact count arithmetic for accuracy thresholds, and float64
probability scoring without a forgiving epsilon or post-hoc temperature.

Advance only if `belief_query` satisfies **all eight checks against each** of
`pooled`, `schema_attention`, `state_token` and `untouched` (32 checks total):

1. Mean unseen macro accuracy gain is at least 0.01.
2. Unseen macro accuracy gain is strictly positive in at least two of three seeds.
3. Mean unseen changed accuracy gain is at least 0.01.
4. Mean unseen endpoint NLL is no worse.
5. Mean unseen endpoint Brier is no worse.
6. Mean seen macro accuracy decline is at most 0.01.
7. Mean seen assigned-retention error increase is at most 0.005.
8. Mean unseen assigned-retention error increase is at most 0.005.

All saved fits and all comparisons must be reported, including failures. A pass
permits a subsequent prospective study; it does not establish significance,
generalization, biological learning, a serving speed advantage or ICLR readiness.
Report actual measured costs descriptively. The shared-cache experiment does not
compare deployed end-to-end latency. A failure closes this continuation recipe
without changing any thresholds or presenting a favorable subgroup as success.
