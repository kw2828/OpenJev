# Dialogue memory: fixed development comparison

Status: pre-execution protocol. Freeze its hash with implementation, tests, source data and encoded features before fitting. This study tests compact streamed memory for OpenJev's supplied-question/candidate interface. It does not run the official full dialogue-state-tracking task, train a general language model, or establish architecture novelty.

## Task and data boundary

Use the official [Schema-Guided Dialogue](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78) train and development data at commit `e852981ae34990f4358979625854259302feaa78`. These are simulated dialogue outlines paraphrased by people, not organic customer logs. Data is CC BY-SA 4.0; keep source text and individual predictions local and attribute the dataset. Official test dialogue contents are not accessed.

Audit the entire training and development sources. Exclude development dialogues whose normalized full public transcript duplicates training; preserve the original source and exclusion receipt. Retain and report within-split duplicates. Choose the 2,048 training dialogue IDs with smallest SHA-256 of `openjev-sgd-v1:` followed by ID, before looking at labels or dropping conversations without categorical questions. Evaluate every remaining development dialogue with categorical questions. This is a limited-data development pilot, not an official full-training benchmark score.

At each labeled USER frame, the caller supplies one question for every categorical slot of the annotated service, with candidates taken only from that service's declared schema. This grants service/slot routing as task input. Candidate IDs are stable strings; `reserved:NOT_MENTIONED` and `reserved:DONTCARE` are distinct from every `value:` ontology ID, including a literal value named `None`. Candidate order follows the schema after the two reserved entries. Each target must be a singleton declared value, DONTCARE, or absent slot. Reject and retain malformed/multivalued/out-of-vocabulary targets; do not silently map them to a convenient class.

The actor sees only public utterances through the current USER turn and the supplied schema question/candidates. The current SYSTEM reply, annotation actions, spans, state, active intent, service results, dialogue service inventory and future turns are excluded. Each model input step encodes the preceding SYSTEM utterance plus the current USER utterance. A missing service frame makes its target unscored; it never resets memory. Evaluator-only transition categories use the preceding labeled USER frame for the same service. No previous gold value is an actor input.

All questions read from a dialogue stream built independently of annotations. The streamed global-memory arms never receive questions or candidates during an update. The carry control may read the supplied question while replaying its public prefix. This is a stronger question-conditioned control, with its additional work disclosed.

## Shared representation and model families

Freeze `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Encode causal utterance pairs, schema questions, and schema-plus-value descriptions independently. Mean-pool frozen encoder states, then L2-normalize the resulting 384-dimensional vectors. Inputs above 254 content tokens are split into nonoverlapping chunks, weighted by content-token count and combined before normalization. Include all tokens; record chunk counts and encoder work. The shared pretrained encoder is a transformer and may have seen public benchmark content during pretraining.

Compare all seven families, with three fixed seeds per family:

1. Current-turn memoryless scorer.
2. Conventional GRU with 128 hidden values.
3. Full causal softmax attention over projected turn keys and values, with a learned nonnegative recency penalty initialized at .1 per real turn. Padding does not advance age.
4. Gated delta-rule memory, key/value width 16.
5. Dense covariance-tracking Kalman associative memory, width 16.
6. The same Kalman memory with key-local innovation-dependent covariance inflation.
7. Learned categorical CARRY/write control, initialized at NOT_MENTIONED.

The three matrix-memory arms share projection/gate shapes and paired initial tensors. They carry a 16×16 association matrix. The Kalman arms additionally carry a 16×16 key covariance initialized to identity. GRU and carry parameter counts differ; disclose them, rather than claiming parameter or compute matching. Attention retains the turn history; report memory costs without presuming fixed-state recurrence is cheaper on these short dialogues.

The delta update and covariance-tracking equations are borrowed mechanisms. [Kalman Delta Networks](https://arxiv.org/abs/2609.07816) motivates explicit uncertainty over associations. Our dense implementation is a small sequential reference, not a reproduction of that paper's scalable kernels or language-model results. Retention is sigmoid(alpha), observation variance is `(1-beta)/beta + .01`, and isotropic process variance is .01. Use the exact gain and Joseph covariance update under this internal model.

The experimental extension adds covariance only along the current normalized key before the gain update. Its magnitude is `.1 * clip(mean(innovation²) / predictive_variance - 1, 0, 10)`. Predictive variance is computed before inflation; recompute the gain afterward. Gradients flow through the residual. This adaptive-filter heuristic tests whether local surprise helps revise stored associations. It is not exact Bayesian conditioning, a calibrated uncertainty estimate over semantic truth, or a claimed invention of adaptive filtering. Eta zero must recover the ordinary Kalman arm.

The carry control predicts a soft distribution over CARRY plus all supplied candidates. It combines the candidate write mass with CARRY mass times its own previous predicted distribution, never a gold previous state. This is an established baseline motivated by the [official SGD tracker](https://github.com/google-research/google-research/blob/5b09c22d73a9d35eb6c5d2a99b95677a45053466/schema_guided_dst/baseline/pred_utils.py), [SUMBT](https://aclanthology.org/P19-1546/) and [SOM-DST](https://aclanthology.org/2020.acl-main.53/). Our implementation replays prefixes separately per question, so its measured runtime does not represent the most efficient incremental dictionary implementation. Include always-NOT_MENTIONED and exact USER-mention-and-carry references, using longest unique bounded ontology-string matches and carrying on ties. Reserved entries are excluded from matching. This deliberately weak literal reference has no negation, yes/no, DONTCARE, or annotation-span parser; the learned carry remains the meaningful language control.

## Training and reporting

Train 12 epochs with AdamW, learning rate .001, weight decay .0001, gradient clipping 1, batch size 32 and four CPU threads. Seeds are 1729, 2718 and 3141. Each seed uses the same dialogue permutations across methods, generated independently of model initialization. Save final weights only for scored evaluation; do not select epochs, seeds or variants using development scores. All 21 final fits and failures remain in the ledger.

Use weighted state cross entropy with equal total training weight for three strata: unmentioned retention; assigned-value retention; and changes (first assignment, revision or clear). Determine weights from the selected training labels only. Revisions include a change between two assigned values; DONTCARE counts as an assigned value and is separately identifiable. This avoids letting abundant empty/carry labels dominate optimization. No privileged update-operation targets or gold teacher forcing are used.

Report micro accuracy, NLL, Brier score, exact transition-bin accuracy, three-stratum macro accuracy, and revision-only accuracy/counts. Report all, seen-service, and unseen-service development panels. Seen/unseen is defined only by membership in the official training schema. Aggregate fit metrics equally across all three seeds; show every point. A bin with no examples is missing evidence, not zero error. Distinguish repeated fits from independent datasets.

Record encoder preprocessing, training wall time, optimizer updates, number of supervised queries processed, parameter counts, saved checkpoint hashes, evaluation wall time and logical retained-state sizes. Evaluation wall time includes batch construction and saving predictions and excludes shared encoder preprocessing. It is not single-request latency or an end-to-end speedup. Do not infer a memory/compute advantage from accuracy alone.

## Prospective development continuation

The innovation variant is the fixed primary proposal; do not replace it with whichever family wins. Require, on **both seen-service and unseen-service panels**:

- At least one percentage point higher three-stratum macro accuracy than ordinary Kalman memory, averaged across fits.
- Mean per-panel micro NLL no worse than ordinary Kalman memory, equally averaging the three fits.
- Strictly higher macro accuracy in at least two of three paired seeds.
- Macro accuracy no worse than gated delta memory.
- Macro accuracy within one percentage point of the strongest of GRU, full attention and learned carry, chosen by their mean macro accuracy on that panel.

All ten checks plus completion of all 21 fits are required to justify a separately frozen confirmation study. These are practical development screens, not statistical significance tests or ICLR acceptance criteria. Report revision accuracy even if it disagrees with the aggregate decision, and do not make a revision-specific advantage claim without improvement there. A pass still needs official-test confirmation, computation comparisons and broader transfer; a failure closes this fixed recipe without changing its thresholds or selecting another winning seed.

Before training, test causal prefix invariance, candidate permutation, padding, gold/metadata isolation, stable IDs, streaming equivalence, finite gradients and covariance equations. The independent reporter must authenticate every saved prediction against the packet labels, verify complete membership and recompute all metrics without model inference. Preserve all preprocessing or execution failures with concrete receipts; do not silently rerun a scored fit.
