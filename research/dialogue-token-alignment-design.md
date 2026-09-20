# Token alignment for semantic branch decisions

Prospective design only. No new cache, model, fitting or evaluation was executed
for this note. The [typed study](dialogue-typed-results.md) remains failed at
5/9 checks. This proposal neither resumes its fits nor changes any earlier rule.

The [saved decomposition](../output/dialogue-typed-decomposition-v1/diagnostic-01/summary.json),
SHA256 `7de563a2a3903707ff694efdfa5e50f753bd28cf237b5fd1ad5290bcd11b0cef`,
sharpens the question. Mean equal-service changed-state branch/value NLL is
1.8331/0.4061 for flat-stratum, 1.8767/0.6558 for flat-balanced,
1.4921/0.3448 for typed-stratum and 1.3236/0.5273 for typed-balanced.
Despite its lower branch loss, typed-balanced increases mean wrong-selected-branch
counts from 225.67 to 242 relative to flat-balanced. Across the three paired fits,
176 previously correct decisions become wrong and 122 wrong decisions become
correct. These are repeated predictions on the same 578 changed rows, not new
independent examples.

All twelve fits rank TRUE above FALSE within the concrete branch on all 29
TRUE-change rows, yet most final selections are NONE or DONTCARE. Thus the
remaining question is how the observation supports a concrete constraint,
no preference, or the resulting unmentioned state. This does not prove that
the models understand Boolean polarity: the panel has no FALSE changes, and
a within-branch preference for TRUE could produce that finding. NONE is a
resulting value, not a privileged absence-of-speech label. Lower loss, branch
mass argmax and the branch of the selected candidate are distinct quantities.

## One comparison

Test whether comparing schema phrases with dialogue phrases before aggregation
helps branch interpretation. The current [conditional scorer](../src/openjev/research/dialogue_conditional_observation.py)
retains dialogue tokens but compresses each schema/candidate to one vector,
then compresses the dialogue to one evidence vector per candidate. This is a
plausible bottleneck, not a conclusion established by the decomposition.

Propose three arms and three paired seeds, nine fresh fits:

| Arm | Role |
|---|---|
| Flat-stratum candidate attention | Strongest existing decision control, reproduced under the new common run |
| Token comparison with opposite-sequence means | Input- and parameter-matched control for the new token representation and comparison network |
| Token comparison with soft alignment | Primary mechanism: retain local cross-text matches before nonlinear comparison |

For each supplied candidate, use its existing exact schema/value string and the
current preceding-SYSTEM/current-USER context. Project both token sequences
through a shared learned 384-to-64 map. Compute their scaled dot-product matrix
and bidirectional soft alignments, adding the respective token log priors before
masked softmax. Apply a shared small network to
`[token, aligned_token, token-aligned_token, token*aligned_token]`, then take
the prior-weighted mean on each side. Project the two aggregate vectors to a
64-dimensional evidence vector. The matched control uses the opposite
sequence's prior-weighted mean in place of each aligned vector; its projection,
comparison and output tensors are identical and remain active.

Use the existing shared query/candidate projections, lexical features, five
public candidate-type flags and privileged previous-value indicator in all
arms. Keep the flat candidate-plus-branch-score normalization and ordinary
three-stratum training weights. Do not add rare weighting, a typed probability
factorization, operation supervision, inferred entailment labels or recurrence.
Candidates remain caller supplied and order equivariant; no learned service or
candidate-ID table is introduced. The inherited literal features retain their
causal register history, so this is not a history-free text task.

The two new arms share their complete initial tensors and row orders per seed.
Copy compatible common tensors to the baseline and disclose unmatched parts.
Equal active parameters between the new arms do not imply equal computation:
alignment adds the pairwise matrix and reductions. The original baseline is a
separate practical control, not an equal-capacity claim. Keep it even if its
historical result is inconvenient. No old fitted checkpoint is resumed.

## Cache, cost and scope

Reuse the completed context-token cache. It explicitly contains no candidate
tokens, so retain raw frozen MiniLM token vectors for each unique existing
schema/candidate string once, using the same pinned revision and chunking
convention. Restrict this addition to the official TRAIN service split used in
the typed study. Encode every supplied candidate independently of row labels;
do not create target-specific proposition paraphrases. This is a new schema-only
cache, not another candidate-plus-context Transformer pass.

The 64-wide trainable head is small enough to be a plausible CPU comparison,
but runtime has not been measured. Interaction storage and work scale with
`batch * candidates * context_tokens * schema_tokens`; this can dominate the
parameter count. Gather ragged data, reuse context projections, and use fixed
candidate microbatches if needed, without truncating tokens or filtering rows.
Trainable projections must be recomputed after optimizer updates. Charge schema
encoding, both caches, gathering, padding, forward/backward, optimizer and I/O;
report actual peak memory and both parameter and operation counts. Do not infer
equal compute or a speedup from shared frozen encoders.

Retain the existing fit/evaluation membership and all held-out-service rows.
They are historically exposed TRAIN development data; official DEV/TEST are
not part of this comparison. Keep changed/retained and per-service results,
selected-branch errors, within-branch errors, repair/harm counts, TRUE/DONTCARE
recall and their supported false-positive denominators. The 29 TRUE changes,
five DONTCARE changes in one schema, and absence of FALSE changes/clears remain
material limits. Previous-gold carry and literal carry remain accuracy controls.

Before fitting, settle the implementation and metadata-derived token workload,
then use a bounded synthetic capacity measurement to fix the batch/update
recipe and whole-run cap. Publish the objective, pairing, cost limits and one
behavioral continuation rule before scientific execution. This note sets no
new numerical efficacy threshold and predicts no runtime. It proposes one
comparison, not another sequence of kernel qualification variants.

Useful evidence would require improved actual branch decisions against both
controls, with changed-state gains that do not merely increase false updates or
damage retention. A loss-only improvement would repeat the present failure.
If the mean-comparison arm matches alignment, extra token access or comparison
capacity explains the result without an alignment-specific benefit. If all
arms fail, representation quality, supervision, context sufficiency and
optimization remain unresolved; failure does not establish a need for memory.
Even success would establish only a stronger conditional observation component.

## Prior art and novelty boundary

[Decomposable Attention, EMNLP 2016](https://aclanthology.org/D16-1244/)
already aligns two token sequences, compares aligned phrases with a shared
network and aggregates them for natural-language inference. Its fixed-embedding
experiments motivate the compact comparison, but used substantial NLI
supervision. This proposal is an adaptation of that established mechanism.

[ColBERT, SIGIR 2020](https://arxiv.org/abs/2004.12832) establishes separately
encoded contextual token representations with late matching. Its MaxSim
retrieval score is neither entailment nor a calibrated candidate probability;
its retrieval speedups do not predict this experiment's cost.

[LITE, 2024](https://arxiv.org/html/2406.17968v1) learns reductions of token
similarity matrices. Its frozen-backbone ablation improves over fixed MaxSim,
while remaining well below its fine-tuned results. Most reported experiments
also use teacher scores. Its expressivity theorem allows learned encoders and
does not prove that fixed MiniLM features contain sufficient task information.

Neither token alignment, a nonlinear comparison head nor their use before a
future memory constitutes a novelty claim. The immediate scientific question
is whether a less destructive use of the same pretrained representation helps
ground branch decisions. A novel recurrent architecture would still require a
separate task showing that learned temporal inference adds value beyond a
strong observation model and explicit-state controls.
