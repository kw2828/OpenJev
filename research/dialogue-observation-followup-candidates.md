# Follow-up candidates depend on the observation-learning result

Source review, 20 September 2026; control clarification, 21 September 2026.
This note uses completed reports and three
primary papers. No active scientific predictions, new dialogue examples,
models, tests or timing experiments were accessed. It is a contingent design,
not a protocol, allocation or efficacy claim.

**Do not select another recurrent operator yet.** The running
[clock-corrected observation-learning comparison](dialogue-observation-learning-protocol-v2.md)
is the appropriate next evidence. Its scalar head already receives its own
candidate belief and entropy. The remaining structural question is whether
that state should influence observation processing earlier, not whether to
add state feedback for the first time.

## What the completed evidence supports

- [Copy V2](dialogue-copy-v2-results.md) found scalar unseen macro accuracy
  72.58%, versus 65.77% for readout and 72.89% for selective memory. Selective
  failed its rule; unseen assigned TRUE accuracy was only 6.62% and DONTCARE
  accuracy zero. This does not establish an expressive limitation of scalar
  belief updates.
- [Token alignment](dialogue-token-alignment-scientific-results.md) improved
  changed-value recognition but failed the retention requirements. The
  [objective study](dialogue-objective-results.md) reduced retained error from
  4.82% to 3.95% with row-uniform training, without the required changed-state
  gain. These used correct previous values, so their tradeoff cannot be
  attributed to recurrent error accumulation.
- [Removing Qwen's lexical bundle](dialogue-qwen-lexical-ablation-results.md)
  improved changed accuracy but failed 11/16 conditions. Current-context
  retained errors increased from 1,260 to 1,437 of 7,241. Better recognition,
  more history and fewer hints did not establish better state commitment.
- The [complete TRAIN history audit](dialogue-history-support-results.md)
  found 34/38 apparent distant references expressed as recent number words;
  the other four were ambiguous cross-service transfers. This closed the
  source-separated proposal branch on this cohort. Distant retention alone
  is not evidence against ordinary scalar carry.

## Three mechanisms and their necessary controls

**1. Earlier feedback from predicted state into text processing: the only
conditional architectural candidate worth prioritizing here.**
The Feedback Transformer mixes previous layer representations into a shared
memory that future layers attend to. Its contribution is an earlier
history-to-representation path, rather than another output gate.
[Fan et al., v3, 25 January 2021, section 3.3](https://arxiv.org/html/2002.09402v3#S3.SS3).
The current [encoder wrapper](../src/openjev/research/dialogue_trainable_encoder.py)
encodes public text before invoking scalar memory; fine-tuning changes its
weights, but its current observation vectors do not depend on the preceding
predicted belief.

A small local adaptation could feed a schema-conditioned, belief-weighted
candidate summary into one observation-attention block, before pooling, while
retaining the existing scalar transition. Use the full predicted distribution,
including NONE/DONTCARE/Boolean candidates, and a permutation-invariant summary.
No previous gold, annotation-availability mask or future turn enters this path.
This is an adaptation of established feedback, not a Feedback Transformer
replication or a novelty claim.

The strongest cheap comparator gives the whole model identical text, schema,
lexical history and autonomous belief, but conditions that same encoder block
on a prior-independent schema summary. Both still receive belief in the head;
match tensor shapes, projections, attention calls, optimizer and paid work.
An ordinary predicted-state input token is a second practical control against
crediting sophisticated feedback for simply exposing state earlier. It must
encode the same complete candidate distribution, including reserved candidates,
through a differentiable representation. An argmax-only token would discard
uncertainty and confound the mechanism comparison with richer information.

Both the feedback and state-token controls must propagate gradients through
prior belief, with the ordinary scalar recurrence still attached. Report
effective gradient-path differences rather than claiming parameter equality
proves equal capacity. The feedback path also changes how future losses train
earlier observations. A gain would therefore support the combined forward and
learning mechanism. Attributing it specifically to forward conditioning would
require a separate control that stops gradients only on the encoder-feedback
branch while retaining the ordinary scalar recurrence. This is a condition on
that stronger claim, not an additional experiment admitted before V2 finishes.

The hypothesis is that earlier conditioning improves revision interpretation
without additional false updates. It is falsified if gains disappear against
those controls, require gold previous state, or worsen retention/recovery from
an earlier model error. Wrong beliefs can bias the next representation and
reinforce themselves. A gain in easy retention alone could therefore conceal
poorer recovery. Whole-stream outcomes remain primary; prior-error subsets
are descriptive, not a replacement population. Per-query encoding and loss of
shared turn computation must be included in cost qualification.

**2. Explicit operation memory: a strong comparator before richer feedback.**
SOM-DST separates CARRYOVER, DELETE, DONTCARE and UPDATE, with value generation
only for UPDATE. Its published training uses ground-truth previous state and
operations. [Kim et al., ACL July 2020, sections 3 and 4.2](https://aclanthology.org/2020.acl-main.53/).
A finite-candidate autonomous adaptation would distinguish no update from
clearing to NONE, with the same learned observation path. This could address
the demonstrated switching/retention tradeoff, but is not a new architecture
or equivalent to the already tested candidate-specific retention rates.

Compare with the ordinary scalar under the same loss, inputs and supervision;
if operation supervision is added, give an equivalent auxiliary target to the
control. Preserve all candidates and predicted-state rollout. Continue only if
it improves changed decisions without increased assigned or unmentioned
retention errors and without degrading proper scores. Otherwise a renamed
carry/write gate has not solved the problem. This comparator becomes relevant
only if observation learning improves semantics while a reproducible update
tradeoff remains; a failed encoder study alone is insufficient.

**3. Pretrained associative attention steering: an expensive missing control.**
Delta-mem reads an online matrix before writing and turns that read into
low-rank attention corrections. Unlike the earlier local delta head, memory
changes backbone computation. [Lei et al., arXiv v1, 12 May 2026, section 3](https://arxiv.org/html/2605.12357v1#S3).
The [pinned release review](dialogue-delta-mem-source-review.md) identifies the
Qwen3-4B TSW adapter, additional training exposure and unqualified MLX port.
Its chat runtime retains KV caches and text, so the tiny matrix is not total
memory.

The necessary cheap recurrence control is that same adapter reset at each
exchange, with identical public context, precision, readout and bounded KV/text
policy; an explicit ledger remains the practical comparator. Query-time writes
must not contaminate subsequent public-state updates. Carried state must improve
decisions beyond reset without retention harm. A reset-only gain identifies
adapter training/capacity, not useful recurrent carry. Neither current Qwen
failure nor a MiniLM improvement admits this port or proves long-memory need.

## Decision after the current study

If number normalization or encoder learning resolves the useful tradeoff,
retain the simple scalar and report that result as observation learning. If
semantic errors remain even with adequate recent context, improve their
representation/support before adding memory. Only a completed result showing
useful observation learning plus a reproducible state-dependent residual would
justify qualifying candidate 1 against the explicit operation control. That
residual's cause would still be a hypothesis, not something aggregate scores
identify automatically.

The earlier gated-delta/Kalman experiments and
[recurrent prior-art review](dialogue-recurrent-prior-art-review.md) already
cover ordinary associative updates. No demonstrated graph-local dependency
motivates a connectome here. Adding topology, another gate or more state without
a matched practical advantage is not the next experiment. All earlier failures
and the current study's frozen decision rule remain unchanged; no thresholds
are selected in this note.
