# Two conditional mechanisms beyond belief pooling

Prospective source review, 21 September 2026. This note uses completed published
results, model source and primary papers. No new warm-study predictions, fit
quality, dialogue examples, checkpoints or numerical experiments were read or
run. It is not a protocol, allocation or outcome claim.

**Do not respond to a pooling failure by automatically increasing memory.** The
current study tests attention over already contextualized, frozen token states.
It cannot establish what feedback inside the encoder would do. Conversely, a
gain would not establish that longer history or a predictive world model is
needed. State conditioning, low-rank steering and fast-weight learning all have
substantial prior art.

## What the existing evidence actually motivates

[Autonomous observation learning](dialogue-observation-learning-v2-results.md)
improved unseen macro accuracy from 76.46% to 79.34%, predominantly through
retention, while changed accuracy slightly declined and raw NLL worsened in
every seed. The completed [calibration control](dialogue-calibration-runtime-v2-results.md)
improved trained-number NLL without changing any choices, but failed 9/11 because
the equally calibrated frozen model worsened against its own raw scores.
Neither result identifies an insufficient recurrent state.

The [earlier conditional studies](dialogue-observation-followup-candidates.md)
also exhibited semantic and switching errors with correct previous state
supplied. The [complete history audit](dialogue-history-support-results.md)
found recent number-word expressions in 34/38 apparent distant-reference cases;
the remaining four were ambiguous transfers. Scalar carry already retains
categorical state indefinitely. Supplied candidate inventories and these short
dialogues therefore provide weak evidence for a long-memory bottleneck. No
agent actions or transition interventions are modeled in the present API.

The [current model](../src/openjev/research/dialogue_warm_pooling.py) already
feeds autonomous belief and entropy into the head. Its state-token control adds
one pooling key/value; it is not a state-conditioned contextual encoder. These
boundaries matter when interpreting its eventual result.

## 1. State feedback inside contextualization

**Mechanism.** Retain the trained scalar transition, but let the previous
predicted candidate distribution influence one upper encoder block before
current text tokens are contextualized. Represent each supplied candidate by
its schema embedding, type and belief mass; retain NONE, DONTCARE and Boolean
candidates. A small adapter can read this permutation-equivariant candidate
bank to modify token attention queries. This changes token-to-token
interpretation, rather than selecting among final token vectors. Initialize
the adapter as an identity and use no previous gold state.

**Nearest prior art.** SOM-DST puts previous dialogue state into its encoder
input and separates carry/delete/dontcare/update operations. Its original
training uses gold previous state, unlike the proposed autonomous rollout.
[Kim et al., ACL 2020, sections 3 and 4.2](https://aclanthology.org/2020.acl-main.53.pdf).
Feedback Transformer lets current layers attend to a memory mixing previous
layer representations. Earlier high-level feedback itself is established.
[Fan et al., arXiv v3, 25 January 2021, section 3.3](https://arxiv.org/html/2002.09402v3#S3.SS3).
Neither paper establishes this finite-candidate adaptation's advantage.

**Decisive matched comparison.** Compare the adapter with both a schema-only
conditioning branch and ordinary candidate/state tokens inside that same upper
encoder block. Every model retains the same full belief in the scalar head;
the state-token comparator receives the same candidate probability bank, not
an argmax label. Match initialization, text, labels, updated layers and training
budget; record any remaining parameter, attention-work and latency differences.
The simple state-input route is the practical comparator, not the current
single pooling KV. Keep the untouched trained model and adapted pooled model
as references for extra training. A feedback-branch stop-gradient ablation is
needed only to distinguish forward conditioning from added temporal credit
assignment, since otherwise the intervention changes both.

**Falsifier.** Require improved revision/changed decisions with retention and
raw proper-score safeguards against the strong state-input comparator, under
complete autonomous rollout. Improvement only over schema attention is
insufficient. Wrong beliefs can bias interpretation and reinforce themselves;
model-specific error-recovery subsets alone cannot establish robustness. A
separately fixed common state-perturbation diagnostic can test recovery, but
does not replace unperturbed task performance. This branch is worth qualifying
only if the completed warm result supports useful state-dependent processing
beyond matched continuation, or a separately justified ambiguity test does.

## 2. Predictive fast state that steers the observation path

**Mechanism.** Preserve explicit candidate belief and add a small per-query
fast predictor. Before the current USER turn, predict a fixed observation
feature from past public context, schema, autonomous belief and the available
SYSTEM turn. Once USER text arrives, use the signed pre-update prediction
residual to steer an internal attention adapter. Update fast state only after
forming the current decision; reset it at dialogue boundaries. No state label
or correctness signal enters its online update. A fixed target representation
prevents the auxiliary target from collapsing jointly with the predictor.

**Nearest prior art.** TTT treats recurrent state as model weights updated by a
learned reconstruction objective; online self-supervision is already its main
mechanism. [Sun et al., arXiv v1, July 2024, sections 2.1-2.3](https://arxiv.org/html/2407.04620v1#S2.SS1).
Delta-mem reads an associative state before writing, then steers attention with
query/output corrections. [Lei et al., arXiv v1, May 2026, sections 3.2-3.3](https://arxiv.org/html/2605.12357v1#S3.SS2).
The possible local distinction is delayed public-observation prediction and
its use in categorical decisions, not new delta algebra. The
[existing note](dialogue-recurrent-prior-art-review.md) already proposed that
distinction; it has not acquired empirical support merely through repetition.

**Decisive matched comparison.** Hold the observation backbone and scalar head
fixed across carried fast state, the same adapter reset each exchange, and a
causal recent-window ridge/RLS predictor feeding the identical residual path.
Give each the same public information and account for state, updates and encoder
work. Any pretrained delta-mem comparison must separately match KV/text
retention: its [released chat runtime](dialogue-delta-mem-source-review.md)
retains those alongside its small matrix. A reset-adapter improvement does not
show useful recurrent carry.

**Falsifier and prerequisite.** Future USER features must first contain useful
predictable structure beyond SYSTEM text and recent-history controls. User
choices may be inherently unpredictable. Better embedding prediction without
better decisions is insufficient; a gain must also survive the linear and
reset controls. This mechanism is not justified by the current SGD evidence,
and passive dialogue prediction is not an action-conditioned world model.

## Decision boundary

Only mechanism 1 is a plausible narrow continuation on this task, and even it
is conditional. Finish and report the current frozen comparison first. If its
advantage disappears against competent pooled/state controls, close that
recipe rather than lower a gate or declare another memory design necessary.

For a stronger recurrent-architecture claim, a different task may be required.
First establish that identical recent observations can require different
decisions because of earlier state, and that controlled actions produce
different future observations. Compare with a strong GRU, finite-window model
and explicit state/filter baseline under matched information and compute.
The task must require nontrivial inference, not merely reward a hand-coded
ledger on an easy counter. That is a prerequisite for a later benchmark choice;
this note specifies neither a new benchmark nor an execution protocol. No combination of the known
pieces above is presented as paper novelty, and no easier success criterion is
substituted for the architecture-performance goal.
