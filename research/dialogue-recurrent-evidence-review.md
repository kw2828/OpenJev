# Evidence needed for an autonomous dialogue-memory claim

Source-only review, written while the alignment campaign is running. No current
fit outputs, official DEV/TEST contents, feature arrays, models or numerical
jobs were inspected. This is a recommendation, not an execution protocol.

The next architecture comparison needs **autonomous state tracking against a
strong semantic scorer with an explicit state ledger**, followed by a single
untouched evaluation. Another win with supplied previous gold would establish
conditional decoding only. The current
[alignment protocol](dialogue-token-alignment-scientific-protocol.md) explicitly
supplies correct previous values, excludes nonadjacent scored transitions and
uses historically exposed original-TRAIN rows. Its result cannot identify
recovery from a model's own mistakes or the value of learned recurrent state.
The inherited literal-register features also carry history, so a readout using
them is not history-free.

The [V2 results](dialogue-copy-v2-results.md) already make normalized scalar
candidate memory the practical starting point: selective retention did not
establish its required advantage, and unseen TRUE/DONTCARE interpretation was
weak. This supports improving and testing the observation model before claiming
that a more elaborate temporal update solves the problem. A new recurrent
mechanism should replace only the update/state representation in the matched
comparison, not quietly receive better schema features or stronger supervision.

One bounded comparison, developed on the already exposed material, should use
these four arms with the same three training seeds and dialogue orders:

| Arm | What it controls |
|---|---|
| Explicit candidate ledger with the ordinary learned scalar carry/write rule | Strong simple temporal control, using the same selected observation encoder and features as the proposed mechanism |
| Proposed recurrent update with that identical observation path | Primary mechanism contrast; active parameter and state differences must be disclosed |
| Conventional causal attention over the retained sequence of per-turn evidence, with a matched candidate scorer | Whether storing and consulting history suffices without the proposed recurrent compression |
| The scalar ledger with the same six-layer MiniLM backbone fine-tuned on fitting dialogues | Practical representation-quality control; this has a different trainable capacity and must not be described as a pure recurrence ablation |

Choose the common observation path using development evidence before this
comparison, then keep it identical in the first three arms. The fourth arm
should fine-tune the existing small encoder rather than introduce an unrelated
large model. Its token computation, optimizer state and any schema re-encoding
must be paid; frozen feature caches are invalid after its weights change.
Keeping this comparator affordable needs a measured training allocation, not a
claim that few downstream parameters imply low cost. Retain literal carry as
an additional deterministic accuracy reference.

Train and evaluate whole public dialogue streams from initial NOT_MENTIONED.
Every recurrent arm consumes its own preceding belief, including mistakes;
gold labels contribute only to the loss. Process every USER turn for every
caller-supplied query, including unscored service-frame gaps, without resetting
state or using annotation availability as an update mask. Supply schema queries
as explicit task input, not the private dialogue service inventory. Do not use
the conditional study's adjacent-gold admission filter for this rollout.
For the history arm, retain all earlier per-turn evidence on these short
dialogues, with no future turns. A separately labeled gold-previous diagnostic
may explain an autonomy gap, but must not replace autonomous scores.

The remaining fresh split is **official SGD TEST, untouched by this project
according to the [recorded provenance](sgd-state-source-review.md)**. Official
DEV has already guided several studies; the internal TRAIN holdout is also
development material. Set all training choices and the primary comparison
before accessing TEST, then evaluate the complete eligible categorical cohort
once, without choosing services or examples for favorable support. Preserve
all repeated-seed outcomes. This is project-level freshness, not a guarantee
against pretrained-model contamination. The published benchmark has been public
for years. [Official SGD repository and format](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

Report autonomous changed-state accuracy, genuine revision accuracy, retained
state errors, correct-branch errors, and joint **categorical** state accuracy,
alongside NLL/Brier and every service's support. Use dialogue-level uncertainty
for paired outcomes, with service-level results visible; seeds do not multiply
the number of examples or unseen services. Show recovery after an earlier
mistake descriptively, without conditioning the primary population on a
model's prior correctness. TRUE/FALSE/DONTCARE and clear counts must remain
explicit; missing support cannot support a polarity or correction claim.
Neither these supplied-query scores nor joint categorical accuracy equal the
official full-DST task, which also includes noncategorical values and routing.

A mechanism claim is disconfirmed if the gain disappears against the matched
scalar ledger or history arm, exists only with gold previous state, or comes
from damaging retention to make more updates. If fine-tuning the simple ledger
accounts for the useful gain, the immediate finding is representation learning,
not novel memory. Compare actual whole-training cost and matched streaming
decision cost, with retained history/state bytes, rather than adding nested
timings or presenting a parameter count as a speed measurement.

The novelty boundary is already substantial. [SUMBT](https://aclanthology.org/P19-1546/)
learns contextual slot/value matching, and
[SOM-DST](https://aclanthology.org/2020.acl-main.53/) uses explicit memory and
selective overwriting while analyzing ground-truth-versus-predicted state
operations. Neither schema attention nor copy/carry is new. As a secondary
robustness check after selecting a system,
[SGD-X](https://arxiv.org/abs/2110.06800) provides five crowdsourced schema
paraphrases, but reuses SGD dialogues; those variants are not five independent
fresh conversation sets. Better caller-supplied candidate decisions would still
not demonstrate open-vocabulary value generation, service routing, an
action-conditioned world model or a new RL algorithm.
