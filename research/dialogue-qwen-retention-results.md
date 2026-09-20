# Why Qwen lost state in the dialogue comparison

The completed saved-output diagnosis identifies a state-update problem worth
testing before adding recurrence. Qwen frequently discards supplied prior
values, while longer history increases assignments to fields that the reference
state leaves unset. The fixed Astra6 review also finds two cases where accepting
an action and changing a preference are not clearly distinguished by the task.
These observations select hypotheses; they do not establish a cause or improve
the [original failed result](dialogue-qwen-observation-results.md).

## Complete-cohort findings

Both arms contain the same 7,819 exposed development rows and receive the correct
previous state. No new predictions were generated. Of these rows, 7,241 retain
their reference value and 578 change it.

| Retained reference type | Rows | Current errors | History4 errors |
| --- | ---: | ---: | ---: |
| NOT_MENTIONED | 4,032 | 461 | 780 |
| TRUE | 551 | 521 | 531 |
| Ordinary OTHER | 2,648 | 273 | 248 |
| DONTCARE | 10 | 5 | 10 |
| FALSE | 0 | 0 | 0 |
| All retained values | 7,241 | 1,260 | 1,569 |

The TRUE failures are mainly resets: current selects NOT_MENTIONED on 355 rows
and DONTCARE on 165; history4 selects them on 358 and 170. No retained FALSE
examples exist, so this cannot establish a general boolean asymmetry.

For previously unset fields, current incorrectly selects an ordinary value on
307 rows and history4 on 510. The selected literal occurs in the latest SYSTEM
utterance on 235/307 current rows, versus 23/307 in the latest USER utterance.
For history4, those counts are 201/510 and 20/510; the earlier SYSTEM window
contains it on 300/510. These overlapping literal flags are consistent with a
proposal-versus-commitment problem, but do not prove one. A string occurrence
does not establish acceptance, negation, relevance or semantic support.

All previous/target/selected type combinations, correctness, transition
subtypes and six services are retained in the
[complete diagnostic summary](../output/dialogue-qwen-retention-diagnostic-v1/run-01/summary.json).
The [15,638 row records](../output/dialogue-qwen-retention-diagnostic-v1/run-01/rows.jsonl)
also preserve prior probability, its fixed bin and competition rank, selected
probability and separate literal flags. Earlier history flags are explicitly
diagnostic-only for current-arm rows: that model did not see those utterances.
TRUE, FALSE, DONTCARE and NOT_MENTIONED have no literal-match interpretation.
No probability threshold, selector or replacement policy was fitted.

## Fixed blinded Astra6 interpretation

The [protocol](dialogue-qwen-retention-diagnostic-protocol.md) was published after
the aggregate Qwen result and before individual case inspection. It selects two
distinct dialogues from each of six error-conditioned strata using a fixed
hash order. The resulting 12 cases belong to 12 dialogues. This is not a random
sample of the evaluation set and cannot estimate error prevalence or Astra6
benchmark accuracy.

Separate current/history4 packets preserve exact public contexts, questions and
candidate descriptions. They omit row routing IDs, reference labels, model
choices, scores and selection strata. Candidate IDs are stable; the canonical
mapping is held separately. Each fresh reviewer received only its own packet
and the same [neutral instructions](dialogue-qwen-review-instructions.md).

Three of four assigned reviews completed. `current-02` failed to spawn with
`agent thread limit reached`, before a review context or response existed.
That failure is preserved without replacement. Both history4 reviews completed
in separate fresh contexts. The parent read no responses until all four
assignments were terminal, then ran the saved-only join.

| Assigned review | Valid choices | Reference agreement | Qwen agreement | Ambiguity flags |
| --- | ---: | ---: | ---: | ---: |
| Current 01 | 12 | 10/12 | 4/12 | 0 |
| Current 02 | Missing | Undefined | Undefined | Undefined |
| History4 01 | 12 | 10/12 | 4/12 | 1 |
| History4 02 | 12 | 10/12 | 4/12 | 2 |

The history4 pair agrees on all 12 choices. All three completed reviewers also
choose the same candidate across the two contexts for every case. Current-arm
within-condition agreement is unavailable because its second review is missing.
The reviews are interpretations from one model family, not independent human
annotations or verified ground truth. All 36 submitted choices are retained;
none is unresolved or structurally invalid. Exact support text is present for
all 36, which checks quotation only, not entailment.

| Fixed selection stratum | Cases | Completed reviewers' common interpretation |
| --- | ---: | --- |
| Retained TRUE, current wrong | 2 | Preserve the supplied unisex/wifi requirement when another field changes |
| Retained OTHER, current wrong | 2 | Preserve Sports while the user confirms details or moves to banking |
| Retained DONTCARE, current wrong | 2 | Accept the proposed playback device; conflicts with both reference labels |
| Previously unset, current right/history4 wrong | 2 | Preserve the authoritative prior state; history introduces preference-versus-property ambiguity |
| Changed TRUE, current wrong | 2 | Explicit requests for a unisex salon imply TRUE |
| Changed OTHER, current wrong | 2 | A concert implies Music; three stars and two rooms refer to different slots |

The two DONTCARE cases matter. In one, SYSTEM asks whether to use the kitchen
speaker and USER accepts. In the other, USER confirms TV playback. All reviewers
and both Qwen arms choose the proposed device, while the reference retains
DONTCARE. Under the public question's current wording, that is a defensible
interpretation. It does not license rewriting the labels: a dataset may store
search preferences separately from the chosen execution device.

The two history-related cases flagged for ambiguity make the same distinction elsewhere:
accepting a particular property or salon need not create a new search constraint
for each of its attributes. The current question does not fully specify that
boundary. Most of the sampled retained TRUE/OTHER failures, however, require
only preserving the supplied previous value, which all completed reviewers do.
Semantic ambiguity therefore does not explain away the full retention failure.

## Next hypothesis and limits

The next control should remove the ten lexical flags and their explanatory text
while preserving raw dialogue, task semantics, previous value, candidate types
and IDs, label ordering, model and scoring. Compare with the saved unchanged
baseline in both context arms over the complete cohort. Report paired repairs
and harms, changed accuracy, both retention subtypes, NLL and Brier, including
every service. Specify the execution allocation and decision rule before any
new inference.

The sampled three-star/two-room case motivates this test: its flags mark the
number 2, Qwen chooses 2 for stars, and all completed reviewers choose 3.
This does not prove feature distraction. Removing the flags also removes genuine
public-prefix information carried by the literal register, so the comparison
would measure the net effect of this input representation. No such ablation has
run yet. Explicit KEEP/SET wording can wait: the existing question already
instructs preservation, and stronger copying alone could hide real updates.

These inspected rows remain development data. Any later architecture claim
needs fresh evaluation, autonomous predicted-state rollouts and a scenario
shift. Neither this next control nor model agreement resolves disputed label
semantics automatically.

In particular, the diagnosis does not promote a recurrent adapter, a connectome
architecture, a calibration claim or a new RL algorithm. It motivates a cleaner
observation/state-update control before learning when and what to write into
memory. The original four failed comparison decisions remain unchanged.

## Verification and artifacts

The diagnostic completed once in **2.871 seconds**, with **275,759,104 bytes**
process peak RSS and **12,902,061 bytes** of output. It uses zero model,
tokenizer, checkpoint or training calls. Its authentication, canonical score
reconstruction and main-report reconciliation reuse the pinned independent
result auditor; this is not another independent audit.

The saved review join completed once in **0.0228 seconds**, with zero model
calls. Those numbers describe local analysis, not Astra6 inference cost. There
were three completed Astra6 agent review tasks and one rejected dispatch. No
API-call count, model token cost or inference latency comparison is claimed.

Synthetic diagnostic checks: 11 passed. The initial lint failure is preserved;
its comment-only fix is AST-equivalent and passes lint. The join's original 16
checks passed; after keeping quote mismatches in valid-choice denominators, 11
affected checks passed with six unchanged checks deselected. Final code passes
lint. Source and data hashes were verified before execution and at completion.

| Artifact | SHA256 |
| --- | --- |
| [Diagnostic receipt](../output/dialogue-qwen-retention-diagnostic-v1/run-01/receipt.json) | `01e96b70c37ac436076c3861e02bccdaa63629209f2eefc13e23c8598525b1e0` |
| [Fixed dispatch](../output/dialogue-qwen-retention-diagnostic-v1/reviews-01/dispatch.json) | `120b3f2384dbfcff82c66e08a1fe6f852f16b0dc83fd61b69fc6e7ba7541d0f4` |
| [Current packet](../output/dialogue-qwen-retention-diagnostic-v1/run-01/current-review.json) | `c3947b689ad1b3efa8e9c1962cbeadac852a6f74ae198d7615988fa2f187d5f3` |
| [History4 packet](../output/dialogue-qwen-retention-diagnostic-v1/run-01/history4-review.json) | `f270cac1b6dfc85fba5e8a3b906732966bd5369bf3e58af5e0c6caa976db08c9` |
| [Joined reviews and all interpretations](../output/dialogue-qwen-retention-diagnostic-v1/join-01/summary.json) | See the receipt's member hashes |
| [Join receipt](../output/dialogue-qwen-retention-diagnostic-v1/join-01/receipt.json) | `8cdf7f45d65e719e869d83cab62aed51b375ccaf4d553d1600dc01f9640d56a4` |

The [diagnostic](../scripts/diagnose_dialogue_qwen_retention.py) and
[joiner](../scripts/join_dialogue_qwen_reviews.py) are reproducible saved-only
tools. Preserve their exclusive output directories when running them again.
The prompt excerpts are derived from SGD and retain its **CC BY-SA 4.0** terms;
see the [dataset provenance and attribution](dialogue-memory-reproduction.md).
Publication does not relicense the dataset or Qwen model.
