# Learning observations with fixed autonomous memory

Prospective scientific protocol. Execution is pending the separately pinned
cost qualification, saved-output audit, allocation and complete source review.
No result is established by this document.

## Question and comparison

Does end-to-end learning of the text encoder improve autonomous slot-state
tracking after a simple number-word lexical correction? Keep normalized scalar
memory fixed and cross frozen/trainable MiniLM with original/number-normalized
lexical observations. This is an observation-learning ablation, not a new
recurrent or connectome architecture.

Use the complete [corrected preparation](dialogue-observation-learning-status.md),
completion `d1461a1ea64b23338b2112d581479798c6618c8ccebfbde24ce131059474cf83`,
plan `4c5b2ddead9626e3c4f90819cb50d829f3894ee1fa1bae4adfe249c0ac178c8e`.
Preserve all 2,017 TRAIN and 2,363 DEV dialogues, candidate identities and public
USER exchanges. Official TEST is excluded. DEV has already informed development.

## Twelve fresh fits

Use seed order 6901, 6902, 6903. Within each seed run frozen_original,
frozen_numbers, trainable_original, trainable_numbers in that order. Every fit
starts from the identical pinned MiniLM checkpoint and the same scalar
initialization for its seed. Retain the same prepared epoch permutations across
the four arms. No historical or qualification-trained weights initialize a fit.

Run twenty complete epochs. A microbatch is one full dialogue, with all of its
within-dialogue recurrence retained for backward. An effective batch has 32
distinct dialogues, except the final one-dialogue tail. Backward each microbatch
immediately, using its weighted endpoint loss sum divided by the same actual
effective batch's total endpoint count. Take one global clip and AdamW update
per effective batch. Preserve the prepared three-stratum weights.

Use encoder learning rate 0.00002, memory learning rate 0.001, weight decay
0.0001 and clipping norm 1. MiniLM uses eager float32 on MPS with dropout off;
the scalar uses float32 on CPU with one CPU thread. Re-encode each dialogue and
its schemas on every visit, under both frozen and trainable arms. Use the
qualified 254-content-token chunking and encoder batch size 32. Encoder inputs
remain original text; number normalization changes only the six designated
lexical features. No cached frozen vectors, cross-dialogue encoder batching,
gold reset, unscored-turn skip or recurrent detach is introduced.

Only separate TRAIN endpoint rows supply labels to the loss. The actor receives
public inputs and supplied question/candidate inventories. The actual incoming,
feature and outgoing beliefs and released mass retain all V2 checks at 2e-6.
Check finite gradients/parameters, absent frozen encoder gradients and the
trainable encoder gradient path. Final encoder digests must preserve frozen
weights and show a change for trainable arms.

| Work | Per fit | All twelve |
| --- | ---: | ---: |
| Optimizer updates | 1,280 | 15,360 |
| TRAIN dialogue forwards/backwards | 40,340 | 484,080 |
| Final DEV dialogue forwards | 2,363 | 28,356 |
| Supervised TRAIN endpoint presentations | 1,034,820 | 12,417,840 |
| Saved DEV endpoint predictions | 62,329 | 747,948 |
| Encoder batch forwards | 73,001 | 876,012 |

These totals exclude separately reported prior preparation and qualification
work. Any new parity or other neural operation must be explicitly included in
the frozen plan and actual work ledger before execution.

## Evaluation and records

Save final-epoch weights only, with six trainable encoder states and twelve
scalar states. Do not save optimizer state for resumption. Evaluate DEV once
per fit after training; do not inspect intermediate DEV quality or select a
checkpoint. Save all endpoint log probabilities and canonical row indices.
Keep labels, candidate values, transition bins and service flags in a separate
shared evaluator ledger. Report both precomputed literal-register controls.

The [scoring contract](dialogue-observation-learning-scoring.md) fixes all
denominators, proper scores, seven primary conditions and descriptive factorial
contrasts. Complete, valid execution of every fit is required. A failed fit
cannot be replaced by another seed or excluded from the study result.

Keep one compact record per effective training batch with order identity,
endpoint denominator, attempted/completed operations, encoder work, actual
state-check counts and maxima, and elapsed time. Evaluation records cover all
public updates and extracted endpoints. Retain partial journals, active progress,
failure receipts and any partial prediction artifact if a run fails. Report
all attempted work, including incomplete work, separately from quality.

## Allocation and stop boundary

Before scientific execution, publish a hash-bound allocation JSON containing
the measured cost completion and independent audit identities, this protocol
and whole-study wall/RSS/MPS/output limits. The scientific plan separately binds
that allocation, the complete source closure, runtime, orders and evaluator
ledger before training starts.
There is no default training allocation. Estimate duration from the complete
prepared workload inventory and measured qualification costs, label the estimate
as a heuristic and include explicit overhead and headroom. Sampled MPS memory
is not a true transient peak and is not additive to process RSS.

Use an exclusive output directory and an independent parent process-group
watchdog matching the frozen wall limit. Coordinate shared compute before
launch. A timeout, resource limit, invalid gradient/state, changed source/input
or execution error stops the run and preserves its evidence. No resume, retry,
replacement seed, trimmed cohort, device fallback or cap extension is permitted
inside this frozen attempt. Saved-output auditing runs no model.

Publish code, aggregate results and receipts. Keep reversible inputs, labels,
individual predictions and qualification weights local. The source dialogue
dataset retains its CC BY-SA 4.0 license. Any eventual pass supports only this
development comparison and a separate untouched confirmation study.
