# Complete-batch observation learning: training path qualified

The fixed cost pilot passed all **40 case/arm cells**, and its independent
saved-output audit agreed. It completed in **75.306 seconds**, including input
authentication, fresh model loading, parity checks, all training/evaluation
work, checkpoint writes and closing checks. This is implementation and cost
evidence using synthetic targets. It is not an accuracy or architecture result.

The [prospective protocol](dialogue-observation-learning-cost-protocol.md) and
[selected-case plan](../output/dialogue-observation-learning-v1/cost-preparation-01/plan.json)
were published before execution. Cases comprise three maximum full batches,
three TRAIN workload extremes, one separate tail and three DEV extremes. Each
starts from the same pretrained MiniLM and fresh normalized scalar memory.

## Measured work

| Operation | Completed |
| --- | ---: |
| Optimizer updates | 52 |
| TRAIN dialogue forwards/backwards | 1,168 |
| DEV forward-only visits | 24 |
| Initial feature-parity dialogue encodings | 412 |
| Total dialogue encodings | 1,604 |
| Actual encoder batch forwards | 3,284 |
| Each of incoming/feature/outgoing/mass checks | 83,732 |
| Qualification checkpoints | 4 |

All measured trainable updates reached the word embeddings and first attention
query with finite nonzero gradients. Frozen encoder gradients remained absent
and frozen tensor digests stayed unchanged. Trainable digests changed. Maximum
initial-vector error was **1.565e-7**, below the 2e-5 limit. Maximum recorded
state normalization error was **2.431e-7**, below the 2e-6 limit. There were no
recorded mass overshoots. Synthetic labels and strata used only scored
positions and public candidate counts, not task answers.

| Arm | Median full-batch update, seconds | Range, seconds |
| --- | ---: | ---: |
| Frozen, original lexical features | 0.808 | 0.750-0.871 |
| Frozen, number-normalized features | 0.848 | 0.771-0.897 |
| Trainable, original lexical features | 1.428 | 1.390-1.599 |
| Trainable, number-normalized features | 1.442 | 1.370-1.607 |

Each row summarizes six measured effective-32 updates across three fixed full
batches, after one warm update per case. These are not six independent datasets.
Update time includes fresh encoding, public-stream recurrence, weighted loss,
backward, gradient checks, clipping, optimizer work and synchronization. Journal
writes, initial parity, loading and checkpoint work are separate components of
the complete 75.306-second phase. All individual warm/measured events remain
in the [saved journal](../output/dialogue-observation-learning-v1/cost-pilot-01/events.jsonl).

Peak process RSS was **1,210,548,224 bytes**. Maximum sampled MPS driver allocation
was **2,352,660,480 bytes**; maximum sampled live allocation was **376,559,104
bytes**. Samples do not establish a true transient peak and must not be added
to process RSS. All stayed within the frozen caps.

Frozen checkpoints were 401,285 bytes each; trainable encoder-plus-memory
checkpoints were 91,286,173 bytes each. They remain local and cannot initialize
scientific fits. Their public hashes are in the completion manifest.

## Audit and next allocation

[Completion](../output/dialogue-observation-learning-v1/cost-pilot-01/completed.json):
`2ba77fed00551a111bdf427d9a53bac5c3edc96dcaa61acae6335c303007bc11`.
[Independent audit](../output/dialogue-observation-learning-v1/cost-audit-01/result-01/receipt.json):
`1b530eca81e650c9bf5874ce5ffb2f3aa9617a24733b4ef58f9dfe400acd611e`.
The audit authenticated 44 sources and seven payloads and independently
reconstructed case selection, work, loss denominators and state-check coverage.
It hashed checkpoint bytes without loading tensors or making model calls.

Actual session 80747 exited 0. The parent recorded **75.639 seconds**, no timeout
and process group 12891 absent. The compute slot was released. No retry occurred.

The complete future twelve-fit inventory has 15,360 updates and 484,080 training
dialogue visits. Applying each arm's largest measured full-batch, single-update
and DEV costs gives a **5.48-hour heuristic**. Adding 25% and ten minutes gives
**7.02 hours**. The prospective allocation is **eight hours**, 8 GiB process RSS,
8 GiB sampled MPS driver allocation and 2 GiB output. Estimated final-checkpoint
storage is 550,124,748 bytes. [Calculation and caveats](../output/dialogue-observation-learning-v1/cost-estimate-01.json).

This is a scheduling heuristic, not a worst-case guarantee or automatic training
admission. Complete runner/reporter source checks and the allocation still must
be bound before execution. The [scientific protocol](dialogue-observation-learning-protocol.md)
and [seven scoring conditions](dialogue-observation-learning-scoring.md) retain
all twelve final fits. Official DEV remains development evidence; official TEST
is sealed. There is no new benchmark accuracy or biological-learning claim.
