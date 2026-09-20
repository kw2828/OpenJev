# Text-encoder learning through autonomous memory: qualification passed

The single real-checkpoint pilot completed all six workload/arm cells in
**8.147 recorded phase seconds**, or **8.471 seconds including process startup
and shutdown**. The trainable path delivered finite nonzero encoder gradients
through the CPU recurrent state; the frozen path preserved its encoder weights.
This is an implementation and cost result. It measures no task accuracy and
establishes no new architecture advantage.

The [protocol](dialogue-finetune-qualification-protocol.md) and source were
published before preparation, and the [complete prepared plan](../output/dialogue-finetune-qualification-v1/preparation-01/plan.json)
was published before execution. Workloads were selected from 2,017 TRAIN
dialogues by fixed ranks of padded attention work, without inspecting quality.
The same pretrained MiniLM encoder and fresh scalar-memory initialization were
used in every cell. All targets were synthetic; all resulting weights were
discarded. [Preparation and execution history](dialogue-finetune-qualification-status.md).

## Measured work and cost

Each optimizer update accumulates four fresh full-dialogue passes of the same
representative. Each cell has one warm update and two measured updates.

| Workload rank | Arm | Warm update, ms | Measured update 1, ms | Measured update 2, ms | Median measured, ms | Evaluation, ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10th | Frozen encoder | 42.82 | 31.60 | 31.80 | 31.70 | 6.88 |
| 10th | Trainable encoder | 221.02 | 127.64 | 127.89 | 127.76 | 7.24 |
| 50th | Frozen encoder | 68.97 | 64.71 | 64.39 | 64.55 | 13.89 |
| 50th | Trainable encoder | 320.36 | 194.92 | 194.24 | 194.58 | 13.87 |
| 90th | Frozen encoder | 91.40 | 91.26 | 96.17 | 93.71 | 22.11 |
| 90th | Trainable encoder | 354.17 | 255.00 | 252.96 | 253.98 | 20.05 |

These timings include fresh text/schema encoding, transfer, sequential memory,
synthetic loss, backward, gradient checks, clipping, optimizer work and device
synchronization. They are only two measured updates per cell, not a latency
distribution or a full-training forecast. Frozen always precedes trainable.
Re-encoding the frozen model is a matched-path control; cached frozen embeddings
would be a cheaper practical baseline. Loading, authentication, parity checks,
garbage collection and output work are included in the complete phase time,
outside the individual update times.

All **18 optimizer updates and 72 dialogue visits** completed. These repeat
three dialogues, not 72 independent examples. Six parity passes and six
evaluation passes bring the total to **84 encoder passes / 168 encoder batch
calls**, processing **4,144 text sequences and 105,952 content tokens**. No
selected text exceeded the 254-content-token chunk size; long-text chunking was
covered by synthetic tests, not by this real-workload pilot.

Peak process RSS was **828,424,192 bytes**. Maximum sampled MPS driver allocation
was **1,224,654,848 bytes**, and sampled live tensor allocation was **371,119,616
bytes**. These are distinct memory measures, not additive; sampling does not
establish the accelerator's transient peak. All recorded limits passed.

## Numerical evidence

Before any update, all context, query and candidate vectors were compared with
the historical frozen features. The largest absolute difference was
**1.416e-7**, below the fixed **2e-5** tolerance. Full encoder and memory tensor
digests matched across all fresh initializations.

Every trainable update had finite nonzero gradients at both the word embeddings
and the first attention-query weight, and every scalar update had finite
nonzero gradients. Updated encoder digests changed only in the trainable arm.
The memory processed every public USER exchange, initialized once per dialogue
at NOT_MENTIONED, with no gold-state reset or within-dialogue detach.

Across 78 sequential state forwards, **4,966 real question updates each passed
incoming-belief, feature-belief, outgoing-belief and released-mass checks**.
This checks the actual recurrent state, not just final softmax outputs.

The completion receipt is
[`a0f91d8f…`](../output/dialogue-finetune-qualification-v1/pilot-01/completed.json).
The full [summary](../output/dialogue-finetune-qualification-v1/pilot-01/summary.json)
and [per-update events](../output/dialogue-finetune-qualification-v1/pilot-01/events.jsonl)
retain every cell and all paid encoder work. The independent
[saved-artifact audit](../output/dialogue-finetune-qualification-v1/pilot-audit-01.json)
agrees on all six cells, 18 joined events, work totals, recorded gradient and
parity witnesses, state-check counts and resource/termination records. It
independently recomputed the measured-update medians. It did not repeat
numerical execution; the gradient and state witnesses remain observations
from the original monitored run.

## What this permits next

Freeze a fresh learning comparison with the memory mechanism held fixed:
frozen versus trainable text encoders, crossed with original versus explicit
number-normalized lexical observations. Match training examples, loss,
initialization seeds and full public histories. The lexical control tests the
specific number-word gap found in the history audit. Keep official TEST sealed
during this development comparison.

This pilot does not automatically admit that training campaign. Its complete
allocation, evaluation measures and continuation rule still need to be fixed.
A better observation model would strengthen the baseline a later recurrent or
connectome mechanism must beat. It would not itself establish novelty for ICLR.
