# Board-aware OpenJev chess models

We trained twelve original models from scratch: a spatial CNN, a recurrent policy, and recurrent policies with current-board reconstruction or future-board prediction. The recurrent variants have **43,726 parameters** each. All weights are available under the [MIT license](../models/chess-spatial-v1/LICENSE).

The new policies outperform the fixed material heuristic on both development panels. **Future prediction did not establish an advantage over the matched spatial controls.**

![Every fit in the spatial chess comparison](assets/chess-spatial-results.png)

| Model | Parameters | Ordinary development | Shifted development |
|---|---:|---:|---:|
| Spatial CNN | 99,214 | 33.02% | 25.33% |
| Recurrent | 43,726 | 32.90% | 25.58% |
| Recurrent + reconstruction | 43,726 | 32.69% | 25.22% |
| Recurrent + future prediction | 43,726 | 32.97% | 25.43% |
| One-ply material heuristic | 0 | 25.61% | 19.51% |

These are mean agreements with a 2,000-node Stockfish teacher across three fits, with 4,096 positions in each panel. They are not win rates or Elo. The shift uses fewer random moves and longer generated games. Comparing these percentages directly with the [earlier flat student](chess-student.md) would also change the training data budget and evaluation positions.

## What future prediction learned

The prediction decoder reached **79.20% changed-square accuracy** on ordinary development and **76.14%** under the shift. But complete successor-board accuracy was only **1.04% and 0.38%**. Overall square accuracy alone would hide those errors.

Prediction improved move agreement over reconstruction by only **0.28 and 0.21 percentage points**, below the frozen two-point requirement. It also did not beat every spatial control. The continuation criterion failed. Predicting pieces during training is therefore not evidence of useful planning in this experiment.

A stronger, 20,000-node engine assessed 128 fixed positions per panel. The prediction model's mean bounded engine-score difference was **0.102 versus 0.226 for the material heuristic** on ordinary development, and **0.132 versus 0.269** under the shift; lower is better. The no-auxiliary recurrent model scored **0.103 and 0.120**, respectively. These are finite-search estimates, can be negative on individual positions, and are not game-theoretic regret or loss probabilities.

## Watch it play

[![First scheduled game, prediction model versus material heuristic](assets/chess-spatial-game-01.gif)](chess-spatial-replay.html)

The first scheduled game was a draw by fivefold repetition. The replay includes all 114 accepted moves; playback speed is illustrative. [Interactive replay](chess-spatial-replay.html) · [GIF receipt](assets/chess-spatial-game-01.json) · [Complete game records](../evidence/chess-spatial-arena-v1/results/).

Each seed-17 variant played the material heuristic and the original GRU once with each color. All four variants finished **one win, three draws, zero losses**. All 16 games completed, with no unfinished or failed games. This small deterministic exhibition does not distinguish their strength or establish Elo. No new Astra, ChessFly or ChessLFM comparison is claimed here; [their earlier comparison](chess.md) remains separate.

## Load a model

From the repository root:

```sh
uv sync --frozen --extra train
uv pip install -r research/requirements-chess.txt
```

```python
import hashlib
from pathlib import Path

import chess
import torch
from openjev.research.chess_spatial import SpatialChess

torch.set_num_threads(2)
plan_hash = hashlib.sha256(
    Path("evidence/chess-spatial-v1/plan.json").read_bytes()
).hexdigest()
model = SpatialChess.load(
    "models/chess-spatial-v1/predict-17/weights.pt",
    expected_plan_sha256=plan_hash,
)
decision = model.choose(chess.Board())
print(decision["choice"], decision["probabilities"])
```

Seed 17 is the first declared seed, not a selected winner. Replace `predict-17` with any mode (`cnn`, `recurrent`, `reconstruct`, `predict`) and seed (`17`, `29`, `43`). CPU inference needs no Stockfish process. Probabilities are an uncalibrated softmax over all legal UCI move IDs. The value output estimates a transformed engine score, not a probability of winning.

## Architecture and evidence

The policy preserves the 8 by 8 board and uses one shared head to score source/destination features, board context, displacement and promotion. The recurrent block refines the current position four times and then resets. There is no persistent game memory, inference search or biological connectome. The auxiliary decoder predicts pieces during training and is absent from move selection.

All fits received the same 32,768 training positions and 1,536 updates. Training and development use separate games, with all exact and color-mirrored duplicates removed, including the earlier study's 5,120 positions. Initialization and batch order match within seed. All arms execute the auxiliary branch; only reconstruction and prediction give its loss nonzero weight. MPS indexing gradients are not bitwise deterministic.

The twelve fits used **182.56 seconds of recorded training time**. Data generation made **44,305 teacher calls**, including 3,345 rollout-only calls, for 88,539,203 reported nodes. The secondary assessment made 1,152 calls and used 22,857,247 reported nodes. Training time excludes data generation, validation and engine assessment.

[Frozen protocol](../research/chess-spatial-study.md) · [Plan](../evidence/chess-spatial-v1/plan.json) · [Results](../evidence/chess-spatial-v1/results/summary.json) · [Lossless evidence archive](../evidence/chess-spatial-v1/results/execution-and-report.tar.gz) · [Manifest and weight checksums](../evidence/chess-spatial-v1/results/manifest.json) · [Paper](../output/pdf/openjev-chess-spatial-study.pdf) · [LaTeX source](../paper/chess-spatial-study.tex).

Audit the archive and every copied checkpoint without running models or Stockfish:

```sh
.venv/bin/python scripts/publish_chess_spatial.py audit
```

To replicate training, obtain Stockfish 19 separately and prepare a fresh plan on an MPS-capable machine. The published plan identifies the original host and binary, so use a new output directory for a replication:

```sh
.venv/bin/python scripts/train_chess_spatial.py prepare \
  --engine /path/to/stockfish --out runs/my-spatial-plan
.venv/bin/python scripts/train_chess_spatial.py run \
  --plan runs/my-spatial-plan/plan.json --out runs/my-spatial-run
.venv/bin/python scripts/train_chess_spatial.py report \
  --plan runs/my-spatial-plan/plan.json --execution runs/my-spatial-run \
  --out runs/my-spatial-report
```

## Next experiment

First test whether additional recurrent steps improve decisions. Only then test a computation gate. Any learned lookahead must compete with scoring exact successor boards at the same cost, since chess already provides a perfect transition simulator. Recurrence, future prediction and learned halting have established prior art; these results do not yet support an ICLR novelty claim.
