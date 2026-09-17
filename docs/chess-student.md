# OpenJev chess students

OpenJev includes nine small, locally trained chess models: a GRU, a synthetic signed recurrent circuit, and a rewired circuit control, each trained with seeds 17, 29 and 43. They score all legal moves and return an uncalibrated probability for each UCI move ID. The original checkpoints are released under the [MIT license](../models/chess-student-v1/LICENSE).

![All nine chess student fits, two simple baselines, and the failed circuit continuation comparison](assets/chess-student-results.png)

The circuit did not establish an advantage. Its mean agreement with Stockfish was **14.36%**, versus **14.19%** for rewiring and **14.52%** for the GRU. The continuation gate required a 3 percentage point gain over rewiring; the observed gain was **0.16 points**, so the gate **failed**. These are development-set teacher-agreement results, not game win rates or Elo estimates.

| Model or baseline | Mean teacher agreement | Parameters |
| --- | ---: | ---: |
| Synthetic circuit | 14.36% | 182,289 |
| Rewired circuit | 14.19% | 182,289 |
| GRU | 14.52% | 206,833 |
| Uniform legal move, exact expectation | 4.50% | None |
| Most frequent training-label move among legal moves | 7.13% | None |

## Load a trained model

From the repository root, using Python 3.11-3.13:

```bash
uv sync --frozen --extra train
uv pip install -r research/requirements-chess.txt
```

```python
import hashlib
from pathlib import Path

import chess
import torch
from openjev.research.chess_student import ChessStudent

torch.set_num_threads(2)
plan_hash = hashlib.sha256(
    Path("evidence/chess-student-v1/plan.json").read_bytes()
).hexdigest()
model = ChessStudent.load(
    "models/chess-student-v1/circuit-17/weights.pt",
    expected_plan_sha256=plan_hash,
)
decision = model.choose(chess.Board())
print(decision["choice"], decision["probabilities"])
```

Replace `circuit-17` with any `<mode>-<seed>` from `circuit`, `rewired`, `gru` and `17`, `29`, `43`. Seed 17 above is the first declared fit, not a selected winner. [Checkpoint checksums](../models/chess-student-v1/checksums.json) identify all nine unchanged training outputs. No Stockfish process is needed for inference.

## Play and render a local game

Continue from the loaded model above:

```python
import json
from openjev.research.chess_arena import greedy_material_policy, play_game

game = play_game(
    model.choose, greedy_material_policy(),
    white_name="OpenJev circuit, seed 17",
    black_name="One-ply material heuristic",
    clock_seconds=300, max_plies=120,
)
Path("runs").mkdir(exist_ok=True)
Path("runs/my-chess-game.json").write_text(json.dumps(game.to_dict(), indent=2))
print(game.result, game.termination)
```

```bash
.venv/bin/python scripts/render_chess_replay.py \
  --game runs/my-chess-game.json --out runs/my-chess-replay \
  --caption "OpenJev circuit seed 17 versus a one-ply material heuristic; recorded local game"
```

The renderer produces an offline HTML replay, GIF, final-position PNG and hash receipt. A game that reaches the 120-ply limit remains unfinished (`*`); the limit is not scored as a draw.

## Training and evidence

All students started from random initialization and received the same 4,096 training positions, three epochs and 192 Adam updates. Stockfish 19 supplied move and bounded-value targets at 2,000 requested search nodes per analysis. Positions came from generated games mixing random legal moves and teacher moves. The 1,024 development positions use different game IDs, with exact board states deduplicated across both splits. All final fits were evaluated; no checkpoint selection or early stopping was used.

Each model performs four recurrent steps within one board evaluation and then resets its hidden state. The circuit has 64 units, 352 synthetic signed edges, sensory-only input injection, and learned edge magnitudes. Rewiring preserves each node's signed in/out degrees. These are supervised models without persistent game memory, transition prediction or reinforcement learning. The circuit is not a downloaded biological connectome. Shared training budgets do not make the GRU and circuit parameter counts equal.

[Frozen plan](../evidence/chess-student-v1/plan.json), [completed report](../evidence/chess-student-v1/results/report/summary.json), [data and learning records](../evidence/chess-student-v1/results/execution/), and [publication manifest](../evidence/chess-student-v1/results/manifest.json) preserve the audit trail. The release contains original OpenJev weights and generated supervision records; external ChessFly, FlyWire, LFM, Qwen and Stockfish assets are excluded. Value-head outputs and legal-move softmax scores have not been calibrated. Correlated positions within games and this single generated distribution limit generalization claims.

Audit the published data, logs, checkpoint identities and recomputed metrics without running models or Stockfish:

```bash
uv pip install matplotlib
.venv/bin/python scripts/publish_chess_student.py audit \
  --plan evidence/chess-student-v1/plan.json \
  --execution evidence/chess-student-v1/results/execution \
  --report evidence/chess-student-v1/results/report \
  --models models/chess-student-v1
```

To redraw the chart without training or engine calls, use a fresh output prefix:

```bash
.venv/bin/python scripts/publish_chess_student.py plot \
  --report evidence/chess-student-v1/results/report \
  --out runs/chess-student-chart
```

For a new training replication, obtain Stockfish 19 separately and prepare a new plan that records its local binary hash:

```bash
.venv/bin/python scripts/train_chess_student.py prepare \
  --engine /path/to/stockfish --out runs/chess-student-replication/plan
.venv/bin/python scripts/train_chess_student.py run \
  --plan runs/chess-student-replication/plan/plan.json \
  --out runs/chess-student-replication/execution
.venv/bin/python scripts/train_chess_student.py report \
  --plan runs/chess-student-replication/plan/plan.json \
  --execution runs/chess-student-replication/execution \
  --out runs/chess-student-replication/report
```

The original frozen plan binds its source files, dependencies and engine binary. A new replication has its own provenance and does not replace these recorded results.
