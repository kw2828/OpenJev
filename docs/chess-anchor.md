# Recurrent chess: more robust at depth, not stronger yet

We trained twelve new models to separate two changes: keeping the original board features in each recurrent update, and training at varied depths. The combination preserves move agreement at sixteen steps, but neither the primary quality gate nor the extra-computation gate passes.

![All twelve fits, move agreement, confidence and measured CPU cost](assets/chess-anchor-results.png)

## What changed

All models encode the visible board, score its legal moves and predict a bounded engine value. They use the same 33,185 policy/value parameters, identical initial weights within each seed, and identical minibatches. Checkpoints retain an unused auxiliary decoder, for 43,726 stored parameters. There is no search or memory carried between moves.

The ordinary residual update adds the previous hidden state. The input-anchored update instead adds the original encoded board. Fixed-depth training always uses four steps; mixed-depth training uses exactly 512 updates each at two, four and six steps. Both regimes therefore use the same 1,536 optimizer updates and 6,144 recurrent iterations per fit.

Each fit uses the original 32,768 training positions for six epochs. We evaluate all final checkpoints, with three seeds per arm, on 2,048 fresh ordinary and 2,048 fresh shifted positions. These exclude 204,356 previously seen state keys, including stored successor targets, and their mirrored equivalents. The generators and shift have been studied before; this is fresh development evidence, not a new environment.

## Move quality

Entries are mean agreement with the 2,000-node Stockfish teacher across three seeds. Agreement is not winning percentage.

| Model | Ordinary, 4 steps | Shifted, 4 steps | Ordinary, 16 steps | Shifted, 16 steps |
|---|---:|---:|---:|---:|
| Residual, fixed training | 33.38% | 26.53% | 19.89% | 16.00% |
| Residual, mixed training | 34.26% | 26.87% | 32.16% | 25.11% |
| Input anchor, fixed training | 33.59% | 26.58% | 25.70% | 19.69% |
| Input anchor, mixed training | 34.54% | 26.25% | 34.47% | 26.16% |

The primary comparison, input anchor versus residual under mixed training at four steps, gains only **0.28 percentage points** on ordinary positions and loses **0.62 points** on shifted positions. The frozen rule required at least two points on both, plus better stronger-engine assessment. It fails.

The conditional 95% game-cluster intervals for those differences are -0.47 to +0.96 points across 37 ordinary games and -1.39 to +0.13 across 27 shifted games. Both include zero.

Going from four to eight steps in the anchored mixed model reduces agreement by **0.13 points on each panel**. That also fails the separate extra-computation rule. Sixteen steps were a declared stress test, not an alternative winner selected afterward.

## Stronger-engine assessment

A fixed 128-position subset per panel receives 20,000-node Stockfish assessment. Lower bounded score loss is better. These finite-search differences can be negative and are not exact game-theoretic regret or winning probabilities.

| Mixed-training model | Ordinary, 4 steps | Shifted, 4 steps | Ordinary, 8 steps | Shifted, 8 steps |
|---|---:|---:|---:|---:|
| Residual | 0.07503 | 0.09708 | 0.07975 | 0.10224 |
| Input anchor | 0.07300 | 0.10272 | 0.07311 | 0.10105 |

The anchor improves ordinary score loss at four steps but worsens shifted score loss. The chart's depth robustness is therefore not evidence of stronger general chess play.

## Cost and scope

The twelve fits used 18,432 optimizer updates and 118.63 seconds of recorded MPS training time, excluding preparation and evaluation. Fresh data required 4,542 teacher calls and 9.084 million requested nodes. Stronger grading required 759 calls and 15.18 million requested nodes. All calls, checkpoints and predictions are retained.

The chart measures full single-position CPU decisions, including board construction, encoding and response validation. Each configuration uses the same 64 selected positions and two PyTorch threads. Three warmups are recorded separately. Equal training iteration budgets do not imply equal wall time.

The report includes pointwise game-cluster bootstrap intervals, conditional on these three fitted seeds. It does not establish uncertainty over all possible training runs. No new game tournament, Elo estimate, calibrated confidence or novel architecture claim follows from this experiment.

Input recall and recurrent chess computation already appear in [End-to-end Algorithm Synthesis with Recurrent Networks](https://arxiv.org/html/2202.05826v3). This is a smaller, different skip-input ablation. The useful finding is that depth robustness and useful extra computation must be tested separately.

## Load and reproduce

- [All twelve original checkpoints](../models/chess-anchor-v1/)
- [Complete evidence and portable integrity audit](../evidence/chess-anchor-v1/results/)
- [Frozen protocol and continuation rules](../research/chess-anchor-study.md)
- [Earlier compute and mate-training results](chess-refinement.md)

```python
from pathlib import Path
import hashlib
import chess
from openjev.research.chess_anchor import AnchorChess

plan = Path("evidence/chess-anchor-v1/plan.json")
model = AnchorChess.load(
    "models/chess-anchor-v1/anchor_mixed-17/weights.pt",
    expected_plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),
    expected_recurrence="anchor",
    expected_seed=17,
)
result = model.choose(chess.Board(), depth=4)
print(result["choice"], result["probabilities"])
```

Seed 17 is the first declared seed, not a selected winner. Scores are an uncalibrated softmax over legal candidates. Use the model's own `forward` or `choose` methods: the older exact-successor diagnostic manually applies a different recurrence and is not compatible with this model.
