# OpenJev chess research plan

**Status: the supervised pilot and puzzle panel completed; the topology continuation gate failed. All 18 games completed.** This document does not establish architectural superiority or an ICLR contribution. The standalone paper draft is [chess-study.tex](../paper/chess-study.tex).

The immediate question is narrow: **with the same teacher data, optimizer, signed degree sequence, and four internal updates, does a structured synthetic circuit learn better chess decisions than its rewired control?** GRU and published players provide context. They do not isolate topology.

## What the current pilot tests

Our [students](../src/openjev/research/chess_student.py) predict legal moves and value from numeric FEN features. Their 64-dimensional hidden state resets for each board and runs four internal updates. This is computation within one position, not memory across moves.

| Arm | Internal computation | What the comparison establishes |
|---|---|---|
| Synthetic circuit | 352 signed edges across 16 sensory, 24 inter, 16 command and 8 motor units | One designed sparse topology |
| Rewired circuit | Same number of edges and each node's signed incoming/outgoing degrees; fixed double-edge swaps | A matched topology control, currently only one graph pair |
| GRU | Dense gated 64-dimensional recurrence | A conventional learned baseline with different capacity and input access |

Both circuits have 182,289 declared parameters; GRU has 206,833. Shared initialization matches, but circuits receive only 16 encoder outputs versus GRU's 64, leaving 48 unused encoder rows. Both heads read all hidden units. The circuit executes a dense matrix, so sparse parameters do not establish a speedup.

[The frozen protocol](../scripts/train_chess_student.py) uses 4,096 training and 1,024 development positions from separate generated games, globally deduplicated by state. Stockfish 19 supplies 2,000-node labels. All three modes use seeds 17/29/43, three epochs, 192 updates, identical per-seed batch ordering, legal-masked cross-entropy and value MSE. Only final checkpoints are evaluated. This is supervised distillation, not RL; bounded centipawn targets are not win probabilities.

## Observed development results

All nine fits completed. The saved report is [the published summary](../evidence/chess-student-v1/results/report/summary.json), bound to training plan `b798fc8a421a6e72479085b990c8439b9069d915ff0cd8c888151ff440e70d45`.

| Arm | Agreement, seeds 17 / 29 / 43 | Mean agreement | Mean value MAE |
|---|---|---|---|
| Synthetic circuit | 14.1602% / 14.4531% / 14.4531% | 14.3555% | 0.578403 |
| Rewired circuit | 14.5508% / 14.0625% / 13.9648% | 14.1927% | 0.578546 |
| GRU | 14.4531% / 14.4531% / 14.6484% | 14.5182% | 0.577060 |

Uniform-legal expected agreement is 4.4997%; the training-move-frequency baseline scores 7.1289%. The circuit's mean gain over rewired is **0.16276 percentage points**, below the frozen 3-point requirement. Its paired difference changes sign across seeds. The other three gate checks pass, but the conjunction fails. This small-data, short-training baseline shows some teacher imitation, not a topology advantage. It neither confirms nor disproves the usefulness of larger circuit models. Do not scale this topology claim from these results.

## What the public-player pilot adds

The [18-game design](../evidence/chess-v1/design.json) fixes a color-paired round robin among our circuit, published ChessFly, ChessLFM and GPT-6 Astra via Codex, plus circuit pairs against OpenJev Qwen, GRU and rewired. Published models have different training data, sizes, representations and compute routes; these games demonstrate behavior rather than isolate architecture. Seed 17 and the first-game replay were selected before training results, not selected for performance. All 18 games reached scored outcomes. Astra won four games and lost two, including one clock loss to ChessFly and one checkmate loss to ChessLFM. Our circuit drew seven and lost five. [Complete games and puzzle scores](../docs/chess.md) retain every outcome.

ChessFly and ChessLFM use published weights with search omitted; Qwen directly scores candidate labels. Astra uses one persistent Codex agent across all requests, retaining prior packet context but receiving no gold labels or score files. Isolation is instruction-based, not an OS sandbox. Its timing includes dispatch, reasoning, tools and polling, unlike local inference. Load/warmup costs are separate. No Astra probabilities or hardware-matched speed claims are invented.

The completed puzzle panel is the first eight valid rows in each of three rating bands from a bounded official [Lichess CSV prefix](https://database.lichess.org/#puzzles). The first recorded opponent move is applied before asking for the first solver move. All 240 policy calls completed without errors, recorded in [the published attempt records](../evidence/chess-v1/results/puzzles.jsonl).

| Policy | Strict first-move matches / 24 |
|---|---:|
| Stockfish, 1,000 nodes | 22 |
| GPT-6 Astra via Codex | 17 |
| ChessLFM direct | 14 |
| ChessFly direct | 7 |
| Greedy material | 5 |
| OpenJev circuit, seed 17 | 2 |
| OpenJev rewired, seed 17 | 2 |
| OpenJev Qwen direct | 2 |
| OpenJev GRU, seed 17 | 1 |
| Seeded uniform random | 1 |

These are first-move matches, not full-puzzle solves. A separately recorded immediate-mate alternative is not folded into this table. The 24 public convenience positions may overlap published models' training; the table is neither a representative benchmark nor a chess rating. Capped games stay unfinished; invalid responses and transport failures stay failed. All 18 games reached scored outcomes; none failed or reached the ply cap.

The text packets for Astra and Qwen include SAN check/checkmate markers. Three of the 24 puzzle positions have an immediate mate revealed by the notation; the numerical policies receive legal masks without SAN. A future matched reasoning comparison should equalize this information.

## Prior art and the claim boundary

Sparse recurrent wiring, [policy distillation](https://arxiv.org/abs/1511.06295), and [extra recurrent computation on chess](https://arxiv.org/abs/2106.04537) are established. Our tanh cell is not a reproduction of [Neural Circuit Policies' continuous-time dynamics](https://github.com/mlech26l/ncps). Published ChessFly and ChessLFM are prior artifacts, not newly invented OpenJev models.

Our circuit is synthetic. ChessFly outcomes cannot establish that biology caused its performance without matched retraining controls. A [learned world model](https://arxiv.org/abs/1912.01603) needs action-conditioned future prediction and evaluated rollouts; the current students implement neither.

## Next: a memory and state-transition diagnostic, not yet frozen or run

Test **whether an action-conditioned predictive state preserves chess history that is absent from the current FEN**. Ordinary board reconstruction is mostly observable already. Repetition provides an exact history-dependent target without an expensive teacher. Use the current-position third-occurrence predicate from [python-chess](https://python-chess.readthedocs.io/en/latest/core.html#chess.Board.is_repetition), not the broader claimable-draw predicate and not an invented chess reward.

A verified construction starts from the standard board. `g1f3 g8f6 f3g1 f6g8` repeated twice and `g1f3 g8f6 b1c3 b8c6 c3b1 c6b8 f3g1 f6g8` reach the same complete FEN, including counters, but only the first history has a third occurrence. The last moves also match. This is an illustrative rule check, not a learned-model result. A benchmark needs many distinct prefixes and cycle families, not these two memorized strings.

1. Generate balanced pairs with identical endpoint observations and previous actions. Keep complete prefix/cycle families within one split. Freeze disjoint game-seed panels and endpoint keys across splits, retaining duplicate endpoints *within* pairs. Use a fixed generation cap and report failures.
2. Compare current-only, persistent GRU, predictive GRU, and a same-sized auxiliary reconstruction control. Match input access, data, optimization and fresh seeds; do not inherit pilot weights. Use exact python-chess replay as reference and per-step state reset as the memory intervention.
3. Predict one-, two- and four-step futures without later observations or future legal masks. Report exact-board and changed-square accuracy, state-field errors, invalid predictions, repetition accuracy, calibration and total compute. Copying unchanged squares must not dominate the metric.
4. Proposed budget: 4,096 training sequences, 512 calibration pairs, 512 test pairs per condition, three seeds and 192 updates per arm. Freeze unseen-prefix and longer-cycle/distractor tests. Validate generation and runtime before committing these counts, without scoring test models.

The diagnostic continuation rule should require at least 90% repetition accuracy on both unseen-prefix and longer-history panels in every fit, at least a 20-point drop under state reset, and demonstrable rollout improvement over the recurrent/reconstruction controls at matched compute. The balanced identical-input deterministic current-only control should stay at its 50% information ceiling; a higher score signals leakage or imbalance. Freeze a practical rollout-effect threshold and game/pair-level uncertainty analysis before execution. Passing would establish learnability and state use, not better chess play or a novel world model.

An ICLR-oriented claim would still need a specific mechanism beyond ordinary recurrence and prediction, a gain on untouched tasks at matched compute, strong memory baselines such as an explicit associative store, and an intervention connecting that mechanism to the gain. If graph topology is revisited later, use multiple structured and degree-preserving random graph seeds, common input access, and parameter/compute controls. The present topology gate remains failed. Adaptive stopping would also need [ACT](https://arxiv.org/abs/1603.08983), fixed-depth, entropy-gated and matched-budget ensemble baselines. Conformal prediction should address a demonstrated decision risk rather than substitute for these tests.
