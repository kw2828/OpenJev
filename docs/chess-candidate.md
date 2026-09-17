# Candidate-conditioned chess

**Status: complete and audited.** All four required engine-loss comparisons and both required game-score thresholds fail. The full evidence package and all twelve original checkpoints are retained.

Native one-ply consequences slightly raise move-label agreement, but the recorded outputs miss the required score-loss and game-score improvements. This experiment does not establish a stronger reference model, an Elo rating or architectural novelty.

![All twelve candidate-policy fits, move quality, game outcomes and cost](assets/chess-candidate-results.png)

[First scheduled replay](chess-candidate-replay.html) · [Full evidence](../evidence/chess-candidate-v2/results/) · [All twelve checkpoints](../models/chess-candidate-v2/) · [Original protocol](../research/chess-candidate-study.md) · [Recovery amendment](../research/chess-candidate-recovery.md)

## What was compared

Four width-32 policies share the same root encoder, recurrent core and move-scoring head. Each uses four root computation steps. Direct scores the root representation; action only adds move information and two shared-core steps. Exact delta also adds a linear projection of the encoded successor-minus-root difference. Full afterstate re-encodes each successor before its two refinement steps.

Delta and full afterstate receive exact native one-ply consequences for every legal move. Their refinement steps process that single successor; they do not roll forward additional chess plies or carry memory between decisions. These are **not learned world models**. Delta is an exact input difference, not an exact nonlinear latent update. No tactical guard or engine search runs inside these policies.

All twelve final fits use the original 32,768 training positions, six epochs, 1,536 updates and seeds 97, 109 and 127, with matched labels, minibatch orders and common initialization. Direct has 33,185 active parameters; each refinement arm has 33,313. All checkpoints store 43,854 including unused modules. Parameters, computation and wall time are unequal.

## Move quality

Entries average all three seeds. Agreement uses 2,048 ordinary and 2,048 shifted development positions. Stronger-engine assessment uses the same fixed 128 positions per panel for every fit, with 20,000-node Stockfish calls.

| Arm | Ordinary agreement | Shifted agreement | Ordinary bounded loss | Shifted bounded loss |
|---|---:|---:|---:|---:|
| Direct | 31.75% | 26.81% | 0.09878 | 0.08163 |
| Action only | 31.54% | 26.45% | 0.09115 | 0.09525 |
| Exact delta | 33.15% | 27.10% | 0.09283 | 0.08768 |
| Full afterstate | 32.19% | 27.10% | 0.10821 | 0.09537 |

Lower signed bounded loss is better. This is the difference between finite-search scores transformed by `tanh(cp / 600)`, not game-theoretic regret or winning probability. Agreement measures matching one 2,000-node reference move.

The frozen rule requires delta to reduce mean bounded loss by **at least 20% against both direct and action only, on both panels**, with positive comparator means. Against direct, delta improves ordinary loss by 6.02% and worsens shifted loss by 7.42%. Against action only, it worsens ordinary loss by 1.85% and improves shifted loss by 7.95%. None reaches the required improvement. Delta's raw shifted centipawn loss also worsens versus direct: 487.20 versus 428.61.

## Paired games

Each opponent contributes 96 games: sixteen fixed six-ply openings, three paired seeds and both colors. Every side receives five minutes without increment. A nonterminal 240-model-ply cap remains unfinished.

| Delta opponent | Wins | Draws | Losses | Unfinished | Failed | Point bounds |
|---|---:|---:|---:|---:|---:|---:|
| Direct | 5 | 86 | 4 | 1 | 0 | 50.00-51.04% |
| Action only | 7 | 83 | 5 | 1 | 0 | 50.52-51.56% |
| Full afterstate | 11 | 74 | 11 | 0 | 0 | 50.00% |

All 96 games remain in each denominator; unresolved games receive zero points in the lower bound and one in the upper bound. The rule also requires **at least 60% lower-bound points against both direct and action only, and zero failed games across all 288**. Both required point thresholds fail. The full-afterstate pairing is descriptive. The successor-permutation panel is a separate diagnostic and cannot replace these criteria.

[![First scheduled candidate game, a fivefold-repetition draw](assets/chess-candidate-game-001.gif)](chess-candidate-replay.html)

The replay is the first scheduled game: delta-97 as White against direct-97, drawn by fivefold repetition after six opening plies and 85 model plies. It was selected before outcomes.

## Cost and recovery

| Arm | Mean full CPU decision | Mean training time per fit |
|---|---:|---:|
| Direct | 0.617 ms | 20.14 s |
| Action only | 2.622 ms | 114.90 s |
| Exact delta | 3.415 ms | 132.24 s |
| Full afterstate | 7.318 ms | 162.07 s |

CPU timing covers the same 128 positions per fit, including board construction, native successor preparation where applicable, inference and response validation, with two PyTorch threads. Three warmups are separate. These are full decision timings on one machine, not cached-batch throughput or a general speed claim. Recorded MPS training totals 1,288.06 seconds; shared cache construction and the discarded attempt are outside those fit times.

The [first attempt failed](../evidence/chess-candidate-v1/failed-attempt/README.md) after 128 updates and 16,384 presentations, without a completed checkpoint or neural evaluation. Recovery starts every model from its original initialization and reuses the same 4,096 evaluation positions byte for byte. They are not a second fresh sample. Their 4,756 teacher calls and 9.512 million requested nodes are counted once. Stronger grading adds 932 calls and 18.64 million requested nodes.

The successful execution records 1,844.56 seconds; the failed attempt adds 457.02 seconds, for 2,301.58 seconds across both attempts. The twelve final fits receive 18,432 optimizer updates; adding the 128 discarded updates gives 18,560 total. The failed duration includes data generation, validation, cache construction and partial training, whose separate duration is unavailable.

These are development results on previously studied generators, conditional on three fitted seeds. Shared successors can leave dependence beyond source-game clusters. Candidate probabilities are uncalibrated legal-menu softmax scores. Every final fit is retained; no best seed is selected.

## What the diagnostics support

Permuting delta's candidate-to-successor assignments reduces agreement from 33.15% to 25.73% on ordinary positions and from 27.10% to 22.27% on shifted positions. Every seed declines on both panels. This demonstrates sensitivity to correct action-successor alignment, without establishing effective planning or learned dynamics. Eight ordinary and 28 shifted single-action positions cannot be permuted and are reported separately.

A posthoc summary finds decreasing training losses in every arm and 231 fivefold-repetition endings among 288 games. The remaining endings are 43 checkmates, ten insufficient-material draws, two stalemates and two unresolved ply caps. These observations do not isolate a representation defect, missing memory or an optimization failure. A further experiment must distinguish those explanations rather than infer a cause from draw frequency.
