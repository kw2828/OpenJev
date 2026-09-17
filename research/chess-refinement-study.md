# Chess refinement experiments

Two bounded follow-ups address failures in the [spatial study](chess-spatial-study.md). Neither is an architectural novelty claim.

## Can inference repair the current models?

The first diagnostic uses the twelve frozen spatial checkpoints and the same 256 ordinary and 256 shifted development positions. These panels have been evaluated before; they are not fresh confirmation.

- Recurrent policies use 2, 4, 8 or 16 internal updates. The untied CNN uses 2 or 4 blocks.
- A native rules control takes an immediate mate when one exists, otherwise the original depth-4 policy.
- Exact-successor controls rank either the top 2, top 4 or all legal moves by the negative value of the resulting opponent-to-move board. These controls use explicit chess transitions and native terminal outcomes.
- Every method reports end-to-end decision time, symbolic transitions and neural evaluations. Three starting-board warmups per configuration are counted separately.
- A fixed 64-position subset per split receives 20,000-node Stockfish assessment for every distinct chosen move and an unrestricted reference. Finite-search score differences can be negative.

The predetermined development gate requires the prediction model's depth-8 policy to improve mean agreement by two percentage points in both panels, with no paired-seed deficit exceeding one point. A failure rejects this particular extra-depth configuration; it does not prove that a future trained gate cannot help. Corrected and broken decisions are both reported.

## Can targeted training repair missed mates?

The preceding 16-game arena missed 15 of 19 available mate-in-one decisions. The targeted training pilot starts from all three published prediction-model checkpoints. It keeps the 43,726-parameter architecture and four inference updates unchanged.

We select 1,024 training, 256 development and 512 confirmation mate-in-one positions from the cached official Lichess prefix. Positions are derived after the opponent's setup move. All legal mating moves are enumerated for labels only. Old puzzle source games and all prior training/evaluation positions and color mirrors are excluded; new splits have disjoint source games and canonical board states. The public prefix is a convenience sample, not a random sample of all chess positions.

Two objectives use identical initial weights and batches:

| Objective | Mate-position loss |
|---|---|
| Single | Cross entropy on the recorded mating move |
| Set | Negative log total probability of every immediate mating move |

Both retain ordinary single-target replay and value regression. Each fit uses twelve epochs, 192 updates and batches of 64 mating plus 64 ordinary positions. The replay pool contains 12,288 distinct original training positions per seed, with no new engine labels. Adam uses a learning rate of 0.0001. Six final fits are evaluated alongside all three unchanged references. There is no early stopping or checkpoint selection.

The future-board auxiliary head is unused during fine-tuning in both arms. At deployment, the network receives only board features and legal UCI moves. It does not enumerate mating outcomes or search successor boards.

The narrow skill gate requires a 20-point mean mate-accuracy gain over the frozen model on both puzzle panels. Retention requires at most a one-point mean agreement drop on each full, reused 4,096-position ordinary panel. A separate set-objective gate requires a five-point advantage over Single on both puzzle panels. These checks do not establish Elo or game-level superiority; stronger-engine and full-game follow-ups are separate.

## Why these comparisons?

[Lichess explicitly accepts every immediate mating move](https://database.lichess.org/#puzzles), so a single recorded answer can penalize another equally correct move. The set loss addresses that label mismatch; it is standard multi-answer supervision, not a claimed new algorithm.

[Prior recurrent-network work](https://arxiv.org/abs/2202.05826) studies extra computation and overthinking. [Large-scale chess distillation](https://arxiv.org/abs/2402.04494) already studies policy, action-value and state-value models. These pilots should identify a concrete failure that a later architectural experiment can address, rather than claim novelty from combining known parts.

Status: implementation and pre-run validation. Results will be appended after the frozen runs finish.
