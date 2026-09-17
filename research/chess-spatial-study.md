# Board-aware recurrent chess study

**Completed:** twelve final fits and both development panels. The prediction criterion failed. [Results, models and replay](../docs/chess-spatial.md).

The first OpenJev chess pilot produced weak policies. Its synthetic circuit matched Stockfish on 14.36% of development positions, and its topology criterion failed. A [post-hoc diagnostic](../evidence/chess-student-v1/posthoc-diagnostic/report.md) found that most choices barely changed when board inputs were shuffled while legal menus were preserved. This motivates improving the representation before making further topology claims.

## Question

Does action-conditioned prediction of the next board improve chess decisions beyond an equally expensive current-board reconstruction objective?

The experiment compares four models:

| Model | Spatial computation | Auxiliary target | Parameters |
|---|---|---|---:|
| CNN | Four independent residual blocks | Zero-weight branch | 99,214 |
| Recurrent | One residual block repeated four times | Zero-weight branch | 43,726 |
| Reconstruct | Same recurrent model | Current pieces | 43,726 |
| Predict | Same recurrent model | Pieces after a legal action | 43,726 |

Each policy encodes the 8 by 8 board from the mover's perspective. One shared head scores each legal candidate from source-square, destination-square, pooled-board, displacement and promotion features. It receives neither engine evaluations nor SAN check/mate hints. The three recurrent arms have identical parameter counts, initial weights, batches and computation. The CNN has the same number of executed residual blocks but more independent parameters.

All four arms execute the auxiliary branch during training. Only the two supervised auxiliary arms give its loss nonzero weight. Four fixed legal actions per position supply successor boards through python-chess. Successor targets retain the original mover's perspective. Changed and unchanged squares receive equal aggregate loss weight. We report changed-square accuracy and exact-board accuracy because copying unchanged squares can otherwise look successful.

## Frozen development experiment

The executable contract is `evidence/chess-spatial-v1/plan.json`, generated before the first new teacher call. It binds sources, environment, exclusions, node budgets, fit seeds and schedule.

- 32,768 fresh training positions; 4,096 development positions; 4,096 shifted positions.
- Separate generated games and globally unique positions, including color-mirrored equivalents. All 5,120 positions from the first student study are excluded.
- Training and ordinary development rollouts use 50% uniformly random moves and 50% teacher moves, up to 64 plies. Shifted games use 10% random moves and up to 96 plies. This is a specified game-distribution shift, not evidence of broad transfer.
- Stockfish 19 labels every visited nonterminal position with 2,000 requested nodes, one thread, 16 MB hash and a cleared table. All rollout-only calls count toward cost.
- Three seeds, four models, six epochs and 128 examples per batch. No early stopping, checkpoint selection, replacement fits or post-result budget extension.
- MPS training uses fixed seeds and batch orders. Its indexing gradients are not bitwise deterministic; exact repeatability is not claimed. All fits use the same device.
- Uniform expectation, training move frequency and the existing one-ply material heuristic provide untrained baselines.
- A fixed random subset of 128 positions per evaluation split receives an additional 20,000-node engine assessment. Each distinct chosen move is assessed with a constrained root search. The unrestricted-minus-constrained bounded score can be negative because finite searches differ. It is a secondary engine assessment, not true game-theoretic regret.

The continuation criterion requires prediction to improve average teacher agreement by at least two percentage points over every control in both splits, lose no more than one percentage point in any paired seed, and exceed 70% changed-square prediction accuracy. This is a descriptive development rule. Both evaluation splits become development evidence once scored. A subsequent confirmation study would need fresh positions and a frozen continuation.

## Interpretation and prior art

A better board-aware policy would be useful, but improvements over the old student also change the data budget and representation. They do not isolate recurrence or prediction. Only the matched auxiliary comparisons test the proposed mechanism.

Recurrence on chess and additional test-time iterations already appear in [Schwarzschild et al.](https://arxiv.org/abs/2106.04537). Stockfish policy/value distillation is established in [ChessBench](https://arxiv.org/abs/2402.04494). Action-conditioned models used for planning are established in [MuZero](https://arxiv.org/abs/1911.08265).

Here the auxiliary decoder predicts board pieces during training; it does not plan, predict rewards, maintain cross-move memory or drive imagined rollouts. A gain would support dynamics-supervised representation learning. The next planning comparison should include exact python-chess successors, since chess already provides a perfect transition simulator.
