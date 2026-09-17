# Chess capacity study

The input-anchor experiment preserved accuracy at greater recurrent depth but did not pass either quality criterion. This study tests whether a larger policy can learn substantially stronger chess before further architectural changes.

## Fixed comparison

| | Small control | Larger policy |
|---|---:|---:|
| Hidden width | 32 | 128 |
| Active policy/value parameters | 33,185 | 439,073 |
| Stored parameters, including unused auxiliary decoder | 43,726 | 591,790 |
| Residual recurrent steps | 4 | 4 |
| Shared training positions | 98,304 | 98,304 |
| Epochs | 8 | 8 |
| Updates per fit | 6,144 | 6,144 |
| Seeds | 53, 67, 83 | 53, 67, 83 |

Both widths train from scratch with the same per-seed minibatch order, legal-move cross entropy plus 0.5 bounded-value MSE, and Adam at 0.001. Width changes the encoder, recurrent core and heads together. This isolates overall capacity within the comparison; parameters, FLOPs and wall time are deliberately not matched. No auxiliary transition objective, mate fine-tuning, search or immediate-mate guard is added.

Training joins the existing 32,768 positions with 65,536 newly generated positions. Each example retains its original source identity. New data excludes all previous input boards, stored behavior/auxiliary successor boards, selected mate puzzles and recorded games, including mirror-equivalent states. Fresh evaluation has 4,096 ordinary and 4,096 shifted positions, with game seeds and exact quotas frozen before generation. Training and ordinary evaluation use the same 50% random rollout generator; shifted evaluation uses 10% random moves. These generators have been studied previously.

The teacher remains Stockfish 19 at 2,000 nodes, one thread, 16 MB hash, cleared per call. All visited nonterminal positions are charged, even when discarded. Actual reported nodes and failed calls remain in the record. Native auxiliary transitions are recorded by the reused generator but do not enter this training loss.

## Evaluation and continuation

All six final fits finish before fresh neural evaluation is exposed. Each is evaluated at its trained depth of four. CPU latency includes board construction, encoding and response validation on the same 128 selected positions, with two PyTorch threads and three separately recorded warmups.

A fixed 128-position subset per panel receives 20,000-node engine assessment for every fit. There is one unrestricted call and one call per distinct selected move per position, at most 1,792 calls or 35.84 million requested nodes. Signed bounded score differences are retained; they are finite-search measurements, not exact game-theoretic regret or winning probabilities.

The larger model must reduce mean bounded score loss by at least 20% on both panels. This relative criterion requires a strictly positive small-model baseline mean; otherwise it is undefined and fails. Teacher agreement and conditional game-cluster intervals are secondary descriptions, not substitutes for the engine criterion.

The same paired seeds play 96 games: sixteen fixed, common six-ply opening prefixes, with both colors per pair. The openings are legally validated, not claimed engine-balanced. Each side has 300 seconds with no increment. A game stops after at most 240 additional plies; a nonterminal cap is unfinished, never a draw. Native automatic draws are recognized with full board history. Every scheduled game, failure and clock cost is retained, with no replacement or fallback policy.

The larger model must earn at least 60% of all available points, using zero for unresolved games in the lower point bound, with no failed games. Both the engine and arena criteria must pass to justify promotion to a stronger baseline. Opening-cluster intervals condition on these three fitted seeds and do not establish an Elo rating.

## Research scope

Capacity and data scaling are established directions, including [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/html/2402.04494v2). This study supplies a stronger controlled baseline for later recurrent/world-model work; it is not itself a novel architecture. A synthetic local preflight suggested that six fits are feasible in minutes of training, excluding data preparation, logging and evaluation. Actual costs will replace that estimate in the report.

Status: implementation and synthetic validation, before any real generation or fitting. The first scheduled larger-model game is the fixed replay selection, regardless of its outcome.
