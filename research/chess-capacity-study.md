# Chess capacity study

The input-anchor experiment preserved accuracy at greater recurrent depth but did not pass either quality criterion. This study tests whether a larger policy can learn substantially stronger chess before further architectural changes.

**Status: completed and audited; continuation failed.** All six fits, twelve evaluations and 96 scheduled games are retained. [Results and chart](../docs/chess-capacity.md) · [Complete evidence](../evidence/chess-capacity-v1/results/) · [All checkpoints](../models/chess-capacity-v1/) · [First scheduled replay](../docs/chess-capacity-replay.html).

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

Capacity and data scaling are established directions, including [Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/html/2402.04494v2). This is a controlled capacity comparison for later recurrent/world-model work, not a novel architecture. Its failed continuation gate does not justify promoting the larger policy as a stronger reference model.

## Recorded results

Across all three seeds, ordinary teacher agreement increases from 35.13% to 35.75%; shifted agreement increases from 26.69% to 26.88%. Mean bounded score loss falls from 0.10547 to 0.08122 on ordinary positions (22.99%) and from 0.15987 to 0.14507 on shifted positions (9.26%). The shifted result misses the required 20%. Raw shifted centipawn loss instead worsens from 177.01 to 231.08, so the bounded improvement is not a general reduction in every error measure.

The larger policy records 18 wins, 64 draws, 12 losses and two unfinished games, with no failed games. Its point bound is 52.08-54.17%, below the required 60% lower bound. Both criteria must pass, so the overall continuation rule fails. The first scheduled replay is a fivefold-repetition draw, retained regardless of outcome.

Training used 36,864 optimizer updates, 147,456 recurrent iterations and 362.53 seconds of MPS fit time. Mean full CPU decision latency was 0.595 ms for width 32 and 0.911 ms for width 128. Fresh data required 80,229 teacher calls and 160.458 million requested nodes; grading required 828 calls and 16.56 million requested nodes. These costs exclude the earlier creation of the reused training labels. Full execution took 814.20 seconds.

The [posthoc finishing diagnostic](../evidence/chess-capacity-v1/finishing-diagnostic/summary.json) is exploratory and separate from these frozen criteria. It counts missed immediate mates in the existing game traces without new model or engine calls, and changes neither outcomes nor the continuation gate. No Elo, calibrated-confidence, novel-architecture or world-model benefit claim follows from this study.
