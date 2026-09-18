# Biological wiring does not pass the controlled chess comparison

All eighteen adapters and three unchanged backbones completed the frozen
comparison. The original reporting program audited all saved evidence without
new model or engine calls. The biological topology fails its continuation rule:
**5 of 30 required checks pass**.

![All twenty-one models, paired seeds and full decision latency](../evidence/chess-connectome-v1/figure/connectome-chess.png)

| Arm | Ordinary bounded loss | Shifted bounded loss | Ordinary agreement | Shifted agreement | Full CPU decision, ms |
|---|---:|---:|---:|---:|---:|
| Frozen direct | 0.09739 | 0.11009 | 33.02% | 24.37% | 0.537 |
| Biological | 0.10198 | 0.10396 | 33.46% | 24.63% | 1.168 |
| Rewire 151 | 0.09539 | 0.09741 | 33.35% | 24.32% | 1.168 |
| Rewire 163 | 0.09099 | 0.10426 | 33.63% | 24.64% | 1.152 |
| Rewire 179 | 0.09664 | 0.10266 | 33.41% | 24.45% | 1.161 |
| Dense | 0.09361 | 0.10694 | 33.45% | 24.56% | 6.242 |
| Node-local | 0.09250 | 0.10490 | 33.45% | 24.35% | 0.948 |

Values average every paired seed (97, 109, 127). Agreement uses 2,048 positions
per panel; loss uses the fixed 128-position subset per panel graded at 20,000
Stockfish nodes. The bounded loss is a signed finite-search score difference,
not exact regret or a winning probability. CPU latency includes the complete
decision on 128 fixed positions with warmups recorded separately. All arms use
dense matrix operations on a shared desktop; topology sparsity is not a claim
of sparse-kernel efficiency.

Biological mean loss is higher than each rewire on ordinary positions. Under
shift it is lower than rewire 163 by 0.29%, but higher than the other two. The
required reduction was at least 10% against every rewire on both panels, with
strictly lower loss in every paired seed and no degradation against the frozen
backbone in any paired seed. The original threshold is unchanged.

The graph is the induced set of all 1,409 descending nodes and 44,090 signed
connections from the pinned packed ChessFly graph. It excludes 430,912 boundary
connections, including original sensory inputs. The shared board interface is
artificial. These findings concern this adapter and training regime; they do
not establish that biological topology cannot help other tasks or interfaces.
There is no arena result or Elo estimate for this study.

Any follow-up needs its own protocol and matched controls. It cannot change
this study's failed criterion.

## What to test next

The input/output projections already learned during this experiment. A
[saved-weight audit](../evidence/chess-connectome-parameter-audit-v1/README.md)
confirms that all eighteen adapters moved away from their zero-output
initialization. It does not establish which computations affected decisions.

A separate [spatial-interface implementation](../research/chess-connectome-interface.md)
can change which board squares feed each neuron while preserving a hard,
tied gather/scatter mapping. A [30-fit comparison](../research/chess-connectome-mapping-study.md)
gives biological, rewired and node-local models the same mapping-search budget.
Its 135 engineering checks and full original-input validation pass. Protocol
preparation is running. No new chess fit or performance result is available yet.

## Evidence

- [Frozen study description](../research/chess-connectome-study.md)
- [Frozen plan](../evidence/chess-connectome-v1/protocol/plan.json)
- [Completed report](../evidence/chess-connectome-v1/results/summary.json)
- [Original report receipt](../evidence/chess-connectome-v1/results/completed.json)
- Original raw outputs and derivative weights remain local under
  `runs/chess-connectome-v1/execution`; no portable execution archive is claimed.

The first report attempt encountered a sandbox-only change in GPU availability.
The environment comparison isolated that one difference. The original report
then ran in the original host environment and passed without modifying its
source, protocol, artifacts or checks. No training was retried.
