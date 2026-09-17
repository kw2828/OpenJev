# Chess student v1: post-hoc diagnosis

The clearest next step is a board-aware, shared candidate scorer with a verified learnability check. The current results do not support spending more on the synthetic circuit topology alone.

This diagnosis uses the existing 4,096 training positions and already scored 1,024 development positions. No fitting, teacher calls, new test set, checkpoint selection, or tuning occurred. Every checkpoint and saved development prediction was verified before analysis. These are descriptive diagnostics, not new evidence of playing strength.

## What the final models learned

All numbers below are means over the three fixed seeds. Every fit completed 192 optimizer updates and 12,288 example presentations. The recorded training time across all nine fits was approximately 2.32 seconds; data generation and evaluation are separate.

| Model | Training teacher agreement | Development agreement | Policy CE, epochs 1 / 2 / 3 | Development value MAE |
|---|---:|---:|---|---:|
| GRU | 21.57% | 14.52% | 3.192 / 2.911 / 2.812 | 0.5771 |
| Circuit | 18.33% | 14.36% | 3.251 / 2.993 / 2.874 | 0.5784 |
| Rewired | 18.31% | 14.19% | 3.252 / 2.997 / 2.877 | 0.5785 |

Training CE is still falling and training agreement remains low. More optimization is plausible, but this does not establish that longer training would improve unseen positions. The training-development gaps are already 7.06 percentage points for GRU and about 4 points for both sparse models.

The value heads remain close to a constant: development predictions have standard deviation 0.0269 for GRU and about 0.013 for both sparse models, versus 0.674 for teacher targets. Predicting the training-set mean value gives development MAE 0.5774, essentially matching every learned head. This is a useful prerequisite failure to address before claiming predictive reasoning or world modeling.

## The models rely weakly on the board features

A single fixed circular permutation of the existing development board features, retaining each example's original legal mask and teacher label, leaves 89.36% of GRU choices, 96.88% of circuit choices and 96.71% of rewired choices unchanged. Teacher agreement changes from 14.52% to 14.52%, 14.36% to 14.13%, and 14.19% to 14.19%, respectively. NLL worsens by 0.063 for GRU and about 0.011 for the sparse models.

This intervention breaks the feature-mask relationship and is a post-hoc sensitivity check. Legal masks still contain substantial position information. It does not establish that the models contain no chess information or provide an unbiased alternative benchmark. It does show that most final move choices survive replacing their numeric board features with another development position's features.

## Effective circuit input capacity differs from GRU

Both sparse models mask the encoder drive to its first 16 of 64 channels. Exactly 48 encoder rows remain identical to initialization and have zero gradient in the diagnostic batch, across all six sparse fits. This disconnects 40,416 of the encoder's 53,888 parameters. All 64 GRU encoder rows changed and receive nonzero gradients.

The sparse core's 416 parameters all receive nonzero gradients. Structured input reachability after updates 1-4 is 16 / 40 / 56 / 64 nodes; rewired reachability is 16 / 63 / 64 / 64. Thus the circuit is connected within the declared four updates, although depth and path lengths differ. The readout uses all 64 states, including sensory states, so it can bypass the intended sensory-to-motor chain.

Structured versus rewired remains matched on the 16-channel bottleneck and parameter count. Sparse versus GRU is not an equal effective-input-capacity comparison. The separate 1,968-output policy head has 127,920 parameters, roughly 70% of a sparse model's registered parameters.

## Errors suggest a useful geometry and capture baseline

The dataset contains 113 training games and 29 development games. Training supplies 895 distinct target UCI moves, while 91 of 1,024 development targets never occur as training labels. Mean agreement on those 91 examples is only 1.10% for GRU/circuit and 1.47% for rewired. The current flat head has no shared move geometry or destination/piece-based scoring, although its input does contain all board information necessary to learn these features.

| Same existing development panel | All 1,024 positions | Teacher move is a capture, n=422 | Teacher move is quiet, n=602 |
|---|---:|---:|---:|
| Exact uniform legal choice expectation | 4.50% | - | - |
| Training move-frequency legal argmax | 7.13% | 1.90% | 10.80% |
| Existing one-ply greedy material, post-hoc | 23.54% | 54.50% | 1.83% |
| GRU | 14.52% | 22.43% | 8.97% |
| Circuit | 14.36% | 20.77% | 9.86% |
| Rewired | 14.19% | 20.46% | 9.80% |

Greedy material is stronger on teacher agreement overall because this panel contains many captures; it fails badly on quiet moves. Its new score is post-hoc and does not imply a win rate. It should become a prespecified baseline in the next experiment, alongside the original frequency and uniform baselines. All three students underperform the simple frequency rule on the 602 quiet targets.

## One bounded next protocol

1. Use a shared legal-candidate scorer with source/destination board embeddings, moving/captured piece, promotion and displacement features. Keep a plain shared-candidate MLP as the positive control before adding recurrence or predictive losses.
2. Match effective encoder input width across recurrence arms. Keep structured versus rewired as a secondary comparison, and explicitly state whether readout can bypass intermediate layers.
3. Freeze training-data size and an optimization schedule before running; retain full loss curves and a training-fit check. The present study leaves both data coverage and optimization unresolved, so increasing epochs alone cannot answer the architecture question.
4. Prespecify capture/quiet and rare/unseen-move slices, plus greedy material and constant-value baselines. Any recurrent/predictive extension should improve board sensitivity and value learning, then show improvement on fresh game-separated data and gameplay. Do not select a model using this repeatedly inspected development panel.

Artifacts: `scope.json`, `summary.json`, `means.json` and `diagnose.py` in this folder preserve source/checkpoint hashes, exact grouping counts, all nine fit results, gradient paths and the fixed permutation definition. Existing study files and checkpoints are unchanged.
