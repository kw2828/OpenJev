# Self-referential policy optimization: Doom adaptation

This study adapts **Self-Referential Policy Optimization for Vision-Language-Action Models** ([paper](https://arxiv.org/abs/2511.15605v2), [official reward implementation](https://github.com/sii-research/siiRL/blob/89d8764b6133ea78557a22eef32c68ad92946d45/siirl/utils/reward_score/embodied.py)). The paper trains vision-language-action policies using successful trajectories as references for failed attempts. OpenJev tests the mechanism in its small structured-observation Doom controller. It does not reproduce OpenVLA, V-JEPA, LIBERO results, or a new algorithm.

## What is tested

Each of the three preserved PPO-history checkpoints initializes three matched variants:

| Variant | Training reward | Trajectory representation |
|---|---|---|
| Group binary | 1 for at least 10 kills, otherwise 0 | None |
| SRPO raw adaptation | Success reward plus similarity credit for failures | Mean and last current observation |
| SRPO latent adaptation | Same success and failure-credit rule | Mean and last learned dynamics encoding |

Each fit receives 16 groups of eight stochastic full episodes. Members of a group share one initial game seed. All variants use the same seed schedule, actor architecture, optimizer, clipping, KL penalty, entropy bonus, and fixed steering. Only rewards differ between binary and raw; latent additionally learns a representation. All three initial fits are retained. The unchanged PPO models and always-fire are evaluation references.

This matches episode and update-opportunity budgets, **not actual interaction counts or compute**. Episode lengths differ; homogeneous groups do not update. The analysis charges encoder fitting to the latent variant and records actual environment interactions, actor training time, and updated groups. The unchanged PPO reference has no additional training, so comparing against it does not isolate the effect of additional compute.

The encoder is a small, one-step next-observation predictor trained only on each corresponding PPO fit's historical training trace. Six current observation features enter an eight-dimensional bottleneck; action enters the decoder. The encoder never receives kill labels, terminal flags, horizon, or evaluation data. Its mean and last pre-action embeddings summarize each trajectory. This is a locally learned dynamics representation, not a pretrained video world model, a recurrent architecture, or evidence that distance measures progress accurately.

## Reward and optimization details

Success means at least 10 kills over the complete Center episode. This is a new sparse goal, distinct from the previous PPO's dense kill/death/fire-cost objective. References come only from successful members of the current eight-trajectory group. They are clustered in standardized embedding space with DBSCAN, radius 0.5 and minimum two samples. If all successes are noise, their mean is the fallback center.

Failure rewards decrease with distance to the nearest successful center, using the pinned official implementation's min-max normalization and `0.6 * sigmoid(10 * (0.5 - normalized_distance))`. Equal distances receive 0.3. No-success groups receive all-zero rewards; all-success groups receive all ones. Both skip optimization without extra replacement rollouts.

There is a documented source discrepancy: the paper describes standardized squared distances and a 0.8 failure-reward scale, while official code at commit `89d8764b6133ea78557a22eef32c68ad92946d45` uses Euclidean min-max distances and 0.6. This study follows that code's reward convention. The implementation is independent; the small DBSCAN special case and its noise fallback are tested.

Group-normalized trajectory advantages feed a clipped actor loss. The implementation uses four full-group updates, equal trajectory weighting, exact categorical KL to the initial policy as a penalty, and an entropy bonus. It never trains the inherited critic or includes padded terminal steps. Those details and the structured encoder are departures from a full reproduction.

## Protocol and stopping rule

[Prospective protocol](../research/protocols/srpo-doom-v1.json). Nine fits have at most 207,360 new training transitions, followed by 624 deterministic evaluation episodes on 24 new paired seeds in each scenario. Maximum wall time is 30 minutes. The smoke run uses separate seeds and reduced settings; it is an engineering check only.

To justify a future fresh confirmation, latent SRPO must gain at least 0.5 mean Center kills against **each** of binary rewards, raw similarity, unchanged PPO, and always-fire. Line kills and both durations must not decline by more than 0.5. Every latent-model episode must also pass the 1 ms p95 decision-time screen. These are descriptive development thresholds, not a statistical efficacy test. The paired crossed-bootstrap 95% intervals are exploratory, unadjusted, and limited by three training fits. Both scenarios were used in earlier research.

No threshold changes, fallback selection, or extra tuning follow a failed gate. A passing pilot would motivate a separately frozen confirmation, not establish efficacy or ICLR novelty.

```sh
uv sync --frozen --extra rl --extra dev
uv run --extra rl python research/srpo_doom.py --smoke --output runs/srpo-smoke-new
uv run --extra rl python research/srpo_doom.py --output evidence/srpo-doom-v1/pilot-new
uv run --extra rl python research/analyze_srpo_doom.py evidence/srpo-doom-v1/pilot-new --output evidence/srpo-doom-v1/analysis-new.json
```

Use new output paths. No paid API calls. Historical evidence and checkpoints remain unchanged.
