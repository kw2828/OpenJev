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

The encoder is a small, one-step next-observation predictor trained only on each corresponding PPO fit's historical training trace. Six current observation features enter an eight-dimensional bottleneck; action enters the decoder. The encoder never receives kill labels, terminal flags, horizon, or evaluation data. Its mean and last pre-action embeddings summarize each trajectory. This is a locally learned dynamics representation, not a pretrained video world model, a recurrent architecture, or evidence that distance measures progress accurately. The raw and latent summaries also differ in dimension and geometry. There is no random-encoder control, so even a positive latent-versus-raw comparison would not isolate the benefit of learning dynamics.

## Reward and optimization details

Success means at least 10 kills over the complete Center episode. This is a new sparse goal, distinct from the previous PPO's dense kill/death/fire-cost objective. References come only from successful members of the current eight-trajectory group. They are clustered in standardized embedding space with DBSCAN, radius 0.5 and minimum two samples. If all successes are noise, their mean is the fallback center.

Failure rewards decrease with distance to the nearest successful center, using the pinned official implementation's min-max normalization and `0.6 * sigmoid(10 * (0.5 - normalized_distance))`. Equal distances receive 0.3. No-success groups receive all-zero rewards; all-success groups receive all ones. Both skip optimization without extra replacement rollouts.

There is a documented source discrepancy: the paper describes standardized squared distances and a 0.8 failure-reward scale, while official code at commit `89d8764b6133ea78557a22eef32c68ad92946d45` uses Euclidean min-max distances and 0.6. This study follows that code's reward convention. The implementation is independent; the small DBSCAN special case and its noise fallback are tested.

Group-normalized trajectory advantages feed a clipped actor loss. The implementation uses four full-group updates, equal trajectory weighting, exact categorical KL to the initial policy as a penalty, and an entropy bonus. It never trains the inherited critic or includes padded terminal steps. Those details and the structured encoder are departures from a full reproduction.

## Protocol and stopping rule

[Prospective protocol](../research/protocols/srpo-doom-v1.json). Nine fits have at most 207,360 new training transitions, followed by 624 deterministic evaluation episodes on 24 new paired seeds in each scenario. Maximum wall time was 30 minutes. The protocol and execution code were frozen at `001c94f` before the pilot. The smoke run uses separate seeds and reduced settings; it is an engineering check only.

To justify a future fresh confirmation, latent SRPO must gain at least 0.5 mean Center kills against **each** of binary rewards, raw similarity, unchanged PPO, and always-fire. Line kills and both durations must not decline by more than 0.5. Every latent-model episode must also pass the 1 ms p95 decision-time screen. These are descriptive development thresholds, not a statistical efficacy test. The paired crossed-bootstrap 95% intervals are exploratory, unadjusted, and limited by three training fits. Both scenarios were used in earlier research.

No threshold changes, fallback selection, or extra tuning follow a failed gate. A passing pilot would motivate a separately frozen confirmation, not establish efficacy or ICLR novelty.

```sh
uv sync --frozen --extra rl --extra dev
uv run --extra rl python research/srpo_doom.py --smoke --output runs/srpo-smoke-new
uv run --extra rl python research/srpo_doom.py --output evidence/srpo-doom-v1/pilot-new
uv run --extra rl python research/analyze_srpo_doom.py evidence/srpo-doom-v1/pilot-new --output evidence/srpo-doom-v1/analysis-new.json
```

Use new output paths. No paid API calls. Historical evidence and checkpoints remain unchanged.


## Completed pilot: continuation criteria not met

The run completed **113,861 new interactions**, 1,152 training episodes, and **624 evaluation episodes** in 463.94 workstation seconds. No further training or fresh confirmation was launched after this result. These are new evaluation seeds, so the frozen PPO means differ from its earlier confirmation report.

| Controller | Center kills | Center game seconds | Line kills | Line game seconds |
|---|---:|---:|---:|---:|
| Group binary | 11.875 | 21.848 | 30.167 | 29.050 |
| SRPO raw adaptation | 11.889 | 21.633 | 30.167 | 29.050 |
| SRPO latent adaptation | 10.653 | 20.476 | 30.167 | 29.050 |
| Frozen PPO history | 11.292 | 20.952 | 30.167 | 29.050 |
| Always fire | 8.542 | 16.843 | 30.167 | 29.050 |

Learned rows average three fits and 24 paired seeds. Always-fire is evaluated once per seed, not treated as three independent runs. Line contains capped episodes. All methods have identical paired kills and duration there; aggregate equality alone is not a proof of identical policies.

| Center contrast: latent adaptation minus control | Kill difference | Exploratory 95% interval |
|---|---:|---|
| Group binary | -1.222 | [-3.236, 0.542] |
| Raw similarity | -1.236 | [-3.208, 0.500] |
| Frozen PPO | -0.639 | [-1.792, 0.694] |
| Always-fire | +2.111 | [0.417, 3.570] |

Latent similarity missed the required Center kill gain against binary rewards, raw similarity and frozen PPO. It also missed Center duration noninferiority against binary rewards and raw similarity. Line and latency screens passed. Maximum latent episode p95 times were 0.455 ms in Center and 0.486 ms in Line, excluding engine advancement and trace I/O. These are local timing screens; light reporting ran concurrently.

The wide intervals do not establish that the published SRPO method is harmful or that all latent rewards fail. They show that this particular small dynamics representation and warm-start adaptation did not earn continuation. The raw and binary variants have almost identical mean performance; their improvements over unchanged PPO are descriptive, not confirmed benefits.

| Additional training cost, all three fits | Interactions | Fine-tuning wall seconds | Encoder fitting seconds | Updated groups / total |
|---|---:|---:|---:|---:|
| Group binary | 37,217 | 94.80 | 0 | 42 / 48 |
| SRPO raw adaptation | 38,237 | 95.32 | 0 | 42 / 48 |
| SRPO latent adaptation | 38,407 | 95.68 | 1.44 | 42 / 48 |

Fine-tuning time includes rollout collection, reward computation, actor updates, trace and checkpoint I/O. Encoder time additionally includes loading historical training data and fitting the encoder. Historical PPO pretraining is shared and excluded from these additional costs. Thus the latent variant had no advantage here despite slightly more interactions and computation. The encoder training loss alone does not validate trajectory distance as progress.

Center mean kills by fit were 11.917, 9.167, 10.875 for latent SRPO; 13.333, 12.000, 10.292 for binary rewards; and 13.333, 11.917, 10.417 for raw similarity. No fit was selected or discarded.

![SRPO pilot and measured additional training costs](../evidence/srpo-doom-v1/pilot-results.png)

[Analysis and all checks](../evidence/srpo-doom-v1/analysis-001.json) · [Run manifest and hashes](../evidence/srpo-doom-v1/pilot-001/manifest.json) · [Integrity review](../evidence/srpo-doom-v1/integrity-review.json). The run folder contains all nine final policy checkpoints, three encoder checkpoints, full rollout arrays, compressed environment traces and training logs. The audit verifies finite checkpoints, unchanged execution sources, raw artifact hashes and identical first-group observations/actions/probabilities across paired variants. All 73 repository tests passed, including six focused reward and actor-loss tests.

The next defensible step would be to validate whether the reward representation orders actual task progress before another policy-training campaign. A sparse goal-based task and a random-encoder control would make that question clearer. Neither is implemented or claimed here. The existing paper PDF still covers studies through the preceding PPO/DQN experiment; this follow-up has its own report.
