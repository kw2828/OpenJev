# JEPA rewards and RL variants in Doom

This bounded pilot tests whether pretrained video similarity improves a small Doom firing policy, and compares several policy-gradient objectives under the same additional interaction budget. The project remains OpenJev.

The protocol is frozen before the study: [jepa-rl-doom-v1.json](../research/protocols/jepa-rl-doom-v1.json). This is an exploratory adaptation, not a new algorithm or a reproduction of the complete SRPO, DAPO or robotics training systems.

## Results

The completed pilot **did not meet its continuation criteria**. All 27 fits received exactly 8,192 additional interactions, totaling **221,184**, followed by **992 evaluation episodes**. Total run time was **14.67 minutes** on this Apple M5 Max Mac with 48 GiB RAM. No paid API calls were used.

![JEPA and RL pilot results](../evidence/jepa-rl-doom-v1/pilot-results.png)

| Method | Center mean kills | Line mean kills | Additional training seconds, 3 fits |
| --- | ---: | ---: | ---: |
| GRPO | 12.12 | 25.38 | 56.9 |
| Dr. GRPO adaptation | 11.88 | 25.38 | 56.9 |
| DAPO loss adaptation | 11.79 | 25.38 | 56.3 |
| RLOO adaptation | 12.12 | 25.38 | 56.3 |
| SRPO pixels | 11.94 | 25.38 | 85.2 |
| SRPO random video | 11.67 | 25.38 | 115.6 |
| SRPO pretrained V-JEPA 2 | 12.40 | 25.38 | 103.2 |
| Sparse PPO | 10.15 | 25.38 | 56.8 |
| Sparse A2C | 11.79 | 25.38 | 55.7 |
| Unchanged PPO | 11.10 | 25.38 | No new training |
| Always fire | 8.19 | 25.38 | No new training |

Pretrained JEPA compared with each predeclared Center control:

| Comparator | JEPA minus control, kills | Exploratory paired 95% interval |
| --- | ---: | ---: |
| GRPO | +0.27 | [-1.02, +1.42] |
| SRPO pixels | +0.46 | [-0.35, +1.44] |
| SRPO random video | +0.73 | [-1.00, +2.62] |
| Sparse PPO | +2.25 | [-0.44, +5.00] |
| Unchanged PPO | +1.29 | [-0.19, +3.02] |
| Always fire | +4.21 | [+2.69, +5.69] |

JEPA's +0.27 kills over GRPO and +0.46 over pixels fall below the predeclared +0.5 threshold. Its intervals against every learned control include zero. The highest mean therefore does not establish a reliable JEPA benefit. Additional local training took 103 seconds for the three JEPA fits versus 57 seconds for GRPO, excluding external pretraining.

All Line methods averaged 25.375 kills. A [post-hoc trace check](../evidence/jepa-rl-doom-v1/line-behavior-diagnostic.json) found all 480 learned-policy episodes matched the corresponding always-fire issued commands and hit/kill events exactly. This scenario is therefore weak evidence for distinguishing these controllers.

These intervals resample both training fits and game seeds. They are exploratory with three warm-start fits, 16 game seeds and multiple comparisons. Means alone are insufficient to establish a reliable advantage. A failed screen ends this pilot without changing its hyperparameters or adding confirmation runs.

The full analysis includes duration, policy latency, per-fit means, all control contrasts, discarded interactions and update counts. The audit verified hashes and interaction budgets, recomputed self-reference rewards, replayed group actor updates from recorded states/actions, and replayed the first encoded trajectory of each visual fit to verify its clip hash. It separately checked sparse PPO/A2C terminal reward traces. This does not validate JEPA's semantic understanding or reproduce SB3 training from scratch.

- [Full analysis](../evidence/jepa-rl-doom-v1/analysis-001.json)
- [Integrity audit](../evidence/jepa-rl-doom-v1/integrity-review.json)
- [Run manifest, versions and hashes](../evidence/jepa-rl-doom-v1/pilot-001/manifest.json)
- [All policy checkpoints and traces](../evidence/jepa-rl-doom-v1/pilot-001/)
- [Engineering checks](../evidence/jepa-rl-doom-v1/engineering-check.json): 83 tests passed, plus a separate nine-method smoke run and encoder checks

## Recorded policy

![First JEPA-trained policy and first evaluation seed](assets/jepa-policy-181000.gif)

First fit, first Center evaluation seed 181000: **13 kills in 19.26 game seconds**. The replay matches the preserved outcome. This small policy was trained using JEPA rewards; JEPA is not running during gameplay. One illustrative recording, not a selected best episode or an inference benchmark. [Recording receipt](assets/jepa-policy-181000.json).

## What is being tested

Nine methods each receive 8,192 additional game interactions for each of three paired historical PPO actors. The action input, small 64x64 policy, fire/wait action space, rule steering and finite horizon stay fixed.

| Method | Training signal and update |
| --- | --- |
| GRPO | Binary terminal success, sample-standardized group rewards, equal trajectory weighting |
| Dr. GRPO adaptation | Centered rewards without standard deviation scaling, fixed-horizon loss normalization |
| DAPO loss adaptation | Asymmetric clipping and mean over real steps; no dynamic sampling or full DAPO recipe |
| RLOO adaptation | Leave-one-out reward baseline, one on-policy REINFORCE update |
| SRPO pixels | In-group success similarity using pooled RGB clips |
| SRPO random video | Same similarity reward with a randomly initialized V-JEPA 2 architecture |
| SRPO pretrained V-JEPA 2 | Same similarity reward using Meta's actual frozen pretrained video encoder |
| Sparse PPO | Actor-critic updates with terminal success reward and a fresh critic |
| Sparse A2C | Actor-critic updates with terminal success reward and a fresh critic |

The goal is at least 10 kills by episode end. Every SRPO arm uses the earlier pinned self-reference reward convention. Group methods share the initial-policy KL penalty and entropy bonus. These implementation choices are stated in the protocol; the names identify adapted objectives, not full published systems.

## How JEPA is used

The official `facebook/vjepa2-vitl-fpc64-256` checkpoint is pinned to revision `b3c1679b7c34d3255ef3547f27c7b226aefab26f`. Its frozen encoder processes 16 uniformly sampled pre-action frames from each completed training episode. The HUD is removed. Final video tokens are averaged and normalized to form a 1,024-dimensional clip representation.

JEPA supplies a retrospective training reward. It is not an action-policy encoder or a learned planning world model. Deployment still uses only the small policy. Sixteen time-spaced episode frames differ from the model's pretraining cadence and 64-frame context, so this is a test of transfer, not an assumed validated Doom representation.

The random-video control has the same architecture and representation size. The pixel control uses the same frames with 8x8 RGB pooling, producing 192 dimensions. This helps distinguish the effect of pretraining from a generic visual similarity reward, although representation geometry still matters.

## Budget and interpretation

- 27 fits, exactly 221,184 new training interactions; no paid API calls.
- 16 fresh game seeds on each of two previously used scenarios, 992 evaluation episodes including unchanged PPO and always-fire controls.
- Group methods use eight stochastic trajectories from a shared initial seed. A final incomplete group is fully charged but excluded from optimization. Homogeneous groups produce no update.
- PPO/A2C use consecutive episode seeds and bootstrap unfinished episodes. Equal interactions do not imply equal effective data, update counts, seed distributions or compute.
- Original environment `training_reward` fields in episode summaries are dense diagnostic totals, not the objective optimized here. Group optimization uses the separately recorded terminal/self-reference rewards; PPO/A2C traces explicitly record sparse learning rewards.
- Training wall time includes collection, preprocessing, encoding, updates and artifact I/O. Model loading is recorded. Historical PPO training and external JEPA pretraining are excluded. These are workstation measurements, not universal speed claims.
- JEPA continues only if it clears the predeclared Center improvement, Line/duration retention and policy-latency thresholds against all six controls. No tuning follows failure. A pass would only justify a separately frozen confirmation study.

Three warm-start fits and 16 evaluation seeds support a pilot, not conference-level novelty or generalization claims. Paired bootstrap intervals are exploratory, without correction for multiple comparisons. Both scenarios were used before; fresh seeds are not unseen tasks.

## Reproduction

Install the existing `rl`, `hosted` and `dev` extras. Download the pinned official Hugging Face safetensors checkpoint into the local cache. The runner uses `local_files_only=True`; large encoder weights are not vendored in this repository. The current encoder path requires Apple MPS.

```sh
uv sync --frozen --extra rl --extra hosted --extra dev
source .venv/bin/activate
python -c "from huggingface_hub import snapshot_download; snapshot_download('facebook/vjepa2-vitl-fpc64-256', revision='b3c1679b7c34d3255ef3547f27c7b226aefab26f', allow_patterns=['config.json','model.safetensors','video_preprocessor_config.json'])"
python research/jepa_rl_doom.py --smoke --output runs/jepa-smoke-new
python research/jepa_rl_doom.py --output runs/jepa-pilot-new
python research/analyze_jepa_rl_doom.py runs/jepa-pilot-new --output runs/jepa-analysis-new.json
```

The short smoke run checks execution and budget handling. It can contain homogeneous groups, so focused tests separately check nonzero optimizer updates. Neither constitutes efficacy evidence.

## What this changes for the research direction

The implementation now supports a controlled video-reward comparison, but this pilot does not justify a new algorithm claim. The continuation screen failed, so no further tuning or confirmation was run. Adding more named optimizers would not resolve the main uncertainty.

A separately designed next experiment should test a decision where memory or prediction is actually necessary, with a matched recurrent actor and an action-conditioned predictive representation. It should include a second environment where always-fire cannot collapse the comparison. That would test whether predictive state helps decisions, which this retrospective clip-reward study does not answer.

## Primary sources

- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985), [Meta code](https://github.com/facebookresearch/vjepa2), [Transformers implementation](https://huggingface.co/docs/transformers/model_doc/vjepa2)
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300), [Dr. GRPO](https://arxiv.org/abs/2503.20783), [DAPO](https://arxiv.org/abs/2503.14476)
- [REINFORCE leave-one-out](https://arxiv.org/abs/2402.14740), [SRPO](https://arxiv.org/abs/2511.15605v2)
- [TRL objective implementation notes](https://huggingface.co/docs/trl/grpo_trainer), [SB3 A2C](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html)
