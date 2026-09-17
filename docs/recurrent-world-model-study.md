# Recurrent prediction on a memory task

**Status: planned, not run. No effectiveness result is available.** This pilot tests whether memory helps a small policy, then whether predicting future observations and latent states adds value beyond reward prediction. It uses MiniGrid Memory rather than the Doom firing task. There is no Astra supervision, imagined-policy training or planning in this stage.

## Question and controls

All conditions share a small observation encoder, a 64-unit GRU, and policy/value readouts. The actor receives the same observation fields and action history in every condition. Shared components start from matched weights for each seed. The current-only condition resets its recurrent state at every step; the others retain state within an episode.

| Condition | Policy memory | Auxiliary training |
| --- | --- | --- |
| `current_ppo` | Reset every step | None |
| `recurrent_ppo` | Retained within an episode | None |
| `reward_prediction` | Retained within an episode | Action-conditioned reward and termination prediction |
| `world_prediction` | Retained within an episode | The same reward and termination objectives, plus future observation and latent-state prediction |

The first comparison tests the benefit of recurrent state under this training recipe. The primary comparison is `world_prediction` against both `recurrent_ppo` and `reward_prediction`. An advantage over ordinary recurrent PPO alone would not isolate the contribution of latent or observation prediction. Extra predictive heads also add training parameters and compute; the design matches interactions, not total training cost.

The two prediction conditions use the same action-conditioned recurrent transition at horizons one, two and four. Starting from the policy's current latent state, the transition rolls forward using recorded actions. It must not consume future observations during this rollout. Future posterior states computed from the actual trajectory provide stop-gradient targets for the full prediction arm. Observation targets are categorical MiniGrid object, color and state channels. Reward and termination supervision comes from the environment. Transitions crossing an episode reset are excluded from predictive targets.

At each horizon, the reward-control loss is reward mean squared error plus termination binary cross-entropy. The full prediction arm adds latent mean squared error and observation cross-entropy. Observation cross-entropy averages four equally weighted groups: object identity, color, object state and agent direction; each image group averages over its visible-window cells. Average the summed objective across valid horizons one, two and four, then multiply by **0.1** before adding it to the PPO loss in either prediction arm. Future latent targets are detached without target normalization. Stop-gradient by itself is not a guarantee against collapse. Report latent feature variation and compare prediction error with copying the current state.

## Environment and budget

Use the native seven-action space and native partial 7 by 7 observation window from [MiniGrid Memory](https://minigrid.farama.org/environments/minigrid/MemoryEnv/). Do not supply the full map, cue identity, correct branch or success/failure positions as policy inputs. Privileged environment fields may identify evaluation outcomes but cannot enter the agent's observations or training loss targets beyond native observations, reward and termination.

Train on **size eleven**, with shifts to **sizes seventeen and twenty-three**. A pretraining geometry check found that size seven can expose the initial cue at the decision point, so it was excluded from the memory comparison. The native view remains seven cells across. Preserve this visibility check in the execution evidence; the size choice is based on environment geometry, not trained-policy scores.

Configure the environment's maximum episode length as **128 steps** for all three sizes. This is an explicit adaptation: native Memory uses a size-dependent limit of `5 * size**2`. The configured limit also sets the reward denominator, so successful return is `1 - 0.9 * steps / 128`. Report success and return separately so time-sensitive reward does not obscure correctness. Increasing map size changes navigation difficulty as well as required memory duration.

| Setting | Planned value |
| --- | --- |
| Training seeds | 17, 29, 43 |
| Environments per fit | 16 |
| Rollout length | 64 steps |
| Rollout updates | 512 |
| Interactions per fit | 524,288 |
| Conditions and fits | Four conditions, three seeds each; 12 fits |
| Total planned interactions | 6,291,456 |
| PPO epochs per rollout | Two |
| Minibatch | Eight complete environment sequences |
| Learning rate | 0.0003 |
| Discount / GAE | 0.99 / 0.95 |
| PPO clipping | 0.2 |
| Value / entropy coefficients | 0.5 / 0.01 |
| Gradient norm cap | 0.5 |
| Recurrent gradient window | Full 64-step rollout, with episode masks |
| Checkpoint selection | Final snapshot only |
| Execution | Local CPU; measured time and model size reported |

Preserve episode order within recurrent minibatches. Carry recurrent state across consecutive rollout boundaries with a detached gradient history; reset it at episode boundaries. Bootstrap time-limit truncations from the final observation and its corresponding recurrent state, while genuine terminal states receive no value bootstrap. Never substitute the next episode's reset observation for the previous episode's final transition.

The budget was increased before freezing after an execution smoke measured roughly 0.11 seconds per rollout update for the controls and 0.16 seconds for full prediction. This is a runtime-based allocation decision; no scored evaluation or checkpoint comparison informed it. Record actual completed interactions and measured training time rather than presenting the estimate as an execution result.

## Evaluation and continuation

Evaluate every final snapshot on the same **128 new environment seeds per size**, using greedy argmax actions. Freeze the seed lists before evaluation; training seeds and evaluation seeds must be disjoint. Include a uniform random-action baseline on the same environment seeds with a frozen action RNG. Preserve per-fit episode results, including success, wrong-goal termination, timeout, return and length. Report all three training fits, their mean, and paired differences on common evaluation seeds; three fits do not justify a broad reliability claim.

The predefined candidate is `world_prediction`. Continue to a separately designed confirmation or planning study only if all of the following hold:

1. Its mean success rate is at least 80% on the training-size evaluation.
2. Its mean success rate exceeds `recurrent_ppo` by at least five percentage points on the same-size evaluation and on each shifted size.
3. Its mean success rate exceeds `reward_prediction` by at least five percentage points on the same-size evaluation and on each shifted size.

These thresholds are a pilot continuation rule, not a statistical significance test. Keep outcomes from every arm even if the gate fails. Do not choose a different checkpoint, seed, shift or comparison after seeing results.

For each recurrent checkpoint, also evaluate an otherwise identical policy whose hidden state is cleared **before every step**. Retain the same current observation, previous-action input, greedy action rule and environment seeds. This tests aggregate reliance on retained state; it does not train a new agent or replace the primary comparisons. It is not a selective removal of cue memory.

Use a shared, fixed uniform-random-action probe of **64 sequences of 16 steps per size** to assess the learned predictive representation. Freeze its seeds and action RNG before scoring, and use the same recorded probe trajectories across conditions and training seeds. Report one-step latent prediction mean squared error beside the persistence baseline that copies the current state, latent feature variance, and future-observation, reward and termination losses. Keep episode-boundary masking consistent with training. These diagnostics evaluate prediction on the probe's state distribution; they are not gameplay efficacy, calibration or evidence of reliable long imagined rollouts, and do not select checkpoints or change the continuation rule.

Record training and evaluation time separately, including auxiliary prediction computation. Report parameter counts and inference cost with and without unused auxiliary heads. A fixed interaction budget can establish a sample-efficiency difference under this recipe; it does not by itself establish a compute-efficiency difference.

## Checks before execution

- Replay a saved sequence under unchanged weights and verify agreement with rollout action log probabilities, including internal episode resets.
- Exercise a true terminal state and a time-limit truncation within a rollout. Check hidden-state resets, final-observation targets, value bootstrap and multi-step target masks.
- Use two different histories ending in the same observation and previous action. The current-only actor must agree; the recurrent actor may retain a difference. Editing future observations must not change earlier actions or logits.
- Freeze code hashes, environment/package versions, selected sizes, seed lists, loss definitions and coefficients, and the continuation rule before scored evaluation. Retain durable run receipts and all final checkpoints.

## What this can establish

A positive result would support a useful predictive auxiliary objective for this small recurrent policy on an adapted MiniGrid task. It would not establish a new recurrent-world-model architecture, general English understanding, transfer to Doom, calibrated probabilities or ICLR novelty. One-step or teacher-forced prediction accuracy also does not establish reliable long imagined trajectories. Planning, control through imagined rollouts, an independent environment and a separate confirmation remain future experiments.

Relevant prior work includes [UNREAL](https://arxiv.org/abs/1611.05397), which studies auxiliary objectives including reward prediction; [Self-Predictive Representations](https://arxiv.org/abs/2007.05929), which predicts future latent targets with learned transitions; [PlaNet](https://arxiv.org/abs/1811.04551), which learns latent dynamics for planning; and [Dreamer](https://arxiv.org/abs/1912.01603), which learns behavior through latent imagination. This pilot is inspired by those mechanisms and does not reproduce their architectures or published results. It does not use pretrained V-JEPA.

## Reproduce

Use the existing `hosted` and `dev` dependencies, then install `minigrid==3.0.0` into the research environment without changing the project lockfile. The frozen plan records exact package versions, environment source and training source hashes. A different runtime requires a new plan; it must not be presented as the same execution.

```sh
uv pip install --python .venv/bin/python minigrid==3.0.0
.venv/bin/python -m pytest tests/test_memory_env.py tests/test_predictive_memory.py -q
.venv/bin/python scripts/recurrent_world_study.py freeze --out runs/recurrent-world-v1/plan.json
.venv/bin/python scripts/recurrent_world_study.py run --plan runs/recurrent-world-v1/plan.json --out runs/recurrent-world-v1/execution
.venv/bin/python scripts/recurrent_world_study.py report --plan runs/recurrent-world-v1/plan.json --run runs/recurrent-world-v1/execution --out runs/recurrent-world-v1/report
```

Fresh output directories are required. An interrupted run retains its started receipts, per-update learning logs and completed fits; inspect the existing process and evidence before taking a recovery action. The runner does not silently restart or select partial fits. Trained checkpoints and raw logs stay local; the public report contains the protocol, all per-fit scores and per-episode evaluation outcomes.
