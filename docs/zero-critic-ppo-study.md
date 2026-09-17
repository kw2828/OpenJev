# Does zero critic initialization improve PPO learning?

**Running; no scored follow-up result yet.** Protocol and sources were frozen in `5a3a08d` before the full run. Seed 7, two-update smoke checks and 25 focused tests passed. This is an optimization comparison of existing architectures, not a new RL algorithm.

The completed [original PPO study](associative-ppo-study.md) achieved mean same-size success of 34.375% for GRU, 0% for the feedforward adapter and 17.1875% for each associative store. Its selective-writing gate failed. The separate [initialization probe](ppo-initialization-probe.md) found nonzero actor and critic gradients on unrewarded first rollouts; zeroing the value head removed both. That observation motivates a training test but does not establish that initialization caused the earlier failures.

## One training change

Run the same four architectures and seeds with the fresh value-head weight and bias set to zero immediately after normal initialization. Keep that head trainable. The wrapper imports the original trainer unchanged and checks that all other parameters, policy logits, recurrent states and random-number state remain identical before optimization. No pretrained checkpoint is used.

| Setting | Frozen value |
| --- | --- |
| Architectures | GRU, feedforward adapter, global writes, selective writes |
| Seeds | 101, 113, 127 for every architecture |
| Environment | Cue-visible MiniGrid Memory, native seven actions and sparse rewards |
| Train / evaluation sizes | 11 / 11, 17, 23 |
| Episode cap / partial view | 128 steps / 7 by 7 |
| Environments / rollout / updates | 16 / 64 / 1,024 |
| Interactions per fit / additional total | 1,048,576 / **12,582,912** |
| PPO epochs / environment minibatch | 2 / 8 |
| Optimizer steps per fit | 4,096 |
| Adam learning rate / epsilon | 0.0003 / 0.00001 |
| Discount / GAE lambda | 0.99 / 0.95 |
| PPO clip / value coefficient / entropy coefficient | 0.2 / 0.5 / 0.01 |
| Gradient norm cap | 0.5 |
| Model selection | Final checkpoint only; every declared fit |

Training data generation, action-sampling RNG reset, optimizer, entropy term, normalization and model code remain unchanged. No shortened panel fits, selective resumes, extra architecture or additional algorithm is part of this comparison.

## Controls and evaluation

Reuse all twelve completed original-head fits. Preparation requires the original execution and report to be complete, and hashes their checkpoints, logs, evaluations and receipts together with the completed initialization probe. Changed inputs or sources invalidate the prepared plan. The controls ran earlier, so the conditions are not contemporaneously interleaved; report runtime separately rather than interpreting timing differences as an initialization benefit.

Evaluate on the **same 128 previously scored paired development seeds per size**. Preserve intact, all-state-reset, cue-swapped and native-start evaluations, plus store-only resets for store architectures. Pair architecture, training seed, condition, size and episode seed. Keep timeouts in the denominator. Native-start transfer remains descriptive because changing the starting pose also changes navigation distance.

The uniform-random reference repeats the same seeded policy and episodes. Its episode contents must match the original reference exactly; timing fields may differ. It is not independent new evidence. Each panel has 55 evaluation records covering 21,120 assigned episodes across all conditions, including that repeated reference.

Reuse existing learning logs for entropy, cumulative completed episodes and rolling success. Discovery summaries must retain rolling-window coverage and censoring limits. These logs do not provide raw advantages, gradient-component histories or exact cumulative reward-event counts.

## Keep the two gates separate

The frozen **optimization continuation gate** requires at least a ten-percentage-point mean same-size success improvement over the corresponding original-head control for at least three of four architectures, with no architecture degrading by more than five percentage points. Publish all paired seeds and shifts, including failures.

Report the **existing selective-writing gate** separately for both initialization conditions, unchanged: selective writes must reach at least 80% mean same-size success and 70% in every fit, exceed each of GRU, adapter and global writes by five percentage points at every size, and lose at least twenty percentage points after same-size store-only resets. An optimization gain does not establish selective-writing superiority or use of memory.

This reused development panel cannot provide independent confirmation. Zeroing the head changes both bootstrap-derived actor advantages and critic gradients into shared features; any training benefit cannot be attributed solely to advantage normalization. A broader architectural claim would also require stronger controls, including the cached-initial-state model that solved the earlier forced-route diagnostic. That control is outside this single-factor experiment.

## Preparation and reproduction

The commands below describe the planned execution. Preparation binds the completed original controls and probe before any full zero-head training starts. Use fresh output paths; existing artifacts are never overwritten.

```sh
.venv/bin/python scripts/zero_critic_ppo_study.py prepare \
  --control-plan evidence/associative-ppo-v1/plan.json \
  --control-run runs/associative-ppo-v1/execution \
  --control-report runs/associative-ppo-v1/report \
  --probe-plan evidence/ppo-initialization-probe-v1/plan.json \
  --probe-run runs/ppo-initialization-probe-v1/execution \
  --out evidence/zero-critic-ppo-v1/plan.json
.venv/bin/python scripts/zero_critic_ppo_study.py run \
  --plan evidence/zero-critic-ppo-v1/plan.json \
  --out runs/zero-critic-ppo-v1/execution
.venv/bin/python scripts/zero_critic_ppo_study.py report \
  --plan evidence/zero-critic-ppo-v1/plan.json \
  --run runs/zero-critic-ppo-v1/execution \
  --out runs/zero-critic-ppo-v1/report
```

The report preserves the inherited per-fit and intervention results under `core/`, and writes paired initialization comparisons and separate gate outcomes to `comparison.json`. Its completion receipt confirms repeated random-episode identity. The wrapper's 25 focused tests cover initialization parity, exception cleanup, preparation requirements, pairing and gate failures. Full reproduction requires the bound local control artifacts in addition to the published plans.
