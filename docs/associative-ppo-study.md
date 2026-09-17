# Do associative stores help autonomous reward learning?

**Status: protocol prepared; no scored RL result yet.** The [supervised diagnostic](associative-learnability.md) established that both global and selective stores can retain a useful cue along forced routes. It did not train navigation or establish a selective-writing advantage. This study tests autonomous PPO from fresh initialization.

## Matched comparison

Use the same four architectures: plain GRU, a feedforward adapter with a similar added parameter count, global associative writes, and input-dependent selective writes. Every arm receives the same cue-visible MiniGrid reset, native seven actions, native sparse rewards, 128-step episode cap, optimizer and interaction budget. No model inherits a supervised checkpoint. There are no auxiliary prediction losses or privileged visibility inputs.

| Setting | Frozen value |
| --- | --- |
| Training seeds | 101, 113, 127 |
| Training / evaluation sizes | Train 11; evaluate 11, 17, 23 |
| Environments / rollout | 16 / 64 steps |
| Updates per fit | 1,024 |
| Interactions per fit / total | 1,048,576 / 12,582,912 |
| PPO epochs / environment minibatch | 2 / 8 |
| Learning rate / entropy coefficient | 0.0003 / 0.01 |
| Discount / GAE lambda | 0.99 / 0.95 |
| Clip / value coefficient / gradient cap | 0.2 / 0.5 / 0.5 |
| Evaluation | Final checkpoint, greedy actions, 128 assigned episodes per size |

The action-sampling RNG is reset after architecture-specific parameter initialization. A fixed randomized order runs all 12 fits. Source and dependency hashes must match the frozen plan. Evaluation seeds are separated from training seed streams, but this finite environment can reuse layouts across seeds. The budget matches interactions, not total computation.

## Interventions and reporting

Every learned fit is evaluated intact, with all state cleared before each action, with the visible cue swapped while preserving the reward target, and under the original random-start rule. Both store architectures also receive store-only resets, preserving the GRU. One uniform-random reference is included. This yields 55 evaluation records and 21,120 assigned episodes across the three sizes.

Report every fit, success, wrong-goal and timeout rates, returns, lengths, action counts and measured training cost. Paired cue swaps retain timeout pairs in the denominator. Count branch reversals only when both paired episodes reach a branch. Write-strength telemetry is descriptive and uses cue-visibility metadata only after the model acts.

The predeclared selective-writing continuation rule requires all of:

- At least 80% mean same-size success and at least 70% for every fit.
- At least a five-percentage-point mean advantage over each of GRU, feedforward adapter and global writes at every size.
- At least a 20-percentage-point same-size success drop with store-only resets.

If both stores improve equally, report the simpler global store's observed comparison with the controls while retaining a failed selective-writing gate. Do not replace the gate with another metric after seeing results. Stronger initial writes, lower losses or isolated successful episodes do not establish a selective-writing advantage.

These are development comparisons of established fast-weight mechanisms. They do not establish architectural novelty or independent confirmation. The earlier supervised seven-action loss failure does not prove that PPO will fail: it is a different learning objective.

## Run

```sh
.venv/bin/python scripts/associative_ppo_study.py run \
  --plan evidence/associative-ppo-v1/plan.json \
  --out runs/associative-ppo-v1/execution
.venv/bin/python scripts/associative_ppo_study.py report \
  --plan evidence/associative-ppo-v1/plan.json \
  --run runs/associative-ppo-v1/execution \
  --out runs/associative-ppo-v1/report
```

[Frozen plan](../evidence/associative-ppo-v1/plan.json) · [Architecture and prior art](associative-memory.md)
