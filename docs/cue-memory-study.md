# Does a visible cue make recurrence useful?

**Status: protocol prepared; scored training has not run.** The [previous memory study](recurrent-world-model-study.md) found that none of its policies acquired a cue that was initially unseen. This follow-up removes that search requirement and tests the same four controllers. It is a development diagnostic, not independent confirmation or evidence of a new algorithm.

## Intervention

Generate the native [MiniGrid Memory](https://minigrid.farama.org/environments/minigrid/MemoryEnv/) map with its original random draws. Then set the agent's starting position to **x=1 on the center row, facing east**, on every reset. The cue is visible in the native 7 by 7 partial observation. Both branch objects are hidden at the start; the cue is hidden at the fork in all four orientations on sizes 11, 17 and 23.

Every controller receives the same starting rule. Map, cue identity, branch assignment, RNG state, seven actions and sparse native rewards are unchanged. No goal identity or privileged map feature enters the model. As before, the 128-step episode cap changes the native time-sensitive reward denominator.

Pinning the start also changes navigation distance and the distribution of states. Comparing this study to the earlier native-start study does not isolate a pure causal effect of cue visibility. The strongest comparisons are between controllers and interventions within this study. Seed separation also does not guarantee unique layouts in this finite environment.

## Matched training

Reuse the exact frozen training functions and model from `recurrent-world-v1` through an isolated import. The new runner substitutes only the documented environment, fresh seed panel, fit-order randomization and study metadata. It hashes the old and new source files together. It does not edit the old source or rewrite its results.

| Setting | Value |
| --- | --- |
| Controllers | Current-only PPO, recurrent PPO, reward/termination prediction, full world prediction |
| Training seeds | 61, 73, 89 |
| Training / evaluation sizes | Train 11; evaluate 11, 17, 23 |
| Interactions | 524,288 per fit; 6,291,456 over 12 fits |
| Rollout / environments / updates | 64 / 16 / 512 |
| PPO epochs / sequence minibatch | 2 / 8 environments |
| Hidden width | 64 |
| Learning rate / entropy coefficient | 0.0003 / 0.01 |
| Auxiliary coefficient / horizons | 0.1 / 1, 2, 4 |
| Selection | Final checkpoint of every fit |
| Evaluation | Greedy actions, 128 assigned seeds per size |

All remaining optimizer, loss, bootstrap, reset, predictive-target and clipping settings are identical to the [original protocol](recurrent-world-model-study.md#frozen-study-design). The current-only controller still receives the previous action. It is a one-action-history control, not a strictly memoryless policy. Extra prediction heads add training cost and gradient-bearing parameters; the experiment matches interactions, not total compute.

The training seed stream starts at `20,000,000 + training_seed * 100,000`; evaluation starts at `40,000,000 + size * 10,000`; prediction probes start at `50,000,000 + size * 10,000`. Fit order uses seed 91637. An execution smoke uses seed 7 for two updates per arm only and does not select a checkpoint or tune the budget.

## Evidence about memory and cue use

Evaluate each recurrent checkpoint with intact state and with state cleared before every action. Preserve current observation and previous action. Use the same assigned episodes for every mode. Include the current-only controller and a uniform random policy, plus the original common random-action prediction probes.

Two additional evaluations use each final checkpoint:

1. **Inconsistent-cue intervention:** swap the visible key/ball cue after reset, while preserving the original stored reward target and both branch objects. Both cue categories are familiar, but their relation to reward is deliberately inconsistent. Measure paired terminal-branch reversal over **all assigned episode pairs**, and report how many pairs reach a branch in both modes. Timeout pairs stay in the denominator. A success drop alone is not sufficient evidence of the expected cue-dependent response. This is a diagnostic intervention, not ordinary task accuracy.
2. **Native-start transfer:** restore the original random starting position, with no retraining. Report all outcomes on the same seed panel. This changes the task distribution and can require searching for a previously unseen cue.

The primary recurrence gate requires recurrent PPO to achieve all of:

- At least 80% mean success on size 11.
- At least 70% success for every individual training fit on size 11.
- At least 20 percentage points above current-only PPO on size 11.
- At least a 20-point drop when the same recurrent checkpoint is reset every step.

This gate tests whether the training recipe develops a useful recurrent mechanism in the adapted task. Swapping establishes cue sensitivity; recurrence controls address reliance on retained state. Neither alone proves a particular internal representation. Larger sizes and native-start transfer are reported separately and do not become alternative success gates after scoring.

The separate full-prediction continuation rule is retained unchanged: at least 80% size-11 success, and at least five-point gains over both recurrent PPO and reward prediction on every size. All fits remain in the report if either gate fails. Passing a pilot gate would justify another experiment, not a novelty or ICLR claim.

## Reproduce

Use the existing environment with `minigrid==3.0.0`. The run writes durable started/completed receipts and requires fresh output directories.

```sh
.venv/bin/python scripts/cue_memory_study.py smoke --out runs/cue-memory-smoke-v1
.venv/bin/python scripts/cue_memory_study.py freeze --out runs/cue-memory-v1/plan.json
.venv/bin/python scripts/cue_memory_study.py run --plan runs/cue-memory-v1/plan.json --out runs/cue-memory-v1/execution
.venv/bin/python scripts/cue_memory_study.py report --plan runs/cue-memory-v1/plan.json --run runs/cue-memory-v1/execution --out runs/cue-memory-v1/report
```

The outer completion receipt covers all training, base evaluations and both added interventions. A `core/completed.json` receipt alone does not mean the follow-up finished. Publication also requires `report-completed.json`, written only after the primary results and every intervention validate. An incomplete report remains an unsuccessful attempt; regenerate into a fresh directory after resolving its cause. Never restart a live or partially observed process because a polling call timed out. Inspect its handle and existing receipts first.
