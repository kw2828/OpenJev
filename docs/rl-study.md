# PPO and DQN firing control

This is a new bounded experiment following the completed command-efficiency studies. It tests actual reinforcement learning: agents collect game transitions and optimize future reward. This is separate from the earlier supervised logistic outcome heads and does not implement proprietary RLCD.

Three families are compared: PPO with current observations, PPO with causal history, and DQN with the same causal history. Every family trains three independent seeds, each for exactly 32,768 interactions in Defend the Center. All policies choose fire/wait with unchanged rule steering. The two PPO variants have identical input and network dimensions; the current-only variant's history positions contain zeros. History contains previous observations, issued fire, observed hit and age since firing. It is handcrafted input, not a learned recurrent network.

The training reward is kills minus death minus 0.01 per issued-fire window. The previous command-efficiency utility remains secondary. The primary continuation rule requires at least one additional mean kill in Center with a positive interval lower endpoint, kill noninferiority on Line, duration noninferiority in both, and a latency screen. The episode horizon is explicitly observed and finite, so reaching the cap terminates value bootstrap. This avoids treating a hidden time limit as an ordinary terminal state.

Development evaluates all three final fits per family on 24 new paired seeds in each scenario. Selection is by family, never by lucky training seed or intermediate checkpoint. At most one family proceeds to 96 new confirmation seeds per scenario; if none qualifies, confirmation is not run. Confirmation includes PPO-current as a reference and both plain rules and the stronger event-memory rule. All prior failed gates remain unchanged.

These are established [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and [vanilla DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) implementations from pinned Stable Baselines3 2.9.0. See [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). Matching interactions does not match optimization compute or on-policy data. The new RL experience also differs from the older supervised datasets, so comparisons do not isolate an algorithm effect at matched training data. No new algorithm, Bayesian advantage, or ICLR novelty is claimed.

[Full prospective protocol](../research/protocols/rl-doom-v1.json).

```sh
uv sync --frozen --extra rl --extra dev
uv run --extra rl python research/rl_doom.py --stage development --output evidence/rl-doom-v1/development-new
uv run --extra rl python research/analyze_rl_doom.py evidence/rl-doom-v1/development-new --output evidence/rl-doom-v1/development-analysis-new.json
# Only when the frozen development selection qualifies a family:
uv run --extra rl python research/rl_doom.py --stage confirmation --development evidence/rl-doom-v1/development-new --output evidence/rl-doom-v1/confirmation-new
uv run --extra rl python research/analyze_rl_doom.py evidence/rl-doom-v1/confirmation-new --output evidence/rl-doom-v1/confirmation-analysis-new.json
```

Use new output paths. No paid API calls. A hard 30-minute wall limit bounds each stage; partial or failed stages cannot establish efficacy. The development budget is 294,912 interactions plus 528 evaluation episodes. Confirmation is at most 1,536 episodes. Protocol, code, runtime versions, final checkpoints, logs and compressed traces are preserved with hashes.


## Simple-control addendum

While training was running, before any development evaluations, we separately froze [three diagnostic policies](../research/protocols/rl-doom-v1-simple-controls.json): always fire, alternating fire/wait, and fire whenever an enemy is visible. They use the same rule steering and hard constraints. They test whether any learned advantage has a trivial firing-pattern explanation. They do not modify the main protocol, family selection, or primary confirmation gate. Each completed stage can run these controls on its same game seeds, with separate traces and receipts. A positive primary gate alone would not establish a nontrivial learned mechanism if these controls explain the result.


## Completed development results

Completed 294,912 training interactions and 528 development evaluation episodes. **PPO history** qualified for one fresh confirmation.

| Controller | Center kills | Center game seconds | Line kills | Line game seconds |
|---|---:|---:|---:|---:|
| PPO current | 8.778 | 17.564 | 30.236 | 29.877 |
| PPO history | 10.278 | 18.812 | 30.500 | 30.031 |
| DQN history | 9.347 | 17.716 | 21.014 | 23.263 |
| Rules | 7.000 | 15.518 | 26.208 | 26.836 |
| Rules + event memory | 7.000 | 15.518 | 26.208 | 26.836 |

Learned rows average all three training fits and matched game seeds. Fixed rules are run once per seed, not repeated as independent data. These finite-horizon durations can include capped episodes.

| Family | Center kills by fit | Line kills by fit |
|---|---|---|
| PPO current | 8.92, 7.58, 9.83 | 30.50, 29.71, 30.50 |
| PPO history | 11.29, 10.04, 9.50 | 30.50, 30.50, 30.50 |
| DQN history | 7.42, 10.83, 9.79 | 30.50, 3.62, 28.92 |

PPO scores are action-selection probabilities. DQN scores are discounted action-value estimates, not probabilities of success. Neither is a calibrated success probability.

Development uses the disclosed selection thresholds, not a statistical superiority claim. The analysis includes descriptive crossed-bootstrap intervals; development selection and small training-seed counts limit their interpretation.


![RL development results](../evidence/rl-doom-v1/development-results.png)

![Training curves for all nine fits](../evidence/rl-doom-v1/learning-curves.png)

Learning curves show trailing means over 20 completed training episodes. They include exploration and are not substitutes for deterministic evaluation. Training froze at `d14a0b4`; the simple-control addendum was finalized at `6a48cf8`; development checkpoints and selection were preserved at `278d529` before confirmation.

## What the simple controls reveal in development

| Simple policy | Center kills | Line kills |
|---|---:|---:|
| Always fire | 8.000 | 30.500 |
| Alternate fire/wait | 8.000 | 30.500 |
| Fire when a target is visible | 7.542 | 29.875 |
| PPO history, three-fit mean | 10.278 | 30.500 |

All 10,728 compared Line decisions from history-PPO matched always-fire. That transfer result therefore needs no learned memory mechanism to explain it. Center's descriptive improvement over the simple policies still requires fresh confirmation and remains separate from the primary comparison against event-memory rules.

The controls also demonstrate a limitation of the old utility. Always-fire and alternating-fire produced identical kills, duration, engine reward and ammo use in all 48 paired development episodes. Yet all 1,319 hit windows under alternating-fire occurred while the command was wait, so the old issued-fire-window metric credited none. This is a timing-sensitive command-window estimand, not a reliable measure of combat performance across different firing schedules. Earlier results remain preserved under their stated metric; they should not be reinterpreted as improved accuracy or combat. The new RL study chose kills as primary before these diagnostics.

[Control summaries](../evidence/rl-doom-v1/development-controls-means.json) · [Action matching and delayed-hit diagnostic](../evidence/rl-doom-v1/development-credit-diagnostic.json) · [Training and evaluation receipt](../evidence/rl-doom-v1/development-001/manifest.json).


## Fresh confirmation

The selected family was **PPO history**. The full confirmation gate **passed**.

| Controller | Center kills | Center game seconds | Line kills | Line game seconds |
|---|---:|---:|---:|---:|
| PPO current | 9.372 | 18.786 | 28.007 | 27.043 |
| PPO history | 11.559 | 21.131 | 28.083 | 27.019 |
| Rules | 6.375 | 14.427 | 25.365 | 25.591 |
| Rules + event memory | 6.375 | 14.427 | 25.365 | 25.591 |

Learned rows average all three training fits and matched game seeds. Fixed rules are run once per seed, not repeated as independent data. These finite-horizon durations can include capped episodes.

| Family | Center kills by fit | Line kills by fit |
|---|---|---|
| PPO current | 10.38, 7.55, 10.19 | 28.08, 27.85, 28.08 |
| PPO history | 12.29, 11.25, 11.14 | 28.08, 28.08, 28.08 |

PPO scores are action-selection probabilities. DQN scores are discounted action-value estimates, not probabilities of success. Neither is a calibrated success probability.

| Primary contrast versus event-memory rules | Difference | 98.75% interval |
|---|---:|---|
| Center kills | +5.184 | [3.889, 6.573] |
| Center game seconds | +6.703 | [4.703, 8.813] |
| Line kills | +2.719 | [1.073, 4.417] |
| Line game seconds | +1.428 | [-0.213, 3.080] |

Intervals resample paired game seeds and the three training fits. Only these four primary contrasts use the within-study multiplicity adjustment. Three fits give limited information about training variability. Secondary comparisons cannot rescue a failed gate.


History-PPO improved Center kills over PPO-current by +2.188 [0.521, 4.063] and duration by +2.345 seconds [0.117, 4.906] in exploratory secondary comparisons. Line's history-versus-current difference was +0.076 kills [-0.219, 0.583], so a benefit of history was not established there. These are handcrafted history inputs, not a learned recurrent architecture.

The old net command utility changed by -0.738 in Center [-2.087, 0.767] and -10.760 in Line [-12.943, -8.636] against event-memory rules. Passing the new combat gate does not pass or replace the earlier utility gates. The reward and endpoint change was frozen before new training.

The selected controller used about 0.118 ms per decision on this workstation, excluding engine advancement, observation extraction and trace I/O. Maximum episode p95 values were 0.576 ms in Center and 0.524 ms in Line. Light reporting work ran concurrently; this is a latency screen, not an isolated systems benchmark.

[Fresh confirmation analysis](../evidence/rl-doom-v1/confirmation-analysis-001.json) · [Preserved run receipt](../evidence/rl-doom-v1/confirmation-001/manifest.json).


## Fresh simple-control comparisons

The three diagnostic controls completed a further 576 episodes on the same confirmation seeds, separately from the 1,536 primary episodes.

| Policy | Center kills | Line kills |
|---|---:|---:|
| PPO history, three-fit mean | 11.559 | 28.083 |
| Always fire | 8.542 | 28.083 |
| Alternate fire/wait | 8.542 | 28.083 |
| Fire when a target is visible | 7.156 | 28.208 |

History-PPO gained +3.017 Center kills over always-fire [2.024, 4.115] and +4.403 over visible-target fire [3.292, 5.542]. These secondary intervals are exploratory and outside the primary four-comparison correction family. They support a useful Center controller beyond these simple patterns, not a new RL method.

All **288 Line learned-policy episodes** matched the always-fire issued-action sequence exactly. The Line gain over event-memory rules therefore has a trivial control explanation. Both always-fire and alternate-fire again matched kills, duration, engine reward and ammo use on every paired seed. The old utility failed to credit all 4,865 hit windows for alternating-fire because they occurred during waits.

![Fresh RL confirmation with simple controls](../evidence/rl-doom-v1/confirmation-results.png)

[Secondary comparisons and trace diagnostics](../evidence/rl-doom-v1/confirmation-controls-analysis-001.json). The earlier development diagnostic analysis 002 only corrects file-handle cleanup in its analyzer; all numerical results match analysis 001 exactly.

## Local RL gameplay

![PPO history, first confirmation seed and first training fit](assets/ppo-history-141000.gif)

First confirmation seed 141000, training fit 0: **11 kills in 25.09 game seconds**, then death. This is a local 10,947-parameter actor/value model using structured engine observations and fixed rule steering. It is separate from the browser Qwen language model. The GIF reproduces the preserved episode's gameplay fields; playback is at game speed, not a model-speed benchmark. [Recording receipt](assets/ppo-history-141000.json).

```sh
uv run --extra rl python research/record_rl_doom.py --run evidence/rl-doom-v1/confirmation-001 --models evidence/rl-doom-v1/development-001 --arm ppo_history --replicate 0 --scenario defend_the_center --seed 141000 --output runs/ppo-replay.gif
uv run --no-project --with matplotlib python research/plot_rl_doom.py evidence/rl-doom-v1/confirmation-analysis-001.json --prefix runs/rl-confirmation --controls evidence/rl-doom-v1/confirmation-controls-means.json
```

## Conclusion

Standard PPO produces a useful **Center combat improvement**, including over the matched current-only PPO and the tested simple controls. The broader cross-scenario result does not establish general learned memory: Line reduces to always-fire. DQN transfer was unstable under this budget. No proprietary RLCD, new RL algorithm, neural recurrence, or ICLR novelty is established.

A next experiment should vary action timing or weapon delay, compare recurrent PPO against history inputs and the simple firing policies, and keep kills/survival as the objective. It needs a new frozen protocol and fresh seeds. This study does not authorize a claim that those extensions already work.
