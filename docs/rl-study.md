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

While training was running, before any development evaluations, we separately froze [two diagnostic policies](../research/protocols/rl-doom-v1-simple-controls.json): always fire and alternating fire/wait. They use the same rule steering and hard constraints. They test whether any learned advantage has a trivial firing-pattern explanation. They do not modify the main protocol, family selection, or primary confirmation gate. Each completed stage can run these controls on its same game seeds, with separate traces and receipts. A positive primary gate alone would not establish a nontrivial learned mechanism if these controls explain the result.
