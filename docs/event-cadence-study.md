# Combined observable-event memory

This is a separate, prospectively frozen follow-up to the ammo-only cadence study. Source and protocol were committed at `337629c`. The original memory study's failed gate remains unchanged.

The fixed candidate pauses for one seven-tic window after either an observed ammo decrease or a hit reported by the preceding window. Ammo feedback also detects misses; hit feedback remains available when a scenario replenishes ammunition. The controller reads neither hidden weapon readiness nor future outcomes. Its steering and aiming threshold match the original rule.

Historical rule traces motivated the combination: 31 Center and 738 Line fire commands immediately following a hit had no next-window hit. These historical traces are exploratory evidence, not the new confirmation set. The fixed candidate and all thresholds were committed before its development run and before analysis of the ongoing ammo-only confirmation.

The new protocol uses 24 development seeds per scenario and five arms: combined events, rules, ammo only, hit only, and rest after every fire command. The combined candidate must improve mean utility by at least 0.5 in both scenarios while losing no more than 0.5 mean kills. It cannot be replaced by a better-looking ablation after development.

If it qualifies, one confirmation uses 96 fresh seeds per scenario, the same five policies, and all five preserved history-MAP fits: 1,920 episodes. Four primary utility contrasts and four kill contrasts use the same predeclared 99.375% bootstrap intervals, utility-gain threshold and kill noninferiority requirement as the ammo-only study. Decision p95 must remain below 1 ms. Both scenarios are used during development, so fresh-seed confirmation is not evidence of unseen-task generalization.

This is a small handwritten memory controller. A successful comparison would establish an engineering improvement for these tasks and this command-cost objective, not Bayesian RL, adaptive computation, or a novel architecture. [Dynamic Action Repetition](https://ojs.aaai.org/index.php/AAAI/article/view/10918) and [FiGAR](https://arxiv.org/abs/1702.06054) are relevant established work on action timing; this study keeps the decision-window duration fixed.

The original rule is deliberately unchanged as a reference. Single-signal ablations test whether both feedback channels are needed. Actual kills, game reward, ammunition use and truncations accompany command-window utility, which is not per-shot accuracy. Net ammo use can hide consumption under replenishment and is reported only as net change.

## Reproduce

Use new directories to preserve existing runs.

```sh
uv run python research/event_cadence_doom.py --stage development --output evidence/event-cadence-doom-v1/development-new
uv run python research/event_cadence_doom.py --stage confirmation --development evidence/event-cadence-doom-v1/development-new --output evidence/event-cadence-doom-v1/confirmation-new
uv run python research/analyze_event_cadence_doom.py evidence/event-cadence-doom-v1/confirmation-new --output evidence/event-cadence-doom-v1/analysis-new.json
```

No paid API calls or new model-weight training are used. The fixed history models retain their original training data. All results will be reported after their corresponding stages finish.

## Development result

The fixed combined-event policy qualified on the 24 development seeds per scenario. Mean utility rose from 2.729 to 3.521 in Center and from 14.042 to 19.865 in Line, with identical kills to rules in each paired episode. Ammo-only matched the combination in Center; hit-only matched it in Line. The command-rest control reached 22.646 utility in Line, showing that a simpler schedule remains a strong competitor. These are development comparisons, not confirmation.

[Full development result](../evidence/event-cadence-doom-v1/development-001/selection.json). Confirmation is running under the frozen protocol.
