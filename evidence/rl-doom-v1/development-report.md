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
