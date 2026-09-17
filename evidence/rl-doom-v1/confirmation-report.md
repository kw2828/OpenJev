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
