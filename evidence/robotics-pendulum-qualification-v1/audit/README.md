# Pendulum observation and control qualification

This is a supplied-physics diagnostic. No neural models were trained.

Short-history qualification: **PASS**.

| Controller | Stationary mean cost | Switched-gain mean cost |
|---|---:|---:|
| current_nominal | 1352.487 | 1351.853 |
| velocity_nominal | 298.186 | 298.136 |
| history_identified | 299.428 | 299.459 |
| recent3_identified | 309.409 | 309.419 |
| known_state_physics | 299.428 | 299.442 |
| uniform | 1277.414 | 1271.112 |

Lower native episode cost is better. Every controller receives the same two initial probe actions, included in cost. MPC candidates and budgets are paired. The known-state reference uses finite random-shooting planning and is not an optimal controller.

Independent Gymnasium replay checked 125,952 saved transitions. Paired episode intervals and every case are in summary.json. They do not measure training-seed or architecture uncertainty.

A pass rules out expanding this clean constant-gain task into a long-memory topology study. A failure retains the fixed diagnostic and requires diagnosis, not a claim that biological recurrence is needed.
