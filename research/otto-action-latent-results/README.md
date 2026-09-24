# Every action-gap result

All values below are independently reconstructed from the saved final predictions.
No model, teacher or native environment was called to create this report.

Means below average three fit seeds over the same cases; ridge repeats one deterministic fit.
Decision gap averages surviving targets within each case, then cases with support.

| Method | Setting | Long-gap log loss | Long-gap decision gap | Normal log loss | Normal decision gap |
|---|---|---:|---:|---:|---:|
| Action recurrent | lambda3 | 1.062800 | 0.456592 | 0.972827 | 0.313148 |
| Action recurrent | lambda4 | 0.949422 | 0.422583 | 0.879789 | 0.327522 |
| Action blind | lambda3 | 1.035081 | 0.502254 | 0.949271 | 0.369341 |
| Action blind | lambda4 | 0.934890 | 0.409249 | 0.850920 | 0.318533 |
| Direct horizon | lambda3 | 1.105050 | 0.466179 | 1.013273 | 0.362855 |
| Direct horizon | lambda4 | 0.974455 | 0.363322 | 0.924862 | 0.325502 |
| Solved ridge | lambda3 | 2.269969 | 0.428260 | 1.840856 | 0.350361 |
| Solved ridge | lambda4 | 2.583536 | 0.369872 | 2.523586 | 0.325304 |

## Long-gap Brier score

This second proper score is retained even when its ranking differs from log loss.

| Method | lambda3 | lambda4 |
|---|---:|---:|
| Action recurrent | 0.526566 | 0.460269 |
| Action blind | 0.517910 | 0.445465 |
| Direct horizon | 0.532503 | 0.478008 |
| Solved ridge | 0.570411 | 0.408210 |

## Every fit and comparison

| Seed | Setting | Control | Long log loss | Long decision gap | Normal log loss | Normal decision gap | Supported cases | Pass |
|---|---|---|---|---|---|---|---|---|
| 323000001 | lambda3 | action_blind | 1.054845 / 1.044462 (fail) | 0.493154 / 0.478195 (fail) | 0.966845 / 0.949029 (fail) | 0.284812 / 0.345611 (pass) | 30 long; 30 normal | False |
| 323000001 | lambda3 | direct_horizon | 1.054845 / 1.073303 (pass) | 0.493154 / 0.392040 (fail) | 0.966845 / 1.012539 (pass) | 0.284812 / 0.374767 (pass) | 30 long; 30 normal | False |
| 323000001 | lambda3 | ridge | 1.054845 / 2.269969 (pass) | 0.493154 / 0.428260 (fail) | 0.966845 / 1.840856 (pass) | 0.284812 / 0.350361 (pass) | 30 long; 30 normal | False |
| 323000001 | lambda4 | action_blind | 0.936596 / 0.903131 (fail) | 0.383588 / 0.448344 (pass) | 0.837223 / 0.843582 (pass) | 0.340883 / 0.316418 (fail) | 36 long; 38 normal | False |
| 323000001 | lambda4 | direct_horizon | 0.936596 / 0.897760 (fail) | 0.383588 / 0.319554 (fail) | 0.837223 / 0.894844 (pass) | 0.340883 / 0.285528 (fail) | 36 long; 38 normal | False |
| 323000001 | lambda4 | ridge | 0.936596 / 2.583536 (pass) | 0.383588 / 0.369872 (fail) | 0.837223 / 2.523586 (pass) | 0.340883 / 0.325304 (fail) | 36 long; 38 normal | False |
| 323000002 | lambda3 | action_blind | 1.023438 / 1.003266 (fail) | 0.379257 / 0.466505 (pass) | 0.953597 / 0.940299 (fail) | 0.314882 / 0.347240 (pass) | 30 long; 30 normal | False |
| 323000002 | lambda3 | direct_horizon | 1.023438 / 1.119064 (pass) | 0.379257 / 0.510668 (pass) | 0.953597 / 1.016774 (pass) | 0.314882 / 0.390574 (pass) | 30 long; 30 normal | True |
| 323000002 | lambda3 | ridge | 1.023438 / 2.269969 (pass) | 0.379257 / 0.428260 (pass) | 0.953597 / 1.840856 (pass) | 0.314882 / 0.350361 (pass) | 30 long; 30 normal | True |
| 323000002 | lambda4 | action_blind | 0.964986 / 0.910996 (fail) | 0.460842 / 0.415077 (fail) | 0.865950 / 0.805355 (fail) | 0.360581 / 0.353541 (fail) | 36 long; 38 normal | False |
| 323000002 | lambda4 | direct_horizon | 0.964986 / 1.039120 (pass) | 0.460842 / 0.405596 (fail) | 0.865950 / 0.983213 (pass) | 0.360581 / 0.353762 (fail) | 36 long; 38 normal | False |
| 323000002 | lambda4 | ridge | 0.964986 / 2.583536 (pass) | 0.460842 / 0.369872 (fail) | 0.865950 / 2.523586 (pass) | 0.360581 / 0.325304 (fail) | 36 long; 38 normal | False |
| 323000003 | lambda3 | action_blind | 1.110117 / 1.057515 (fail) | 0.497365 / 0.562062 (pass) | 0.998039 / 0.958486 (fail) | 0.339750 / 0.415173 (pass) | 30 long; 30 normal | False |
| 323000003 | lambda3 | direct_horizon | 1.110117 / 1.122782 (pass) | 0.497365 / 0.495830 (fail) | 0.998039 / 1.010507 (pass) | 0.339750 / 0.323223 (fail) | 30 long; 30 normal | False |
| 323000003 | lambda3 | ridge | 1.110117 / 2.269969 (pass) | 0.497365 / 0.428260 (fail) | 0.998039 / 1.840856 (pass) | 0.339750 / 0.350361 (pass) | 30 long; 30 normal | False |
| 323000003 | lambda4 | action_blind | 0.946686 / 0.990542 (pass) | 0.423321 / 0.364325 (fail) | 0.936196 / 0.903823 (fail) | 0.281102 / 0.285641 (pass) | 36 long; 38 normal | False |
| 323000003 | lambda4 | direct_horizon | 0.946686 / 0.986487 (pass) | 0.423321 / 0.364817 (fail) | 0.936196 / 0.896529 (fail) | 0.281102 / 0.337215 (pass) | 36 long; 38 normal | False |
| 323000003 | lambda4 | ridge | 0.946686 / 2.583536 (pass) | 0.423321 / 0.369872 (fail) | 0.936196 / 2.523586 (pass) | 0.281102 / 0.325304 (pass) | 36 long; 38 normal | False |

Each metric cell shows candidate / control. All six conditions per cell must pass.

## Costs

| Method | Seed | Fit seconds | Updates | Parameters |
|---|---|---:|---:|---:|
| action_recurrent | 323000001 | 1.608178 | 320 | 8299 |
| action_blind | 323000001 | 1.200333 | 320 | 8299 |
| direct_horizon | 323000001 | 1.192860 | 320 | 8107 |
| action_blind | 323000002 | 1.069554 | 320 | 8299 |
| direct_horizon | 323000002 | 1.129428 | 320 | 8107 |
| action_recurrent | 323000002 | 1.130674 | 320 | 8299 |
| direct_horizon | 323000003 | 1.198789 | 320 | 8107 |
| action_recurrent | 323000003 | 1.109922 | 320 | 8299 |
| action_blind | 323000003 | 1.158010 | 320 | 8299 |

Ridge: one fit, 0.004526 seconds, 3,024 coefficients.

| Method | Seed | Panel | Cases | Inference seconds |
|---|---|---|---:|---:|
| action_recurrent | 323000001 | gap | 68 | 0.028016 |
| action_recurrent | 323000001 | normal | 68 | 0.042801 |
| action_recurrent | 323000002 | gap | 68 | 0.027467 |
| action_recurrent | 323000002 | normal | 68 | 0.045626 |
| action_recurrent | 323000003 | gap | 68 | 0.031570 |
| action_recurrent | 323000003 | normal | 68 | 0.049280 |
| action_blind | 323000001 | gap | 68 | 0.030639 |
| action_blind | 323000001 | normal | 68 | 0.041152 |
| action_blind | 323000002 | gap | 68 | 0.027435 |
| action_blind | 323000002 | normal | 68 | 0.041523 |
| action_blind | 323000003 | gap | 68 | 0.027057 |
| action_blind | 323000003 | normal | 68 | 0.043618 |
| direct_horizon | 323000001 | gap | 68 | 0.040303 |
| direct_horizon | 323000001 | normal | 68 | 0.061636 |
| direct_horizon | 323000002 | gap | 68 | 0.040701 |
| direct_horizon | 323000002 | normal | 68 | 0.056954 |
| direct_horizon | 323000003 | gap | 68 | 0.063921 |
| direct_horizon | 323000003 | normal | 68 | 0.068595 |
| ridge | 323000001 | gap | 68 | 0.009651 |
| ridge | 323000001 | normal | 68 | 0.012832 |
| ridge | 323000002 | gap | 68 | 0.012885 |
| ridge | 323000002 | normal | 68 | 0.010876 |
| ridge | 323000003 | gap | 68 | 0.009720 |
| ridge | 323000003 | normal | 68 | 0.010935 |

Fit times include construction, optimization, final parameter checks and checkpoint writing. The whole producer additionally includes data preparation, scoring, all prediction views and publication of its artifacts.
Inference is batch one, with Python validation and feature/input preparation, excluding model construction/loading. No end-to-end serving or hardware-independent speed claim.

## Support and interpretation

* lambda3, short: 30 originating cases, 30 with decision support, 120/120 nonterminal targets.
* lambda3, long: 30 originating cases, 30 with decision support, 117/120 nonterminal targets.
* lambda3, all: 30 originating cases, 30 with decision support, 237/240 nonterminal targets.
* lambda4, short: 38 originating cases, 38 with decision support, 146/152 nonterminal targets.
* lambda4, long: 38 originating cases, 36 with decision support, 143/152 nonterminal targets.
* lambda4, all: 38 originating cases, 38 with decision support, 289/304 nonterminal targets.

Shading shows the range across three fits, not a confidence interval. All three fits share the same evaluation cases. The absorbing-found suffix is scored for outcomes and excluded from decisions.
Normal forecasts keep finite learned logits after observed found while freezing state. They do not take the available exact terminal-probability shortcut; this is an implementation limitation shared by all controls.

[Protocol](../otto-action-latent-protocol.md) | [Interpretation](../otto-action-latent-results.md) | [Complete evidence and checkpoints](https://github.com/kw2828/OpenJev/releases/tag/otto-action-latent-v1)
