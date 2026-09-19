# One engineering case: saved-output review

No scientific qualification gate was evaluated.

Native cost per issued action; lower is better. PRE [0,80), POST [80,200), MOVE [100,150).

| Role | Total | PRE | POST | MOVE |
|---|---:|---:|---:|---:|
| nominal | 0.11349797 | 0.12685897 | 0.10459064 | 0.05677829 |
| adaptive | 0.11448848 | 0.12966094 | 0.10437350 | 0.06446815 |
| frozen | 0.11848501 | 0.12927116 | 0.11129425 | 0.05727800 |
| public_gain | 0.13394715 | 0.18065342 | 0.10280964 | 0.04530133 |
| true_state | 0.14254395 | 0.16854546 | 0.12520961 | 0.12392000 |
| zero | 0.17858525 | 0.18744342 | 0.17267980 | 0.17957590 |

Identifier mean absolute error uses the estimate available at the decision root, before its action.

| Role | Total | PRE | POST | MOVE |
|---|---:|---:|---:|---:|
| adaptive | 0.18250000 | 0.21437500 | 0.16125000 | 0.14600000 |
| frozen | 0.42825000 | 0.09562500 | 0.65000000 | 0.65000000 |

Post-observation errors against the just-completed transition's gain are reported separately in JSON.

- One reused seed410 engineering case, one gain-switch direction and fixed goals; no independent cohort or qualification gate.
- Recorded native costs describe this case, not a learned-model advantage or a confirmed adaptation benefit.
- Root estimates use only completed past transitions; update estimates are separately compared with the gain of the transition just observed.
- MOVE is fixed [100,150), the first complete goal interval after the switch; it is not a selected favorable interval.
- Known-physics planning and identification are charged; equal candidate budgets are not equal total compute.
- Shared-host wall times include instrumentation; nested times must not be added to enclosing times.
- This reviewer authenticates saved audit claims and recomputes arithmetic; it performs no native replay or inference and does not attest subprocess exits.

Outer completion SHA: `064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050`.
