# Reacher inherited-model evaluation

Continuation gate: **FAIL** (9/17 checks).

6 unchanged inherited fits; 0 new fits; 177,600 native transitions replayed.

Inherited fitting: 209.151973 s. Prior invalid attempt including fitting: 359.138964 s. New evaluation: 247.419424 s.

| Panel | Arm | Mean episode cost |
|---|---|---:|
| full | free-271 | 11.543374 |
| full | free-283 | 11.533182 |
| full | free-293 | 11.725332 |
| full | residual-271 | 11.246180 |
| full | residual-283 | 10.659529 |
| full | residual-293 | 11.716704 |
| full | known_state | 7.505277 |
| full | particle | 7.664983 |
| full | zero | 11.725279 |
| full | uniform | 43.056249 |
| ordinary | free-271 | 11.570778 |
| ordinary | free-283 | 11.569238 |
| ordinary | free-293 | 11.725332 |
| ordinary | residual-271 | 11.284419 |
| ordinary | residual-283 | 10.788644 |
| ordinary | residual-293 | 11.724707 |
| ordinary | free-271-reset | 11.395031 |
| ordinary | free-283-reset | 11.584336 |
| ordinary | free-293-reset | 11.725332 |
| ordinary | residual-271-reset | 11.100172 |
| ordinary | residual-283-reset | 10.941544 |
| ordinary | residual-293-reset | 11.649545 |
| ordinary | known_state | 7.505277 |
| ordinary | particle | 7.674705 |
| ordinary | zero | 11.725279 |
| ordinary | uniform | 43.056249 |
| shift | free-271 | 11.540547 |
| shift | free-283 | 11.600225 |
| shift | free-293 | 11.725332 |
| shift | residual-271 | 11.266660 |
| shift | residual-283 | 10.953008 |
| shift | residual-293 | 11.700722 |
| shift | free-271-reset | 11.469327 |
| shift | free-283-reset | 11.635351 |
| shift | free-293-reset | 11.725332 |
| shift | residual-271-reset | 11.275734 |
| shift | residual-283-reset | 11.075855 |
| shift | residual-293-reset | 11.642150 |
| shift | known_state | 7.505277 |
| shift | particle | 7.727983 |
| shift | zero | 11.725279 |
| shift | uniform | 43.056249 |

- All final fits are inherited byte-for-byte without selection or new optimization. The original attempt remains invalid and stopped; its partial evaluation is not a result of this study.
- Fresh concrete seeds and generator initial states are checked across every role and against both prior plans. Controller/panel pairing intentionally reuses named streams; no samples are drawn to build the manifest.
- Native replay checks all inherited training and fresh evaluation transitions; saved prediction/control arithmetic is recomputed without learned-model, policy or MPC calls. Recorded training checks do not rerun optimization.
- The treatment uses supplied actuator-cost and noise-distribution knowledge. Privileged raw state and realized actuator noise remain audit-only for learned models.
- Inherited-fit and new-evaluation costs are separate. Prior invalid-attempt cost already contains the inherited fits. Phase receipts provide source-bound logged chronology, not an independent process observer.
- All paired seeds and ordinary/shift panels retain the parent criteria. Reset interventions and case-paired intervals are descriptive; intervals are conditional on inherited fits, not architecture uncertainty.
- Setup and batch decision times are charged; per-case latency is amortized, not a deadline. The execution cap is cooperative. Physics references retain their supplied-model privilege; floor scores are placeholders.
- This is a known-reward-structure qualification, not a new RL algorithm, memory advantage, connectome result or ICLR claim.
