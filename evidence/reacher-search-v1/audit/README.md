# Reacher adaptive-search saved-output audit

Primary search gate: **PASS** (8/8).

6 inherited fits; 66 control rows; 64 diagnostic roots; 544,928 native transitions checked.

| Panel | Arm | Mean native cost |
|---|---|---:|
| full | free-271-rs64 | 11.584336 |
| full | free-271-rs256 | 11.585418 |
| full | free-271-cem256 | 10.173851 |
| full | residual-271-rs64 | 11.210039 |
| full | residual-271-rs256 | 11.195551 |
| full | residual-271-cem256 | 7.981945 |
| full | residual-283-rs64 | 10.450696 |
| full | residual-283-rs256 | 10.402530 |
| full | residual-283-cem256 | 8.231077 |
| full | free-283-rs64 | 11.323092 |
| full | free-283-rs256 | 11.323223 |
| full | free-283-cem256 | 8.695079 |
| full | free-293-rs64 | 11.642625 |
| full | free-293-rs256 | 11.642740 |
| full | free-293-cem256 | 9.154482 |
| full | residual-293-rs64 | 11.642626 |
| full | residual-293-rs256 | 11.642585 |
| full | residual-293-cem256 | 8.504949 |
| full | known_state | 7.259323 |
| full | particle | 7.403523 |
| full | zero | 11.642618 |
| full | uniform | 42.437480 |
| ordinary | free-271-rs64 | 11.595801 |
| ordinary | free-271-rs256 | 11.596905 |
| ordinary | free-271-cem256 | 10.313631 |
| ordinary | residual-271-rs64 | 11.230326 |
| ordinary | residual-271-rs256 | 11.222442 |
| ordinary | residual-271-cem256 | 8.026593 |
| ordinary | residual-283-rs64 | 10.668850 |
| ordinary | residual-283-rs256 | 10.634733 |
| ordinary | residual-283-cem256 | 8.316879 |
| ordinary | free-283-rs64 | 11.382123 |
| ordinary | free-283-rs256 | 11.380550 |
| ordinary | free-283-cem256 | 8.804996 |
| ordinary | free-293-rs64 | 11.642625 |
| ordinary | free-293-rs256 | 11.642740 |
| ordinary | free-293-cem256 | 9.268433 |
| ordinary | residual-293-rs64 | 11.642626 |
| ordinary | residual-293-rs256 | 11.642585 |
| ordinary | residual-293-cem256 | 8.621930 |
| ordinary | known_state | 7.259323 |
| ordinary | particle | 7.450230 |
| ordinary | zero | 11.642618 |
| ordinary | uniform | 42.437480 |
| shift | free-271-rs64 | 11.601916 |
| shift | free-271-rs256 | 11.603026 |
| shift | free-271-cem256 | 10.402877 |
| shift | residual-271-rs64 | 11.184077 |
| shift | residual-271-rs256 | 11.184038 |
| shift | residual-271-cem256 | 8.067491 |
| shift | residual-283-rs64 | 10.775356 |
| shift | residual-283-rs256 | 10.744520 |
| shift | residual-283-cem256 | 8.342890 |
| shift | free-283-rs64 | 11.443972 |
| shift | free-283-rs256 | 11.441095 |
| shift | free-283-cem256 | 8.863866 |
| shift | free-293-rs64 | 11.642625 |
| shift | free-293-rs256 | 11.642740 |
| shift | free-293-cem256 | 9.378412 |
| shift | residual-293-rs64 | 11.642626 |
| shift | residual-293-rs256 | 11.642585 |
| shift | residual-293-cem256 | 8.680470 |
| shift | known_state | 7.259323 |
| shift | particle | 7.490604 |
| shift | zero | 11.642618 |
| shift | uniform | 42.437480 |

- All six final fits are inherited unchanged and unselected. No new training, prediction-cohort or memory/reset qualification is performed.
- The primary search gate measures actual native control cost. Control competence is separately reported for every planner. Historical 17 checks remain historical.
- Diagnostic regret is relative to the evaluated finite union, not a globally optimal controller. Duplicate identity slots are retained and unique sequences counted.
- Rank correlation excludes undefined constant-vector roots and reports coverage. Four noise branches and case bootstrap intervals are descriptive conditional on the fixed fits.
- Native replay checks stored actions and physical outcomes. Proposal reconstruction reads recorded model scores; it does not independently verify neural forward outputs.
- Raw and clipped predictions are retained separately. Supplied-physics references have model/state privileges; their scored values are not learned predictions.
- Wall time includes setup, proposal generation, scoring, state updates and storage. Counts describe scored candidates and imagined steps, not total FLOPs or single-agent deadlines.
- The prior cumulative cost already includes fitting. New evaluation cost is added once. Phase receipts establish source-bound logged chronology, not an independent observer.
