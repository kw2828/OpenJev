# Training-parent error-order screen

Screen **FAIL**: 223/368 checks. All 81 rows, 45 fits and 864 held-out window predictions retained.

| Method | Test position RMSE (m) | Test rotation RMSE (rad) | Train position RMSE (m) | Train rotation RMSE (rad) |
|---|---:|---:|---:|---:|
| base | 0.009592749 | 0.01514202 | 0.009584049 | 0.01499249 |
| bias | 0.01735411 | 0.0168185 | 0.01709039 | 0.01588163 |
| ridge_summary | 0.008702953 | 0.01399143 | 0.007307471 | 0.01135197 |
| ridge_ordered | 0.00892108 | 0.01454177 | 0.006175765 | 0.009428716 |
| summary | 0.00896492 | 0.01460632 | 0.006825041 | 0.009516077 |
| recurrent | 0.009162216 | 0.0145173 | 0.008064734 | 0.01114783 |
| shuffled | 0.009254179 | 0.01491953 | 0.007866584 | 0.0114445 |
| noerror | 0.00926897 | 0.01489813 | 0.008268166 | 0.01210871 |
| error_shuffled | 0.009193838 | 0.01465644 | 0.007984069 | 0.01145132 |

Training-parent development screen only, not old external-panel qualification or new architecture efficacy.

Saved neural outputs and gradients/optimizer updates are source/test-bound, not independently replayed.

Orders/permutations are checked for exact legal membership and pairing; random seed generation is not rerun.

Ridge normal equations and saved predictions are verified without another fit or solve.

Token arithmetic uses explicit scale-dependent floating-point bounds; causal pre-assimilation provenance remains source/test-bound.

Windows, folds and seeds are correlated; 368 conjunctive checks are not independent significance tests.
