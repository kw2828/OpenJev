# OTTO direct-readout diagnostic

**REUSED_DEV_FAIL: ridge passed 4/6 paired cells.**

All five methods and three fit seeds use the same 36 previously exposed DEV paths. These paths were held out from training, but this is reused development evidence. Neither TEST nor confirmation is admitted. No autonomous performance, calibrated probabilities, statistical equivalence or architecture novelty is established.

![Every method, fit seed, measured adaptation cost and ridge check](methods.png)

Dots show all three fits and diamonds their equal means. Fits share evaluation paths; the dots are not independent evaluation samples or confidence intervals.

## All fifteen DEV views

| Method | Seed | lambda3 full | lambda3 later | lambda4 full | lambda4 later |
| --- | --- | --- | --- | --- | --- |
| pretrained | 309000001 | 0.0798382926 | 0.0913428781 | 0.0904198849 | 0.113973803 |
| pretrained | 309000002 | 0.0850218004 | 0.0950800996 | 0.0990128359 | 0.119042352 |
| pretrained | 309000003 | 0.092632142 | 0.103958783 | 0.106483161 | 0.130698902 |
| action_residual_only | 309000001 | 0.0754457507 | 0.093023653 | 0.0738535204 | 0.0910002579 |
| action_residual_only | 309000002 | 0.0939268719 | 0.10398258 | 0.0950616968 | 0.107156073 |
| action_residual_only | 309000003 | 0.0685513677 | 0.0754695084 | 0.100753136 | 0.119571799 |
| ols | 309000001 | 0.0735982413 | 0.0854984439 | 0.0896511792 | 0.11775147 |
| ols | 309000002 | 0.0798004287 | 0.0812124884 | 0.0892852147 | 0.108140486 |
| ols | 309000003 | 0.0621279302 | 0.0684855454 | 0.0842716324 | 0.102540289 |
| ridge | 309000001 | 0.0725876571 | 0.0843693353 | 0.0894845376 | 0.117529254 |
| ridge | 309000002 | 0.0775622497 | 0.0789056346 | 0.0896419197 | 0.108548486 |
| ridge | 309000003 | 0.0586615782 | 0.0642781837 | 0.0851942598 | 0.103434384 |
| full_joint | 309000001 | 0.104673406 | 0.104631234 | 0.114528844 | 0.141549415 |
| full_joint | 309000002 | 0.0968813451 | 0.0938126016 | 0.103331224 | 0.123883102 |
| full_joint | 309000003 | 0.0720600833 | 0.0824941296 | 0.0875546326 | 0.0990349934 |

Raw gap is teacher score of the selected legal action minus the minimum legal teacher score. Lower is better. Full includes all nonquery actions; later includes nonquery steps >=5. Cases and collectors retain their registered weighting, including zero-support cases.

## Every paired ridge comparison

| Seed | Setting | Supported later cases | Parent later/full | Adam74 later/full | Cell |
| --- | --- | --- | --- | --- | --- |
| 309000001 | lambda3 | 6/6 | PASS / PASS | PASS / PASS | PASS |
| 309000001 | lambda4 | 6/6 | FAIL / PASS | FAIL / FAIL | FAIL |
| 309000002 | lambda3 | 6/6 | PASS / PASS | PASS / PASS | PASS |
| 309000002 | lambda4 | 6/6 | PASS / PASS | FAIL / PASS | FAIL |
| 309000003 | lambda3 | 6/6 | PASS / PASS | PASS / PASS | PASS |
| 309000003 | lambda4 | 6/6 | PASS / PASS | PASS / PASS | PASS |

Each cell requires >=5% later-gap reduction, strict improvement, and no full-gap regression against both the parent and Adam74 residual head. OLS and full joint remain descriptive comparators; neither replaces the registered ridge candidate.

## All six adaptation times

| Seed | Solver | Extract s | Design s | Solve s | Export s | Validate s | Standalone total s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 309000001 | ols | 2.10247079 | 0.067647083 | 0.282059375 | 0.000775583 | 2.60181304 | 5.05476587 |
| 309000001 | ridge | 2.10247079 | 0.067647083 | 0.186825375 | 0.000934541 | 2.80356425 | 5.16144204 |
| 309000002 | ols | 2.17218425 | 0.031797291 | 0.185201 | 0.001107333 | 2.55245037 | 4.94274025 |
| 309000002 | ridge | 2.17218425 | 0.031797291 | 0.200053583 | 0.001128625 | 2.61762188 | 5.02278562 |
| 309000003 | ols | 2.00082096 | 0.036164041 | 0.159886375 | 0.000917333 | 2.61683033 | 4.81461904 |
| 309000003 | ridge | 2.00082096 | 0.036164041 | 0.176320583 | 0.000930208 | 2.57766454 | 4.79190033 |

Each standalone total charges the shared cache and design fully to that solver, so summing both totals double-counts shared work. Parent pretraining, historical Adam training, collection, DEV evaluation and independent audit are outside these totals. These are measured qualified CPU costs, not optimized inference-speed benchmarks.

## Cached versus ordinary TRAIN loss

| Seed | Method | Float64 cached data loss | Exported-head cached loss | Ordinary float32 output loss | Ordinary minus exported |
| --- | --- | --- | --- | --- | --- |
| 309000001 | pretrained | 2.42926735e-05 | 2.42926735e-05 | 2.42926761e-05 | 2.67002645e-12 |
| 309000001 | action_residual_only | 2.16236805e-05 | 2.16236805e-05 | 2.16236771e-05 | -3.37762626e-12 |
| 309000001 | ols | 2.0427352e-05 | 2.0427352e-05 | 2.04273534e-05 | 1.38703778e-12 |
| 309000001 | ridge | 2.04821452e-05 | 2.04821452e-05 | 2.04821414e-05 | -3.77916857e-12 |
| 309000002 | pretrained | 2.63627961e-05 | 2.63627961e-05 | 2.63627991e-05 | 3.03805487e-12 |
| 309000002 | action_residual_only | 2.3860197e-05 | 2.3860197e-05 | 2.38601971e-05 | 9.8802646e-14 |
| 309000002 | ols | 2.06035449e-05 | 2.06035449e-05 | 2.0603545e-05 | 1.22189273e-13 |
| 309000002 | ridge | 2.0691253e-05 | 2.0691253e-05 | 2.06912538e-05 | 8.21273689e-13 |
| 309000003 | pretrained | 2.71893596e-05 | 2.71893596e-05 | 2.71893576e-05 | -1.9607314e-12 |
| 309000003 | action_residual_only | 2.06900164e-05 | 2.06900164e-05 | 2.06900189e-05 | 2.59372665e-12 |
| 309000003 | ols | 1.93347717e-05 | 1.93347717e-05 | 1.9334772e-05 | 2.51453443e-13 |
| 309000003 | ridge | 1.94405372e-05 | 1.94405372e-05 | 1.94405371e-05 | -8.30313466e-14 |

Loss sums the episode-weighted legal nonquery centered MSE and all-action prequery centered MSE, divided by 64². Unsupported episodes remain in the denominator 54. Ridge regularization is excluded from this data-loss comparison. Float64 cache arithmetic and ordinary float32 outputs are distinct; all twelve TRAIN parity checks passed before DEV was opened.

OLS uses the retained numerical subspace at rcond=1e-10; ridge fixes lambda=1e-4 and penalizes all 87 contrast increment coordinates, including biases. The least-squares objective is not an action-gap optimum. No new solver or model claim is made.

## Authenticated evidence

Closure: [closure-01.json](<../../output/otto-direct-readout-v1/closure-01.json>) (`6b06c9e12ac348e5d0cdfbffd4e9a1bf86cc11aa108eacb856a60d8ba325bc87`).

- plan: [registration-01.json](<../../output/otto-direct-readout-v1/registration-01.json>) (`4a4bcd699da834bb6005ae10daf298beb194ac5d83f3e3ae6d66eaf6a7be1393`)
- producer receipt: [receipt.json](<../../output/otto-direct-readout-v1/training-01/receipt.json>) (`a44103cbb518a5c46e6e6787786da8b390810b7717fe41464fa31fc41d73521d`)
- producer terminal: [training-native-01.terminal.json](<../../output/otto-direct-readout-v1/training-native-01.terminal.json>) (`83d886c978dd808a869e27f7c89390eb3662df9d014b658a0744f60ae30469b1`)
- audit receipt: [receipt.json](<../../output/otto-direct-readout-v1/audit-01/receipt.json>) (`8d20ead711c3ea9dddcb0c8cdc0d3b22c327dd10078de4f709aa425cc78a6f83`)
- audit terminal: [audit-native-01.terminal.json](<../../output/otto-direct-readout-v1/audit-native-01.terminal.json>) (`3d3743d1007004ffeb5032494701da89e23c8370545f32429bd76e1e09f3d6d2`)
- audit: [audit.json](<../../output/otto-direct-readout-v1/audit-01/audit.json>) (`1b57c16e28da2f4f24927594e02f79fa623220589416080450d1055f09c567cf`)
- fit records: [fits.json](<../../output/otto-direct-readout-v1/training-01/fits.json>)

Both original process closures were authenticated before reporting. This renderer reads saved JSON and opaque hashes only: zero array/checkpoint decodes, model calls, solver calls, optimizer updates, teacher calls or simulator calls. Previous studies remain closed.
