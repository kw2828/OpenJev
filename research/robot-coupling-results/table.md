# Robot coupling saved results

Conditional offline prediction given future realized measured torque; 32 observed steps, 128 forecast steps (12.8 s). New causal per-recording preprocessing. Internal DEV after two-rate selection; not an official benchmark score. Dots show every selected fit; diamonds show arithmetic family means, not an ensemble. No CONFIRM or official TEST access. Direct-history ridge reads all future torques jointly; autoregressive models consume them stepwise. That ridge is a high-resource conditional reference, not a causally matched world model.

Saved screen outcome: **DO_NOT_ADVANCE_COUPLING**, 20/55 conditions.

## Every candidate rate and reference

Both rates and both reported horizons are retained below. Selection uses pooled DEV H128; these are screening results.

| Recording | Model | Rate | Seed | Horizon | Status | Standardized RMSE | Physical RMSE (deg) | Error |
|---|---|---:|---:|---:|---|---:|---:|---|
| 21H_54M | Chain memory | 0.0001 | 8101 | 64 | PASS | 0.677485 | 19.7905 |  |
| 21H_54M | Chain memory | 0.0001 | 8101 | 128 | PASS | 0.825206 | 24.6967 |  |
| 21H_54M | Rewired memory | 0.0001 | 8101 | 64 | PASS | 0.680952 | 19.8483 |  |
| 21H_54M | Rewired memory | 0.0001 | 8101 | 128 | PASS | 0.828286 | 24.7479 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8101 | 64 | PASS | 0.680302 | 19.8322 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8101 | 128 | PASS | 0.825263 | 24.6175 |  |
| 21H_54M | GRU residual | 0.0001 | 8101 | 64 | PASS | 0.66061 | 19.2203 |  |
| 21H_54M | GRU residual | 0.0001 | 8101 | 128 | PASS | 0.823788 | 24.6042 |  |
| 21H_54M | Quadratic AR2 | 0.0001 | 8101 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.0001 | 8101 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Chain memory | 0.001 | 8101 | 64 | PASS | 0.674271 | 21.2674 |  |
| 21H_54M | Chain memory | 0.001 | 8101 | 128 | PASS | 0.781552 | 25.0614 |  |
| 21H_54M | Rewired memory | 0.001 | 8101 | 64 | PASS | 0.672798 | 21.1192 |  |
| 21H_54M | Rewired memory | 0.001 | 8101 | 128 | PASS | 0.778401 | 24.902 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8101 | 64 | PASS | 0.709933 | 22.1428 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8101 | 128 | PASS | 0.844804 | 27.0843 |  |
| 21H_54M | GRU residual | 0.001 | 8101 | 64 | PASS | 0.66475 | 19.1879 |  |
| 21H_54M | GRU residual | 0.001 | 8101 | 128 | PASS | 0.837066 | 24.7298 |  |
| 21H_54M | Quadratic AR2 | 0.001 | 8101 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.001 | 8101 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Chain memory | 0.0001 | 8102 | 64 | PASS | 0.661532 | 19.0514 |  |
| 21H_54M | Chain memory | 0.0001 | 8102 | 128 | PASS | 0.797375 | 23.3434 |  |
| 21H_54M | Rewired memory | 0.0001 | 8102 | 64 | PASS | 0.661288 | 19.0527 |  |
| 21H_54M | Rewired memory | 0.0001 | 8102 | 128 | PASS | 0.797709 | 23.3556 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8102 | 64 | PASS | 0.664613 | 19.1364 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8102 | 128 | PASS | 0.799013 | 23.3428 |  |
| 21H_54M | GRU residual | 0.0001 | 8102 | 64 | PASS | 0.643636 | 18.055 |  |
| 21H_54M | GRU residual | 0.0001 | 8102 | 128 | PASS | 0.782763 | 22.373 |  |
| 21H_54M | Quadratic AR2 | 0.0001 | 8102 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.0001 | 8102 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Chain memory | 0.001 | 8102 | 64 | PASS | 0.660161 | 20.0317 |  |
| 21H_54M | Chain memory | 0.001 | 8102 | 128 | PASS | 0.73783 | 22.2646 |  |
| 21H_54M | Rewired memory | 0.001 | 8102 | 64 | PASS | 0.657016 | 19.9171 |  |
| 21H_54M | Rewired memory | 0.001 | 8102 | 128 | PASS | 0.734324 | 22.133 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8102 | 64 | PASS | 0.686521 | 20.3609 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8102 | 128 | PASS | 0.775828 | 22.9946 |  |
| 21H_54M | GRU residual | 0.001 | 8102 | 64 | PASS | 0.656444 | 19.0995 |  |
| 21H_54M | GRU residual | 0.001 | 8102 | 128 | PASS | 0.716918 | 21.1967 |  |
| 21H_54M | Quadratic AR2 | 0.001 | 8102 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.001 | 8102 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Chain memory | 0.0001 | 8103 | 64 | PASS | 0.672264 | 19.6878 |  |
| 21H_54M | Chain memory | 0.0001 | 8103 | 128 | PASS | 0.819502 | 24.4204 |  |
| 21H_54M | Rewired memory | 0.0001 | 8103 | 64 | PASS | 0.674191 | 19.7516 |  |
| 21H_54M | Rewired memory | 0.0001 | 8103 | 128 | PASS | 0.820839 | 24.4505 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8103 | 64 | PASS | 0.673638 | 19.7418 |  |
| 21H_54M | Chain instantaneous | 0.0001 | 8103 | 128 | PASS | 0.816787 | 24.3046 |  |
| 21H_54M | GRU residual | 0.0001 | 8103 | 64 | PASS | 0.656155 | 18.9512 |  |
| 21H_54M | GRU residual | 0.0001 | 8103 | 128 | PASS | 0.806555 | 24.0581 |  |
| 21H_54M | Quadratic AR2 | 0.0001 | 8103 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.0001 | 8103 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Chain memory | 0.001 | 8103 | 64 | PASS | 0.709202 | 21.8069 |  |
| 21H_54M | Chain memory | 0.001 | 8103 | 128 | PASS | 0.832679 | 25.5361 |  |
| 21H_54M | Rewired memory | 0.001 | 8103 | 64 | PASS | 0.70697 | 21.8611 |  |
| 21H_54M | Rewired memory | 0.001 | 8103 | 128 | PASS | 0.830693 | 25.7093 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8103 | 64 | PASS | 0.750183 | 21.9379 |  |
| 21H_54M | Chain instantaneous | 0.001 | 8103 | 128 | PASS | 0.888493 | 26.5828 |  |
| 21H_54M | GRU residual | 0.001 | 8103 | 64 | PASS | 0.666439 | 20.2531 |  |
| 21H_54M | GRU residual | 0.001 | 8103 | 128 | PASS | 0.761336 | 23.3822 |  |
| 21H_54M | Quadratic AR2 | 0.001 | 8103 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Quadratic AR2 | 0.001 | 8103 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 21H_54M | Frozen linear AR2 | n/a | None | 64 | PASS | 0.730026 | 21.2211 |  |
| 21H_54M | Frozen linear AR2 | n/a | None | 128 | PASS | 1.08777 | 30.8077 |  |
| 21H_54M | Frozen quadratic AR2 | n/a | None | 64 | FAILED | n/a | n/a | {'type': 'NonfiniteReference'} |
| 21H_54M | Frozen quadratic AR2 | n/a | None | 128 | FAILED | n/a | n/a | {'type': 'NonfiniteReference'} |
| 21H_54M | Persistence | n/a | None | 64 | PASS | 1.13736 | 32.6754 |  |
| 21H_54M | Persistence | n/a | None | 128 | PASS | 1.23294 | 35.4162 |  |
| 21H_54M | Constant velocity | n/a | None | 64 | PASS | 2.0169 | 58.1456 |  |
| 21H_54M | Constant velocity | n/a | None | 128 | PASS | 3.53034 | 103.584 |  |
| 21H_54M | Direct ridge (1) | n/a | None | 64 | PASS | 0.560843 | 16.214 |  |
| 21H_54M | Direct ridge (1) | n/a | None | 128 | PASS | 0.710309 | 21.1847 |  |
| 21H_54M | Direct ridge (100) | n/a | None | 64 | PASS | 0.504435 | 15.3211 |  |
| 21H_54M | Direct ridge (100) | n/a | None | 128 | PASS | 0.595124 | 18.1417 |  |
| 22H_10M | Chain memory | 0.0001 | 8101 | 64 | PASS | 0.722668 | 21.7493 |  |
| 22H_10M | Chain memory | 0.0001 | 8101 | 128 | PASS | 0.883114 | 27.2672 |  |
| 22H_10M | Rewired memory | 0.0001 | 8101 | 64 | PASS | 0.72267 | 21.7006 |  |
| 22H_10M | Rewired memory | 0.0001 | 8101 | 128 | PASS | 0.88235 | 27.2078 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8101 | 64 | PASS | 0.723721 | 21.7566 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8101 | 128 | PASS | 0.881425 | 27.1761 |  |
| 22H_10M | GRU residual | 0.0001 | 8101 | 64 | PASS | 0.702499 | 21.1736 |  |
| 22H_10M | GRU residual | 0.0001 | 8101 | 128 | PASS | 0.84961 | 26.1407 |  |
| 22H_10M | Quadratic AR2 | 0.0001 | 8101 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.0001 | 8101 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Chain memory | 0.001 | 8101 | 64 | PASS | 0.760154 | 24.204 |  |
| 22H_10M | Chain memory | 0.001 | 8101 | 128 | PASS | 0.905551 | 29.5695 |  |
| 22H_10M | Rewired memory | 0.001 | 8101 | 64 | PASS | 0.755545 | 24.006 |  |
| 22H_10M | Rewired memory | 0.001 | 8101 | 128 | PASS | 0.890992 | 29.0998 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8101 | 64 | PASS | 0.767205 | 23.9048 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8101 | 128 | PASS | 0.893627 | 28.9572 |  |
| 22H_10M | GRU residual | 0.001 | 8101 | 64 | PASS | 0.756479 | 23.0935 |  |
| 22H_10M | GRU residual | 0.001 | 8101 | 128 | PASS | 0.908798 | 29.2034 |  |
| 22H_10M | Quadratic AR2 | 0.001 | 8101 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.001 | 8101 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Chain memory | 0.0001 | 8102 | 64 | PASS | 0.693348 | 20.6678 |  |
| 22H_10M | Chain memory | 0.0001 | 8102 | 128 | PASS | 0.838459 | 25.3632 |  |
| 22H_10M | Rewired memory | 0.0001 | 8102 | 64 | PASS | 0.69405 | 20.642 |  |
| 22H_10M | Rewired memory | 0.0001 | 8102 | 128 | PASS | 0.838427 | 25.3542 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8102 | 64 | PASS | 0.697454 | 20.784 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8102 | 128 | PASS | 0.840623 | 25.4281 |  |
| 22H_10M | GRU residual | 0.0001 | 8102 | 64 | PASS | 0.67518 | 19.5582 |  |
| 22H_10M | GRU residual | 0.0001 | 8102 | 128 | PASS | 0.828301 | 24.2153 |  |
| 22H_10M | Quadratic AR2 | 0.0001 | 8102 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.0001 | 8102 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Chain memory | 0.001 | 8102 | 64 | PASS | 0.746007 | 23.2854 |  |
| 22H_10M | Chain memory | 0.001 | 8102 | 128 | PASS | 0.806069 | 25.361 |  |
| 22H_10M | Rewired memory | 0.001 | 8102 | 64 | PASS | 0.745284 | 23.2873 |  |
| 22H_10M | Rewired memory | 0.001 | 8102 | 128 | PASS | 0.808606 | 25.4944 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8102 | 64 | PASS | 0.773853 | 23.589 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8102 | 128 | PASS | 0.858362 | 26.5367 |  |
| 22H_10M | GRU residual | 0.001 | 8102 | 64 | PASS | 0.703241 | 21.2054 |  |
| 22H_10M | GRU residual | 0.001 | 8102 | 128 | PASS | 0.763562 | 22.3906 |  |
| 22H_10M | Quadratic AR2 | 0.001 | 8102 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.001 | 8102 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Chain memory | 0.0001 | 8103 | 64 | PASS | 0.712905 | 20.9249 |  |
| 22H_10M | Chain memory | 0.0001 | 8103 | 128 | PASS | 0.838463 | 25.0568 |  |
| 22H_10M | Rewired memory | 0.0001 | 8103 | 64 | PASS | 0.71342 | 20.9347 |  |
| 22H_10M | Rewired memory | 0.0001 | 8103 | 128 | PASS | 0.837895 | 25.0205 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8103 | 64 | PASS | 0.712484 | 20.9424 |  |
| 22H_10M | Chain instantaneous | 0.0001 | 8103 | 128 | PASS | 0.837358 | 25.0322 |  |
| 22H_10M | GRU residual | 0.0001 | 8103 | 64 | PASS | 0.70042 | 19.9548 |  |
| 22H_10M | GRU residual | 0.0001 | 8103 | 128 | PASS | 0.855672 | 24.703 |  |
| 22H_10M | Quadratic AR2 | 0.0001 | 8103 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.0001 | 8103 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Chain memory | 0.001 | 8103 | 64 | PASS | 0.762981 | 23.557 |  |
| 22H_10M | Chain memory | 0.001 | 8103 | 128 | PASS | 0.847084 | 25.9795 |  |
| 22H_10M | Rewired memory | 0.001 | 8103 | 64 | PASS | 0.762458 | 23.5635 |  |
| 22H_10M | Rewired memory | 0.001 | 8103 | 128 | PASS | 0.843885 | 25.9597 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8103 | 64 | PASS | 0.792561 | 23.4711 |  |
| 22H_10M | Chain instantaneous | 0.001 | 8103 | 128 | PASS | 0.903901 | 26.729 |  |
| 22H_10M | GRU residual | 0.001 | 8103 | 64 | PASS | 0.723216 | 21.5246 |  |
| 22H_10M | GRU residual | 0.001 | 8103 | 128 | PASS | 0.806074 | 24.4611 |  |
| 22H_10M | Quadratic AR2 | 0.001 | 8103 | 64 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Quadratic AR2 | 0.001 | 8103 | 128 | FAILED | n/a | n/a | {'message': 'not a completed trained model', 'type': 'FailedTrainingAttempt'} |
| 22H_10M | Frozen linear AR2 | n/a | None | 64 | PASS | 0.772146 | 22.3071 |  |
| 22H_10M | Frozen linear AR2 | n/a | None | 128 | PASS | 1.08615 | 30.8539 |  |
| 22H_10M | Frozen quadratic AR2 | n/a | None | 64 | FAILED | n/a | n/a | {'type': 'NonfiniteReference'} |
| 22H_10M | Frozen quadratic AR2 | n/a | None | 128 | FAILED | n/a | n/a | {'type': 'NonfiniteReference'} |
| 22H_10M | Persistence | n/a | None | 64 | PASS | 1.26554 | 35.9032 |  |
| 22H_10M | Persistence | n/a | None | 128 | PASS | 1.34667 | 38.3842 |  |
| 22H_10M | Constant velocity | n/a | None | 64 | PASS | 2.29293 | 64.9265 |  |
| 22H_10M | Constant velocity | n/a | None | 128 | PASS | 4.09268 | 118.118 |  |
| 22H_10M | Direct ridge (1) | n/a | None | 64 | PASS | 0.60865 | 16.7951 |  |
| 22H_10M | Direct ridge (1) | n/a | None | 128 | PASS | 0.770213 | 22.8406 |  |
| 22H_10M | Direct ridge (100) | n/a | None | 64 | PASS | 0.509184 | 15.2417 |  |
| 22H_10M | Direct ridge (100) | n/a | None | 128 | PASS | 0.641891 | 20.043 |  |

## Selection

| Family | Selected rate / ridge | Candidate eligibility |
|---|---|---|
| Chain memory | 0.001 | 0.0001: eligible; 0.001: eligible |
| Rewired memory | 0.001 | 0.0001: eligible; 0.001: eligible |
| Chain instantaneous | 0.0001 | 0.0001: eligible; 0.001: eligible |
| GRU residual | 0.001 | 0.0001: eligible; 0.001: eligible |
| Quadratic AR2 | n/a | 0.0001: INELIGIBLE; 0.001: INELIGIBLE |
| Direct ridge | direct_ridge_100 | [{'arm': 'direct_ridge_1', 'eligible': True, 'pooled_dev_h128_rmse': 0.7408663143923557}, {'arm': 'direct_ridge_100', 'eligible': True, 'pooled_dev_h128_rmse': 0.6189495092448053}] |

## Costs

All trained attempts are shown; request timing exists only for selected eligible recipes. Missing timing is not zero.

| Model | Rate | Seed | Fit status | Weights (B) | State (B) | Buffers (B) | Normalizer (B) | Request median (ms) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Chain memory | 0.0001 | 8101 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Rewired memory | 0.0001 | 8101 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Chain instantaneous | 0.0001 | 8101 | PASS | 1064 | 72 | 160 | 192 | 3.72594 |
| GRU residual | 0.0001 | 8101 | PASS | 5184 | 112 | 0 | 192 | n/a |
| Quadratic AR2 | 0.0001 | 8101 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Chain memory | 0.001 | 8101 | PASS | 1064 | 112 | 160 | 192 | 4.44617 |
| Rewired memory | 0.001 | 8101 | PASS | 1064 | 112 | 160 | 192 | 4.38235 |
| Chain instantaneous | 0.001 | 8101 | PASS | 1064 | 72 | 160 | 192 | n/a |
| GRU residual | 0.001 | 8101 | PASS | 5184 | 112 | 0 | 192 | 1.99277 |
| Quadratic AR2 | 0.001 | 8101 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Chain memory | 0.0001 | 8102 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Rewired memory | 0.0001 | 8102 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Chain instantaneous | 0.0001 | 8102 | PASS | 1064 | 72 | 160 | 192 | 3.63744 |
| GRU residual | 0.0001 | 8102 | PASS | 5184 | 112 | 0 | 192 | n/a |
| Quadratic AR2 | 0.0001 | 8102 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Chain memory | 0.001 | 8102 | PASS | 1064 | 112 | 160 | 192 | 4.42152 |
| Rewired memory | 0.001 | 8102 | PASS | 1064 | 112 | 160 | 192 | 4.39808 |
| Chain instantaneous | 0.001 | 8102 | PASS | 1064 | 72 | 160 | 192 | n/a |
| GRU residual | 0.001 | 8102 | PASS | 5184 | 112 | 0 | 192 | 1.98527 |
| Quadratic AR2 | 0.001 | 8102 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Chain memory | 0.0001 | 8103 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Rewired memory | 0.0001 | 8103 | PASS | 1064 | 112 | 160 | 192 | n/a |
| Chain instantaneous | 0.0001 | 8103 | PASS | 1064 | 72 | 160 | 192 | 3.72608 |
| GRU residual | 0.0001 | 8103 | PASS | 5184 | 112 | 0 | 192 | n/a |
| Quadratic AR2 | 0.0001 | 8103 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Chain memory | 0.001 | 8103 | PASS | 1064 | 112 | 160 | 192 | 4.34902 |
| Rewired memory | 0.001 | 8103 | PASS | 1064 | 112 | 160 | 192 | 4.44829 |
| Chain instantaneous | 0.001 | 8103 | PASS | 1064 | 72 | 160 | 192 | n/a |
| GRU residual | 0.001 | 8103 | PASS | 5184 | 112 | 0 | 192 | 1.99885 |
| Quadratic AR2 | 0.001 | 8103 | FAILED | 7800 | 72 | 0 | 192 | n/a |
| Frozen linear AR2 | n/a | None | reference | 1200 | 144 | 0 | 192 | 0.243375 |
| Frozen quadratic AR2 | n/a | None | reference | 15600 | 144 | 0 | 192 | n/a |
| Persistence | n/a | None | reference | 0 | 48 | 0 | 192 | 0.00891648 |
| Constant velocity | n/a | None | reference | 0 | 240 | 0 | 192 | 0.0142499 |
| Direct ridge (100) | n/a | None | reference | 5904384 | 1536 | 0 | 192 | 0.0243335 |

Storage covers declared weights, explicit recurrent/history state, buffers and normalizers. Request inputs/outputs are reported in plotted-values.json; temporary workspace is not measured. Models differ in size and precision, so this is not an equal-memory or hardware-independent speed claim.
