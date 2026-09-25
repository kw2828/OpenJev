# Fixed-checkpoint confirmation

**DO_NOT_CONFIRM_HISTORY_INITIALIZATION**, 3/5 conditions.

| Criterion | Passed |
|---|---|
| primary_recipes_complete | True |
| equal_file_mean_5pct_vs_both_locals | False |
| each_file_within_2pct_best_local | False |
| latency_within_125pct_last_two | True |
| complete_frontier_not_dominated | True |

| Family | Fixed rate | File 1 H128 RMSE | File 2 H128 RMSE | Equal-file mean | Request ms | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|
| Last two | 0.001 | 0.617194 | 0.684764 | 0.650979 | 4.89031 | 2600 |
| Local affine | 0.003 | 0.591883 | 0.660706 | 0.626295 | 4.40615 | 4088 |
| Temporal affine | 0.003 | 0.610736 | 0.637046 | 0.623891 | 5.67333 | 4088 |
| Dense bounded | 0.001 | 0.622578 | 0.684829 | 0.653703 | 6.20379 | 3464 |
| Dense unbounded | 0.003 | 0.716145 | 0.753901 | 0.735023 | 7.15998 | 3464 |
| GRU32 | 0.003 | 0.716931 | 0.761932 | 0.739432 | 2.93129 | 24056 |
| Legacy instant | 0.001 | 0.628127 | 0.721249 | 0.674688 | 5.89988 | 4296 |
| GRU10 | 0.003 | 0.629438 | 0.736384 | 0.682911 | 2.99715 | 5488 |
| Causal ridge 1 | - | 0.753 | 0.835218 | 0.794109 | 0.986 | 3565264 |
| Causal ridge 100 | - | 0.661038 | 0.725718 | 0.693378 | 0.912208 | 3565264 |
| Linear AR2 | - | 1.00941 | 1.0253 | 1.01735 | 0.413021 | 1536 |
| Persistence | - | 1.28446 | 1.25636 | 1.27041 | 0.00966608 | 240 |

All 112 H64/H128 score rows, all 56 forecast attempts and all 28 cost attempts remain in the evidence. Means and family costs above are saved audited scalars. Failed cases are retained. Training costs in models.json are historical; this confirmation performs zero fitting.
