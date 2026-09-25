# Frozen-backbone observer development

**OBSERVER_DEVELOPMENT_FAIL**, 0/5 conditions.

| Criterion | Passed |
|---|---|
| complete_forecasts_and_costs | False |
| equal_four_file_mean_5pct_vs_best_control | False |
| each_file_within_2pct_best_simple | False |
| latency_within_150pct_last_two | False |
| complete_frontier_not_dominated | False |

| Family | Selected/fixed rate | File 1 RMSE | File 2 RMSE | File 3 RMSE | File 4 RMSE | Equal-file mean | Request ms | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Local affine | 0.003 | 0.60918 | 0.67141 | 0.617453 | 0.675793 | 0.643459 | 4.00575 | 4088 |
| Temporal affine | 0.001 | 0.604369 | 0.679029 | 0.609073 | 0.674383 | 0.641713 | 4.06094 | 4088 |
| Observer learned | None | FAILED / incomplete | FAILED / incomplete | FAILED / incomplete | FAILED / incomplete | FAILED / incomplete | FAILED / incomplete | incomplete |
| Last two | - | 0.618315 | 0.678468 | 0.617194 | 0.684764 | 0.649685 | 3.96379 | 2600 |
| Observer fixed | - | 5.4018e+07 | 3.40661e+08 | 3.17052e+08 | 7.48037e+06 | 1.79803e+08 | 5.6144 | 2888 |
| Observer zero | - | 0.661995 | 0.753771 | 0.656122 | 0.787687 | 0.714894 | 5.52235 | 2888 |
| Joint local | 0.003 | 0.622122 | 0.657327 | 0.591883 | 0.660706 | 0.63301 | 3.99235 | 4088 |
| Joint temporal | 0.003 | 0.576806 | 0.635978 | 0.610736 | 0.637046 | 0.615141 | 3.98775 | 4088 |
| Dense bounded | 0.001 | 0.636765 | 0.672438 | 0.622578 | 0.684829 | 0.654152 | 4.56485 | 3464 |
| Dense unbounded | 0.003 | 0.649673 | 0.69214 | 0.716145 | 0.753901 | 0.702965 | 4.53777 | 3464 |
| GRU32 | 0.003 | 0.742666 | 0.791812 | 0.716931 | 0.761932 | 0.753335 | 2.20437 | 24056 |
| Legacy instant | 0.001 | 0.625821 | 0.679592 | 0.628127 | 0.721249 | 0.663697 | 4.34956 | 4296 |
| GRU10 | 0.003 | 0.682249 | 0.72576 | 0.629438 | 0.736384 | 0.693458 | 2.13025 | 5488 |
| Causal ridge 1 | - | 0.744225 | 0.811347 | 0.753 | 0.835218 | 0.785947 | 0.556459 | 3565264 |
| Causal ridge 100 | - | 0.675447 | 0.740766 | 0.661038 | 0.725718 | 0.700742 | 0.53825 | 3565264 |
| Linear AR2 | - | 1.08777 | 1.08615 | 1.00941 | 1.0253 | 1.05216 | 0.239396 | 1536 |
| Persistence | - | 1.23294 | 1.34667 | 1.28446 | 1.25636 | 1.28011 | 0.00920903 | 240 |

All 416 H64/H128 rows (both learned rates), 208 forecast attempts and 43 cost attempts are retained. Table means and family costs are saved audited scalars. Rates use only the original two DEV files. All four files are exposed development data. Frozen backbone and inherited training costs remain historical. New training is limited to the affine head or observer gain; no model is declared novel by this result.
