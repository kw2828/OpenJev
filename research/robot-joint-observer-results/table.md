# Joint observer development

**JOINT_OBSERVER_DEVELOPMENT_FAIL**, 4/5 conditions; all five required.

| Criterion | Passed |
|---|---|
| complete_forecasts_and_costs | True |
| equal_four_file_mean_5pct_vs_best_control | False |
| each_file_within_2pct_best_simple | True |
| latency_within_150pct_last_two | True |
| complete_frontier_not_dominated | True |

| Family | Rate | File 1 | File 2 | File 3 | File 4 | Equal-file mean | Request ms | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Joint observer long | 0.001 | 0.613356 | 0.677499 | 0.60359 | 0.681334 | 0.643945 | 5.78325 | 2888 |
| Joint observer short | 0.001 | 0.617327 | 0.674844 | 0.615214 | 0.68529 | 0.648169 | 4.44381 | 2888 |
| Continued last two | 0.001 | 0.623437 | 0.671105 | 0.621381 | 0.679332 | 0.648814 | 4.08763 | 2600 |
| Continued temporal | 0.001 | 0.616637 | 0.678464 | 0.610158 | 0.678768 | 0.646007 | 4.14331 | 4088 |
| Frozen position .003 | 0.003 | 0.61347 | 0.686064 | 0.61386 | 0.68475 | 0.649536 | 5.63792 | 2888 |
| Position fixed | None | 0.624302 | 0.692757 | 0.623255 | 0.70105 | 0.660341 | 5.69115 | 2888 |
| Frozen local affine | 0.003 | 0.60918 | 0.67141 | 0.617453 | 0.675793 | 0.643459 | 4.11954 | 4088 |
| Frozen temporal affine | 0.001 | 0.604369 | 0.679029 | 0.609073 | 0.674383 | 0.641713 | 4.13967 | 4088 |
| Failed prior observer | no complete parent recipe | diagnostic | diagnostic | diagnostic | diagnostic | ineligible | not retimed | see parent |
| Frozen last two | None | 0.618315 | 0.678468 | 0.617194 | 0.684764 | 0.649685 | 4.1109 | 2600 |
| Observer fixed | None | 5.4018e+07 | 3.40661e+08 | 3.17052e+08 | 7.48037e+06 | 1.79803e+08 | 5.9395 | 2888 |
| Observer zero | None | 0.661995 | 0.753771 | 0.656122 | 0.787687 | 0.714894 | 5.65765 | 2888 |
| Historical joint local | 0.003 | 0.622122 | 0.657327 | 0.591883 | 0.660706 | 0.63301 | 3.95573 | 4088 |
| Historical joint temporal | 0.003 | 0.576806 | 0.635978 | 0.610736 | 0.637046 | 0.615141 | 4.05146 | 4088 |
| Dense bounded | 0.001 | 0.636765 | 0.672438 | 0.622578 | 0.684829 | 0.654152 | 4.58831 | 3464 |
| Dense unbounded | 0.003 | 0.649673 | 0.69214 | 0.716145 | 0.753901 | 0.702965 | 4.54587 | 3464 |
| GRU32 | 0.003 | 0.742666 | 0.791812 | 0.716931 | 0.761932 | 0.753335 | 2.17967 | 24056 |
| Legacy instant | 0.001 | 0.625821 | 0.679592 | 0.628127 | 0.721249 | 0.663697 | 4.37244 | 4296 |
| GRU10 | 0.003 | 0.682249 | 0.72576 | 0.629438 | 0.736384 | 0.693458 | 2.2804 | 5488 |
| Causal ridge 1 | None | 0.744225 | 0.811347 | 0.753 | 0.835218 | 0.785947 | 0.582896 | 3565264 |
| Causal ridge 100 | None | 0.675447 | 0.740766 | 0.661038 | 0.725718 | 0.700742 | 0.599041 | 3565264 |
| Linear AR2 | None | 1.08777 | 1.08615 | 1.00941 | 1.0253 | 1.05216 | 0.266645 | 1536 |
| Persistence | None | 1.23294 | 1.34667 | 1.28446 | 1.25636 | 1.28011 | 0.00997947 | 240 |
| Frozen position .001 | 0.001 | 0.615415 | 0.687323 | 0.615899 | 0.685486 | 0.651031 | 5.67115 | 2888 |

All 584 score rows retain both horizons and every declared parent rate: 488 unchanged parent rows plus 96 fresh rows. All 292 attempts and 61 current costs remain visible. No new rate selection or parent reselection occurs. The frozen .001 alias refers to the existing observer_position .001 rows and checkpoint; score rows retain their original identities.

All four recordings are exposed development. Twelve fresh fits have per-attempt prefix and all-parameter gradient diagnostics, raw last-gradient banks and aggregate summaries under study/. The child has no diagnostic probes. Native clipping is unchanged and diagnostic float64 norms do not repair the optimizer.
