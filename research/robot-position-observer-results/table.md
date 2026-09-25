# Position-only observer development

**POSITION_OBSERVER_DEVELOPMENT_FAIL**, 3/5 conditions; all five required.

| Criterion | Passed |
|---|---|
| complete_forecasts_and_costs | True |
| equal_four_file_mean_5pct_vs_best_control | False |
| each_file_within_2pct_best_simple | False |
| latency_within_150pct_last_two | True |
| complete_frontier_not_dominated | True |

| Family | Rate | File 1 | File 2 | File 3 | File 4 | Equal-file mean | Request ms | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Position learned | 0.003 | 0.61347 | 0.686064 | 0.61386 | 0.68475 | 0.649536 | 5.29515 | 2888 |
| Position fixed | None | 0.624302 | 0.692757 | 0.623255 | 0.70105 | 0.660341 | 5.33894 | 2888 |
| Local affine | 0.003 | 0.60918 | 0.67141 | 0.617453 | 0.675793 | 0.643459 | 3.96394 | 4088 |
| Temporal affine | 0.001 | 0.604369 | 0.679029 | 0.609073 | 0.674383 | 0.641713 | 3.84273 | 4088 |
| Prior observer learned | no complete parent recipe | diagnostic | diagnostic | diagnostic | diagnostic | ineligible | not retimed | see parent |
| Last two | None | 0.618315 | 0.678468 | 0.617194 | 0.684764 | 0.649685 | 3.87163 | 2600 |
| Observer fixed | None | 5.4018e+07 | 3.40661e+08 | 3.17052e+08 | 7.48037e+06 | 1.79803e+08 | 5.37762 | 2888 |
| Observer zero | None | 0.661995 | 0.753771 | 0.656122 | 0.787687 | 0.714894 | 5.17985 | 2888 |
| Joint local | 0.003 | 0.622122 | 0.657327 | 0.591883 | 0.660706 | 0.63301 | 3.829 | 4088 |
| Joint temporal | 0.003 | 0.576806 | 0.635978 | 0.610736 | 0.637046 | 0.615141 | 3.84469 | 4088 |
| Dense bounded | 0.001 | 0.636765 | 0.672438 | 0.622578 | 0.684829 | 0.654152 | 4.378 | 3464 |
| Dense unbounded | 0.003 | 0.649673 | 0.69214 | 0.716145 | 0.753901 | 0.702965 | 4.33623 | 3464 |
| GRU32 | 0.003 | 0.742666 | 0.791812 | 0.716931 | 0.761932 | 0.753335 | 2.10294 | 24056 |
| Legacy instant | 0.001 | 0.625821 | 0.679592 | 0.628127 | 0.721249 | 0.663697 | 4.18283 | 4296 |
| GRU10 | 0.003 | 0.682249 | 0.72576 | 0.629438 | 0.736384 | 0.693458 | 2.08633 | 5488 |
| Causal ridge 1 | None | 0.744225 | 0.811347 | 0.753 | 0.835218 | 0.785947 | 0.520937 | 3565264 |
| Causal ridge 100 | None | 0.675447 | 0.740766 | 0.661038 | 0.725718 | 0.700742 | 0.528125 | 3565264 |
| Linear AR2 | None | 1.08777 | 1.08615 | 1.00941 | 1.0253 | 1.05216 | 0.239646 | 1536 |
| Persistence | None | 1.23294 | 1.34667 | 1.28446 | 1.25636 | 1.28011 | 0.00906258 | 240 |

All 488 score rows retain both horizons and every declared rate: 416 unchanged parent rows plus 72 new rows. All 244 attempts and 46 current costs remain visible. No parent recipe is reselected. Only the new learned gain uses the two original DEV files for rate selection; all four files are exposed development.

Seven FIT-only no-update probes and six per-fit diagnostic bundles are retained under study/. Finite gradient entries and a nonfinite native clipping norm are distinct outcomes. Diagnostics do not establish a learning-rate effect before any optimizer update.
