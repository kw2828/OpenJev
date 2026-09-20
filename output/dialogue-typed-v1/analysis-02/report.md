# Typed/support conditional development study

Official TRAIN service-held-out development data, historically exposed in prior fits. Gold previous value supplied. References have accuracy only. All twelve fits retained; no recurrent, calibration, novelty or fresh-confirmation claim.

Fixed mechanism screen: **FAIL**, 5/9 checks passed.
Technical completion and source/row/weight bindings passed for all twelve fits.

| Arm | Seed | Held-out changed equal-service NLL | TRUE recall | DONTCARE recall | Retained error |
|---|---:|---:|---:|---:|---:|
| flat_stratum | 6101 | 2.272072569878079 | 2/29 | 0/5 | 250/7241 |
| flat_stratum | 6102 | 2.060626118451783 | 2/29 | 0/5 | 395/7241 |
| flat_stratum | 6103 | 2.3846907079358415 | 1/29 | 0/5 | 243/7241 |
| flat_balanced | 6101 | 2.6175565533351066 | 3/29 | 0/5 | 201/7241 |
| flat_balanced | 6102 | 2.570636737389644 | 0/29 | 0/5 | 150/7241 |
| flat_balanced | 6103 | 2.409476983848998 | 1/29 | 1/5 | 123/7241 |
| typed_stratum | 6101 | 1.5539218793445144 | 3/29 | 0/5 | 322/7241 |
| typed_stratum | 6102 | 2.023521587621444 | 2/29 | 0/5 | 253/7241 |
| typed_stratum | 6103 | 1.9332988401366216 | 0/29 | 0/5 | 196/7241 |
| typed_balanced | 6101 | 2.194626788274188 | 1/29 | 0/5 | 174/7241 |
| typed_balanced | 6102 | 1.680111960459023 | 7/29 | 0/5 | 322/7241 |
| typed_balanced | 6103 | 1.677833684673681 | 0/29 | 0/5 | 269/7241 |

All panel, transition/value, service and dialogue means and factorial contrasts are in summary.json.
Reference outputs are accuracy-only. Three seeds are repeated optimizations, not independent service samples.
Runtime, normalization extrema and initializer digests are authenticated execution witnesses; no model or checkpoint replay was performed.

Whole-run wall time: 551.820233 seconds. Process-lifetime peak RSS: 1919205376 bytes.

| Fit | Training seconds | Evaluation seconds | Fit seconds |
|---|---:|---:|---:|
| flat_stratum-6101 | 45.134301 | 0.617528 | 45.755589 |
| flat_balanced-6101 | 44.682975 | 0.656772 | 45.343957 |
| typed_stratum-6101 | 44.555020 | 0.582414 | 45.144051 |
| typed_balanced-6101 | 44.964287 | 0.847858 | 45.817611 |
| flat_balanced-6102 | 45.000566 | 0.592229 | 45.598388 |
| typed_stratum-6102 | 45.460545 | 0.647997 | 46.112336 |
| typed_balanced-6102 | 44.912528 | 0.551815 | 45.468044 |
| flat_stratum-6102 | 44.253948 | 0.574735 | 44.832422 |
| typed_stratum-6103 | 45.158336 | 0.562863 | 45.726961 |
| typed_balanced-6103 | 45.225050 | 0.592060 | 45.822408 |
| flat_stratum-6103 | 44.426012 | 0.621375 | 45.051091 |
| flat_balanced-6103 | 44.289638 | 0.759721 | 45.053131 |

Training includes optimizer setup, training and checkpoint write; evaluation includes actor/model evaluation and prediction write; fit adds receipt preparation/payload hashing. Whole run also pays shared authentication, cache load and initialization. RSS is process-lifetime high-water, not a sum of fit peaks. Inherited preprocessing/preflight are separate costs, not included or claimed free.

This corrected saved reader validates all six split-projection fields. The original pre-scoring failure is preserved; no training or metric changes occurred.
