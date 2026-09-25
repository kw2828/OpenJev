# Complete saved-scalar tables

All values below come from the closed independent audit and saved timing records. No models were rerun.

## Every record

| Record | Old 16 | Old 64 | New 16 | New 64 | Candidate seed mean | Fixed strongest control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100mV-realization-3-period-0 | 0.042680784 | 0.042681215 | 0.117282115 | 0.117282749 | 0.038673968 | 0.056435983 |
| 100mV-realization-3-period-1 | 0.041590318 | 0.041590396 | 0.117080428 | 0.117079232 | 0.036555986 | 0.053935844 |
| 100mV-realization-4-period-0 | 0.040545045 | 0.040545561 | 0.098556342 | 0.098551395 | 0.038617582 | 0.055067676 |
| 100mV-realization-4-period-1 | 0.039879664 | 0.039880031 | 0.115777548 | 0.115785900 | 0.037412954 | 0.053248762 |
| 100mV-realization-5-period-0 | 0.040104193 | 0.040105958 | 0.122118118 | 0.122139257 | 0.036655671 | 0.053349424 |
| 100mV-realization-5-period-1 | 0.039820607 | 0.039822368 | 0.120248628 | 0.120239599 | 0.035655765 | 0.051810157 |
| 200mV-realization-3-period-0 | 0.060992156 | 0.060992337 | 0.156767317 | 0.156765133 | 0.047904905 | 0.053183623 |
| 200mV-realization-3-period-1 | 0.061440100 | 0.061440043 | 0.158357686 | 0.158367470 | 0.048459812 | 0.054753290 |
| 200mV-realization-4-period-0 | 0.062200304 | 0.062200544 | 0.132967092 | 0.132918389 | 0.047396215 | 0.056128669 |
| 200mV-realization-4-period-1 | 0.063517159 | 0.063517790 | 0.132667527 | 0.132665220 | 0.048577440 | 0.058089840 |
| 200mV-realization-5-period-0 | 0.061681899 | 0.061682735 | 0.213585585 | 0.213581793 | 0.048763070 | 0.057266282 |
| 200mV-realization-5-period-1 | 0.063040554 | 0.063040310 | 0.213056953 | 0.213054937 | 0.050140626 | 0.059397323 |

## All eligible families

The candidate is included for comparison; the other 17 families are controls. The two old capped cells remain diagnostic.

| Family | Equal-record / equal-seed RMSE |
| --- | ---: |
| `tanh_feedback-lr0.0003` | 0.042901166 |
| `tanh_output_only-lr0.001` | 0.055222239 |
| `varx96-ridge1e-06` | 0.058295792 |
| `varx64-ridge1e-06` | 0.059372292 |
| `varx96-ridge0.001` | 0.060517967 |
| `varx64-ridge0.001` | 0.062152680 |
| `affine_output_only-lr0.0001` | 0.063955487 |
| `folded_affine_feedback` | 0.064588272 |
| `affine_feedback-lr0.0001` | 0.064588272 |
| `native_varx` | 0.064629567 |
| `varx32-ridge1e-06` | 0.064629567 |
| `varx32-ridge0.001` | 0.070323106 |
| `author_bla28` | 0.095731482 |
| `varx96-ridge0.1` | 0.101160700 |
| `varx64-ridge0.1` | 0.105388968 |
| `varx32-ridge0.1` | 0.127522738 |
| `new64` | 0.141535923 |
| `new16` | 0.141538778 |

## Every candidate and strongest-control seed

| Seed | Candidate | Strongest control |
| --- | ---: | ---: |
| 9201 | 0.043493356 | 0.054708879 |
| 9202 | 0.042293333 | 0.055740500 |
| 9203 | 0.042916811 | 0.055217339 |

## All 96 timed calls

Times are milliseconds. Slot order rotates the four cells; each occupies each position six times. One prior warmup per cell is retained in the complete evidence.

| Slot | Record | Start | Old 16 | Old 64 | New 16 | New 64 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 100mV-realization-3-period-0 | 0 | 80.716166 | 227.461209 | 77.800000 | 73.816000 |
| 1 | 100mV-realization-3-period-0 | 7936 | 68.825250 | 202.611250 | 86.838042 | 119.882792 |
| 2 | 100mV-realization-3-period-1 | 0 | 72.309125 | 228.255708 | 89.941083 | 170.857750 |
| 3 | 100mV-realization-3-period-1 | 7936 | 66.536708 | 177.609958 | 92.204417 | 84.333750 |
| 4 | 100mV-realization-4-period-0 | 0 | 105.118833 | 309.439625 | 81.443042 | 172.224959 |
| 5 | 100mV-realization-4-period-0 | 7936 | 64.752208 | 80.058292 | 99.957792 | 90.843958 |
| 6 | 100mV-realization-4-period-1 | 0 | 62.460084 | 179.313875 | 76.777791 | 182.240042 |
| 7 | 100mV-realization-4-period-1 | 7936 | 61.323417 | 61.079875 | 86.350375 | 198.729708 |
| 8 | 100mV-realization-5-period-0 | 0 | 62.139041 | 170.605583 | 79.235333 | 78.878375 |
| 9 | 100mV-realization-5-period-0 | 7936 | 67.799292 | 170.669375 | 119.543208 | 102.814584 |
| 10 | 100mV-realization-5-period-1 | 0 | 66.162667 | 183.539750 | 81.087417 | 83.692709 |
| 11 | 100mV-realization-5-period-1 | 7936 | 64.552041 | 157.997458 | 84.326750 | 155.487375 |
| 12 | 200mV-realization-3-period-0 | 0 | 71.892708 | 173.659666 | 78.051875 | 71.258250 |
| 13 | 200mV-realization-3-period-0 | 7936 | 69.855625 | 198.596334 | 86.525000 | 219.208792 |
| 14 | 200mV-realization-3-period-1 | 0 | 71.042042 | 156.015625 | 89.535500 | 89.031709 |
| 15 | 200mV-realization-3-period-1 | 7936 | 75.368208 | 189.916500 | 66.295709 | 213.009417 |
| 16 | 200mV-realization-4-period-0 | 0 | 63.297000 | 182.297500 | 66.919167 | 69.498084 |
| 17 | 200mV-realization-4-period-0 | 7936 | 63.723083 | 163.207459 | 105.303750 | 142.578959 |
| 18 | 200mV-realization-4-period-1 | 0 | 64.038375 | 186.828666 | 53.073583 | 54.679500 |
| 19 | 200mV-realization-4-period-1 | 7936 | 61.599542 | 164.664791 | 64.763000 | 272.474833 |
| 20 | 200mV-realization-5-period-0 | 0 | 63.163791 | 220.614166 | 63.692292 | 128.614583 |
| 21 | 200mV-realization-5-period-0 | 7936 | 60.993625 | 208.791333 | 72.094709 | 104.021041 |
| 22 | 200mV-realization-5-period-1 | 0 | 63.663416 | 216.924125 | 61.269000 | 195.785959 |
| 23 | 200mV-realization-5-period-1 | 7936 | 62.212166 | 194.614458 | 69.294084 | 277.805167 |

## Solver work within the 96 timed calls

These are totals over 24 calls per cell, excluding warmups and untimed scoring calls. Complete traces for all forecasts are in the evidence archive.

| Cell | Directions | Accepted steps | Trajectories | Jacobians | Trial attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| old16 | 384 | 383 | 831 | 408 | 423 |
| old64 | 1054 | 1030 | 2390 | 1078 | 1312 |
| new16 | 346 | 336 | 1189 | 370 | 819 |
| new64 | 576 | 552 | 2249 | 600 | 1649 |
