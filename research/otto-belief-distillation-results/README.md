# Every belief-distillation comparison

Saved independent-audit scalars only. Positive reductions favor recurrent soft. Fit seeds share the same development cases. No case-level data is duplicated here.

| Seed | Setting | Control | NLL reduction | Decision-gap reduction | KL reduction | Effect-error reduction | Normal NLL reduction | Normal gap reduction |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 327000001 | lambda3 | Recurrent + sampled | +31.03% | +20.72% | +91.25% | +81.21% | +7.97% | +1.81% |
| 327000002 | lambda3 | Recurrent + sampled | +29.23% | -4.10% | +89.02% | +80.12% | +10.22% | +15.54% |
| 327000003 | lambda3 | Recurrent + sampled | +35.22% | -0.15% | +94.17% | +82.32% | +15.30% | +28.49% |
| 327000001 | lambda4 | Recurrent + sampled | +35.56% | +12.23% | +91.76% | +87.65% | +10.18% | +24.15% |
| 327000002 | lambda4 | Recurrent + sampled | +28.42% | -3.46% | +90.21% | +87.32% | +13.08% | +13.97% |
| 327000003 | lambda4 | Recurrent + sampled | +36.88% | +28.51% | +93.08% | +78.06% | +10.28% | +26.29% |
| 327000001 | lambda3 | Action blind + soft | +0.49% | +47.37% | +18.97% | +19.02% | +0.94% | +26.09% |
| 327000002 | lambda3 | Action blind + soft | -1.41% | +39.44% | -19.12% | +16.88% | +1.16% | +21.76% |
| 327000003 | lambda3 | Action blind + soft | +0.40% | +41.43% | +28.27% | +38.22% | -0.41% | +21.39% |
| 327000001 | lambda4 | Action blind + soft | +6.64% | +34.96% | +35.21% | -12.74% | +2.08% | +26.70% |
| 327000002 | lambda4 | Action blind + soft | -2.51% | +43.50% | +3.26% | +4.09% | +1.38% | +25.70% |
| 327000003 | lambda4 | Action blind + soft | +5.98% | +30.46% | +35.36% | -26.48% | +2.78% | +29.57% |
| 327000001 | lambda3 | Direct + soft | +4.55% | +28.92% | +20.56% | +3.01% | +0.68% | +1.21% |
| 327000002 | lambda3 | Direct + soft | +1.73% | +21.54% | -9.21% | -21.96% | +0.87% | +29.02% |
| 327000003 | lambda3 | Direct + soft | -0.87% | +9.52% | +19.97% | +19.13% | -1.05% | +6.87% |
| 327000001 | lambda4 | Direct + soft | +6.35% | -0.46% | +22.07% | -69.27% | -0.05% | +10.48% |
| 327000002 | lambda4 | Direct + soft | -0.32% | +6.97% | +4.68% | -46.46% | +1.88% | +33.33% |
| 327000003 | lambda4 | Direct + soft | +6.56% | +0.37% | +9.20% | -90.11% | +0.09% | +17.13% |

## Original gate

| Seed | Setting | Control | Passed conditions | Cell passes |
|---|---|---|---|---|
| 327000001 | lambda3 | recurrent_sampled | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000001 | lambda3 | action_blind_soft | gap_support, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | False |
| 327000001 | lambda3 | direct_soft | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000001 | lambda4 | recurrent_sampled | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000001 | lambda4 | action_blind_soft | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000001 | lambda4 | direct_soft | gap_support, gap_log_score, normal_support, normal_log_score, normal_decision_gap | False |
| 327000002 | lambda3 | recurrent_sampled | gap_support, gap_log_score, normal_support, normal_log_score, normal_decision_gap | False |
| 327000002 | lambda3 | action_blind_soft | gap_support, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | False |
| 327000002 | lambda3 | direct_soft | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000002 | lambda4 | recurrent_sampled | gap_support, gap_log_score, normal_support, normal_log_score, normal_decision_gap | False |
| 327000002 | lambda4 | action_blind_soft | gap_support, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | False |
| 327000002 | lambda4 | direct_soft | gap_support, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | False |
| 327000003 | lambda3 | recurrent_sampled | gap_support, gap_log_score, normal_support, normal_log_score, normal_decision_gap | False |
| 327000003 | lambda3 | action_blind_soft | gap_support, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | False |
| 327000003 | lambda3 | direct_soft | gap_support, gap_decision_gap, normal_support, normal_decision_gap | False |
| 327000003 | lambda4 | recurrent_sampled | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000003 | lambda4 | action_blind_soft | gap_support, gap_log_score, gap_decision_gap, normal_support, normal_log_score, normal_decision_gap | True |
| 327000003 | lambda4 | direct_soft | gap_support, gap_log_score, normal_support, normal_log_score, normal_decision_gap | False |

## All method and seed scalars

| Method | Seed | Setting | Long NLL | Long gap | Long KL | Long effect E | Normal NLL | Normal gap |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Recurrent + soft | 327000001 | lambda3 | 0.848942441 | 0.163814338 | 0.031723428 | 0.026021759 | 0.674983880 | 0.116571668 |
| Recurrent + soft | 327000001 | lambda4 | 0.816885435 | 0.286045187 | 0.036983631 | 0.019666633 | 0.637052929 | 0.156994239 |
| Recurrent + soft | 327000002 | lambda3 | 0.854453270 | 0.194353172 | 0.041468504 | 0.026709331 | 0.671410163 | 0.124614179 |
| Recurrent + soft | 327000002 | lambda4 | 0.865388383 | 0.231663146 | 0.041610760 | 0.016731251 | 0.636826988 | 0.134222753 |
| Recurrent + soft | 327000003 | lambda3 | 0.863049412 | 0.202446704 | 0.029598505 | 0.019852093 | 0.681056693 | 0.132788465 |
| Recurrent + soft | 327000003 | lambda4 | 0.831474854 | 0.262253724 | 0.038773250 | 0.022063816 | 0.637111633 | 0.138312410 |
| Recurrent + sampled | 327000001 | lambda3 | 1.230935499 | 0.206626305 | 0.362573753 | 0.138506014 | 0.733434721 | 0.118726496 |
| Recurrent + sampled | 327000001 | lambda4 | 1.267650438 | 0.325885383 | 0.448989007 | 0.159262477 | 0.709280682 | 0.206985803 |
| Recurrent + sampled | 327000002 | lambda3 | 1.207376813 | 0.186705264 | 0.377541889 | 0.134332241 | 0.747865713 | 0.147535636 |
| Recurrent + sampled | 327000002 | lambda4 | 1.208957756 | 0.223909603 | 0.424965857 | 0.131933002 | 0.732634897 | 0.156027505 |
| Recurrent + sampled | 327000003 | lambda3 | 1.332288671 | 0.202152384 | 0.508021496 | 0.112291078 | 0.804061378 | 0.185686833 |
| Recurrent + sampled | 327000003 | lambda4 | 1.317336468 | 0.366831934 | 0.560122636 | 0.100545194 | 0.710135901 | 0.187654168 |
| Action blind + soft | 327000001 | lambda3 | 0.853117643 | 0.311240443 | 0.039149310 | 0.032132651 | 0.681354987 | 0.157723986 |
| Action blind + soft | 327000001 | lambda4 | 0.874967372 | 0.439778666 | 0.057081721 | 0.017444224 | 0.650560094 | 0.214187490 |
| Action blind + soft | 327000002 | lambda3 | 0.842579856 | 0.320920178 | 0.034811267 | 0.032132651 | 0.679304721 | 0.159272605 |
| Action blind + soft | 327000002 | lambda4 | 0.844219076 | 0.410046505 | 0.043013689 | 0.017444224 | 0.645713438 | 0.180653162 |
| Action blind + soft | 327000003 | lambda3 | 0.866549194 | 0.345646496 | 0.041262174 | 0.032132651 | 0.678258329 | 0.168927329 |
| Action blind + soft | 327000003 | lambda4 | 0.884378876 | 0.377106398 | 0.059979671 | 0.017444224 | 0.655306265 | 0.196390472 |
| Direct + soft | 327000001 | lambda3 | 0.889414228 | 0.230452873 | 0.039933038 | 0.026830650 | 0.679599927 | 0.118001771 |
| Direct + soft | 327000001 | lambda4 | 0.872271420 | 0.284725828 | 0.047454921 | 0.011618305 | 0.636711399 | 0.175378144 |
| Direct + soft | 327000002 | lambda3 | 0.869462749 | 0.247720148 | 0.037972825 | 0.021900625 | 0.677328774 | 0.175556865 |
| Direct + soft | 327000002 | lambda4 | 0.862646367 | 0.249029275 | 0.043652148 | 0.011424138 | 0.649012878 | 0.201335123 |
| Direct + soft | 327000003 | lambda3 | 0.855564661 | 0.223738817 | 0.036983201 | 0.024548201 | 0.673996838 | 0.142590488 |
| Direct + soft | 327000003 | lambda4 | 0.889852641 | 0.263233450 | 0.042703495 | 0.011605767 | 0.637678141 | 0.166911433 |

## Recorded fit and inference costs

| Method | Seed | Fit seconds | Updates | Exposures |
|---|---|---:|---:|---:|
| recurrent_soft | 327000001 | 10.035812 | 2640 | 83680 |
| recurrent_sampled | 327000001 | 9.311512 | 2640 | 83680 |
| action_blind_soft | 327000001 | 9.633712 | 2640 | 83680 |
| direct_soft | 327000001 | 9.940225 | 2640 | 83680 |
| recurrent_sampled | 327000002 | 9.513766 | 2640 | 83680 |
| action_blind_soft | 327000002 | 9.453725 | 2640 | 83680 |
| direct_soft | 327000002 | 9.976411 | 2640 | 83680 |
| recurrent_soft | 327000002 | 9.625133 | 2640 | 83680 |
| action_blind_soft | 327000003 | 9.501330 | 2640 | 83680 |
| direct_soft | 327000003 | 9.842351 | 2640 | 83680 |
| recurrent_soft | 327000003 | 9.533597 | 2640 | 83680 |
| recurrent_sampled | 327000003 | 9.509343 | 2640 | 83680 |

| Method | Seed | Panel | Cases | Inference seconds |
|---|---|---|---:|---:|
| recurrent_soft | 327000001 | gap | 187 | 0.083848 |
| recurrent_soft | 327000001 | normal | 187 | 0.135866 |
| recurrent_soft | 327000001 | opposite | 187 | 0.078289 |
| recurrent_soft | 327000002 | gap | 187 | 0.083575 |
| recurrent_soft | 327000002 | normal | 187 | 0.126333 |
| recurrent_soft | 327000002 | opposite | 187 | 0.080145 |
| recurrent_soft | 327000003 | gap | 187 | 0.100217 |
| recurrent_soft | 327000003 | normal | 187 | 0.131345 |
| recurrent_soft | 327000003 | opposite | 187 | 0.078233 |
| recurrent_sampled | 327000001 | gap | 187 | 0.078073 |
| recurrent_sampled | 327000001 | normal | 187 | 0.121796 |
| recurrent_sampled | 327000001 | opposite | 187 | 0.093916 |
| recurrent_sampled | 327000002 | gap | 187 | 0.091495 |
| recurrent_sampled | 327000002 | normal | 187 | 0.136257 |
| recurrent_sampled | 327000002 | opposite | 187 | 0.081040 |
| recurrent_sampled | 327000003 | gap | 187 | 0.081527 |
| recurrent_sampled | 327000003 | normal | 187 | 0.124415 |
| recurrent_sampled | 327000003 | opposite | 187 | 0.082329 |
| action_blind_soft | 327000001 | gap | 187 | 0.088313 |
| action_blind_soft | 327000001 | normal | 187 | 0.133649 |
| action_blind_soft | 327000001 | opposite | 187 | 0.077853 |
| action_blind_soft | 327000002 | gap | 187 | 0.074330 |
| action_blind_soft | 327000002 | normal | 187 | 0.124821 |
| action_blind_soft | 327000002 | opposite | 187 | 0.079776 |
| action_blind_soft | 327000003 | gap | 187 | 0.075477 |
| action_blind_soft | 327000003 | normal | 187 | 0.120606 |
| action_blind_soft | 327000003 | opposite | 187 | 0.076739 |
| direct_soft | 327000001 | gap | 187 | 0.101958 |
| direct_soft | 327000001 | normal | 187 | 0.139310 |
| direct_soft | 327000001 | opposite | 187 | 0.108545 |
| direct_soft | 327000002 | gap | 187 | 0.108095 |
| direct_soft | 327000002 | normal | 187 | 0.142329 |
| direct_soft | 327000002 | opposite | 187 | 0.104522 |
| direct_soft | 327000003 | gap | 187 | 0.106247 |
| direct_soft | 327000003 | normal | 187 | 0.139313 |
| direct_soft | 327000003 | opposite | 187 | 0.111213 |

[Main interpretation](../otto-belief-distillation-results.md) | [Compact scalar JSON](summary.json)
