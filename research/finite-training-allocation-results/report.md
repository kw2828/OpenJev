# Recurrent learning under one training-time allowance

![Every fit and its measured cost](benchmark.png)

Three methods share the same 352-parameter recurrent model, fresh data and paired initial parameters. Each has stage deadlines at elapsed 10 and 40 seconds from one original start. Late updates restore both parameters and optimizer state; all attempted work is retained and charged.

[Protocol](../finite-training-allocation-protocol.md) · [All saved numbers](summary.json) · [Original evidence receipt](receipt.json)

| Criterion | Continuous joint | Restarted joint | Prefix then joint |
|---|---|---|---|
| SHORT_HORIZON_LEARNING | PASS 24/24 | PASS 24/24 | PASS 24/24 |
| BLIND_EXTRAPOLATION | FAIL 15/21 | FAIL 16/21 | FAIL 18/21 |
| OBSERVED_FILTERING_EXTRAPOLATION | PASS 8/8 | PASS 8/8 | PASS 8/8 |

**Allocation advance: FAIL, 16/21 conditions.**

Every original criterion must pass. Against each control, all six paired long-horizon regret differences must be nonpositive, both horizon means must improve at least 10%, and mean measured fit time must be within 5%. Favorable means do not rescue failed conditions.

| Advance condition | Outcome |
|---|---|
| BLIND_EXTRAPOLATION | FAIL |
| OBSERVED_FILTERING_EXTRAPOLATION | PASS |
| SHORT_HORIZON_LEARNING | PASS |
| joint_continuous_430261001_h4_nonpositive_regret_difference | FAIL |
| joint_continuous_430261001_h8_nonpositive_regret_difference | FAIL |
| joint_continuous_430261002_h4_nonpositive_regret_difference | PASS |
| joint_continuous_430261002_h8_nonpositive_regret_difference | PASS |
| joint_continuous_430261003_h4_nonpositive_regret_difference | PASS |
| joint_continuous_430261003_h8_nonpositive_regret_difference | PASS |
| joint_continuous_h4_ten_percent_mean_regret | PASS |
| joint_continuous_h8_ten_percent_mean_regret | PASS |
| joint_continuous_mean_fit_time_within_five_percent | PASS |
| joint_restart_430261001_h4_nonpositive_regret_difference | FAIL |
| joint_restart_430261001_h8_nonpositive_regret_difference | FAIL |
| joint_restart_430261002_h4_nonpositive_regret_difference | PASS |
| joint_restart_430261002_h8_nonpositive_regret_difference | PASS |
| joint_restart_430261003_h4_nonpositive_regret_difference | PASS |
| joint_restart_430261003_h8_nonpositive_regret_difference | PASS |
| joint_restart_h4_ten_percent_mean_regret | PASS |
| joint_restart_h8_ten_percent_mean_regret | PASS |
| joint_restart_mean_fit_time_within_five_percent | PASS |

## Every training allocation

| Arm | Seed | Accepted joint | Accepted prefix | All attempted | Timed s | Overrun s | Final summary s | Full fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_continuous | 430261001 | 5061 | 0 | 5063 | 40.009427 | 0.009427 | 0.001789 | 40.248998 |
| joint_continuous | 430261002 | 4631 | 0 | 4633 | 40.004682 | 0.004682 | 0.002396 | 40.228726 |
| joint_continuous | 430261003 | 4703 | 0 | 4705 | 40.005416 | 0.005416 | 0.001515 | 40.221250 |
| joint_restart | 430261001 | 5043 | 0 | 5045 | 40.006753 | 0.006753 | 0.002879 | 40.241443 |
| joint_restart | 430261002 | 5014 | 0 | 5016 | 40.003753 | 0.003753 | 0.001538 | 40.242626 |
| joint_restart | 430261003 | 4814 | 0 | 4816 | 40.013828 | 0.013828 | 0.002366 | 40.261002 |
| prefix_then_joint | 430261001 | 3877 | 1665 | 5544 | 40.004495 | 0.004495 | 0.001286 | 40.236668 |
| prefix_then_joint | 430261002 | 3801 | 1597 | 5400 | 40.010665 | 0.010665 | 0.002045 | 40.253431 |
| prefix_then_joint | 430261003 | 3533 | 1587 | 5122 | 40.010617 | 0.010617 | 0.001965 | 40.243257 |

Fit seconds include construction, both allocation stages, discarded attempts, boundary checkpoints, final diagnostics and durable full trace; final fit-row publication and shared preprocessing remain included in outer producer time.

The timing columns are nested, not additive. Equal eligibility is not exact FLOP or wall-time equality. Continuous Adam carries moments; restarted Adam retains the next accepted batch but discards moments. Prefix training freezes the cost head and starts joint training at batch zero. Each fit has its own clock, so the joint controls can have different boundary states and batch counts.

## Every endpoint metric

| Arm | Seed | H | Cases | Blind MSE | Regret | Survival MAE | Observed MSE | Observed survival MAE | Observed KL | Shuffled regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| joint_continuous | 430261001 | 1 | 120 | 0.05496535 | 0.255228 | 0.002768777 | 0.05496535 | 0.002768777 | 0.04106666 | 0.5524824 |
| joint_continuous | 430261001 | 2 | 120 | 0.05781974 | 0.2305879 | 0.003632501 | 0.05946952 | 0.00293102 | 0.05867371 | 0.5848046 |
| joint_continuous | 430261001 | 4 | 120 | 0.06666629 | 0.2674916 | 0.005608283 | 0.06361098 | 0.002984728 | 0.06790785 | 0.5832731 |
| joint_continuous | 430261001 | 8 | 120 | 0.07578898 | 0.33459 | 0.006648065 | 0.069061 | 0.002926432 | 0.08130532 | 0.4913363 |
| joint_continuous | 430261002 | 1 | 120 | 0.05509245 | 0.2012151 | 0.00339327 | 0.05509245 | 0.00339327 | 0.04391175 | 0.6227818 |
| joint_continuous | 430261002 | 2 | 120 | 0.06546218 | 0.2402 | 0.004544471 | 0.06683915 | 0.003202311 | 0.06724426 | 0.6499268 |
| joint_continuous | 430261002 | 4 | 120 | 0.06359735 | 0.2379691 | 0.006534284 | 0.06325035 | 0.002894696 | 0.06355319 | 0.6405336 |
| joint_continuous | 430261002 | 8 | 120 | 0.06530244 | 0.2872586 | 0.007553377 | 0.06275745 | 0.002902595 | 0.06200349 | 0.542372 |
| joint_continuous | 430261003 | 1 | 120 | 0.00257703 | 0.0009020022 | 0.002880808 | 0.00257703 | 0.002880808 | 0.006510271 | 0.6208036 |
| joint_continuous | 430261003 | 2 | 120 | 0.002783152 | 0.0012427 | 0.003828146 | 0.003147808 | 0.003053066 | 0.008849153 | 0.6070824 |
| joint_continuous | 430261003 | 4 | 120 | 0.003168045 | 0.0006044552 | 0.005857074 | 0.003682719 | 0.00272182 | 0.008005972 | 0.5675634 |
| joint_continuous | 430261003 | 8 | 120 | 0.003926581 | 0.001043509 | 0.007075555 | 0.002901888 | 0.002672609 | 0.008989087 | 0.4934873 |
| joint_restart | 430261001 | 1 | 120 | 0.05481301 | 0.255228 | 0.002752795 | 0.05481301 | 0.002752795 | 0.04184314 | 0.5524831 |
| joint_restart | 430261001 | 2 | 120 | 0.05754737 | 0.2305879 | 0.003670612 | 0.05919703 | 0.003011634 | 0.05910993 | 0.5848046 |
| joint_restart | 430261001 | 4 | 120 | 0.06645737 | 0.2674916 | 0.005537591 | 0.06362769 | 0.00321934 | 0.06843101 | 0.5832731 |
| joint_restart | 430261001 | 8 | 120 | 0.07554347 | 0.3410856 | 0.006724549 | 0.06897682 | 0.00306678 | 0.08187425 | 0.4913818 |
| joint_restart | 430261002 | 1 | 120 | 0.05161204 | 0.2007876 | 0.003524583 | 0.05161204 | 0.003524583 | 0.04169722 | 0.6070051 |
| joint_restart | 430261002 | 2 | 120 | 0.0616579 | 0.2103402 | 0.004833193 | 0.0622194 | 0.003317157 | 0.0671379 | 0.6337508 |
| joint_restart | 430261002 | 4 | 120 | 0.05954982 | 0.2156978 | 0.006835937 | 0.05756962 | 0.003003011 | 0.04992744 | 0.5849187 |
| joint_restart | 430261002 | 8 | 120 | 0.0613605 | 0.2832039 | 0.008523572 | 0.05837112 | 0.002850849 | 0.05846193 | 0.5123272 |
| joint_restart | 430261003 | 1 | 120 | 0.00125797 | 0.000833822 | 0.003197489 | 0.00125797 | 0.003197489 | 0.005961173 | 0.6209489 |
| joint_restart | 430261003 | 2 | 120 | 0.001229312 | 0.001205278 | 0.004212493 | 0.001591614 | 0.003200752 | 0.007913395 | 0.6070824 |
| joint_restart | 430261003 | 4 | 120 | 0.001162865 | 0.0005908906 | 0.006385275 | 0.00179357 | 0.002883139 | 0.005450945 | 0.5603872 |
| joint_restart | 430261003 | 8 | 120 | 0.001105105 | 0.001043509 | 0.008399327 | 0.001194238 | 0.002861318 | 0.006369131 | 0.4941257 |
| prefix_then_joint | 430261001 | 1 | 120 | 0.05098034 | 0.2198323 | 0.003546116 | 0.05098034 | 0.003546116 | 0.04626314 | 0.6269651 |
| prefix_then_joint | 430261001 | 2 | 120 | 0.06021388 | 0.2978544 | 0.004502804 | 0.06035672 | 0.003278781 | 0.04798198 | 0.6222264 |
| prefix_then_joint | 430261001 | 4 | 120 | 0.06647984 | 0.2990542 | 0.006006317 | 0.06726278 | 0.002984668 | 0.05634206 | 0.5881915 |
| prefix_then_joint | 430261001 | 8 | 120 | 0.06845844 | 0.3457995 | 0.007238245 | 0.06876937 | 0.002945774 | 0.05995401 | 0.4956156 |
| prefix_then_joint | 430261002 | 1 | 120 | 0.001247939 | 0.0008205492 | 0.003349944 | 0.001247939 | 0.003349944 | 0.005808413 | 0.6447476 |
| prefix_then_joint | 430261002 | 2 | 120 | 0.001159675 | 0.001226632 | 0.004411421 | 0.001556322 | 0.003267122 | 0.008494623 | 0.6233615 |
| prefix_then_joint | 430261002 | 4 | 120 | 0.001123822 | 0.0005800413 | 0.006425922 | 0.001769865 | 0.003038182 | 0.005212583 | 0.582225 |
| prefix_then_joint | 430261002 | 8 | 120 | 0.0009616555 | 0.0009352059 | 0.008933714 | 0.001199233 | 0.00296954 | 0.005982383 | 0.5071528 |
| prefix_then_joint | 430261003 | 1 | 120 | 0.001234143 | 0.0008209524 | 0.003345281 | 0.001234143 | 0.003345281 | 0.006717276 | 0.6447476 |
| prefix_then_joint | 430261003 | 2 | 120 | 0.001154973 | 0.001226632 | 0.004436587 | 0.002407231 | 0.003262116 | 0.01214508 | 0.630167 |
| prefix_then_joint | 430261003 | 4 | 120 | 0.001111191 | 0.0005800413 | 0.00647852 | 0.002482141 | 0.003078984 | 0.008845938 | 0.582225 |
| prefix_then_joint | 430261003 | 8 | 120 | 0.0008957895 | 0.0010026 | 0.009001323 | 0.001331468 | 0.002992895 | 0.006622356 | 0.5071528 |

Arithmetic means of separately fitted models on common cases, not an ensemble or significance test.

## Evidence and limits

The original engineering log reports **269 passing tests**. Engineering includes the complete independent audit of its saved small run. Science uses fresh histories and all nine final checkpoints precede DEV generation. The independent audit decodes 27 parameter and 27 optimizer boundaries, reconstructs targets, metrics and work records, and makes no model, optimizer or environment calls.

| Original supervised phase | Seconds |
|---|---:|
| qualify | 18.006 |
| fit | 364.204 |
| audit | 4.564 |

Complete traces and checkpoints are in the evidence release; this display report keeps their hashes, boundary records and aggregate work without duplicating every attempted update. Historical optimizer execution and timings remain source-qualified attestations, not a replay.

These are synthetic learning results with a privileged initial readout. They do not establish a new architecture, biological wiring advantage, calibration, scenario-shift robustness or native transfer. Earlier failed experiments remain closed.
