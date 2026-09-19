# Independent completed-results review

The declared continuation rule passes **25/25**. This review authenticated the four externally supplied hashes below, recomputed means from all 51 control rows and each row's 64 saved episode costs, and independently reconstructed all 25 threshold comparisons. It made no model, policy, native-simulator, training or network calls. It is a saved-output arithmetic and interpretation review, not another native replay audit.

## Authenticated inputs

- plan: `evidence/reacher-geometry-memory-v1/protocol/plan.json`; SHA-256 `23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee`.
- execution completion: `runs/reacher-geometry-memory-v1/attempt/completed.json`; SHA-256 `58268dea7b15b24d59e4a779a1a2cbe61a08e350a175a6ef7929c5b16db4749a`.
- audit receipt: `evidence/reacher-geometry-memory-v1/audit/receipt.json`; SHA-256 `cb64ded7bbcc4ad6e342f9ac3b431e085e754f77075bb2e4852d6b82d29a54c6`.
- audit summary: `evidence/reacher-geometry-memory-v1/audit/summary.json`; SHA-256 `921c7f12f73169dd788fbe8204ee5377949e3a5c8b7cd88023887a4da6e5753d`.

The receipt binds the summary, execution completion and frozen plan; receipt and summary costs agree. All twelve inherited fits, three observation panels, three paired fits per learned family and five reference controllers were included.

## Recomputed native cost

Lower is better. Family means average all three fits, with all 64 paired cases retained.

| Controller | Full | Six-step gaps | Ten-step gaps |
|---|---:|---:|---:|
| Persistent GRU | 5.537576278 | 5.709233056 | 6.296307290 |
| Current-packet GRU | 8.034106147 | 9.240608652 | 9.740743131 |
| Cached-angle GRU | 7.902707986 | 8.378622202 | 8.507414413 |
| Cached-angle MLP | 6.705505976 | 6.975331278 | 7.234154665 |
| Known-state physics | 4.568716923 | 4.568716923 | 4.568716923 |
| Particle physics | 4.731064889 | 4.786676833 | 4.892595462 |
| Public kinematics | 4.580680081 | 4.640030379 | 4.755555101 |
| Zero action | 11.830063897 | 11.830063897 | 11.830063897 |
| Uniform action | 42.796284457 | 42.796284457 | 42.796284457 |

Persistent cost reduction is `100 * (comparator mean - persistent mean) / comparator mean`:

| Comparator | Full | Six-step gaps | Ten-step gaps |
|---|---:|---:|---:|
| Current-packet GRU | 31.074146% | 38.215833% | 35.361120% |
| Cached-angle GRU | 29.928117% | 31.859524% | 25.990354% |
| Cached-angle MLP, descriptive | 17.417473% | 18.151084% | 12.964160% |

The independent 25-check reconstruction contains four gap-family mean thresholds, twelve paired-fit nonregression checks, two full-sensing mean limits, six persistent-versus-zero competence checks and one known-state ordinary-panel competence check. Every name, left/right numeric quantity and Boolean matched the saved gate. All 25 pass without using a secondary comparison to rescue any primary check.

Every paired fit improves against all three learned comparators in all three panels. Gap-panel reductions versus cached GRU range from 20.909636% to 38.750143%; versus MLP, from 5.951331% to 26.753031%. Persistent ordinary costs for pair0/1/2 are 5.698578205, 6.324741285 and 5.104379678; shifted costs are 6.315874998, 6.859158938 and 5.713887935. These means do not imply dominance on every episode: the smallest persistent-versus-MLP shifted improvement wins 38 of 64 cases.

Saved conditional paired-case 95% intervals for persistent minus cached GRU are [-3.129126216, -2.210239835] for ordinary and [-2.564162377, -1.867245605] for shift. These are native-cost differences, not percentages. They condition on the three existing fits and training corpus; this review generated no new bootstrap, and these are neither training-seed population intervals nor multiplicity-corrected confirmation.

## Physics, costs and diagnostics

Physics remains stronger. Persistent cost is 23.043010% higher than public kinematics on ordinary and 32.398998% higher on shift; it is 24.963598% and 37.813469% higher than known-state physics. Known-state, public kinematics and particle references now use CEM256 with the same geometry objective/horizon/block settings. Known-state is privileged; public references still have supplied simulator dynamics. They are not optimal stochastic-control oracles.

Physics whole-row wall is 6.25-6.64 times the persistent-family average. Persistent amortized whole-row values are 7.026124, 6.925095 and 6.553136 ms per case/action for full/ordinary/shift; public kinematics values are 44.051074, 43.248732 and 43.221202 ms. These are measured shared-host batched throughput with setup, native stepping and evidence storage, not isolated inference latency or a cross-machine benchmark.

All three GRUs have 36,805 parameters and 193,294,049,280 counted dense affine MACs per complete row. The MLP has 36,599 parameters but 316,534,090,496 counted MACs per row, about 1.64 times as many. These counters exclude substantial non-affine work. Equal candidate counts are not equal total compute, even with similar parameter counts. All original learned heads still execute under geometry scoring.

Execution cost is 2,109.204781 seconds; independent audit cost is 1,384.438953 seconds, or 58.227396 minutes together. The 51 row totals sum to 2,088.180367 seconds: learned rows 823.577845 seconds, physics rows 1,263.543862 seconds and floor rows 1.058659 seconds. Component timings are nested. The 7,884.216771-second cumulative execution lineage already includes the historical geometry study and its cache ancestry; do not add those histories again. Historical audits, preparation and publication are separate.

The authenticated audit reports zero maximum error for 163,200 executed native transitions, 78,741,504 nominal candidate transitions, 28,800 separately counted selected nominal transitions and 310,464 separately counted public-observer transitions. The native total excluding observer replay is 78,933,504. This review checked those recorded counts; it did not repeat native replay. No new fits or optimizer updates occurred in this study.

All learned rows report zero reward-score clipping. Persistent blackout angle MSE is 0.005288031 on ordinary and 0.016316530 on shift, versus cached GRU 0.049562400/0.118342293 and cached MLP 0.020925924/0.047530168. These are on-policy descriptive errors from different closed-loop states and actions, not common-input causal comparisons or proof of a particular hidden-state mechanism. This study collected no new held-out prediction episodes or common-root diagnostic bank.

## Supported claim and next comparison

These trained persistent real-observation update policies outperform both trained reset-GRU controls under fixed geometry scoring on this task. All GRUs remain recurrent within imagined rollouts, and the models were separately trained with different real-assimilation policies and validity gating. The result therefore does not isolate recurrence as a binary architectural property.

The substantial full-angle-sensing gains are compatible with hidden-motion estimation: instantaneous angle packets still omit velocity. Retaining the last angle alone is insufficient in these comparisons, but a two-valid-observation estimate with elapsed time and causal action history remains unresolved. That is the next useful matched comparison, under the same geometry interface and planning budget with fresh declared evaluation streams.

Do not claim biological or connectome superiority, Bayesian inference, calibrated uncertainty, new architecture, general robotics performance, independent multi-environment confirmation or isolated latency gains. The older cache-study failure and geometry-scoring success remain separate historical claims. The present passed gate authorizes a focused follow-up; it does not by itself establish a novel ICLR contribution.
