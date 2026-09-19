# Independent review: two-observation control study

**Completed execution and audit; continuation failed, 24/25.** Independently summing every saved reward reproduces all 42 row means, all three-fit family means, and all 25 exact inclusive comparisons. Persistent recurrence improves on the two-observation model by **4.443796%** in the six-step-gap panel and **2.939287%** in the ten-step-gap panel. The latter does not meet the frozen 3% requirement. Rounding, alternate aggregation, or secondary results must not convert it into a pass.

The failed check is `gap_mean/shift/two_observation_gru`: persistent mean cost `5.893156421237040` exceeds its allowed threshold `5.889470184985484` by `0.003686236251556`. The two-observation comparator mean is `6.071618747407715`. This is a negative continuation decision, not an equivalence test or evidence that memory has no value.

## All families and references

Lower mean negative native reward is better. Every learned family retains all three paired fits, each with the same 64 cases and 50 decisions.

| Method | Full sensing | Six-step gaps | Ten-step gaps |
|---|---:|---:|---:|
| Persistent GRU | 5.291520 | 5.415830 | 5.893156 |
| Two-observation GRU | 5.351848 | 5.667691 | 6.071619 |
| One-observation cached GRU | 7.289267 | 7.801957 | 8.104161 |
| Privileged known-state physics | 4.478741 | 4.478741 | 4.478741 |
| Public particle-filter physics | 4.598499 | 4.636065 | 4.773625 |
| Public kinematic physics | 4.501670 | 4.522096 | 4.643636 |
| Zero command | 11.052549 | 11.052549 | 11.052549 |
| Uniform command | 42.578826 | 42.578826 | 42.578826 |

Persistent versus two-observation paired cost reductions are **4.679%, 2.973%, 5.844%** in ordinary gaps and **4.093%, 0.958%, 3.896%** in shifted gaps. Thus all six paired gap inequalities pass, but the shift-family minimum does not. Full-sensing means favor persistence by 1.127%, while pair0 favors the two-observation model by 2.730%; retain that heterogeneity.

The two-observation family improves over the one-observation cache by 27.356% in ordinary gaps and 25.080% in shifted gaps. That is a useful secondary result for this trained reconstruction recipe. It does not isolate velocity estimation or establish equivalence to persistent memory. All three physics references beat the persistent family in every panel; they use supplied dynamics, and the known-state row also receives privileged state. These finite-search references are not optimal stochastic-control bounds.

## Cost and verification scope

All nine models have 36,805 parameters and use CEM256 geometry scoring. Equal parameters, updates and proposal counts do not match total compute: the two-observation arm reconstructs its public history, including padded operations and original heads. Across all nine rows per learned family, batched/amortized decision times are 6.591 ms per case-decision for persistent GRU, 9.665 ms for two-observation GRU, and 7.008 ms for cached GRU. The two-observation total is 46.647% higher than persistent in this execution. These are shared-host instrumented timings, not isolated latency or an optimized architecture speed claim.

Three new history fits perform 1,152 updates each, 3,456 total, and take 836.722095 seconds. Their original paired initialization/data/orders remain inherited; six comparator fits are reused. Execution takes 2,771.517904 seconds, including 1,894.821317 seconds of control-row work. The independent audit reports 1,355.968478 seconds. Nested phases must not be added twice. Actual outer process receipts show exit0 for execution (2,772.479273 seconds) and audit (1,356.927038 seconds), within the frozen 7,200/3,600-second caps. Historical cumulative execution cost is 10,655.734675 seconds; the current audit is additional.

This review checked all 8,119 payload hashes, totaling 20,073,002,692 bytes, and exact 8,120-file membership including completion. All 116 current source files match their snapshots, frozen source commit, and published commit. All nine deployment tensors match their before/after snapshots and canonical identities; inherited prefit snapshots also match. The JSON contains every per-case cost, model weight path/file hash/canonical hash, source identity, and process receipt.

The completed audit reports zero maximum error for 134,400 executed transitions, 78,741,504 nominal candidate transitions, 28,800 selected nominal transitions, and 310,464 public-observer transitions. Including observers, this is 79,215,168 checked transitions. This review authenticated those audit outputs; it did not replay native physics or neural predictions. The local process receipts and source-bound driver are integrity evidence, not external process attestation.

The three fits share historical training data, initialization and order pairings. No biological, Bayesian, cross-environment, or novel-architecture superiority follows. No threshold was changed, no seed/fit was selected, and no new inference, optimization, simulation, or RNG draw occurred.

## Bound identities

- Plan: `e3196e1e3ed6b513ad65918a1f7780def1f708867c6ec7aaa61ccba668624038`.
- Freeze: `07071a8d711cd546e02f0df35ee4a9e839231ca925d7b091c240d9f92e373061`.
- Execution completion: `cf90efcf33d7f23f4f7777c534c3285ebbb4ef90f604019f96cd8e4c24b63bfc`.
- Audit receipt: `ce77c492ce9266b0c9cabb6f404ad10f55f8739806204b6bcaf49c21e2fbf24f`.
- Audit summary: `dc66fd1097dad508dded44940e5bab8eb7625dff38b6c44c597e7330b8f0ffd3`.
- Published commit: `a0852048625b9fcf13b8f0884254e134c7dff1b7`.
- Review JSON: `a685eeead30dca4caa391b37bbd7760ad5b97a877e95106d51ea0384c3ec2be5`.
