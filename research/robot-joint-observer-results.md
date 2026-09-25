# Joint observer and dynamics: small fresh-control gains, development gate failed

**`JOINT_OBSERVER_DEVELOPMENT_FAIL`: 4 of 5 criteria passed.** Jointly training the long observer and its transition produced H128 standardized RMSE **0.643945**, averaged equally across four exposed recordings. This was 0.652% lower than the matched short observer, 0.750% lower than continued last-two, and 0.319% lower than continued temporal. It remained **4.682% higher than the strongest archived control**, historical joint temporal at 0.615141. The registered requirement was at least 5% below that best control, or at most 0.584384. The independent evidence audit passed; the scientific criterion did not.

The [frozen protocol](robot-joint-observer-protocol.md) required all five criteria. Complete forecasts and costs, the per-file fresh-control guard, the latency cap and nondominance passed. The accuracy requirement against the best of all 22 controls failed. This result does not admit the proposed transfer step; no transfer experiment was launched.

| Registered criterion | Outcome |
|---|---|
| Complete eligible H64/H128 forecasts and all 61 current costs | Pass |
| At least 5% lower equal-file H128 mean than the best control | **Fail** |
| Every file within 2% of its best fresh simple control | Pass |
| Full-request latency at most 1.5 times continued last-two | Pass |
| No eligible control dominates error, latency and persistent storage together | Pass |

The table reports each file's mean over seeds 8101, 8102 and 8103, followed by their equal-four-file mean. Values are standardized RMSE, lower is better; these are individual-model averages, not ensembles. The first two files were original DEV; the latter two were previously exposed confirmation files and are now development data.

| Family | Rate | 21H_54M | 22H_10M | 22H_41M | 22H_50M | Equal-file mean | Request ms | Persistent bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Fresh long observer** | .001 | 0.613356 | 0.677499 | 0.603590 | 0.681334 | **0.643945** | 5.783 | 2,888 |
| Fresh short observer | .001 | 0.617327 | 0.674844 | 0.615214 | 0.685290 | 0.648169 | 4.444 | 2,888 |
| Continued last-two | .001 | 0.623437 | 0.671105 | 0.621381 | 0.679332 | 0.648814 | 4.088 | 2,600 |
| Continued temporal | .001 | 0.616637 | 0.678464 | 0.610158 | 0.678768 | 0.646007 | 4.143 | 4,088 |
| Archived position observer, frozen .001 alias | .001 | 0.615415 | 0.687323 | 0.615899 | 0.685486 | 0.651031 | 5.671 | 2,888 |
| Archived position observer, selected recipe | .003 | 0.613470 | 0.686064 | 0.613860 | 0.684750 | 0.649536 | 5.638 | 2,888 |
| **Historical joint temporal** | .003 | 0.576806 | 0.635978 | 0.610736 | 0.637046 | **0.615141** | 4.051 | 4,088 |

The long observer won 8 of 12 paired file/seed comparisons against the short observer, 6 of 12 against continued last-two, and 7 of 12 against continued temporal. Its file means beat those controls on three, two and three files, respectively. Seed 8102 was worse than the short observer on three of four files, while seed 8103 was better on all four. These descriptive comparisons show heterogeneous effects; the shared robot, recordings and seeds are not independent replications supporting a significance claim.

[Full-range benchmark](robot-joint-observer-results/benchmark.png) and [readable detail](robot-joint-observer-results/benchmark-detail.png) retain all 24 displayed families, including the failed historical observer. The full figure preserves extreme finite errors; the detail view marks values above its range. The [complete table](robot-joint-observer-results/table.md) covers every family; its evidence package retains both horizons, earlier rates and all current cost slots.

All twelve fresh fits completed 4,096 updates: **49,152 additional optimizer updates** in total. Every fresh arm used the same per-seed archived last-two transition, fresh empty Adam state, the same paired batches of 16, H128 MSE, float32 arithmetic and fixed .001 learning rate. There was no new rate search, optimizer continuation, clipping repair or retry. The long and short observers each trained 662 values, including the common 590-value transition and 72-value gain. The short observer uses three recent positions and one correction; the long observer uses thirty corrections. Continued last-two and continued temporal train 590 and 962 values. Historical joint temporal remains a separate, previously selected reference, not the warm start of the fresh temporal arm.

Equal update exposure did not mean equal compute. Long-observer request latency was **5.783250 ms**, versus **4.087625 ms** for continued last-two, a **1.414819×** ratio. These are medians across the three instance medians, each measured from twenty current samples after three warmups. A request includes normalization, casts, conditioning, operator preparation, H128 rollout, validation, denormalization and deadline checks on one thread. Persistent bytes include parameters, buffers, the 12-value state and normalizers, with no sharing; they exclude temporary workspace, loading and Python overhead. Passing nondominance reflects an accuracy/storage/latency tradeoff, not overall superiority to the best historical model.

The experiment supplies modest evidence that this jointly trained long initializer can improve the declared fresh operating point. It does not establish that the frozen transition was the cause of the previous failure, or that older observations alone explain the gain. Long and short conditioning share parameter counts but differ in function, computation and gradient path. Joint observer training is conventional; there is no novelty, verified physical-state recovery, stability, calibration or closed-loop-control claim. All four recordings are exposed measurements from one robot/day, and realized future torques are supplied. Official TEST remained closed.

The original run exited successfully after **1,245.721 seconds**; its one independent audit exited successfully after **13.293 seconds**. The audit verified all **57 final states**, twelve initial/Adam/diagnostic bundles and **48 new forecast banks**, independently reconstructing the 96 new score rows and five criteria. It retained **488 parent rows and 244 parent attempts unchanged**, with no parent forecast replay. Totals are **584 score rows, 292 attempts and 61 current costs**, across **24 displayed and 23 eligible families**. All 24 failed score rows belong to the older learned-observer attempts and remain visible; no fresh fit failed. The frozen .001 position alias uses existing rows and checkpoints without adding or relabeling scores.

Qualification attempt 01 is preserved: Ruff passed, 217 tests passed and one rate-mutation fixture failed because its mutation left the value unchanged. Only that fixture was corrected. Attempt 02 passed Ruff and all **218 tests** before registration and the original run. The package retains both attempts, source snapshots, original closures, twelve all-parameter training diagnostic bundles and the 63 registered source files. There were no additional diagnostic probes, backward/optimizer replay or repeated timing during the audit.

See the [agreeing audit](robot-joint-observer-results/audit.json), [file manifest](robot-joint-observer-results/manifest.json), [evidence README](robot-joint-observer-results/README.md), [original run receipt](robot-joint-observer-results/engineering/run-process-01.json) and [original audit receipt](robot-joint-observer-results/engineering/audit-process-01.json). Four measured target-window banks and external measurement arrays remain excluded under their descriptors and dataset terms. The next decision remains prospective; this negative development result does not reopen the earlier confirmation or justify a new performance claim.

A [predictive-state routing research note](predictive-state-routing-design.md) records one possible next question, its substantial prior art and the required supervision-matched dense control. The routing mechanism and benchmark are still unspecified; no follow-up experiment has been launched.
