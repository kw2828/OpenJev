# OpenJev experiment archive

Completed studies, including failures and controls. A passing implementation check is not an architecture result.

## Complete experiment list

- [Computing only required schema-fill rows](dialogue-token-required-fill-results.md): all sixteen numerical comparisons and **15/16** speed requirements pass. All twelve larger cells meet 1.10x; minimum candidate/readout remains below its floor at **0.845x**. Overall qualification fails despite removing all seven unused fill rows in that layout. No training admission. [All paired timings](../output/dialogue-token-required-fill-v1/figure-01/timing.png).

- [Consolidating supported dialogue evidence before projection](dialogue-token-consolidation-results.md): all sixteen numerical comparisons and **15/16** speed requirements pass. All twelve larger cells meet 1.10x, including **1.61-1.87x** on the fixed real-mask layout. Minimum candidate/readout reaches **0.897x**, below its 0.90 floor, so overall qualification fails. No training admission. [All paired timings](../output/dialogue-token-consolidation-v1/figure-02/timing.png).

- [Separating observation and recurrent-state head computation](dialogue-token-factoring-results.md): all sixteen numerical comparisons pass and **14/16** speed requirements pass. Large candidate/scalar reaches 1.070x and real-mask slot/readout 1.005x, below the required 1.10x. Overall qualification fails; no training admission. [All paired timings](../output/dialogue-token-factoring-v1/figure-01/timing.png).

- [Projecting dialogue evidence before scattering](dialogue-token-projection-results.md): all sixteen numerical comparisons pass, with **1.28-1.45x** paired median speed ratios on the four real-mask cells. Overall admission still fails: **10/16** speed cells pass, including only 3/8 original medium/large cells. No training restart or task-quality claim. [All paired timings](../output/dialogue-token-projection-v1/figure-01/timing.png).

- [Mask-packed dialogue pooling](dialogue-token-packing-results.md): all sixteen output/gradient checks pass, but **0/12 larger cells** meets the speed requirement. The fixed 160-update run fails admission, including all four cells with a preselected real training mask. Values and labels remain synthetic. [Every paired timing measurement](../output/dialogue-token-packing-v1/figure-01/timing.png).

- [Time-batched dialogue pooling](dialogue-token-batching-results.md): all twelve output/gradient checks pass, but only **1/8 medium/large cells** meets the speed requirement. The fixed 120-update engineering run fails admission. No training restart. [All paired timing measurements](../output/dialogue-token-batching-v1/figure-01/timing.png).

- [Shared-token dialogue study](dialogue-token-results.md): preparation completed in **17.88 seconds**, with independently verified representations. Training then exceeded its fixed **3,600-second** cap at **7/12 complete fits**, with one partial fit preserved. No task predictions were scored. [Prospective batching qualification](dialogue-token-batching-protocol.md).

- [Candidate-conditioned dialogue encoding cost screen](dialogue-joint-capacity-results.md): a valid **512-text** probe projects **20.91 minutes**, exceeding the fixed **12-minute** admission limit. Disk projection passes at 1.74 GiB. Independent audit confirms the stop; no full encoding, training or new accuracy result. [Shared-token follow-up design](dialogue-token-evidence-design.md).

[![Measured capacity probe and projected full cost, with the unchanged admission thresholds](../output/dialogue-joint-v1/figure/capacity.png)](dialogue-joint-capacity-results.md)

- [Corrected candidate dialogue memory](dialogue-copy-v2-results.md): **15 fresh fits**, valid internal probabilities, independently audited **7/13** scientific checks passed. Selective unseen macro remains **72.89%**, versus scalar's **72.58%**; the proposed mechanism still fails. The numerical correction changes no continuation decision. All **56,202,005** executed question positions passed each state-normalization check.

[![All fifteen corrected candidate-memory fits](../output/dialogue-copy-v2/report-01/comparison.png)](dialogue-copy-v2-results.md)

- [Internal dialogue-memory qualification](dialogue-evidence-qualification-results.md): stopped after the first scalar fit reproduced all **62,329** saved predictions but failed internal normalization checks on **7,989** prior states. The old output metrics remain descriptive; this motivated the corrected fresh training above. No counterfactual efficacy result was produced by the failed qualification.

- [Historical candidate dialogue memory with lexical observations](dialogue-copy-results.md): **15 fits**. Selective memory improves unseen macro accuracy to **72.89%**, versus literal carry's **65.76%**. Simpler scalar memory reaches **72.58%** and beats selective on seen services. The primary fails its rule with **7/13 checks passed** and still trails literal carry on unseen revisions. Its internal normalization defect is addressed by the fresh corrected study above; retain these outputs as historical evidence.

[![All fifteen candidate-memory fits and both deterministic references](../output/dialogue-copy-v1/report-01/comparison.png)](dialogue-copy-results.md)

- [Recurrent dialogue memory](dialogue-memory-results.md): **21 fits** on a supplied-schema SGD subtask. Surprise-adaptive Kalman memory scores **71.65% / 53.98%** seen/unseen macro accuracy and fails its fixed continuation rule with **8/11 checks passed**. A literal-match-and-carry reference scores **65.76%** on unseen services and beats all neural models there. No new architecture advantage established.

[![All dialogue memory models and deterministic references](../output/dialogue-memory-v1/visualization-01/comparison.png)](dialogue-memory-results.md)

- [Robot dynamics opportunity screen](drive-qualification-results.md): **252 episodes and 75,600 simulated transitions**. Exact parameters improve late tracking by only 3.76% and 9.49%; **6/10 requirements pass**. Strong calibration controls close this fixed configuration before any neural fit.

[![All robot observers and paired seeds in the completed dynamics screen](../output/drive-qualification-v1/report-01/paired-costs.png)](drive-qualification-results.md)

- [Shared-context text scoring](../research/shared-prefix-results.md): a cache prototype is **2.08x faster** across the four-question workloads, with every selected answer preserved in 1,296 synthetic request evaluations. One-question caching is slower. This is a serving benchmark, not a new model or calibration result.

[![Shared-prefix prototype: all methods and workload cells](../output/shared-prefix-v1/report-01/warm-latency.png)](../research/shared-prefix-results.md)

- [Does ordered error memory help?](../research/pose-innovation-results.md): **45 fits across nine methods**, evaluated on twelve held-out training parents. The recurrent head lowers error by about 4%, but simple summary ridge beats it on both endpoints in all nine paired comparisons. **223/368 checks passed; the recipe failed.**

[![All nine error-history methods and held-out training results](../output/pose-innovation-v1/visualization-01/physical-errors.png)](../research/pose-innovation-results.md)

- [Can past errors choose the blend?](../research/pose-probe-blend-results.md): **0/17 requirements passed**. Simple inverse-error weights beat the fitted probe on all four measurements. The [earlier capacity diagnosis](../research/pose-capacity-results.md) found 17-21% potential improvement when given future answers; the past-error rule did not realize it.

- [Parent-excluded selector training](../research/pose-crossfit-results.md): 30 neural fits produced small improvements, but only **2/17 requirements passed**. The [earlier coordination screen](../research/pose-coordination-results.md) passed 1/17. Neither established an architecture advantage.

- [Recency and robust adaptation](../research/pose-support-results.md): with the same trained weights, position error falls **36.6% / 15.4%** versus no adaptation. Plain rotation improves; zigzag rotation regresses **7.6%**. Recent-five explains most of the position gain. **5/17 requirements pass**, so no new architecture advantage is established.

[![Recency, robust updates and controls with equal total support weight](../output/pose-support-v1/visualization-01/physical-errors.png)](../research/pose-support-results.md)

- [Learning through context adaptation](../research/pose-adaptation-results.md): twelve new fits. Disabling adaptation in the **same trained model** lowers both physical errors on both robot archives for every seed. Only **1/17 requirements** passed; the adaptation recipe failed.

- [Geometry and recurrent memory](../research/pose-transport-results.md): 12 new fits with physical position/rotation errors and stronger motion baselines. Rotating memory improved the matched model by only 0.1% to 1.4%; **345/541 checks passed**, so the continuation rule failed.

- [Residual dynamics and online correction](../research/residual-dynamics-results.md): six new fits and 500 ms forecasts. Error fell 52% on one archive and rose 27% on another; **36/49 checks passed**, so the continuation rule failed. The gain is concentrated in one trajectory, and simple motion prediction explains much of the position improvement. [Figure](../output/residual-dynamics-v1/visualization-03/forecast-results.png).

- [Action-conditioned robot memory](../research/action-filter-results.md): 15 new fits on published simulated-robot data. A linear predictor beat every neural model; the proposed memory had 18.3 times the GRU's forecast error. [Comparison figure](../output/action-filter-v1/visualization-01/forecast-results.png). No architecture advantage established.

[![Same memories, better confidence: all five associative families solve every card game](../output/card-calibration-v1/visualization-01/calibration.png)](../research/card-calibration-results.md)

[![First fixed card game under baseline, calibrated and hard probabilities](../output/card-calibration-v1/visualization-01/fixed-public-replay.gif)](../research/card-calibration-results.md)

- [Calibrated card memory](../research/card-calibration-results.md): with the same weights, all five associative-memory families improve from **35-46% to 100% completion**, and aggregate prediction loss falls **36.3%**. The overall rule still fails **11/12** because the GRU regresses. Simple memory references remain more efficient; no new architecture advantage is established. [Earlier controller comparison](../research/card-controller-comparison.md) · [Original training](../research/card-memory-pilot.md).
- [Route memory](../research/mystery-path-memory-qualification.md): full memory reached 76.2% success versus 29.7% with the last 32 transitions, but missed the fixed 80% requirement. These are rule-based controls; no neural-model result is claimed. [Recorded GIF](../output/mystery-path-qualification-v1/visualization-01/episode.gif).
- [Planning diagnostic](../research/reacher-common-root-diagnostic.md): a longer horizon lowered mean branch cost by 7.33%, but missed the consistency rule; accurate dynamics lowered selected branch cost by 2.51%. All three comparisons failed, so this tracking setup is closed. [Earlier plan-memory test](../research/reacher-proposal-memory.md).
- [Error-gated memory pilot](../research/reacher-innovation-pilot.md): twelve fits learned, but normalized-error gating failed its continuation rule (21/29 checks). Its gain over age-only gating was below 0.13%.
- [Earlier robot reward-prediction study](../research/reacher-reward-residual-control.md): 9/17 continuation checks passed. [Biological learning and JEPA follow-up](../research/connectome-learning-program.md).
- [Chess action-outcome supervision](../docs/chess-continuation.md): nine fits and 192 games; no established gameplay improvement.
- [Astra-trained text students](../docs/codex-astra-distillation.md): routing improved; BoolQ remained near chance.
- [Recurrent world models](../docs/recurrent-world-model-study.md) and [associative-memory PPO](../docs/associative-ppo-study.md): no established memory advantage. [Initialization follow-up](../docs/zero-critic-ppo-study.md).
- [Doom PPO/DQN](../docs/rl-study.md), [JEPA + RL](../docs/jepa-rl-study.md), [all figures](../docs/benchmarks.md) and [paper sources](../paper/README.md).
