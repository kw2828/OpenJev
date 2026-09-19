# OpenJev experiment archive

Completed studies, including failures and controls. A passing implementation check is not an architecture result.

## Complete experiment list

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
