# OpenJev experiment archive

Studies and frozen work awaiting execution, including failures and controls. A passing implementation check is not an architecture result.

## Complete experiment list

- [Complete-stream literal-history audit](dialogue-history-support-results.md): all **51,741 training endpoints in 2,017 dialogues**, completed in **2.57 seconds with zero model calls**. Of 38 apparent distant SYSTEM-only references, 34 already express the target number as words within four exchanges; four involve ambiguous older cross-service party-size transfer. The source-separated proposal branch is closed on this cohort. This is diagnostic evidence, not a performance result. [All case assessments](dialogue-history-support-case-review.md) · [Frozen protocol](dialogue-history-support-protocol.md).

- [Qwen lexical-input ablation](dialogue-qwen-lexical-ablation-results.md): all **15,638 decisions completed in 41.3 minutes**. Removing flags improves changed accuracy **83.74% → 85.64%** for current context and **82.35% → 84.60%** with history. Retained error rises **17.40% → 19.85%** and **21.67% → 21.68%** respectively, while overall log loss worsens. **5/16 checks passed; both arms fail**, with independent agreement on 84 primary cells and all decisions. [Figure](../output/dialogue-qwen-lexical-ablation-v1/figure-01/comparison.png) · [Execution record](dialogue-qwen-lexical-ablation-status.md) · [Frozen protocol](dialogue-qwen-lexical-ablation-protocol.md).

- [Qwen retention diagnosis and blinded Astra6 review](dialogue-qwen-retention-results.md): all **15,638 saved decisions** reconciled. Retained TRUE errors are **521/551** current and **531/551** with history; previously unset errors rise **461 → 780**. Three completed reviewers agree on all 12 fixed sampled choices, including two defensible answers that conflict with reference labels. One planned reviewer could not start and remains missing. This error-conditioned review is diagnostic, not new accuracy or architecture evidence.

- [Qwen observation and recent history](dialogue-qwen-observation-results.md): all **15,638 decisions completed in 53.8 minutes**. Current-exchange Qwen reaches **83.74%** changed accuracy but **17.40%** retained error, versus **58.59% / 2.46%** for the historical corrected-flat mean. Four exchanges worsen changed accuracy to **82.35%** and retained error to **21.67%**. Both behavioral comparisons and both proper-score nonregression checks fail; the independent audit agrees. Correct previous values are supplied on exposed development rows. [Chart](../output/dialogue-qwen-observation-v1/figure-02/comparison.png) · [Execution history](dialogue-qwen-observation-status.md) · [Prospective protocol](dialogue-qwen-observation-protocol.md).

  [Pretrained recurrent control review](dialogue-delta-mem-source-review.md): the pinned delta-mem checkpoint matches all **324 expected tensor shapes** for the same Qwen3-4B backbone, with **4.87 million active adapter parameters**. Numerical compatibility remains untested, and MLX's existing delta kernel has different recurrence semantics. Its chat runtime retains ordinary KV history. The failed observation comparison does not promote this adapter to a quality study; diagnose retention first.

- [Paired training-objective comparison](dialogue-objective-results.md): all six fits completed once in **74.67 minutes**. Uniform training lowers retained error **4.82% → 3.95%** versus corrected stratum, but changed accuracy is **75.78% versus 75.89%**. Historical corrected flat remains stronger on pooled overall accuracy and log loss. **6/13 checks passed; continuation failed**, with independent agreement on all decisions. No architecture or autonomous-memory claim. [Decision figure](../output/dialogue-objective-v1/figure-01/objective-decisions.png) · [Overall figure](../output/dialogue-objective-v1/figure-01/objective-overall.png) · [Execution history](dialogue-objective-status.md).

- [Fixed training-weight correction](dialogue-weight-prior-results.md): all nine fits gain pooled overall accuracy and lose changed accuracy. Corrected aligned reaches **93.75%** overall and **75.89%** changed accuracy; corrected flat reaches **94.66%** overall and **58.59%** changed. Independent agreement on 90 primary cells and 270 paired cells. This is a fixed readout tradeoff, with no new fitting or architecture advantage. [All-fit figure](../output/dialogue-weight-prior-v1/figure-01/weight-prior.png).

- [Saved-distribution commitment diagnostic](dialogue-commitment-results.md): all five constructions across three seeds, with independent agreement on 75 primary cells and 225 paired cells. Mean mass plus aligned alternatives reaches **80.68%** changed accuracy versus mean's **78.60%**, but retained error rises and overall improvement is only **0.0554 pp**. Retention recovery versus aligned reverses under equal-service weighting. No fitting, new architecture or gate reversal. [All-seed figure](../output/dialogue-commitment-v1/figure-03/commitment.png).

- [Token alignment scientific campaign](dialogue-token-alignment-scientific-results.md): **all nine fits completed in 69.19 minutes** under the separately published two-hour allocation. Changed-value accuracy is **83.45% aligned / 78.60% token mean / 71.91% baseline**. Every paired seed improves on changed values, but retained-value harms lower overall accuracy and the rule fails **18/22**. Main report and independent audit agree. [All-nine figure](../output/dialogue-token-alignment-scientific-v1/figure-01/alignment.png) · [Launch and resource history](dialogue-token-alignment-scientific-status.md).

- [Token alignment cost screen](dialogue-token-alignment-capacity-results.md): exact frozen schema cache and two matched 124,482-parameter models pass implementation checks. All **72 synthetic cost events** complete in **20.84 seconds**, but the nine-fit study projects to **77.27 minutes**, above the fixed **48-minute admission threshold**. No scientific fits or new quality result. The zero-call metadata failure and its correction are preserved. [Timing figure](../output/dialogue-token-alignment-v1/capacity-figure-02/capacity.png).

- [Why typed loss improves while decisions worsen](dialogue-typed-decomposition-results.md): no new fits. Exact saved-probability decomposition shows **176 correct-to-wrong versus 122 wrong-to-correct events**; 164 of the new mistakes choose the wrong branch. The largest favorable loss contribution is among still-wrong rows. Two independent numerical readers agree within their declared scopes. Original **5/9 failure** unchanged. [Figure](../output/dialogue-typed-decomposition-v1/figure-01/typed-decomposition.png) · [Next observation-model design](dialogue-token-alignment-design.md).

- [Typed decisions and rare-category support](dialogue-typed-results.md): **12 fresh fits and 27,600 updates** complete in **551.82 seconds**. Typed-balanced reduces changed equal-service NLL **26.92%**, but changed accuracy falls **50.52% to 47.40%** and retained error rises **2.18% to 3.52%** against flat-balanced. The flat/stratum control reaches **70.88%** changed accuracy. **5/9 requirements pass; the recipe fails.** Independent arithmetic agrees. The original pre-scoring reader failure and its separate correction are preserved. [All-fit figure](../output/dialogue-typed-v1/figure-01/typed-observation.png).

- [Conditional decision error diagnosis](dialogue-conditional-error-results.md): a saved-output analysis finds that all nine fits choose the previous NOT_MENTIONED value on every unseen TRUE update. The small favorable mean NLL difference comes from TRUE/DONTCARE probability changes despite incorrect choices, offset by an ordinary-value regression. No new training or change to the failed study.

- [Conditional observation with the correct previous value](dialogue-conditional-results.md): all **9 fits** and **30,240 updates** complete in **489.08 seconds**. Candidate attention improves unseen-change NLL in two seeds and worsens in one, failing the fixed consistency rule despite a 1.07% favorable mean. Unseen-change accuracy is **68.64%**, versus **69.74%** for slot attention; mean pooling has the lowest average NLL. Independent audit verifies all metrics and costs. No memory or architecture advantage. [Figure](../output/dialogue-conditional-v1/figure-01/comparison.png).

- [Sharing state-weight columns within each dialogue forward](dialogue-token-shared-columns-results.md): all sixteen numerical comparisons and **15/16** speed requirements pass. Minimum cells all pass, but medium candidate/scalar reaches **1.088x**, below its 1.10x threshold. The single 160-update qualification fails admission. No training restart. [All paired timings](../output/dialogue-token-shared-columns-v1/figure-01/timing.png).

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
