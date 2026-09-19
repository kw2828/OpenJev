# A stronger candidate-memory baseline; selective retention fails its test

**Corrected replication completed:** [15 fresh normalized fits](dialogue-copy-v2-results.md) pass internal-state checks and reproduce the scientific failure, with 7/13 requirements passed. Use that study for the corrected scalar comparison. This page preserves the historical V1 results.

**Numerical erratum:** a [later internal-state replay](dialogue-evidence-qualification-results.md) reproduced the first scalar fit's saved outputs but found substantial drift in its recurrent probability mass. The metrics below remain records of the frozen implementation; scalar comparisons do not establish the intended normalized-memory mechanism. A correction requires a new training study. The failed continuation decision is unchanged.

All **15 fits** completed. The proposed selective memory reaches **72.89%** three-stratum macro accuracy on unseen services, above literal copying's **65.76%**. However, a simpler scalar update reaches nearly the same unseen score, **72.58%**, and wins on seen services. The proposed method passes **7 of 13** predeclared requirements and fails the continuation rule.

These are results on previously exposed official development data for a supplied-service, supplied-slot, finite-candidate task. Official test contents remain untouched. They support a stronger practical baseline, not a new architecture or an ICLR-ready contribution.

![All fifteen final fits, literal carry, and always-unmentioned references](../output/dialogue-copy-v1/report-01/comparison.png)

## What changed

The [earlier experiment](dialogue-memory-results.md) included shared compressed memories and a learned carry control that already retained exact categorical candidate state. Every learned family lost to literal carry on unseen services. This new comparison uses revised candidate scorers and adds the same public exact-match, literal-history, reserved-type, and simple yes/no observations to the learned controls. It uses the same frozen MiniLM embeddings and dataset subset, with a new fixed 20-epoch recipe. The earlier study used 12 epochs and different models, so cross-study gains cannot be attributed to exact storage or any other single change.

The primary model can release a different fraction of each candidate's retained probability. Its scalar control releases the same fraction from every candidate. They share parameter shapes and initialization; given identical beliefs and scorer outputs, total released mass is equal. The [protocol](dialogue-copy-protocol.md) fixes this comparison and all thresholds before training. Copying and selective overwrite are established mechanisms, as described in the [source review](dialogue-copy-source-review.md).

## Results

Means include all three seeds, without checkpoint or seed selection. Macro accuracy equally weights unmentioned retention, assigned retention, and changed states. There are 29,236 seen-service and 33,093 unseen-service scored questions, with 241 and 203 revisions respectively. No development clear examples exist.

| Model | Seen macro | Unseen macro | Seen revisions | Unseen revisions |
|---|---:|---:|---:|---:|
| Readout with literal-history features | 73.19% | 65.77% | 57.26% | 62.56% |
| Scalar candidate memory | 79.12% | 72.58% | 55.05% | 72.58% |
| **Selective candidate memory, primary** | **78.01%** | **72.89%** | **56.15%** | **73.56%** |
| Selective, lexical/history features removed | 71.49% | 51.78% | 40.25% | 39.41% |
| Candidate GRU | 79.76% | 68.68% | 56.43% | 70.61% |
| Literal mention and carry | 54.75% | 65.76% | 49.79% | 79.80% |
| Always NOT_MENTIONED | 33.33% | 33.33% | 0.00% | 0.00% |

The selective model beats literal carry by **7.13 percentage points** on unseen macro accuracy. Removing its eight lexical/history observations lowers unseen macro accuracy by **21.11 points**. That bundled ablation supports the usefulness of those observations, but does not identify whether exact matches, literal state, or boolean cues cause each improvement.

Within the new feature representation, scalar memory is 6.81 points above the readout on unseen macro accuracy. The readout still has history through the deterministic literal register; it is not a wholly stateless system. Candidate GRU has the highest seen macro accuracy and overall micro accuracy, while scalar/selective transfer better by unseen macro accuracy. There is no universal winner across these metrics.

## Why the primary failed

The selective-minus-scalar macro difference is **-1.12 points on seen services** and **+0.31 points on unseen services**. Both miss the required +0.5-point margin. Selective loses all three paired seen comparisons and wins two of three unseen comparisons.

Its mean seen micro NLL is also worse: **0.55345 versus 0.52303**. Unseen NLL improves slightly, **0.74290 versus 0.75156**. The primary trails the conventional GRU on seen macro accuracy, and its unseen revision accuracy trails literal carry by **6.24 points**, outside the allowed one-point deficit.

These are all six failed checks: seen macro margin, seen NLL, seen paired wins, seen conventional-control comparison, unseen macro margin, and unseen revision deficit. All 15 fits completed, and the other six panel checks passed. The fixed recipe is closed; a better result from another metric or seed does not change that decision. Full unrounded checks and proper scores are in the [generated report](../output/dialogue-copy-v1/report-01/report.md) and [aggregate JSON](../output/dialogue-copy-v1/report-01/summary.json).

## Cost and verification

The run completed in **1,273.36 seconds** on the local CPU, with **19,200 optimizer updates**, **15,522,300 supervised question presentations**, and **934,935 saved development predictions**. There were no failed or replacement fits and no external model API calls. Non-GRU heads register 99,458 parameters; candidate GRU registers 103,411. Unused output-head parameters and gradient-count limits are disclosed in the protocol.

Shared encoder preparation previously cost 15.26 seconds for 44,763 texts and 1,337,813 tokens; it was reused here. New lexical preparation cost 2.27 seconds, with zero encoder or neural calls. Evaluation includes batch assembly and saving predictions, and excludes both preparation stages. Model order was fixed, and these are single local executions per fit, not a randomized latency benchmark or live serving-speed claim. The generated report retains every model's training/evaluation time and real/padded work counts.

All **107 synthetic tests** and lint checks passed before freezing. An [independent saved-only audit](../output/dialogue-copy-v1/independent-audit-01/receipt.json), implemented without importing the main metric helpers, recomputed every fit's metrics and all 13 decisions. All agree; the largest floating-point difference was **4.44e-16**. It verified all 48 original execution files and the 13 scientific source files against the prospectively published source commit. This verifies saved-output arithmetic and provenance, not an independent replay of training.

The [execution metadata](../output/dialogue-copy-v1/execution-01/manifest.json) binds the original local weights and predictions by hash; those payloads remain local. [Reproduction instructions](dialogue-copy-reproduction.md) provide the complete data-to-report path. The public source is [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/e852981ae34990f4358979625854259302feaa78), licensed CC BY-SA 4.0. The supplied-schema boundary, human-paraphrased simulated dialogues, frozen transformer's unknown pretraining overlap, repeated development exposure, and small revision sample limit the claim.

## Implication for the next experiment

Keep the present scalar model with its shared lexical/history inputs as a stronger reference for future work. The evidence does not isolate an exact-storage advantage or justify further claims for this selective-retention rule. A useful next hypothesis must address the remaining correction failures and beat that simple reference under a new frozen comparison. Any untouched confirmation needs a separately specified protocol. This study does not establish RL, connectome learning, a recurrent world model, calibrated uncertainty, or architecture novelty.
