# A trained starting point does not rescue belief-guided pooling

21 September 2026. **The complete warm-start study failed its continuation rule:
13 of 32 conditions passed.** Belief-guided attention has lower mean unseen
macro accuracy and worse probability scores than every control. This is a
competent-start negative result, not just the cold pilot's failure to learn.

![All fifteen results across three paired seeds](../output/dialogue-warm-pooling-v1/figure-01/render-01/warm-pooling.png)

## Complete result

Equal means over all three seeds, on the same 33,093 unseen-service DEV endpoints,
including 2,543 changed decisions. Macro accuracy equally averages unmentioned
retention, assigned retention and changed accuracy. NLL and multiclass Brier
average endpoints; lower is better. No calibration or probability repair.

| Method | Unseen macro accuracy | Changed accuracy | NLL | Brier |
| --- | ---: | ---: | ---: | ---: |
| Untouched trained model | 79.3402% | 80.1940% | 0.8272065 | 0.3387461 |
| Pooled continuation | 79.4154% | 80.3644% | 0.8443938 | 0.3395880 |
| Schema attention | 79.5686% | 80.3644% | 0.8423191 | 0.3366751 |
| State token | 79.3901% | 80.0629% | 0.8472903 | 0.3385357 |
| Belief-guided attention | 79.3097% | 80.3251% | 0.8496693 | 0.3408533 |

Belief guidance trails pooled continuation by **0.106 percentage points** and
schema attention by **0.259 points**. Its NLL and Brier are worse than both of
those controls in every seed. Changed accuracy improves only 0.131 points over
the untouched model and 0.262 points over the state token, below the required
one-point gains; it trails pooled and schema attention on that measure too.

The thirteen passing conditions are twelve seen/retention tolerance checks and
one positive-in-two-seeds macro check against the untouched model. **None of
the required mean accuracy gains or proper-score comparisons passes.** These
are three training seeds on shared exposed development examples, not three
independent test sets. Schema attention's descriptively highest macro accuracy
is not a new architecture win: its mean NLL also exceeds the untouched model.

## What the stronger experiment established

The [small cold pilot](dialogue-belief-pooling-pilot-results.md) used fresh heads,
a pretrained frozen encoder and much less data; all fits scored zero changed
accuracy. This follow-up instead restored all three original trained encoder
and scalar-memory checkpoints. It used all 2,017 original TRAIN dialogues and
all 2,363 DEV dialogues, preserving the original three-stratum weighted loss.

For each checkpoint, the untouched cached scalar and all four zero-initialized
attention variants reproduced **every one of the 62,329 original DEV predictions
exactly**: zero log-probability differences, answer changes or top-tie changes.
All fifteen complete replays passed before any optimizer was created. This
confirms that initial numerical changes did not explain the comparison.

Every adapted arm then trained for five epochs with a fresh matched optimizer,
320 updates per fit. Encoders stayed frozen at their trained weights. All twelve
final fits were evaluated, without early stopping or checkpoint selection.
Attention arms share parameter dimensions and forward geometry; their backward
paths differ. Pooled continuation is a cheaper reference with unused attention
parameters explicitly recorded. Training from restored weights is not exact
optimizer resumption.

The belief-query arm uses its own full candidate-probability-weighted embedding
summary to steer token reads. Schema attention uses a uniform candidate mean;
the state-token arm puts the belief summary in an extra pooling key/value.
Every head already receives its own previous probability and entropy. No gold
prior state enters the actor. This comparison tests where state enters pooling
after a frozen encoder, not every possible form of recurrent contextualization.

## Decision and next direction

**Close this pooling recipe.** The better initialization removed a major
limitation of the pilot but did not reveal an advantage. Do not add epochs,
select the favorable seed or use calibration to reverse the fixed decision.
These results do not establish that belief feedback generally fails, or that
any larger recurrent architecture is needed.

The [source review of possible mechanisms](dialogue-warm-pooling-next-mechanisms.md)
distinguishes internal encoder feedback from predictive fast state, with strong
prior art and explicit controls. Neither becomes the next experiment merely
because pooling failed. Existing history audits also provide weak evidence of
a long-memory bottleneck on this supplied-candidate task.

The next research step should qualify a task where earlier observations change
the correct decision despite identical recent observations, and actions change
future observations or rewards. A strong explicit-state/filter reference and a
trained GRU must be included before attributing gains to biological wiring,
predictive memory or a new recurrent architecture. Official SGD TEST remains
unopened; no novelty, transfer or ICLR-readiness claim follows from this study.

## Execution and independent verification

The complete native supervised run finished in **1,301.172 seconds (21.69
minutes)**, including all loading, three caches, fifteen qualification paths,
121,020 training-dialogue visits, 28,356 adapted evaluation-dialogue visits and
process cleanup. The fixed allocation was 7,200 seconds; there was one complete
run and no restart. These cached development timings are not serving latency.

There were **45 model/runner checks** and **72 reporting/metric checks**. The
report completed with an actual zero exit code, followed by a separate audit
with a zero exit code. The audit independently recomputed **180 metric cells,
45 macro summaries and all 32 conditions**. All **2,523 scalar checks** agreed;
maximum numerical difference was 5.56e-17. Its provenance authentication reuses
the qualified reader. Secondary service/type/recovery tables and runtime counts
are authenticated, but not independently recomputed. Initialization parity is
producer evidence, not an additional independent neural replay.

Five reporter lint findings were corrected before its final source freeze and
the 72 tests were rerun. No changes were made to the frozen experiment or
scoring arithmetic after outcomes were opened. The result figure was rendered
and visually checked; all fifteen primary results remain visible.

- [Prospective protocol](dialogue-warm-pooling-protocol.md) and [source/data/checkpoint plan](../output/dialogue-warm-pooling-v1/freeze-01/plan.json).
- [Actual run exit](../output/dialogue-warm-pooling-v1/run-evidence-01/actual-exit.json) and [complete execution manifest](../output/dialogue-warm-pooling-v1/run-evidence-01/completed.json).
- [All fifteen scores and thirty-two conditions](../output/dialogue-warm-pooling-v1/report-01/report.md).
- [Independent numerical audit](../output/dialogue-warm-pooling-v1/audit-01/result-01/summary.json).
- [Exact plotted values and figure receipt](../output/dialogue-warm-pooling-v1/figure-01/render-01/receipt.json).

Raw checkpoints, trained feature caches and predictions remain local under
`runs/dialogue-warm-pooling-v1/run-01`; public artifacts contain their hashes,
counts and complete scored summaries.
