# Training and inference budget comparison: harness qualified

The complete four-condition evaluator and independent auditor pass **95 new
fabricated tests plus lint**, in addition to the [71 arithmetic tests](fsm-author-budget-factorial-qualification.md)
already published. This is implementation evidence, not a new forecasting result.
The original larger-budget fit is still running. Its final checkpoint, successful
process closure and independent audit must be available before registering and
launching the comparison.

The [fixed protocol](fsm-author-nllfr-factorial-protocol.md) compares the original
and fresh longer-trained checkpoints, each with 16 and 64 state-estimation
directions. It separates extra training from extra prediction-time work, keeps
all four conditions and preserves failures or worse forecasts.

| Qualification | Passing tests | Evidence |
| --- | ---: | --- |
| Evaluator and original-process supervisor | 39 | [Original receipt](../output/fsm-author-budget-factorial-engineering-v1/full-producer-qualification-01/receipt.json) |
| Independent auditor | 55 | [Corrected-fixture receipt](../output/fsm-author-budget-factorial-engineering-v1/full-auditor-qualification-02/receipt.json) |
| Producer files consumed by the complete auditor | 1 | [Original receipt](../output/fsm-author-budget-factorial-engineering-v1/full-cross-schema-qualification-01/receipt.json) |

The final integration test writes actual fabricated NPZ and JSON files through
the evaluator, then passes them through the completed audit. It exercises all
1,536 forecast slots, 48 record slots, four warmups, 96 timed slots and 312
unchanged reference banks. Numerical inference and parent admission are explicit
stubs; scoring, runtime-record validation, file schemas, parity checks, storage,
continuation rules and complete output inventories remain active. The separate
arithmetic tests cover the real numerical implementations.

Source review caught a runtime-record format mismatch and missing native-unit
overflow handling in the new auditor before qualification. Its first test run
then passed 54 tests and failed one boundary fixture: a one-ULP increase over the
threshold rounded back to the threshold during averaging. Only that fixture was
corrected, with an explicit check of its reduced mean. The scientific rule and
auditor were unchanged. [The original failure and correction remain recorded](../output/fsm-author-budget-factorial-engineering-v1/full-auditor-qualification-observation.json).

## What remains before a result

The [larger-budget fit](fsm-author-nllfr-budget-protocol.md) must close once, with
its original files and training sources unchanged. Its audit must establish
finite validated output and exact agreement with the original first 10,000 loss
entries. A prefix mismatch stops this comparison before development evaluation.
A finite fit that reaches its cap can support only an incomplete diagnostic.

Next, register the exact checkpoints and audit closures, publish that registration,
and run one supervised comparison. Both new-checkpoint policies remain eligible
accuracy controls when their forecasts and training are complete. Missing timing
evidence blocks overall continuation without removing a strong accurate control.
All four existing continuation rules remain unchanged.

The prepared [FIT-audit launcher](../output/fsm-author-nllfr-budget-engineering-v1/audit-original.py)
rejects an active training process. Its [qualification record](../output/fsm-author-nllfr-budget-engineering-v1/audit-wrapper-qualification-02/receipt.json)
also retains a [correction to an earlier premature lint claim](../output/fsm-author-nllfr-budget-engineering-v1/audit-wrapper-prefit-check-correction.json).
No empirical audit was launched by that check.

The [registration helper](../output/fsm-author-budget-factorial-engineering-v1/prepare-registration.py)
is also ready. It binds the exact audited checkpoints, the 37 qualified producer
sources and a separate four-source auditor freeze. Four fabricated guard tests
and an actual rejection of the still-running fit passed; no registration was
created. Its [preparation record](../output/fsm-author-budget-factorial-engineering-v1/registration-preparation.json)
records the reviewed source and checks. Successful assembly remains conditional
on both original parent processes closing successfully.

The [factorial audit launcher](../output/fsm-author-budget-factorial-engineering-v1/audit-original.py)
passes [14 focused guard and subprocess checks](../output/fsm-author-budget-factorial-engineering-v1/audit-wrapper-qualification-01/receipt.json).
It runs one audit child with a one-hour limit, retains failures and rejects a
second attempt. Closed incomplete evaluator outcomes remain auditable, while
missing or live prerequisites prevent launch. A successful audit of incomplete
evidence remains scientifically incomplete. These checks use fabricated metadata
and tiny stub children; the empirical factorial audit has not run.

No new development or reserved-data evaluation, training restart, candidate
selection or performance claim is part of this qualification.

[Source and qualification closure](../output/fsm-author-budget-factorial-engineering-v1/full-harness-qualification-closure.json)
