# Observation learning with corrected elapsed-time accounting

This directory is a separate version of the matched twelve-fit comparison.
The [fresh twelve-fit execution completed](../../research/dialogue-observation-learning-v2-status.md)
and passed technical validation. **Scientific continuation failed 6/7:** unseen
macro accuracy improves, but log loss worsens in every paired seed.
[Results and figures](../../research/dialogue-observation-learning-v2-results.md)
include every arm and seed. The [complete report](report-01/report.md) and
[independent scoring audit](result-audit-01/result-02/receipt.json) agree.
The [first attempt remains failed](../../research/dialogue-observation-learning-timing-failure.md).
Its five completed fits and partial sixth are not scored, resumed or used to
initialize this version.

The [prospective protocol](../../research/dialogue-observation-learning-protocol-v2.md)
keeps the original science and replaces inconsistent elapsed timers with a
shared suspend-inclusive deadline. The separate allocation retains an eight-hour
limit. A temporary idle-sleep assertion lasts only while the supervisor runs;
deadline correctness does not depend on that assertion preventing every sleep.

Worker qualification: **42 synthetic tests passed**, including tiny CPU model
fixtures and injected clocks. The eight scientific/data-route functions remain
AST-identical to V1. [Initial receipt](worker-preflight-01/receipt.json) and
[lint correction and equivalence receipt](worker-preflight-02/receipt.json)
preserve the actual attempts. This is implementation evidence, not task quality.

The [clock qualification](../dialogue-clock-qualification-v1/README.md) records
the helper and supervisor tests, including the real no-op process smoke.
The V2 reporter has passing evidence for **47 synthetic cases**, and the
independent auditor for **28 cases**. Initial clock-fixture read-count mistakes
and their targeted corrections remain in the
[reporter](report-preflight-02/receipt.json) and
[auditor](audit-preflight-02/receipt.json) preflight histories. Production scoring
was unchanged by those fixture corrections. Peer source review is clear.
The [source review](scientific-source-review-01/receipt.json) binds all 64 files
and verifies unchanged ASTs for 21 scientific and scoring functions. The
separate metadata freeze and its independent audit passed before training;
actual execution receipts are linked in the status page.

The [actual result figures](figure-01/render-01/receipt.json) require both complete
V2 report and audit receipts and retain the failed scientific condition. Source,
qualification logs, render receipts and rejection receipts are published;
invented report/audit fixtures and synthetic images stay local. Their hashes
remain in the preserved receipts.

The [first audit launch](result-audit-01/result-01/failed.json) selected base Python
by resolving the virtual-environment interpreter symlink and failed on NumPy
import before reading any saved inputs. Its exact invocation is preserved.
The [prospective launcher correction](result-audit-01/launcher-correction-01.json)
used the unchanged auditor, inputs and limits through `.venv/bin/python` in a
new output directory. That first actual audit evaluation completed successfully;
it did not repeat training or change the scientific rule.

Publish plans, allocations, receipts, aggregate reports and original source
files. Keep duplicate frozen source snapshots, reversible actor tokens, target
rows, lexical arrays, individual predictions and fitted weights local. Manifests
bind local payloads regardless of Git membership. The unchanged preparation
comes from the Schema-Guided Dialogue dataset under CC BY-SA 4.0; its attribution
and publication terms remain in the [V1 evidence note](../dialogue-observation-learning-v1/README.md).
