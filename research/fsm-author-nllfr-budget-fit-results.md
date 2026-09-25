# Larger-budget reference: training complete

The original fresh NL-LFR fit reached the author's small-change stopping rule at
**51,065 iterations**, within its registered 100,000-iteration limit. The process
finished in **8,014.90 seconds**. The [independent FIT audit](../output/fsm-author-nllfr-budget-audit-v1/audit.json)
passes with `FIT_ONLY_COMPLETE`: all first 10,000 loss entries match the original
run exactly, and both endpoint objectives replay independently.

| Native training objective | Loss |
| --- | ---: |
| Common initialization | 294.125218 |
| Original 10,000-iteration cap | 53.197659 |
| New 51,065-iteration stopping point | 33.969512 |

This is a training result. The stopping flag is not a stationarity or global
optimum certificate, and lower training loss does not establish better forecasts.
There were no development forecasts or model-selection calls in this fit.

The [comparison registration](fsm-author-nllfr-factorial-registration.json) now
binds both audited checkpoints, 37 qualified producer sources and 19 prerequisites.
The [independent auditor freeze](../output/fsm-author-budget-factorial-engineering-v1/auditor-freeze.json)
binds its four sources separately. Registration is published before one original
evaluation of both checkpoints at 16 and 64 context-estimation steps, following
the unchanged [four-condition protocol](fsm-author-nllfr-factorial-protocol.md).

The candidate's earlier exposed-development advantage remains provisional until
that comparison and its independent audit close. This result establishes no
architecture advantage, untouched generalization or useful sensitivity penalty.

[Original process and audit observation](../output/fsm-author-nllfr-budget-engineering-v1/fit-completion-observation.json)
and [registration assembly](../output/fsm-author-budget-factorial-engineering-v1/registration-assembly.json).
