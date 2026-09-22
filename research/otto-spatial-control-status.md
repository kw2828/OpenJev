# Spatial control: evaluation complete, audit running

The original fixed evaluation completed **all 1,152 episodes**, with exit code 0, no timeout and successful process cleanup. This establishes execution completion. The performance verdict remains pending until the separately counted saved-output audit finishes.

The [frozen protocol](otto-spatial-control-protocol.md) retains fifteen final learned models and analytic control over 72 paired environmental cases. It permits no replacement seeds, checkpoint selection or extension after seeing the results. No new training or external model calls occurred in this evaluation.

The worker completed 1,155 environment resets, 1,738,066 environment steps, 1,736,024 learned value calls, 2,042 analytic decisions and 15 model loads. Every attempted operation returned. Worker time was 9,850.384311 seconds; the enclosing supervisor measured 9,852.806477 seconds. Peak recorded worker memory was 681,590,784 bytes. These counts and enclosing intervals are execution records, not comparative efficacy or isolated latency claims.

The full saved-output audit was dispatched once against the closed original outputs. It checks source and checkpoint identities, public-state updates, recorded decisions, complete episode accounting and the frozen comparisons. It makes additional, separately counted saved-checkpoint calls, with no training or new environment collection. Its neural algebra is shared with the qualified NumPy deployment implementation; the geometry, filtering and metric reconstruction are independently implemented. No audit completion is claimed in this status snapshot.

The results renderer and evidence packer passed **37 fabricated tests** and Ruff. These checks qualify reporting infrastructure; they do not test policy effectiveness. The full renderer and packer have not yet been run on these empirical results.

| Record | Evidence |
| --- | --- |
| Frozen source and input identities | [Plan](../output/otto-spatial-control-v1/plan-01.json) |
| Original successful execution | [Worker receipt](../output/otto-spatial-control-v1/run-01/receipt.json), [supervisor terminal](../output/otto-spatial-control-v1/process-01.terminal.json) |
| Tool observations and audit dispatch | [Lifecycle witness](../output/otto-spatial-control-v1/worker-completion-audit-dispatch-01.json) |
| Reporting checks | [Engineering receipt](../output/otto-spatial-control-v1/publication-engineering-01/receipt.json) |

The earlier [scalar fitting result](otto-spatial-study-results.md) and [deployment qualification](otto-spatial-control-qualification.md) retain their original scopes. No connectome, recurrent-model or other novel architecture advantage is established by this execution status.
