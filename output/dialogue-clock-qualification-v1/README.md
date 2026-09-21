# Suspend-inclusive clock qualification

The old training attempt exposed a difference between awake elapsed time and
time including system sleep. These checks qualify new elapsed-time and process
supervision code; they do not establish a model result.

| Component | Check | Result |
| --- | --- | --- |
| Clock helper | 24 injected-clock and backend-binding tests | Passed in 0.07 s |
| Supervisor | 13 lifecycle tests, including a native no-op process | Passed in 0.37 s |
| Worker | 42 synthetic lifecycle and tiny CPU-route tests | Passed in 2.05 s |

The native no-op smoke ran under `/usr/bin/caffeinate -i`. Its wrapper exited 0
and was confirmed absent; the worker process group also disappeared. The native
parent elapsed interval was **0.037140334 seconds**. No actual system sleep was
induced, so native sleep/wake behavior is supported by the OS contract and
injected-clock tests, not a hardware sleep experiment.

The helper and supervisor each initially had lint findings. Those attempts
remain preserved. Annotation/comment-only production corrections were followed
by lint checks with unchanged behavioral ASTs; passed numerical tests and the
native smoke were not repeated.

- Helper: [initial preflight](preflight-01/receipt.json),
  [final lint and AST check](preflight-02/receipt.json).
- Supervisor: [initial preflight](supervisor-preflight-01/receipt.json),
  [final lint, AST check and preserved native receipts](supervisor-preflight-02/receipt.json).
- Worker: [qualification and limits](../dialogue-observation-learning-v2/README.md).

All original 53 scientific source files remain unchanged. A new attempt requires
its own complete source freeze, allocation, real terminal evidence and all
twelve fits. A failed clock never authorizes fallback to civil or awake time.
