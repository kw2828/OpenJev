# Terminal packet correction for the saved diagnostic

The first saved diagnostic stopped before producing rows or a summary. Its reader required a nonempty legal-action list after a completed teacher episode. The native interface correctly returns an empty list at termination. The chronological join encounters this packet while reading through TRAIN episodes to reach the fixed VALID episode.

[Original failure witness](../output/otto-return-consistency-v1/execution-witness.json) records session 79651, completion chunk `38fc0d`, exit 1. Original source, tests, design, plan and failed output remain unchanged. This is a reader defect, not a failed learning experiment or new numerical result.

## Exact correction

Version two uses a separate, explicit adapter around the pinned original diagnostic. Only public-packet eligibility changes:

- A completed packet requires hit `-2` and `valid_actions=[]`.
- A nonterminal packet retains the original nonempty, sorted, exact in-bounds legal-action list.
- Grid position, step, hit/done type checks, chronological joins and posterior identities remain enforced.

No terminal record is skipped. No new row is selected. The target cohort remains all 144 saved model-prefix rows from steps 0 through 7 of the same two teacher episodes. Physical units, branch floors, scalar calculations, strict near-tie selection, summaries and interpretation limits are unchanged from the [original design](otto-return-consistency-design.md).

The adapter authenticates its inherited source before import. The new plan pins all eight sources: the original five, new adapter, new focused tests and this correction design. It pins the same five completed scientific input records, the current runtime, and an exclusive `otto-return-consistency-v2` output. The original 180-second, 2-GiB RSS and 64-MiB output limits apply independently to the new diagnostic. There are still no model, training, posterior-replay or simulator calls.

Qualify the actual terminal-empty contract, rejection of terminal-nonempty and nonterminal-empty packets, chronological traversal through a completed TRAIN episode, and unchanged arithmetic across all 144 synthetic rows before freezing and executing version two. Preserve any new failure without changing its output.

This correction changes no condition or result in the completed scalar-return study. Any successful diagnostic remains descriptive evidence from two episodes, not autonomous improvement or an architecture admission.
