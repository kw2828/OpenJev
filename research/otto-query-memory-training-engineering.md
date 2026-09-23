# Query-memory training and audit qualification

This page tracks implementation checks for the frozen
[query-memory experiment](otto-query-memory-protocol.md). Fabricated tests and
capacity estimates do not establish that the memory mechanism improves decisions.

The new training path uses the complete TRAIN census, accumulates gradients over
chronological chunks, and updates parameters once per batch of six complete
episodes. Each fit seed has one common pretrained model and five branches. The
memory branches preserve all eight slow-model tensors and optimize only the
224-parameter key projection. Final checkpoints and all TRAIN predictions must
be saved before DEV is decoded.

The producer reports a pending continuation decision. A separate audit checks
the original process completion, file identities, saved predictions and metrics.
It recomputes the DEV conditions without calling a model, optimizer, teacher or
simulator. TEST stays unused unless this audit and its original supervisor both
complete successfully and every DEV condition passes.

The capacity screen measures fabricated batches through the actual training
kernel. It uses only completed TRAIN episode lengths from the collection to
estimate the fixed training workload. Its estimate includes a twofold multiplier
and 600 seconds of overhead; admission requires at most 16,200 seconds against
the unchanged 21,600-second training limit. This remains a heuristic, not a
runtime guarantee.

Source review caught a runtime mismatch before execution: the capacity checker
initially attempted to authenticate a historical TensorFlow run against the
active PyTorch interpreter. The revised checker verifies the saved native
runtime and completed run separately from the current training runtime.

All **528 fabricated tests and lint pass**. The final qualification binds 39
source/configuration files, with matching hashes before and after execution.
The first attempt passed every test but reported nine style findings; those were
corrected before a complete second qualification.

| Attempt | Tests | Lint | Scalar audit estimate |
| --- | --- | --- | --- |
| [01](../output/otto-query-memory-training-engineering-v1/attempt-01/receipt.json) | 528 passed | 9 findings | 27.82 seconds |
| [02](../output/otto-query-memory-training-engineering-v1/attempt-02/receipt.json) | 528 passed | Passed | 22.87 seconds |

The scalar estimate extrapolates one fabricated 13,128-row view to 3,780,864
rows with a twofold multiplier. Both estimates are below the 480-second scalar
allowance, with 120 seconds separately reserved for metadata and file checks
inside the unchanged 600-second audit limit. These are engineering estimates,
not scientific speed results. They do not replace the separate training-capacity
screen, which requires the completed collection's TRAIN lengths.

Source review also tightened admission: every auditor component must be present
in the authenticated qualification before any fitting. The audit independently
derives operation counts from the fixed schedules and checks recorded batch
totals against them. Model inference, gradients, physical timing and the recorded
training/AUX losses remain authenticated producer evidence; the audit does not
replay them.

No scientific training or DEV evaluation has run at this publication.
