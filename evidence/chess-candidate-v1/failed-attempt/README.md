# Candidate chess v1: failed attempt

This attempt failed with `Broken pipe` after **128 recorded optimizer updates** of the first fit, `action_only-97`. There are **zero complete checkpoints and zero neural evaluations**. This is a retained failed attempt, not a chess-performance result.

The journal records 16,384 training presentations and 484,721 candidate evaluations. The first progress print occurs immediately after flushing update 128, which is consistent with the error. The receipt does not contain a traceback, so that callsite is an inference.

Fresh-data generation completed: 4,096 positions, 4,756 Stockfish calls, 9,512,000 requested nodes and 9,497,297 reported nodes. The training cache was built for 32,768 roots and 968,036 legal successors. Native baseline summaries exist; trained-model predictions, latency measurements, stronger-engine grading and games do not.

`execution.tar.gz` preserves all 13 execution files losslessly. `manifest.json` records every original file's SHA-256 and byte size. The archiver verified each decompressed member against the original bytes and checked that the execution remained unchanged. `summary.json` retains the failed attempt's partial work and data costs.

Any recovery needs a separate, explicit protocol amendment. The failed attempt must remain in cumulative accounting, including these 128 extra updates. No checkpoint exists to resume. Reusing the completed evaluation data requires binding the same bytes and disclosing that reuse; it does not justify claiming newly generated data or dropping this failure.
