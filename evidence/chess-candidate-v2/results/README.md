# Candidate-conditioned chess development evidence

Engine-score gate: **FAILED**. Game gate: **FAILED**. Combined continuation: **FAILED**.

[Summary](summary.json) · [Plan](plan.json) · [All raw evidence](execution-and-report.tar.gz) · [All twelve checkpoints](../../../models/chess-candidate-v2/README.md) · [Hash manifest](manifest.json)

The archive preserves every execution and report file, including evaluation positions, generator traces, raw engine calls, training logs, predictions, original weights and all 288 game JSON/PGN traces. Derived cache metadata and fingerprints are retained; the temporary float feature buffers are not archived. Unfinished or failed games retain unresolved points; they are never relabeled draws. This is a development mechanism comparison on known generators, conditional on three training seeds. Native one-ply consequences are not learned dynamics. Mapping corruption remains diagnostic only. No Elo estimate, new architecture or novelty claim is established.

This is the separately frozen recovery of an output-pipe failure. All twelve models start from fresh initialization. The failed attempt's 128 optimizer updates were discarded and count as study overhead, not extra training of a published model. The same 4,096 previously generated evaluation positions are reused byte-for-byte, with no earlier neural predictions; they are not a second fresh sample. Teacher calls and data-generation time are counted once. [Preserved failed attempt](../../chess-candidate-v1/failed-attempt/README.md).
