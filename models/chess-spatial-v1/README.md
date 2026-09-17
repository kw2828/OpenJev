# OpenJev spatial chess checkpoints

Twelve original supervised chess models, released under MIT. Modes: `cnn`, `recurrent`, `reconstruct`, `predict`; seeds: `17`, `29`, `43`. All final fits are included. `predict-17` is the first declared seed, not a selected winner.

See [loading instructions, measured results and limitations](../../docs/chess-spatial.md). The [publication manifest](../../evidence/chess-spatial-v1/results/manifest.json) binds each checkpoint to its training receipt and exact weight hash. CPU inference uses `SpatialChess.load`; it needs no engine process.

These models score supplied legal moves with an uncalibrated softmax. Their value estimates are transformed engine scores, not winning probabilities. No Elo, calibrated confidence, biological connectome, persistent game memory or novel planning algorithm is established.
