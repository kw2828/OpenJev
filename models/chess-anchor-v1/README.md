# OpenJev anchor chess checkpoints

License: MIT. All 12 final checkpoints are retained.

This is a fresh-initialization supervised ablation on fixed Stockfish-labeled positions. Residual and fixed-encoder-anchor recurrence share the same architecture, heads and seed-specific initial tensors. Fixed training uses depth 4; mixed training uses equal counts at depths 2, 4 and 6. There is no cross-move memory, search, reinforcement learning, Elo estimate or novelty claim.

Each model has 33,185 active parameters and 43,726 stored parameters. The auxiliary head is unused. Training uses legal-move cross-entropy plus 0.5 bounded-value MSE. These are original final weights, without selection or a promoted winner alias.

Frozen plan SHA-256: `5b578800cab3d1b0d07fb776ee3b9a3b56fe8fd12b57b301886511acbf7dfaf2`.
Anchor source SHA-256: `791be15f6b20bb40c2ff075ef824b4c725887592d32b966fe1fef38ac115210b`.
Spatial initializer source SHA-256: `49e6e53d33cd52511c0e4a08074b9a7f0dc2244727df2879f3fb7fd1aa886ab4`.

| Checkpoint | Recurrence | Training depths | Seed | Paired residual control |
| --- | --- | --- | --- | --- |
| [anchor_fixed-17](anchor_fixed-17/weights.pt) | anchor | 4 | 17 | residual_fixed-17 |
| [anchor_fixed-29](anchor_fixed-29/weights.pt) | anchor | 4 | 29 | residual_fixed-29 |
| [anchor_fixed-43](anchor_fixed-43/weights.pt) | anchor | 4 | 43 | residual_fixed-43 |
| [anchor_mixed-17](anchor_mixed-17/weights.pt) | anchor | 2 / 4 / 6 | 17 | residual_mixed-17 |
| [anchor_mixed-29](anchor_mixed-29/weights.pt) | anchor | 2 / 4 / 6 | 29 | residual_mixed-29 |
| [anchor_mixed-43](anchor_mixed-43/weights.pt) | anchor | 2 / 4 / 6 | 43 | residual_mixed-43 |
| [residual_fixed-17](residual_fixed-17/weights.pt) | residual | 4 | 17 | residual_fixed-17 |
| [residual_fixed-29](residual_fixed-29/weights.pt) | residual | 4 | 29 | residual_fixed-29 |
| [residual_fixed-43](residual_fixed-43/weights.pt) | residual | 4 | 43 | residual_fixed-43 |
| [residual_mixed-17](residual_mixed-17/weights.pt) | residual | 2 / 4 / 6 | 17 | residual_mixed-17 |
| [residual_mixed-29](residual_mixed-29/weights.pt) | residual | 2 / 4 / 6 | 29 | residual_mixed-29 |
| [residual_mixed-43](residual_mixed-43/weights.pt) | residual | 2 / 4 / 6 | 43 | residual_mixed-43 |

Load with `AnchorChess.load(path, expected_plan_sha256=..., expected_recurrence=..., expected_seed=...)`. The distinct checkpoint format rejects old spatial models.
