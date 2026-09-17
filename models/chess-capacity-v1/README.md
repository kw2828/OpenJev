# OpenJev capacity chess checkpoints

License: MIT. All six final checkpoints are retained.

These residual spatial policies use four internal updates on a fully visible board. Training uses 98,304 fixed Stockfish-labeled positions, eight epochs, legal-move cross-entropy and 0.5 bounded-value MSE. There is no cross-move memory, search, RL, Elo estimate or novelty claim.

Width 32: 33,185 active / 43,726 stored parameters. Width 128: 439,073 active / 591,790 stored. The unused auxiliary head remains in both checkpoints. Data, batches, optimizer updates and depth are matched; parameter count, FLOPs and wall time are not. Equal seed numbers do not imply identical tensors across different widths.

Frozen plan SHA-256: `19f9be2a3c03fd2c4cab15ee87096b5fc0e883cd349625a4c0f0ea5b11e4dacb`.

| Checkpoint | Width | Seed |
| --- | --- | --- |
| [width128-53](width128-53/weights.pt) | 128 | 53 |
| [width128-67](width128-67/weights.pt) | 128 | 67 |
| [width128-83](width128-83/weights.pt) | 128 | 83 |
| [width32-53](width32-53/weights.pt) | 32 | 53 |
| [width32-67](width32-67/weights.pt) | 32 | 67 |
| [width32-83](width32-83/weights.pt) | 32 | 83 |

Load with `AnchorChess.load(path, expected_plan_sha256=..., expected_recurrence="residual", expected_seed=...)`. Weights are copied byte for byte from the completed execution; no winner alias or selected checkpoint is substituted.

Source hashes:
- `scripts/chess_capacity_study.py`: `d3ab4f96aa86915161bef08c2cf9f572441ce8bf8b69bdd686cb58b30e5f245e`
- `scripts/chess_compute_study.py`: `3f3495f75a232cce52ee5b570dccc1c98c833086f46963272bc98b653ec66582`
- `src/openjev/research/chess_anchor.py`: `791be15f6b20bb40c2ff075ef824b4c725887592d32b966fe1fef38ac115210b`
- `src/openjev/research/chess_anchor_data.py`: `3bc6d724ab5448aa7d859f1b392697e89b7dea625fdbe10f35f0a907976fc85c`
- `src/openjev/research/chess_anchor_eval.py`: `c9069096a805d867e780f214528f5c2340b796238ae5433e5de9dc80c0d45210`
- `src/openjev/research/chess_arena.py`: `a9a5be6ccb982d2c43b4365a0f0e9dd68c3a8c4fb11ea8c5cf950076b7046702`
- `src/openjev/research/chess_capacity_arena.py`: `86dde83de5b85333f79011e2d023265f33c81bf68eed4d80dff3262e6c5d887e`
- `src/openjev/research/chess_capacity_data.py`: `1c4a3fdf0853544a0f4dd2ee34ea495d02245bee150b9ae6404b77ec773d6ad3`
- `src/openjev/research/chess_compute.py`: `a1ec1a4618a92b4dc37a9a8f510f5f37ba45081e9e0db0d89714c689f88f537f`
- `src/openjev/research/chess_spatial.py`: `49e6e53d33cd52511c0e4a08074b9a7f0dc2244727df2879f3fb7fd1aa886ab4`
- `src/openjev/research/chess_spatial_baselines.py`: `9824212fdfe5678fafe2591eeb5f3962790713bd1b4dece4803648b696ae8942`
- `src/openjev/research/chess_spatial_data.py`: `70d6e94799cd59c7e4e097eb783a44e3288cad903fb7d4637e3fabd86003cf3e`
- `tests/test_chess_capacity_arena.py`: `6b293aa01d14bf38b2a42f0329c8308de6973bcca928f5a3afe7ec6cf20aeb44`
- `tests/test_chess_capacity_data.py`: `ffc4b52c36e0b22769db5710995b31818133a832a3fd0935603ce1a90119bf7c`
- `tests/test_chess_capacity_study.py`: `2d6270113c5ca81b2d0c2b5262f5202c507cff1b138df153fa564f17acb6cc1c`
