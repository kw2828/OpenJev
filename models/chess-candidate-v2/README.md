# OpenJev candidate chess checkpoints

License: MIT. All twelve final fits are retained.

Four width-32 controls share root initialization, 32,768 training positions, six epochs and 1,536 updates. Root depth is four; refinement depth is two. Direct has 33,185 active parameters, each refinement arm 33,313; all store 43,854 including unused heads. Parameters, FLOPs and wall time are not matched.

Delta and full-afterstate policies receive native one-ply consequences in the root player's perspective. These are not learned dynamics, multi-ply search, RL or memory between moves. No Elo or novelty claim follows. All scores are uncalibrated candidate-relative softmax values.

Frozen plan SHA-256: `0cdcf5b9f9e64b5dde6f0a951fa4249475d050470e9036ce5c8ab76cf181c106`.

| Checkpoint | Arm | Seed |
| --- | --- | --- |
| [action_only-109](action_only-109/weights.pt) | action_only | 109 |
| [action_only-127](action_only-127/weights.pt) | action_only | 127 |
| [action_only-97](action_only-97/weights.pt) | action_only | 97 |
| [delta-109](delta-109/weights.pt) | delta | 109 |
| [delta-127](delta-127/weights.pt) | delta | 127 |
| [delta-97](delta-97/weights.pt) | delta | 97 |
| [direct-109](direct-109/weights.pt) | direct | 109 |
| [direct-127](direct-127/weights.pt) | direct | 127 |
| [direct-97](direct-97/weights.pt) | direct | 97 |
| [full_afterstate-109](full_afterstate-109/weights.pt) | full_afterstate | 109 |
| [full_afterstate-127](full_afterstate-127/weights.pt) | full_afterstate | 127 |
| [full_afterstate-97](full_afterstate-97/weights.pt) | full_afterstate | 97 |

This is the separately frozen recovery of an output-pipe failure. All twelve models start from fresh initialization. The failed attempt's 128 optimizer updates were discarded and count as study overhead, not extra training of a published model. The same 4,096 previously generated evaluation positions are reused byte-for-byte, with no earlier neural predictions; they are not a second fresh sample. Teacher calls and data-generation time are counted once.

[Preserved failed attempt](../../evidence/chess-candidate-v1/failed-attempt/README.md).

Load with `CandidateChess.load(path, expected_plan_sha256=..., expected_arm=..., expected_seed=..., expected_width=32, expected_root_depth=4, expected_branch_depth=2)`. All original bytes are copied; no winner alias or selected seed.

Source hashes:
- `scripts/chess_candidate_recovery.py`: `3d968cf761e25625a1d0ae4f764c3d0f4bf6ac6fdeddf98cb16c5df0e939b3b6`
- `scripts/chess_candidate_study.py`: `aa192ee3a93da336afb4f6e27eb4ad57059d7a7acfcf2e2874924764b0d2e3d6`
- `scripts/chess_compute_study.py`: `3f3495f75a232cce52ee5b570dccc1c98c833086f46963272bc98b653ec66582`
- `scripts/preflight_chess_candidate.py`: `c84b49b54acac197473337c56bec82d556471867aa0a544632e78d91169c6c51`
- `scripts/preflight_chess_candidate_128.py`: `251764c844b2fcc8d33530548db7ce6f3f009f82d520b78a59f6ed4c6df3c436`
- `src/openjev/research/chess_anchor.py`: `791be15f6b20bb40c2ff075ef824b4c725887592d32b966fe1fef38ac115210b`
- `src/openjev/research/chess_anchor_data.py`: `3bc6d724ab5448aa7d859f1b392697e89b7dea625fdbe10f35f0a907976fc85c`
- `src/openjev/research/chess_anchor_eval.py`: `c9069096a805d867e780f214528f5c2340b796238ae5433e5de9dc80c0d45210`
- `src/openjev/research/chess_arena.py`: `a9a5be6ccb982d2c43b4365a0f0e9dd68c3a8c4fb11ea8c5cf950076b7046702`
- `src/openjev/research/chess_candidate.py`: `3f09247154d5ed47d300c00fb71729a3808d4504b344af62423571b8f12cd3fa`
- `src/openjev/research/chess_candidate_arena.py`: `65e96d4fe993042404f1a542bdf0d207f731703e5295d9478e2203529a5f27ae`
- `src/openjev/research/chess_candidate_attempt.py`: `fdf25a76f62bf6c87deb849f908e18a6fe2c3c934e146402efeae1b01b12417d`
- `src/openjev/research/chess_candidate_data.py`: `d62a7ce582a9fe44c20f0637bf0e7bd8fc2cc0c673cb693b0d428fdcaefc956e`
- `src/openjev/research/chess_candidate_eval.py`: `c32adb06f38969e2af8c85c8606e0f988996a38a61e3bd870b53efd28aebe60e`
- `src/openjev/research/chess_capacity_arena.py`: `86dde83de5b85333f79011e2d023265f33c81bf68eed4d80dff3262e6c5d887e`
- `src/openjev/research/chess_capacity_data.py`: `1c4a3fdf0853544a0f4dd2ee34ea495d02245bee150b9ae6404b77ec773d6ad3`
- `src/openjev/research/chess_compute.py`: `a1ec1a4618a92b4dc37a9a8f510f5f37ba45081e9e0db0d89714c689f88f537f`
- `src/openjev/research/chess_spatial.py`: `49e6e53d33cd52511c0e4a08074b9a7f0dc2244727df2879f3fb7fd1aa886ab4`
- `src/openjev/research/chess_spatial_baselines.py`: `9824212fdfe5678fafe2591eeb5f3962790713bd1b4dece4803648b696ae8942`
- `src/openjev/research/chess_spatial_data.py`: `70d6e94799cd59c7e4e097eb783a44e3288cad903fb7d4637e3fabd86003cf3e`
- `tests/test_chess_candidate.py`: `a75252e8c17397bf2a378826f5e8392c8f8119fc0ff54a56b8c28452ee7298fc`
- `tests/test_chess_candidate_arena.py`: `ea0691eec975d9413dfac9c4a1201de691fe811c7932715ae613553283ecb23f`
- `tests/test_chess_candidate_attempt.py`: `d80984b35d2b4c39bd85b0532820c606e974c8ef07c1d5685ab44621ffd609b6`
- `tests/test_chess_candidate_data.py`: `50c66d970ce858cda8cbae1b9e24554f87ba4668324d77da4c1c7611ea43b542`
- `tests/test_chess_candidate_eval.py`: `4e7ca667f1c3139e6730d2e25fb84cd953c95c092705b45b38eaa0354fae3007`
- `tests/test_chess_candidate_integration.py`: `54e75c3d9dc46b8808fb706cf3474642f4668ea700e5af0ff5fe328802be9eae`
- `tests/test_chess_candidate_study.py`: `1bf9fa7d8cc58ce84d8a7a6ffca60e40e818a048ad7861e87ad55608dc4ec82f`
