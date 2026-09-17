# OpenJev chess student checkpoints

Nine original final checkpoints trained from random initialization by OpenJev. These weights are released under the included MIT license. They do not contain ChessFly, FlyWire, LFM, Qwen or Stockfish model assets.

Modes: `circuit`, `rewired`, `gru`; seeds: `17`, `29`, `43`. Each `<mode>-<seed>/weights.pt` preserves the original training output bytes.

The synthetic circuit continuation gate failed. All fits are included without selection. The GRU has 206,833 parameters; each circuit has 182,289. Training used bounded Stockfish-generated move/value labels, not biological data.

See [usage and results](../../docs/chess-student.md) and [audit records](../../evidence/chess-student-v1/results/manifest.json). `checksums.json` records every weight hash.
