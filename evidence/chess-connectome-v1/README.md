# Frozen connectome comparison

The [plan](protocol/plan.json) was prepared before fresh panel generation, training or neural evaluation. Its SHA-256 is `5543877d88d2c00b7085c93f2f01602f2d0b3439486b05661d1a8511af7f81db`.

The local study fitted all 18 adapters and evaluated all 21 models, including three unchanged backbones. The original report audit completed on September 18, 2026. The biological topology failed its frozen criterion: 5 of 30 required checks passed. Read the [results](../../docs/chess-connectome.md), [completed summary](results/summary.json), and [design and reproduction commands](../../research/chess-connectome-study.md).

The frozen exclusion snapshot contains 3,826,350 natural board states from prior work and the entire ChessBench source file plus legal successors. Admission also excludes their mirrors. The compressed snapshot is included for reproducible exposure checks.

Source and test hashes are bound in the plan. The relevant 309-test suite passed, followed by 139 passing checks for the final new training, data and runner files; Ruff passed. Tests use synthetic fixtures, not performance results.

Graph assets and derivative trained checkpoints stay in local `runs/`. This protocol package contains no biological graph arrays or derivative weights.
