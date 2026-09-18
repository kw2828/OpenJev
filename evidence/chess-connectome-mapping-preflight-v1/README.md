# Synthetic mapping-search timing check

Six MPS cases completed: ordinary updates, fixed-map proposals and learned-map
proposals, each with sparse-mode and node-local recurrence. Every case used
eight warmup and eight timed updates. Two fixture/accounting tests and Ruff
passed before the run.

The graph and labels are synthetic. No biological asset, real training row,
teacher call or retained checkpoint was used.

- [Measurements and limits](profile.json)
- [Source/output hashes](receipt.json)
- [Prospective experiment](../../research/chess-connectome-mapping-study.md)

The 1,223.30-second full-workload projection is a linear estimate from repeated
starting boards with padded candidate menus. It excludes data preparation,
journals, checkpoints and evaluation. This is engineering evidence, not a
measured full-run duration or evidence of chess performance.
