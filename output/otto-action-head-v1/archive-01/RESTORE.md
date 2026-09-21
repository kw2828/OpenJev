# Restore the complete immutable action-head run

Archive: `otto-action-head-v1-run-01.tar.gz`

SHA-256: `3ba2808c9b52b8973d62356df6ebc1d486bb393673cdbf63f4bc2605f7cf3b8a`

1. Verify the archive SHA-256 against this manifest before extraction.
2. Create a new empty directory outside any existing scientific run.
3. Extract with `tar -xzf otto-action-head-v1-run-01.tar.gz -C /absolute/path/to/empty-directory`.
4. The resulting `run-01/` contains all 49 original files, including the original receipt and 12 model checkpoints.
5. Compare every restored file size and SHA-256 to the `members` entries in `manifest.json`. The 48 payload hashes must also equal `run-01/receipt.json`; verify the receipt itself against `751eecb6e1eccb6918fdf7f121dc240f94fc0f08b000d10e0398339a05a0780f`.

No model loader is needed to restore or verify this archive. NPZ contents were never decoded during packaging. Study source, plan, parent terminal, and independent audit remain separate repository evidence. This archive preserves the completed run; it does not alter its scientific result.
