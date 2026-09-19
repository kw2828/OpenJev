## Fixed memories, better action selection

The same 18 learned checkpoints and two public-memory references played three fixed selection policies on 64 fresh paired decks: 3,840 games, 392,810 native actions, no retraining or retry.

The unseen-card tie preference (C) passed all three continuation conditions, expanded into 13 detailed checks. Mean learned C-minus-B return was +0.422877, with positive differences in all 18 fits. The best learned family, ordinary diagonal Kalman memory, finished 83/192 games (43.2%), versus zero under the original and stable-only pickers. Both simple memory references finished 64/64 under C.

The proposed local innovation memory did not beat ordinary Kalman memory. This is a controller result, not a new architecture, calibrated uncertainty or an ICLR-ready claim. The original architecture result remains FAIL, 1/6. Stable scoring alone worsened all learned fits and reduced both reference completion rates to zero.

The run completed in 204.96 seconds. Full saved-output and independent public-transition audits passed. Timings include instrumentation and storage; A has an extra diagnostic scoring pass, so they do not establish an intrinsic speedup. The 3,840 games reuse 64 paired layouts rather than independent decks.

### Contents and verification

The archive contains every new raw trajectory and receipt, source snapshot, fixed protocol and seeds, all result tables, a benchmark chart and the preselected first-game GIF. Each archive member was read back and checked against manifest.json. receipt.json contains archive and manifest SHA256 values.

The GIF uses gated-delta pair 0 on the first fresh case, selected before outcomes. None of the three policies finishes that particular game. No frames expose hidden cards.

### Inherited training dependency

This evaluation reuses the 18 checkpoints and training data from [the original pilot release](https://github.com/kw2828/OpenJev/releases/tag/research-card-memory-pilot-v1). Obtain both evidence archives and unpack them into the repository root to reconstruct the complete chain. The original archive SHA256 is `dbdc50851322e2dc02b744e43eac83b315f83bcfa607b9931ef932e2f69b980f`.

The prospective controller code and criteria were pushed at `eef9ab6` before native execution. Saved-only report and figure programs accept the explicit SHA256 bindings recorded in the receipts. Runtime and source changes require a separately identified reproduction; do not overwrite these frozen evidence files.
