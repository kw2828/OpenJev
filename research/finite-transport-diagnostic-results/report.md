# Saved recurrent transport diagnostic

This is a descriptive diagnostic of all 27 saved states from the completed equal-time synthetic study. It does not train, rerun forecasting, select a fit or advance an architecture.

[Protocol](../finite-transport-diagnostic-protocol.md) · [All saved metrics](summary.json) · [Evidence manifest](manifest.json) · [Child evidence archive](evidence.tar.gz)

![All checkpoints and existing errors](benchmark.png)

The three geometry panels retain initial, boundary and final states for all nine fits. Transition quantities are arithmetic averages over four actions. The regret panel uses the original independently audited H2/H8 rows; no evaluation was repeated.

| Original phase | Status | Seconds |
|---|---|---:|
| Qualification 01 | FAILED: 3 assertions, 50 passed | 0.942113 |
| Qualification 02 | PASS: 53 tests | 0.935183 |
| Diagnostic | PASS: 27 checkpoints, 108 arrays | 0.509708 |

Test-only analytical float64 bounds: u=eps/2, gamma_n=n*u/(1-n*u); normalization uses 4*gamma_n and projection allows two versus one rounded matrix products. No fitted residual cutoff, core change or scientific threshold change.

The first failure, its complete source snapshot, original logs and receipts are retained. Only qualification 02 admitted checkpoint decoding. The diagnostic made zero model, optimizer or generator calls and created no training updates or evaluation cases.

The Grams average over all uniformly weighted action sequences. They propagate surviving mass without normalization, so attenuation combines transport mixing and hazard loss. Full eight-dimensional propagation precedes projection. Small signed eigenvalues are retained. The rank-at-most-three emission and head spectra have structural zero modes, which are not learned collapse.

Large sensitivity does not establish correctness, and small sensitivity does not establish lost task information. The earlier retention failure and the closed H4-supervision follow-up remain unchanged. No causal, biological, robotics, novelty or architecture-improvement claim follows from these plots.

**Dependency:** [the complete parent release](https://github.com/kw2828/OpenJev/releases/tag/finite-training-allocation-v1) is required for the 27 original checkpoints and their evidence. This archive contains the child study only, including raw geometry.json and both complete source snapshots. Current registered source hashes and the complete external parent identity are in the manifest and receipt. The executable environment is not bundled and this package does not authorize a rerun.
