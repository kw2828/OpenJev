# OTTO readout/state ablation

**MECHANISM_DEV_FAIL: 3/6 paired seed/setting cells passed.**

This compares continuing training of the action residual head (116 parameters), both readouts (232), or all GRU/readout parameters (6,112), against the same pretrained parents. All nine fits used 40 epochs, 360 updates, matched orders, and final checkpoints only.

The 12 views use the same 18 previously exposed DEV paths (4,816 rows), P4 teacher queries, and three fit seeds. Lower teacher-score gap is better. These are OTTO fixed-path teacher-score imitation results, not autonomous performance, independent held-out evidence, statistical equivalence, or a novelty claim.

![All arms, fit seeds, training costs, and comparison checks](methods.png)

Dots show each fit; black diamonds are equal means across the three fits. No confidence interval is inferred from three fits on shared paths. The pretrained parent has no continuation training cost; its original pretraining cost is excluded.

## All twelve views

Values below are the saved case-weighted raw gaps. Full covers all nonquery actions; later covers nonquery steps >=5, after the first post-start teacher query.

| Arm | Fit seed | lambda3 full | lambda3 later | lambda4 full | lambda4 later | View seconds |
| --- | --- | --- | --- | --- | --- | --- |
| pretrained | 309000001 | 0.0642943963 | 0.0754422871 | 0.0957248924 | 0.105356751 | 0.464457625 |
| pretrained | 309000002 | 0.0564876525 | 0.0665799087 | 0.105048203 | 0.111298087 | 0.448810833 |
| pretrained | 309000003 | 0.0915056234 | 0.0988648198 | 0.129506915 | 0.141324248 | 0.466868125 |
| action_residual_only | 309000001 | 0.0705214681 | 0.0762236771 | 0.117266228 | 0.127671289 | 0.42675575 |
| action_residual_only | 309000002 | 0.046261517 | 0.0519883519 | 0.0911799343 | 0.0989724144 | 0.516558667 |
| action_residual_only | 309000003 | 0.0531717704 | 0.0600301418 | 0.118361985 | 0.127199042 | 0.443787917 |
| both_readouts | 309000001 | 0.0704903021 | 0.0759601846 | 0.104375114 | 0.113877517 | 0.482059375 |
| both_readouts | 309000002 | 0.0502935392 | 0.0561639682 | 0.0952448896 | 0.100658369 | 0.537365292 |
| both_readouts | 309000003 | 0.0984546594 | 0.109350295 | 0.146056021 | 0.139388854 | 0.438222333 |
| full_joint | 309000001 | 0.077757574 | 0.0840905512 | 0.136862761 | 0.131028309 | 0.414018084 |
| full_joint | 309000002 | 0.0494127397 | 0.0558552862 | 0.0797164838 | 0.0843956871 | 0.5572575 |
| full_joint | 309000003 | 0.0463351968 | 0.0448989681 | 0.109357236 | 0.12059105 | 0.403369375 |

## Complete paired decision

The fixed full-joint candidate must improve later gap by at least 5% against each control with strict improvement, and must not regress full gap. Every seed/setting cell must pass. A zero control gap cannot pass the strict-improvement check. No failed cell is excluded.

| Fit seed | Setting | Later supported cases | vs parent later/full | vs residual later/full | vs both later/full | Cell |
| --- | --- | --- | --- | --- | --- | --- |
| 309000001 | lambda3 | 3/3 | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL |
| 309000001 | lambda4 | 3/3 | FAIL / FAIL | FAIL / FAIL | FAIL / FAIL | FAIL |
| 309000002 | lambda3 | 3/3 | PASS / PASS | FAIL / FAIL | FAIL / PASS | FAIL |
| 309000002 | lambda4 | 3/3 | PASS / PASS | PASS / PASS | PASS / PASS | PASS |
| 309000003 | lambda3 | 3/3 | PASS / PASS | PASS / PASS | PASS / PASS | PASS |
| 309000003 | lambda4 | 3/3 | PASS / PASS | PASS / PASS | PASS / PASS | PASS |

## Readout proximity, descriptive only

These saved comparisons ask whether a readout-only later gap is at most 5% above full joint. They do not establish equivalence or select a replacement candidate. Relative excess is undefined when the full-joint gap is zero.

| Fit seed | Setting | Readout | Relative excess | No more than 5% worse |
| --- | --- | --- | --- | --- |
| 309000001 | lambda3 | action_residual_only | -0.0935524143 | yes |
| 309000001 | lambda3 | both_readouts | -0.0966858515 | yes |
| 309000001 | lambda4 | action_residual_only | -0.0256205738 | yes |
| 309000001 | lambda4 | both_readouts | -0.13089379 | yes |
| 309000002 | lambda3 | action_residual_only | -0.069231303 | yes |
| 309000002 | lambda3 | both_readouts | 0.00552646117 | yes |
| 309000002 | lambda4 | action_residual_only | 0.172718865 | no |
| 309000002 | lambda4 | both_readouts | 0.192695656 | no |
| 309000003 | lambda3 | action_residual_only | 0.337004932 | no |
| 309000003 | lambda3 | both_readouts | 1.43547457 | no |
| 309000003 | lambda4 | action_residual_only | 0.0547967026 | no |
| 309000003 | lambda4 | both_readouts | 0.155880594 | no |

## Measured cost

| Arm | Trainable parameters | Fit seed | Continuation seconds |
| --- | --- | --- | --- |
| action_residual_only | 116 | 309000001 | 94.6490608 |
| action_residual_only | 116 | 309000002 | 93.2642126 |
| action_residual_only | 116 | 309000003 | 82.3918007 |
| both_readouts | 232 | 309000001 | 147.018055 |
| both_readouts | 232 | 309000002 | 154.638248 |
| both_readouts | 232 | 309000003 | 156.245292 |
| full_joint | 6112 | 309000001 | 168.585427 |
| full_joint | 6112 | 309000002 | 160.307893 |
| full_joint | 6112 | 309000003 | 175.264457 |

Saved worker wall times: producer 1240.7547 s; independent audit 4.22611421 s. Per-fit and per-view timers cover their recorded worker regions; these are not a matched inference-speed benchmark or end-to-end compute estimates.

## Evidence and scope

Original closure: [closure-01.json](<../../output/otto-readout-state-ablation-v1/closure-01.json>). External SHA-256: `5b1a11c236f0f4e91edf60dbb025ce0071ec4e0ec758fb2311ee59868dc59065`.

- plan: [registration-01.json](<../../output/otto-readout-state-ablation-v1/registration-01.json>) (`1dcd63d7b191e55991713e9ee7342d4d704f9727901b6ecabf58b52558bb490d`)
- producer receipt: [receipt.json](<../../output/otto-readout-state-ablation-v1/training-01/receipt.json>) (`bf21e4ccc5f5afb0ab0ba6b3c548e025993bc8fe1ba1373a0d377f71c200f469`)
- producer terminal: [training-native-01.terminal.json](<../../output/otto-readout-state-ablation-v1/training-native-01.terminal.json>) (`f5c21f03267862cca2adf5f023601e99dd18d052590d02b0526ee08b8cfc0e3a`)
- audit receipt: [receipt.json](<../../output/otto-readout-state-ablation-v1/audit-01/receipt.json>) (`1eaa13e7be7899dcda48bb79cf40697202401377971c802bf2964204b97fdfc8`)
- audit terminal: [audit-native-01.terminal.json](<../../output/otto-readout-state-ablation-v1/audit-native-01.terminal.json>) (`52de0fc727219af58bf9f8dbc01673597b3f985482919d47b5860a14f1ab0707`)
- audit: [audit.json in the evidence archive](https://github.com/kw2828/OpenJev/releases/tag/otto-readout-state-ablation-v1) (`6aca7b25e4d0bfe85e461b2c9dec14437e508a7b4d765b60fc776a0417e8f62d`)

The original producer and independent auditor both completed under their original supervisors. The audit checked all nine fits, 3,240 updates, 19,440 episode exposures, and twelve views. Publication reads saved JSON and verifies opaque payload hashes; it performs no array/checkpoint decoding, model calls, training, teacher calls, or simulator calls. TEST and confirmation remain unadmitted.
