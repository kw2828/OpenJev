# Does latent prediction improve recurrent control?

**Frozen and launched September 18, 2026; results pending.** The
[launch receipt](../evidence/reacher-objective-ablation-v1/launch.json) identifies
the single execution attempt. This comparison holds the residual
GRU and CEM256 planner fixed, then changes the training objective. It follows the
[completed adaptive-search result](reacher-adaptive-search.md). It is a development
experiment, not a JEPA reproduction or a test of biological wiring.

| Arm | Training objective | Deployment |
|---|---|---|
| Anchor | Existing observation and reward loss | Student GRU + CEM256 |
| Raw | Anchor + future observation prediction | Same |
| Latent | Anchor + EMA-teacher future state prediction | Same |

There are three paired initializations and nine fits. Every fit uses the same
768 training episodes, 48 epochs, 1,152 optimizer updates and paired minibatch
orders. Raw and Latent share auxiliary horizons 1, 3 and 7, masks and 4,160 added
predictor parameters. Their gradient paths and training computation differ.
The teacher and predictor are excluded from deployment.

Auxiliary weights come from a frozen training-only gradient-norm rule. All nine
final checkpoints, including their optimizer and teacher state, must pass
authentication and restoration before collecting fresh evaluation data.
Evaluation covers 64 paired cases with full sensing, six missing observations
and ten missing observations, plus history-reset interventions. There are 45
learned controller rows, 12 reference rows and 96 fresh prediction episodes.

## Continuation rule

Latent prediction must reduce family mean native cost by at least 5% against
both Anchor and Raw on both gap panels, with no paired fit worse. Its history
reset must worsen family cost by at least 5%, with positive reset effects in
every fit. The separate zero-action and physics-reference checks also apply.
All 31 prespecified checks must pass; every fit will be reported.

A reset penalty establishes intervention sensitivity, not an advantage over
a separately trained current-observation model. A positive result still needs
that comparator, a compute-matched Raw treatment, untouched confirmation cases
and a second environment. Connectome and rewired controls follow only after a
useful learning mechanism is established.

## Engineering evidence and fixed limits

The [preflight receipt](../evidence/reacher-objective-ablation-v1/preflight.json)
records **199 passing checks**. The full synthetic rehearsal ran nine tiny fits
and 57 control rows; its independent audit replayed **9,350 native transitions
with zero discrepancy**, while neural and optimizer calls were prohibited.
Synthetic inheritance checks replace production lineage only in this rehearsal.
The [complete engineering archive](https://github.com/kw2828/OpenJev/releases/download/research-reacher-objective-ablation-v1/engineering-whole-tree-v1.tar.gz)
contains the saved tree. These are implementation checks, not efficacy results.
The receipt also records the earlier failed engineering audit and its unavailable
temporary artifacts.

The size-matched capacity measurement projects 16 minutes of training. Earlier
controller timings project another 7.9 minutes of controls and references.
These estimates exclude remaining storage, prediction and restoration overhead.
The frozen limits are **3,600 seconds for execution and 300 seconds for audit**.
Preparation costs are reported separately from research execution costs.
There are no retries, replacement seeds, resumes or cap extensions.

The [executable protocol](../evidence/reacher-objective-ablation-v1/protocol/plan.json)
binds all 42 source files, runtime, data lineage, random streams and criteria.
Its SHA-256 is:

```text
dca778153230375a5a0e790390d328d69857dd4c2d3260b61fb9f214fa1d32f2
```

The [original design](reacher-objective-ablation-draft.md) preserves the reasoning
and detailed limits. The [runner](../scripts/reacher_objective_study.py) and
[independent auditor](../scripts/audit_reacher_objective_study.py) are separate.
The auditor reconstructs recorded gradient norms, first/last EMA updates, costs
and native outcomes. It does not independently rerun every training update or
neural prediction.
