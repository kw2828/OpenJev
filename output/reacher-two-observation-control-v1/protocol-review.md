# Independent prospective protocol review

**Result: no material blocker found within this source-only definition module.** This is a review of a prospective protocol, not approval to prepare, freeze or run a scientific study. No protocol functions, tests, model constructors, random generators, training code or native simulator calls were executed for this review.

Reviewed source identities:

- `src/openjev/research/reacher_two_observation_protocol.py`: `572ec977c234be0d1c68ee9ac130abd3bf5132cb3854126a0d8f88ce0aa9362c`.
- `tests/test_reacher_two_observation_protocol.py`: `e7165e09c7e85a3ed1b451b9f2f59354e20f0daa916b44e19f0c7192ee15db0f`.
- The previous frozen geometry-memory plan is `23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee`. All 90 source files bound by that plan still match their saved hashes.

## Scope and proposed gate

The definitions retain three original paired initializations, the original public training data and complete minibatch orders. Only the three two-observation GRUs receive new fitting: 768 episodes, 48 epochs, batch 32, 1,152 updates per fit and 3,456 new updates total. The six inherited persistent/cached GRUs receive no new updates. All nine final actual classes must be restored before fresh control begins. The source manifests name all six inherited five-file fit directories plus the original training, initialization and order files: 44 inherited members, before lineage sidecars.

Each of three panels contains nine learned rows and five references, giving 42 rows: 27 learned, nine physics and six action-floor rows. All learned and physics rows use geometry scoring and CEM256 with horizon 12, action blocks of three, all four paid proposal banks, the paid mean, global selection and terminal truncation. Shared innovations do not make later score-adaptive proposal banks identical.

The proposed primary gate expands to exactly 25 distinct checks:

| Check | Count | Required cost relationship |
| --- | ---: | --- |
| Persistent family versus each of two comparators on each gap panel | 4 | Persistent mean <= 0.97 times comparator mean |
| Same comparisons for every paired fit | 12 | Persistent fit <= paired comparator fit |
| Full-sensing family versus each comparator | 2 | Persistent mean <= 1.02 times comparator mean |
| Every persistent fit versus zero actions on each gap panel | 6 | Persistent fit <= 0.90 times zero cost |
| Known-state physics versus zero actions on ordinary gaps | 1 | Physics cost <= 0.90 times zero cost |

All checks are required. Particle and public-kinematic references remain descriptive under this particular proposed rule. Family checks use arithmetic means of all three fit costs, not selected fits or averages of percentage changes. Failure does not establish equivalence or prove that explicit history explains the earlier result. The descriptive two-observation versus cached comparison cannot rescue a failed primary gate.

## Work accounting

The static arithmetic agrees with the unchanged loss and the component's full padded reconstruction contract. For 64 cases, 50 decisions and summed shortened horizon 534:

| Scheduled successful work | Count |
| --- | ---: |
| Executed native control transitions | 134,400 |
| Learned candidate evaluations | 22,118,400 |
| Learned imagined candidate transitions | 236,224,512 |
| Physics candidate evaluations | 7,372,800 |
| Nominal physics candidate transitions | 78,741,504 |
| Selected learned / physics advances | 86,400 / 28,800 |
| Candidate / selected geometry samples | 314,966,016 / 115,200 |
| Candidate / selected native physics substeps | 157,483,008 / 57,600 |
| History-arm real assimilations during control | 28,800 |
| Their paid observation updates / replay transitions | 345,600 / 316,800 |
| New-training real assimilation samples | 5,529,600 |
| Their paid observation updates / replay transitions | 66,355,200 / 60,825,600 |
| Additional training prefix and five-step rollout advances | 30,965,760 |
| Total new-training transition / analytic-cost samples | 91,791,360 each |

All 12 observation updates and 11 transitions per reconstruction are charged, including padding and both unchanged heads. These are successful forward counts, not total FLOPs or wall-time predictions. Backward, Adam, failed prefixes, copies, validation, native steps, audit replay and storage still require the future measured cost contract. Equal fitting updates or candidate budgets do not establish equal compute.

## Preparation boundaries

The module imports only the standard library and contains no preparation, seed derivation, generator construction or execution entry point. It offers 508 symbolic root roles and 636 generator roles including the particle-filter children. Numerical state separation is explicitly unfinished. Its required future exclusion scope includes all authenticated historical scored and engineering streams, full-size own namespaces and known literal engineering calls, including 410. Namespace strings alone are not treated as freshness evidence.

Production settings and all scientific constants are exact. Engineering mode only reduces specified sizes and retains all nine model identities, all 42 rows, all 25 checks and the horizon/physics contract. There are no guessed execution/audit caps. The prospective status cannot be changed to frozen while passing this validator.

`validate_preparation_bindings` checks an externally supplied metadata structure and always returns `execution_authorized: false`. It requires the exact prerequisite/cache identities; complete original paired data/order/initialization identities; all six actual inherited classes and checkpoint hashes; exact independently supplied source maps preserving the old 90 hashes and including at least 14 additive files; explicit runtime, separation, capacity, rehearsal and audit-contract bindings. It does not claim to authenticate those bytes, positive prior qualification, checkpoint tensor semantics, actual generator states or adequacy of caps. Synthetic placeholder hashes accepted by its tests therefore demonstrate schema behavior only. A future runner/auditor must fulfill these obligations independently before scientific execution.

The supplied tests cover the exact gate, lineups, counts, pure API/import surface, production drift, bounded engineering dimensions, matching Adam/training constants and malformed external bindings. The author reported 60 passing synthetic tests and Ruff success; this reviewer read them but did not rerun them. No scientific readiness or effectiveness result follows from that component validation.

## Interpretation limit

A future pass would support persistent control beyond this particular trained bounded two-observation recipe and the trained single-observation cache on this task and objective. Shared historical fitting inputs, three paired fits and a single environment do not establish independent training replication, isolated velocity inference, Bayesian inference, biological wiring or architectural novelty. Reconstruction also intentionally costs more neural work, so any eventual utility or speed advantage must be reported with those measured costs.
