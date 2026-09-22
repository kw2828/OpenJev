# Teacher-cost cohort: all 558 states supported

**The fixed training cohort passed its support check: 558/558 selected states, with no repairs or omissions.** It covers all 144 historical learner trajectories, including 144 initial states and retained prefixes as late as step 2,187. This establishes usable starting states for a larger teacher-cost study. No new continuation labels, policy training or autonomous evaluation occurred in this phase.

The [prospective protocol](otto-teacher-cohort-protocol.md) chose at most four evenly spaced retained prefixes per episode, including reset. Ten short episodes contributed fewer than four rows. Selection used identity and prefix order, never posterior mass, teacher scores or outcomes. There are 280 lambda3 and 278 lambda4 states, with four, three and two eligible actions at 398, 107 and 53 states respectively.

## Completed work

| Measurement | Result |
| --- | ---: |
| Original retained TRAIN rows inspected | 4,596 |
| Episodes represented | 144/144 |
| Captured states supported / unsupported | 558 / 0 |
| Public resets / replayed updates | 144 / 119,212 |
| Support constructor attempts / returns | 558 / 558 |
| Complete attempt/return events | 239,828 |
| Original parent elapsed time | 16.8542 seconds |
| Worker peak RSS | 141,705,216 bytes |
| Worker outputs, including receipt | 70,090,786 bytes |
| Independent audit parent elapsed time | 11.2027 seconds |

Every reconstructed public posterior matched its original float64 hash and mass. The extractor captured each selected belief without normalization, clipping or a second assimilation of its last observation. Each captured belief then passed the unchanged teacher snapshot's support conditions. Sampling, native simulation, analytic action selection, learned inference and optimization counts were all zero.

The new extractor passed 20 fabricated tests; the execution wrapper passed four failure tests, including preservation of interrupted work and demotion of a late failed completion. The first engineering attempt preserved a lint failure in the wrapper even though all 20 extractor tests passed. The final wrapper tests and lint passed before empirical selection. No empirical run was repeated.

The separately authored [saved-record auditor](../scripts/audit_otto_teacher_cohort.py) agreed with the complete selection, all 239,828 public/support events, every captured posterior hash and support predicate, both original process records, and the pinned source/input closures. It made no new teacher or simulator calls. It did **not** recompute intermediate posterior arithmetic; that remains checked by the original reconstruction against historical witnesses. Its audit agreement is an integrity result, not evidence of useful labels or a better policy.

## Evidence and next experiment

[Frozen plan](../output/otto-teacher-cohort-v1/plan-01.json) · [Worker receipt](../output/otto-teacher-cohort-v1/run-01/receipt.json) · [Original supervisor](../output/otto-teacher-cohort-v1/supervision-01.terminal.json) · [Independent audit](../output/otto-teacher-cohort-v1/audit-01/receipt.json) · [Complete new-phase archive](https://github.com/kw2828/OpenJev/releases/tag/otto-teacher-cohort-v1).

The release includes the full public replay journal and exact captured arrays. Re-running the auditor also requires the historical inputs identified in the frozen plan; this archive does not replace those inherited studies.

The [learning comparison](otto-action-cost-followup.md) can now use this complete cohort. Its two arms should share the ordinary policy, features, episode weights, D4 augmentation, initialization and optimizer work. Compare analytic preference targets with continuous teacher-continuation costs under the same centered regression loss. Use a global TRAIN-derived scale, preserving small gaps rather than turning nearly tied actions into confident labels. This is a new common training objective, not an exact replication of the older cross-entropy recipe.

Label sampling, fitting and fresh full-horizon evaluation still need their own fixed allocation before execution. Qualification does not establish a recurrent, connectome or novel architecture advantage. The supplied exact posterior already summarizes observation history, so this interface cannot demonstrate that recurrence recovers missing memory.
