# Does remembering the previous plan help?

**Not reliably in this test.** Carrying the previous action sequence into a new search passed only **2 of 6** predeclared comparisons. No state/dynamics setting passed against both a fresh search and the simpler last-action control. This is one reused engineering case, with no trained model or architecture claim.

![All nine controller costs and complete execution times](../output/reacher-proposal-memory-engineering-v1/review-01/figure.png)

## What changed

We tested three ways to start the same 256-candidate, 12-step planning search:

- **Fresh search (`cold`):** the original proposals centered on zero.
- **Last action (`repeat_last`):** proposals centered on the last executed command.
- **Previous plan (`shift_plan`):** proposals centered on the previous selected sequence, shifted by one executed action and projected into the existing three-action blocks.

Only the initial random proposal centers changed. The seven fixed anchors, random inputs, search budget, scoring, physics and later search updates stayed the same. Memory updated only after an action was acknowledged by the simulator and the observation was processed. An observed target change reset the center; hidden motor-strength changes did not trigger a reset.

Each method ran with nominal dynamics and public state estimates, then with known current motor strength, then with known current state and motor strength. Those references had no access to future targets or disturbances. All nine runs used the same seed-410 starting case, 200 actions and supplied disturbances. Their subsequent trajectories differ because their actions differ.

## Results

Native cost per action, lower is better. The predeclared support rule required the previous-plan method to reduce whole-episode cost by at least 3% against **both** controls in **all three** settings.

| State and dynamics available | Fresh search | Last action | Previous plan | Gain over fresh / last action |
|---|---:|---:|---:|---:|
| Public estimate, nominal gain | 0.113498 | 0.135375 | 0.127553 | -12.38% / +5.78% |
| Public estimate, known gain | 0.133947 | 0.137009 | 0.134355 | -0.30% / +1.94% |
| Known state and gain | 0.142544 | 0.132820 | 0.135823 | +4.72% / -2.26% |

The previous plan helps against one baseline in some settings, but offers no consistent advantage over both. Neither a favorable setting nor a shorter time window rescues the failed comparison. [All predeclared interval results](../output/reacher-proposal-memory-engineering-v1/review-01/report.md).

This closes the tested proposal-memory explanation. It does not show that recurrent world models cannot help: this experiment retains a plan, not learned beliefs about dynamics. It also does not establish that more accurate state hurts control. The controller uses a short horizon and finite search, and the resulting trajectories are different.

## Verification and cost

All nine runs completed and passed independent native replay. The three fresh-search episodes and **9,600 saved decision arrays** exactly matched their corresponding earlier runs. An independent saved-output review authenticated **5,496 run files**, all 20 bound source files against the pre-run Git commit, and the unchanged source sets from the two earlier closed studies.

The run executed **1,800 real transitions, 5,377,536 candidate transitions and 1,800 selected-plan advances**. Execution took **89.63 seconds** and replay **84.82 seconds**; the complete enclosing process took **175.44 seconds**. Individual 200-action runs took 9.79-10.14 seconds including setup, trace writes and hashing. These shared-host measurements do not establish a deployment speedup. There was no training or online parameter identification.

[Frozen protocol](../output/reacher-proposal-memory-engineering-v1/protocol.json) · [Independent results review](../output/reacher-proposal-memory-engineering-v1/results-review.md) · [Implementation review](../output/reacher-proposal-memory-engineering-v1/implementation-review.md) · [Audit review](../output/reacher-proposal-memory-engineering-v1/audit-review.md).

[Download the complete evidence](https://github.com/kw2828/OpenJev/releases/tag/research-reacher-proposal-memory-v1): all nine raw runs, original shared inputs, source snapshots, replay reports and the three complete earlier fresh-search reference runs. The archive preserves original path bindings and runtime assumptions; it is a research record, not a standalone application installer.

## Next decision

Retain fresh search as the reference. Diagnose whether the planner can turn better state and dynamics information into better decisions before spending on learned adaptation. The existing fresh-case qualification and its thresholds remain unchanged. [Decision and next diagnostic](../output/reacher-proposal-memory-engineering-v1/next-decision.md).

[Earlier changing-dynamics result](reacher-tracking-engineering.md) · [Saved observer and candidate diagnostics](../output/persistent-dynamics-diagnosis-v1/README.md) · [Trained memory comparison](reacher-two-observation-control.md).
