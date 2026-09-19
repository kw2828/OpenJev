# Independent proposal-memory results review

**Completed engineering evidence; descriptive support check failed, 2/6 passed.** All nine rows are included. This is one reused seed410 case, not a scientific qualification or a new architecture result.

## Independent verification

Authenticated 5,496 exact run members, including 5,454 row payloads (162,518,475 bytes), all nine completion receipts and all nine replay-audit receipts. The 20 live source files, saved source snapshots and prospective Git objects match commit `a9b689396207d62cd5285403296f05aa12744126`. Both old 116-source and 136-source protected maps are unchanged. The retained outer process receipt records exit 0; no retry or failed artifact is present in the accepted run.

All cases match the original saved targets, motor-gain schedule, action-noise arrays, reset seed and 200 shared innovation stems. All three cold episodes and their 9,600 saved decision arrays reproduce their earlier counterparts exactly. The completed audits report 1,800 real transitions, 5,377,536 nominal candidate transitions and 1,800 selected advances, with zero native state error. Maximum independent float32 geometry difference is 2.384185791015625e-7. This review repeated only file authentication and saved-array arithmetic, with zero model, simulator or RNG calls.

## All native costs

Per-action native costs, lower is better. FULL=[0,200), PRE=[0,80), POST=[80,200), MOVE=[100,150). These intervals use issued-action indices and native rewards, including their actual target-event timing.

| Role | Proposal | FULL | PRE | POST | MOVE |
|---|---|---:|---:|---:|---:|
| nominal | cold | 0.11349797 | 0.12685897 | 0.10459064 | 0.05677829 |
| nominal | repeat_last | 0.13537533 | 0.12393802 | 0.14300020 | 0.17230865 |
| nominal | shift_plan | 0.12755328 | 0.12323546 | 0.13043182 | 0.14882919 |
| public_gain | cold | 0.13394715 | 0.18065342 | 0.10280964 | 0.04530133 |
| public_gain | repeat_last | 0.13700914 | 0.15919771 | 0.12221677 | 0.11069363 |
| public_gain | shift_plan | 0.13435511 | 0.14734161 | 0.12569745 | 0.12533576 |
| true_state | cold | 0.14254395 | 0.16854546 | 0.12520961 | 0.12392000 |
| true_state | repeat_last | 0.13282033 | 0.14158475 | 0.12697739 | 0.14087845 |
| true_state | shift_plan | 0.13582295 | 0.15241640 | 0.12476065 | 0.12492437 |

## Fixed full-episode comparisons

Improvement is 100 × (comparator cost - shifted-plan cost) / comparator cost. The threshold is inclusive: shifted-plan cost must be at most 0.97 × comparator cost.

| Role | Shift versus | Improvement | Meets 3% |
|---|---|---:|---|
| nominal | cold | -12.383750% | No |
| nominal | repeat_last | +5.778048% | Yes |
| public_gain | cold | -0.304568% | No |
| public_gain | repeat_last | +1.937121% | No |
| true_state | cold | +4.715036% | Yes |
| true_state | repeat_last | -2.260661% | No |

The full-plan versus repeat-last control is decisive. With nominal dynamics, shifting beats repeat-last by 5.78% but remains 12.38% worse than cold planning. With known gain, its 1.94% improvement over repeat-last misses the fixed margin and it is slightly worse than cold. With known state and gain, shifting improves on cold by 4.72%, but shifting is 2.26% worse than repeat-last. None of these roles passes both comparisons. Favorable PRE or POST windows cannot rescue the specified full-episode check.

Cold nominal remains the lowest full native cost among these nine exposed rows. That is a description of this reused case, not evidence that approximate state or incorrect dynamics are generally preferable. Methods share initial case and innovations but follow different closed-loop states, so these differences alone do not identify whether proposal centering, score mismatch or early trajectory changes caused the result.

## Cost and next action

Execution took 89.627339s; independent native replay took 84.818830s. Inclusive driver time was 174.446318s, within the two separate 600s caps; external process time was 175.437341s. The sum of whole-row times was 89.516767s. Proposal prepare/commit calls contributed 0.174725s in total, nested within execution, not an additional cost. Each row uses the same paid candidate budget and selected-action replay. Shared-host instrumented wall times are not isolated latency measurements.

Close this warm-proposal hypothesis for the tested configuration and retain cold planning as the reference. Do not launch neural training or the held fresh-case qualification on this result. A useful next decision is whether to prospectively redesign the control task/controller pair so that correct persistent dynamics offer a demonstrable utility margin over the simplest public-state baseline. First use the already-saved same-root candidate diagnostics to identify a specific objective or planning limitation; if those do not support one concrete correction, change the task instead of accumulating more warm-start variants. Any new task or correction needs its own fixed comparison and fresh validation, without relaxing this failed 3% check.

Warm-starting CEM is conventional optimizer memory. This run neither validates nor rules out biological recurrence, learned latent memory, persistent dynamics inference, or broader MPC warm-starting. There is no confidence interval, cross-case generalization claim, parameter learning, or mechanism novelty here.

## Receipt bindings

- Outer completed: `82785582063738b97588e6487397e4bddcf58e98f9c352e8ce74f85ae834c5ec`.
- Execution completed: `7e57d4cb3006567c925a09059f63a0fcd4f39ce0b16f8441df81ef3eac0eaeb9`.
- Protocol: `7ffea079cfaea91eb71ce091db54123fa8d0666966325488a729f61f08fe2711`.
- Process completed: `3dce483782faa4605df126fda6051ee54a9d57271e21f5fac041f5fd295a923a`.
- Numerical review: `a9e110e0696f730d7a5f145f4789b6f05162dbc8b4014111af32d16ae7535761`.

The JSON records all row and audit hashes, all 20 source identities, protected plan hashes, interval reward components and comparison arithmetic. Local Git objects and retained process receipts support the stated integrity check; they do not independently attest remote publication or operating-system execution.
