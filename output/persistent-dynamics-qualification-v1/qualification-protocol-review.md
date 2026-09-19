# A small first screen for useful gain adaptation

19 September 2026. **Prospective review only.** No qualification case, seed, simulator trajectory, controller decision or neural fit was generated. This note reads the proposed [tracking design](design-review.md), [classical comparator](classical-comparator-review.md), reviewed [wrapper](wrapper-review.md), and completed engineering capacity timing/count records. It does not read the running two-observation study's results.

**Start with four cases, nominal tracking and both switch directions, retaining all six controllers. Advance to sixteen fresh cases and all five regimes only if the small screen passes.** Freeze the same definitions and thresholds for both stages before the first screen. The screen is a development admission decision, not confirmation or an architecture result. Its cases must not be pooled with confirmation cases or used to tune the gain grid, observer, target process, planner or thresholds.

## Fixed task and comparisons

Keep 200 actions, full sensing, fixed damping, gain values 0.7/1.3, independent normalized actuator noise 0.05, and the original native distance/control weights. Install a new public target at roots 0, 50, 100, 150, with reward for action `t` evaluated against the target already visible at root `t`. The first target and every change are shared across all controllers and regimes for each case.

One explicit target-generation proposal is uniform area in the reachable annulus `0.08 <= radius <= 0.18`, with angle uniform on `[0,2*pi)`, rejecting subsequent targets less than 0.10 away from the previous public target. This creates repeated movement without looking at any controller's state or outcome. It is a new declared target process, not native Reacher's distribution. Give target generation its own stream, independent of reset, gain regime, switch time, actuator noise and search innovations. Rejection uses only target geometry. An enclosing implementation must bound failed sampling attempts and preserve failures, rather than replace a hard case. No future target preview is allowed.

Balance hidden changes at roots 80 and 120. Four screening cases give two cases per time; sixteen confirmation cases give eight. Use the same case's switch-time assignment in both directions, independent of the public target process. All policy rows share exogenous noise and initial search innovations, but their later actions, physical states and adaptive CEM banks can diverge. No controller gets a change flag or future gain schedule.

All six rows are necessary:

| Row | Required information and purpose |
|---|---|
| Public nominal | Public endpoint-velocity observer, nominal gain 1. Tests whether gain can simply be ignored. |
| Public rolling ID | Same observer, 21-point gain bank, 20-transition residual window. Tests conventional online identification. |
| Public ID then freeze | Same estimator through packet 40, then retain the gain selected at root 40 for actions 40 onward. Continue the same public state observer; stop gain-bank updates. Tests the value of continuing identification after a hidden change. |
| Public current-gain reference | Same public state observer, true current gain only. Isolates gain knowledge from velocity privilege. |
| True-state/current-gain reference | True current angles/velocity and current gain, without future changes/noise/goals. Tests supplied-physics controller competence. |
| Zero command | The same event/noise process, with zero issued commands. Supplies a fixed competence floor. |

The true-gain rows are finite-search references, not optimal bounds. Gain at planning root `t` must come from the explicit privileged current value, not the last transition record: the wrapper installs the next gear immediately before stepping. Never copy the live hidden-gain model into a public identifier.

## H12 is a controlled limitation

Keep the reviewed geometry/CEM256 controller unchanged in this qualification: four paid 64-candidate stages, horizon 12, block 3, paid final mean, earliest global best, stepwise reward clipping and terminal shortening. A 12-step horizon sees 0.24 seconds, while one target interval lasts 1 second. This can hide benefits that require longer acceleration/braking coordination. A fixed short horizon is nevertheless a valid first test of whether gain information helps this existing controller; gain already changes the immediate action-conditioned acceleration.

Do not give the oracle a longer horizon, add a terminal value only to one row, or raise the horizon after inspecting a failed result. Require native tracking competence first. If every physics row is poor, or the true-state reference greatly outperforms the same-observer gain reference, the unresolved issue is planner/observer adequacy, not evidence for a neural architecture. A negative result means this fixed task/controller did not qualify. It does not prove that adaptive control is useless. A stronger MPC comparison would need its own prospectively matched horizon/budget across all rows; it is not an automatic rescue stage of this screen.

## Exact proposed continuation arithmetic

These are proposed practical thresholds to accept before any screen outcome. They extend the design's 5% oracle and 3% ID margins with an absolute floor and a first-movement check. All checks are required; secondary plots cannot rescue a failure. No epsilon, rounding, discarded case or selected controller substitutes for a comparison.

For controller `m`, case `i`, and action window `W_i`, define

```text
c_i(m,W) = -sum(native_rewards_i[m,t] for t in W_i) / len(W_i)
cbar(m,W) = mean over every case of c_i(m,W)
```

Each case therefore has equal weight even though change 80 and change 120 leave different post-change lengths. Cost includes actual normalized noisy control effort; do not use clipped imagined reward for qualification. For a changed case with switch root `tau`, use:

- `PRE = [0,tau)`.
- `POST = [tau,200)`, including the first action under changed gain.
- `MOVE = [100,150)` for change 80, and `[150,200)` for change 120. Each is the first complete new-goal interval after the hidden change, exactly 50 actions.

Apply the following separately to the low-to-high and high-to-low switched regimes:

1. **Gain knowledge matters.** For both baselines `B` in `{public nominal, public ID then freeze}`, require `cbar(current-gain,POST) <= 0.95*cbar(B,POST)` **and** `cbar(B,POST)-cbar(current-gain,POST) >= 0.001`. The second threshold is 0.001 native-cost unit per action, an explicit engineering materiality choice that prevents a large percentage of an almost-zero tail from passing. Also require current-gain mean `MOVE` cost no higher than `B`, and nonworse `POST` mean within each switch-time stratum.
2. **Actual online ID helps.** Against both `B`, require `cbar(rolling ID,POST) <= 0.97*cbar(B,POST)` **and** `cbar(B,POST)-cbar(rolling ID,POST) >= 0.001`. Require nonworse mean `MOVE` cost and nonworse `POST` mean in both switch-time strata. Require rolling-ID `PRE` mean no more than 1.05 times each baseline. Retain individual paired costs even though this qualification is a case-average criterion, not a worst-case guarantee.
3. **The identifier recovers in time.** At planning roots `tau+30` through `tau+49`, require the case-mean absolute gain error to be at most 0.1 at **every** offset. Also require at least 75% of cases to have average absolute error at most 0.1 over those 20 roots: 3/4 in screening and 12/16 in confirmation. Root `tau+30` may use exactly 30 completed post-change commands; root `tau` cannot use the first changed outcome prematurely. Audit-only true gain grades estimates but never enters an update. Retain gain error by switch time and all cases.

Additional requirements:

- **Ordinary and no-change competence:** on the 50-action, static-target, gain 1 check and the 200-action nominal tracking regime, require the true-state/current-gain reference's full-episode mean cost to be at most 0.90 times zero-command cost. Require rolling ID to be no more than 1.05 times public nominal on both. In confirmation also require that same 5% non-regression on stationary gain 0.7 and 1.3. Report the public current-gain gap to the privileged state reference; a large gap is evidence of state-estimation difficulty, not adaptation difficulty.
- **Identification engineering prerequisite:** a separately fixed pulse/coast diagnostic with legal public inputs must identify both stationary gains to mean absolute error at most 0.1 over its final 20 roots after at least 30 completed transitions. A zero-command **and zero-noise** diagnostic must expose non-identification, not an apparent correct estimate from a prior or tie rule. These checks precede interpreting a control failure. Their exact action array/case count must be recorded before running; it cannot be chosen by estimator error.
- **Compute usefulness on this platform:** propose a separate 10% incremental execution budget: for each switched direction, summed whole control-row wall time of rolling ID must be at most 1.10 times public nominal over the same cases. Include model/grid construction, observer updates, identification simulations, CEM, native stepping, copies, checks and trace writes. Also report decision-only/adaptation-only time and stored bytes. This is a deployment-cost admission rule, not a conversion of native reward to seconds. If the team does not adopt this budget before launch, report the two axes separately and do not claim that the extra compute is worthwhile. Shared-host timing is not isolated latency or real-time certification.

For percent improvement displays, use `100*(cbar(B)-cbar(A))/cbar(B)`. Do not average casewise percentages, mix sum-based and per-action denominators, or substitute zero-command cost as the percentage denominator. If the baseline mean is not strictly positive, mark the percentage undefined and the improvement criterion non-qualifying. The inequalities above remain the primary arithmetic. Retain distance and control-effort components and absolute changes alongside totals; a benefit from less effort alone must not be described as lower tracking error.

Screening passes only if all these applicable checks pass on its four cases after all rows finish and saved replay succeeds. A pass authorizes the fixed fresh-case confirmation, not a neural run. Confirmation must independently pass the same checks plus its stationary-regime safeguards. These are practical sample-average admission thresholds, not statistical significance tests. Neither stage establishes unseen continuous-gain generalization or a need for long memory. If the four-transition residual-window diagnostic matches 20 transitions, report that directly. If the oracle passes but ID fails, diagnose the ordinary estimator before promoting a neural alternative.

## Workload and practical staging

Both stages retain the ordinary 50-action check. The first screen uses nominal tracking and two switched regimes; confirmation adds both stationary non-unit gains. Confirmation uses sixteen entirely fresh cases, not the screen's four plus twelve new ones.

| Coverage | Four-case screen | Sixteen-case confirmation |
|---|---:|---:|
| Tracking regimes | 3 | 5 |
| Batched controller rows, including ordinary check | 24 | 36 |
| Actual episodes | 96 | 576 |
| Actual native decisions | 15,600 | 100,800 |
| CEM candidate nominal transitions | 38,584,320 | 249,937,920 |
| Selected-first-action nominal transitions | 13,000 | 84,000 |
| Gain-ID nominal-transition upper bound | 68,040 | 433,440 |

Here `sum_t min(12,T-t)` is 534 for `T=50` and 2334 for `T=200`. Candidate work is `N*5 planned rows*256*(R*2334+534)`. Selected work is `N*5*(R*200+50)`. Gain-ID bounds allow the rolling row to score every returned transition, including terminal, and the frozen row to score the first 40: `N*21*(R*(200+40)+50+40)`. If the final packet is not assimilated for ID, bind the smaller actual count explicitly. Each nominal transition contains two native RK4 substeps. Fixed excited diagnostics and engineering checks are additional, not hidden in these totals.

Completed source-matched engineering profiling provides a useful scale, not a budget guarantee. The prior known-state CEM256 row measured 64 cases by 50 actions, 8,749,056 candidate transitions, 144.123 seconds whole-call execution and 133.738 seconds independent saved replay. Its raw row occupied 707,255,503 bytes. Linear candidate-work scaling gives:

| Proxy using that one old row | Screen | Confirmation |
|---|---:|---:|
| Execution | 635.6 s / 10.6 min | 4,117.2 s / 68.6 min |
| Independent replay | 589.8 s / 9.8 min | 3,820.5 s / 63.7 min |
| Raw per-row artifact scaling | 3.12 GB | 20.20 GB |

Sources: [measurement](../reacher-two-observation-capacity-v1/attempt-02/execution/measurement.json), SHA256 `4a361dc72e0d4f27759c75459fc52209c86437579de01f65695f7d5206b92c99`; [repaired saved audit](../reacher-two-observation-capacity-v1/attempt-02-audit-repair-01/measurement-audit.json), SHA256 `c037b4b5914f45b2252b98b9c93aff5aab81746618dc2e0237373f2b7509cd88`. Only engineering timings, counts and storage metadata were used.

The new wrapper, gain banks, target events, batch size and record schema differ. These proxies exclude new ID overhead, final whole-tree hashing and any new trace growth, and are not selected caps. A single retained source-matched engineering row plus saved replay is enough to check the estimate before choosing explicit limits; no new general preparation framework is needed. Preserve every failed or timed-out attempt, with no automatic retries or horizon changes. Storage can be reduced by a prospectively specified representation, but required evidence must not be discarded after seeing a result.

This staged qualification answers whether gain information and conventional online identification improve this controller's cost at an acceptable measured overhead. It does not answer whether connectome wiring, fast weights, normalized innovation or a recurrent world model is novel or effective.
