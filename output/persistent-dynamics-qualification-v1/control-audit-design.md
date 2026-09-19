# Compact saved-output audit for tracking control

Prospective engineering contract, September 19, 2026. No new simulation, scoring, model inference or scientific allocation was performed for this note. The completed pulse probe establishes approximate noiseless identification only; it does not qualify tracking control. Keep the existing 116 study and 136 pilot sources unchanged.

## Minimum evidence

Use the existing `PhysicsGeometryCEM` numerical contract through `GainPlanningBank`. Save complete native **executed** episode states, but omit candidate qpos/qvel trajectories from successful decision files. The independent auditor can reconstruct candidate trajectories transiently from authenticated roots and issued sequences. This verifies their scores and selected action, not an unrecorded internal trajectory or a learned mechanism.

At execution level, bind the request/configuration, source bytes, native XML/model fingerprints, runtime, explicit engineering or scientific stream ledger, allowed gain vector, case schedules, exact file membership/hashes, caps, actual terminal process status and failed attempts. Public information and audit-only schedules must be separate. Hash checks and serialization belong inside enclosing wall caps.

For each row retain:

- Native record: all public packets and issued commands; post-target-event decision states; each transition's applied noise/action, actual gain/gears, reward target, next target, native pre-event endpoint, distance/control reward and termination flags. Keep initial integration state and declared reset identity.
- Observer history: public three-point observer roots; identifier update records with their pre-transition roots, actual issued command, observed endpoint packet, 21 candidate cosine/sine predictions, residual vector, 20-entry rolling sums, prior/new gain and exactly-flat flag. Save which updates actually ran. A frozen-estimate row must explicitly bind the last included transition and subsequent retained gain.
- Planning setup/final snapshots: nominal binary hash, gain-specific model hashes and fixed configuration for every eager bank member; all member and aggregate work counters; inclusive setup/plan/observer/row times. Unused eager models still cost setup.

For each planned decision `t` retain one compact NPZ plus metadata:

| Field | Required identity/shape for one case |
|---|---|
| Root | qpos/qvel float64 `[1,4]`, current public target float32 `[1,2]`, step, row identity |
| Gain | exact float64 allowed value/index and selected member model hash |
| Innovations | bound full `SearchInputs`: initial, unused random-extra, three CEM arrays, their source role and hashes; store once and share across rows |
| Proposals | float32 `[1,256,H,2]` sequences, or lossless scored chunk arrays with exact expansion/truncation rule |
| Scores | original per-offset geometry rewards float32 `[1,256,H]`; original accumulated scores float32 or their exact float64 promotion |
| Selection | global IDs 0..255, chosen int64 ID, issued float32 action and full chosen sequence |
| Selected re-advance | qpos/qvel before/after, predicted cosine/sine, geometry reward and its distinct work counters |
| Work | four ordered 64-candidate bank summaries plus selected summary, callback/input/root identities and elapsed costs |

`H=min(12, steps-t)`. Keeping per-offset rewards costs 12 KiB at H12 per case/decision and makes the exact clipping/reduction auditable even when NumPy/Torch trigonometry differs slightly. No need to save every candidate angle, geometry component or physical state. Do not infer successful work from zero-filled arrays. On failure retain the compact completed prefix, masks/cursor/substep counts, failed scratch native state, attempted/completed counters and original exception. A failed search cannot be accepted as a completed decision.

## Independent checks

1. Reconstruct proposals with independent array arithmetic, not `search()` or a controller. Initial chunks use the declared 0.25/0.75 scales and seven anchors, then clip/cast to float32. Stable previous-generation top-eight elites determine float64 mean/population standard deviation with 0.001 floor. Three refinement banks retain the final paid mean at ID255. Verify every saved sequence exactly, including near-terminal truncation. Pair innovations across rows; later banks legitimately differ with scores.
2. Verify each recorded score by sequential float32 accumulation of per-offset rewards clipped to [-2.5,0]. Verify earliest global argmax, selected sequence and first action exactly. This reconstructs a saved decision; it does not make another online decision or replace a near tie.
3. Independently replay every nominal candidate: copy authenticated nominal physics, install declared gain, reset private data, restore root qpos/qvel and time `t*dt`, forward once; issue commands with two separate native substeps and one postconstraint refresh per offset. No realized action noise, future gain, future target or estimator update enters a branch. Reconstruct float32 cosine/sine and approximate planar FK with lengths 0.10/0.11 plus the unchanged expected clipped normalized-action cost exactly once. Compare per-offset rewards with a prospectively declared float32 arithmetic tolerance (existing independent geometry tolerance: rtol 2e-6, atol 5e-7). Retain maximum error. Recorded score/selection checks remain exact. Never rerank from tolerant replay or silently repair scores.
4. Replay the separately paid selected one-action prediction from the original root. Its qpos/qvel and quantized angles must equal its chosen candidate's first step exactly; geometry arithmetic can use the same declared tolerance. It must not replace the real observer state.
5. Reconstruct public roots from observed packets: principal angles initially, shortest signed increments, startup velocity zero, then one backward difference, then `(3*d_latest-d_previous)/(2*dt)`. Identify from the *previous* root and issued command before assimilating the endpoint. Recompute all candidate residuals and rolling sums; exactly flat retains the previous estimate, otherwise earliest minimum. Residual separation is not calibrated confidence. Public rows must match this root; privileged-state rows alone may use actual decision qpos/qvel.
6. Enforce row gain permissions: nominal uses 1; rolling uses only completed past identifier updates; frozen uses precisely the agreed cutoff estimate; current-parameter oracle and state-plus-parameter oracle alone receive actual current gain. Audit schedules never enter public state construction. Cross-row root estimates can diverge because earlier actions differ; only exogenous initial state, noise, target/gain schedules and innovations are paired.

## Event and work boundaries

Actual transition `t` installs `200*g_t` before native stepping and uses the target visible in packet `t`. Native endpoint/reward are saved **before** installing target `t+1`; the next decision state/packet is saved afterward. Target events preserve arm qpos/qvel and physical time, set target qpos/current goal and target velocity zero, then forward. A gain switch alters the model parameter, not saved integration state or the observer. Every planning branch holds the current allowed gain/target throughout H even if a real event lies inside that horizon. Native integration state alone cannot restore changed gears.

For each successful one-case planned decision: 256 candidate sequences plus one selected restart; candidate transitions `256*H`, selected transitions 1; native substeps twice these counts; 257 resets and forwards; postconstraint calls equal transitions; five geometry calls and `256*H+1` geometry samples. Attempted equals completed only on success. Sum adapter members to bank lifetime; separate candidate, selected, identifier and real-plant counts. Identification adds 21 nominal transitions and 42 substeps per actual update, including updates before an estimate is frozen. Observer arithmetic itself makes no native call. The zero-action row incurs no planner/identifier simulation unless the protocol explicitly specifies one.

For any complete S-decision planned row, `sum(H)=12*S-66` when S>=12. Thus candidate work is 597,504 transitions for S200, or 136,704 for S50, plus S selected transitions. The proposed 80 main cases with five planned rows plus 16 ordinary cases would require 249,937,920 candidate transitions and 84,000 selected transitions; actual plant work across six rows would be 100,800 transitions. These are sizing formulas, not an authorized protocol. Independent candidate replay pays comparable native work and must have its own measured cap before launch.

Wall costs must keep enclosing times and nested components distinct: bank setup includes eager copies, plan wall includes search/journal copying, estimator wall includes its native bank, row wall includes public reconstruction, native execution, validation, hashes and trace I/O. Do not sum nested times twice or present batched/shared-host wall time as isolated latency. Full completion and scientific qualification are separate; an audited failed gate remains failed.

Source basis: `reacher_geometry_physics.py`, `reacher_adaptive_search.py`, `reacher_tracking_control.py`, `reacher_tracking_dynamics.py`, `reacher_tracking_identification.py`, and independent geometry arithmetic in `scripts/audit_reacher_geometry_study.py`. The final compact writer and audit must share an explicit serialized schema; this note does not itself authenticate a future run or permit preparation.
