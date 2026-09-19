# Optional rollout pruning for geometry-scored CEM

Written 2026-09-19 05:41:02 UTC from source inspection. No current outcomes, saved candidate traces, capacity measurements, models or native simulations were inspected or executed. This is an optional future implementation design, not an algorithmic novelty or speed claim. The current 116 bound sources and scientific comparison remain unchanged.

## Decision and precise scope

**Exact elite/proposal/winner preservation is possible on finite, admissible rollouts. A generally equivalent replacement of the current fail-closed, fully recorded execution is not established.** A pruned tail can conceal an invalid prediction that would stop the full evaluator, and its uncomputed scores/angles cannot honestly appear as exact diagnostics. Do not install pruning in the current study or return invented complete score arrays through its existing callback interface.

The frozen kernel proposes four generations of64 candidates. The first three generations each supply the next generation's eight elites. The final generation contains63 random candidates plus one fully evaluated proposal mean. The executed winner is the best candidate across all256, with earliest global ID winning ties. Elites are current-generation only, not a retained global elite pool.

## Bound the actual clipped sum

For candidate `i`, actual native-step horizon is `H=min(12,50-t)`. Expanded commands repeat each quantized action block three times and truncate at `H`. Let `c_j` be the existing expected clipped-action cost for command `a_j`; its implemented output is in `[0,2]`. Geometry supplies nonnegative distance `d_j`, so the planner's per-step **clipped** cost is

`q_j = min(d_j + c_j, 2.5) >= c_j`.

In exact arithmetic, after `k` evaluated steps,

`L_i(k) = sum(j<k, q_j) + sum(k<=j<H, c_j)`

is a lower bound on final cost. Equivalently, its negative is an upper bound on the reward sum. Unclipped accumulated distance-plus-action cost is not the optimized quantity; it can overstate a lower bound when clipping is active. Use the exact expanded float32 commands, fixed noise configuration and actual truncated horizon. Do not use block costs without their repetition counts, include unused trailing steps, or compare a mean prefix against a summed incumbent. Dividing every complete candidate by a common horizon is mathematically rank-equivalent but changes the frozen arithmetic and is unnecessary.

The implementation accumulates clipped rewards **sequentially in float32**, then exposes that result in float64. A separately reduced future sum is not automatically a sound bound for those machine scores. One conservative construction starts from the actual float32 prefix reward accumulator and repeatedly adds `-c_j` in the original future-step order. Floating-point addition is monotone on finite operands, and each actual clipped reward is at most `-c_j`, so this yields an upper envelope if the precomputed action costs are identical to, or conservatively below, the scorer's values. Different batch shapes/math kernels require verification or outward-rounded intervals. An arbitrary epsilon or float64 summation is not a proof of safety.

## Incumbents, ties and CEM semantics

Keep a completed global incumbent and, for generations0-2, eight fully scored current-generation candidates. A deterministic seed schedule can fully evaluate the first eight candidate IDs before attempting pruning. Until eight complete candidates exist, a global incumbent alone is insufficient: a candidate unable to win globally may still change the current generation's elite set and all later proposals.

For a reward upper bound `U`, the conservative rule in generations0-2 is:

`prune only if U < current_eighth_score AND U < global_best_score`.

In generation3, only the global winner matters, so use `U < global_best_score`. Protect the final proposal-mean slot from pruning and fully evaluate its entire horizon, retaining its original global ID255 and its paid work. Fully evaluate the ultimately selected candidate and retain the separate selected one-step advance from the real root.

Do not use `<=` blindly. Equality may belong to an earlier ID and change the stable elite order or global winner. Keeping every equality candidate is the simplest safe policy; lexicographic score/ID certificates could later prune provably losing ties. Incumbents must be exact completed scores, not another candidate's optimistic bound. As further candidates complete, the completed eighth-best threshold can only improve, so an earlier strict exclusion remains safe.

Preserve the **ordered** eight elites from the original stable descending-score sort, not merely their set. Fit moments from the same clipped float32 command chunks converted to float64; retain population standard deviation (`ddof=0`), floor`1e-3`, no smoothing and no retained elites. Preserve every proposal, anchor overwrite, final mean, command quantization and full `SearchInputs` identity, including unused draws. Pruning must not request replacement candidates or alter random allocation. These invariants support an induction over generations, conditional on exact survivor scores and valid bound certificates.

## Why this needs a separate implementation

- The existing callback promises a finite exact score for every candidate, and its receipts claim every scheduled rollout transition occurred. A future pruned kernel needs explicit complete/pruned statuses, evaluated prefix lengths, bounds and incumbent/elite IDs at exclusion. Never disguise an upper bound or sentinel as an observed score, or count256 proposals as256 complete neural rollouts.
- Full evaluation rejects nonfinite predicted angles or learned reward values and angle-pair norms below`1e-6`. A dominated candidate can still encounter either defect later. Skipping that tail changes failure behavior. Exact whole-program equivalence would require a sound tail-validity certificate or continued validation of all tails; neither is supplied by a positive action-cost bound. The strongest justified initial claim is selection equivalence on independently verified admissible traces.
- Current learned scoring evaluates a full bank together. Producing eight completed incumbents early requires a different evaluation schedule; merely masking rows while still calling the full batch saves no neural arithmetic. Compaction can reduce vectorization and change floating-point kernels. Compare pruning against an unpruned evaluator using the same microbatch schedule as well as the original full-bank evaluator. Do not assume mathematically equal predictions are bitwise equal after reshaping batches.
- Compute the bound using the known actuator expectation, not a new uncharged model. Precomputing all candidate action costs, conservative arithmetic, survivor gathers/scatters, threshold maintenance, validation and recording are paid work. Reusing precomputed costs in geometry is another source-bound change; it must not silently double-charge or omit the action penalty. Original learned-head work remains part of every evaluated learned transition.

Apply the same bound, tie policy, paid mean and proof obligations to the supplied-physics CEM references. The current physics adapter simulates the whole bank before its batched geometry call; early incumbents and prefix distances therefore require a separately charged change of scoring schedule. Its nominal branches remain private, and the selected advance still restarts from the original observer root. Physics tail failures can likewise be hidden by pruning. Different achievable savings do not make wall time matched; report actual completed candidate/selected native steps and gate costs separately.

## Bounded future validation

First use only complete, authenticated saved traces under a prospectively fixed candidate-evaluation schedule. Reconstruct every exclusion certificate and verify identical ordered elites, moments, proposals and global winner. Report skipped-transition opportunities and gate-operation counts as **replay-derived work estimates**, not measured speed or a general failure-equivalence proof. Include saturation, equal scores/earlier IDs, late incumbent improvement, final mean wins, terminal short horizons, and deliberately invalid hidden tails that demonstrate the validity limitation.

Only afterward consider a separate live engineering benchmark with fixed inputs/weights and matched scheduling controls. Measure complete decision/row wall time, original heads, bound construction, validation, copies, storage, failed-prefix preservation and selected advances for both learned and physics methods. Preserve all cases, including cases where pruning is slower. Any future efficacy comparison and admission rule belongs in a new protocol.

Source basis, all under `src/openjev/research/`:

| File | SHA-256 |
| --- | --- |
| `reacher_geometry_reward.py` | `62c650a4414cb56c0a88923993f1713b15ec8e956141f49f6137a0271b701c4e` |
| `reacher_reward_residual.py` | `72aadee9db1362f258edd4813e56e517c64fdb74dd9f54b0b67a21f2a00d9308` |
| `reacher_adaptive_search.py` | `c4041429363240a6bcbc5b520cc71aeaa8208c58e61844a14d797293842054ba` |
| `reacher_geometry_control.py` | `eaeaa95320fbc957bc61fe1aeadd4e3ea5b1d3cb39e62122cc921ddeb2214738` |
| `reacher_geometry_physics.py` | `31332b0226a27a44602548bd8a8679bf0c4456d6f9f4ca505dfdebff28aab101` |
