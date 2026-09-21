# Proposed branch-value qualification, before fitting

This is an engineering proposal, not a new scientific result or an execution protocol. The completed symmetry pilot and all 54 original conditions remain unchanged. Its saved diagnostic found no exactly-zero-mass packets or exact position/posterior-hash repeats, while 164 of 213 failed learned searches alternated between two positions throughout their last 256 decisions. Those observations motivate testing action/value computation; they do not establish causation or missing memory.

## One numerical contract first

Implement a new pure NumPy module, `otto_value_branches.py`, and synthetic tests. Accept only the public float64 53x53 belief, public position, known float64 4x107x107 likelihood kernel and explicit eligible actions. No simulator, sampled source, seed, trajectory or learned checkpoint input is needed for this qualification.

Use **released RLPolicy branch semantics**, not SAI semantics. For all four clamped successor positions x', form each of four nonfound hit branches:

\[
u_{a,h}(s)=b(s)L_h(s,x'),\quad m_{a,h}=\sum_su_{a,h}(s),\quad
w_{a,h}=\max(m_{a,h},10^{-10}),\quad z_{a,h}=u_{a,h}/w_{a,h}.
\]

The frozen kernel's zero origin excludes found-source mass from nonfound branches. Preserve that zero exactly. Zero branches stay zero. Do not renormalize the sixteen weights, repair a subnormalized belief, or replace the numerical likelihood kernel. Center each z on its successor position exactly as the qualified public view does. Score `1 + sum_h(w * C(z, successor_position, known_kernel))`. Compute all four scores, then select only from the declared eligible IDs with the existing strict first-within-1e-10 rule. A blocked direction still has a legitimate stay-and-observe branch; its exclusion is a policy choice.

Expose raw centered u, raw masses, floored weights, centered normalized/subnormalized z and successor coordinates in an immutable result. Keep float64 reference arithmetic separate from an explicitly labeled float32 cast/reduction route. No TensorFlow model or native environment is required.

## What homogeneity does and does not permit

For `C_x(v) = min_j dot(alpha_j(x, lambda), v)`, with no additive bias and coefficients independent of v, positive homogeneity gives `w*C_x(u/w) = C_x(u)` for every positive w, including the declared floor. A fused structured score can therefore use unnormalized branches **in exact arithmetic**. Position, boundaries and supplied kernel may condition coefficients; belief-dependent conditioning, square-root belief features, ordinary biased MLPs and additive output offsets invalidate this shortcut.

This is not a proof of bit-identical floating-point scores or choices. Casting before versus after division changes rounding and underflow, and a tiny score change can split a tie. Qualify the explicit route first. Compare float64 explicit/fused values with fixed `atol=rtol=1e-12` on bounded synthetic coefficients; require exact selected actions separately. Report float32 discrepancies and ties independently. A failed fused-action comparison retains the explicit route; do not widen tolerances or silently repair costs.

Required fixtures: zero and subnormalized beliefs; sparse and asymmetric beliefs; interior and all four boundaries; zero/subfloor/exact-floor/above-floor branch masses; immediate finding; nonuniform action masses; tied and separated costs; and a biased-MLP counterexample that must fail the homogeneity identity. Test input immutability, all sixteen branches, and unchanged sum-of-weight semantics.

Keep found termination separate from zero-input model behavior. The found outcome has zero continuation and is excluded by the kernel origin. An arbitrary biased value model can return a nonzero value on a zero nonfound branch, which the released numerical policy still weights by 1e-10. Do not suppress that call or force its value to zero. Only the homogeneous min-linear form guarantees `C(0)=0`; it does not guarantee `C(delta_origin)=0`. A terminal value override belongs to an explicit found-state contract, not to generic branch evaluation.

## Subsequent fitting decision

Only after this mechanical qualification should a new pilot compare the min-linear value with a parameter-budget-matched MLP using **the same explicit branches, public inputs, scalar targets, TRAIN states, fitting seeds and action set**. A bias-free homogeneous MLP is a stronger later control if a gain is attributed specifically to piecewise linear structure. Charge branch construction and complete inference, not only network calls.

Do not label existing relative SAI action preferences as cost-to-go. SAI performs two normalize-only-above-threshold operations and a log distance/entropy potential; it is neither the RLPolicy floor route nor automatically homogeneous off the simplex. Scalar SAI-potential distillation would be a separately declared approximation experiment. To claim value improvement, use matched remaining-search-cost or Bellman targets, preserve censored trajectories without treating truncation as found/zero continuation, and freeze that target recipe separately before fitting.

Loisy and Heinonen already use complete observation-branch backups and neural value learning, and describe alpha-vector value representations. Their reward-maximization convention becomes a min-linear representation after changing sign to remaining cost. Learned templates do not inherit PBVI guarantees merely from their shape. [Primary paper, Sections 2 and 3](https://arxiv.org/html/2302.00706v2).

For any later autonomous test, retain analytic control, fresh evaluation cases, all seeds and full capped-search cost. Continue reporting the same oscillation/mass diagnostic. A fixed no-immediate-reversal rule or a separately budgeted analytic fallback can serve as intervention controls; both change the policy and neither proves better memory. Stop architectural escalation if the value heads fail full-belief competence or a simple intervention explains the gain. Do not admit compact recurrence from this qualification alone.

Suggested ownership: architecture agent owns the new pure branch module/tests; a second reviewer independently derives floor, centering and terminal fixtures; root freezes the eventual target/data/compute protocol. No fitting, simulator call or evaluation is part of this next engineering step.
