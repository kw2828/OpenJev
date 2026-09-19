# Optional exact streaming implementation of the two-observation control

Written 2026-09-19 05:36:32 UTC, from source inspection only. This is a prospective optimization note, not an implementation, measurement, qualification or amendment to the current study. No capacity measurements, scientific outcomes, models or native environments were used. The current 116 bound sources and reconstruction comparison remain unchanged.

## Decision

**Yes, two recurrent branches can reproduce the rebuilt roots and predictions mathematically, for reachable states under fixed parameters.** They must each start from zero at an actual valid observation and consume only the actual public packet/action sequence from that anchor. This is reuse of deterministic computation, not a new architecture or a stronger information source. It does not establish bitwise floating-point or optimizer-trajectory equivalence without tests.

Let `R(b,t)` be the unchanged parent GRU state obtained by zero initialization, assimilation of valid packet `p_b`, then chronological parent advances on issued commands `a_b ... a_(t-1)` and assimilations of actual packets through `p_t`. The current history model returns `R(b,t)`, where `b` is the older of the last two valid observations, or zero before the second observation. Its left-padding operations are executed but masked out of the state; in finite real arithmetic they are identities.

Maintain branches `older=R(u,t)` and `newest=R(v,t)` for the latest two valid indices `u<v`. Before a second valid packet, both logical identities may refer to the one branch `R(0,t)`.

1. At startup, require visible `p_0`, and initialize the branch with the unchanged parent assimilation from zero.
2. On a missing real packet, advance both branches using the preceding **acknowledged issued command**, then assimilate the sanitized packet. Their anchors remain unchanged.
3. On a valid real packet at `t`, advance and assimilate the branch previously anchored at `v`; promote it to `older=R(v,t)`. Discard the branch anchored before `v`. Create `newest=R(t,t)` by assimilating `p_t` from zero.
4. Plan solely from `older`. Candidate states are private. A candidate terminal, candidate action or predicted observation never updates either real branch.

Induction establishes the invariant: missing packets append the same operations as reconstruction; a new valid packet selects precisely the previously newest anchor and creates the next zero-start branch. Identical roots, fixed weights and identical candidate commands therefore produce identical mathematical predictions and scores. CEM action equality additionally depends on preserving numerical scores, ordering and the earliest-best tie rule.

## State and boundary requirements

Retain the actual sanitized suffix, indices, presence masks, issued commands and static target as audit evidence initially. Preserve the current twelve-packet/eleven-command limit, age tolerance, visible startup, valid reacquisition and terminal50 rules. A streaming cache could physically continue longer, but accepting such a history would change this comparator.

The full parent state is `hidden` **and** `packet`. Even on missing measurements, parent assimilation replaces the packet and its public validity/age while retaining the transitioned hidden vector. Carrying only hidden state while leaving a stale packet changes the next transition, which consumes `action + packet[4:]`. Each branch must have the actual sanitized packet at every real boundary. Newest-branch initialization uses the valid packet's actual target and zero age.

Use an explicit real-observation commit boundary. The current controller already computes an active-branch one-step selected advance. It may be reused after the enclosing native trace acknowledges that exact command, provided the root and parameter version still match. Advance the background newest branch on that same acknowledged command. Alternatively advance both inside the next real-observation operation and charge the repeated active computation. Do not infer real commitment merely because a generic `advance` was called: the planner also calls it on counterfactual candidates.

A distinct actual class/configuration and resume format must bind both anchors, branch states, public evidence, clocks, pending command and parameter identity. Do not load this behavior under the old class name or treat a compatible weight tensor schema as complete restoration authority. Existing resistance to arbitrary corruption of previous hidden/predicted state does **not** automatically apply to cached branch state. Equivalence is for correctly maintained reachable caches; shape/hash checks alone cannot prove their neural values without replay.

## Counterexamples and training caveats

- One persistent branch with a renamed anchor is insufficient. An attainable scalar GRU example has observation update `U(h,p)=h/2+x(p)/2` and transition `T(h,a)=h/2`. After three consecutive valid observations, the persistent root retains `x(p_0)/32`; reconstruction from `p_1` has no such term. The separately zero-started newest branch is what removes the obsolete dependency.
- Resetting a branch at a missing packet discards its valid anchor. Assimilating predicted angles as measurements, substituting applied noisy actions for issued commands, or carrying a candidate terminal changes the function and information boundary.
- A cached state produced under parameters `theta_old` is generally unequal to reconstruction under updated `theta_new`. Discard or reconstruct all branches after a parameter update. The current sequence loss holds parameters fixed through a complete minibatch forward and updates once afterward, so a fresh per-sequence cache is feasible; cross-minibatch reuse is not justified.
- Reusing a prefix in an autograd graph can preserve the mathematical gradient: contributions from every downstream root/loss must add at the shared prefix. Do not detach branches, update tensors in place, drop earlier loss references or backpropagate/update parameters halfway through the sequence. Removing an obsolete branch from future roots must not erase gradients from losses already using it. The newest branch must start from zero, never clone an older branch's hidden vector.
- Masked padding and discarded readouts have zero influence only under the stated finite, deterministic computation. Overflow, NaNs, stochastic layers, mutable buffers or different precision could break that argument. Shared graph accumulation and changed batch shapes may also change floating-point rounding, clipping decisions and later Adam updates. Claim algebraic equivalence, not exact final fitted weights, until independently checked.

## Work and validation

With both branches maintained uniformly, a fifty-decision deployment can use one startup assimilation plus two parent assimilations at each of the other49 roots: at most99 instead of600 observation-update sample calls per case. Reusing the acknowledged active selected advance requires50 active selected advances plus49 background advances:99 instead of550 replay advances plus50 selected advances. These counts include full parent heads and the analytic reward skip; they do not silently elide unused readouts. Startup aliasing could reduce work further but need not be the first implementation.

The unchanged CEM256 search already pays `256 * sum(min(12,50-t)) = 136704` candidate transitions per case. Consequently the removed real-boundary work is less than1% of the current GRU64 affine-MAC total, using the existing operation formula. This is **not** a wall-time forecast. Training has no CEM search, so eliminating repeated reconstruction can have a much larger arithmetic effect. Validation, branch/public-buffer copies, hooks, graph storage, readouts, I/O and failed work still require measured accounting. Separately removing history buffers from candidate state might save additional work, but should be identified and tested as a second optimization.

Before any deployment experiment, compare the same fixed tensors on synthetic histories: startup; consecutive valid packets; six- and ten-step gaps; twelve-slot valid reacquisition; heterogeneous batch schedules; changed older past with identical retained suffix; zero/nonzero command alignment; poisoned missing placeholders; unsupported overflow; and terminal50. Compare every root, both branch anchors, selected advance and a common private candidate bank, checking mutation isolation and actual-command acknowledgement. Preserve dtype/device/batch shapes first; record exact equality and maximum discrepancies rather than assuming a tolerance proves identical CEM decisions.

Training validation should compare all unchanged sequence-loss terms and named parameter/input gradients, then a few paired optimizer steps with the same original initialization, batch order and Adam settings. Any discrepancy requires diagnosis before a claim of exact training equivalence. A first useful future comparison is an independently source-bound deployment optimization on the **same** checkpoint, with parity and complete row timing. Do not replace current traces or retroactively revise their cost claims.

Source basis (SHA-256):

- `reacher_two_observation_history.py`: `67420a414941a16123f9ebdd82b2f073f947e45be73d4039c261335cb0e8d1aa`
- `reacher_world_models.py`: `84a4a9f9122d0e8099bac19e04619faff22adabf0acf516eff9d5367ded2060e`
- `reacher_reward_residual.py`: `72aadee9db1362f258edd4813e56e517c64fdb74dd9f54b0b67a21f2a00d9308`
- `reacher_two_observation_control.py`: `73247ad5d46733b27a5968128ff5f8e3dcde795f9c0596be83cd583eea2fc77f`

All four source files are under `src/openjev/research/`. This note claims neither biological superiority nor novelty; it isolates a possible implementation cost of the known bounded-history control.
