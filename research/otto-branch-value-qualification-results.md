# Observation-branch value readout: synthetic qualification

**The new numerical readout is implemented and its focused checks pass. No model has been trained or evaluated with it.** The completed [720-search comparison](otto-symmetry-head-results.md) still fails its 2/54 continuation rule.

The next candidate evaluates possible observations before choosing an action. It constructs all sixteen action/hit branches from the public belief and known observation kernel, then evaluates a scalar value for each branch. The default calculation retains the reference policy's probability floor, subnormalized inputs, blocked stay-and-observe branches and final action restriction. It does not use hidden source coordinates or simulator state.

[Implementation](../src/openjev/research/otto_value_branches.py) · [Independent scalar-oracle tests](../tests/test_otto_value_branches.py) · [Design and subsequent fitting requirements](otto-branch-value-qualification-design.md)

## What passed

The initial full suite passed **57 tests**. Three lint findings led to mechanical corrections: an invalid input container now raises `TypeError`, imports were ordered, and the scalar test oracle spells its existing floor with `max`. All **24 affected tests** passed afterward, with 33 deselected, and final lint passed. The final numerical implementation is unchanged from the initial passing suite. A separate launcher syntax error occurred before any test command; its source and failure record are retained.

Independent source review checked the calculations, and a second readback reconstructed the exact earlier source/test hashes by reversing only those documented corrections. The evidence distinguishes the initial full run from the final affected-test run.

| Check | Scope |
| --- | --- |
| Independent scalar oracle | Dense/asymmetric and sparse beliefs, all sixteen branches, interior, edges and corners |
| Tiny branch probabilities | Exactly zero, below the floor, at the floor and above it, without renormalization |
| Value calculation | Generic callback, common or action-conditioned minimum-linear banks, raw four-action costs |
| Float64 shortcut | Fixed `atol=rtol=1e-12` on bounded synthetic coefficients, with selected actions checked separately |
| Zero inputs | A biased value remains weighted by the positive floor; the minimum-linear value alone guarantees zero at zero input |
| Ownership and selection | Immutable owned arrays, explicit eligible IDs and the strict first-ID tie rule |
| Float32 limits | Separate cast/reduction behavior, a changed-action tie example and an underflow counterexample to shortcut equivalence |

[Initial test/lint receipt](../output/otto-branch-value-qualification-v1/engineering-01/receipt.json) · [Final affected-test/lint receipt](../output/otto-branch-value-qualification-v1/engineering-01/receipt-02.json) · [Final source review](../output/otto-branch-value-qualification-v1/root-source-review.json)

## The shortcut remains optional

A minimum of linear values with fixed coefficients and no additive bias is positively homogeneous. In exact arithmetic, multiplying a normalized branch value by its normalization weight gives the value of the unnormalized branch. This can remove normalization work from a future specialized implementation. The current module exposes both routes for comparison; it does not demonstrate a speed improvement.

Floating-point calculations can differ. One deliberately extreme float32 fixture loses a tiny branch when casting the raw branch before evaluation, while the explicit route retains a contribution. Ordinary biased networks also fail the algebraic identity. **The explicit route remains the default.** These checks do not establish TensorFlow parity, correctness for every coefficient bank, or a qualified deployed policy.

Full observation backups and alpha-vector value representations are established POMDP methods, including in [Loisy and Heinonen's olfactory-search benchmark, Sections 2 and 3](https://arxiv.org/html/2302.00706v2). This module is an engineering foundation, not a novelty result or a recurrent world model.

Before fitting, freeze a separate protocol comparing a structured value bank with an MLP under identical explicit branches, public inputs, scalar targets, training states, seeds and complete computation accounting. Existing relative action preferences must not be relabeled as remaining-search cost. Any autonomous comparison needs fresh evaluation cases and the analytic control. Compact recurrent memory remains a later decision contingent on competent full-belief control.
