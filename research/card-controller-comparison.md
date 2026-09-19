# Stable scoring and exploration with frozen memories

This follows the [learned card-memory pilot](card-memory-pilot.md), whose proposed update failed its continuation rule. Saved-state analysis showed that tiny probability row-sum differences changed many first-card choices. This comparison tests the controller while keeping every trained weight fixed.

## Three policies, the same models and fresh decks

- **A, original:** reproduce the previous picker exactly.
- **B, stable:** normalize probability rows in float64 and treat scores within `1e-12` of the maximum as tied. Select the first eligible index.
- **C, explore:** use B's scores and tie set, but prefer an unseen first card among tied pair endpoints. The second-card rule stays identical to B.

A versus B measures normalization and numerical tie handling together. C versus B isolates the added exploration preference. C may select the higher-index unseen endpoint of a tied pair first; it cannot select an endpoint outside that score tie set.

All 18 final checkpoints and both symbolic references play all three policies on the same 64 fresh reset seeds: 3,840 games, with no retraining or checkpoint selection. Every seed is disjoint from the previous train, development, evaluation and engineering sets. Episode order is fixed before execution.

The prospective exploration criterion requires C-minus-B mean return of at least 0.03 across all learned fits, nonnegative mean differences in every family, and positive differences in at least two of each family's three paired fits. The original architecture result remains failed regardless of this outcome.

The actor receives only public observations and the same seen/matched/phase bookkeeping. A separate evaluator hashes the hidden layout immediately after reset to verify paired deck identity; only the digest is retained, and the actor never receives the hidden ranks. Public reveals are used to independently reconstruct and check complete layouts afterward.

Timing includes the complete instrumented controller. A deliberately calls the frozen original chooser after computing diagnostic scores, so it has an extra scoring pass. These timings do not establish an intrinsic speedup from B or C. A pure saved-output test reproduced all 130,730 original decisions before any fresh games.

[Protocol](../evidence/card-controllers-v1/protocol.json) · [Fresh seeds and order](../evidence/card-controllers-v1/inputs.json) · [Prior diagnostic](../output/card-memory-pilot-v1/picker-diagnostic-01.json)

Results are pending the fixed run. This is a controller comparison, not an architectural novelty claim or a substitute for further held-out and scenario-shift validation.
