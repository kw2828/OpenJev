# Independent results-component review

**No material blocker found.** This is a static review of a pure arithmetic component, not a scientific result or permission to prepare a study. The reviewer did not execute the helper/tests, allocate RNG, read production outcomes, or make model/native calls.

Reviewed identities:

- `src/openjev/research/reacher_two_observation_results.py`: `59e5499726d20332623470b9d5f516076958ed1f845b188d6496ef4102dc3717`.
- `tests/test_reacher_two_observation_results.py`: `8f8db6df5e4d43a35a7686c761c591dc974e2521372c04f315aabd7d1516e61c`.
- Referenced prospective protocol: `572ec977c234be0d1c68ee9ac130abd3bf5132cb3854126a0d8f88ce0aa9362c`.

`evaluate_rows(plan, rows)` first validates the unchanged prospective settings, then requires exactly 42 distinct declared `(panel, label)` rows. Every row must contain exactly `panel`, `label`, `case_ids` and `native_rewards`. All three panels retain nine learned fits and five references. Duplicate, missing or unknown rows fail; row input ordering may vary but returned rows are ordered by the complete protocol manifest. There is no selected-fit or best-case path.

Case IDs must be explicit, nonempty strings, unique within a row and identically ordered across every fit, panel and reference. The helper sums all 50 rewards for every case in float64, retains each case cost and ID, computes every fit mean over all cases, and computes each family mean as the equal mean of all three fit means. Because every row has the same case count, this matches the declared balanced family aggregation. Original reward arrays, ID containers and protocol settings are not mutated or aliased in outputs.

The complete gate is reconstructed from the validated protocol manifest: four 3% gap-family checks, twelve paired-fit nonworsening checks, two full-sensing 2% bounds, six persistent-versus-zero competence checks and one known-state ordinary-gap competence check. All use direct inclusive `left <= multiplier * control_mean`, with no epsilon, rounding or percentage conversion. All 25 must pass. A failing pair therefore remains fatal even when the family mean improves. Particle/public-kinematic and secondary comparisons cannot rescue the gate; their full costs remain in the returned controls. Zero comparator cost uses the same literal inequality and does not fabricate a percentage improvement.

Reward inputs require an actual numeric NumPy array of the exact declared shape. Boolean, complex, object and list inputs are rejected. Nonfinite values, float64 conversion/sum overflow, negative total episode cost, nonfinite row/family means and overflowing criterion bounds are rejected rather than producing implicit winners. Malformed identities, case count/order differences, incomplete fields and modified criterion constants are also rejected. The caller still owns native authenticity: these checks cannot prove that arbitrary finite arrays or case labels came from the declared executed trajectories.

The tests meaningfully exercise a failed pair despite favorable family means, an isolated competence failure, inclusive boundaries and the next representable value above them, the final time step of an individual case, all references, reordered row input, output-copy behavior, malformed/nonfinite evidence and criterion tampering. The author reported 37 synthetic tests and Ruff success; the reviewer read those tests without rerunning them. This validation is separate from effectiveness evidence.

The component imports only NumPy, standard-library helpers and the already reviewed pure protocol. Its status is `arithmetic_completed`, and it explicitly excludes byte/source/model authentication, native replay, training/search verification, statistical inference, equivalence and launch authority. Future whole-study integration must authenticate the rows and IDs, validate all saved work, and preserve its failed gate exactly. The full independent auditor should check the declared 25 inequalities and source binding rather than treating this helper's returned booleans as evidence on their own.
