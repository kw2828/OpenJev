# Saved common-prefix return consistency diagnostic

Status: proposed descriptive protocol, outside the frozen scalar-return study. Root review and externally pinned successful audit evidence are required before a separately recorded execution. No inference, training, environment, policy or simulator call is proposed.

## Fixed scope

Include **all 144 existing parity rows**: `min8`, `mlp8` and `homogeneous8`, each with fitting seeds 10101, 10102 and 10103, on the first eight TRAIN and first eight VALID rows. Require that exact Cartesian product. Retain all sixteen action/observation branches, including blocked-action scores and zero-raw-mass branches. Do not select a subset by outcome or error.

These are **16 mechanically selected parity prefixes**, not representative validation coverage. All are steps 0 through 7 of exactly two length-three, initial-hit-one teacher episodes: TRAIN `train:lambda3:910001:teacher` (79 total steps) and VALID `valid:lambda3:930001:teacher` (14 total steps). No length-four/five settings, other initial-hit strata, or later prefixes occur. Only eight are VALID prefixes, versus 1,109 states in the complete validation set. Neither 144 model-prefix rows nor 16 prefixes are independent evaluation episodes.

## Input authentication and teacher-action join

A new diagnostic plan must pin this design, its implementation/runtime, exclusive output path, and external SHA-256 identities for the completed scalar-return plan, worker receipt, successful supervisor terminal and independent audit receipt. Require matching source/plan identities and audit agreement. Verify all 44 producer payload hashes and the complete audit payload closure before numerical decoding. Missing, partial, changed or failed evidence stops execution without replacement.

Use `otto-return-value-v1/run-01/parity.jsonl`, `train-rows.jsonl` and `valid-rows.jsonl`. Reach the original symmetry teacher's `collection-transitions.jsonl` through the pinned `prior_receipt` lineage and authenticate its original payload closure. Other summaries establish provenance and the unchanged study decision, never selection.

Join parity to metadata by split and `row_index`; require identical `episode_id`, `prefix_index` and posterior SHA. Verify `target=(total_steps-prefix_index)/64` and current public step. The teacher action is from the original transition with **the same episode ID, `kind='step'` and `step=prefix_index+1`**. Require its `posterior_before` hash and allowed actions to match the current metadata, and its action to be eligible. Never use the preceding action. Stream only the required TRAIN/VALID teacher records; do not decode DAgger or EVAL.

## Quantities and action selection

For each saved row:

- `C = 64 * scalar_numpy`: predicted current physical cost.
- `Y = 64 * metadata.target`: realized teacher return.
- `p[a,h] = raw_masses[a,h]`, `w[a,h] = weights[a,h]`, `v[a,h] = values_numpy[4*a+h]`, already in physical cost units.
- `b[a,h] = w[a,h]*v[a,h]`; reconstructed `Q[a] = 1 + sum_h b[a,h]`.

Validate finite quantities, declared shapes, nonnegative raw masses and `w=max(p,1e-10)`. Compare reconstructed float64 costs with saved `costs_numpy` at the existing `atol=rtol=1e-10` bound and retain the differences. No renormalization, clipping, zero-branch suppression or model recomputation.

Use **saved costs** for selection. Among eligible actions, choose the smallest numeric ID whose cost differs from the eligible minimum by **strictly less than 1e-10**. Require exact agreement with `action_numpy`; retain the minimum, tie IDs and chosen-minus-minimum gap. Numerical comparison tolerance never changes this action rule. A near-tie choice need not equal the exact argmin.

For teacher action `a_T` and greedy action `a_G`, save:

1. Current return error: `C-Y`.
2. Teacher-action residual: `costs_numpy[a_T]-C`.
3. Greedy-action residual: `costs_numpy[a_G]-C`.
4. Predicted switching advantage: `costs_numpy[a_T]-costs_numpy[a_G]`.
5. Four observation contributions: `b[a_T,h]-b[a_G,h]`.

Check the contributions sum to the switching advantage within the same arithmetic bound, and teacher residual minus greedy residual equals that advantage. Preserve all sixteen branch products. Keep signed values and tiny negative advantages from the near-tie convention; never clamp them.

## Reporting and limits

Publish every joined row, branch record and input identity. Summaries may show signed mean, mean absolute and RMS return error and residuals, mean switching advantage and teacher-disagreement count, separately for every fit/split with its eight-prefix denominator. Retain all three seeds; any family mean equally averages them. No confidence intervals, significance tests or best-seed selection.

Targets are noisy **Monte Carlo returns under the analytic teacher**, not optimal values, counterfactual action returns or branch labels. Lower current-state scalar error does not establish correct action ranking. Residuals measure consistency of the deployed floored backup, not exact physical Bellman errors: subfloor branch inputs can be subnormalized and biased zero-input values are retained. Teacher-action residual measures internal consistency, not observed action-value error. Even an exact teacher value can have a nonzero greedy residual if switching improves the policy. Teacher disagreement is not automatically policy error.

A common inconsistency would motivate examining shared targets and branch coverage; a min8-specific pattern would motivate a matched representation-versus-coverage question. Neither identifies causation or an architecture limitation from sixteen selected prefixes. Full VALID branch analysis would require separately scoped checkpoint inference.

There is **no efficacy gate, architecture admission, policy repair or revised continuation rule**. All original decisions remain unchanged. No original file is modified. Use new exclusive outputs, record source/input/output hashes, and preserve failures.
