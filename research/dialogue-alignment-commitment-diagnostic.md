# Saved-distribution commitment diagnostic

Prospective posthoc specification, 20 September 2026. No diagnostic calculation, model call or fitting was performed for this note. The completed token-alignment campaign remains **FAIL, 18/22 checks passed**. This diagnostic cannot reverse that decision, choose a new winner, or establish autonomous memory performance.

The [audited aggregates](../output/dialogue-token-alignment-scientific-v1/report-01/summary.json) motivate one question: can token-mean's probability of retaining the supplied previous candidate reduce aligned's retention harm while preserving the observed changed-state advantage of aligned's alternative ranking? Against mean, aligned adds 181/75/19 unmentioned-retention errors across seeds, while assigned retention improves in aggregate. These are repeated predictions on the same exposed rows, not independent examples or evidence of an internal recurrent cause.

## Inputs and fixed constructions

Use all 13,599 saved evaluation rows, all three seeds 6201/6202/6203, and the nine authenticated saved distributions. Join only by the exact canonical `row_indices`. The previous candidate is `previous_current_index` from the existing authenticated `evaluation-rows.jsonl`, an explicitly privileged input already supplied to every model. Current labels and transition groups are used only for descriptive scoring from those same rows. No new gold lookup, corpus, cache, checkpoint or encoder access is needed.

For each seed, `M` is token_mean and `A` is token_aligned. Construct exactly:

| Name | Stay/change mass source | Ranking among alternatives |
|---|---|---|
| MM | M | M, self reconstruction |
| AA | A | A, self reconstruction |
| MA | M | A, primary diagnostic combination |
| AM | A | M, reverse combination |

Flat-stratum remains a reported reference for every seed. Compare all four constructions with both existing controls, flat-stratum and token-mean, and show differences from aligned. Do not mix flat into another construction. No weights, thresholds, seed selection, temperature, confidence clamp or outcome-dependent routing are fitted.

## Numerical definition

Promote saved float32 logs to float64. Require the original schema `[13599,12]`, finite nonpositive supported logs, negative infinity on padding, exact canonical row order, valid supported previous indices and original probability-mass error at most `2e-6`. Every row has at least one alternative. Reject violations; do not repair input support or add floors.

For each source X and row with supported candidates V and previous index j:

```text
Z_X       = logsumexp(l_X[c], c in V)
q_X[c]    = l_X[c] - Z_X
stay_X    = q_X[j]
change_X  = logsumexp(q_X[c], c in V excluding j)
rank_X[c] = q_X[c] - change_X, c != j

hybrid_(X,Y)[j] = stay_X
hybrid_(X,Y)[c] = change_X + rank_Y[c], c != j
```

Implement logsumexp by subtracting the maximum. Compute change mass directly from alternative logs, never by subtracting an exponentiated stay probability from one. Padding remains negative infinity. There is no epsilon, probability floor or final clipping. This explicit per-row normalization changes original raw-log NLL by `Z_X`; record its maximum absolute value and keep original raw-log report values distinct from normalized diagnostic values. It does not constitute recalibration.

Check all four constructed distributions for finite supported logs and float64 unit mass within `1e-12`. Check each self reconstruction against `q_X` with `atol=1e-10, rtol=1e-12`. For self cells, use canonical `q_X` for scored output after that identity check, avoiding cancellation-induced tie changes. Require their choices to reproduce the original saved-log argmax exactly. For cross cells, choose the first maximum in original candidate order, with exact floating equality only and no tolerance-based tie enlargement. Record exact tie counts. This is deterministic tie handling, not candidate-permutation invariance at ties. Stop and preserve a failed receipt on any numerical or identity mismatch, without silently changing tolerances.

Here, alternative ranking includes the full conditional distribution and its
concentration, not just the order of candidate scores. A hybrid's hard keep
decision compares the mass donor's stay/change odds with the largest
conditional alternative probability. Therefore MA need not make the same
hard keep decision as M. With exactly two supported candidates, there is only
one alternative: MA reconstructs M and AM reconstructs A. Include that
closed-form case in the synthetic checks. This clarification does not change
the four numerical constructions.

## Reporting and interpretation

Publish every construction and seed on all/seen-service/heldout-service panels, each with all, changed, retained, unmentioned-retention and assigned-retention strata. Keep inherited support membership fixed. The primary heldout panel has 7,819 rows: 578 changed, 7,241 retained, comprising 4,032 unmentioned retention and 3,209 assigned retention. Changed rows include 29 TRUE and five DONTCARE targets, with no FALSE or NONE changes. Show per-service breakdowns without selecting services afterward.

Report exact accuracy/error counts, wrong-selected-branch versus wrong-value counts, raw-target NLL of the constructed normalized distributions, Brier score, TRUE/DONTCARE recall and their inherited supported false-positive denominators. Preserve undefined zero-support cells. Give row, equal-service and equal-dialogue summaries, three-seed means and every paired seed difference. Pairwise correct-to-wrong/wrong-to-correct tables must use the same rows. Report both retained harm recovered relative to AA and changed advantage retained relative to MM; do not hide their tradeoff in a new weighted score.

The hypothesis is weakened if MA does not reduce retention errors, or if any reduction consumes its changed-state advantage over MM. A favorable pattern would justify discussing a separately trained commitment/value factorization with matched evidence and controls. It would not prove that a hidden gate caused the original errors: these factors are derived from final categorical distributions, not the model's internal branch gate. No minimum useful gain, new continuation threshold or diagnostic pass/fail rule is declared here; those require a separate discussion before another study. Both cross combinations are posthoc algebraic counterfactuals, not newly trained or deployed models. The cost of obtaining both source predictions is not free.

## Execution envelope

The diagnostic and independent primary-metric audit each have a checked
60-second wall limit, a 1 GiB process-lifetime peak-RSS limit and a 64 MiB
output limit. They use exclusive output directories and preserve failure
records. These bounds cover saved-output arithmetic, not model inference.
No actual diagnostic run is admitted until source review and synthetic tests
finish and the implementation, tests and this specification are hash-bound.
Any input-path or implementation failure is retained separately; no result
is silently overwritten. The independent audit recomputes the primary
panels with separate arithmetic rather than importing the producer's
construction or scoring functions.

## Frozen saved-input identities

Authenticate the following external pins before decoding rows or prediction arrays. The completion manifest supplies exact payload sizes and all nine prediction hashes; verify each selected payload and repeat hashes after scoring. Record diagnostic source, this note, input identities, work scope and terminal status in a new exclusive output directory. Do not modify any earlier receipt or result.

| Artifact | SHA256 |
|---|---|
| `training-01/completed.json` | `e7222147935b1f1604d31da83c940ed428ae3745dfddec374cfa64f7ddd1ce7c` |
| Training plan | `e700080ee2dd28c83c0c13a0dad2640cdc83fb1992efb4bfd3777a90111877c1` |
| `training-01/evaluation-rows.jsonl` | `56e572f6df34cf81ccc38db137d82a699eb3f4f84509af39682d7d963e62a4d0` |
| `report-01/receipt.json` | `7591182a5c07c9cd4ee4035128b6c0900d7d9177ba89efccfe92bfd528c8a6c5` |
| `report-01/summary.json` | `b4b8c1d3f6e9d1e98096c2684a5bca84b011496f89044e3f3eb31260f2fbbdc3` |
| `audit-01/result-01/receipt.json` | `8773cf48d0b07b10f6bb920549d3628a4716beb814c98185df021081046b19bd` |

Paths above are relative to `output/dialogue-token-alignment-scientific-v1/`. Within `training-01/fits/`, the pinned `predictions.npz` members are:

| Fit | SHA256 |
|---|---|
| flat_stratum-6201 | `dbccd26acbaa4622de3b2ecd66a752f7a06d67cc9598c880dc933ee1c8ba27ec` |
| flat_stratum-6202 | `1ff1ae1b6a8f746fd086a63e7ecfa355751e2c1bcb5a6f7cdeaa8cba42444537` |
| flat_stratum-6203 | `773ee404db257cfa81c5de0fe75bc6e495de10759c2094c9ac06857e256caf40` |
| token_mean-6201 | `1979aa6c082415d208d4ea5ab8dfff738a76c0f09efa569199a083f573e16dc3` |
| token_mean-6202 | `ec1534af8240f36585c5c6512414cf2eba9c13d27ffac7083c0e7411738ea72d` |
| token_mean-6203 | `2b20d456f367bf0fd2cff57ee0684faba454f0aeea54042cef956620b5098c9b` |
| token_aligned-6201 | `bae0149549750430358c52ef0ecff2b2e237704ff442fc232d6ea2a912d4a170` |
| token_aligned-6202 | `0067cbf5e9fb38a24a221873d27addea3f1cd3a5e9bda8125806afbb9bbe42d6` |
| token_aligned-6203 | `066b953cd9073fc4a927feeb77e0a6c44225f255957a4a57f038fd3745bcfb0a` |
