# Prospective execution control: compute only required schema fill

Follow-up: this proposal was implemented and [measured](dialogue-token-required-fill-results.md).
All numerical checks and 15/16 speed requirements pass, but the small
candidate/readout workload still fails. The original prospective design follows.

Status: source-only proposal. No implementation, numerical tests, new timing,
model calls or training admission. The consolidation qualification and every
earlier run remain closed. This proposes one general public-mask optimization,
not a repeat of its failed cell.

The [consolidation completion](../output/dialogue-token-consolidation-v1/qualification-01/completed.json)
has SHA256 `51be227a111793b99d238d4dbe91f3d83db58a71c4a0e0ac1a47cfc8562fd368`.
All 81 listed payload hashes and byte sizes were checked. Its
[summary](../output/dialogue-token-consolidation-v1/qualification-01/summary.json)
passes all 16 parity checks and 15 speed requirements, but fails engineering
admission. `minimum-candidate-readout` has median paired original/consolidated
whole-update time ratio **0.8970424854**, below **0.90**. All twelve larger cells
pass; the four geometry ratios range from **1.6084873464 to 1.8701879084**.
These measurements establish neither quality nor full-training admission.

The failed cell's saved forward/validation/loss region is lower in each of its
four measured pairs, while backward/clipping is higher in each pair. Those
broad regions do not isolate fill, indexing, allocation, autograd or monitoring.
Separate phase medians need not add to a whole-update median. Neither proximity
to the threshold nor comparison with earlier runs establishes a cause or
justifies another measurement of this cell.

## Source-proven unused work

The frozen [consolidated implementation](../src/openjev/research/dialogue_token_consolidated.py)
has SHA256 `03fa31be4ac35ff302fe34b367ef8a632416b7ff42c4aa11c331d7b25bf26ec8`.
It computes one exogenous fill vector for every schema coordinate `(b,q,c)`,
expands this across time, then overwrites every supported candidate on every
real turn with its pooled-evidence vector. Thus a fill row contributes nowhere
if that dialogue has no padded turns and the candidate is supported.

The exact required-row predicate is:

```text
needs_fill[b,q,c] = not all_t(valid[b,t]) or not candidate_mask[b,q,c]
```

This depends only on public masks. It uses no method name, case size, timing,
labels, predictions or learned values. A single padded turn makes fill necessary
for all schema coordinates of that dialogue. Unsupported candidates retain
their fill even though output logits are masked. Dummy NONE candidates use
ordinary support. Token-mask holes do not change the predicate because a valid
real turn must already have nonempty token support.

| Fixed mask layout | All schema rows | Required fill rows | Removed rows | Removed first-layer MACs |
|---|---:|---:|---:|---:|
| Minimum | 7 | 0 | 7 | 176,512 |
| Medium | 3,840 | 3,025 | 815 | 20,551,040 |
| Large | 3,840 | 3,025 | 815 | 20,551,040 |
| Geometry | 3,840 | 3,789 | 51 | 1,286,016 |

Counts use the frozen synthetic layout formulas and authenticated public
geometry snapshot, without generating values or reading quality outputs.
MACs count `removed_rows * 394 * 64` for the exogenous first layer. Geometry
removes only 1.328125% of fill rows. These are logical operations, not a kernel
profile, elapsed-time estimate, corpus-density estimate or memory-traffic claim.

## One additive implementation, if pursued

Keep grouped attention, raw-evidence normalization, supported-row concatenation,
both supported projections and their deterministic scatter order unchanged.
Gather only required schema rows for fill. Compute their exact existing inputs:
`u0 = tanh(turn_projection.bias)`, projected query and candidate vectors, their
three products, and zero lexical features. Apply the same existing feature
columns and bias, then place those vectors at their original schema indices.

Use initialized zero storage for omitted rows, expand the schema baseline
across time and apply the existing unique supported-destination scatter. Every
omitted destination must be overwritten before any recurrent step consumes it.
No uninitialized storage is permitted. The all-required case may retain the
existing full fill path; the zero-required case skips the fill projection.
Both decisions must follow the predicate, with no workload-specific threshold.

Keep the full ordered recurrence and actual incoming, feature, outgoing and
departure-mass checks. All-padding requires all schema fill rows and retains
the old empty projection and safe zero-gradient connections. When no fill is
needed, supported paths still use the same parameters and schema inputs, but
zero-versus-absent input and parameter gradients require explicit verification.
Preserve parameter names, initialization, ordinary step behavior and ownership.
Do not cache masks, projections or outcomes across calls or updates.

Dense time-by-schema hidden storage remains necessary in this design. Mask
reduction, index extraction, gathers and selective scatter are new paid work;
they may cost more than the removed small projection. Existing attention,
supported projections, state feature computation and monitoring remain. Report
actual required fill rows, fill calls, MACs and indexing payloads separately
from the old full-fill reference. Do not claim a peak-memory reduction.

## Bounded decision

This is a credible dead-work elimination candidate. Its effect on the speed
failure remains unmeasured. Test it alone to separate its cost from other
optimizations.

If implemented, first verify all four arms in float32 and float64 against the
original implementation: outputs, loss, every input and parameter gradient
including exact None membership, actual monitor coverage, supported and masked
schema features, mixed padding, all-padding, token holes, dummy NONE, nonzero
biases, generic widths, causal prefixes and inherited step behavior. Explicitly
witness zero required fill calls when all rows are overwritten and exact fill
coverage otherwise. Floating reduction and failure boundaries are not promised
bitwise equivalent over all finite inputs.

Then freeze one new complete sixteen-cell qualification with unchanged cells,
seeds, tolerances, paired reset procedure, four measured pairs, whole-update
accounting, 300-second cap, 6 GiB RSS limit and speed requirements. Include all
mask/index/counter work inside timing. No selective retiming, threshold change,
retry, quality inspection or reopening of the failed training attempt follows
from this proposal. A further failure closes this implementation under that
rule; a pass would establish bounded engineering admission only.
