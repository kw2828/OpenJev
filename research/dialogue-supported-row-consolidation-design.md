# Next execution candidate: consolidate supported evidence rows

Status: prospective source-only design. No implementation, new measurement or
training admission. The completed factoring qualification passed all sixteen
parity checks and fourteen speed checks, but failed its fixed admission rule.
That run and every earlier qualification and incomplete training run remain
closed.

## Evidence and limits

This proposal follows the authenticated
[factoring completion receipt](../output/dialogue-token-factoring-v1/qualification-01/completed.json)
and its [saved summary](../output/dialogue-token-factoring-v1/qualification-01/summary.json).
The completion SHA256 is
`47f5bdc270a00a58e8b2755531bb93db1dae3d9b72cd5caa0c81ec8432dfa700`;
all 76 payload hashes and byte sizes were independently checked.

The two failures were `large-candidate-scalar`, with a median paired
original/factored update-time ratio of **1.070468**, and
`geometry-slot-readout`, at **1.004527**. Both required at least **1.10**.
For the latter, median forward/validation/loss time fell from 95.17 to
58.67 milliseconds, while backward/clipping rose from 82.02 to 113.29
milliseconds. The large candidate-scalar cell improved both broad phases but
still missed the paired whole-update threshold.

These saved regions do not isolate matrix operations, indexing, allocation,
autograd or monitoring. They do not prove a cause for either failure, and
separate phase medians need not sum to the median update time. The geometry
uses synthetic values and one predetermined public mask layout, not a
representative corpus workload or a quality evaluation.

## Preserve the function, consolidate two operations

Keep the existing public-mask groups, attention scores, softmax, weighted raw
token sums and 384-dimensional L2 normalization unchanged. Retain each group's
supported normalized evidence rows and concatenate them into one `S x 384`
tensor, where `S` counts supported real-turn candidate positions, including
supported dummy NONE candidates.

Apply the existing biased turn projection and tanh once to all supported rows.
Gather the corresponding projected query, candidate and lexical inputs in
exactly the same deterministic row order. Construct the same 394 exogenous
features and apply the existing first-layer columns and bias once to all `S`
rows. Scatter the resulting hidden vectors using their aligned destination
indices. Do not replicate raw token sequences along query or candidate axes.

Keep the full-schema skipped-cell fill once per forward, including turn bias,
projected schema interactions and zero lexical features. Retain every ordered
state update and actual incoming, feature, outgoing and departure-mass check.
The feature monitor must still inspect the live belief and entropy consumed
by the split operation. Preserve padding carry, masked support, empty-path
zero-versus-absent gradients, parameter names and initialization. No activation
or outcome may be cached across updates.

## Work and memory tradeoff

The grouped attention work and mathematical multiply-accumulate counts remain
unchanged. Only the turn and exogenous projection calls are consolidated:

| Fixed shape | Supported rows | Calls for each supported-row projection | New float32 pooled concatenation |
|---|---:|---:|---:|
| Minimum | 42 | 2 to 1 | 0.065 MB |
| Medium | 43,613 | 3 to 1 | 66.99 MB |
| Large | 69,461 | 6 to 1 | 106.69 MB |
| Geometry | 13,555 | 69 to 1 | 20.82 MB |

The exogenous projection counts exclude the unchanged single schema-fill
operation. The current implementation concatenates 64-wide hidden rows;
this proposal instead concatenates 384-wide supported evidence before the
two projections. For geometry, those concatenation payloads are 3.47 and
20.82 MB respectively, while the avoided dense 384-wide tensor would be
135.66 MB. Payload sizes use decimal MB and do not predict peak memory or
physical memory traffic. Concatenation, gathers, retained gradients and their
allocation lifetimes must all be paid and measured.

Fewer projection calls and separate gradient-accumulation branches could help,
but larger intermediates can erase that benefit. The large cell consolidates
only six calls while introducing a supported buffer exceeding 100 MB. There
is no speed claim. Matrix shape changes also change floating reduction order;
algebraic equivalence is not bitwise optimizer-trajectory equivalence.

## One prospective qualification

If implemented, freeze one new bounded sixteen-cell protocol before execution.
Keep every existing workload, seed, numerical tolerance, optimizer-state reset,
whole-update accounting rule, cap and speed gate unchanged. Validate outputs,
loss, all input and parameter gradients, normalization, monitoring coverage,
padding and inherited step behavior. Include all concatenation, indexing,
allocation and counter work in timing.

Do not selectively retime the failed cells, choose execution paths from their
outcomes, relax thresholds or resume earlier runs. A passing engineering
qualification would still require a separately declared representative cost
check before scientific training. This is an execution hypothesis for the same
model, with no new evidence about prediction quality or architecture novelty.
