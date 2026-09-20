# Next execution candidate: separate observation and state terms

Follow-up: this proposal was implemented and [measured](dialogue-token-factoring-results.md).
It passes 14/16 speed requirements but fails the full qualification. The
original prospective design follows.

Status: source-based design, unimplemented and unmeasured. The
[projected-pooling comparison](dialogue-token-projection-results.md) passes
10/16 speed checks, including all four real-mask cases, but fails its fixed
admission rule. That attempt remains closed. This design targets work left
unchanged by it; it is not a retry or a new scientific result.

## Split an existing linear layer

The head's first layer currently consumes 396 features: six 64-dimensional
observation/schema terms, ten lexical values, own belief and entropy. Only
the last two depend on the recurrent state. Write its preactivation as

`h = Linear_observation([u, q, c, u*q, u*c, q*c, lexical]) + Linear_state([belief, entropy])`

The first term uses the existing weight's first 394 columns and its bias.
The second uses its final two columns without a second bias. Apply the
existing `tanh` only after adding both terms, followed by the unchanged output
layer, transition and state normalization. Readout resets its belief before
computing the state terms, as before.

Compute the first term within each public-mask pooling group, after the
original 384-dimensional pooling normalization and turn projection. Use one
394-to-64 operation per group, not six separate small matrix products. Scatter
this 64-dimensional preactivation rather than the projected turn vector.
All lexical/schema gathers, concatenations, column slicing, bias handling and
scattering must be included in measured cost.

Skipped entries need their correct schema-dependent first term with
`u=tanh(turn_projection.bias)` and zero lexical values. A zero fill is
incorrect. Compute a full-schema fill once per forward and broadcast it over
turns, then replace supported real-turn positions. No computed tensor or
gradient may be reused after an optimizer update.

## Keep monitoring on the consumed state features

A feature-container adapter can retain the original Linear and Tanh children,
parameter objects, names and initialization. Its ordinary full-feature call
must delegate unchanged for the inherited single-step API. Its explicit split
call accepts the actual `[belief, entropy]` tensor first and the precomputed
observation term second. The existing feature pre-hook then checks precisely
the own-belief value consumed by that operation through `[..., -2]`.

Do not create a fake full feature tensor merely for the monitor, disable the
hook, or audit a detached substitute instead of the actual input. Retain
every incoming, feature, outgoing and scalar departure-mass check in turn
order, including dummy NONE support and padded-state carry. Tests must prove
the split path and the ordinary inherited path both use the same checks.

## Cost hypothesis and limits

For the fixed geometry, the original first layer processes 88,320 dense
positions with 396 input and 64 output dimensions: **2,238,382,080**
multiply-accumulates. Grouping the 13,555 supported positions, conservatively
adding all 3,840 schema positions for skipped-cell fill, and retaining a
two-feature state projection on all 88,320 positions gives **449,937,280**
multiply-accumulates for that layer. This is an algebraic work estimate,
excluding other layers and operations, not a runtime or hardware-FLOP result.

The smaller and denser cells have less avoidable work. They remain required;
the sparse geometry gain cannot justify dropping them. More operations or
unfavorable memory layouts can erase the arithmetic savings.

If implemented, freeze a distinct source closure and bounded protocol. Keep
all sixteen workloads, sample seeds, numerical tolerances, complete-update
accounting, canonical optimizer-state resets, and original speed thresholds.
Require output, every input/parameter gradient, all-padding zero-versus-None,
feature-monitor coverage and ordinary-step parity before timing. Any measured
benefit remains an execution result for the same model. A passing cost screen
would still need a separately declared representative check before scientific
training; the earlier incomplete training cannot resume.
