# Prospective execution control: share state columns within each forward

Status: source-only design. No implementation, tests, model calls, new timing
or training admission. The required-fill qualification and all earlier
qualification and incomplete training attempts remain closed.

The [required-fill completion](../output/dialogue-token-required-fill-v1/qualification-01/completed.json)
has SHA256 `465e6ec7a6326c063582262529522f7ef66d0449327363fc3e7ce9def46b3665`.
Its 86 listed payload hashes and byte sizes were checked. The
[summary](../output/dialogue-token-required-fill-v1/qualification-01/summary.json)
passes all sixteen numerical checks and fifteen speed requirements, but fails
engineering admission. Minimum candidate/readout has median paired
original/required-fill update-time ratio **0.8454861177**, below **0.90**.
All twelve larger cells pass, including geometry ratios **1.3303358254 to
1.6549745518**. This does not authorize full training or establish task quality.

The failed cell uses no schema-fill projection. In its four saved measured
updates, the separate median forward/validation/loss regions are 2.288 ms for
original and 2.499 ms for required-fill; backward/clipping is 1.250 ms and
1.655 ms respectively. These regions do not isolate slicing, gradient
accumulation, matrix kernels, indexing, allocation or monitoring. Phase medians
need not sum to the whole-update median. There is no causal or controlled
comparison with timing in another run, and no justification for retiming only
this cell.

## A repeated parameter view in the ordered head

The frozen [factored feature container](../src/openjev/research/dialogue_token_factored.py)
has SHA256 `d817ae4646da4bc4cb4380cf64e678c927237f71f7a2ee27bfd198e5008b6f42`.
Every split feature call evaluates:

```text
tanh(exogenous + linear(actual_belief_entropy, feature.weight[:, -2:]))
```

The sequential head invokes this expression once per time position. The
parameter values are unchanged within a forward, yet each invocation creates
a fresh slice view and its differentiable ancestry. The current required-fill
implementation, SHA256
`f52ad0ca268017e7903da0e076632391f38d1ec28a2b56c6a1721a2d098f54b4`,
inherits this path unchanged.

Create `state_columns = feature.weight[:, -2:]` once per forward in the caller's
current gradient context. Pass that same view explicitly to every ordered
state-feature call. Its separate linear-use contributions can accumulate at
the shared view before propagation through shared slice ancestry, instead of
having independently created slice paths. This is an autograd-graph hypothesis,
not a measured allocation or kernel claim. A view is not a copied weight tensor.

| Fixed shape | Time positions | State-column slice expressions, current to proposed | State linear calls |
|---|---:|---:|---:|
| Minimum | 6 | 6 to 1 | 6 unchanged |
| Medium | 15 | 15 to 1 | 15 unchanged |
| Large | 23 | 23 to 1 | 23 unchanged |
| Geometry | 23 | 23 to 1 | 23 unchanged |

The default shared view has 128 logical elements in a 64-by-2 strided view of
the existing 64-by-396 parameter. That describes shape, not new storage or
saved backward bytes. Pooling, fill, supported projections, state arithmetic,
all nonlinearities, normalization and every monitor check remain unchanged.

## Explicit local ownership and unchanged observed inputs

Use a private typed per-turn record containing the existing exogenous tensor
and the shared state-column view. A narrow head adapter must recognize this
record explicitly, never by tensor width. Its feature call must still take the
actual live belief/entropy tensor as argument zero, preserving the existing
monitor hook. Reuse the same Linear and Tanh child objects, parameter names,
initialization draws and scalar/readout transition equations. Ordinary
full-feature calls and inherited one-step behavior must delegate unchanged.

The view must remain differentiable. Do not detach, clone, enter an extra
no-grad context, change Parameter registration, use a buffer or retain a tensor
on the model between calls. Create a fresh view for each forward, including
after loading weights or an optimizer update. No public mask, workload name,
timing, outcome or recurrent value chooses whether to share it. Do not bypass
state computation in readout or padded rows.

The local record avoids a new model-wide cache and accidental reuse between
interleaved forward graphs. It does not make the current monitored wrapper
thread-safe: audit counters, active-check state and last-work metadata are
already mutable. Concurrent monitored calls still require independent instances
or external serialization. No parameter mutation may occur between a forward
and backward that depends on it. Exceptions must not leave a stored view on
the model.

Sharing gradient ancestry can change floating addition order. Neither bitwise
optimizer trajectories nor equivalence over all extreme finite inputs is
claimed. The all-padding graph and every zero-versus-absent gradient must remain
intact. Record actual source-level view expressions and logical dimensions,
alongside the unchanged state linear calls and MACs. Do not label those counts
as measured backward nodes, physical traffic, peak memory or elapsed savings.

## One bounded test of this mechanism

The source duplication is concrete enough to justify one additive control.
It does not establish that this will repair the failed cell. Keep the change
limited to sharing the state-column view, with no bundled monitor relaxation,
readout shortcut, extra packing or case-specific route.

Before freezing, check all four arms in float32 and float64 against the
original implementation: outputs, finite loss, every input and parameter
gradient, exact None membership, mixed and all padding, dummy NONE, nonzero
biases, generic widths, causal prefixes, candidate/query permutation and
ordinary inherited steps. Inspect actual feature-hook inputs and witness that
the same fresh view reaches every split call within one forward. Test two
forward graphs before a combined backward, and fresh forwards/backwards after
parameter changes. Confirm no persistent view, new tensor registration or
initialization draw. Preserve ordinary monitor counts and numerical tolerances.

If implemented, freeze one complete sixteen-cell qualification with the same
workloads, seeds, four measured pairs, canonical model/optimizer resets and
full-update accounting. Retain the **0.90** floor for each minimum cell,
**1.10** for each medium/large and geometry cell, **6 GiB** RSS limit and
**300-second** whole-command cap. Charge view creation, records, dispatch,
validation and diagnostics. No selective retiming, retries, cap or gate
changes. These exposed engineering fixtures do not provide fresh task-quality
evidence. A pass would support bounded implementation admission only.

## Implementation follow-up

The [shared-column implementation and status](dialogue-token-shared-columns-status.md)
now record 113 passing focused tests and a published 71-source qualification
freeze. The timed attempt has not started because another independent training
job is using the host. The design above is the original prospective rationale;
there is still no measured speed effect or training admission.
