# Direct readout solve on frozen recurrent features

This is a separate development diagnostic following the completed comparable-cost
study. It preserves all earlier studies and their closed TEST/confirmation panels.
The question is whether a direct quadratic solve provides a stronger readout
control before another recurrent architecture is introduced.

## Fixed comparison

Use all three original 80-epoch parents (309000001, 309000002, 309000003), the
same 54 TRAIN episodes and existing Adam74/full-joint40 checkpoints. Extract
one canonical TRAIN feature cache per parent: period four, batch one, chronological
chunks of 32, frozen predictor, CPU float32. Labels remain separate from model
inputs. Hidden states on later queries precede assimilation of that query's
teacher score. First queries are unsupported. Charge every actual extraction
operation, including readouts the unchanged predictor still executes.

Fit two head increments per parent, with all six backbone tensors unchanged:
minimum-norm least squares and ridge with fixed coefficient **0.0001** on all
87 action-contrast increment coordinates, including biases. Use a fixed
orthonormal four-action contrast basis and float64 SVD least squares with
`rcond=1e-10`. No regularization search, selected seeds, optimizer continuation,
retries, replacement checkpoints or best-of-solver selection is admitted.

The objective is the [documented quadratic](otto-linear-probe-proposal.md).
Use the census's float64 episode weights directly, dividing nonquery rows by
their legal-action count and later queries by four. It is a canonical batch-one
surrogate, not the literal historical shuffled batch-six float32 Adam program.
Report parent, Adam74, OLS and ridge objectives on the same cache; OLS minimizes
the unpenalized objective and ridge minimizes a distinct penalized objective.
Report numerical rank, spectra, conditioning, residual norms and objective values.

Export each solved head to ordinary float32 inference. Validate all four heads
(parent, Adam74, OLS, ridge) against their normalized cached affine predictions
on every supported TRAIN row, with `atol=1e-5, rtol=1e-5`. Report maximum absolute
and tolerance-scaled discrepancy. This tolerance establishes numerical agreement,
not bitwise identity or equal action choices near ties. Any parity failure closes
the scientific attempt as failed and prevents DEV evaluation.

## Development evaluation and decision

After every solve and TRAIN export check completes, evaluate all five methods
(including historical full-joint40) on the same **already exposed 36 DEV paths**
from the comparable-cost study. Reusing this panel is explicit. It is held out
from fitting, but is neither fresh evidence nor independent confirmation. No
new teacher, simulator or Astra calls occur. All 15 views are retained.

Ridge is the fixed candidate; OLS is a control, not a fallback winner. The
diagnostic continuation rule requires at least 5% lower later case-weighted raw
teacher-cost gap than both Adam74 and the parent, with no full-gap regression
against either, in all six seed/setting comparisons. A zero comparator cannot
establish a positive relative improvement. Publish all checks whether they pass
or fail. Full-joint40 supplies a reference, not a selected comparator.

Successful fitting or lower TRAIN loss alone does not establish decision quality.
Even a diagnostic pass requires separately registered untouched paths and
autonomous validation before an effectiveness claim. This study tests a
conventional optimization baseline, not a new architecture or learning rule.

## Costs and execution

Report cache extraction, design construction, each solve, float32 export,
ordinary TRAIN validation and DEV inference separately. Charge the entire common
cache/design cost to each solver when stating standalone adaptation cost. Keep
historical Adam/joint fit costs labeled historical; no matched wall-clock or
end-to-end deployment speedup is claimed. Shared pretraining and data collection
remain disclosed prior costs.

Before decoding empirical arrays, qualify changed modules and runner controls
using fabricated tests, lint and a fabricated numerical capacity probe; bind their
bytes, all parent/comparator/data descriptors, the original successful process
closures and the runtime in a committed registration. Numerical work uses one
CPU thread. The original worker has 1,800 seconds, 4 GiB peak RSS and 1 GiB output;
the saved-output audit has 600 seconds, 2 GiB RSS and 512 MiB output. Use the existing
suspend-aware process-group supervisor and exclusive attempt paths. Preserve
failed attempts without scientific retries or extensions.

The independent audit reconstructs cached scalar losses, exported-head and frozen
tensor witnesses, prediction metrics and all paired checks from saved outputs.
It performs no model inference or fitting. Reports require both original process
closures and unchanged sources. TRAIN parity is producer evidence backed by
fabricated cross-path tests; the audit does not rerun a model to reproduce it.
