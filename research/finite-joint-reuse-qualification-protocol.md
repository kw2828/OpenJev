# Qualification: reuse computation inside one joint objective

This is a separate engineering mechanism after the
[equal-update feasibility stop](finite-update-learning-stop-results.md).
That stop remains closed. Its 90-second projection threshold, update counts
and frozen sources do not change, and no scientific comparison is admitted
by this component qualification.

The new `finite_joint_reuse.joint_objective` computes the unchanged weighted
joint loss while sharing probability construction and one endpoint prefix
filter. For a nonempty endpoint minibatch, the frozen implementation builds
the same probability fields five times. The new helper builds them once and
shares one eligible-prefix state between separate blind and observed
continuations. Every shared tensor remains attached to autograd and lives
only within that call. There is no persistent cache, optimizer change,
detachment or replacement of model methods.

Keep all-attempt prefix likelihood separate from the endpoint prefix pass.
Its reset arithmetic uses `emission.sum()/8`, whereas endpoint filtering
uses `(emission/8).sum()`. Preserve both operation orders. Also retain the
per-horizon readout calls and prediction before each future observation is
conditioned upon. Found events are scored once and later padding is ignored.

The objective remains `n/b * (e/S * endpoint_loss + prefix_NLL_sum/E)`.
Denominators are global attempted cases, surviving endpoint cases and valid
prefix events. Empty endpoint batches retain the graph-zero term for every
model parameter. Work is split into five disjoint blocks: shared fields,
endpoint prefix, prefix likelihood, blind continuation and observed
continuation. Shared operations are counted once. Counts are not measured
speedups or complete backward FLOPs.

Use only fabricated public tokens and fixed targets, with model seed 943101.
No world generator, scientific split, teacher request, saved checkpoint or
held-out performance metric is used. Test all three model arms and horizons
1, 2 and 8, including partial batches, batch size 64, no eligible endpoints,
found/padded histories, ownership, failures and recomputation after state
changes. Compare values and losses with the frozen functions at absolute
and relative tolerance 1e-10, and raw gradients, clipped gradients and three
Adam steps at 1e-9. Sharing changes gradient accumulation order, so these
checks do not claim bitwise equivalence or identical long training runs.

Freeze source and runtime before one native-supervised qualification. Run
lint and the declared component test file in that order, with a 180-second
cap covering startup through process cleanup. Disable ambient pytest plugins
and bytecode writes; use one numerical thread. Bound logs to 8 MiB at command
boundaries. Record peak parent and child RSS after termination, without a
continuous memory-limit claim. Any failed command or incomplete native
closure stops this attempt. Retain original logs, hashes and snapshots; no
in-place edits or reruns of this registration.

A pass establishes only the checked numerical equivalence properties. A
fresh integration, independently audited saved outputs and a bounded timing
comparison are still required before a speed or feasibility claim. This is
implementation efficiency, not an architectural or scientific result.
