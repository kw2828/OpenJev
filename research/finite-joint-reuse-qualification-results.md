# Joint computation reuse: component qualification

**QUALIFICATION_PASS: 44 fabricated tests passed.** This establishes the checked numerical equivalence of a per-call computation-sharing helper. It is not a speed benchmark, empirical fit or scientific advancement.

The original native-supervised process completed in 4.066847041s under its 180-second cap. Pytest reported 3.11s; that time is nested within the whole process. Lint also passed. All 80 registered sources and their original snapshot copies remain unchanged.

The prototype API is `finite_joint_reuse.joint_objective(model, prefix, lengths, endpoint_positions, actions, observations, targets, *, total_attempts, total_survivors, total_events)`. It preserves the global reduction `n/b * (e/S * endpoint_loss + prefix_NLL_sum/E)`, with the same three model arms. Tests use fabricated public tokens and fixed targets at horizons 1, 2 and 8, including terminal/padded histories, empty endpoint batches and partial batches. No saved weights or scientific data are loaded by the qualification.

| Checked work for a nonempty endpoint batch | Separate routes | Reuse |
|---|---:|---:|
| Probability-field construction | 5 | 1 |
| Eligible endpoint-prefix filtering | 2 | 1 |
| All-attempt prefix likelihood | 1 | 1 |

Shared tensors stay attached to autograd and live only within a call. The all-attempt prefix likelihood retains its own reset arithmetic, and both continuations retain separate per-horizon readouts and prediction before observation assimilation. These operation counts do not measure speed or complete backward work.

Values and losses are checked at absolute/relative tolerance 1e-10; raw gradients, clipped gradients and three Adam steps at 1e-9. Graph sharing can change floating-point accumulation order. The checks do not imply bitwise equivalence or identical longer training.

The [previous equal-update feasibility stop](finite-update-learning-stop-results.md) remains failed and closed. This qualification does not change its 90-second projection threshold or registered update counts. A fresh integration, independently audited saved outputs and a bounded throughput comparison are still needed before a speed or feasibility claim.

[Frozen protocol](finite-joint-reuse-qualification-protocol.md) · [Summary](finite-joint-reuse-qualification-results/summary.json) · [Manifest](finite-joint-reuse-qualification-results/manifest.json) · [Complete evidence archive](finite-joint-reuse-qualification-results/evidence.tar.gz) · [Publication receipt](finite-joint-reuse-qualification-results/receipt.json)

The archive includes the complete child qualification, all 80 current source files and their snapshot copies, plus the explicitly bound prior-stop evidence and publication. Historical array/checkpoint artifacts are copied as opaque bytes only. The publisher performs zero array decodes, model calls or numerical replays. Installed packages and the interpreter remain external dependencies.
