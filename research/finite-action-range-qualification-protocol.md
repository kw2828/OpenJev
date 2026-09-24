# Qualify the action-error contrast loss before empirical learning

This engineering step implements the [prospective loss-control hypothesis](finite-decision-error-next.md).
It does not train on scientific cases, reuse evaluation cases or change the
parent's 14/15 failure. Its only inputs are fabricated numerical witnesses.

The two unchanged architectures use the task-independent random head and exact
352-parameter model type. Each has three externally identified loss arms:
ordinary blind-cost MSE, twice blind-cost MSE, and the mean squared action-error
range divided by four. The range is computed per case and time step before
averaging. Use PyTorch amax/amin, whose subgradients share exact ties. The
ordinary arm returns the qualified original loss tensor unchanged.

The wrapper calls the qualified shared forward objective once. For nonempty
endpoints, add only the alternative-minus-original blind MSE, weighted by the
existing eligible/total-survivors and total-attempts/batch factors. Preserve all
observed-cost, event, survival and prefix terms. Empty endpoint batches return
the original loss without a mean of an empty tensor. Extra loss arithmetic has
separate counters; model forward counts keep their original meaning.

Qualification must cover hand-derived centered four-action inequalities and
regret bounds, gradients away from ties, exact-tie/permutation behavior,
ordinary-arm value/gradient/optimizer parity for both transports, and full,
partial and zero-endpoint normalization. Verify identical initial model/head
and unchanged prefix-stage boundaries across loss variants within each
architecture. Keep the optimizer reset, parameter order, clipping, paired
batch sequence and checkpoint controller intact. Fabricated numerical checks
are engineering evidence, not effectiveness or throughput measurements.

Freeze the six new source/protocol files with the inherited source closure,
the published diagnostic authenticator and next-hypothesis note. Bind all 152
sources and the exact runtime before one native qualification attempt. Use
single numerical threads, no ambient pytest plugins, fixed selected tests,
exclusive outputs and the existing native supervisor with a 120-second failure
cap. Preserve partial outputs on failure and do not edit, retry or relax the
registered attempt. Require its original clean process closure before claiming
qualification. Archive the complete source snapshot and original receipt.

A successful qualification establishes numerical integration only. A separate
protocol, complete cohort runner/auditor, feasibility check and registration
are required before the proposed thirty-fit learning comparison. Neither
architectural novelty nor an empirical improvement follows from this step.
