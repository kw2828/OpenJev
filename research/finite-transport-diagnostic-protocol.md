# Saved-state transport and readout diagnostic

Prospective diagnostic **finite-transport-diagnostic-v1**. The completed
[equal-time comparison](finite-training-allocation-results.md) improves means
but fails its advance rule. Its weak prefix fit already has poor H1/H2
predictions. A short-horizon pass against the uniform reference does not
establish accurate short forecasting, and the current result does not isolate
a long-horizon memory defect.

This is a descriptive analysis of existing learned parameters. It trains no
model, creates no cases, evaluates no new trajectories and promotes no
candidate. It tests which learning or representation mechanism is worth
investigating after the equal-time comparison.

## Fixed evidence and complete coverage

Use only the closed **finite-training-allocation-v1** parent registered with
SHA256 `2867fc976559808ceda2b21818bdbf394c343f4871a95e6b24bd789a85b10145`.
Authenticate its successful original qualification, fit and audit closures,
complete payload inventories and source pins before any checkpoint decode.

Include all three arms, all three seeds and initial/boundary/final snapshots:
**27 NPZ checkpoints, 108 float64 parameter arrays**. Every checkpoint must
have exactly transition logits `[4,8,8]`, emission logits `[4,8]`, hazard
logits `[4,8]` and cost logits `[4,8]`. Reject malformed/nonfinite arrays.
Keep the original 36 endpoint metric rows and nine prefix-likelihood rows
beside the geometry. No selected fit or post-hoc fitted cutoff is allowed.

Reconstruct the declared learned probabilities in NumPy: column softmax for
T and O, sigmoid for h, and `C = .25 - softmax_decision(cost_logits)`.
No true-world matrix, hidden state, teacher label, optimizer or model object
enters this analysis. Parent files remain byte-identical.

## Quantities fixed before decoding

Let Q be a deterministic orthonormal Helmert basis of the seven-dimensional
zero-sum subspace of the eight-state mass vector.

For each action's column-stochastic transition T, report:

- Dobrushin contraction coefficient, one half the largest L1 distance between
  any two columns. This describes T, not the full conditioned filter.
- All eight column entropies and their minimum, mean and maximum.
- All singular values of `Q.T @ T @ Q`. Count values above each fixed absolute
  threshold **1e-8, 1e-6, 1e-4, 1e-2** as a sensitivity table, not a success rule.
- Maximum row-sum deviation from one and `L1(T @ uniform - uniform)`. These
  measure a property of learned transport, not error against true dynamics.

For O and C, report every pairwise Euclidean column distance and their summary,
and the singular spectrum after multiplication by Q. Both have rank at most
three on this subspace by construction. Their other zero modes are structural;
they must not be described as learned collapse.

For each action define the unnormalized surviving-state operator
`A_a = diag(1 - h_a) @ T_a`. Starting separately from `G0 = I8` and
`G0 = C.T @ C`, recurse in the full eight-state space:

`G_(k+1) = (1/4) * sum_a A_a.T @ G_k @ A_a`.

Measure `Q.T @ G_k @ Q` only at horizons **0,1,2,4,8**. Report its trace and
all raw eigenvalues, plus counts above **1e-12, 1e-10, 1e-8, 1e-6**. Preserve
small signed roundoff values; reject eigenvalues below -1e-12 rather than
silently projecting the matrix onto a different positive cone.

The I8 start measures surviving-state sensitivity; the cost start measures
sensitivity of model-predicted costs. Each is an exact uniform average over
action sequences of that length. Projecting between recurrence steps is
forbidden: hazard-related mass changes couple contrast and total mass. The
recursion avoids enumerating a large branch bank.

## Qualification and execution

Before checkpoint decoding, pin the complete 52-file source closure, runtime,
original evidence inventory, protocol and tests. Qualification uses fabricated
arrays only. It checks uniform and identity/permutation transport, known
spectra and entropy, explicit short-sequence enumeration versus Gram recursion,
simultaneous state-permutation invariance, orthonormal basis invariance,
head-only permutation effects, structural rank limits and invalid inputs.

Both qualification and diagnosis use original native supervision with
**90-second caps**, single-thread arithmetic, 1 GiB worker RSS and 32 MiB phase
output. Outputs and registrations are exclusive. Keep failed attempts and
their source snapshots; no silent rerun, replacement snapshot or extended cap.
Diagnosis requires the original successful qualification on identical sources
and evidence. Qualification executes only the selected fabricated diagnostic
tests, not the parent's training tests. Exactly 27 scientific checkpoint
archives are decoded in diagnosis. No model, optimizer or generator is called.

Read results for interpretation only after the original diagnostic process
closes successfully. Publish all quantities and original closures, whether or
not the geometry suggests a useful next step. Presentation uses saved JSON;
it does not decode the checkpoints again.

## What the result can establish

These metrics describe the learned system's geometry. They are invariant to
simultaneous latent-state permutations and an orthonormal change of contrast
basis, not arbitrary latent reparameterizations. A low rank or small spectrum
does not by itself establish lost task information: the true cost may not
require all seven directions. A large spectrum does not establish correctness.
Uniform-action weighting also differs from the empirical evaluation cases.

Associations with existing error rows are exploratory. They cannot establish
causal optimization failure, true-state recovery, mutual information or a
biological advantage. The earlier retentive-initialization failure remains
relevant negative evidence. A doubly stochastic transition constraint is only
a possible later hypothesis; it allows uniform mixing and supplies additional
correct structure from this particular world. This diagnostic does not admit
that training experiment or reopen the closed H4-supervision follow-up.
