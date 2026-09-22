# Native query-gate integration qualification

This is an engineering prerequisite for a learned computation gate. It does not
train a model, establish autonomous competence, or test a novel architecture.
The next scientific comparison remains a learned recurrent gate against a
matched stateless gate and fixed neural/analytic endpoints.

## Fixed scope

- Use the unchanged qualified legacy TensorFlow CPU runtime, one numerical
  thread, original 13,390,849-parameter checkpoint and byte-verified tensors.
- Reuse the released-reference setup and forward recorder. Preserve its
  historical setup allocation metadata; report the physical cost of this new
  qualification separately. Do not infer latency per episode from that old
  allocation or make a speed claim from this probe.
- Three fixed modes: always query, never query, alternate queries using a
  carried counter reset at each episode. No training, threshold search, or
  outcome-dependent mode selection.
- Six cases per mode: sensing length 3 with initial hits 1, 2, 3 and seeds
  1110001, 1110002, 1110003; sensing length 4 with hits 1, 2, 3 and seeds
  1120001, 1120002, 1120003. Reuse each case across all three modes.
- End each case at source discovery or after 16 actual steps. Maximum 18 native
  resets, 288 native steps, and 288 TensorFlow policy forwards including the
  separate reference computation on every queried step. Never-query episodes
  must make zero TensorFlow policy forwards.

## Invariants

The gate receives only the declared public feature vector and its own carried
state, before the neural score is computed. Its analytic backend and original
neural backend bind once to the gate's read-only public view. Independent
analytic and restricted neural reference actors compute scores without creating
pending counterfactual actions. All filters then receive the actual selected
action and public observation, including a terminal observation.

Require byte-identical public/native belief arrays after initialization and
every update. Require exact independent analytic score/action agreement on every
decision and original float32 restricted-neural score/action agreement whenever
queried. Preserve the original float32 near-tie selector. Never-query chooses
the analytic endpoint. Query counts, state transitions, reset behavior, feature
blindness, query age and attempted/completed operation accounting must agree
with the declared mode. Backend purity remains a trusted callable contract, not
a process sandbox.

## Execution and evidence

Freeze this protocol, helper/tests, qualifier, inherited scientific sources,
runtime identity, original native inputs, fabricated-test receipt, and the
scoped seed-reservation receipt before numerical imports. That seed check covers
220 OTTO protocols, scripts and plan/seed records; it is not a claim to have
searched every historical step log. The immutable plan records exact hashes.

Use one original suspend-inclusive supervisor with a 180-second cap, 4 GiB RSS
and 64 MiB output. The cap includes setup and all native/model operations. Record
attempts before calls, returned operations, scores, features, recurrent state,
actual observations and per-channel cost. Preserve incomplete attempts and
failure receipts. No replacement seeds, silent retry or cap extension.

PASS requires every declared case to finish its bounded probe, all invariants
to pass, and the original supervisor to close successfully. A PASS admits
integration only. Short capped probes cannot estimate full-horizon success,
the value of learned recurrence, or the utility-versus-compute frontier.
