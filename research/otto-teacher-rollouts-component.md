# Paired teacher continuations

The [sampler](../src/openjev/research/otto_teacher_rollouts.py) passes **54 fabricated tests** and Ruff. It starts each candidate action from the same public belief and estimates the capped search cost of following that action with the fixed analytic teacher. A subsequent [native comparison](otto-teacher-sampler-qualification-results.md) passed all 461 fixed cases, including 408 simulator steps. No training labels, fitted models or improved task results are established by these checks.

For each declared replicate, the sampler draws one hidden source from the supplied public posterior. All eligible first actions share that source and the same sequence of observation uniforms. Each action uses its own position-dependent probabilities, so shared uniforms can produce different observed hits. A separate source channel prevents observation draws from changing the sampled source. Seed, anchor and replicate identities are fixed-width uint32 values in an explicit PCG64 namespace.

Each action gets a fresh [public-only teacher snapshot](otto-teacher-snapshot-component.md). Its forced first move counts toward the horizon; later moves use the unchanged analytic teacher. Source discovery terminates immediately without an odor draw. Every public observation is assimilated, including success at the horizon and the last observation of an unsuccessful capped rollout. Cycling does not shorten the cap.

The caller declares every eligible first action and every replicate before sampling. Processing order is canonical. The returned panel is complete or the function raises; a failure cannot become an unsuccessful capped label. Detached events retain attempted operations and already completed records when a later operation fails. One validation snapshot and every per-action snapshot are counted explicitly. Durable evidence and resource enforcement remain the caller's responsibility.

## Verification

The [tests](../tests/test_otto_teacher_rollouts.py) cover categorical boundaries and zero mass, independently reconstructed streams, signed kernel coordinates, action-dependent observations, source privacy, unchanged global random generators, canonical ordering, found and capped trajectories, final updates, exact operation accounting and interrupted execution. All cases use tiny fabricated states. These checks do not establish useful action targets or full native transition parity.

- Original tool: `bf7ec5`, exit 0; 54 tests passed in 0.14 seconds.
- Captured test-process time: 0.373 seconds; Ruff: 0.080 seconds.
- [Receipt](../output/otto-teacher-rollouts-engineering-v1/attempt-01/receipt.json): `ea9c185ef82370fc0d682f618b7f0ea20a89315a84fd86c2a8dac84d0ce9ed2d`.
- Sampler SHA256: `0fb4f0e4e45e0bed0c40c90c3bff5b19f58b888a838b1f155ab4ff6f0a777612`.
- Test SHA256: `10cd8d4576bc6b62f24ce4b2a2c2cfc7852d89f65cbef7af0e26849887dbf470`.

All 219 frozen spatial-study source hashes remained unchanged before and after the run. There were zero native-simulator calls, learned-model calls, empirical-record reads, training calls or training-label collection calls. Analytic choices on synthetic states are included in the test runtime.

The separately frozen comparison with the original observation generator passed its fixed checklist. The [learning comparison](otto-action-cost-followup.md) remains to be specified and evaluated separately.
