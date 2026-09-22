# Teacher continuation: public snapshot initializer

The [snapshot initializer](../src/openjev/research/otto_teacher_snapshot.py) passes **31 fabricated-state tests** and Ruff. It prepares the arbitrary-anchor initialization required by the [conditional action-cost experiment](otto-action-cost-followup.md). It does not generate continuations, train a model or establish better task performance.

`TeacherSnapshot(public, belief, kernel)` constructs the existing in-bounds analytic teacher at an already-assimilated public state. It preserves the supplied position, absolute public step and float64 belief entries. Initialization neither resets to the center prior nor applies the last observation again. The caller must establish the input posterior's public provenance.

The belief must be finite, nonnegative and normalized within the existing categorical tolerance of `1e-10`, with exactly zero mass at the current nonterminal cell. Accepted roundoff is copied exactly, without renormalization. Zero and subfloor beliefs, terminal anchors and malformed public packets are rejected. The teacher owns its copied posterior and immutable observation kernel; the interface accepts no simulator, hidden source, random stream or learned model.

The component inherits the existing analytic choice and public-update logic, with `allow_stay=False`. A prescribed first update is allowed before any choice is pending; the future sampler must separately validate its declared first action. The caller owns the continuation horizon and must assimilate its final packet. Inherited `reset` starts a new center-prior episode rather than restoring the anchor. Construct a fresh snapshot to restart an arbitrary continuation.

The [tests](../tests/test_otto_teacher_snapshot.py) check exact copying without re-assimilation, arbitrary steps, named public packets, defensive ownership, boundary choices, a hand-calculated posterior after a forced action, pending-action errors, terminal handling, final nonterminal updates, reset semantics and invalid inputs. They use fabricated kernels and beliefs. These checks do not establish native observation-generation parity, correct random pairing, or rollout cost accounting; those remain requirements for the full sampler.

The first captured attempt passed all 31 tests but failed Ruff's import-order check. Only the import order changed before the second attempt, which passed both checks. Both attempts remain available:

- [First receipt](../output/otto-teacher-snapshot-engineering-v1/attempt-01/receipt.json), original tool chunk `c2d3da`, exit 1. Receipt SHA-256: `79f0cce949444108a8dc1bf9d9446dd66dcde0584a1f0f9573b622afab28693a`.
- [Passing receipt](../output/otto-teacher-snapshot-engineering-v1/attempt-02/receipt.json), original tool chunk `0148c8`, exit 0. Receipt SHA-256: `75b9b55b9a72a617161414219f5825838462f5c62a3fee3eeec24ba0e86adfc4`. [Test log](../output/otto-teacher-snapshot-engineering-v1/attempt-02/pytest.log) and [code-check log](../output/otto-teacher-snapshot-engineering-v1/attempt-02/ruff.log).

All 219 frozen spatial-study source hashes remained unchanged before and after both attempts. No empirical inputs, native simulator calls, learned-model calls, training or label collection were used. Synthetic analytic choices are included in the test runtime. The original spatial-control audit remains a separate process; these component tests do not replace it or admit another empirical study.
