# Independent proposal-memory auditor review

**Clear for the enclosing bounded engineering run.** No material issue was found in the additive auditor's causal reconstruction, input binding, memory accounting or retained native checks. I changed no source or test file. This review does not authorize scientific qualification or establish a warm-start benefit.

| Reviewed file | SHA-256 |
|---|---|
| `src/openjev/research/reacher_proposal_memory_audit.py` | `5a416b7006805f11c18a0d8f8cce9f38fa6239737b8d99e363d7fe7f83b8373e` |
| `tests/test_reacher_proposal_memory_audit.py` | `e28173f5a0ac94503c77797ddcf1817bde56a44dd2528857ff3305418840a80f` |
| Unchanged inherited `src/openjev/research/reacher_tracking_audit.py` | `a17911455347a05883d789aaead1731775cc1840bef1df7a04cbcedea7b2dee6` |

## Independent source checks

- **Causal centers:** the auditor derives each center from the previous selected sequence, the corresponding actually issued command and the previous/current public targets. The selected sequence comes from the saved bank and exact selected ID, whose first command must equal the native command. It does not import or invoke the proposal-memory implementation. Startup has no invented history. Target-change reset uses only equality of observed current targets. No future gain, change flag or target schedule enters center reconstruction.
- **Alignment and terminal behavior:** `shift_plan` removes one real action, hold-pads only when the retained tail is shorter than the actual current horizon, and takes explicit float64 column means within the current blocks. `repeat_last` uses the acknowledged first command, not a terminal candidate action. Centers retain their full declared block shape with inactive terminal blocks exactly zero. All three modes and observed-target resets are covered.
- **Input identity:** original external innovation files remain hash-bound before and after replay. Independent reconstruction changes only initial slots 7 through 63; the raw anchor slots, unused random-extra bank and all three CEM innovation banks are preserved. Both original and transformed dtype/shape/content identities must match the trace. Only the independently derived inputs enter the inherited exact CEM proposal/selection checker. No global rebinding or bypass of that checker exists.
- **Memory bookkeeping:** prepare traces include the preceding acknowledged commit only after step zero; the final snapshot adds the final commit and second snapshot. Expected retained payload bytes account for shared pending/trace target identity rather than double-counting it, while distinguishing the previous committed target from the next prepared target. The formulas match the producer's 72-byte startup, 176-byte first committed state and 280-byte subsequent shifted preparation at H12/block3. They measure retained NumPy payloads, not Python/JSON overhead or process memory. Copy/hash counts, column means, projected blocks, hold-padding and reset counts are reconstructed independently.
- **Time and row integrity:** setup remains constant; cumulative times are finite and monotonic. Prepare-plus-commit time must fit within the writer's separate proposal-memory interval. Initial setup/snapshot work and final snapshot work are bounded by enclosing row times. Disjoint row components include proposal-memory time once. Exact successful membership adds the two proposal-memory snapshots to the prior row schema; extra or failed payloads reject the row. External inputs and completion bytes are rechecked after replay.
- **Original checks retained:** the additive audit still performs continuous native episode replay, reward-before-target-event checks, explicit current-gain privilege checks, independent public roots, exact saved-score CEM selection, every candidate and selected-root native replay, eager model fingerprints/setup counts, native/geometry work accounting and deadline checks. Admission is restricted to `nominal`, `public_gain` and `true_state`; no identifying role or hidden identifier can enter this matrix. Independent geometry tolerance does not alter exact saved-score selection.

## Reviewer validation

After the author's stable handoff, I ran once from the OpenJev checkout:

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_proposal_memory_audit.py
57 passed in 1.83s

.venv/bin/ruff check src/openjev/research/reacher_proposal_memory_audit.py tests/test_reacher_proposal_memory_audit.py
All checks passed!
```

The suite uses only synthetic inputs and engineering seed 410. It includes all nine tiny real-writer H3/T3 combinations with native replay, exact old/new cold episode and decision-array parity for all three roles, terminal and target-change cases, and corruption of centers, identities, roots, caches, counters, byte totals, timing and membership. Source/test hashes above were independently checked. No nine-row T200 run, scientific input, fitted model or production evaluation was read or executed for this review.

The enclosing driver still owns original case/source/runtime/process authentication, exact nine-cell completeness, full T200 cold parity and finite execution/audit caps. This auditor validates saved consistency within that boundary; it does not certify scientific freshness, control effectiveness, optimality or comparable wall time. The inherited `native_control.mean_cost` remains an episode total despite its name, so windowed per-action reporting must use the saved reward list and declared denominator.
