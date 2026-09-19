# Tracking policy: independent engineering review

No material blocker was found for the bounded engineering run. This review covers controller information boundaries and orchestration, not task qualification or controller effectiveness.

Reviewed source:

- `src/openjev/research/reacher_tracking_policy.py`: `835a764012be2a0a8e6841129e4474b82760885005bad4cad248e74911f8d08a`.
- `tests/test_reacher_tracking_policy.py`: `01ef3e52bfc34dba26a53ea5839d3b0b9933bb2312ef734ade93686efcbedf2f`.

## Findings

The nominal, adaptive, frozen and zero roles reject both oracle arguments before planning. `public_gain` requires the current gain but uses the same public angle/velocity observer. `true_state` alone accepts native arm position and velocity, requires the current public target and zero target velocity, and quantizes its target to the identical public float32 value before planning. The caller must still supply only the current true gain and authenticate the nominal model; the component cannot verify that provenance itself.

One decision creates one pending selected action. A second decision is rejected until the caller acknowledges that actual issued command and provides the next visible packet. A mismatched acknowledgement, repeated observation or missing packet is rejected before state updates. Equality is checked after float32 quantization, matching the wrapper's actual issued-command representation. The caller must acknowledge the issued command, not the hidden noisy applied action.

Planning does not update the common observer or the identifier. During observation, the common public observer advances once. Adaptive identification uses its own previous observer root and the completed command/packet pair, then requires exact agreement with the updated common observer. Its duplicate observation work is explicit and paid. The frozen role updates on transition indices 0 through `freeze_after-1`, then retains that inferred gain while its common public observer continues to advance. Public target changes do not reset the observer, gain estimate or residual history.

The last native observation is still processed after the final selected action; subsequent decisions are rejected. Exceptions after a planning or observation operation begins mark the policy terminal. If identification fails after the common observer updates, that partial state remains visible in the failed snapshot and cannot be used for another decision. The outer driver should preserve its attempted command/packet and original error.

Per-method timing is nested accounting: decision time includes bank planning and trace preparation; observation time includes identification and the duplicate observer. These values overlap the bank/identifier timers and must not be added to them. They exclude initial role/input validation and root preparation. An outer row timer must therefore charge the complete calls, construction, final snapshots, native execution and storage. The same outer deadline must be checked before and after operations: `observe` has no deadline parameter, and the zero role does not consult the planner deadline.

## Executed checks

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_tracking_policy.py
14 passed in 0.57s

.venv/bin/python -m ruff check src/openjev/research/reacher_tracking_policy.py tests/test_reacher_tracking_policy.py
All checks passed!
```

The tests substitute a fake planning bank, use handcrafted public packets and an independently constructed nominal model, and exercise the real small three-gain identifier where required. They verify role rejection, target precision, causal pending-action sequencing, freeze counts, common-observer agreement, isolation and terminal planning failure. No scientific cohort, fitted model, training, new scientific seed, control benchmark or efficacy result was used. This memo does not replace the upcoming native closed-loop artifact audit.
