# Independent tracking control-audit review

Status: clear for the planned retained engineering loop. This is a source review and focused synthetic/native engineering test result, not task qualification or evidence that online identification improves control.

Reviewed source: `src/openjev/research/reacher_tracking_audit.py`, SHA-256 `a17911455347a05883d789aaead1731775cc1840bef1df7a04cbcedea7b2dee6`.

Reviewed tests: `tests/test_reacher_tracking_audit.py`, SHA-256 `70bd461be94542af8fa675aa3d4d612171074ac896300aaf3347164c137f37a1`.

## Findings and resolution

The first pass found incomplete authentication of identifier startup work, setup copies, retained residual storage and the frozen identifier's final observer state. The author added narrow checks and corruption tests. I reviewed the changes and found no remaining material blocker in the requested numerical, information-boundary or accounting scope. I changed neither source nor tests.

- **Artifact integrity:** the row must have exactly four fixed payloads plus three files per decision, with an additional completed marker. Every payload is hashed before and after replay. Unexpected failure files or other payloads reject the row. External innovation NPZ/JSON files are bound to the writer's manifest and caller-supplied paths, checked on both sides of replay, and independently checked for exact array members and dtype/shape identities. The caller still authenticates the completion hash, source/runtime, initial state/reset allocation and target/gain/noise schedules against the enclosing run manifest. This module does not establish that broader provenance by itself.
- **Native event timing:** continuous replay restores the recorded initial integration state, applies the current decision's gain, executes the issued command plus explicit clipped actuator noise, and checks native state and reward before any next-target event. Target installation and the following public packet are checked separately. Applied-action squared cost is charged once, in normalized action units, without multiplying it by actuator gain. All transition and terminal flags must agree.
- **Planner reconstruction:** all four paid CEM banks are independently rebuilt from the full-horizon innovations, truncated only at the declared terminal boundary. Checks preserve the seven anchors, initial scales, stable eight-elite selection, float32 scored chunks, minimum standard deviation, paid final mean, clipping, and earliest global argmax. The selected action is replayed from the original root and checked against its candidate's first transition.
- **Exact selection versus tolerant arithmetic:** recorded raw rewards must reproduce recorded scores exactly by sequential float32 addition of per-step rewards clipped to `[-2.5, 0]`. Those exact recorded scores determine every elite and selected ID. Independent NumPy geometry permits `rtol=2e-6, atol=5e-7`; that tolerance cannot change the saved-score selection path. Replayed candidate angles and proposal arrays are exact. Native state and identifier prediction comparisons use absolute tolerance `1e-10` with zero relative tolerance.
- **Public identification and gain freeze:** public roots independently reconstruct wrapped angles and causal backward-difference velocity, including zero initial velocity. Every identification transition starts from the preceding public root, consumes the actual issued command, and compares to the subsequently received packet. Its rolling residual sum and earliest minimum/exactly-flat retention rule are independently checked. The frozen arm performs only its declared first `freeze_after` updates; its gain and private identifier observer remain at that cutoff while the common public observer continues.
- **Privilege:** only `public_gain` and `true_state` may use the actual current gain. Only `true_state` may use native arm position/velocity. All other planning roots and all gain-identification roots must equal the reconstructed public estimates. Current target precision is shared across roles. Nominal planning reconstructs solver state from explicit qpos/qvel; it is not a complete future-trajectory oracle.
- **Paid work:** the audit checks all candidate and selected native transitions, two native substeps per transition, forward/reset/postconstraint work, geometry calls/samples, per-gain lifetime totals, eager model/data copies, initial zero work, retained identifier residual scalars and the identifier cutoff state. Nested setup/update/planner times must fit inside their parent measured intervals. Total row time includes caller bookkeeping and storage according to the writer's declared scope; nested times must not be summed again.

## Independent validation

Run from the OpenJev checkout after the author's stable handoff:

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_tracking_audit.py
38 passed in 1.31s

.venv/bin/ruff check src/openjev/research/reacher_tracking_audit.py tests/test_reacher_tracking_audit.py
All checks passed!
```

The focused suite uses engineering seed 410 and tiny horizons, including all six actual writer roles, complete candidate/selected replay, target and gain events, public derivative alignment, cached-gain freeze, setup/state corruption, exact membership and external-input tampering. This review did not run the proposed six-arm 200-decision loop or consume scientific data, fitted checkpoints or scored streams. The auditor itself imports no controller/search/identifier or Torch implementation and performs no action selection, model inference or RNG draw; its native replay is explicitly paid audit work.

## Integration limits

`native_control.mean_cost` currently contains **total episode cost**, `-sum(rewards)`, and intentionally equals the writer's `native_cost`. The enclosing screen must divide by the declared interval length when reporting per-action costs or comparing unequal horizons. Windowed post-switch metrics must use the returned full reward list, not this total.

The enclosing engineering driver must authenticate its explicit initial state, gain/target/noise schedules, shared innovations, complete six-role membership, source/runtime and process terminal states. A completed row audit proves consistency of the saved evidence within that boundary. It does not certify scientific freshness, gain identifiability, useful adaptation, calibrated uncertainty or matched wall time. Full-size runtime/storage and cap behavior still need the separately authorized retained engineering loop.
