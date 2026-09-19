# Tracking rollout and engineering driver review

No remaining material blocker was found before the bounded engineering run. The driver was inspected without executing it. Tests used fake policies, fake episodes and explicit synthetic file bytes, with **zero native/model/optimizer/RNG calls** and no scientific artifacts.

Reviewed final bytes:

- `src/openjev/research/reacher_tracking_rollout.py`: `33432dd1b88fa7eee5ed8d86c7c26832eb3d01558391c27eba7d41d74ce30742`.
- `tests/test_reacher_tracking_rollout.py`: `3926ffdd5a0b9f9022867685928616c4f547a2dd7cb812260344fc8ef6aa5544`.
- `output/persistent-dynamics-qualification-v1/run_control_engineering.py`: `aeb94b726292103a97d62bdd1c5c06d7f945380a1b644a17623ea182cc534062`.

## Findings fixed before execution

The first test run reproduced four concrete defects: oracle gains could be read from a caller-mutated schedule after the episode copied its original values; `started.json` writing was outside the failure guard; completion writing could cross the deadline without demotion; and a failed failure-receipt write could replace the original `BaseException`. The source owner fixed all four. The final regression suite also covers an observation failure after native action execution, so the preserved native prefix can be longer than the observed-policy prefix.

The gain schedule is now copied and made read-only before episode construction and current-gain lookup. Only `public_gain` and `true_state` receive its current scalar; only `true_state` reads the privileged current decision state. Other roles receive no native state, future target, noise or gain schedule. All roles use the same declared innovation stems. The actual stored native issued command and returned packet are checked against the selected action and native public record before a copied acknowledgement is sent to the policy.

Compact decision files retain all candidate sequences, scores, predicted angles and raw geometry rewards, plus the selected index and separate selected native root advance. Current public/root state, selected gain, innovation identities, planner configuration and callback metadata remain available for independent proposal/native reconstruction. This review does not itself certify that reconstruction; that belongs to the independent auditor.

Outputs are exclusive. Failure retains completed decision/observation files, partial episode/controller state, the active phase/step and the original error. A late completion is retained under a partial filename and no longer appears as `completed.json`. Receipt-preservation errors are attached to the original exception rather than replacing it. A failure before output allocation remains a preflight rejection, not a completed attempt.

## Enclosing driver

The fixed engineering recipe has one reset410 case, six roles, 200 actual actions per role, current-target events, one hidden-gain switch, and shared precomputed innovation/noise arrays. All six rows complete before saved-output audits start. No efficacy gate or model selection is applied. Source snapshots are hash-checked against their pre-run capture, and the existing scientific/pilot source maps are checked before and after the work.

The workload arithmetic is correct:

- Five planned roles, each with `256 * (189*12 + 1+...+11)` candidate native transitions: **2,987,520** total.
- Five planned roles times 200 separately paid selected advances: **1,000**.
- Six roles times 200 executed plant transitions: **1,200**.
- Adaptive 200 plus frozen 40 identifier updates, each with 21 candidates: **5,040**.

Execution and audit have separate cooperative 600-second deadlines. Source snapshot writes are inside the execution phase; final source verification is inside the audit phase. Both phase completions are checked after writing and demoted on a late failure. Initial protected-source preflight and initial source hashing precede the execution timer. This is not a process-preemption guarantee, and the measurement must retain actual completion/failure and outer runtime rather than treating a configured cap as measured latency.

The row's whole wall time includes preparation, calls, copies, closure and payload hashing, while explicitly excluding its final receipt write. The enclosing deadline still checks that write. Internal setup/decision/identifier/planner times overlap and must not be summed into a second total. Single-case call and serialization overhead must be measured from this actual engineering run; prior batched timings are not a substitute.

## Executed synthetic checks

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_tracking_rollout.py
19 passed in 0.61s

.venv/bin/python -m ruff check src/openjev/research/reacher_tracking_rollout.py tests/test_reacher_tracking_rollout.py output/persistent-dynamics-qualification-v1/run_control_engineering.py
All checks passed!
```

The initial 18-test version had 14 passes and the four intended regression failures. The final suite adds observation-failure coverage and passes all 19. It does not execute the engineering driver, native dynamics, a learned model, a real controller or the scientific qualification.
