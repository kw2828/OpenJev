# Independent tracking-wrapper review

19 September 2026. **Clear after one narrow failure-accounting correction.** This is an engineering source review and focused test rerun, not a task qualification, controller comparison or scientific result. No training, learned-model call, production corpus/checkpoint read or scientific stream was used. Native calls were restricted to the new tests' explicit seed410 fixtures.

Reviewed final files:

- [`reacher_tracking_dynamics.py`](../../src/openjev/research/reacher_tracking_dynamics.py), SHA256 `d1bd6780f4e8d6b9601de0738e5cd3f7bc7088b98f95e3857514d5c80c5dfba2`.
- [`test_reacher_tracking_dynamics.py`](../../tests/test_reacher_tracking_dynamics.py), SHA256 `550fe18c06254befd09b3c962d20d4af71921fd2561887673e78b8eb03af35fd`.

The review compared the implementation with the [prospective tracking design](design-review.md), [public-comparator boundary](classical-comparator-review.md), unchanged public-packet helper and installed native Reacher step/reward/reset code and XML.

## Checked behavior

**Public information.** Reset and step return only float32 eight-channel packets: angle cosine/sine when visible, current public target, validity and elapsed measurement age. Missing angles are zero. Hidden gain, disturbance realizations, native velocity, rewards and future events are confined to explicitly privileged audit methods. Different hidden gains/noise produce identical missing-observation packets when current public information matches. Audit methods are not a security sandbox; the eventual runner must never pass their results or the episode object to public controllers.

**Goal and reward timing.** Action `t` advances physics with target `t` and gain `g_t`. Native reward and `native_after` are captured before any target event. A changed target is then installed, target velocity is zeroed, and packet `t+1` is emitted. Arm position, velocity and physical time are preserved across the goal event. Saved decision states and transition states remain distinct, allowing an auditor to avoid grading an action against an unseen goal. Targets are checked against the native target-joint box; reachable-target sampling is an enclosing protocol obligation.

**Gain, noise and control cost.** Gain changes motor gear to `200*g_t`; damping remains fixed. Commands are clipped then quantized to float32, explicit supplied noise is added in float64, and the normalized applied control is clipped once more. Native reward charges that applied control squared once, rather than the gain-scaled torque. The tests compare every transition in a manual native schedule, including a gain change, target event, noise and clipping. Gain also scales disturbance torque physically; this is a task property, not a removed confound.

**Ownership and time limits.** Input arrays are copied and made read-only; public packets and audit records are independent copies. The nominal-model factory creates a fresh gear200 model without copying the live plant. Explicit horizons override the native50 limit; a51-action test verifies nontermination at50 and strict termination at51. Single-use episodes prevent a reset from silently replacing partial evidence. The source supports the proposed longer horizon, but this review did not run a200-action task qualification.

**Failure records.** Invalid pre-attempt commands leave the successful prefix unchanged. Once an attempt starts, failures are terminal; attempted commands, native returns, completed public decisions and available native state remain distinguishable. Reset, interrupted step, target-event and reward-metadata failures preserve the original exception. No failed episode can resume through the public step/reset API.

The initial implementation combined `native_returned=True` with reward-field conversions in one `dict.update` call. A missing native information field could raise before the update, falsely excluding a returned native step from the count. The author corrected this by setting the return flag immediately, saving native state, and only then parsing reward metadata. A new test performs a real engineering step, removes one returned information field, and verifies one native return, zero completed public decisions, retained state and the original `KeyError`.

## Independent checks run

From the OpenJev checkout:

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_tracking_dynamics.py
37 passed in 0.25s

.venv/bin/ruff check src/openjev/research/reacher_tracking_dynamics.py tests/test_reacher_tracking_dynamics.py
All checks passed
```

The first collection attempt used `.venv/bin/python` and stopped before any test because that environment lacks `mujoco`. The established `.venv-robotics` interpreter was then supplied by the author and used successfully. No package installation or dependency change was made. Final source/test hashes match the reviewed correction. The reviewer wrote only this note.

## Integration limits

At a planning boundary, the live native model and latest transition record can still contain the *previous action's* gain: `g_t` is installed immediately before the next native step. A future privileged current-parameter reference must receive the explicit current schedule value, stripped of all future values. It must not copy the live plant or read the previous transition gain and mistake it for the current gain. A public comparator must receive neither.

Future schedule generation, fresh RNG allocation, gain-identifiability tests, meaningful post-change cost, controller timing, independent full-record replay and scientific continuation criteria remain outside this wrapper. The tests neither establish adaptation efficacy nor authorize a neural architecture claim.
