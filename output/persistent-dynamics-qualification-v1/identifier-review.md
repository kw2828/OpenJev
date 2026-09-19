# Public tracking identifier: independent engineering review

Reviewed the additive source and tests without production data or scientific outcomes. No material blocker was found for the proposed once-only pulse diagnostic.

Source bindings:

- `src/openjev/research/reacher_tracking_identification.py`: `0798bb0918b5c7147eb9930c5bce1f66b77897ebdbdac58c7573f5d72f7fb971`.
- `tests/test_reacher_tracking_identification.py`: `97b2aa5ddb075a46c4104478342fbf3fd6b35de31c0586c235d9a05927dcc210`.

## Causal and numerical boundaries

Each gain candidate starts from the same **previous** public observer estimate and receives the issued command for that completed transition. The returned packet is first validated using a private observer copy. That copy cannot affect the simulator root; it becomes current only after the prediction bank and residual calculation finish. No actual actuator noise, native velocity, gain schedule or future target enters this interface.

The observer uses zero startup velocity, then one backward difference, then the three-point backward derivative. It accumulates wrapped angle increments rather than resetting to a principal angle on every packet. Target changes update the public target coordinates while preserving arm position, velocity, residual history and gain estimate. Initial winding and displacements larger than pi remain ambiguous, as documented.

Every completed update pays for every gain's private model/data root reset, forward call, native transition and post-constraint refresh. The residual is mean squared error across four cosine/sine coordinates; the bank objective sums only the latest declared number of completed transition residuals. An exactly flat whole bank retains the previous estimate; otherwise the first minimum wins. Residual spread is correctly not called confidence. Inputs, model copies, estimates and returned traces do not alias the caller's arrays.

Native failure makes the identifier terminal. Requested and completed work are distinguished. A partially interrupted `_step` may have executed some native substeps without returning; completed counters are therefore completed helper-call accounting, not an independently measured partial native-work count. `last_trace` is the last successful update. An enclosing diagnostic must retain its attempted command/packet and exception when recording a failure.

Regular sampling and correct issued-command alignment are caller obligations: visible packets have age zero and contain no absolute timestamp that could prove there was exactly one elapsed transition. The nominal-model factory/provenance also belongs to the caller. This component rejects altered actuator gears but does not authenticate the entire supplied physics model against an external XML hash.

## Executed engineering checks

From the OpenJev root:

```text
PYTHONPATH=src:scripts .venv-robotics/bin/python -m pytest -q tests/test_reacher_tracking_identification.py
27 passed in 0.21s

.venv/bin/python -m ruff check src/openjev/research/reacher_tracking_identification.py tests/test_reacher_tracking_identification.py
All checks passed!
```

The focused tests use native engineering resets with seed410, public synthetic packets and explicit commands. They cover independent candidate-step equality, derivative/wrapping behavior, target events, copies, finite-window arithmetic, flat-bank prior retention, rejection and terminal failure. No fitting, production checkpoint, scored cohort, scientific RNG allocation or policy/controller evaluation was performed by this review.

## Proposed pulse diagnostic interpretation

The fixed six-regime, 80-transition, zero-noise pulse diagnostic is a useful engineering test of whether this public-state approximation can identify the supplied gain under known excitation. The predeclared last-ten-transition gain-MAE threshold of 0.1 and the zero-command flat-bank negative controls should remain fixed. Failure should be retained without tuning this attempt.

Passing would establish this limited identification behavior only. It would not establish control relevance, robustness to observation gaps or actuator noise, optimal state estimation, or a learned adaptation advantage. The later tracking qualification still needs a common-public-observer true-gain comparator and separate privileged-state reference, paid identification work, and post-change control costs.
