# RockSample runtime works; legacy memory crosses episode boundaries

21 September 2026. The single supervised engineering qualification completed
with actual exit **0**, **19 recorded checks** and **34 individual environment
steps**: 18 checks passed and one reproduced a known wrapper defect. Parent
elapsed time was **7.287974833 seconds**, including child cleanup; worker time
was **6.819516292 seconds**. Peak worker RSS was **895,303,680 bytes**. The
600-second, 8-GiB, 128-MiB and 2,000-step limits were respected.

This establishes compatibility and specific interface behavior, not policy
performance, learning efficacy, sensor calibration or a recurrent-architecture
advantage. There were no model calls or training updates. Five constructors
used the same map seed and produced one map; injected states were engineering
fixtures, not policy-generated trajectories.

## Observed behavior

| Check | Recorded result and consequence |
| --- | --- |
| Ordinary interface | 33 observation channels and 16 actions for RockSample(11,11). Reset, movement and sampling carry no check readings. A distance-zero check gives the correct signed reading, which disappears on the next move. |
| Sampling | Good, bad, depleted and empty-square fixtures returned raw rewards `+10`, `-10`, `-10` and `0`. Sampling depleted the matching rock. |
| Natural termination | Raw `step_env` returned the east-boundary terminal observation. Public Gymnax `step` returned the exact keyed **reset** observation/state while preserving `done=True` and the exit reward. The constructor map remained unchanged. |
| Timeout | TimeLimit returned a fresh reset observation. When natural exit and the horizon coincided, natural termination took precedence: `time_limit_reached=True`, `truncated=False`, with the exit reward retained. |
| Previous action | The 49-channel concatenated interface retained the terminating action one-hot after natural reset. Timeout reset cleared all 16 action channels. A reset cannot be detected by assuming every `done` observation has a zero action suffix. |
| Legacy memory defect | `PerfectMemoryWrapper` retained all planted signed readings across natural reset, attaching old-episode evidence to the new episode. Explicit reset and timeout reset cleared the readings. This is now reproduced, not merely a source suspicion. |
| Factory information | `get_env(perfect_memory=True)` exposed true current rock qualities. `get_transformer_env(perfect_memory=True)` used latest-reading memory, including a hidden-map lookup on sampling. The same flag does not provide equivalent information. |
| Vectorization and rewards | Two lanes agreed with scalar and lane-permuted references. All four actual factory routes worked. Factory return rewards were normalized; `info.reward` retained raw task rewards. |

The legacy memory wrapper is **not qualified as a clean public-history
baseline**. The fully observable wrapper is a privileged-quality reference.
An ordinary baseline in this runtime must receive only the declared observation,
previous action and episode boundary, with no simulator state or map lookup.
It must reset its own recurrent/filter state at every boundary before consuming
the returned next-episode observation. This qualification did not evaluate the
new approximate public filter or any learned policy.

## Compatibility correction preserved

The original [setup-01 import](../output/rocksample-runtime-v1/setup-01/import-01.json)
succeeded, but its resolver selected Gymnax **1.0.0**. The subsequent
[source compatibility check](../output/rocksample-runtime-v1/setup-01/compatibility.json)
rejected that runtime before environment construction: its six-result step API
and required `state.time` conflicted with the pinned POBAX five-result contract
and RockSample state. It was not an environment-performance failure.

[Setup-02](../output/rocksample-runtime-v1/setup-02/plan.json) created a separate
runtime with Gymnax **0.0.9**, constraining the other resolved packages to their
previous versions. Runtime-01 and its setup records remain preserved. The
qualified run used Python **3.12.13**, JAX/JAXlib **0.6.2** and CPU only; the
[import receipt](../output/rocksample-runtime-v1/qualification-01/imports.json)
records the full versions and hashes of the imported sources. Upstream POBAX
remained at `a5e1d62d14e4efe783885b9d4f19cffa2a568eec`, unchanged.

## Evidence and reproduction

- [Prospective qualification plan](../output/rocksample-runtime-v1/qualification-plan-01.json)
  and [qualification source](../scripts/qualify_rocksample_runtime.py).
- [Complete check witnesses](../output/rocksample-runtime-v1/qualification-01/checks.jsonl)
  and [summary](../output/rocksample-runtime-v1/qualification-01/summary.json).
- [Actual supervised command](../output/rocksample-runtime-v1/qualification-process-01.launch.json)
  and [successful terminal receipt](../output/rocksample-runtime-v1/qualification-process-01.terminal.json),
  including reaping and absent process group.
- [Completion manifest](../output/rocksample-runtime-v1/qualification-01/completed.json):
  SHA-256 `b233dbd09714ae574626cdd2f58b3efde80c0127759af8f399147845ece4bcb2`.
- [Locked dependencies](rocksample-runtime-requirements.lock.txt) and the
  unchanged [earlier source review](rocksample-source-opportunity-review.md).

The recorded command uses `tmp/pobax-runtime-02/bin/python` and the native-clock
supervisor. Reproduction requires the pinned checkout on `PYTHONPATH`, that
locked runtime, and fresh exclusive output/supervision paths; the completed
directory must not be reused. No reproduction was run for this note.

A separate saved-only review verified all four completion payload hashes and
exact membership, five planned source hashes, 22 imported upstream source
hashes, the installed Gymnax source identity, parent/worker clock and command
joins, the 19-check/34-step totals, and the recorded boundary/defect witnesses.
It found no inconsistency. This review did not rerun transitions or independently
reproduce every internal assertion. The bounded qualification opens an
engineering route to further public-input checks; it does not admit training
or reopen the failed dialogue pooling recipe.
