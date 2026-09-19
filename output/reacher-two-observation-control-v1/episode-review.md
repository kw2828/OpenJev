# Independent episode-wrapper review

**No material blocker found in the reviewed component.** This was static source/test inspection only. The reviewer did not run tests, construct a model/environment, allocate seeds, replay native dynamics or read scientific outcomes.

Reviewed identities:

- `src/openjev/research/reacher_two_observation_episode.py`: `8ca706e61925588438255eaad56b8127b3dc0375b2be3e99cd31c07def2a7098`.
- `tests/test_reacher_two_observation_episode.py`: `696e5a5f5aa543cdec0fe6e0f93d72371525bd90ea047b78fd2dfb7869e02d8d`.

## Actual episode and public-input boundaries

`learned_control(plan, model, panel, inputs_by_step, out, deadline, progress, *, cases)` admits the reviewed actual-class controller, exactly 50 saved innovation stems and explicit `ControlCase` objects. All `.npz/.json` input files must exist before native setup. The wrapper has no protocol seed lookup or independent innovation generation; reset/noise seeds and immutable schedules come from the supplied cases and are recorded in `started.json`. Authenticity and freshness of those inputs remain enclosing-runner obligations.

Every episode starts with an independently copied actual public packet. `_public_record` reads only the native record's `policy` allowlist of packets/commands, checks exact prefix lengths, validates masked float32 public fields and compares the returned packet against the recorded packet. Privileged native arrays may be copied for storage but never enter the controller arguments. Each selected action is checked against CEM selection, passed to the native wrapper, then compared exactly with the command actually recorded as issued. The next decision acknowledges that recorded command. This prevents silently retaining a planner command different from the executed command, including unnoticed native clipping.

The loop performs exactly decisions 0 through 49. After every native step it requires the expected step index and termination only at step 50. Complete success therefore contains 50 commands, 51 observed packets, 50 root/carried state pairs and no terminal-packet assimilation or fifty-first decision. The history controller owns private candidate state; this wrapper does not install candidate-terminal history or expose the native audit record to it.

## Saved evidence and timing

Every successful decision writes the original trace, full scoring arrays/metadata and full controller metadata in `controller-decisions/NNN.json`. Final files retain native episodes, every root/carried state field, original learned-head selected predictions, complete per-step and aggregate work, and the external input-file hashes and loaded identities. The original learned reward is kept separate from the geometry reward used for planning. Before/after load hashes catch an input file changing during its read; the enclosing audit must also compare all saved input identities against the authenticated root manifest and regenerate their declared innovations.

The row output is exclusive. On success its completion manifest hashes all row payload files, excluding only the completion receipt itself. This is an observed-file inventory, not an independent expected-schema validator: the full-study auditor must require the exact declared membership, all 50 traces and no failed/partial extras, then authenticate the external input files. The component makes no claim to perform that complete audit.

Decision timing includes input loads/hashes, controller computation and trace writes. Native timing includes per-case stepping plus record copying/validation. Setup, final aggregation, cleanup, final payload writes and member hashing are included in final row wall time. Nested timings are explicitly labeled and must not be added twice. The final `completed.json` write is explicitly excluded from its own timestamp; the outer execution wall/cap must charge that operation and enforce the final global deadline. No total-FLOPs or isolated-latency claim is made.

## Failure and cleanup

Controller decisions are saved before native stepping. A failure partway through a native batch therefore preserves both the completed model-decision count and each case's actual native step count, rather than pretending they agree. The existing partial serializer preserves ragged case prefixes. Available controller state, root/carried states, inputs, completed work and active phase timing are retained separately. Controller, native, trace-I/O and cleanup failures do not trigger fallback commands or retries.

All constructed environments are closed, including a failed-reset environment. Each close is attempted once; secondary cleanup or preservation errors are attached to the original exception. Independent best-effort partial writes allow other evidence to survive one failed write. A pre-existing output is refused without overwriting it. Missing late input files are rejected before native setup.

The 23 author-reported fake-only tests cover full command/packet alignment with privileged canaries, terminal behavior, manifests, input mutation, ragged native failure, controller failure, partial trace failure, reset/cleanup failure, secondary-error preservation and deadline interruption. The reviewer read but did not rerun them. The tests replace the native environment, controller and search serialization; they establish orchestration behavior, not real model/native integration, complete source-bound audit or scientific effectiveness. A retained whole-pipeline engineering rehearsal and capacity measurement are still required before a prospective scored study can be frozen.
