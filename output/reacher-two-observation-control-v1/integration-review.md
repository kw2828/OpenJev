# Two-observation replay: integration review

**Source review clear: no material blocker found.** This review made no training/model/simulator call, evaluation draw or current scientific outcome read. All 90 source hashes still match frozen geometry-memory plan `23c93e4adfbb45cf224383ffa31d7323df2405807eb3f1c4c75bb27378bcd9ee`. Component preparation is conditional; a future training study requires the current scientific gate to pass and a separately reviewed, frozen protocol.

Reviewed [component](../../src/openjev/research/reacher_two_observation_history.py) SHA256 `67420a414941a16123f9ebdd82b2f073f947e45be73d4039c261335cb0e8d1aa` and [tests](../../tests/test_reacher_two_observation_history.py) SHA256 `79fdc10fc66909832df077d57ef8e91ae86f1166721d03b576af7b46f2dda99c`. The author reports 49 synthetic tests passed and Ruff clean. This reviewer inspected those tests and verified both source hashes, but did not rerun model calls.

## Scope and compatibility

The proposed `TwoObservationHistoryGRUWorldModel` is an **eight-input, parameter-identical raw-history replay comparator**, distinct from the older [12-feature finite-difference proposal](../reacher-cache-ablation-v1/two-observation-design.md). It rebuilds learned state from the penultimate actually visible packet, the latest visible packet, intervening real missing packets and issued commands. It does not receive native velocity, realized noise, future visibility or computed motion labels. All retained missing packets keep their actual validity and age, with angular placeholders sanitized to zero.

Inheriting the residual GRU constructor preserves its parameter names, order, shapes and 36,805 parameters at hidden size 64. The public signatures remain `initial`, `assimilate`, `advance`, so the unchanged [sequence loss](../../src/openjev/research/reacher_world_models.py) can consume the existing eight-field packets, two-field issued commands and total native rewards. Original target masks and reward supervision remain unchanged. The five-step training rollout horizon is not enlarged by this component.

The frozen cache trainer and geometry-memory controller use exact-class admission. They cannot accept the new class by changing a configuration flag. A future additive trainer/control adapter must bind its actual class, complete configuration, paired initialization, optimizer state, data/order hashes and resume cursor. An identical tensor schema does not authorize substituting an old fitted checkpoint. The deployment payload alone is not a resumable training checkpoint.

## State and causal boundary

| State field | Per-case shape and dtype |
|---|---|
| `hidden`, `packet` | `[h]`, `[8]` float32 |
| `real_packets`, `real_actions` | `[12,8]`, `[11,2]` float32 |
| `real_present`, `real_indices` | `[12]` bool, `[12]` int64 |
| `real_index`, `imagined_depth` | `[1]` int64 each |
| `real_target`, `pending_action` | `[2]` float32 each |

This state is `4h + 644` bytes per case, or 900 bytes at h64, excluding tensor objects, graphs, copies and allocator overhead. A future saved-output auditor must reconstruct every root independently from the actual packet/command prefix; the two visible anchors and their indices must be derivable from this evidence.

At each real boundary, discard old hidden/predicted packet values, append only the previous issued command and current real packet, and retain history starting at the older of the two latest valid observations. With only one observation, retain that first anchor. Require startup or exactly one selected-action advance; a multi-step search terminal cannot become real evidence. The caller must additionally verify that the stored pending action was actually issued.

The twelve-slot boundary needs an explicit test: the last missing packet in a ten-gap can fill all slots; append the next valid packet temporarily before shifting the anchor, then retain the eleven intervening commands. Reject a true anchor-to-current span above eleven instead of silently discarding the necessary anchor. The current separated six-/ten-gap schedules fit this contract; arbitrary future schedules need separate validation.

Imagination must preserve all real history and clocks, update only private predicted state/depth/pending action, and leave siblings and roots unchanged. Re-advance the selected command from the untouched root. Terminal observation 50 may be represented, but no advance beyond it is permitted. Existing loss and control loops stop at that boundary.

## Work and gradients

For **A successful real-assimilation samples** and **D ordinary advance samples** (including selected actions and imagined candidates), charge:

| Paid work | Samples |
|---|---:|
| Observation-update GRU calls | `12A` |
| Transition GRU calls | `11A + D` |
| All GRU-cell calls | `23A + D` |
| Linear-layer calls, including both original heads | `44A + 4D` |
| Expected clipped-action-cost calls when residual reward is enabled | `11A + D` |
| Dense affine MAC estimate | `12A*3h(h+8) + (11A+D)*(5h²+25h)` |

Startup-masked calls are still paid. Discarded replay readouts still execute, but they are not additional supervised targets. Gradients must reach the student's observation-update and transition parameters through reconstruction; old hidden-state graphs must not carry earlier neural history across real boundaries. Public input tensors normally require no gradients, but there is no reason to detach reconstructed student activations. Original prediction/reward heads receive their normal loss gradients from scored advances.

These formulas describe completed forward calls, not failure-prefix work, backward/Adam cost or total FLOPs. A future adapter must also report reconstruction time, validation, window shifts, every per-candidate buffer check/copy, trace serialization, retained-state bytes and peak memory separately. Equal parameters or mathematical transition counts do not equalize these costs. Do not attribute an instrumentation or copying penalty to an information-theoretic advantage.

## Remaining integration checks

The component tests cover constructor/tensor parity, deployment restore, causal replay/action ordering, the full-window reacquisition boundary, discarded-state poisoning, candidate isolation, bounded gradients, hook-based call counts and unchanged fifty-step loss/terminal enumeration. None of these checks establishes effectiveness.

Future integration still needs a new exact-class trainer with paired initialization/order and audited optimizer resume; a control adapter with complete reconstruction and candidate-copy costs; saved-state audits against native-derived public prefixes; retained failure-prefix accounting; and a complete runner/auditor rehearsal before choosing a capacity cap. Register the isolated constructor/test seed410 in that future engineering ledger. This comparator tests sufficiency of short explicit public history, not a particular velocity algorithm, biological wiring or architectural novelty.
