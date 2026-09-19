# Independent public-only controller review

No material blocker found in the bounded static review. This is component engineering, not a study result, protocol freeze, native integration test or authorization to launch a scored run.

## Reviewed identity and evidence scope

- `src/openjev/research/reacher_two_observation_control.py`: SHA-256 `73247ad5d46733b27a5968128ff5f8e3dcde795f9c0596be83cd583eea2fc77f`.
- `tests/test_reacher_two_observation_control.py`: SHA-256 `88646ffe9e61e4750fbaa4d32131ffc120070d3479e339832225eecb2016af0c`.

The reviewer read both files plus the unchanged CEM, geometry-bank, selected-advance, public-packet and state-copy helpers and the new model's real-history contract. No source edit, test rerun, model call, native step, training or network action occurred. The author's reported 24 passing synthetic tests in 1.74 seconds and clean Ruff are author-supplied validation; this memo is an independent source assessment.

## Public information and exact class

Only the exact `TwoObservationHistoryGRUWorldModel` is admitted. It must use eval mode, residual reward enabled, finite CPU float32 parameters, no registered buffers, matching dt/noise/optional width and the complete model configuration. Nonengineering use requires horizon12/block3 and geometry-only scoring. A complete weight/configuration identity is checked before reconstruction and again before state commit.

The controller accepts only a float32 `[case,8]` public packet, caller-supplied immutable `SearchInputs`, and the previously issued float32 command. The first decision rejects an antecedent command; subsequent decisions require exact equality with both the prior selection and carried pending action. This prevents silently replaying the planner's command when the caller actually issued something different. The adapter does not know applied actuator noise and does not accept native state, velocity or a simulator handle.

The model sanitizes unavailable angular placeholders before checking known fields, including discarding NaN placeholders when validity is zero. It verifies integer real clocks, age, static target, startup and exactly-one-selected-advance phases. The caller still bears responsibility for authenticating that the packet came from the public wrapper and that the acknowledged command was actually issued. The component cannot attest to an external environment by itself.

## Private history and unchanged search

Every real decision reconstructs from the bounded actual public packet/action suffix. Candidate roots use copied/index-selected state, and each advance keeps private imagined hidden/packet state while cloning actual-history buffers unchanged. Root hashes are checked by the unchanged scorer. Predictions never become real evidence.

The adapter calls the existing `search('cem256', ...)` and geometry bank kernel directly, without replacing a global method or frozen registry. It requires four paid banks of64, preserves the final paid mean candidate, truncates at the fifty-action boundary and retains original sequential float32 clipping to [-2.5,0]. Original learned angle/reward heads and the residual action-cost skip still execute; geometry replaces the planning reward rather than adding the expected action cost twice.

The selected command is advanced exactly once from the same real root, through the unchanged selected-prediction kernel, with its model and geometry work recorded separately. The controller commits independent copies only after scoring, snapshots, identity checks and the final cap check. Returned decisions, NumPy snapshots and the original caller plan cannot alias the live history.

The source tests compare complete CEM scores/sequences/selections with the frozen kernel at horizons12,3,1; compare candidate trajectories and sequential scores directly; perturb one candidate without changing its neighbors; exercise full-window reacquisition and actual issued actions; and check a fixed toy geometry policy. The toy clipping case makes all256 scores tie and verifies the existing earliest-global-ID selection. These are meaningful numerical engineering tests, not evidence of task effectiveness.

## Work, timing and failure semantics

Real assimilation pays all twelve observation updates and eleven complete parent transitions, including startup padding. Four learned Linear head calls and the residual analytic action-cost operation for each replayed transition are retained. Successful decision accounting includes those reconstruction calls, every CEM candidate transition and the separate selected advance. Hooks record attempted and completed GRUCell/Linear sample calls; no method or registry patch is used.

Copy/state payload counters are explicitly incomplete memory/work estimates, not peak memory or total FLOPs. Reconstruction, search, selected advance, validation, hooks and snapshots are inside measured controller wall time. Native steps and artifact I/O remain the caller's separately charged responsibility. Equal model parameters or candidate budgets do not match total compute because this history reconstruction adds work.

A failed decision is terminal and does not advance the real controller cursor or commit the candidate state. Completed and partial bank prefixes, original roots, selected outputs when available, phase timers and neural counters are preserved under exclusive failure paths. Existing failure directories are rejected rather than overwritten. Temporary hooks are removed even on exceptions, and errors from evidence preservation are attached without intentionally retrying the decision.

The nonmodule residual-cost skip cannot be completely observed by a module hook. Failure reporting therefore provides conservative lower/upper completed-sample bounds: a returned advance establishes its cost was paid, while a completed reward-head Linear alone does not prove the subsequent analytic operation finished. This avoids claiming an interrupted model method completed. Reconstruction bounds use the next observation-update attempt as evidence that the preceding full transition returned.

## Remaining integration boundary

There is no new environment wrapper, stream selection, checkpoint loader, fit, optimizer or study runner here. Future integration must authenticate the actual-class checkpoint, public wrapper and issued-action sequence; bind sources and RNG roles; retain full arrays/partial attempts; charge native and artifact work; and pass a complete engineering rehearsal and independent saved-output audit under declared caps. The component's strict twelve-packet/eleven-command limit rejects longer unsupported history instead of truncating it. These boundaries and the added reconstruction cost must remain explicit in any later comparison with persistent recurrence.
