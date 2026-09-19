# Next experiment: does persistent state beat the matched GRU controls under geometry scoring?

Read-only design after the complete geometry-scoring audit, 2026-09-18. This proposal is unfrozen and unrun. No model, native-environment, training or new evaluation-stream calls were made. Only this note was written.

**Run the existing trained current-packet and cached-angle GRUs through the same geometry scorer before fitting a ranking head.** This is a zero-fit test using stronger within-architecture controls. A new head would change the interface that just made the recurrent predictions useful, while leaving the memory comparison unresolved.

## What the completed evidence establishes

The geometry audit receipt was authenticated at `9cd5fdde29601ae5227afb9b3725be1fc855d4c7ba72aa7ca4c83248ae8de72e`; its bound summary hash is `b8691b3e71707890c1fce1d9bae027b0d67e6f2d86023e3eba0ffa2ec80e0d47`. All 25 prospective checks passed. The audit replayed 329,088 recorded native transitions. The results retain all fits:

| Fixed model and score | Full sensing | Six missing steps | Ten missing steps |
|---|---:|---:|---:|
| Persistent GRU, learned reward | 8.6378 | 8.7382 | 8.8163 |
| Persistent GRU, geometry | 5.4941 | 5.6182 | 6.2881 |
| Cached MLP, learned reward | 8.0838 | 8.2155 | 8.3050 |
| Cached MLP, geometry | 6.9667 | 7.2556 | 7.4572 |

Values are mean native episode cost over all three fits and 64 paired cases; lower is better. Geometry lowers GRU cost by 35.7%/28.7% on the gap panels. With both families using geometry, GRU cost is 22.6%/15.7% lower than cached MLP, with every paired fit improving. This supports the particular scoring intervention and useful recurrent predictions. It does not yet isolate persistent state from a GRU-versus-MLP difference.

The result is not free additional accuracy: GRU whole-row time rises from 4.615 to 7.051 ms per case/decision on ordinary gaps, including cache/state validation and copying inside the controller, trace storage, native stepping and geometry work. Global source/checkpoint validation and copying are outside these row timings. Cached MLP geometry takes 7.492 ms by the same accounting. These are shared-host amortized row timings, not isolated latency or total FLOPs.

The older cache study remains a failed 15/28 result under its learned-reward interface. Its audit was independently authenticated at `d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790`, with summary `621f083f648802e1e133c5f3486fed6f16aed24126e839e360b17be4d1f3a24c`. Its already-trained GRU controls were not tested by the completed geometry intervention. That omission is now the most direct uncertainty to resolve.

## The exact available controls

Use all three original fit pairs for four declared classes, 12 inherited checkpoints total. No new fit, checkpoint selection, encoder substitution or reward-head fitting.

| Actual class / registry kind | State available at each real decision | What its contrast answers |
|---|---|---|
| `GRUResidualRewardWorldModel` / `residual_gru` | Retains action-conditioned hidden state; assimilates visible observations; missing observations preserve the predicted hidden state | Persistent state-update policy under test |
| `EncodedCurrentGRUWorldModel` / `encoded_current_gru` | Erases hidden state and encodes the current sanitized packet from zero, including missing packets | Whether the same GRU machinery can act from current input alone |
| `CachedObservationGRUWorldModel` / `cached_gru` | Erases hidden state and encodes last actually observed angles plus current target, validity and age | Whether one explicit measurement cache explains the gain within the same architecture |
| `CachedObservationMLPWorldModel` / `cached_mlp` | Same actual-angle cache, no learned state across real assimilation | Continuity with the strongest previously measured simple baseline; secondary comparison |

The three GRUs each have **36,805 parameters**, identical tensor schemas and paired initial tensors. All original fits saw the same 768 training episodes, minibatch order, loss and 1,152 updates. Cached MLP has 36,599 parameters. These are separately trained final weights; equal initial tensors do not make this a same-weights deployment intervention.

All three GRUs retain recurrent state during an imagined candidate rollout. The intended treatment is persistent learned state **across real decision boundaries**, not recurrence inside planning. The persistent implementation also uses a validity-gated observation update, whereas the reset controls encode every real packet. A positive result therefore supports this trained persistent update policy over these trained reset policies. It does not identify a particular gate, hidden coordinate, Bayesian filter, or velocity representation as the cause.

Full sensing still supplies angles without velocities. A full-sensing recurrent gain can therefore be meaningful; it cannot be called a gain specific to blackouts. The cache is an actual past measurement, not a current-angle estimate, and neither reset control carries older issued actions across real boundaries.

## One bounded prospective comparison

Use a new protocol and new domain-separated evaluation streams, excluded from every consumed scored and engineering registry. Evaluate all four families and all three fit pairs again on the same fresh 64 cases per panel: full sensing, six-step gaps and ten-step gaps. Reuse the native 50-step horizon, dt=.02, noise=.05, gap onsets/phase rule and public eight-field packet. Pair reset states, actuator disturbances, sensing schedules and planner innovations across every row.

Use **geometry scoring only** in this next primary comparison: unchanged model transitions and original reward-head computation, approximate XML forward kinematics, expected clipped-action cost charged once, per-step clipping [-2.5,0], sequential float32 summation, CEM256, horizon12, action-block3 and terminal truncation. The completed learned-versus-geometry result is historical context, not an extra control row needed to answer this question. Save the original learned reward predictions for descriptive checks, but do not use them to choose methods.

Primary comparisons are persistent GRU versus encoded-current GRU and versus cached GRU. Cached MLP remains a secondary practical comparator. Repeat persistent GRU and cached MLP on the fresh cases: evaluating only the two new controls and joining them to old means would break case pairing and reuse exposed evaluation as if it were fresh.

A concrete proposed all-required gate, to freeze before any new draws:

- On each gap panel, persistent GRU mean cost must be at least 3% lower than each of the two reset GRU controls: four checks.
- Persistent GRU must not lose any paired fit against either reset control on either gap panel: twelve checks. Exact ties are allowed here, but cannot alone satisfy the mean margin.
- Full-sensing persistent mean cost may be at most 2% higher than each reset GRU control: two checks.
- Every persistent fit must beat zero action by at least 10% on both gap panels: six checks.
- The declared known-state physics reference must beat zero action by at least 10% on the ordinary panel: one check.

This is a proposed 25-check continuation rule, not a retrospective redefinition of the just-completed gate. A failed check stops escalation to a memory-mechanism claim. Report every row and conditional paired-case interval; intervals over cases condition on the three saved fits and shared training corpus. They are not uncertainty over new training seeds.

## Search and computation controls that matter

**Same architecture comparisons:** use the identical CEM256 implementation, action parameterization, initial64 bank, four paid64-candidate stages, eight elites, standard-deviation floor, paid final mean, global-best retention and tie ordering. Later proposals adapt to each model, so they need not be identical. Count every imagined transition and geometry call; retain all root/carried states and candidate evidence. Cache validation and copying differ despite equal GRU parameters and neural dimensions.

**Physics comparisons:** the current known-state, particle and public-kinematic references use only the common64 proposals and nominal native-distance scoring. The learned arms use adaptive256 proposals and geometry/expected-action scoring. Consequently, the current 5.6182 GRU versus 7.6200 known-state ordinary cost is not evidence that learned dynamics beat correctly informed physics.

For the next comparison, use the same CEM256/horizon/block and the same declared geometry score for all three supplied-physics references. Preserve each information boundary: true state only for `known_state`; public packets/actions only for particle and kinematic estimates. No reference receives realized future actuator noise. Preserve the nominal rollout assumption and report it. This gives 36 learned rows plus 15 reference/floor rows, **51 rows and 163,200 executed native transitions**. No new counterfactual branch-label dataset is needed for this primary test.

This reference change requires an explicit adapter. `PhysicsMPC.plan` currently returns a summed native-distance-plus-command-square cost, without the per-step values needed for identical clipping. Do not clip that aggregate or silently call it the same objective. A new physics callback must retain nominal per-step predicted angles, apply the same geometry/expected-action score and float32 accumulation, and expose all256 paid proposals to the existing CEM kernel. Its simulator knowledge is privileged even for observation-only state estimates. Matching candidate count does not match CPU time.

**Compute claims:** make the first result a candidate-budget-matched mechanism comparison, not an equal-wall-time claim. Charge complete row and decision costs, all head work, geometry, cache/filter work, initialization and storage. Report all points, not a selected frontier. If persistence is better and faster at this fixed budget, that establishes dominance at this measured operating point. If it is slower, retain the utility result but do not claim efficiency. A later equal-total-compute comparison would need a separately profiled, frozen candidate/restart budget before new evaluation, including physics/filter costs; it is unnecessary to establish the first memory contrast. Do not pad a cheap arm with sleep or equate K×H with total compute.

## Small implementation surface and stopping decision

The existing cache trainer registry already binds all four actual classes and all checkpoint metadata. Load the 12 exact checkpoints directly from the completed cache study, authenticate their original class/settings/initialization/order identities, and restore deployment tensors only. No trainer or Adam restoration is needed. Restore every model before drawing the fresh evaluation cohort; snapshot before/after tensors.

The frozen geometry controller deliberately accepts only `residual_gru` and `cached_mlp`; its protocol, runner and auditor also hardcode six fits and48 diagnostic roots. Setting a different plan field or monkeypatching `KINDS` would not be a valid extension. Add new, separately bound controller/protocol/runner/auditor files, retaining the old geometry formula and CEM kernels unchanged. Reuse the cache controller's existing exact-class identities and saved-state accounting. Keep actual public validity/age untouched, update real state exactly once per packet, isolate every candidate, and carry the selected one-step action from the real root rather than a candidate terminal. Synthetic tests must cover all four classes and geometry parity before the prospective freeze. The 80 completed-study sources stay immutable.

Interpret the next outcome narrowly:

- If cached GRU matches or beats persistence, the present GRU-versus-MLP advantage does not establish persistent-memory value. Keep the useful geometry controller and defer a new recurrent or biological architecture claim.
- If persistence clears both reset-GRU controls, advance to the already-written **two-valid-observation plus issued-action-history** comparison under the now-established geometry interface. A last-angle cache is not a strong velocity/history baseline. Start with matched feature controls and fresh training, not posthoc hidden resets that create a deployment distribution shift.
- If persistence wins only with full sensing, or the gap benefit changes sign by fit, retain those results without declaring robust blackout memory.

The rank-head proposal in [conditional-followups.md](conditional-followups.md) remains plausible, but its conditional trigger does not make it the immediate priority. Geometry has already supplied a useful scalar interface. Testing the existing within-GRU controls resolves a more specific causal alternative with zero new fitting. Learned return/ranking heads, terminal critics, adaptive computation and connectomes should follow a demonstrated remaining bottleneck. No new architecture, biological superiority, calibrated uncertainty or cross-environment generality is established by either the completed result or this proposal.

Source pointers: `src/openjev/research/reacher_world_models.py::GRUWorldModel`; `reacher_cached_observation.py::_PublicBoundary.assimilate/advance`; `reacher_cache_training.py::REGISTRY/SEMANTICS`; `reacher_cache_control.py::model_identity`; `reacher_geometry_control.py::_identity/score_search`; `reacher_adaptive_search.py::search`; `reacher_physics_control.py::PhysicsMPC.plan`; the two authenticated audit summaries above. No new literature claims are introduced by this source-level design.
