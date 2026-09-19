# Two public observations before another recurrent mechanism

Prospective design, unfrozen and unrun. Only this memo was written. No fitting, model calls, native calls, evaluation draws or current execution/result reads were performed. The current cache protocol was read and authenticated at SHA-256 `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa`; its 70 frozen sources are unchanged. This proposal is conditional on that study's terminal, audited outcome.

**Recommended question:** can two actual angle observations and their elapsed time explain the remaining value of persistent learned state? Start with a small, matched observation-feature ablation. If persistence still wins, add ordered issued-action context before interpreting the residual as superior hidden-motion inference. A finite difference is an interval-average motion feature, not an identified instantaneous velocity.

## Decision after the cache study

| Terminal outcome | Next decision |
|---|---|
| Persistent recurrence qualifies against both explicit-cache families, with the frozen competence and full-sensing checks intact | A two-observation comparison is justified. Preserve all current fits and outcomes; train new comparators and use fresh paired evaluation cases. |
| The persistent-versus-cache gate fails | Report the failed gate. Do not rename failure as cache equivalence. Do not automatically add a velocity model to recover a preferred conclusion. A smaller noninferiority or efficiency study can be separately justified and frozen. |
| Predictive gains persist but native utility remains inconclusive | Keep prediction and control conclusions separate. The completed prior reward diagnostic motivates checking reward/ranking error, not automatically enlarging recurrent state. |
| Motion features match persistence and improve their no-motion controls | Report that two-observation information is sufficient within the tested setting and tolerance. This does not establish native velocity identification or architectural novelty. |
| Persistence still wins the minimal motion comparison | Test the ordered-action control below before advancing to a new latent estimator. The minimal motion arm forgets previous issued commands at real boundaries. |

The timing of this decision does not permit selecting fit seeds, widths or margins from current outcomes. If the next comparison is pursued, finalize the gate and cost cap before any new evaluation draws.

## Causal measurement state and feature contract

Keep the external interface unchanged: a public packet has eight fields `[cos(q1), cos(q2), sin(q1), sin(q2), goal_x, goal_y, valid, age_seconds]`; an issued command has two fields. The training targets and their masks remain the original public packets and total rewards. Internal cached features never become extra observations or labels.

At real decision index `t`, retain the two latest **actually valid** public angle measurements. Let their indices be `p < l <= t`, with orientations `z_p` and `z_l`. Retain a count capped at two, integer indices, static public target, real decision index and imagined depth. These are explicit nonlearned episode state. Every new episode clears them. A first visible packet establishes only `l`; it does not create a fictitious previous measurement.

At real assimilation:

1. Sanitize the public packet first. If `valid=0`, the first four external fields are zero, regardless of supplied placeholders.
2. At a valid packet, move the old last measurement to the previous slot, record the new measurement at index `t`, and set age to zero. At a missing packet, neither measured orientation nor either measurement timestamp changes.
3. Check public age against `(t-l)*dt`, using the existing frozen tolerances `rtol=1e-5`, `atol=1e-6`, and require exactly zero age for a valid packet. Require the initial packet to be valid and the public target to remain static. No future observation schedule or time-to-reappearance is supplied.
4. With two observations, compute each angular change directly from measured sine/cosine pairs:

   `s_delta = s_l*c_p - c_l*s_p`

   `c_delta = c_l*c_p + s_l*s_p`

   `delta = principal_angle(atan2(s_delta, c_delta))`

   Define `principal_angle` deterministically on `[-pi, pi)`, including the exact positive-pi tie. Compute the arithmetic in float64, then cast the feature to the model's float32 input. This avoids subtracting two separately wrapped `atan2` estimates. It does not resolve a real movement of more than pi between observations.
5. Let `Delta=(l-p)*dt`, `omega_interval=delta/Delta`, and supply `dt*omega_interval=delta/(l-p)` as the two motion inputs. This fixed scaling uses only the known step size. There is no evaluation-fitted normalization, unwrapping using hidden velocity, extrapolation from a future observation, or learned clipping threshold.
6. With fewer than two observations, set both motion features and interval to zero and set the two-observation flag to zero. A zero measured angular change with flag one remains distinguishable from unavailable motion. During a blackout, interval and interval-motion features remain fixed while current age increases.

The 12 root encoder features, in fixed order, are:

`[last_measured_cos1, last_measured_cos2, last_measured_sin1, last_measured_sin2, goal_x, goal_y, current_valid, current_age_seconds, dt*omega1, dt*omega2, measured_interval_seconds, has_two_measurements]`.

Six consecutive missing packets produce a **seven-step** interval between the last visible packet and the first visible packet after the gap. Ten missing packets produce an eleven-step interval. Do not divide by six or ten, and do not substitute one timestep at reacquisition.

This is an average displacement over the measurement interval and can be stale or aliased. Acceleration, collisions/joint limits and unobserved actuator disturbances prevent it from being interpreted as current native `qvel`. Availability is a data-presence flag, not a confidence estimate. Never resolve multi-turn ambiguity by choosing the branch nearest privileged state or the eventual trajectory. A two-orientation history contains the same basic measured information as this representation; improvement establishes useful extra history, not proof that the network internally performs velocity inference.

## The smallest matched comparison: five arms, three fit pairs

Use one new 12-input constructor per architecture family and train all arms from fresh paired initializations. The within-family motion contrast must keep interval and availability metadata identical: otherwise motion is confounded with extra information about the observation schedule.

| Arm | Root feature differences | Learned state across real assimilation |
|---|---|---|
| `persistent_gru12` | Original sanitized public eight fields, plus four zero-padded inputs | Preserve the existing persistent GRU's valid-measurement update and action-transition semantics. |
| `cache_metadata_gru12` | Last-measurement cache plus actual interval/availability; only the two motion columns are zero | Erase hidden state, then encode every root from zero, including missing roots. |
| `motion_gru12` | Same as cache-metadata GRU, with both measured motion columns enabled | Identical reset-and-encode semantics. |
| `cache_metadata_mlp12` | Same feature contract as the cache-metadata GRU | No learned state crosses a real boundary. |
| `motion_mlp12` | Same as cache-metadata MLP, with both motion columns enabled | No learned state crosses a real boundary. |

The `cache_metadata` controls are new controls, not a relabeling of the current eight-feature cache fits. They remember two timestamps, but do not expose the previous orientation or its displacement to the model. Retain the same explicit state schema in the motion and no-motion variants, so their stored-state and feature-construction cost can also be compared transparently.

Keep hidden size 64 for all three GRUs and the existing transition/reward/observation heads. Expanding the observation-update input from eight to twelve adds `3*64*4=768` parameters, yielding **37,573** parameters. A 12-feature MLP with width 108 has **37,697**, 0.33% more. Its parameter formula is `3*w*w + (feature_dim+13)*w + 5`. The original totals were 36,805 and 36,599. Zero-padded or disabled feature columns have allocated parameters but do not add active information; say so rather than claiming equal effective capacity.

Within each fit pair, store one complete GRU initialization and one MLP initialization, and copy them byte-for-byte to their respective variants. Keep minibatch order paired across all five arms. A shared constructor seed by itself is inadequate for comparison with an eight-input constructor: larger input matrices consume different random draws and shift later-layer initialization. Bind fresh classes, input layout, enabled columns, all tensors and optimizer states. Do not load an eight-feature fitted checkpoint and reinterpret it as this model.

The padded persistent arm is a fresh capacity bridge, not a numerical continuation of the old persistent fit. Its extra zero-input columns do not change its information set. Match unchanged tensors through the stored initialization, not by claiming the old and new constructor seeds have the same meaning.

## Actions, imagination and selected-action separation

In the minimal 12-feature study, **only the currently proposed/issued command** enters each `advance`, exactly as in the existing models. The two-observation features do not receive past-command summaries, realized actuator noise, applied controls or native reward components. The persistent GRU can retain earlier issued actions through its actual transition state. Explicit motion arms cannot. This asymmetry is the main remaining interpretation limit if persistence wins; it is not concealed as an equal-information comparison.

Every new real packet discards learned hidden/predicted state in the explicit-history arms. On a real root, the MLP's private encoder starts with the actual last observation and motion metadata. On later imagined steps it uses predicted angle features, static target, validity zero and incremented age, while the two-observation interval/motion/availability metadata stays attached to the real root. Those stale metadata do not become newly measured velocity. The GRU initializes its hidden state from the same root features and then evolves through the candidate's actions.

Candidate expansion deep-copies the explicit measurement buffers. `advance` changes only private imagined state/depth, never real measurement timestamps or caches. It may not assimilate predicted angles as new measurements. After search, reconstruct the selected one-step state by advancing the issued command from the untouched real root. Only that state may precede the next real assimilation. Require startup or exactly one selected-action depth; reject multi-step candidate terminal states. The controller audit, rather than the model class alone, proves that the retained action was the one actually issued.

Keep action clipping, known-noise actuator-cost correction, reward head, Anchor loss and actual-validity masks unchanged. Do not simultaneously add velocity supervision, a physical transition model, a geometry reward head or a new reward scale. Those would answer different questions.

## Stronger control if actions could explain a residual gap

Before attributing a persistent win to a new latent mechanism, supply explicit ordered issued actions to **both** motion and no-motion controls in a separate declared comparison. Prefer the raw command window over an action sum: reversing two commands can change the state even when their sum is identical.

For the currently declared, separated six- and ten-step gaps, the last **11 issued commands** cover all commands between the penultimate valid measurement and the current root. Verify this from observed indices for every root; do not assume it for arbitrary future sensing schedules. A ten-step blackout needs ten post-last-observation commands at its final missing root, plus the preceding command between the two stored measurements. At reacquisition, eleven commands separate the two visible measurements. If a new schedule violates this bound, reject the declared contract or freeze a longer window before use.

At root `t`, use chronological commands `a_(t-11),...,a_(t-1)`, zero-padding negative indices and supplying eleven explicit availability bits. No `a_t` or future command enters root features. This adds 22 action values and 11 masks, giving 45 features. Both matched cache-metadata and motion arms receive the identical window. The persistent bridge can receive the same public ordered window in addition to its recurrent state; label this stronger fresh comparator rather than reusing earlier scores.

Keep actual action history separate from a branch's hypothetical command history. During imagination, a private branch may append its own proposed command for its next imagined step. It must not mutate the real history. At the next real assimilation, commit only the previously selected issued action, verified against the saved public command; discard hypothetical histories. State schemas should distinguish these two buffers, or derive the branch buffer from the immutable root plus its declared prefix. Missing observations do not prevent committing an actual issued command.

With 45 encoder features, a GRU64 has **43,909** parameters. MLP112 has **44,133**, 0.51% more. This is a larger but still bounded alternative, not a free addition to the 12-feature study. It changes workload and needs its own capacity rehearsal, registered initialization, actual operation counts, timing and gate. Do not silently append it after inspecting the new test outcomes and pool the best result.

Success of this stronger arm would support sufficiency of two observations plus short explicit action history. It would not isolate a velocity algorithm. Persistence winning both arms could still reflect better estimation, nonlinear history compression, optimization or reward modeling, rather than a novel recurrent mechanism.

## Evaluation, limits and continuation

For the first study, retain 768 fixed public training episodes, 48 epochs, batches of 32, the unchanged optimizer/loss and 1,152 updates per fit. Fifteen fits require 17,280 optimizer updates. Keep the current planner's CEM256, horizon 12, action block 3, native 50-step episodes, ordinary six-missing and shifted ten-missing panels. Proposed fresh evaluation scope is 64 paired control cases across all three panels, all five existing references, and 96 common public-history prediction episodes. This is 60 control rows, including references. These dimensions are proposed, not a launch authorization.

Use independent new scored and engineering namespaces and exclude all previous full-size scored/engineering streams, including the current cache study after its lineage is authenticated. Pair reset states, actuator disturbances, sensor schedules, the initial CEM bank and CEM innovations. Later score-adaptive proposal banks may differ. Keep all fits, no test-driven early stopping or favorable-fit selection.

Training-update matching and near parameter matching are not compute matching. Report full fitting time, actual root encodes, all imagined transitions, feature arithmetic, copies, state bytes, search/proposal work, scoring and selected-state advance. Rehearse the complete new artifact tree before choosing an execution/audit cap. Do not reduce the scientific comparison to fit an unexpectedly small runtime budget. Wall time on a shared host is throughput evidence, not isolated latency or a FLOP estimate.

Predeclare the GRU motion-versus-cache-metadata contrast as the primary mechanism test; retain the MLP contrast as an independently reported corroborating comparison rather than selecting whichever succeeds. A useful prospective criterion would combine a fixed native-cost improvement margin over the no-motion control, agreement of all three paired fits, full-sensing nondegradation, and reference competence. Exact margins require a separate freeze.

If the intended claim is that explicit motion **matches** persistence, define a noninferiority test in advance. For example, choose a maximum tolerable relative cost excess `delta` before new evaluations, average paired fit differences by case, and require the one-sided bound for motion-minus-persistent cost to lie below `delta` on both gap panels, with a separate all-fit condition. A conventional failure to show persistence superiority is not this test. Case-conditional intervals from three fits do not establish broad training-seed robustness. Do not describe equal updates, a passed prediction metric or a nonsignificant utility difference as equivalence.

The next recurrent mechanism is justified only after a robust residual advantage survives these explicit-history controls and a targeted reward/planning diagnosis. No current outcome can establish anatomical connectivity, plasticity or a second-environment result.

## Implementation and independent-audit boundary

The frozen code supplies useful interfaces but **cannot accept this new model by a configuration switch**. `reacher_cached_observation.py` hardcodes eight-field encoder/state widths, exact state membership and exact-class configuration identities. The frozen trainer and controller also use exact-class registries. Monkeypatching those registries, weakening identity checks or mutating `_cached` flags would change frozen semantics.

Implement a new observation-history model module and new explicit trainer/control/protocol/auditor adapters. Preserve the original eight-field public packet and model `initial/assimilate/advance` signatures, so the existing pure Anchor loss, analytic action-cost function, CEM machinery and artifact serialization can be reused without editing their sources. New adapters must have their own exact class registry and checkpoint configuration. Do not subclass the private cache boundary and bypass its state-key checks as an informal compatibility shortcut.

Minimum synthetic tests and independent saved-state assertions:

1. Reconstruct both measurement slots, indices, flag, interval, motion features and public age from the actual packet prefix. Check all real roots; no neural replay is needed for this claim.
2. Test first and second observations, stationary measured motion versus unavailable motion, exact pi tie, crossing the -pi/pi boundary, an irregular interval, seven/eleven-step reacquisition and a deliberately aliased multi-turn example. The aliased case must demonstrate the limitation, not silently choose the true branch.
3. Poison old hidden/predicted angles and unavailable public angular placeholders. If the declared measurement cache/current packet is unchanged, an explicit-history real root must be unchanged. Change a genuinely observed earlier orientation and verify the motion feature changes only when it belongs to the retained pair.
4. Verify the no-motion arm shares interval/availability, root call counts, initialization and order with the motion arm. The only within-family feature differences must be the two declared motion columns. Original public observation-target masks remain byte-identical.
5. Ensure candidate mutation cannot affect sibling or real caches, an imagined prediction cannot update a measurement timestamp, a selected one-step state can assimilate once, and a multi-step terminal cannot. Audit selected state against the actual issued command.
6. For the stronger action control, reconstruct each actual command window directly from `policy.commands`; perturb future commands and realized noise without changing current root features. Reverse two past commands and require their order to remain distinguishable. Verify branch-only appends and the eleven-step coverage bound.
7. Authenticate all five actual model identities, feature orders/scales, active columns, paired initial tensors, all minibatch orders and optimizer cursors. Charge feature extraction and state copying; verify exact all-fit/all-panel membership before calculating any gate.

The reviewed cache model/trainer/control/protocol and original observation/world-model sources all matched the current authenticated plan. No source, protocol, checkpoint or execution artifact was changed for this design.
