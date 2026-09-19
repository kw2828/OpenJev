# Geometry-scored real-observation memory study

Continuation: **PASS** (25/25).

All twelve inherited fits and 51 control rows are retained. Zero new fits or training updates. All planned rows use CEM256; total wall time and information are not matched.

| Panel | Policy | Mean native cost | Whole-row wall (s) | Amortized wall (ms/action/case) |
|---|---|---:|---:|---:|
| full | residual_gru-pair0 | 5.629319 | 22.213 | 6.941521 |
| full | residual_gru-pair1 | 5.966741 | 22.853 | 7.141644 |
| full | residual_gru-pair2 | 5.016669 | 22.385 | 6.995208 |
| full | encoded_current_gru-pair0 | 8.010211 | 23.672 | 7.397557 |
| full | encoded_current_gru-pair1 | 8.119734 | 23.881 | 7.462676 |
| full | encoded_current_gru-pair2 | 7.972373 | 23.422 | 7.319384 |
| full | cached_gru-pair0 | 7.809205 | 24.189 | 7.559015 |
| full | cached_gru-pair1 | 8.039507 | 23.885 | 7.464217 |
| full | cached_gru-pair2 | 7.859412 | 23.388 | 7.308858 |
| full | cached_mlp-pair0 | 6.708122 | 24.619 | 7.693500 |
| full | cached_mlp-pair1 | 6.675464 | 23.768 | 7.427652 |
| full | cached_mlp-pair2 | 6.732931 | 23.831 | 7.447243 |
| full | known_state | 4.568717 | 140.597 | 43.936673 |
| full | particle | 4.731065 | 143.958 | 44.986733 |
| full | public_kinematic | 4.580680 | 140.963 | 44.051074 |
| full | zero | 11.830064 | 0.179 | 0.056031 |
| full | uniform | 42.796284 | 0.181 | 0.056448 |
| ordinary | residual_gru-pair0 | 5.698578 | 22.391 | 6.997275 |
| ordinary | residual_gru-pair1 | 6.324741 | 21.930 | 6.852981 |
| ordinary | residual_gru-pair2 | 5.104380 | 22.160 | 6.925030 |
| ordinary | encoded_current_gru-pair0 | 9.063089 | 23.235 | 7.260798 |
| ordinary | encoded_current_gru-pair1 | 9.515180 | 23.291 | 7.278306 |
| ordinary | encoded_current_gru-pair2 | 9.143557 | 23.352 | 7.297504 |
| ordinary | cached_gru-pair0 | 8.255982 | 23.227 | 7.258376 |
| ordinary | cached_gru-pair1 | 8.546184 | 23.595 | 7.373348 |
| ordinary | cached_gru-pair2 | 8.333701 | 23.712 | 7.410126 |
| ordinary | cached_mlp-pair0 | 6.947214 | 23.511 | 7.347285 |
| ordinary | cached_mlp-pair1 | 7.010055 | 23.871 | 7.459557 |
| ordinary | cached_mlp-pair2 | 6.968725 | 23.959 | 7.487283 |
| ordinary | known_state | 4.568717 | 140.802 | 44.000660 |
| ordinary | particle | 4.786677 | 142.564 | 44.551290 |
| ordinary | public_kinematic | 4.640030 | 138.396 | 43.248732 |
| ordinary | zero | 11.830064 | 0.165 | 0.051560 |
| ordinary | uniform | 42.796284 | 0.168 | 0.052628 |
| shift | residual_gru-pair0 | 6.315875 | 21.811 | 6.815825 |
| shift | residual_gru-pair1 | 6.859159 | 20.495 | 6.404536 |
| shift | residual_gru-pair2 | 5.713888 | 20.605 | 6.439047 |
| shift | encoded_current_gru-pair0 | 9.483682 | 23.118 | 7.224429 |
| shift | encoded_current_gru-pair1 | 10.128282 | 21.760 | 6.800149 |
| shift | encoded_current_gru-pair2 | 9.610265 | 21.814 | 6.816724 |
| shift | cached_gru-pair0 | 8.469254 | 22.287 | 6.964696 |
| shift | cached_gru-pair1 | 8.672560 | 22.153 | 6.922828 |
| shift | cached_gru-pair2 | 8.380430 | 22.051 | 6.890936 |
| shift | cached_mlp-pair0 | 7.249306 | 22.165 | 6.926688 |
| shift | cached_mlp-pair1 | 7.293202 | 22.342 | 6.981851 |
| shift | cached_mlp-pair2 | 7.159956 | 22.637 | 7.074022 |
| shift | known_state | 4.568717 | 138.785 | 43.370243 |
| shift | particle | 4.892595 | 139.171 | 43.490849 |
| shift | public_kinematic | 4.755555 | 138.308 | 43.221202 |
| shift | zero | 11.830064 | 0.176 | 0.055036 |
| shift | uniform | 42.796284 | 0.189 | 0.059128 |

Every row contains 64 paired cases and 50 decisions. Wall time includes setup, native stepping and trace storage on a shared host.

| Gate check | Measured cost | Required upper bound | Result |
|---|---:|---:|---|
| gap_mean/ordinary/encoded_current_gru | 5.709233 | 8.963390 | PASS |
| gap_pair/ordinary/encoded_current_gru/pair0 | 5.698578 | 9.063089 | PASS |
| gap_pair/ordinary/encoded_current_gru/pair1 | 6.324741 | 9.515180 | PASS |
| gap_pair/ordinary/encoded_current_gru/pair2 | 5.104380 | 9.143557 | PASS |
| gap_mean/shift/encoded_current_gru | 6.296307 | 9.448521 | PASS |
| gap_pair/shift/encoded_current_gru/pair0 | 6.315875 | 9.483682 | PASS |
| gap_pair/shift/encoded_current_gru/pair1 | 6.859159 | 10.128282 | PASS |
| gap_pair/shift/encoded_current_gru/pair2 | 5.713888 | 9.610265 | PASS |
| full_mean/encoded_current_gru | 5.537576 | 8.194788 | PASS |
| gap_mean/ordinary/cached_gru | 5.709233 | 8.127264 | PASS |
| gap_pair/ordinary/cached_gru/pair0 | 5.698578 | 8.255982 | PASS |
| gap_pair/ordinary/cached_gru/pair1 | 6.324741 | 8.546184 | PASS |
| gap_pair/ordinary/cached_gru/pair2 | 5.104380 | 8.333701 | PASS |
| gap_mean/shift/cached_gru | 6.296307 | 8.252192 | PASS |
| gap_pair/shift/cached_gru/pair0 | 6.315875 | 8.469254 | PASS |
| gap_pair/shift/cached_gru/pair1 | 6.859159 | 8.672560 | PASS |
| gap_pair/shift/cached_gru/pair2 | 5.713888 | 8.380430 | PASS |
| full_mean/cached_gru | 5.537576 | 8.060762 | PASS |
| competence/ordinary/residual_gru-pair0 | 5.698578 | 10.647058 | PASS |
| competence/ordinary/residual_gru-pair1 | 6.324741 | 10.647058 | PASS |
| competence/ordinary/residual_gru-pair2 | 5.104380 | 10.647058 | PASS |
| competence/shift/residual_gru-pair0 | 6.315875 | 10.647058 | PASS |
| competence/shift/residual_gru-pair1 | 6.859159 | 10.647058 | PASS |
| competence/shift/residual_gru-pair2 | 5.713888 | 10.647058 | PASS |
| competence/ordinary/known_state | 4.568717 | 10.647058 | PASS |

| Panel / contrast | Mean cost difference | Percent cost change | Conditional paired-case 95% interval | Pair 0 / 1 / 2 differences |
|---|---:|---:|---|---|
| full / persistent_minus_encoded_current_gru | -2.496530 | -31.074% | [-2.977526, -2.041690] | -2.380893 / -2.152993 / -2.955704 |
| full / persistent_minus_cached_gru | -2.365132 | -29.928% | [-2.824215, -1.924333] | -2.179886 / -2.072766 / -2.842743 |
| full / persistent_minus_cached_mlp | -1.167930 | -17.417% | [-1.455187, -0.886910] | -1.078804 / -0.708723 / -1.716262 |
| full / cached_gru_minus_encoded_current_gru | -0.131398 | -1.636% | [-0.202425, -0.068779] | -0.201007 / -0.080226 / -0.112961 |
| full / cached_gru_minus_cached_mlp | 1.197202 | 17.854% | [0.874203, 1.517801] | 1.101082 / 1.364043 / 1.126481 |
| ordinary / persistent_minus_encoded_current_gru | -3.531376 | -38.216% | [-4.089386, -2.984670] | -3.364511 / -3.190439 / -4.039177 |
| ordinary / persistent_minus_cached_gru | -2.669389 | -31.860% | [-3.129126, -2.210240] | -2.557403 / -2.221443 / -3.229321 |
| ordinary / persistent_minus_cached_mlp | -1.266098 | -18.151% | [-1.544728, -0.990735] | -1.248636 / -0.685313 / -1.864345 |
| ordinary / cached_gru_minus_encoded_current_gru | -0.861986 | -9.328% | [-1.146619, -0.602572] | -0.807108 / -0.968996 / -0.809856 |
| ordinary / cached_gru_minus_cached_mlp | 1.403291 | 20.118% | [1.005989, 1.795782] | 1.308767 / 1.536130 / 1.364976 |
| shift / persistent_minus_encoded_current_gru | -3.444436 | -35.361% | [-3.947403, -2.958304] | -3.167807 / -3.269123 / -3.896377 |
| shift / persistent_minus_cached_gru | -2.211107 | -25.990% | [-2.564162, -1.867246] | -2.153379 / -1.813401 / -2.666542 |
| shift / persistent_minus_cached_mlp | -0.937847 | -12.964% | [-1.193701, -0.682836] | -0.933431 / -0.434043 / -1.446068 |
| shift / cached_gru_minus_encoded_current_gru | -1.233329 | -12.662% | [-1.544210, -0.945806] | -1.014428 / -1.455722 / -1.229835 |
| shift / cached_gru_minus_cached_mlp | 1.273260 | 17.601% | [0.891198, 1.641374] | 1.219948 / 1.379358 / 1.220473 |

Intervals are copied from the authenticated audit without new resampling. They condition on the three saved fits and training corpus; they are not population-level seed uncertainty or multiplicity-corrected confirmation.

| Paid scope | Wall seconds |
|---|---:|
| inheritance_copy_wall_seconds | 0.094706 |
| restore_wall_seconds | 0.242930 |
| innovation_generation_and_storage_seconds | 2.003180 |
| control_row_wall_seconds | 2088.180367 |
| execution_wall_seconds | 2109.204781 |
| audit_validation_wall_seconds | 1384.438953 |
| Prior geometry lineage, including its cache ancestry | 5775.011990 |
| Cumulative through this execution | 7884.216771 |

Phase and kernel timings are nested. Do not add the cache parent a second time to the geometry lineage. Engineering preparation and publication are separate costs.

| Work scope | Count |
|---|---:|
| learned_candidate_sequences | 29,491,200 |
| learned_candidate_transitions | 314,966,016 |
| learned_selected_advances | 115,200 |
| physics_candidate_sequences | 7,372,800 |
| physics_candidate_transitions | 78,741,504 |
| physics_selected_advances | 28,800 |
| physics_candidate_integration_substeps | 157,483,008 |
| physics_selected_integration_substeps | 57,600 |
| executed_native_control_transitions | 163,200 |
| public_observer_native_transitions | 310,464 |

The nominal planning replay count is separate from executed native-control replay and public-observer reconstruction. The selected one-action predictions are charged separately from candidate trajectories.

- Zero new fits: all twelve inherited models and all 51 control rows are retained. The older cache failure and score-intervention success remain separate claims.
- The GRU arms are all recurrent within imagined rollouts. This compares separately trained real-assimilation policies and validity gating, not recurrence in isolation.
- All nonfloor controllers use CEM256 with the same geometry interface, horizon 12 and block 3. Equal candidate budgets do not match total compute or information.
- Physics references use supplied nominal dynamics. Known-state has privileged current qpos/qvel; public observers do not. None is an optimal stochastic-control oracle.
- Geometry approximates native RK4 cached-body reward; distance at projected mean angles is not expected distance.
- Conditional paired-case intervals fix these three saved fits and training corpus. They are not training-seed population uncertainty or multiplicity-corrected confirmation.
- Shared-host whole-row wall includes setup, native stepping and evidence storage. Amortized throughput is not isolated latency, FLOPs or a cross-machine speed result.
- The replay uses ordinary case0, pair0, every saved action and all four classes by a fixed pre-outcome rule. It is not a selected winner or aggregate result.
- Saved qpos is used only for schematic centerline drawing. Public-availability shading concerns decision inputs; no clean hidden angles are supplied to the controller.
- No connectome advantage, biological superiority, novel architecture, calibrated uncertainty or cross-environment generality is established.
- Zero new fits; all12 inherited checkpoints and51 control rows retained. Historical cache15/28 failure and geometry25/25 success remain separate claims.
- Before/after observed tensors are equal. No trainer/optimizer is invoked by this audit; absence of intermediate updates is source-bound, not proven by endpoints alone.
- Learned predictions/hidden vectors are saved source-bound evidence, not new neural inference. Public cache fields, clocks, chosen actions and actual-class configurations are checked.
- All three GRU arms are recurrent within imagined rollouts. Treatment is the separately trained real-assimilation update policy, including its validity gating, not recurrence in isolation.
- Geometry is independently derived in NumPy. Final-angle FK approximates native RK4 cached-body reward, and projected mean-angle distance is not expected distance.
- Native control replay, every nominal planning candidate/selected step, and causal public-observer propagation are independently checked and separately counted.
- Physics uses the same paid CEM256 and geometry objective but nominal zero-disturbance dynamics and supplied simulator knowledge. It is not an optimal stochastic-control oracle.
- All original learned heads remain executed. Geometry/cache/observer/trace work is charged; equal candidate budgets are not equal total compute.
- Conditional paired-case bootstrap intervals fix these three saved fits and training corpus; they do not establish training-seed population uncertainty or multiplicity-corrected confirmation.
- Shared-host batched wall times include recorded evidence work; they are not isolated latency, total FLOPs or cross-machine speed.
- Passing supports these trained persistent update policies over declared reset controls on this task; retained angles, velocity inference and explicit two-observation/action-history alternatives remain unresolved.
- No biological/connectome superiority, calibrated uncertainty, novel architecture or cross-environment robotics generality is established.
