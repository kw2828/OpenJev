# Reacher explicit-cache comparison

Completed scored development study. Every fit and reference is retained.

Prespecified continuation: **FAIL** (15/28 checks).

## Native control costs

Lower is better. Three-fit spread is descriptive, not a confidence interval.

| Panel | Family | Mean of 3 fits | Smallest fit mean | Largest fit mean |
| --- | --- | --- | --- | --- |
| full | Persistent GRU | 8.79416 | 8.54458 | 9.08567 |
| full | Encoded-current GRU | 8.66283 | 8.44304 | 8.84195 |
| full | Cached-angle GRU | 8.79879 | 8.52964 | 9.04576 |
| full | Packet MLP | 8.62448 | 8.32765 | 9.0282 |
| full | Cached-angle MLP | 8.19658 | 8.0677 | 8.2955 |
| ordinary | Persistent GRU | 8.9008 | 8.57133 | 9.42876 |
| ordinary | Encoded-current GRU | 8.68701 | 8.44265 | 8.87063 |
| ordinary | Cached-angle GRU | 8.87392 | 8.63879 | 9.12534 |
| ordinary | Packet MLP | 8.78126 | 8.49851 | 9.21069 |
| ordinary | Cached-angle MLP | 8.33833 | 8.21875 | 8.45401 |
| shift | Persistent GRU | 8.97656 | 8.61932 | 9.64273 |
| shift | Encoded-current GRU | 8.73839 | 8.41003 | 8.9037 |
| shift | Cached-angle GRU | 8.93436 | 8.69759 | 9.13391 |
| shift | Packet MLP | 8.94385 | 8.6748 | 9.3639 |
| shift | Cached-angle MLP | 8.47669 | 8.38887 | 8.57905 |

## Every control fit

Decision milliseconds are per case after batch amortization; state bytes are payload per case at one real root.

| Panel | Fit | Native cost | Decision ms/case | Whole-row ms/case/decision | Root state bytes | Parameters | Full row s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | residual_gru-pair0 | 8.54458 | 2.22669 | 2.29987 | 288 | 36805 | 7.35959 |
| full | residual_gru-pair1 | 9.08567 | 2.13965 | 2.20955 | 288 | 36805 | 7.07056 |
| full | residual_gru-pair2 | 8.75223 | 2.11907 | 2.18863 | 288 | 36805 | 7.00363 |
| full | encoded_current_gru-pair0 | 8.84195 | 2.5798 | 2.65289 | 320 | 36805 | 8.48926 |
| full | encoded_current_gru-pair1 | 8.44304 | 2.56691 | 2.6372 | 320 | 36805 | 8.43904 |
| full | encoded_current_gru-pair2 | 8.7035 | 2.53126 | 2.60125 | 320 | 36805 | 8.324 |
| full | cached_gru-pair0 | 9.04576 | 2.58385 | 2.6558 | 336 | 36805 | 8.49855 |
| full | cached_gru-pair1 | 8.52964 | 2.52864 | 2.59944 | 336 | 36805 | 8.3182 |
| full | cached_gru-pair2 | 8.82096 | 2.52333 | 2.59396 | 336 | 36805 | 8.30067 |
| full | packet_mlp-pair0 | 9.0282 | 2.67709 | 2.73639 | 32 | 36599 | 8.75646 |
| full | packet_mlp-pair1 | 8.5176 | 2.57232 | 2.62986 | 32 | 36599 | 8.41555 |
| full | packet_mlp-pair2 | 8.32765 | 2.60472 | 2.66308 | 32 | 36599 | 8.52186 |
| full | cached_mlp-pair0 | 8.22653 | 3.11269 | 3.17326 | 112 | 36599 | 10.1544 |
| full | cached_mlp-pair1 | 8.2955 | 2.97343 | 3.03413 | 112 | 36599 | 9.70921 |
| full | cached_mlp-pair2 | 8.0677 | 2.9779 | 3.03858 | 112 | 36599 | 9.72346 |
| ordinary | residual_gru-pair0 | 8.57133 | 2.18035 | 2.25097 | 288 | 36805 | 7.20312 |
| ordinary | residual_gru-pair1 | 9.42876 | 2.17244 | 2.24284 | 288 | 36805 | 7.17709 |
| ordinary | residual_gru-pair2 | 8.70231 | 2.12078 | 2.19144 | 288 | 36805 | 7.01262 |
| ordinary | encoded_current_gru-pair0 | 8.87063 | 2.57303 | 2.64381 | 320 | 36805 | 8.46019 |
| ordinary | encoded_current_gru-pair1 | 8.44265 | 2.54697 | 2.61832 | 320 | 36805 | 8.37861 |
| ordinary | encoded_current_gru-pair2 | 8.74773 | 2.51194 | 2.58204 | 320 | 36805 | 8.26254 |
| ordinary | cached_gru-pair0 | 9.12534 | 2.59048 | 2.66308 | 336 | 36805 | 8.52186 |
| ordinary | cached_gru-pair1 | 8.63879 | 2.57045 | 2.64266 | 336 | 36805 | 8.45651 |
| ordinary | cached_gru-pair2 | 8.85762 | 2.5026 | 2.57418 | 336 | 36805 | 8.23739 |
| ordinary | packet_mlp-pair0 | 9.21069 | 2.63352 | 2.69561 | 32 | 36599 | 8.62595 |
| ordinary | packet_mlp-pair1 | 8.63457 | 2.62531 | 2.68399 | 32 | 36599 | 8.58876 |
| ordinary | packet_mlp-pair2 | 8.49851 | 2.59197 | 2.65026 | 32 | 36599 | 8.48083 |
| ordinary | cached_mlp-pair0 | 8.34224 | 3.01843 | 3.07977 | 112 | 36599 | 9.85527 |
| ordinary | cached_mlp-pair1 | 8.45401 | 3.05465 | 3.11664 | 112 | 36599 | 9.97324 |
| ordinary | cached_mlp-pair2 | 8.21875 | 2.96747 | 3.0273 | 112 | 36599 | 9.68737 |
| shift | residual_gru-pair0 | 8.61932 | 2.16172 | 2.23112 | 288 | 36805 | 7.13958 |
| shift | residual_gru-pair1 | 9.64273 | 2.20975 | 2.28026 | 288 | 36805 | 7.29683 |
| shift | residual_gru-pair2 | 8.66765 | 2.08618 | 2.15611 | 288 | 36805 | 6.89956 |
| shift | encoded_current_gru-pair0 | 8.90144 | 2.4911 | 2.56037 | 320 | 36805 | 8.19317 |
| shift | encoded_current_gru-pair1 | 8.41003 | 2.54762 | 2.61785 | 320 | 36805 | 8.37712 |
| shift | encoded_current_gru-pair2 | 8.9037 | 2.45217 | 2.52066 | 320 | 36805 | 8.06611 |
| shift | cached_gru-pair0 | 9.13391 | 2.54244 | 2.61338 | 336 | 36805 | 8.36281 |
| shift | cached_gru-pair1 | 8.69759 | 2.57075 | 2.64213 | 336 | 36805 | 8.4548 |
| shift | cached_gru-pair2 | 8.97159 | 2.58359 | 2.65474 | 336 | 36805 | 8.49518 |
| shift | packet_mlp-pair0 | 9.3639 | 2.64989 | 2.70998 | 32 | 36599 | 8.67193 |
| shift | packet_mlp-pair1 | 8.79287 | 2.64841 | 2.70889 | 32 | 36599 | 8.66845 |
| shift | packet_mlp-pair2 | 8.6748 | 2.63545 | 2.70063 | 32 | 36599 | 8.64203 |
| shift | cached_mlp-pair0 | 8.46216 | 3.01826 | 3.07864 | 112 | 36599 | 9.85164 |
| shift | cached_mlp-pair1 | 8.57905 | 3.05008 | 3.11139 | 112 | 36599 | 9.95644 |
| shift | cached_mlp-pair2 | 8.38887 | 2.98693 | 3.04719 | 112 | 36599 | 9.751 |

## Every reference

Physics references use 64 proposals; learned controls use 256. Public kinematics has supplied physics and public observations; its result is descriptive only.

| Panel | Reference | Native cost | Decision ms/case | Whole-row ms/case/decision | Full row s |
| --- | --- | --- | --- | --- | --- |
| full | Known-state physics | 8.17178 | 8.27017 | 8.33633 | 26.6762 |
| full | Particle physics | 8.30065 | 8.9795 | 9.18319 | 29.3862 |
| full | Zero action | 12.4011 | 2.34905e-05 | 0.0695452 | 0.222544 |
| full | Uniform action | 43.438 | 4.68869e-05 | 0.0517129 | 0.165481 |
| full | Public kinematics | 8.13355 | 8.36114 | 8.43827 | 27.0025 |
| ordinary | Known-state physics | 8.17178 | 8.27127 | 8.33782 | 26.681 |
| ordinary | Particle physics | 8.32464 | 8.91022 | 9.11115 | 29.1557 |
| ordinary | Zero action | 12.4011 | 1.98565e-05 | 0.0541267 | 0.173205 |
| ordinary | Uniform action | 43.438 | 4.85555e-05 | 0.0526942 | 0.168621 |
| ordinary | Public kinematics | 8.16114 | 8.33881 | 8.41931 | 26.9418 |
| shift | Known-state physics | 8.17178 | 8.1746 | 8.24319 | 26.3782 |
| shift | Particle physics | 8.39508 | 8.82023 | 9.03982 | 28.9274 |
| shift | Zero action | 12.4011 | 1.47914e-05 | 0.0660689 | 0.211421 |
| shift | Uniform action | 43.438 | 5.36326e-05 | 0.0507187 | 0.1623 |
| shift | Public kinematics | 8.25081 | 8.33031 | 8.41155 | 26.917 |

## Native cost versus full control-row time

See utility-vs-cost.png. All 45 learned fit/panel points and 15 reference points are shown, with a logarithmic time axis and no selected frontier.

1000 * recorded control row wall seconds / (cases * real decisions); includes row setup, native steps and saved traces. Global validation/hashing and training are separate.

Times are shared-host and amortized, not isolated latency. The comparison is not matched total compute. Training costs remain in the separate table below.

## Held-out predictions

Endpoint angle MSE uses observed roots and endpoints. Counts for each mask are retained in report.json.

| Fit | Horizon 1 MSE | Horizon 3 MSE | Horizon 7 MSE | Reward MSE | Blackout angle MSE |
| --- | --- | --- | --- | --- | --- |
| residual_gru-pair0 | 0.00221479 | 0.02943 | 0.120635 | 0.00417234 | 0.0290787 |
| encoded_current_gru-pair0 | 0.0145167 | 0.0886554 | 0.193869 | 0.00562662 | 0.364787 |
| cached_gru-pair0 | 0.0145284 | 0.0888146 | 0.193137 | 0.00466628 | 0.103173 |
| packet_mlp-pair0 | 0.0150565 | 0.0942139 | 0.233378 | 0.00551451 | 0.381549 |
| cached_mlp-pair0 | 0.0148721 | 0.0917767 | 0.231185 | 0.0046061 | 0.10153 |
| encoded_current_gru-pair1 | 0.0146585 | 0.0894648 | 0.196795 | 0.00576449 | 0.366075 |
| cached_gru-pair1 | 0.0146815 | 0.0895063 | 0.195867 | 0.00485202 | 0.105996 |
| packet_mlp-pair1 | 0.0151199 | 0.093109 | 0.241233 | 0.00599456 | 0.378424 |
| cached_mlp-pair1 | 0.0149849 | 0.0926533 | 0.227709 | 0.00517086 | 0.102182 |
| residual_gru-pair1 | 0.00237998 | 0.0397231 | 0.137171 | 0.0045218 | 0.039591 |
| cached_gru-pair2 | 0.0146321 | 0.0886945 | 0.193403 | 0.00449759 | 0.10544 |
| packet_mlp-pair2 | 0.0146139 | 0.0914685 | 0.227007 | 0.00603921 | 0.384348 |
| cached_mlp-pair2 | 0.0145039 | 0.0891619 | 0.234349 | 0.00521165 | 0.101394 |
| residual_gru-pair2 | 0.00176549 | 0.0169973 | 0.0916404 | 0.00390392 | 0.0192855 |
| encoded_current_gru-pair2 | 0.014617 | 0.0886263 | 0.193515 | 0.0054881 | 0.369195 |

## Training and deployment work

Equal optimizer updates do not imply equal training compute. The cache controls reset learned state at every real packet while retaining explicit public measurements. Their bookkeeping and unconditional encoding are charged.

| Fit | Fit wall s | Training s | Forward s | Backward s | Updates | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| residual_gru-pair0 | 28.1621 | 27.9342 | 13.9027 | 11.1544 | 1152 | 36805 |
| encoded_current_gru-pair0 | 40.2441 | 40.0163 | 26.3882 | 10.6973 | 1152 | 36805 |
| cached_gru-pair0 | 42.6135 | 42.3789 | 28.4898 | 10.8752 | 1152 | 36805 |
| packet_mlp-pair0 | 28.208 | 28.0643 | 16.4794 | 8.60891 | 1152 | 36599 |
| cached_mlp-pair0 | 43.2239 | 43.069 | 31.2772 | 8.75038 | 1152 | 36599 |
| encoded_current_gru-pair1 | 42.5033 | 42.2468 | 27.9497 | 11.0591 | 1152 | 36805 |
| cached_gru-pair1 | 46.2896 | 46.0164 | 30.8671 | 11.8371 | 1152 | 36805 |
| packet_mlp-pair1 | 37.5456 | 37.3521 | 21.5325 | 11.9041 | 1152 | 36599 |
| cached_mlp-pair1 | 48.5324 | 48.3553 | 34.9385 | 9.96941 | 1152 | 36599 |
| residual_gru-pair1 | 28.8093 | 28.557 | 14.1178 | 11.3773 | 1152 | 36805 |
| cached_gru-pair2 | 43.7055 | 43.4469 | 29.2526 | 11.0597 | 1152 | 36805 |
| packet_mlp-pair2 | 28.351 | 28.2002 | 16.5383 | 8.67948 | 1152 | 36599 |
| cached_mlp-pair2 | 42.9895 | 42.8356 | 31.1076 | 8.70104 | 1152 | 36599 |
| residual_gru-pair2 | 38.9056 | 38.5859 | 18.6098 | 15.919 | 1152 | 36805 |
| encoded_current_gru-pair2 | 51.8243 | 51.5183 | 33.6485 | 13.9861 | 1152 | 36805 |

| Recorded study cost | Seconds |
| --- | --- |
| training_preparation_seconds | 0.0133023 |
| fit_wall_seconds | 591.908 |
| restore_wall_seconds | 0.312753 |
| prediction_collection_seconds | 0.477738 |
| prediction_model_seconds | 1.58405 |
| innovation_generation_and_storage_seconds | 2.05567 |
| control_row_wall_seconds | 630.002 |
| control_setup_seconds | 4.1588 |
| control_decision_seconds | 615.899 |
| control_native_step_seconds | 6.5001 |
| new_execution_wall_seconds | 1234.34 |
| cumulative_attempt_wall_seconds | 4792.09 |
| audit_validation_wall_seconds | 36.7182 |

Whole execution includes validation, copying, all fitting/restoration, collection, planning, native stepping, traces and final hashes. Timings are nested and not all additive. Shared-host batch throughput, not isolated latency. Audits/publication are separate.

Actual operation counts, search counts, tensor sizes, prediction costs, setup/native costs and prior costs are retained in report.json.

## Every primary and secondary paired comparison

Treatment minus control native cost: negative favors treatment. Dots in paired-comparisons.png retain all three fit pairs. Intervals are the audited case-bootstrap 95% percentiles, conditional on the three saved fits and one corpus. They do not measure fit uncertainty or independent confirmation, and no new bootstrap was drawn.

Persistent-versus-cache contrasts are primary (3% improvement in gap panels, at most 2% degradation under full sensing, plus the complete 28-check gate). Cache-versus-current contrasts are secondary and cannot rescue a failed gate.

| Panel | Purpose | Treatment - control | Pair0 | Pair1 | Pair2 | Mean difference | Conditional case interval |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | primary | Persistent GRU - Cached-angle GRU | -0.501181 | 0.55603 | -0.0687353 | -0.00462858 | -0.249487, 0.2512 |
| full | primary | Persistent GRU - Cached-angle MLP | 0.318048 | 0.790174 | 0.684525 | 0.597582 | 0.246924, 0.964287 |
| full | secondary | Cached-angle GRU - Encoded-current GRU | 0.20381 | 0.0865982 | 0.117466 | 0.135958 | 0.0404124, 0.233469 |
| full | secondary | Cached-angle MLP - Packet MLP | -0.801668 | -0.2221 | -0.259945 | -0.427904 | -0.662539, -0.204185 |
| ordinary | primary | Persistent GRU - Cached-angle GRU | -0.554014 | 0.789968 | -0.155304 | 0.0268832 | -0.251799, 0.319175 |
| ordinary | primary | Persistent GRU - Cached-angle MLP | 0.229091 | 0.974749 | 0.483567 | 0.562469 | 0.170377, 0.970209 |
| ordinary | secondary | Cached-angle GRU - Encoded-current GRU | 0.254715 | 0.196136 | 0.109884 | 0.186912 | 0.0870622, 0.285933 |
| ordinary | secondary | Cached-angle MLP - Packet MLP | -0.868453 | -0.18056 | -0.279761 | -0.442925 | -0.663438, -0.248352 |
| shift | primary | Persistent GRU - Cached-angle GRU | -0.514594 | 0.945135 | -0.303945 | 0.0421986 | -0.266662, 0.364797 |
| shift | primary | Persistent GRU - Cached-angle MLP | 0.157153 | 1.06368 | 0.278779 | 0.499871 | 0.0685728, 0.95254 |
| shift | secondary | Cached-angle GRU - Encoded-current GRU | 0.232473 | 0.287556 | 0.0678935 | 0.195974 | 0.0954591, 0.297074 |
| shift | secondary | Cached-angle MLP - Packet MLP | -0.901742 | -0.213819 | -0.285927 | -0.467163 | -0.695673, -0.266159 |

## All 28 continuation checks

| Check | Left | Operator | Threshold | Result |
| --- | --- | --- | --- | --- |
| ordinary: persistent mean improves cached_gru by 3% | 8.9008 | <= | 8.6077 | FAIL |
| ordinary/pair0: persistent nonworse than cached_gru | 8.57133 | <= | 9.12534 | PASS |
| ordinary/pair1: persistent nonworse than cached_gru | 9.42876 | <= | 8.63879 | FAIL |
| ordinary/pair2: persistent nonworse than cached_gru | 8.70231 | <= | 8.85762 | PASS |
| ordinary: persistent mean improves cached_mlp by 3% | 8.9008 | <= | 8.08818 | FAIL |
| ordinary/pair0: persistent nonworse than cached_mlp | 8.57133 | <= | 8.34224 | FAIL |
| ordinary/pair1: persistent nonworse than cached_mlp | 9.42876 | <= | 8.45401 | FAIL |
| ordinary/pair2: persistent nonworse than cached_mlp | 8.70231 | <= | 8.21875 | FAIL |
| ordinary/pair0: persistent beats zero by 10% | 8.57133 | <= | 11.161 | PASS |
| ordinary/pair1: persistent beats zero by 10% | 9.42876 | <= | 11.161 | PASS |
| ordinary/pair2: persistent beats zero by 10% | 8.70231 | <= | 11.161 | PASS |
| ordinary/known_state: physics beats zero by 10% | 8.17178 | <= | 11.161 | PASS |
| ordinary/particle: physics beats zero by 10% | 8.32464 | <= | 11.161 | PASS |
| shift: persistent mean improves cached_gru by 3% | 8.97656 | <= | 8.66633 | FAIL |
| shift/pair0: persistent nonworse than cached_gru | 8.61932 | <= | 9.13391 | PASS |
| shift/pair1: persistent nonworse than cached_gru | 9.64273 | <= | 8.69759 | FAIL |
| shift/pair2: persistent nonworse than cached_gru | 8.66765 | <= | 8.97159 | PASS |
| shift: persistent mean improves cached_mlp by 3% | 8.97656 | <= | 8.22239 | FAIL |
| shift/pair0: persistent nonworse than cached_mlp | 8.61932 | <= | 8.46216 | FAIL |
| shift/pair1: persistent nonworse than cached_mlp | 9.64273 | <= | 8.57905 | FAIL |
| shift/pair2: persistent nonworse than cached_mlp | 8.66765 | <= | 8.38887 | FAIL |
| shift/pair0: persistent beats zero by 10% | 8.61932 | <= | 11.161 | PASS |
| shift/pair1: persistent beats zero by 10% | 9.64273 | <= | 11.161 | PASS |
| shift/pair2: persistent beats zero by 10% | 8.66765 | <= | 11.161 | PASS |
| shift/known_state: physics beats zero by 10% | 8.17178 | <= | 11.161 | PASS |
| shift/particle: physics beats zero by 10% | 8.39508 | <= | 11.161 | PASS |
| full: persistent mean degrades cached_gru at most 2% | 8.79416 | <= | 8.97476 | PASS |
| full: persistent mean degrades cached_mlp at most 2% | 8.79416 | <= | 8.36051 | FAIL |

## Limits

- Saved final tensors, Adam state, update cursor and log chain are authenticated; neural training and optimizer transitions are not independently rerun.
- Actual model class and history semantics bind to configuration and source even when GRU state-dictionary shapes match.
- Public cache values, actual-measurement indices and real/imagined phase transitions are checked; learned hidden values and neural predictions are not recomputed.
- Native replay validates actions/outcomes. CEM reconstructs all proposals and earliest-best decisions from saved scores without learned-model calls.
- Public kinematics is a supplied-physics reference with zero initial velocity and interval-average wrapped velocities; neither learned nor privileged true-state control.
- Equal optimizer updates and paired GRU initialization are not equal training compute. MLP width, actual operations and state payload bytes are reported separately.
- All 15 fits and 60 control rows are retained. Bootstrap intervals condition on three paired fits and one shared training corpus.
- This single-environment architecture comparison cannot establish biological, JEPA or general robotics novelty.
- A persistent-history advantage beyond both cached controls does not identify velocity inference or a novel mechanism. Failure of superiority does not establish cache equivalence.
- Current-to-cache contrasts within GRU and MLP families are descriptive; neither can rescue a failed primary gate.
- Native cost is lower-is-better; all family means weight the three fits equally, without selecting a winner.
- Learned controls use CEM256; supplied-physics references use 64 proposals. These are not matched-total-compute comparisons.
- Decision times are shared-host batch-amortized throughput, not isolated single-case latency. Setup, native steps and full row time are separate.
- The utility-versus-cost plot uses whole recorded control row time, including setup, native steps and traces, amortized per case and decision. Training remains separate; neither total compute nor hardware-isolated latency is matched.
- State sizes count explicit tensor payloads; they are not peak resident or accelerator memory. Counts and affine MAC estimates are not measured FLOPs.
- Prediction horizons 1/3/7 use observed root/endpoint masks. Hidden blackout angles are labels for audit diagnostics only.

## Artifact bindings

- plan_sha256: `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa`
- audit_receipt_sha256: `d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790`
- audit_summary_sha256: `621f083f648802e1e133c5f3486fed6f16aed24126e839e360b17be4d1f3a24c`
- execution_completed_sha256: `08f5f700d7214d58a26c4b89368da1ecdfe9afb87ce1ff3628713eebdc4e3d5a`
- execution_member_count: `5062`
- frozen_source_count: `70`
