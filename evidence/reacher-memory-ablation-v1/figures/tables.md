# Reacher memory comparison

Completed scored development study. Every fit and reference is retained.

Prespecified continuation: **PASS** (25/25 checks).

## Native control costs

Lower is better. Three-fit spread is descriptive, not a confidence interval.

| Panel | Family | Mean of 3 fits | Smallest fit mean | Largest fit mean |
| --- | --- | --- | --- | --- |
| full | Persistent GRU | 8.0852 | 7.53238 | 8.75574 |
| full | Current-packet GRU | 8.72656 | 8.02363 | 9.19347 |
| full | 3-packet GRU | 8.57286 | 7.99404 | 9.44029 |
| full | Packet MLP | 8.05829 | 7.93331 | 8.16575 |
| ordinary | Persistent GRU | 8.09429 | 7.5773 | 8.7864 |
| ordinary | Current-packet GRU | 8.9208 | 8.08453 | 9.54661 |
| ordinary | 3-packet GRU | 8.69272 | 8.06193 | 9.48537 |
| ordinary | Packet MLP | 8.1807 | 8.1128 | 8.27136 |
| shift | Persistent GRU | 8.22819 | 7.65422 | 9.03846 |
| shift | Current-packet GRU | 9.04191 | 8.16572 | 9.76013 |
| shift | 3-packet GRU | 8.80097 | 8.18303 | 9.51614 |
| shift | Packet MLP | 8.37259 | 8.23929 | 8.5148 |

## Every control fit

Decision milliseconds are per case after batch amortization; state bytes are payload per case at one real root.

| Panel | Fit | Native cost | Decision ms/case | Whole-row ms/case/decision | Root state bytes | Parameters | Full row s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full | residual_gru-pair0 | 7.53238 | 2.02759 | 2.09513 | 288 | 36805 | 6.7044 |
| full | residual_gru-pair1 | 8.75574 | 1.98803 | 2.05568 | 288 | 36805 | 6.57817 |
| full | residual_gru-pair2 | 7.96749 | 1.99901 | 2.06424 | 288 | 36805 | 6.60558 |
| full | current_gru-pair0 | 9.19347 | 2.00373 | 2.08154 | 288 | 36805 | 6.66092 |
| full | current_gru-pair1 | 8.96258 | 1.99987 | 2.06571 | 288 | 36805 | 6.61028 |
| full | current_gru-pair2 | 8.02363 | 1.99896 | 2.06505 | 288 | 36805 | 6.60817 |
| full | bounded_gru-pair0 | 8.28426 | 2.41612 | 2.48397 | 419 | 36805 | 7.94871 |
| full | bounded_gru-pair1 | 9.44029 | 2.42845 | 2.49556 | 419 | 36805 | 7.98579 |
| full | bounded_gru-pair2 | 7.99404 | 2.45116 | 2.52497 | 419 | 36805 | 8.07991 |
| full | packet_mlp-pair0 | 8.16575 | 2.38888 | 2.44278 | 32 | 36599 | 7.81691 |
| full | packet_mlp-pair1 | 8.0758 | 2.37949 | 2.4356 | 32 | 36599 | 7.79393 |
| full | packet_mlp-pair2 | 7.93331 | 2.3784 | 2.43249 | 32 | 36599 | 7.78397 |
| ordinary | residual_gru-pair0 | 7.5773 | 2.15377 | 2.22503 | 288 | 36805 | 7.12011 |
| ordinary | residual_gru-pair1 | 8.7864 | 2.1144 | 2.18303 | 288 | 36805 | 6.98571 |
| ordinary | residual_gru-pair2 | 7.91917 | 2.1193 | 2.1875 | 288 | 36805 | 7.00002 |
| ordinary | current_gru-pair0 | 9.54661 | 2.1653 | 2.23276 | 288 | 36805 | 7.14482 |
| ordinary | current_gru-pair1 | 9.13127 | 2.12136 | 2.18842 | 288 | 36805 | 7.00293 |
| ordinary | current_gru-pair2 | 8.08453 | 2.16396 | 2.2496 | 288 | 36805 | 7.19873 |
| ordinary | bounded_gru-pair0 | 8.53086 | 2.59406 | 2.66561 | 419 | 36805 | 8.52995 |
| ordinary | bounded_gru-pair1 | 9.48537 | 2.61629 | 2.6886 | 419 | 36805 | 8.6035 |
| ordinary | bounded_gru-pair2 | 8.06193 | 2.61211 | 2.68311 | 419 | 36805 | 8.58594 |
| ordinary | packet_mlp-pair0 | 8.1128 | 2.55761 | 2.61438 | 32 | 36599 | 8.36602 |
| ordinary | packet_mlp-pair1 | 8.27136 | 2.59233 | 2.65168 | 32 | 36599 | 8.48538 |
| ordinary | packet_mlp-pair2 | 8.15794 | 2.5694 | 2.62625 | 32 | 36599 | 8.404 |
| shift | residual_gru-pair0 | 7.65422 | 2.20336 | 2.27832 | 288 | 36805 | 7.29062 |
| shift | residual_gru-pair1 | 9.03846 | 2.14 | 2.20898 | 288 | 36805 | 7.06872 |
| shift | residual_gru-pair2 | 7.9919 | 2.13599 | 2.20589 | 288 | 36805 | 7.05885 |
| shift | current_gru-pair0 | 9.76013 | 2.19228 | 2.25974 | 288 | 36805 | 7.23116 |
| shift | current_gru-pair1 | 9.19988 | 2.1372 | 2.20231 | 288 | 36805 | 7.04739 |
| shift | current_gru-pair2 | 8.16572 | 2.14404 | 2.20959 | 288 | 36805 | 7.0707 |
| shift | bounded_gru-pair0 | 8.70375 | 2.6241 | 2.69483 | 419 | 36805 | 8.62347 |
| shift | bounded_gru-pair1 | 9.51614 | 2.59517 | 2.66691 | 419 | 36805 | 8.53412 |
| shift | bounded_gru-pair2 | 8.18303 | 2.61859 | 2.68896 | 419 | 36805 | 8.60467 |
| shift | packet_mlp-pair0 | 8.23929 | 2.56171 | 2.61917 | 32 | 36599 | 8.38133 |
| shift | packet_mlp-pair1 | 8.5148 | 2.54053 | 2.59768 | 32 | 36599 | 8.31259 |
| shift | packet_mlp-pair2 | 8.36369 | 2.57534 | 2.63396 | 32 | 36599 | 8.42868 |

## Every reference

Physics references use 64 proposals; learned controls use 256. Public kinematics has supplied physics and public observations; its result is descriptive only.

| Panel | Reference | Native cost | Decision ms/case | Whole-row ms/case/decision | Full row s |
| --- | --- | --- | --- | --- | --- |
| full | Known-state physics | 7.69909 | 7.75504 | 7.82009 | 25.0243 |
| full | Particle physics | 7.80201 | 8.37383 | 8.56558 | 27.4099 |
| full | Zero action | 11.3728 | 1.45174e-05 | 0.0487602 | 0.156033 |
| full | Uniform action | 42.7737 | 5.05601e-05 | 0.0488963 | 0.156468 |
| full | Public kinematics | 7.7072 | 8.05946 | 8.13584 | 26.0347 |
| ordinary | Known-state physics | 7.69909 | 8.22953 | 8.29623 | 26.5479 |
| ordinary | Particle physics | 7.84979 | 9.07418 | 9.25456 | 29.6146 |
| ordinary | Zero action | 11.3728 | 1.45972e-05 | 0.0549486 | 0.175836 |
| ordinary | Uniform action | 42.7737 | 4.6836e-05 | 0.0525881 | 0.168282 |
| ordinary | Public kinematics | 7.74679 | 8.53145 | 8.61131 | 27.5562 |
| shift | Known-state physics | 7.69909 | 8.17805 | 8.24497 | 26.3839 |
| shift | Particle physics | 7.91254 | 8.79956 | 8.979 | 28.7328 |
| shift | Zero action | 11.3728 | 1.47011e-05 | 0.0506627 | 0.162121 |
| shift | Uniform action | 42.7737 | 5.03894e-05 | 0.0524918 | 0.167974 |
| shift | Public kinematics | 7.82562 | 8.23075 | 8.3109 | 26.5949 |

## Native cost versus full control-row time

See utility-vs-cost.png. All 36 learned fit/panel points and 15 reference points are shown, with a logarithmic time axis and no selected frontier.

1000 * recorded control row wall seconds / (cases * real decisions); includes row setup, native steps and saved traces. Global validation/hashing and training are separate.

Times are shared-host and amortized, not isolated latency. The comparison is not matched total compute. Training costs remain in the separate table below.

## Held-out predictions

Endpoint angle MSE uses observed roots and endpoints. Counts for each mask are retained in report.json.

| Fit | Horizon 1 MSE | Horizon 3 MSE | Horizon 7 MSE | Reward MSE | Blackout angle MSE |
| --- | --- | --- | --- | --- | --- |
| residual_gru-pair0 | 0.00216489 | 0.0295206 | 0.117588 | 0.00394721 | 0.0295765 |
| current_gru-pair0 | 0.0126883 | 0.0790418 | 0.178255 | 0.00559283 | 0.35874 |
| bounded_gru-pair0 | 0.0025253 | 0.0409881 | 0.164761 | 0.00485105 | 0.227509 |
| packet_mlp-pair0 | 0.0126489 | 0.0808083 | 0.212897 | 0.00554428 | 0.383808 |
| current_gru-pair1 | 0.0123649 | 0.0779485 | 0.177814 | 0.00533011 | 0.358083 |
| bounded_gru-pair1 | 0.00243187 | 0.0388755 | 0.167456 | 0.00482069 | 0.224872 |
| packet_mlp-pair1 | 0.0125548 | 0.0797304 | 0.218869 | 0.00578957 | 0.380517 |
| residual_gru-pair1 | 0.00157535 | 0.0200408 | 0.108675 | 0.00404101 | 0.0227951 |
| bounded_gru-pair2 | 0.00237061 | 0.0381741 | 0.161298 | 0.00443686 | 0.226532 |
| packet_mlp-pair2 | 0.0125533 | 0.079047 | 0.219598 | 0.00570555 | 0.377188 |
| residual_gru-pair2 | 0.00193002 | 0.0304998 | 0.124387 | 0.00361523 | 0.0333421 |
| current_gru-pair2 | 0.0124751 | 0.0782134 | 0.177701 | 0.00530302 | 0.358181 |

## Training and deployment work

Equal optimizer updates do not imply equal training compute. The bounded-history model rebuilds its state at each real packet.

| Fit | Fit wall s | Training s | Forward s | Backward s | Updates | Parameters |
| --- | --- | --- | --- | --- | --- | --- |
| residual_gru-pair0 | 31.3831 | 31.0625 | 15.2539 | 12.383 | 1152 | 36805 |
| current_gru-pair0 | 33.6355 | 33.3262 | 17.5283 | 12.2765 | 1152 | 36805 |
| bounded_gru-pair0 | 97.8845 | 96.9778 | 59.0289 | 33.9016 | 1152 | 36805 |
| packet_mlp-pair0 | 30.4272 | 30.24 | 17.6159 | 9.2761 | 1152 | 36599 |
| current_gru-pair1 | 33.2136 | 32.8968 | 17.2959 | 12.1718 | 1152 | 36805 |
| bounded_gru-pair1 | 89.9932 | 89.1533 | 54.4463 | 30.8546 | 1152 | 36805 |
| packet_mlp-pair1 | 38.2224 | 38.0111 | 21.9321 | 12.0874 | 1152 | 36599 |
| residual_gru-pair1 | 32.6644 | 32.3692 | 15.817 | 13.1666 | 1152 | 36805 |
| bounded_gru-pair2 | 79.7921 | 79.0945 | 48.4165 | 27.3635 | 1152 | 36805 |
| packet_mlp-pair2 | 28.3058 | 28.1524 | 16.5181 | 8.66284 | 1152 | 36599 |
| residual_gru-pair2 | 28.6598 | 28.406 | 14.0763 | 11.3005 | 1152 | 36805 |
| current_gru-pair2 | 30.3672 | 30.1193 | 16.0004 | 11.089 | 1152 | 36805 |

| Recorded study cost | Seconds |
| --- | --- |
| training_preparation_seconds | 0.0346476 |
| fit_wall_seconds | 554.549 |
| restore_wall_seconds | 0.224923 |
| prediction_collection_seconds | 0.459262 |
| prediction_model_seconds | 1.01702 |
| innovation_generation_and_storage_seconds | 1.92797 |
| control_row_wall_seconds | 519.142 |
| control_setup_seconds | 3.4741 |
| control_decision_seconds | 507.328 |
| control_native_step_seconds | 5.26901 |
| new_execution_wall_seconds | 1083.97 |
| cumulative_attempt_wall_seconds | 3557.75 |
| audit_validation_wall_seconds | 30.9662 |

Whole execution includes validation, copying, all fitting/restoration, collection, planning, native stepping, traces and final hashes. Timings are nested and not all additive. Shared-host batch throughput, not isolated latency. Audits/publication are separate.

Actual operation counts, search counts, tensor sizes, prediction costs, setup/native costs and prior costs are retained in report.json.

## All 25 continuation checks

| Check | Left | Operator | Threshold | Result |
| --- | --- | --- | --- | --- |
| ordinary: persistent mean improves current by 5% | 8.09429 | <= | 8.47476 | PASS |
| ordinary/pair0: persistent strictly beats current | 7.5773 | < | 9.54661 | PASS |
| ordinary/pair1: persistent strictly beats current | 8.7864 | < | 9.13127 | PASS |
| ordinary/pair2: persistent strictly beats current | 7.91917 | < | 8.08453 | PASS |
| ordinary: persistent mean nonworse than MLP | 8.09429 | <= | 8.1807 | PASS |
| ordinary/pair0: persistent beats zero by 10% | 7.5773 | <= | 10.2355 | PASS |
| ordinary/pair1: persistent beats zero by 10% | 8.7864 | <= | 10.2355 | PASS |
| ordinary/pair2: persistent beats zero by 10% | 7.91917 | <= | 10.2355 | PASS |
| ordinary/known_state: physics beats zero by 10% | 7.69909 | <= | 10.2355 | PASS |
| ordinary/particle: physics beats zero by 10% | 7.84979 | <= | 10.2355 | PASS |
| shift: persistent mean improves current by 5% | 8.22819 | <= | 8.58981 | PASS |
| shift/pair0: persistent strictly beats current | 7.65422 | < | 9.76013 | PASS |
| shift/pair1: persistent strictly beats current | 9.03846 | < | 9.19988 | PASS |
| shift/pair2: persistent strictly beats current | 7.9919 | < | 8.16572 | PASS |
| shift: persistent mean nonworse than MLP | 8.22819 | <= | 8.37259 | PASS |
| shift/pair0: persistent beats zero by 10% | 7.65422 | <= | 10.2355 | PASS |
| shift/pair1: persistent beats zero by 10% | 9.03846 | <= | 10.2355 | PASS |
| shift/pair2: persistent beats zero by 10% | 7.9919 | <= | 10.2355 | PASS |
| shift/known_state: physics beats zero by 10% | 7.69909 | <= | 10.2355 | PASS |
| shift/particle: physics beats zero by 10% | 7.91254 | <= | 10.2355 | PASS |
| shift: persistent mean improves bounded by 3% | 8.22819 | <= | 8.53695 | PASS |
| shift/pair0: persistent nonworse than bounded | 7.65422 | <= | 8.70375 | PASS |
| shift/pair1: persistent nonworse than bounded | 9.03846 | <= | 9.51614 | PASS |
| shift/pair2: persistent nonworse than bounded | 7.9919 | <= | 8.18303 | PASS |
| full: persistent mean degrades current at most 2% | 8.0852 | <= | 8.90109 | PASS |

## Limits

- Saved final tensors, Adam state, update cursor and log chain are authenticated; neural training and optimizer transitions are not independently rerun.
- Actual model class and history semantics bind to configuration and source even when GRU state-dictionary shapes match.
- Public history buffers, missing-observation resets and phase transitions are checked; learned hidden values and neural predictions are not recomputed.
- Native replay validates actions/outcomes. CEM reconstructs all proposals and earliest-best decisions from saved scores without learned-model calls.
- Public kinematics is a supplied-physics reference with zero initial velocity and interval-average wrapped velocities; neither learned nor privileged true-state control.
- Equal optimizer updates and paired GRU initialization are not equal training compute. MLP width, actual operations and state payload bytes are reported separately.
- All12 fits and51 control rows are retained. Bootstrap intervals condition on three paired fits and one shared training corpus.
- This single-environment architecture comparison cannot establish biological, JEPA or general robotics novelty.
- A persistent-history advantage could reflect retention of last visible angles or velocity inference; this study does not separate those mechanisms. Public kinematics supplies physics and is not a hold-last-angle ablation.
- Native cost is lower-is-better; all family means weight the three fits equally, without selecting a winner.
- Learned controls use CEM256; supplied-physics references use 64 proposals. These are not matched-total-compute comparisons.
- Decision times are shared-host batch-amortized throughput, not isolated single-case latency. Setup, native steps and full row time are separate.
- The utility-versus-cost plot uses whole recorded control row time, including setup, native steps and traces, amortized per case and decision. Training remains separate; neither total compute nor hardware-isolated latency is matched.
- State sizes count explicit tensor payloads; they are not peak resident or accelerator memory. Counts and affine MAC estimates are not measured FLOPs.
- Prediction horizons 1/3/7 use observed root/endpoint masks. Hidden blackout angles are labels for audit diagnostics only.

## Artifact bindings

- plan_sha256: `05f09ee5425190f5d652aedb10af1641037035d85e906d04d86baf33552a5786`
- audit_receipt_sha256: `2589383326a2312d7f738821da7fd11f82b9bee1b7168aba2e7f44aaa4946d5d`
- audit_summary_sha256: `f12d56c9dbc2571515bfaa099ca58f3b3dc92e518e82219aebbf3c668af3ff0e`
- execution_completed_sha256: `d63866e590799b46e88461197aa074b641f2a35c795bbf30e7cb04cba4db9ee1`
- execution_member_count: `4087`
- frozen_source_count: `58`
