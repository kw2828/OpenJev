# Frozen-transition geometry-scoring study

Continuation: **PASS** (25/25).

All six inherited fits and all 51 control rows are retained. Zero new training updates. The parent cache gate remains failed (15/28).

| Panel | Policy | Mean native cost | Whole-row wall (s) | Amortized wall (ms/action/case) |
|---|---|---:|---:|---:|
| full | residual_gru-pair0--learned | 8.322402 | 14.495 | 4.529757 |
| full | residual_gru-pair1--learned | 8.843054 | 14.401 | 4.500173 |
| full | residual_gru-pair2--learned | 8.747960 | 14.419 | 4.505822 |
| full | residual_gru-pair0--geometry | 5.562172 | 21.793 | 6.810400 |
| full | residual_gru-pair1--geometry | 5.830230 | 21.967 | 6.864748 |
| full | residual_gru-pair2--geometry | 5.089814 | 22.091 | 6.903436 |
| full | cached_mlp-pair0--learned | 8.197920 | 16.107 | 5.033509 |
| full | cached_mlp-pair1--learned | 8.071525 | 16.318 | 5.099477 |
| full | cached_mlp-pair2--learned | 7.981959 | 16.338 | 5.105509 |
| full | cached_mlp-pair0--geometry | 7.177392 | 23.522 | 7.350640 |
| full | cached_mlp-pair1--geometry | 6.860641 | 23.908 | 7.471113 |
| full | cached_mlp-pair2--geometry | 6.861972 | 23.836 | 7.448782 |
| full | known_state | 7.619962 | 27.365 | 8.551633 |
| full | particle | 7.788715 | 30.158 | 9.424315 |
| full | zero | 12.274553 | 0.174 | 0.054382 |
| full | uniform | 42.760681 | 0.167 | 0.052340 |
| full | public_kinematic | 7.609722 | 27.673 | 8.647913 |
| ordinary | residual_gru-pair0--learned | 8.372377 | 15.000 | 4.687424 |
| ordinary | residual_gru-pair1--learned | 9.152351 | 14.561 | 4.550275 |
| ordinary | residual_gru-pair2--learned | 8.689774 | 14.741 | 4.606438 |
| ordinary | residual_gru-pair0--geometry | 5.514401 | 22.568 | 7.052521 |
| ordinary | residual_gru-pair1--geometry | 6.193268 | 22.402 | 7.000599 |
| ordinary | residual_gru-pair2--geometry | 5.146883 | 22.722 | 7.100633 |
| ordinary | cached_mlp-pair0--learned | 8.316235 | 16.831 | 5.259644 |
| ordinary | cached_mlp-pair1--learned | 8.228325 | 16.240 | 5.075154 |
| ordinary | cached_mlp-pair2--learned | 8.101978 | 16.599 | 5.187074 |
| ordinary | cached_mlp-pair0--geometry | 7.485209 | 23.864 | 7.457645 |
| ordinary | cached_mlp-pair1--geometry | 7.183409 | 23.683 | 7.400790 |
| ordinary | cached_mlp-pair2--geometry | 7.098104 | 24.372 | 7.616246 |
| ordinary | known_state | 7.619962 | 27.129 | 8.477961 |
| ordinary | particle | 7.838700 | 29.201 | 9.125358 |
| ordinary | zero | 12.274553 | 0.169 | 0.052815 |
| ordinary | uniform | 42.760681 | 0.170 | 0.053026 |
| ordinary | public_kinematic | 7.638609 | 26.876 | 8.398851 |
| shift | residual_gru-pair0--learned | 8.442902 | 13.943 | 4.357051 |
| shift | residual_gru-pair1--learned | 9.341694 | 14.562 | 4.550645 |
| shift | residual_gru-pair2--learned | 8.664378 | 14.429 | 4.509202 |
| shift | residual_gru-pair0--geometry | 6.117706 | 21.721 | 6.787779 |
| shift | residual_gru-pair1--geometry | 6.894631 | 22.106 | 6.908102 |
| shift | residual_gru-pair2--geometry | 5.851857 | 22.037 | 6.886586 |
| shift | cached_mlp-pair0--learned | 8.379553 | 16.107 | 5.033295 |
| shift | cached_mlp-pair1--learned | 8.340355 | 16.332 | 5.103731 |
| shift | cached_mlp-pair2--learned | 8.195180 | 16.246 | 5.076724 |
| shift | cached_mlp-pair0--geometry | 7.730163 | 24.162 | 7.550740 |
| shift | cached_mlp-pair1--geometry | 7.289433 | 23.833 | 7.447919 |
| shift | cached_mlp-pair2--geometry | 7.352090 | 23.704 | 7.407356 |
| shift | known_state | 7.619962 | 25.264 | 7.894902 |
| shift | particle | 7.899669 | 27.158 | 8.486897 |
| shift | zero | 12.274553 | 0.156 | 0.048812 |
| shift | uniform | 42.760681 | 0.158 | 0.049234 |
| shift | public_kinematic | 7.718255 | 26.860 | 8.393890 |

Timing includes setup, native steps and trace work on a shared host. It is not isolated latency or matched total compute.

| Gate check | Measured cost | Required upper bound | Result |
|---|---:|---:|---|
| ordinary: GRU geometry mean improves learned by 3% | 5.618184 | 8.476022 | PASS |
| ordinary/pair0: GRU geometry nonworse than learned | 5.514401 | 8.372377 | PASS |
| ordinary/pair1: GRU geometry nonworse than learned | 6.193268 | 9.152351 | PASS |
| ordinary/pair2: GRU geometry nonworse than learned | 5.146883 | 8.689774 | PASS |
| ordinary/pair0/learned: GRU beats zero by 10% | 8.372377 | 11.047098 | PASS |
| ordinary/pair1/learned: GRU beats zero by 10% | 9.152351 | 11.047098 | PASS |
| ordinary/pair2/learned: GRU beats zero by 10% | 8.689774 | 11.047098 | PASS |
| ordinary/pair0/geometry: GRU beats zero by 10% | 5.514401 | 11.047098 | PASS |
| ordinary/pair1/geometry: GRU beats zero by 10% | 6.193268 | 11.047098 | PASS |
| ordinary/pair2/geometry: GRU beats zero by 10% | 5.146883 | 11.047098 | PASS |
| ordinary/known_state: physics beats zero by 10% | 7.619962 | 11.047098 | PASS |
| ordinary/particle: physics beats zero by 10% | 7.838700 | 11.047098 | PASS |
| shift: GRU geometry mean improves learned by 3% | 6.288065 | 8.551835 | PASS |
| shift/pair0: GRU geometry nonworse than learned | 6.117706 | 8.442902 | PASS |
| shift/pair1: GRU geometry nonworse than learned | 6.894631 | 9.341694 | PASS |
| shift/pair2: GRU geometry nonworse than learned | 5.851857 | 8.664378 | PASS |
| shift/pair0/learned: GRU beats zero by 10% | 8.442902 | 11.047098 | PASS |
| shift/pair1/learned: GRU beats zero by 10% | 9.341694 | 11.047098 | PASS |
| shift/pair2/learned: GRU beats zero by 10% | 8.664378 | 11.047098 | PASS |
| shift/pair0/geometry: GRU beats zero by 10% | 6.117706 | 11.047098 | PASS |
| shift/pair1/geometry: GRU beats zero by 10% | 6.894631 | 11.047098 | PASS |
| shift/pair2/geometry: GRU beats zero by 10% | 5.851857 | 11.047098 | PASS |
| shift/known_state: physics beats zero by 10% | 7.619962 | 11.047098 | PASS |
| shift/particle: physics beats zero by 10% | 7.899669 | 11.047098 | PASS |
| full: GRU geometry degrades learned at most 2% | 5.494072 | 8.810562 | PASS |

| Paid scope | Wall seconds |
|---|---:|
| restore_wall_seconds | 0.107451 |
| diagnostic_root_wall_seconds | 22.945260 |
| control_row_wall_seconds | 940.637655 |
| innovation_generation_and_storage_seconds | 2.115397 |
| execution_wall_seconds | 982.919834 |
| audit_validation_wall_seconds | 121.522991 |
| Inherited cumulative prior attempts | 4792.092156 |
| Cumulative through this execution | 5775.011990 |

No new training. Inherited training and previous studies are prior costs, not deployment latency.

- Zero-fit score intervention using all six inherited models. The failed parent cache gate (15/28) remains unchanged.
- Geometry approximates native reward. Final-angle forward kinematics is not exact RK4 cached-body reward, and projected angle distance is not expected distance.
- All fit pairs and all 51 rows are retained. Three saved fit pairs do not establish population-level seed uncertainty.
- The 25-check gate tests this scoring intervention. It establishes no architecture, recurrence or biological novelty.
- Diagnostics use exposed prior histories, a finite candidate union and four noise branches. They are descriptive and cannot rescue a failed gate.
- Timing is measured whole-row wall time, including setup, native steps and saved traces, amortized over cases and actions. Shared-host throughput is not isolated latency or matched total compute.
- The four-panel replay is one preselected case, not an aggregate result or a selected winning episode. Saved native qpos is used only for schematic drawing, never as model input.
- This zero-fit intervention preserves all six inherited students and their failed parent 15/28 result; it cannot rescue that earlier claim.
- Before/after observed tensors are equal. No trainer or optimizer is invoked by this audit; absence of intermediate updates is source-bound, not proven by endpoints alone.
- Neural predictions and latent vectors are saved source-bound evidence, not independently recomputed. Cache/public-clock fields and same-bank mode invariance are checked.
- Geometry is independent NumPy forward kinematics with clipped-action expected cost once. Final-angle FK is not exact native RK4 cached-body reward; projected mean angles do not imply expected distance.
- Native replay validates saved actions/outcomes. CEM is reconstructed from all paid proposals and saved scores with exact sequential float32 clipping.
- Diagnostic histories were already exposed in the prior completed cache study. Rank and regret describe only the finite union and four shared noise branches; they are outside the gate.
- Both learned and geometry modes retain original motion/reward heads. Geometry adds arithmetic and recording work; equal candidate counts are not equal compute.
- All six fits, twelve policies and 51 control rows are retained. Conditional episode bootstrap intervals do not capture training-corpus or seed-population uncertainty.
- Timing is shared-host batched throughput including trace work, not isolated per-case latency, total FLOPs or cross-machine speed.
- A positive gate would support this specific approximate scoring intervention, not biological novelty, a new world-model architecture, or general robotics capability.
