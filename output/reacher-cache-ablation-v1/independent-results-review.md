# Independent review of completed cache study

**The execution and audit completed successfully; the scientific continuation gate failed: 15/28 checks passed, 13 failed.** Persistent recurrence did not establish the required advantage beyond explicit last-observation caching. The cached MLP has lower mean cost than persistence on every panel and in all three paired fits on both gap panels. This is a negative result for the prespecified incremental-recurrence claim, not proof that recurrence is universally unnecessary or that all architectures are equivalent.

## Authentication and verification scope

- Plan SHA-256: `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa`.
- Audit receipt SHA-256: `d1a6e486fde8823f5a760af8036e79af4fc57b452bc5fa0c8d184093ce2a1790`.
- Audit summary SHA-256: `621f083f648802e1e133c5f3486fed6f16aed24126e839e360b17be4d1f3a24c`.
- Completed execution SHA-256: `08f5f700d7214d58a26c4b89368da1ecdfe9afb87ce1ff3628713eebdc4e3d5a`.
- Independently recomputed episode costs from all 60 hash-bound native reward arrays, all five family means, all paired-fit signs, all 28 gate comparisons and all 21 descriptive contrast point estimates. Initial integration states and actuator-noise arrays match across all 60 rows. All 70 frozen source hashes remain unchanged.
- Bound execution membership is exact: 5,062 members plus the completed receipt, with no partial/failed artifacts in this completed execution. The audit has no failed receipt. This describes the completed scored attempt, not an assertion that every historical development attempt succeeded.
- Replay coverage reconciles to **235,200** transitions: 38,400 inherited training + 4,800 common prediction + 192,000 control. All 62 authenticated cohort records report maximum absolute replay error **0.0**. This review verified the saved audit evidence and array dimensions; it did not rerun native replay or learned models. The audit also makes no claim to independently retrain neural weights.
- This review made zero model/native calls, new rollouts, fits or random draws. It authenticated 66 numerical/provenance input files plus all 70 frozen source files. Existing bootstrap intervals below are copied from the authenticated audit; point estimates were independently recomputed, with no new bootstrap or gate.

## Means from native reward arrays

Lower total episode cost is better. Family means retain all three fits and all 64 paired cases; values are rounded to 12 decimal places.

| Family/reference | Full sensing | Ordinary gap | Shifted gap |
|---|---:|---:|---:|
| residual_gru | 8.794160068535 | 8.900800161579 | 8.976562801688 |
| encoded_current_gru | 8.662830809749 | 8.687005129750 | 8.738389805467 |
| cached_gru | 8.798788645781 | 8.873916912057 | 8.934364241177 |
| packet_mlp | 8.624482029215 | 8.781255396416 | 8.943854733844 |
| cached_mlp | 8.196577747313 | 8.338330722207 | 8.476692221625 |
| known_state | 8.171782490696 | 8.171782490696 | 8.171782490696 |
| particle | 8.300647938365 | 8.324635041368 | 8.395079826573 |
| zero | 12.401115364103 | 12.401115364103 | 12.401115364103 |
| uniform | 43.437963452001 | 43.437963452001 | 43.437963452001 |
| public_kinematic | 8.133552742942 | 8.161143754888 | 8.250812956154 |

Known-state and particle references use supplied physics; the known-state controller additionally receives native state. The public kinematic reference is descriptive and uses public measurements with nominal physics. Similar family means do not establish equivalence to these references.

## Primary comparisons and fit consistency

Percent excess is `100*(persistent/comparator - 1)`: positive values mean persistence is worse. The frozen gap requirement was at least 3% better in the family mean and nonworse in every paired fit.

| Panel | Comparator | Persistent cost excess | Paired differences, pairs 0/1/2 | Persistent nonworse |
|---|---|---:|---|---:|
| full | cached_gru | -0.0526% | -0.501181, +0.556030, -0.068735 | 2/3 |
| full | cached_mlp | +7.2906% | +0.318048, +0.790174, +0.684525 | 0/3 |
| ordinary | cached_gru | +0.3029% | -0.554014, +0.789968, -0.155304 | 2/3 |
| ordinary | cached_mlp | +6.7456% | +0.229091, +0.974749, +0.483567 | 0/3 |
| shift | cached_gru | +0.4723% | -0.514594, +0.945135, -0.303945 | 2/3 |
| shift | cached_mlp | +5.8970% | +0.157153, +1.063679, +0.278779 | 0/3 |

All four 3%-improvement checks fail. Against cached GRU, pair 1 fails the nonworse condition on both gaps; pairs 0 and 2 pass. Against cached MLP, all six gap-fit checks fail. Full-sensing nondegradation passes against cached GRU but fails against cached MLP. The remaining ten competence checks all pass: six persistent-versus-zero and four known-state/particle-versus-zero checks. Thus the failure is not explained by an incompetent reference floor.

Exact failed gate comparisons follow; a check required `left <= allowed maximum`.

| Failed check | Left | Allowed maximum |
|---|---:|---:|
| ordinary: persistent mean improves cached_gru by 3% | 8.900800161579 | 8.607699404695 |
| ordinary/pair1: persistent nonworse than cached_gru | 9.428757118126 | 8.638789158465 |
| ordinary: persistent mean improves cached_mlp by 3% | 8.900800161579 | 8.088180800541 |
| ordinary/pair0: persistent nonworse than cached_mlp | 8.571329151827 | 8.342237732242 |
| ordinary/pair1: persistent nonworse than cached_mlp | 9.428757118126 | 8.454007696181 |
| ordinary/pair2: persistent nonworse than cached_mlp | 8.702314214783 | 8.218746738198 |
| shift: persistent mean improves cached_gru by 3% | 8.976562801688 | 8.666333313942 |
| shift/pair1: persistent nonworse than cached_gru | 9.642725771463 | 8.697591107497 |
| shift: persistent mean improves cached_mlp by 3% | 8.976562801688 | 8.222391454977 |
| shift/pair0: persistent nonworse than cached_mlp | 8.619315710291 | 8.462162431014 |
| shift/pair1: persistent nonworse than cached_mlp | 9.642725771463 | 8.579046498344 |
| shift/pair2: persistent nonworse than cached_mlp | 8.667646923309 | 8.388867735519 |
| full: persistent mean degrades cached_mlp at most 2% | 8.794160068535 | 8.360509302259 |

## Secondary cache-information contrasts

Treatment minus its matching current-only control, negative is better. These were prespecified descriptive contrasts and cannot rescue the failed primary gate. The intervals condition on these three fitted pairs and the shared training corpus.

| Panel | Treatment minus control | Mean difference | Relative change | Paired differences 0/1/2 | Saved conditional 95% interval |
|---|---|---:|---:|---|---|
| full | gru_cache_information | +0.135957836 | +1.5694% | +0.203810, +0.086598, +0.117466 | [+0.040412, +0.233469] |
| full | mlp_cache_information | -0.427904282 | -4.9615% | -0.801668, -0.222100, -0.259945 | [-0.662539, -0.204185] |
| ordinary | gru_cache_information | +0.186911782 | +2.1516% | +0.254715, +0.196136, +0.109884 | [+0.087062, +0.285933] |
| ordinary | mlp_cache_information | -0.442924674 | -5.0440% | -0.868453, -0.180560, -0.279761 | [-0.663438, -0.248352] |
| shift | gru_cache_information | +0.195974436 | +2.2427% | +0.232473, +0.287556, +0.067893 | [+0.095459, +0.297074] |
| shift | mlp_cache_information | -0.467162512 | -5.2233% | -0.901742, -0.213819, -0.285927 | [-0.695673, -0.266159] |

The direction differs by architecture: caching improves the MLP mean on all panels but worsens the reset-and-encode GRU mean. Neither a generic cache benefit nor a generic recurrence benefit follows. Full-sensing differences do not isolate information at a visible root: identical available angles can still yield different fitted weights because blackout-training gradients differ. The historical gated current-GRU arm is not present, so this study does not separately estimate gate removal.

## Every fitted model

| Fit | Full cost | Ordinary cost | Shift cost |
|---|---:|---:|---:|
| residual_gru-pair0 | 8.544581154 | 8.571329152 | 8.619315710 |
| residual_gru-pair1 | 9.085669552 | 9.428757118 | 9.642725771 |
| residual_gru-pair2 | 8.752229500 | 8.702314215 | 8.667646923 |
| encoded_current_gru-pair0 | 8.841952031 | 8.870628164 | 8.901436414 |
| encoded_current_gru-pair1 | 8.443041194 | 8.442653122 | 8.410034763 |
| encoded_current_gru-pair2 | 8.703499205 | 8.747734103 | 8.903698239 |
| cached_gru-pair0 | 9.045761710 | 9.125343551 | 9.133909901 |
| cached_gru-pair1 | 8.529639388 | 8.638789158 | 8.697591107 |
| cached_gru-pair2 | 8.820964839 | 8.857618027 | 8.971591715 |
| packet_mlp-pair0 | 9.028201057 | 9.210690600 | 9.363904012 |
| packet_mlp-pair1 | 8.517595419 | 8.634567773 | 8.792865117 |
| packet_mlp-pair2 | 8.327649611 | 8.498507817 | 8.674795073 |
| cached_mlp-pair0 | 8.226533110 | 8.342237732 | 8.462162431 |
| cached_mlp-pair1 | 8.295495701 | 8.454007696 | 8.579046498 |
| cached_mlp-pair2 | 8.067704432 | 8.218746738 | 8.388867736 |

## Next decision

**Do not advance the biological/recurrent-architecture branch on this result.** The condition for escalating to two-observation motion features or a new latent mechanism was not met. Preserve the failed gate and report the cached MLP as the strongest observed learned family here, without promoting its descriptive advantage to a broadly confirmed winner.

The prior reward diagnostic remains useful as a separate, explicitly exploratory explanation test. Its large angle-MSE gains translated to smaller reward-prediction gains and little utility separation, with most remaining reward discrepancy in the learned residual. That does not prove reward modeling is the bottleneck, and its numerical gains cannot be transferred to this new cohort.

If one small follow-up is warranted, prioritize the already drafted **zero-fit frozen-transition reward intervention**: original learned reward versus explicitly approximate angle-decoded geometry scoring, first on common-root native rankings and then on fresh paired closed-loop cases. Retain all three fits per chosen family, unchanged CEM budget and actuator penalty, and include the cached MLP as a declared comparator if adapting the design to this completed study. Do not select a favorable checkpoint or tune a hybrid to this result. This is a new diagnostic hypothesis, not continuation approval under the failed memory gate. A separately frozen noninferiority/efficiency comparison is the alternative if simplifying deployment is the objective.

The geometry score must retain the prior documented RK4 cached-position timing caveat and decoder-exploitation checks. No new mechanism, velocity inference, connectome advantage or ICLR-level generalization has been demonstrated.

## Cost boundary

Execution wall time was 1234.341376 s, including 591.907669 s across all 15 fits; the independent audit was another 36.718210 s. Components are nested, not additive. All 17,280 optimizer updates were retained. Equal updates and approximately matched parameter counts do not make computation equal; shared-host, batched timings are not isolated latency.
