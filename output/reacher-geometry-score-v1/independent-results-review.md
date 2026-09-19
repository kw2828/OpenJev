# Independent review of the completed geometry-scoring intervention

Reviewed 2026-09-18. This review used completed saved outputs only. It made no learned-model, policy, native-environment, training or new evaluation-stream calls and changed no frozen source. Native replay results below are the authenticated independent auditor's results, not a second replay performed by this review.

**The scoring intervention passed all 25 prospective checks. Replacing the learned scalar reward with angle-derived geometry and the unchanged expected actuator cost materially improves all three persistent-GRU fits. The result supports useful predictions whose control value depends on the scoring interface. It does not yet establish that persistent memory, biological wiring or a new architecture caused the advantage.**

## Authentication and coverage

| Artifact | SHA-256 |
|---|---|
| Frozen plan | `94c90f7585303f4edd88e1b0cb0dc8a488a2e2f07237aae4cfb489ade2068a2b` |
| Completed audit receipt | `9cd5fdde29601ae5227afb9b3725be1fc855d4c7ba72aa7ca4c83248ae8de72e` |
| Bound audit summary | `b8691b3e71707890c1fce1d9bae027b0d67e6f2d86023e3eba0ffa2ec80e0d47` |
| Execution completion | `b73f5da7b8863a733a1a384f0ca6588e5fdb7e8f8a7d88490ac55b1219b86b73` |

Sources are the plan and audit under `evidence/reacher-geometry-score-v1/` and completed execution `runs/reacher-geometry-score-v1/attempt`. I checked the matching execution/member manifests, hashed all 873 execution members read by this review, and verified all 80 frozen source hashes. This is not a claim that this review reread every one of the 11,782 execution members.

The review independently recomputed all 51 control-row costs from saved native reward arrays, all 25 gate inequalities, the 12 published comparison point estimates, and diagnostic metrics for 48 roots by 12 policies. It checked all row timing receipts. Coverage reconciles to 163,200 control transitions plus 165,888 diagnostic transitions, totaling **329,088 native transitions**, with authenticated replay maximum absolute error **0**. A further 9,408 public-observer transitions are separately reported. There were six inherited checkpoints, zero new fits and zero optimizer updates.

## Utility and fit consistency

Mean native episode cost, lower is better. Each learned family/mode includes all three fits and 64 paired cases per panel.

| Controller | Full sensing | Six-step gaps | Ten-step gaps |
|---|---:|---:|---:|
| Persistent GRU, learned score | 8.637805 | 8.738167 | 8.816325 |
| Persistent GRU, geometry | 5.494072 | 5.618184 | 6.288065 |
| Cached MLP, learned score | 8.083801 | 8.215512 | 8.305029 |
| Cached MLP, geometry | 6.966668 | 7.255574 | 7.457229 |
| Known-state physics | 7.619962 | 7.619962 | 7.619962 |
| Particle-filter physics | 7.788715 | 7.838700 | 7.899669 |
| Public-kinematic physics | 7.609722 | 7.638609 | 7.718255 |
| Zero action | 12.274553 | 12.274553 | 12.274553 |
| Uniform action | 42.760681 | 42.760681 | 42.760681 |

Geometry lowers GRU cost by **36.40%, 35.71% and 28.68%** across full/ordinary/shift. Individual-fit gap reductions are 32.33%-40.77% ordinary and 26.20%-32.46% shifted. Cached MLP also improves in every fit and panel, with family reductions of 13.82%, 11.68% and 10.21%.

With both using geometry, GRU costs are **21.14%, 22.57% and 15.68%** lower than cached MLP. Every paired fit favors GRU, including the weakest shifted comparison, a 5.42% reduction. These cross-family comparisons are descriptive secondary evidence; the primary 25-check gate tests the GRU scoring intervention and competence floors. No gate failed: nine score-utility checks, twelve GRU-mode competence checks, and four physics-reference checks passed.

Published paired-case 95% intervals for GRU geometry minus its own learned score are [-3.5486, -2.6545] ordinary and [-2.9290, -2.1186] shifted. Against geometry MLP they are [-1.8937, -1.3727] and [-1.4244, -0.9198]. These intervals condition on the three saved fits and shared training corpus. They do not estimate uncertainty over new training seeds, and the secondary contrasts are not a multiplicity-adjusted confirmation. This review did not generate a new bootstrap.

The full-sensing improvement is at least as large as the gap improvement. Full sensing exposes angles, not velocity, so history can still be useful, but this result cannot be described as a blackout-specific memory effect. The previous cache experiment remains a failed 15/28 study under its original learned-score interface.

## What the exposed diagnostic does and does not explain

These are prior completed cache-study trajectories from GRU pair 0: eight cases per panel, two decision roots each. They are exposed development histories, not the fresh geometry controller's trajectories. Each root has 76 identity slots, deduplicated to 65-76 unique sequences, mean 72. Four shared fresh noise futures evaluate each unique sequence. Candidate outcomes are correlated through those common futures.

The following averages retain all three fits and all 16 roots per panel. Regret is against the best native mean return in this finite union, not a global optimum. Return bias is predicted minus native return.

| Panel and GRU score | Common-bank rank correlation | Selected union regret | Selected raw-return bias |
|---|---:|---:|---:|
| Full, learned | 0.991036 | 0.070730 | +0.442708 |
| Full, geometry | 0.995976 | 0.015896 | -0.004181 |
| Ordinary, learned | 0.991376 | 0.079403 | +0.498435 |
| Ordinary, geometry | 0.995382 | 0.032069 | +0.006903 |
| Shift, learned | 0.992342 | 0.061163 | +0.498819 |
| Shift, geometry | 0.995315 | 0.024837 | +0.014467 |

GRU common-bank raw-return MSE falls from 0.41784 to 0.13269 ordinary and 0.40438 to 0.12400 shifted. MLP union regret also falls, from 0.09784 to 0.07355 ordinary and 0.08497 to 0.05901 shifted. Thus the scorer helps both representations; the recurrent model remains stronger on these roots after the intervention.

Rank correlation was already high. A small aggregate rank change can still affect the chosen action, while broad action-cost variation can dominate this statistic. The large optimism reduction is informative but not a complete causal explanation: a candidate-independent constant bias cannot change fixed-horizon rankings when clipping is inactive. Geometry control clipping is zero in every fit/panel, and learned-score clipping is zero or at most 0.0461% of steps. Neither clipping nor a simple constant reward shift adequately describes the observed intervention. The common-root comparison avoids comparing different on-policy inputs, but its limited exposed support cannot establish global calibration or explain the entire closed-loop gain.

The geometry score is approximate planar forward kinematics from predicted angles. Native RK4 cached-position timing prevents treating it as exact native reward reconstruction. Its empirical success is evidence for this approximation in this task, not numerical equivalence to the simulator's reward.

## Cost and physics comparisons

Execution took **982.920 s**, within its 2,400 s cap. The independent audit took **121.523 s**, within its 600 s cap. Their sum is **1,104.443 s**, excluding engineering and publication work. The reported cumulative execution lineage is 5,775.012 s, including 4,792.092 s of prior execution; it is not total project compute or a sum including all audits. Diagnostic roots took 22.945 s and control rows 940.638 s, nested within execution.

Amortized decision time per case, excluding row setup and native stepping but including loading, assimilation, search, selected advancement, diagnostic copying/hashing and trace storage:

| Family/score | Full ms | Ordinary ms | Shift ms |
|---|---:|---:|---:|
| GRU learned | 4.441 | 4.543 | 4.403 |
| GRU geometry | 6.789 | 6.980 | 6.791 |
| MLP learned | 5.019 | 5.113 | 5.011 |
| MLP geometry | 7.363 | 7.431 | 7.409 |

These are batched, shared-host measurements, not isolated single-case latency or FLOPs. The geometry intervention adds about 54% to GRU decision time on the gap panels. The geometry function itself accounts for about 0.33 ms per case/decision; much of the observed difference includes the larger recorded diagnostics and bookkeeping. All original learned-head work remains present. There were 29,491,200 learned-control candidate evaluations and 314,966,016 imagined model transitions; geometry processed 157,483,008 candidate steps plus selected one-step evaluations. Equal candidate budgets do not mean equal total computation.

**The physics references use RS64, not CEM64.** They select once from the common initial 64 candidates; learned controllers use adaptive CEM256. Physics also rolls nominal noise-free trajectories and charges issued-command square cost, whereas the geometry controller charges expected clipped actuator cost. Particle and public-kinematic references additionally estimate the current state. These are useful operational baselines, but their poorer cost is not evidence that learned dynamics outperform matched-search, matched-objective native MPC or an oracle.

## One next experiment

I recommend the zero-fit **persistent versus encoded-current versus cached-angle GRU comparison under the same geometry scorer**, with cached MLP secondary, as specified in [next-experiment-design.md](next-experiment-design.md). Use all three inherited pairs, fresh paired evaluation cases, identical CEM256 and score arithmetic, complete cost accounting, and separately frozen criteria. Repeat existing geometry arms on the same new cohort rather than joining old means to new controls. A matched-search physics adapter can make the reference comparison interpretable, but its cost must remain explicit.

This takes priority over the frozen-dynamics ranking-head option in [conditional-followups.md](conditional-followups.md): geometry already supplies an effective scalar interface, and an unresolved within-GRU control is available without fitting anything. Persistent, current-packet and cached GRUs share tensor dimensions and paired initializations, though their separately trained final weights and real-assimilation rules differ. A positive comparison would support that trained persistent update policy, not a unique latent-memory mechanism. If a cached GRU closes the gap, retain the successful scorer and stop the stronger memory claim. If persistence survives, the next mechanism test should use two valid observations with elapsed time and issued-action history before introducing biological wiring.

No material numerical reporting blocker was found. The defensible outcome is a successful, bounded scoring intervention with promising recurrent predictions, conditional on this simulator, task family, inherited training corpus and three fits. Architecture novelty, calibrated uncertainty, general robotics capability and superiority to matched native planning remain unestablished.
