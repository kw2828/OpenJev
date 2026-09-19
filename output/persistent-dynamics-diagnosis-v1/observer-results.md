# Saved tracking observer diagnosis

**No observer indexing, angle-wrapping or implemented finite-difference bug was found.** The adaptive gain estimate nevertheless reaches the upper grid endpoint before the actual gain change. These are descriptive findings from one completed engineering case per role, not qualification or efficacy evidence.

Authentication: run completion `064a0e7157088a6625bfc53afd1022d6d16232dc85dbf6cbde3238c1314bc050`; all six row completions and audits; all **3,624 row payloads / 90,356,361 bytes**. The 116-source scientific set, 136-source pilot set and all 14 engineering sources plus snapshots matched their recorded hashes before and after analysis. No simulator, model, optimizer or RNG was called. Only the three new diagnosis files were written.

Machine-readable results: `observer-results.json`, SHA `e6a6037671cc3f503d432217614524b46e67f22edb5337294107fef70bc8b864`. Reproducible calculation source: `observer_diagnosis.py`, SHA `3ff9b6c04a1452ae1f068abf634b6a3e30371a13c1040516ef0c9270118bb49d` (Ruff clean). The JSON retains every row member hash and all interval calculations.

## State and timing checks

The saved public position/velocity roots and returned-observation estimates match an independent implementation of the intended causal equations **exactly**. Position RMSE is 7.95e-9 to 1.01e-8 radians across the six histories. No transition/joint has an angle-aliasing discrepancy; the largest actual one-step displacement is 0.17592 radians, well below pi. Float32 angle quantization contributes only 1.03e-6 to 1.32e-6 rad/s RMSE to the current velocity calculation.

The existing velocity estimate is zero at startup, a backward difference after one transition, then `(3*latest_delta - previous_delta)/(2*dt)`. It extrapolates to endpoint velocity; the simple backward difference estimates the preceding interval average. These differ under changing acceleration, actuator noise and joint-limit dynamics without implying a code error.

Public-observer velocity RMSE, rad/s, against the saved native endpoint velocity:

| Role | 0:50 | 50:80 | 80:100 | 100:150 | 150:200 | All 200 roots |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | .06668 | .29699 | .10153 | .11361 | .31986 | .21018 |
| Adaptive | .06662 | .29895 | .10418 | .11699 | .20697 | .17243 |
| Frozen | .06796 | .26516 | .10000 | .11396 | .19726 | .16024 |
| Public gain | .06159 | .05016 | .25785 | .10558 | .20945 | .14741 |
| True state | .06311 | .05232 | .41770 | .11381 | .11514 | .15939 |
| Zero | .05879 | .04806 | .09291 | .09075 | .10708 | .08366 |

Windows use decision roots `[start, stop)`, before their corresponding issued commands. The gain switches at action80; targets change at decisions50,100,150. `true_state` maintains a diagnostic public observer but actually plans with native velocity, whose saved-root RMSE is zero. The rows follow different action-dependent trajectories, so their observer RMSE differences are not a paired causal effect of estimator design.

## Alternative causal differences, same saved histories

Every alternative was computed without changing any saved action or rerunning the gain bank. Startup uses zero and lower-order fallback when insufficient history exists.

| Role | Current 3-packet endpoint | Last-interval difference | 4-packet endpoint | 3-interval secant |
|---|---:|---:|---:|---:|
| Nominal | .21018 | .26781 | .23719 | .57917 |
| Adaptive | .17243 | .25928 | .17731 | .60139 |
| Frozen | .16024 | .22746 | .16845 | .50433 |
| Public gain | .14741 | .25511 | .15809 | .64096 |
| True state | .15939 | .22849 | .18181 | .53913 |
| Zero | .08366 | .11617 | .07391 | .22405 |

These are whole-history velocity RMSEs. The current rule has the lowest error among these four fixed choices on all five planned-role histories. The higher-order rule is better only on the zero-command history. This does not prove the current observer is optimal, and no alternative control outcome follows from these offline errors.

## Gain drift and excitation

The true gain is .7 until action80. Adaptive planning gains at decisions0,20,39,40,50,60,79 are **1.0, .9, .65, .65, .75, .85, 1.5**. The first upper-endpoint planning decision is74. The frozen role retains .65 after its40 updates.

Adaptive returned-observation estimates, compared with the gain of the just-completed transition:

| Transition window | Mean inferred gain | Gain MAE | Upper endpoint1.5 | Zero issued commands | Flat rolling banks |
|---|---:|---:|---:|---:|---:|
| 0:50 | .79700 | .12100 | 0/50 | 5/50 | 0/50 |
| 50:80 | 1.08667 | .38667 | 7/30 | 4/30 | 0/30 |
| 80:100 | 1.38000 | .16000 | 11/20 | 8/20 | 0/20 |
| 100:150 | 1.38200 | .14600 | 27/50 | 4/50 | 0/50 |
| 150:200 | 1.32900 | .17500 | 17/50 | 0/50 | 0/50 |

Pre-switch MAE across all80 updates is .220625. The apparent closer estimate after the upward switch cannot by itself establish successful adaptation: the estimate had already drifted upward. The rolling20-transition objective also mixes old/new regimes immediately after the switch by design.

There are21 adaptive updates with exactly zero command and an exactly flat **current-transition** residual bank, but no exactly flat **rolling** bank. Previously informative residuals remain in the window. This is consistent with the implemented prior-retention rule, not a tie-handling defect. Overall, adaptive command squared-energy averages .007356 per transition versus hidden-noise energy .005680. During80:100 they are .001094 versus .004669. The latter is privileged diagnostic context, not information available to identification.

Before the switch, mean rolling second-best-minus-best SSE is 1.56e-7; mean whole-bank SSE range is 2.28e-5. The exact gain-ranking arithmetic and retained20-transition sums are verified. These small but nonzero differences are not confidence estimates. Mean squared endpoint-grid prediction sensitivity falls from2.66e-6 in50:80 to4.09e-7 in80:100. It is only a finite-difference sensitivity statistic; it is not Fisher information or an identifiability certificate.

In50:80, adaptive velocity RMSE rises to .29895 rad/s and six roots exceed the elbow's nominal ±3-radian soft-limit threshold. These events coincide with gain drift, but the records do not isolate whether state error, hidden noise, nonlinear limit dynamics, command selection or their combination causes the drift or control costs.

## Next mechanism implication

Do not treat a different finite-difference convention as an established repair, or the post-switch gain proximity as learned adaptation. The narrow next classical mechanism to test is **joint treatment of public state uncertainty and gain**: for example, a prospectively specified finite-window fit of gain with the window's initial velocity as a nuisance parameter, using only public angles and issued actions. Keep the same planning kernel, nominal/current-gain common-observer controls and true-state reference, and pay for all fitting work. This is a conventional system-identification check, not a new architecture claim.

That proposal still needs its own fixed implementation and engineering qualification. No controller, gain estimate, threshold or scientific protocol was changed here. A fresh two-direction task qualification remains necessary before claiming persistent adaptation or advancing a neural method.
