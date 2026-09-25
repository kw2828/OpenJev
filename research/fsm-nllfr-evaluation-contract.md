# Proposed FIT-only NL-LFR continuation contract

**Source review only; unregistered and unrun.** This document proposes the next reference check after the [audited BLA comparison](fsm-author-bla-results.md). It neither authorizes a fit nor changes the fixed residual candidate. No measurement arrays, saved weights or model calls were used to write it. The 100/200 mV DEV realizations are already exposed; all 300 mV and official-test measurements remain closed.

The useful next question is whether a faithful, FIT-restricted author nonlinear model, with a qualified causal initializer, removes the residual's advantage. A poor periodic-model result with unreliable state estimation would not answer that question.

## Evidence and fixed comparison

The completed author BLA adaptation scored **0.09573148182781799** mean standardized RMSE. The fixed tanh-feedback candidate scored **0.04290116627954671** and the strongest old control, tanh output-only, **0.05522223941878766**. All four continuation checks passed; adding BLA did not change the strongest control. These are exposed-development results, not an untouched author-baseline reproduction. The BLA result alone does not identify whether its weakness came from its dynamics, fitting objective or short-context state estimation.

Keep `tanh_feedback-lr0.0003`, seeds 9201/9202/9203, byte-identical. Retain every complete old control in the strongest-control comparison, not just BLA. Use the same twelve DEV records, separate periods, C100/H128, starts `0,256,...,7936`, common FIT scoring scales, channel metrics, equal-record then equal-seed reduction, and paired-seed rule already defined in the [evaluation contract](fsm-author-evaluation-contract.md). Do not select a replacement candidate.

Evidence identities:

- [BLA registration](fsm-author-bla-registration.json): SHA256 `03e01bf97630d4555338fbaa63ca15f749fd96df64e268940bbc8511d302c0a4`, pre-fit commit `5bad1abd`.
- Original independent BLA audit: SHA256 `9854b3e92ae8f333598a4204e4eb92b8447a07952cc77137ea9bd24c7e363022`; 384 new request replays and 300 old forecast banks rescored.
- [Linear-controls registration](fsm-linear-controls-registration.json): SHA256 `996e492b7db79754fec9ac79239563431076673fe378ca696d240ca07d53c364`. Its pinned candidate checkpoint identities and forecast banks remain authoritative.

## What the author recipe actually trains

The pinned [FSM notebook](https://github.com/merijnfloren/fsm-benchmark-data/blob/539a12fef384b086a8562b500498b2fa3899ef70/baseline_results/fit_NonlinearLFR.ipynb) fits a pooled BLA28, connects a 16-input/8-output static ReLU network with two width-64 hidden layers, then runs unweighted BFGS with `rtol=1e-3`, `atol=1e-5`, at most 10,000 iterations. It uses all three amplitudes and their complete training realizations. The adaptation must instead supply only the twelve admitted 100/200 mV FIT records. Do not execute its eager benchmark loader or use supplied models. Reuse our already fitted, pinned pooled BLA as the initialization; do not silently refit or choose another BLA.

The author [constructor](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/static/_nonlin_funcs.py) defaults to seed 42. Set it explicitly. The resulting network has 5,768 parameters; the eight dynamic matrices contribute 1,705, giving **7,473 trainable parameters**. Verify these counts against the exported array roster before fitting. Original BLA and normalization are retained for training provenance but are not trainable.

The [nonlinear optimizer](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_nonlin_lfr.py#L538-L585) jointly updates the dynamic and network arrays. It simulates the period-averaged normalized FIT signals, prepending 820 samples at N=8192. Initial states come from the original BLA's periodic trajectory. Its loss uses **all 4,097 RFFT bins**, including DC and Nyquist, rather than only the 3,839 excited BLA lines. Disabling frequency weighting does not make that loss identical to our per-request H128 MSE. Preserve this objective, including its endpoint weighting, in the first author-method adaptation. The neural notebook does not invoke the separate latent-signal inference-and-learning routine described in the [2025 paper](https://arxiv.org/html/2503.14409v2).

Periodic tail inputs are legitimate within the complete FIT periods used for parameter estimation. They are forbidden when initializing a DEV request. At inference, neither `offset=1000` from the notebook's evaluation nor the optimizer's periodic BLA state is available. This is an intentional task adaptation, not a finding that the original evaluation leaks under its own periodic task.

The [model source](https://github.com/merijnfloren/freq-statespace/blob/a79e8c567b018a6c9462528fc1e10b77fd19b3e2/src/freq_statespace/_model_structures.py#L394-L449) implements:

```text
z_j = Cz x_j + Dzu u_j
w_j = NN(z_j)
yhat_j = Cy x_j + Dyu u_j + Dyw w_j
x_(j+1) = A x_j + Bu u_j + Bw w_j
```

There is no instantaneous feedback from w into z. Output is emitted before the state update. A stable A alone does not guarantee stable nonlinear dynamics: the local transition derivative also contains `Bw J_NN Cz`. No pole clipping, activation substitution or stability repair belongs in this reference fit.

## One causal initializer, not DEV-selected alternatives

Use normalized observed pairs `y[s+1:s+100]` and `u[s+1:s+100]`. Omit `y[s]` because its matching input is unavailable. Estimate the state **at s+1**, then run all 99 nonlinear transitions to obtain the state at s+100. Only then consume the first forecast input `u[s+100]`. Do not accidentally use the BLA adapter's already-propagated end-of-context state as the optimizer's starting-time state.

The proposed warmstart is the minimum-norm observability solution using the **final NL-LFR linear submodel** `A,Bu,Cy,Dyu`, with the existing fixed `rcond=1e-12`. It is an approximation that initially omits w, not an exact state estimate. The retained original `_bla` describes the training initialization heuristic; it is not the chosen request initializer. A rank-deficient final linear submodel is reported, not repaired or used to select another seed.

With model weights frozen, minimize the sum of squared errors of the full nonlinear 99-step context simulation in the author's normalized output coordinates. The implementation proposal is a bounded analytic-Jacobian damped Gauss-Newton solve: 16 outer iterations, at most eight declared trial scales per iteration, one starting state, no restarts. Use the exact [causal solver contract](fsm-nllfr-causal-contract.md): column-scaled augmented least squares with damping 1e-3 and rcond 1e-12; trial scales 1 through 1/128; Armijo coefficient 1e-4 plus strict loss reduction; scaled-gradient tolerance 1e-8. Zero Jacobian, stall and iteration cap are explicit statuses. Do not silently replace this with a finite-difference optimizer whose uncounted residual evaluations change the work budget.

For state sensitivity S, start `S_0=I`. For each known input, the output Jacobian is `(Cy+Dyw J_NN Cz) S_j`; propagate `S_(j+1)=(A+Bw J_NN Cz) S_j`. Declare the ReLU derivative at zero explicitly. Accepted steps must not increase the declared context objective. This establishes only a bounded numerical policy, not convergence to a global state estimate, observability or stability.

Retain initial/final context loss, every accepted/trial step, Jacobian singular values/rank at the declared cutoff, finite-state maxima, stop reason and exact work counts. A capped or stalled state solve must stay visible. No failed solve may quietly fall back to zero state or a BLA forecast. The declared policy returns the last accepted finite iterate, including the seed when no step was accepted. Finite stalled/capped estimates may be scored with their true status; nonfinite seed/current trajectory or failed solve is a failed request. This does not establish convergence.

## Qualification that separates inference from dynamics

All acceptance fixtures below use fabricated parameters and inputs. Freeze their arrays, tolerances and operation budgets before executing them.

1. **Exact recurrence and export.** An independent NumPy loop and the pinned author simulator must agree at fixed float64 `rtol=atol=1e-10`, including nonzero `Dyu`, `Dyw`, state and bias, active/inactive ReLUs, batch one and multiple requests. Check all exported layers, normalizers, shapes and final state. Use explicit author input axes `(H,3,B)`, state `(28,B)` and `offset=None`. Its saved state trajectory is pre-update, so its last row is not the final state.
2. **Linear reduction.** Set `Bw=Dyw=0` while retaining a nontrivial network. Context estimation must reduce to the qualified linear least-squares forecast within `1e-8`, including a rank-deficient case. This catches an incorrect state time, direct-feedthrough subtraction or reshape without requiring full state identifiability.
3. **Known nonlinear state.** Use independently generated, observable, stable fabricated systems with nonzero feedback and direct nonlinear output. Test both exact generating-state rollout and state estimation from observations. Require finite predictions, nonincreasing accepted-step loss, and fixed `1e-6` normalized forecast agreement on the declared well-conditioned recovery fixture. Include a 28-state geometry witness. A failed fixture blocks empirical use; do not weaken its tolerance post hoc.
4. **Derivative and failure checks.** Compare analytic Jacobians with centered differences away from ReLU kinks at `rtol=1e-5, atol=1e-7`; separately test the declared zero derivative. Exercise ill-conditioned/rank-deficient context, rejected trial steps, capped work, nonfinite inputs and overflow. Failure must retain diagnostics without changing weights, clipping states or substituting a different initializer.
5. **Information boundary.** Mutating dropped `y[s]`, unavailable inputs or future target storage cannot alter the inferred state. Changing a future-input suffix cannot alter earlier predictions. Inputs/state must not alias or mutate, and no request cache may retain observations or inferred state.

For the eventual measured fit, keep two sanity checks distinct from primary DEV forecasting. First, independently reconstruct the frozen native FIT spectral objective and compare initial/final values, rather than relying solely on the author's stop flag. Second, retain the nonlinear context loss at the linear warmstart and after the one solve. A large loss reduction confirms that the correction did work; it does not prove that its inferred state will forecast accurately.

A later, separately registered FIT-only prefix holdout could help diagnose low context loss but poor forecasting. It is **deferred from this first continuation**: the immediate qualification uses known-state fabricated nonlinear systems and retains seed/final context diagnostics, without adding an empirical initializer-selection exercise.

## Fairness, practical budget and failure interpretation

The initial continuation should use the exact seed-42 author recipe as a conventional reference, with no DEV search. The existing residual was selected from 36 fits and evaluated across three seeds on these same DEV records. The two procedures therefore do not have matched hyperparameter search, loss, parameter count, initialization or optimization exposure. A win against this one author seed is not evidence of architecture superiority or robustness across author initializations. Report its status as a single FIT-only adaptation.

The nonlinear training path processes 9,012 time steps across six realizations per objective call. A dense float64 inverse-Hessian approximation for 7,473 coordinates alone is about **447 MB**, before gradients, simulated trajectories and compiled workspace. The 20-second BLA fit is not a useful runtime bound for this stage. Qualify the real production dimensions on fabricated arrays, retain compilation/first-call costs, then freeze a native/external wall cap before any measured fit. An incomplete cap is a compute-limit result, not proof of an ineffective baseline. Do not reduce model width or replace BFGS mid-run.

Use the existing isolated, pinned CPU float64 environment and preserve its installed-source identities. Preserve initial/final author archives, explicit numeric arrays, the original BLA identity, native objective trace, raw stopping flag, complete exception/timeout receipts and all request diagnostics. Distinguish small-change stopping from stationarity or optimum claims. Observe every declared DEV record even when another record fails, unless an original fatal process error prevents further work; preserve the resulting incomplete roster.

Request costs must include slicing, normalization, final-linear warmstart, nonlinear state solve, all 128 forecast steps, validation and denormalization. Report initializer and rollout components in addition to full-request latency. Keep JIT compilation outside steady-state timing only if it is separately recorded; do not hide a warm compiled request cache. Count deployed matrices/network, normalizers, state and any retained factorization. If an original BLA copy remains in the deployed object, charge it; if only the final model is exported, document that distinction. Training optimizer memory is separate from inference storage.

If synthetic initialization and FIT diagnostics succeed but the nonlinear reference remains weak, the actionable mismatch is its periodic objective versus our conditional task. A subsequent **separately registered** remedy would refine the same NL-LFR on the common FIT C100/H128 loss with the qualified initializer. Give fresh continuation controls the same additional training exposure, keep the original author-method result visible, and call it task-adapted NL-LFR. Do not retroactively replace the reference, retune on DEV, or claim the periodic method was fairly matched merely because both methods saw the same files.

## Frozen decision and audit checklist

Before the one fit, freeze source/runtime/data identities, original BLA checkpoint, neural seed/recipe, initializer policy, time limits, exact inference roster and diagnostic policy. The reserved members stay outside all decode interfaces. Do not import the supplied author weights or notebook loader.

A completed nonlinear reference enters the existing strongest-control pool. Recompute the same four development continuation rules: candidate mean at least 5% lower; each candidate seed better using paired seeds for stochastic controls and a common deterministic score otherwise; no record seed-mean over 2% worse; both amplitudes improve. The single author seed is the same fixed reference for all candidate seeds. Do not reinterpret it as three matched seeds. Any incomplete reference blocks an affirmative claim about surviving this reference check.

Authenticate original closure before decoding saved arrays. Independently replay all nonlinear forecasts from exported numeric parameters and retained physical contexts, check exact starts/targets against the parent audited banks, recompute metrics and rules, and verify parameter hashes before/after inference. Count state-optimization steps separately from model-training updates. Preserve failed records and diagnostics. Timing remains implementation-specific; a cross-process comparison does not establish a new accuracy/compute frontier.

This continuation is a stronger conventional-control check. It establishes no novel recurrent mechanism, physical-state identification, closed-loop control result, official benchmark record or untouched scenario generalization.

## Source identities checked for this note

Implementation pin: [`freq-statespace` a79e8c567b018a6c9462528fc1e10b77fd19b3e2](https://github.com/merijnfloren/freq-statespace/tree/a79e8c567b018a6c9462528fc1e10b77fd19b3e2). Local source-only hashes:

| Source | SHA256 |
|---|---|
| `_nonlin_lfr.py` | `d16b6dc7f85a68e78ae2c5b1f27081c41763681e13841f2137081ffbe74e464c` |
| `_model_structures.py` | `3dc27b23b049acaa4ed3cb01d9eca452f1156ded7e7f97f390d6cb61c13fcd70` |
| `static/_nonlin_funcs.py` | `622903a00f9c640f4d35e959aff6a2c1f02e3fa1a7fa0f1da9361bc14ce9359f` |
| `_data_manager.py` | `c48e0c0ad948018216007923dde652f926c2cfafae885bb42d9dbc21b45e81b5` |

The [earlier source review](fsm-author-reference-qualification.md) records the distinction from the 2024 Adam recipe and the 2025 inference-and-learning paper. Preserve GPL-3.0-or-later notices for the isolated implementation and the separate CC BY 4.0 measurement attribution.
