# Proposed next direction: jointly learn the prefix observer and transition

**Proposed, unregistered and unrun.** This note recommends one performance experiment, not an automatic norm-precision factorial, a changed result, or a new architecture claim. No models, numerical arrays, training or tests were executed to prepare it.

## Why this is the next useful question

The [audited position-observer study](robot-position-observer-results/audit.json) failed overall, with 3/5 criteria passing despite all six fits completing. Its equal-four-file mean error was 0.6495359, almost the last-two model's 0.6496855; the strongest jointly trained temporal initializer achieved 0.6151412. The candidate also required 5.295 ms per request versus 3.872 ms for last-two. Avoiding the observed training failures did not produce a useful performance advantage. These are saved audited scalars, not new calculations or forecasts.

There is a specific plausible mismatch. The [transition](../src/openjev/research/structured_robot_transition.py) was trained to start from `z31 = [q31, q31-q30]`. Its first six coordinates produce positions, but the remaining six are learned state, not supervised physical velocity. The [observer](../src/openjev/research/robot_position_observer.py) instead rolls this same transition through thirty observed-prefix corrections, while all 590 transition parameters remain frozen. Its gain must discover useful states in coordinates that were not trained jointly with that correction process.

This is an inference from the implementation, not an established explanation of the failure. The gain already receives gradients through the complete frozen prefix and H128 forecast, and it already optimizes forecast MSE. The proposed change is to allow the transition and state estimator to adapt together, not to claim that gradients or forecast supervision were previously absent. The stronger archived temporal model is suggestive but does not isolate freezing: its initialization, optimization path and training history differ.

## One intervention and a small matched comparison

Keep the existing 12-state `dense_mlp` transition, readout `Pz = z[:6]`, normalization and forecast interface. Train its parameters `theta` jointly with the same 72-entry constant gain `K`, initialized at `[I;0]`:

\[
z_1=[q_1,q_1-q_0],\qquad
z_t^- = f_\theta(z_{t-1},u_{t-1}),\qquad
z_t=z_t^-+K(q_t-Pz_t^-),\quad t=2,\ldots,31.
\]

After conditioning, use `z[t+1] = f_theta(z[t],u[t])` without any position correction. Thus `u31` first predicts `q32`; no future position enters the forecast. Minimize the existing normalized H128 position MSE, with no auxiliary prefix loss, teacher-forced forecast, learned readout, new latent dimension or gain clipping. Preserve the current transition parameterization and identical native gradient-clipping/failure policy across every arm; do not combine this experiment with a safer-norm intervention. Implement a new joint-training class because the qualified frozen subclass deliberately rejects trainable backbone parameters; do not change that frozen source or bypass its validation.

| Fresh arm | Trainable parameters | Purpose |
| --- | ---: | --- |
| Joint thirty-correction observer | 662 | Candidate: useful prefix information through a jointly trained observer |
| Joint one-correction observer | 662 | Same cell and gain capacity, with only the most recent innovation |
| Continued last-two model | 590 | Controls for further optimization of the common backbone |
| Continued temporal-affine model | 962 | Strong existing history encoder with the same additional training budget |

For the one-correction control, start at `z30 = [q30,q30-q29]`, predict with `u30`, then set `z31 = f_theta(z30,u30) + K(q31-P f_theta(z30,u30))`, evaluating that prediction once. This is a **three-position local initializer**, using q29, q30, q31 and u30. Both observer arms share the same 590 cell parameters and 72 gain parameters initially, with 12-state storage; they differ in prefix access and recurrence depth. Their initial functions and effective function classes need not match. This comparison does not by itself separate information, optimization conditioning and recovery of the causal preprocessing state.

Use seeds 8101/8102/8103 and the exact same per-seed archived last-two `.001` transition as the common weight warm start for **every arm**, including temporal affine. All four arms start with fresh, empty Adam state; none resumes an archived optimizer. The temporal head starts at zero. Give all four arms the same 4,096 paired updates, batch16, H128 loss, Adam settings and clipping, with one prospectively fixed learning rate `.001`: **12 fresh fits**, no rate search or retry. This is the lower previously used rate, chosen before any new result to bound the development budget, not evidence of optimal tuning or equal search opportunity to historical two-rate-selected models. Each arm inherits the same original backbone-training exposure and receives the same additional window exposure. Charge actual training time; equal update counts do not imply equal compute.

Retain the archived frozen observer at `.001` as the specific freeze/unfreeze reference, and keep the previously selected frozen and jointly trained controls visible as historical performance references. Do not replace the fresh temporal control with a better-selected warm start. Inference timing must include all thirty versus one prefix steps, preparation, normalization, casts and H128 rollout. The two observer arms have 2,888 persistent numeric bytes under the existing accounting; last-two has 2,600 and temporal affine 4,088. Temporary/autograd workspace and training time must remain separate.

## What would justify continuation

Before any run, register a single candidate and the complete reference roster. Keep the existing substantive utility standard: complete finite forecasts and current costs; at least 5% lower equal-four-file mean than the strongest declared fresh or retained control; no file more than 2% worse than the best of fresh one-correction, last-two and temporal-affine controls; full-request latency at most 1.5 times concurrent last-two; and no control dominating error, latency and persistent bytes. Report every file and paired seed, not only the pooled result. All four recordings are exposed development data. Official TEST stays closed, and no success here restores their confirmation status.

Interpret the comparison before adding variants. If joint training helps only relative to the frozen observer, while continued last-two or the one-correction control explains the gain, the result supports ordinary retraining or recent-error correction, not useful long-prefix memory. If the thirty-correction arm beats the one-correction control but loses to temporal affine, it has not established the sought performance improvement. Only a material advantage over the strongest controls at the declared total cost would justify planning new unexposed confirmation. Failure rejects this fixed operating point; it does not authorize an immediate norm, gain or loss search on these same outputs.

## Closest prior art and claim boundary

[Beintema, Toth and Schoukens (2021)](https://proceedings.mlr.press/v144/beintema21a.html) learn a historical-input/output state encoder together with a state-space model using multi-step simulation loss. [Deep Subspace Encoders (2023)](https://arxiv.org/abs/2210.14816) develops this reconstructability-map approach with truncated prediction losses and an explicit treatment of estimation issues. [Forgione and Piga (2021)](https://arxiv.org/abs/1911.13034) jointly estimate neural dynamics and subsequence initial conditions with a consistency regularizer, and discuss limitations of naive fitting objectives.

Joint observer/dynamics training and innovation correction are conventional ideas. The proposed contribution is an informative, parameter-matched prefix-depth comparison on this existing conditional-forecast task, if it works. No Kalman calibration, global observer stability, physical latent-state identification, biological mechanism, closed-loop control benefit or architectural novelty follows from the equations or from a development pass. The existing bounded transition operators do not bound the Jacobian or the observation-correction loop.
