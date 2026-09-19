# Error-gated correction: an empirical question, not established novelty

**Prospective screen; unfrozen and unrun.** The five primary sources below do not justify calling prediction-error-gated fast/slow latent dynamics a new architecture. They already cover its main algebraic ingredients. A narrower OpenJev question remains testable: **does uncertainty-normalized innovation improve recovery after missing observations beyond elapsed-time gating and conventional learned filtering?** This note does not establish that the combination is absent from all prior work.

This extends the existing [architecture note](../../research/reacher-architecture-mechanism-options.md) and [two-paper screen](mechanism-prior-art.md). No current study outcomes, checkpoints or evaluation artifacts were read. No model, training, simulator or scored-RNG call was made, and no frozen source was changed.

## Equation-level overlap in five primary sources

Notation below is harmonized for comparison; equations are not implementation specifications.

| Verified primary version | Relevant equation and overlap | Established scope and limit |
|---|---|---|
| Shaj et al., [ac-RKN, arXiv:2010.10201v2, 5 November 2020](https://arxiv.org/html/2010.10201v2), Sections 3-4 | `mu_next^- = A_t mu_t^+ + b(a_t)`; `P_next^- = A_t P_t^+ A_t^T + Q`; correction is `mu_t^+ = mu_t^- + K_t(w_t - H mu_t^-)`, with uncertainty-dependent Kalman gain. A missing observation skips correction. | Action-conditioned robot dynamics prediction, including missing inputs. Real-time control is explicitly future work. The forward experiment uses RMSE, so learned latent uncertainty is not itself demonstrated output calibration. |
| Revach et al., [KalmanNet, arXiv:2107.10043v3, 11 March 2022](https://arxiv.org/html/2107.10043v3), Eq. 5a and Sections III-A-C | `x_t^+ = x_t^- + K_theta(features_t, memory) e_t`. Features include current innovation and past prediction/correction differences; a recurrent network learns the gain. | State estimation with partially known dynamics, not an action-conditioned MPC result. An error/history-dependent gate multiplying innovation is already a special case of learned gain correction. KalmanNet need not output an explicit covariance. |
| Shaj et al., [MTS3, arXiv:2310.18534v3, 4 December 2023](https://arxiv.org/html/2310.18534v3), Eqs. 5-9 | Fast dynamics use `A z + B a + C mu_slow`; marginal transition noise includes `Q + C P_slow C^T`. A slower latent belief updates every H steps using uncertainty-weighted observation aggregation. | Fast/slow action-conditioned probabilistic prediction is established. Appendix F explicitly leaves hierarchical control untested. Replacing the clock with an error-dependent gate changes the schedule, not the existence of this hierarchy. |
| Gu and Dao, [Mamba, arXiv:2312.00752v2, 31 May 2024](https://arxiv.org/html/2312.00752v2), Section 3.2 and Theorem 1 | Input-dependent discretization yields, in its scalar special case, `h_t = (1-g_t)h_(t-1) + g_t x_t`. More generally, input-dependent transition/read/write parameters select what state retains. | Content-dependent timescales and forgetting are established for sequence models. Feeding innovation into this gate is a feature choice; neither uncertainty calibration nor robotic control follows from the recurrence. |
| Yang et al., [Gated DeltaNet, arXiv:2412.06464v3, 6 March 2025](https://arxiv.org/html/2412.06464v3), Eq. 10 | `M_t = alpha_t M_(t-1) + beta_t (v_t - alpha_t M_(t-1) k_t) k_t^T`, algebraically rearranged from the paper. Memory receives a prediction residual plus adaptive decay. | Error-corrective fast memory and selective forgetting are established; evaluations concern language/retrieval. Its key-value residual is not automatically a physical sensor innovation or a calibrated covariance. |

Putting `fast/slow`, `prediction error`, `uncertainty` and `selectivity` in one diagram therefore does not establish a distinct mechanism. Even `g(e,P) K e` can be absorbed into a learned gain `K'`. A claim would require evidence about a specific constraint or information pathway that these conventional explanations do not account for.

## A precise candidate worth falsifying

Keep an action-conditioned fast state `z`, a slower state `c`, and a predictive uncertainty estimate. Before a **real visible** packet arrives, predict its four public cosine/sine features and innovation covariance:

```text
e_t = public_angles_t - decoder(z_t^-)
S_t = predictive innovation covariance computed before seeing that packet
u_t = e_t^T (S_t + epsilon I)^(-1) e_t
g_t = sigmoid(b + w_u log(1 + u_t) + w_d d_t)
z_t^+ = z_t^- + K_t e_t
c_t^+ = c_t^- + eta g_t tanh(U e_t),  0 < eta < 1
```

This is a proposed conventional filtering variant, not a reproduction of the five papers. The only proposed extra pathway is routing unexpected residual information into the slower state, which conditions later action transitions. `d_t` is elapsed time between the last two actual visible packets, reconstructed from public history. The current visible packet's age is zero, so using that field alone would silently remove the intended elapsed-time control.

During blackouts, propagate the fast dynamics with issued commands and uncertainty; do not manufacture an innovation from a prediction. Hold `c` fixed during missing observations and candidate imagination in the initial design. Each candidate receives private state. A correction is performed once when a real packet arrives, before selecting the next action; it does not advance physical time. Startup has a declared prior and no fabricated previous measurement.

The current [environment](../../src/openjev/research/robotics_reacher.py) draws independent actuator noise per decision, held over its two native substeps. It has no injected persistent hidden gain or damping. Consequently, `c` is a representation of recent motion or model error, not evidence of online identification of changing physics. A large residual may be a random disturbance. Visible angles are clean; high uncertainty after a gap is not a reason to ignore a trustworthy returned measurement.

`S_t` must describe uncertainty in the predicted observation, not the variance of a hidden-state vector copied into another coordinate system. Four cosine/sine coordinates are constrained and correlated; do not assume `u_t` follows chi-square with four independent degrees of freedom. Positive variance is not calibration. If learned jointly, variance inflation can suppress the gate; use a proper predictive objective, fixed numerical bounds and training-only calibration choices, with the same uncertainty head/loss in all relevant ablations.

## Strongest next experiment: correction at equal information and age

Use the existing full/six-gap/ten-gap sensing conditions and iid actuator noise. On a fresh common recorded-history diagnostic, select reacquisition boundaries by schedule alone. Match the public suffix, commands, root and planner innovations; neither errors nor eventual return may select cases. First reconstruct each mechanism arm from the same last-two-valid-observation/action suffix, so older hidden information cannot explain a difference. Treat any persistent-history reference separately.

Within one fixed fast/slow cell, train four paired update variants: **constant gate, elapsed-time-only gate, raw-error gate and uncertainty-normalized-error gate**. Each retains and computes the same state, decoder, uncertainty head and gate inputs; only the specified input pathway is disabled. Calibrate any constant/scale using training development only. This isolates error information, its normalization, and elapsed time without attributing extra parameters or a different training loss to the gate.

Necessary conventional controls are a single-rate recurrent model with matched total state/parameters, a two-rate GRU with ordinary gates, and an ac-RKN-style action-conditioned filter. A learned-gain/KalmanNet-style correction is required before claiming the residual gate beats ordinary learned filtering. The trained bounded-history GRU and public-kinematic supplied-physics control remain useful references. A renamed generic Mamba block is not a substitute for these close controls.

The decisive contrast is normalized-error versus age-only correction **within the same elapsed-gap stratum**, with the raw-error arm testing whether normalization matters. At common roots, measure subsequent public-angle prediction error and candidate-action ranking after reacquisition. Then test native closed-loop episode cost with unchanged geometry scoring and CEM256. Closed-loop branches share resets, schedules and exogenous noise, but may choose different actions and receive different later observations.

Reject the proposed explanation if ordinary gain correction or age-only gating explains the benefit, uncertainty normalization adds nothing, or better prediction/ranking fails to improve native utility. Report every fit, state bytes, all predictor/covariance/gate work, reconstruction and candidate-copy overhead, complete training and controller time. Parameter or transition-count matching is not wall-time matching. Practical margins, multiplicity, freshness and stopping rules still require a separately frozen protocol; none are selected here.

If the running bounded-history comparison eventually explains the earlier memory gain, this remains defensible only as a better inductive bias for the **same finite public information**. It would not support a need for persistent memory, a connectome advantage, calibrated uncertainty, or architectural novelty. No biological wiring intervention is justified by this screen.
