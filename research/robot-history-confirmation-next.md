# Next development question: state correction with frozen dynamics

Design proposal only. No implementation, fitting or measured result is claimed here.

The [reserved-recording confirmation](robot-history-confirmation-results.md) rejects the fixed temporal-feature recipe. Its mean advantage over an equally sized local initializer is only 0.38%, and it is worse on the first recording. This does not establish whether the problem is inadequate history features, transition learning, noise, or a recording-specific regularity.

A bounded follow-up can separate state estimation from transition relearning. For each existing seed, freeze the same last-two model's dense transition and train only an initializer. Compare the existing 372-parameter local and temporal heads against a 72-parameter constant observation-correction gain. Every learned initializer starts from the same frozen transition and receives the same training windows and forecast loss.

For the observer candidate, initialize at observed time 1 using `concat(q1, q1-q0)`. For times 2 through 31, predict once with the previous torque and correct using only the newly available position:

\[
z_t^- = f(z_{t-1},u_{t-1}),\qquad
z_t = z_t^- + K\,(q_t-z_t^-[:6]),\quad K\in\mathbb{R}^{12\times6}.
\]

At time 31, stop corrections. The first forecast torque is `u31`, predicting `q32`. The model must not read future positions or innovations. The bottom six state coordinates are learned increments, not verified physical velocity.

The fixed-gain control `K = [I; I]` is essential: it tests whether repeatedly anchoring predictions to observed positions explains any improvement without learning a gain. Include the unmodified last-two initializer and both refitted affine heads with that same frozen transition. If the fixed-gain or simple affine control explains the benefit, report it. A zero-gain prefix rollout can describe drift, but is not the main quality baseline.

Use original FIT recordings for optimization. All four already exposed DEV/CONFIRM recordings may be declared development evaluation data, with their exposure history explicit. Do not tune on the two confirmation recordings and then call them untouched again. Official TEST remains closed. Before running, freeze the fit budget, learning-rate choices, complete cost comparison, continuation margins and failure handling in a separate protocol.

Charge all thirty prefix transition/correction steps, then all forecast steps, parameter storage and normalization. Freezing dynamics saves parameter updates but does not eliminate the computation or differentiation needed to train the gain. A useful candidate must improve quality against the strongest simple initializer at acceptable full-request cost. No stability, calibration, Bayesian-posterior or novelty guarantee follows from this observer form.

Historical-input state encoders have established [prior art](https://proceedings.mlr.press/v144/beintema21a.html). [KalmanNet](https://arxiv.org/abs/2107.10043) also combines model structure with learned state estimation; a constant gain is a simpler baseline, not a replication of its recurrent gain estimator. Only if this controlled comparison reveals a reproducible limitation should a more elaborate recurrent or connectome constraint be proposed, with matched learned and rewired controls and new unexposed evidence.
