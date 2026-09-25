# Proposed predictive-state routing investigation

**Prospective, unregistered, and not ready to launch. No architectural novelty is established.** This note changes no completed study or admission criterion. No experiment or transfer was run for this note.

The [joint-observer study](robot-joint-observer-results.md) failed its development gate: 4/5 criteria passed. Long conditioning improved mean error by only 0.750% over continued last-two and 0.319% over continued temporal, remained 4.682% worse than historical temporal, and cost 1.41482 times last-two latency. The failed gate does not admit the [proposed transfer](robot-observer-transfer-plan.md). The earlier [chain-connectivity comparison](robot-coupling-results.md) also lost to rewiring. These results motivate a different question; they do not establish that state representation or connectivity caused the failures.

## Hypothesis and causal contract

Investigate whether **routing arrived-observation errors between explicitly predictive state modules** preserves useful information better than an equally supervised dense recurrent model. Modules would represent features of output responses at different horizons. An action-conditioned decoder would read those features from the recurrent state. The experimental ingredient would be a sparse directed correction circuit between modules, not joint observer training, a longer prefix, or a biological name for an ordinary gain matrix.

At time `t`, predict the observation from the previous corrected state and already applied input `u[t-1]`. Form innovation `e[t] = y[t] - predicted_y[t]` only when `y[t]` arrives, then update memory. Autonomous forecasting receives no later observations and consumes supplied future inputs causally: output at horizon `h` cannot use inputs after that horizon.

During training, a decoder may receive a logged future input prefix and predict features of its corresponding observed future outputs. Those outputs are training targets only. Logged data do not supply outcomes under arbitrary reference actions or unexecuted counterfactual input sequences. Such targets must not be fabricated, nor may future inputs enter the online state update. Realized-torque forecasting alone cannot establish command-driven control performance.

## Prior art and the necessary control

[PSIM (2016)](https://arxiv.org/abs/1512.08836) learns filters in predictive coordinates. [Predictive-State Decoders (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/file/61b4a64be663682e8cb037d9719ad8cd-Paper.pdf) already supervises ordinary recurrent states with future-feature targets. [PSRNN (2017)](https://proceedings.neurips.cc/paper/2017/file/2bb0502c80b7432eee4c5847a5fd077b-Paper.pdf) already combines predictive states with bilinear observation gating. None of those ingredients is a novelty claim here; sparse routing itself still needs a closer prior-art review once specified.

The strong control is a dense recurrent model with **identical auxiliary predictive supervision**, decoder, available observations/inputs, optimization exposure, and matched state/parameter budgets. Otherwise auxiliary supervision could explain any gain. Degree-matched rewiring is additionally required before attributing improvement to connectivity. A learned graph is not evidence of a biological connectome. Stability would require analysis of the complete correction and transition dynamics, not just bounded component matrices; [contracting implicit RNNs](https://proceedings.mlr.press/v120/revay20a.html) provide relevant established machinery.

## Cheap falsifier and unresolved choices

Propose one three-seed, single-budget candidate-versus-dense comparison on newly designated TRAIN/DEV data. Candidate stopping margins to freeze before data access: at least 5% lower equal-recording autonomous forecast error, no recording more than 2% worse, and no more than 10% higher complete-request latency or persistent numeric storage. If the dense control matches the candidate or these margins fail, stop this routing proposal. Lower auxiliary loss alone is insufficient.

Before implementation, decide the actual routing equation, module semantics, sparsity construction, horizons, decoder, loss weights, and fair budget matching. Select the benchmark, causal input interpretation, whole-recording splits, untouched confirmation set, and final stopping margins prospectively. Do not recycle exposed robot recordings as fresh confirmation. No efficacy, stability, control, or publication-readiness claim follows from this note.
