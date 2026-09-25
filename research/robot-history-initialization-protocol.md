# Robot history initialization: prospective development protocol

Draft implementation protocol. This study has not been registered or run. The reflection-capacity campaign continues under its existing registration. No partial outcomes from that campaign were used to choose this design. A separate committed source, data and qualification registration is required before fitting.

## Question and models

Test whether a cheap summary of the observed prefix improves initialization of the same compact transition. This is an information ablation, not a novel memory architecture. The [design note](robot-history-initialization-design.md) identifies the relevant history-encoder, observer and Kalman-network prior art. An innovation observer and online weight adaptation are deferred.

Use three primary arms around the unchanged `StructuredRobotTransition('dense_mlp')`:

| Arm | Extra initialization features | Parameters | Persistent numeric bytes |
| --- | --- | ---: | ---: |
| Last two | Original initializer | 590 | 2,600 |
| Local affine | Recent position, difference and preceding torque, plus fixed local square/tanh features | 962 | 4,088 |
| Temporal affine | Identical recent inputs, plus older position slope and older torque mean | 962 | 4,088 |

All arms retain twelve float32 forecast-state values. Bytes include learned parameters, state and four float64 normalization vectors; temporary workspace, request arrays and optimizer state are separate. No retained prefix or prepared-operator cache is allowed.

For a normalized 32-sample context, let `dq = q[31]-q[30]` and `c = concat(q[31], dq, u[30])`. The local features are `concat(c, dq*dq, tanh(u[30]))`. The temporal features are `concat(c, slope(q[0:30]), mean(u[0:30]))`. Compute each slope using weights `t-14.5` for `t=0..29`, divided by `2247.5`. Both feature vectors have thirty entries.

Both affine arms add `W*features+b` to the original twelve-value initializer. Their 12-by-30 weights and twelve biases start at zero. The shared transition parameters are identical for each paired seed, so all three initial functions agree. After training, the correction may alter all twelve latent coordinates, including the first six. Do not claim that the learned initial state preserves the final observed position exactly or represents a Bayesian posterior.

The two affine heads have equal stored parameter counts and common recent inputs. Their feature spaces and optimization conditioning differ. This is not a claim of identical effective function capacity. The local extra features are functional, with no padding or deliberately inactive weights.

No initializer uses `u[31]`: that torque first enters the forecast to predict `q[32]`. Scored forecasts receive no later position, innovation, teacher-forcing correction or reference-model state. Each request starts independently. Input and parameter nonfinites are errors, with no clipping or repair.

## Training and evidence

Use the same seven saved FIT recordings, two already-exposed DEV recordings, FIT-only normalization, three paired batch files, context32 and horizon128 as the closed structured study. No raw MAT decode, changed filtering, CONFIRM or official TEST access. The inputs remain realized measured torques, so this is conditional forecasting rather than verified command-driven control.

Fit all three arms concurrently as an experimental comparison, starting each from scratch. Execute fits sequentially on the host. Seeds are 8101, 8102 and 8103; learning rates are .001 and .003. This gives eighteen fresh attempts. Every attempt receives 4,096 updates, batch16, the original normalized position MSE, Adam betas (.9,.999), epsilon 1e-8 and gradient clipping at1. Use the original paired windows, including offset520000. Add no prefix reconstruction loss or feature statistics fitted on DEV.

Per-fit and whole-run deadlines, launcher, exact inherited evidence roster and resource provenance will be frozen in the registration after implementation qualification. Do not start a scientific run with these items unresolved. Qualification uses fabricated inputs only. It must verify the slope independently, exact initial-function/common-parameter parity, causal torque alignment, gradients, finite guards, request isolation and complete storage accounting.

Retain all attempted fits, initial/final weights, optimizer states and update traces. Close all eighteen attempts before this campaign reads DEV. This barrier does not erase prior development exposure. Reuse the same22 windows per DEV file, all H64/H128 scores and all three seeds at both rates. Select one learning rate per arm using pooled H128 standardized SSE over both files and all seeds; break ties toward the lower rate. A failed seed makes that rate ineligible. Never select the best seed or average predictions into an ensemble.

Keep the existing GRU10, GRU32, legacy scheduler, bounded/unbounded dense and causal-ridge references visible, with their exact source identity and original tuning budget. Retime selected controls on the current host; their fitting times remain historical. Do not silently refit or grant a larger search to any control.

## Development decision

The primary candidate is temporal affine. No observer or other model can substitute for it after seeing outcomes. Define each file's error as the mean H128 standardized RMSE across its three selected seed fits, then take an equal-weight mean of the two files.

Advance only if all of the following hold:

1. All three primary arms have complete finite selected recipes.
2. Temporal affine lowers the equal-file mean by at least5% versus both the local-affine and unaugmented controls.
3. Neither file's temporal-affine mean is more than2% worse than the better local-control mean on that file.
4. Its complete-request median latency is at most1.25 times the concurrent unaugmented control.
5. No declared comparison control is no worse in equal-file error, full-request latency and persistent numeric storage, with a strict advantage on at least one axis.

The 5%, 2% and25% margins are prospective development utility choices, not significance thresholds. This measures one fixed-exposure operating point, not a complete scaling frontier. A quality gain bought with extra latency is a tradeoff, not a speedup. Report per-file means, every paired seed difference, both rates, H64 errors and per-joint errors even though they are not dozens of separate pass conditions.

Timing uses batch1 on the first registered DEV window, three warmups and twenty repetitions per selected fit. Include normalization, conversion, prefix features, conditioning, validation, transition preparation, all128 forecast steps, denormalization and finite checks. Use the same deadline-callback policy, runtime and five single-thread settings for every arm. Aggregate each family's three fit medians by their median. Report raw durations, training time, shared-host identity and temporary versus persistent storage.

## Mechanism and continuation

Predeclare one fixed permutation of paired older position/torque samples at indices0..29, preserving indices30/31 and the boundary torque. Publish all ordinary and permuted forecasts for the selected primary recipes. The local arms must be exactly invariant. The temporal torque mean is order invariant in real arithmetic, although floating-point reduction order can change its last bits; its slope can change materially. Corruption is an off-distribution diagnostic, not an alternative way to pass the primary decision or proof of temporal reasoning. Freeze the exact permutation before evaluation.

The saved signals have undergone causal filtering. A useful history summary might recover preprocessing state, unobserved physical state or a task-specific statistical regularity; this experiment cannot identify which. A physical-memory claim would need a separately designed comparison with matched raw/filtered information and verified action semantics.

Independently replay saved checkpoints and reconstruct scores and the full decision before reporting an outcome. A pass only supports a separately frozen confirmation study. It does not establish architectural novelty, calibrated uncertainty, biological-wiring value, or an ICLR-ready result. A failure rejects this specific information contrast; adding more variants to the same DEV outcomes requires a new justified hypothesis rather than changing its rule.
