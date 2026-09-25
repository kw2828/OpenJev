# Robot reflection capacity: development protocol v1

The [four-reflection experiment](robot-structured-results.md) failed its original rule, passing 23/61 conditions. Its matched bounded dense and MLP-gated controls were more accurate. This study tests whether increasing reflector capacity recovers useful performance under the same training recipe. It does not replace the failed study or revise its rule.

## Hypothesis and limits

A product of four reflections satisfies `rank(Q-I) <= 4`. Increasing the count to twelve permits a broader operator family, but also adds 176 parameters and more sequential operations. An improvement would support this capacity intervention, not prove that rank alone caused the earlier failure.

Reflection products are established prior art in [orthogonal RNNs](https://proceedings.mlr.press/v70/mhammedi17a.html). [DeltaProduct](https://arxiv.org/abs/2502.10297) explicitly studies a reflection-count tradeoff between expressivity and efficiency. State-dependent parameter-varying recurrence also has precedents such as [ReLiNet](https://www.ijcai.org/proceedings/2023/385). This is a task-specific ablation, not a new Householder construction or a novelty claim.

Our twelve-reflection transition is still not an arbitrary bounded dense operator. In metric coordinates it has the form `diag(rho(g))*Q(g)`, with orthogonal Q, positive bounded row scales, fixed vector anchors and bounded free vector coordinates. The same scalar mixture controls all prototypes and forcing. Twelve reflections give positive determinant for Q; they do not remove these other restrictions. State-dependent gates prevent a claim of associative parallel-scan execution.

## Model and initialization

Keep the frozen structured-cell equations, two-way compact reset-GRU gate, forcing weights and biases, diagonal metric, decay parameterization and explicit twelve-value state. The gate reads current predicted position and current torque. Context initialization still uses only the last two observed positions; no prefix encoder or scheduler memory is added.

Use `ReflectionCapacityRobotTransition(12, seed)`. Its first four raw reflection vectors and every common parameter equal the original four-reflection initializer exactly. Append four independently trainable identical pairs from `.01*randn(2,4,11)` using a separate local generator with seed `seed ^ 0x43415031`. The complete anchor sequence is `(0,0,6,6)` repeated three times. Prototype tanh transforms precede mixture interpolation, just as in the original cell.

Identical reflection pairs cancel in real arithmetic, so both models start at the same `.999I` metric-space transition. Different reflection counts are not claimed to produce bitwise-identical float32 trajectories. The new four-reflection implementation delegates the old numerical path and is qualified against its predictions, final states and gradients. The twelve-reflection implementation is checked against an independent materialized operator and gate oracle, finite differences, causal chunking and forced-state bounds before fitting.

The new model has 806 parameters, matching bounded dense, versus 630 for the old four-reflection model. It retains twelve state values, no buffers and no prepared cache. Float32 parameters plus state and four float64 normalization vectors total 3,464 persistent numeric bytes. There is no remaining storage advantage over bounded dense; the MLP-gated dense control uses only 2,600 bytes. The norm bound concerns forced trajectories in exact arithmetic, not incremental contraction, gradient bounds or robot safety.

## Data, controls and training

Authenticate the complete closed structured study and original independent audit before decoding arrays. Use its eleven inherited FIT/DEV/reference descriptors. Do not decode raw MAT files, reconstruct different preprocessing, concatenate recordings or open internal CONFIRM or official TEST.

Use the same seven saved FIT recordings, FIT-only normalizers, linear initializer, causal ridge banks and three paired batch files. Verify the parent normalizers and batch sampling contract. Reuse context32, horizon128, 4,096 updates, batch16, normalized position MSE, Adam betas (.9,.999), epsilon 1e-8 and gradient clipping at 1. The two learning rates remain .001 and .003; seeds remain 8101,8102,8103 with batch-window offset 520000. Only six twelve-reflection fits are new. Do not warm-start them from trained checkpoints.

Before each new fit, verify that the common initial parameters and first four raw vectors equal the corresponding saved four-reflection initialization. Preserve every new initial/final checkpoint, Adam state, update trace and failure. Per-fit cap is 1,800 seconds; whole campaign cap is 10,800 seconds. Use a suspend-aware deadline in addition to the inherited training checks. No restart, rescue, increased update budget or seed promotion. Numerical failures stay in the roster; programming/schema failures stop the campaign.

Copy all 36 parent fit recipes, including both rates and all three seeds for four reflections, bounded dense, unbounded dense, dense MLP, GRU32 and legacy instant scheduling. Preserve their initial/final weights, Adam state, trace and fit receipts byte-for-byte, with the full original fit record and original source identity. Their historical fitting times are not fresh measurements. This produces 42 fit records, with 36 cached and six fresh.

Close the six fresh attempts and copy all cached fits before decoding DEV for this campaign. This barrier does not undo earlier exposure: the same two DEV recordings already informed model design. Evaluate all 42 fits and four frozen references on the same 22 windows per recording. Context q/u[0..31] precedes the forecast; first future torque u[31] predicts q[32]. Later torque inputs cannot influence earlier targets. Future inputs are measured realized torques, not verified issued commands. This remains offline conditional forecasting.

Keep all 184 H64/H128 metric rows and 92 forecast files if every forecast is finite. Select one learning rate per family using pooled H128 standardized RMSE over both DEV recordings and all three seeds, breaking ties to the lower rate. Reselect all cached families from their unchanged six-fit recipes. Select the causal ridge penalty using the same original criterion. Never select a seed or average predictions into an unregistered ensemble. Missing/nonfinite evidence fails the relevant conditions.

## Prospective continuation rule

All 69 conditions are required to qualify a new confirmation protocol. One condition requires eligible recipes for all seven learned families and causal ridge. On each DEV recording, the twelve-reflection candidate must:

- Lower mean H128 error by at least 5% versus four reflections and have no higher error in any of the three paired seeds: four conditions.
- Stay within 2% of the mean error and within 5% of every paired-seed error for each of bounded dense, unbounded dense, dense MLP, GRU32 and legacy instant scheduling: twenty conditions.
- Beat frozen linear and persistence mean error by at least 5%, and stay within 5% of the selected causal ridge: three conditions.
- Have mean physical RMSE on each of six position channels no more than 10% worse than GRU32: six conditions.

These are 33 conditions per recording, plus eligibility, giving 67 accuracy conditions. Two additional conditions require full-request eager CPU latency no more than 1.05 times bounded dense and no more than 1.05 times dense MLP. Report accuracy and compute counts separately as diagnostics, but a partial pass never opens confirmation. If accuracy recovers while compute fails, report that exact outcome. Do not substitute a native result or omit the faster MLP control.

Retime every selected fit and all four references on the same current host: 25 resource rows. Use batch1, the first registered DEV window, three warmups and 20 repetitions. Each family latency is the median of its three individual fit medians. Include normalization, float conversion, conditioning, validation, prototype/operator preparation, gate computation, full 128-step rollout, denormalization and finite-output checks. Every timed prediction also includes the same suspend-aware deadline callback. Clear parameter gradients before inference. Set Torch and all five CPU/BLAS thread controls to one.

Report raw durations, exact numeric storage, input 9,216B and output 6,144B payloads, runtime and shared-host identity. Exclude Python object overhead, temporary workspace, optimizer state and model/disk loading from the deployment-storage count. A local CPU comparison does not establish general hardware throughput.

## Evidence and interpretation

Freeze source, qualification, parent closure and data hashes in a committed registration before fitting. Preserve original launch and terminal process receipts, all attempted fits and all scores. Independently replay saved checkpoints, reconstruct metrics and all 69 conditions before publication. Keep failed qualifications and run failures rather than retrying through another entry point.

Even a complete development pass is not evidence of a novel architecture or an ICLR-ready result. It only supports a separately frozen confirmation study; a credible contribution still needs untouched evaluation, scenario transfer, a strong mechanism-specific comparison and a second environment. The full project objective remains stronger performance from a defensible architecture contribution, not merely passing this ablation.
