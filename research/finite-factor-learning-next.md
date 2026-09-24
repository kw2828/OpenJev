# Proposed next diagnostic: share the learned prefix filter with the forecast model

**Proposal only. Not registered or executed.** The completed [factor diagnostic](finite-factor-learning-results/report.md) passed all three criteria with an exact prefix posterior and learned operators. The learned-prefix/exact-operator and jointly learned cells failed each criterion overall. This supports testing the prefix-learning path under this recipe. It does not establish an irreducible GRU limitation, convergence, or a transfer result.

## Mechanism

Replace the GRU prefix encoder with a learned eight-state filter. Share its action-observation operators with the forecast rollout. The public input remains the same nine rows: the initial odor, followed by eight action/odor pairs. No oracle posterior is supplied at the forecast boundary or deployment.

The reset observation needs its own treatment. In the [world source](../src/openjev/research/finite_observation_world.py), it occurs before any transition or hazard. Start from the disclosed, fixed uniform prior `u[s] = 1/8`. Learn a reset emission `E0[o,s]`, normalized over the four ordinary odors for each state, and initialize:

`b0[s] = E0[initial_odor,s] * u[s] / sum_s(E0[initial_odor,s] * u[s])`.

Row zero is not a dummy action and must not introduce a found hazard. For each later public action/odor pair, apply the learned nonnegative branch operator and normalize:

`bt = B[action,odor] @ b_previous / sum(B[action,odor] @ b_previous)`.

For each action/current state, normalize the four ordinary-event destination branches and one found branch jointly. Keep the existing blind operator `A[action] = sum_odor B[action,odor]`. Blind forecasts propagate unnormalized surviving mass; observed forecasts emit costs, survival and the five-event distribution before assimilating the current observation. Found is absorbing: later cost and survival are zero, while the event distribution assigns probability one to found.

Keep the exact, fixed cost readout `C` in all arms. This remains privileged knowledge of the synthetic task, as does the common reset prior. The proposed learned filter does not receive the true emission, transition operators, hidden state, or posterior targets.

## Three matched arms

| Arm | Prefix processing | Forecast operators |
| --- | --- | --- |
| Shared filter | Learned `E0` and `B` | The same `B` |
| Untied filter | Learned `E0` and a separate `B_prefix` | Learned `B_forecast` |
| Existing GRU prefix | Existing 31-to-28 GRU and eight-state projection | Learned `B_forecast` |

Pair the initial forecast operators across all arms. Pair `E0` across the two filter arms, and initialize the untied `B_prefix` as an exact copy of the shared operator logits. Check that the two filter arms' initial forecasts agree within an explicit floating-point bound; the GRU prefix is not expected to match them. Use domain-separated initialization streams and preserve all seeds and final fits.

Shared versus untied filtering isolates the sharing constraint more directly. Comparing either filter with the GRU also changes the encoder family, parameter count and arithmetic; it is not a compute-matched architectural comparison. Retain the exact/exact implementation check without promoting an oracle-input model into the deployable candidate.

## Training and evaluation

Use identical newly reserved TRAIN cases and fresh DEV cases across arms, with the same base sensor law, 512 TRAIN attempts and 128 DEV attempts, and minima of 256 and 64 retained cases. Reserve new data and fit seeds before registration; do not reuse the completed DEV set for selection. Found prefixes are excluded without replacement, as before.

Keep three paired fits, 480 epochs, batch size 64, Adam at 0.003, gradient clipping at 5, and training horizons H1/H2. Preserve the existing four-component objective: blind cost MSE, observed cost MSE, half the sum of survival MSEs, and observed five-event soft cross-entropy. Add neither posterior supervision nor a prefix likelihood objective in this comparison. Complete every final checkpoint before DEV generation; evaluate H1/H2/H4/H8 without checkpoint selection.

Retain the three [existing criteria](finite-factor-learning-protocol.md#three-separate-criteria), unchanged and required for every fit seed:

- **SHORT_HORIZON_LEARNING:** at H1/H2, blind cost MSE and regret at most half the positive uniform reference; observed KL at most 0.1 nats.
- **BLIND_EXTRAPOLATION:** the same cost thresholds at H4/H8, plus H8 blind survival MAE at most 0.05.
- **OBSERVED_FILTERING_EXTRAPOLATION:** observed KL at H4/H8 at most 0.1 nats. This route receives intervening observations.

Report all arm/seed outcomes and paired differences. These absolute criteria do not by themselves establish superiority of shared filtering. Keep the existing prefix-shuffle diagnostic descriptive. Report parameter counts, precision, storage, updates, case exposures, prefix/forecast operations, inference time and complete phase time. Equal data and 480 epochs do not equalize compute; shared filtering backpropagates through the eight prefix updates as well as the future rollout.

## Interpretation limits

A shared-filter pass with an untied-filter failure would be evidence for this constraint under the frozen recipe, not a general architecture result. If both filters pass, the result would not isolate sharing. Any failure remains a result under the fixed optimization budget, not proof that the model class cannot learn the task.

The fixed centered cost readout has rank at most three and only four decision signatures across eight states. Accurate costs therefore do not identify a unique hidden-state posterior. Event forecasts add constraints without guaranteeing identifiability. Learned reset emissions and operators can remain observationally equivalent; internal probability consistency does not prove that they recover the true dynamics.

This proposal uses classical filtering structure. The existing protocol's references to [Predictive State Inference Machines](https://proceedings.mlr.press/v48/sun16.html), [HMM learning](https://arxiv.org/abs/0811.4413), and [value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html) remain relevant context, not claimed implementations or guarantees. No novelty, calibration, robotics, native-environment or broad architectural gain is implied.
