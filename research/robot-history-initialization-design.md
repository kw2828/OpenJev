# Robot history initialization: prospective design

Status: the first comparison is now [closed and independently audited](robot-history-initialization-results.md), passing all five predeclared development criteria. This design was originally written while the six-fit reflection-capacity experiment was running, and its hypotheses preceded those outputs. The capacity experiment is [closed and negative](robot-reflection-capacity-results.md). The history result qualifies a separately registered confirmation comparison; confirmation data remains unopened.

The [three-arm initializer](../src/openjev/research/robot_history_initializer.py) passes 59 fabricated component tests, and the [integrated qualification](robot-history-initialization-qualification-results/README.md) passes 145 checks including those 59. The unchanged [pre-fit protocol](robot-history-initialization-protocol.md) specifies the completed local-versus-history comparison and defers the observers discussed below. The remainder preserves the prospective rationale; the measured outcome and its limits are in the [results report](robot-history-initialization-results.md).

The next useful question is whether the transition model starts with enough information. The current structured models receive a 32-sample context but initialize their twelve-value state from only the final two positions. Their dynamics are recurrent; their initialization does not assimilate the earlier observations or torques. Increasing the number of reflections does not address this distinction.

## Prior art that changes the comparison

| Source and method inspected | Implication for OpenJev |
| --- | --- |
| [Beintema et al., nonlinear state-space identification using deep encoder networks, L4DC 2021, Section 2.3, Eq. 3](https://proceedings.mlr.press/v144/beintema21a/beintema21a.pdf) | A history encoder initializes a nonlinear state model from past inputs and outputs, then a multistep simulation loss trains the model on overlapping sections. This is a direct baseline for our missing-history question. Adding an encoder and truncated rollout loss is established system identification, not a new world-model architecture. |
| [Becker et al., Recurrent Kalman Networks, ICML 2019, Sections 2.2-2.4](https://proceedings.mlr.press/v97/becker19a/becker19a.pdf) | RKN already combines state-dependent mixtures of locally linear transitions with observation updates and a factorized uncertainty representation. Its covariance retains observation-memory cross terms; dropping every cross term would stop observations from updating the corresponding memory through the Kalman gain. A new observer must be compared against this family before claiming a novel combination of recurrent dynamics and uncertainty. |
| [Revach et al., KalmanNet, TSP 2022, Sections III-A through III-C](https://arxiv.org/html/2107.10043v3) | KalmanNet learns an innovation gain using recurrent features while retaining a partly known state model. Differences between observations, predictions and posterior updates provide its inputs. It does not explicitly propagate a covariance in that formulation. A learned residual correction or recurrent gain alone is therefore not novel or evidence of calibrated uncertainty. |

These papers do not establish that their mechanisms improve our robot task. Their filtering, observation and model-knowledge assumptions differ. We should borrow the comparisons and causal interfaces, then measure their value.

## Smallest informative comparison

Use the compact dense MLP transition as the common dynamics model, subject to the closed capacity result. Compare its current last-two-position initializer with a fixed-size history encoder and an innovation observer. Include a learned constant-gain observer before a more expressive gain network. Train each complete model from scratch under a shared window sequence, loss and tuning budget; retain the original initializer as a concurrent reference. Keep GRU10, GRU32 and causal ridge as external controls. A larger GRU alone is not an adequate recurrent baseline: the existing GRU32 recipe is worse than GRU10.

For an observer, predict the next latent state from the previous posterior and preceding torque, then correct it with the newly observed position residual. All corrections occur inside the observed prefix. Forecasting begins after the final correction, consumes the boundary torque, and receives no later position or error. Observation assimilation changes the initializer; it must not silently introduce teacher forcing during the scored horizon.

A full history encoder can have substantially more parameters than the 590-parameter transition. Report that cost and include a model with comparable parameter count that still sees only the final two positions. Also report request latency including prefix processing. Matching latent-state size alone does not match parameter storage or computation.

Before any fit, freeze the exact encoder size, observer equation, initialization, two learning rates, three seeds, training budget and continuation threshold. Do not choose these after inspecting partial DEV results. Use the same previously exposed DEV recordings for this development comparison; a pass can only propose a separate confirmation protocol.

## What would identify a useful memory mechanism?

Predeclare a paired intervention that permutes older position/torque observations while preserving the final two and their timing. It is a diagnostic corruption, not a realistic deployment distribution. Compare it with writes disabled and with the ordinary ordered prefix. If extra parameters or an unordered summary explains the gain, report that explanation.

Require a benefit over both the history encoder and constant-gain observer before investigating a more elaborate error-driven memory. Any fast-weight extension must also beat a delta/NLMS or RLS update on the same features, with all coefficients, covariance state, pending observations and writes charged. Reset state between requests and distinguish adaptation of hidden state from adaptation of model weights.

The existing causal filtering of measured positions adds another confound: history might recover preprocessing state instead of unobserved physical dynamics. These data also contain realized torques, not verified issued commands. A useful result here remains conditional forecasting. A robotics control claim needs an environment with authenticated action timing and closed-loop interventions.

The contribution still to be earned is a reproducible advantage from a specific memory mechanism across independent conditions, at a useful total cost. Biology, recurrence, prediction residuals and uncertainty are research motivations, not evidence of novelty.
