# Prior art for the prospective bounded-history control

Written 2026-09-19 03:11:28 UTC (2026-09-18 20:11:28 America/Los_Angeles). Primary-source review only, accompanying [history-control-design.md](history-control-design.md). No current geometry-memory execution or audit output was inspected. The proposed follow-up remains conditional on the current comparison passing its complete prospective gate; no new experiment is authorized by this note.

**Recommendation:** treat last-two-valid-observation reconstruction as a strong conventional control. Replaying a public observation/action suffix through a recurrent model is established practice. Our useful question is whether the already specified persistent model retains a control advantage after the same GRU is trained to reconstruct its state from that bounded suffix. This is a test of the tested history restriction, training recipe and compute tradeoff, not an architecture novelty claim.

## Five relevant original sources

### 1. DRQN: zero-start subsequences and persistent episode state are distinct

Hausknecht and Stone, **Deep Recurrent Q-Learning for Partially Observable MDPs** (2015), arXiv:1507.06527. The paper replaces a DQN fully connected layer with an LSTM and studies ordinary and flickering Atari. Its “Stable Recurrent Updates” section, PDF page 3, contrasts episode-sequential training that carries hidden state with randomly sampled subsequences that start from zero. It reports similar performance for those update variants on its tested games and uses randomized updates thereafter. Its recurrent-versus-frame-stack comparison does not establish a universal benefit from recurrence.

**Relevance:** subsequence training with a zero initial state is old; it does not by itself enforce a bounded deployment history. Our proposed real-boundary reset must be active during training and deployment, with the entire allowed suffix reconstructed. [Original paper and methods](https://arxiv.org/pdf/1507.06527#page=3).

### 2. R2D2: burn-in addresses state mismatch, not a hard memory budget

Kapturowski et al., **Recurrent Experience Replay in Distributed Reinforcement Learning** (ICLR 2019). Section 3 studies recurrent-state staleness under distributed replay. Stored recurrent states can become inconsistent with updated weights; burn-in unrolls a replay prefix to obtain a starting state before the part used for updates. The paper compares zero initialization, stored state, burn-in and their combination, including DMLab ablations; its overall agent is evaluated on Atari-57 and DMLab-30.

**Relevance:** reconstructing a state from recent inputs is established. A stored starting state can contain earlier history, so stored-state-plus-burn-in is not our strict suffix-only control. Our reconstruction also participates in the ordinary predictive training graph; do not describe it as a replication of R2D2's update procedure. Methods verified in the [original submitted PDF, Section 3, pages 3-5](https://openreview.net/references/pdf?id=Hy7PKCFCQ#page=3); the [accepted-paper endpoint](https://openreview.net/forum?id=r1lyTjAqYX) required browser verification during this check, so this note does not attribute final-version hyperparameters to the inspected draft.

### 3. Recurrent model-free baselines: history length and implementation matter

Ni, Eysenbach and Salakhutdinov, **Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs** (ICML 2022; arXiv:2110.05038, inspected v3 dated 2022-06-05). Sections 4 and 5.2 vary recurrent architecture, inputs and training context length. Short, medium and long contexts have task-dependent effects; a longer context is not uniformly better. The paper explicitly discusses recovering hidden velocity from consecutive positions. Appendix A reports storage and training-cost considerations, including masked variable-length sequences.

**Relevance:** compare conventional histories before attributing gains to a specialized mechanism. Their context-length ablation is defined as sequence length supplied during training, not automatically a hard bound on online state. Their model-free actor-critic results do not establish our predictive-model/CEM outcome. [Methods and context ablations](https://arxiv.org/html/2110.05038v3#S4), [system details](https://arxiv.org/html/2110.05038v3#A1), [official proceedings](https://proceedings.mlr.press/v162/ni22a.html).

### 4. PlaNet: recurrent filtering and private imagined planning are established

Hafner et al., **Learning Latent Dynamics for Planning from Pixels** (ICML 2019; arXiv:1811.04551). Sections 2-3 define an approximate state belief inferred from past observations/actions, a recurrent state-space model with deterministic and stochastic components, and CEM model-predictive control. Candidate futures start from the current belief; the next real observation permits replanning. Experiments cover six DeepMind Control Suite tasks, including pixel-based Reacher.

**Relevance:** recurrent state inference plus imagined action search is already known. Our deterministic GRU, public angle packets and geometry score differ from PlaNet's stochastic latent model, image reconstruction and learned reward. A successful bounded-history comparison would not reproduce PlaNet or establish a new world-model architecture. It would isolate the permitted real-history source more narrowly while retaining imagined recurrence. [Original methods, pages 2-4](https://proceedings.mlr.press/v97/hafner19a/hafner19a.pdf#page=2), [official proceedings](https://proceedings.mlr.press/v97/hafner19a.html).

### 5. Finite-window POMDP control has theory, with assumptions we have not met

Kara and Yüksel, **Near Optimality of Finite Memory Feedback Policies in Partially Observed Markov Decision Processes** (JMLR 23, published February 2022; arXiv:2010.07452). Sections 1 and 3 construct approximate belief models from finite observation/action windows. Near-optimality bounds relate window length to controlled filter stability. The setup assumes known dynamics/measurement channels and finite observation and action sets; some bounds impose additional regularity and discount conditions.

**Relevance:** forgetting older history can be principled when the filter forgets its initialization. The theorem does not certify two observations as sufficient for our continuous-action, learned-model, finite-horizon CEM controller. Nor does it automatically cover our validity-dependent anchor. We have not established the required stability or approximation conditions. [Original paper, assumptions and construction](https://jmlr.org/papers/volume23/20-1152/20-1152.pdf#page=3), [Section 3 finite-window construction](https://jmlr.org/papers/volume23/20-1152/20-1152.pdf#page=10).

## What the proposed control actually tests

These are design inferences for OpenJev, not findings from the cited studies:

- **Information restriction:** at each real packet, erase learned carry state, retain the last two valid public angle observations and every issued command from the older observation to now, and reconstruct with the same GRU weights. Preserve measurement validity, elapsed ages, static public target and startup masks. Under the declared ten-missing-packet maximum, the suffix needs at most twelve packets/eleven commands. This is not simply “two-frame stacking.”
- **Training restriction:** train the reconstruction class from the paired original initial weights using matched data, order, losses and optimizer updates. Deployment-only truncation of a trained persistent model would confound lost history with an unfamiliar state distribution. No reward, native velocity, applied disturbance or hidden physics state enters its encoder.
- **Mechanism contrast:** shared recurrent operators avoid lag-specific command weights that would remain inactive during six-step-gap training but first activate under ten-step-gap evaluation. Longer replay is still a distribution shift. The comparison restricts retained real evidence; it does not eliminate recurrence from state reconstruction or imagined planning, prove exact state recovery, or isolate a biological mechanism.
- **Compute contrast:** hold candidate budget, horizon, geometry score and paired innovations fixed. Charge every repeated reconstruction step, padding computation, original head, candidate rollout, selected-action advance, copy and trace write. Equal parameter count and optimization updates are not equal training compute; equal CEM budgets are not equal total controller cost.

Circular finite-difference velocity remains a useful transparent companion: divide the wrapped angular displacement by the actual interval between valid observations, retain availability and staleness, and acknowledge angular aliasing and acceleration. It is not a substitute for the raw-history control or an exact current-velocity oracle.

If bounded reconstruction closes the practical gap under a separately frozen comparison, simple recent public history may explain the tested benefit. Failure of a superiority threshold alone does not establish equivalence. If persistent state remains better, the supported claim is an advantage over this trained bounded-history recipe on these tasks and costs. Neither outcome, by itself, is new recurrence, Bayesian inference, mechanistic identification or connectome superiority.
