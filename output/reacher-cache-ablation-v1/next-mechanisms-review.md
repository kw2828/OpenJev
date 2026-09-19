# Possible mechanisms after the explicit-cache comparison

Primary-source review, 2026-09-18. Prospective, unfrozen and unrun. The ongoing cache study is identified by plan SHA-256 `7868daa12242df37f020946f9d3b279811a0e97547eef4d8b179da6e9394cefa` and source commit `d0d9c34`, supplied by the coordinating task. Its partial outputs were not inspected. This review made no model, native-environment, training or API-evaluation calls and changes no protocol or source. Only this memo was written.

The existing [hidden-motion review](../../research/reacher-hidden-motion-followup.md) already covers RKN, action-conditional RKN, differentiable filters, KalmanNet, PlaNet and recurrent RL baselines. Those are still necessary conventional comparators. The five papers below add particular mechanisms; they are not evidence that another large architecture is warranted.

## Five primary papers and what they actually establish

### 1. KalMamba: probabilistic filtering with a parallel sequence backbone

Becker, Freymuth and Neumann, [KalMamba: Towards Efficient Probabilistic State Space Models for RL under Uncertainty](https://arxiv.org/html/2406.15131v1), arXiv:2406.15131, especially Sections 4.1-4.3.

**Established mechanism.** A Mamba backbone processes observation/action history and predicts the parameters of a latent linear-Gaussian transition. The model uses diagonal dynamics/covariances and associative scans for Kalman filtering and smoothing. Training uses smoothed beliefs and a variational objective; acting uses causal filtered beliefs with SAC. The backbone is regularized toward the filtered belief to discourage information bypassing the probabilistic state. Its reported parallelism targets sequence computation, especially longer sequences.

**Closest precedent and limit.** This is a direct extension of the filtering family already reviewed, particularly the variational recurrent Kalman network, rather than evidence that a new uncertainty model is needed. Its training smoother has future information that an OpenJev controller cannot receive. SAC and variational training also differ from our fixed-data Anchor-loss/CEM setting. Neither parallel sequence training nor its results imply lower scalar decision latency for a 50-step CPU task. [Primary paper](https://arxiv.org/html/2406.15131v1).

**Potential small test, our proposal.** Keep the action-conditioned filtering baseline, decoder and planner fixed. Compare a GRU that parameterizes the transition with a small causal linear-state backbone. Hold the covariance parameterization and training objective equal. Only attempt this if profiling shows the sequential backbone is a meaningful bottleneck and the cache study leaves useful unresolved history. A complete KalMamba reproduction would be a separate project.

### 2. Gated DeltaNet: correct an addressed association instead of adding another trace

Yang, Kautz and Hatamizadeh, [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/html/2412.06464v1), arXiv:2412.06464, Section 3.1; [official implementation](https://github.com/NVlabs/GatedDeltaNet).

**Established mechanism.** The matrix memory update combines decay with an error-correcting key/value write:

`S_t = alpha_t S_(t-1) + beta_t (v_t - alpha_t S_(t-1) k_t) k_t^T`.

The subtracted term removes the old association addressed by the current key; the scalar decay permits broader forgetting. The paper also develops chunkwise parallel training. Its empirical setting is language/sequence modeling, not noisy robot state estimation.

**Closest precedent and limit.** The closest comparators are a gated additive outer-product memory and a delta-rule memory without decay. This is an established fast-state mechanism, not our invention. Its learned key/value residual is not automatically a physical sensor innovation, a covariance estimate, or biological plasticity. Efficient GPU kernels do not imply a CPU speed advantage. [Primary paper, Equation 8](https://arxiv.org/html/2412.06464v1#S3.SS1).

**Potential small test, our proposal.** Put a small addressed correction memory beside the common explicit cache. Update it only from a newly available public observation and its causal prediction error. Compare error-correcting writes with additive writes using the same encoders, matrix dimensions, decay and parameter budget. Keep a GRU with comparable total state/parameters as a capacity control. A gain would concern how corrections are stored, not prove a general world-model architecture advantage.

### 3. Skip RNN: reduce updates, without pretending the world stopped

Campos et al., [Skip RNN: Learning to Skip State Updates in Recurrent Neural Networks](https://arxiv.org/html/1708.06834v1), arXiv:1708.06834, Section 3.

**Established mechanism.** A binary gate selects a recurrent update or copies the old state. A learned update probability accumulates across skipped steps; training uses a straight-through gradient estimator. An optional loss penalizes the number of updates. The skip schedule can avoid evaluating the full recurrent cell. This is temporal sparsity, not a sparse anatomical connectivity result.

**Closest precedent and limit.** It supplies the obvious learned-gate baseline for any claim that selective updates save computation. The paper does not turn the gate into calibrated uncertainty. Copying a dynamical belief across a physical step can discard issued-action effects and elapsed time, so a literal whole-state skip is a poor control design for our task. [Primary paper, Equations 2-8](https://arxiv.org/html/1708.06834v1).

**Potential small test, our proposal.** Always advance the action-conditioned physical prediction and clock; gate only an optional correction module. Compare always-update, observation-validity-only, periodic, random-at-matched-count, and learned gating. Preserve identical mandatory transitions. Measure actual gathered/scattered execution: evaluating both branches and masking their outputs does not provide the intended savings. Test this only if the optional correction is both useful and costly.

### 4. Adaptive Computation Time: separate internal refinement from physical time

Graves, [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/html/1603.08983v1), arXiv:1603.08983, Section 2.

**Established mechanism.** ACT repeatedly applies shared recurrent parameters at one input timestep. A learned halting unit defines the stopping weights and remainder; the final state/output is a weighted combination of intermediate values. A ponder penalty discourages excessive computation. The first internal update is flagged so repeated computation is distinguishable from another input timestep.

**Closest precedent and limit.** This is a direct precedent for paying for extra recurrent computation. A halting probability is not a calibrated probability of a correct action or a predictive covariance. It also does not establish that high predictive entropy identifies reducible errors. The original sequence experiments are not an action-conditioned controller comparison. [Primary paper, Equations 3-10](https://arxiv.org/html/1603.08983v1).

**Potential small test, our proposal.** Permit one versus three shared-parameter refinement iterations before planning, with a cheap gate charged to both paths. Compare always-one, always-three, ACT-style halting, age/innovation threshold, and uncertainty-based gating at matched total work. Internal refinement must leave the real clock, issued-action history and measurement count unchanged. It must not call the existing physical `advance` repeatedly or repeatedly count the same observation as independent evidence.

### 5. Backpropamine: test episodic plasticity before attributing anything to wiring

Miconi, Rawal, Clune and Stanley, [Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity](https://arxiv.org/html/2002.10585v1), arXiv:2002.10585, Section 3; [author implementation](https://github.com/uber-research/backpropamine).

**Established mechanism.** Effective connections combine fixed weights with a trainable scaling of an episodic Hebbian trace. A network-produced modulation signal controls trace writes; another variant modulates an eligibility trace. Structural parameters are optimized between episodes, while the plastic trace changes within an episode and resets at its start. The paper compares plastic/nonplastic and modulated/unmodulated variants. Its RL examples include previous reward as an input.

**Closest precedent and limit.** Differentiable Hebbian plasticity and ordinary recurrent memory are the immediate comparators. This supports a biologically inspired learning rule, not superiority of an anatomical connectome. An episode-varying trace is extra state and work. Its reward-input setting cannot silently replace OpenJev's current packet-and-command-only control inputs. [Primary paper, Equations 1-5](https://arxiv.org/html/2002.10585v1).

**Potential small test, our proposal.** Compare no plasticity, fixed-rate plasticity, and an observation-innovation-modulated trace on the same small recurrent backbone. Keep all three on public inputs, with equal slow-parameter and dynamic-state budgets as far as possible. First use a common dense or fixed sparse topology. A later sparse-versus-biological topology factor needs degree/edge-count controls and matched plasticity; changing topology and write rule together would confound the result.

## Conditional decision after the current audit

The following branches were written without inspecting the ongoing outcomes:

| Completed cache result | Next decision |
| --- | --- |
| Persistent recurrence satisfies the complete frozen gate against both cache families | There is evidence for useful information beyond a single retained angle measurement under this recipe. First test the previously proposed two-valid-measurement/interval-motion comparator. If that leaves a residual gap, test one correction-memory mechanism with its conventional filtering controls. |
| Persistent recurrence fails the gate, while cache controls appear close | Preserve the failure. It is not proof of equivalence. Establish a predeclared practical noninferiority margin if the next goal is to simplify the controller. Do not introduce plasticity as a post-hoc rescue. |
| A result is driven by one fit, one comparator or only full sensing | Investigate the already identified optimization/information confound on a separately frozen follow-up. Do not select a favorable pair or describe it as specific to bridging blackouts. |
| History helps, but extra fixed-depth computation does not | Do not train an adaptive-depth gate. A gate cannot allocate a benefit that the slow path has not demonstrated. |
| Fixed extra computation improves some cases and its benefit can be predicted causally | Gate that specific computation. Uncertainty gating is only one candidate, alongside age, actual-observation innovation, learned value of computation and periodic allocation. |

More thinking cannot reveal the realization of unobserved future actuator noise. Large variance can be irreducible. A proposed uncertainty gate should be judged on whether it predicts a useful improvement from the extra computation, not merely on whether it turns on during blackouts. The shared task's reward is a training target; it is not currently a controller input.

## My smallest useful mechanism proposal

Only after a residual history benefit survives the cache and two-measurement controls, test **a small correction memory updated by actual observation innovations**. This is an engineering combination of known mechanisms, not a novelty claim.

Keep the common explicit angle cache, action-conditioned mean transition, reward head and CEM interface. Before the real observation at time `t`, obtain the causal predicted angle features from the previous actual action. When the observation is available, compute its four-component sine/cosine residual against that prediction. Encode a key from the prior belief and issued-action context and a value from that residual. Update a small matrix state with a delta-rule write; the matrix's readout supplies a residual correction to the common estimator. On missing observations, make no measurement-driven write. A declared action-conditioned prior still advances normally.

This proposal deliberately uses an observed prediction error rather than treating any learned key/value mismatch as a physical innovation. No true velocity, native state, realized actuator disturbance, future measurement or unissued action enters a real write. Predicted candidate trajectories may use private copies but cannot write imagined observations into the real memory. Reset the matrix at episode start. Slow trained parameters remain fixed at deployment; the matrix is audited episodic state, not an undeclared optimizer update or extra RL interaction.

Start with three within-backbone variants: no correction memory, additive writes, and delta writes. Share encoders/readouts, decay and training data where meaningful. Add a larger ordinary GRU as a capacity control and retain the strongest explicit-history model plus conventional action-conditioned filtering. The error-correcting versus additive contrast tests write semantics; the GRU and explicit-history controls test whether the extra state was necessary. If covariance is added later, both the conventional filter and new estimator need the same known action-noise treatment.

Do not add adaptive depth, uncertainty loss, plastic topology and online data collection in this first comparison. Fixed sensing gaps and fixed physical dynamics may require only state estimation, not learning new dynamics. A claim about rapid dynamics adaptation needs a separately declared change in hidden dynamics, not merely a longer blackout.

## Matched evaluation and continuation requirements

- Freeze the numerical utility margin, all-fit consistency rule, full-sensing degradation allowance, shift, total compute budget and stopping rule before drawing new evaluation cases. Keep the current study's outcome intact.
- Pair the fixed public training corpus, all fit seeds/orders and evaluation exogenous noise. Use the same initial search proposals and innovations; allow later score-adaptive CEM banks to differ. Do not grant a new architecture extra search samples without matching that budget in its comparators.
- Report every paired fit, total training and deployment time, all gate/refinement calls, matrix-state bytes, branch-copy cost and any uncertainty samples. Matrix state is quadratic in its two dimensions; low parameter count is not a complete memory/compute budget. Sparse masking alone does not establish faster execution.
- Separate causal prediction, audit-only hidden-motion diagnostics and native control utility. Better observation likelihood or smaller innovations are insufficient if control does not improve. Extra-depth comparisons must include the cost of deciding to use the slow path.
- For a gated method, first show that a fixed slow path has a repeatable benefit. Then compare utility at matched measured computation against always-fast, always-slow, periodic/random allocation and a simple age/innovation gate. An uncertainty signal needs held-out calibration, and calibration alone does not prove that extra computation is useful.
- For biological topology, cross the same write mechanism with conventional, degree-matched random and rewired connectivity. Match recurrent state, edges and stability controls. Continue toward a biological claim only if topology adds a repeatable benefit beyond those factors and transfers to a second environment.

The most useful next result could be that a two-measurement feature or a standard filter explains the remaining advantage. The cited papers already establish filtering, selective computation and plastic memory. Our contribution would have to be a narrower demonstrated mechanism under fair comparisons, rather than their combination or a connectome label.
