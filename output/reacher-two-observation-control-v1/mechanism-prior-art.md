# Two mechanisms worth separating from a connectome claim

**Prospective literature note, September 18, 2026. No justified architectural novelty emerges from these papers alone.** No running-study outcomes were read, and no model, training or simulator call was made. This is neither a frozen protocol nor authorization for another experiment. Future training remains conditional on the current scientific gate and separate review.

The existing [learning roadmap](../../research/connectome-learning-program.md), [robotics survey](../../research/connectome-robotics-next-experiments.md) and [architecture decision note](../../research/reacher-architecture-mechanism-options.md) already cover slow context, plastic synapses, state-space recurrence and topology-matched controls. Two additional primary papers fill narrower gaps: intrinsic neuronal adaptation and an action-conditioned probabilistic filtering baseline. The prepared [two-observation replay component](integration-review.md) supplies the public-information control; it has no effectiveness result yet.

## 1. Intrinsic adaptation, distinct from changing synaptic weights

**Source:** Bellec et al., [*Long short-term memory and learning-to-learn in networks of spiking neurons*, arXiv:1803.09574v4, 25 December 2018](https://arxiv.org/html/1803.09574v4), NeurIPS 2018; Sections 2-6.

**Established result and limit:** LSNN neurons raise their firing threshold after activity and let it decay, giving each adapting neuron an additional state. The paper reports sequence classification and meta-learning, including goal navigation with deployment weights held fixed. Adaptation improves its reported non-adapting spiking controls; this is not uniform superiority over LSTMs. Training uses BPTT, which the authors explicitly do not claim is biologically realistic. The navigation experiment is not a robotic world-model/MPC evaluation, and learned rewiring is not a measured fly connectome.

**Our proposed intervention:** isolate activity-dependent negative feedback in a rate-model transition before considering any biological graph. Retain the GRU gates, subtract `beta * b_t` from the candidate activation's pre-tanh input, and update a bounded per-unit activity trace once per issued-action interval:

`b_(t+1) = rho * b_t + (1-rho) * sigmoid(h_t)`.

Use fixed nonnegative `beta` and decay coefficients for the first comparison, selected only from training development. Observation correction does not advance the adaptation clock. This is an engineering analogue, not an LSNN reproduction. It requires an explicit cell/configuration change; an unchanged GRU checkpoint is insufficient.

**Falsifiable comparison:** compare feedback enabled versus `beta=0`, paying for the same trace updates in both. Retain an ordinary GRU and a fast/slow recurrent control, matching total recurrent-state size and trainable-parameter budget and declaring any remaining mismatch. A feedback-off ablation alone does not control effective capacity. Initially rebuild both hidden and adaptation state from exactly the same retained public packet/action suffix as the two-observation control. Each imagined candidate then gets a private copy. Hold observation inputs, targets, training recipe, geometry scoring and CEM fixed. Ask whether feedback reduces action-ranking error through the gap and native control cost without delaying reacquisition. Report full computation and state-copy costs, not only parameter counts.

**If short history explains the gain:** the remaining question is whether this cell is a better inductive bias for the same information. It would not establish a need for long memory. Failure against the bounded GRU closes this candidate on the present task; extending the task after that would require an independently motivated, separately declared question.

Also retain a constant-feedback control, with its value fixed using training data only, while still paying for the unused activity trace. The proposed sigmoid trace has a positive mean, so a gain over `beta=0` could reflect an additional bias rather than activity-dependent adaptation. This is a design deduction from the proposed equation, not a result of the LSNN paper.

## 2. Action-conditioned filtering: the necessary conventional alternative

**Source:** Shaj et al., [*Action-Conditional Recurrent Kalman Networks For Forward and Inverse Dynamics Learning*, arXiv:2010.10201v2, 5 November 2020](https://arxiv.org/html/2010.10201v2), CoRL 2020; Sections 4-6 and Appendix B.

**Established result and limit:** ac-RKN adds action-dependent latent transitions to a recurrent Kalman model and learns robot forward/inverse dynamics. Its experiments include hydraulic, pneumatic and electric robots with missing observations. When measurements are absent, it propagates the prior and skips the measurement update. Its decoder can carry predicted observation values, but those are not new sensor evidence. The paper explicitly leaves real-time control and reward-predicting model-based RL to future work; prediction results therefore do not establish closed-loop utility.

**Our proposed intervention and ablation:** use only public angle packets, validity/age, goals and issued commands. Learn an action-conditioned latent mean/covariance transition; assimilate actual visible angles only. Decode angles for the unchanged geometry scorer. First keep CEM selection based on the decoded mean, so covariance-aware planning is not another simultaneous intervention. Compare learned uncertainty-dependent measurement gain with a fixed-gain filter using the same action model, and with the bounded GRU. Reconstruct all filter state from the same bounded public suffix for the initial information-matched comparison. Do not import robot velocity inputs from an inverse-dynamics example or feed predictions back as measured packets.

**If short history explains the gain:** a compact filtering state may still offer accuracy or measured computation benefits, but persistent neural memory is unnecessary to that argument. A covariance output is not automatically calibrated; calibration needs held-out measurement errors and its own checks. The inherited independent actuator noise does not justify a hidden persistent-physics adaptation claim.

## What would support a biological claim later?

Only after a cell-level benefit, cross adaptation on/off with a documented biological graph and multiple signed-degree-, role- and role-mixing-matched rewires. Preserve sensory/motor interfaces and the decay-rate multiset; include within-role decay-assignment shuffles. A specific interaction is:

`adaptation cost reduction on biological graph - mean reduction on matched rewires`.

If the interaction is absent, any benefit belongs to adaptation, not the connectome. Unmeasured cellular decay assignments must be called engineered, not anatomical. No graph should be chosen because it wins an exposed evaluation. Neither selected paper establishes this interaction, so it remains a falsifiable proposal, not an ICLR novelty claim.

No upstream code is copied or licensed by this note. An implementation should independently follow the attributed equations or first verify the exact official code revision and its license. The immediate missing evidence remains a properly trained bounded-history comparison, not another architecture name.
