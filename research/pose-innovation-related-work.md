# What this error-history screen can and cannot add

Primary-source check, September 19, 2026. The following distinctions use the
papers' public abstracts and our implementation. They are not reproductions or
an exhaustive novelty search.

| Prior work | Relevant established idea | Consequence for OpenJev |
|---|---|---|
| [KalmanNet](https://arxiv.org/abs/2107.10043), Revach et al. | A dedicated recurrent network can be inserted into a structured filtering algorithm for partially known nonlinear dynamics. | A recurrent correction component is not itself a new contribution. Our direct forecast-error head is not a learned Kalman gain or a state estimator. |
| [Rapid Motor Adaptation](https://arxiv.org/abs/2107.04034), Kumar et al. | A base policy and adaptation module support rapid adaptation to changing robot conditions. | Adding a history-dependent adaptation module does not establish novelty. We need a specific mechanism and a fair comparison with simple history models. |
| [Continual Robot Policy Learning via Variational Neural Dynamics](https://arxiv.org/abs/2606.27353), Xing et al. | An analytical physics prior and neural residual are conditioned on hidden dynamics inferred by a recurrent encoder. The inferred condition also affects the policy; recurring conditions can be recognized rather than refitted. | Recurrence plus physics plus residuals is already studied. Our current screen tests whether explicit ordered errors carry useful additional information, before attempting a dynamics or policy update. |
| [Self-adapting Robotic Agents through Online Continual Reinforcement Learning with World Model Feedback](https://arxiv.org/abs/2603.04029), Domberg and Schildbach | World-model residuals trigger adaptation of a DreamerV3-based agent, with task and internal signals used to assess progress. | Error-triggered self-improvement is not a novelty claim. Our current run changes a small offline correction head, not a deployed policy or world-model weights. |

The most consequential evaluation warning comes from
[Raghavan and Singh's controlled feedback study](https://arxiv.org/abs/2609.02811).
In their differential-drive tracking task, model rankings based on an
uninterrupted imagined rollout differ from rankings under repeated measurement
updates and control. Their reported result is specific to that setup; we have
not reproduced it.

For OpenJev, the implication is a proposed design requirement: a useful
forecasting result must eventually survive a control evaluation with an
explicit sensing/update schedule and all adaptation costs counted. The
training-parent error-order screen is a cheap prerequisite, not a substitute.
A passing head would still require feedback integration, stronger baselines,
fresh conditions and closed-loop validation. A failing head provides no basis
for adding connectome terminology or claiming biological learning.

The [frozen experiment protocol](pose-innovation-protocol.md) distinguishes
ordinary public history, explicit error information, whole-history order and
error alignment using separate controls. Those distinctions define a testable
mechanism. They do not by themselves establish a paper contribution.
