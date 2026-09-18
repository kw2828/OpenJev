# What PlaNet and TD-MPC2 suggest for the reaching experiment

Primary methods and official implementations checked September 18, 2026.
These are possible later experiments, not changes to the frozen
[reward-residual comparison](reacher-reward-residual-pilot.md) or reproduced
paper results.

## The relevant differences

Our controller scores one bank of 64 candidate sequences using a 12-step sum
of clipped predicted rewards. It has no terminal value. Its GRU learns observed
angles and total reward, with an additional five-step open-loop loss.

[PlaNet](https://arxiv.org/pdf/1811.04551) also plans using predicted rewards
without a policy or value network. Therefore absence of terminal value alone
does not explain our failure. Its [official defaults](https://github.com/google-research/planet/blob/master/planet/scripts/configs.py)
instead use ten iterations of 1,000 candidates and 100 elites. The
[planner](https://github.com/google-research/planet/blob/master/planet/control/planning.py)
iteratively refits a Gaussian proposal and returns its final mean. This is
substantially more search than our single bank. PlaNet's Reacher setting uses
action repeat four: a horizon of 12 spans 48 underlying environment steps.
Our horizon spans 12 native decisions, or 0.24 seconds. Holding a candidate
command constant for three imagined decisions does not make these physical
lookaheads equal.

[TD-MPC2, equations 3 and 6 and Table 8](https://arxiv.org/html/2310.16828v2),
uses a short horizon of three, six search iterations, 512 candidates and 24
policy proposals. It adds terminal Q to rollout reward and trains latent
consistency, reward prediction and temporal-difference value objectives.
Reward/value targets use a transformed soft classification representation.
The [official agent](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/tdmpc2.py)
also carries the previous plan forward. The
[Q implementation](https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/world_model.py)
uses sampled-Q averages for planning and a minimum for TD targets. These are
several coupled mechanisms; importing one does not reproduce TD-MPC2.

## Controlled follow-ups

1. **Search allocation at fixed learned weights.** Compare one bank of 256
   sequences with four CEM iterations of 64, under the same horizon and action
   blocks. Retain the 64-candidate controller as a cheaper reference. Count
   anchors and any final rescoring within the budget, and charge proposal
   updates to timing. Use fresh paired native episodes and separately specified
   common-state action-ranking cases. Higher predicted reward with worse
   native return would indicate model exploitation, not useful planning.

2. **Training distance versus planning distance.** Compare five-step and
   12-step rollout losses with the same backbone, data and reward treatment.
   Cross them with both planning horizons. Equal optimizer updates cost more
   under longer rollouts; report those transition computations and include a
   compute-matched control. This is conventional multistep prediction, not
   JEPA or PlaNet latent overshooting. PlaNet Appendix D reports that latent
   overshooting slightly worsened its RSSM, so it is not a guaranteed fix.

3. **Terminal continuation value after competent prediction.** Compare
   reward-only scoring against reward plus a trained continuation estimate.
   Our objective ends after 50 steps, so value must condition on remaining
   time and equal zero at that scoring boundary. A value trained on recorded
   behavior is not automatically optimal Q. A TD-MPC-style actor/critic or new
   online exploration requires a separately budgeted experiment. PlaNet and
   TD-MPC2 update their data through interaction; our fixed dataset makes
   out-of-distribution imagined actions a separate concern.

These are established methods or ordinary adaptations. A later architecture
claim would require a specific biological-versus-rewired interaction under
the same corrected objective and planner, a strong conventional recurrent
baseline, and a memory intervention that actually changes control. Better
search alone would not establish a connectome advantage.
