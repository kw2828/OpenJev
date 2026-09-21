# Prospective coverage-design review

**Design review only. No collection, fitting, inference or evaluation was performed to write this note.** The [new protocol](../../research/otto-coverage-protocol.md) fixes a two-stage comparison. Collection requires its own pinned executable plan; training/evaluation remain subject to a separate complete audit, qualification and freeze.

The recent [capacity screen](../../research/otto-capacity-results.md) passed 6/12 conditions, with all TRAIN criteria passing and all required VALID gains missing. The earlier [Bellman control](../../research/otto-bellman-control-results.md) passed 19/42 conditions and failed the overall rule. Neither says ordinary networks cannot work. The coverage comparison keeps the original width-eight model and backup procedure fixed to test one next factor.

## What this identifies

Compare all 5,589 original teacher states against a 5,589-row mixture containing 2,799 teacher and 2,790 new student states, from the same three original checkpoints. Hold original regime/hit counts, optimization/update/refresh counts, parameterization, branch arithmetic, fitting seeds and autonomous cases fixed. All fitting seeds receive the same pooled mixture.

This identifies the effect of the declared exposure replacement within a bootstrapped learning procedure. It does not isolate geometric support from changed sample weighting or changed self-generated targets. Equal index permutations do not imply equal examples, and equal optimizer/target counts do not imply equal compute. Low self-target MSE can coexist with wrong action ranking.

Student-state training is not new or untested here: the [symmetry/action-head study](../../research/otto-symmetry-head-results.md) already pooled student states and failed, with 2/54 conditions passed. The new comparison uses scalar values and observation-backup targets rather than direct action preferences. [OTTO Algorithm 1](https://auroreloisy.github.io/papers/Loisy2023a_EurPhysJE_drl-benchmark.pdf) already supplies exploration/replay/target-network precedent. No novelty follows from using these components.

## Decisive controls and failure points

- Use unchanged original collectors, both supported kernels and all three seeds. Never choose a collector from new outcomes. Preserve all 72 found/censored paths and final updates.
- Compute the six stratum quotas from original TRAIN metadata only, by exact integer largest remainder. Preserve original stratum totals; split each student quota evenly across collectors. Hash-select exact row identities without replacement or quality filtering. Any short quota fails rather than admitting more collection.
- Uniform row weighting and pooling candidate states mean longer trajectories supply more candidates. That is the declared occupancy-weighted intervention, not equal episode weighting. Byte-identical features remain separate exposures, and selected rows may be redundant; do not manufacture diversity through an unregistered deduplication rule.
- Student rows receive no censored Monte Carlo labels. An API-compatible zero placeholder must never reach optimization: backup refresh one completes first, and every consumed target is joined to its saved target checkpoint and selected row.
- Restore the same original parameters and `c0`, with fresh independent Adam in both arms. Eight refreshes in each of six fits require 48 complete refreshes and 268,272 target forwards, twice the target-generation count of the prior MC-versus-backup comparison. Record this cost explicitly.
- Keep fresh EVAL entirely outside collection and fitting. Retain analytic and unchanged references, all three fitting seeds, all strata and eight paired blocks. Lambda5 is an unseen supplied kernel, not a hidden-model adaptation claim.

The fixed 42 conditions retain the strongest essential safeguard: each mixture fit must reach at least 95% weighted success and no more than 1.05 times analytic moves in every setting. Improvement over failed students alone cannot pass. The other 24 conditions require improvement against both teacher-only backup and unchanged controls, consistent block gains and nonworse complete deployed cost. A failure cannot be rescued by training loss, one favorable setting, a different amortization horizon or selected seed.

Collection time is physical work incurred once and charged C/3 per mixture fit, never C/6 across both treatments. Publish shared and dataset-specific preparation, all target/fitting work and complete deployment cost separately. Amortized H=1/100/10000 views retain C and do not change the efficacy decision.

The 900-second collection cap is an execution bound, not a runtime prediction. All 72 trajectories may reach 2,188 steps; 74 resets and 157,536 steps are the hard count limits. Quota sufficiency is unknown before collection and is a legitimate stopping outcome. The saved audit is separately bounded. Writing the protocol is not execution, and a successful collection does not substitute for Stage 2 qualification and its independently frozen plan.

If fresh control does not improve despite this fixed exposure change, retain the failed recipe and consider a separately controlled spatial-readout or feature-scaling study. This result would not prove that all coverage, recurrent state or connectome mechanisms are ineffective. The present full public belief and ordinary model do not establish a memory bottleneck.
