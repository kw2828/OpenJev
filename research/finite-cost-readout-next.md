# Next: replicate the learned readout, then test longer blind supervision

**Proposal only. Neither step is registered or executed.** The
[completed readout study](finite-cost-readout-results.md) changes the immediate
priority: retain the learned head. It passes short-horizon learning and
observed filtering, while five blind-extrapolation conditions fail. A new
architecture would currently obscure a useful result that needs replication.

## First, replicate on untouched seeds and data

Use six fits: learned bounded readout versus the initially function-matched
fixed softened control, with three new fit seeds and a fresh TRAIN/DEV attempt
pool. Keep the H2 recipe, 480 epochs, batch size 64, Adam 0.003, global clip 5,
coefficient-one prefix NLL, paired filter initialization and attempt orders.
Preserve all attempts, terminal masks and global endpoint/event denominators.
Save every final checkpoint before generating DEV. Do not warm-start or select
the promising seed from the completed experiment.

Before data generation, freeze these continuation requirements: the learned
arm must again pass SHORT_HORIZON_LEARNING and OBSERVED_FILTERING_EXTRAPOLATION,
and lower H4 and H8 regret against its paired softened control in every seed.
Report BLIND_EXTRAPOLATION unchanged, including any failures. No mean can
rescue a failed seed and no replacement seed is allowed. This narrower
replication repeats the trainability comparison; it does not repeat the
original exact-C anchor comparison. That limitation stays explicit.

If replication fails, preserve the outcome and investigate stability under a
new proposal. Do not execute the second step automatically or select only the
successful fits. A successful replication permits a separate registration of
the second step; it establishes no transfer or novelty by itself.

## Then, separate longer supervision from more optimization

Compare nine new fits with a learned head in every arm:

1. H2 blind and observed training for 480 epochs, the replicated reference.
2. Blind cost and survival supervision through H4 for 480 epochs. Average each
   blind component over H1-H4 and retain its original objective weight.
   Observed supervision stays at H1/H2; prefix NLL stays coefficient one.
3. The original H2 objective with additional updates. Fix its epoch count using
   a separately qualified engineering-only timing procedure before scientific
   data generation, to approximately match the H4 arm's training budget.

Use a common fresh H4 target bank, filter/readout initialization and paired
minibatch permutations. H2 arms receive only H1/H2 targets and action/evidence
slices. Extra epochs use a predeclared continuation of the same order schedule.
Keep every final checkpoint and evaluate a separate fresh DEV population.

The question is whether longer blind supervision improves H8 decisions beyond
spending similar compute on the original objective. Before execution, specify
the engineering timing sample, deterministic epoch calculation, allowed
budget-ratio range, runtime caps and behavior if that range fails. A mismatch
must be reported; it cannot be repaired with outcome-dependent extra epochs.

Keep all three existing absolute criteria visible. H4 becomes trained-horizon
performance for the extended arm and must not be described as its unseen
extrapolation. H8 remains unseen during training. Require the short and observed
criteria, the existing absolute H8 cost/regret/survival thresholds for every
seed, and per-seed H8 regret improvement over both controls. Freeze the complete
acceptance rule before starting; H4/H8 means alone are insufficient.

Report actual wall time, parameter storage, updates, attempt and prefix-event
exposures, supervised forecast steps and recurrence work. The extra-update
control also receives more prefix exposures. Approximate compute matching does
not equalize supervision or optimization geometry, and a single-process timing
comparison is not an inference-speed claim.

## Scope of the possible contribution

Multistep model training is established. [PlaNet](https://arxiv.org/abs/1811.04551)
introduced latent overshooting as a multistep variational objective. The
proposed comparison directly supervises costs and survival; it does not copy
PlaNet's complete objective, planning system or results. Likewise,
[value equivalence](https://proceedings.neurips.cc/paper/2020/hash/3bb585ea00014b0e3ebe4c6dd165a358-Abstract.html)
already motivates learning models for useful decisions. These sources are
context, not a novelty claim for training a decision head.

The synthetic world and privileged readout initialization remain limitations
through both steps. Once a stable baseline passes, scenario shifts and a
second environment are necessary before native-task claims. Recurrent
factorization, connectome wiring, calibrated uncertainty and RL should then
address a measured weakness against this stronger baseline, with their own
matched controls. The current evidence does not support an ICLR-level
architecture claim.
