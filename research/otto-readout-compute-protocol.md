# Readout versus recurrent adaptation at calibrated training cost

This separately registered development experiment follows the mixed
[readout/state ablation](otto-readout-state-ablation-results.md). It asks whether
updating recurrent weights still helps when the cheaper action readout receives
more training. It does not reopen either closed study, their reserved confirmation
seeds, or the original TEST data. It makes no architecture novelty claim.

## Recipe frozen before fresh collection

Both arms start from all three original 80-epoch parents, fit seeds 309000001,
309000002 and 309000003, and use the original 54 TRAIN paths. The action-residual
arm updates 116 parameters for **74 epochs**. Full joint updates 6,112 parameters
for **40 epochs**. Each has the same 6,112-parameter inference network.

The allocation is fixed from the prior published mean continuation times:
`floor(40 * 168.052592264 / 90.10169136066666) = 74`. This predicts comparable
continuation time; it does not establish equal computation. Each fit must finish
its prescribed epochs. No live timer, result, intermediate checkpoint or DEV
score changes its epoch allocation. Report all three measured residual/joint
time ratios; all must lie in **[0.90, 1.10]** to call costs comparable.

Retain the existing complete-episode kernel: Adam 0.003, gradient clipping 5,
six episodes per batch, chronological 32-step chunks with carry detachment,
P4 observations, legal-centered nonquery MSE plus equally weighted all-four
prequery MSE in /64 units. Restart the PCG64(fit_seed) episode-permutation stream
and optimizer for every arm. Their first 40 epochs have the same episode orders.
There are six fits, 3,078 Adam updates and 18,468 episode exposures. This matches
the training dataset, not the number of exposures. There is no feature cache,
new loss, learning-rate schedule or selected intermediate checkpoint.

Fit timing includes construction, optimizer setup, complete training, validation
and checkpoint serialization within the producer's recorded fit interval. Shared
pretraining, TRAIN collection and process setup are reported separately rather
than credited to either continuation. Report all measured worker and collection
costs. This comparison concerns the qualified implementations on one CPU runtime,
not the best possible implementation of either learning rule.

## Fresh development paths

Collect 36 complete paths: six environment cases in each of lambda3 and lambda4,
each run with analytic, neural and period4-hold fixed collectors. Exact seeds are
318000001 through 318000006 and 319000001 through 319000006, respectively. Use
the same environment, teacher, horizon 2,188, initial hit `1 + case % 3`, and
rotating collector order as the qualified residual collector. The seed review
records the scope of its repository-local lexical scan; it is not a proof of
globally unused numeric blocks.

Use the existing complete-census annotation boundary: fix an unqueried action
before its teacher annotation and never let annotation refresh the held policy's
cache. This provides targets along fixed collectors' paths, not rollouts driven
by either new learner. One teacher score is paid for every retained state.
No Astra calls are required. No old TEST or reserved confirmation is opened.

Register this protocol, both collection/training implementations, qualification
evidence and seeds before collection. The numerical training plan is finalized
only after the original collection and metadata bridge close, binding the actual
fresh data bytes to the already fixed recipe. Complete all six final fits before
decoding fresh DEV for evaluation. Evaluate all six fits and all three unchanged
parents, nine views, using the canonical frozen predictor at batch one/chunk32.
No candidate is promoted or selected between views.

## Decision rule and interpretation

Full joint is the fixed candidate. For each fit seed and each setting, require
at least a **5% reduction in later case-weighted raw teacher-cost gap** against
both residual74 and the unchanged parent, with no full-episode gap regression
against either. There are six paired cells with four checks each. A zero
comparator gap cannot establish positive relative improvement. All cells and
all three actual timing-ratio checks must pass the overall continuation rule.
Publish efficacy and cost comparability separately, including every failed cell.

These are fresh registered development paths held out from training. They are
not the earlier study's confirmation panel or a new environment. Repeated fit
seeds share the same environment cases; do not treat them as independent data.
Report case-weighted and episode/row-weighted support, raw gaps, agreement,
centered MSE and prequery MSE. No p-value or equivalence claim is planned.

A pass supports a further hypothesis about recurrent weight adaptation at the
measured cost. It does not establish online adaptation, a connectome advantage,
a learned world model, autonomous control improvement or a conference result.
A fail remains a useful limit and cannot be repaired through seed replacement,
extra epochs or post-result threshold changes.

## Qualification and closure

All changed sources get fabricated control-flow, seed, timing and independent
metric tests plus lint. Reuse the unchanged qualified numerical training kernel.
Capacity uses only authenticated historical TRAIN lengths and fabricated values,
one epoch for each arm. Admission requires
`2 * 3 * (74 * residual_epoch_seconds + 40 * joint_epoch_seconds) + 240 <= 4050`.
It performs no empirical array or checkpoint decoding. The native preflight is
metadata-only under the original native interpreter.

Collection has a 3,600-second suspend-aware cap, 4 GiB RSS and 1 GiB output, with
36 x 2,188 maxima for state-dependent calls. The metadata bridge has 60 seconds.
Training/evaluation has 5,400 seconds, 4 GiB and 512 MiB; independent saved-output
audit has 600 seconds, 2 GiB and 128 MiB. Use one numerical CPU thread and the
qualified process-group supervisor. Failed scientific attempts retain their
partial evidence and do not retry or extend these bounds.

The independent audit reconstructs histories, unchanged/frozen parameter
witnesses, all training orders and counters, scalar metrics, paired checks and
cost comparability from saved artifacts. It does no model, optimizer, teacher or
native calls. Official reporting requires both original worker closures and
unchanged source/input descriptors. Qualification is engineering evidence only;
it cannot substitute for that experiment.
