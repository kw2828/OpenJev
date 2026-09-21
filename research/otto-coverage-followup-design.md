# Proposed TRAIN-state coverage control

**PROPOSED. Not frozen, admitted or executed.** This note creates no new collection,
training or evaluation run. The completed Bellman-control study is independently
audited: 64,603,745 comparisons agree, including all 42 conditions. Its failed
continuation decision, original criteria and earlier failures remain unchanged.

## Why test coverage next

The audited study reports 19/42 continuation conditions passed: 0/18 absolute
competence conditions and 19/24 comparisons with the weaker learned controls.
Backup-family success is 64.30%, 39.78% and 38.62% for lambda3/4/5, versus analytic
control's 100% in all three settings. On lambda5, unchanged checkpoints perform
better than the backup family: 42.99% success and 1,262.43 capped moves, versus
38.62% and 1,374.68. These audited aggregates do not admit this proposed next
experiment or identify the cause of the competence failure.

All three backup target means and maxima decrease across refreshes; final
negative-target counts are zero. Epoch-40 online MSE against each arm's delayed
targets is 0.00017018, 0.00015769 and 0.00015552. Small error against self-generated
targets does not establish correct action ordering, a small final Bellman
residual, or reliable values on student-visited states. These losses also cannot
be compared directly with MC losses against different labels. Coverage is one
testable explanation, not an identified cause. Compressing 11,025 spatial entries
into eight measurements before a nonlinearity remains a competing explanation.

Student-visited training data is not new to this project. The earlier
[symmetry/action-head study](otto-symmetry-head-results.md) pooled 4,596 prefixes
from 144 student trajectories with teacher data, then failed its 54-condition
rule with 2 conditions passed across 720 evaluations. It trained direct action
preferences. This proposal instead isolates coverage within the current fixed
MLP scalar-value model and observation-backup objective. The earlier failure
remains relevant; it neither validates this proposal nor leaves student-state
training generally untested.

## One intervention, same ordinary model

Restore the same three original `mlp8` checkpoints, seeds 10101/10102/10103, with
their exact parameters and `c0`. For each seed, compare two observation-backup
continuations:

- **Teacher only:** all 5,589 original TRAIN teacher prefixes.
- **Fixed mixture:** exactly 5,589 prefixes, combining a prospectively fixed
  teacher-prefix quota with fresh student-visited TRAIN prefixes.

Both arms use the same architecture, public posterior, known kernels, features,
sixteen explicit branches, mass floors, signed values and eligible-action rule.
Retain fresh Adam, 40 epochs, batch 128, clipping, paired index permutations and
eight delayed target refreshes per fit. This gives 1,760 updates per fit and
10,560 across six fits. Use uniform row MSE and the fixed final checkpoint.
No hyperparameter search or validation-based selection is proposed.

Each arm generates its own delayed targets. Thus this tests a data-coverage
intervention within one learning procedure; it does not hold realized target
values constant or constitute an architecture-only comparison. Both arms pay
for target construction. Equal row/update counts do not imply equal total cost.

## Fresh collection without evaluation reuse

Freeze disjoint TRAIN seeds, the mixture quota, collector order, episode budget,
prefix-selection rule and initial-hit allocation before collection. Use all
three unchanged starting policies as fixed collectors. Pool their public
trajectories once; every fitting seed receives the same mixture dataset. Do not
choose a successful collector or preferentially retain successful trajectories.
Retain found and horizon-censored episodes and their final public updates;
censoring is not a zero-return or zero-continuation label.

Use only training-supported kernels for fitting. Exclude every old EVAL case,
trajectory and label, including the now-exposed Bellman-control evaluation.
Original VALID remains descriptive and outside fitting or schedule selection.
Fresh evaluation cases must be frozen separately, paired across every arm, and
retain the three unchanged checkpoints and analytic controller. Keep all fit
seeds, strata, blocks and failures visible.

## Costs, interpretation and the next decision

Charge collection, reconstruction, target generation, fitting, diagnostics and
complete deployed control separately. Report total physical work and deployment
amortization with the collection charge included. Specify resource caps before
admission; neither collection nor target cost may disappear behind equal updates.

Preserve an absolute analytic competence anchor as a **prospective candidate**:
at least 95% success and no more than 1.05 times analytic capped moves for every
fit seed and setting. Its complete comparison, consistency and cost rules still
require a frozen protocol. No new coverage-gain threshold is established here.

A coverage gain without competence would remain insufficient for architecture
claims. If the mixture does not help, test wider ordinary MLP and small CNN
readouts against width eight using identical data and a common frozen target
sequence. They are controls for capacity and spatial processing before the
proposed specialized spatial readout. Failure of this coverage recipe would not
prove that coverage is irrelevant, that compression caused the failure, or that
memory/connectome mechanisms are ruled out.

Sources: [frozen continuation protocol](otto-bellman-control-protocol.md),
[spatial-readout hypothesis](otto-spatial-readout-hypothesis.md), and completed
worker [summary](../output/otto-bellman-control-v1/run-01/summary.json),
[fit curves](../output/otto-bellman-control-v1/run-01/fit-curves.jsonl), and
[target statistics](../output/otto-bellman-control-v1/run-01/target-refreshes.jsonl).
The full [independent audit summary](../output/otto-bellman-control-v1/audit-01/summary.json)
and [audit receipt](../output/otto-bellman-control-v1/audit-01/receipt.json) bind the
completed study and its successful supervised process.
