# Bounded token alignment cost measurement

This is a synthetic engineering measurement for the
[token alignment design](dialogue-token-alignment-design.md), with no model
quality result. Do not train on real examples unless this measurement admits
the complete planned comparison. All earlier study outcomes remain final.

## Fixed comparison

Use `flat_stratum`, `token_mean`, and `token_aligned`, with fresh seeds
6201, 6202, and 6203. Preserve the typed study's fit/evaluation membership,
20 epochs, effective batch 256, AdamW learning rate .001, weight decay .0001,
gradient clip 1, CPU float32, four intra-op threads, one inter-op thread,
and deterministic operations. Fix row microbatches at 32 for all arms,
including both controls. Use all candidates and tokens without truncation.
Accumulate weighted loss sums divided by the actual effective-batch row
count; clip and update once per effective batch. Candidate chunks feed one
final supported-candidate softmax, never separate chunk normalizations.

The baseline is the original `DialogueTypedObservation("flat")`, including
its branch gate and candidate attention. Share only named semantically
compatible query/candidate projection, feature, head and gate initial tensors.
Copy the complete new-model initializer between token mean and alignment.
The two token models have identical registered parameter counts; the old
baseline is a separate practical control with different capacity.

## Workload and measurement

Authenticate original metadata and completed schema-cache geometry. Hash
existing float payloads without decoding their values. Generate and freeze
all 20 epoch permutations per seed. Every arm uses these exact orders.
Enumerate actual row microbatch padding costs for all 6,900 training updates
and 162 evaluation batches per arm across the three seeds, including tails.
Partition training and evaluation into three deterministic workload strata
using padded pairwise positions. Select the first batch attaining the maximum
geometry cost within each stratum. Record the boundaries, representatives,
full counts and total workload before model execution.
Record component maxima as well: the largest pairwise matrix need not maximize
baseline attention work or comparison-network work. A stratum representative
is not a universal worst case for every arm.

Use deterministic synthetic features through the same actor assembler, valid
synthetic candidate targets, positive normalized token priors, and supported
previous-value indicators. Do not load real float features or compute actual
task accuracy. For each of three arms and each train/evaluation stratum, run
one warm event and three measured events. An event covers one full effective
batch with its row microbatches. Training includes assembly, validation,
forward, weighted loss, backward, gradient checks, clipping, AdamW and logging.
Evaluation includes assembly, forward, invariants and output materialization.
There are 18 arm/phase/stratum cells and 72 events, including 36 optimizer
updates. Synthetic objectives are numerical checks, not evidence of efficacy.

Use each cell's largest measured duration times its actual stratum population
to project total training and evaluation. Add observed non-measured remainder
cost conservatively, including authentication, setup and warm events. Report
separately any serialization or end-of-study hashing that the probe does not
measure. Include the already measured schema preparation cost in the report.
This extrapolation is a heuristic, not a guaranteed runtime bound or a clean
hardware benchmark. Record available process-load observations.

## Fixed limits and decision

One exclusive freeze and one exclusive probe attempt, each at most 300 seconds,
6 GiB process-lifetime peak RSS, and 64 MiB output. Preserve a failed attempt;
no alternative batch sizes, replacement seeds, kernel variants, or reduced
support are authorized by this measurement.

Admit the complete nine-fit scientific study only if projected study cost is
at most 2,880 seconds, leaving 20% headroom under a 3,600-second whole-study
ceiling, and all numerical, memory, and coverage checks pass. The future study
must enforce its actual wall/memory limits. Admission does not authorize a
claim of quality, speedup, novelty, recurrence, or calibration.

If admitted, publish a separate scientific protocol before any fit. Its
continuation rule compares alignment with both controls on the 578 primary
changed rows: three-seed mean accuracy at least 2 percentage points higher
and wrong-selected-branch rate at least 2 points lower; neither quantity may
worsen in any paired seed. Mean retained-state error and supported TRUE and
DONTCARE false-positive rates may each increase by no more than .5 points.
All nine fits must complete before scoring. Report equal-service metrics,
losses, Brier scores, repair/harm counts and sparse TRUE/DONTCARE recalls as
supporting evidence. This is an engineering continuation rule on historically
exposed TRAIN data, not statistical confirmation.
